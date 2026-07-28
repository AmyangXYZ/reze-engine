// Composite: HDR scene + bloom pyramid → Filmic tone map → gamma → swapchain.
// Bloom tint/intensity applied at combine (EEVEE treats them as combine-stage params, not prefilter).
//
// The shader is a TEMPLATE: buildCompositeShader() emits either the base pass or
// a variant with a user background effect injected (setBackgroundEffect). The
// effect is background mode 3, a sibling of the 360 equirect (mode 2) — it reuses
// the same per-pixel view-ray reconstruction and composites in display space
// under the scene, so it never affects lighting, bloom, or tonemapping.

/** What a user background effect must define, documented once:
 *
 *    fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f
 *
 *  - `ray`  — normalized world-space view direction of this pixel (left-handed,
 *             +Z forward; identical to what the 360 skybox samples by).
 *  - `uv`   — 0..1 across the canvas, origin bottom-left (shadertoy-style).
 *  - `time` — seconds since the effect was applied.
 *  - `bgResolution()` — canvas size in pixels, for aspect correction.
 *  - declared params arrive as `params.<name>` (f32 or vec3f).
 *  Return display-space sRGB + alpha, 0..1. The effect is a LAYER: it is
 *  over-composited onto the base background (solid color / 360 equirect /
 *  transparent) and sits behind the scene — alpha 0 lets the base show
 *  through, so e.g. a starfield returns stars with a transparent sky. */
export type CompositeEffectSource = {
  /** User WGSL defining `background(...)` (plus any helpers it wants). */
  wgsl: string
  /** Codegen'd `struct BgParams {...}` + binding decl; empty when no params. */
  paramsDecl: string
}

const COMPOSITE_HEAD = /* wgsl */ `
// Pipeline-override constant: the engine creates two composite pipelines, one
// with APPLY_GAMMA=false (gamma=1 fast path) and one with APPLY_GAMMA=true.
// The 'if (APPLY_GAMMA)' below is resolved at pipeline-compile time — the
// dead branch is dropped by the shader compiler (no runtime branch, no pow
// invocation on Safari's Metal backend in the common case).
override APPLY_GAMMA: bool = true;

@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;   // bloomUpTexture mip 0 (full pyramid top)
@group(0) @binding(2) var bloomSamp: sampler;
@group(0) @binding(3) var<uniform> viewU: array<vec4<f32>, 10>;
// Aux mask/alpha texture. .r = bloom mask (unused here; bloom blit uses it).
// .g = accumulated canvas alpha (what hdr.a carried before the HDR format
// became rg11b10ufloat). We unpremultiply HDR by this alpha for tonemap, then
// re-premultiply the tonemapped color for output so the premultiplied canvas
// alphaMode composites the WebGPU surface over the page background correctly.
@group(0) @binding(4) var maskTex: texture_2d<f32>;
// Filmic tone curve baked to a WIDTH×1 r16float LUT (bakeFilmicLut on the CPU side).
// Domain: log2(linear) mapped to [0,13]. Replaces the old array<f32,14> that was indexed
// by a runtime u32 (a Metal-backend smell — dynamic local-array indexing lowers to a
// per-invocation copy/switch) and interpolated piecewise-linearly (C0, so segment slope
// discontinuities showed as Mach bands in smooth skin/shadow gradients). The LUT is a
// monotone-cubic (Fritsch–Carlson) fit through the same 14 anchors — same values, C1
// continuity kills the banding — sampled with hardware linear filtering.
@group(0) @binding(5) var filmicLut: texture_2d<f32>;
// viewU[0] = (exposure, invGamma, _, _);  viewU[1] = (tint.rgb, intensity)
// viewU[2] = (background.rgb, mode) — display-space sRGB, composited UNDER the
//            scene post-tonemap. BASE-layer mode: 0 transparent (DOM shows),
//            1 solid color, 2 = 360 equirect skybox sampled by view ray. A user
//            WGSL effect is a separate LAYER over the base (viewU[6].y flag).
// viewU[3] = (camera right, tanHalfFov·aspect); viewU[4] = (camera up, tanHalfFov);
// viewU[5] = (camera forward, _) — refreshed per frame while skybox/effect active.
// viewU[6] = (time seconds, effect on/off, canvas width, canvas height).
// viewU[7] = (grade offset.rgb, contrast);  viewU[8] = (grade power.rgb, saturation);
// viewU[9] = (grade slope.rgb, grade on/off) — see grade() below.
// invGamma = 1/gamma precomputed on CPU — avoids a per-pixel divide.
@group(0) @binding(6) var bgEquirect: texture_2d<f32>;

// Must match FILMIC_LUT_WIDTH in engine.ts (bakeFilmicLut).
const FILMIC_LUT_W: f32 = 256.0;

fn filmic(x: f32) -> f32 {
  // Reference checkpoints (Blender 3.6 Filmic MHC, sobotka/filmic-blender
  // look_medium-high-contrast.spi1d): linear 0.18 → ~0.395, linear 1.0 → ~0.83.
  // NOTE: version-pinned to Blender 3.6 — 4.x defaults to AgX, not Filmic.
  let t = clamp(log2(max(x, 1e-10)) + 10.0, 0.0, 13.0);
  // Map t∈[0,13] to the texel-center of baked sample j = t·(W-1)/13.
  let u = (t * (FILMIC_LUT_W - 1.0) / 13.0 + 0.5) / FILMIC_LUT_W;
  // textureSampleLevel (explicit LOD, no derivatives) is legal in non-uniform flow.
  return textureSampleLevel(filmicLut, bloomSamp, vec2f(u, 0.5), 0.0).r;
}

/** Canvas size in pixels — for user background effects (aspect correction). */
fn bgResolution() -> vec2f { return viewU[6].zw; }

/** Color grading, applied to the tonemapped SCENE (not the background — see the
 *  call site). The core is ASC CDL, the film-industry interchange standard:
 *
 *      out = (in · slope + offset) ^ power        then saturation  (SOP → SAT)
 *
 *  Using the real standard rather than invented controls means a look authored
 *  here maps onto any grading tool. slope/offset/power are derived on the CPU
 *  from the UI's shadow/midtone/highlight colors (see setColorGrading), so the
 *  per-pixel cost is one mul-add, one pow, one lerp. */
fn grade(c: vec3f) -> vec3f {
  var x = pow(max(c * viewU[9].xyz + viewU[7].xyz, vec3f(0.0)), viewU[8].xyz);
  // Contrast pivots on 0.5 — display-referred midpoint, since we grade post-Filmic.
  x = (x - vec3f(0.5)) * viewU[7].w + vec3f(0.5);
  // Rec.709 luma, matching the ASC SAT node.
  let luma = dot(x, vec3f(0.2126, 0.7152, 0.0722));
  return max(mix(vec3f(luma), x, viewU[8].w), vec3f(0.0));
}
`

const COMPOSITE_BODY = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let coord = vec2<i32>(fragCoord.xy);
  let hdr = textureLoad(hdrTex, coord, 0);
  let alpha = textureLoad(maskTex, coord, 0).g;
  let a = max(alpha, 1e-6);
  let straight = hdr.rgb / a;
  let fullSz = vec2f(textureDimensions(hdrTex));
  // Bloom is at half-res (pyramid mip 0). Sampler interpolates back to full-res UVs.
  // fragCoord.xy is already at pixel center (e.g. 0.5, 0.5 for first pixel).
  let bloomUv = fragCoord.xy / max(fullSz, vec2f(1.0));
  let tint = viewU[1].xyz;
  let intensity = viewU[1].w;
  let bloom = textureSampleLevel(bloomTex, bloomSamp, bloomUv, 0.0).rgb * tint * intensity;
  let combined = straight + bloom;
  let exposed = combined * exp2(viewU[0].x);
  let tm = vec3f(filmic(exposed.r), filmic(exposed.g), filmic(exposed.b));
  var disp = max(tm, vec3f(0.0));
  // Grade the SCENE only, before the display gamma. Deliberately not applied to
  // the background: it keeps a picked background color exactly as picked, and —
  // load-bearing — leaves green-screen mode's key color unshifted so chroma
  // keying still works. Skipped entirely when the grade is neutral.
  if (viewU[9].w > 0.5) {
    disp = grade(disp);
  }
  if (APPLY_GAMMA) {
    disp = pow(disp, vec3f(viewU[0].y));
  }
  // Composite over the background in display space (premultiplied out). The
  // background is TWO layers: a base (transparent / solid color / 360 equirect)
  // and an optional user WGSL effect over-composited onto it.
  let bg = viewU[2];
  var bgA = select(0.0, 1.0, bg.w > 0.5);
  var bgPm = bg.rgb * bgA;  // premultiplied accumulator
  let fxOn = viewU[6].y > 0.5;
  if ((bg.w > 1.5 || fxOn) COVERAGE_GATE) {
    // The equirect and any effect both need this pixel's world-space view ray,
    // rebuilt from the camera basis. The dome sits at infinity (no parallax) —
    // PhotoDome-style, display-only.
    let ndc = vec2f(fragCoord.x / fullSz.x * 2.0 - 1.0, 1.0 - fragCoord.y / fullSz.y * 2.0);
    let dir = normalize(viewU[5].xyz + ndc.x * viewU[3].w * viewU[3].xyz + ndc.y * viewU[4].w * viewU[4].xyz);
    if (bg.w > 1.5) {
      // LH world (+Z forward): longitude = atan2(x, z), Babylon-PhotoDome convention.
      let su = 0.5 + atan2(dir.x, dir.z) * 0.15915494309;  // 1/(2π)
      let sv = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) * 0.31830988618;  // 1/π
      bgPm = textureSampleLevel(bgEquirect, bloomSamp, vec2f(su, sv), 0.0).rgb;
    }
    BG_EFFECT_CALL
  }
  return vec4f(disp * alpha + bgPm * (1.0 - alpha), alpha + bgA * (1.0 - alpha));
}
`

// Base variant: no effect installed, the flag is never set — the ray block only
// runs for the equirect, and there is nothing to add. (`dir` may go unused when
// this compiles with mode<2 shaders; WGSL is fine with an unused let.)
const NO_EFFECT_CALL = `_ = dir;`

// uv flipped to bottom-left origin (shadertoy convention); clamped so a stray
// effect can't push negatives/NaN into the premultiplied composite. Standard
// OVER onto the base layer.
const EFFECT_CALL = /* wgsl */ `
    if (fxOn) {
      let bgUv = vec2f(fragCoord.x / fullSz.x, 1.0 - fragCoord.y / fullSz.y);
      let fx = clamp(background(dir, bgUv, viewU[6].x), vec4f(0.0), vec4f(1.0));
      bgPm = fx.rgb * fx.a + bgPm * (1.0 - fx.a);
      bgA = fx.a + bgA * (1.0 - fx.a);
    }
`

// Derivative builtins are illegal in non-uniform control flow (WGSL uniformity
// analysis rejects the pipeline), so the coverage gate below can only wrap
// effect code that doesn't use them. Checked textually at build time.
const USES_DERIVATIVES = /\b(?:fwidth|dpdx|dpdy)(?:Fine|Coarse)?\s*\(/

/** Skip the whole background block (equirect sample + effect) behind pixels the
 *  model fully covers — the composite multiplies the result by (1 - alpha) = 0
 *  there anyway, and on a full-screen effect that's a third or more of the frame
 *  (the cost Safari feels most). The equirect uses explicit-LOD sampling, which
 *  is always legal in non-uniform flow; only derivative-using effects must keep
 *  uniform control flow and forgo the gate. */
function coverageGate(effect?: CompositeEffectSource | null): string {
  const gated = !effect || !USES_DERIVATIVES.test(effect.wgsl)
  return gated ? "&& alpha < 0.999" : ""
}

export function buildCompositeShader(effect?: CompositeEffectSource | null): string {
  if (!effect)
    return (
      COMPOSITE_HEAD +
      COMPOSITE_BODY.replace("BG_EFFECT_CALL", NO_EFFECT_CALL).replace("COVERAGE_GATE", coverageGate(null))
    )
  return (
    COMPOSITE_HEAD +
    "\n// ── user background effect (setBackgroundEffect) ──\n" +
    effect.paramsDecl +
    "\n" +
    effect.wgsl +
    "\n" +
    COMPOSITE_BODY.replace("BG_EFFECT_CALL", EFFECT_CALL.trim()).replace("COVERAGE_GATE", coverageGate(effect))
  )
}

/** Kept for compatibility with existing imports (the base, no-effect shader). */
export const COMPOSITE_SHADER_WGSL = buildCompositeShader(null)
