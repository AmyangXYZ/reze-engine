// Composite: HDR scene + bloom pyramid → Filmic tone map → gamma → swapchain.
// Bloom tint/intensity applied at combine (EEVEE treats them as combine-stage params, not prefilter).
//
// The shader is a TEMPLATE: buildCompositeShader() emits either the base pass or
// a variant with user WGSL injected at one or both effect MOUNTS (setEffect).
// The two mounts are the same idea on either side of the scene:
//
//   background(...)  under the scene — a sibling of the 360 equirect (mode 2),
//                    reusing the same per-pixel view-ray reconstruction.
//   foreground(...)  over the finished frame, handed the scene's depth in metres
//                    so it can be occluded by whatever it passes behind.
//
// Both composite in display space, so neither affects lighting, bloom, or
// tonemapping, and both are captured by offline export like any background.

/** What user effect WGSL may define, documented once. A file declares its own
 *  mounts by which of these it defines — defining both is how one file is one
 *  weather system (dark sky behind, rain in front):
 *
 *    fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f
 *    fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f
 *
 *  - `ray`   — normalized world-space view direction of this pixel (left-handed,
 *              +Z forward; identical to what the 360 skybox samples by).
 *  - `uv`    — 0..1 across the canvas, origin bottom-left (shadertoy-style).
 *  - `time`  — seconds since the effect was applied.
 *  - `depth` — FOREGROUND ONLY. Camera-space distance in metres of whatever the
 *              scene drew at this pixel, the far plane where it drew nothing.
 *              Compare a particle's own distance against it and the model
 *              occludes it; fog needs no comparison at all, its alpha simply IS
 *              a function of distance.
 *  - `bgResolution()` — canvas size in pixels, for aspect correction.
 *  - declared params arrive as `params.<name>` (f32 or vec3f), shared by both.
 *
 *  Return display-space sRGB + alpha, 0..1. Both mounts are alpha-composited
 *  LAYERS, so alpha is what decides how much they replace: a background effect
 *  at alpha 1 covers the base (solid color / 360 equirect / transparent) and at
 *  0 lets it through, which is how a starfield is stars over the user's color;
 *  a foreground at alpha 1 covers the frame. No mode flag anywhere — the alpha
 *  channel already says it. */
export type CompositeEffectSource = {
  /** The user's WGSL verbatim: helpers plus whichever entry points it defines. */
  wgsl: string
  /** Codegen'd `struct EffectParams {...}` + binding decl; empty when no params. */
  paramsDecl: string
  /** Defines `fn background(...)` — mount under the scene. */
  hasBackground: boolean
  /** Defines `fn foreground(...)` — mount over the finished frame. */
  hasForeground: boolean
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
@group(0) @binding(3) var<uniform> viewU: array<vec4<f32>, 11>;
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
//            WGSL effect is a separate LAYER over the base — no mode of its own,
//            and no on/off uniform either: the pipeline is REBUILT per effect, so
//            the compiled variant IS the flag.
// viewU[3] = (camera right, tanHalfFov·aspect); viewU[4] = (camera up, tanHalfFov);
// viewU[5] = (camera forward, _) — refreshed per frame while skybox/effect active.
// viewU[6] = (time seconds, view transform id, canvas width, canvas height).
// viewU[7] = (grade offset.rgb, contrast);  viewU[8] = (grade power.rgb, saturation);
// viewU[9] = (grade slope.rgb, grade on/off) — see grade() below.
// viewU[10] = (camera world position, _) — refreshed with the basis above.
// invGamma = 1/gamma precomputed on CPU — avoids a per-pixel divide.
@group(0) @binding(6) var bgEquirect: texture_2d<f32>;
// The scene pass's own MSAA depth buffer, bound depth-only. NOT an extra
// render target: with neither depth of field nor a foreground effect active the
// scene pass discards depth (TBDR tile memory never spills) and this binding is
// never read. Either feature makes the pass store it instead, and both read
// sample 0 — the DoF gather, and linearDepth() for the depth handed to
// foreground().
@group(0) @binding(8) var depthTex: texture_depth_multisampled_2d;
// dofU[0] = (enabled, focusDistance, focusRange, aperture)
// dofU[1] = (maxBlurRadiusPx, bladeCount, sampleCount, anamorphicRatio)
// dofU[2] = (projA, projB, _, _) — z-buffer → camera-space depth inversion,
//           viewZ = projB / (z - projA), rebuilt per frame because near/far
//           track the camera radius. Cleared depth (1.0) inverts to the far
//           plane, so empty sky reads as maximally defocused background.
@group(0) @binding(9) var<uniform> dofU: array<vec4<f32>, 3>;

// Must match FILMIC_LUT_WIDTH in engine.ts (bakeFilmicLut).
const FILMIC_LUT_W: f32 = 256.0;

fn linearDepth(coord: vec2<i32>) -> f32 {
  let z = textureLoad(depthTex, coord, 0);
  // projA > 1 for every valid z in [0,1], so the divisor never crosses zero.
  return clamp(dofU[2].y / (z - dofU[2].x), 0.05, 100000.0);
}

/** Signed circle of confusion in device pixels — negative in front of the
 *  focus band, positive behind it, zero inside it. */
fn circleOfConfusion(depth: f32) -> f32 {
  let focus = max(dofU[0].y, 0.05);
  let halfRange = max(dofU[0].z * 0.5, 0.01);
  let delta = depth - focus;
  let outside = max(abs(delta) - halfRange, 0.0);
  let radius = min(outside / max(depth, 0.05) * dofU[0].w * dofU[1].x, dofU[1].x);
  return select(-radius, radius, delta >= 0.0);
}

/** Premultiplied scene color (HDR + bloom) at an arbitrary pixel, for the
 *  bokeh gather. Explicit-LOD sampling only — legal in non-uniform flow. */
fn sceneSample(coord: vec2<i32>, fullSzI: vec2<i32>, fullSz: vec2f) -> vec4f {
  let p = clamp(coord, vec2<i32>(0), fullSzI - vec2<i32>(1));
  let sAlpha = textureLoad(maskTex, p, 0).g;
  let sHdr = textureLoad(hdrTex, p, 0).rgb / max(sAlpha, 1e-6);
  let sUv = (vec2f(p) + vec2f(0.5)) / fullSz;
  let sBloom = textureSampleLevel(bloomTex, bloomSamp, sUv, 0.0).rgb * viewU[1].xyz * viewU[1].w;
  return vec4f((sHdr + sBloom) * sAlpha, sAlpha);
}

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

/** The sRGB display encoding — Blender's "Standard" view transform, which is what
 *  NPR work uses: no film curve at all, so the colours a graph computes are the
 *  colours that land. Not a 2.2 power law; sRGB has a linear toe. */
fn srgb_encode(x: f32) -> f32 {
  let c = max(x, 0.0);
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

/** Which display transform, chosen per frame at viewU[6].y (0 filmic, 1 standard).
 *  A uniform branch rather than a pipeline variant: switching is rare, and both
 *  arms are cheap enough that specialising the pipeline would buy nothing. */
fn viewTransform(c: vec3f) -> vec3f {
  if (viewU[6].y > 0.5) {
    return vec3f(srgb_encode(c.r), srgb_encode(c.g), srgb_encode(c.b));
  }
  return vec3f(filmic(c.r), filmic(c.g), filmic(c.b));
}

/** Canvas size in pixels — for user effects (aspect correction). */
fn bgResolution() -> vec2f { return viewU[6].zw; }

/** The camera's world position. */
fn bgCameraPos() -> vec3f { return viewU[10].xyz; }

/** Where in the WORLD the scene drew this pixel — the depth handed to
 *  foreground() turned into a place. Without it an effect can only think in
 *  distances from the lens, which is no use to anything that belongs somewhere:
 *  fog lying on the ground has to know where the ground is.
 *
 *  depth measures along the VIEW AXIS, not along the ray, so it is divided by
 *  the ray's projection onto camera-forward before being walked out. At the far
 *  plane (nothing drawn) this lands a very long way off, which is what a sky
 *  should do to anything reading it. */
fn bgWorldPos(ray: vec3f, depth: f32) -> vec3f {
  let axis = max(dot(normalize(ray), viewU[5].xyz), 1e-4);
  return bgCameraPos() + normalize(ray) * (depth / axis);
}

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

  // ── Depth of field ──
  // Single-pass golden-angle gather over a polygonal (bladed) disk, in
  // premultiplied HDR before tonemap. Near-field taps that see focused
  // background are heavily down-weighted so a sharp subject doesn't bleed into
  // a blurred foreground; the reverse leak (background bokeh over the subject
  // edge) is damped less — that soft halo is what real lenses do. Scene layer
  // only: the composited background (solid / 360 / effect) stays sharp, which
  // is invisible while it sits at infinity behind a far-blurred stage.
  var scenePm = vec4f(combined * alpha, alpha);
  if (dofU[0].x > 0.5) {
    let centerDepth = linearDepth(coord);
    let centerCoc = circleOfConfusion(centerDepth);
    let radius = abs(centerCoc);
    if (radius > 0.35) {
      let fullSzI = vec2<i32>(fullSz);
      let sampleCount = clamp(dofU[1].z, 6.0, 24.0);
      let blades = clamp(dofU[1].y, 3.0, 12.0);
      let sector = 6.28318530718 / blades;
      var accum = scenePm;
      var weightSum = 1.0;
      for (var i = 0u; i < 24u; i++) {
        if (f32(i) >= sampleCount) { break; }
        let fi = f32(i) + 0.5;
        let ring = sqrt(fi / sampleCount);
        let angle = fi * 2.39996323;
        let localAngle = (fract((angle + 3.14159265359) / sector) - 0.5) * sector;
        let polygonRadius = cos(3.14159265359 / blades) / max(cos(localAngle), 0.01);
        var disk = vec2f(cos(angle), sin(angle)) * ring * polygonRadius;
        disk.x *= max(dofU[1].w, 0.25);
        let sp = coord + vec2<i32>(round(disk * radius));
        let cp = clamp(sp, vec2<i32>(0), fullSzI - vec2<i32>(1));
        let sampleDepth = linearDepth(cp);
        let sampleCoc = circleOfConfusion(sampleDepth);
        var w = 1.0;
        if (centerCoc < 0.0 && sampleDepth > centerDepth + dofU[0].z) {
          w *= 0.08;
        } else if (centerCoc > 0.0 && sampleDepth > centerDepth + dofU[0].z * 2.0) {
          w *= 0.35;
        }
        // A tap only contributes where its own blur circle reaches this pixel.
        let sampleRadius = abs(sampleCoc);
        w *= mix(0.2, 1.0, smoothstep(ring * radius - 1.0, ring * radius + 1.0, sampleRadius));
        accum += sceneSample(cp, fullSzI, fullSz) * w;
        weightSum += w;
      }
      scenePm = mix(scenePm, accum / max(weightSum, 1e-5), smoothstep(0.35, 1.75, radius));
    }
  }
  let sceneAlpha = scenePm.a;
  let sceneStraight = scenePm.rgb / max(sceneAlpha, 1e-6);

  let exposed = sceneStraight * exp2(viewU[0].x);
  var disp = max(viewTransform(exposed), vec3f(0.0));
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
  // This pixel's world-space view ray, rebuilt from the camera basis — what the
  // equirect samples by, and what both effect mounts navigate by. The dome sits
  // at infinity (no parallax): PhotoDome-style, display-only. Hoisted out of the
  // branch below because the foreground mount is past the end of it; it is pure
  // arithmetic on uniforms, which every backend sinks into whatever reads it.
  let ndc = vec2f(fragCoord.x / fullSz.x * 2.0 - 1.0, 1.0 - fragCoord.y / fullSz.y * 2.0);
  let dir = normalize(viewU[5].xyz + ndc.x * viewU[3].w * viewU[3].xyz + ndc.y * viewU[4].w * viewU[4].xyz);
  if (BACKGROUND_COND) {
    if (bg.w > 1.5) {
      // LH world (+Z forward): longitude = atan2(x, z), Babylon-PhotoDome convention.
      let su = 0.5 + atan2(dir.x, dir.z) * 0.15915494309;  // 1/(2π)
      let sv = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) * 0.31830988618;  // 1/π
      bgPm = textureSampleLevel(bgEquirect, bloomSamp, vec2f(su, sv), 0.0).rgb;
    }
    BACKGROUND_CALL
  }
  // The frame, premultiplied: scene over background. A var, not the return
  // expression, because the foreground mount composites onto it.
  var outRgb = disp * sceneAlpha + bgPm * (1.0 - sceneAlpha);
  var outA = sceneAlpha + bgA * (1.0 - sceneAlpha);
  FOREGROUND_CALL
  return vec4f(outRgb, outA);
}
`

// uv flipped to bottom-left origin (shadertoy convention); clamped so a stray
// effect can't push negatives/NaN into the premultiplied composite. Standard
// OVER onto the base layer. No `if` around it: the pipeline is rebuilt per
// effect, so this text only exists in variants whose WGSL defines background().
const BACKGROUND_CALL = /* wgsl */ `
    let bgUv = vec2f(fragCoord.x / fullSz.x, 1.0 - fragCoord.y / fullSz.y);
    let bgFx = clamp(background(dir, bgUv, viewU[6].x), vec4f(0.0), vec4f(1.0));
    bgPm = bgFx.rgb * bgFx.a + bgPm * (1.0 - bgFx.a);
    bgA = bgFx.a + bgA * (1.0 - bgFx.a);
`

// Same OVER, one layer later — onto the finished frame rather than onto the
// base. Ungated by design: a foreground runs at every pixel, including the ones
// the model covers, because covering them is the point.
const FOREGROUND_CALL = /* wgsl */ `
  let fgUv = vec2f(fragCoord.x / fullSz.x, 1.0 - fragCoord.y / fullSz.y);
  // The scene's own depth, so the effect can tell what is in front of it: a
  // petal compares its distance against this and lets the model take the pixel,
  // and fog's alpha is nothing but a function of it. Pixels the scene never drew
  // read the far plane, so distance fog closes over the backdrop too.
  let fgFx = clamp(foreground(dir, fgUv, viewU[6].x, linearDepth(coord)), vec4f(0.0), vec4f(1.0));
  outRgb = fgFx.rgb * fgFx.a + outRgb * (1.0 - fgFx.a);
  outA = fgFx.a + outA * (1.0 - fgFx.a);
`

// Derivative builtins are illegal in non-uniform control flow (WGSL uniformity
// analysis rejects the pipeline), so the coverage gate below can only wrap
// effect code that doesn't use them. Checked textually at build time.
const USES_DERIVATIVES = /\b(?:fwidth|dpdx|dpdy)(?:Fine|Coarse)?\s*\(/

/** The condition on the background block (equirect sample + background effect).
 *
 *  Two jobs. It skips the block behind pixels the model fully covers — the
 *  composite multiplies the result by (1 - alpha) = 0 there anyway, and on a
 *  full-screen effect that's a third or more of the frame (the cost Safari feels
 *  most). And with no background effect compiled in, it also skips the block
 *  entirely unless the equirect needs it.
 *
 *  The equirect uses explicit-LOD sampling, which is always legal in non-uniform
 *  flow; only derivative-using effects must keep uniform control flow and forgo
 *  the coverage half. The test is textual over the whole file, so a foreground
 *  that uses fwidth costs the background its gate — conservative, and only ever
 *  in the direction of correctness. (The foreground mount itself sits in uniform
 *  flow, so derivatives are always legal there.) */
function backgroundCondition(effect?: CompositeEffectSource | null): string {
  // sceneAlpha, not alpha: the bokeh gather spreads coverage, so a pixel the
  // sharp scene fully covered can end up needing background behind its blur.
  const coverage = "sceneAlpha < 0.999"
  if (!effect?.hasBackground) return `bg.w > 1.5 && ${coverage}`
  return USES_DERIVATIVES.test(effect.wgsl) ? "true" : coverage
}

export function buildCompositeShader(effect?: CompositeEffectSource | null): string {
  const body = COMPOSITE_BODY.replace("BACKGROUND_COND", backgroundCondition(effect))
    .replace("BACKGROUND_CALL", effect?.hasBackground ? BACKGROUND_CALL.trim() : "")
    .replace("FOREGROUND_CALL", effect?.hasForeground ? FOREGROUND_CALL.trim() : "")
  if (!effect) return COMPOSITE_HEAD + body
  return (
    COMPOSITE_HEAD +
    "\n// ── user effect (setEffect) ──\n" +
    effect.paramsDecl +
    "\n" +
    effect.wgsl +
    "\n" +
    body
  )
}

/** Kept for compatibility with existing imports (the base, no-effect shader). */
export const COMPOSITE_SHADER_WGSL = buildCompositeShader(null)
