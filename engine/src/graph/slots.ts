// fs() shell assembly. The graph computes only `final_color`; everything around it —
// texture sample, alpha discard, lighting context, built-in gates, and the FSOut/mask
// write — is composed here from two orthogonal, engine-owned axes:
//   RenderClass ("auto" | "eye" | "hair")  — stencil/cull/draw-order + eye rear-gate +
//                                             hair over-eyes alpha. See render-class.ts.
//   AlphaMode   ("opaque" | "hashed")       — standard threshold discard vs Wyman hashed.
// A style group supplies both; this file turns them into the WGSL shell around fsBody.
// See docs/style-groups-spec.md §5, §7.

import { NODES_WGSL } from "../shaders/materials/nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL, commonFsOutWgsl } from "../shaders/materials/common"
import { sceneIdWriteWgsl } from "../shaders/passes/scene-contract"
import type { AlphaMode, RenderClass } from "./render-class"

// ── Module-scope declarations ──
// hair: the over-eyes pipeline-override constant (a second pipeline is compiled with
// IS_OVER_EYES=1). hashed: the Wyman & McGuire object-space hashed-alpha helpers.
const HAIR_OVER_EYES_DECL = `override IS_OVER_EYES: bool = false;

`

const HASHED_ALPHA_DECLS = `fn _hash3d_wm(a: vec3f) -> f32 {
  return _hash33(a).x * 0.5 + 0.5;
}
fn hashed_alpha_threshold(co: vec3f) -> f32 {
  let alphaHashScale: f32 = 1.0;
  let max_deriv = max(length(dpdx(co)), length(dpdy(co)));
  let pix_scale = 1.0 / max(alphaHashScale * max_deriv, 1e-6);
  let pix_scale_log = log2(pix_scale);
  let px_lo = exp2(floor(pix_scale_log));
  let px_hi = exp2(ceil(pix_scale_log));
  let a_lo = _hash3d_wm(floor(px_lo * co));
  let a_hi = _hash3d_wm(floor(px_hi * co));
  let fac = fract(pix_scale_log);
  let x = mix(a_lo, a_hi, fac);
  let a = min(fac, 1.0 - fac);
  let one_a = 1.0 - a;
  let denom = 1.0 / max(2.0 * a * one_a, 1e-6);
  let one_x = 1.0 - x;
  let case_lo = (x * x) * denom;
  let case_mid = (x - 0.5 * a) / max(one_a, 1e-6);
  let case_hi = 1.0 - (one_x * one_x) * denom;
  var threshold = select(case_hi, select(case_lo, case_mid, x >= a), x < one_a);
  return clamp(threshold, 1e-6, 1.0);
}

`

// Eye's built-in rear-view gate (open-shell PMX heads don't occlude the eye from behind,
// so gate by camera-vs-face hemisphere via the 頭 bone; discard drops color + depth +
// the stencil stamp together). Injected between the view vectors and the lighting setup.
const EYE_REAR_GATE = `
  if (material.headBoneIndex >= 0.0) {
    let hm = skinMats[u32(material.headBoneIndex)];
    let faceDir = -normalize(hm[2].xyz);
    if (dot(faceDir, v) < -0.15) { discard; }
  }
`

// ── Adjust-tier uniforms — fixed 16-vec4f block (256 B) shared by the single material
// bind group layout; non-graph draws bind a zero buffer. ──
export const STYLE_UNIFORMS_WGSL = `struct StyleUniforms { p: array<vec4f, 16> };
@group(2) @binding(4) var<uniform> style: StyleUniforms;

`

function decls(renderClass: RenderClass, alphaMode: AlphaMode): string {
  return (alphaMode === "hashed" ? HASHED_ALPHA_DECLS : "") + (renderClass === "hair" ? HAIR_OVER_EYES_DECL : "")
}

// fs() header up to and including the graph body's context locals. Composed so the
// hand-written material shaders' local names (tex_color, n, v, l, sun, amb, shadow) are
// preserved exactly — the registry's context nodes and emit functions reference them.
function prelude(renderClass: RenderClass, alphaMode: AlphaMode): string {
  const discard =
    alphaMode === "hashed"
      ? "  if (alpha < hashed_alpha_threshold(input.restPos)) { discard; }"
      : "  if (alpha < 0.001) { discard; }"
  const gate = renderClass === "eye" ? EYE_REAR_GATE : ""
  // Double-sided shading, winding-independent: a normal pointing away from the
  // camera means we're seeing the surface's other side — flip it. Only genuinely
  // camera-averted fragments change (front surfaces have dot(n,v) > 0 and are
  // untouched), unlike a front_facing test, which PMX's clockwise winding
  // inverts. Fixes inner skirt linings (裙子白1-style flipped-normal copies)
  // shading near-black through sheer outer layers. Eye is exempt: it FRONT-culls
  // by design, so its visible fragments are back faces with intended normals.
  const flip =
    renderClass === "eye" ? "" : "\n  n = select(-n, n, dot(n, v) >= 0.0);"
  return `@fragment fn fs(input: VertexOutput) -> FSOut {
  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  // MMD alpha semantics: material alpha × texture alpha.
  let alpha = material.alpha * tex_s.a;
${discard}

  var n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);${flip}
${gate}
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);
  let tex_color = tex_s.rgb;

`
}

// Tail of fs(): consumes `final_color` + locals, writes FSOut. hashed forces output
// alpha to 1 (the discard already did the cutout); hair scales alpha for the over-eyes
// pass when IS_OVER_EYES is compiled true.
function epilogue(renderClass: RenderClass, alphaMode: AlphaMode): string {
  const alphaBase = alphaMode === "hashed" ? "1.0" : "alpha"
  // Empty while ids are off, so the epilogue is exactly what it was. The values
  // ride in the per-draw material uniform (see MaterialUniforms), which is what
  // keeps this working through the indirect-draw path.
  const ID_WRITE = sceneIdWriteWgsl("out", "u32(material.materialId)", "u32(material.objectId)")
  // The positional lights, ADDED to whatever the graph decided and tinted by
  // the surface's own texture colour — a red lamp on a blue dress must not wash
  // it toward white. Deliberately outside the graph: this layer does not
  // re-ramp, because two ramped terminators crossing a cheek read as plastic.
  //
  // With no lights rzLightsDiffuse returns exactly zero and its loop never
  // runs, so every scene that declares none renders bit-for-bit as it did
  // before lights existed. That is the property this whole feature is gated on.
  // Modulated by `albedo`, a named local that TODAY is exactly tex_color — the
  // raw texture. That is an approximation with a known failure: a graph that
  // recolours a surface is lit by a colour it no longer shows. The local exists
  // so the fix has an address — when graphs gain an optional albedo output
  // (material-track era), it lands here and every light is corrected at once,
  // instead of a hunt through the epilogue's string templates.
  const LIT = ` + rzLightsDiffuse(input.worldPos, n) * albedo`
  const ALBEDO = `  let albedo = tex_color;
`
  if (renderClass === "hair") {
    return `${ALBEDO}  var outAlpha = ${alphaBase};
  if (IS_OVER_EYES) { outAlpha = ${alphaBase} * 0.25; }

  var out: FSOut;
  out.color = vec4f(final_color${LIT}, outAlpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
${ID_WRITE}  return out;
`
  }
  return `${ALBEDO}  var out: FSOut;
  out.color = vec4f(final_color${LIT}, ${alphaBase});
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
${ID_WRITE}  return out;
`
}

/**
 * Assemble a full WGSL module: shared node library + bindings/skinning prelude, optional
 * StyleUniforms, render-class/alpha-mode declarations, the composed fs() shell, and the
 * graph's node body.
 */
export function assembleModule(
  renderClass: RenderClass,
  alphaMode: AlphaMode,
  fsBody: string,
  includeStyleUniforms: boolean,
): string {
  return (
    NODES_WGSL +
    COMMON_MATERIAL_PRELUDE_WGSL +
    // Exactly where it sat inside the prelude constant. It moved out because
    // the id attachment is probed at init and this module is assembled per
    // compile, which is late enough to know the answer.
    commonFsOutWgsl() +
    (includeStyleUniforms ? STYLE_UNIFORMS_WGSL : "") +
    decls(renderClass, alphaMode) +
    prelude(renderClass, alphaMode) +
    fsBody +
    "\n" +
    epilogue(renderClass, alphaMode) +
    "}\n"
  )
}
