// Per-slot WGSL templates. The graph computes only `final_color`; everything the slot
// owns — texture sample, MMD alpha semantics, discard, lighting context, FSOut/mask
// writes, and built-in pass effects like hair's over-eyes stencil variant — lives here,
// always on, graph-invisible. Mirrors the hand-written material shaders line-for-line.

import { NODES_WGSL } from "../shaders/materials/nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "../shaders/materials/common"
import type { MaterialPreset } from "../engine"

export type SlotTemplate = {
  /** Module-scope declarations (pipeline-override constants, helper fns). */
  decls: string
  /** Optional replacement for the standard fs() prelude (custom alpha semantics). */
  prelude?: string
  /** Tail of fs(): consumes `final_color` + template locals, writes FSOut. */
  epilogue: string
}

const DEFAULT_EPILOGUE = `  var out: FSOut;
  out.color = vec4f(final_color, alpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
`

const DEFAULT_TEMPLATE: SlotTemplate = { decls: "", epilogue: DEFAULT_EPILOGUE }

// Hair's built-in over-eyes effect (see hair.ts): the engine compiles two pipeline
// variants from the same module — normal opaque hair, and a stencil-matched re-draw
// at 25% alpha so eyes read through the silhouette. Always on for the hair slot.
const HAIR_TEMPLATE: SlotTemplate = {
  decls: `override IS_OVER_EYES: bool = false;

`,
  epilogue: `  var outAlpha = alpha;
  if (IS_OVER_EYES) { outAlpha = alpha * 0.25; }

  var out: FSOut;
  out.color = vec4f(final_color, outAlpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
`,
}

// Stockings' built-in alpha behavior (see stockings.ts): Wyman & McGuire hashed
// alpha testing on bind-pose restPos replaces the standard threshold discard —
// sort-independent through self-overlap, dither pinned to the fabric. The graph
// only computes final_color; the hash gate and the alpha=1 output are slot-owned.
const STOCKINGS_TEMPLATE: SlotTemplate = {
  decls: `fn _hash3d_wm(a: vec3f) -> f32 {
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

`,
  prelude: `@fragment fn fs(input: VertexOutput) -> FSOut {
  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  // Hashed alpha test (EEVEE "Hashed" blend) instead of the standard threshold.
  let alpha = material.alpha * tex_s.a;
  if (alpha < hashed_alpha_threshold(input.restPos)) { discard; }

  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);
  let tex_color = tex_s.rgb;

`,
  epilogue: `  var out: FSOut;
  out.color = vec4f(final_color, 1.0);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
`,
}

// Eye's built-in rear-view gate (see eye.ts): open-shell PMX heads don't occlude
// the eye from behind, so it would draw (and stamp the see-through stencil) through
// the back of the head. Gate by camera-vs-face hemisphere via the 頭 bone's skinning
// matrix. Discarding here drops color, depth, and the stencil stamp together. The
// stencil-stamp pipeline state + front-face cull are slot-owned in createSlotPipeline;
// the graph only computes the eye's shading (Principled + emission).
const EYE_TEMPLATE: SlotTemplate = {
  decls: "",
  prelude: `@fragment fn fs(input: VertexOutput) -> FSOut {
  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  let alpha = material.alpha * tex_s.a;
  if (alpha < 0.001) { discard; }

  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);

  if (material.headBoneIndex >= 0.0) {
    let hm = skinMats[u32(material.headBoneIndex)];
    let faceDir = -normalize(hm[2].xyz);
    if (dot(faceDir, v) < -0.15) { discard; }
  }

  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);
  let tex_color = tex_s.rgb;

`,
  epilogue: DEFAULT_EPILOGUE,
}

export const SLOT_TEMPLATES: Partial<Record<MaterialPreset, SlotTemplate>> = {
  hair: HAIR_TEMPLATE,
  stockings: STOCKINGS_TEMPLATE,
  eye: EYE_TEMPLATE,
}

export function slotTemplate(slot: MaterialPreset): SlotTemplate {
  return SLOT_TEMPLATES[slot] ?? DEFAULT_TEMPLATE
}

// Adjust-tier uniforms — fixed 16-vec4f block (256 B) so the single shared material
// bind group layout serves every graph; non-graph presets bind a zero buffer.
export const STYLE_UNIFORMS_WGSL = `struct StyleUniforms { p: array<vec4f, 16> };
@group(2) @binding(4) var<uniform> style: StyleUniforms;

`

// fs() prelude — identical to the hand-written materials so template locals keep the
// exact names the registry's context nodes and emit functions reference.
const FS_PRELUDE = `@fragment fn fs(input: VertexOutput) -> FSOut {
  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  // MMD alpha semantics: material alpha × texture alpha (hair/lace textures cut
  // their shapes in the alpha channel).
  let alpha = material.alpha * tex_s.a;
  if (alpha < 0.001) { discard; }

  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);
  let tex_color = tex_s.rgb;

`

export function assembleModule(slot: MaterialPreset, fsBody: string, includeStyleUniforms: boolean): string {
  const tmpl = slotTemplate(slot)
  return (
    NODES_WGSL +
    COMMON_MATERIAL_PRELUDE_WGSL +
    (includeStyleUniforms ? STYLE_UNIFORMS_WGSL : "") +
    tmpl.decls +
    (tmpl.prelude ?? FS_PRELUDE) +
    fsBody +
    "\n" +
    tmpl.epilogue +
    "}\n"
  )
}
