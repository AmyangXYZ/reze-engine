// M_Stockings — 仿深空之眼渲染预设v1.0_by_小绿毛猫 "M_Stockings". A bbox-gradient ×
// facing-rim mask drives a Mix Shader between an HSV-boosted emission and a sheen
// Principled BSDF. Wyman hashed-alpha testing replaces the graph's Alpha=0.95 (which
// would require TAA to hide the dither dots across every pixel).

import { NODES_WGSL } from "./nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "./common"

export const STOCKINGS_SHADER_WGSL = /* wgsl */ `

${NODES_WGSL}
${COMMON_MATERIAL_PRELUDE_WGSL}

// Principled params from dump (Alpha=0.95 intentionally dropped — see hash note below).
const STOCK_METALLIC: f32 = 0.1;
const STOCK_SPECULAR: f32 = 1.0;
const STOCK_ROUGHNESS: f32 = 0.5;
const STOCK_SHEEN: f32 = 0.7017999887466431;
const STOCK_SHEEN_TINT: f32 = 0.5;
// NPR mask ramps
const STOCK_RAMP002_P1: f32 = 0.9565;  // EASE [0→black, 0.9565→white]
const STOCK_RAMPFACE_P1: f32 = 0.5435; // EASE [0→black, 0.5435→white]
const STOCK_LW_BLEND: f32 = 0.4;       // Layer Weight Blend

// Wyman & McGuire "Hashed Alpha Testing" (2017) — object-space hash with derivative-aware
// pixel-scale selection, matches Blender EEVEE prepass_frag.glsl::hashed_alpha_threshold.
// Key property: dither pattern is stable in object space (doesn't swim) and stays at
// one-pixel frequency regardless of view distance, which makes it tolerable without TAA.
// The hash grid is fed the bind-pose restPos (not worldPos), so the dots stay pinned to
// the fabric even under root (センター) motion — worldPos would crawl across the surface.
// Since MMD skinning is rigid, length(dpdx(restPos)) == length(dpdx(worldPos)), so the
// pixel-scale selection is unaffected by the space change.
// Grid samples reuse the PCG _hash33 from nodes.ts (integer-lattice inputs, Metal-friendly)
// instead of a fract(sin()) hash — sin compiles to a slow transcendental on Apple's Metal
// backend and this runs per-fragment on every stockings pixel.
fn _hash3d_wm(a: vec3f) -> f32 {
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
  // CDF remap so discard-probability = (1 - alpha) uniformly across scale transitions
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

// Smoothstep-based EASE ramp (Blender VALTORGB EASE) — 2 stops, saturate+smoothstep between
fn ramp_ease_s(f: f32, p0: f32, p1: f32) -> f32 {
  let t = saturate((f - p0) / max(p1 - p0, 1e-6));
  return t * t * (3.0 - 2.0 * t);
}

@fragment fn fs(input: VertexOutput) -> FSOut {
  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);

  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  let tex_rgb = tex_s.rgb;
  // Alpha HASHED (Blender EEVEE "Hashed" blend mode) per preset author's note — self-overlap
  // on the stockings produces sort cracks under alpha blend. Wyman-style object-space hash +
  // depth-write is sort-independent. NOTE: Principled.Alpha=0.95 from the dump is DROPPED;
  // it relies on TAA to smooth the 5%-everywhere dither, and without TAA it shows as a
  // pervasive dot pattern. Hash now gates only on texture/material alpha.
  let combined_alpha = material.alpha * tex_s.a;
  if (combined_alpha < hashed_alpha_threshold(input.restPos)) { discard; }

  // ═══ NPR MASK ═══ TEX_COORD.Generated → Mapping(Rot=0,π/2,π/2, Loc=(1,1,1)) → Gradient.
  // The Blender mapping reduces to gradient.x = 1 - input.y (rot swaps axes, loc offsets).
  // We approximate Generated with UV since Y-up PMX has no object bbox in pipeline state.
  let gen_coord = vec3f(input.uv, 0.0);
  let mapped = mapping_point(gen_coord, vec3f(1.0), vec3f(0.0, 1.5708, 1.5708), vec3f(1.0));
  let gradient = tex_gradient_linear(mapped);

  // Ramp.001 LINEAR [0→black, 0.5→white, 1.0→black] — triangular peak at 0.5
  let ramp001 = 1.0 - abs(2.0 * gradient - 1.0);
  let ramp002 = ramp_ease_s(ramp001, 0.0, STOCK_RAMP002_P1);

  let facing = layer_weight_facing(STOCK_LW_BLEND, n, v);
  let ramp_face = ramp_ease_s(facing, 0.0, STOCK_RAMPFACE_P1);

  // Mix.001: MIX blend Fac=0.5, A=white, B=ramp_face
  let mix001 = mix(1.0, ramp_face, 0.5);
  // Mix: LIGHTEN blend Fac=0.5, A=mix001, B=ramp002
  let lighten = max(mix001, ramp002);
  let mask = mix(mix001, lighten, 0.5);

  // ═══ EMISSION SHADER ═══ Hue=0.5 (identity), Sat=1.0, Val=5.0 (5× brightness), Fac=1.
  let emission = hue_sat_id(1.0, 5.0, 1.0, tex_rgb);

  // ═══ PRINCIPLED BSDF with sheen ═══ metallic=0.1, sheen=0.7, sheen_tint=0.5.
  let principled = eval_principled(
    PrincipledIn(tex_rgb, STOCK_METALLIC, STOCK_SPECULAR, STOCK_ROUGHNESS, 1e30, STOCK_SHEEN, STOCK_SHEEN_TINT),
    n, l, v, sun, amb, shadow
  );

  // MIX SHADER: Shader=Emission, Shader_001=Principled, Fac=mask
  let final_color = mix(emission, principled, mask);

  var out: FSOut;
  out.color = vec4f(final_color, 1.0);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}

`
