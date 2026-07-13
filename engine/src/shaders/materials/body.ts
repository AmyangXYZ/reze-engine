// M_Body — 仿深空之眼渲染预设v1.0_by_小绿毛猫 "M_Body". Toon + warm rim + rim1/rim2
// stack mixed 50/50 against a Principled BSDF with noise-bumped normal.

import { NODES_WGSL } from "./nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "./common"

export const BODY_SHADER_WGSL = /* wgsl */ `

${NODES_WGSL}
${COMMON_MATERIAL_PRELUDE_WGSL}

const BODY_ROUGHNESS: f32 = 0.3;
const BODY_SPECULAR: f32 = 0.5;
const BODY_MIX_NPR: f32 = 0.5;
const BODY_SPEC_CLAMP: f32 = 10.0;
const BODY_RIM2_LAYER_BLEND: f32 = 0.20000000298023224;
const BODY_RIM2_POW: f32 = 1.4300000667572021;
const BODY_RIM2_BG: vec3f = vec3f(1.0, 0.4303792119026184, 0.3315804898738861);
const BODY_WARM_STR: f32 = 0.30000001192092896;

// smoothstep-based ramp: t*t*(3-2*t) between two color stops
fn ramp_ease(f: f32, p0: f32, c0: vec4f, p1: f32, c1: vec4f) -> vec4f {
  let t = saturate((f - p0) / max(p1 - p0, 1e-6));
  let ss = t * t * (3.0 - 2.0 * t);
  return mix(c0, c1, ss);
}

@fragment fn fs(input: VertexOutput) -> FSOut {
  let alpha = material.alpha;
  if (alpha < 0.001) { discard; }

  let n = normalize(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);

  let tex_color = textureSample(diffuseTexture, diffuseSampler, input.uv).rgb;

  // ═══ NPR STACK ═══
  let ndotl_raw = shader_to_rgb_diffuse(n, l, sun, amb, shadow);
  let toon = ramp_constant(ndotl_raw, 0.0, vec4f(0,0,0,1), 0.2966, vec4f(1,1,1,1)).r;

  let shadow_tint = hue_sat_id(2.0, 0.3499999940395355, 1.0, tex_color);
  let lit_tint = hue_sat_id(1.5, 1.0, 1.0, tex_color);
  let toon_color = mix_blend(toon, shadow_tint, lit_tint);
  let bc = bright_contrast(toon_color, 0.1, 0.2);

  let emission3 = bc * 4.0;

  let warm_input = clamp(toon + 0.5, 0.0, 1.0);
  let warm_color = ramp_cardinal(warm_input, 0.2409,
    vec4f(0.2426, 0.068, 0.0588, 1.0), 0.4663,
    vec4f(0.6677, 0.5024, 0.5126, 1.0)).rgb;
  let warm_emission = warm_color * BODY_WARM_STR;

  let rim1_str = fresnel(2.0, n, v) * layer_weight_facing(0.24000005424022675, n, v);
  let rim1 = vec3f(0.984157919883728, 0.6110184788703918, 0.5736401677131653) * rim1_str;

  let facing_raw = layer_weight_facing(BODY_RIM2_LAYER_BLEND, n, v);
  let facing_pow = math_power(facing_raw, BODY_RIM2_POW);
  let rim2_fac = ramp_ease(facing_pow, 0.0, vec4f(0,0,0,1), 0.5052, vec4f(1,1,1,1)).r;
  let rim2_mixed = mix(emission3, BODY_RIM2_BG, rim2_fac);

  let npr_stack = rim1 + rim2_mixed + warm_emission;

  // ═══ PRINCIPLED BSDF with noise bump ═══
  // Mapping loc=rot=0 in the Blender graph folds to a plain scale multiply.
  // Height in rest space (restPos) so the bump rides with the skin; geometric basis stays
  // worldPos so the perturbed normal matches the world-space n (see face.ts note).
  let noise_val = tex_noise_d2(input.restPos * vec3f(1.0, 1.0, 1.5), 1.0);
  let noise_ramp = ramp_linear(noise_val, 0.0, vec4f(0,0,0,1), 1.0, vec4f(1,1,1,1)).r;
  let bumped_n = bump_lh(0.324644535779953, noise_ramp, n, input.worldPos);

  let principled_base = mix_blend(noise_ramp, bc, vec3f(0.6831911206245422, 0.19474034011363983, 0.13732507824897766));
  let p_emission = bc * 0.2;

  let principled = eval_principled(
    PrincipledIn(principled_base, 0.0, BODY_SPECULAR, BODY_ROUGHNESS, BODY_SPEC_CLAMP, 0.0, 0.0),
    bumped_n, l, v, sun, amb, shadow
  ) + p_emission;

  let final_color = mix(npr_stack, principled, BODY_MIX_NPR);

  var out: FSOut;
  out.color = vec4f(final_color, alpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}

`
