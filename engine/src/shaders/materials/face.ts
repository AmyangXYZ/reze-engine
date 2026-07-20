// M_Face — 仿深空之眼渲染预设v1.0_by_小绿毛猫 "M_Face". Toon + warm rim + dual fresnel
// rim + BT.601 bright-tex gate, mixed 50/50 against a Principled BSDF with noise bump.

import { NODES_WGSL } from "./nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "./common"

export const FACE_SHADER_WGSL = /* wgsl */ `

${NODES_WGSL}
${COMMON_MATERIAL_PRELUDE_WGSL}

const FACE_SPECULAR: f32 = 0.5;
const FACE_ROUGHNESS: f32 = 0.3;
const FACE_MIX_NPR: f32 = 0.5;
const FACE_SPEC_CLAMP: f32 = 10.0;
const FACE_RIM2_POW: f32 = 0.6300000548362732;
const FACE_RIM2_BG: vec3f = vec3f(1.0, 0.4684903025627136, 0.3698573112487793);
const FACE_WARM_STR: f32 = 0.30000001192092896;
const FACE_BRIGHT_TEX_THRESH: f32 = 0.9300000071525574;

@fragment fn fs(input: VertexOutput) -> FSOut {
  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  // MMD alpha semantics: material alpha × texture alpha. Hair/lace/accessory
  // textures cut their shapes in the alpha channel — ignoring it renders each
  // card's full quad with the texture's padding color (white shimmer on dark
  // hair for models whose textures pad with white).
  let alpha = material.alpha * tex_s.a;
  if (alpha < 0.001) { discard; }

  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);

  let tex_color = tex_s.rgb;

  // ═══ NPR STACK ═══
  let ndotl_raw = shader_to_rgb_diffuse(n, l, sun, amb, shadow);
  // ramp_constant_edge_aa: avoids binary fac shimmer on terminator (fwidth + smoothstep).
  let toon = ramp_constant_edge_aa(ndotl_raw, 0.2966, vec4f(0,0,0,1), vec4f(1,1,1,1)).r;

  let shadow_tint = hue_sat(0.46000000834465027, 2.0, 0.3499999940395355, 1.0, tex_color);
  let lit_tint = hue_sat(0.46000000834465027, 1.600000023841858, 1.5, 1.0, tex_color);
  let toon_color = mix_blend(toon, shadow_tint, lit_tint);
  let bc = bright_contrast(toon_color, 0.1, 0.2);

  let emission3 = bc * 2.5;

  let warm_input = clamp(toon * 0.5 + 0.5, 0.0, 1.0);
  let warm_color = ramp_cardinal(warm_input, 0.2409,
    vec4f(0.2426, 0.068, 0.0588, 1.0), 0.4663,
    vec4f(0.6677, 0.5024, 0.5126, 1.0)).rgb;
  let warm_emission = warm_color * FACE_WARM_STR;

  let rim1_str = fresnel(2.0, n, v) * layer_weight_facing(0.24, n, v);
  let rim1 = vec3f(0.984157919883728, 0.6110184788703918, 0.5736401677131653) * rim1_str;

  let rim2_raw = fresnel(1.45, n, v) * layer_weight_fresnel(0.61, n, v);
  let rim2_fac = math_power(rim2_raw, FACE_RIM2_POW);
  let rim2_mixed = mix(emission3, FACE_RIM2_BG, rim2_fac);

  // Blender implicitly converts Color → Float via BT.601 grayscale when a color output
  // feeds a Math node's Value input. Using tex_color.r instead fires on R-dominant skin
  // and produces firefly speckles on near-white R pixels — color_to_value matches the
  // Blender socket semantic and only gates genuinely near-white painted features.
  let tex_gate = math_greater_than(color_to_value(tex_color), FACE_BRIGHT_TEX_THRESH);
  let bright_emit = vec3f(tex_gate) * 3.0;

  let npr_stack = rim1 + rim2_mixed + bright_emit + warm_emission;

  // ═══ PRINCIPLED BSDF with noise bump ═══
  // Height sampled in rest space (restPos) so the micro-bump doesn't swim under motion;
  // bump_lh's geometric basis stays in worldPos so the perturbed normal lands in the same
  // world frame as n (screen-space height gradient still pairs correctly with it).
  let noise_val = tex_noise_d2(input.restPos * vec3f(1.0, 1.0, 1.5), 1.0);
  let noise_ramp = ramp_linear(noise_val, 0.0, vec4f(0,0,0,1), 1.0, vec4f(1,1,1,1)).r;
  let bumped_n = bump_lh(0.324644535779953, noise_ramp, n, input.worldPos);

  let principled_base = mix_blend(noise_ramp, bc, vec3f(0.6832, 0.1947, 0.1373));
  let p_emission = bc * 0.2;

  let principled = eval_principled(
    PrincipledIn(principled_base, 0.0, FACE_SPECULAR, FACE_ROUGHNESS, FACE_SPEC_CLAMP, 0.0, 0.0),
    bumped_n, l, v, sun, amb, shadow
  ) + p_emission;

  let final_color = mix(npr_stack, principled, FACE_MIX_NPR);

  var out: FSOut;
  out.color = vec4f(final_color, alpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}

`
