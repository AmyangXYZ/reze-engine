// M_Hair — 仿深空之眼渲染预设v1.0_by_小绿毛猫 "M_Hair". Toon + fresnel rim + bevel +
// bright-tex gate, mixed 80/20 NPR/PBR. MixShader.001 Fac=0.2 keeps Principled subtle.

import { NODES_WGSL } from "./nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "./common"

export const HAIR_SHADER_WGSL = /* wgsl */ `

${NODES_WGSL}
${COMMON_MATERIAL_PRELUDE_WGSL}

// Pipeline-override: the engine compiles two variants — the normal opaque hair pipeline
// (IS_OVER_EYES=false) and a second pipeline that re-draws hair fragments stencil-matched
// against the eye stamp with 25% alpha so eyes read through the hair silhouette. Resolved
// at pipeline-compile time; the dead branch is dropped by the shader compiler.
override IS_OVER_EYES: bool = false;

const HAIR_SPECULAR: f32 = 1.0;
const HAIR_ROUGHNESS: f32 = 0.3;
// EEVEE Light Clamp equivalent (same value as body/face/cloth_rough). Hair
// is the firefly-prone material: physics-animated with the noisiest blended
// normals, so grazing NV spikes the GGX term and flashes white through
// bloom without this.
const HAIR_SPEC_CLAMP: f32 = 10.0;
const HAIR_TEX_GATE_THRESH: f32 = 0.15000000596046448;
const HAIR_RIM2_POW: f32 = 0.6300000548362732;
const HAIR_MIX_BG: vec3f = vec3f(0.1673291176557541);
const HAIR_MIX_NPR: f32 = 0.2;

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
  let hue_sat_shadow = hue_sat_id(1.2, 0.5, 1.0, tex_color);
  let hue_sat_002 = hue_sat(0.48, 1.2, 0.7, 1.0, hue_sat_shadow);
  let hue_sat_001 = hue_sat_id(1.5, 1.0, 1.0, tex_color);

  let ndotl_raw = shader_to_rgb_diffuse(n, l, sun, amb, shadow);
  let ramp_008 = ramp_constant(ndotl_raw, 0.0, vec4f(0,0,0,1), 0.2966, vec4f(1,1,1,1)).r;

  let mix_004 = mix_blend(ramp_008, hue_sat_002, hue_sat_001);
  let bc = bright_contrast(mix_004, 0.1, 0.2);

  let bevel_z = clamp(n.y, 0.0, 1.0);
  let mix_003 = mix_blend(bevel_z, bc, hue_sat_002);

  let rim2_raw = fresnel(1.45, n, v) * layer_weight_fresnel(0.61, n, v);
  let rim2_fac = math_power(rim2_raw, HAIR_RIM2_POW);
  let mix_shader_002 = mix(mix_003, HAIR_MIX_BG, rim2_fac);

  // GREATER_THAN on a color input uses BT.601 luminance — same socket-semantic fix as face.ts.
  let tex_gate = math_greater_than(color_to_value(tex_color), HAIR_TEX_GATE_THRESH);
  let gate_emit = vec3f(tex_gate) * 0.1;

  let npr_stack = mix_shader_002 + gate_emit;

  // ═══ PRINCIPLED BSDF ═══
  // Graph has a noise→normal_map bump (Strength=0.1) on Principled.Normal, but MixShader.001
  // weights Principled at only 0.2 — the bumped spec × that weight is imperceptible, so we
  // drop the subtree and keep plain n (saves a tex_noise + bump_lh per hair fragment).
  let principled = eval_principled(
    PrincipledIn(bc, 0.0, HAIR_SPECULAR, HAIR_ROUGHNESS, HAIR_SPEC_CLAMP, 0.0, 0.0),
    n, l, v, sun, amb, shadow
  );

  let final_color = mix(npr_stack, principled, HAIR_MIX_NPR);

  var outAlpha = alpha;
  if (IS_OVER_EYES) { outAlpha = alpha * 0.25; }

  var out: FSOut;
  out.color = vec4f(final_color, outAlpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}

`
