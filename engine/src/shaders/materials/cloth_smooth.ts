// M_Smooth_Cloth — NPR toon + bevel + overlay-boosted emission mixed 10/90 against
// a plain Principled BSDF. Bump subtree is dead in the Blender graph.

import { NODES_WGSL } from "./nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "./common"

export const CLOTH_SMOOTH_SHADER_WGSL = /* wgsl */ `

${NODES_WGSL}
${COMMON_MATERIAL_PRELUDE_WGSL}

const CLOTH_SPECULAR: f32 = 0.8;
const CLOTH_ROUGHNESS: f32 = 0.5;
const CLOTH_TOON_EDGE: f32 = 0.2966;
const CLOTH_MIX04_MUL: f32 = 0.5;
const NPR_EMIT_STR: f32 = 18.200000762939453;
const NPR_MIX_SHADER_FAC: f32 = 0.8999999761581421;

@fragment fn fs(input: VertexOutput) -> FSOut {
  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);

  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  let tex_rgb = tex_s.rgb;
  let out_alpha = material.alpha * tex_s.a;
  if (out_alpha < 0.001) { discard; }

  // ═══ NPR STACK ═══
  let lum_shade = shader_to_rgb_diffuse(n, l, sun, amb, shadow);
  let ramp008 = ramp_constant_edge_aa(lum_shade, CLOTH_TOON_EDGE, vec4f(0,0,0,1), vec4f(1,1,1,1));
  let mix04_fac = math_multiply(ramp008.r, CLOTH_MIX04_MUL);

  let dark_tex = hue_sat_id(1.0, 0.19999998807907104, 1.0, tex_rgb);
  let mix04 = mix_blend(mix04_fac, dark_tex, tex_rgb);

  let bevel_z = clamp(n.y, 0.0, 1.0);
  let mix03 = mix_blend(bevel_z, mix04, dark_tex);

  let hue004 = hue_sat_id(0.800000011920929, 2.0, 1.0, mix03);
  let npr_rgb = mix_overlay(1.0, mix03, hue004);
  let npr_emission = npr_rgb * NPR_EMIT_STR;

  // ═══ PRINCIPLED BSDF ═══
  let principled_base = hue_sat_id(1.0, 0.800000011920929, 1.0, tex_rgb);
  let principled = eval_principled(
    PrincipledIn(principled_base, 0.0, CLOTH_SPECULAR, CLOTH_ROUGHNESS, 10.0, 0.0, 0.0),
    n, l, v, sun, amb, shadow
  );

  // MixShader.001: Shader=自发光.005, Shader_001=原理化BSDF, Fac=0.9
  let final_color = mix(npr_emission, principled, NPR_MIX_SHADER_FAC);

  var out: FSOut;
  out.color = vec4f(final_color, out_alpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}

`
