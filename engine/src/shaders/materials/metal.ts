// M_Metal — Metallic Principled (Metallic=1.0, Specular=1.0, Roughness=0.3) with a
// reflection-coord Voronoi pattern driving base color for metallic sparkle, plus an
// NPR toon/overlay emission stack mixed at MixShader Fac=0.6967.
//
// Graph's base color chain: 纹理坐标.Reflection → 矢量运算.007(CROSS, Vec2=(0,1,0)) →
// 沃罗诺伊纹理(F1, Color out) → 颜色渐变(linear) → 混合.005. The dumper did not capture
// the VectorMath op — CROSS is assumed based on the hardcoded (0,1,0) Vector_001
// constant (MULTIPLY would zero X/Z producing 1D bands; CROSS produces horizontal ring
// patterns consistent with metallic anisotropy).

import { NODES_WGSL } from "./nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "./common"

export const METAL_SHADER_WGSL = /* wgsl */ `

${NODES_WGSL}
${COMMON_MATERIAL_PRELUDE_WGSL}

const METAL_SPECULAR: f32 = 1.0;
const METAL_METALLIC: f32 = 1.0;
const METAL_ROUGHNESS: f32 = 0.3;
const METAL_TOON_EDGE: f32 = 0.2966;
const METAL_MIX04_MUL: f32 = 0.5;
const METAL_EMIT_STR: f32 = 8.100000381469727;
const METAL_MIX_SHADER_FAC: f32 = 0.6967;
const METAL_VORONOI_SCALE: f32 = 4.3;

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
  let tex_tint = hue_sat_id(1.0, 0.800000011920929, 1.0, tex_rgb);
  let lum_shade = shader_to_rgb_diffuse(n, l, sun, amb, shadow);
  let ramp008 = ramp_constant_edge_aa(lum_shade, METAL_TOON_EDGE, vec4f(0,0,0,1), vec4f(1,1,1,1));
  let mix04_fac = math_multiply(ramp008.r, METAL_MIX04_MUL);

  let dark_tex = hue_sat_id(1.0, 0.19999998807907104, 1.0, tex_tint);
  let mix04 = mix_blend(mix04_fac, dark_tex, tex_tint);

  let hue004 = hue_sat_id(1.0, 2.0, 1.0, mix04);
  let npr_rgb = mix_overlay(1.0, mix04, hue004);
  let npr_emission = npr_rgb * METAL_EMIT_STR;

  // ═══ PRINCIPLED BSDF (metallic=1, voronoi-driven base) ═══
  // Reflection-coord Voronoi produces the metallic sparkle variation.
  // VALTORGB takes Color → Fac via Blender's BT.601 implicit color_to_value.
  let refl_dir = reflect(-v, n);
  let voro_input = cross(refl_dir, vec3f(0.0, 1.0, 0.0));
  let voro_rgb = tex_voronoi_color(voro_input, METAL_VORONOI_SCALE);
  let voro_scalar = color_to_value(voro_rgb);
  let voro_ramp = ramp_linear(voro_scalar, 0.0, vec4f(0,0,0,1), 1.0, vec4f(1,1,1,1)).r;
  let hue006 = hue_sat_id(1.5, 1.2999999523162842, 1.0, tex_tint);
  let albedo = mix_blend(voro_ramp, vec3f(voro_ramp), hue006);

  // metallic=1 collapses f0 = mix(dielectric, albedo, 1) = albedo; diffuse_weight = 0.
  let principled = eval_principled(
    PrincipledIn(albedo, METAL_METALLIC, METAL_SPECULAR, METAL_ROUGHNESS, 1e30, 0.0, 0.0),
    n, l, v, sun, amb, shadow
  );

  let final_color = mix(npr_emission, principled, METAL_MIX_SHADER_FAC);

  var out: FSOut;
  out.color = vec4f(final_color, out_alpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}

`
