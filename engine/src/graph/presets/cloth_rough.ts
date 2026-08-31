// M_Rough_Cloth as a ShaderGraph — port of shaders/materials/cloth_rough.ts.
// NPR graph identical to M_Smooth_Cloth, but the noise bump subtree IS live on
// Principled.Normal (weave bump in rest space) and Roughness is raised to 0.8187.
// The tex_noise node hits the detail=2 peephole → tex_noise_d2.

import type { ShaderGraph } from "../schema"

export const CLOTH_ROUGH_GRAPH: ShaderGraph = {
  version: 1,
  name: "Rough Cloth",
  tags: ["cloth_rough"],
  nodes: [
    { id: "tex", type: "texture" },
    // Base colour: the texture times the PMX material's diffuse. MMD authors
    // tint per MATERIAL rather than per texel — an eyebrow sharing the face
    // atlas is painted once and coloured here — so the diffuse belongs in the
    // base of every look.
    { id: "mat_diffuse", type: "material_diffuse" },
    { id: "tex_base", type: "mix/multiply", inputs: { fac: 1.0 } },
    { id: "geo", type: "geometry" },
    { id: "str", type: "shader_to_rgb_diffuse" },
    { id: "ramp_008", type: "ramp_constant_aa", inputs: { edge: 0.2966 } },
    { id: "mix04_fac", type: "math/multiply", inputs: { b: 0.5 } },
    { id: "dark_tex", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.0, value: 0.19999998807907104, fac: 1.0 } },
    { id: "mix_004", type: "mix/blend" },
    { id: "sep_n", type: "separate_xyz" },
    { id: "bevel_clamp", type: "math/clamp01" },
    { id: "mix_003", type: "mix/blend" },
    { id: "hue_004", type: "hue_sat", inputs: { hue: 0.5, saturation: 0.800000011920929, value: 2.0, fac: 1.0 } },
    { id: "npr_overlay", type: "mix/overlay", inputs: { fac: 1.0 } },
    { id: "npr_emit", type: "emission", inputs: { strength: 18.200000762939453 } },
    { id: "noise", type: "tex_noise", inputs: { scale: 17.7 } },
    { id: "noise_ramp", type: "ramp_linear", inputs: { pos0: 0.0, pos1: 1.0 } },
    { id: "bump", type: "bump", inputs: { strength: 1.0 } },
    { id: "principled_base", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.0, value: 0.800000011920929, fac: 1.0 } },
    {
      id: "principled",
      type: "principled",
      inputs: { metallic: 0.0, specular_ior_level: 0.8, roughness: 0.8187, spec_clamp: 10.0, sheen_weight: 0.0, sheen_tint: 0.0 },
    },
    { id: "mix_shader_001", type: "mix_shader", inputs: { fac: 0.8999999761581421 } },
  ],
  links: [
    { from: { node: "str", socket: "value" }, to: { node: "ramp_008", socket: "fac" } },
    { from: { node: "ramp_008", socket: "fac_out" }, to: { node: "mix04_fac", socket: "a" } },
    { from: { node: "tex", socket: "color" }, to: { node: "tex_base", socket: "a" } },
    { from: { node: "mat_diffuse", socket: "color" }, to: { node: "tex_base", socket: "b" } },
    { from: { node: "tex_base", socket: "color" }, to: { node: "dark_tex", socket: "color" } },
    { from: { node: "mix04_fac", socket: "value" }, to: { node: "mix_004", socket: "fac" } },
    { from: { node: "dark_tex", socket: "color" }, to: { node: "mix_004", socket: "a" } },
    { from: { node: "tex_base", socket: "color" }, to: { node: "mix_004", socket: "b" } },
    { from: { node: "geo", socket: "normal" }, to: { node: "sep_n", socket: "vector" } },
    { from: { node: "sep_n", socket: "y" }, to: { node: "bevel_clamp", socket: "a" } },
    { from: { node: "bevel_clamp", socket: "value" }, to: { node: "mix_003", socket: "fac" } },
    { from: { node: "mix_004", socket: "color" }, to: { node: "mix_003", socket: "a" } },
    { from: { node: "dark_tex", socket: "color" }, to: { node: "mix_003", socket: "b" } },
    { from: { node: "mix_003", socket: "color" }, to: { node: "hue_004", socket: "color" } },
    { from: { node: "mix_003", socket: "color" }, to: { node: "npr_overlay", socket: "a" } },
    { from: { node: "hue_004", socket: "color" }, to: { node: "npr_overlay", socket: "b" } },
    { from: { node: "npr_overlay", socket: "color" }, to: { node: "npr_emit", socket: "color" } },
    { from: { node: "geo", socket: "rest_pos" }, to: { node: "noise", socket: "vector" } },
    { from: { node: "noise", socket: "value" }, to: { node: "noise_ramp", socket: "fac" } },
    { from: { node: "noise_ramp", socket: "fac_out" }, to: { node: "bump", socket: "height" } },
    { from: { node: "geo", socket: "normal" }, to: { node: "bump", socket: "normal" } },
    { from: { node: "tex_base", socket: "color" }, to: { node: "principled_base", socket: "color" } },
    { from: { node: "principled_base", socket: "color" }, to: { node: "principled", socket: "base_color" } },
    { from: { node: "bump", socket: "vector" }, to: { node: "principled", socket: "normal" } },
    { from: { node: "npr_emit", socket: "color" }, to: { node: "mix_shader_001", socket: "a" } },
    { from: { node: "principled", socket: "color" }, to: { node: "mix_shader_001", socket: "b" } },
  ],
  output: { node: "mix_shader_001", socket: "color" },
}
