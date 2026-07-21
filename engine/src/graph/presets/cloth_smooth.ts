// M_Smooth_Cloth as a StyleGraph — port of shaders/materials/cloth_smooth.ts.
// NPR toon + bevel + overlay-boosted emission (18.2×) mixed 10/90 against a plain
// Principled BSDF. The Blender graph's dead bump subtree is omitted (as in the hand
// port). hue_sat nodes with hue=0.5 compile to the hue_sat_id specialization.

import type { StyleGraph } from "../schema"

export const CLOTH_SMOOTH_GRAPH: StyleGraph = {
  version: 1,
  name: "Smooth Cloth",
  slot: "cloth_smooth",
  nodes: [
    { id: "tex", type: "texture" },
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
    { id: "principled_base", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.0, value: 0.800000011920929, fac: 1.0 } },
    {
      id: "principled",
      type: "principled",
      inputs: { metallic: 0.0, specular: 0.8, roughness: 0.5, spec_clamp: 10.0, sheen: 0.0, sheen_tint: 0.0 },
    },
    { id: "mix_shader_001", type: "mix_shader", inputs: { fac: 0.8999999761581421 } },
  ],
  links: [
    { from: { node: "str", socket: "value" }, to: { node: "ramp_008", socket: "fac" } },
    { from: { node: "ramp_008", socket: "fac_out" }, to: { node: "mix04_fac", socket: "a" } },
    { from: { node: "tex", socket: "color" }, to: { node: "dark_tex", socket: "color" } },
    { from: { node: "mix04_fac", socket: "value" }, to: { node: "mix_004", socket: "fac" } },
    { from: { node: "dark_tex", socket: "color" }, to: { node: "mix_004", socket: "a" } },
    { from: { node: "tex", socket: "color" }, to: { node: "mix_004", socket: "b" } },
    { from: { node: "geo", socket: "normal" }, to: { node: "sep_n", socket: "vector" } },
    { from: { node: "sep_n", socket: "y" }, to: { node: "bevel_clamp", socket: "a" } },
    { from: { node: "bevel_clamp", socket: "value" }, to: { node: "mix_003", socket: "fac" } },
    { from: { node: "mix_004", socket: "color" }, to: { node: "mix_003", socket: "a" } },
    { from: { node: "dark_tex", socket: "color" }, to: { node: "mix_003", socket: "b" } },
    { from: { node: "mix_003", socket: "color" }, to: { node: "hue_004", socket: "color" } },
    { from: { node: "mix_003", socket: "color" }, to: { node: "npr_overlay", socket: "a" } },
    { from: { node: "hue_004", socket: "color" }, to: { node: "npr_overlay", socket: "b" } },
    { from: { node: "npr_overlay", socket: "color" }, to: { node: "npr_emit", socket: "color" } },
    { from: { node: "tex", socket: "color" }, to: { node: "principled_base", socket: "color" } },
    { from: { node: "principled_base", socket: "color" }, to: { node: "principled", socket: "base" } },
    { from: { node: "npr_emit", socket: "color" }, to: { node: "mix_shader_001", socket: "a" } },
    { from: { node: "principled", socket: "color" }, to: { node: "mix_shader_001", socket: "b" } },
  ],
  output: { node: "mix_shader_001", socket: "color" },
}
