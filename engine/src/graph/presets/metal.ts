// M_Metal as a ShaderGraph — port of shaders/materials/metal.ts.
// a metallic Lit (metallic 1, smoothness 0.45) with a
// reflection-coord voronoi driving base color (metallic sparkle), plus an NPR
// toon/overlay emission stack mixed at Fac=0.6967. The voronoi Color→ramp Fac
// link goes through Blender's implicit BT.601 conversion (color_to_value).

import type { ShaderGraph } from "../schema"

export const METAL_GRAPH: ShaderGraph = {
  version: 1,
  name: "Metal",
  tags: ["metal"],
  nodes: [
    { id: "tex", type: "texture" },
    // Base colour: the texture times the PMX material's diffuse. MMD authors
    // tint per MATERIAL rather than per texel — an eyebrow sharing the face
    // atlas is painted once and coloured here — so the diffuse belongs in the
    // base of every look.
    { id: "mat_diffuse", type: "material_diffuse" },
    { id: "tex_base", type: "mix/multiply", inputs: { fac: 1.0 } },
    { id: "geo", type: "geometry" },
    { id: "tex_tint", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.0, value: 0.800000011920929, fac: 1.0 } },
    { id: "str", type: "lambert" },
    { id: "ramp_008", type: "ramp_cardinal", inputs: { pos0: 0.2466, pos1: 0.3466 } },
    { id: "mix04_fac", type: "math/multiply", inputs: { b: 0.5 } },
    { id: "dark_tex", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.0, value: 0.19999998807907104, fac: 1.0 } },
    { id: "mix_004", type: "mix/blend" },
    { id: "hue_004", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.0, value: 2.0, fac: 1.0 } },
    { id: "npr_overlay", type: "mix/overlay", inputs: { fac: 1.0 } },
    { id: "npr_emit", type: "vector_math/scale", inputs: { scale: 8.100000381469727 } },
    { id: "voro_cross", type: "vect_cross", inputs: { b: [0, 1, 0] } },
    { id: "voro", type: "tex_voronoi/color", inputs: { scale: 4.3 } },
    { id: "voro_ramp", type: "ramp_linear", inputs: { pos0: 0.0, pos1: 1.0 } },
    { id: "hue_006", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.5, value: 1.2999999523162842, fac: 1.0 } },
    { id: "albedo", type: "mix/blend" },
    {
      id: "principled",
      type: "lit",
      inputs: { metallic: 1.0, smoothness: 0.4523, specular: 1.0 },
    },
    { id: "mix_shader_001", type: "mix/blend", inputs: { fac: 0.6967 } },
  ],
  links: [
    { from: { node: "tex", socket: "color" }, to: { node: "tex_base", socket: "a" } },
    { from: { node: "mat_diffuse", socket: "color" }, to: { node: "tex_base", socket: "b" } },
    { from: { node: "tex_base", socket: "color" }, to: { node: "tex_tint", socket: "color" } },
    { from: { node: "str", socket: "value" }, to: { node: "ramp_008", socket: "fac" } },
    { from: { node: "ramp_008", socket: "fac_out" }, to: { node: "mix04_fac", socket: "a" } },
    { from: { node: "tex_tint", socket: "color" }, to: { node: "dark_tex", socket: "color" } },
    { from: { node: "mix04_fac", socket: "value" }, to: { node: "mix_004", socket: "fac" } },
    { from: { node: "dark_tex", socket: "color" }, to: { node: "mix_004", socket: "a" } },
    { from: { node: "tex_tint", socket: "color" }, to: { node: "mix_004", socket: "b" } },
    { from: { node: "mix_004", socket: "color" }, to: { node: "hue_004", socket: "color" } },
    { from: { node: "mix_004", socket: "color" }, to: { node: "npr_overlay", socket: "a" } },
    { from: { node: "hue_004", socket: "color" }, to: { node: "npr_overlay", socket: "b" } },
    { from: { node: "npr_overlay", socket: "color" }, to: { node: "npr_emit", socket: "a" } },
    { from: { node: "geo", socket: "reflection" }, to: { node: "voro_cross", socket: "a" } },
    { from: { node: "voro_cross", socket: "vector" }, to: { node: "voro", socket: "vector" } },
    { from: { node: "voro", socket: "color" }, to: { node: "voro_ramp", socket: "fac" } },
    { from: { node: "tex_tint", socket: "color" }, to: { node: "hue_006", socket: "color" } },
    { from: { node: "voro_ramp", socket: "fac_out" }, to: { node: "albedo", socket: "fac" } },
    { from: { node: "voro_ramp", socket: "fac_out" }, to: { node: "albedo", socket: "a" } },
    { from: { node: "hue_006", socket: "color" }, to: { node: "albedo", socket: "b" } },
    { from: { node: "albedo", socket: "color" }, to: { node: "principled", socket: "base_color" } },
    { from: { node: "npr_emit", socket: "vector" }, to: { node: "mix_shader_001", socket: "a" } },
    { from: { node: "principled", socket: "color" }, to: { node: "mix_shader_001", socket: "b" } },
  ],
  output: { node: "mix_shader_001", socket: "color" },
}
