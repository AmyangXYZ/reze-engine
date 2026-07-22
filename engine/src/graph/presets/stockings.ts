// M_Stockings as a StyleGraph — port of shaders/materials/stockings.ts.
// A bbox-gradient × facing-rim mask drives a Mix Shader between an HSV-boosted
// emission (val 5×) and a sheen Principled. The hashed-alpha discard and the
// alpha=1 output are slot-owned (see STOCKINGS_TEMPLATE in slots.ts) — the graph
// computes only the radiance. Blender's Generated coord is approximated with UV,
// as in the hand port. The grayscale mask feeds Mix Shader Fac through Blender's
// implicit BT.601 color→float conversion (equal-component vector → same scalar).

import type { StyleGraph } from "../schema"

export const STOCKINGS_GRAPH: StyleGraph = {
  version: 1,
  name: "Stockings",
  tags: ["stockings"],
  nodes: [
    { id: "tex", type: "texture" },
    { id: "geo", type: "geometry" },
    { id: "map", type: "mapping", inputs: { loc: [1.0, 1.0, 1.0], rot: [0.0, 1.5708, 1.5708] } },
    { id: "grad", type: "tex_gradient" },
    { id: "ramp_001", type: "ramp_tri" },
    { id: "ramp_002", type: "ramp_cardinal", inputs: { pos0: 0.0, pos1: 0.9565 } },
    { id: "facing", type: "layer_weight/facing", inputs: { blend: 0.4 } },
    { id: "ramp_face", type: "ramp_cardinal", inputs: { pos0: 0.0, pos1: 0.5435 } },
    { id: "mix_001", type: "mix/blend", inputs: { fac: 0.5, a: [1.0, 1.0, 1.0] } },
    { id: "mask", type: "mix/lighten", inputs: { fac: 0.5 } },
    { id: "emission_hs", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.0, value: 5.0, fac: 1.0 } },
    {
      id: "principled",
      type: "principled",
      inputs: {
        metallic: 0.1,
        specular: 1.0,
        roughness: 0.5,
        spec_clamp: 1e30,
        sheen: 0.7017999887466431,
        sheen_tint: 0.5,
      },
    },
    { id: "mix_shader_001", type: "mix_shader" },
  ],
  links: [
    { from: { node: "geo", socket: "uv" }, to: { node: "map", socket: "vector" } },
    { from: { node: "map", socket: "vector" }, to: { node: "grad", socket: "vector" } },
    { from: { node: "grad", socket: "value" }, to: { node: "ramp_001", socket: "fac" } },
    { from: { node: "ramp_001", socket: "value" }, to: { node: "ramp_002", socket: "fac" } },
    { from: { node: "facing", socket: "value" }, to: { node: "ramp_face", socket: "fac" } },
    { from: { node: "ramp_face", socket: "fac_out" }, to: { node: "mix_001", socket: "b" } },
    { from: { node: "mix_001", socket: "color" }, to: { node: "mask", socket: "a" } },
    { from: { node: "ramp_002", socket: "fac_out" }, to: { node: "mask", socket: "b" } },
    { from: { node: "tex", socket: "color" }, to: { node: "emission_hs", socket: "color" } },
    { from: { node: "tex", socket: "color" }, to: { node: "principled", socket: "base" } },
    { from: { node: "emission_hs", socket: "color" }, to: { node: "mix_shader_001", socket: "a" } },
    { from: { node: "principled", socket: "color" }, to: { node: "mix_shader_001", socket: "b" } },
    { from: { node: "mask", socket: "color" }, to: { node: "mix_shader_001", socket: "fac" } },
  ],
  output: { node: "mix_shader_001", socket: "color" },
}
