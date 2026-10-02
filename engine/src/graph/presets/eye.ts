// Eye as a ShaderGraph — port of shaders/materials/eye.ts. The published preset
// author's instruction: "keep eyes in the default nodegraph, add emission 1.5".
// So it's the default Lit plus an emission of the diffuse texture at
// 1.5× (the emission, decomposed as a separate scale +
// Add Shader — the emission feeds bloom pre-tonemap).
//
// The rear-view gate and the see-through stencil stamp are slot-owned (built-in eye
// behavior, see EYE_TEMPLATE in slots.ts + createSlotPipeline) — not in this graph.

import type { ShaderGraph } from "../schema"

export const EYE_GRAPH: ShaderGraph = {
  version: 1,
  name: "Eye",
  tags: ["eye"],
  nodes: [
    { id: "tex", type: "texture" },
    // Base colour: the texture times the PMX material's diffuse. MMD authors
    // tint per MATERIAL rather than per texel — an eyebrow sharing the face
    // atlas is painted once and coloured here — so the diffuse belongs in the
    // base of every look.
    { id: "mat_diffuse", type: "material_diffuse" },
    { id: "tex_base", type: "mix/multiply", inputs: { fac: 1.0 } },
    {
      id: "principled",
      type: "lit",
      inputs: { metallic: 0.0, smoothness: 0.2929 },
    },
    { id: "emission", type: "vector_math/scale", inputs: { scale: 1.5 } },
    { id: "add", type: "vector_math/add" },
    // The iris glow follows the key round her head: full (1.5) lit from the
    // front, easing to 0.2 of it lit from behind - a fixed 1.5 left the eyes
    // bright white in a backlit face. head_basis.forward · L, half-Lambert,
    // times the sun's shadow at the eye, so an eye in a shadowed face dims too.
    { id: "eye_hb", type: "head_basis" },
    { id: "eye_lt", type: "light" },
    { id: "eye_fl", type: "vector_math/dot" },
    { id: "eye_half", type: "math/multiply_add", inputs: { b: 0.5, c: 0.5 } },
    { id: "eye_lvl", type: "map_range", inputs: { from_min: 0.0, from_max: 1.0, to_min: 0.2, to_max: 1.0 } },
    { id: "eye_sh", type: "math/multiply" },
    { id: "eye_str", type: "math/multiply", inputs: { b: 1.5 } },
  ],
  links: [
    { from: { node: "tex", socket: "color" }, to: { node: "tex_base", socket: "a" } },
    { from: { node: "mat_diffuse", socket: "color" }, to: { node: "tex_base", socket: "b" } },
    { from: { node: "tex_base", socket: "color" }, to: { node: "principled", socket: "base_color" } },
    { from: { node: "tex_base", socket: "color" }, to: { node: "emission", socket: "a" } },
    { from: { node: "principled", socket: "color" }, to: { node: "add", socket: "a" } },
    { from: { node: "emission", socket: "vector" }, to: { node: "add", socket: "b" } },
    { from: { node: "eye_hb", socket: "forward" }, to: { node: "eye_fl", socket: "a" } },
    { from: { node: "eye_lt", socket: "direction" }, to: { node: "eye_fl", socket: "b" } },
    { from: { node: "eye_fl", socket: "value" }, to: { node: "eye_half", socket: "a" } },
    { from: { node: "eye_half", socket: "value" }, to: { node: "eye_sh", socket: "a" } },
    { from: { node: "eye_lt", socket: "shadow" }, to: { node: "eye_sh", socket: "b" } },
    { from: { node: "eye_sh", socket: "value" }, to: { node: "eye_lvl", socket: "value" } },
    { from: { node: "eye_lvl", socket: "value" }, to: { node: "eye_str", socket: "a" } },
    { from: { node: "eye_str", socket: "value" }, to: { node: "emission", socket: "scale" } },
  ],
  output: { node: "add", socket: "vector" },
}
