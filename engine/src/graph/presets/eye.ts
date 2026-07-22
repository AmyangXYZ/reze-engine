// Eye as a StyleGraph — port of shaders/materials/eye.ts. The published preset
// author's instruction: "keep eyes in the default nodegraph, add emission 1.5".
// So it's the default Principled BSDF plus an Emission of the diffuse texture at
// 1.5× (Blender's Principled Emission socket, decomposed as a separate Emission +
// Add Shader — the emission feeds bloom pre-tonemap).
//
// The rear-view gate and the see-through stencil stamp are slot-owned (built-in eye
// behavior, see EYE_TEMPLATE in slots.ts + createSlotPipeline) — not in this graph.

import type { StyleGraph } from "../schema"

export const EYE_GRAPH: StyleGraph = {
  version: 1,
  name: "Eye",
  tags: ["eye"],
  nodes: [
    { id: "tex", type: "texture" },
    {
      id: "principled",
      type: "principled",
      inputs: { metallic: 0.0, specular: 0.5, roughness: 0.5, spec_clamp: 1e30, sheen: 0.0, sheen_tint: 0.0 },
    },
    { id: "emission", type: "emission", inputs: { strength: 1.5 } },
    { id: "add", type: "add_shader" },
  ],
  links: [
    { from: { node: "tex", socket: "color" }, to: { node: "principled", socket: "base" } },
    { from: { node: "tex", socket: "color" }, to: { node: "emission", socket: "color" } },
    { from: { node: "principled", socket: "color" }, to: { node: "add", socket: "a" } },
    { from: { node: "emission", socket: "color" }, to: { node: "add", socket: "b" } },
  ],
  output: { node: "add", socket: "color" },
}
