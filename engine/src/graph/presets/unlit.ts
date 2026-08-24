// Unlit — the texture, at its own brightness, and nothing else.
//
// What FOOTAGE needs. A media plane carries pixels somebody already finished
// somewhere else: a gradient painted in Photoshop, a title card, a rendered
// element. Shading it is not a stylistic choice but a mistake — the scene's sun
// would dim one side of a card that has no side, and the world colour would
// tint artwork whose colour is the point. Blender's answer is the same one:
// wire the image straight into an Emission shader and let it out at the value
// it was authored at.
//
// Emission rather than Principled with roughness 1: emission is radiance, so it
// leaves the light loop entirely instead of being an unusually flat surface
// inside it. That is also what lets a card sit in front of a lamp without
// picking up its highlight.

import type { ShaderGraph } from "../schema"

export const UNLIT_GRAPH: ShaderGraph = {
  version: 1,
  name: "Unlit",
  tags: ["unlit", "plane"],
  nodes: [
    { id: "tex", type: "texture" },
    // The PMX material colour still multiplies in, so a plane can be tinted or
    // faded through the same dial every other material uses rather than needing
    // one of its own.
    { id: "mat", type: "material_diffuse" },
    { id: "base", type: "mix/multiply", inputs: { fac: 1.0 } },
    { id: "emit", type: "emission", inputs: { strength: 1.0 } },
  ],
  links: [
    { from: { node: "tex", socket: "color" }, to: { node: "base", socket: "a" } },
    { from: { node: "mat", socket: "color" }, to: { node: "base", socket: "b" } },
    { from: { node: "base", socket: "color" }, to: { node: "emit", socket: "color" } },
  ],
  output: { node: "emit", socket: "color" },
}
