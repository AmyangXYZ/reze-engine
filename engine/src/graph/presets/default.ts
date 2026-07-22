// Default material as a StyleGraph — Blender's new-material template verbatim:
// Image Texture → Principled BSDF (Metallic 0, Specular 0.5, Roughness 0.5) → Output.
// This is both the port of shaders/materials/default.ts (must match it pixel-for-pixel;
// see tests/graph.test.mjs) and the blank-canvas starter graph an editor offers when
// the user creates a new style.

import type { StyleGraph } from "../schema"

export const DEFAULT_GRAPH: StyleGraph = {
  version: 1,
  name: "Principled BSDF",
  tags: ["default"],
  nodes: [
    { id: "tex", type: "texture" },
    {
      id: "principled",
      type: "principled",
      inputs: { metallic: 0.0, specular: 0.5, roughness: 0.5, spec_clamp: 10.0, sheen: 0.0, sheen_tint: 0.0 },
    },
  ],
  links: [{ from: { node: "tex", socket: "color" }, to: { node: "principled", socket: "base" } }],
  output: { node: "principled", socket: "color" },
}
