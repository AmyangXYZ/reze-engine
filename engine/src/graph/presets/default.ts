// Default material — the neutral base used two ways: the ungrouped fallback (a material
// in no style group renders this) and the blank-canvas starter the editor's "New graph"
// begins from, so the two always agree. MMD-correct PBSDF base: diffuse texture × the
// PMX material diffuse color → Lit (metallic 0, smoothness 0.29).
// The material-color multiply is what keeps untextured/solid-color materials from
// rendering white (they carry their color in material.diffuse, not a texture).

import type { ShaderGraph } from "../schema"

export const DEFAULT_GRAPH: ShaderGraph = {
  version: 1,
  name: "Lit",
  tags: ["default"],
  nodes: [
    { id: "tex", type: "texture" },
    { id: "mat", type: "material_diffuse" },
    { id: "base", type: "mix/multiply", inputs: { fac: 1.0 } }, // texture × material diffuse
    {
      id: "principled",
      type: "lit",
      inputs: { metallic: 0.0, smoothness: 0.2929 },
    },
  ],
  links: [
    { from: { node: "tex", socket: "color" }, to: { node: "base", socket: "a" } },
    { from: { node: "mat", socket: "color" }, to: { node: "base", socket: "b" } },
    { from: { node: "base", socket: "color" }, to: { node: "principled", socket: "base_color" } },
  ],
  output: { node: "principled", socket: "color" },
}
