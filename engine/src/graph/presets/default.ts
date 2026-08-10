// Default material — the neutral base used two ways: the ungrouped fallback (a material
// in no style group renders this) and the blank-canvas starter the editor's "New graph"
// begins from, so the two always agree. MMD-correct PBSDF base: diffuse texture × the
// PMX material diffuse color → Principled BSDF (Metallic 0, Specular 0.5, Roughness 0.5).
// The material-color multiply is what keeps untextured/solid-color materials from
// rendering white (they carry their color in material.diffuse, not a texture).

import type { ShaderGraph } from "../schema"

export const DEFAULT_GRAPH: ShaderGraph = {
  version: 1,
  name: "Principled BSDF",
  tags: ["default"],
  nodes: [
    { id: "tex", type: "texture" },
    { id: "mat", type: "material_diffuse" },
    { id: "base", type: "mix/multiply", inputs: { fac: 1.0 } }, // texture × material diffuse
    {
      id: "principled",
      type: "principled",
      inputs: { metallic: 0.0, specular_ior_level: 0.5, roughness: 0.5, spec_clamp: 10.0, sheen_weight: 0.0, sheen_tint: 0.0 },
    },
  ],
  links: [
    { from: { node: "tex", socket: "color" }, to: { node: "base", socket: "a" } },
    { from: { node: "mat", socket: "color" }, to: { node: "base", socket: "b" } },
    { from: { node: "base", socket: "color" }, to: { node: "principled", socket: "base_color" } },
  ],
  output: { node: "principled", socket: "color" },
}
