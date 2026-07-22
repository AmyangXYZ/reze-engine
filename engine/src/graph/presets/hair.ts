// M_Hair — 仿深空之眼渲染预设v1.0_by_小绿毛猫, as a StyleGraph. Node ids mirror the
// Blender node names referenced in shaders/materials/hair.ts comments; compiled with
// { inlineParams: true } this graph must reproduce HAIR_SHADER_WGSL's fragment body
// (golden test #1 — see tests/graph.test.mjs).
//
// Inherited authoring decisions from the hand port: the noise→bump subtree on
// Principled.Normal is omitted (imperceptible at 0.2 mix weight), and Blender's bevel
// node is approximated by saturate(normal.y) — Blender Z-up ⇒ engine Y-up.
//
// The object is pure JSON (no functions/undefined) — JSON.stringify round-trips it,
// which is how reze.design will ship additional presets.

import type { StyleGraph } from "../schema"

export const HAIR_GRAPH: StyleGraph = {
  version: 1,
  name: "Hair",
  tags: ["hair"],
  nodes: [
    { id: "tex", type: "texture" },
    { id: "geo", type: "geometry" },
    { id: "hs_shadow", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.2, value: 0.5, fac: 1.0 } },
    { id: "hs_002", type: "hue_sat", inputs: { hue: 0.48, saturation: 1.2, value: 0.7, fac: 1.0 } },
    { id: "hs_001", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.5, value: 1.0, fac: 1.0 } },
    { id: "str", type: "shader_to_rgb_diffuse" },
    { id: "ramp_008", type: "ramp_constant", inputs: { pos0: 0.0, color0: [0, 0, 0, 1], pos1: 0.2966, color1: [1, 1, 1, 1] } },
    { id: "mix_004", type: "mix/blend" },
    { id: "bc", type: "bright_contrast", inputs: { bright: 0.1, contrast: 0.2 } },
    { id: "sep_n", type: "separate_xyz" },
    { id: "bevel_clamp", type: "math/clamp01" },
    { id: "mix_003", type: "mix/blend" },
    { id: "fres", type: "fresnel", inputs: { ior: 1.45 } },
    { id: "lw", type: "layer_weight/fresnel", inputs: { blend: 0.61 } },
    { id: "rim_mul", type: "math/multiply" },
    { id: "rim_pow", type: "math/power", inputs: { b: 0.6300000548362732 } },
    {
      id: "mix_shader_002",
      type: "mix_shader",
      inputs: { b: [0.1673291176557541, 0.1673291176557541, 0.1673291176557541] },
    },
    { id: "gate", type: "math/greater_than", inputs: { b: 0.15000000596046448 } },
    { id: "gate_scale", type: "math/multiply", inputs: { b: 0.1 } },
    { id: "npr_add", type: "mix/add_emit" },
    {
      id: "principled",
      type: "principled",
      inputs: { metallic: 0.0, specular: 1.0, roughness: 0.3, spec_clamp: 10.0, sheen: 0.0, sheen_tint: 0.0 },
    },
    { id: "mix_shader_001", type: "mix_shader", inputs: { fac: 0.2 } },
  ],
  links: [
    { from: { node: "tex", socket: "color" }, to: { node: "hs_shadow", socket: "color" } },
    { from: { node: "hs_shadow", socket: "color" }, to: { node: "hs_002", socket: "color" } },
    { from: { node: "tex", socket: "color" }, to: { node: "hs_001", socket: "color" } },
    { from: { node: "str", socket: "value" }, to: { node: "ramp_008", socket: "fac" } },
    { from: { node: "ramp_008", socket: "fac_out" }, to: { node: "mix_004", socket: "fac" } },
    { from: { node: "hs_002", socket: "color" }, to: { node: "mix_004", socket: "a" } },
    { from: { node: "hs_001", socket: "color" }, to: { node: "mix_004", socket: "b" } },
    { from: { node: "mix_004", socket: "color" }, to: { node: "bc", socket: "color" } },
    { from: { node: "geo", socket: "normal" }, to: { node: "sep_n", socket: "vector" } },
    { from: { node: "sep_n", socket: "y" }, to: { node: "bevel_clamp", socket: "a" } },
    { from: { node: "bevel_clamp", socket: "value" }, to: { node: "mix_003", socket: "fac" } },
    { from: { node: "bc", socket: "color" }, to: { node: "mix_003", socket: "a" } },
    { from: { node: "hs_002", socket: "color" }, to: { node: "mix_003", socket: "b" } },
    { from: { node: "fres", socket: "value" }, to: { node: "rim_mul", socket: "a" } },
    { from: { node: "lw", socket: "value" }, to: { node: "rim_mul", socket: "b" } },
    { from: { node: "rim_mul", socket: "value" }, to: { node: "rim_pow", socket: "a" } },
    { from: { node: "mix_003", socket: "color" }, to: { node: "mix_shader_002", socket: "a" } },
    { from: { node: "rim_pow", socket: "value" }, to: { node: "mix_shader_002", socket: "fac" } },
    { from: { node: "tex", socket: "color" }, to: { node: "gate", socket: "a" } },
    { from: { node: "gate", socket: "value" }, to: { node: "gate_scale", socket: "a" } },
    { from: { node: "mix_shader_002", socket: "color" }, to: { node: "npr_add", socket: "a" } },
    { from: { node: "gate_scale", socket: "value" }, to: { node: "npr_add", socket: "b" } },
    { from: { node: "bc", socket: "color" }, to: { node: "principled", socket: "base" } },
    { from: { node: "npr_add", socket: "color" }, to: { node: "mix_shader_001", socket: "a" } },
    { from: { node: "principled", socket: "color" }, to: { node: "mix_shader_001", socket: "b" } },
  ],
  output: { node: "mix_shader_001", socket: "color" },
  // Adjust-tier params deliberately absent: which sockets deserve sliders (and what
  // they're called) is preset-author curation done in reze.design, not engine data.
  // The exposed-param mechanism itself is covered by tests/graph.test.mjs.
}
