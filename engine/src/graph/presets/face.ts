// M_Face as a ShaderGraph — port of shaders/materials/face.ts (仿深空之眼 "M_Face").
// Toon + warm rim + dual fresnel rim + BT.601 bright-tex gate, mixed 50/50 against a
// Principled BSDF with noise bump. Structurally close to body, but: the toon uses the
// edge-AA ramp, hue tints are 0.46 (full hue_sat, not the 0.5 _id specialization),
// rim2 is the hair-style fresnel×layer_weight_fresnel rim, and there's an extra
// near-white texture gate that emits (freckle/highlight paint). Plain material — no
// built-in slot effect, so it uses the default prelude/epilogue.

import type { ShaderGraph } from "../schema"

export const FACE_GRAPH: ShaderGraph = {
  version: 1,
  name: "Face",
  tags: ["face"],
  nodes: [
    { id: "tex", type: "texture" },
    { id: "geo", type: "geometry" },

    // ── toon shading ──
    { id: "str", type: "shader_to_rgb_diffuse" },
    { id: "toon", type: "ramp_constant_aa", inputs: { edge: 0.2966 } },
    { id: "shadow_tint", type: "hue_sat", inputs: { hue: 0.46000000834465027, saturation: 2.0, value: 0.3499999940395355, fac: 1.0 } },
    { id: "lit_tint", type: "hue_sat", inputs: { hue: 0.46000000834465027, saturation: 1.600000023841858, value: 1.5, fac: 1.0 } },
    { id: "toon_color", type: "mix/blend" },
    { id: "bc", type: "bright_contrast", inputs: { bright: 0.1, contrast: 0.2 } },
    { id: "emission3", type: "emission", inputs: { strength: 2.5 } },

    // ── warm rim (toon*0.5+0.5 → cardinal ramp) ──
    { id: "warm_mul", type: "math/multiply", inputs: { b: 0.5 } },
    { id: "warm_add", type: "math/add", inputs: { b: 0.5 } },
    { id: "warm_clamp", type: "math/clamp01" },
    {
      id: "warm_ramp",
      type: "ramp_cardinal",
      inputs: {
        pos0: 0.2409,
        color0: [0.2426, 0.068, 0.0588, 1.0],
        pos1: 0.4663,
        color1: [0.6677, 0.5024, 0.5126, 1.0],
      },
    },
    { id: "warm_emit", type: "emission", inputs: { strength: 0.30000001192092896 } },

    // ── rim1: fresnel × facing ──
    { id: "rim1_fres", type: "fresnel", inputs: { ior: 2.0 } },
    { id: "rim1_lw", type: "layer_weight/facing", inputs: { blend: 0.24 } },
    { id: "rim1_str", type: "math/multiply" },
    { id: "rim1", type: "emission", inputs: { color: [0.984157919883728, 0.6110184788703918, 0.5736401677131653] } },

    // ── rim2: fresnel × fresnel-layer-weight, powered (hair-style) ──
    { id: "rim2_fres", type: "fresnel", inputs: { ior: 1.45 } },
    { id: "rim2_lw", type: "layer_weight/fresnel", inputs: { blend: 0.61 } },
    { id: "rim2_raw", type: "math/multiply" },
    { id: "rim2_pow", type: "math/power", inputs: { b: 0.6300000548362732 } },
    { id: "rim2_mix", type: "mix_shader", inputs: { b: [1.0, 0.4684903025627136, 0.3698573112487793] } },

    // ── near-white texture gate emission ──
    { id: "gate", type: "math/greater_than", inputs: { b: 0.9300000071525574 } },
    { id: "gate_scale", type: "math/multiply", inputs: { b: 3.0 } },

    // ── npr stack (rim1 + rim2 + bright_emit + warm) ──
    { id: "npr_add1", type: "add_shader" },
    { id: "npr_add2", type: "add_shader" },
    { id: "npr_stack", type: "add_shader" },

    // ── principled with noise bump ──
    { id: "map", type: "mapping", inputs: { scl: [1.0, 1.0, 1.5] } },
    { id: "noise", type: "tex_noise", inputs: { scale: 1.0 } },
    { id: "noise_ramp", type: "ramp_linear", inputs: { pos0: 0.0, pos1: 1.0 } },
    { id: "bump", type: "bump", inputs: { strength: 0.324644535779953 } },
    { id: "principled_base", type: "mix/blend", inputs: { b: [0.6832, 0.1947, 0.1373] } },
    { id: "p_emit", type: "emission", inputs: { strength: 0.2 } },
    {
      id: "principled",
      type: "principled",
      inputs: { metallic: 0.0, specular_ior_level: 0.5, roughness: 0.3, spec_clamp: 10.0, sheen_weight: 0.0, sheen_tint: 0.0 },
    },
    { id: "p_sum", type: "add_shader" },

    { id: "mix_shader_001", type: "mix_shader", inputs: { fac: 0.5 } },
  ],
  links: [
    // toon
    { from: { node: "str", socket: "value" }, to: { node: "toon", socket: "fac" } },
    { from: { node: "tex", socket: "color" }, to: { node: "shadow_tint", socket: "color" } },
    { from: { node: "tex", socket: "color" }, to: { node: "lit_tint", socket: "color" } },
    { from: { node: "toon", socket: "fac_out" }, to: { node: "toon_color", socket: "fac" } },
    { from: { node: "shadow_tint", socket: "color" }, to: { node: "toon_color", socket: "a" } },
    { from: { node: "lit_tint", socket: "color" }, to: { node: "toon_color", socket: "b" } },
    { from: { node: "toon_color", socket: "color" }, to: { node: "bc", socket: "color" } },
    { from: { node: "bc", socket: "color" }, to: { node: "emission3", socket: "color" } },
    // warm rim
    { from: { node: "toon", socket: "fac_out" }, to: { node: "warm_mul", socket: "a" } },
    { from: { node: "warm_mul", socket: "value" }, to: { node: "warm_add", socket: "a" } },
    { from: { node: "warm_add", socket: "value" }, to: { node: "warm_clamp", socket: "a" } },
    { from: { node: "warm_clamp", socket: "value" }, to: { node: "warm_ramp", socket: "fac" } },
    { from: { node: "warm_ramp", socket: "color" }, to: { node: "warm_emit", socket: "color" } },
    // rim1
    { from: { node: "rim1_fres", socket: "value" }, to: { node: "rim1_str", socket: "a" } },
    { from: { node: "rim1_lw", socket: "value" }, to: { node: "rim1_str", socket: "b" } },
    { from: { node: "rim1_str", socket: "value" }, to: { node: "rim1", socket: "strength" } },
    // rim2
    { from: { node: "rim2_fres", socket: "value" }, to: { node: "rim2_raw", socket: "a" } },
    { from: { node: "rim2_lw", socket: "value" }, to: { node: "rim2_raw", socket: "b" } },
    { from: { node: "rim2_raw", socket: "value" }, to: { node: "rim2_pow", socket: "a" } },
    { from: { node: "emission3", socket: "color" }, to: { node: "rim2_mix", socket: "a" } },
    { from: { node: "rim2_pow", socket: "value" }, to: { node: "rim2_mix", socket: "fac" } },
    // gate
    { from: { node: "tex", socket: "color" }, to: { node: "gate", socket: "a" } },
    { from: { node: "gate", socket: "value" }, to: { node: "gate_scale", socket: "a" } },
    // npr stack
    { from: { node: "rim1", socket: "color" }, to: { node: "npr_add1", socket: "a" } },
    { from: { node: "rim2_mix", socket: "color" }, to: { node: "npr_add1", socket: "b" } },
    { from: { node: "npr_add1", socket: "color" }, to: { node: "npr_add2", socket: "a" } },
    { from: { node: "gate_scale", socket: "value" }, to: { node: "npr_add2", socket: "b" } },
    { from: { node: "npr_add2", socket: "color" }, to: { node: "npr_stack", socket: "a" } },
    { from: { node: "warm_emit", socket: "color" }, to: { node: "npr_stack", socket: "b" } },
    // principled + noise bump
    { from: { node: "geo", socket: "rest_pos" }, to: { node: "map", socket: "vector" } },
    { from: { node: "map", socket: "vector" }, to: { node: "noise", socket: "vector" } },
    { from: { node: "noise", socket: "value" }, to: { node: "noise_ramp", socket: "fac" } },
    { from: { node: "noise_ramp", socket: "fac_out" }, to: { node: "bump", socket: "height" } },
    { from: { node: "geo", socket: "normal" }, to: { node: "bump", socket: "normal" } },
    { from: { node: "noise_ramp", socket: "fac_out" }, to: { node: "principled_base", socket: "fac" } },
    { from: { node: "bc", socket: "color" }, to: { node: "principled_base", socket: "a" } },
    { from: { node: "bc", socket: "color" }, to: { node: "p_emit", socket: "color" } },
    { from: { node: "principled_base", socket: "color" }, to: { node: "principled", socket: "base_color" } },
    { from: { node: "bump", socket: "vector" }, to: { node: "principled", socket: "normal" } },
    { from: { node: "principled", socket: "color" }, to: { node: "p_sum", socket: "a" } },
    { from: { node: "p_emit", socket: "color" }, to: { node: "p_sum", socket: "b" } },
    // final mix
    { from: { node: "npr_stack", socket: "color" }, to: { node: "mix_shader_001", socket: "a" } },
    { from: { node: "p_sum", socket: "color" }, to: { node: "mix_shader_001", socket: "b" } },
  ],
  output: { node: "mix_shader_001", socket: "color" },
}
