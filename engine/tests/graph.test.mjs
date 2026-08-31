// Graph compiler tests. Run: npm test (node --test with the extensionless-import hook).
// The hair snapshot is the compiler's contract: it must stay semantically line-for-line
// with shaders/materials/hair.ts (the hand-written ground truth) until the pixel-golden
// harness replaces it as the equivalence proof.

import { test } from "node:test"
import assert from "node:assert/strict"
import { compileGraph, validateGraph } from "../dist/graph/compile.js"
import { HAIR_GRAPH } from "../dist/graph/presets/hair.js"
import { DEFAULT_GRAPH } from "../dist/graph/presets/default.js"
import { CLOTH_SMOOTH_GRAPH } from "../dist/graph/presets/cloth_smooth.js"
import { CLOTH_ROUGH_GRAPH } from "../dist/graph/presets/cloth_rough.js"
import { METAL_GRAPH } from "../dist/graph/presets/metal.js"
import { BODY_GRAPH } from "../dist/graph/presets/body.js"
import { STOCKINGS_GRAPH } from "../dist/graph/presets/stockings.js"
import { EYE_GRAPH } from "../dist/graph/presets/eye.js"
import { FACE_GRAPH } from "../dist/graph/presets/face.js"

const HAIR_BODY_INLINE = [
  "  let n_fres = fresnel(1.45, n, v); // @node:fres",
  "  let n_lw = layer_weight_fresnel(0.61, n, v); // @node:lw",
  "  let n_rim_mul = math_multiply(n_fres, n_lw); // @node:rim_mul",
  "  let n_rim_pow = math_power(n_rim_mul, 0.6300000548362732); // @node:rim_pow",
  "  let n_sep_n = n; // @node:sep_n",
  "  let n_bevel_clamp = saturate(n_sep_n.y); // @node:bevel_clamp",
  "  let n_str = shader_to_rgb_diffuse(n, l, sun, amb, shadow); // @node:str",
  "  let n_ramp_008 = ramp_constant(n_str, 0.0, vec4f(0.0, 0.0, 0.0, 1.0), 0.2966, vec4f(1.0, 1.0, 1.0, 1.0)); // @node:ramp_008",
  "  let n_gate = math_greater_than(color_to_value(tex_color), 0.15000000596046448); // @node:gate",
  "  let n_gate_scale = math_multiply(n_gate, 0.1); // @node:gate_scale",
  "  let n_tex_base = mix_multiply(1.0, tex_color, material.diffuseColor); // @node:tex_base",
  "  let n_hs_001 = hue_sat_id(1.5, 1.0, 1.0, n_tex_base); // @node:hs_001",
  "  let n_hs_shadow = hue_sat_id(1.2, 0.5, 1.0, n_tex_base); // @node:hs_shadow",
  "  let n_hs_002 = hue_sat(0.48, 1.2, 0.7, 1.0, n_hs_shadow); // @node:hs_002",
  "  let n_mix_004 = mix_blend(n_ramp_008.r, n_hs_002, n_hs_001); // @node:mix_004",
  "  let n_bc = bright_contrast(n_mix_004, 0.1, 0.2); // @node:bc",
  "  let n_mix_003 = mix_blend(n_bevel_clamp, n_bc, n_hs_002); // @node:mix_003",
  "  let n_mix_shader_002 = mix(n_mix_003, vec3f(0.1673291176557541), n_rim_pow); // @node:mix_shader_002",
  "  let n_npr_add = n_mix_shader_002 + vec3f(n_gate_scale); // @node:npr_add",
  "  let n_principled = eval_principled(PrincipledIn(n_bc, 0.0, 1.0, 0.3, 10.0, 0.0, 0.0), n, l, v, sun, amb, shadow); // @node:principled",
  "  let n_mix_shader_001 = mix(n_npr_add, n_principled, 0.2); // @node:mix_shader_001",
  "  let final_color = n_mix_shader_001; // @node:mix_shader_001",
].join("\n")

test("default graph: MMD-correct neutral base (texture × material diffuse → PBSDF)", () => {
  const r = compileGraph(DEFAULT_GRAPH, { inlineParams: true })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  assert.equal(
    r.fsBody,
    [
      "  let n_base = mix_multiply(1.0, tex_color, material.diffuseColor); // @node:base",
      "  let n_principled = eval_principled(PrincipledIn(n_base, 0.0, 0.5, 0.5, 10.0, 0.0, 0.0), n, l, v, sun, amb, shadow); // @node:principled",
      "  let final_color = n_principled; // @node:principled",
    ].join("\n"),
  )
  // The ungrouped default renders neutral — no stencil / render-class overrides.
  assert.ok(!r.wgsl.includes("IS_OVER_EYES"))
})

test("cloth_smooth graph matches the hand-written shader (snapshot)", () => {
  const r = compileGraph(CLOTH_SMOOTH_GRAPH, { inlineParams: true })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  assert.equal(
    r.fsBody,
    [
      "  let n_sep_n = n; // @node:sep_n",
      "  let n_bevel_clamp = saturate(n_sep_n.y); // @node:bevel_clamp",
      "  let n_str = shader_to_rgb_diffuse(n, l, sun, amb, shadow); // @node:str",
      "  let n_ramp_008 = ramp_constant_edge_aa(n_str, 0.2966, vec4f(0.0, 0.0, 0.0, 1.0), vec4f(1.0, 1.0, 1.0, 1.0)); // @node:ramp_008",
      "  let n_mix04_fac = math_multiply(n_ramp_008.r, 0.5); // @node:mix04_fac",
      "  let n_tex_base = mix_multiply(1.0, tex_color, material.diffuseColor); // @node:tex_base",
      "  let n_dark_tex = hue_sat_id(1.0, 0.19999998807907104, 1.0, n_tex_base); // @node:dark_tex",
      "  let n_mix_004 = mix_blend(n_mix04_fac, n_dark_tex, n_tex_base); // @node:mix_004",
      "  let n_mix_003 = mix_blend(n_bevel_clamp, n_mix_004, n_dark_tex); // @node:mix_003",
      "  let n_hue_004 = hue_sat_id(0.800000011920929, 2.0, 1.0, n_mix_003); // @node:hue_004",
      "  let n_npr_overlay = mix_overlay(1.0, n_mix_003, n_hue_004); // @node:npr_overlay",
      "  let n_npr_emit = n_npr_overlay * 18.200000762939453; // @node:npr_emit",
      "  let n_principled_base = hue_sat_id(1.0, 0.800000011920929, 1.0, n_tex_base); // @node:principled_base",
      "  let n_principled = eval_principled(PrincipledIn(n_principled_base, 0.0, 0.8, 0.5, 10.0, 0.0, 0.0), n, l, v, sun, amb, shadow); // @node:principled",
      "  let n_mix_shader_001 = mix(n_npr_emit, n_principled, 0.8999999761581421); // @node:mix_shader_001",
      "  let final_color = n_mix_shader_001; // @node:mix_shader_001",
    ].join("\n"),
  )
})

test("metal graph matches the hand-written shader (snapshot)", () => {
  const r = compileGraph(METAL_GRAPH, { inlineParams: true })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  assert.equal(
    r.fsBody,
    [
      "  let n_str = shader_to_rgb_diffuse(n, l, sun, amb, shadow); // @node:str",
      "  let n_ramp_008 = ramp_constant_edge_aa(n_str, 0.2966, vec4f(0.0, 0.0, 0.0, 1.0), vec4f(1.0, 1.0, 1.0, 1.0)); // @node:ramp_008",
      "  let n_mix04_fac = math_multiply(n_ramp_008.r, 0.5); // @node:mix04_fac",
      "  let n_tex_base = mix_multiply(1.0, tex_color, material.diffuseColor); // @node:tex_base",
      "  let n_tex_tint = hue_sat_id(1.0, 0.800000011920929, 1.0, n_tex_base); // @node:tex_tint",
      "  let n_dark_tex = hue_sat_id(1.0, 0.19999998807907104, 1.0, n_tex_tint); // @node:dark_tex",
      "  let n_hue_006 = hue_sat_id(1.5, 1.2999999523162842, 1.0, n_tex_tint); // @node:hue_006",
      "  let n_mix_004 = mix_blend(n_mix04_fac, n_dark_tex, n_tex_tint); // @node:mix_004",
      "  let n_hue_004 = hue_sat_id(1.0, 2.0, 1.0, n_mix_004); // @node:hue_004",
      "  let n_npr_overlay = mix_overlay(1.0, n_mix_004, n_hue_004); // @node:npr_overlay",
      "  let n_npr_emit = n_npr_overlay * 8.100000381469727; // @node:npr_emit",
      "  let n_voro_cross = vect_math_cross(reflect(-v, n), vec3f(0.0, 1.0, 0.0)); // @node:voro_cross",
      "  let n_voro = tex_voronoi_color(n_voro_cross, 4.3); // @node:voro",
      "  let n_voro_ramp = ramp_linear(color_to_value(n_voro), 0.0, vec4f(0.0, 0.0, 0.0, 1.0), 1.0, vec4f(1.0, 1.0, 1.0, 1.0)); // @node:voro_ramp",
      "  let n_albedo = mix_blend(n_voro_ramp.r, vec3f(n_voro_ramp.r), n_hue_006); // @node:albedo",
      "  let n_principled = eval_principled(PrincipledIn(n_albedo, 1.0, 1.0, 0.3, 1e+30, 0.0, 0.0), n, l, v, sun, amb, shadow); // @node:principled",
      "  let n_mix_shader_001 = mix(n_npr_emit, n_principled, 0.6967); // @node:mix_shader_001",
      "  let final_color = n_mix_shader_001; // @node:mix_shader_001",
    ].join("\n"),
  )
})

test("cloth_rough graph matches the hand-written shader (key terms)", () => {
  const r = compileGraph(CLOTH_ROUGH_GRAPH, { inlineParams: true })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  assert.ok(r.fsBody.includes("let n_noise = tex_noise_d2(input.restPos, 17.7);"))
  assert.ok(r.fsBody.includes("let n_bump = bump_lh(1.0, n_noise_ramp.r, n, input.worldPos);"))
  assert.ok(
    r.fsBody.includes(
      "eval_principled(PrincipledIn(n_principled_base, 0.0, 0.8, 0.8187, 10.0, 0.0, 0.0), n_bump, l, v, sun, amb, shadow)",
    ),
  )
  assert.ok(r.fsBody.includes("mix(n_npr_emit, n_principled, 0.8999999761581421)"))
})

test("body graph matches the hand-written shader (key terms)", () => {
  const r = compileGraph(BODY_GRAPH, { inlineParams: true })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  const expect = [
    "let n_map = mapping_point(input.restPos, vec3f(0.0), vec3f(0.0), vec3f(1.0, 1.0, 1.5));",
    "let n_noise = tex_noise_d2(n_map, 1.0);",
    "let n_bump = bump_lh(0.324644535779953, n_noise_ramp.r, n, input.worldPos);",
    "let n_rim1 = vec3f(0.984157919883728, 0.6110184788703918, 0.5736401677131653) * n_rim1_str;",
    "let n_rim2_pow = math_power(n_rim2_lw, 1.4300000667572021);",
    "let n_toon_color = mix_blend(n_toon.r, n_shadow_tint, n_lit_tint);",
    "let n_emission3 = n_bc * 4.0;",
    "let n_warm_add = math_add(n_toon.r, 0.5);",
    "let n_warm_emit = n_warm_ramp.rgb * 0.30000001192092896;",
    "let n_rim2_mix = mix(n_emission3, vec3f(1.0, 0.4303792119026184, 0.3315804898738861), n_rim2_ramp.r);",
    "let n_npr_stack = n_npr_add1 + n_warm_emit;",
    "eval_principled(PrincipledIn(n_principled_base, 0.0, 0.5, 0.3, 10.0, 0.0, 0.0), n_bump, l, v, sun, amb, shadow)",
    "let n_p_sum = n_principled + n_p_emit;",
    "let n_mix_shader_001 = mix(n_npr_stack, n_p_sum, 0.5);",
  ]
  for (const line of expect) assert.ok(r.fsBody.includes(line), `missing: ${line}`)
})

test("stockings graph: radiance in graph, hashed alpha from alphaMode", () => {
  const r = compileGraph(STOCKINGS_GRAPH, { inlineParams: true, alphaMode: "hashed" })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  const expect = [
    "let n_map = mapping_point(vec3f(input.uv, 0.0), vec3f(1.0), vec3f(0.0, 1.5708, 1.5708), vec3f(1.0));",
    "let n_ramp_001 = 1.0 - abs(2.0 * n_grad - 1.0);",
    "let n_mix_001 = mix_blend(0.5, vec3f(1.0), vec3f(n_ramp_face.r));",
    "let n_mask = mix_lighten(0.5, n_mix_001, vec3f(n_ramp_002.r));",
    "let n_emission_hs = hue_sat_id(1.0, 5.0, 1.0, n_tex_base);",
    "eval_principled(PrincipledIn(n_tex_base, 0.1, 1.0, 0.5, 1e+30, 0.7017999887466431, 0.5), n, l, v, sun, amb, shadow)",
    "mix(n_emission_hs, n_principled, color_to_value(n_mask))",
  ]
  for (const line of expect) assert.ok(r.fsBody.includes(line), `missing: ${line}`)
  // Slot-owned behaviors: Wyman hash gate replaces the alpha threshold; alpha out = 1.
  assert.ok(r.wgsl.includes("if (alpha < hashed_alpha_threshold(input.restPos)) { discard; }"))
  // The ALPHA is what this test is about — hashed forces it to 1. The colour
  // carries the positional-light layer beside final_color, and the dissolve's
  // burning edge after it; both are zero in a scene using neither. Asserted as
  // the two ENDS of the expression rather than the whole of it, because pinning
  // the whole of it is what made this test fail the first time anything was
  // added to the epilogue — which the note it replaces predicted.
  assert.ok(r.wgsl.includes("out.color = vec4f(final_color + rzLightsDiffuse(input.worldPos, n) * albedo"))
  assert.ok(r.wgsl.includes(", 1.0);"))
  assert.ok(!r.wgsl.includes("if (alpha < 0.001)"))
})

test("eye graph: default Principled + emission, rear-gate from renderClass", () => {
  const r = compileGraph(EYE_GRAPH, { inlineParams: true, renderClass: "eye" })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  assert.equal(
    r.fsBody,
    [
      "  let n_tex_base = mix_multiply(1.0, tex_color, material.diffuseColor); // @node:tex_base",
      "  let n_emission = n_tex_base * 1.5; // @node:emission",
      "  let n_principled = eval_principled(PrincipledIn(n_tex_base, 0.0, 0.5, 0.5, 1e+30, 0.0, 0.0), n, l, v, sun, amb, shadow); // @node:principled",
      "  let n_add = n_principled + n_emission; // @node:add",
      "  let final_color = n_add; // @node:add",
    ].join("\n"),
  )
  // Slot-owned: rear-view gate in the prelude, standard alpha epilogue.
  assert.ok(r.wgsl.includes("if (dot(faceDir, v) < -0.15) { discard; }"))
  assert.ok(r.wgsl.includes("out.color = vec4f(final_color + rzLightsDiffuse(input.worldPos, n) * albedo"))
  assert.ok(r.wgsl.includes(", alpha);"))
})

test("face graph matches the hand-written shader (key terms)", () => {
  const r = compileGraph(FACE_GRAPH, { inlineParams: true })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  const expect = [
    "let n_toon = ramp_constant_edge_aa(n_str, 0.2966, vec4f(0.0, 0.0, 0.0, 1.0), vec4f(1.0, 1.0, 1.0, 1.0));",
    "let n_shadow_tint = hue_sat(0.46000000834465027, 2.0, 0.3499999940395355, 1.0, n_tex_base);",
    "let n_lit_tint = hue_sat(0.46000000834465027, 1.600000023841858, 1.5, 1.0, n_tex_base);",
    "let n_toon_color = mix_blend(n_toon.r, n_shadow_tint, n_lit_tint);",
    "let n_emission3 = n_bc * 2.5;",
    "let n_warm_mul = math_multiply(n_toon.r, 0.5);",
    "let n_warm_add = math_add(n_warm_mul, 0.5);",
    "let n_warm_emit = n_warm_ramp.rgb * 0.30000001192092896;",
    "let n_rim1 = vec3f(0.984157919883728, 0.6110184788703918, 0.5736401677131653) * n_rim1_str;",
    "let n_rim2_raw = math_multiply(n_rim2_fres, n_rim2_lw);",
    "let n_rim2_pow = math_power(n_rim2_raw, 0.6300000548362732);",
    "let n_rim2_mix = mix(n_emission3, vec3f(1.0, 0.4684903025627136, 0.3698573112487793), n_rim2_pow);",
    "let n_gate = math_greater_than(color_to_value(tex_color), 0.9300000071525574);",
    "let n_npr_add2 = n_npr_add1 + vec3f(n_gate_scale);",
    "let n_map = mapping_point(input.restPos, vec3f(0.0), vec3f(0.0), vec3f(1.0, 1.0, 1.5));",
    "let n_noise = tex_noise_d2(n_map, 1.0);",
    "let n_bump = bump_lh(0.324644535779953, n_noise_ramp.r, n, input.worldPos);",
    "let n_principled_base = mix_blend(n_noise_ramp.r, n_bc, vec3f(0.6832, 0.1947, 0.1373));",
    "eval_principled(PrincipledIn(n_principled_base, 0.0, 0.5, 0.3, 10.0, 0.0, 0.0), n_bump, l, v, sun, amb, shadow)",
    "let n_p_sum = n_principled + n_p_emit;",
    "let n_mix_shader_001 = mix(n_npr_stack, n_p_sum, 0.5);",
  ]
  for (const line of expect) assert.ok(r.fsBody.includes(line), `missing: ${line}`)
})

test("hair graph compiles clean, nothing pruned", () => {
  const r = compileGraph(HAIR_GRAPH, { inlineParams: true })
  assert.equal(r.ok, true)
  assert.deepEqual(r.diagnostics, [])
  assert.deepEqual(r.prunedNodes, [])
})

test("hair inline body matches the hand-written shader (snapshot)", () => {
  const r = compileGraph(HAIR_GRAPH, { inlineParams: true })
  assert.equal(r.fsBody, HAIR_BODY_INLINE)
})

test("hair renderClass: over-eyes override present, style block absent when inlined", () => {
  const r = compileGraph(HAIR_GRAPH, { inlineParams: true, renderClass: "hair" })
  assert.ok(r.wgsl.includes("override IS_OVER_EYES: bool = false;"))
  assert.ok(r.wgsl.includes("if (IS_OVER_EYES) { outAlpha = alpha * 0.25; }"))
  assert.ok(!r.wgsl.includes("StyleUniforms"))
})

// The shipped preset carries no params (slider curation is reze.design's job);
// this variant exposes four sockets to exercise the StyleUniforms path end to end.
const HAIR_WITH_PARAMS = {
  ...HAIR_GRAPH,
  params: [
    { id: "npr_mix", label: "Realism", target: { node: "mix_shader_001", socket: "fac" }, kind: "float", default: 0.2 },
    { id: "rim", label: "Rim Power", target: { node: "rim_pow", socket: "b" }, kind: "float", default: 0.6300000548362732 },
    { id: "shadow_edge", label: "Shadow Edge", target: { node: "ramp_008", socket: "pos1" }, kind: "float", default: 0.2966 },
    { id: "gloss", label: "Gloss", target: { node: "principled", socket: "roughness" }, kind: "float", default: 0.3 },
  ],
}

test("live mode: params become style.p reads, slots pack 4 floats into one vec4", () => {
  const r = compileGraph(HAIR_WITH_PARAMS)
  assert.equal(r.ok, true)
  assert.ok(r.wgsl.includes("struct StyleUniforms { p: array<vec4f, 16> };"))
  assert.deepEqual(
    r.slotMap.map((s) => s.expr),
    ["style.p[0].x", "style.p[0].y", "style.p[0].z", "style.p[0].w"],
  )
  assert.ok(r.fsBody.includes("math_power(n_rim_mul, style.p[0].y)"))
  assert.ok(r.fsBody.includes("mix(n_npr_add, n_principled, style.p[0].x)"))
})

test("param defaults inline to the same WGSL as the paramless preset", () => {
  const a = compileGraph(HAIR_WITH_PARAMS, { inlineParams: true })
  const b = compileGraph(HAIR_GRAPH, { inlineParams: true })
  assert.equal(a.fsBody, b.fsBody)
})

test("compile is deterministic and JSON round-trip is lossless", () => {
  const a = compileGraph(HAIR_GRAPH)
  const b = compileGraph(JSON.parse(JSON.stringify(HAIR_GRAPH)))
  assert.equal(a.wgsl, b.wgsl)
})

test("previewNode overrides output and prunes downstream work", () => {
  const r = compileGraph(HAIR_GRAPH, { inlineParams: true, previewNode: { node: "rim_pow", socket: "value" } })
  assert.equal(r.ok, true)
  assert.ok(r.fsBody.endsWith("let final_color = vec3f(n_rim_pow); // @node:rim_pow"))
  assert.ok(r.prunedNodes.includes("principled"))
  assert.ok(r.prunedNodes.includes("mix_shader_001"))
  assert.ok(!r.fsBody.includes("eval_principled"))
})

test("exposing a hue slider disables the hue_sat_id specialization", () => {
  const graph = {
    version: 1,
    name: "t",
    nodes: [
      { id: "tex", type: "texture" },
      { id: "hs", type: "hue_sat", inputs: { hue: 0.5, saturation: 1.2, value: 1.0, fac: 1.0 } },
    ],
    links: [{ from: { node: "tex", socket: "color" }, to: { node: "hs", socket: "color" } }],
    output: { node: "hs", socket: "color" },
    params: [
      { id: "hue", label: "Hue", target: { node: "hs", socket: "hue" }, kind: "float", default: 0.5 },
    ],
  }
  const live = compileGraph(graph)
  assert.ok(live.fsBody.includes("hue_sat(style.p[0].x, 1.2, 1.0, 1.0, tex_color)"))
  const inline = compileGraph(graph, { inlineParams: true })
  assert.ok(inline.fsBody.includes("hue_sat_id(1.2, 1.0, 1.0, tex_color)"))
})

test("mix with literal fac 0/1 collapses to a passthrough", () => {
  const graph = {
    version: 1,
    name: "t",
    nodes: [
      { id: "tex", type: "texture" },
      { id: "m0", type: "mix/blend", inputs: { fac: 0, b: [1, 0, 0] } },
      { id: "m1", type: "mix/blend", inputs: { fac: 1, b: [0, 1, 0] } },
    ],
    links: [
      { from: { node: "tex", socket: "color" }, to: { node: "m0", socket: "a" } },
      { from: { node: "m0", socket: "color" }, to: { node: "m1", socket: "a" } },
    ],
    output: { node: "m1", socket: "color" },
  }
  const r = compileGraph(graph, { inlineParams: true })
  assert.ok(r.fsBody.includes("let n_m0 = tex_color;"))
  assert.ok(r.fsBody.includes("let n_m1 = vec3f(0.0, 1.0, 0.0);"))
})

test("cycles are rejected with the cycle named", () => {
  const graph = {
    version: 1,
    name: "t",
    nodes: [
      { id: "a", type: "mix/blend", inputs: { b: [0, 0, 0] } },
      { id: "b", type: "mix/blend", inputs: { b: [0, 0, 0] } },
    ],
    links: [
      { from: { node: "a", socket: "color" }, to: { node: "b", socket: "a" } },
      { from: { node: "b", socket: "color" }, to: { node: "a", socket: "a" } },
    ],
    output: { node: "b", socket: "color" },
  }
  const r = compileGraph(graph)
  assert.equal(r.ok, false)
  assert.ok(r.diagnostics.some((d) => d.message.includes("cycle")))
})

test("validation names the offending node", () => {
  const graph = {
    version: 1,
    name: "t",
    nodes: [
      { id: "tex", type: "texture" },
      { id: "hs", type: "hue_sat" },
      { id: "bad", type: "does_not_exist" },
    ],
    links: [
      { from: { node: "tex", socket: "color" }, to: { node: "hs", socket: "color" } },
      { from: { node: "tex", socket: "nope", }, to: { node: "hs", socket: "fac" } },
      { from: { node: "geo_missing", socket: "normal" }, to: { node: "hs", socket: "hue" } },
    ],
    output: { node: "hs", socket: "color" },
    params: [
      { id: "p", label: "P", target: { node: "hs", socket: "color" }, kind: "color", default: [1, 1, 1] },
    ],
  }
  const d = validateGraph(graph)
  assert.ok(d.some((x) => x.nodeId === "bad" && x.message.includes("unknown node type")))
  assert.ok(d.some((x) => x.message.includes('no output socket "nope"')))
  assert.ok(d.some((x) => x.message.includes('"geo_missing" doesn\'t exist')))
  assert.ok(d.some((x) => x.message.includes("sliders only override literals")))
})

test("vector cannot implicitly feed a float socket (not a Blender conversion)", () => {
  const graph = {
    version: 1,
    name: "t",
    nodes: [
      { id: "geo", type: "geometry" },
      { id: "m", type: "math/clamp01" },
    ],
    links: [{ from: { node: "geo", socket: "normal" }, to: { node: "m", socket: "a" } }],
    output: { node: "m", socket: "value" },
  }
  const r = compileGraph(graph)
  assert.equal(r.ok, false)
  assert.ok(r.diagnostics.some((d) => d.message.includes("type mismatch")))
})

test("light exposes the sun as values a ported NPR graph can build its own diffuse from", () => {
  // Blender NPR presets don't call a diffuse closure — they ramp dot(N, L)
  // themselves. That needs the direction as a value, which only this node gives.
  const graph = {
    version: 1,
    name: "t",
    nodes: [
      { id: "geo", type: "geometry" },
      { id: "lt", type: "light" },
      { id: "d", type: "vector_math/dot" },
      { id: "half", type: "math/multiply_add", inputs: { b: 0.5, c: 0.5 } },
      { id: "tint", type: "mix/multiply", inputs: { fac: 1 } },
    ],
    links: [
      { from: { node: "geo", socket: "normal" }, to: { node: "d", socket: "a" } },
      { from: { node: "lt", socket: "direction" }, to: { node: "d", socket: "b" } },
      { from: { node: "d", socket: "value" }, to: { node: "half", socket: "a" } },
      { from: { node: "half", socket: "value" }, to: { node: "tint", socket: "a" } },
      { from: { node: "lt", socket: "color" }, to: { node: "tint", socket: "b" } },
    ],
    output: { node: "tint", socket: "color" },
  }
  const r = compileGraph(graph)
  assert.deepEqual(r.diagnostics, [])
  // The four sockets read the slot template's own locals, so no node is emitted
  // for the light itself.
  assert.ok(r.fsBody.includes("vector_dot(n, l)"))
  assert.ok(r.fsBody.includes("sun"))
  assert.ok(!r.fsBody.includes("n_lt ="))
})

test("light's shadow and ambient reach a graph that wants to tint its own shadow", () => {
  const graph = {
    version: 1,
    name: "t",
    nodes: [
      { id: "lt", type: "light" },
      { id: "mix", type: "mix/blend" },
      { id: "tex", type: "texture" },
    ],
    links: [
      { from: { node: "lt", socket: "shadow" }, to: { node: "mix", socket: "fac" } },
      { from: { node: "lt", socket: "ambient" }, to: { node: "mix", socket: "a" } },
      { from: { node: "tex", socket: "color" }, to: { node: "mix", socket: "b" } },
    ],
    output: { node: "mix", socket: "color" },
  }
  const r = compileGraph(graph)
  assert.deepEqual(r.diagnostics, [])
  assert.ok(r.fsBody.includes("mix_blend(shadow, amb, tex_color)"))
})

test("a math op accepts all three of Blender's Value sockets, used or not", () => {
  // Sockets belong to the node, not the operation. A transcriber maps them
  // socket-for-socket without knowing which ones the op reads.
  const graph = {
    version: 1,
    name: "t",
    nodes: [
      { id: "tex", type: "texture" },
      { id: "m", type: "math/add", inputs: { b: 0.25, c: 9 } },
      { id: "mix", type: "mix/multiply", inputs: { fac: 1 } },
    ],
    links: [
      { from: { node: "tex", socket: "alpha" }, to: { node: "m", socket: "a" } },
      { from: { node: "tex", socket: "color" }, to: { node: "mix", socket: "a" } },
      { from: { node: "m", socket: "value" }, to: { node: "mix", socket: "b" } },
    ],
    output: { node: "mix", socket: "color" },
  }
  const r = compileGraph(graph)
  assert.deepEqual(r.diagnostics, [])
  // c is inert on add — declared, unread, and absent from the emission.
  assert.ok(r.fsBody.includes("math_add(tex_s.a, 0.25)"))
  assert.ok(!r.fsBody.includes("9.0"))
})

test("divide keeps its own default when the parity pass adds the sockets it lacked", () => {
  const r = compileGraph({
    version: 1,
    name: "t",
    nodes: [
      { id: "tex", type: "texture" },
      { id: "d", type: "math/divide" },
      { id: "mix", type: "mix/multiply", inputs: { fac: 1 } },
    ],
    links: [
      { from: { node: "tex", socket: "alpha" }, to: { node: "d", socket: "a" } },
      { from: { node: "tex", socket: "color" }, to: { node: "mix", socket: "a" } },
      { from: { node: "d", socket: "value" }, to: { node: "mix", socket: "b" } },
    ],
    output: { node: "mix", socket: "color" },
  })
  assert.deepEqual(r.diagnostics, [])
  // Not 0.0 — widening must not overwrite a declared default.
  assert.ok(r.fsBody.includes("math_divide(tex_s.a, 1.0)"))
})

test("an RGBA literal fits a colour socket, dropping alpha as Blender does", () => {
  const r = compileGraph({
    version: 1,
    name: "t",
    nodes: [
      { id: "tex", type: "texture" },
      { id: "mix", type: "mix/multiply", inputs: { fac: 1, b: [0.25, 0.5, 0.75, 1] } },
    ],
    links: [{ from: { node: "tex", socket: "color" }, to: { node: "mix", socket: "a" } }],
    output: { node: "mix", socket: "color" },
  })
  assert.deepEqual(r.diagnostics, [])
  assert.ok(r.fsBody.includes("vec3f(0.25, 0.5, 0.75)"))
})

test("a ramp stop keeps its alpha, since vec4 is the one socket that has one", () => {
  const r = compileGraph({
    version: 1,
    name: "t",
    nodes: [
      { id: "lt", type: "light" },
      { id: "geo", type: "geometry" },
      { id: "d", type: "vector_math/dot" },
      { id: "r", type: "ramp_linear", inputs: { color1: [1, 0, 0, 0.5] } },
    ],
    links: [
      { from: { node: "geo", socket: "normal" }, to: { node: "d", socket: "a" } },
      { from: { node: "lt", socket: "direction" }, to: { node: "d", socket: "b" } },
      { from: { node: "d", socket: "value" }, to: { node: "r", socket: "fac" } },
    ],
    output: { node: "r", socket: "color" },
  })
  assert.deepEqual(r.diagnostics, [])
  assert.ok(r.fsBody.includes("vec4f(1.0, 0.0, 0.0, 0.5)"))
})

test("a colour literal still cannot sit on a float socket", () => {
  const r = compileGraph({
    version: 1,
    name: "t",
    nodes: [
      { id: "tex", type: "texture" },
      { id: "m", type: "math/add", inputs: { b: [1, 0, 0, 1] } },
      { id: "mix", type: "mix/multiply", inputs: { fac: 1 } },
    ],
    links: [
      { from: { node: "tex", socket: "alpha" }, to: { node: "m", socket: "a" } },
      { from: { node: "tex", socket: "color" }, to: { node: "mix", socket: "a" } },
      { from: { node: "m", socket: "value" }, to: { node: "mix", socket: "b" } },
    ],
    output: { node: "mix", socket: "color" },
  })
  assert.equal(r.ok, false)
  assert.ok(r.diagnostics.some((d) => d.message.includes("doesn't fit float socket")))
})
