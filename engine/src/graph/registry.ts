// Node registry: one entry per Blender-equivalent node backed by NODES_WGSL (nodes.ts).
// The registry adds no WGSL of its own — a node type exists here only if its function
// already exists (validated against EEVEE) in the shader library, or is a WGSL builtin.
// Semantics are frozen Blender 3.6 legacy-EEVEE; enum modes (math op, mix blend type,
// ramp interpolation) are part of the type string so they are unambiguously topology.

import type { SocketValue } from "./schema"

export type SockT = "float" | "color" | "vector" | "vec4"

export type InputSpec = {
  type: SockT
  default?: SocketValue
  /** Socket is meaningless as a literal (e.g. the color being processed) — must be linked. */
  requiresLink?: boolean
  /** Unlinked fallback is a template local, not a literal (e.g. principled.normal → n). */
  contextDefault?: string
}

export type NodeSpec = {
  inputs: Record<string, InputSpec>
  outputs: Record<string, SockT>
  /** RHS expression for this node's `let`, from resolved arg expressions keyed by socket. */
  emit?: (args: Record<string, string>) => string
  /** Context nodes tap template locals directly — no `let` emitted. */
  contextOutputs?: Record<string, string>
  /** Swizzle applied to the node's variable per output socket (default: none). */
  outputSelect?: Record<string, string>
}

// ─── Literal formatting ───────────────────────────────────────────────
// Deterministic: same graph JSON → byte-identical WGSL. String(x) is JS shortest
// round-trip, so full-precision Blender constants (0.15000000596046448) survive.

export function fmtFloat(x: number): string {
  if (!Number.isFinite(x)) throw new Error(`non-finite literal: ${x}`)
  const s = String(x)
  return /[.e]/.test(s) ? s : s + ".0"
}

export function fmtValue(value: SocketValue, type: SockT): string {
  if (typeof value === "number") {
    if (type === "float") return fmtFloat(value)
    if (type === "color" || type === "vector") return `vec3f(${fmtFloat(value)})`
    return `vec4f(vec3f(${fmtFloat(value)}), 1.0)`
  }
  if (value.length === 3) {
    const [x, y, z] = value
    if (type === "vec4") return `vec4f(${fmtFloat(x)}, ${fmtFloat(y)}, ${fmtFloat(z)}, 1.0)`
    if (type === "float") throw new Error(`vector literal on float socket`)
    // All-equal shorthand matches the hand-written shaders (vec3f(0.167…)).
    if (x === y && y === z) return `vec3f(${fmtFloat(x)})`
    return `vec3f(${fmtFloat(x)}, ${fmtFloat(y)}, ${fmtFloat(z)})`
  }
  if (type !== "vec4") throw new Error(`vec4 literal on ${type} socket`)
  return `vec4f(${value.map(fmtFloat).join(", ")})`
}

/** Does a literal's shape fit a socket type? (Scalar splats onto color/vector/vec4.) */
export function literalFits(value: SocketValue, type: SockT): boolean {
  if (typeof value === "number") return true
  if (value.length === 3) return type !== "float"
  return type === "vec4"
}

// ─── Implicit socket conversions (Blender-faithful) ──────────────────
// vec4 appears only on ramp stop-color literals — never linkable, so conversions
// cover float/color/vector. vector→float is NOT implicit in Blender; rejected.

export function canConvert(from: SockT, to: SockT): boolean {
  if (from === to) return true
  if (from === "color" && to === "float") return true // BT.601 via color_to_value
  if (from === "float" && (to === "color" || to === "vector")) return true
  if ((from === "color" && to === "vector") || (from === "vector" && to === "color")) return true
  return false
}

export function convert(from: SockT, to: SockT, expr: string): string {
  if (from === to) return expr
  if (from === "color" && to === "float") return `color_to_value(${expr})`
  if (from === "float") return `vec3f(${expr})`
  return expr // color ↔ vector: bit-identical vec3f pass-through
}

// ─── Registry ─────────────────────────────────────────────────────────

const F = (d: number, requiresLink = false): InputSpec => ({ type: "float", default: d, requiresLink })
const C = (d: [number, number, number] = [1, 1, 1], requiresLink = false): InputSpec => ({
  type: "color",
  default: d,
  requiresLink,
})
const V = (d: [number, number, number] = [0, 0, 0], requiresLink = false): InputSpec => ({
  type: "vector",
  default: d,
  requiresLink,
})
const V4 = (d: [number, number, number, number]): InputSpec => ({ type: "vec4", default: d })

const RAMP_INPUTS: Record<string, InputSpec> = {
  fac: F(0.5, true),
  pos0: F(0),
  color0: V4([0, 0, 0, 1]),
  pos1: F(1),
  color1: V4([1, 1, 1, 1]),
}
const RAMP_OUTPUTS: Record<string, SockT> = { color: "color", alpha: "float", fac_out: "float" }
// fac_out (.r) matches how a grayscale ramp feeds a scalar consumer in the hand ports —
// routing through the BT.601 color→float conversion instead would change the value.
const RAMP_SELECT = { color: ".rgb", alpha: ".a", fac_out: ".r" }

export const NODE_REGISTRY: Record<string, NodeSpec> = {
  // ── Context inputs (template locals; no emission) ──
  texture: {
    inputs: {},
    outputs: { color: "color", alpha: "float" },
    contextOutputs: { color: "tex_color", alpha: "tex_s.a" },
  },
  geometry: {
    inputs: {},
    outputs: {
      normal: "vector",
      view: "vector",
      world_pos: "vector",
      rest_pos: "vector",
      uv: "vector",
      // Blender Texture Coordinate → Reflection (view ray mirrored on the normal);
      // drives env-tracking patterns like metal's voronoi sparkle.
      reflection: "vector",
    },
    contextOutputs: {
      normal: "n",
      view: "v",
      world_pos: "input.worldPos",
      rest_pos: "input.restPos",
      uv: "vec3f(input.uv, 0.0)",
      reflection: "reflect(-v, n)",
    },
  },
  // PMX material's diffuse color (the authored base tint). Multiply the diffuse texture
  // by this for the MMD-correct base — untextured materials carry their color here, so a
  // texture-only base would render them white.
  material_diffuse: {
    inputs: {},
    outputs: { color: "color" },
    contextOutputs: { color: "material.diffuseColor" },
  },

  // The PMX material's sphere map, which is where an MMD model keeps its
  // highlights — every PMX ships one, and hair without it reads flat. The mode
  // is the material's own (.sph multiplies the shaded base, .spa adds a
  // highlight), so a graph asks for the effect and the model decides which; a
  // material with no sphere texture is an exact no-op.
  sphere_map: {
    inputs: { base: C([0, 0, 0], true), strength: F(1) },
    outputs: { color: "color" },
    emit: (a) => `pmx_sphere_map(${a.base}, ${a.strength}, n)`,
  },

  // ── Literals as nodes (for editor ergonomics; inlined literals work too) ──
  value: { inputs: { value: F(0) }, outputs: { value: "float" }, emit: (a) => a.value },
  rgb: { inputs: { color: C() }, outputs: { color: "color" }, emit: (a) => a.color },

  // ── Color ──
  hue_sat: {
    inputs: { hue: F(0.5), saturation: F(1), value: F(1), fac: F(1), color: C([1, 1, 1], true) },
    outputs: { color: "color" },
    emit: (a) => `hue_sat(${a.hue}, ${a.saturation}, ${a.value}, ${a.fac}, ${a.color})`,
  },
  bright_contrast: {
    inputs: { color: C([1, 1, 1], true), bright: F(0), contrast: F(0) },
    outputs: { color: "color" },
    emit: (a) => `bright_contrast(${a.color}, ${a.bright}, ${a.contrast})`,
  },
  invert: {
    inputs: { fac: F(1), color: C([1, 1, 1], true) },
    outputs: { color: "color" },
    emit: (a) => `invert(${a.fac}, ${a.color})`,
  },
  ramp_constant: {
    inputs: RAMP_INPUTS,
    outputs: RAMP_OUTPUTS,
    outputSelect: RAMP_SELECT,
    emit: (a) => `ramp_constant(${a.fac}, ${a.pos0}, ${a.color0}, ${a.pos1}, ${a.color1})`,
  },
  ramp_linear: {
    inputs: RAMP_INPUTS,
    outputs: RAMP_OUTPUTS,
    outputSelect: RAMP_SELECT,
    emit: (a) => `ramp_linear(${a.fac}, ${a.pos0}, ${a.color0}, ${a.pos1}, ${a.color1})`,
  },
  ramp_cardinal: {
    inputs: RAMP_INPUTS,
    outputs: RAMP_OUTPUTS,
    outputSelect: RAMP_SELECT,
    emit: (a) => `ramp_cardinal(${a.fac}, ${a.pos0}, ${a.color0}, ${a.pos1}, ${a.color1})`,
  },
  ramp_constant_aa: {
    inputs: { fac: F(0.5, true), edge: F(0.5), color0: V4([0, 0, 0, 1]), color1: V4([1, 1, 1, 1]) },
    outputs: RAMP_OUTPUTS,
    outputSelect: RAMP_SELECT,
    emit: (a) => `ramp_constant_edge_aa(${a.fac}, ${a.edge}, ${a.color0}, ${a.color1})`,
  },
  // Blender ColorRamp LINEAR with 3 stops black→white→black (triangular peak at 0.5).
  // The only >2-stop ramp the presets use; folded to closed form like the hand port.
  ramp_tri: {
    inputs: { fac: F(0.5, true) },
    outputs: { value: "float" },
    emit: (a) => `1.0 - abs(2.0 * ${a.fac} - 1.0)`,
  },

  // ── Math (enum op in type string) ──
  "math/add": { inputs: { a: F(0), b: F(0) }, outputs: { value: "float" }, emit: (a) => `math_add(${a.a}, ${a.b})` },
  "math/multiply": {
    inputs: { a: F(0), b: F(0) },
    outputs: { value: "float" },
    emit: (a) => `math_multiply(${a.a}, ${a.b})`,
  },
  "math/power": {
    inputs: { a: F(0), b: F(1) },
    outputs: { value: "float" },
    emit: (a) => `math_power(${a.a}, ${a.b})`,
  },
  "math/greater_than": {
    inputs: { a: F(0), b: F(0.5) },
    outputs: { value: "float" },
    emit: (a) => `math_greater_than(${a.a}, ${a.b})`,
  },
  "math/clamp01": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `saturate(${a.a})` },

  // ── Mix (blend type in type string) ──
  "mix/blend": {
    inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() },
    outputs: { color: "color" },
    emit: (a) => `mix_blend(${a.fac}, ${a.a}, ${a.b})`,
  },
  "mix/overlay": {
    inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() },
    outputs: { color: "color" },
    emit: (a) => `mix_overlay(${a.fac}, ${a.a}, ${a.b})`,
  },
  "mix/multiply": {
    inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() },
    outputs: { color: "color" },
    emit: (a) => `mix_multiply(${a.fac}, ${a.a}, ${a.b})`,
  },
  "mix/lighten": {
    inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() },
    outputs: { color: "color" },
    emit: (a) => `mix_lighten(${a.fac}, ${a.a}, ${a.b})`,
  },
  "mix/linear_light": {
    inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() },
    outputs: { color: "color" },
    emit: (a) => `mix_linear_light(${a.fac}, ${a.a}, ${a.b})`,
  },
  // Emission-add: color + scalar-gated emission. Stand-in for Blender's
  // Emission → Add Shader pair in the ShaderToRGB era (see hair's bright-tex gate).
  "mix/add_emit": {
    inputs: { a: C([0, 0, 0], true), b: F(0) },
    outputs: { color: "color" },
    emit: (a) => `${a.a} + vec3f(${a.b})`,
  },
  // Blender Emission shader: color × strength. In the ShaderToRGB-era ports the
  // emission result feeds a Mix Shader directly as radiance. Color may be a literal
  // (body's rim tints) or linked.
  emission: {
    inputs: { color: C([1, 1, 1]), strength: F(1) },
    outputs: { color: "color" },
    emit: (a) => `${a.color} * ${a.strength}`,
  },
  // Blender Add Shader — radiance sum of two evaluated shading results.
  add_shader: {
    inputs: { a: C([0, 0, 0], true), b: C([0, 0, 0], true) },
    outputs: { color: "color" },
    emit: (a) => `${a.a} + ${a.b}`,
  },
  // Mix Shader — plain lerp between two evaluated shading results.
  mix_shader: {
    inputs: { fac: F(0.5), a: C([0, 0, 0], true), b: C([0, 0, 0]) },
    outputs: { color: "color" },
    emit: (a) => `mix(${a.a}, ${a.b}, ${a.fac})`,
  },

  // ── View-dependent scalars ──
  fresnel: { inputs: { ior: F(1.45) }, outputs: { value: "float" }, emit: (a) => `fresnel(${a.ior}, n, v)` },
  "layer_weight/fresnel": {
    inputs: { blend: F(0.5) },
    outputs: { value: "float" },
    emit: (a) => `layer_weight_fresnel(${a.blend}, n, v)`,
  },
  "layer_weight/facing": {
    inputs: { blend: F(0.5) },
    outputs: { value: "float" },
    emit: (a) => `layer_weight_facing(${a.blend}, n, v)`,
  },

  // ── Lighting capture ──
  shader_to_rgb_diffuse: {
    inputs: {},
    outputs: { value: "float" },
    emit: () => `shader_to_rgb_diffuse(n, l, sun, amb, shadow)`,
  },

  // ── Vector ──
  separate_xyz: {
    inputs: { vector: V([0, 0, 0], true) },
    outputs: { x: "float", y: "float", z: "float" },
    outputSelect: { x: ".x", y: ".y", z: ".z" },
    emit: (a) => a.vector,
  },
  vect_cross: {
    // b may be a literal constant (metal crosses the reflection dir with (0,1,0)).
    inputs: { a: V([0, 0, 0], true), b: V([0, 1, 0]) },
    outputs: { vector: "vector" },
    emit: (a) => `vect_math_cross(${a.a}, ${a.b})`,
  },
  mapping: {
    inputs: { vector: V([0, 0, 0], true), loc: V([0, 0, 0]), rot: V([0, 0, 0]), scl: V([1, 1, 1]) },
    outputs: { vector: "vector" },
    emit: (a) => `mapping_point(${a.vector}, ${a.loc}, ${a.rot}, ${a.scl})`,
  },
  bump: {
    // Screen-space bump; world position comes from context (matches bump_lh's port).
    inputs: { strength: F(0.1), height: F(0, true), normal: V([0, 0, 0], true) },
    outputs: { vector: "vector" },
    emit: (a) => `bump_lh(${a.strength}, ${a.height}, ${a.normal}, input.worldPos)`,
  },

  // ── Procedural textures ──
  tex_noise: {
    inputs: { vector: V([0, 0, 0], true), scale: F(5), detail: F(2), roughness: F(0.5), distortion: F(0) },
    outputs: { value: "float" },
    emit: (a) => `tex_noise(${a.vector}, ${a.scale}, ${a.detail}, ${a.roughness}, ${a.distortion})`,
  },
  tex_gradient: {
    inputs: { vector: V([0, 0, 0], true) },
    outputs: { value: "float" },
    emit: (a) => `tex_gradient_linear(${a.vector})`,
  },
  "tex_voronoi/f1": {
    inputs: { vector: V([0, 0, 0], true), scale: F(5) },
    outputs: { value: "float" },
    emit: (a) => `tex_voronoi_f1(${a.vector}, ${a.scale})`,
  },
  "tex_voronoi/color": {
    inputs: { vector: V([0, 0, 0], true), scale: F(5) },
    outputs: { color: "color" },
    emit: (a) => `tex_voronoi_color(${a.vector}, ${a.scale})`,
  },

  // ── Principled BSDF (frozen 3.6 legacy-EEVEE semantics: eval_principled port) ──
  // normal defaults to the template's shading normal; link a bump/normal_map chain
  // to perturb it (body/cloth_rough noise bump).
  principled: {
    inputs: {
      base: C([0.8, 0.8, 0.8], true),
      metallic: F(0),
      specular: F(0.5),
      roughness: F(0.5),
      spec_clamp: F(1e30),
      sheen: F(0),
      sheen_tint: F(0),
      normal: { type: "vector", contextDefault: "n" },
    },
    outputs: { color: "color" },
    emit: (a) =>
      `eval_principled(PrincipledIn(${a.base}, ${a.metallic}, ${a.specular}, ${a.roughness}, ${a.spec_clamp}, ${a.sheen}, ${a.sheen_tint}), ${a.normal}, l, v, sun, amb, shadow)`,
  },
}
