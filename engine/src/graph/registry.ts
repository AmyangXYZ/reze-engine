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
  if (type === "float") throw new Error(`color literal on float socket`)
  // Blender's colour sockets are RGBA, so a ported literal arrives with four
  // components whatever it feeds. Only a stop colour keeps the alpha; every other
  // socket here is a vec3 and drops it, which is what Blender does downstream.
  if (type !== "vec4") return fmtValue([value[0], value[1], value[2]], type)
  return `vec4f(${value.map(fmtFloat).join(", ")})`
}

/** Does a literal's shape fit a socket type? (Scalar splats onto color/vector/vec4.) */
export function literalFits(value: SocketValue, type: SockT): boolean {
  if (typeof value === "number") return true
  // 3 and 4 components both fit anything but a float — see fmtValue on RGBA.
  return type !== "float"
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

/** Principled v2 reflectance, evaluated at compile time when it can be. */
function foldSpecular(ior: string, level: string): string {
  const i = Number(ior)
  const l = Number(level)
  if (!Number.isFinite(i) || !Number.isFinite(l)) return `principled_specular(${ior}, ${level})`
  const r = (i - 1) / Math.max(i + 1, 1e-6)
  // Snapped to 9 significant digits: the arithmetic leaves double noise
  // (0.2² · 2 · 0.5 / 0.08 = 0.5000000000000001) that is meaningless in an f32
  // and only makes the emitted shader harder to read.
  return fmtFloat(Number(((r * r * 2 * Math.max(l, 0)) / 0.08).toPrecision(9)))
}

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
  // The scene's key light. Blender NPR presets rarely use a diffuse closure —
  // they build their own term, typically dot(normal, direction) pushed through a
  // ramp or a soft threshold band, because that is what gives an anime shader its
  // hard terminator. Reaching that term needs the direction as a value, which no
  // other node exposes: `shader_to_rgb*` and `bsdf_diffuse` bake the whole closure
  // and hand back a result. `shadow` is the same cascade sample those closures
  // take, so a graph can tint its own shadow instead of accepting theirs.
  //
  // A ported graph reads `direction` where the Blender original read an Attribute
  // fed by a light empty; the empty and our sun mean the same thing.
  light: {
    inputs: {},
    outputs: { direction: "vector", color: "color", ambient: "color", shadow: "float" },
    contextOutputs: { direction: "l", color: "sun", ambient: "amb", shadow: "shadow" },
  },

  // The head bone's world basis — what an SDF face shadow is built on.
  //
  // That technique compares a face-shaped distance field against the light's
  // angle IN THE HEAD'S OWN FRAME, so the shadow sweeps across the face as the
  // light moves and stays put as the head turns. Without the frame there is
  // nothing to measure the angle against, and the effect cannot be expressed at
  // all. `right` also carries the sign that mirrors the field's U coordinate,
  // which is how one half-face texture serves both sides.
  //
  // Read from the 頭 bone's skinning matrix, already bound to the fragment stage
  // for the eye's rear-view gate. A model without that bone falls back to bone 0
  // rather than sampling out of bounds; the face shadow is then wrong, not unsafe.
  head_basis: {
    inputs: {},
    outputs: { forward: "vector", right: "vector", up: "vector" },
    contextOutputs: {
      forward: "(-normalize(skinMats[u32(max(material.headBoneIndex, 0.0))][2].xyz))",
      right: "normalize(skinMats[u32(max(material.headBoneIndex, 0.0))][0].xyz)",
      up: "normalize(skinMats[u32(max(material.headBoneIndex, 0.0))][1].xyz)",
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

  // ── Blender 5.2 Math node, full operation set ──
  "math/absolute": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_absolute(${a.a})` },
  "math/sqrt": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_sqrt(${a.a})` },
  "math/inversesqrt": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_inversesqrt(${a.a})` },
  "math/exponent": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_exponent(${a.a})` },
  "math/sign": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_sign(${a.a})` },
  "math/round": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_round(${a.a})` },
  "math/floor": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_floor(${a.a})` },
  "math/ceil": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_ceil(${a.a})` },
  "math/truncate": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_truncate(${a.a})` },
  "math/fraction": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_fraction(${a.a})` },
  "math/sine": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_sine(${a.a})` },
  "math/cosine": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_cosine(${a.a})` },
  "math/tangent": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_tangent(${a.a})` },
  "math/arcsine": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_arcsine(${a.a})` },
  "math/arccosine": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_arccosine(${a.a})` },
  "math/arctangent": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_arctangent(${a.a})` },
  "math/radians": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_radians(${a.a})` },
  "math/degrees": { inputs: { a: F(0) }, outputs: { value: "float" }, emit: (a) => `math_degrees(${a.a})` },
  "math/subtract": { inputs: { a: F(0), b: F(0) }, outputs: { value: "float" }, emit: (a) => `math_subtract(${a.a}, ${a.b})` },
  "math/divide": { inputs: { a: F(0), b: F(1) }, outputs: { value: "float" }, emit: (a) => `math_divide(${a.a}, ${a.b})` },
  "math/logarithm": { inputs: { a: F(0), b: F(1) }, outputs: { value: "float" }, emit: (a) => `math_logarithm(${a.a}, ${a.b})` },
  "math/minimum": { inputs: { a: F(0), b: F(0) }, outputs: { value: "float" }, emit: (a) => `math_minimum(${a.a}, ${a.b})` },
  "math/maximum": { inputs: { a: F(0), b: F(0) }, outputs: { value: "float" }, emit: (a) => `math_maximum(${a.a}, ${a.b})` },
  "math/less_than": { inputs: { a: F(0), b: F(0) }, outputs: { value: "float" }, emit: (a) => `math_less_than(${a.a}, ${a.b})` },
  "math/modulo": { inputs: { a: F(0), b: F(1) }, outputs: { value: "float" }, emit: (a) => `math_modulo(${a.a}, ${a.b})` },
  "math/floored_modulo": { inputs: { a: F(0), b: F(1) }, outputs: { value: "float" }, emit: (a) => `math_floored_modulo(${a.a}, ${a.b})` },
  "math/snap": { inputs: { a: F(0), b: F(1) }, outputs: { value: "float" }, emit: (a) => `math_snap(${a.a}, ${a.b})` },
  "math/pingpong": { inputs: { a: F(0), b: F(1) }, outputs: { value: "float" }, emit: (a) => `math_pingpong(${a.a}, ${a.b})` },
  "math/arctan2": { inputs: { a: F(0), b: F(0) }, outputs: { value: "float" }, emit: (a) => `math_arctan2(${a.a}, ${a.b})` },
  "math/multiply_add": { inputs: { a: F(0), b: F(0), c: F(0) }, outputs: { value: "float" }, emit: (a) => `math_multiply_add(${a.a}, ${a.b}, ${a.c})` },
  "math/compare": { inputs: { a: F(0), b: F(0), c: F(0) }, outputs: { value: "float" }, emit: (a) => `math_compare(${a.a}, ${a.b}, ${a.c})` },
  "math/smooth_min": { inputs: { a: F(0), b: F(0), c: F(0) }, outputs: { value: "float" }, emit: (a) => `math_smooth_min(${a.a}, ${a.b}, ${a.c})` },
  "math/smooth_max": { inputs: { a: F(0), b: F(0), c: F(0) }, outputs: { value: "float" }, emit: (a) => `math_smooth_max(${a.a}, ${a.b}, ${a.c})` },
  "math/wrap": { inputs: { a: F(0), b: F(0), c: F(0) }, outputs: { value: "float" }, emit: (a) => `math_wrap(${a.a}, ${a.b}, ${a.c})` },

  // ── Blender 5.2 Vector Math node ──
  "vector_math/normalize": { inputs: { a: V([0, 0, 0], true) }, outputs: { vector: "vector" }, emit: (a) => `vector_normalize(${a.a})` },
  "vector_math/absolute": { inputs: { a: V([0, 0, 0], true) }, outputs: { vector: "vector" }, emit: (a) => `vector_absolute(${a.a})` },
  "vector_math/floor": { inputs: { a: V([0, 0, 0], true) }, outputs: { vector: "vector" }, emit: (a) => `vector_floor(${a.a})` },
  "vector_math/ceil": { inputs: { a: V([0, 0, 0], true) }, outputs: { vector: "vector" }, emit: (a) => `vector_ceil(${a.a})` },
  "vector_math/fraction": { inputs: { a: V([0, 0, 0], true) }, outputs: { vector: "vector" }, emit: (a) => `vector_fraction(${a.a})` },
  "vector_math/add": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_add(${a.a}, ${a.b})` },
  "vector_math/subtract": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_subtract(${a.a}, ${a.b})` },
  "vector_math/multiply": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_multiply(${a.a}, ${a.b})` },
  "vector_math/divide": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_divide(${a.a}, ${a.b})` },
  "vector_math/cross": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_cross(${a.a}, ${a.b})` },
  "vector_math/project": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_project(${a.a}, ${a.b})` },
  "vector_math/reflect": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_reflect(${a.a}, ${a.b})` },
  "vector_math/minimum": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_minimum(${a.a}, ${a.b})` },
  "vector_math/maximum": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_maximum(${a.a}, ${a.b})` },
  "vector_math/modulo": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_modulo(${a.a}, ${a.b})` },
  "vector_math/snap": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_snap(${a.a}, ${a.b})` },
  "vector_math/dot": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { value: "float" }, emit: (a) => `vector_dot(${a.a}, ${a.b})` },
  "vector_math/distance": { inputs: { a: V([0, 0, 0], true), b: V() }, outputs: { value: "float" }, emit: (a) => `vector_distance(${a.a}, ${a.b})` },
  "vector_math/length": { inputs: { a: V([0, 0, 0], true) }, outputs: { value: "float" }, emit: (a) => `vector_length(${a.a})` },
  "vector_math/scale": { inputs: { a: V([0, 0, 0], true), scale: F(1) }, outputs: { vector: "vector" }, emit: (a) => `vector_scale(${a.a}, ${a.scale})` },
  "vector_math/multiply_add": { inputs: { a: V([0, 0, 0], true), b: V(), c: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_multiply_add(${a.a}, ${a.b}, ${a.c})` },
  "vector_math/faceforward": { inputs: { a: V([0, 0, 0], true), b: V(), c: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_faceforward(${a.a}, ${a.b}, ${a.c})` },
  "vector_math/refract": { inputs: { a: V([0, 0, 0], true), b: V(), ior: F(1.45) }, outputs: { vector: "vector" }, emit: (a) => `vector_refract(${a.a}, ${a.b}, ${a.ior})` },
  "vector_math/wrap": { inputs: { a: V([0, 0, 0], true), b: V(), c: V() }, outputs: { vector: "vector" }, emit: (a) => `vector_wrap(${a.a}, ${a.b}, ${a.c})` },

  // ── Blender 5.2 Mix (Color) blend set ──
  "mix/add": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_add(${a.fac}, ${a.a}, ${a.b})` },
  "mix/subtract": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_subtract(${a.fac}, ${a.a}, ${a.b})` },
  "mix/darken": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_darken(${a.fac}, ${a.a}, ${a.b})` },
  "mix/difference": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_difference(${a.fac}, ${a.a}, ${a.b})` },
  "mix/exclusion": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_exclusion(${a.fac}, ${a.a}, ${a.b})` },
  "mix/screen": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_screen(${a.fac}, ${a.a}, ${a.b})` },
  "mix/soft_light": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_soft_light(${a.fac}, ${a.a}, ${a.b})` },
  "mix/dodge": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_dodge(${a.fac}, ${a.a}, ${a.b})` },
  "mix/burn": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_burn(${a.fac}, ${a.a}, ${a.b})` },
  "mix/divide": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_divide(${a.fac}, ${a.a}, ${a.b})` },
  "mix/hue": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_hue(${a.fac}, ${a.a}, ${a.b})` },
  "mix/saturation": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_saturation(${a.fac}, ${a.a}, ${a.b})` },
  "mix/value": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_value(${a.fac}, ${a.a}, ${a.b})` },
  "mix/color": { inputs: { fac: F(0.5), a: C([1, 1, 1], true), b: C() }, outputs: { color: "color" }, emit: (a) => `mix_color_blend(${a.fac}, ${a.a}, ${a.b})` },

  // ── RGB Curves, sampled. See rgb_curve in nodes.ts for why five. ──
  rgb_curve: {
    inputs: {
      color: C([1, 1, 1], true),
      fac: F(1),
      y0: F(0),
      y1: F(0.25),
      y2: F(0.5),
      y3: F(0.75),
      y4: F(1),
    },
    outputs: { color: "color" },
    emit: (a) => `mix(${a.color}, rgb_curve(${a.color}, ${a.y0}, ${a.y1}, ${a.y2}, ${a.y3}, ${a.y4}), ${a.fac})`,
  },
  // UV Map — one layer on a PMX, which is the mesh UV the texture node uses.
  uv_map: {
    inputs: {},
    outputs: { uv: "vector" },
    contextOutputs: { uv: "vec3f(input.uv, 0.0)" },
  },

  // ── Normal Map, Vector Transform ──
  normal_map: {
    inputs: { color: C([0.5, 0.5, 1], true), strength: F(1) },
    outputs: { normal: "vector" },
    emit: (a) => `node_normal_map(${a.color}, ${a.strength}, n, input.worldPos, input.uv)`,
  },
  "vector_transform/world_to_camera": {
    inputs: { vector: V([0, 0, 0], true) },
    outputs: { vector: "vector" },
    emit: (a) => `vector_world_to_camera(${a.vector})`,
  },
  "vector_transform/camera_to_world": {
    inputs: { vector: V([0, 0, 0], true) },
    outputs: { vector: "vector" },
    emit: (a) => `vector_camera_to_world(${a.vector})`,
  },
  "vector_transform/point_world_to_camera": {
    inputs: { vector: V([0, 0, 0], true) },
    outputs: { vector: "vector" },
    emit: (a) => `point_world_to_camera(${a.vector})`,
  },

  // ── Shader nodes. Shaders travel as RGB here (see shader_to_rgb_diffuse), so
  // a "transparent" shader is the colour that contributes nothing; the actual
  // cutout is the group's alphaMode, which is where transparency belongs in a
  // rasteriser without order-independent blending.
  bsdf_transparent: { inputs: {}, outputs: { color: "color" }, emit: () => `vec3f(0.0)` },
  bsdf_diffuse: {
    inputs: { color: C([0.8, 0.8, 0.8], true) },
    outputs: { color: "color" },
    emit: (a) => `${a.color} * shader_to_rgb_lit(n, l, sun, amb, shadow)`,
  },

  // ── Nodes with no meaning on a PMX, answered honestly with a constant ──
  // Each returns what Blender returns for the absent case, so a graph that reads
  // one degrades to a sensible look instead of failing to compile. Documented
  // rather than silently wrong.
  //
  // attribute: MMD meshes carry no vertex colour layer, so Color reads white and
  // Fac reads 1 — the identity for the multiply these usually feed.
  attribute: {
    inputs: {},
    outputs: { color: "color", fac: "float" },
    contextOutputs: { color: "vec3f(1.0)", fac: "1.0" },
  },
  // object_info: one model, one instance — Random is the only field with a real
  // use (per-instance variation) and there are no instances to vary.
  object_info: {
    inputs: {},
    outputs: { location: "vector", color: "color", random: "float" },
    contextOutputs: { location: "vec3f(0.0)", color: "vec3f(1.0)", random: "0.0" },
  },
  // light_path: this is a raster pass, so every shaded fragment IS a camera ray.
  light_path: {
    inputs: {},
    outputs: { is_camera_ray: "float", is_shadow_ray: "float", ray_depth: "float" },
    contextOutputs: { is_camera_ray: "1.0", is_shadow_ray: "0.0", ray_depth: "0.0" },
  },

  // ── Image Texture on a style-group slot ──
  // The PMX gives a material ONE image; this style needs several, so the extra
  // maps ride on the group. Slot is part of the type because it selects a
  // binding, which is topology rather than a value. The uv input defaults to the
  // mesh UV, matching an unlinked Vector socket in Blender.
  "tex_image/0": {
    inputs: { uv: { type: "vector", contextDefault: "vec3f(input.uv, 0.0)" } },
    outputs: { color: "color", alpha: "float" },
    outputSelect: { color: ".rgb", alpha: ".a" },
    emit: (a) => `group_tex0(${a.uv}.xy)`,
  },
  "tex_image/1": {
    inputs: { uv: { type: "vector", contextDefault: "vec3f(input.uv, 0.0)" } },
    outputs: { color: "color", alpha: "float" },
    outputSelect: { color: ".rgb", alpha: ".a" },
    emit: (a) => `group_tex1(${a.uv}.xy)`,
  },
  "tex_image/2": {
    inputs: { uv: { type: "vector", contextDefault: "vec3f(input.uv, 0.0)" } },
    outputs: { color: "color", alpha: "float" },
    outputSelect: { color: ".rgb", alpha: ".a" },
    emit: (a) => `group_tex2(${a.uv}.xy)`,
  },
  "tex_image/3": {
    inputs: { uv: { type: "vector", contextDefault: "vec3f(input.uv, 0.0)" } },
    outputs: { color: "color", alpha: "float" },
    outputSelect: { color: ".rgb", alpha: ".a" },
    emit: (a) => `group_tex3(${a.uv}.xy)`,
  },

  // ── Blender 5.2 colour utilities ──
  separate_color: {
    inputs: { color: C([1, 1, 1], true) },
    outputs: { r: "float", g: "float", b: "float" },
    outputSelect: { r: ".r", g: ".g", b: ".b" },
    emit: (a) => a.color,
  },
  "separate_color/hsv": {
    inputs: { color: C([1, 1, 1], true) },
    outputs: { h: "float", s: "float", v: "float" },
    outputSelect: { h: ".x", s: ".y", v: ".z" },
    emit: (a) => `rgb_to_hsv(${a.color})`,
  },
  "separate_color/hsl": {
    inputs: { color: C([1, 1, 1], true) },
    outputs: { h: "float", s: "float", l: "float" },
    outputSelect: { h: ".x", s: ".y", l: ".z" },
    emit: (a) => `rgb_to_hsl(${a.color})`,
  },
  combine_color: {
    inputs: { r: F(0), g: F(0), b: F(0) },
    outputs: { color: "color" },
    emit: (a) => `vec3f(${a.r}, ${a.g}, ${a.b})`,
  },
  "combine_color/hsv": {
    inputs: { h: F(0), s: F(0), v: F(0) },
    outputs: { color: "color" },
    emit: (a) => `hsv_to_rgb(vec3f(${a.h}, ${a.s}, ${a.v}))`,
  },
  "combine_color/hsl": {
    inputs: { h: F(0), s: F(0), l: F(0) },
    outputs: { color: "color" },
    emit: (a) => `hsl_to_rgb(vec3f(${a.h}, ${a.s}, ${a.l}))`,
  },
  combine_xyz: {
    inputs: { x: F(0), y: F(0), z: F(0) },
    outputs: { vector: "vector" },
    emit: (a) => `vec3f(${a.x}, ${a.y}, ${a.z})`,
  },
  gamma: {
    inputs: { color: C([1, 1, 1], true), gamma: F(1) },
    outputs: { color: "color" },
    emit: (a) => `node_gamma(${a.color}, ${a.gamma})`,
  },
  // Map Range: the interpolation is topology, as everywhere else here. The
  // clamped form is Blender's default, which is why it carries the bare name.
  map_range: {
    inputs: { value: F(0, true), from_min: F(0), from_max: F(1), to_min: F(0), to_max: F(1) },
    outputs: { value: "float" },
    emit: (a) => `map_range_clamped(${a.value}, ${a.from_min}, ${a.from_max}, ${a.to_min}, ${a.to_max})`,
  },
  "map_range/linear": {
    inputs: { value: F(0, true), from_min: F(0), from_max: F(1), to_min: F(0), to_max: F(1) },
    outputs: { value: "float" },
    emit: (a) => `map_range_linear(${a.value}, ${a.from_min}, ${a.from_max}, ${a.to_min}, ${a.to_max})`,
  },
  "map_range/smoothstep": {
    inputs: { value: F(0, true), from_min: F(0), from_max: F(1), to_min: F(0), to_max: F(1) },
    outputs: { value: "float" },
    emit: (a) => `map_range_smooth(${a.value}, ${a.from_min}, ${a.from_max}, ${a.to_min}, ${a.to_max})`,
  },
  // Vector Rotate — the rotation TYPE is topology.
  "vector_rotate/axis_angle": {
    inputs: { vector: V([0, 0, 0], true), center: V(), axis: V([0, 0, 1]), angle: F(0) },
    outputs: { vector: "vector" },
    emit: (a) => `vector_rotate_axis(${a.vector}, ${a.center}, ${a.axis}, ${a.angle})`,
  },
  "vector_rotate/euler_xyz": {
    inputs: { vector: V([0, 0, 0], true), center: V(), rotation: V() },
    outputs: { vector: "vector" },
    emit: (a) => `vector_rotate_euler(${a.vector}, ${a.center}, ${a.rotation})`,
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
  // Blender ColorRamp LINEAR with three arbitrary stops. A two-stop ramp cannot
  // express a shadow that passes through a colour on its way — a warm terminator
  // between cool shadow and lit skin is three stops, and that middle band is
  // where a lot of NPR character lives. Decomposing it into two ramps and a
  // select costs three extra nodes and stops reading as a ramp.
  ramp_linear_3: {
    inputs: {
      fac: F(0.5, true),
      pos0: F(0),
      color0: V4([0, 0, 0, 1]),
      pos1: F(0.5),
      color1: V4([0.5, 0.5, 0.5, 1]),
      pos2: F(1),
      color2: V4([1, 1, 1, 1]),
    },
    outputs: RAMP_OUTPUTS,
    outputSelect: RAMP_SELECT,
    emit: (a) =>
      `ramp_linear3(${a.fac}, ${a.pos0}, ${a.color0}, ${a.pos1}, ${a.color1}, ${a.pos2}, ${a.color2})`,
  },
  // Blender ColorRamp LINEAR with 3 stops black→white→black (triangular peak at 0.5).
  // Folded to closed form like the hand port.
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
  /**
   * Shader → RGB on a white diffuse closure, reduced to a scalar.
   *
   * Rec.709 luminance, kept exactly as it was: every shipped preset ramps this,
   * and changing the weights would move every terminator in the library.
   */
  shader_to_rgb_diffuse: {
    inputs: {},
    outputs: { value: "float" },
    emit: () => `shader_to_rgb_diffuse(n, l, sun, amb, shadow)`,
  },

  /**
   * The same closure as a COLOUR, which is what Blender's Shader to RGB
   * actually returns.
   *
   * A warm sun against a cool ambient radiates warm light and cool shadow, and
   * that colour IS the look in most of this style. Reducing it to luminance
   * first — the only thing reachable before — leaves every ramp working on grey.
   *
   * Linking this into a scalar socket still yields a float, because Blender's
   * own implicit Color → Value conversion (BT.601) happens at the link. So the
   * two nodes differ only in whether the graph wants the colour or the scalar,
   * and each gets Blender's answer for its own question.
   */
  shader_to_rgb: {
    inputs: {},
    outputs: { color: "color" },
    emit: () => `shader_to_rgb_lit(n, l, sun, amb, shadow)`,
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
  /**
   * Principled BSDF, Blender 5.2 socket names.
   *
   * v2 (4.0+) renamed most of what a graph touches and re-derived one of them,
   * which is exactly where a transcribed port goes quietly wrong: Specular
   * became "Specular IOR Level" and no longer sets reflectance directly — IOR
   * does, and the level scales it. So f0 is computed the v2 way here, and the
   * defaults land on 0.04 for ior 1.5 at level 0.5, matching both versions at
   * their defaults while tracking v2 when a preset moves either.
   *
   * Emission Strength defaults to 0 in v2 where 3.6's Emission was visible, so a
   * naive port turns every emissive material black. Following v2 means honouring
   * that default and letting the preset say otherwise.
   *
   * Not implemented, and not silently wrong — a graph setting these gets the
   * base BSDF rather than a wrong approximation of them: Coat, Transmission,
   * Subsurface, Anisotropic, Thin Film. They need path-traced or
   * multi-scatter machinery this renderer does not have.
   */
  principled: {
    inputs: {
      base_color: C([0.8, 0.8, 0.8], true),
      metallic: F(0),
      roughness: F(0.5),
      ior: F(1.5),
      specular_ior_level: F(0.5),
      sheen_weight: F(0),
      sheen_tint: F(0),
      emission_color: C([1, 1, 1]),
      emission_strength: F(0),
      normal: { type: "vector", contextDefault: "n" },
      /** Ours, not Blender's: caps firefly speculars from noise-bumped NDF
       *  aliasing, which EEVEE hides behind TAA and we have none. */
      spec_clamp: F(1e30),
    },
    outputs: { color: "color" },
    emit: (a) => {
      // Fold the reflectance when both sockets are literals, which is nearly
      // always. It keeps the emitted WGSL as tight as the 3.6 form was — and at
      // the defaults it folds to exactly 0.5, which is what 3.6's Specular held,
      // so the two versions agree byte-for-byte where a preset changed nothing.
      const spec = foldSpecular(a.ior, a.specular_ior_level)
      const bsdf =
        `eval_principled(PrincipledIn(${a.base_color}, ${a.metallic}, ` +
        `${spec}, ${a.roughness}, ` +
        `${a.spec_clamp}, ${a.sheen_weight}, ${a.sheen_tint}), ${a.normal}, l, v, sun, amb, shadow)`
      // v2 defaults Emission Strength to 0, which is the overwhelming case. Emit
      // the term only when it can do something, so the common shader carries no
      // dead add and the output stays readable.
      return a.emission_strength === "0.0" ? bsdf : `${bsdf} + ${a.emission_color} * ${a.emission_strength}`
    },
  },
}

// ─── Blender socket parity ────────────────────────────────────────────
// Blender's Math node carries three Value inputs whatever operation is selected,
// its Vector Math node three Vectors plus a Scale, and Vector Rotate the union of
// every rotation type's inputs. Sockets are part of the node, not of the mode.
//
// Declaring the same set here is what makes a port mechanical: a transcriber maps
// socket-for-socket and never has to know which inputs a given operation happens
// to read. The emit functions read only what their operation uses, so the added
// sockets change no generated WGSL — an unread one is inert.
//
// Applied as a pass rather than written into each entry so that every operation
// KEEPS the default it already declared (divide's b is 1, not 0). Only genuinely
// missing sockets are added.
function widen(prefix: string, sockets: Record<string, InputSpec>): void {
  for (const [key, spec] of Object.entries(NODE_REGISTRY)) {
    if (!key.startsWith(prefix)) continue
    for (const [name, def] of Object.entries(sockets)) {
      if (!(name in spec.inputs)) spec.inputs[name] = def
    }
  }
}

widen("math/", { a: F(0), b: F(0), c: F(0) })
widen("vector_math/", { a: V(), b: V(), c: V(), scale: F(1) })
widen("vector_rotate/", {
  vector: V([0, 0, 0], true),
  center: V(),
  axis: V([0, 0, 1]),
  angle: F(0),
  rotation: V(),
})
