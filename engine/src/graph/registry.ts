// Node registry: one entry per Blender-equivalent node backed by NODES_WGSL (nodes.ts).
// The registry adds no WGSL of its own — a node type exists here only if its function
// already exists (validated against EEVEE) in the shader library, or is a WGSL builtin.
// Semantics are frozen Blender 3.6 legacy-EEVEE; enum modes (math op, mix blend type,
// ramp interpolation) are part of the type string so they are unambiguously topology.

import type { SocketValue } from "./schema"

export type SockT = "float" | "color" | "vector" | "vec4"

type InputSpec = {
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
  /** The node shades by the scene's light without naming it: its graph takes
   *  the lamps and the fog as the four shading nodes' graphs do. */
  takesLight?: boolean
}

// ─── Literal formatting ───────────────────────────────────────────────
// Deterministic: same graph JSON → byte-identical WGSL. String(x) is JS shortest
// round-trip, so full-precision Blender constants (0.15000000596046448) survive.

function fmtFloat(x: number): string {
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

const uberNode = (slot: number): NodeSpec => ({
  inputs: {
    base: C([1, 1, 1], true),
    metallic: F(0),
    roughness: F(0.6),
    occlusion: F(1),
    normal: { type: "vector", contextDefault: "n" },
    // The key light: this draw's, by default — the sun, or the light on the
    // character's own layer — its way and its colour (the game's _LocalLightDir,
    // _LocalLightColor). Linked, a look supplies its own (ag_key_light).
    direction: { type: "vector", contextDefault: "l" },
    light_color: { type: "color", contextDefault: "sun" },
    // Unity's v into the ramp: the property map's alpha, which picks a
    // surface's band (skin 0.702, hair and pale cloth 0.506).
    row: F(0.702),
    receive_shadow: F(1),
    rim_mask: F(1),
    rim_mid: F(0.448),
    rim_width: F(0.11),
    rim_tint: C([0.9623, 0.9214, 0.9214]),
    rim_intensity: F(1),
    rim_albedo: F(0.9),
    rim_in_light: F(1),
    emission: C([0, 0, 0]),
    reflection: F(1),
    // THE STUDIO AMBIENT (_GlobalIlluminationOverride): the game blends a
    // character's ambient from its own reflection cube's smallest mip — near
    // neutral, 0.4–0.9 across the newer skins — over the scene's light, by the
    // material's blend (bodies 0.8, hair 1, faces 0.5). It is what keeps a
    // character clean on a coloured stage.
    studio: C([0.6, 0.6, 0.6]),
    studio_blend: F(0),
    // A face's SDF shade (ag_face_sdf); left at -1, the material reads N·L.
    shade: F(-1),
    // 1: the game's FACE_MODE — no GGX, the environment at a flat 0.0157.
    face: F(0),
    // A face's SDF highlight (ag_face_spec), in place of GGX.
    face_spec: F(0),
    // 1: hair (_ANISOTROPIC_SPECULAR) — its ring (ag_hair_ring) in place of GGX,
    // the environment at a flat 0.0157.
    hair: F(0),
    hair_spec: C([0, 0, 0]),
  },
  outputs: { color: "color" },
  emit: (a) =>
    `ag_uber(${slot}u, ${a.base}, ${a.metallic}, ${a.roughness}, ${a.occlusion}, ${a.normal}, ${a.direction}, ${a.light_color}, ${a.row}, ${a.receive_shadow}, ` +
    `${a.rim_mask}, ${a.rim_mid}, ${a.rim_width}, ${a.rim_tint}, ${a.rim_intensity}, ${a.rim_albedo}, ${a.rim_in_light}, ${a.emission}, ${a.reflection}, ${a.studio}, ${a.studio_blend}, ${a.shade}, ${a.face}, ${a.face_spec}, ${a.hair}, ${a.hair_spec}, input.worldPos, v)`,
  takesLight: true,
})
const UBER_NODES: Record<string, NodeSpec> = Object.fromEntries([0, 1, 2, 3].map((s) => [`ag_uber/${s}`, uberNode(s)]))
// How much a colour reads as skin (ag_skin): what picks a surface's ramp row in
// a general look, where the game would read its property mask.
const AG_SKIN: NodeSpec = {
  inputs: { color: C([1, 1, 1], true) },
  outputs: { value: "float" },
  emit: (a) => `ag_skin(${a.color})`,
}
// The face's SDF shade, the image on the slot the type names. An empty slot
// reads white: the whole face takes one shade from where the key light stands
// to the head, the game's flat face for a model without its SDF.
const faceSdfNode = (slot: number): NodeSpec => ({
  inputs: {
    uv: { type: "vector", contextDefault: "vec3f(input.uv, 0.0)" },
    direction: { type: "vector", contextDefault: "l" },
    normal: { type: "vector", contextDefault: "n" },
    smoothness: F(0.1),
    invert: F(0),
  },
  outputs: { value: "float" },
  emit: (a) => `ag_face_sdf(${slot}u, ${a.uv}.xy, ${a.direction}, ${a.normal}, ${a.smoothness}, ${a.invert})`,
})
const FACE_SDF_NODES: Record<string, NodeSpec> = Object.fromEntries([0, 1, 2, 3].map((s) => [`ag_face_sdf/${s}`, faceSdfNode(s)]))
// The newer SDF (_SDFType 1): a painted face normal, not a threshold field.
const faceSdfNewNode = (slot: number): NodeSpec => ({
  ...faceSdfNode(slot),
  emit: (a) => `ag_face_sdf_new(${slot}u, ${a.uv}.xy, ${a.direction}, ${a.normal}, ${a.smoothness}, ${a.invert})`,
})
const FACE_SDF_NEW_NODES: Record<string, NodeSpec> = Object.fromEntries([0, 1, 2, 3].map((s) => [`ag_face_sdf_new/${s}`, faceSdfNewNode(s)]))
// The face's highlight from the newer SDF's green and blue.
const faceSpecNode = (slot: number): NodeSpec => ({
  inputs: {
    uv: { type: "vector", contextDefault: "vec3f(input.uv, 0.0)" },
    direction: { type: "vector", contextDefault: "l" },
    normal: { type: "vector", contextDefault: "n" },
    anisotropy: F(1),
    shift: F(0),
  },
  outputs: { value: "float" },
  emit: (a) => `ag_face_spec(${slot}u, ${a.uv}.xy, ${a.direction}, ${a.normal}, v, ${a.anisotropy}, ${a.shift})`,
})
// The hair's angel ring, the band on the slot's image.
const hairRingNode = (slot: number): NodeSpec => ({
  inputs: {
    uv: { type: "vector", contextDefault: "vec3f(input.uv, 0.0)" },
    normal: { type: "vector", contextDefault: "n" },
    color: C([1, 1, 1]),
    anisotropy: F(1),
    shift: F(0),
  },
  outputs: { color: "color" },
  emit: (a) => `ag_hair_ring(${slot}u, ${a.uv}.xy, ${a.normal}, v, ${a.color}, ${a.anisotropy}, ${a.shift})`,
})
const HAIR_RING_NODES: Record<string, NodeSpec> = Object.fromEntries([0, 1, 2, 3].map((s) => [`ag_hair_ring/${s}`, hairRingNode(s)]))
// The game's pupil (SimPipeline/Character/Eye): unlit relief, matcap and
// glints — slots: 0 depth, 1 normal, 2 matcap, 3 glint mask.
const AG_EYE: NodeSpec = {
  inputs: {
    height_scale: F(0.2), min_layer: F(8), max_layer: F(32), intensity: F(0.9),
    main_color: C([1, 1, 1]), matcap_color: C([1, 1, 1]), matcap_pow: F(1),
    mask_scale: F(0.15), mask_soft: F(0.014),
    mask2: F(0), mask2_color: C([1, 1, 1]), mask3: F(0), mask3_color: C([1, 1, 1]),
    blink: F(0), blink_scale: F(10), scale_speed: F(0.1), rotate_speed: F(0.6), blink_angle: F(5),
    level: F(1),
  },
  outputs: { color: "color" },
  emit: (a) =>
    `ag_eye(input.uv, n, v, input.worldPos, ${a.height_scale}, ${a.min_layer}, ${a.max_layer}, ${a.intensity}, ${a.main_color}, ` +
    `${a.matcap_color}, ${a.matcap_pow}, ${a.mask_scale}, ${a.mask_soft}, ${a.mask2}, ${a.mask2_color}, ${a.mask3}, ${a.mask3_color}, ` +
    `${a.blink}, ${a.blink_scale}, ${a.scale_speed}, ${a.rotate_speed}, ${a.blink_angle}, ${a.level})`,
}
const FACE_SPEC_NODES: Record<string, NodeSpec> = Object.fromEntries([0, 1, 2, 3].map((s) => [`ag_face_spec/${s}`, faceSpecNode(s)]))

// A packed property map on the slot the type names — the game's material
// contract: R metal, G perceptual roughness (answered as smoothness, Unity's
// word for 1 − r), B occlusion, A emission mask, each remapped through the
// material's own min/max (see rz_property_map). The four
// plain values are what the surface is WITHOUT a map, so one look serves a
// converted set that brought maps and a hand-made stage that brought none.
// Defaults are the game's: the full 0–1 range, and the glow off (min = max = 1).
const propertyMapNode = (slot: number): NodeSpec => ({
  inputs: {
    uv: { type: "vector", contextDefault: "vec3f(input.uv, 0.0)" },
    metallic: F(0),
    smoothness: F(0.5),
    occlusion: F(1),
    emission: F(0),
    metal_min: F(0),
    metal_max: F(1),
    rough_min: F(0),
    rough_max: F(1),
    ao_min: F(0),
    ao_max: F(1),
    emission_min: F(1),
    emission_max: F(1),
  },
  outputs: { metallic: "float", smoothness: "float", occlusion: "float", emission: "float" },
  outputSelect: { metallic: ".x", smoothness: ".y", occlusion: ".z", emission: ".w" },
  emit: (a) =>
    `rz_property_map(group_tex${slot}(${a.uv}.xy), rz_group_bound${slot}(), vec4f(${a.metallic}, ${a.smoothness}, ${a.occlusion}, ${a.emission}), ` +
    `vec4f(${a.metal_min}, ${a.rough_min}, ${a.ao_min}, ${a.emission_min}), vec4f(${a.metal_max}, ${a.rough_max}, ${a.ao_max}, ${a.emission_max}))`,
})
const PROPERTY_MAP_NODES: Record<string, NodeSpec> = Object.fromEntries([0, 1, 2, 3].map((s) => [`property_map/${s}`, propertyMapNode(s)]))

export const NODE_REGISTRY: Record<string, NodeSpec> = {
  // ── Context inputs (template locals; no emission) ──
  texture: {
    inputs: {},
    outputs: { color: "color", alpha: "float" },
    contextOutputs: { color: "tex_color", alpha: "tex_s.a" },
  },
  // The scene clock, in seconds. A material is otherwise a function of position
  // alone — this is what lets one SCROLL: ripples on water, a conveyor, a sign.
  // It is the same clock the effects read, so a paused scene holds still and an
  // export steps it frame by frame.
  time: {
    inputs: {},
    outputs: { value: "float" },
    contextOutputs: { value: "camera.time" },
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
      /**
       * How much WORLD one pixel covers here, in units.
       *
       * What a texture gets free from its mipmap and a procedural pattern has
       * no way to ask for. Past the point where this exceeds a pattern's
       * wavelength the pattern is under-sampled, and a bump taken from it is
       * differencing two uncorrelated samples — random per 2x2 quad, which on a
       * receding plane stretches along the depth axis and reads as scan lines.
       * Fade a layer out as this approaches its wavelength and the surface goes
       * quietly smooth instead, which is what distance should look like.
       */
      footprint: "float",
    },
    contextOutputs: {
      normal: "n",
      view: "v",
      world_pos: "input.worldPos",
      rest_pos: "input.restPos",
      uv: "vec3f(input.uv, 0.0)",
      reflection: "reflect(-v, n)",
      footprint: "max(length(dpdx(input.worldPos)), length(dpdy(input.worldPos)))",
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
  // WHAT THE SURFACE IS MADE OF, when whoever wrote the model said so.
  //
  // PMX carries a specular colour its own renderer barely uses, and a stage
  // converted out of a game engine packs its material's (metal, roughness,
  // occlusion) there — the same triple that game's property map holds, averaged
  // per material. One shared look can then read it and a whole stage keeps ONE
  // pipeline, where a look per material was what cost a garden its frame rate.
  //
  // A hand-authored PMX leaves it at whatever its exporter wrote, so a graph
  // that reads this is choosing to trust the model. Nothing else does.
  // How opaque the PMX said this material is, before its texture's alpha is
  // multiplied in. A graph that computes its own opacity still wants it: glass
  // is see-through by the amount its author chose and a mirror at a grazing
  // angle, and without this every pane in a stage gets the same transparency
  // whatever its material says.
  material_alpha: {
    inputs: {},
    outputs: { value: "float" },
    contextOutputs: { value: "material.alpha" },
  },

  material_specular: {
    inputs: {},
    outputs: { color: "color" },
    contextOutputs: { color: "material.specular" },
  },
  material_shininess: {
    inputs: {},
    outputs: { value: "float" },
    contextOutputs: { value: "material.shininess" },
  },

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
  // UV Map — the mesh UV the texture node uses, and the second set: a PMX's
  // second additional UV (xy) where it has one, else the first UV again.
  uv_map: {
    inputs: {},
    outputs: { uv: "vector", uv2: "vector" },
    contextOutputs: { uv: "vec3f(input.uv, 0.0)", uv2: "vec3f(input.uv2, 0.0)" },
  },
  // Whether each group image slot holds a real map, 1 or 0. A slot nobody
  // filled is a 1×1 white stand-in, and white is not "no map" to a normal or
  // a property map — this is how a look tells the two apart and keeps the
  // surface's own normal where the model brought none.
  image_bound: {
    inputs: {},
    outputs: { slot0: "float", slot1: "float", slot2: "float", slot3: "float" },
    contextOutputs: {
      slot0: "select(0.0, 1.0, rz_group_bound0())",
      slot1: "select(0.0, 1.0, rz_group_bound1())",
      slot2: "select(0.0, 1.0, rz_group_bound2())",
      slot3: "select(0.0, 1.0, rz_group_bound3())",
    },
  },

  // ── Normal Map, Vector Transform ──
  // Strength scales the map's tilt, so above 1 steepens it (Unity's
  // _NormalScale, Blender's strength); 0 is the surface's own normal.
  normal_map: {
    inputs: { color: C([0.5, 0.5, 1], true), strength: F(1) },
    outputs: { normal: "vector" },
    emit: (a) => `node_normal_map(${a.color}, ${a.strength}, n, input.worldPos, input.uv)`,
  },
  // A normal map packed as game engines store one (DXT5nm/BC5: X in alpha, Y
  // in green, Z rebuilt). Link the image's colour and its alpha.
  "normal_map/packed": {
    inputs: { color: C([1, 0.5, 1], true), alpha: F(0.5), strength: F(1) },
    outputs: { normal: "vector" },
    emit: (a) => `node_normal_map_packed(vec4f(${a.color}, ${a.alpha}), ${a.strength}, n, input.worldPos, input.uv)`,
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

  // ── Nodes with no meaning on a PMX, answered honestly with a constant ──
  // Each returns what Blender returns for the absent case, so a graph that reads
  // one degrades to a sensible look instead of failing to compile. Documented
  // rather than silently wrong.
  //
  // attribute: the vertex colour. A PMX carries none of its own, so a model
  // keeps it in its first additional UV (RGBA) — what a converted game stage
  // writes there, a plant's occlusion in the alpha. A model without one reads
  // white and alpha 1, the identity for the multiply these usually feed.
  attribute: {
    inputs: {},
    outputs: { color: "color", fac: "float", alpha: "float" },
    contextOutputs: { color: "input.vcolor.rgb", fac: "color_to_value(input.vcolor.rgb)", alpha: "input.vcolor.a" },
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

  // ── Aether Gazer's character shading ──
  // The game's character shader (SimPipeline/Character/Debug, which most of its
  // characters wear), taken apart into the steps it runs — see ag_* in
  // nodes.ts for each step's derivation from the decompiled pass. A graph
  // composes them as the game did: a ramp lookup driven by the key light, a
  // matcap, a reflection, the rim and the fill, with the ramp and matcap
  // images on the style group's slots.

  // The key light: CharacterEffect's direction, held in a frame that turns
  // with the camera about the vertical (sim_TowardMatrix) — the game lights a
  // character from where the viewer is, not from the scene's sun.
  ag_key_light: {
    inputs: { inclination: F(24), azimuth: F(-28) },
    outputs: { direction: "vector" },
    emit: (a) => `ag_key_light(${a.inclination}, ${a.azimuth})`,
  },
  // Where on the shadow ramp this point falls: N·L under the mask's occlusion,
  // and the sun's cast shadow on top when the character receives it.
  ag_shade_term: {
    inputs: {
      normal: { type: "vector", contextDefault: "n" },
      direction: { type: "vector", requiresLink: true },
      occlusion: F(1),
      occlusion_scale: F(1),
      receive_shadow: F(1),
    },
    outputs: { value: "float" },
    emit: (a) => `ag_shade_term(${a.normal}, ${a.direction}, ${a.occlusion}, ${a.occlusion_scale}, ${a.receive_shadow}, input.worldPos)`,
    takesLight: true,
  },
  // A ramp image's coordinate: u from the shade term, v the ramp's row (the
  // game's property map picks it per texel; a PMX carries one per material).
  ag_ramp_uv: {
    inputs: { value: F(0), row: F(0.5) },
    outputs: { uv: "vector" },
    emit: (a) => `ag_ramp_uv(${a.value}, ${a.row})`,
  },
  // The view-space reflection the matcap, the reflection map, the rim and the
  // fill all read: its uv into a sphere image, and how much it faces the eye.
  ag_view_reflection: {
    inputs: { normal: { type: "vector", contextDefault: "n" } },
    outputs: { uv: "vector", facing: "float" },
    outputSelect: { uv: ".xyz", facing: ".w" },
    emit: (a) => `ag_view_reflection(${a.normal}, v)`,
  },
  // The matcap's lift: a metal's shine, at least 1, where the mask says.
  ag_matcap: {
    inputs: { color: C([1, 1, 1], true), matcap: C([1, 1, 1]), strength: F(1), mask: F(1) },
    outputs: { color: "color" },
    emit: (a) => `ag_matcap(${a.color}, ${a.matcap}, ${a.strength}, ${a.mask})`,
  },
  // The rim: the two fixed screen-side directions CharacterEffect sets, cut to
  // the silhouette, laid on as a colour dodge where the mask allows.
  ag_rim: {
    inputs: {
      color: C([1, 1, 1], true),
      facing: F(1),
      uv: { type: "vector", requiresLink: true },
      rim_color: C([0.8584906, 0.8584906, 0.8584906]),
      threshold: F(0.1),
      fade: F(0.113),
      range: F(0.444),
      inclination: F(0),
      azimuth1: F(0.8499963),
      azimuth2: F(1),
      mask: F(1),
    },
    outputs: { color: "color" },
    emit: (a) =>
      `ag_rim(${a.color}, ${a.facing}, ${a.uv}, ${a.rim_color}, ${a.threshold}, ${a.fade}, ${a.range}, ${a.inclination}, ${a.azimuth1}, ${a.azimuth2}, ${a.mask})`,
  },
  // The fill: a colour laid toward the silhouette, inner to outer.
  ag_fill: {
    inputs: {
      color: C([1, 1, 1], true),
      facing: F(1),
      inner: C([0, 0, 0]),
      outer: C([0, 0, 0]),
      amount: F(0),
      softness: F(0),
    },
    outputs: { color: "color" },
    emit: (a) => `ag_fill(${a.color}, ${a.facing}, ${a.inner}, ${a.outer}, ${a.amount}, ${a.softness})`,
  },

  // Aether Gazer's current character shading (PBR/Uber) as one closure, the way
  // Principled is one: the ramp image on the group slot named by the type.
  // Defaults are 104903's skin.
  ...UBER_NODES,
  ag_skin: AG_SKIN,
  ...FACE_SDF_NODES,
  ...FACE_SDF_NEW_NODES,
  ...FACE_SPEC_NODES,
  ag_eye: AG_EYE,
  ...HAIR_RING_NODES,
  ...PROPERTY_MAP_NODES,

  /**
   * Lit — Unity's Lit (URP's PBR), term for term: what a stage, a prop or
   * the weather is shaded with, so a surface authored with Unity's inputs
   * looks here as it does there. See urp_lit in nodes.ts for the formula.
   *
   * smoothness is Unity's (1 − perceptual roughness). emission is added on
   * top, as HDR colour. alpha < 1 is Transparent with Preserve Specular
   * Lighting — the diffuse fades, the reflection does not; the `alpha`
   * output is the coverage to hand the graph's opacity, on a group blending
   * "premultiplied".
   */
  lit: {
    inputs: {
      base_color: C([0.8, 0.8, 0.8], true),
      metallic: F(0),
      smoothness: F(0.5),
      /** A dielectric's reflectance, as Unreal's and HDRP's Specular: 0.5 is
       *  URP's fixed 0.04, 1 is 0.08 — glossy cloth, lacquer, wet skin. */
      specular: F(0.5),
      occlusion: F(1),
      emission: C([0, 0, 0]),
      alpha: F(1),
      normal: { type: "vector", contextDefault: "n" },
    },
    outputs: { color: "color", alpha: "float" },
    outputSelect: { color: ".rgb", alpha: ".a" },
    emit: (a) =>
      `urp_lit(${a.base_color}, ${a.metallic}, ${a.smoothness}, ${a.specular}, ${a.occlusion}, ${a.emission}, ${a.alpha}, ` +
      `${a.normal}, l, v, sun, amb, shadow, input.worldPos, true)`,
    takesLight: true,
  },

  /**
   * The scene's reflection along a direction — the stage's own reflection
   * probe, box-projected to the stage's bounds, where the engine captured one;
   * the sky where it did not. Roughness picks the blur on the game's curve.
   * Its default vector is this surface's reflection, so a glass shell or a
   * water surface wires only what it bends.
   */
  /**
   * The highlights alone: the sun's and the lamps' specular on a GGX lobe
   * (URP's, α = r²), untinted. A water or glass look that computes its own
   * reflection multiplies this by its reflectance for the glints.
   */
  glossy_direct: {
    inputs: { smoothness: F(0.9), normal: { type: "vector", contextDefault: "n" } },
    outputs: { color: "color" },
    emit: (a) => `rz_glossy_direct(${a.normal}, l, v, sun, shadow, input.worldPos, 1.0 - (${a.smoothness}))`,
    takesLight: true,
  },
  /**
   * The lamps, handed to the graph — Shader Graph's Additional Lights: their
   * diffuse at `normal`, and a Blinn-Phong highlight pow(N·H, exponent), both
   * untinted. A graph that reads them combines them as its shader does (the
   * game's water adds the diffuse into its body before its colour, and glints
   * each lamp at exponent 128·smoothness), and owns the lamps' diffuse: the
   * layer the engine otherwise adds after the graph is not added.
   */
  additional_lights: {
    inputs: { exponent: F(64), normal: { type: "vector", contextDefault: "n" } },
    outputs: { diffuse: "color", specular: "color" },
    outputSelect: { diffuse: ".d", specular: ".s" },
    emit: (a) => `rz_additional_lights(input.worldPos, ${a.normal}, v, ${a.exponent})`,
    takesLight: true,
  },
  reflection_probe: {
    inputs: {
      vector: { type: "vector", contextDefault: "reflect(-v, n)" },
      smoothness: F(1),
    },
    outputs: { color: "color" },
    emit: (a) => `rzProbeSpecular(${a.vector}, input.worldPos, 1.0 - (${a.smoothness}))`,
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

  // ── View-dependent scalars ──
  fresnel: { inputs: { ior: F(1.45) }, outputs: { value: "float" }, emit: (a) => `fresnel(${a.ior}, n, v)` },
  // The normal is an INPUT with the shading normal as its default, so these read
  // like every other node when nothing is wired and can be handed a perturbed
  // normal when something is. Water is why: its ripples live in a bump node, and
  // a fresnel that quietly read the flat surface normal gave a pool whose
  // reflection and opacity were both perfectly smooth however the bump was
  // tuned — the ripples existed and touched nothing anybody could see.
  "layer_weight/fresnel": {
    inputs: { blend: F(0.5), normal: { type: "vector", contextDefault: "n" } },
    outputs: { value: "float" },
    emit: (a) => `layer_weight_fresnel(${a.blend}, ${a.normal}, v)`,
  },
  "layer_weight/facing": {
    inputs: { blend: F(0.5), normal: { type: "vector", contextDefault: "n" } },
    outputs: { value: "float" },
    emit: (a) => `layer_weight_facing(${a.blend}, ${a.normal}, v)`,
  },


  // ── Lighting capture ──
  /**
   * Lambert — URP's LightingLambert for the main light, plus the ambient:
   * what a white surface facing `normal` receives. A toon look ramps `value`
   * (its luminance) into a terminator, or tints by `color`. Shader Graph's
   * custom-lighting idiom, in one node.
   */
  lambert: {
    inputs: { normal: { type: "vector", contextDefault: "n" } },
    outputs: { color: "color", value: "float" },
    outputSelect: { color: ".rgb", value: ".a" },
    emit: (a) => `urp_lambert(${a.normal}, l, sun, amb, shadow)`,
  },
  /**
   * Hands this surface to the screen-space scattering pass: the colour passes
   * through, and the pixel is blurred afterwards the way ray-mmd blurs skin —
   * red furthest, never across a depth step. Strength scales the width; 1 is
   * ray-mmd's skin, up to 4. It has to lie on the path to the output to take effect.
   */
  subsurface: {
    inputs: { color: C([1, 1, 1], true), strength: F(1) },
    outputs: { color: "color" },
    emit: (a) => `rz_subsurface(${a.strength}, ${a.color})`,
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
  // Its strength is a SLOPE, not a screen effect — see bump_world. Use it for
  // anything whose detail belongs to the surface rather than to the frame:
  // water, sand, hammered metal. `bump` holds its size on screen, which is what
  // skin and cloth want and what water cannot use.
  "bump/world": {
    inputs: { strength: F(0.1), height: F(0, true), normal: V([0, 0, 0], true) },
    outputs: { vector: "vector" },
    emit: (a) => `bump_world(${a.strength}, ${a.height}, ${a.normal}, input.worldPos)`,
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
