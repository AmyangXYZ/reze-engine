// The data behind the overlay pass: what an overlay primitive IS, the unit
// wireframes the pass draws, and the three builders that turn a posed model into
// a list of them.
//
// The builders are pure — (model, physics, options) -> OverlayPrimitive[] — so a
// host can call them itself, print the result, diff two poses, or hand a list
// straight to setOverlay without the engine's live layers in the way. The pass
// is one caller of them.
//
// Everything they emit is WORLD space, root transform included, so a list is
// meaningful on its own.

import { Quat, Vec3, Mat4 } from "./math"
import { RigidbodyShape, RigidbodyType } from "./physics/types"
import type { Rigidbody } from "./physics/types"
import type { Model } from "./model"
import type { RezePhysics } from "./physics"

export type RGBA = readonly [number, number, number, number]
export type Vec3Tuple = readonly [number, number, number]

/** The unit wireframes the overlay pass carries, each a list of line segments. */
export type OverlayShape =
  | "line"
  | "dashedLine"
  | "axes"
  | "circle"
  | "dot"
  | "link"
  | "sphere"
  | "box"
  | "capsule"
  | "solidSphere"
  | "solidBox"
  | "solidCapsule"

export const OVERLAY_SHAPES: readonly OverlayShape[] = [
  "line",
  "dashedLine",
  "axes",
  "circle",
  "dot",
  "link",
  "sphere",
  "box",
  "capsule",
  "solidSphere",
  "solidBox",
  "solidCapsule",
]

/**
 * One wireframe copy. `scale` is in world units and means half-extents for
 * `box`, radius for `sphere` and `capsule`, and (width, length, width) for
 * `line`, `dashedLine` and `axes`.
 */
export interface OverlayPrimitive {
  shape: OverlayShape
  /** World position of the shape's origin. */
  position: Vec3Tuple
  /** World rotation, quaternion xyzw. Default identity. */
  rotation?: readonly [number, number, number, number]
  /** World-unit size, per the shape's own reading above. 1e-4–1e4. Default [1,1,1]. */
  scale?: Vec3Tuple
  /** Capsule half-height along local +Y, world units. 0–1e4. Ignored by every
   *  other shape. Default 0. */
  extent?: number
  /** Linear rgba, each 0–1. */
  color: RGBA
  /** Stroke width in device pixels. Ignored by the filled shapes (`circle`,
   *  `dot`), whose weight is in their geometry. 0.5–24. Default 3. */
  thickness?: number
}

/** The shapes that are solid geometry rather than stroked outlines. They draw
 *  through their own pipeline: no depth write, no culling, so a translucent
 *  volume reads as a volume and never hides the rig behind it. */
export const OVERLAY_SOLID_SHAPES: ReadonlySet<OverlayShape> = new Set<OverlayShape>([
  "solidSphere",
  "solidBox",
  "solidCapsule",
])

/** Floats per instance in the pass's instance stream. */
export const OVERLAY_INSTANCE_FLOATS = 16
/** Interleaved pos(3), dir(3), caps(2), side(1), t(1), mode(1) per vertex. */
export const OVERLAY_VERTEX_FLOATS = 11

/** Radius of the marker's solid centre, as a fraction of the ring's. */
const DOT_RADIUS = 0.42
/** Segments in a marker's ring and centre. They are small, always face the
 *  camera, and there are only a few hundred, so smoothness is nearly free. */
const MARKER_SEG = 48

/** Segments around a sphere's great circles and a capsule's rings. */
const RING_SEG = 32
/** Segments in one half-circle of a capsule cap (ring → pole → ring). */
const CAP_SEG = 16
/** Dashes in a dashed unit line. */
const DASHES = 7

export interface OverlayGeometry {
  vertices: Float32Array<ArrayBuffer>
  /** Where each shape's vertices start, and how many it has. */
  ranges: Record<OverlayShape, { first: number; count: number }>
}

/** Builds every unit wireframe back to back in one line-list vertex buffer. */
export function buildOverlayShapes(): OverlayGeometry {
  const v: number[] = []
  const ranges = {} as Record<OverlayShape, { first: number; count: number }>

  // Placement mode for the shape being emitted — see `placeWorld` in the shader.
  let b = 0
  // A stroked segment: 6 vertices carrying each endpoint, the offset to the
  // other, both cap signs, which side of the quad it is on, and its parameter
  // along the segment. c0/c1 are the capsule cap signs — see the shader.
  const seg = (p0: Vec3Tuple, p1: Vec3Tuple, c0 = 0, c1 = 0) => {
    const dx = p1[0] - p0[0],
      dy = p1[1] - p0[1],
      dz = p1[2] - p0[2]
    // prettier-ignore
    v.push(
      p0[0], p0[1], p0[2],  dx,  dy,  dz, c0, c1, -1, 0, b,
      p0[0], p0[1], p0[2],  dx,  dy,  dz, c0, c1,  1, 0, b,
      p1[0], p1[1], p1[2], -dx, -dy, -dz, c1, c0, -1, 1, b,
      p0[0], p0[1], p0[2],  dx,  dy,  dz, c0, c1,  1, 0, b,
      p1[0], p1[1], p1[2], -dx, -dy, -dz, c1, c0,  1, 1, b,
      p1[0], p1[1], p1[2], -dx, -dy, -dz, c1, c0, -1, 1, b,
    )
  }
  // A filled triangle, for the shapes that are geometry rather than strokes.
  const tri = (p0: Vec3Tuple, p1: Vec3Tuple, p2: Vec3Tuple) => {
    for (const p of [p0, p1, p2]) v.push(p[0], p[1], p[2], 0, 0, 0, 0, 0, 0, 0, b)
  }
  // A filled triangle whose vertices carry their own cap signs — the solid
  // capsule needs it, since one quad of its wall straddles both hemispheres.
  const triC = (p0: Vec3Tuple, c0: number, p1: Vec3Tuple, c1: number, p2: Vec3Tuple, c2: number) => {
    v.push(p0[0], p0[1], p0[2], 0, 0, 0, c0, 0, 0, 0, b)
    v.push(p1[0], p1[1], p1[2], 0, 0, 0, c1, 0, 0, 0, b)
    v.push(p2[0], p2[1], p2[2], 0, 0, 0, c2, 0, 0, 0, b)
  }
  // One rim vertex of the marker's ring: a direction, and which rim it is on.
  const rim = (d: Vec3Tuple, side: number) => {
    v.push(d[0], d[1], d[2], 0, 0, 0, 0, 0, side, 0, b)
  }
  const open = (shape: OverlayShape) => {
    ranges[shape] = { first: v.length / OVERLAY_VERTEX_FLOATS, count: 0 }
  }
  const close = (shape: OverlayShape) => {
    ranges[shape].count = v.length / OVERLAY_VERTEX_FLOATS - ranges[shape].first
  }

  // line — origin to +Y. scale.y is its length; point it with the rotation.
  open("line")
  seg([0, 0, 0], [0, 1, 0])
  close("line")

  // dashedLine — the same run, broken up. A dash is geometry here rather than a
  // fragment test, which is what a line pass can express without a shader for it.
  open("dashedLine")
  for (let i = 0; i < DASHES; i++) {
    const a = i / DASHES
    const b = (i + 0.5) / DASHES
    seg([0, a, 0], [0, b, 0])
  }
  close("dashedLine")

  // axes — a cross through the origin, one segment per axis, ±scale.
  open("axes")
  seg([-1, 0, 0], [1, 0, 0])
  seg([0, -1, 0], [0, 1, 0])
  seg([0, 0, -1], [0, 0, 1])
  close("axes")

  // circle — the bone marker's ring, facing the camera. Triangles between an
  // inner and an outer rim, which the shader offsets radially in SCREEN space:
  // the radius is world, the stroke is the same pixel width as the links, and
  // the two meet without a step. `side` names the rim, -1 inner and +1 outer.
  b = 4
  open("circle")
  for (let i = 0; i < MARKER_SEG; i++) {
    const a0 = (i / MARKER_SEG) * Math.PI * 2
    const a1 = ((i + 1) / MARKER_SEG) * Math.PI * 2
    const d0: Vec3Tuple = [Math.cos(a0), Math.sin(a0), 0]
    const d1: Vec3Tuple = [Math.cos(a1), Math.sin(a1), 0]
    rim(d0, -1)
    rim(d0, 1)
    rim(d1, 1)
    rim(d0, -1)
    rim(d1, 1)
    rim(d1, -1)
  }
  close("circle")

  // dot — the marker's solid centre. A plain camera-facing fan: mode 3, NOT the
  // ring's 4, whose radial offset is undefined at a fan's centre vertex.
  b = 3
  open("dot")
  for (let i = 0; i < MARKER_SEG; i++) {
    const a0 = (i / MARKER_SEG) * Math.PI * 2
    const a1 = ((i + 1) / MARKER_SEG) * Math.PI * 2
    tri([0, 0, 0], [Math.cos(a0), Math.sin(a0), 0], [Math.cos(a1), Math.sin(a1), 0])
  }
  close("dot")
  b = 2

  // link — the two lines from a marker's edge to its child. scale.x is the
  // half-width at the base (the marker's radius), scale.y the distance to the
  // child, and the rotation points +Y at it. Single segments, so there are no
  // joins: the stroke can be as thick as it likes.
  open("link")
  seg([-1, 0, 0], [0, 1, 0])
  seg([1, 0, 0], [0, 1, 0])
  close("link")
  b = 0

  // sphere — three great circles of radius 1.
  open("sphere")
  for (let plane = 0; plane < 3; plane++) {
    for (let i = 0; i < RING_SEG; i++) {
      const a0 = (i / RING_SEG) * Math.PI * 2
      const a1 = ((i + 1) / RING_SEG) * Math.PI * 2
      const c0 = Math.cos(a0),
        s0 = Math.sin(a0)
      const c1 = Math.cos(a1),
        s1 = Math.sin(a1)
      if (plane === 0) seg([0, c0, s0], [0, c1, s1])
      else if (plane === 1) seg([s0, 0, c0], [s1, 0, c1])
      else seg([c0, s0, 0], [c1, s1, 0])
    }
  }
  close("sphere")

  // box — the 12 edges of a cube of half-extent 1.
  open("box")
  for (let axis = 0; axis < 3; axis++) {
    for (let a = -1; a <= 1; a += 2) {
      for (let b = -1; b <= 1; b += 2) {
        const p0: number[] = [0, 0, 0]
        const p1: number[] = [0, 0, 0]
        p0[axis] = -1
        p1[axis] = 1
        p0[(axis + 1) % 3] = p1[(axis + 1) % 3] = a
        p0[(axis + 2) % 3] = p1[(axis + 2) % 3] = b
        seg(p0 as unknown as Vec3Tuple, p1 as unknown as Vec3Tuple)
      }
    }
  }
  close("box")

  // capsule — radius 1, with both hemispheres AT the origin and the cap sign
  // saying which way `extent` pushes them. Nothing here is scaled along Y, so a
  // long thin body keeps round caps.
  open("capsule")
  for (const capSign of [1, -1]) {
    for (let i = 0; i < RING_SEG; i++) {
      const a0 = (i / RING_SEG) * Math.PI * 2
      const a1 = ((i + 1) / RING_SEG) * Math.PI * 2
      seg([Math.cos(a0), 0, Math.sin(a0)], [Math.cos(a1), 0, Math.sin(a1)], capSign, capSign)
    }
    for (let plane = 0; plane < 2; plane++) {
      for (let i = 0; i < CAP_SEG; i++) {
        const a0 = (i / CAP_SEG) * Math.PI
        const a1 = ((i + 1) / CAP_SEG) * Math.PI
        const c0 = Math.cos(a0),
          s0 = Math.sin(a0) * capSign
        const c1 = Math.cos(a1),
          s1 = Math.sin(a1) * capSign
        if (plane === 0) seg([c0, s0, 0], [c1, s1, 0], capSign, capSign)
        else seg([0, s0, c0], [0, s1, c1], capSign, capSign)
      }
    }
  }
  // The cylinder's four side lines. Both endpoints are the same point on the
  // ring — the cap signs are what separate them, one pushed to +extent and one
  // to -extent, so the line is exactly as long as the body is.
  for (const [x, z] of [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
  ]) {
    seg([x, 0, z], [x, 0, z], 1, -1)
  }
  close("capsule")

  // ── Solid volumes ────────────────────────────────────────────────
  // Mode 5: placed in world like their wireframe twins, but filled, so they
  // skip the stroke extrusion. Same size conventions — half-extents for the
  // box, radius for the sphere, radius plus `extent` for the capsule.
  b = 5
  const LAT = 10
  const LON = 24

  open("solidSphere")
  for (let y = 0; y < LAT; y++) {
    const p0 = (y / LAT) * Math.PI - Math.PI / 2
    const p1 = ((y + 1) / LAT) * Math.PI - Math.PI / 2
    for (let x = 0; x < LON; x++) {
      const t0 = (x / LON) * Math.PI * 2
      const t1 = ((x + 1) / LON) * Math.PI * 2
      const at = (phi: number, theta: number): Vec3Tuple => [
        Math.cos(phi) * Math.cos(theta),
        Math.sin(phi),
        Math.cos(phi) * Math.sin(theta),
      ]
      tri(at(p0, t0), at(p1, t0), at(p1, t1))
      tri(at(p0, t0), at(p1, t1), at(p0, t1))
    }
  }
  close("solidSphere")

  open("solidBox")
  for (let axis = 0; axis < 3; axis++) {
    for (const face of [-1, 1]) {
      const corner = (u: number, w: number): Vec3Tuple => {
        const q: number[] = [0, 0, 0]
        q[axis] = face
        q[(axis + 1) % 3] = u
        q[(axis + 2) % 3] = w
        return q as unknown as Vec3Tuple
      }
      tri(corner(-1, -1), corner(1, -1), corner(1, 1))
      tri(corner(-1, -1), corner(1, 1), corner(-1, 1))
    }
  }
  close("solidBox")

  open("solidCapsule")
  for (let x = 0; x < LON; x++) {
    const t0 = (x / LON) * Math.PI * 2
    const t1 = ((x + 1) / LON) * Math.PI * 2
    const r0: Vec3Tuple = [Math.cos(t0), 0, Math.sin(t0)]
    const r1: Vec3Tuple = [Math.cos(t1), 0, Math.sin(t1)]
    // The wall: one quad from the top cap's rim to the bottom cap's.
    triC(r0, 1, r0, -1, r1, -1)
    triC(r0, 1, r1, -1, r1, 1)
    // The two hemispheres, each entirely on its own cap.
    for (const capSign of [1, -1]) {
      for (let y = 0; y < LAT; y++) {
        const p0 = (y / LAT) * (Math.PI / 2)
        const p1 = ((y + 1) / LAT) * (Math.PI / 2)
        const at = (phi: number, theta: number): Vec3Tuple => [
          Math.cos(phi) * Math.cos(theta),
          Math.sin(phi) * capSign,
          Math.cos(phi) * Math.sin(theta),
        ]
        triC(at(p0, t0), capSign, at(p1, t0), capSign, at(p1, t1), capSign)
        triC(at(p0, t0), capSign, at(p1, t1), capSign, at(p0, t1), capSign)
      }
    }
  }
  close("solidCapsule")
  b = 0

  return { vertices: new Float32Array(v), ranges }
}

/** Packs one primitive into the pass's instance layout at `offset` floats. */
export function writeOverlayInstance(p: OverlayPrimitive, out: Float32Array, offset: number): void {
  const r = p.rotation ?? IDENTITY_ROT
  const s = p.scale ?? UNIT_SCALE
  out[offset + 0] = r[0]
  out[offset + 1] = r[1]
  out[offset + 2] = r[2]
  out[offset + 3] = r[3]
  out[offset + 4] = p.position[0]
  out[offset + 5] = p.position[1]
  out[offset + 6] = p.position[2]
  out[offset + 7] = p.extent ?? 0
  out[offset + 8] = s[0]
  out[offset + 9] = s[1]
  out[offset + 10] = s[2]
  out[offset + 11] = p.thickness ?? 3
  out[offset + 12] = p.color[0]
  out[offset + 13] = p.color[1]
  out[offset + 14] = p.color[2]
  out[offset + 15] = p.color[3]
}

const IDENTITY_ROT: readonly [number, number, number, number] = [0, 0, 0, 1]
const UNIT_SCALE: Vec3Tuple = [1, 1, 1]

/**
 * How big this model is, from its bone positions — the reference every default
 * size here is a fraction of, so one set of numbers suits a 1-unit model and a
 * 100-unit one.
 */
function skeletonExtent(world: readonly Mat4[]): number {
  let minY = Infinity,
    maxY = -Infinity,
    minX = Infinity,
    maxX = -Infinity
  for (const m of world) {
    const v = m.values
    if (v[12] < minX) minX = v[12]
    if (v[12] > maxX) maxX = v[12]
    if (v[13] < minY) minY = v[13]
    if (v[13] > maxY) maxY = v[13]
  }
  return Math.max(maxY - minY, maxX - minX, 1e-4)
}

// ──────────────────────────────────────────────────────────────────
// Bones

/** What a bone is driven BY, which is what the colour says. Decided most
 *  specific first: selected, ik, physics, plain.
 *
 *  Four, not the six this started with. Append and twist had classes of their
 *  own and are plain now: a colour has to earn a category, and on a 147-bone
 *  skeleton those two spent their distinction on bones nobody was hunting for.
 *  Both are still on the Bone, for a panel to show where it matters. */
export type BoneClass = "plain" | "ik" | "physics" | "selected"

/**
 * The overlay's colours, and the only place they live.
 *
 * NOT an option a host passes. An overlay means the same thing in every product
 * that draws one — orange is a bone the solver drives, whichever app you are
 * looking at — and a colour each app picks for itself is a colour that drifts.
 * A host that wants its own look builds its own list through setOverlay.
 *
 * Every value is one step of the same ramp (Tailwind's 500), so the classes read
 * as a set and hue is the only thing that separates them.
 *
 * Red and green are never used together — that pair is the confusion most colour
 * blindness causes. Red is spent entirely on `selected`, the one distinction
 * that must never be ambiguous; green is not used at all.
 *
 * A physics-driven bone and a dynamic body share orange on purpose. They are the
 * same fact seen twice — the solver owns this — and colouring them apart would
 * invent a difference that is not there.
 *
 * The CATEGORIES are MMD's, straight out of the PMX file — IK chains, 付与親,
 * 軸制限, a body's physics mode. The colours are not: MMD has no canonical
 * palette for them.
 */
export const DEFAULT_BONE_PALETTE: Record<BoneClass, RGBA> = {
  /** Nothing in particular drives it — most of the skeleton, so it is the colour
   *  the whole overlay reads as. */
  plain: [0.231, 0.51, 0.965, 1], // blue-500
  /** Has an IK chain (ikLinks): the leg IK bones and friends. Grey because an IK
   *  bone is a CONTROL rather than a deformer — nothing is skinned to it, and it
   *  moves a chain instead of the mesh.
   *
   *  Dark grey, not light. Leg IK sits on the floor and reaches almost nothing,
   *  so these are the smallest markers in the rig drawn against the ground — and
   *  gray-400 there measured 1.02:1, which is not a dim marker but no marker. */
  ik: [0.294, 0.333, 0.388, 1], // gray-600
  /** A dynamic rigidbody moves it — hair, skirt, anything the solver owns. Blue,
   *  the same as plain: which bones the solver owns is the rigidbody layer's
   *  question, and it already answers it. Kept as a class so a panel can pick it
   *  out, spending no colour on it here. */
  physics: [0.231, 0.51, 0.965, 1], // blue-500
  /** The one you picked. Red is spent entirely here and used nowhere else, so
   *  nothing on screen can be mistaken for a selection. */
  selected: [0.937, 0.267, 0.267, 1], // red-500
}

export interface BoneOverlayOptions {
  /** Name of the bone drawn in the `selected` colour. */
  selected?: string | null
  /** Marker radius for a bone that governs the whole model, world units. Every
   *  other bone shrinks by how much still hangs off it — see REACH in
   *  boneOverlay. Held between 0.55x and 1.0x. Defaults to 0.5% of the
   *  skeleton's extent, which suits any model scale. */
  markerSize?: number
  /** Stroke width of the links, device pixels. 0.5–24. Default 4. */
  thickness?: number
  /** Length for a leaf whose parent has no length either, world units. Default 0.25. */
  tipLength?: number
  /** Bone names to draw. Omit for all of them. */
  include?: readonly string[]
}

/**
 * The skeleton, MMD's way: a ring with a solid centre at every bone, and two
 * lines from that ring converging on the child's.
 *
 * Ring and centre are filled geometry facing the camera, so a marker is a circle
 * from any angle and can be heavy without a stroke wide enough to tear itself
 * apart. The two link lines leave the ring at its edge — the taper is what says
 * which way the bone points, where a single line between centres would not.
 */
export function boneOverlay(model: Model, options: BoneOverlayOptions = {}): OverlayPrimitive[] {
  const bones = model.getSkeleton().bones
  const world = model.getWorldMatrices()
  const palette = DEFAULT_BONE_PALETTE
  const tipLength = options.tipLength ?? 0.25
  const include = options.include ? new Set(options.include) : null
  const extent = skeletonExtent(world)
  const markerSize = options.markerSize ?? extent * 0.005
  const thickness = options.thickness ?? 4

  const physicsBones = new Set<number>()
  for (const rb of model.getRigidbodies()) {
    if (rb.type !== RigidbodyType.Static && rb.boneIndex >= 0) physicsBones.add(rb.boneIndex)
  }

  // Marker radius per bone, from its REACH: the longest chain still hanging off
  // it, head to furthest descendant.
  //
  // Not the bone's own length, which is what this measured before. A bone's own
  // length says nothing about its place in the rig — every segment of a hair
  // strand is the same short segment, so a strand's ROOT scored no larger than
  // its tip, and nothing stopped a child's segment being longer than its
  // parent's and drawing bigger. Reach cannot do either: a parent's reach is its
  // child's plus the step between them, so a marker never grows going down a
  // chain, and a bone that governs an arm, a strand or a whole side of the body
  // is marked for what it carries rather than for the gap to its first child.
  const depth = new Int32Array(bones.length)
  for (let i = 0; i < bones.length; i++) {
    let d = 0
    let c = bones[i].parentIndex
    // Guarded: a malformed file can cycle, and this must not hang on one.
    while (c >= 0 && c < bones.length && d < bones.length) {
      d++
      c = bones[c].parentIndex
    }
    depth[i] = d
  }
  const deepestFirst = Array.from(bones.keys()).sort((a, b) => depth[b] - depth[a])
  const reach = new Float32Array(bones.length)
  for (const i of deepestFirst) {
    const p = bones[i].parentIndex
    if (p < 0 || p >= world.length || i >= world.length) continue
    const m = world[p].values
    const cm = world[i].values
    const step = Math.hypot(cm[12] - m[12], cm[13] - m[13], cm[14] - m[14])
    if (reach[i] + step > reach[p]) reach[p] = reach[i] + step
  }

  // Square-rooted, so the spread stays legible: without it the root bone's reach
  // is the whole model and everything below it collapses into the floor.
  const sizes = new Float32Array(bones.length)
  const referenceReach = Math.max(extent * 0.2, 1e-6)
  for (let i = 0; i < bones.length; i++) {
    const r = Math.sqrt(Math.max(reach[i], tipLength) / referenceReach)
    sizes[i] = markerSize * Math.min(Math.max(r, 0.55), 1.0)
  }

  // MMD stacks control bones on one point — 全ての親, センター and グルーブ all sit
  // at the hips. Sized per bone they draw as a nest of rings, so a position gets
  // ONE marker, the largest asked for there. Links are still drawn for every
  // bone: it is the markers that overlap, not the connections.
  const grid = Math.max(extent * 1e-4, 1e-6)
  const key = (m: Float32Array) =>
    `${Math.round(m[12] / grid)},${Math.round(m[13] / grid)},${Math.round(m[14] / grid)}`
  const markerAt = new Map<string, number>()
  for (let i = 0; i < bones.length && i < world.length; i++) {
    if (include && !include.has(bones[i].name)) continue
    const k = key(world[i].values)
    const held = markerAt.get(k)
    if (held === undefined || sizes[i] > sizes[held]) markerAt.set(k, i)
  }

  const root = rootTransform(model)
  const out: OverlayPrimitive[] = []
  const head = new Vec3(0, 0, 0)
  const tail = new Vec3(0, 0, 0)
  const dir = new Vec3(0, 0, 0)

  for (let i = 0; i < bones.length && i < world.length; i++) {
    const bone = bones[i]
    if (include && !include.has(bone.name)) continue
    const m = world[i].values
    head.setXYZ(m[12], m[13], m[14])

    const selected = options.selected != null && bone.name === options.selected
    const cls: BoneClass = selected
      ? "selected"
      : bone.ikLinks && bone.ikLinks.length > 0
        ? "ik"
        : physicsBones.has(i)
          ? "physics"
          : "plain"
    const color = palette[cls]
    const size = selected ? sizes[i] * 1.5 : sizes[i]
    const stroke = selected ? thickness * 1.6 : thickness

    // One link per bone, from its PARENT down to it — so every parent-child
    // edge is drawn exactly once and branches all show.
    //
    // Not "from this bone to its first child", which is what this did and what
    // made a thigh point at the wrong place: children are ordered by bone INDEX,
    // not by anatomy, so 左足's children[0] is 左腿物理, a physics bone hanging
    // off the thigh, and 左ひざ — the actual knee — came second. There is no
    // ordering of children that is reliably the continuation. Drawing all of
    // them removes the need to guess.
    const parent = bone.parentIndex
    if (parent >= 0 && parent < world.length) {
      const pm = world[parent].values
      tail.setXYZ(pm[12], pm[13], pm[14])
      const span = Math.hypot(head.x - tail.x, head.y - tail.y, head.z - tail.z)
      if (span > 1e-5) {
        dir.setXYZ((head.x - tail.x) / span, (head.y - tail.y) / span, (head.z - tail.z) / span)
        const rot = Quat.fromUnitVectors(UP, dir)
        out.push(
          applyRoot(root, {
            shape: "link",
            // The taper starts at the PARENT's marker and narrows onto this one,
            // so its base half-width is the parent's radius.
            position: [tail.x, tail.y, tail.z],
            rotation: [rot.x, rot.y, rot.z, rot.w],
            scale: [sizes[parent], span, sizes[parent]],
            color,
            thickness: stroke,
          }),
        )
      }
    }
    // Only the bone that owns this position draws the marker.
    if (markerAt.get(key(m)) !== i) continue
    out.push(
      applyRoot(root, {
        shape: "circle",
        position: [head.x, head.y, head.z],
        scale: [size, size, size],
        color,
        // The same stroke the links carry, so they read as one line.
        thickness: stroke,
      }),
    )
    out.push(
      applyRoot(root, {
        shape: "dot",
        position: [head.x, head.y, head.z],
        scale: [size * DOT_RADIUS, size * DOT_RADIUS, size * DOT_RADIUS],
        color,
      }),
    )
  }
  return out
}

// ──────────────────────────────────────────────────────────────────
// Rigidbodies

export type RigidbodyClass = "static" | "dynamic" | "selected"

/** Solid volumes, so alpha does the work a stroke used to: low enough to see the
 *  body through them and the rig behind them. Cool follows its bone, warm is
 *  driven by the solver. */
export const DEFAULT_RIGIDBODY_PALETTE: Record<RigidbodyClass, RGBA> = {
  static: [0.055, 0.647, 0.914, 0.28], // sky-500
  dynamic: [0.976, 0.451, 0.086, 0.38], // orange-500
  selected: [0.937, 0.267, 0.267, 0.55], // red-500
}

/** The joint cross, the dashed lines to the bodies it holds, and its selection. */
export const DEFAULT_JOINT_PALETTE = {
  cross: [0.055, 0.647, 0.914, 1] as RGBA, // sky-500
  link: [0.055, 0.647, 0.914, 0.5] as RGBA, // sky-500
  selected: [0.937, 0.267, 0.267, 1] as RGBA, // red-500
}

/** The mesh wireframe. Yellow, because it has to carry over pale skin and dark
 *  cloth alike and the rig owns the blues and greens. */
export const DEFAULT_VERTEX_COLOR: RGBA = [0.918, 0.702, 0.031, 0.95] // yellow-500

export interface RigidbodyOverlayOptions {
  /** Stroke width in device pixels. 0.5–24. Default 2.5. */
  thickness?: number
  /** Name of the body drawn in the `selected` colour. */
  selected?: string | null
  /** Body names to draw. Omit for all of them. */
  include?: readonly string[]
}

/**
 * The sphere, box or capsule each rigidbody actually is, where it actually is.
 *
 * With physics running the transforms come from the simulation, so a body drawn
 * here is the body the solver collides — which is the point of drawing them
 * while tuning cloth. Without it they fall back to the bone-driven bind pose.
 */
export function rigidbodyOverlay(
  model: Model,
  physics: RezePhysics | null,
  options: RigidbodyOverlayOptions = {},
): OverlayPrimitive[] {
  const bodies = model.getRigidbodies()
  const palette = DEFAULT_RIGIDBODY_PALETTE
  const extent = skeletonExtent(model.getWorldMatrices())
  const thickness = options.thickness ?? 2.5
  const include = options.include ? new Set(options.include) : null
  const root = rootTransform(model)

  const pos = new Vec3(0, 0, 0)
  const rot = new Quat(0, 0, 0, 1)
  const out: OverlayPrimitive[] = []

  for (let i = 0; i < bodies.length; i++) {
    const rb = bodies[i]
    if (include && !include.has(rb.name)) continue
    resolveBodyTransform(model, physics, i, pos, rot)

    const selected = options.selected != null && rb.name === options.selected
    const cls: RigidbodyClass = selected ? "selected" : rb.type === RigidbodyType.Static ? "static" : "dynamic"
    const color = palette[cls]

    if (rb.shape === RigidbodyShape.Sphere) {
      const r = rb.size.x
      out.push(
        applyRoot(root, {
          shape: "solidSphere",
          position: [pos.x, pos.y, pos.z],
          rotation: [rot.x, rot.y, rot.z, rot.w],
          scale: [r, r, r],
          color,
          thickness: selected ? thickness * 1.8 : thickness,
        }),
      )
    } else if (rb.shape === RigidbodyShape.Box) {
      out.push(
        applyRoot(root, {
          shape: "solidBox",
          position: [pos.x, pos.y, pos.z],
          rotation: [rot.x, rot.y, rot.z, rot.w],
          // PMX box size is half-extents already.
          scale: [rb.size.x, rb.size.y, rb.size.z],
          color,
          thickness: selected ? thickness * 1.8 : thickness,
        }),
      )
    } else {
      const r = rb.size.x
      out.push(
        applyRoot(root, {
          shape: "solidCapsule",
          position: [pos.x, pos.y, pos.z],
          rotation: [rot.x, rot.y, rot.z, rot.w],
          scale: [r, r, r],
          // PMX capsule size.y is the CYLINDER's length, caps not included.
          extent: rb.size.y * 0.5,
          color,
          thickness: selected ? thickness * 1.8 : thickness,
        }),
      )
    }
  }
  return out
}

// ──────────────────────────────────────────────────────────────────
// Joints

export interface JointOverlayOptions {
  /** Stroke width in device pixels. 0.5–24. Default 2.5. */
  thickness?: number
  /** Name of the joint drawn in the `selected` colour. */
  selected?: string | null
  /** Cross arm length, world units. Defaults to 0.6% of the skeleton's extent. */
  size?: number
  /** Draw the dashed lines to the bodies a joint constrains. Default true. */
  links?: boolean
  /** Joint names to draw. Omit for all of them. */
  include?: readonly string[]
}

/**
 * An axis cross at each joint, plus a dashed line to each body it constrains.
 *
 * The cross is where and how the constraint frame sits; the two dashed lines
 * are the pair it holds together, which is the question actually being asked of
 * a skirt with two hundred of them. Dashed, because a relationship should not
 * read as a thing.
 */
export function jointOverlay(
  model: Model,
  physics: RezePhysics | null,
  options: JointOverlayOptions = {},
): OverlayPrimitive[] {
  const joints = model.getJoints()
  const bodies = model.getRigidbodies()
  const extent = skeletonExtent(model.getWorldMatrices())
  const thickness = options.thickness ?? 2.5
  const size = options.size ?? extent * 0.006
  const color = DEFAULT_JOINT_PALETTE.cross
  const linkColor = DEFAULT_JOINT_PALETTE.link
  const selectedColor = DEFAULT_JOINT_PALETTE.selected
  const links = options.links ?? true
  const include = options.include ? new Set(options.include) : null
  const root = rootTransform(model)

  const jointPos = new Vec3(0, 0, 0)
  const bodyPos = new Vec3(0, 0, 0)
  const bodyRot = new Quat(0, 0, 0, 1)
  const out: OverlayPrimitive[] = []

  for (const joint of joints) {
    if (include && !include.has(joint.name)) continue
    const selected = options.selected != null && joint.name === options.selected
    const crossColor = selected ? selectedColor : color

    // A joint's frame is authored in bind pose, so drawing it there would leave
    // every joint on a simulating skirt behind. Carry it on body A instead:
    //   local = bindA⁻¹ · bindJoint,  posed = poseA · local.
    jointPos.setXYZ(joint.position.x, joint.position.y, joint.position.z)
    const rot = Quat.fromEuler(joint.rotation.x, joint.rotation.y, joint.rotation.z)
    const a = joint.rigidbodyIndexA
    if (a >= 0 && a < bodies.length) {
      resolveBodyTransform(model, physics, a, bodyPos, bodyRot)
      const bind = bodies[a]
      const bindRot = Quat.fromEuler(bind.shapeRotation.x, bind.shapeRotation.y, bind.shapeRotation.z)
      const localPos = Quat.rotateVecInv(
        bindRot,
        new Vec3(
          joint.position.x - bind.shapePosition.x,
          joint.position.y - bind.shapePosition.y,
          joint.position.z - bind.shapePosition.z,
        ),
      )
      const localRot = Quat.conjugateInto(bindRot, new Quat(0, 0, 0, 1)).multiply(rot)
      const carried = Quat.rotateVec(bodyRot, localPos)
      jointPos.setXYZ(bodyPos.x + carried.x, bodyPos.y + carried.y, bodyPos.z + carried.z)
      rot.set(bodyRot.multiply(localRot).normalize())
    }

    out.push(
      applyRoot(root, {
        shape: "axes",
        position: [jointPos.x, jointPos.y, jointPos.z],
        rotation: [rot.x, rot.y, rot.z, rot.w],
        scale: [size, size, size],
        color: crossColor,
        thickness: selected ? thickness * 1.8 : thickness,
      }),
    )

    if (!links) continue
    for (const idx of [joint.rigidbodyIndexA, joint.rigidbodyIndexB]) {
      if (idx < 0 || idx >= bodies.length) continue
      resolveBodyTransform(model, physics, idx, bodyPos, bodyRot)
      const line = lineBetween(
        [jointPos.x, jointPos.y, jointPos.z],
        [bodyPos.x, bodyPos.y, bodyPos.z],
        selected ? selectedColor : linkColor,
        true,
      )
      if (line) {
        line.thickness = thickness
        out.push(applyRoot(root, line))
      }
    }
  }
  return out
}

/** A segment from `a` to `b`, solid or dashed. Null when the two coincide. */
export function lineBetween(a: Vec3Tuple, b: Vec3Tuple, color: RGBA, dashed = false): OverlayPrimitive | null {
  const dx = b[0] - a[0],
    dy = b[1] - a[1],
    dz = b[2] - a[2]
  const length = Math.hypot(dx, dy, dz)
  if (length < 1e-6) return null
  const dir = new Vec3(dx / length, dy / length, dz / length)
  const rot = Quat.fromUnitVectors(UP, dir)
  return {
    shape: dashed ? "dashedLine" : "line",
    position: a,
    rotation: [rot.x, rot.y, rot.z, rot.w],
    scale: [1, length, 1],
    color,
  }
}

const UP = new Vec3(0, 1, 0)

// ──────────────────────────────────────────────────────────────────
// Shared

/**
 * Where body `index` is right now: the simulation's answer when physics is
 * running, the bone-driven bind transform when it is not.
 */
function resolveBodyTransform(
  model: Model,
  physics: RezePhysics | null,
  index: number,
  outPos: Vec3,
  outRot: Quat,
): void {
  if (physics) {
    // RigidBodyStore is [...model bodies, floor], so a model index is a store
    // index and needs no mapping.
    const store = physics.getStore()
    if (index < store.count) {
      const i3 = index * 3
      const i4 = index * 4
      outPos.setXYZ(store.positions[i3], store.positions[i3 + 1], store.positions[i3 + 2])
      outRot.setXYZW(store.orientations[i4], store.orientations[i4 + 1], store.orientations[i4 + 2], store.orientations[i4 + 3])
      return
    }
  }
  const rb: Rigidbody = model.getRigidbodies()[index]
  const world = model.getWorldMatrices()
  if (rb.bodyOffsetMatrix && rb.boneIndex >= 0 && rb.boneIndex < world.length) {
    const m = world[rb.boneIndex].multiply(rb.bodyOffsetMatrix)
    outPos.setXYZ(m.values[12], m.values[13], m.values[14])
    outRot.set(m.toQuat().normalize())
    return
  }
  outPos.setXYZ(rb.shapePosition.x, rb.shapePosition.y, rb.shapePosition.z)
  outRot.set(Quat.fromEuler(rb.shapeRotation.x, rb.shapeRotation.y, rb.shapeRotation.z))
}

interface RootTransform {
  identity: boolean
  position: Vec3
  rotation: Quat
  scale: number
}

/** The model's scene placement, which bone and body transforms do NOT carry —
 *  getSkinMatrices applies it on the way to the GPU, so an overlay has to too. */
function rootTransform(model: Model): RootTransform {
  const p = model.position
  const r = model.rotation
  const s = model.scale
  const identity =
    p.x === 0 && p.y === 0 && p.z === 0 && r.x === 0 && r.y === 0 && r.z === 0 && r.w === 1 && s === 1
  return { identity, position: p, rotation: r, scale: s }
}

function applyRoot(root: RootTransform, p: OverlayPrimitive): OverlayPrimitive {
  if (root.identity) return p
  const local = new Vec3(p.position[0] * root.scale, p.position[1] * root.scale, p.position[2] * root.scale)
  const rotated = Quat.rotateVec(root.rotation, local)
  const rot = p.rotation ?? IDENTITY_ROT
  const worldRot = root.rotation.multiply(new Quat(rot[0], rot[1], rot[2], rot[3])).normalize()
  const s = p.scale ?? UNIT_SCALE
  p.position = [rotated.x + root.position.x, rotated.y + root.position.y, rotated.z + root.position.z]
  p.rotation = [worldRot.x, worldRot.y, worldRot.z, worldRot.w]
  p.scale = [s[0] * root.scale, s[1] * root.scale, s[2] * root.scale]
  if (p.extent) p.extent = p.extent * root.scale
  return p
}

