import { POINTS_FLOATS } from "./shaders/points-api"

// The CPU half of `#points`: which bones match, and where they are this frame.

/** Indices of the bones whose names start with `prefix`, in rig order. By
 *  index, because fifty wicks may share one name and a name lookup keeps one. */
export function bonesWithPrefix(bones: { name: string }[], prefix: string): number[] {
  const out: number[] = []
  bones.forEach((b, i) => {
    if (b.name.startsWith(prefix)) out.push(i)
  })
  return out
}

/** Where a model stands: the same scale, rotation and position the cast's
 *  anchors are placed by. */
export type PointPlacement = {
  scale: number
  rotation: { x: number; y: number; z: number; w: number }
  position: { x: number; y: number; z: number }
}

function place(v: [number, number, number], at: PointPlacement): [number, number, number] {
  const s = at.scale
  const x = v[0] * s
  const y = v[1] * s
  const z = v[2] * s
  // v + 2w(q×v) + 2q×(q×v)
  const q = at.rotation
  const tx = 2 * (q.y * z - q.z * y)
  const ty = 2 * (q.z * x - q.x * z)
  const tz = 2 * (q.x * y - q.y * x)
  return [
    x + q.w * tx + (q.y * tz - q.z * ty) + at.position.x,
    y + q.w * ty + (q.z * tx - q.x * tz) + at.position.y,
    z + q.w * tz + (q.x * ty - q.y * tx) + at.position.z,
  ]
}

/**
 * Point `slot`: the bone's head and its tail's end, posed by the bone's world
 * matrix (model space, column-major) and placed by the model. A bone with no
 * tail is a point with no length — its tip is its head.
 */
export function writeBonePoint(
  out: Float32Array,
  slot: number,
  world: Float32Array,
  tail: [number, number, number] | undefined,
  at: PointPlacement,
): void {
  const head: [number, number, number] = [world[12], world[13], world[14]]
  const t = tail ?? [0, 0, 0]
  const tip: [number, number, number] = [
    head[0] + world[0] * t[0] + world[4] * t[1] + world[8] * t[2],
    head[1] + world[1] * t[0] + world[5] * t[1] + world[9] * t[2],
    head[2] + world[2] * t[0] + world[6] * t[1] + world[10] * t[2],
  ]
  const b = 4 + slot * 8
  out.set(place(head, at), b)
  out.set(place(tip, at), b + 4)
}

/** A zeroed points buffer's CPU copy. */
export function pointsData(): Float32Array {
  return new Float32Array(POINTS_FLOATS)
}
