// The shadow volumes, as data — a list the engine iterates rather than one
// hardcoded box, so a second cascade is a list entry and not a rewrite.
//
// Pure math, its own module for the same reason param-track.ts is: the engine
// class needs a GPU to construct, and the one thing that ever goes WRONG with a
// shadow volume is arithmetic — a snap that stops snapping, an eye that lands
// inside the near plane. Headless tests can hold this half to golden values.
//
// The arithmetic is the shipped single-volume code, operation for operation:
// float addition is not associative, so "the same formula, reordered" is not
// the same matrix, and cascade 0 must be BIT-IDENTICAL to the volume every
// published scene was lit by.

import { Mat4, Vec3 } from "./math"

type ShadowCascade = {
  /** World units across the ortho box, both axes. */
  span: number
  /** How far behind the target the light's eye sits, along -sunDir. */
  back: number
  /** Ortho near/far, in world units from that eye. */
  near: number
  far: number
  /** Texels per side of this cascade's map — sets the snap quantum. */
  mapSize: number
}

/**
 * The list, inner to outer. INVARIANT the sampler and the cull both lean on:
 * each cascade's box must CONTAIN the previous one (same snapped target, wider
 * span, deeper reach), because
 *
 *   - the sampler falls from cascade i to i+1 at the box edge, which is only
 *     seamless if i+1 covers where i ends, and
 *   - the cull tests ONE frustum — the outermost — and the rasterizer clips
 *     each cascade to its own box. That is the same argument that made
 *     single-volume shadow culling exact: anything rejected was contributing
 *     nothing anywhere. Concentric containment is what keeps it true for a
 *     LIST. tests/shadow-cascades.test.mjs pins it.
 */
export const SHADOW_CASCADES: readonly ShadowCascade[] = [
  // The shipped volume: 64 units at 4096² ≈ 64 texels/unit — crisp contact
  // shadows on the ground catcher (2048 read visibly blurry).
  { span: 64, back: 72, near: 1, far: 140, mapSize: 4096 },
  // The stage volume: 4× the span on each side, so a set piece 100 units out
  // still throws and receives shade instead of popping lit at the near box's
  // edge. 2048² over 256 units ≈ 8 texels/unit — soft, and read at distances
  // where soft is what a shadow looks like anyway. 16 MB where the near map
  // is 64. Depth reach scales with the span (same eye direction, deeper box),
  // keeping the containment invariant checkable from the specs alone.
  { span: 256, back: 288, near: 1, far: 560, mapSize: 2048 },
]

/**
 * One cascade's view-projection, following the camera target.
 *
 * The target is snapped to this cascade's OWN texel quantum in the light's
 * right/up plane, so a moving volume doesn't shimmer its shadow edges while
 * running — each cascade snaps to its own grid, coarser maps snapping coarser.
 *
 * Writes the 16 floats into `out` at `offset` and returns `out`.
 */
export function buildShadowVP(
  target: { x: number; y: number; z: number },
  sunDirection: { x: number; y: number; z: number },
  cascade: ShadowCascade,
  out: Float32Array,
  offset: number,
): Float32Array {
  const dir = new Vec3(sunDirection.x, sunDirection.y, sunDirection.z)
  dir.normalize()
  const up = Math.abs(dir.y) > 0.99 ? new Vec3(0, 0, -1) : new Vec3(0, 1, 0)

  const t = new Vec3(target.x, target.y, target.z)
  const right = Vec3.crossInto(up, dir, new Vec3(0, 0, 0)).normalize()
  const upv = Vec3.crossInto(dir, right, new Vec3(0, 0, 0))
  const texel = cascade.span / cascade.mapSize
  const tr = Math.round(t.dot(right) / texel) * texel
  const tu = Math.round(t.dot(upv) / texel) * texel
  const td = t.dot(dir)
  const snapped = new Vec3(
    right.x * tr + upv.x * tu + dir.x * td,
    right.y * tr + upv.y * tu + dir.y * td,
    right.z * tr + upv.z * tu + dir.z * td,
  )

  const eye = new Vec3(snapped.x - dir.x * cascade.back, snapped.y - dir.y * cascade.back, snapped.z - dir.z * cascade.back)
  const view = Mat4.lookAt(eye, snapped, up)
  const half = cascade.span / 2
  // The shadow map keeps the NON-reversed convention (orthographicLh maps z to
  // [0,1] front-to-back) — reversing it buys nothing for an ortho box and the
  // +2 pipeline depth bias is signed against this direction.
  const proj = Mat4.orthographicLh(-half, half, -half, half, cascade.near, cascade.far)
  const vp = proj.multiply(view)
  out.set(vp.values, offset)
  return out
}
