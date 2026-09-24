import { Mat4, Vec3 } from "./math"

/**
 * THE SUN'S SHADOW FOLLOWS THE CAMERA, the way Blender's does.
 *
 * Each cascade is fitted every frame to a slice of the view frustum: the near
 * one to the stretch around what the camera is looking at, the far one to the
 * whole of what it sees. Whatever is in view is in a cascade, and whatever
 * could throw a shadow onto it is in the cascade's depth range, which is fitted
 * to the scene's bounds along the light. A room forty metres across with its
 * window frames behind the camera shadows its floor as the game does; a lone
 * dancer on an empty floor keeps a crisp near map.
 *
 * The earlier shape was two fixed boxes around the camera target, 64 and 256
 * units across, and a stage reached past them: the floor near the windows lay
 * outside every box and was drawn lit, the shadows stopping at the box's edge
 * in a straight line.
 *
 * INVARIANT the sampler and the cull both lean on: the outer cascade CONTAINS
 * the inner one. The sampler falls from cascade 0 to 1 at the box edge, which
 * is only seamless if 1 covers where 0 ends, and the cull tests the OUTERMOST
 * frustum alone. Both hold because the outer slice is the whole frustum, of
 * which the inner slice is a part, and both take the same depth range.
 * tests/shadow-cascades.test.mjs pins it.
 */
export type ShadowCascade = {
  /** Texels per side of this cascade's map — sets the snap quantum. */
  mapSize: number
}

export const SHADOW_CASCADES: readonly ShadowCascade[] = [{ mapSize: 4096 }, { mapSize: 2048 }]

/** How far past the camera's point of interest the near cascade reaches, in
 *  world units: the dancer and the floor around her, at the near map's texel. */
export const NEAR_REACH = 40

/** The view the cascades are fitted to: the camera's eye and basis (world
 *  space, left-handed, +Z forward as the projection is), its vertical field of
 *  view, aspect and clip planes, and how far away the thing it looks at is. */
export type ShadowView = {
  eye: { x: number; y: number; z: number }
  right: { x: number; y: number; z: number }
  up: { x: number; y: number; z: number }
  forward: { x: number; y: number; z: number }
  fov: number
  aspect: number
  near: number
  far: number
  /** Distance from the eye to the camera target, along the view. */
  focus: number
}

/** World-space box around everything drawn, or null for an empty scene. */
export type ShadowBounds = { min: [number, number, number]; max: [number, number, number] } | null

type XYZ = { x: number; y: number; z: number }

/**
 * Where each cascade's slice of the view starts and ends, as distances along
 * the view: [near, split] and [near, farFit]. The far end is where the scene
 * ends, so an empty floor with a dancer on it keeps a short, sharp frustum
 * rather than the camera's far plane.
 */
/** The deepest a cascade reaches along the view, in world units — the far
 *  plane's old cap, which is where every stage's shadows were fitted before. */
const SHADOW_FAR = 8000

export function cascadeSlices(view: ShadowView, bounds: ShadowBounds): [number, number][] {
  let farFit = view.near + 200
  if (bounds) {
    let deepest = 0
    for (let i = 0; i < 8; i++) {
      const cx = (i & 1 ? bounds.max : bounds.min)[0] - view.eye.x
      const cy = (i & 2 ? bounds.max : bounds.min)[1] - view.eye.y
      const cz = (i & 4 ? bounds.max : bounds.min)[2] - view.eye.z
      deepest = Math.max(deepest, cx * view.forward.x + cy * view.forward.y + cz * view.forward.z)
    }
    farFit = deepest + 1
  }
  // Never past SHADOW_FAR, whatever the camera's far plane: a stage's sky dome
  // a kilometre out is in the scene's bounds, and fitting the far cascade to it
  // would spread the stage's shadow map across the sky.
  farFit = Math.min(view.far, SHADOW_FAR, Math.max(view.near + 1, farFit))
  const split = Math.min(farFit, Math.max(view.near + 8, view.focus + NEAR_REACH))
  return [
    [view.near, split],
    [view.near, farFit],
  ]
}

/** The bounding sphere of a frustum slice: on the view axis, at the depth that
 *  balances the near and far rectangles' corners. */
function sliceSphere(view: ShadowView, n: number, f: number): { center: Vec3; radius: number } {
  const t = Math.tan(view.fov / 2)
  const k2 = t * t * (1 + view.aspect * view.aspect)
  let depth: number
  let radius: number
  if (k2 >= (f - n) / (f + n)) {
    depth = f
    radius = f * Math.sqrt(k2)
  } else {
    depth = 0.5 * (f + n) * (1 + k2)
    radius = 0.5 * Math.sqrt((f - n) * (f - n) + 2 * (f * f + n * n) * k2 + (f + n) * (f + n) * k2 * k2)
  }
  const center = new Vec3(
    view.eye.x + view.forward.x * depth,
    view.eye.y + view.forward.y * depth,
    view.eye.z + view.forward.z * depth,
  )
  return { center, radius }
}

/** The eight corners of a slice, for tests and for the fit's own checks. */
export function sliceCorners(view: ShadowView, n: number, f: number): XYZ[] {
  const out: XYZ[] = []
  for (const d of [n, f]) {
    const h = d * Math.tan(view.fov / 2)
    const w = h * view.aspect
    for (const sy of [-1, 1])
      for (const sx of [-1, 1])
        out.push({
          x: view.eye.x + view.forward.x * d + view.right.x * w * sx + view.up.x * h * sy,
          y: view.eye.y + view.forward.y * d + view.right.y * w * sx + view.up.y * h * sy,
          z: view.eye.z + view.forward.z * d + view.right.z * w * sx + view.up.z * h * sy,
        })
  }
  return out
}

/**
 * One cascade's view-projection: an orthographic box around the slice's
 * sphere, snapped to the map's texel grid in the light's right/up plane so a
 * moving camera doesn't shimmer its shadow edges, and reaching along the light
 * from the nearest thing in the scene to the farthest, so everything that
 * could cast into the slice does.
 *
 * Writes the 16 floats into `out` at `offset` and returns `out`.
 */
export function fitShadowVP(
  view: ShadowView,
  slice: [number, number],
  sunDirection: XYZ,
  bounds: ShadowBounds,
  cascade: ShadowCascade,
  out: Float32Array,
  offset: number,
): Float32Array {
  const dir = new Vec3(sunDirection.x, sunDirection.y, sunDirection.z)
  dir.normalize()
  const up = Math.abs(dir.y) > 0.99 ? new Vec3(0, 0, -1) : new Vec3(0, 1, 0)
  const right = Vec3.crossInto(up, dir, new Vec3(0, 0, 0)).normalize()
  const upv = Vec3.crossInto(dir, right, new Vec3(0, 0, 0))

  const { center, radius } = sliceSphere(view, slice[0], slice[1])
  // A texel of the map, in world units; the radius is rounded up onto the
  // grid too, so the box's size does not drift with the fov by fractions.
  const texel = Math.max((2 * radius) / cascade.mapSize, 1e-4)
  const half = Math.ceil(radius / texel) * texel
  const tr = Math.round(center.dot(right) / texel) * texel
  const tu = Math.round(center.dot(upv) / texel) * texel
  const td = center.dot(dir)
  const snapped = new Vec3(
    right.x * tr + upv.x * tu + dir.x * td,
    right.y * tr + upv.y * tu + dir.y * td,
    right.z * tr + upv.z * tu + dir.z * td,
  )

  // Along the light: from the scene's nearest point to its farthest, with a
  // margin so a caster on the box's own face is not clipped. With nothing to
  // fit, the sphere itself.
  let zmin = td - half
  let zmax = td + half
  if (bounds) {
    for (let i = 0; i < 8; i++) {
      const z =
        (i & 1 ? bounds.max : bounds.min)[0] * dir.x + (i & 2 ? bounds.max : bounds.min)[1] * dir.y + (i & 4 ? bounds.max : bounds.min)[2] * dir.z
      zmin = Math.min(zmin, z)
      zmax = Math.max(zmax, z)
    }
  }
  const margin = 2 + 0.02 * (zmax - zmin)
  const back = td - zmin + margin
  const far = back + (zmax - td) + margin
  const eye = new Vec3(snapped.x - dir.x * back, snapped.y - dir.y * back, snapped.z - dir.z * back)
  const viewM = Mat4.lookAt(eye, snapped, up)
  const proj = Mat4.orthographicLh(-half, half, -half, half, 1, far + 1)
  out.set(proj.multiply(viewM).values, offset)
  return out
}

/** Every cascade's view-projection in a row, inner to outer. */
export function buildShadowCascades(view: ShadowView, sunDirection: XYZ, bounds: ShadowBounds, out: Float32Array): Float32Array {
  const slices = cascadeSlices(view, bounds)
  for (let i = 0; i < SHADOW_CASCADES.length; i++) fitShadowVP(view, slices[i], sunDirection, bounds, SHADOW_CASCADES[i], out, i * 16)
  return out
}
