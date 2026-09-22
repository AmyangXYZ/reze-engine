// The sun's shadow cascades, fitted to the camera — the fit math alone, without
// a GPU. What the sampler and the cull lean on is pinned here: every slice of
// the view is inside its cascade, the outer cascade contains the inner, the
// scene's depth is inside every cascade's range, and a sub-texel camera move
// leaves the box where it was.
import { test } from "node:test"
import assert from "node:assert/strict"
import { SHADOW_CASCADES, NEAR_REACH, buildShadowCascades, cascadeSlices, fitShadowVP, sliceCorners } from "../dist/shadow-cascades.js"

/** A camera looking down +Z from 60 units back, a little above the floor. */
function view(overrides = {}) {
  return {
    eye: { x: 0, y: 14, z: -60 },
    right: { x: 1, y: 0, z: 0 },
    up: { x: 0, y: 1, z: 0 },
    forward: { x: 0, y: 0, z: 1 },
    fov: Math.PI / 4,
    aspect: 16 / 9,
    near: 0.5,
    far: 5000,
    focus: 60,
    ...overrides,
  }
}

/** A room 500 units across around the origin — a game stage. */
const ROOM = { min: [-250, -1, -250], max: [250, 60, 250] }
/** A dancer alone. */
const DANCER = { min: [-6, 0, -6], max: [6, 18, 6] }

const SUNS = [
  { x: 0.951, y: -0.196, z: -0.241 },
  { x: -0.3, y: -0.8, z: 0.5 },
  { x: 0, y: -1, z: 0 },
  { x: 0.01, y: -0.999, z: 0 },
]

function project(vp, p) {
  const x = vp[0] * p.x + vp[4] * p.y + vp[8] * p.z + vp[12]
  const y = vp[1] * p.x + vp[5] * p.y + vp[9] * p.z + vp[13]
  const z = vp[2] * p.x + vp[6] * p.y + vp[10] * p.z + vp[14]
  const w = vp[3] * p.x + vp[7] * p.y + vp[11] * p.z + vp[15]
  return { x: x / w, y: y / w, z: z / w }
}

const inside = (n) => Math.abs(n.x) <= 1.0001 && Math.abs(n.y) <= 1.0001 && n.z >= -0.0001 && n.z <= 1.0001

test("every corner of a slice lands inside its cascade", () => {
  for (const bounds of [ROOM, DANCER, null]) {
    for (const sun of SUNS) {
      const v = view()
      const slices = cascadeSlices(v, bounds)
      const out = buildShadowCascades(v, sun, bounds, new Float32Array(32))
      for (let i = 0; i < SHADOW_CASCADES.length; i++) {
        const vp = out.subarray(i * 16, i * 16 + 16)
        for (const c of sliceCorners(v, slices[i][0], slices[i][1])) {
          const n = project(vp, c)
          assert.ok(inside(n), `cascade ${i}, sun ${JSON.stringify(sun)}: corner ${JSON.stringify(c)} projects to ${JSON.stringify(n)}`)
        }
      }
    }
  }
})

test("the outer cascade contains the inner — the cull and the sampler both lean on it", () => {
  for (const bounds of [ROOM, DANCER]) {
    for (const sun of SUNS) {
      const v = view()
      const slices = cascadeSlices(v, bounds)
      const out = buildShadowCascades(v, sun, bounds, new Float32Array(32))
      const outer = out.subarray(16, 32)
      for (const c of sliceCorners(v, slices[0][0], slices[0][1])) {
        assert.ok(inside(project(outer, c)), `inner corner ${JSON.stringify(c)} outside the outer cascade`)
      }
    }
  }
})

test("the scene's whole depth is inside every cascade's range along the light", () => {
  for (const sun of SUNS) {
    const v = view()
    const out = buildShadowCascades(v, sun, ROOM, new Float32Array(32))
    for (let i = 0; i < SHADOW_CASCADES.length; i++) {
      const vp = out.subarray(i * 16, i * 16 + 16)
      // The room's corners that fall inside the box laterally must fall inside
      // its depth too: a window frame across the room still casts.
      for (let k = 0; k < 8; k++) {
        const c = { x: (k & 1 ? ROOM.max : ROOM.min)[0], y: (k & 2 ? ROOM.max : ROOM.min)[1], z: (k & 4 ? ROOM.max : ROOM.min)[2] }
        const n = project(vp, c)
        assert.ok(n.z >= -0.0001 && n.z <= 1.0001, `cascade ${i}: room corner ${JSON.stringify(c)} at depth ${n.z}`)
      }
    }
  }
})

test("the near slice reaches past the point of interest and the far slice ends at the scene", () => {
  const v = view()
  const [near, far] = cascadeSlices(v, ROOM)
  assert.equal(near[0], v.near)
  assert.ok(Math.abs(near[1] - (v.focus + NEAR_REACH)) < 1e-9)
  assert.ok(far[1] > 300 && far[1] < 320, `the far slice ends where the room does: ${far[1]}`)
  const [, alone] = cascadeSlices(v, DANCER)
  assert.ok(alone[1] < 70, `a dancer alone keeps a short frustum: ${alone[1]}`)
})

test("the snap holds: a sub-texel camera move does not change the matrix", () => {
  const sun = SUNS[0]
  for (let i = 0; i < SHADOW_CASCADES.length; i++) {
    const v = view()
    const slice = cascadeSlices(v, ROOM)[i]
    const a = fitShadowVP(v, slice, sun, ROOM, SHADOW_CASCADES[i], new Float32Array(16), 0)
    // The texel is set by the slice's sphere; a move a fifth of it along the
    // light's right lands on the same grid point.
    // The box's half-size is the inverse of the ortho scale, which is the
    // length of the matrix's first row (the row also carries the rotation).
    const half = 1 / Math.hypot(a[0], a[4], a[8])
    const texel = (2 * half) / SHADOW_CASCADES[i].mapSize
    const right = { x: -sun.z, y: 0, z: sun.x }
    const rl = Math.hypot(right.x, right.z)
    // A step this small can cross a grid line in at most ONE direction, so
    // one of the two moves must land on the same box.
    const step = texel / 5
    const same = [1, -1].filter((sign) => {
      const moved = view({ eye: { x: v.eye.x + (right.x / rl) * step * sign, y: v.eye.y, z: v.eye.z + (right.z / rl) * step * sign } })
      const b = fitShadowVP(moved, slice, sun, ROOM, SHADOW_CASCADES[i], new Float32Array(16), 0)
      // Within float noise: a move that is perpendicular to the light in exact
      // arithmetic is not quite in floating point, and the depth term wobbles
      // at the 1e-13 level. A texel step shows up at 1e-4.
      return [...a].every((x, k) => Math.abs(x - b[k]) < 1e-6)
    })
    assert.ok(same.length >= 1, `cascade ${i}: a sub-texel move shimmered the box both ways`)
    // And a full texel always re-aims it.
    const far = view({ eye: { x: v.eye.x + (right.x / rl) * texel * 1.5, y: v.eye.y, z: v.eye.z + (right.z / rl) * texel * 1.5 } })
    const c = fitShadowVP(far, slice, sun, ROOM, SHADOW_CASCADES[i], new Float32Array(16), 0)
    assert.ok([...a].some((x, k) => Math.abs(x - c[k]) > 1e-6), `cascade ${i}: a texel move must re-aim the box`)
  }
})

test("the list is inner to outer and nonempty", () => {
  assert.ok(SHADOW_CASCADES.length >= 1)
  const v = view()
  const slices = cascadeSlices(v, ROOM)
  for (let i = 1; i < slices.length; i++) assert.ok(slices[i][1] >= slices[i - 1][1])
})
