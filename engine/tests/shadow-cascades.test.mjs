// The shadow volumes. Run: npm test.
//
// Cascade 0 must be BIT-IDENTICAL to the single volume every published scene
// was lit by — the refactor that made volumes a list moved arithmetic, and
// float addition is not associative, so "the same formula" is a claim to prove,
// not to assert. The golden here IS the shipped code, re-derived verbatim.

import { test } from "node:test"
import assert from "node:assert/strict"
import { SHADOW_CASCADES, buildShadowVP } from "../dist/shadow-cascades.js"
import { Mat4, Vec3 } from "../dist/math.js"

/** The single-volume arithmetic exactly as engine.ts shipped it, literals and
 *  operation order included. If this and buildShadowVP ever disagree, the
 *  MODULE is what changed — this stays the record of what scenes were lit by. */
function shippedVolume(target, sunDirection) {
  const t = new Vec3(target.x, target.y, target.z)
  const dir = new Vec3(sunDirection.x, sunDirection.y, sunDirection.z)
  dir.normalize()
  const up = Math.abs(dir.y) > 0.99 ? new Vec3(0, 0, -1) : new Vec3(0, 1, 0)
  const right = Vec3.crossInto(up, dir, new Vec3(0, 0, 0)).normalize()
  const upv = Vec3.crossInto(dir, right, new Vec3(0, 0, 0))
  const texel = 64 / 4096
  const tr = Math.round(t.dot(right) / texel) * texel
  const tu = Math.round(t.dot(upv) / texel) * texel
  const td = t.dot(dir)
  const snapped = new Vec3(
    right.x * tr + upv.x * tu + dir.x * td,
    right.y * tr + upv.y * tu + dir.y * td,
    right.z * tr + upv.z * tu + dir.z * td,
  )
  const eye = new Vec3(snapped.x - dir.x * 72, snapped.y - dir.y * 72, snapped.z - dir.z * 72)
  const view = Mat4.lookAt(eye, snapped, up)
  const proj = Mat4.orthographicLh(-32, 32, -32, 32, 1, 140)
  return proj.multiply(view).values
}

/** Camera targets that exercise the snap: origin, mid-dance, far root motion,
 *  a target that sits exactly on a texel boundary, and one just off it. */
const TARGETS = [
  { x: 0, y: 11, z: 0 },
  { x: 3.7123, y: 12.05, z: -8.44 },
  { x: 412.9, y: 10.2, z: -397.6 },
  { x: 0.015625, y: 11, z: 0.015625 },
  { x: 0.0157, y: 11, z: 0.0161 },
]
/** Sun directions including the near-vertical branch (|dir.y| > 0.99). */
const SUNS = [
  { x: -0.35, y: -0.8, z: 0.49 },
  { x: 0.01, y: -0.9999, z: 0.005 },
  { x: 0.7, y: -0.2, z: 0.7 },
]

test("cascade 0 is bit-identical to the shipped single volume", () => {
  const c0 = SHADOW_CASCADES[0]
  assert.deepEqual(
    { span: c0.span, back: c0.back, near: c0.near, far: c0.far, mapSize: c0.mapSize },
    { span: 64, back: 72, near: 1, far: 140, mapSize: 4096 },
    "cascade 0's numbers ARE the published look — changing them is a look change, not a refactor",
  )
  for (const t of TARGETS) {
    for (const sun of SUNS) {
      const out = new Float32Array(16)
      buildShadowVP(t, sun, c0, out, 0)
      const golden = shippedVolume(t, sun)
      for (let i = 0; i < 16; i++) {
        assert.equal(out[i], golden[i], `element ${i} at target ${JSON.stringify(t)}, sun ${JSON.stringify(sun)}`)
      }
    }
  }
})

test("the snap holds: a sub-texel move does not change the matrix", () => {
  for (const c of SHADOW_CASCADES) {
    const texel = c.span / c.mapSize
    const a = new Float32Array(16)
    const b = new Float32Array(16)
    // Nudge by a third of a texel along world x — the snapped light-plane
    // coordinates land on the same grid point. A straight-down sun keeps the
    // depth term out of it: depth tracks the target continuously by design, so
    // any sun with an x component would change the matrix through depth alone.
    const sunFlat = { x: 0, y: -1, z: 0 }
    buildShadowVP({ x: 5, y: 11, z: 3 }, sunFlat, c, a, 0)
    buildShadowVP({ x: 5 + texel / 3, y: 11, z: 3 }, sunFlat, c, b, 0)
    assert.deepEqual([...a], [...b], `cascade span ${c.span}: sub-texel move shimmered the volume`)
    // And a whole-texel move DOES change it — the snap is a grid, not a freeze.
    buildShadowVP({ x: 5 + texel, y: 11, z: 3 }, sunFlat, c, b, 0)
    assert.notDeepEqual([...a], [...b], `cascade span ${c.span}: a full texel move must re-aim the volume`)
  }
})

test("each cascade contains the previous — the cull and the sampler both lean on it", () => {
  // The invariant is over the SPECS, which is what makes it checkable without
  // picking a target: same center, wider span, deeper reach on both sides.
  for (let i = 1; i < SHADOW_CASCADES.length; i++) {
    const inner = SHADOW_CASCADES[i - 1]
    const outer = SHADOW_CASCADES[i]
    assert.ok(outer.span > inner.span, `cascade ${i} must widen`)
    assert.ok(outer.back >= inner.back, `cascade ${i} must reach at least as far behind`)
    assert.ok(
      outer.back - outer.near >= inner.back - inner.near,
      `cascade ${i} must start no later in front of the light`,
    )
    assert.ok(
      outer.far - outer.back >= inner.far - inner.back,
      `cascade ${i} must reach at least as deep past the target`,
    )
  }
})

test("the list is inner to outer and nonempty", () => {
  assert.ok(SHADOW_CASCADES.length >= 1)
  for (let i = 1; i < SHADOW_CASCADES.length; i++) {
    assert.ok(SHADOW_CASCADES[i].span > SHADOW_CASCADES[i - 1].span)
  }
})
