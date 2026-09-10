// The mirror camera. Run: npm test.
//
// The claim buildMirrorCamera makes is algebraic — view' = view × R with R the
// Householder reflection about (n, d) — and a sign error in it hides perfectly:
// the picture still looks like a reflection, of the wrong thing. So the tests
// check the ALGEBRA: drawing a point through the mirror camera must land where
// the camera would draw that point's reflection.
//
// The planes below are deliberately not all horizontal. A floor was the only
// plane this supported once, and every bug that generalising it could introduce
// lives in the terms that vanish when n = (0, 1, 0).

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { buildMirrorCamera, reflectionAboutPlane, planeFromPointNormal } from "../dist/reflection.js"
import { Mat4, Vec3 } from "../dist/math.js"

/** Column-major 4x4 times (x,y,z,1). */
function xform(m, o, p) {
  const [x, y, z] = p
  return [
    m[o + 0] * x + m[o + 4] * y + m[o + 8] * z + m[o + 12],
    m[o + 1] * x + m[o + 5] * y + m[o + 9] * z + m[o + 13],
    m[o + 2] * x + m[o + 6] * y + m[o + 10] * z + m[o + 14],
    m[o + 3] * x + m[o + 7] * y + m[o + 11] * z + m[o + 15],
  ]
}

/** A plausible camera block: view at 0, projection at 16, eye at 32. */
function cameraBlock(eye, target) {
  const block = new Float32Array(40)
  const view = Mat4.lookAt(new Vec3(...eye), new Vec3(...target), new Vec3(0, 1, 0))
  block.set(view.values, 0)
  const proj = Mat4.orthographicLh(-20, 20, -20, 20, 0.1, 200)
  block.set(proj.values, 16)
  block[32] = eye[0]
  block[33] = eye[1]
  block[34] = eye[2]
  block[35] = 1080
  return block
}

/** p - 2(n·p + d)n — the reflection the matrix is supposed to be. */
function reflect(plane, p) {
  const t = 2 * (plane[0] * p[0] + plane[1] * p[1] + plane[2] * p[2] + plane[3])
  return [p[0] - t * plane[0], p[1] - t * plane[1], p[2] - t * plane[2]]
}

/** Unit-normal planes: the floor, a raised floor, a wall behind, a wall to the
 *  side, and one tilted like a standing mirror leaning back on its foot. */
const PLANES = [
  new Float32Array([0, 1, 0, 0]),
  new Float32Array([0, 1, 0, -1.5]),
  new Float32Array([0, 0, 1, 4]),
  new Float32Array([1, 0, 0, 2.5]),
  planeFromPointNormal(-1, 1.2, -0.5, 0.35, 0.12, 0.93, new Float32Array(4)),
]

const POINTS = [
  [0, 12, 0],
  [3.5, 0.0, -2.2],
  [-8, 25, 14],
  [0.01, 0.001, 0.01],
]

test("the mirror camera draws a point where the camera draws its reflection", () => {
  for (const plane of PLANES) {
    const cam = cameraBlock([10, 18, -25], [0, 11, 0])
    const mir = buildMirrorCamera(cam, plane, new Float32Array(40))
    for (const p of POINTS) {
      const a = xform(mir, 0, p) // mirror view of the point
      const b = xform(cam, 0, reflect(plane, p)) // camera view of its reflection
      for (let i = 0; i < 4; i++) {
        assert.ok(Math.abs(a[i] - b[i]) < 1e-4, `plane=${plane} p=${p}: component ${i}: ${a[i]} vs ${b[i]}`)
      }
    }
  }
})

test("a point ON the plane lands in the same place through either camera", () => {
  const cam = cameraBlock([6, 14, -20], [0, 10, 2])
  // One point on each plane, found by reflecting an arbitrary point and taking
  // the midpoint — which is on the plane whatever the plane is.
  for (const plane of PLANES) {
    const seed = [4.2, 3.1, -1.3]
    const r = reflect(plane, seed)
    const onPlane = [(seed[0] + r[0]) / 2, (seed[1] + r[1]) / 2, (seed[2] + r[2]) / 2]
    const mir = buildMirrorCamera(cam, plane, new Float32Array(40))
    const a = xform(cam, 0, onPlane)
    const b = xform(mir, 0, onPlane)
    for (let i = 0; i < 4; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-4, `plane=${plane} component ${i}`)
  }
})

test("reflecting twice is the identity", () => {
  const cam = cameraBlock([10, 18, -25], [0, 11, 0])
  for (const plane of PLANES) {
    const once = buildMirrorCamera(cam, plane, new Float32Array(40))
    const twice = buildMirrorCamera(once, plane, new Float32Array(40))
    for (let i = 0; i < 36; i++) {
      assert.ok(Math.abs(twice[i] - cam[i]) < 1e-4, `plane=${plane} element ${i}: ${twice[i]} vs ${cam[i]}`)
    }
  }
})

test("the projection and target height ride along unchanged; the eye mirrors", () => {
  const cam = cameraBlock([10, 18, -25], [0, 11, 0])
  const plane = new Float32Array([0, 1, 0, -2])
  const mir = buildMirrorCamera(cam, plane, new Float32Array(40))
  for (let i = 16; i < 32; i++) assert.equal(mir[i], cam[i], `projection element ${i - 16}`)
  assert.ok(Math.abs(mir[32] - 10) < 1e-5)
  // y = 2 is the plane; the eye at 18 lands at 2*2 - 18 = -14.
  assert.ok(Math.abs(mir[33] - -14) < 1e-5, `eye y: ${mir[33]}`)
  assert.ok(Math.abs(mir[34] - -25) < 1e-5)
  assert.equal(mir[35], 1080)
})

test("a wall reflects across it and leaves the other two axes alone", () => {
  // The case a floor-only fold could never express, stated in plain numbers so
  // a regression reads as a wrong coordinate rather than a failed epsilon.
  const wall = new Float32Array([0, 0, 1, 4]) // z = -4
  assert.deepEqual(reflect(wall, [3, 5, 0]), [3, 5, -8])
  assert.deepEqual(reflect(wall, [3, 5, -4]), [3, 5, -4])
})

test("the reflection matrix is its own inverse and flips handedness", () => {
  for (const plane of PLANES) {
    const r = reflectionAboutPlane(plane)
    const p = [1.5, 7, -4]
    const once = xform(r, 0, p)
    const expected = reflect(plane, p)
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(once[i] - expected[i]) < 1e-5)
    const back = xform(r, 0, once.slice(0, 3))
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(back[i] - p[i]) < 1e-5)
    // det = -1: the winding flip is why the outline (cullMode back) sits the
    // mirror out — this pins that the flip is real, not folklore.
    const det =
      r[0] * (r[5] * r[10] - r[6] * r[9]) -
      r[4] * (r[1] * r[10] - r[2] * r[9]) +
      r[8] * (r[1] * r[6] - r[2] * r[5])
    assert.ok(Math.abs(det + 1) < 1e-5, `det=${det}`)
  }
})

test("planeFromPointNormal normalizes, so a scaled basis vector still places the plane", () => {
  const out = planeFromPointNormal(0, 2, 0, 0, 7, 0, new Float32Array(4))
  assert.deepEqual([...out], [0, 1, 0, -2])
  // The point is on the plane it just made, whatever length the normal had.
  assert.ok(Math.abs(out[0] * 0 + out[1] * 2 + out[2] * 0 + out[3]) < 1e-6)
})

test("the camera reports its pose the same way in both modes", () => {
  // For a host writing the shot out to something else — an AE composition, a
  // .vmd, a log. Both modes already hold the five channels MMD states a camera
  // in; neither should make a caller take a view matrix apart to get them, and
  // neither should make it ask which mode is driving first.
  const src = readFileSync(new URL("../src/camera.ts", import.meta.url), "utf8")
  const body = src.replace(/\/\/[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "")
  const fn = body.slice(body.indexOf("getPose(): CameraPose"), body.indexOf("setVmdPose"))
  // VMD-driven: the stored pose, unfactored.
  assert.match(fn, /if \(this\.vmdDriven\)/)
  assert.match(fn, /distance: this\._vmdDistance/)
  // Orbiting: the same five. Distance NEGATIVE, because in a VMD the camera sits
  // behind its target and a host reading one shape must not get two conventions.
  assert.match(fn, /distance: -this\.radius/)
  // z is the ROLL, not a hard zero. It was zero for as long as an orbit could
  // not lean; now that it can, a hard zero here would report a rolled shot as a
  // level one — and this is the channel the AE rig and the VMD writer read, so
  // the lean would survive on screen and vanish from every export of it.
  assert.match(fn, /rotation: new Vec3\(this\.beta - Math\.PI \/ 2, -this\.alpha, this\.roll\)/)
  // A COPY, not the live vectors — a caller sampling once a frame into an array
  // would otherwise end up with one pose repeated however many times it read.
  assert.match(fn, /target: new Vec3\(this\._vmdTarget\.x/)
})
