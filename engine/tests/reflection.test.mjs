// The mirror camera. Run: npm test.
//
// The claim buildMirrorCamera makes is algebraic — view' = view × R with R the
// reflection about y = h — and it is written as unrolled arithmetic for
// exactness, which is precisely the kind of code a sign error hides in. So the
// tests check the ALGEBRA: drawing a point through the mirror camera must land
// where the camera would draw that point's reflection.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { buildMirrorCamera, reflectionAboutY } from "../dist/reflection.js"
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

const POINTS = [
  [0, 12, 0],
  [3.5, 0.0, -2.2],
  [-8, 25, 14],
  [0.01, 0.001, 0.01],
]

test("the mirror camera draws a point where the camera draws its reflection", () => {
  for (const h of [0, 1.5]) {
    const cam = cameraBlock([10, 18, -25], [0, 11, 0])
    const mir = buildMirrorCamera(cam, h, new Float32Array(40))
    for (const p of POINTS) {
      const reflected = [p[0], 2 * h - p[1], p[2]]
      const a = xform(mir, 0, p) // mirror view of the point
      const b = xform(cam, 0, reflected) // camera view of its reflection
      for (let i = 0; i < 4; i++) {
        assert.ok(Math.abs(a[i] - b[i]) < 1e-4, `h=${h} p=${p}: component ${i}: ${a[i]} vs ${b[i]}`)
      }
    }
  }
})

test("a point ON the plane lands in the same place through either camera", () => {
  const cam = cameraBlock([6, 14, -20], [0, 10, 2])
  const mir = buildMirrorCamera(cam, 0, new Float32Array(40))
  const onPlane = [4.2, 0, -1.3]
  const a = xform(cam, 0, onPlane)
  const b = xform(mir, 0, onPlane)
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(a[i] - b[i]) < 1e-4, `component ${i}`)
})

test("reflecting twice is the identity", () => {
  const cam = cameraBlock([10, 18, -25], [0, 11, 0])
  const once = buildMirrorCamera(cam, 0.75, new Float32Array(40))
  const twice = buildMirrorCamera(once, 0.75, new Float32Array(40))
  for (let i = 0; i < 36; i++) {
    assert.ok(Math.abs(twice[i] - cam[i]) < 1e-5, `element ${i}: ${twice[i]} vs ${cam[i]}`)
  }
})

test("the projection and target height ride along unchanged; the eye mirrors", () => {
  const cam = cameraBlock([10, 18, -25], [0, 11, 0])
  const mir = buildMirrorCamera(cam, 2, new Float32Array(40))
  for (let i = 16; i < 32; i++) assert.equal(mir[i], cam[i], `projection element ${i - 16}`)
  assert.equal(mir[32], 10)
  assert.equal(mir[33], 2 * 2 - 18)
  assert.equal(mir[34], -25)
  assert.equal(mir[35], 1080)
})

test("the reflection matrix is its own inverse and flips handedness", () => {
  const r = reflectionAboutY(3)
  const p = [1.5, 7, -4]
  const once = xform(r, 0, p)
  assert.deepEqual(once.slice(0, 3), [1.5, 2 * 3 - 7, -4])
  const back = xform(r, 0, once.slice(0, 3))
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(back[i] - p[i]) < 1e-6)
  // det = -1: the winding flip is why the outline (cullMode back) sits the
  // mirror out — this pins that the flip is real, not folklore.
  const det = r[0] * r[5] * r[10]
  assert.equal(det, -1)
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
  // Orbiting: the same five, with no roll. Distance NEGATIVE, because in a VMD
  // the camera sits behind its target and a host reading one shape must not get
  // two conventions.
  assert.match(fn, /distance: -this\.radius/)
  assert.match(fn, /rotation: new Vec3\(this\.beta - Math\.PI \/ 2, -this\.alpha, 0\)/)
  // A COPY, not the live vectors — a caller sampling once a frame into an array
  // would otherwise end up with one pose repeated however many times it read.
  assert.match(fn, /target: new Vec3\(this\._vmdTarget\.x/)
})
