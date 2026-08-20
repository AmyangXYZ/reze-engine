import { test } from "node:test"
import assert from "node:assert/strict"
import { VMDWriter } from "../dist/vmd-writer.js"
import { VMDLoader } from "../dist/vmd-loader.js"
import { CameraAnimation } from "../dist/camera-animation.js"
import { Vec3 } from "../dist/math.js"

const kf = (frame, distance, target, rotation, fov) => ({
  frame,
  distance,
  target: new Vec3(...target),
  rotation: new Vec3(...rotation),
  fov,
})

test("a written camera VMD parses back to the same keyframes", () => {
  const frames = [
    kf(0, -35.5, [0, 10, 0], [0, 0, 0], 30),
    kf(60, -12.25, [1.5, 14, -2.5], [0.1, -0.2, 0.05], 45),
    kf(120, -50, [-3, 8.75, 4], [-0.3, 1.2, 0], 22),
  ]
  const buf = new VMDWriter().writeCamera(frames)
  const back = VMDLoader.loadCameraFromBuffer(buf)

  assert.equal(back.length, frames.length)
  for (let i = 0; i < frames.length; i++) {
    const a = frames[i]
    const b = back[i]
    assert.equal(b.frame, a.frame)
    // f32 round trip: compare at single precision, not exact equality.
    assert.ok(Math.abs(b.distance - a.distance) < 1e-5, `distance ${i}`)
    for (const axis of ["x", "y", "z"]) {
      assert.ok(Math.abs(b.target[axis] - a.target[axis]) < 1e-5, `target.${axis} ${i}`)
      assert.ok(Math.abs(b.rotation[axis] - a.rotation[axis]) < 1e-5, `rotation.${axis} ${i}`)
    }
    assert.equal(b.fov, a.fov)
    assert.equal(b.interpolation.length, 24)
  }
})

test("keyframes are sorted on the way out", () => {
  const buf = new VMDWriter().writeCamera([
    kf(120, -20, [0, 0, 0], [0, 0, 0], 30),
    kf(0, -30, [0, 0, 0], [0, 0, 0], 30),
    kf(60, -25, [0, 0, 0], [0, 0, 0], 30),
  ])
  assert.deepEqual(VMDLoader.loadCameraFromBuffer(buf).map((f) => f.frame), [0, 60, 120])
})

test("an authored keyframe needs no interpolation table", () => {
  // The whole point of making it optional: this must not throw, and must
  // produce a curve the sampler can read.
  const buf = new VMDWriter().writeCamera([kf(0, -30, [0, 10, 0], [0, 0, 0], 30)])
  const back = VMDLoader.loadCameraFromBuffer(buf)
  assert.equal(back[0].interpolation.length, 24)
  assert.ok(back[0].interpolation.some((b) => b !== 0), "should carry the linear default, not zeros")
})

test("a round-tripped track samples to the pose it was built from", () => {
  const frames = [
    kf(0, -35, [0, 10, 0], [0, 0, 0], 30),
    kf(30, -20, [0, 12, 0], [0, 0.5, 0], 40),
  ]
  const back = VMDLoader.loadCameraFromBuffer(new VMDWriter().writeCamera(frames))
  const anim = new CameraAnimation(back)
  const at0 = anim.sample(0)
  assert.ok(Math.abs(at0.distance - -35) < 1e-5)
  assert.ok(Math.abs(at0.target.y - 10) < 1e-5)
  // fov is degrees in the file, radians out of the sampler.
  assert.ok(Math.abs(at0.fov - (30 * Math.PI) / 180) < 1e-5)
  // Endpoint at the last keyframe, whatever the curve does between.
  const at1 = anim.sample(1)
  assert.ok(Math.abs(at1.distance - -20) < 1e-5)
})

test("camera bytes carry no bone or morph frames", () => {
  const buf = new VMDWriter().writeCamera([kf(0, -30, [0, 10, 0], [0, 0, 0], 30)])
  const view = new DataView(buf)
  assert.equal(view.getUint32(50, true), 0, "bone count")
  assert.equal(view.getUint32(54, true), 0, "morph count")
  assert.equal(view.getUint32(58, true), 1, "camera count")
  assert.equal(buf.byteLength, 30 + 20 + 4 + 4 + 4 + 61)
})
