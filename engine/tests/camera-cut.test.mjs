import { test } from "node:test"
import assert from "node:assert/strict"
import { CameraAnimation } from "../dist/camera-animation.js"
import { Vec3 } from "../dist/math.js"

const kf = (frame, distance, fov) => ({ frame, distance, target: new Vec3(0, 10, 0), rotation: new Vec3(0, 0, 0), fov })

test("keys one frame apart are a cut: the earlier key holds", () => {
  const anim = new CameraAnimation([kf(0, -40, 30), kf(10, -50, 28), kf(11, -100, 20), kf(21, -60, 40)])
  const mid = anim.sample(10.5 / 30)
  assert.equal(mid.distance, -50)
  assert.ok(Math.abs(mid.fov - (28 * Math.PI) / 180) < 1e-9)
  assert.ok(Math.abs(anim.sample(11 / 30).distance + 100) < 0.01)
})

test("keys further apart still interpolate", () => {
  const anim = new CameraAnimation([kf(0, -40, 30), kf(2, -60, 30)])
  const d = anim.sample(1 / 30).distance
  assert.ok(d < -40 && d > -60)
})
