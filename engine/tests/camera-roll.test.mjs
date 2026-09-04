// Orbit roll. Run: npm test.
//
// Roll tips the up vector about the eye→target line, which is three cross
// products written as unrolled arithmetic — precisely the kind of code a sign
// error hides in, and one did: `right` was computed as worldUp × forward instead
// of forward × worldUp, so `up = right × forward` came out pointing DOWN and the
// camera turned upside down the instant roll left zero. It looked fine at roll 0
// because that takes a separate branch with world up handed straight to lookAt.
//
// So the test is CONTINUITY: a hair of roll must be a hair of change. Anything
// that inverts, mirrors or spins the basis fails it by a mile.

import { test } from "node:test"
import assert from "node:assert/strict"
import { Camera } from "../dist/camera.js"
import { Vec3 } from "../dist/math.js"

const mk = () => new Camera(0.7, 1.1, 30, new Vec3(0, 10, 0))

test("a hair of roll is a hair of change", () => {
  const c = mk()
  c.roll = 0
  const flat = [...c.getViewMatrix().values]
  c.roll = 0.002
  const rolled = [...c.getViewMatrix().values]
  for (let i = 0; i < 16; i++) {
    assert.ok(
      Math.abs(flat[i] - rolled[i]) < 0.05,
      `element ${i} jumped ${flat[i].toFixed(3)} -> ${rolled[i].toFixed(3)} for a tenth of a degree`,
    )
  }
})

test("roll turns the horizon and leaves the eye where it was", () => {
  const c = mk()
  const eye = c.getEyePosition()
  c.roll = 0.3
  const after = c.getEyePosition()
  // Roll is an attitude, not a move: the camera stays put and keeps looking at
  // the same point. Only which way is up changes.
  assert.ok(Math.abs(eye.x - after.x) < 1e-9)
  assert.ok(Math.abs(eye.y - after.y) < 1e-9)
  assert.ok(Math.abs(eye.z - after.z) < 1e-9)
})

test("up stays up", () => {
  // The view matrix's second ROW is the camera's up axis in world terms. World
  // up must still be up-ish in the image at a sane lean; the inverted basis put
  // it at −1, which is the bug this exists for.
  const c = mk()
  for (const roll of [0, 0.05, -0.05, 0.3, -0.3]) {
    c.roll = roll
    const v = c.getViewMatrix().values
    // Column-major: row 1 of the rotation is (v[1], v[5], v[9]).
    const upDotWorldUp = v[5]
    assert.ok(upDotWorldUp > 0.4, `roll ${roll}: camera up is ${upDotWorldUp.toFixed(3)} against world up`)
  }
})

test("the up axis turns by exactly the roll angle", () => {
  // The strongest statement available: roll is a rotation of the up axis about
  // the view axis, so the angle between the untilted up and the rolled one IS
  // the roll. Exact, sign-agnostic, and it pins the magnitude that the
  // continuity test above only bounds.
  const c = mk()
  const upOf = () => {
    const v = c.getViewMatrix().values
    return [v[1], v[5], v[9]]
  }
  c.roll = 0
  const u0 = upOf()
  for (const roll of [0.05, -0.05, 0.3, -0.3, 0.7]) {
    c.roll = roll
    const u = upOf()
    const dot = u0[0] * u[0] + u0[1] * u[1] + u0[2] * u[2]
    const ang = Math.acos(Math.min(1, Math.max(-1, dot)))
    assert.ok(
      Math.abs(ang - Math.abs(roll)) < 1e-6,
      `roll ${roll}: up turned ${ang.toFixed(6)} rad, not ${Math.abs(roll)}`,
    )
  }
})
