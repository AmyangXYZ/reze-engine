// Math primitive tests. Run: npm test (node --test; build first — tests import dist/).
// Covers the consolidated primitives added for rig/MiKaPo/studio plus the pre-existing
// slerp/euler/bezier code, which had no tests before.

import { test } from "node:test"
import assert from "node:assert/strict"
import { Vec3, Quat, Mat4, easeInOut } from "../dist/math.js"
import { bezierInterpolate, interpolateControlPoints } from "../dist/animation.js"

const EPS = 1e-6

function assertNear(actual, expected, eps = EPS, msg = "") {
  assert.ok(
    Math.abs(actual - expected) < eps,
    `${msg} expected ${expected}, got ${actual} (eps ${eps})`
  )
}

function assertVecNear(v, x, y, z, eps = EPS, msg = "") {
  assertNear(v.x, x, eps, `${msg} [x]`)
  assertNear(v.y, y, eps, `${msg} [y]`)
  assertNear(v.z, z, eps, `${msg} [z]`)
}

// Same rotation up to double cover: |dot| ≈ 1.
function assertQuatSameRotation(a, b, eps = EPS, msg = "") {
  const d = Math.abs(Quat.dot(a, b))
  assert.ok(1 - d < eps, `${msg} quats differ: |dot| = ${d} (${JSON.stringify(a)} vs ${JSON.stringify(b)})`)
}

const ORDERS = ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"]

// Fixed non-degenerate angle triples (middle angle away from ±90° for every order).
const ANGLE_SAMPLES = [
  [0.3, -0.4, 0.5],
  [-1.1, 0.2, 0.9],
  [0.05, 1.0, -1.2],
  [-0.7, -0.6, -0.5],
  [0, 0, 0],
]

const X_AXIS = new Vec3(1, 0, 0)
const Y_AXIS = new Vec3(0, 1, 0)
const Z_AXIS = new Vec3(0, 0, 1)

test("fromEulerOrder matches explicit axis-angle composition for all six orders", () => {
  const axisFor = { X: X_AXIS, Y: Y_AXIS, Z: Z_AXIS }
  for (const order of ORDERS) {
    for (const [x, y, z] of ANGLE_SAMPLES) {
      const angleFor = { X: x, Y: y, Z: z }
      let expected = Quat.identity()
      for (const letter of order) {
        expected = expected.multiply(Quat.fromAxisAngle(axisFor[letter], angleFor[letter]))
      }
      const actual = Quat.fromEulerOrder(x, y, z, order)
      assertQuatSameRotation(actual, expected, EPS, `${order} ${[x, y, z]}`)
    }
  }
})

test("fromEuler is fromEulerOrder YXZ (MMD/PMX convention)", () => {
  for (const [x, y, z] of ANGLE_SAMPLES) {
    assertQuatSameRotation(Quat.fromEuler(x, y, z), Quat.fromEulerOrder(x, y, z, "YXZ"))
  }
})

test("toEulerOrder round-trips fromEulerOrder for all six orders", () => {
  for (const order of ORDERS) {
    for (const [x, y, z] of ANGLE_SAMPLES) {
      const q = Quat.fromEulerOrder(x, y, z, order)
      const e = Quat.toEulerOrder(q, order)
      const back = Quat.fromEulerOrder(e.x, e.y, e.z, order)
      assertQuatSameRotation(q, back, 1e-5, `${order} ${[x, y, z]}`)
    }
  }
})

test("toEulerOrder recovers the exact angles away from gimbal lock", () => {
  for (const order of ORDERS) {
    const q = Quat.fromEulerOrder(0.3, -0.4, 0.5, order)
    const e = Quat.toEulerOrder(q, order)
    assertVecNear(e, 0.3, -0.4, 0.5, 1e-6, order)
  }
})

test("toEulerOrder survives gimbal lock without NaN", () => {
  const half = Math.PI / 2
  for (const order of ORDERS) {
    const q = Quat.fromEulerOrder(
      order[1] === "X" ? half : 0.3,
      order[1] === "Y" ? half : 0.3,
      order[1] === "Z" ? half : 0.3,
      order
    )
    const e = Quat.toEulerOrder(q, order)
    assert.ok(Number.isFinite(e.x) && Number.isFinite(e.y) && Number.isFinite(e.z), order)
    const back = Quat.fromEulerOrder(e.x, e.y, e.z, order)
    assertQuatSameRotation(q, back, 1e-5, `${order} at gimbal lock`)
  }
})

test("rotateVec agrees with the rotation matrix, rotateVecInv inverts it", () => {
  const q = Quat.fromEulerOrder(0.4, -1.0, 0.7, "YXZ")
  const m = Mat4.fromQuat(q.x, q.y, q.z, q.w)
  const samples = [new Vec3(1, 0, 0), new Vec3(0, 1, 0), new Vec3(0, 0, 1), new Vec3(0.3, -2, 1.5)]
  for (const v of samples) {
    const byQuat = Quat.rotateVec(q, v)
    const byMat = Vec3.transformMat4RotationInto(v, m.values, new Vec3(0, 0, 0))
    assertVecNear(byQuat, byMat.x, byMat.y, byMat.z, 1e-6, "quat vs mat")
    const roundTrip = Quat.rotateVecInv(q, byQuat)
    assertVecNear(roundTrip, v.x, v.y, v.z, 1e-6, "inv round-trip")
  }
})

test("rotateVecInto is aliasing-safe (out === v)", () => {
  const q = Quat.fromEulerOrder(0.2, 0.3, -0.4, "ZYX")
  const v = new Vec3(1, 2, 3)
  const expected = Quat.rotateVec(q, v)
  Quat.rotateVecInto(q, v, v)
  assertVecNear(v, expected.x, expected.y, expected.z)
})

test("fromUnitVectors takes from onto to", () => {
  const pairs = [
    [new Vec3(1, 0, 0), new Vec3(0, 1, 0)],
    [new Vec3(0, 0, 1), new Vec3(0, 1, 0)],
    [new Vec3(1, 0, 0), new Vec3(1, 0, 0)], // parallel → identity
  ]
  const n = (v) => v.normalize()
  pairs.push([n(new Vec3(1, 2, -0.5)), n(new Vec3(-0.3, 0.4, 2))])
  for (const [from, to] of pairs) {
    const q = Quat.fromUnitVectors(from, to)
    const rotated = Quat.rotateVec(q, from)
    assertVecNear(rotated, to.x, to.y, to.z, 1e-6, "fromUnitVectors")
  }
})

test("fromUnitVectors antiparallel branch produces a 180° rotation, no NaN", () => {
  const cases = [
    [new Vec3(1, 0, 0), new Vec3(-1, 0, 0)],
    [new Vec3(0, 1, 0), new Vec3(0, -1, 0)],
    [new Vec3(0, 0, 1), new Vec3(0, 0, -1)],
  ]
  for (const [from, to] of cases) {
    const q = Quat.fromUnitVectors(from, to)
    assert.ok([q.x, q.y, q.z, q.w].every(Number.isFinite))
    const rotated = Quat.rotateVec(q, from)
    assertVecNear(rotated, to.x, to.y, to.z, 1e-6, "antiparallel")
  }
})

test("fromBasis maps the standard basis onto the given axes", () => {
  assertQuatSameRotation(Quat.fromBasis(X_AXIS, Y_AXIS, Z_AXIS), Quat.identity())
  const q = Quat.fromEulerOrder(-0.6, 0.8, 1.4, "XYZ")
  const bx = Quat.rotateVec(q, X_AXIS)
  const by = Quat.rotateVec(q, Y_AXIS)
  const bz = Quat.rotateVec(q, Z_AXIS)
  const rebuilt = Quat.fromBasis(bx, by, bz)
  assertQuatSameRotation(rebuilt, q, 1e-6, "fromBasis")
})

test("twistAroundAxis decomposes q into swing · twist", () => {
  const q = Quat.fromEulerOrder(0.7, -0.9, 0.3, "YXZ").normalize()
  for (const axis of [X_AXIS, Y_AXIS, Z_AXIS]) {
    const twist = Quat.twistAroundAxis(q, axis)
    // twist is a pure rotation about the axis: it keeps the axis fixed
    const kept = Quat.rotateVec(twist, axis)
    assertVecNear(kept, axis.x, axis.y, axis.z, 1e-6, "twist keeps axis")
    // swing = q · twist⁻¹ recomposes to q
    const swing = q.multiply(Quat.conjugateInto(twist, Quat.identity()))
    const recomposed = swing.multiply(twist)
    assertQuatSameRotation(recomposed, q, 1e-6, "swing·twist")
    // the twist of the swing about the same axis is identity
    const residual = Quat.twistAroundAxis(swing, axis)
    assertQuatSameRotation(residual, Quat.identity(), 1e-6, "swing has no twist")
  }
})

test("twistAroundAxis singularity returns identity", () => {
  // 180° about X is entirely perpendicular to Y: no well-defined twist about Y
  const q = Quat.fromAxisAngle(X_AXIS, Math.PI)
  const twist = Quat.twistAroundAxis(q, Y_AXIS)
  assertQuatSameRotation(twist, Quat.identity())
})

test("lookRotation maps +Z to forward and keeps up on the +Y side", () => {
  const cases = [
    [new Vec3(0, 0, 1), new Vec3(0, 1, 0)],
    [new Vec3(1, 0, 0), new Vec3(0, 1, 0)],
    [new Vec3(0.5, 0.2, -0.8), new Vec3(0, 1, 0)],
    [new Vec3(0, 0.99, 0.1), new Vec3(0, 1, 0)],
  ]
  for (const [forward, up] of cases) {
    const q = Quat.lookRotation(forward, up)
    const f = forward.clone ? forward.clone() : new Vec3(forward.x, forward.y, forward.z)
    f.normalize()
    const z = Quat.rotateVec(q, Z_AXIS)
    assertVecNear(z, f.x, f.y, f.z, 1e-6, "forward")
    const y = Quat.rotateVec(q, Y_AXIS)
    assert.ok(y.dot(up) > 0, "up hemisphere")
  }
})

test("lookRotation handles forward parallel to up without NaN", () => {
  for (const forward of [new Vec3(0, 1, 0), new Vec3(0, -1, 0)]) {
    const q = Quat.lookRotation(forward, new Vec3(0, 1, 0))
    assert.ok([q.x, q.y, q.z, q.w].every(Number.isFinite))
    const z = Quat.rotateVec(q, Z_AXIS)
    assertVecNear(z, forward.x, forward.y, forward.z, 1e-6, "degenerate forward")
  }
})

test("nlerp hits endpoints and matches slerp for small angles", () => {
  const a = Quat.fromEulerOrder(0.1, 0.2, 0.3, "YXZ")
  const b = Quat.fromEulerOrder(0.15, 0.22, 0.28, "YXZ")
  assertQuatSameRotation(Quat.nlerp(a, b, 0), a)
  assertQuatSameRotation(Quat.nlerp(a, b, 1), b)
  for (const t of [0.25, 0.5, 0.75]) {
    const n = Quat.nlerp(a, b, t)
    const s = Quat.slerp(a, b, t)
    assertQuatSameRotation(n, s, 1e-6, `t=${t}`)
  }
})

test("nlerp takes the short path across hemispheres", () => {
  const a = Quat.fromAxisAngle(Y_AXIS, 0.2)
  const bNeg = Quat.fromAxisAngle(Y_AXIS, 0.4)
  bNeg.x = -bNeg.x; bNeg.y = -bNeg.y; bNeg.z = -bNeg.z; bNeg.w = -bNeg.w
  const mid = Quat.nlerp(a, bNeg, 0.5)
  assertQuatSameRotation(mid, Quat.fromAxisAngle(Y_AXIS, 0.3), 1e-6, "hemisphere flip")
})

test("slerp interpolates angle linearly (pre-existing code, first coverage)", () => {
  const a = Quat.identity()
  const b = Quat.fromAxisAngle(Y_AXIS, Math.PI / 2)
  const mid = Quat.slerp(a, b, 0.5)
  assertQuatSameRotation(mid, Quat.fromAxisAngle(Y_AXIS, Math.PI / 4), 1e-6, "slerp midpoint")
  const out = Quat.identity()
  Quat.slerpInto(a, b, 0.5, out)
  assertQuatSameRotation(out, mid, 1e-6, "slerpInto agrees")
})

test("angleTo measures rotation difference, insensitive to double cover", () => {
  const q = Quat.fromEulerOrder(0.3, 0.5, -0.2, "YXZ")
  const step = Quat.fromAxisAngle(Y_AXIS, 0.5)
  assertNear(Quat.angleTo(q, q.multiply(step)), 0.5, 1e-6, "angleTo")
  const neg = new Quat(-q.x, -q.y, -q.z, -q.w)
  assertNear(Quat.angleTo(q, neg), 0, 1e-6, "double cover")
})

test("mirrorZ converts handedness consistently for vectors and rotations", () => {
  const q = Quat.fromEulerOrder(0.4, -0.3, 0.8, "ZYX")
  const v = new Vec3(0.5, -1.2, 2)
  // involutive
  assertQuatSameRotation(Quat.mirrorZ(Quat.mirrorZ(q)), q)
  const mv = Vec3.mirrorZ(Vec3.mirrorZ(v))
  assertVecNear(mv, v.x, v.y, v.z)
  // mirror(q) acting on mirror(v) equals mirror of q acting on v
  const lhs = Quat.rotateVec(Quat.mirrorZ(q), Vec3.mirrorZ(v))
  const rhs = Vec3.mirrorZ(Quat.rotateVec(q, v))
  assertVecNear(lhs, rhs.x, rhs.y, rhs.z, 1e-6, "conjugation identity")
})

test("Into variants return out and write the same values as the allocating forms", () => {
  const a = Quat.fromEulerOrder(0.1, 0.4, -0.6, "YXZ")
  const b = Quat.fromEulerOrder(-0.2, 0.1, 0.3, "YXZ")
  const outQ = Quat.identity()
  assert.equal(Quat.nlerpInto(a, b, 0.3, outQ), outQ)
  assertQuatSameRotation(outQ, Quat.nlerp(a, b, 0.3))
  const outV = Vec3.zeros()
  assert.equal(Quat.toEulerOrderInto(a, "YXZ", outV), outV)
  const e = Quat.toEulerOrder(a, "YXZ")
  assertVecNear(outV, e.x, e.y, e.z)
  const outQ2 = Quat.identity()
  assert.equal(Quat.fromEulerOrderInto(0.1, 0.4, -0.6, "YXZ", outQ2), outQ2)
  assertQuatSameRotation(outQ2, a)
})

test("bezierInterpolate: endpoints, linearity, monotonic S-curve", () => {
  // endpoints clamp to the curve's y range
  assertNear(bezierInterpolate(0.3, 0.7, 0.1, 0.9, 0), 0, 1e-3, "t=0")
  assertNear(bezierInterpolate(0.3, 0.7, 0.1, 0.9, 1), 1, 1e-3, "t=1")
  // control points on the diagonal → identity curve
  for (const t of [0.2, 0.5, 0.8]) {
    assertNear(bezierInterpolate(0.25, 0.75, 0.25, 0.75, t), t, 1e-3, "linear")
  }
  // ease-in-out control points → monotonically increasing
  let prev = -1
  for (let t = 0; t <= 1.001; t += 0.1) {
    const y = bezierInterpolate(0.7, 0.3, 0.1, 0.9, t)
    assert.ok(y >= prev - 1e-4, `monotonic at t=${t}`)
    prev = y
  }
})

test("interpolateControlPoints evaluates 127-space VMD control points", () => {
  // VMD linear default: (20, 20) / (107, 107) — identity curve
  const linear = [{ x: 20, y: 20 }, { x: 107, y: 107 }]
  for (const t of [0, 0.3, 0.5, 0.9, 1]) {
    assertNear(interpolateControlPoints(linear, t), t, 1e-3, "linear cps")
  }
})

test("easeInOut endpoints and midpoint", () => {
  assertNear(easeInOut(0), 0)
  assertNear(easeInOut(0.5), 0.5)
  assertNear(easeInOut(1), 1)
})
