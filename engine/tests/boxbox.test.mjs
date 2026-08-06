// Box–box narrowphase: SAT + face clipping. MMD dress rigs are box panels, so
// this pair is the majority of collidable pairs on those models — it was
// silently unimplemented, which is what let skirt layers pass through each
// other. These pin the contract the solver relies on: normals run A→B, depth
// is positive when overlapping, and a face-on-face contact yields a real
// manifold rather than a single pivot point.

import { test } from "node:test"
import assert from "node:assert/strict"

const { RigidBodyStore } = await import("../dist/physics/body.js")
const { ContactPool, generateContacts } = await import("../dist/physics/contact.js")
const { Vec3, Mat4 } = await import("../dist/math.js")
const { RigidbodyShape, RigidbodyType } = await import("../dist/physics/types.js")

const shaped = (shape, sx, sy, sz) => ({
  name: "b", englishName: "b", boneIndex: -1, group: 0, collisionMask: 0xffff,
  shape, size: new Vec3(sx, sy, sz),
  shapePosition: new Vec3(0, 0, 0), shapeRotation: new Vec3(0, 0, 0),
  mass: 1, linearDamping: 0, angularDamping: 0, restitution: 0, friction: 0.5,
  type: RigidbodyType.Dynamic, aligned: false, bodyOffsetMatrixInverse: Mat4.identity(),
})

// Two unit boxes, placed and oriented by hand. The store is built from the
// descriptors, then poses are written directly — narrowphase reads only
// positions/orientations/size, so this needs no bones or stepping.
const box = (hx, hy, hz) => shaped(RigidbodyShape.Box, hx, hy, hz)
// PMX capsules are Y-aligned in body space: size.x is the radius, size.y the
// segment length between the cap centres.
const capsule = (r, len) => shaped(RigidbodyShape.Capsule, r, len, r)

function pair(posA, quatA, posB, quatB, hA = [1, 1, 1], hB = [1, 1, 1], mk = [box, box]) {
  const store = new RigidBodyStore([mk[0](...hA), mk[1](...hB)])
  store.positions.set(posA, 0)
  store.positions.set(posB, 3)
  store.orientations.set(quatA, 0)
  store.orientations.set(quatB, 4)
  const pool = new ContactPool()
  generateContacts(store, 0, 1, pool)
  return { store, pool, contacts: Array.from({ length: pool.count }, (_, i) => pool.get(i)) }
}

const I = [0, 0, 0, 1]
// Quaternion for a rotation of `ang` about a unit axis.
const q = (ax, ay, az, ang) => {
  const s = Math.sin(ang / 2)
  return [ax * s, ay * s, az * s, Math.cos(ang / 2)]
}

test("clearly separated boxes produce no contact", () => {
  const { pool } = pair([0, 0, 0], I, [5, 0, 0], I)
  assert.equal(pool.count, 0)
})

test("face overlap yields a four-point manifold", () => {
  // B's -X face is 0.5 inside A's +X face.
  const { contacts } = pair([0, 0, 0], I, [1.5, 0, 0], I)
  assert.equal(contacts.length, 4, "a face-on-face patch needs four points, not one")
  for (const c of contacts) {
    assert.ok(Math.abs(c.nx - 1) < 1e-5, `normal should run A→B along +X, got ${c.nx}`)
    assert.ok(Math.abs(c.depth - 0.5) < 1e-5, `depth should be the 0.5 overlap, got ${c.depth}`)
  }
})

test("normal always runs A→B, whichever way the pair is ordered", () => {
  const fwd = pair([0, 0, 0], I, [1.5, 0, 0], I)
  const rev = pair([1.5, 0, 0], I, [0, 0, 0], I)
  assert.ok(fwd.contacts[0].nx > 0.99)
  assert.ok(rev.contacts[0].nx < -0.99, "reversed pair must mirror the normal")
})

test("the shallowest axis wins, so a thin overlap picks that face", () => {
  // Overlaps 0.2 on X and 1.5 on Y — X is the way out.
  const { contacts } = pair([0, 0, 0], I, [1.8, 0.5, 0], I)
  assert.ok(contacts.length > 0)
  assert.ok(Math.abs(contacts[0].nx) > 0.99, "should separate along X, the shallow axis")
  assert.ok(Math.abs(contacts[0].depth - 0.2) < 1e-5)
})

test("a rotated box still contacts, with a finite normal and depth", () => {
  const { contacts } = pair([0, 0, 0], I, [1.6, 0, 0], q(0, 0, 1, Math.PI / 4))
  assert.ok(contacts.length > 0, "a 45° box overlapping a face must generate contact")
  for (const c of contacts) {
    assert.ok(Number.isFinite(c.depth) && Number.isFinite(c.nx + c.ny + c.nz))
    assert.ok(Math.abs(Math.hypot(c.nx, c.ny, c.nz) - 1) < 1e-4, "normal must stay unit length")
    assert.ok(c.depth > 0)
  }
})

test("edge-on-edge overlap produces a finite contact", () => {
  // Both boxes tipped so an edge of each faces the other. A rotated 45° about Z
  // spans 1.414 along X, B rotated about X still spans 1.0 — so they touch
  // below 2.414 apart, and 2.2 gives a real overlap to resolve.
  const { contacts } = pair([0, 0, 0], q(0, 0, 1, Math.PI / 4), [2.2, 0, 0], q(1, 0, 0, Math.PI / 4))
  assert.ok(contacts.length >= 1, "crossed edges that overlap must generate contact")
  for (const c of contacts) {
    assert.ok(Number.isFinite(c.depth) && c.depth > -0.05)
    assert.ok(Math.abs(Math.hypot(c.nx, c.ny, c.nz) - 1) < 1e-4)
  }
})

test("the manifold never exceeds four points", () => {
  for (const ang of [0, 0.1, 0.3, Math.PI / 6, Math.PI / 4]) {
    const { pool } = pair([0, 0, 0], I, [1.5, 0, 0], q(1, 0, 0, ang))
    assert.ok(pool.count <= 4, `${pool.count} points at ${ang} rad — solver rows are not free`)
  }
})

test("deep overlap stays finite instead of exploding", () => {
  const { contacts } = pair([0, 0, 0], I, [0.05, 0.05, 0.05], I)
  assert.ok(contacts.length > 0)
  for (const c of contacts) {
    assert.ok(Number.isFinite(c.depth) && Number.isFinite(c.nx))
    assert.ok(Math.abs(Math.hypot(c.nx, c.ny, c.nz) - 1) < 1e-4)
  }
})

test("thin panels overlapping face-on — the dress case — collide", () => {
  // Two flat plates, the shape MMD skirt rigs are built from.
  const { contacts } = pair([0, 0, 0], I, [0, 0, 0.15], I, [0.34, 0.25, 0.1], [0.34, 0.25, 0.1])
  assert.ok(contacts.length > 0, "skirt panel against skirt panel must generate contact")
  assert.ok(Math.abs(contacts[0].nz) > 0.99, "plates separate along their thin axis")
  assert.ok(Math.abs(contacts[0].depth - 0.05) < 1e-5)
})

test("contact points sit between the two bodies, not at their centres", () => {
  const { contacts, store } = pair([0, 0, 0], I, [1.5, 0, 0], I)
  for (const c of contacts) {
    // rA is CG→point for A; the point must be out near A's +X face.
    assert.ok(c.rAx > 0.4, `contact should sit on the touching face, rAx=${c.rAx}`)
    assert.ok(c.rBx < -0.4, `and on B's opposite face, rBx=${c.rBx}`)
  }
  assert.equal(store.count, 2)
})


// --- Capsule–box manifold --------------------------------------------------
// A capsule against a box face touches along a LINE. Reporting one point of it
// let a flat panel pivot about that point and rotate into the body — dress
// through leg. These pin the multi-point manifold and the swap path that
// serves box-capsule from the same detector.

// A wide flat plate with a capsule lying along its top face, overlapping 0.05.
const plateAndCapsule = (order) => {
  const lying = q(0, 0, 1, Math.PI / 2) // Y-aligned capsule laid along X
  return order === "capsule-first"
    ? pair([0, 0.45, 0], lying, [0, 0, 0], I, [0.3, 2], [2, 0.2, 2], [capsule, box])
    : pair([0, 0, 0], I, [0, 0.45, 0], lying, [2, 0.2, 2], [0.3, 2], [box, capsule])
}

test("capsule-box stays a single deepest contact, on purpose", () => {
  // A manifold along the touching segment was tried and reverted: it cut
  // penetration but raised idle jitter on dense rigs, because each extra
  // contact is another shove from the position-correction pass that the joint
  // springs return. Pinned so the trade is a decision, not a drift.
  const { contacts } = plateAndCapsule("capsule-first")
  assert.equal(contacts.length, 1)
  assert.ok(contacts[0].depth > 0)
  assert.ok(Math.abs(Math.hypot(contacts[0].nx, contacts[0].ny, contacts[0].nz) - 1) < 1e-4)
})

test("the swapped box-capsule path flips every contact it produced", () => {
  // flipNormalsFrom re-anchors everything the swapped call emitted, not just
  // the last contact. With capsule-box back to one point this passes either
  // way — it is kept because giving that detector a manifold again is exactly
  // the change that would silently reverse the extra normals, pulling the pair
  // together instead of apart.
  const { contacts } = plateAndCapsule("box-first")
  assert.ok(contacts.length >= 1)
  for (const c of contacts) {
    // A is the box, B the capsule above it: every normal must run +Y.
    assert.ok(c.ny > 0.99, `normal must run box→capsule (+Y), got ny=${c.ny}`)
  }
})

test("both orderings agree on the contact count", () => {
  assert.equal(plateAndCapsule("capsule-first").contacts.length,
    plateAndCapsule("box-first").contacts.length)
})
