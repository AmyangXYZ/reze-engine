// Gravity and wind: both are accelerations on dynamic bodies, and both are
// world-wide. The properties worth pinning are the ones a scene depends on —
// that still air changes nothing, that wind pushes the way it is pointed, and
// that gusting is driven by simulated time so an exported take matches its
// preview frame for frame.

import { test } from "node:test"
import assert from "node:assert/strict"

const { World } = await import("../dist/physics/world.js")
const { RigidBodyStore } = await import("../dist/physics/body.js")
const { SolverCache } = await import("../dist/physics/solver.js")
const { ContactPool } = await import("../dist/physics/contact.js")
const { Vec3 } = await import("../dist/math.js")
const { RigidbodyType, RigidbodyShape } = await import("../dist/physics/types.js")
const { Mat4 } = await import("../dist/math.js")

/** One free-falling sphere, far from anything it could collide with. */
const makeStore = () =>
  new RigidBodyStore([
    {
      name: "b",
      englishName: "b",
      boneIndex: -1,
      group: 0,
      collisionMask: 0,
      shape: RigidbodyShape.Sphere,
      size: new Vec3(0.5, 0, 0),
      shapePosition: new Vec3(0, 100, 0),
      shapeRotation: new Vec3(0, 0, 0),
      mass: 1,
      linearDamping: 0,
      angularDamping: 0,
      restitution: 0,
      friction: 0,
      type: RigidbodyType.Dynamic,
      bodyOffsetMatrixInverse: Mat4.identity(),
    },
  ])

/** Integrate `frames` fixed steps and return the body's velocity. */
function run(wind, frames = 60, gravity = new Vec3(0, -98, 0)) {
  const store = makeStore()
  const world = new World(gravity)
  world.setWind(wind)
  const cache = new SolverCache([])
  const contacts = new ContactPool()
  for (let f = 0; f < frames; f++) world.step(store, [], cache, contacts, 1 / 60)
  return { x: store.linearVelocities[0], y: store.linearVelocities[1], z: store.linearVelocities[2] }
}

test("still air leaves gravity alone", () => {
  const none = run(null)
  const off = run({ direction: new Vec3(1, 0, 0), strength: 0 })
  const zeroDir = run({ direction: new Vec3(0, 0, 0), strength: 50 })
  // One second of 98 downward.
  assert.ok(Math.abs(none.y + 98) < 0.5, `expected ~-98, got ${none.y}`)
  assert.equal(none.x, 0)
  assert.deepEqual(off, none, "zero strength must be identical to no wind")
  assert.deepEqual(zeroDir, none, "a zero direction vector must be identical to no wind")
})

test("wind pushes along its direction and scales with strength", () => {
  const weak = run({ direction: new Vec3(1, 0, 0), strength: 10 })
  const strong = run({ direction: new Vec3(1, 0, 0), strength: 20 })
  assert.ok(weak.x > 0, "wind along +x must move the body along +x")
  assert.ok(Math.abs(weak.x - 10) < 0.5, `one second at 10 should reach ~10, got ${weak.x}`)
  assert.ok(Math.abs(strong.x - 2 * weak.x) < 0.5, "twice the strength, twice the velocity")
  // Vertical motion is gravity's business; wind across should not disturb it.
  assert.ok(Math.abs(strong.y - weak.y) < 1e-3, "sideways wind must not change fall speed")
})

test("direction is normalised, so length is not a second strength dial", () => {
  const unit = run({ direction: new Vec3(1, 0, 0), strength: 10 })
  const long = run({ direction: new Vec3(7, 0, 0), strength: 10 })
  assert.ok(Math.abs(unit.x - long.x) < 1e-3, `${unit.x} vs ${long.x}`)
})

test("gusting varies the push without reversing it, and averages to the set strength", () => {
  const steady = run({ direction: new Vec3(1, 0, 0), strength: 10 }, 600)
  const gusty = run({ direction: new Vec3(1, 0, 0), strength: 10, turbulence: 1, frequency: 0.5 }, 600)
  assert.ok(gusty.x > 0, "a gust must never blow the body backwards")
  // Ten seconds is many gust periods; the mean should land near steady.
  assert.ok(
    Math.abs(gusty.x - steady.x) / steady.x < 0.25,
    `gusting should average near steady: ${gusty.x} vs ${steady.x}`,
  )
  assert.notEqual(gusty.x, steady.x, "turbulence must actually change the result")
})

test("gusts run on simulated time, so a take reproduces exactly", () => {
  const wind = { direction: new Vec3(1, 0, 0), strength: 10, turbulence: 1, frequency: 0.5 }
  const a = run(wind, 137)
  const b = run(wind, 137)
  assert.deepEqual(a, b, "same steps in, same velocity out — an export must match its preview")
})

test("getWind reports back what was set", () => {
  const world = new World(new Vec3(0, -98, 0))
  assert.equal(world.getWind(), null)
  world.setWind({ direction: new Vec3(0, 0, 3), strength: 12, turbulence: 0.4, frequency: 0.8 })
  const w = world.getWind()
  assert.ok(Math.abs(w.strength - 12) < 1e-4)
  assert.ok(Math.abs(w.direction.z - 1) < 1e-4, "direction comes back normalised")
  assert.ok(Math.abs(w.turbulence - 0.4) < 1e-6)
  assert.ok(Math.abs(w.frequency - 0.8) < 1e-6)
  world.setWind(null)
  assert.equal(world.getWind(), null)
})
