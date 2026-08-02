// LocomotionController tests against a stub model — the controller only touches
// getClip() and setBlendPose(), so the whole state machine runs headless.

import { test } from "node:test"
import assert from "node:assert/strict"
import { LocomotionController } from "../dist/locomotion.js"

function makeStub() {
  const clips = {
    idle: { frameCount: 90 }, // 3s
    run: { frameCount: 30 }, // 1s
    sprint: { frameCount: 24 }, // 0.8s
  }
  return {
    lastEntries: null,
    getClip(name) {
      return clips[name] ?? null
    },
    setBlendPose(entries) {
      this.lastEntries = entries
    },
    clearBlendPose() {
      this.lastEntries = null
    },
  }
}

function makeController(stub, options) {
  return new LocomotionController(stub, { idle: "idle", run: "run", sprint: "sprint" }, options)
}

function weightsByName(stub) {
  const out = {}
  for (const e of stub.lastEntries) out[e.name] = (out[e.name] ?? 0) + e.weight
  return out
}

const STEPS = (controller, n, dt = 1 / 60) => {
  let pose
  for (let i = 0; i < n; i++) pose = controller.update(dt)
  return pose
}

test("at rest the pose is pure idle and the character does not move", () => {
  const stub = makeStub()
  const c = makeController(stub)
  const pose = STEPS(c, 60)
  const w = weightsByName(stub)
  assert.equal(w.idle, 1)
  assert.equal(w.run, 0)
  assert.equal(pose.position.x, 0)
  assert.equal(pose.position.z, 0)
})

test("weights always sum to 1 through ramps", () => {
  const stub = makeStub()
  const c = makeController(stub)
  c.setMove(0, 1)
  for (let i = 0; i < 120; i++) {
    c.update(1 / 60)
    const w = weightsByName(stub)
    const sum = w.idle + w.run + w.sprint
    assert.ok(Math.abs(sum - 1) < 1e-9, `sum=${sum} at step ${i}`)
  }
})

test("forward input reaches full run and moves along +Z at runSpeed", () => {
  const stub = makeStub()
  const c = makeController(stub, { runSpeed: 6 })
  c.setMove(0, 1)
  STEPS(c, 120) // 2s: level fully ramped
  const before = { x: c.getPosition().x, z: c.getPosition().z }
  const pose = STEPS(c, 60) // 1 more second at full speed
  const w = weightsByName(stub)
  assert.ok(w.run > 0.999, `run weight ${w.run}`)
  assert.ok(Math.abs(pose.position.z - before.z - 6) < 0.05, `dz=${pose.position.z - before.z}`)
  assert.ok(Math.abs(pose.position.x - before.x) < 1e-6)
})

test("sprint ramps past run into the sprint slot", () => {
  const stub = makeStub()
  const c = makeController(stub)
  c.setMove(0, 1, true)
  STEPS(c, 240)
  const w = weightsByName(stub)
  assert.ok(w.sprint > 0.999, `sprint weight ${w.sprint}`)
  assert.equal(w.idle, 0)
})

test("yaw eases toward the input heading and wraps correctly", () => {
  const stub = makeStub()
  const c = makeController(stub)
  c.setMove(-1, 0) // heading -PI/2 (toward -X)
  const pose = STEPS(c, 300)
  assert.ok(Math.abs(pose.yaw - -Math.PI / 2) < 1e-3, `yaw=${pose.yaw}`)
  // rotation quat = rotY(yaw + PI) by default
  const expected = Math.sin((pose.yaw + Math.PI) / 2)
  assert.ok(Math.abs(pose.rotation.y - expected) < 1e-6)
})

test("run and sprint share the gait phase (same normalized time)", () => {
  const stub = makeStub()
  const c = makeController(stub)
  c.setMove(0, 1, true)
  STEPS(c, 100)
  const run = stub.lastEntries.find((e) => e.name === "run")
  const sprint = stub.lastEntries.find((e) => e.name === "sprint")
  const runPhase = run.time / 1.0 // runDur = 1s
  const sprintPhase = sprint.time / 0.8 // sprintDur = 0.8s
  assert.ok(Math.abs(runPhase - sprintPhase) < 1e-9)
})

test("releasing input blends back to idle and stops", () => {
  const stub = makeStub()
  const c = makeController(stub)
  c.setMove(0, 1)
  STEPS(c, 120)
  c.setMove(0, 0)
  STEPS(c, 120)
  const w = weightsByName(stub)
  assert.ok(w.idle > 0.999)
  const z1 = c.getPosition().z
  STEPS(c, 30)
  assert.equal(c.getPosition().z, z1)
})

test("clip times stay within each clip's duration", () => {
  const stub = makeStub()
  const c = makeController(stub)
  c.setMove(1, 0.3, true)
  for (let i = 0; i < 600; i++) {
    c.update(1 / 60)
    for (const e of stub.lastEntries) {
      const dur = stub.getClip(e.name).frameCount / 30
      assert.ok(e.time >= 0 && e.time < dur + 1e-9, `${e.name} time=${e.time}`)
    }
  }
})

test("L-R-L direction reversals never drift fore/aft", () => {
  const stub = makeStub()
  const c = makeController(stub)
  // run forward, then alternate strict left/right several times
  c.setMove(0, 1)
  STEPS(c, 120)
  const zRef = c.getPosition().z
  for (let i = 0; i < 6; i++) {
    c.setMove(i % 2 === 0 ? -1 : 1, 0)
    STEPS(c, 45)
  }
  // movement was strictly along ±X the whole time: z must not have moved at all
  assert.equal(c.getPosition().z, zRef)
})

test("pivot gate: no translation while the body is far from the input heading", () => {
  const stub = makeStub()
  const c = makeController(stub)
  c.setMove(0, 1)
  STEPS(c, 120)
  const x1 = c.getPosition().x
  const z1 = c.getPosition().z
  c.setMove(0, -1) // 180° reversal → pivot in place first
  STEPS(c, 5) // yaw still > 45° off after 5 frames at turnResponse 10
  assert.equal(c.getPosition().x, x1)
  assert.equal(c.getPosition().z, z1)
})

test("tank mode: steer rotates at steerRate, standing or moving", () => {
  const stub = makeStub()
  const c = makeController(stub, { steerRate: 2 })
  c.setDrive(0, 1)
  const pose = STEPS(c, 60) // 1s standing steer
  assert.ok(Math.abs(pose.yaw - 2) < 0.01, `yaw=${pose.yaw}`)
  assert.equal(pose.position.x, 0)
  assert.equal(pose.position.z, 0)
})

test("tank mode: W runs along the facing, curving while steering", () => {
  const stub = makeStub()
  const c = makeController(stub, { runSpeed: 6, steerRate: 2 })
  c.setDrive(1, 0)
  STEPS(c, 120)
  const z1 = c.getPosition().z
  const pose = STEPS(c, 60)
  assert.ok(Math.abs(pose.position.z - z1 - 6) < 0.05, "straight run at runSpeed")
  // now steer while running: path curves, yaw changes
  c.setDrive(1, 1)
  const before = pose.yaw
  const p2 = STEPS(c, 30)
  assert.ok(p2.yaw > before + 0.5, "yaw increased while running")
})

test("tank mode: backpedal moves backward at backpedalScale, never sprints", () => {
  const stub = makeStub()
  const c = makeController(stub, { runSpeed: 6, backpedalScale: 0.5 })
  c.setDrive(-1, 0, true) // sprint requested but backpedaling
  STEPS(c, 120)
  const z1 = c.getPosition().z
  const pose = STEPS(c, 60)
  assert.ok(Math.abs(z1 - pose.position.z - 3) < 0.05, `backpedal dz=${z1 - pose.position.z}`)
  const w = weightsByName(stub)
  assert.equal(w.sprint, 0)
})

test("huge dt is clamped (tab-switch guard)", () => {
  const stub = makeStub()
  const c = makeController(stub, { runSpeed: 6 })
  c.setMove(0, 1)
  STEPS(c, 240)
  const z1 = c.getPosition().z
  c.update(5) // 5s frozen tab → treated as 0.1s
  assert.ok(c.getPosition().z - z1 < 6 * 0.11, `dz=${c.getPosition().z - z1}`)
})
