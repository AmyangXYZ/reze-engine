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

const DEG = Math.PI / 180
const RING = (prefix) => [
  { clip: `${prefix}F`, angle: 0, speed: 60 },
  { clip: `${prefix}R45`, angle: 45 * DEG, speed: 60 },
  { clip: `${prefix}R`, angle: 90 * DEG, speed: 50 },
  { clip: `${prefix}R135`, angle: 135 * DEG, speed: 60 },
  { clip: `${prefix}B`, angle: 180 * DEG, speed: 60 },
  { clip: `${prefix}L135`, angle: -135 * DEG, speed: 60 },
  { clip: `${prefix}L`, angle: -90 * DEG, speed: 50 },
  { clip: `${prefix}L45`, angle: -45 * DEG, speed: 60 },
]

function makeStrafeStub() {
  const clips = { idle: { frameCount: 90 } }
  for (const e of [...RING("run_"), ...RING("sprint_")]) clips[e.clip] = { frameCount: e.clip.startsWith("run_") ? 30 : 24 }
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

function makeStrafeController(stub) {
  return new LocomotionController(stub, {
    idle: "idle",
    run: "run_F",
    strafeRun: RING("run_"),
    strafeSprint: RING("sprint_"),
  })
}

test("strafe mode: body holds the facing while moving sideways on the right clip", () => {
  const stub = makeStrafeStub()
  const c = makeStrafeController(stub)
  c.setFacing(0) // face +Z
  c.setMove(1, 0) // move +X = the character's right
  const pose = STEPS(c, 240)
  assert.ok(Math.abs(pose.yaw) < 1e-6, `yaw held, got ${pose.yaw}`)
  const w = weightsByName(stub)
  assert.ok(w["run_R"] > 0.999, `pure-right clip drives: ${JSON.stringify(w)}`)
  assert.ok(c.getPosition().x > 0)
  assert.equal(Math.round(c.getPosition().z * 1e6), 0)
})

test("strafe mode: a diagonal blends the adjacent ring pair, weights sum to 1", () => {
  const stub = makeStrafeStub()
  const c = makeStrafeController(stub)
  c.setFacing(0)
  c.setMove(Math.sin(22.5 * DEG), Math.cos(22.5 * DEG)) // halfway F..R45
  STEPS(c, 240)
  const w = weightsByName(stub)
  assert.ok(Math.abs(w["run_F"] - 0.5) < 0.01 && Math.abs(w["run_R45"] - 0.5) < 0.01, JSON.stringify(w))
  let sum = 0
  for (const e of stub.lastEntries) sum += e.weight
  assert.ok(Math.abs(sum - 1) < 1e-9)
})

test("strafe mode: sprint engages the sprint ring; root speed uses authored clip speed", () => {
  const stub = makeStrafeStub()
  const c = makeStrafeController(stub)
  c.setFacing(0)
  c.setMove(-1, 0, true) // sprint left
  STEPS(c, 240)
  const w = weightsByName(stub)
  assert.ok(w["sprint_L"] > 0.999, JSON.stringify(w))
  const x1 = c.getPosition().x
  STEPS(c, 60) // 1s at authored 50 u/s for the pure-side clip
  assert.ok(Math.abs(x1 - c.getPosition().x - 50) < 0.5, `dx=${x1 - c.getPosition().x}`)
})

test("strafe mode: releasing input settles into idle without drift", () => {
  const stub = makeStrafeStub()
  const c = makeStrafeController(stub)
  c.setFacing(0)
  c.setMove(0, -1) // backpedal
  STEPS(c, 120)
  c.setMove(0, 0)
  STEPS(c, 120)
  const w = weightsByName(stub)
  assert.ok(w.idle > 0.999)
  const z = c.getPosition().z
  STEPS(c, 30)
  assert.equal(c.getPosition().z, z)
})

const TURN_CLIPS = [
  { clip: "turn_L90", angle: (-90 * Math.PI) / 180, exitTime: 1.2 },
  { clip: "turn_R90", angle: (90 * Math.PI) / 180, exitTime: 1.2 },
  { clip: "turn_L180", angle: -Math.PI, exitTime: 1.6 },
]

function makeTurnStub() {
  const clips = { idle: { frameCount: 90 }, run: { frameCount: 30 } }
  for (const t of TURN_CLIPS) clips[t.clip] = { frameCount: 68 }
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

test("turn clips: a reversal from standstill plays the nearest clip and transfers its yaw", () => {
  const stub = makeTurnStub()
  const c = new LocomotionController(stub, { idle: "idle", run: "run", turnInPlace: TURN_CLIPS }, { turnTimeScale: 1 })
  c.setMove(0, 1)
  STEPS(c, 120) // face +Z, then release and settle
  c.setMove(0, 0)
  STEPS(c, 120)
  const p0 = { x: c.getPosition().x, z: c.getPosition().z }
  c.setMove(0, -1) // 180° reversal from idle
  c.update(1 / 60)
  const w = weightsByName(stub)
  assert.ok(w["turn_L180"] !== undefined, `turn clip active: ${JSON.stringify(w)}`)
  // during the turn: no translation, yaw frozen
  const yawBefore = STEPS(c, 30).yaw
  assert.ok(Math.abs(yawBefore) < 1e-6, `root yaw frozen during turn, got ${yawBefore}`)
  assert.equal(c.getPosition().x, p0.x)
  assert.equal(c.getPosition().z, p0.z)
  // run past exitTime: angle transfers, then she runs off toward -Z
  const pose = STEPS(c, 120)
  assert.ok(Math.abs(Math.abs(pose.yaw) - Math.PI) < 0.2, `yaw transferred, got ${pose.yaw}`)
  assert.ok(c.getPosition().z < p0.z - 1, "moving -Z after the turn")
})

test("turn clips: small corrections keep the instant pivot (no clip)", () => {
  const stub = makeTurnStub()
  const c = new LocomotionController(stub, { idle: "idle", run: "run", turnInPlace: TURN_CLIPS })
  c.setMove(0, 1)
  STEPS(c, 120)
  c.setMove(1, 1) // 45° — under the ~100° threshold
  STEPS(c, 10)
  const w = weightsByName(stub)
  assert.ok(!("turn_L90" in w) && !("turn_R90" in w) && !("turn_L180" in w), JSON.stringify(w))
})

test("turn clips: weights sum to 1 throughout the turn", () => {
  const stub = makeTurnStub()
  const c = new LocomotionController(stub, { idle: "idle", run: "run", turnInPlace: TURN_CLIPS }, { turnTimeScale: 1 })
  c.setMove(0, -1)
  for (let i = 0; i < 90; i++) {
    c.update(1 / 60)
    let sum = 0
    for (const e of stub.lastEntries) sum += e.weight
    assert.ok(Math.abs(sum - 1) < 1e-9, `sum=${sum} at step ${i}`)
  }
})

test("run-turn: a moving reversal plays the plant-and-turn along its authored profile", () => {
  const clips = {
    idle: { frameCount: 90 },
    run: { frameCount: 30 },
    rt: { frameCount: 41 },
  }
  const stub = {
    lastEntries: null,
    getClip(n) { return clips[n] ?? null },
    setBlendPose(e) { this.lastEntries = e },
    clearBlendPose() { this.lastEntries = null },
  }
  const c = new LocomotionController(stub, {
    idle: "idle",
    run: "run",
    runTurn: [
      { clip: "rt", angle: Math.PI, exitTime: 1.2, forward: [0, 8, 14, 16, 12, 6, -2], gear: "run", foot: "L" },
      { clip: "rt", angle: -Math.PI, exitTime: 1.2, forward: [0, 8, 14, 16, 12, 6, -2], gear: "run", foot: "L" },
    ],
  })
  c.setMove(0, 1)
  STEPS(c, 120) // full run toward +Z
  const z0 = c.getPosition().z
  c.setMove(0, -1) // reversal while running
  c.update(1 / 60)
  const w = weightsByName(stub)
  assert.ok(w["rt"] !== undefined, `run-turn active: ${JSON.stringify(w)}`)
  // mid-turn: root has overrun forward along the OLD heading (profile positive)
  STEPS(c, 30) // 0.5s in
  assert.ok(c.getPosition().z > z0 + 5, `overran forward, dz=${c.getPosition().z - z0}`)
  // after exit: yaw reversed, profile returned, and she runs out toward -Z
  const pose = STEPS(c, 120)
  assert.ok(Math.abs(Math.abs(pose.yaw) - Math.PI) < 0.2, `yaw=${pose.yaw}`)
  assert.ok(c.getPosition().z < z0, `ran back past the plant, dz=${c.getPosition().z - z0}`)
})

test("stop clips: releasing at speed plays the skid along its profile, then idle", () => {
  const clips = { idle: { frameCount: 90 }, run: { frameCount: 30 }, st: { frameCount: 45 } }
  const stub = {
    lastEntries: null,
    getClip(n) { return clips[n] ?? null },
    setBlendPose(e) { this.lastEntries = e },
    clearBlendPose() {},
  }
  const c = new LocomotionController(stub, {
    idle: "idle",
    run: "run",
    stop: [{ clip: "st", exitTime: 1.5, forward: [0, 10, 20, 28, 33, 35, 35], gear: "run", foot: "L" }],
  })
  c.setMove(0, 1)
  STEPS(c, 120)
  const z0 = c.getPosition().z
  c.setMove(0, 0)
  c.update(1 / 60)
  const w = weightsByName(stub)
  assert.ok(w["st"] !== undefined, `stop clip active: ${JSON.stringify(w)}`)
  const pose = STEPS(c, 120) // 2s > exitTime
  assert.ok(Math.abs(pose.position.z - z0 - 35) < 0.5, `skid distance dz=${pose.position.z - z0}`)
  const w2 = weightsByName(stub)
  assert.ok(w2.idle > 0.999, `settled to idle: ${JSON.stringify(w2)}`)
  assert.equal(pose.speedLevel, 0)
})

test("stop clips: re-pressing input interrupts the stop and resumes", () => {
  const clips = { idle: { frameCount: 90 }, run: { frameCount: 30 }, st: { frameCount: 45 } }
  const stub = {
    lastEntries: null,
    getClip(n) { return clips[n] ?? null },
    setBlendPose(e) { this.lastEntries = e },
    clearBlendPose() {},
  }
  const c = new LocomotionController(stub, {
    idle: "idle",
    run: "run",
    stop: [{ clip: "st", exitTime: 1.5, forward: [0, 10, 20, 28, 33, 35, 35], gear: "run", foot: "L" }],
  })
  c.setMove(0, 1)
  STEPS(c, 120)
  c.setMove(0, 0)
  STEPS(c, 20) // ~0.33s into the stop
  c.setMove(0, 1) // resume!
  const p1 = STEPS(c, 1)
  // The stop pose lingers only as a fading ghost over live locomotion,
  // and the breakout restarts from a low level (fresh walk-up, no drift)
  const w = weightsByName(stub)
  assert.ok(w["st"] > 0 && w["st"] < 1, `ghost fading: ${JSON.stringify(w)}`)
  assert.ok(p1.speedLevel <= 0.4, `breakout restarts low: ${p1.speedLevel}`)
  STEPS(c, 24) // past the 0.25s fade
  const w2 = weightsByName(stub)
  assert.ok(!("st" in w2) || w2["st"] === 0, `ghost gone: ${JSON.stringify(w2)}`)
  const z1 = c.getPosition().z
  STEPS(c, 60)
  assert.ok(c.getPosition().z > z1 + 1, "running again")
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
