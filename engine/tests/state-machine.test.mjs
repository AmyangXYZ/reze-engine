// AnimationStateMachine tests against a stub model — the machine only touches
// getClip() and setBlendPose(), so transitions and fades run headless.

import { test } from "node:test"
import assert from "node:assert/strict"
import { AnimationStateMachine } from "../dist/state-machine.js"

function makeStub() {
  const clips = {
    idle: { frameCount: 90 }, // ~2.97s span (89/30)
    walk: { frameCount: 30 },
    skill: { frameCount: 60 }, // ~1.97s
  }
  return {
    lastEntries: null,
    getClip(n) {
      return clips[n] ?? null
    },
    setBlendPose(e) {
      this.lastEntries = e
    },
    clearBlendPose() {},
  }
}

const STEPS = (m, n, dt = 1 / 60) => {
  for (let i = 0; i < n; i++) m.update(dt)
}

function weightsByName(stub) {
  const out = {}
  for (const e of stub.lastEntries ?? []) {
    if (e.weight > 1e-6) out[e.name] = (out[e.name] ?? 0) + e.weight
  }
  return out
}

test("initial clip state plays and loops", () => {
  const stub = makeStub()
  const m = new AnimationStateMachine(stub, { idle: { clip: "idle" } }, [], { initial: "idle" })
  STEPS(m, 30)
  const w = weightsByName(stub)
  assert.ok(w.idle > 0.999, JSON.stringify(w))
  const t1 = stub.lastEntries.find((e) => e.name === "idle").time
  STEPS(m, 240) // past the ~2.97s loop point
  const t2 = stub.lastEntries.find((e) => e.name === "idle").time
  assert.ok(t2 < 3, `looped: ${t2}`)
  assert.ok(t1 > 0.4, `advancing: ${t1}`)
})

test("when-transition crossfades with weights summing to 1", () => {
  const stub = makeStub()
  let go = false
  const m = new AnimationStateMachine(
    stub,
    { idle: { clip: "idle" }, walk: { clip: "walk" } },
    [{ from: "idle", to: "walk", when: () => go, fade: 0.3 }],
    { initial: "idle" }
  )
  STEPS(m, 10)
  go = true
  STEPS(m, 9) // 0.15s: mid-fade
  const w = weightsByName(stub)
  assert.equal(m.state, "walk")
  assert.ok(w.idle > 0.05 && w.walk > 0.05, `both live: ${JSON.stringify(w)}`)
  assert.ok(Math.abs(w.idle + w.walk - 1) < 1e-6, `sums to 1: ${JSON.stringify(w)}`)
  STEPS(m, 12) // past the fade
  const w2 = weightsByName(stub)
  assert.ok(!w2.idle && w2.walk > 0.999, `settled: ${JSON.stringify(w2)}`)
})

test("exitTime holds a transition until the state has aged", () => {
  const stub = makeStub()
  const m = new AnimationStateMachine(
    stub,
    { idle: { clip: "idle" }, walk: { clip: "walk" } },
    [{ from: "idle", to: "walk", exitTime: 0.5, fade: 0.1 }],
    { initial: "idle" }
  )
  STEPS(m, 20) // 0.33s — too early
  assert.equal(m.state, "idle")
  STEPS(m, 12) // crosses 0.5s
  assert.equal(m.state, "walk")
})

test("non-loop clip with unconditional transition returns near clip end", () => {
  const stub = makeStub()
  const m = new AnimationStateMachine(
    stub,
    { skill: { clip: "skill", loop: false }, idle: { clip: "idle" } },
    [{ from: "skill", to: "idle", fade: 0.25 }],
    { initial: "skill" }
  )
  STEPS(m, 60) // 1s: mid-skill
  assert.equal(m.state, "skill")
  STEPS(m, 50) // ~1.83s > 1.97 - 0.25
  assert.equal(m.state, "idle")
  const w = weightsByName(stub)
  assert.ok(w.skill > 0.05, `skill still fading out: ${JSON.stringify(w)}`)
})

test("wildcard transitions fire from any state but never self-loop", () => {
  const stub = makeStub()
  let alarm = false
  const m = new AnimationStateMachine(
    stub,
    { idle: { clip: "idle" }, walk: { clip: "walk" }, skill: { clip: "skill" } },
    [{ from: "*", to: "skill", when: () => alarm, fade: 0.1 }],
    { initial: "walk" }
  )
  STEPS(m, 5)
  alarm = true
  STEPS(m, 1)
  assert.equal(m.state, "skill")
  const t1 = m.stateTime
  STEPS(m, 30) // alarm stays true; must not re-enter and reset the clock
  assert.ok(m.stateTime > t1, "no self-restart")
})

test("go() forces a transition and onEnter/onExit fire in order", () => {
  const stub = makeStub()
  const calls = []
  const m = new AnimationStateMachine(
    stub,
    {
      idle: { clip: "idle", onExit: (to) => calls.push(`exit-idle->${to}`) },
      skill: { clip: "skill", onEnter: (from) => calls.push(`enter-skill<-${from}`) },
    },
    [],
    { initial: "idle" }
  )
  STEPS(m, 5)
  m.go("skill", 0.2)
  assert.equal(m.state, "skill")
  assert.deepEqual(calls, ["exit-idle->skill", "enter-skill<-idle"])
  STEPS(m, 6)
  const w = weightsByName(stub)
  assert.ok(w.idle > 0.05 && w.skill > 0.05, `fading: ${JSON.stringify(w)}`)
})

test("speed scales the clip clock", () => {
  const stub = makeStub()
  const m = new AnimationStateMachine(stub, { idle: { clip: "idle", speed: 2 } }, [], { initial: "idle" })
  STEPS(m, 30) // 0.5s of wall time
  const t = stub.lastEntries.find((e) => e.name === "idle").time
  assert.ok(Math.abs(t - 1) < 0.05, `2x speed: ${t}`)
})

test("delegate state merges its entries under the fade", () => {
  const stub = makeStub()
  let go = false
  const m = new AnimationStateMachine(
    stub,
    {
      loco: {
        entries: () => [
          { name: "idle", time: 0.5, weight: 0.4 },
          { name: "walk", time: 0.2, weight: 0.6 },
        ],
      },
      skill: { clip: "skill" },
    },
    [{ from: "loco", to: "skill", when: () => go, fade: 0.2 }],
    { initial: "loco" }
  )
  STEPS(m, 5)
  let w = weightsByName(stub)
  assert.ok(Math.abs(w.idle - 0.4) < 1e-6 && Math.abs(w.walk - 0.6) < 1e-6, JSON.stringify(w))
  go = true
  STEPS(m, 6) // 0.1s: mid-fade
  w = weightsByName(stub)
  assert.ok(w.skill > 0.05, `incoming: ${JSON.stringify(w)}`)
  const total = w.idle + w.walk + w.skill
  assert.ok(Math.abs(total - 1) < 1e-6, `sums to 1: ${total}`)
  // the delegate's internal ratio survives the group scale
  assert.ok(Math.abs(w.idle / w.walk - 0.4 / 0.6) < 1e-3, `ratio kept: ${JSON.stringify(w)}`)
})

test("unknown states throw", () => {
  const stub = makeStub()
  assert.throws(() => new AnimationStateMachine(stub, { idle: { clip: "idle" } }, [], { initial: "nope" }))
  const m = new AnimationStateMachine(stub, { idle: { clip: "idle" } }, [], { initial: "idle" })
  assert.throws(() => m.go("nope"))
})
