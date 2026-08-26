// The strip: when an effect is alive, and how much of it. Run: npm test.
//
// Pure, so every edge can be checked exhaustively — which is the reason it is
// its own file. The failure mode is a beat missed in a video someone posts, and
// an off-by-one at an edge looks like nothing at all in an editor. Every case
// below is one a person can produce by dragging, which is the bar a schedule
// has to survive.

import { test } from "node:test"
import assert from "node:assert/strict"
import { effectState } from "../dist/effect-schedule.js"

const near = (got, want, what) => assert.ok(Math.abs(got - want) < 1e-9, `${what}: ${got} != ${want}`)

test("no window runs with the scene, exactly as before windows existed", () => {
  assert.deepEqual(effectState(null, 1, 0), { weight: 1, time: 0 })
  assert.deepEqual(effectState(null, 1, 30), { weight: 1, time: 30 })
  // Influence with no window is a permanent level — the same dial, not a second
  // one. This is what lets an effect be dimmed without being scheduled.
  assert.deepEqual(effectState(null, 0.5, 10), { weight: 0.5, time: 10 })
})

test("the edges are inclusive, and the clock is the effect's own", () => {
  const w = { start: 10, end: 20 }
  assert.deepEqual(effectState([w], 1, 9.99), { weight: 0, time: 0 }, "before the start")
  assert.deepEqual(effectState([w], 1, 10), { weight: 1, time: 0 }, "the start is ALIVE, at local zero")
  assert.deepEqual(effectState([w], 1, 15), { weight: 1, time: 5 }, "inside, the clock is local")
  assert.deepEqual(effectState([w], 1, 20), { weight: 1, time: 10 }, "the end is still alive")
  assert.deepEqual(effectState([w], 1, 20.01), { weight: 0, time: 0 }, "past the end")
})

test("a late entry plays its OWN opening", () => {
  // The reason the window owns the clock rather than sitting beside a
  // visibility flag: an effect entering at 60s should start at zero, not join
  // the scene a minute in and skip whatever it does on its first frame — which
  // for anything with a grid is its only chance to seed.
  assert.deepEqual(effectState([{ start: 60 }], 1, 60), { weight: 1, time: 0 })
  assert.deepEqual(effectState([{ start: 60 }], 1, 61.5), { weight: 1, time: 1.5 })
})

test("an open window never ends", () => {
  assert.deepEqual(effectState([{ start: 1 }], 1, 10_000), { weight: 1, time: 9_999 })
})

test("blend in ramps from zero to the level, over its own length", () => {
  const w = { start: 10, end: 100, blendIn: 4 }
  assert.equal(effectState([w], 1, 10).weight, 0, "a blend STARTS at zero")
  near(effectState([w], 1, 12).weight, 0.5, "halfway")
  assert.equal(effectState([w], 1, 14).weight, 1, "complete exactly on its length")
  assert.equal(effectState([w], 1, 50).weight, 1, "then flat")
  // Zero is a HARD CUT, which is right for a flash and wrong for a glow — the
  // author's call, and the default.
  assert.equal(effectState([{ start: 10, blendIn: 0 }], 1, 10).weight, 1)
  assert.equal(effectState([{ start: 10 }], 1, 10).weight, 1)
})

test("blend out reaches zero ON the end, and needs an end to measure from", () => {
  const w = { start: 0, end: 100, blendOut: 4 }
  assert.equal(effectState([w], 1, 95).weight, 1, "before it, flat")
  assert.equal(effectState([w], 1, 96).weight, 1, "it begins at end - length")
  near(effectState([w], 1, 98).weight, 0.5, "halfway")
  assert.equal(effectState([w], 1, 100).weight, 0, "zero on the end frame")
  // With no end there is nothing to ramp toward, and inventing an origin would
  // fade an effect out at a time nobody asked for.
  assert.equal(effectState([{ start: 0, blendOut: 4 }], 1, 5000).weight, 1)
})

test("blends ramp toward INFLUENCE, not toward 1", () => {
  // Blender's meaning. Ramping to 1 would make a half-strength effect jump
  // above its own setting on the way in and then drop back to it.
  const w = { start: 0, end: 100, blendIn: 10 }
  near(effectState([w], 0.5, 5).weight, 0.25, "halfway to half")
  near(effectState([w], 0.5, 50).weight, 0.5, "settles AT influence")
})

test("overlapping blends degrade to a triangle instead of breaking", () => {
  // Reachable by dragging the edges together under two long blends. The MINIMUM
  // of the ramps stays in range and stays smooth; multiplying them would dip
  // toward zero in the middle of a short strip, which is not a fade.
  const w = { start: 0, end: 10, blendIn: 100, blendOut: 100 }
  near(effectState([w], 1, 0).weight, 0, "starts at zero")
  near(effectState([w], 1, 5).weight, 0.05, "peaks in the middle")
  near(effectState([w], 1, 10).weight, 0, "ends at zero")
  for (let t = 0; t <= 10; t += 0.25) {
    const v = effectState([w], 1, t).weight
    assert.ok(v >= 0 && v <= 1, `out of range at ${t}: ${v}`)
  }
})

test("nonsense a person can drag into is empty, not inverted", () => {
  // A negative length makes every ramp produce garbage, so the answer to the
  // left edge crossing the right is "nothing plays".
  assert.deepEqual(effectState([{ start: 20, end: 10 }], 1, 15), { weight: 0, time: 0 })
  assert.deepEqual(effectState([{ start: 10, end: 10 }], 1, 10), { weight: 0, time: 0 })
})

test("influence out of range is clamped, and NaN is off", () => {
  // A hand-edited document, or a slider bound to something that went wrong.
  // WGSL leaves NaN comparisons indeterminate, so a NaN that reached a uniform
  // would not even fail the same way on every GPU.
  assert.equal(effectState(null, 5, 0).weight, 1)
  assert.equal(effectState(null, -2, 0).weight, 0)
  assert.equal(effectState(null, NaN, 0).weight, 0)
})

test("a scrub before zero is silent", () => {
  assert.deepEqual(effectState([{ start: 0, end: 10 }], 1, -1), { weight: 0, time: 0 })
})

test("the engine evaluates schedules before anything reads them", () => {
  // Order in the frame is the contract: the sim's clock, the particle uniform,
  // the light dispatch and the field draw all read what this writes. Pinned as
  // source because there is no device here to render a frame with.
  const src = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  const body = src.replace(/\/\/[^\n]*/g, "")
  const at = body.indexOf("this.evaluateEffectSchedules()")
  assert.ok(at > 0, "the frame must evaluate schedules")
  for (const after of ["this.stepSim(", "this.stepParticles(", "this.emitLights("]) {
    assert.ok(body.indexOf(after, at) > at, `${after} must come after the evaluation`)
  }
})

test("an unscheduled effect keeps what a caller set", () => {
  // The manual path — an animation's progress, a skill firing — writes
  // influence and time itself. Evaluating a window it does not have would
  // fight that caller for the field every frame.
  const src = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  const from = src.indexOf("private evaluateEffectSchedules")
  const fn = src.slice(from, src.indexOf("\n  }", from))
  assert.match(fn, /if \(!fx\.window \|\| fx\.window\.length === 0\) \{\s*fx\.weight = fx\.influence\s*continue/)
})

import { readFileSync } from "node:fs"

test("a lane holds many strips, and each one restarts the effect's clock", () => {
  // The whole reason a lane is a list. A hit placed at bar 8 and again at bar 24
  // has to play its opening BOTH times — resuming halfway through itself is the
  // failure this shape exists to prevent.
  const lane = [
    { start: 10, end: 13 },
    { start: 30, end: 33 },
  ]
  assert.deepEqual(effectState(lane, 1, 10), { weight: 1, time: 0 }, "first firing, at its own zero")
  assert.deepEqual(effectState(lane, 1, 12), { weight: 1, time: 2 })
  assert.deepEqual(effectState(lane, 1, 30), { weight: 1, time: 0 }, "SECOND firing starts at zero again")
  assert.deepEqual(effectState(lane, 1, 32), { weight: 1, time: 2 })
})

test("the gaps between strips are silent", () => {
  const lane = [
    { start: 10, end: 13 },
    { start: 30, end: 33 },
  ]
  assert.deepEqual(effectState(lane, 1, 9), { weight: 0, time: 0 }, "before the first")
  assert.deepEqual(effectState(lane, 1, 20), { weight: 0, time: 0 }, "between them")
  assert.deepEqual(effectState(lane, 1, 40), { weight: 0, time: 0 }, "after the last")
})

test("each strip carries its own blends", () => {
  const lane = [
    { start: 0, end: 10, blendIn: 4 },
    { start: 20, end: 30, blendOut: 4 },
  ]
  near(effectState(lane, 1, 2).weight, 0.5, "the first strip eases in")
  assert.equal(effectState(lane, 1, 20).weight, 1, "the second cuts in")
  near(effectState(lane, 1, 28).weight, 0.5, "and eases out")
})

test("an empty lane is unscheduled, not silent", () => {
  // [] arrives from a document whose last strip was dragged away. It has to mean
  // the same as never having been scheduled — an effect that vanished because
  // its lane was emptied would look like a broken effect.
  assert.deepEqual(effectState([], 1, 42), { weight: 1, time: 42 })
})

test("strips out of order still fire in time order", () => {
  // The document holds what the user laid down, and nothing sorts it. A lane
  // written back to front must behave the same as one written in order.
  const lane = [
    { start: 30, end: 33 },
    { start: 10, end: 13 },
  ]
  assert.deepEqual(effectState(lane, 1, 11), { weight: 1, time: 1 })
  assert.deepEqual(effectState(lane, 1, 31), { weight: 1, time: 1 })
})

test("where strips overlap, the one most recently entered wins", () => {
  // They are not meant to overlap — that is the rule every NLE enforces within
  // a track — but a hand-edited document can, and the answer has to be one
  // strip rather than a blend nobody asked for.
  const lane = [
    { start: 0, end: 20 },
    { start: 10, end: 30 },
  ]
  assert.deepEqual(effectState(lane, 1, 5), { weight: 1, time: 5 }, "only the first contains it")
  assert.deepEqual(effectState(lane, 1, 15), { weight: 1, time: 5 }, "both do — the later start wins")
  assert.deepEqual(effectState(lane, 1, 25), { weight: 1, time: 15 }, "only the second")
})

test("strips are evaluated against the TRANSPORT, never engine uptime", () => {
  // The bug this replaced: sceneClock only accumulates delta. It does not move
  // when you scrub, does not stop when you pause, and is really "how long has
  // this page been open" — so a strip at frame 100 fired once, moments after
  // load, and was dead for the rest of the session.
  const src = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  const from = src.indexOf("private evaluateEffectSchedules")
  const fn = src.slice(from, src.indexOf("\n  }", from))
  assert.match(fn, /const transport = this\.transportTime\(\)/)
  assert.match(fn, /effectState\(fx\.window, fx\.influence, transport\)/)
  // The epoch is still expressed against sceneClock, deliberately: the mounts
  // integrate on a smooth monotonic clock, and what changes is the VALUE handed
  // to them, not the clock they run on.
  assert.match(fn, /fx\.epochScene = this\.sceneClock - at\.time/)
})
