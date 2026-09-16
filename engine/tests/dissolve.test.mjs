// Taking the cast apart: the cycle an effect declares, and when it runs. Run: npm test.
//
// The body's dissolve is not drawn by the effect — the material shell throws the
// model away — so the effect declares the timing and the engine performs it. What
// this pins is WHEN: an effect with clips vanishes on each clip and stands whole
// between them, and one without keeps the repeating cycle it always had.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import {
  DISSOLVE_PARAMS,
  dissolveConstants,
  dissolveCycleOf,
  sampleDissolveCycle,
  scheduledDissolve,
} from "../dist/effect-schedule.js"

const near = (got, want, what) => assert.ok(Math.abs(got - want) < 1e-9, `${what}: ${got} != ${want}`)
// Teleportation's own numbers.
const TIMINGS = { apart: 0.5, gone: 0.65, back: 0.35, whole: 3 }
const CYCLE = dissolveCycleOf(TIMINGS)

test("the four durations become four moments, and the cycle starts whole", () => {
  assert.deepEqual(CYCLE, { period: 4.5, breakAt: 3, hiddenAt: 3.5, backAt: 4.15, doneAt: 4.5 })
  assert.equal(dissolveCycleOf({ apart: 0, gone: 0, back: 0, whole: 0 }), null, "nothing is not a cycle")
})

test("a source's constants are read, and a dial of the same name replaces them", () => {
  const wgsl = [
    "const DISSOLVE_APART = 0.50;   // she comes apart",
    "const DISSOLVE_GONE = 0.65;",
    "const DISSOLVE_BACK = 0.35;",
    "const DISSOLVE_WHOLE = 3.00;",
  ].join("\n")
  assert.deepEqual(dissolveConstants(wgsl), TIMINGS)
  // An effect that declares them as dials writes no constants at all.
  assert.deepEqual(dissolveConstants("#param float DISSOLVE_APART 0.5 0.05 3"), { apart: 0, gone: 0, back: 0, whole: 0 })
  assert.deepEqual(
    DISSOLVE_PARAMS.map(([k, name]) => [k, name]),
    [
      ["apart", "DISSOLVE_APART"],
      ["gone", "DISSOLVE_GONE"],
      ["back", "DISSOLVE_BACK"],
      ["whole", "DISSOLVE_WHOLE"],
    ],
  )
})

test("the cycle is whole, then goes, then comes back", () => {
  near(sampleDissolveCycle(CYCLE, 0), 1, "standing there")
  near(sampleDissolveCycle(CYCLE, 3), 1, "the departure begins whole")
  near(sampleDissolveCycle(CYCLE, 3.25), 0.5, "halfway apart")
  near(sampleDissolveCycle(CYCLE, 3.6), 0, "gone")
  near(sampleDissolveCycle(CYCLE, 4.325), 0.5, "halfway back")
  near(sampleDissolveCycle(CYCLE, 4.5), 1, "whole again, and the next turn begins")
  near(sampleDissolveCycle(CYCLE, 7.5), 1, "one period on, the same place")
  near(sampleDissolveCycle(CYCLE, 7.75), 0.5, "and the same departure")
})

test("outside every clip she is whole", () => {
  const lane = [{ start: 10, end: 14.5 }]
  near(scheduledDissolve(lane, CYCLE, 0), 1, "before")
  near(scheduledDissolve(lane, CYCLE, 9.99), 1, "just before")
  near(scheduledDissolve(lane, CYCLE, 20), 1, "after")
  near(scheduledDissolve([], CYCLE, 12), 1, "no lane at all")
})

test("a clip opens on the departure, so the motes fall inside it", () => {
  const lane = [{ start: 10, end: 14.5 }]
  near(scheduledDissolve(lane, CYCLE, 10), 1, "whole at the clip's first frame")
  near(scheduledDissolve(lane, CYCLE, 10.25), 0.5, "coming apart a quarter second in")
  near(scheduledDissolve(lane, CYCLE, 10.6), 0, "gone")
  near(scheduledDissolve(lane, CYCLE, 11.325), 0.5, "coming back")
  near(scheduledDissolve(lane, CYCLE, 11.5), 1, "whole again, well inside the clip")
})

test("a second clip is a second teleport, and a long one repeats", () => {
  const two = [
    { start: 10, end: 14.5 },
    { start: 30, end: 34.5 },
  ]
  near(scheduledDissolve(two, CYCLE, 30.25), 0.5, "the second clip departs at its own start")
  near(scheduledDissolve(two, CYCLE, 25), 1, "and she is whole in between")
  // An open clip runs on: the cycle repeats every period from its start.
  const open = [{ start: 10 }]
  near(scheduledDissolve(open, CYCLE, 10.25), 0.5, "first turn")
  near(scheduledDissolve(open, CYCLE, 14.75), 0.5, "one period later")
})

test("the engine evaluates dissolves before the cast is written", () => {
  const src = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  const body = src.replace(/\/\/[^\n]*/g, "")
  const at = body.indexOf("this.evaluateDissolves()")
  assert.ok(at > 0, "the frame must evaluate dissolves")
  assert.ok(body.indexOf("this.updateCameraUniforms()", at) > at, "and do it before the cast is written")
  // The effect's own cycle: its dials where it declares them, its clips when it has them.
  const fn = src.slice(src.indexOf("private evaluateDissolves"), src.indexOf("\n  }", src.indexOf("private evaluateDissolves")))
  assert.match(fn, /scheduledDissolve\(fx\.window, cycle, transport\)/)
  assert.match(fn, /sampleDissolveCycle\(cycle, this\.sceneClock\)/)
  assert.match(src, /const slot = fx\.paramLayout\.get\(name\)\n\s*if \(slot\) t\[key\] = Math\.max\(0, fx\.paramsData\[slot\.offset\]\)/)
  assert.match(src, /dissolve: d\.dissolve \? dissolveConstants\(authored\) : null/)
})
