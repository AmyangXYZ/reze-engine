// Material parameters over time. Run: npm test.
//
// The sampler is pure and lives alone precisely so it can be tested here:
// everything else in the feature needs a GPU. What is checked is the part with
// edge cases — the ends, one key, keys at the same instant, mixed types — and
// the property the whole design rests on, that the value is a function of the
// scene clock and nothing else.

import { test } from "node:test"
import assert from "node:assert/strict"
import { paramChanged, sampleParamTrack } from "../dist/param-track.js"

const K = (t, v) => ({ t, v })

test("between keys it interpolates linearly", () => {
  const keys = [K(0, 0), K(2, 10)]
  assert.equal(sampleParamTrack(keys, 0), 0)
  assert.equal(sampleParamTrack(keys, 1), 5)
  assert.equal(sampleParamTrack(keys, 1.5), 7.5)
  assert.equal(sampleParamTrack(keys, 2), 10)
})

test("outside the track it HOLDS, never extrapolates", () => {
  // A parameter that ran off its keys and kept going would leave the scene
  // somewhere its author never described. The last key is the last thing said.
  const keys = [K(1, 4), K(3, 8)]
  assert.equal(sampleParamTrack(keys, -100), 4)
  assert.equal(sampleParamTrack(keys, 0.99), 4)
  assert.equal(sampleParamTrack(keys, 3.01), 8)
  assert.equal(sampleParamTrack(keys, 1e6), 8)
})

test("one key is a constant, and no keys is nothing at all", () => {
  assert.equal(sampleParamTrack([K(5, 3)], 0), 3)
  assert.equal(sampleParamTrack([K(5, 3)], 1e6), 3)
  // null, not 0: "this track says nothing" and "this track says zero" are
  // different, and the caller writes a uniform on one and not the other.
  assert.equal(sampleParamTrack([], 0), null)
})

test("two keys at the same instant are a STEP, later wins", () => {
  // The span is zero there. Dividing by it would put a NaN in a uniform every
  // frame, which is the kind of thing that shows up as one wrong material.
  const keys = [K(0, 0), K(1, 1), K(1, 9), K(2, 10)]
  assert.equal(sampleParamTrack(keys, 0.5), 0.5)
  assert.equal(sampleParamTrack(keys, 1), 9, "at the instant itself, the later key")
  assert.equal(sampleParamTrack(keys, 1.5), 9.5, "and the next segment starts from it")
  for (const t of [0, 0.5, 1, 1.5, 2]) assert.ok(Number.isFinite(sampleParamTrack(keys, t)))
})

test("vectors interpolate componentwise", () => {
  const keys = [K(0, [0, 10, 100]), K(1, [1, 20, 300])]
  assert.deepEqual(sampleParamTrack(keys, 0.5), [0.5, 15, 200])
  assert.deepEqual(sampleParamTrack(keys, 0), [0, 10, 100])
  assert.deepEqual(sampleParamTrack(keys, 1), [1, 20, 300])
})

test("a scalar and a vector in one track holds rather than guessing", () => {
  // An authoring mistake. Inventing a conversion would produce a plausible
  // wrong value; holding the segment's start produces an obvious one.
  const keys = [K(0, 1), K(1, [5, 5, 5])]
  assert.equal(sampleParamTrack(keys, 0.5), 1)
})

test("the value depends on the clock and nothing else", () => {
  // The property an offline export rests on: stepped at any rate, in any order,
  // the same time gives the same value. Anything reading wall time or holding
  // state between calls would fail this.
  const keys = [K(0, 0), K(1, 3), K(2.5, -2), K(4, 8)]
  const forward = []
  for (let t = 0; t <= 4.0001; t += 0.25) forward.push(sampleParamTrack(keys, t))
  const shuffled = [3.25, 0.5, 4, 1.75, 0].map((t) => sampleParamTrack(keys, t))
  const again = []
  for (let t = 0; t <= 4.0001; t += 0.25) again.push(sampleParamTrack(keys, t))
  assert.deepEqual(again, forward, "same clock, same values, on a second pass")
  assert.deepEqual(shuffled, [3.25, 0.5, 4, 1.75, 0].map((t) => sampleParamTrack(keys, t)))
})

test("binary search finds the right segment in a long track", () => {
  // Sorted keys are searched, not scanned, so a thousand-key track costs what a
  // four-key one does. A wrong midpoint shows up as a value from the wrong
  // segment, which is what this catches.
  const keys = Array.from({ length: 1000 }, (_, i) => K(i, i))
  for (const t of [0, 1, 499.5, 500, 998.25, 999]) {
    assert.equal(sampleParamTrack(keys, t), t, `at t=${t}`)
  }
})

test("paramChanged is what stops a still scene writing every frame", () => {
  assert.equal(paramChanged(1, 1), false)
  assert.equal(paramChanged(1, 1.0000001), true)
  assert.equal(paramChanged([1, 2, 3], [1, 2, 3]), false)
  assert.equal(paramChanged([1, 2, 3], [1, 2, 4]), true)
  // A first sample always writes: there is nothing on the GPU to compare to.
  assert.equal(paramChanged(0, null), true)
  assert.equal(paramChanged(null, null), false)
})
