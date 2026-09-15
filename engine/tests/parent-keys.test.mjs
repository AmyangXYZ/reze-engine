// Parent keys: which hold a keyed model is in at one moment. Run: npm test.
//
// Pure, so the boundary can be checked for real. A switch that lands one frame
// late is a ball that leaves her hand after it should have, in a video someone
// posts — and nothing in an editor looks wrong about it.

import { test } from "node:test"
import assert from "node:assert/strict"
import { parentKeyIndex, parentKeySpan } from "../dist/parent-keys.js"

const keys = [{ time: 0 }, { time: 4 }, { time: 5 }]

test("no keys is -1, and the first key holds before its own time", () => {
  assert.equal(parentKeyIndex([], 3), -1)
  assert.equal(parentKeyIndex([{ time: 2 }], 0), 0)
  assert.equal(parentKeyIndex([{ time: 2 }, { time: 6 }], 1), 0)
})

test("a key holds from its time until the next key's", () => {
  assert.equal(parentKeyIndex(keys, 3.9), 0)
  assert.equal(parentKeyIndex(keys, 4), 1, "the switch time is the new hold")
  assert.equal(parentKeyIndex(keys, 4.5), 1)
  assert.equal(parentKeyIndex(keys, 5), 2)
  assert.equal(parentKeyIndex(keys, 100), 2, "the last key holds to the end")
})

test("a clock summed from frame deltas reaches a key on its frame, and not a frame early", () => {
  for (const fps of [30, 60, 120]) {
    let t = 0
    for (let f = 0; f < 4 * fps; f++) t += 1 / fps
    assert.equal(parentKeyIndex(keys, t), 1, `${fps}fps: frame ${4 * fps} is the switch`)
    assert.equal(parentKeyIndex(keys, 4 - 1 / fps), 0, `${fps}fps: the frame before is still the old hold`)
  }
})

test("a tweened key is arrived at across the time since the previous key", () => {
  const tweened = [{ time: 0 }, { time: 2, tween: true }, { time: 4 }]
  assert.deepEqual(parentKeySpan(tweened, 0), { index: 0, toward: 0 }, "at the previous key nothing has moved")
  assert.deepEqual(parentKeySpan(tweened, 1), { index: 0, toward: 0.5 })
  assert.deepEqual(parentKeySpan(tweened, 2), { index: 1, toward: 0 }, "at its own time the tweened key is in force")
  assert.deepEqual(parentKeySpan(tweened, 3), { index: 1, toward: 0 }, "a key that does not tween switches")
  assert.deepEqual(parentKeySpan([], 1), { index: -1, toward: 0 })
})
