import { test } from "node:test"
import assert from "node:assert/strict"
import { AnimationState } from "../dist/animation.js"

const key = (frame, weight) => ({ frame, weight })

function clip({ bones = [], morphs = [], frameCount = 0 } = {}) {
  return {
    boneTracks: new Map(bones),
    morphTracks: new Map(morphs),
    frameCount,
  }
}

test("an expression file overwrites the body motion's morphs, and leaves its bones", () => {
  const state = new AnimationState()
  state.loadAnimation("dance", clip({
    bones: [["センター", [{ frame: 0 }, { frame: 60 }]]],
    morphs: [["まばたき", [key(0, 0), key(30, 1)]], ["笑い", [key(0, 1)]]],
    frameCount: 60,
  }))
  state.setMorphTracks("dance", new Map([["あ", [key(0, 0), key(10, 1)]]]), 40)
  const after = state.getAnimationClip("dance")
  // The face is entirely the expression file's — a half-overridden face is
  // nobody's intent, so the motion's own morphs do not survive alongside it.
  assert.deepEqual([...after.morphTracks.keys()], ["あ"])
  // …and the body is untouched.
  assert.deepEqual([...after.boneTracks.keys()], ["センター"])
  assert.equal(after.boneTracks.get("センター").length, 2)
})

test("with no expression file, the motion's own morphs are what plays", () => {
  const state = new AnimationState()
  state.loadAnimation("dance", clip({ morphs: [["まばたき", [key(0, 1)]]], frameCount: 60 }))
  assert.deepEqual([...state.getAnimationClip("dance").morphTracks.keys()], ["まばたき"])
})

test("the clip covers the longer of the two — an expression running past the body is not truncated", () => {
  const state = new AnimationState()
  state.loadAnimation("dance", clip({ frameCount: 60 }))
  state.setMorphTracks("dance", new Map([["あ", [key(0, 1)]]]), 90)
  assert.equal(state.getAnimationClip("dance").frameCount, 90)
  // …and a shorter expression never shrinks the body motion.
  state.setMorphTracks("dance", new Map([["い", [key(0, 1)]]]), 10)
  assert.equal(state.getAnimationClip("dance").frameCount, 90)
})

test("the two files may arrive in either order", () => {
  const state = new AnimationState()
  // Expression first: the clip is created rather than dropped on the floor.
  state.setMorphTracks("dance", new Map([["あ", [key(0, 1)]]]), 40)
  assert.deepEqual([...state.getAnimationClip("dance").morphTracks.keys()], ["あ"])
  assert.equal(state.getAnimationClip("dance").frameCount, 40)
})

test("a fresh motion load replaces the whole clip, expression included", () => {
  const state = new AnimationState()
  state.loadAnimation("dance", clip({ morphs: [["まばたき", [key(0, 1)]]], frameCount: 60 }))
  state.setMorphTracks("dance", new Map([["あ", [key(0, 1)]]]), 60)
  // Loading a new body motion is a new clip, not an accumulation — the host
  // re-applies the expression file it still holds, so the state stays legible.
  state.loadAnimation("dance", clip({ morphs: [["ウィンク", [key(0, 1)]]], frameCount: 30 }))
  assert.deepEqual([...state.getAnimationClip("dance").morphTracks.keys()], ["ウィンク"])
  assert.equal(state.getAnimationClip("dance").frameCount, 30)
})

test("a replacement reaches playback, because the evaluator re-reads the stored clip", () => {
  const state = new AnimationState()
  state.loadAnimation("dance", clip({ morphs: [["まばたき", [key(0, 1)]]], frameCount: 60 }))
  state.play("dance")
  state.setMorphTracks("dance", new Map([["あ", [key(0, 1)]]]), 60)
  assert.deepEqual([...state.getCurrentClip().morphTracks.keys()], ["あ"])
})
