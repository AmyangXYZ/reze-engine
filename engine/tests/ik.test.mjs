import { test } from "node:test"
import assert from "node:assert/strict"
import { VMDWriter } from "../dist/vmd-writer.js"
import { VMDLoader } from "../dist/vmd-loader.js"

test("IK block round-trips through writer and loader", () => {
  const clip = {
    boneTracks: new Map(),
    morphTracks: new Map(),
    frameCount: 0,
    ikTracks: new Map([
      ["左足ＩＫ", [{ frame: 0, enabled: false }, { frame: 30, enabled: true }]],
      ["右足ＩＫ", [{ frame: 0, enabled: false }]],
    ]),
  }
  const back = VMDLoader.loadIkFromBuffer(new VMDWriter().write(clip))
  assert.equal(back.length, 2)
  assert.equal(back[0].frame, 0)
  assert.deepEqual(
    back[0].states.map((s) => [s.boneName, s.enabled]).sort(),
    [["右足ＩＫ", false], ["左足ＩＫ", false]].sort(),
  )
  assert.equal(back[1].frame, 30)
  assert.deepEqual(back[1].states, [{ boneName: "左足ＩＫ", enabled: true }])
})

test("a clip with no IK state writes no trailing block", () => {
  const bare = { boneTracks: new Map(), morphTracks: new Map(), frameCount: 0 }
  assert.deepEqual(VMDLoader.loadIkFromBuffer(new VMDWriter().write(bare)), [])
})

test("IK state survives loadClip → exportVmd", async () => {
  const { AnimationState } = await import("../dist/animation.js")
  const state = new AnimationState()
  const ikTracks = new Map([["左足ＩＫ", [{ frame: 0, enabled: false }]]])
  state.loadAnimation("c", { boneTracks: new Map(), morphTracks: new Map(), ikTracks, frameCount: 0 })
  const stored = state.getAnimationClip("c")
  assert.ok(stored.ikTracks, "clip lost its IK tracks on the way into AnimationState")
  const back = VMDLoader.loadIkFromBuffer(new VMDWriter().write(stored))
  assert.equal(back[0].states[0].boneName, "左足ＩＫ")
  assert.equal(back[0].states[0].enabled, false)
})
