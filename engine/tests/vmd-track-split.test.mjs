import { test } from "node:test"
import assert from "node:assert/strict"
import { VMDWriter } from "../dist/vmd-writer.js"
import { VMDLoader } from "../dist/vmd-loader.js"
import { Vec3, Quat } from "../dist/math.js"
import { rawInterpolationToBoneInterpolation } from "../dist/animation.js"

const IP = rawInterpolationToBoneInterpolation(new Uint8Array(64).fill(20))

function clip() {
  return {
    boneTracks: new Map([
      ["センター", [
        { boneName: "センター", frame: 0, rotation: new Quat(0, 0, 0, 1), translation: new Vec3(0, 0, 0), interpolation: IP },
        { boneName: "センター", frame: 30, rotation: new Quat(0, 0, 0, 1), translation: new Vec3(1, 2, 3), interpolation: IP },
      ]],
    ]),
    morphTracks: new Map([
      ["まばたき", [
        { morphName: "まばたき", frame: 0, weight: 0 },
        { morphName: "まばたき", frame: 15, weight: 1 },
        { morphName: "まばたき", frame: 30, weight: 0 },
      ]],
    ]),
    ikTracks: new Map([["左足ＩＫ", [{ frame: 0, enabled: false }]]]),
    frameCount: 30,
  }
}

/** Bone and morph counts straight out of the header, without a full parse. */
function counts(buf) {
  const view = new DataView(buf)
  const bones = view.getUint32(50, true)
  const morphs = view.getUint32(54 + bones * 111, true)
  return { bones, morphs }
}

test("all: both halves, as before", () => {
  const { bones, morphs } = counts(new VMDWriter().write(clip()))
  assert.equal(bones, 2)
  assert.equal(morphs, 3)
})

test("no options is identical to tracks: all", () => {
  const a = Buffer.from(new VMDWriter().write(clip()))
  const b = Buffer.from(new VMDWriter().write(clip(), { tracks: "all" }))
  assert.ok(a.equals(b), "default must stay byte-identical")
})

test("motion: bones only, no morph frames", () => {
  const { bones, morphs } = counts(new VMDWriter().write(clip(), { tracks: "motion" }))
  assert.equal(bones, 2)
  assert.equal(morphs, 0)
})

test("morphs: morph frames only, no bones", () => {
  const { bones, morphs } = counts(new VMDWriter().write(clip(), { tracks: "morphs" }))
  assert.equal(bones, 0)
  assert.equal(morphs, 3)
})

test("split halves parse back to the right frames", () => {
  const motion = VMDLoader.loadFromBuffer(new VMDWriter().write(clip(), { tracks: "motion" }))
  const morphs = VMDLoader.loadFromBuffer(new VMDWriter().write(clip(), { tracks: "morphs" }))
  const allBone = motion.flatMap((k) => k.boneFrames)
  const allMorph = motion.flatMap((k) => k.morphFrames)
  assert.equal(allBone.length, 2)
  assert.equal(allMorph.length, 0)
  assert.equal(morphs.flatMap((k) => k.boneFrames).length, 0)
  assert.equal(morphs.flatMap((k) => k.morphFrames).length, 3)
})

test("IK rides with motion, not with morphs", () => {
  // The IK block only exists when the writer emitted one; a morphs-only file
  // must not carry bone state.
  const motionIk = VMDLoader.loadIkFromBuffer(new VMDWriter().write(clip(), { tracks: "motion" }))
  const morphIk = VMDLoader.loadIkFromBuffer(new VMDWriter().write(clip(), { tracks: "morphs" }))
  assert.equal(motionIk.length, 1)
  assert.equal(motionIk[0].states[0].boneName, "左足ＩＫ")
  assert.equal(motionIk[0].states[0].enabled, false)
  assert.equal(morphIk.length, 0)
})
