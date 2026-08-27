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

// ── the host-authored-FK regression: setIKEnabled(false) must leave the legs ──
import { readFileSync, existsSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"
import { PmxLoader } from "../dist/pmx-loader.js"
import { Quat } from "../dist/math.js"

const here = dirname(fileURLToPath(import.meta.url))
const MODEL = join(here, "../../web/public/models/托特/托特.pmx")

test("IK off leaves host-written chain-link rotations alone", { skip: !existsSync(MODEL) }, () => {
  // The MiKaPo case: no clip, the host writes leg FK by hand, IK disabled
  // engine-wide. The link wipe that feeds the solver a clean start used to run
  // anyway and erased the pose every frame — on any model that still has its
  // IK chains, which the bundled demo model does not.
  const buf = readFileSync(MODEL)
  const model = PmxLoader.loadFromBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
  const q = new Quat(0.3, 0, 0, Math.sqrt(1 - 0.09)).normalize()
  const KNEE = "左ひざ"

  assert.ok(model.getSkeleton().bones.some((b) => b.name === KNEE), "the rig has a knee")
  // The written rotation must survive update with IK off: the ankle child moves
  // with the knee bend, so compare the FOOT's world position against rest.
  model.resetAllBones()
  model.update(1 / 60, false)
  const rest = model.getBoneWorldPosition("左足首")
  model.rotateBones({ [KNEE]: q }, 0)
  model.update(1 / 60, false)
  const bent = model.getBoneWorldPosition("左足首")
  const moved = Math.hypot(bent.x - rest.x, bent.y - rest.y, bent.z - rest.z)
  assert.ok(moved > 0.5, `a hand-bent knee moves the ankle with IK off (moved ${moved.toFixed(3)})`)

  // And with IK ON and no clip suspended, the wipe still runs — the solver
  // path keeps its clean start (the knee limit-cycle fix stays fixed).
  model.resetAllBones()
  model.rotateBones({ [KNEE]: q }, 0)
  model.update(1 / 60, true)
  const solved = model.getBoneWorldPosition("左足首")
  const wiped = Math.hypot(solved.x - rest.x, solved.y - rest.y, solved.z - rest.z)
  assert.ok(wiped < moved * 0.5, `with IK on, the link wipe still precedes the solve (moved ${wiped.toFixed(3)})`)
})
