// A VMD's five section counts are positional: a reader reaches each block by
// walking every count before it. Both writers used to stop early -- writeCamera
// after the camera block, write() after the morph block whenever a clip carried
// no IK state -- so anything that kept reading ran off the end of the buffer and
// parsed whatever followed as the block it expected.
//
// That is not theoretical. The camera file is カメラ・照明, camera AND lighting,
// so MMD looks for the light block exactly where the file used to stop; it lit
// the scene from past the end of the file, and users reported a stray coloured
// light on camera-only exports.
//
// These tests walk the table the way MMD does and assert the file ends on a
// complete one.

import { test } from "node:test"
import assert from "node:assert/strict"
import { VMDWriter } from "../dist/vmd-writer.js"
import { Vec3, Quat } from "../dist/math.js"
import { rawInterpolationToBoneInterpolation } from "../dist/animation.js"

const IP = rawInterpolationToBoneInterpolation(new Uint8Array(64).fill(20))

/** Record sizes, in file order. IK is variable and handled separately. */
const SECTIONS = [
  ["bone", 111],
  ["morph", 23],
  ["camera", 61],
  ["light", 28],
  ["selfShadow", 9],
]

/**
 * Walk the section table positionally, exactly as a reader with no bounds
 * checking would. Throws with the section name if the file runs out.
 */
function walk(buf) {
  const view = new DataView(buf)
  const end = buf.byteLength
  let o = 50 // 30-byte signature + 20-byte model name
  const counts = {}
  for (const [name, size] of SECTIONS) {
    assert.ok(o + 4 <= end, `file ended before the ${name} count (at ${o} of ${end})`)
    const n = view.getUint32(o, true)
    o += 4
    assert.ok(o + n * size <= end,
      `${name} block claims ${n} records (${n * size} bytes) but only ${end - o} remain`)
    o += n * size
    counts[name] = n
  }
  assert.ok(o + 4 <= end, `file ended before the IK count (at ${o} of ${end})`)
  counts.ik = view.getUint32(o, true)
  o += 4
  return { counts, consumed: o, end }
}

const camKf = (frame) => ({
  frame,
  distance: -35.5,
  target: new Vec3(0, 10, 0),
  rotation: new Vec3(0, 0, 0),
  fov: 30,
})

function clip({ bones = true, morphs = true, ik = false } = {}) {
  return {
    boneTracks: bones
      ? new Map([["センター", [
          { boneName: "センター", frame: 0, rotation: new Quat(0, 0, 0, 1), translation: new Vec3(0, 0, 0), interpolation: IP },
        ]]])
      : new Map(),
    morphTracks: morphs
      ? new Map([["まばたき", [{ morphName: "まばたき", frame: 0, weight: 1 }]]])
      : new Map(),
    ikTracks: ik ? new Map([["左足ＩＫ", [{ frame: 0, enabled: false }]]]) : new Map(),
    frameCount: 30,
  }
}

test("a camera VMD ends on a complete section table", () => {
  const buf = new VMDWriter().writeCamera([camKf(0), camKf(60), camKf(120)])
  const { counts, consumed, end } = walk(buf)
  assert.equal(counts.bone, 0)
  assert.equal(counts.morph, 0)
  assert.equal(counts.camera, 3)
  // The regression: light is the block immediately after camera, and it is the
  // one MMD reads from a カメラ・照明 file.
  assert.equal(counts.light, 0)
  assert.equal(counts.selfShadow, 0)
  assert.equal(counts.ik, 0)
  assert.equal(consumed, end, "no bytes left over after a full walk")
})

test("a camera VMD with no keyframes still carries every count", () => {
  const { counts, consumed, end } = walk(new VMDWriter().writeCamera([]))
  assert.equal(counts.camera, 0)
  assert.equal(counts.light, 0)
  assert.equal(consumed, end)
})

test("a motion VMD ends on a complete section table, with or without IK", () => {
  for (const ik of [false, true]) {
    const buf = new VMDWriter().write(clip({ ik }))
    const { counts, consumed, end } = walk(buf)
    assert.equal(counts.bone, 1, `bones, ik=${ik}`)
    assert.equal(counts.morph, 1, `morphs, ik=${ik}`)
    assert.equal(counts.camera, 0, `camera, ik=${ik}`)
    assert.equal(counts.light, 0, `light, ik=${ik}`)
    assert.equal(counts.selfShadow, 0, `self-shadow, ik=${ik}`)
    assert.equal(counts.ik, ik ? 1 : 0, `ik count, ik=${ik}`)
    if (!ik) assert.equal(consumed, end, "no bytes left over when there is no IK block")
  }
})

test("each split half ends on a complete section table", () => {
  for (const tracks of ["all", "motion", "morphs"]) {
    const buf = new VMDWriter().write(clip({ ik: true }), { tracks })
    const { counts } = walk(buf)
    assert.equal(counts.camera, 0, `camera, tracks=${tracks}`)
    assert.equal(counts.light, 0, `light, tracks=${tracks}`)
    assert.equal(counts.selfShadow, 0, `self-shadow, tracks=${tracks}`)
  }
})
