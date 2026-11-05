// A motion can hide its model: VMD's IK/display block carries a 表示 flag per
// record, and motions use it to make a character vanish for a stretch (MMD
// and Blender both honour it). The loader always parsed it; the clip now keeps
// it, and the writer must hand it back — a record written as "shown" would
// bring her back mid-hide.

import { test } from "node:test"
import assert from "node:assert/strict"
import { VMDWriter } from "../dist/vmd-writer.js"
import { VMDLoader } from "../dist/vmd-loader.js"

const clip = (extra) => ({ boneTracks: new Map(), morphTracks: new Map(), frameCount: 300, ...extra })

test("a hide and a show round-trip through a written VMD", () => {
  const visibility = [
    { frame: 0, visible: true },
    { frame: 90, visible: false },
    { frame: 150, visible: true },
  ]
  const back = VMDLoader.loadIkFromBuffer(new VMDWriter().write(clip({ visibility })))
  assert.deepEqual(
    back.map((r) => ({ frame: r.frame, visible: r.visible })),
    visibility,
  )
})

test("an IK switch inside a hidden stretch is written hidden", () => {
  const visibility = [
    { frame: 30, visible: false },
    { frame: 120, visible: true },
  ]
  const ikTracks = new Map([["左足ＩＫ", [{ frame: 60, enabled: false }, { frame: 200, enabled: true }]]])
  const back = VMDLoader.loadIkFromBuffer(new VMDWriter().write(clip({ visibility, ikTracks })))
  assert.deepEqual(
    back.map((r) => [r.frame, r.visible, r.states.length]),
    [
      [30, false, 0],
      [60, false, 1],
      [120, true, 0],
      [200, true, 1],
    ],
  )
})

test("a clip with no visibility writes every IK record shown", () => {
  const ikTracks = new Map([["右足ＩＫ", [{ frame: 10, enabled: false }]]])
  const back = VMDLoader.loadIkFromBuffer(new VMDWriter().write(clip({ ikTracks })))
  assert.deepEqual(
    back.map((r) => [r.frame, r.visible]),
    [[10, true]],
  )
})
