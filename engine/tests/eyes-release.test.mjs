// Eyes on the camera, handed back.
//
// The solve writes 左目/右目 and holds 両目 at identity, and a clip only writes the
// bones it KEYS. setEyeTracking(null) used to clear the request and nothing else,
// so on a motion that keys no eyes — most dances — or a model with no motion at
// all, the eyes kept the last look the camera drew for good.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const { PmxLoader } = await import("../dist/pmx-loader.js")
const { Quat, Vec3 } = await import("../dist/math.js")

/** Whatever models this machine happens to have, same as the loader suite. */
const findModels = () => {
  const roots = [
    join(here, "../../web/public/models"),
    join(here, "../../../MiKaPo/public/models"),
    join(here, "../../../reze-studio/public/models"),
    join(here, "../../../reze-design/public/models"),
  ].filter(existsSync)
  const out = []
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry)
      if (statSync(p).isDirectory()) walk(p)
      else if (p.toLowerCase().endsWith(".pmx")) out.push(p)
    }
  }
  for (const r of roots) walk(r)
  return out
}

const toAB = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)

/** The first model on disk with a head and both eye bones. */
const findEyedRig = () => {
  for (const path of findModels()) {
    let model
    try {
      model = PmxLoader.loadFromBuffer(toAB(readFileSync(path)))
    } catch {
      continue
    }
    const names = model.getSkeleton().bones.map((b) => b.name)
    if (["頭", "左目", "右目"].every((n) => names.includes(n))) return path
  }
  return null
}

const RIG = findEyedRig()
const SKIP = { skip: RIG ? false : "no model with eye bones on this machine" }

const load = () => {
  const model = PmxLoader.loadFromBuffer(toAB(readFileSync(RIG)))
  const bones = model.getSkeleton().bones
  const index = (name) => bones.findIndex((b) => b.name === name)
  return { model, left: index("左目"), right: index("右目") }
}

const turn = (q) => 2 * Math.acos(Math.min(1, Math.abs(q.w)))

/** Track the eyes toward a point well off to her left and in front, for a few frames. */
const trackAside = (model) => {
  model.setEyeTracking({})
  // Model space; she rests facing -Z, head somewhere around y 15.
  model.setGazeTarget(new Vec3(25, 15, -30))
  for (let i = 0; i < 3; i++) model.update(1 / 60, false)
}

test("eyes handed back with no motion return to rest", SKIP, () => {
  const { model, left, right } = load()
  trackAside(model)
  assert.ok(turn(model.getBoneLocalRotation(left)) > 1e-3, "the solve should have turned 左目 at all")

  model.setEyeTracking(null)
  model.update(1 / 60, false)
  for (const [name, idx] of [["左目", left], ["右目", right]]) {
    const r = turn(model.getBoneLocalRotation(idx))
    assert.ok(r < 1e-6, `${name} kept ${(r * 180) / Math.PI}° of the tracked look after tracking was turned off`)
  }
})

test("eyes handed back under a motion that keys them take the motion's look", SKIP, () => {
  const { model, left } = load()
  // 10° about Y on 左目, held across the clip.
  const HALF = (10 * Math.PI) / 180 / 2
  const key = (frame) => ({
    boneName: "左目",
    frame,
    rotation: new Quat(0, Math.sin(HALF), 0, Math.cos(HALF)),
    translation: new Vec3(0, 0, 0),
    interpolation: {
      rotation: [{ x: 20, y: 20 }, { x: 107, y: 107 }],
      translationX: [{ x: 20, y: 20 }, { x: 107, y: 107 }],
      translationY: [{ x: 20, y: 20 }, { x: 107, y: 107 }],
      translationZ: [{ x: 20, y: 20 }, { x: 107, y: 107 }],
    },
  })
  model.loadClip("look", { boneTracks: new Map([["左目", [key(0), key(30)]]]), morphTracks: new Map(), frameCount: 30 })
  model.show("look")
  model.seek(0)
  trackAside(model)

  model.setEyeTracking(null)
  model.update(0, false)
  const r = model.getBoneLocalRotation(left)
  const gap = 2 * Math.acos(Math.min(1, Math.abs(r.x * 0 + r.y * Math.sin(HALF) + r.z * 0 + r.w * Math.cos(HALF))))
  assert.ok(gap < 1e-4, `左目 is ${(gap * 180) / Math.PI}° off the motion's key after tracking was turned off`)
})
