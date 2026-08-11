// Bone morphs (PMX type 2) and the pose, which have to compose in that order.
//
// A bone morph is an OFFSET on top of whatever the pose sources produced, so the
// previous frame's offset has to come off before the current frame's pose goes
// on. It used to come off afterwards, inside applyBoneMorphs: the restore
// overwrote the freshly animated locals with a snapshot taken a frame earlier,
// and the re-snapshot immediately below saved that same stale value again. Every
// bone any bone morph touched was pinned to the first frame the model was posed
// at — permanently, and at any weight, since the restore ran over the touched set
// before the weight test.
//
// It surfaced as "the arms don't animate on some models", because arms are what
// these morphs touch: T-Pose / A-Pose / ShouderBlend / ElbowBlend adjusters on 腕
// and ひじ are near-universal on character models, and nothing else in the rig is
// a comparably popular bone-morph target.

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

/** The first model on disk carrying a bone morph, plus a bone that morph moves. */
const findRigWithBoneMorph = () => {
  for (const path of findModels()) {
    let model
    try {
      model = PmxLoader.loadFromBuffer(toAB(readFileSync(path)))
    } catch {
      continue
    }
    const bones = model.getSkeleton().bones
    for (const morph of model.morphing?.morphs ?? []) {
      if (morph.type !== 2 || !morph.boneOffsets) continue
      for (const off of morph.boneOffsets) {
        const bone = bones[off.boneIndex]
        // A morph whose offset is zero would prove nothing either way.
        const moves =
          Math.hypot(...off.translation) > 1e-6 || Math.abs(off.rotation[3]) < 0.999999
        if (bone && moves)
          return { path, model, boneName: bone.name, boneIndex: off.boneIndex, morphName: morph.name }
      }
    }
  }
  return null
}

const RIG = findRigWithBoneMorph()

// MMD's default bezier handles — linear enough for "did this bone move at all".
const LINEAR = () => [
  { x: 20, y: 20 },
  { x: 107, y: 107 },
]
const interpolation = () => ({
  rotation: LINEAR(),
  translationX: LINEAR(),
  translationY: LINEAR(),
  translationZ: LINEAR(),
})

// 45° about X, as a unit quaternion.
const HALF = (45 * Math.PI) / 180 / 2

/** A clip that swings one bone through 45° over 30 frames. */
const swingClip = (boneName) => ({
  boneTracks: new Map([
    [
      boneName,
      [
        {
          boneName,
          frame: 0,
          rotation: new Quat(0, 0, 0, 1),
          translation: new Vec3(0, 0, 0),
          interpolation: interpolation(),
        },
        {
          boneName,
          frame: 30,
          rotation: new Quat(Math.sin(HALF), 0, 0, Math.cos(HALF)),
          translation: new Vec3(0, 0, 0),
          interpolation: interpolation(),
        },
      ],
    ],
  ]),
  morphTracks: new Map(),
  frameCount: 30,
})

test(
  "a bone a bone morph touches still follows its clip",
  { skip: RIG ? false : "no model with a bone morph on this machine" },
  () => {
    const { model, boneName, boneIndex, morphName } = RIG
    model.loadClip("swing", swingClip(boneName))
    model.show("swing")

    // The local rotation, not a world position: a bone rotating about its own
    // pivot does not move its own origin, only its children's.
    //
    // Frame 0 first, because that is what did the pinning — whatever pose landed
    // first became the snapshot every later frame was restored to.
    model.seek(0)
    model.update(0, false)
    const a = model.getBoneLocalRotation(boneIndex)
    const start = [a.x, a.y, a.z, a.w]

    model.seek(1) // frame 30 at 30fps
    model.update(0, false)
    const b = model.getBoneLocalRotation(boneIndex)

    const delta = Math.hypot(b.x - start[0], b.y - start[1], b.z - start[2], b.w - start[3])
    assert.ok(
      delta > 1e-3,
      `${boneName} (touched by morph "${morphName}") held frame 0's rotation at frame 30 — ` +
        `bone morphs are pinning the animated pose`,
    )
  },
)

test(
  "a bone morph on a model with no clip does not compound",
  { skip: RIG ? false : "no model with a bone morph on this machine" },
  () => {
    // The reason the undo exists at all. A stage has no clip to rewrite its
    // bones, so an offset re-added every frame would walk the door off its hinge.
    const { path, boneName, boneIndex, morphName } = RIG
    const model = PmxLoader.loadFromBuffer(toAB(readFileSync(path)))
    model.setMorphWeight(morphName, 1)

    model.update(0, false)
    const r = model.getBoneLocalRotation(boneIndex)
    const first = [r.x, r.y, r.z, r.w]

    for (let i = 0; i < 10; i++) model.update(1 / 60, false)
    const after = model.getBoneLocalRotation(boneIndex)

    const drift = Math.hypot(after.x - first[0], after.y - first[1], after.z - first[2], after.w - first[3])
    assert.ok(drift < 1e-5, `${boneName} drifted ${drift} over 10 idle frames — the offset is compounding`)
  },
)
