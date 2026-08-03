// Physics invariant suite against a real PMX rig, run fully headless. These
// pin the behaviors every optimization must preserve: finiteness, settling,
// determinism, kinematic tracking, teleport recovery, and (for broadphase
// work) exact candidate-pair equivalence with the brute-force filter.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const MODEL = join(here, "../../web/public/models/托特/托特.pmx")
const VMD = join(here, "../../web/public/unity-fbx-locomotion/vmd/Run_Lfoot.vmd")
const hasAssets = existsSync(MODEL) && existsSync(VMD)

const { PmxLoader } = await import("../dist/pmx-loader.js")
const { VMDLoader } = await import("../dist/vmd-loader.js")
const { RezePhysics } = await import("../dist/physics/physics.js")
const { aabbOverlap } = await import("../dist/physics/contact.js")

const toAB = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
const pmxBuf = hasAssets ? readFileSync(MODEL) : null

const DT = 1 / 60

function makeSim({ vmd = false } = {}) {
  const model = PmxLoader.loadFromBuffer(toAB(pmxBuf))
  const physics = new RezePhysics(model.getRigidbodies(), model.getJoints())
  if (vmd) {
    const frames = VMDLoader.loadFromBuffer(toAB(readFileSync(VMD)))
    model.loadClip("clip", model.buildClipFromVmdKeyFrames(frames))
    model.play("clip", { loop: true })
  }
  const step = () => {
    model.update(DT)
    physics.step(DT, model.getWorldMatrices(), model.getBoneInverseBindMatrices())
  }
  return { model, physics, store: physics.store, step }
}

const allFinite = (arr) => {
  for (let i = 0; i < arr.length; i++) if (!Number.isFinite(arr[i])) return false
  return true
}

const maxSpeed = (store) => {
  const v = store.linearVelocities
  let m = 0
  for (let i = 0; i < store.count; i++) {
    if (store.type[i] !== 1) continue
    const s = Math.hypot(v[i * 3], v[i * 3 + 1], v[i * 3 + 2])
    if (s > m) m = s
  }
  return m
}

test("idle sim stays finite and settles", { skip: !hasAssets }, () => {
  const sim = makeSim()
  for (let i = 0; i < 30; i++) sim.step()
  const early = maxSpeed(sim.store)
  for (let i = 0; i < 570; i++) sim.step() // 10s total
  assert.ok(allFinite(sim.store.positions), "positions finite")
  assert.ok(allFinite(sim.store.orientations), "orientations finite")
  const late = maxSpeed(sim.store)
  assert.ok(late < Math.max(0.5, early * 0.5), `settling: early=${early.toFixed(3)} late=${late.toFixed(3)}`)
})

test("animated sim stays finite", { skip: !hasAssets }, () => {
  const sim = makeSim({ vmd: true })
  for (let i = 0; i < 600; i++) sim.step()
  assert.ok(allFinite(sim.store.positions), "positions finite")
  assert.ok(allFinite(sim.store.orientations), "orientations finite")
})

test("simulation is deterministic", { skip: !hasAssets }, () => {
  const a = makeSim({ vmd: true })
  const b = makeSim({ vmd: true })
  for (let i = 0; i < 240; i++) {
    a.step()
    b.step()
  }
  assert.deepEqual(Array.from(a.store.positions), Array.from(b.store.positions))
  assert.deepEqual(Array.from(a.store.orientations), Array.from(b.store.orientations))
})

test("kinematic bodies track their bones through a translation", { skip: !hasAssets }, () => {
  const sim = makeSim()
  for (let i = 0; i < 120; i++) sim.step()
  const store = sim.store
  const kin = []
  for (let i = 0; i < store.count; i++) if (store.type[i] !== 1 && store.boneIndex[i] >= 0) kin.push(i)
  assert.ok(kin.length > 0, "model has kinematic anchors")
  const before = kin.map((i) => store.positions[i * 3])
  // Persistent +8 X shift applied to every bone world matrix before each step,
  // small enough per-frame logic treats it as continuous after the first jump.
  for (let f = 0; f < 90; f++) {
    sim.model.update(DT)
    const mats = sim.model.getWorldMatrices()
    for (const m of mats) m.values[12] += 8
    sim.physics.step(DT, mats, sim.model.getBoneInverseBindMatrices())
  }
  for (let k = 0; k < kin.length; k++) {
    const dx = sim.store.positions[kin[k] * 3] - before[k]
    assert.ok(Math.abs(dx - 8) < 0.5, `anchor ${kin[k]} moved ${dx.toFixed(3)}, expected ~8`)
  }
  assert.ok(allFinite(sim.store.positions), "positions finite after shift")
})

test("teleport (large jump) recovers without exploding", { skip: !hasAssets }, () => {
  const sim = makeSim()
  for (let i = 0; i < 120; i++) sim.step()
  const jumps = sim.physics.teleportCount ?? 0
  for (let f = 0; f < 120; f++) {
    sim.model.update(DT)
    const mats = sim.model.getWorldMatrices()
    for (const m of mats) m.values[12] += 500 // far beyond continuous motion
    sim.physics.step(DT, mats, sim.model.getBoneInverseBindMatrices())
  }
  assert.ok((sim.physics.teleportCount ?? 0) > jumps, "teleport path taken")
  assert.ok(allFinite(sim.store.positions), "positions finite")
  assert.ok(maxSpeed(sim.store) < 50, `no runaway velocities: ${maxSpeed(sim.store).toFixed(2)}`)
})

test("the built-in floor keeps sunken cloth above y=0", { skip: !hasAssets }, () => {
  const sim = makeSim()
  for (let i = 0; i < 120; i++) sim.step()
  // Sink the skeleton 6 units — a sitting-height pose: the skirt and hair
  // would hang well below y=0 without the floor; the ground box catches them.
  // (Sinking further drags KINEMATIC anchors under the floor and jointed
  // cloth legitimately follows them — that's not a floor failure.)
  for (let f = 0; f < 300; f++) {
    sim.model.update(DT)
    const mats = sim.model.getWorldMatrices()
    for (const m of mats) m.values[13] -= 6
    sim.physics.step(DT, mats, sim.model.getBoneInverseBindMatrices())
  }
  assert.ok(allFinite(sim.store.positions), "positions finite")
  let minY = Infinity
  for (let i = 0; i < sim.store.count; i++) {
    if (sim.store.type[i] !== 1) continue
    const y = sim.store.positions[i * 3 + 1]
    if (y < minY) minY = y
  }
  // Sphere/capsule centers sit a radius above the face; small transient
  // penetration is fine — sinking metres below is not.
  assert.ok(minY > -1.0, `dynamic bodies rest near the floor: minY=${minY.toFixed(2)}`)
})

test("broadphase pairs match the brute-force filtered sweep", { skip: !hasAssets }, () => {
  const sim = makeSim()
  for (let i = 0; i < 120; i++) sim.step()
  const store = sim.store
  store.updateAabbs()
  // Ground truth: prebuilt filtered pair list + AABB test (the pre-SAP pipeline).
  const truth = new Set()
  const pairs = store.getCollisionPairs()
  for (let p = 0; p < pairs.length; p += 2) {
    if (aabbOverlap(store, pairs[p], pairs[p + 1])) truth.add(pairs[p] * 65536 + pairs[p + 1])
  }
  // Whatever pair source findContacts uses must produce the same set. Until a
  // SAP lands this trivially passes; afterwards it is the equivalence proof.
  const candidate = new Set()
  const src = store.sweepPairs ? store.sweepPairs() : pairs
  for (let p = 0; p < src.length; p += 2) {
    const i = Math.min(src[p], src[p + 1])
    const j = Math.max(src[p], src[p + 1])
    if (aabbOverlap(store, i, j)) candidate.add(i * 65536 + j)
  }
  assert.equal(candidate.size, truth.size)
  for (const key of truth) assert.ok(candidate.has(key), `missing pair ${Math.floor(key / 65536)},${key % 65536}`)
})
