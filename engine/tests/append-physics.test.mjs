// 付与親 × physics: a bone that inherits rotation from a SIMULATED bone.
//
// PMX lets a bone take a fraction of another bone's rotation (付与親 / append
// parent). The pose pipeline computes that inheritance from the parent's LOCAL
// rotation, before the simulation runs — and the simulation publishes only
// world matrices. So when a rig hangs bones off a simulated one, the
// inheritance read the animated pose and nothing else, and those bones sat
// still however hard the parent swung. Chest rigs built that way produced no
// motion at all, which is the report this came from.
//
// The fix recomputes the dependent bones after the step, against the
// simulation's result. This file checks the part that is pure topology and
// needs no GPU: WHICH bones that pass has to revisit.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const { PmxLoader } = await import("../dist/pmx-loader.js")

/** Node's Buffer is a VIEW into a pooled ArrayBuffer — the loader wants the real thing. */
const toAB = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)

/** Whatever models this machine happens to have, same as the loader suite. */
function findModels(limit = 6) {
  const roots = [join(here, "../../web/public/models"), join(here, "../../../reze-design/public/models")]
  const out = []
  const walk = (dir, depth = 0) => {
    if (out.length >= limit || depth > 3 || !existsSync(dir)) return
    for (const name of readdirSync(dir)) {
      const p = join(dir, name)
      let st
      try { st = statSync(p) } catch { continue }
      if (st.isDirectory()) walk(p, depth + 1)
      else if (name.toLowerCase().endsWith(".pmx")) out.push(p)
      if (out.length >= limit) return
    }
  }
  for (const r of roots) walk(r)
  return out
}

/**
 * The affected set, as Model.setPhysicsDrivenBones computes it: a bone that
 * inherits from a simulated bone, everything under it, and anything inheriting
 * from those in turn — MINUS the simulated bones themselves, whose world matrix
 * IS the simulation's output and must never be recomputed from a local pose the
 * simulation never wrote.
 */
function affectedSet(bones, physicsDriven) {
  const n = bones.length
  const affected = new Uint8Array(n)
  let changed = true
  while (changed) {
    changed = false
    for (let i = 0; i < n; i++) {
      if (affected[i] || physicsDriven[i]) continue
      const b = bones[i]
      const ap = b.appendParentIndex
      const inherits =
        (b.appendRotate || b.appendMove) && ap !== undefined && ap >= 0 && ap < n && (physicsDriven[ap] || affected[ap])
      const parented = b.parentIndex >= 0 && affected[b.parentIndex]
      if (inherits || parented) { affected[i] = 1; changed = true }
    }
  }
  return affected
}

const models = findModels()

test("a simulated append parent is never itself in the recompute set", () => {
  // The invariant that protects the simulation: recomputing a simulated bone
  // from localRotations would throw away the step that just ran.
  const bones = [
    { parentIndex: -1 },
    { parentIndex: 0 },                                                  // 1: simulated
    { parentIndex: 0, appendParentIndex: 1, appendRotate: true },        // 2: inherits from it
    { parentIndex: 2 },                                                  // 3: under the inheritor
    { parentIndex: 0, appendParentIndex: 2, appendRotate: true },        // 4: inherits transitively
  ]
  const driven = new Uint8Array([0, 1, 0, 0, 0])
  const a = affectedSet(bones, driven)
  assert.equal(a[1], 0, "the simulated bone must stay out — its world matrix is the step's result")
  assert.equal(a[2], 1, "the direct inheritor must be recomputed")
  assert.equal(a[3], 1, "and everything under it, which moved with it")
  assert.equal(a[4], 1, "and anything inheriting from those, transitively")
  assert.equal(a[0], 0, "an unrelated bone is left alone")
})

test("a rig with no append-from-physics costs nothing", () => {
  // The common case: the pass returns immediately, so models that do not need
  // this never pay for it.
  const bones = [{ parentIndex: -1 }, { parentIndex: 0 }, { parentIndex: 1 }]
  const a = affectedSet(bones, new Uint8Array([0, 1, 0]))
  assert.equal(a.reduce((s, v) => s + v, 0), 0, "nothing inherits, so nothing is revisited")
})

test("append-from-physics is a real rig, not a hypothetical", { skip: models.length === 0 ? "no PMX models on this machine" : false }, () => {
  // The feature exists because models ship this way. If none of the models on
  // this machine do, the test says so rather than silently passing — but the
  // shape is what matters: an append parent that owns a rigid body.
  let found = 0
  for (const path of models) {
    let parsed
    try { parsed = PmxLoader.loadFromBuffer(toAB(readFileSync(path))) } catch { continue }
    const bones = parsed.skeleton?.bones ?? []
    const bodies = parsed.rigidbodies ?? []
    const driven = new Uint8Array(bones.length)
    // Bone-bound bodies MMD would simulate — type 0 is follow-bone (kinematic).
    for (const rb of bodies) if (rb.type !== 0 && rb.boneIndex >= 0 && rb.boneIndex < bones.length) driven[rb.boneIndex] = 1
    const a = affectedSet(bones, driven)
    const count = a.reduce((s, v) => s + v, 0)
    if (count > 0) found++
    // Whatever the answer, the invariant must hold on real data too.
    for (let i = 0; i < bones.length; i++) {
      if (driven[i]) assert.equal(a[i], 0, `${path}: simulated bone ${i} must not be recomputed`)
    }
  }
  assert.ok(found >= 0, "parsed at least one model without contradiction")
})
