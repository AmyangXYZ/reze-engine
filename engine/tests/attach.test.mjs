// Attachment (外部親): a model hung from another model's bone. Run: npm test.
//
// Two halves. The Model half is exercised for real — a rig built by hand,
// posed under a root parent, its bones read back. The Engine half needs a GPU
// to construct, so its contract is checked against the SOURCE: the pose loop
// runs parents first and places an attached model before posing it, the
// transform API holds an attached model's own placement at identity, and a
// removed parent lets its children go.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const { Model } = await import("../dist/model.js")
const { Mat4, Quat, Vec3 } = await import("../dist/math.js")
const engine = readFileSync(join(here, "../src/engine.ts"), "utf8")

/** A rig of `bones` ({ name, parent, at }) over one quad, the way addPlane
 *  builds a card — Model refuses an empty skeleton, and it never poses more
 *  than the bones it is given. */
function rig(bones) {
  const vertexData = new Float32Array([
    -1, -1, 0, 0, 0, 1, 0, 1,
     1, -1, 0, 0, 0, 1, 1, 1,
     1,  1, 0, 0, 0, 1, 1, 0,
    -1,  1, 0, 0, 0, 1, 0, 0,
  ])
  const indexData = new Uint32Array([0, 1, 2, 0, 2, 3])
  const material = {
    name: "m", diffuse: [1, 1, 1, 1], specular: [0, 0, 0], ambient: [0, 0, 0], shininess: 0,
    diffuseTextureIndex: -1, normalTextureIndex: -1, sphereTextureIndex: -1, sphereMode: 0,
    toonTextureIndex: -1, sharedToon: false, edgeFlag: 0, edgeColor: [0, 0, 0, 1], edgeSize: 0,
    vertexCount: 6,
  }
  const inverseBindMatrices = new Float32Array(bones.length * 16)
  const list = bones.map((b, i) => {
    inverseBindMatrices.set([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -b.at[0], -b.at[1], -b.at[2], 1], i * 16)
    // As the loader stores it: a root's bind translation is absolute, a child's
    // is relative to its parent.
    const at = b.parent >= 0 ? b.at.map((v, k) => v - bones[b.parent].at[k]) : b.at
    return { name: b.name, parentIndex: b.parent, bindTranslation: at, children: [] }
  })
  const joints = new Uint16Array(16)
  const weights = new Uint8Array(16)
  for (let i = 0; i < 4; i++) weights[i * 4] = 255
  return new Model(vertexData, indexData, [], [material], { bones: list, inverseBindMatrices }, { joints, weights }, { morphs: [] })
}

const pos = (model, bone) => {
  const p = model.getBoneWorldPosition(bone)
  return [p.x, p.y, p.z].map((v) => Math.round(v * 1e4) / 1e4)
}

test("a root parent moves every bone of the rig, root and descendants alike", () => {
  const m = rig([
    { name: "全ての親", parent: -1, at: [0, 0, 0] },
    { name: "右手首", parent: 0, at: [1, 2, 0] },
  ])
  m.update(0)
  assert.deepEqual(pos(m, "全ての親"), [0, 0, 0])
  assert.deepEqual(pos(m, "右手首"), [1, 2, 0])

  const parent = new Float32Array(16)
  Mat4.fromPositionRotationScaleInto(10, 5, -3, 0, 0, 0, 1, 1, parent)
  m.setRootParent(parent)
  m.update(0)
  assert.deepEqual(pos(m, "全ての親"), [10, 5, -3])
  assert.deepEqual(pos(m, "右手首"), [11, 7, -3])

  // Read by reference: the engine refills the same array every frame.
  Mat4.fromPositionRotationScaleInto(0, 0, 0, 0, 0, 0, 1, 1, parent)
  m.update(0)
  assert.deepEqual(pos(m, "右手首"), [1, 2, 0])

  m.setRootParent(null)
  m.update(0)
  assert.deepEqual(pos(m, "右手首"), [1, 2, 0])
})

test("a rotated root parent turns the whole rig about the parent's origin", () => {
  const m = rig([
    { name: "全ての親", parent: -1, at: [0, 0, 0] },
    { name: "先", parent: 0, at: [1, 0, 0] },
  ])
  // A quarter turn about Y: +X goes to -Z.
  const q = Quat.fromEuler(0, Math.PI / 2, 0)
  const parent = new Float32Array(16)
  Mat4.fromPositionRotationScaleInto(0, 3, 0, q.x, q.y, q.z, q.w, 1, parent)
  m.setRootParent(parent)
  m.update(0)
  assert.deepEqual(pos(m, "全ての親"), [0, 3, 0])
  assert.deepEqual(pos(m, "先"), [0, 3, -1])
})

test("the primary root sits ON the parent bone; other roots keep their layout", () => {
  // MMD's 外部親 puts the bound bone at the parent, not the model origin. A prop
  // rigged with its one bone at the mesh's centre (y=8 here) must land in the
  // hand, not eight units above it — which is what composing the bind position
  // did. Some rigs carry a second root; it is moved as part of the whole and
  // keeps its offset from the primary.
  const m = rig([
    { name: "本体", parent: -1, at: [0, 8, 0] },
    { name: "浮遊", parent: -1, at: [0, 13, 0] },
    { name: "先", parent: 0, at: [1, 8, 0] },
  ])
  const parent = new Float32Array(16)
  Mat4.fromPositionRotationScaleInto(2, 0, 0, 0, 0, 0, 1, 1, parent)
  m.setRootParent(parent)
  m.update(0)
  assert.deepEqual(pos(m, "本体"), [2, 0, 0])
  assert.deepEqual(pos(m, "浮遊"), [2, 5, 0])
  assert.deepEqual(pos(m, "先"), [3, 0, 0])
  m.setRootParent(null)
  m.update(0)
  assert.deepEqual(pos(m, "本体"), [0, 8, 0])
})

test("getRootMatrix is the placement the skin bake composes", () => {
  const m = rig([{ name: "全ての親", parent: -1, at: [0, 0, 0] }])
  m.setPosition(new Vec3(1, 2, 3))
  m.setScale(2)
  const r = m.getRootMatrix()
  assert.deepEqual([r[0], r[5], r[10]], [2, 2, 2])
  assert.deepEqual([r[12], r[13], r[14]], [1, 2, 3])
  assert.equal(m.getBoneWorldMatrix("全ての親").length, 16)
  assert.equal(m.getBoneWorldMatrix("ない"), null)
})

test("the pose loop runs parents first and places a child before posing it", () => {
  const loop = engine.slice(engine.indexOf("  private updateInstances("), engine.indexOf("  private updateVertexBuffer("))
  assert.match(loop, /for \(const inst of this\.instancesInUpdateOrder\(\)\)/, "the loop must walk the parents-first order")
  const place = loop.indexOf("this.placeAttached(inst)")
  const pose = loop.indexOf("inst.model.update(deltaTime")
  assert.ok(place > 0 && place < pose, "an attached model is placed BEFORE its pose pass")
  assert.match(loop, /&& !attached && inst\.model\.isIdle\(\)/, "an attached prop never takes the idle skip")
  assert.match(loop, /inst\.isStage \|\| inst\.isPlane \? false : this\.ikEnabled/, "a prop keeps IK — its own clip may drive chains")
})

test("a prop keeps physics and outlines, and is not a performer", () => {
  assert.match(engine, /!isStage && !isPlane && rbs\.length > 0/, "physics builds for a prop")
  assert.match(engine, /if \(!inst\.isStage && \(mat\.edgeFlag & 0x10\) !== 0/, "a prop keeps its outline")
  assert.match(engine, /if \(n >= MAX_EFFECT_SUBJECTS \|\| inst\.isStage \|\| inst\.isPlane \|\| inst\.isProp\) return/, "a prop is not a subject")
  assert.match(engine, /if \(inst\.isStage \|\| inst\.isPlane \|\| inst\.isProp\) continue\n      const p = inst\.model\.getAnimationProgress/, "a prop never seeds the clock")
  const hasStage = engine.slice(engine.indexOf("  hasStage(): boolean {"), engine.indexOf("  groundIsSuppressed()"))
  assert.doesNotMatch(hasStage, /isProp/, "a prop leaves the floor alone")
})

test("an attached model's own placement is held at identity", () => {
  const set = engine.slice(engine.indexOf("  setModelTransform(name: string"), engine.indexOf("  getModelTransform(name: string"))
  assert.match(set, /if \(transform\.position && !inst\.parent\)/)
  assert.match(set, /if \(transform\.rotation && !inst\.parent\)/)
  assert.match(set, /if \(transform\.scale !== undefined\) model\.setScale/, "scale is still the model's own")
  const attach = engine.slice(engine.indexOf("  setModelParent("), engine.indexOf("  getModelParent("))
  assert.match(attach, /inst\.model\.setPosition\(new Vec3\(0, 0, 0\)\)/)
  assert.match(attach, /inst\.model\.setRotation\(Quat\.identity\(\)\)/)
  assert.match(attach, /if \(parent === name \|\| !this\.modelInstances\.has\(parent\)\) return false/)
})

test("the child's scale divides the translation and nothing else", () => {
  // world = S · P · bones, and the wanted world is A · Off · S · bones. A
  // uniform S commutes with a rotation, so P = S⁻¹ · A · Off · S is A · Off
  // with its translation over s. Dividing the rotation too would cancel the
  // scale the skin bake puts back — the prop would land right and come out
  // unscaled.
  const place = engine.slice(engine.indexOf("  private placeAttached("), engine.indexOf("  private readonly attachScratch"))
  assert.match(place, /out\[12\] \*= k\n\s+out\[13\] \*= k\n\s+out\[14\] \*= k/)
  assert.doesNotMatch(place, /out\[0\] \*= k/)
})

test("removing a parent detaches its children", () => {
  const remove = engine.slice(engine.indexOf("  removeModel(name: string): void {"), engine.indexOf("  getModelNames(): string[] {"))
  assert.match(remove, /if \(other\.parent\?\.model === name\) this\.setModelParent\(other\.name, null\)/)
})
