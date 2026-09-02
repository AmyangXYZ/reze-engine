// The product promise, executable: a PMX that goes through the document and back
// out is the same file. Everything the editor does sits on top of this, so if it
// fails nothing above it can be trusted. Run: npm test.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync } from "node:fs"
import { readPmxDocument, writePmxDocument } from "../dist/pmx-document.js"

const FIXTURE = new URL("./fixtures/reze.pmx", import.meta.url)

function load() {
  const buf = readFileSync(FIXTURE)
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
}

test("a PMX round-trips byte for byte", { skip: !existsSync(FIXTURE) && "no fixture" }, () => {
  const original = new Uint8Array(load())
  const out = new Uint8Array(writePmxDocument(readPmxDocument(load())))

  assert.equal(out.length, original.length, "the written file is a different size")
  // Report WHERE, not just that it differs — a byte offset is the only useful
  // thing to know when a section's field order has drifted.
  for (let i = 0; i < original.length; i++) {
    if (out[i] !== original[i]) {
      assert.fail(`byte ${i} of ${original.length} differs: wrote 0x${out[i].toString(16)}, expected 0x${original[i].toString(16)}`)
    }
  }
})

test("the document carries what the renderer drops", { skip: !existsSync(FIXTURE) && "no fixture" }, () => {
  const doc = readPmxDocument(load())
  // Each of these is something Model does not keep, and each is something a save
  // would silently delete if the writer were fed from Model instead.
  assert.ok(doc.displayFrames.length > 0, "no display frames — MMD groups its sliders by these")
  assert.ok(doc.morphs.some((m) => m.panel > 0), "no morph panel assignments")
  assert.ok(doc.bones.some((b) => b.nameEn.length > 0), "no English bone names")
  assert.ok(doc.materials.some((m) => m.memo.length > 0 || m.nameEn.length > 0), "no material memos or English names")
  assert.equal(doc.indices.length % 3, 0, "the index buffer is not whole faces")
  assert.equal(
    doc.materials.reduce((n, m) => n + m.indexCount, 0),
    doc.indices.length,
    "material index runs do not tile the index buffer",
  )
})

test("an edit changes what it names and nothing else", { skip: !existsSync(FIXTURE) && "no fixture" }, () => {
  const before = readPmxDocument(load())
  const target = before.bones.findIndex((b) => b.name === "左足")
  assert.ok(target >= 0, "fixture has no 左足 to rename")

  const edited = readPmxDocument(load())
  edited.bones[target].name = "left leg"
  const after = readPmxDocument(writePmxDocument(edited))

  assert.equal(after.bones[target].name, "left leg", "the rename did not survive")
  assert.equal(after.bones[target].nameEn, before.bones[target].nameEn, "the English name went with it")

  // Everything a save has historically thrown away, still here and still equal.
  assert.equal(after.displayFrames.length, before.displayFrames.length)
  assert.deepEqual(
    after.displayFrames.map((f) => [f.name, f.special, f.elements.length]),
    before.displayFrames.map((f) => [f.name, f.special, f.elements.length]),
    "display frames changed — MMD groups its sliders by these",
  )
  assert.deepEqual(
    after.morphs.map((m) => [m.name, m.panel, m.type, m.offsets.length]),
    before.morphs.map((m) => [m.name, m.panel, m.type, m.offsets.length]),
    "morph panels or offsets changed",
  )
  assert.equal(after.vertices.length, before.vertices.length)
  assert.equal(after.indices.length, before.indices.length)
  assert.deepEqual(
    after.materials.map((m) => m.indexCount),
    before.materials.map((m) => m.indexCount),
    "material index runs moved — face order is document identity",
  )
  // Vertex indices are identity too: every vertex morph must still point at the
  // vertex it pointed at.
  const vertexMorph = before.morphs.find((m) => m.type === 1 && m.offsets.length > 0)
  if (vertexMorph) {
    const i = before.morphs.indexOf(vertexMorph)
    assert.deepEqual(
      after.morphs[i].offsets.map((o) => o.index),
      vertexMorph.offsets.map((o) => o.index),
      "a vertex morph's targets moved",
    )
  }
})
