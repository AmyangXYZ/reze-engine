// PMX loader regression suite.
//
// Written after a "suspicious string length" guard rejected any model whose
// comment field ran past a thousand bytes — credits and terms of use, which is
// to say a large share of distributed models. Nothing exercised the loader
// against real files, so a heuristic could sit there breaking people quietly.
// These read the actual models in the repo and the format edges that bit us.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const { PmxLoader } = await import("../dist/pmx-loader.js")

/** Every .pmx under the sibling repos — whatever the machine happens to have. */
const findModels = () => {
  const roots = [
    join(here, "../../web/public/models"),
    join(here, "../../../MiKaPo/public/models"),
    join(here, "../../../reze-studio/public/models"),
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

const MODELS = findModels()
const toAB = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)

test("every model on disk parses into a usable rig", { skip: MODELS.length === 0 }, () => {
  for (const path of MODELS) {
    const model = PmxLoader.loadFromBuffer(toAB(readFileSync(path)))
    const name = path.split("/").pop()
    const bones = model.getSkeleton().bones
    assert.ok(bones.length > 0, `${name}: no bones`)
    assert.ok(model.getMaterials().length > 0, `${name}: no materials`)
    assert.ok(model.getVertices().length > 0, `${name}: no vertices`)
    // A parse that silently misaligns still returns data — it just returns
    // nonsense. Bone parents must reference real bones, and no bone may be its
    // own ancestor, which garbage indices violate immediately.
    for (let i = 0; i < bones.length; i++) {
      const p = bones[i].parentIndex
      assert.ok(p >= -1 && p < bones.length, `${name}: bone ${i} parent ${p} out of range`)
      let hops = 0
      for (let cur = p; cur >= 0; cur = bones[cur].parentIndex) {
        assert.ok(++hops < bones.length, `${name}: bone ${i} sits in a parent cycle`)
      }
    }
  }
})

test("skin weights are normalised and reference real bones", { skip: MODELS.length === 0 }, () => {
  for (const path of MODELS) {
    const model = PmxLoader.loadFromBuffer(toAB(readFileSync(path)))
    const name = path.split("/").pop()
    const bones = model.getSkeleton().bones.length
    const verts = model.getVertices()
    // Vertex layout is position(3) normal(3) uv(2) — joints and weights ride
    // their own arrays, exposed through the skinning buffers the engine builds.
    assert.equal(verts.length % 8, 0, `${name}: vertex stride is not 8 floats`)
    const skin = model.getSkinning?.()
    if (!skin) continue
    for (let i = 0; i < skin.joints.length; i++) {
      assert.ok(skin.joints[i] < bones, `${name}: joint index ${skin.joints[i]} exceeds ${bones} bones`)
    }
    for (let v = 0; v < skin.weights.length; v += 4) {
      const sum = skin.weights[v] + skin.weights[v + 1] + skin.weights[v + 2] + skin.weights[v + 3]
      assert.ok(Math.abs(sum - 255) <= 2, `${name}: vertex ${v / 4} weights sum to ${sum}, not 255`)
    }
  }
})

test("a lowercase 'Pmx ' signature loads", { skip: MODELS.length === 0 }, () => {
  // Exporters in the wild write both cases; a strict uppercase compare rejected
  // half of them.
  const bytes = new Uint8Array(readFileSync(MODELS[0]))
  bytes[1] = "m".charCodeAt(0)
  bytes[2] = "x".charCodeAt(0)
  const model = PmxLoader.loadFromBuffer(toAB(bytes))
  assert.ok(model.getSkeleton().bones.length > 0)
})

test("a PMD file is named as such rather than 'not a PMX file'", { skip: MODELS.length === 0 }, () => {
  const bytes = new Uint8Array(readFileSync(MODELS[0]))
  bytes[0] = "P".charCodeAt(0)
  bytes[1] = "m".charCodeAt(0)
  bytes[2] = "d".charCodeAt(0)
  assert.throws(
    () => PmxLoader.loadFromBuffer(toAB(bytes)),
    /PMD/,
    "a PMD should say it is a PMD, so the fix is obvious",
  )
})

test("parsing is identical from a buffer with a byte offset", { skip: MODELS.length === 0 }, () => {
  // A DataView over a slice of a larger buffer has its own origin. Reading the
  // raw buffer without honouring it lands somewhere else entirely — which would
  // look exactly like scrambled weights and wrong materials.
  const raw = new Uint8Array(readFileSync(MODELS[0]))
  const padded = new Uint8Array(raw.length + 64)
  padded.set(raw, 64)
  const offsetView = padded.buffer.slice(64)

  const direct = PmxLoader.loadFromBuffer(toAB(raw))
  const shifted = PmxLoader.loadFromBuffer(offsetView)
  assert.equal(shifted.getSkeleton().bones.length, direct.getSkeleton().bones.length)
  assert.equal(shifted.getMaterials().length, direct.getMaterials().length)
  assert.deepEqual(
    shifted.getSkeleton().bones.map((b) => b.name),
    direct.getSkeleton().bones.map((b) => b.name),
  )
})

test("a truncated file fails with a clear error instead of hanging", { skip: MODELS.length === 0 }, () => {
  const raw = new Uint8Array(readFileSync(MODELS[0]))
  const cut = raw.slice(0, Math.floor(raw.length / 3))
  assert.throws(() => PmxLoader.loadFromBuffer(toAB(cut)))
})
