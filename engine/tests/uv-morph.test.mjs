// UV morphs (PMX type 3) move the base UV, the way a vertex morph moves the
// position: a texture that scrolls or spins as a morph is keyed, a pattern
// swapped by sliding to another part of the sheet. They were parsed and never
// applied, so every such morph did nothing.
//
// Both paths apply them: the GPU morph pass carries five offsets an entry (xyz
// uv) and writes the UV beside the position; the CPU path adds them into the
// same touched range.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const { PmxLoader } = await import("../dist/pmx-loader.js")
const { MORPH_COMPUTE_WGSL } = await import("../dist/shaders/passes/morph.js")

test("the morph pass reads five offsets an entry and writes the UV beside the position", () => {
  assert.match(MORPH_COMPUTE_WGSL, /let o5 = i \* 5u;/)
  assert.match(MORPH_COMPUTE_WGSL, /vtx\[vo \+ 6u\] = tu;/)
  assert.match(MORPH_COMPUTE_WGSL, /vtx\[vo \+ 7u\] = tv;/)
  // The normal is the pass's to leave alone.
  assert.doesNotMatch(MORPH_COMPUTE_WGSL, /vtx\[vo \+ [345]u\]/)
})

test("a vertex-position write lands in the compute base at five floats a vertex", () => {
  const engine = readFileSync(join(here, "../src/engine.ts"), "utf8")
  assert.match(engine, /writeBuffer\(inst\.gpuMorph\.baseBuf, index \* 20, scratch\)/)
})

/** Whatever models this machine happens to have, as the other model suites. */
const findModels = () => {
  const roots = [
    join(here, "../../web/public/models"),
    join(here, "../../../MiKaPo/public/models"),
    join(here, "../../../reze-studio/public/models"),
    join(here, "../../../reze-design/public/models"),
    // A folder of models to try besides these, for a run on a machine that has one.
    ...(process.env.REZE_TEST_MODELS ? [process.env.REZE_TEST_MODELS] : []),
  ].filter(existsSync)
  const out = []
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry)
      if (statSync(p).isDirectory()) walk(p)
      else if (p.toLowerCase().endsWith(".pmx")) out.push(p)
    }
  }
  roots.forEach(walk)
  return out
}

const load = (path) => {
  const buf = readFileSync(path)
  return PmxLoader.loadFromBuffer(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength))
}

const withUv = findModels()
  .map((path) => ({ path, model: load(path) }))
  .filter(({ model }) => model.getMorphing().morphs.some((m) => m.type === 3 && m.uvOffsets?.length))

test("a UV morph's offsets reach the compute data, and the CPU path moves the UV by them", { skip: withUv.length === 0 && "no model with UV morphs here" }, () => {
  for (const { path, model } of withUv) {
    const data = model.buildMorphComputeData()
    assert.ok(data, `${path}: has morph data`)
    let uv = 0
    for (let i = 0; i < data.entryCount; i++) if (data.colOffset[i * 5 + 3] !== 0 || data.colOffset[i * 5 + 4] !== 0) uv++
    assert.ok(uv > 0, `${path}: UV entries in the compute data`)
    assert.equal(data.base.length, data.vertexCount * 5)

    const morphs = model.getMorphing().morphs
    const idx = morphs.findIndex((m) => m.type === 3 && m.uvOffsets.some((o) => Math.abs(o.offset[0]) + Math.abs(o.offset[1]) > 1e-3))
    const off = morphs[idx].uvOffsets.find((o) => Math.abs(o.offset[0]) + Math.abs(o.offset[1]) > 1e-3)
    const at = off.vertexIndex * 8 + 6
    const before = [...model.getVertices().slice(at, at + 2)]
    model.setMorphWeight(morphs[idx].name, 0.5)
    const after = [...model.getVertices().slice(at, at + 2)]
    // Other UV morphs may share the vertex; this one alone is weighted.
    assert.ok(Math.abs(after[0] - (before[0] + off.offset[0] * 0.5)) < 1e-3, `${path}: u moved by half the offset`)
    assert.ok(Math.abs(after[1] - (before[1] + off.offset[1] * 0.5)) < 1e-3, `${path}: v moved by half the offset`)
  }
})
