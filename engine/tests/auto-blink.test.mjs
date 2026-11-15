// Auto blink: she blinks on her own every few seconds, on her closing morph.
//
// The weights are float32. The blink tells the motion's fresh writes from its
// own by comparing against what it last wrote, and a float64 kept there never
// matched, so its own value passed for the motion's: the eyes stuck half shut
// after the first blink.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const { PmxLoader } = await import("../dist/pmx-loader.js")

const findModels = () => {
  const roots = [
    join(here, "../../web/public/models"),
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

const RIG = findModels().find((path) => {
  try {
    const names = PmxLoader.loadFromBuffer(toAB(readFileSync(path))).getMorphing().morphs.map((m) => m.name)
    return names.includes("まばたき")
  } catch {
    return false
  }
})
const SKIP = { skip: RIG ? false : "no model with まばたき on this machine" }

/** Step 20 s at 60 fps, the まばたき weight each frame. */
const run = (model) => {
  const idx = model.getMorphing().morphs.findIndex((m) => m.name === "まばたき")
  const out = []
  for (let i = 0; i < 1200; i++) {
    model.update(1 / 60, false)
    out.push(model.getMorphWeights()[idx])
  }
  return out
}

test("she blinks several times in 20 s, shutting fully and opening again", SKIP, () => {
  const model = PmxLoader.loadFromBuffer(toAB(readFileSync(RIG)))
  assert.equal(model.setAutoBlink({}), true)
  const w = run(model)
  let blinks = 0
  for (let i = 1; i < w.length; i++) if (w[i] >= 0.99 && w[i - 1] < 0.99) blinks++
  assert.ok(blinks >= 3, `blinks: ${blinks}`)
  // Open far more than shut.
  const open = w.filter((x) => x < 0.05).length
  assert.ok(open > w.length * 0.9, `open frames: ${open}/${w.length}`)
  assert.equal(w[w.length - 1] < 1, true)
})

test("off hands the morph back to what it was", SKIP, () => {
  const model = PmxLoader.loadFromBuffer(toAB(readFileSync(RIG)))
  model.setAutoBlink({})
  run(model)
  model.setAutoBlink(null)
  model.update(1 / 60, false)
  const idx = model.getMorphing().morphs.findIndex((m) => m.name === "まばたき")
  assert.equal(model.getMorphWeights()[idx], 0)
})

test("it owns the blink morph over a motion's own", SKIP, () => {
  const model = PmxLoader.loadFromBuffer(toAB(readFileSync(RIG)))
  model.setAutoBlink({})
  // A host or morph motion holding the eyes shut is overridden, and blinks still run.
  const idx = model.getMorphing().morphs.findIndex((m) => m.name === "まばたき")
  const w = []
  for (let i = 0; i < 1200; i++) {
    model.getMorphWeights()[idx] = 1
    model.update(1 / 60, false)
    w.push(model.getMorphWeights()[idx])
  }
  assert.ok(w.filter((x) => x === 0).length > 1000, "open between blinks")
  assert.ok(w.some((x) => x >= 0.99), "still blinks")
})

test("a shorter interval blinks more often", SKIP, () => {
  const count = (interval) => {
    const model = PmxLoader.loadFromBuffer(toAB(readFileSync(RIG)))
    model.setAutoBlink({ interval, double: false })
    const w = run(model)
    let n = 0
    for (let i = 1; i < w.length; i++) if (w[i] >= 0.99 && w[i - 1] < 0.99) n++
    return n
  }
  assert.ok(count(1.5) > count(8), `${count(1.5)} vs ${count(8)}`)
})

// A double is a flutter: two quick blinks with open eyes between, close enough
// to read as one gesture. 0.6 s apart at full length they read as two blinks,
// and at 15% one came every half minute, so nobody saw one.
test("doubles come often enough, quick and close", SKIP, () => {
  const model = PmxLoader.loadFromBuffer(toAB(readFileSync(RIG)))
  model.setAutoBlink({ interval: 4, double: true })
  const w = run(model)
  const shut = []
  for (let i = 1; i < w.length; i++) if (w[i] >= 0.99 && w[i - 1] < 0.99) shut.push(i)
  const pairs = shut.filter((f, k) => k > 0 && f - shut[k - 1] <= 20)
  assert.ok(pairs.length >= 1, `a double within 20 s (${shut.join(",")})`)
  for (const f of pairs) {
    const between = w.slice(f - 18, f - 3)
    assert.ok(between.some((x) => x < 0.05), "open between the two")
  }
})
