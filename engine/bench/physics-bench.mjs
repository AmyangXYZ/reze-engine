// Headless physics benchmark: load a PMX, build its physics world, step it for
// N simulated seconds, report per-step cost. The workload mirrors the engine's
// frame loop exactly — model.update() then physics.step() with the same
// arguments — so numbers here predict cpuPhysicsMs in the browser.
//
//   node bench/physics-bench.mjs <model.pmx> [--seconds 10] [--vmd <clip.vmd>] [--models 1]
//
// With --vmd the clip loops during the run (kinematic bodies move: the running
// workload). Without it the model holds bind pose (the idle-page workload).
// --models N simulates N independent copies (the 3-character demo scene).

import { readFileSync } from "node:fs"
import { basename } from "node:path"
import { PmxLoader, VMDLoader } from "../dist/index.js"

const args = process.argv.slice(2)
const modelPath = args.find((a) => !a.startsWith("--"))
if (!modelPath) {
  console.error("usage: node bench/physics-bench.mjs <model.pmx> [--seconds 10] [--vmd <clip.vmd>] [--models 1]")
  process.exit(1)
}
const flag = (name, dflt) => {
  const i = args.indexOf(`--${name}`)
  return i >= 0 ? args[i + 1] : dflt
}
const seconds = Number(flag("seconds", "10"))
const vmdPath = flag("vmd", null)
const nModels = Number(flag("models", "1"))

const toArrayBuffer = (buf) => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)

// RezePhysics is not exported from the package root (browser apps never build
// one directly) — reach into dist for the bench.
const { RezePhysics } = await import("../dist/physics/physics.js")

const pmxBuf = readFileSync(modelPath)
const vmdFrames = vmdPath ? VMDLoader.loadFromBuffer(toArrayBuffer(readFileSync(vmdPath))) : null

const sims = []
for (let m = 0; m < nModels; m++) {
  const model = PmxLoader.loadFromBuffer(toArrayBuffer(pmxBuf))
  const physics = new RezePhysics(model.getRigidbodies(), model.getJoints())
  if (vmdFrames) {
    const clip = model.buildClipFromVmdKeyFrames(vmdFrames)
    model.loadClip("bench", clip)
    model.play("bench", { loop: true })
  }
  sims.push({ model, physics })
}

const rbs = sims[0].model.getRigidbodies()
const dynamic = rbs.filter((r) => r.type !== 0).length
console.log(`model: ${basename(modelPath)} ×${nModels}`)
console.log(`bodies: ${rbs.length} (${dynamic} dynamic), joints: ${sims[0].model.getJoints().length}`)
console.log(`workload: ${vmdFrames ? `animated (${basename(vmdPath)})` : "bind pose (idle)"}, ${seconds}s simulated @60fps`)

const DT = 1 / 60
const steps = Math.round(seconds / DT)
const WARMUP = 60

const physMs = new Float64Array(steps)
const animMs = new Float64Array(steps)
for (let i = -WARMUP; i < steps; i++) {
  let pTotal = 0
  let aTotal = 0
  for (const s of sims) {
    const t0 = performance.now()
    s.model.update(DT)
    const t1 = performance.now()
    s.physics.step(DT, s.model.getWorldMatrices(), s.model.getBoneInverseBindMatrices())
    const t2 = performance.now()
    aTotal += t1 - t0
    pTotal += t2 - t1
  }
  if (i >= 0) {
    physMs[i] = pTotal
    animMs[i] = aTotal
  }
}

const stats = (arr) => {
  const sorted = Float64Array.from(arr).sort()
  const sum = arr.reduce((a, b) => a + b, 0)
  return {
    mean: sum / arr.length,
    p50: sorted[Math.floor(arr.length * 0.5)],
    p95: sorted[Math.floor(arr.length * 0.95)],
    max: sorted[arr.length - 1],
  }
}
const fmt = (s) => `mean ${s.mean.toFixed(2)}  p50 ${s.p50.toFixed(2)}  p95 ${s.p95.toFixed(2)}  max ${s.max.toFixed(2)}`
console.log(`physics ms/frame: ${fmt(stats(physMs))}`)
console.log(`anim    ms/frame: ${fmt(stats(animMs))}`)

// NaN guard: a benchmark on an exploded sim measures garbage.
const pos = sims[0].physics.store?.positions
if (pos && Array.from(pos.slice(0, 30)).some((v) => !Number.isFinite(v))) {
  console.error("WARNING: non-finite body positions after run — sim exploded, numbers invalid")
  process.exit(2)
}
