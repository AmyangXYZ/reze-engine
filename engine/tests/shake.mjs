// Cloth-shake metric. Not a test — a measuring tool, run by hand:
//
//   node --import ./tests/register.mjs tests/shake.mjs <model.pmx> [more.pmx...]
//   CLIP=none node --import ./tests/register.mjs tests/shake.mjs <model.pmx>
//
// Jitter is OSCILLATION, not speed. Peak-velocity metrics mislead badly here —
// swinging cloth is fast and looks fine; shaking cloth can be slow and looks
// broken. Two measures, both per dynamic body:
//
//   flip  — fraction of frames on which linear velocity reverses direction.
//           Smooth swinging cloth almost never reverses; shaking cloth does
//           constantly. This is the one that tracks the eye.
//   jerk  — mean |Δv|/dt, the acceleration it takes to produce that motion.
//
// CLIP names a VMD in web/public/unity-fbx-locomotion/vmd (default Idle);
// CLIP=none runs the rig with no animation at all, which isolates the
// settling defect from the animation-driven one.

import { readFileSync, existsSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const VMD_DIR = join(here, "../../web/public/unity-fbx-locomotion/vmd")

const { PmxLoader } = await import("../dist/pmx-loader.js")
const { VMDLoader } = await import("../dist/vmd-loader.js")
const { RezePhysics } = await import("../dist/physics/physics.js")

const toAB = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)
const DT = 1 / 60

// 120 frames of settling are discarded so the metric measures the steady
// state, not the drop from the bind pose.
const WARMUP = 120
const FRAMES = 420

export function shake(path, clip = process.env.CLIP ?? "Idle") {
  const model = PmxLoader.loadFromBuffer(toAB(readFileSync(path)))
  const physics = new RezePhysics(model.getRigidbodies(), model.getJoints())
  const store = physics.store
  const RB = model.getRigidbodies()
  if (clip !== "none") {
    const vmd = join(VMD_DIR, `${clip}.vmd`)
    if (!existsSync(vmd)) throw new Error(`no such clip: ${vmd}`)
    const f = VMDLoader.loadFromBuffer(toAB(readFileSync(vmd)))
    model.loadClip("c", model.buildClipFromVmdKeyFrames(f))
    model.play("c", { loop: true })
  }

  const N = store.count
  const prev = new Float32Array(N * 3)
  const flips = new Float64Array(N)
  const jerk = new Float64Array(N)
  let samples = 0
  for (let i = 0; i < FRAMES; i++) {
    model.update(DT)
    physics.step(DT, model.getWorldMatrices(), model.getBoneInverseBindMatrices())
    const v = store.linearVelocities
    if (i >= WARMUP) {
      samples++
      for (let b = 0; b < N; b++) {
        if (store.invMass[b] <= 0) continue
        const b3 = b * 3
        const dx = v[b3] - prev[b3], dy = v[b3 + 1] - prev[b3 + 1], dz = v[b3 + 2] - prev[b3 + 2]
        jerk[b] += Math.hypot(dx, dy, dz) / DT
        const dot = v[b3] * prev[b3] + v[b3 + 1] * prev[b3 + 1] + v[b3 + 2] * prev[b3 + 2]
        const m1 = Math.hypot(v[b3], v[b3 + 1], v[b3 + 2])
        const m0 = Math.hypot(prev[b3], prev[b3 + 1], prev[b3 + 2])
        // Only count a reversal when there is real motion to reverse.
        if (m0 > 1e-3 && m1 > 1e-3 && dot < 0) flips[b] += 1
      }
    }
    prev.set(v)
  }

  const rows = []
  for (let b = 0; b < N; b++) {
    if (store.invMass[b] <= 0) continue
    rows.push({ index: b, name: RB[b]?.name ?? `#${b}`, flip: flips[b] / samples, jerk: jerk[b] / samples })
  }
  const dyn = rows.length
  const meanFlip = rows.reduce((a, r) => a + r.flip, 0) / dyn
  const meanJerk = rows.reduce((a, r) => a + r.jerk, 0) / dyn
  const bad = rows.filter((r) => r.flip > 0.25).length
  return { rows, dyn, meanFlip, meanJerk, bad }
}

if (process.argv[2]) {
  for (const p of process.argv.slice(2)) {
    const s = shake(p)
    console.log(`\n${p.split("/").at(-1)}  ${s.dyn} dynamic bodies`)
    console.log(
      `  mean flip-rate ${s.meanFlip.toFixed(3)}   mean jerk ${s.meanJerk.toFixed(1)}   bodies flipping >25% of frames: ${s.bad}`,
    )
    console.log(`  worst shakers:`)
    for (const r of s.rows.sort((a, b) => b.flip - a.flip).slice(0, 8))
      console.log(`    flip ${r.flip.toFixed(3)}  jerk ${String(Math.round(r.jerk)).padStart(5)}   ${r.name}`)
  }
}
