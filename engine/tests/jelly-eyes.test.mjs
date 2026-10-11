// Jelly eyes: the springs behind the 果冻眼 wobble (src/jelly-eyes.ts).
//
// Most of this runs on a fake model — a head, two eye bones, a blink morph and
// a ring of iris vertices on each eye — so the spring's behaviour is pinned
// exactly whatever models this machine has. The last test reads a real PMX
// when one is on disk, to check the irises are found where the eyes are.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const { JellyEyes, clampStrength, clearJellyEyes } = await import("../dist/jelly-eyes.js")

// ── A fake model: just what JellyEyes reads ──

const IDENTITY = () => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])

const fakeModel = () => {
  const bones = [
    { name: "頭", parentIndex: -1, bindTranslation: [0, 15, 0] },
    { name: "左目", parentIndex: 0, bindTranslation: [0.5, 0.2, -0.5] },
    { name: "右目", parentIndex: 0, bindTranslation: [-0.5, 0.2, -0.5] },
  ]
  // A ring of eight vertices around each eye, radius 0.2 × 0.3, carried fully
  // by its eye bone; one face vertex on the head.
  const pos = []
  const joints = []
  const weights = []
  for (const [bone, cx] of [[1, 0.5], [2, -0.5]]) {
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2
      pos.push(cx + 0.2 * Math.cos(a), 15.2 + 0.3 * Math.sin(a), -0.5)
      joints.push(bone, 0, 0, 0)
      weights.push(255, 0, 0, 0)
    }
  }
  pos.push(0, 14, -0.6)
  joints.push(0, 0, 0, 0)
  weights.push(255, 0, 0, 0)
  const morphs = [{ name: "まばたき" }, { name: "ウィンク" }, { name: "ウィンク右" }]
  const w = new Float32Array(morphs.length)
  const mats = bones.map(() => IDENTITY())
  return {
    weights: w,
    mats,
    getSkeleton: () => ({ bones }),
    getMorphing: () => ({ morphs }),
    getGeometry: () => ({ positions: new Float32Array(pos), uvs: new Float32Array(0), indices: new Uint32Array(0) }),
    getSkinning: () => ({ joints: new Uint16Array(joints), weights: new Uint8Array(weights) }),
    getEffectiveMorphWeights: () => w,
    getBoneWorldMatrixAt: (i) => mats[i] ?? null,
  }
}

/** Yaw an eye bone's world matrix by `a` radians about +Y. */
const yaw = (m, a) => {
  const c = Math.cos(a)
  const s = Math.sin(a)
  m.set([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1])
}

const read = (jelly) => {
  const o = new Float32Array(72)
  jelly.write(o)
  return {
    o,
    strength: o[51],
    squash: [o[64], o[68]],
    offset: [[o[65], o[66]], [o[69], o[70]]],
    closure: [o[58], o[62]],
  }
}

/** Run the springs for `seconds` at 60 fps, collecting the left eye's squash. */
const run = (jelly, model, seconds) => {
  const trace = []
  for (let t = 0; t < seconds; t += 1 / 60) {
    jelly.step(1 / 60, model)
    trace.push(read(jelly).squash[0])
  }
  return trace
}

test("clampStrength: default 1, held to 0..2", () => {
  assert.equal(clampStrength(undefined), 1)
  assert.equal(clampStrength(0.5), 0.5)
  assert.equal(clampStrength(-1), 0)
  assert.equal(clampStrength(5), 2)
  assert.equal(clampStrength(0), 0)
  const model = fakeModel()
  assert.equal(new JellyEyes(model, { idle: false }).strength, 1)
  assert.equal(new JellyEyes(model, { strength: 9 }).strength, 2)
})

test("a blink squashes as the lid moves, opening too, and settles", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { strength: 1, idle: false })
  jelly.step(1 / 60, model)
  model.weights[0] = 1
  const closing = run(jelly, model, 0.2)
  assert.ok(Math.max(...closing) > 0.03, `closing squash ${Math.max(...closing)}`)
  assert.deepEqual(read(jelly).closure, [1, 1])
  // Held shut it comes back to rest: only a change of closure squashes.
  const held = run(jelly, model, 4)
  assert.ok(Math.abs(held.at(-1)) < 0.01, `held shut should settle, ended at ${held.at(-1)}`)
  // Opening — the half you can see — squashes again, then settles.
  model.weights[0] = 0
  const opening = run(jelly, model, 4)
  assert.ok(Math.max(...opening) > 0.03, `opening squash ${Math.max(...opening)}`)
  assert.ok(Math.abs(opening.at(-1)) < 0.01, `should settle, ended at ${opening.at(-1)}`)
  const end = read(jelly)
  assert.ok(Math.abs(end.squash[0] - end.squash[1]) < 1e-6)
})

test("a held half-close (a smile) leaves the iris at rest", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { idle: false })
  model.weights[0] = 0.26
  jelly.step(1 / 60, model)
  const trace = run(jelly, model, 2)
  assert.ok(trace.every((x) => Math.abs(x) < 1e-6), `squash while smiling ${Math.max(...trace.map(Math.abs))}`)
})

test("a wink moves only its own eye", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { idle: false })
  jelly.step(1 / 60, model)
  model.weights[1] = 1 // ウィンク — the left eye
  run(jelly, model, 0.1)
  const r = read(jelly)
  assert.ok(r.squash[0] > 0.05, `left squash ${r.squash[0]}`)
  assert.equal(r.squash[1], 0)
  assert.deepEqual(r.closure, [1, 0])
})

test("a seek (dt > 0.25 s) lands on the pose instead of springing to it", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { idle: false })
  model.weights[0] = 1
  jelly.step(1 / 60, model)
  model.weights[0] = 0
  jelly.step(0.5, model)
  assert.equal(read(jelly).squash[0], 0)
  // A turn of the eye over a seek leaves no lag either.
  yaw(model.mats[1], 0.4)
  jelly.step(0.5, model)
  assert.deepEqual(read(jelly).offset[0], [0, 0])
  // A negative delta (scrubbing back) is a seek too.
  // Landing shut over a seek is no blink, so no squash.
  model.weights[0] = 1
  jelly.step(-1 / 60, model)
  assert.equal(read(jelly).squash[0], 0)
})

test("her head turning swells it in place; her eyes turning (eye tracking) does not", () => {
  const swell = (bone) => {
    const model = fakeModel()
    const jelly = new JellyEyes(model, { idle: false })
    jelly.step(1 / 60, model)
    yaw(model.mats[bone], 0.3)
    let peak = 0
    for (let i = 0; i < 30; i++) {
      jelly.step(1 / 60, model)
      const r = read(jelly)
      peak = Math.max(peak, r.o[55])
      assert.deepEqual(r.offset[0], [0, 0], "the iris never moves")
    }
    for (let i = 0; i < 600; i++) jelly.step(1 / 60, model)
    return { peak, end: read(jelly).o[55] }
  }
  const head = swell(0)
  assert.ok(head.peak > 0.01, `a head turn should swell it, peak ${head.peak}`)
  // It ripples, not one dip: the squash crosses back and forth several times.
  {
    const model = fakeModel()
    const jelly = new JellyEyes(model, { idle: false })
    jelly.step(1 / 60, model)
    yaw(model.mats[0], 0.3)
    const trace = run(jelly, model, 4)
    let crossings = 0
    for (let i = 1; i < trace.length; i++) if (Math.sign(trace[i]) !== Math.sign(trace[i - 1]) && trace[i] !== 0) crossings++
    assert.ok(crossings >= 4, `a knock should ripple, crossed rest ${crossings} times`)
  }
  assert.ok(Math.abs(head.end) < 0.005, `should settle, ended at ${head.end}`)
  assert.equal(swell(1).peak, 0, "an eye turn alone must not")
})

test("idle taps: a small bounce every few seconds, settling between, on the scene clock", () => {
  const model = fakeModel()
  const trace = (idle) => {
    const jelly = new JellyEyes(model, { idle })
    const out = []
    for (let t = 0; t < 12; t += 1 / 60) {
      jelly.step(1 / 60, model)
      out.push(read(jelly).squash[0])
    }
    return out
  }
  const a = trace(true)
  const peak = Math.max(...a.map(Math.abs))
  assert.ok(peak > 0.02, `it should tap while nothing else moves, peak ${peak}`)
  assert.ok(peak < 0.1, `and stay small, peak ${peak}`)
  // Rests between taps: somewhere it has settled almost still.
  assert.ok(a.some((x, i) => i > 60 && Math.abs(x) < 0.003), "it should settle between taps")
  // The same clock taps the same way.
  assert.deepEqual(trace(true), a)
  assert.ok(trace(false).every((x) => x === 0), "idle off: nothing")
  // Paused (no time passing) it holds still.
  const jelly = new JellyEyes(model, {})
  for (let i = 0; i < 200; i++) jelly.step(1 / 60, model)
  const held = read(jelly).squash[0]
  jelly.step(0, model)
  jelly.step(0, model)
  assert.ok(Math.abs(read(jelly).squash[0] - held) < 1e-9)
})

test("write(): layout, and strength scaling (0 reads as off)", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { strength: 2, idle: false })
  jelly.step(1 / 60, model)
  model.weights[0] = 1
  run(jelly, model, 0.1)
  const o = new Float32Array(80).fill(7)
  jelly.write(o)
  // Words outside 48..71 are left alone.
  assert.equal(o[47], 7)
  assert.equal(o[72], 7)
  // [0] left centre + strength, [1] right centre: the ring centres.
  assert.ok(Math.abs(o[48] - 0.5) < 1e-5 && Math.abs(o[49] - 15.2) < 1e-5 && Math.abs(o[50] + 0.5) < 1e-5)
  assert.equal(o[51], 2)
  assert.ok(Math.abs(o[52] + 0.5) < 1e-5 && Math.abs(o[53] - 15.2) < 1e-5)
  // [2]/[3] half extents, closure, eye bone.
  assert.ok(Math.abs(o[56] - 0.2) < 1e-5 && Math.abs(o[57] - 0.3) < 1e-5)
  assert.equal(o[58], 1)
  assert.equal(o[59], 1)
  assert.ok(Math.abs(o[60] - 0.2) < 1e-5 && Math.abs(o[61] - 0.3) < 1e-5)
  assert.equal(o[63], 2)
  // [4]/[5] squash scaled by strength: twice what strength 1 writes.
  assert.ok(o[64] > 0.05, `squash ${o[64]}`)
  const squash2 = o[64]
  jelly.strength = 1
  jelly.write(o)
  assert.ok(Math.abs(o[64] * 2 - squash2) < 1e-5, `squash × strength ${squash2} vs ${o[64]}`)
  assert.ok(Math.abs(o[68] - o[64]) < 1e-6)

  jelly.strength = 0
  jelly.write(o)
  assert.equal(o[51], 0, "strength 0 is the shader's off")
  assert.equal(o[64], 0)

  clearJellyEyes(o)
  assert.ok(o.subarray(48, 72).every((x) => x === 0))
  assert.equal(o[47], 7)
  assert.equal(o[72], 7)
})

// ── A real rig, when this machine has one ──

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
const { PmxLoader } = await import("../dist/pmx-loader.js")

const findEyedRig = () => {
  for (const path of findModels()) {
    let model
    try {
      model = PmxLoader.loadFromBuffer(toAB(readFileSync(path)))
    } catch {
      continue
    }
    const names = model.getSkeleton().bones.map((b) => b.name)
    if (["頭", "左目", "右目"].every((n) => names.includes(n))) return model
  }
  return null
}

const RIG = findEyedRig()

test("a real rig: irises found on either side, near the eye bones", { skip: RIG ? false : "no model with eye bones on this machine" }, () => {
  const jelly = new JellyEyes(RIG, {})
  jelly.step(1 / 60, RIG)
  const o = new Float32Array(72)
  jelly.write(o)
  const [lx, ly, , , rx, ry] = [o[48], o[49], o[50], o[51], o[52], o[53]]
  assert.ok(lx > 0 && rx < 0, `left iris on +X, right on −X: ${lx}, ${rx}`)
  assert.ok(Math.abs(ly - ry) < 0.5, `irises level: ${ly}, ${ry}`)
  for (const r of [o[56], o[57], o[60], o[61]]) assert.ok(r > 0 && r < Math.abs(lx - rx), `iris radius ${r}`)
  assert.ok(o.every(Number.isFinite))
})
