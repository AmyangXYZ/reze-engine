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

// `tips`: the irises hang on 左目先/右目先 (children of the eye bones) instead
// of the eye bones themselves, as some converted rigs do.
const fakeModel = ({ tips = false, layers = null } = {}) => {
  const bones = [
    { name: "頭", parentIndex: -1, bindTranslation: [0, 15, 0] },
    { name: "左目", parentIndex: 0, bindTranslation: [0.5, 0.2, -0.5] },
    { name: "右目", parentIndex: 0, bindTranslation: [-0.5, 0.2, -0.5] },
    { name: "左目先", parentIndex: 1, bindTranslation: [0, 0, -0.1] },
    { name: "右目先", parentIndex: 2, bindTranslation: [0, 0, -0.1] },
  ]
  // A ring of eight vertices around each eye, radius 0.2 × 0.3, carried fully
  // by its eye bone; one face vertex on the head.
  const pos = []
  const joints = []
  const weights = []
  for (const [bone, cx] of tips ? [[3, 0.5], [4, -0.5]] : [[1, 0.5], [2, -0.5]]) {
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
  // The last one lifts the LEFT iris (its ring is vertices 0–7) by 0.1, as a
  // rig's pupil-up morph does.
  const morphs = [
    { name: "まばたき" },
    { name: "ウィンク" },
    { name: "ウィンク右" },
    { name: "瞳上", vertexOffsets: [0, 1, 2, 3, 4, 5, 6, 7].map((i) => ({ vertexIndex: i, positionOffset: [0, 0.1, 0] })) },
  ]
  const w = new Float32Array(morphs.length)
  const mats = bones.map(() => IDENTITY())
  return {
    weights: w,
    mats,
    getSkeleton: () => ({ bones }),
    getMorphing: () => ({ morphs }),
    getGeometry: () => ({ positions: new Float32Array(pos), uvs: new Float32Array(0), indices: new Uint32Array(layers ? layers.indices : []) }),
    ...(layers ? { getMaterials: () => layers.materials } : {}),
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
  const o = new Float32Array(96)
  jelly.write(o)
  return {
    o,
    strength: o[51],
    // How far each eye's scale is from rest, across plus down.
    squash: [Math.abs(o[64] - 1) + Math.abs(o[65] - 1), Math.abs(o[68] - 1) + Math.abs(o[69] - 1)],
    scale: [[o[64], o[65]], [o[68], o[69]]],
    twist: [o[66], o[70]],
    offset: [[o[72], o[73]], [o[76], o[77]]],
    tremble: [[o[74], o[75]], [o[78], o[79]]],
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

/** A blink as a VMD keys it: down over four frames, then held. */
const close = (jelly, model, morph, to) => {
  const from = model.weights[morph]
  for (let i = 1; i <= 4; i++) {
    model.weights[morph] = from + ((to - from) * i) / 4
    jelly.step(1 / 60, model)
  }
}

test("a blink widens the iris as the lids close, rings out after, and settles", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { strength: 1, idle: false })
  jelly.step(1 / 60, model)
  close(jelly, model, 0, 1)
  const shut = read(jelly)
  assert.deepEqual(shut.closure, [1, 1])
  assert.ok(shut.scale[0][0] > 1.1, `wider while closing, x ${shut.scale[0][0]}`)
  assert.ok(shut.scale[0][1] < 1, `and pressed flatter, y ${shut.scale[0][1]}`)
  // Opening rings: across, it swings past rest and back several times.
  close(jelly, model, 0, 0)
  const xs = []
  for (let t = 0; t < 3; t += 1 / 60) {
    jelly.step(1 / 60, model)
    xs.push(read(jelly).scale[0][0] - 1)
  }
  let crossings = 0
  for (let i = 1; i < xs.length; i++) if (Math.sign(xs[i]) !== Math.sign(xs[i - 1]) && xs[i] !== 0) crossings++
  assert.ok(crossings >= 3, `it should ring, crossed rest ${crossings} times`)
  const end = read(jelly)
  assert.ok(end.squash[0] < 1e-6, `should settle, ended at ${end.squash[0]}`)
  assert.equal(end.twist[0], 0)
  assert.deepEqual(end.offset[0], [0, 0])
  assert.ok(Math.abs(end.squash[0] - end.squash[1]) < 1e-6)
})

test("a closing expression rolls the iris down about its eye bone on a spring: angles add, it swings past rest on opening", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { idle: false })
  jelly.step(1 / 60, model)
  const deg = (o, e) => (o[83 + e * 4] * 180) / Math.PI
  const hold = (s) => { for (let t = 0; t < s; t += 1 / 60) jelly.step(1 / 60, model) }
  let r = read(jelly)
  assert.equal(deg(r.o, 0), 0)
  // The pivot is the eye bone's head: 左目 at (0.5, 15.2, −0.5).
  assert.deepEqual([...r.o.subarray(80, 83)].map((x) => +x.toFixed(5)), [0.5, 15.2, -0.5])
  close(jelly, model, 0, 1) // まばたき
  hold(3)
  r = read(jelly)
  assert.ok(Math.abs(deg(r.o, 0) + 15) < 0.05 && Math.abs(deg(r.o, 1) + 15) < 0.05, `held, blink settles at −15°: ${deg(r.o, 0)}`)
  assert.ok(Math.abs(r.o[90]) < 1e-4, "and stays back once the spring settles: a shut lid sits close in front")
  close(jelly, model, 1, 1) // ウィンク adds −5° to the left only
  hold(3)
  r = read(jelly)
  assert.ok(Math.abs(deg(r.o, 0) + 20) < 0.05 && Math.abs(deg(r.o, 1) + 15) < 0.05)
  // Opening, it swings up past rest, back and forth, and settles.
  model.weights[1] = 0
  close(jelly, model, 0, 0)
  const trace = []
  for (let t = 0; t < 4; t += 1 / 60) {
    jelly.step(1 / 60, model)
    trace.push(deg(read(jelly).o, 0))
  }
  assert.ok(Math.max(...trace) > 3, `it should swing up past rest, peak ${Math.max(...trace)}`)
  let crossings = 0
  for (let i = 1; i < trace.length; i++) if (Math.sign(trace[i]) !== Math.sign(trace[i - 1]) && trace[i] !== 0) crossings++
  assert.ok(crossings >= 3, `it should wobble, crossed rest ${crossings} times`)
  assert.ok(Math.abs(trace.at(-1)) < 0.05, `and settle, ended at ${trace.at(-1)}`)
})

test("a held half-close (a smile) presses the iris flatter and holds still", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { idle: false })
  model.weights[0] = 0.26
  jelly.step(1 / 60, model)
  for (let i = 0; i < 120; i++) jelly.step(1 / 60, model)
  const a = read(jelly)
  jelly.step(1 / 60, model)
  const b = read(jelly)
  assert.ok(Math.abs(a.scale[0][1] - (1 - 0.18 * 0.26)) < 1e-6, `the lid's press, y ${a.scale[0][1]}`)
  assert.equal(a.scale[0][0], 1)
  assert.deepEqual(a.scale, b.scale, "held, not moving")
})

test("a wink moves only its own eye", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { idle: false })
  jelly.step(1 / 60, model)
  close(jelly, model, 1, 1) // ウィンク — the left eye
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
  assert.equal(read(jelly).offset[0][0], 0)
  // A negative delta (scrubbing back) is a seek too.
  // Landing shut over a seek is no blink: only the lid's press, no spring.
  model.weights[0] = 1
  jelly.step(-1 / 60, model)
  assert.equal(read(jelly).scale[0][0], 1)
  assert.equal(read(jelly).twist[0], 0)
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
      peak = Math.max(peak, r.squash[0])
      assert.equal(r.offset[0][0], 0, "the iris never moves")
    }
    for (let i = 0; i < 600; i++) jelly.step(1 / 60, model)
    return { peak, end: read(jelly).squash[0] }
  }
  const head = swell(0)
  assert.ok(head.peak > 0.01, `a head turn should swell it, peak ${head.peak}`)
  // It ripples, not one dip: the squash crosses back and forth several times.
  {
    const model = fakeModel()
    const jelly = new JellyEyes(model, { idle: false })
    jelly.step(1 / 60, model)
    yaw(model.mats[0], 0.3)
    const trace = []
    for (let t = 0; t < 4; t += 1 / 60) {
      jelly.step(1 / 60, model)
      trace.push(read(jelly).scale[0][1] - 1)
    }
    let crossings = 0
    for (let i = 1; i < trace.length; i++) if (Math.sign(trace[i]) !== Math.sign(trace[i - 1]) && trace[i] !== 0) crossings++
    assert.ok(crossings >= 4, `a knock should ripple, crossed rest ${crossings} times`)
  }
  assert.ok(Math.abs(head.end) < 0.005, `should settle, ended at ${head.end}`)
  assert.equal(swell(1).peak, 0, "an eye turn alone must not")
})

test("idle taps: a knock every 4–8 s, settling between, on the scene clock", () => {
  const model = fakeModel()
  const trace = (idle) => {
    const jelly = new JellyEyes(model, { idle })
    const out = []
    for (let t = 0; t < 20; t += 1 / 60) {
      jelly.step(1 / 60, model)
      out.push(read(jelly).squash[0])
    }
    return out
  }
  const a = trace(true)
  const peak = Math.max(...a.map(Math.abs))
  assert.ok(peak > 0.04, `it should tap noticeably, peak ${peak}`)
  assert.ok(peak < 0.3, `and within the squash's reach, peak ${peak}`)
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

test("the centre follows a morph that moves the iris, so it scales in place", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { idle: false })
  jelly.step(1 / 60, model)
  assert.ok(Math.abs(read(jelly).o[49] - 15.2) < 1e-5, "at rest, the ring's centre")
  model.weights[3] = 1
  jelly.step(1 / 60, model)
  const o = read(jelly).o
  assert.ok(Math.abs(o[49] - 15.3) < 1e-5, `left centre should rise with the morph, at ${o[49]}`)
  assert.ok(Math.abs(o[53] - 15.2) < 1e-5, `the right iris it does not touch stays, at ${o[53]}`)
  model.weights[3] = 0.5
  jelly.step(1 / 60, model)
  assert.ok(Math.abs(read(jelly).o[49] - 15.25) < 1e-5, "half the weight, half the way")
})

test("irises hung on the eye bones' tips are found, on those bones", () => {
  const model = fakeModel({ tips: true })
  const jelly = new JellyEyes(model, { idle: false })
  jelly.step(1 / 60, model)
  const o = read(jelly).o
  assert.ok(Math.abs(o[48] - 0.5) < 1e-5 && Math.abs(o[49] - 15.2) < 1e-5, "left iris centre")
  assert.ok(Math.abs(o[56] - 0.2) < 1e-5 && Math.abs(o[57] - 0.3) < 1e-5, "left iris size, not the fallback")
  assert.equal(o[59], 3, "the left iris scales by 左目先")
  assert.equal(o[63], 4, "the right by 右目先")
})

test("write(): layout, and strength scaling (0 reads as off)", () => {
  const model = fakeModel()
  const jelly = new JellyEyes(model, { strength: 2, idle: false })
  jelly.step(1 / 60, model)
  close(jelly, model, 0, 1)
  const o = new Float32Array(104).fill(7)
  jelly.write(o)
  // Words outside 48..95 are left alone.
  assert.equal(o[47], 7)
  assert.equal(o[96], 7)
  // [0] left centre + strength, [1] right centre + highlight ids (none here).
  assert.ok(Math.abs(o[48] - 0.5) < 1e-5 && Math.abs(o[49] - 15.2) < 1e-5 && Math.abs(o[50] + 0.5) < 1e-5)
  assert.equal(o[51], 2)
  assert.ok(Math.abs(o[52] + 0.5) < 1e-5 && Math.abs(o[53] - 15.2) < 1e-5)
  assert.equal(o[55], 0)
  // [2]/[3] half extents, closure, eye bone.
  assert.ok(Math.abs(o[56] - 0.2) < 1e-5 && Math.abs(o[57] - 0.3) < 1e-5)
  assert.equal(o[58], 1)
  assert.equal(o[59], 1)
  assert.ok(Math.abs(o[60] - 0.2) < 1e-5 && Math.abs(o[61] - 0.3) < 1e-5)
  assert.equal(o[63], 2)
  // [4]/[5] scale, twist: strength 1 departs from rest twice as far as 0.5
  // (2 holds at the bounds).
  jelly.strength = 1
  jelly.write(o)
  const wide1 = o[64] - 1
  const twist1 = o[66]
  jelly.strength = 0.5
  jelly.write(o)
  assert.ok(Math.abs((o[64] - 1) * 2 - wide1) < 1e-5, `scale × strength ${wide1} vs ${o[64] - 1}`)
  assert.ok(Math.abs(o[66] * 2 - twist1) < 1e-6)
  assert.ok(Math.abs(o[68] - o[64]) < 1e-6)

  jelly.strength = 0
  jelly.write(o)
  assert.equal(o[51], 0, "strength 0 is the shader's off")
  assert.equal(o[64], 1)
  assert.equal(o[66], 0)

  clearJellyEyes(o)
  assert.ok(o.subarray(48, 96).every((x) => x === 0))
  assert.equal(o[47], 7)
  assert.equal(o[96], 7)
})

// The ring of each eye drawn twice — an iris material, then a highlight sheet
// on the same vertices — after a face material, as MME rigs lay them out.
const layered = (names = ["face", "eye", "eyelight"]) => ({
  materials: [
    { name: names[0], vertexCount: 3 },
    { name: "empty", vertexCount: 0 },
    { name: names[1], vertexCount: 6 },
    { name: names[2], vertexCount: 6 },
  ],
  indices: [16, 16, 16, 0, 1, 2, 8, 9, 10, 0, 2, 4, 8, 10, 12],
})

test("a highlight sheet over the iris is found by its draw id; a painted-in highlight is not", () => {
  const sheet = new JellyEyes(fakeModel({ layers: layered() }), { idle: false })
  // Draw ids count only the materials that draw: face 1, eye 2, eyelight 3.
  assert.deepEqual(sheet.highlights, [3])
  const o = new Float32Array(96)
  sheet.write(o)
  assert.equal(o[55], 3)
  // Named as one, even alone.
  const named = new JellyEyes(fakeModel({ layers: { materials: [{ name: "face", vertexCount: 3 }, { name: "ハイライト", vertexCount: 6 }], indices: [16, 16, 16, 0, 1, 2, 8, 9, 10] } }), { idle: false })
  assert.deepEqual(named.highlights, [2])
  // One iris per eye, side by side, is not a sheet over the other.
  const sides = new JellyEyes(fakeModel({ layers: { materials: [{ name: "瞳1", vertexCount: 3 }, { name: "瞳2", vertexCount: 3 }], indices: [0, 2, 4, 8, 10, 12] } }), { idle: false })
  assert.deepEqual(sides.highlights, [])
})

test("the highlight trembles: held random spots, about ten a second, on the scene clock", () => {
  const model = fakeModel()
  const trace = (idle) => {
    const jelly = new JellyEyes(model, { idle })
    const out = []
    for (let t = 0; t < 2; t += 1 / 60) {
      jelly.step(1 / 60, model)
      out.push(read(jelly).tremble[0][0])
    }
    return out
  }
  const a = trace(true)
  let changes = 0
  for (let i = 1; i < a.length; i++) if (a[i] !== a[i - 1]) changes++
  assert.ok(changes >= 14 && changes <= 20, `about 9 new spots a second, saw ${changes} in 2 s`)
  // Small: within 3% of the iris's radius (0.25 mean here).
  assert.ok(a.every((x) => Math.abs(x) <= 0.03 * 0.25 + 1e-9), `tremble ${Math.max(...a.map(Math.abs))}`)
  assert.ok(a.some((x) => Math.abs(x) > 0.002), "but there")
  assert.deepEqual(trace(true), a, "the same clock trembles the same way")
  assert.ok(trace(false).every((x) => x === 0), "idle off: still")
  // The iris itself does not tremble.
  const jelly = new JellyEyes(model, {})
  for (let i = 0; i < 60; i++) jelly.step(1 / 60, model)
  assert.deepEqual(read(jelly).offset[0], [0, 0])
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
  const o = new Float32Array(96)
  jelly.write(o)
  const [lx, ly, , , rx, ry] = [o[48], o[49], o[50], o[51], o[52], o[53]]
  assert.ok(lx > 0 && rx < 0, `left iris on +X, right on −X: ${lx}, ${rx}`)
  assert.ok(Math.abs(ly - ry) < 0.5, `irises level: ${ly}, ${ry}`)
  for (const r of [o[56], o[57], o[60], o[61]]) assert.ok(r > 0 && r < Math.abs(lx - rx), `iris radius ${r}`)
  assert.ok(o.every(Number.isFinite))
})
