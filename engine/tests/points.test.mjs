// `#points`: named bones as points an effect can put something on. Run: npm test.
//
// No GPU here, so the WGSL is checked as text — every module an effect's source
// lands in must resolve the names exactly once — and the CPU half is checked by
// value: which bones match, and where a posed, placed bone's head and tip go.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"
import { parseDirectives } from "../dist/shaders/directives.js"
import { MAX_EFFECT_POINTS, POINTS_FLOATS, pointsApi } from "../dist/shaders/points-api.js"
import { bonesWithPrefix, writeBonePoint } from "../dist/effect-points.js"
import { buildParticleComputeShader, buildParticleRenderShader, PARTICLE_POINTS_BINDING } from "../dist/shaders/passes/particles.js"
import { buildTrailShader } from "../dist/shaders/passes/trails.js"
import { buildFieldShader, EFFECT_SCENE_API } from "../dist/shaders/passes/composite.js"
import { buildSimShader } from "../dist/shaders/passes/grid.js"
import { buildLightEmitShader } from "../dist/shaders/lights.js"
import { anchorAliasWgsl } from "../dist/shaders/anchor-table.js"

test("#points names a bone prefix, and nothing declares none", () => {
  assert.equal(parseDirectives("#points flame").directives.points, "flame")
  assert.equal(parseDirectives("// no points here").directives.points, null)
  // Prose mentioning it declares nothing, the #anchor rule.
  assert.equal(parseDirectives("// mentioning #points flame in a sentence").directives.points, null)
  assert.ok(parseDirectives("#points").errors.length > 0, "a prefix is required")
})

test("every module an effect lands in resolves the points API exactly once", () => {
  const CAST = { subjects: 4, samples: 128, base: 12, trailBase: 108, slots: 8, alias: [0], trailCount: 1 }
  const P = { wgsl: "fn particleInit(id: u32, s: f32) -> Particle { var q: Particle; return q; }", count: 64, blend: "alpha", bloom: false, paramsDecl: "", gridSize: 0, cover: false }
  const modules = {
    field: buildFieldShader({ wgsl: "fn foreground(r: vec3f, uv: vec2f, t: f32, d: f32) -> vec4f { return vec4f(0.0); }", paramsDecl: "", hasBackground: false, hasForeground: true, gridSize: 0 }),
    "particle compute": buildParticleComputeShader(P, CAST),
    "particle render": buildParticleRenderShader(P, CAST),
    trail: buildTrailShader({ wgsl: "fn trailWidth(u: f32, a: f32) -> f32 { return 1.0; }", slots: 1, ribbonSlots: [0], blend: "additive", bloom: true }, CAST),
    grid: buildSimShader("fn gridStep(uv: vec2f, prev: vec4f, dt: f32) -> vec4f { return prev; }", 64, CAST),
    emit: buildLightEmitShader("fn lightEmit(i: u32, time: f32) -> RzLight { var l: RzLight; return l; }", EFFECT_SCENE_API + anchorAliasWgsl([0]), CAST),
  }
  for (const [name, src] of Object.entries(modules)) {
    for (const decl of [/struct RzPoint\b/g, /fn rzPointCount\(/g, /fn rzPoint\(/g]) {
      assert.equal((src.match(decl) ?? []).length, 1, `${name}: ${decl} must appear exactly once`)
    }
  }
  // Only the particle stages read the buffer; the rest are stubs reading zero.
  for (const name of ["particle compute", "particle render"]) {
    assert.match(modules[name], new RegExp(`@binding\\(${PARTICLE_POINTS_BINDING}\\) var<storage, read> _rzPoints`))
  }
  for (const name of ["field", "trail", "grid", "emit"]) {
    assert.doesNotMatch(modules[name], /_rzPoints/)
    assert.match(modules[name], /fn rzPointCount\(\) -> u32 \{ return 0u; \}/)
  }
})

test("the shader reads the layout the CPU writes", () => {
  // Header vec4 (count), then two vec4s a point: head at 4 + 8i, tip at 8 + 8i.
  const wgsl = pointsApi(true, 0, 15)
  assert.match(wgsl, /return min\(u32\(_rzPoints\[0\]\.x\), 256u\)/)
  assert.match(wgsl, /p\.pos = _rzPoints\[1u \+ i \* 2u\]\.xyz;/)
  assert.match(wgsl, /p\.tip = _rzPoints\[2u \+ i \* 2u\]\.xyz;/)
  assert.equal(POINTS_FLOATS, 4 + MAX_EFFECT_POINTS * 8)
})

test("bones match by prefix, by index, duplicates included", () => {
  const bones = [{ name: "センター" }, { name: "flame.01" }, { name: "flame.02" }, { name: "flame" }, { name: "flame.02" }, { name: "noflame" }]
  assert.deepEqual(bonesWithPrefix(bones, "flame"), [1, 2, 3, 4])
  assert.deepEqual(bonesWithPrefix(bones, "torch"), [])
})

test("a point is the bone's head and tail, posed and placed", () => {
  const out = new Float32Array(POINTS_FLOATS)
  // A bone standing at (1, 2, 3), turned 90° about y (x → -z), tail 2 up and 1 along x.
  const world = new Float32Array([0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 2, 3, 1])
  const at = { scale: 2, rotation: { x: 0, y: 0, z: 0, w: 1 }, position: { x: 10, y: 0, z: 0 } }
  writeBonePoint(out, 1, world, [1, 2, 0], at)
  const b = 4 + 8
  const near = (a, e) => a.forEach((v, k) => assert.ok(Math.abs(v - e[k]) < 1e-5, `${[...a]} vs ${e}`))
  near(out.subarray(b, b + 3), [12, 4, 6])
  // Tail turned: (1,2,0) → (0, 2, -1); tip (1, 4, 2); placed ×2 + (10,0,0).
  near(out.subarray(b + 4, b + 7), [12, 8, 4])
  // A model turned 180° about y moves both, as the cast's anchors are moved.
  const turned = { scale: 1, rotation: { x: 0, y: 1, z: 0, w: 0 }, position: { x: 0, y: 0, z: 0 } }
  writeBonePoint(out, 0, world, undefined, turned)
  near(out.subarray(4, 7), [-1, 2, -3])
  near(out.subarray(8, 11), [-1, 2, -3])
})

// ── The loader keeps a bone's tail ──

const here = dirname(fileURLToPath(import.meta.url))
const { PmxLoader } = await import("../dist/pmx-loader.js")
const findModels = () => {
  const roots = [join(here, "../../web/public/models"), join(here, "../../../reze-studio/public/models")].filter(existsSync)
  const out = []
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry)
      if (statSync(p).isDirectory()) walk(p)
      else if (p.toLowerCase().endsWith(".pmx")) out.push(p)
    }
  }
  for (const r of roots) walk(r)
  return out.slice(0, 4)
}
const MODELS = findModels()
const toAB = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)

test("the loader keeps every bone's tail as a finite offset", { skip: MODELS.length === 0 }, () => {
  // A tail that names no bone (-1) is a bone with no tail, which weapons and
  // props are full of; every tail that IS stated must be a finite offset.
  let long = 0
  for (const path of MODELS) {
    const bones = PmxLoader.loadFromBuffer(toAB(readFileSync(path))).getSkeleton().bones
    for (const b of bones.filter((x) => x.tail)) {
      assert.ok(b.tail.length === 3 && b.tail.every(Number.isFinite), `${path} ${b.name}: ${b.tail}`)
      if (Math.hypot(...b.tail) > 0) long++
    }
  }
  assert.ok(long > 0, "a real rig carries tails with length")
})
