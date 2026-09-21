// The positional-light layer. Run: npm test.
//
// The property everything else rests on: a scene with NO lights renders exactly
// as it did before lights existed. That is not a nicety — the accessors are
// spliced into every material and the ground, so if the zero case cost anything
// or changed anything, the feature would have to be paid for by every scene
// that never asked for it.
//
// No GPU here, so the shading itself is checked by reading the emitted WGSL and
// by reimplementing the falloff in JS against the same layout constants. What
// that can catch is the whole failure class this file cares about: a loop that
// runs when the count is zero, an accessor that indexes the wrong slot, a
// falloff that never reaches zero at the radius.

import { test } from "node:test"
import { readFileSync } from "node:fs"
import assert from "node:assert/strict"
import { LIGHT_GRID_BASE, LIGHT_GRID_CELLS, LIGHT_HEADER, LIGHT_MASK_WORDS, LIGHT_STRIDE, LIGHTS_FLOATS, MAX_LIGHTS, lightsApi } from "../dist/shaders/lights.js"
import { buildLightGrid } from "../dist/light-grid.js"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "../dist/shaders/materials/common.js"
import { groundShaderWgsl } from "../dist/shaders/passes/ground.js"

const wgsl = lightsApi(0, 6)

test("the buffer is the header, the records, then the grid", () => {
  assert.equal(LIGHT_GRID_BASE, LIGHT_HEADER + MAX_LIGHTS * LIGHT_STRIDE)
  assert.equal(LIGHTS_FLOATS, LIGHT_GRID_BASE + LIGHT_GRID_CELLS * LIGHT_MASK_WORDS)
  // vec4-aligned, all three: the shader reads the buffer as vec4u.
  assert.equal(LIGHT_HEADER % 4, 0, "the header must not push the records off a vec4 boundary")
  assert.equal(LIGHT_STRIDE % 4, 0, "a record must be a whole number of vec4s")
  assert.equal(LIGHT_GRID_BASE % 4, 0, "the grid must start on a vec4")
})

test("a cell's lamp bits are one vec4u, and the shader walks all four words", () => {
  // The walk is unrolled over m.x..m.w. Raising the cap past 128 without
  // widening both would drop lamps 128 and up in silence.
  assert.equal(LIGHT_MASK_WORDS, 4)
  assert.equal(MAX_LIGHTS, LIGHT_MASK_WORDS * 32)
  const body = wgsl.slice(wgsl.indexOf("fn rzLightsDiffuse"))
  for (const [w, base] of [["x", 0], ["y", 32], ["z", 64], ["w", 96]]) {
    assert.match(body, new RegExp(`_rzLightWord\\(m\\.${w}, ${base}u, p, n\\)`))
  }
})

test("the accessors read the slots the writer writes", () => {
  // The engine writes position at b+0..2, radius at b+3, colour at b+4..6.
  // These are the reads. They are in two files and can drift apart in silence —
  // the symptom would be a light with someone else's radius.
  assert.match(wgsl, new RegExp(`return bitcast<vec4f>\\(_rzLights\\[${LIGHT_HEADER / 4}u \\+ i \\* ${LIGHT_STRIDE / 4}u \\+ k\\]\\)`))
  assert.match(wgsl, /fn rzLightPos\(i: u32\) -> vec3f \{ return _rzLightVec\(i, 0u\)\.xyz; \}/)
  assert.match(wgsl, /fn rzLightRadius\(i: u32\) -> f32 \{ return _rzLightVec\(i, 0u\)\.w; \}/)
  assert.match(wgsl, /fn rzLightColor\(i: u32\) -> vec3f \{ return _rzLightVec\(i, 1u\)\.xyz; \}/)
})

test("the count is clamped in the shader, not only by the writer", () => {
  // The buffer is fixed size. A count past the cap — a stale write, a caller
  // reaching in — would read past the records into whatever follows, so the
  // shader clamps rather than trusting the number it was handed.
  assert.match(wgsl, /return min\(u32\(bitcast<f32>\(_rzLights\[0\]\.x\)\), RZ_MAX_LIGHTS\)/)
  // And the document's share can never exceed the whole.
  assert.match(wgsl, /return min\(u32\(bitcast<f32>\(_rzLights\[0\]\.y\)\), rzLightCount\(\)\)/)
})

test("the loop is bounded by the count, so zero lights runs nothing", () => {
  const body = wgsl.slice(wgsl.indexOf("fn rzLightsDiffuse"))
  assert.match(body, /let count = rzLightCount\(\);/)
  // The grid is consulted only when the document placed lamps, and the
  // effects' walk starts where the document's end: with none, neither runs.
  assert.match(body, /if \(docs > 0u\) \{/)
  assert.match(body, /for \(var i = docs; i < count; i = i \+ 1u\)/)
  // Starts at zero and only ever accumulates inside the loop, so with no
  // lights it returns exactly vec3f(0.0) — and adding that to a colour is an
  // exact float operation, which is what makes "bit-identical" true rather
  // than "close enough".
  assert.match(body, /var acc = vec3f\(0\.0\);/)
  assert.match(body, /return acc;/)
})

test("both surfaces that shade get the same accessors", () => {
  // Materials via the shared prelude, the ground in its own module. A lamp
  // lighting the cast and not the floor under her is the failure here.
  assert.match(COMMON_MATERIAL_PRELUDE_WGSL, /fn rzLightsDiffuse\(/)
  assert.match(groundShaderWgsl(), /fn rzLightsDiffuse\(/)
  assert.match(groundShaderWgsl(), /let lamps = rzLightsDiffuse\(i\.worldPos, n\);/)
  // Same binding in both, or one of them reads the wrong buffer.
  for (const src of [COMMON_MATERIAL_PRELUDE_WGSL, groundShaderWgsl()]) {
    assert.match(src, /@group\(0\) @binding\(6\) var<storage, read> _rzLights: array<vec4u>;/)
  }
})

/** The shader's falloff and cone, reimplemented against the same constants.
 *  `aim` and `cone` default to what a POINT light stores. */
function contribution(light, p, n) {
  const aim = light.aim ?? [0, 0, 0]
  const cone = light.cone ?? [-1, -1]
  const d = [light.pos[0] - p[0], light.pos[1] - p[1], light.pos[2] - p[2]]
  const dist = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
  if (dist >= light.radius) return 0
  const inv = 1 / Math.max(dist, 1e-4)
  const toLight = [d[0] * inv, d[1] * inv, d[2] * inv]
  const ndl = Math.max(n[0] * toLight[0] + n[1] * toLight[1] + n[2] * toLight[2], 0)
  if (ndl <= 0) return 0
  const t = Math.min(Math.max(dist / Math.max(light.radius, 1e-4), 0), 1)
  const falloff = 1 - t * t
  const axis = -(toLight[0] * aim[0] + toLight[1] * aim[1] + toLight[2] * aim[2])
  const lit = Math.min(Math.max((axis - cone[0]) / Math.max(cone[1] - cone[0], 1e-4), 0), 1)
  return ndl * falloff * falloff * lit * lit
}

/** The cosine pair setLights stores for a cone of `deg` degrees, inner 80% of it. */
const coneOf = (deg) => [Math.cos(((deg / 2) * Math.PI) / 180), Math.cos(((deg * 0.8 * 0.5) * Math.PI) / 180)]

test("a light's reach ENDS at its radius", () => {
  // Pure inverse-square never reaches zero, so every light would touch every
  // fragment and the cap would be the only thing bounding the cost. The window
  // is what makes the radius mean what it says — and what a cull could later
  // be derived from.
  const light = { pos: [0, 0, 0], radius: 5 }
  const n = [0, 0, -1]
  assert.ok(contribution(light, [0, 0, 1], n) > 0, "inside the radius it lights")
  assert.equal(contribution(light, [0, 0, 5], n), 0, "AT the radius it is exactly zero")
  assert.equal(contribution(light, [0, 0, 9], n), 0, "past it, still zero")
})

test("a surface facing away takes nothing", () => {
  const light = { pos: [0, 0, 0], radius: 5 }
  assert.equal(contribution(light, [0, 0, 1], [0, 0, 1]), 0, "back to the light")
})

test("the falloff is finite at the source", () => {
  // An unbounded 1/r² is infinite where the light sits, and a lamp inside
  // geometry would blow the frame out rather than look bright.
  const light = { pos: [0, 0, 0], radius: 5 }
  const v = contribution(light, [0, 0, 1e-5], [0, 0, -1])
  assert.ok(Number.isFinite(v) && v <= 1.0, `contribution at the source was ${v}`)
})

test("intensity is usable at the scale a scene is actually built at", () => {
  // The bug this replaced: a windowed real inverse-square is measured in world
  // units, an MMD character is ~18 of them tall, and a lamp a couple of units
  // off her shoulder divided by nearly 40. Intensity 4 landed as 0.06 and
  // nothing on screen changed. A light well inside its own radius has to
  // deliver most of its intensity, or the dial is a lie.
  const light = { pos: [0, 12, -6], radius: 25 }
  const chest = contribution(light, [0, 12, 0], [0, 0, -1])
  assert.ok(chest > 0.5, `six units in, a 25-unit light delivered ${chest.toFixed(3)} of its intensity`)
  // And still nothing at all past the radius.
  assert.equal(contribution(light, [0, 12, 20], [0, 0, -1]), 0)
})

// ── The lightEmit mount ──

import { RZ_LIGHT_STRUCT_WGSL, buildLightEmitShader, hasLightEmit } from "../dist/shaders/lights.js"
import { parseDirectives } from "../dist/shaders/directives.js"
import { EFFECT_SCENE_API, buildFieldShader } from "../dist/shaders/passes/composite.js"
import { anchorAliasWgsl } from "../dist/shaders/anchor-table.js"

/** What the engine hands the builder: the scene API plus this effect's alias. */
const API = EFFECT_SCENE_API + anchorAliasWgsl([0])
/** The cast shape the emit module turns into constants — two trailed anchors,
 *  so RZ_TRAIL_SLOTS is a number a hosted trail loop could actually be wrong about. */
const CAST = { trailCount: 2 }
import { buildParticleComputeShader, buildParticleRenderShader } from "../dist/shaders/passes/particles.js"
import { buildTrailShader } from "../dist/shaders/passes/trails.js"

const EMIT = `#lights 3
fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  l.pos = vec3f(f32(i) * 2.0, 10.0 + time, 0.0);
  l.color = vec3f(1.0, 0.5, 0.2);
  l.intensity = 3.0;
  l.radius = 20.0;
  return l;
}`

test("an effect declares how many lights it emits", () => {
  assert.equal(Math.min(parseDirectives(EMIT).directives.lights, MAX_LIGHTS), 3)
  const lights = (src) => Math.min(parseDirectives(src).directives.lights, MAX_LIGHTS)
  assert.equal(lights("// nothing here"), 0)
  // Clamped, not rejected: the same choice #particles makes.
  assert.equal(lights("#lights 999"), MAX_LIGHTS)
  // Prose mentioning one must not declare anything, the #anchor rule.
  assert.equal(lights("// mentioning #lights 4 in a sentence"), 0)
  assert.equal(hasLightEmit(EMIT), true)
  assert.equal(hasLightEmit("fn background() {}"), false)
})

test("the emit shader writes the slots the material shader reads", () => {
  const src = buildLightEmitShader(EMIT, API, CAST)
  // Same stride and header on both sides of the buffer, expressed against the
  // same constants — this is the seam where a writer and a reader drift.
  assert.match(src, new RegExp(`let b = ${LIGHT_HEADER}u \\+ \\(u32\\(_rzLightU\\[0\\]\\.y\\) \\+ i\\) \\* ${LIGHT_STRIDE}u;`))
  assert.match(src, /_rzLightsOut\[b \+ 3u\] = select\(0\.0, max\(l\.radius, 0\.0\), finite && _rzLightU\[0\]\.w > 0\.0\);/)
  // Colour x intensity, the same product the CPU writer stores — through the
  // sanitized local, since the raw product is what the guard exists to check.
  assert.match(src, /_rzLightsOut\[b \+ 4u\] = c\.x;/)
  assert.ok(src.includes(EMIT), "the author's source is spliced in verbatim")
})

test("the emit shader guards its dispatch tail", () => {
  const src = buildLightEmitShader(EMIT, API, CAST)
  // A workgroup is 64 wide and a count rarely is. Without this the tail threads
  // write into whatever slots follow — another effect's lights, silently.
  assert.match(src, /if \(i >= u32\(_rzLightU\[0\]\.z\)\) \{ return; \}/)
})

test("time is a PARAMETER, so the same source compiles in every module", () => {
  // An effect that emits lights AND draws something has its whole source
  // spliced into the field, particle, trail or grid module too — where
  // lightEmit is dead code that still has to resolve. Those modules already
  // define rzTime differently, so lightEmit's SIGNATURE must not need it.
  assert.match(buildLightEmitShader(EMIT, API, CAST), /let l = lightEmit\(i, _rzLightU\[0\]\.x\);/)
})

test("the emit module hosts the whole file, so the whole API resolves in it", () => {
  // The other half of the rule above, and the one that was missing: the
  // signature must not DEPEND on rzTime, but the module must still DEFINE it,
  // because a trail effect that grows a lamp at its tip compiles its ribbon
  // code here too. Hand Ribbon failed on exactly this — rzTime, rzFalloff,
  // rzViewportHeight and RZ_TRAIL_SLOTS, none of them called by lightEmit.
  const src = buildLightEmitShader(EMIT, API, CAST)
  for (const name of ["rzTime", "rzDt", "rzFalloff", "rzViewportHeight", "rzValueNoise", "rzCurlNoise"]) {
    assert.match(src, new RegExp(`fn ${name}\\(`), `${name} must resolve in the emit module`)
  }
  assert.match(src, /const RZ_TRAIL_SLOTS: i32 = 2;/)
  // Once each. EFFECT_SCENE_API is in this module too, so a helper added to
  // both blocks is a redefinition rather than a convenience.
  for (const name of ["rzTime", "rzFalloff", "rzHash11", "rzHash31"]) {
    const n = (src.match(new RegExp(`fn ${name}\\(`, "g")) ?? []).length
    assert.equal(n, 1, `${name} is defined ${n} times in the emit module`)
  }
})

test("a lamp can pulse on the beat and light up on a note", () => {
  // Audio and score are the reason an effect owns a light at all: a document
  // light is placed once, and this one can answer the music. Without these the
  // mount is only a way to move a lamp along a bone.
  const src = buildLightEmitShader(EMIT, API, CAST)
  assert.match(src, /fn rzAudioLevel\(/)
  assert.match(src, /fn rzNoteVelocity\(/)
  // On the same bindings the particle and trail modules use, so the layout in
  // engine.ts is one convention rather than three.
  assert.match(src, /@group\(0\) @binding\(4\) var<storage, read> _rzAudio/)
  assert.match(src, /@group\(0\) @binding\(5\) var<storage, read> _rzMidi/)
})

test("the emit stage can read the cast, so a lamp can aim at someone", () => {
  // Stage Lights points its beams at rzSubject().root. A light that could not
  // ask where she is could only sit where the fixture hangs, which is the one
  // place a follow-spot never is.
  const src = buildLightEmitShader(EMIT, API, CAST)
  assert.match(src, /fn rzSubject\(/)
  assert.match(src, /fn rzTrail\(/)
  // The alias too, or an effect sharing the anchor table reads someone else's
  // bones — the bug that put Footprints' prints on her hands.
  assert.equal((src.match(/fn _rzSlot\(/g) ?? []).length, 1)
})

test("RzLight resolves in every module a source is spliced into", () => {
  // Exactly once each: absent is a compile error on a function the author was
  // right to write, twice is a redefinition.
  const CAST = { subjects: 4, samples: 128, base: 12, trailBase: 108, slots: 8, reversedZ: false, alias: [0], trailCount: 1 }
  const P = { wgsl: "fn particleInit(id: u32, s: f32) -> Particle { var q: Particle; return q; }", count: 64, blend: "alpha", bloom: false }
  const modules = {
    field: buildFieldShader({ wgsl: "fn foreground(r: vec3f, uv: vec2f, t: f32, d: f32) -> vec4f { return vec4f(0.0); }", paramsDecl: "", hasBackground: false, hasForeground: true, gridSize: 0 }),
    "particle compute": buildParticleComputeShader(P, CAST),
    "particle render": buildParticleRenderShader(P, CAST),
    trail: buildTrailShader({ wgsl: "fn trailWidth(u: f32, a: f32) -> f32 { return 1.0; }", slots: 1, ribbonSlots: [0], blend: "additive", bloom: true }, CAST),
    emit: buildLightEmitShader(EMIT, API, CAST),
  }
  for (const [name, src] of Object.entries(modules)) {
    assert.equal((src.match(/struct RzLight\b/g) ?? []).length, 1, `${name} must declare RzLight exactly once`)
  }
  assert.match(RZ_LIGHT_STRUCT_WGSL, /struct RzLight/)
})

test("the slot base is a uniform, never baked into the text", () => {
  const src = buildLightEmitShader(EMIT, API, CAST)
  assert.match(src, /u32\(_rzLightU\[0\]\.y\)/)
  // Baking it would mean recompiling every emitting effect whenever a scene
  // gained or lost a document light — a shader rebuild triggered by moving a
  // lamp. The builder takes no base at all, so it cannot regress to that.
  assert.doesNotMatch(buildLightEmitShader(EMIT, API, CAST), /\+ \d+u\) \* 8u/)
})

test("the writable view of the lights buffer exists only in the emit stage", () => {
  // Everything that SHADES reads the buffer read-only. One writable binding, in
  // a compute pass that runs before the pass reading it.
  assert.match(buildLightEmitShader(EMIT, API, CAST), /var<storage, read_write> _rzLightsOut/)
  assert.doesNotMatch(COMMON_MATERIAL_PRELUDE_WGSL, /read_write.*_rzLights/)
  assert.doesNotMatch(groundShaderWgsl(), /read_write.*_rzLights/)
})

test("the emit write is sanitized: hosted code cannot poison the frame", () => {
  // lightEmit is USER WGSL, and this buffer feeds every fragment of every
  // material — one NaN position would poison the whole frame, and WGSL leaves
  // max(NaN, 0) indeterminate, so it would not even fail the same way on every
  // GPU. The one write site checks, and a light that fails writes zeros.
  const src = buildLightEmitShader(EMIT, API, CAST)
  assert.match(src, /let finite = l\.pos\.x == l\.pos\.x/, "the NaN self-equality check must guard the write")
  assert.match(src, /select\(vec3f\(0\.0\), max\(l\.color \* l\.intensity, vec3f\(0\.0\)\), finite\)/,
    "colour must be clamped at zero — this layer is additive, and negative light darkens")
  assert.match(src, /select\(0\.0, max\(l\.radius, 0\.0\), finite && _rzLightU\[0\]\.w > 0\.0\)/)
})

test("the header has exactly one writer", () => {
  // setLights used to upload the header region too, leaving two CPU mirrors of
  // the count — correct on the GPU only by queue ordering, and a trap on the
  // CPU: the first path that uploads lightsData whole would zero the count.
  const engineSrc = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  const at = engineSrc.indexOf("setLights(")
  const body = engineSrc.slice(at, engineSrc.indexOf("\n  }", at))
  assert.match(body, /LIGHT_HEADER \* 4,\s*\n\s*this\.lightsData\.buffer/, "setLights must write records only, offset past the header")
  // TWO offset-0 writes are legitimate: the one-time zero-fill at buffer
  // creation (count 0, before anything renders) and allocateLightSlots. A
  // third is someone writing the header from a new place — the trap returning.
  const writes = [...engineSrc.matchAll(/writeBuffer\(\s*this\.lightsBuffer,\s*0,/g)].length
  assert.equal(writes, 2, `offset-0 lights-buffer writes: want init zero-fill + allocateLightSlots only, found ${writes}`)
})

test("a point light is a spot with no cone", () => {
  // The branchless trick the whole loop rests on: aim (0,0,0) with both cosines
  // at -1 divides by the floor and clamps to 1, so a point light pays one dot
  // product and takes the SAME path a spot does. If this drifts, every point
  // light in every scene goes dark at once.
  const light = { pos: [0, 0, 0], radius: 5 }
  const plain = contribution(light, [0, 0, 1], [0, 0, -1])
  const spelled = contribution({ ...light, aim: [0, 0, 0], cone: [-1, -1] }, [0, 0, 1], [0, 0, -1])
  assert.equal(plain, spelled)
  assert.ok(plain > 0)
})

test("a spot lights inside its cone and nothing outside it", () => {
  // A 60° lamp at the origin aimed along +z, and a wall two units down its axis.
  const light = { pos: [0, 0, 0], radius: 10, aim: [0, 0, 1], cone: coneOf(60) }
  const n = [0, 0, -1]
  const axis = contribution(light, [0, 0, 2], n)
  const inside = contribution(light, [0.3, 0, 2], n)
  const edge = contribution(light, [1.15, 0, 2], n) // ~30° off, the outer edge
  const outside = contribution(light, [4, 0, 2], n) // ~63°, past it
  assert.ok(axis > 0, "the axis is lit")
  assert.ok(inside > 0 && inside <= axis, "inside the cone, no brighter than the axis")
  assert.ok(edge < axis * 0.5, `the edge falls off (edge ${edge.toFixed(4)} vs axis ${axis.toFixed(4)})`)
  assert.equal(outside, 0, "past the outer angle, exactly nothing")
})

test("a spot aimed away lights nothing in front of it", () => {
  const light = { pos: [0, 0, 0], radius: 10, aim: [0, 0, -1], cone: coneOf(60) }
  assert.equal(contribution(light, [0, 0, 2], [0, 0, -1]), 0)
})

test("the cap clears a game stage's rig", () => {
  // A Unity stage arrives with a lamp per fixture: a resort brought 33, a lit
  // interior brings past a hundred. The loop runs over the scene's count, so
  // this bounds the buffer and the worst case rather than the ordinary one, and
  // the buffer is storage — 128 records is 8 KiB.
  assert.ok(MAX_LIGHTS >= 128, `MAX_LIGHTS is ${MAX_LIGHTS}`)
  // Header, 128 records, and 32k cells of four words — about half a megabyte,
  // bound once to every material.
  assert.equal(LIGHTS_FLOATS * 4, 532544)
})

test("the record holds a spot's aim and cone where the writer puts them", () => {
  // engine.ts writes aim at b+8..10 and the cosines at b+11, b+12. These are the
  // reads; drift shows up as a spot pointing somewhere else.
  // Word 8 is the third vec4's x; 11 its w; 12 the fourth vec4's x.
  assert.match(wgsl, /fn rzLightAim\(i: u32\) -> vec3f \{ return _rzLightVec\(i, 2u\)\.xyz; \}/)
  assert.match(wgsl, /fn rzLightCone\(i: u32\) -> vec2f \{ return vec2f\(_rzLightVec\(i, 2u\)\.w, _rzLightVec\(i, 3u\)\.x\); \}/)
  const body = wgsl.slice(wgsl.indexOf("fn _rzLightOne"))
  assert.match(body, /if \(dist >= pr\.w\) \{ return vec3f\(0\.0\); \}/, "out of reach is skipped before the rest")
  assert.match(body, /aim \* aim/, "the cone edge is squared like the falloff")
})

// ── The grid ──

/** A lamp as setLights stores it, and as the grid builder reads it back. */
function lamp(pos, radius, aim, deg) {
  if (!aim) return { x: pos[0], y: pos[1], z: pos[2], radius, ax: 0, ay: 0, az: 0, cosOuter: -1, pos, cone: [-1, -1] }
  const len = Math.hypot(...aim)
  const a = aim.map((v) => v / len)
  const cone = coneOf(deg)
  return { x: pos[0], y: pos[1], z: pos[2], radius, ax: a[0], ay: a[1], az: a[2], cosOuter: cone[0], pos, aim: a, cone }
}

/** The shader's lookup, in f32 where the shader is: the header stores the
 *  origin and 1/cell as floats. */
function maskAt(grid, p) {
  const f = Math.fround
  const inv = f(1 / grid.cell)
  const c = [0, 1, 2].map((k) => Math.floor(f(f(p[k] - f(grid.origin[k])) * inv)))
  const inside = c.every((v, k) => v >= 0 && v < grid.dims[k])
  if (!inside) return grid.outside
  const at = ((c[2] * grid.dims[1] + c[1]) * grid.dims[0] + c[0]) * LIGHT_MASK_WORDS
  return grid.cells.subarray(at, at + LIGHT_MASK_WORDS)
}
const has = (mask, i) => ((mask[i >> 5] >>> (i & 31)) & 1) === 1

/** Deterministic, so a failure reproduces. */
function rng(seed) {
  let s = seed >>> 0
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296)
}

test("the grid never drops a lamp from a point it lights", () => {
  // The one property the grid must hold: conservative. A bit missing from a
  // cell is a hole in the light; a spare bit costs a few instructions.
  // x340-shaped: a cluster of small lamps, a few mid-sized, and stage spots
  // reaching hundreds of units from far outside the cluster.
  const r = rng(7)
  const lamps = []
  for (let i = 0; i < 40; i++) lamps.push(lamp([r() * 60 - 30, r() * 10, r() * 60 - 30], 3 + r() * 8))
  for (let i = 0; i < 6; i++) lamps.push(lamp([r() * 200 - 100, 20 + r() * 20, r() * 200 - 100], 20 + r() * 30))
  for (let i = 0; i < 6; i++) {
    lamps.push(lamp([-120 + r() * 40, 50, 60 + r() * 60], 307, [0.6 + r() * 0.2, -0.7, r() * 0.6 - 0.3], 40 + r() * 50))
  }
  const grid = buildLightGrid(lamps)
  let lit = 0
  for (let s = 0; s < 40000; s++) {
    const p = [r() * 500 - 250, r() * 80 - 10, r() * 500 - 250]
    const m = maskAt(grid, p)
    lamps.forEach((l, i) => {
      // Facing the lamp head-on, so N·L never hides a missing bit.
      const d = [l.x - p[0], l.y - p[1], l.z - p[2]]
      const len = Math.hypot(...d) || 1
      if (contribution(l, p, d.map((v) => v / len)) > 0) {
        lit++
        assert.ok(has(m, i), `lamp ${i} lights (${p.map((v) => v.toFixed(2))}) but its bit is missing`)
      }
    })
  }
  assert.ok(lit > 10000, `the sample must actually hit lit points (hit ${lit})`)
})

test("the grid actually separates lamps", () => {
  // Conservative is trivially satisfied by setting every bit; that would be
  // correct and exactly as slow as before. Two lamps far apart must not share
  // the cells around either one.
  const grid = buildLightGrid([lamp([0, 0, 0], 5), lamp([100, 0, 0], 5)])
  assert.ok(has(maskAt(grid, [1, 0, 0]), 0) && !has(maskAt(grid, [1, 0, 0]), 1))
  assert.ok(has(maskAt(grid, [99, 0, 0]), 1) && !has(maskAt(grid, [99, 0, 0]), 0))
  // A spot claims the cells in its beam and not the ones behind it.
  const spot = buildLightGrid([lamp([0, 0, 0], 50, [0, -1, 0], 30), lamp([0, 0, 0], 5)])
  assert.ok(has(maskAt(spot, [0, -20, 0]), 0), "down the beam")
  assert.ok(!has(maskAt(spot, [0, 20, 0]), 0), "behind the lamp")
})

test("the grid stays inside its cell budget and places a lone lamp", () => {
  const r = rng(3)
  const many = Array.from({ length: 128 }, () => lamp([r() * 1000, r() * 1000, r() * 1000], 1 + r() * 400))
  const grid = buildLightGrid(many)
  assert.ok(grid.dims[0] * grid.dims[1] * grid.dims[2] <= LIGHT_GRID_CELLS)
  assert.equal(grid.cells.length, grid.dims[0] * grid.dims[1] * grid.dims[2] * LIGHT_MASK_WORDS)
  const one = buildLightGrid([lamp([5, 5, 5], 2)])
  assert.ok(has(maskAt(one, [5, 5, 5]), 0))
})

test("no lamps is an empty grid, and a lamp with no reach sets nothing", () => {
  const none = buildLightGrid([])
  assert.deepEqual(none.dims, [0, 0, 0])
  assert.equal(none.cells.length, 0)
  const dark = buildLightGrid([lamp([0, 0, 0], 0), lamp([10, 0, 0], 4)])
  assert.ok(!has(maskAt(dark, [0, 0, 0]), 0))
  assert.ok(!has(dark.outside, 0))
})

test("the shader and the builder agree on the cell order", () => {
  // x fastest, then y, then z — the builder's index and the shader's. Drift
  // here shows as lamps lighting the wrong corner of the stage.
  assert.match(wgsl, /\(ci\.z \* dims\.y \+ ci\.y\) \* dims\.x \+ ci\.x/)
  assert.match(wgsl, new RegExp(`_rzLights\\[${LIGHT_GRID_BASE / 4}u \\+ `))
  const src = readFileSync(new URL("../src/light-grid.ts", import.meta.url), "utf8")
  assert.match(src, /\(\(z \* dims\[1\] \+ y\) \* dims\[0\] \+ x\) \* LIGHT_MASK_WORDS/)
})
