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
import { LIGHT_HEADER, LIGHT_STRIDE, LIGHTS_FLOATS, MAX_LIGHTS, lightsApi } from "../dist/shaders/lights.js"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "../dist/shaders/materials/common.js"
import { groundShaderWgsl } from "../dist/shaders/passes/ground.js"

const wgsl = lightsApi(0, 6)

test("the buffer is exactly the header plus the records", () => {
  assert.equal(LIGHTS_FLOATS, LIGHT_HEADER + MAX_LIGHTS * LIGHT_STRIDE)
  // vec4-aligned, both of them: a later pass wanting to read these as vec4s
  // must not have to repack the buffer to do it.
  assert.equal(LIGHT_HEADER % 4, 0, "the header must not push the records off a vec4 boundary")
  assert.equal(LIGHT_STRIDE % 4, 0, "a record must be a whole number of vec4s")
})

test("the accessors read the slots the writer writes", () => {
  // The engine writes position at b+0..2, radius at b+3, colour at b+4..6.
  // These are the reads. They are in two files and can drift apart in silence —
  // the symptom would be a light with someone else's radius.
  assert.match(wgsl, new RegExp(`fn rzLightPos\\(i: u32\\) -> vec3f \\{\\s*let b = ${LIGHT_HEADER}u \\+ i \\* ${LIGHT_STRIDE}u;`))
  assert.match(wgsl, new RegExp(`fn rzLightRadius\\(i: u32\\) -> f32 \\{ return _rzLights\\[${LIGHT_HEADER}u \\+ i \\* ${LIGHT_STRIDE}u \\+ 3u\\]`))
  assert.match(wgsl, new RegExp(`let b = ${LIGHT_HEADER}u \\+ i \\* ${LIGHT_STRIDE}u \\+ 4u;`))
})

test("the count is clamped in the shader, not only by the writer", () => {
  // The buffer is fixed size. A count past the cap — a stale write, a caller
  // reaching in — would read past the records into whatever follows, so the
  // shader clamps rather than trusting the number it was handed.
  assert.match(wgsl, /return min\(u32\(_rzLights\[0\]\), RZ_MAX_LIGHTS\)/)
})

test("the loop is bounded by the count, so zero lights runs nothing", () => {
  const body = wgsl.slice(wgsl.indexOf("fn rzLightsDiffuse"))
  assert.match(body, /let count = rzLightCount\(\);/)
  assert.match(body, /for \(var i = 0u; i < count; i = i \+ 1u\)/)
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
    assert.match(src, /@group\(0\) @binding\(6\) var<storage, read> _rzLights: array<f32>;/)
  }
})

/** The shader's falloff, reimplemented against the same constants. */
function contribution(light, p, n) {
  const d = [light.pos[0] - p[0], light.pos[1] - p[1], light.pos[2] - p[2]]
  const dist = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
  const inv = 1 / Math.max(dist, 1e-4)
  const ndl = Math.max(n[0] * d[0] * inv + n[1] * d[1] * inv + n[2] * d[2] * inv, 0)
  if (ndl <= 0) return 0
  const t = Math.min(Math.max(dist / Math.max(light.radius, 1e-4), 0), 1)
  const falloff = 1 - t * t
  return ndl * falloff * falloff
}

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

import { RZ_LIGHT_STRUCT_WGSL, buildLightEmitShader, hasLightEmit, parseLightCount } from "../dist/shaders/lights.js"
import { EFFECT_SCENE_API, buildFieldShader } from "../dist/shaders/passes/composite.js"
import { anchorAliasWgsl } from "../dist/shaders/anchor-table.js"

/** What the engine hands the builder: the scene API plus this effect's alias. */
const API = EFFECT_SCENE_API + anchorAliasWgsl([0])
import { buildParticleComputeShader, buildParticleRenderShader } from "../dist/shaders/passes/particles.js"
import { buildTrailShader } from "../dist/shaders/passes/trails.js"

const EMIT = `// @lights 3
fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  l.pos = vec3f(f32(i) * 2.0, 10.0 + time, 0.0);
  l.color = vec3f(1.0, 0.5, 0.2);
  l.intensity = 3.0;
  l.radius = 20.0;
  return l;
}`

test("an effect declares how many lights it emits", () => {
  assert.equal(parseLightCount(EMIT, MAX_LIGHTS), 3)
  assert.equal(parseLightCount("// nothing here", MAX_LIGHTS), 0)
  // Clamped, not rejected: the same choice @particles makes.
  assert.equal(parseLightCount("// @lights 999", MAX_LIGHTS), MAX_LIGHTS)
  // Mid-sentence prose must not declare anything, the @anchor rule.
  assert.equal(parseLightCount("// mentioning @lights 4 in a sentence", MAX_LIGHTS), 0)
  assert.equal(hasLightEmit(EMIT), true)
  assert.equal(hasLightEmit("fn background() {}"), false)
})

test("the emit shader writes the slots the material shader reads", () => {
  const src = buildLightEmitShader(EMIT, API)
  // Same stride and header on both sides of the buffer, expressed against the
  // same constants — this is the seam where a writer and a reader drift.
  assert.match(src, new RegExp(`let b = ${LIGHT_HEADER}u \\+ \\(u32\\(_rzLightU\\.y\\) \\+ i\\) \\* ${LIGHT_STRIDE}u;`))
  assert.match(src, /_rzLightsOut\[b \+ 3u\] = select\(0\.0, max\(l\.radius, 0\.0\), finite\);/)
  // Colour x intensity, the same product the CPU writer stores — through the
  // sanitized local, since the raw product is what the guard exists to check.
  assert.match(src, /_rzLightsOut\[b \+ 4u\] = c\.x;/)
  assert.ok(src.includes(EMIT), "the author's source is spliced in verbatim")
})

test("the emit shader guards its dispatch tail", () => {
  const src = buildLightEmitShader(EMIT, API)
  // A workgroup is 64 wide and a count rarely is. Without this the tail threads
  // write into whatever slots follow — another effect's lights, silently.
  assert.match(src, /if \(i >= u32\(_rzLightU\.z\)\) \{ return; \}/)
})

test("time is a PARAMETER, so the same source compiles in every module", () => {
  // An effect that emits lights AND draws something has its whole source
  // spliced into the field, particle, trail or grid module too — where
  // lightEmit is dead code that still has to resolve. Those modules already
  // define rzTime differently or not at all, so lightEmit must not need it.
  assert.match(buildLightEmitShader(EMIT, API), /let l = lightEmit\(i, _rzLightU\.x\);/)
  assert.doesNotMatch(buildLightEmitShader(EMIT, API), /fn rzTime\(\)/)
})

test("the emit stage can read the cast, so a lamp can aim at someone", () => {
  // Stage Lights points its beams at rzSubject().root. A light that could not
  // ask where she is could only sit where the fixture hangs, which is the one
  // place a follow-spot never is.
  const src = buildLightEmitShader(EMIT, API)
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
    emit: buildLightEmitShader(EMIT, API),
  }
  for (const [name, src] of Object.entries(modules)) {
    assert.equal((src.match(/struct RzLight\b/g) ?? []).length, 1, `${name} must declare RzLight exactly once`)
  }
  assert.match(RZ_LIGHT_STRUCT_WGSL, /struct RzLight/)
})

test("the slot base is a uniform, never baked into the text", () => {
  const src = buildLightEmitShader(EMIT, API)
  assert.match(src, /u32\(_rzLightU\.y\)/)
  // Baking it would mean recompiling every emitting effect whenever a scene
  // gained or lost a document light — a shader rebuild triggered by moving a
  // lamp. The builder takes no base at all, so it cannot regress to that.
  assert.doesNotMatch(buildLightEmitShader(EMIT, API), /\+ \d+u\) \* 8u/)
})

test("the writable view of the lights buffer exists only in the emit stage", () => {
  // Everything that SHADES reads the buffer read-only. One writable binding, in
  // a compute pass that runs before the pass reading it.
  assert.match(buildLightEmitShader(EMIT, API), /var<storage, read_write> _rzLightsOut/)
  assert.doesNotMatch(COMMON_MATERIAL_PRELUDE_WGSL, /read_write.*_rzLights/)
  assert.doesNotMatch(groundShaderWgsl(), /read_write.*_rzLights/)
})

test("the emit write is sanitized: hosted code cannot poison the frame", () => {
  // lightEmit is USER WGSL, and this buffer feeds every fragment of every
  // material — one NaN position would poison the whole frame, and WGSL leaves
  // max(NaN, 0) indeterminate, so it would not even fail the same way on every
  // GPU. The one write site checks, and a light that fails writes zeros.
  const src = buildLightEmitShader(EMIT, API)
  assert.match(src, /let finite = l\.pos\.x == l\.pos\.x/, "the NaN self-equality check must guard the write")
  assert.match(src, /select\(vec3f\(0\.0\), max\(l\.color \* l\.intensity, vec3f\(0\.0\)\), finite\)/,
    "colour must be clamped at zero — this layer is additive, and negative light darkens")
  assert.match(src, /select\(0\.0, max\(l\.radius, 0\.0\), finite\)/)
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
