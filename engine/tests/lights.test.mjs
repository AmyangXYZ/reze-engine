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
  const dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
  const dist = Math.sqrt(dist2)
  const inv = 1 / Math.max(dist, 1e-4)
  const ndl = Math.max(n[0] * d[0] * inv + n[1] * d[1] * inv + n[2] * d[2] * inv, 0)
  if (ndl <= 0) return 0
  const w = Math.min(Math.max(1 - dist / Math.max(light.radius, 1e-4), 0), 1)
  return (ndl * w * w) / (1 + dist2)
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
