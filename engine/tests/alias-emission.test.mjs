// Every assembled effect module that CALLS _rzSlot must DEFINE it, once. Run: npm test.
//
// The engine's shaders are strings assembled from pieces in different files, so
// "function referenced but never defined" is not a compile error anywhere until
// a real GPU device sees the finished module — at which point the symptom is
// "effects fail to install" with a WGSL diagnostic naming a helper the author
// never wrote. That exact hole shipped once: the composite accessors were routed
// through _rzSlot and the definition was spliced into particles and trails but
// not into composite, which took down every effect install. This test assembles
// each module the way the engine does and checks the contract at the string
// level, where it is cheap and headless.

import { test } from "node:test"
import assert from "node:assert/strict"
import { buildCompositeShader, buildFieldShader } from "../dist/shaders/passes/composite.js"
import { buildParticleComputeShader, buildParticleRenderShader } from "../dist/shaders/passes/particles.js"
import { buildTrailShader } from "../dist/shaders/passes/trails.js"

const effect = {
  wgsl: "fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f { return vec4f(0.0); }",
  paramsDecl: "",
  hasBackground: true,
  hasForeground: false,
  simSize: 0,
}
const cast = { subjects: 4, samples: 128, base: 12, trailBase: 108, slots: 8, alias: [0, 1], reversedZ: false }
const particleSrc = { wgsl: "", count: 64, blend: "alpha", bloom: false }
const trailSrc = { wgsl: "", slots: 1, ribbonSlots: [0], blend: "additive", bloom: false }

const defs = (s) => (s.match(/fn _rzSlot\(/g) ?? []).length
const refs = (s) => (s.match(/_rzSlot\(/g) ?? []).length - defs(s)

for (const [name, code] of [
  ["composite (no effect)", buildCompositeShader(null)],
  ["composite (with effect)", buildCompositeShader(effect)],
  ["field layer", buildFieldShader(effect)],
  ["particle compute", buildParticleComputeShader(particleSrc, cast)],
  ["particle render", buildParticleRenderShader(particleSrc, cast)],
  ["trail shader", buildTrailShader(trailSrc, cast)],
]) {
  test(`${name}: _rzSlot is defined exactly once wherever it is called`, () => {
    assert.ok(refs(code) > 0, "the accessors should route through the alias — if this fails, the aliasing was removed")
    assert.equal(defs(code), 1, `${defs(code)} definitions of _rzSlot — a module needs exactly one`)
  })
}

test("the ribbon occlusion compare follows the depth convention", () => {
  // Manual WGSL compare — the one depth test the pipeline-level depthCompare
  // flip cannot reach. Unflipped on a reversed-Z device it drew ribbons only
  // when OCCLUDED, which shipped once and looked like ribbons simply gone.
  const fwd = buildTrailShader(trailSrc, cast)
  const rev = buildTrailShader(trailSrc, { ...cast, reversedZ: true })
  assert.match(fwd, /in\.clip\.z > sceneD/)
  assert.match(rev, /in\.clip\.z < sceneD/)
})

test("the assembled trail module resolves ribbon -> local slot -> scene slot", () => {
  // The full chain in one string, which is the only place all three index
  // spaces meet. A mixed file: one ribbon, belonging to local anchor 1,
  // which the scene table has placed at slot 3.
  const mixed = buildTrailShader(
    { ...trailSrc, slots: 1, ribbonSlots: [1] },
    { ...cast, alias: [7, 3] },
  )
  assert.match(mixed, /fn _rzRibbonSlot\(ribbon: i32\)/, "ribbon -> local must be emitted")
  assert.match(mixed, /case 0: \{ return 1; \}/, "ribbon 0 belongs to local anchor 1")
  assert.match(mixed, /case 1: \{ return 3; \}/, "local anchor 1 lives at scene slot 3")
  assert.match(mixed, /let slot = _rzRibbonSlot\(ribbon\);/, "the vertex shader must go through the mapping")
})
