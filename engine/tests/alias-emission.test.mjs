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
import { buildSimShader } from "../dist/shaders/passes/grid.js"
import { buildLightEmitShader } from "../dist/shaders/lights.js"
import { EFFECT_SCENE_API } from "../dist/shaders/passes/composite.js"
import { anchorAliasWgsl } from "../dist/shaders/anchor-table.js"

const effect = {
  wgsl: "fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f { return vec4f(0.0); }",
  paramsDecl: "",
  hasBackground: true,
  hasForeground: false,
  simSize: 0,
  trailCount: 0,
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
  // The two modules added after this guard was written — and the sim module is
  // the cautionary tale: absent from this list, its missing _rzSlot shipped and
  // every gridStep effect failed to install until the WGSL validator caught it.
  ["grid step", buildSimShader("fn gridStep(uv: vec2f) -> vec4f { return vec4f(rzAnchor(0, 0).pos, 0.0); }", 256, { ...cast, trailCount: 0 })],
  ["light emit", buildLightEmitShader("#lights 1\nfn lightEmit(i: u32, t: f32) -> RzLight { var l: RzLight; return l; }", EFFECT_SCENE_API + anchorAliasWgsl([0]), { trailCount: 0 })],
]) {
  test(`${name}: _rzSlot is defined exactly once wherever it is called`, () => {
    assert.ok(refs(code) > 0, "the accessors should route through the alias — if this fails, the aliasing was removed")
    assert.equal(defs(code), 1, `${defs(code)} definitions of _rzSlot — a module needs exactly one`)
  })
}

test("ribbons draw as scene geometry: bloom mask written, no hand-rolled depth", () => {
  // This used to assert the manual compare matched the depth convention —
  // unflipped on a reversed-Z device it drew ribbons only when OCCLUDED, which
  // shipped once and looked like ribbons simply gone. That compare is gone with
  // the layer it belonged to: ribbons draw inside the scene pass now, so the
  // hardware tests depth from the pipeline and the trap cannot recur.
  //
  // What the shader must still do is write the AUX target. Mask 1 is what puts
  // a ribbon through the bloom gate, and without it they would draw in HDR and
  // never bloom — which is the entire reason they were moved.
  const src = buildTrailShader(trailSrc, cast)
  assert.doesNotMatch(src, /textureLoad\(sceneDepth/, "hardware depth replaced the hand-rolled compare")
  assert.match(src, /@location\(1\) aux/, "ribbons must write the scene's aux target")
  // And the mask is the AUTHOR's call, as it already is for particles: #bloom
  // opts in. A ribbon that never asked to bloom must not start blooming just
  // because it moved into the scene pass.
  assert.match(src, /o\.aux = vec4f\(0\.0/, "no #bloom means no bloom mask")
  const lit = buildTrailShader({ ...trailSrc, bloom: true }, cast)
  assert.match(lit, /o\.aux = vec4f\(1\.0/, "#bloom is what puts a ribbon through the gate")
})

test("a FIELD effect's _rzSlot carries its real alias, not the identity", () => {
  // The half of the contract the test above cannot see. It checks that _rzSlot
  // is defined once wherever it is called — which the field module satisfied
  // while defining it as the IDENTITY, hardcoded, forever. That is correct for
  // an effect that owns the anchor table and silently wrong for one sharing it:
  // its slot 0 addressed scene slot 0, which belongs to whichever effect was
  // installed first. Footprints composed after Hand Ribbon read the hand trails
  // and drew its prints on her hands.
  //
  // Only ever visible with TWO anchor-declaring effects, which is why a year of
  // single-effect scenes never showed it.
  const shared = buildFieldShader({
    wgsl: "fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f { return vec4f(0.0); }",
    paramsDecl: "",
    hasBackground: false,
    hasForeground: true,
    simSize: 0,
    alias: [2, 3, 4, 5],
  })
  assert.match(shared, /case 0: \{ return 2; \}/, "local slot 0 must resolve to scene slot 2")
  assert.match(shared, /case 3: \{ return 5; \}/, "local slot 3 must resolve to scene slot 5")
  assert.doesNotMatch(
    shared,
    /fn _rzSlot\(local: i32\) -> i32 \{ return local; \}/,
    "a shared table must not compile to the identity — that is the bug this covers",
  )
  // And an effect that owns the table still folds away to nothing.
  const alone = buildFieldShader({
    wgsl: "fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f { return vec4f(0.0); }",
    paramsDecl: "",
    hasBackground: true,
    hasForeground: false,
    simSize: 0,
    alias: [0, 1],
  })
  assert.match(alone, /fn _rzSlot\(local: i32\) -> i32 \{ return local; \}/, "identity must stay free")
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
