// `particleOrient(p, id) -> mat3x3f`: a particle quad laid in the author's own
// plane instead of facing the camera. Run: npm test.
//
// No GPU here, so the WGSL is checked as text: the entry point is detected by
// its declaration, the vertex stage takes the quad's axes from it only when it
// is declared, and an effect without one keeps the camera-facing billboard.

import { test } from "node:test"
import assert from "node:assert/strict"
import { buildParticleRenderShader, particleEntryPoints } from "../dist/shaders/passes/particles.js"

const CAST = { subjects: 4, samples: 128, base: 12, trailBase: 108, slots: 8, alias: [0], trailCount: 0 }
const BODY = `fn particleInit(id: u32, s: f32) -> Particle { var q: Particle; return q; }
fn particleStep(p: Particle, dt: f32) -> Particle { return p; }
fn particleShade(p: Particle, uv: vec2f) -> vec4f { return vec4f(1.0); }`
const ORIENT = `
fn particleOrient(p: Particle, id: u32) -> mat3x3f { return mat3x3f(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0)); }`
const src = (wgsl) => ({ wgsl, count: 16, blend: "alpha", bloom: false, paramsDecl: "", gridSize: 0, cover: false, orient: particleEntryPoints(wgsl).orient })

test("particleOrient is found by its declaration, not by a call or a mention", () => {
  assert.equal(particleEntryPoints(BODY + ORIENT).orient, true)
  assert.equal(particleEntryPoints(BODY).orient, false)
  assert.equal(particleEntryPoints(BODY + "\n// particleOrient(p, id) would lay it flat").orient, false)
})

test("the quad takes the author's axes only when the effect declares them", () => {
  const laid = buildParticleRenderShader(src(BODY + ORIENT), CAST)
  assert.match(laid, /let o = particleOrient\(p, ii\);\s*right = o\[0\];\s*up = o\[1\];/)
  const facing = buildParticleRenderShader(src(BODY), CAST)
  assert.ok(!/particleOrient\(p, ii\)/.test(facing), "a billboard never calls it")
  // both keep the camera's axes as the default the author's replace
  for (const code of [laid, facing]) assert.match(code, /var right = vec3f\(cam\.view\[0\]\[0\]/)
})
