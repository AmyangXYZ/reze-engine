// `#textures N`: pictures the host hands an effect, read as rzTexture(i, uv) in
// the particle shading. Run: npm test.
//
// No GPU here, so the WGSL is checked as text: the directive parses, the render
// stage binds exactly N textures and one sampler at their slots, and every
// other module an effect lands in resolves the names as stubs — and an effect
// that declares none carries no bindings at all.

import { test } from "node:test"
import assert from "node:assert/strict"
import { parseDirectives } from "../dist/shaders/directives.js"
import { textureApi } from "../dist/shaders/texture-api.js"
import {
  buildParticleComputeShader,
  buildParticleRenderShader,
  PARTICLE_TEXTURE_BINDING,
  PARTICLE_TEXTURE_SAMPLER_BINDING,
} from "../dist/shaders/passes/particles.js"
import { buildTrailShader } from "../dist/shaders/passes/trails.js"
import { buildFieldShader, EFFECT_SCENE_API } from "../dist/shaders/passes/composite.js"
import { buildSimShader } from "../dist/shaders/passes/grid.js"
import { buildLightEmitShader } from "../dist/shaders/lights.js"
import { anchorAliasWgsl } from "../dist/shaders/anchor-table.js"

test("#textures takes a count from 1 to 4, and nothing declares none", () => {
  assert.equal(parseDirectives("#textures 2").directives.textures, 2)
  assert.equal(parseDirectives("// no pictures").directives.textures, 0)
  for (const bad of ["#textures 0", "#textures 5", "#textures 1.5", "#textures x", "#textures"]) {
    assert.ok(parseDirectives(bad).errors.length > 0, `${bad} is an error`)
  }
})

test("the render stage binds exactly N textures and one sampler; others get stubs", () => {
  const CAST = { subjects: 4, samples: 128, base: 12, trailBase: 108, slots: 8, alias: [0], trailCount: 1 }
  const P = (n) => ({ wgsl: "fn particleInit(id: u32, s: f32) -> Particle { var q: Particle; return q; }", count: 64, blend: "alpha", bloom: false, paramsDecl: "", gridSize: 0, cover: false, textures: n })
  const render = buildParticleRenderShader(P(2), CAST)
  const bindings = [...render.matchAll(/@binding\((\d+)\) var _rzTex(\d+): texture_2d<f32>/g)].map((m) => Number(m[1]))
  assert.deepEqual(bindings, [PARTICLE_TEXTURE_BINDING, PARTICLE_TEXTURE_BINDING + 1])
  assert.ok(render.includes(`@binding(${PARTICLE_TEXTURE_SAMPLER_BINDING}) var _rzTexSampler: sampler;`))
  assert.equal((render.match(/fn rzTexture\(/g) ?? []).length, 1)
  assert.equal((render.match(/fn rzTextureLod\(/g) ?? []).length, 1)

  // none declared: the names resolve, and nothing is bound
  const plain = buildParticleRenderShader(P(0), CAST)
  assert.ok(!/_rzTex/.test(plain), "no bindings without #textures")
  assert.equal((plain.match(/fn rzTexture\(/g) ?? []).length, 1)

  const elsewhere = {
    "particle compute": buildParticleComputeShader(P(2), CAST),
    field: buildFieldShader({ wgsl: "fn foreground(r: vec3f, uv: vec2f, t: f32, d: f32) -> vec4f { return vec4f(0.0); }", paramsDecl: "", hasBackground: false, hasForeground: true, gridSize: 0 }),
    trail: buildTrailShader({ wgsl: "fn trailWidth(u: f32, a: f32) -> f32 { return 1.0; }", slots: 1, ribbonSlots: [0], blend: "additive", bloom: true }, CAST),
    grid: buildSimShader("fn gridStep(uv: vec2f, prev: vec4f, dt: f32) -> vec4f { return prev; }", 64, CAST),
    emit: buildLightEmitShader("fn lightEmit(i: u32, time: f32) -> RzLight { var l: RzLight; return l; }", EFFECT_SCENE_API + anchorAliasWgsl([0]), CAST),
  }
  for (const [name, code] of Object.entries(elsewhere)) {
    assert.equal((code.match(/fn rzTexture\(/g) ?? []).length, 1, `${name} resolves rzTexture once`)
    assert.equal((code.match(/fn rzTextureLod\(/g) ?? []).length, 1, `${name} resolves rzTextureLod once`)
    assert.ok(!/_rzTex\d/.test(code), `${name} binds no textures`)
  }
})

test("an out-of-range slot reads zero, and every slot is sampled in uniform control flow", () => {
  const api = textureApi(2, 0, 17, 21)
  // both slots sampled unconditionally, then selected
  assert.ok(/let s0 = textureSample\(_rzTex0/.test(api) && /let s1 = textureSample\(_rzTex1/.test(api))
  assert.ok(/var c = vec4f\(0\.0\);/.test(api))
})
