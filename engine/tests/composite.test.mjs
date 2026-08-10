// Composite-template tests. Run: npm test.
//
// buildCompositeShader assembles WGSL by string substitution, and a substitution
// that silently fails to fire produces a shader that either does nothing or does
// not compile — neither of which any unit test elsewhere would notice. So the
// contract asserted here is: every placeholder is consumed in every variant, the
// mounts the source declares are the mounts that get called, and the background
// block's condition is the right one for each combination.

import { test } from "node:test"
import assert from "node:assert/strict"
import { buildCompositeShader, COMPOSITE_SHADER_WGSL } from "../dist/shaders/passes/composite.js"

const BACKGROUND = "fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f { return vec4f(0.0); }"
const FOREGROUND = "fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f { return vec4f(0.0); }"

const source = (wgsl, hasBackground, hasForeground) =>
  buildCompositeShader({ wgsl, paramsDecl: "", hasBackground, hasForeground })

/** Anything of the form WORD_WORD left in the emitted WGSL is an unfired
 *  substitution — real WGSL in this file is lower/camel case. */
const PLACEHOLDER = /\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g
/** The template's own compile-time constants, which legitimately look like one. */
const ALLOWED = new Set(["APPLY_GAMMA", "FILMIC_LUT_W"])

// Comments are prose and name things that don't exist in this file (engine-side
// constants, the placeholders themselves) — scan the code the GPU sees.
const leftovers = (src) =>
  [...(src.replace(/\/\/[^\n]*/g, "").match(PLACEHOLDER) ?? [])].filter((w) => !ALLOWED.has(w))

test("every variant consumes every placeholder", () => {
  assert.deepEqual(leftovers(COMPOSITE_SHADER_WGSL), [], "base pass")
  assert.deepEqual(leftovers(source(BACKGROUND, true, false)), [], "background only")
  assert.deepEqual(leftovers(source(FOREGROUND, false, true)), [], "foreground only")
  assert.deepEqual(leftovers(source(`${BACKGROUND}\n${FOREGROUND}`, true, true)), [], "both mounts")
})

test("the declared mounts are the ones called", () => {
  const calls = (src) => ({
    background: /clamp\(background\(/.test(src),
    foreground: /clamp\(foreground\(/.test(src),
  })
  assert.deepEqual(calls(COMPOSITE_SHADER_WGSL), { background: false, foreground: false })
  assert.deepEqual(calls(source(BACKGROUND, true, false)), { background: true, foreground: false })
  assert.deepEqual(calls(source(FOREGROUND, false, true)), { background: false, foreground: true })
  assert.deepEqual(calls(source(`${BACKGROUND}\n${FOREGROUND}`, true, true)), { background: true, foreground: true })
})

test("the foreground is handed the scene's depth", () => {
  const src = source(FOREGROUND, false, true)
  assert.match(src, /foreground\(dir, fgUv, viewU\[6\]\.x, linearDepth\(coord\)\)/)
})

test("a foreground alone leaves the background block gated on the equirect", () => {
  // Nothing was injected under the scene, so the block must stay skippable —
  // regressing this would run the equirect branch on every frame for an effect
  // that has no business there.
  assert.match(source(FOREGROUND, false, true), /if \(bg\.w > 1\.5 && sceneAlpha < 0\.999\) \{/)
})

test("a background effect keeps the coverage gate, and derivatives forfeit it", () => {
  // Derivative builtins are illegal in non-uniform control flow, so an effect
  // using them has to run unconditionally or the pipeline fails to compile.
  assert.match(source(BACKGROUND, true, false), /if \(sceneAlpha < 0\.999\) \{/)
  const derivative = "fn background(r: vec3f, uv: vec2f, t: f32) -> vec4f { return vec4f(fwidth(uv.x)); }"
  assert.match(source(derivative, true, false), /if \(true\) \{/)

  // The test is textual over the whole file, so a foreground's derivative costs
  // the background its gate. Conservative, and only ever toward correctness.
  const mixed = `${BACKGROUND}\nfn foreground(r: vec3f, uv: vec2f, t: f32, d: f32) -> vec4f { return vec4f(dpdx(uv.x)); }`
  assert.match(source(mixed, true, true), /if \(true\) \{/)
})

test("user code and its params land ahead of the entry points", () => {
  const src = buildCompositeShader({
    wgsl: BACKGROUND,
    paramsDecl: "struct EffectParams {\n  density: f32,\n}\n",
    hasBackground: true,
    hasForeground: false,
  })
  assert.ok(src.indexOf("struct EffectParams") < src.indexOf(BACKGROUND), "params declared before the user's code")
  assert.ok(src.indexOf(BACKGROUND) < src.indexOf("@fragment"), "user's code before the entry point that calls it")
})
