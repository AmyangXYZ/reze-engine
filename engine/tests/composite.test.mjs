// Composite- and field-template tests. Run: npm test.
//
// The two shaders are assembled differently and so fail differently. The
// COMPOSITE is built by string substitution, and a substitution that silently
// fails to fire produces a shader that either does nothing or does not compile —
// neither of which any unit test elsewhere would notice. The FIELD shader is
// where the user's WGSL actually lands, so ordering (params, then user code,
// then the entry points that call it) is its contract.
//
// The division matters and is the thing these tests pin: since the field pass,
// the composite never calls a user mount at all — it samples the layer the field
// pass drew. A test asserting the composite calls background() was asserting the
// old architecture, and stayed red for it.

import { test } from "node:test"
import assert from "node:assert/strict"
import {
  buildCompositeShader,
  buildFieldShader,
  COMPOSITE_SHADER_WGSL,
  parseEffectAnchors,
} from "../dist/shaders/passes/composite.js"

const BACKGROUND = "fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f { return vec4f(0.0); }"
const FOREGROUND = "fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f { return vec4f(0.0); }"

const effect = (wgsl, hasBackground, hasForeground, paramsDecl = "") => ({
  wgsl,
  paramsDecl,
  hasBackground,
  hasForeground,
  gridSize: 0,
})

const source = (wgsl, hasBackground, hasForeground) => buildCompositeShader(effect(wgsl, hasBackground, hasForeground))
const field = (wgsl, hasBackground, hasForeground) => buildFieldShader(effect(wgsl, hasBackground, hasForeground))

/** Anything of the form WORD_WORD left in the emitted WGSL is an unfired
 *  substitution — real WGSL in this file is lower/camel case. */
const PLACEHOLDER = /\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b/g
/** The template's own compile-time constants, which legitimately look like one. */
const ALLOWED = new Set(["APPLY_GAMMA", "FILMIC_LUT_W", "RZ_MAX_ANCHORS", "RZ_TRAIL_SAMPLES", "RZ_GRID_SIZE"])

// Comments are prose and name things that don't exist in this file (engine-side
// constants, the placeholders themselves) — scan the code the GPU sees.
const leftovers = (src) =>
  [...(src.replace(/\/\/[^\n]*/g, "").match(PLACEHOLDER) ?? [])].filter((w) => !ALLOWED.has(w))

test("every variant consumes every placeholder", () => {
  assert.deepEqual(leftovers(COMPOSITE_SHADER_WGSL), [], "base pass")
  assert.deepEqual(leftovers(source(BACKGROUND, true, false)), [], "background only")
  assert.deepEqual(leftovers(source(FOREGROUND, false, true)), [], "foreground only")
  assert.deepEqual(leftovers(source(`${BACKGROUND}\n${FOREGROUND}`, true, true)), [], "both mounts")
  // The field shader interpolates rather than substitutes, but an unfired
  // template hole reads the same way in the output.
  assert.deepEqual(leftovers(field(BACKGROUND, true, false)), [], "field, background only")
  assert.deepEqual(leftovers(field(FOREGROUND, false, true)), [], "field, foreground only")
  assert.deepEqual(leftovers(field(`${BACKGROUND}\n${FOREGROUND}`, true, true)), [], "field, both mounts")
})

test("the declared mounts are the ones called, in the field pass", () => {
  const calls = (src) => ({
    background: /clamp\(background\(/.test(src),
    foreground: /clamp\(foreground\(/.test(src),
  })
  assert.deepEqual(calls(field(BACKGROUND, true, false)), { background: true, foreground: false })
  assert.deepEqual(calls(field(FOREGROUND, false, true)), { background: false, foreground: true })
  assert.deepEqual(calls(field(`${BACKGROUND}\n${FOREGROUND}`, true, true)), { background: true, foreground: true })

  // And the composite calls NEITHER, in any variant: it samples the layer the
  // field pass drew. Calling a mount here as well would run the effect twice.
  for (const [name, src] of [
    ["base", COMPOSITE_SHADER_WGSL],
    ["background", source(BACKGROUND, true, false)],
    ["foreground", source(FOREGROUND, false, true)],
    ["both", source(`${BACKGROUND}\n${FOREGROUND}`, true, true)],
  ]) {
    assert.deepEqual(calls(src), { background: false, foreground: false }, `composite (${name})`)
  }
})

test("the foreground is handed the scene's depth", () => {
  // The field pass stands in for a full-resolution pixel, so the depth read is
  // at the reconstructed full-res coordinate, clamped inside the buffer.
  assert.match(field(FOREGROUND, false, true), /foreground\(dir, uv, _rzFieldClock\.x, linearDepth\(/)
})

test("a mount is handed ITS OWN clock, never the shared one", () => {
  // viewU[6].x is measured from the FIRST installed effect's epoch, so every
  // later effect started mid-stream — and an effect whose lightEmit read its
  // own epoch disagreed with its own background() about what time it was.
  // Nothing in the field module may reach for it again.
  for (const [what, src] of [
    ["background", field(BACKGROUND, true, false)],
    ["foreground", field(FOREGROUND, false, true)],
  ]) {
    assert.match(src, /_rzFieldClock\.x/, `${what} must read the per-effect clock`)
    // BOTH comment forms stripped: the note explaining why this rule exists
    // names viewU[6].x, and a stripper that only handles // would fail on the
    // documentation of the very thing it is checking.
    const code = src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "")
    assert.doesNotMatch(code, /viewU\[6\]\.x/, `${what} must not read the shared clock`)
  }
})

test("a foreground alone leaves the background block gated on the equirect", () => {
  // Nothing was injected under the scene, so the block must stay skippable —
  // regressing this would run the equirect branch on every frame for an effect
  // that has no business there.
  assert.match(source(FOREGROUND, false, true), /if \(bg\.w > 1\.5 && sceneAlpha < 0\.999\) \{/)
})

test("the composite is ONE shader, whatever effects are installed", () => {
  // It has no variants left. Effects draw inside the scene pass now — the field
  // layer is blitted there, so nothing in this shader depends on which of them
  // exist. The gate is the equirect's alone, and it is the same in every
  // variant: behind a fully covered pixel the result is multiplied by
  // (1 - alpha) = 0 anyway, and on a full-screen dome that is a third of the
  // frame.
  const derivative = "fn background(r: vec3f, uv: vec2f, t: f32) -> vec4f { return vec4f(fwidth(uv.x)); }"
  const variants = [
    COMPOSITE_SHADER_WGSL,
    source(BACKGROUND, true, false),
    source(FOREGROUND, false, true),
    source(derivative, true, false),
    source(`${BACKGROUND}\n${FOREGROUND}`, true, true),
  ]
  for (const v of variants) {
    assert.match(v, /if \(bg\.w > 1\.5 && sceneAlpha < 0\.999\) \{/)
    assert.equal(v, variants[0], "the composite must not vary with the effect list")
  }
})

test("the composite never samples the field layer", () => {
  // It is drawn INTO the scene pass now. Reading it here as well would draw
  // every field effect twice — and compositing it here at all is what kept the
  // mount from ever reaching the bloom pyramid.
  for (const name of ["fieldBgTex", "fieldFgTex", "fieldBgHalfTex", "fieldFgHalfTex", "rzFieldMerge"]) {
    assert.ok(!COMPOSITE_SHADER_WGSL.includes(name), `${name} must be gone from the composite`)
  }
})

test("user code and its params land ahead of the entry points", () => {
  const src = buildFieldShader(effect(BACKGROUND, true, false, "struct EffectParams {\n  density: f32,\n}\n"))
  assert.ok(src.indexOf("struct EffectParams") < src.indexOf(BACKGROUND), "params declared before the user's code")
  assert.ok(
    src.indexOf(BACKGROUND) < src.indexOf("@fragment fn fieldFs"),
    "user's code before the entry point that calls it",
  )
})

// The bg* names are a COMPATIBILITY CONTRACT, not a deprecation with an end
// date. A published link is immutable, so a scene pinning an effect that calls
// bgWorldPos has to keep compiling forever — deleting one of these does not
// break a test somewhere, it breaks scenes already shared with other people.
// Every alias must also still be reachable BEFORE user code, since that is where
// the effect calls it from.
test("every bg* alias survives, ahead of the user's code", () => {
  const src = buildCompositeShader(null)
  const aliases = ["bgResolution", "bgCameraPos", "bgSubjectCount", "bgSubjectPos", "bgWorldPos"]
  for (const name of aliases) {
    assert.match(src, new RegExp(`fn ${name}\\s*\\(`), `${name} is pinned by published effects and must not be removed`)
  }
  // Reachable BEFORE the user's code means before it in the FIELD shader, which
  // is the module the user's code is spliced into.
  const withUser = field(BACKGROUND, true, false)
  for (const name of aliases) {
    assert.match(withUser, new RegExp(`fn ${name}\\s*\\(`), `${name} missing from the field shader`)
    assert.ok(withUser.indexOf(`fn ${name}`) < withUser.indexOf(BACKGROUND), `${name} declared before user code`)
  }
})

test("the rz* API is what the aliases delegate to", () => {
  const src = buildCompositeShader(null)
  for (const name of ["rzResolution", "rzCameraPos", "rzSubjectCount", "rzSubjectHip", "rzWorldPos", "rzProject"]) {
    assert.match(src, new RegExp(`fn ${name}\\s*\\(`), `${name} missing`)
  }
  // Delegation, not two copies: one body per value, so the alias cannot drift.
  assert.match(src, /fn bgCameraPos\(\) -> vec3f \{ return rzCameraPos\(\); \}/)
  assert.match(src, /fn bgSubjectPos\(i: i32\) -> vec3f \{ return rzSubjectHip\(i\); \}/)
})

// The pragma is a CONTRACT with effect authors: slot N is the Nth declaration.
// Anything that quietly adds or drops one shifts every slot after it, so an
// effect that was reading a hand starts reading a head.
test("@anchor declares slots in order, and only at the start of a line", () => {
  const src = [
    "// @anchor 左手首 trail",
    "  //  @anchor 頭",
    "// mentioning @anchor 右手首 mid-sentence must not add a slot",
    "fn foreground(r: vec3f, uv: vec2f, t: f32, d: f32) -> vec4f {",
    "  // @anchor 右足ＩＫ",
    "  return vec4f(0.0);",
    "}",
  ].join("\n")
  assert.deepEqual(parseEffectAnchors(src, 8), [
    { bone: "左手首", trail: true },
    { bone: "頭", trail: false },
    { bone: "右足ＩＫ", trail: false },
  ])
  // Past the cap the extras are dropped, never wrapped: the slots that DID fit
  // keep the meaning the author gave them.
  assert.deepEqual(parseEffectAnchors(src, 2), [
    { bone: "左手首", trail: true },
    { bone: "頭", trail: false },
  ])
  assert.deepEqual(parseEffectAnchors("fn background() {}", 8), [])
})
