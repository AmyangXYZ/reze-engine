// The WGSL surface published effects are written against. Run: npm test.
//
// An effect's source is spliced into EVERY module its mounts compile into, so a
// constant that exists in one module and not another is a compile error in a
// scene that worked yesterday — and the engine's own code will not reference it,
// so nothing else here notices it is gone. That happened: RZ_TRAIL_SLOTS was
// renamed to RZ_MAX_ANCHORS in the particle module during the anchor-alias work,
// which read as a tidy-up and was actually a breaking change to author surface.
// A library ribbon effect stopped installing with "unresolved value
// 'RZ_TRAIL_SLOTS'".
//
// Beta waives compatibility for DELIBERATE breaks. This test is about the other
// kind: renames that nobody decided to make.

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
  // Two trailed anchors, so RZ_TRAIL_SLOTS is a number the assertions below can
  // tell apart from the anchor address space.
  trailCount: 2,
}
const cast = {
  subjects: 4,
  samples: 128,
  base: 12,
  trailBase: 108,
  slots: 8,
  trailCount: 2,
  alias: [0, 1],
  reversedZ: false,
}
const particleSrc = { wgsl: "", count: 64, blend: "alpha", bloom: false }
const trailSrc = { wgsl: "", slots: 2, ribbonSlots: [0, 1], blend: "additive", bloom: false }

/** Modules an effect's own WGSL is spliced into, and what each must define. */
const MODULES = [
  ["composite", buildCompositeShader(effect)],
  ["field layer", buildFieldShader(effect)],
  ["particle compute", buildParticleComputeShader(particleSrc, cast)],
  ["particle render", buildParticleRenderShader(particleSrc, cast)],
  ["trail shader", buildTrailShader(trailSrc, cast)],
]

/** Accessors every module must expose, because one effect file reaches all of
 *  them and an author cannot know which module a helper of theirs lands in. */
const ACCESSORS = [
  "rzSubjectCount",
  "rzTrailCount",
  "rzTrail",
  // Utilities, in EVERY module. rzHash11 lived only in the particle and trail
  // modules, so an effect that used it in a background silently failed to
  // compile — the gap that produced this line.
  "rzHash11",
  "rzHash13",
]

for (const [name, code] of MODULES) {
  test(`${name}: the cast accessors an effect may call are all present`, () => {
    for (const fn of ACCESSORS) {
      assert.match(code, new RegExp(`fn ${fn}\\(`), `${name} is missing ${fn} — effects calling it will not compile`)
    }
  })
}

/** What the GPU sees. A comment naming a constant is prose, not a definition —
 *  the sibling check in composite.test.mjs strips them for the same reason. */
const code_only = (src) => src.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "")

test("RZ_TRAIL_SLOTS is defined wherever trails are reachable, and means trail COUNT", () => {
  // Not the anchor address space: that is RZ_MAX_ANCHORS, and conflating the
  // two is the original latent trail bug. An author looping `for s in
  // 0..RZ_TRAIL_SLOTS` must iterate their ribbons, not eight.
  for (const [name, src] of MODULES) {
    const code = code_only(src)
    if (!/RZ_TRAIL_SLOTS/.test(code)) continue
    assert.match(
      code,
      /const RZ_TRAIL_SLOTS: i32 = 2;/,
      `${name} defines RZ_TRAIL_SLOTS as something other than the trail count`,
    )
  }
  // The two modules that actually record/draw trails must define it.
  for (const [name, src] of MODULES.filter(([n]) => n.startsWith("particle") || n === "trail shader")) {
    assert.match(code_only(src), /const RZ_TRAIL_SLOTS: i32 =/, `${name} must define RZ_TRAIL_SLOTS — published effects use it`)
  }
})

test("RZ_MAX_ANCHORS is the address space and is 8, not the trail count", () => {
  for (const [name, src] of MODULES) {
    const code = code_only(src)
    if (!/RZ_MAX_ANCHORS/.test(code)) continue
    assert.match(code, /const RZ_MAX_ANCHORS: i32 = 8;/, `${name} has the wrong anchor address space`)
  }
})
