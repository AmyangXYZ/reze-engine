// One effect file, every module. Run: npm test.
//
// An effect's source is spliced WHOLE into every module it has a mount in. A
// file with a trail and a lightEmit compiles its ribbon code inside the light
// module, where it never runs and still has to resolve. So the question this
// file answers is not "does each module have what its own mount needs" — it is
// "does every module have what ANY mount's code might name".
//
// It was not close. Before this test there were 78 author-facing names and only
// 14 existed in all six modules; the particle modules had no rzAnchor at all,
// so a spark could ask where a trail had been but not where a wrist is. Two
// separately-written implementations of the same cast buffer is how that
// happened, and nothing noticed because the engine's own code never calls the
// author API — only someone else's effect does, at install time, in a browser.
//
// The guard is deliberately two-sided. A NEW partial name fails, which is the
// drift. A LISTED name that has since become universal also fails, so the list
// cannot quietly grow into a description of whatever the code happens to do.

import { test } from "node:test"
import assert from "node:assert/strict"
import { EFFECT_SCENE_API, buildFieldShader } from "../dist/shaders/passes/composite.js"
import { buildSimShader } from "../dist/shaders/passes/grid.js"
import { buildParticleComputeShader, buildParticleRenderShader } from "../dist/shaders/passes/particles.js"
import { buildTrailShader } from "../dist/shaders/passes/trails.js"
import { buildLightEmitShader } from "../dist/shaders/lights.js"
import { anchorAliasWgsl } from "../dist/shaders/anchor-table.js"

const alias = [0, 1, 2, 3, 4, 5, 6, 7]
const cast = { subjects: 4, samples: 128, base: 12, trailBase: 108, slots: 8, trailCount: 2, alias, reversedZ: false }
const wgsl = "// a file with no mounts of its own — this is about the PRELUDES\n"
const psrc = { wgsl, count: 64, blend: "additive", bloom: false, slots: 2, ribbonSlots: [0, 1] }

/** Every module an author's source can be spliced into. Adding a mount to the
 *  engine and not to this list is the one way to slip past this test. */
const MODULES = {
  field: buildFieldShader({ wgsl, paramsDecl: "", hasBackground: true, hasForeground: false, gridSize: 0, alias, trailCount: 2 }),
  grid: buildSimShader(wgsl, 256, cast),
  "particle sim": buildParticleComputeShader(psrc, cast),
  "particle draw": buildParticleRenderShader(psrc, cast),
  trail: buildTrailShader(psrc, cast),
  "light emit": buildLightEmitShader(wgsl, EFFECT_SCENE_API + anchorAliasWgsl(alias), cast),
}

const strip = (s) => s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "")
// Names an author could write: the rz and bg functions, the RZ_ constants, and
// the two mount structs that appear in the signatures they are handed. (Line
// comments, not a block: "rz*" followed by "/bg" would close one mid-sentence.)
const declared = (src) => {
  const c = strip(src)
  return new Set([
    ...[...c.matchAll(/\bfn\s+(rz[A-Za-z0-9_]*|bg[A-Za-z0-9_]*)\s*\(/g)].map((m) => m[1]),
    ...[...c.matchAll(/\b(?:const|let|var)\s+(RZ_[A-Z0-9_]+)\b/g)].map((m) => m[1]),
    ...[...c.matchAll(/\bstruct\s+(Rz[A-Za-z0-9_]*|Particle)\b/g)].map((m) => m[1]),
  ])
}

const SETS = Object.fromEntries(Object.entries(MODULES).map(([k, v]) => [k, declared(v)]))
const NAMES = [...new Set(Object.values(SETS).flatMap((s) => [...s]))].sort()
const everywhere = (n) => Object.values(SETS).every((s) => s.has(n))

/**
 * Names that legitimately exist in only some modules, and why.
 *
 * Each one is a capability a module genuinely does not have — a texture it does
 * not bind, a buffer that means nothing to it — not an oversight. The cost is
 * real and is the point of writing them down: an effect whose file uses one of
 * these AND has a mount in a module without it does not compile, and that is a
 * thing to fix by closing the gap, not by adding a line here.
 */
const EXCEPTIONS = {
  // The grid is a texture pair. Giving it to the particle and trail modules
  // means binding a 1x1 of zeroes in both, which is cheap and probably right —
  // it is simply not done yet.
  RZ_GRID_SIZE: "grid textures are bound only by the field and grid modules",
  rzGrid: "same",
  rzGridPrev: "same",
  rzGridSize: "same",
  rzGridTexel: "same",
  rzGridFrame: "same",
  // The id attachment exists only where the scene pass wrote it.
  rzObjectAt: "reads the id attachment, which only the field module binds",
  rzMaterialAt: "same",
  rzFieldMerge: "composites the two field layers; nothing else has them",
  // The particle and trail modules keep a camera STRUCT rather than the view
  // uniform, so anything derived from the full uniform is missing there. The
  // fix is a wider CameraU, not a second implementation.
  rzResolution: "needs the view uniform; the particle and trail modules have only CameraU",
  rzWorldPos: "same",
  rzSubjectHip: "same",
  bgResolution: "alias of rzResolution",
  bgCameraPos: "alias, defined beside the others it aliases",
  bgSubjectCount: "same",
  bgSubjectPos: "same",
  bgWorldPos: "same",
  // Ribbon geometry. Pure math that could be shared; nothing outside a trail
  // has yet had a reason to call it.
  rzSpline: "trail geometry helpers, used by the ribbon builder",
  rzSplineTangent: "same",
  rzTangentAt: "same",
  rzTrailAt: "same",
  rzTurnRadius: "same",
  RZ_REF_SPAN: "the ribbon's line-integral reference span",
  RZ_SLOTS: "how many ribbons THIS module draws; RZ_TRAIL_SLOTS is the author-facing count",
}

test("every author-facing name is in every module, or listed as an exception", () => {
  const surprises = NAMES.filter((n) => !everywhere(n) && !(n in EXCEPTIONS))
  assert.deepEqual(
    surprises,
    [],
    `defined in some modules and not others, with no reason recorded: ${surprises
      .map((n) => `${n} (in ${Object.keys(SETS).filter((k) => SETS[k].has(n)).join(", ")})`)
      .join("; ")}`,
  )
})

test("the exception list has no entries that are no longer true", () => {
  const stale = Object.keys(EXCEPTIONS).filter((n) => everywhere(n) || !NAMES.includes(n))
  assert.deepEqual(stale, [], `listed as partial but is not: ${stale.join(", ")}`)
})

test("the names an effect is most likely to reach for are universal", () => {
  // Not a restatement of the test above: this is the floor. If closing a gap
  // ever means DELETING a name from a module to make the sets agree, these are
  // the ones that must not be deleted.
  const CORE = [
    // Where the dancers and their bones are — the thing effects are anchored to.
    "rzAnchor",
    "RzAnchor",
    "rzSubject",
    "RzSubject",
    "rzSubjectCount",
    "rzSubjectId",
    "rzTrail",
    "rzTrailCount",
    "RZ_MAX_ANCHORS",
    "RZ_TRAIL_SLOTS",
    // The clock, and the two structs that appear in signatures the author writes.
    "rzTime",
    "rzDt",
    "Particle",
    "RzLight",
    // Utilities. rzHash11 lived only in the particle and trail modules once, so
    // a background that used it failed to compile for no visible reason.
    "rzHash11",
    "rzHash13",
    "rzHash21",
    "rzHash31",
    "rzFalloff",
    "rzValueNoise",
    "rzCurlNoise",
    "rzViewportHeight",
    // The camera, in every module that can draw or place anything.
    "rzCameraPos",
    "rzCameraRight",
    "rzCameraUp",
    "rzCameraForward",
    "rzProject",
  ]
  for (const name of CORE) {
    const missing = Object.keys(SETS).filter((k) => !SETS[k].has(name))
    assert.deepEqual(missing, [], `${name} is missing from: ${missing.join(", ")}`)
  }
})

test("nothing is declared twice in one module", () => {
  // The failure mode of sharing blocks: two of them both carrying rzFalloff is
  // not a duplicate definition an author can see, it is a shader that will not
  // compile with an error pointing at engine code.
  for (const [name, src] of Object.entries(MODULES)) {
    const c = strip(src)
    const counts = new Map()
    for (const m of c.matchAll(/\b(?:fn|struct)\s+((?:rz|bg|Rz)[A-Za-z0-9_]*|Particle)\b/g)) {
      counts.set(m[1], (counts.get(m[1]) ?? 0) + 1)
    }
    const dupes = [...counts].filter(([, n]) => n > 1).map(([k, n]) => `${k}×${n}`)
    assert.deepEqual(dupes, [], `${name} declares: ${dupes.join(", ")}`)
  }
})
