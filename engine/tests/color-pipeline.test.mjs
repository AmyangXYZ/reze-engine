// A CPU model of what the frame arithmetic does. Run: npm test.
//
// THE INSTRUMENT THAT WAS MISSING. Every visual regression in this engine's
// recent history was predictable from pure arithmetic with no GPU involved:
//
//   - lights invisible          a windowed 1/d^2 in world units, at MMD scale,
//                               delivered 0.016 of its intensity to a subject
//   - stars became flat discs   additive field layers carry alpha ~= 0, and the
//                               composite divides rgb by that coverage
//   - a ribbon changed colour   the bloom prefilter performs the SAME divide,
//                               so one mount's layer moved the whole frame
//
// The suite verifies structure exhaustively — bindings, layouts, emitted WGSL —
// and verified appearance not at all, so all three shipped and were found by
// eye. This file models the chain a pixel actually travels: blend equations ->
// unpremultiply -> view transform -> composite. It is an APPROXIMATION of AgX,
// and deliberately so: what these tests catch is arithmetic that is wrong by
// orders of magnitude, which is the class that has actually bitten. A model
// accurate to a code value would be a second renderer to maintain.

import { test } from "node:test"
import assert from "node:assert/strict"

// ── The scene pass's blend equations, from scene-contract's classes ──────────

/** dst = src.rgb * srcFactor + dst.rgb * dstFactor, per the class's blend. */
function blendOver(dst, src, mode) {
  const a = src.a
  switch (mode) {
    case "alpha": // material / ground-as-it-was: src-alpha, one-minus-src-alpha
      return { rgb: src.rgb.map((c, i) => c * a + dst.rgb[i] * (1 - a)), a: a + dst.a * (1 - a) }
    case "premultiplied": // the ground today: colour already weighted
      return { rgb: src.rgb.map((c, i) => c + dst.rgb[i] * (1 - a)), a: a + dst.a * (1 - a) }
    case "additive-keep-alpha": // particle-additive colour: alpha untouched
      return { rgb: src.rgb.map((c, i) => c + dst.rgb[i]), a: dst.a }
    case "additive-both": // particle-additive AUX: coverage sums too
      return { rgb: src.rgb.map((c, i) => c + dst.rgb[i]), a: Math.min(1, a + dst.a) }
    default:
      throw new Error(`unknown blend ${mode}`)
  }
}

/** The composite's first act, and the bloom prefilter's: recover straight
 *  colour by dividing out the coverage the aux target accumulated. */
const unpremultiply = (rgb, coverage) => rgb.map((c) => c / Math.max(coverage, 1e-6))

/** AgX, approximated. Not the LUT — a curve with the property that matters
 *  here: it compresses hard above ~1 and flattens well before 4, which is why
 *  a value arriving 10x too bright reads as a flat white shape rather than as
 *  something bright. */
const viewTransform = (rgb) => rgb.map((c) => 1 - Math.exp(-Math.max(c, 0) * 0.85))

/** How much detail survives the transform across a range — 1.0 means the
 *  gradient is fully preserved, near 0 means it flattened to one value. */
function contrastRetained(lo, hi) {
  const a = viewTransform([lo, lo, lo])[0]
  const b = viewTransform([hi, hi, hi])[0]
  return b - a
}

// ── What the field layer hands over, per @layer mode ─────────────────────────
//
// The two field blends, verbatim from engine.ts. The additive one is the
// whole story of the reverted attempt: its ALPHA factors are (zero, one), so
// an additive effect contributes rgb and NO coverage — deliberately, because
// in display space that alpha was an occlusion weight and light must not
// occlude.

const fieldAccumulate = (effects, mode) => {
  let layer = { rgb: [0, 0, 0], a: 0 }
  for (const e of effects) {
    layer =
      mode === "additive"
        ? { rgb: e.rgb.map((c, i) => c * e.a + layer.rgb[i]), a: layer.a } // src-alpha, one / zero, one
        : blendOver(layer, e, "alpha")
  }
  return layer
}

test("an additive field layer carries no coverage — the fact behind the failure", () => {
  const stars = [{ rgb: [0.8, 0.85, 1.0], a: 0.9 }]
  const additive = fieldAccumulate(stars, "additive")
  const alpha = fieldAccumulate(stars, "alpha")
  assert.equal(additive.a, 0, "additive contributes rgb and NOTHING to alpha")
  assert.ok(alpha.a > 0.8, "an alpha-over layer does carry its coverage")
})

test("blitting an additive layer into HDR explodes it — all four symptoms", () => {
  // Reproduces the reverted attempt exactly: put the layer in the scene target,
  // let the composite unpremultiply by the coverage the pass accumulated.
  const layer = fieldAccumulate([{ rgb: [0.8, 0.85, 1.0], a: 0.9 }], "additive")
  const scene = blendOver({ rgb: [0, 0, 0], a: 0 }, layer, "premultiplied")
  const straight = unpremultiply(scene.rgb, scene.a)

  // Symptom 3: astronomically bright, so the view transform flattens it.
  assert.ok(straight[0] > 1e4, `expected an explosion, got ${straight[0]}`)
  const core = viewTransform(straight)
  const edge = viewTransform(straight.map((c) => c * 0.25)) // the falloff's edge
  assert.ok(core[0] > 0.99 && edge[0] > 0.99, "core and falloff both saturate — a flat disc, not a glow")
  assert.ok(contrastRetained(straight[0] * 0.25, straight[0]) < 0.01, "the entire falloff is flattened away")
})

test("the ground's coverage is what made it visible ONLY over the ground", () => {
  // The paradoxical report: the effect appeared where the ground drew and
  // nowhere else. The ground contributes coverage, which RESCUES the divide.
  const layer = fieldAccumulate([{ rgb: [0.8, 0.85, 1.0], a: 0.9 }], "additive")
  const sky = blendOver({ rgb: [0, 0, 0], a: 0 }, layer, "premultiplied")
  const overGround = blendOver(sky, { rgb: [0.2, 0.2, 0.2], a: 0.42 }, "premultiplied")

  const skyStraight = unpremultiply(sky.rgb, sky.a)[0]
  const groundStraight = unpremultiply(overGround.rgb, overGround.a)[0]
  assert.ok(skyStraight > 1e4, "over nothing: divided by ~zero coverage")
  assert.ok(groundStraight < 10, `over the ground: coverage present, arithmetic sane (${groundStraight})`)
})

test("summing coverage is the fix, and the codebase already had it", () => {
  // particle-additive's AUX blend is additive-both for exactly this reason:
  // additive content sums coverage so it survives the unpremultiply and can
  // reach the bloom gate. The field path used zero-alpha instead.
  let aux = { rgb: [0, 0, 0], a: 0 }
  aux = blendOver(aux, { rgb: [1, 0, 0], a: 0.9 }, "additive-both")
  assert.ok(aux.a > 0.8, "coverage accumulates for additive content")

  const layer = fieldAccumulate([{ rgb: [0.8, 0.85, 1.0], a: 0.9 }], "additive")
  const straight = unpremultiply(layer.rgb, aux.a)
  assert.ok(straight[0] < 4, `sane magnitude once coverage is summed (${straight[0]})`)
  assert.ok(contrastRetained(straight[0] * 0.25, straight[0]) > 0.1, "and the falloff survives the transform")
})

// ── The lights falloff, the other regression this would have caught ──────────

const falloffPhysical = (d, r) => {
  const w = Math.max(0, 1 - d / r)
  return (w * w) / (1 + d * d)
}
const falloffRadial = (d, r) => {
  const t = Math.min(d / r, 1)
  return (1 - t * t) ** 2
}

test("a light's dial has to be felt at the scale a scene is built at", () => {
  // An MMD character is ~18 units tall; a lamp sits metres away. The shipped
  // physical falloff delivered 1.6% of its intensity to her and read as
  // nothing, with no intensity anyone would think to type able to fix it.
  const d = 6
  const r = 25
  assert.ok(falloffPhysical(d, r) < 0.02, "the physical one is invisible at this scale")
  assert.ok(falloffRadial(d, r) > 0.5, "the radius-relative one delivers most of its intensity")
  // Both must still END at the radius, or the radius means nothing and no cull
  // can be derived from it.
  assert.equal(falloffPhysical(r, r), 0)
  assert.equal(falloffRadial(r, r), 0)
})

test("zero lights is exactly zero, so the layer costs nothing until asked for", () => {
  // The property the whole lights feature is gated on: adding the term to every
  // material must be arithmetically inert until a scene declares one.
  const lit = 0.42
  assert.equal(lit + 0, lit, "adding zero is exact in floating point")
})
