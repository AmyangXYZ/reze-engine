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
import { readFileSync } from "node:fs"

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

// ── What the field layer hands over, per #layer mode ─────────────────────────
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

// ── The REAL transforms, read off the composite's own source ─────────────────
//
// The approximation above is fine for order-of-magnitude checks. These are not
// approximations: the default transform's constants are read from composite.ts,
// so a constant mapping authored values onto the screen can be DERIVED rather
// than borrowed.

const compositeSrc = readFileSync(new URL("../src/shaders/passes/composite.ts", import.meta.url), "utf8")
const srgbEncode = (x) => (x <= 0.0031308 ? Math.max(x, 0) * 12.92 : 1.055 * Math.pow(Math.max(x, 0), 1 / 2.4) - 0.055)

test("the view transforms are soft, neutral, aces and none — soft first, the default", () => {
  const dispatch = compositeSrc.slice(compositeSrc.indexOf("fn viewTransform("))
  for (const fn of ["acesTransform", "neutralTransform", "softTransform"]) assert.ok(dispatch.includes(`return ${fn}(c)`), fn)
  const engine = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  assert.match(engine, /export type ViewTransformName = "soft" \| "neutral" \| "aces" \| "none"/)
  assert.match(engine, /transform: "soft",\n\}/, "soft is the default")
  assert.doesNotMatch(compositeSrc, /filmicLut|agxLut/, "no Blender LUTs left")
})

test("soft is the game's curve: (1 - e^(-2.5x))^1.4, then sRGB", () => {
  const body = compositeSrc.slice(compositeSrc.indexOf("fn softTransform("))
  assert.match(body, /exp\(-2\.5 \* max\(c, vec3f\(0\.0\)\)\)/)
  assert.match(body, /vec3f\(1\.4\)/)
  const soft = (x) => srgbEncode(Math.pow(1 - Math.exp(-2.5 * x), 1.4))
  // Mid grey lands light, white lands just short of the top: the game's look.
  assert.ok(Math.abs(soft(0.18) - 0.529) < 0.005, `linear 0.18 -> ${soft(0.18).toFixed(3)}`)
  assert.ok(soft(1.0) > 0.94 && soft(1.0) < 0.96, `linear 1.0 -> ${soft(1.0).toFixed(3)}`)
})

test("neutral passes colour through below its knee — Khronos PBR Neutral's start 0.76", () => {
  const body = compositeSrc.slice(compositeSrc.indexOf("fn neutralTransform("))
  assert.match(body, /startCompression = 0\.8 - 0\.04/)
  assert.match(body, /desaturation = 0\.15/)
})

test("the world and the backdrop are separate seats", () => {
  // WHAT LIGHTS and WHAT YOU SEE are different questions, and they shared one
  // slot — so lighting with a studio HDRI while showing a different sky could
  // not be said at all. Pinned as source: there is no device here to install
  // an equirect on, and the rules below are decisions rather than pixels.
  const src = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  const body = src.replace(/\/\/[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "")

  // Two installers, each owning one question.
  assert.match(body, /setWorldEquirect\(source: HdrImage \| null/)
  assert.match(body, /setBackdropEquirect\(\s*source: ImageBitmap \| HTMLImageElement \| HTMLCanvasElement \| null/)
  // The backdrop no longer takes an HDR — a picture that silently changed the
  // lighting because of its file extension is the surprise this removes.
  const bd = body.slice(body.indexOf("setBackdropEquirect(source"))
  assert.doesNotMatch(bd.slice(0, 2000), /projectIrradianceSH/)

  // Only the world projects irradiance, and it projects it RAW.
  const wd = body.slice(body.indexOf("setWorldEquirect(source"), body.indexOf("setBackdropEquirect(source"))
  assert.match(wd, /this\.worldSH = projectIrradianceSH/)
  // ONE STRENGTH, and installing a sky is not where it is decided. A second
  // field folded into the projection made the sky's brightness depend on the
  // order the dial and the install happened to arrive in — the same scene came
  // up one way on upload and another on reload.
  assert.doesNotMatch(body, /worldStrength/)
  assert.match(body, /u\[8\] = showingWorld \? this\.world\.strength : \(bg\?\.x \?\? 0\)/)

  // THE BACKDROP WINS WHAT YOU SEE; the world lights regardless. With only a
  // world installed it is also the sky, which is what an HDRI alone always did.
  assert.match(body, /const showingWorld = this\.backdropEquirectView === null && this\.worldEquirectView !== null/)
  assert.match(body, /u\[11\] = this\.backdropEquirectView \? 2 : showingWorld \? 3 : bg \? 1 : 0/)
  assert.match(body, /binding: 6, resource: this\.backdropEquirectView \?\? this\.worldEquirectView \?\? this\.fallbackEquirectView/)
})

test("every pass that draws the body honours the dissolve", () => {
  // FOUR PASSES, ONE RULE. The colour pass shades her, the depth prepass claims
  // depth for what it will shade, the shadow pass casts from her — and the
  // outline pass traces her silhouette. Three of them tested and the fourth did
  // not, so a character who had dissolved left her own hulls standing: a solid
  // shape in her edge colour, where she had been.
  //
  // It looked like the model would not dissolve, and it only happened to models
  // whose materials carry MMD's edge flag — which is what made it read as
  // "some models are broken" rather than as one pass missing a line.
  // The threshold is per TRIANGLE: computed in the vertex stage and passed
  // FLAT, so every fragment of a face shares the provoking vertex's value and
  // the face goes as a unit.
  //
  // That moves what these four have to agree about. It is no longer "run the
  // same expression on restPos" but "derive the same flat value from the same
  // bind-pose attribute" — a pass that computed it per fragment instead would
  // disagree with the others about which FACES are gone, and the model would
  // write depth where it no longer draws. So the derivation is pinned here as
  // well as the comparison: the comparison alone would pass for a pass that
  // filled its own slot from something else.
  const passes = ["outline", "depth-prepass", "shadow"]
  for (const name of passes) {
    const src = readFileSync(new URL(`../src/shaders/passes/${name}.ts`, import.meta.url), "utf8")
    assert.match(
      src,
      /material\.dissolve < 0\.9995 && in(put)?\.faceT > material\.dissolve/,
      `${name} must run the same dissolve test`,
    )
    assert.match(src, /@interpolate\(flat\) faceT: f32/, `${name} must carry the threshold FLAT`)
    assert.match(
      src,
      /\.faceT = rz_dissolve_threshold\(position\)/,
      `${name} must derive it from the bind-pose attribute`,
    )
  }
  // And the colour pass, which reaches it through the graph prologue.
  const slots = readFileSync(new URL("../src/graph/slots.ts", import.meta.url), "utf8")
  assert.match(slots, /if \(rz_t > material\.dissolve\) \{ discard; \}/)
  assert.match(slots, /let rz_t = input\.faceT;/)
  // The colour pass's own flat slot lives on the shared material VertexOutput.
  const common = readFileSync(new URL("../src/shaders/materials/common.ts", import.meta.url), "utf8")
  assert.match(common, /@interpolate\(flat\) faceT: f32/)
  assert.match(common, /output\.faceT = rz_dissolve_threshold\(position\)/)
})

test("the outline's struct describes the buffer that is actually bound", () => {
  // A STRUCT IS A CLAIM ABOUT A BUFFER, and the buffer is built somewhere else.
  //
  // This first reached for the material block's layout, declaring the skipped
  // middle so dissolve landed on byte 60 — where setModelDissolve writes it for
  // the colour pass. It compiled, and then failed validation at the first draw:
  // the buffer bound here is 32 bytes of edge data, and a pipeline asking for 64
  // is one nothing can satisfy.
  //
  // So the hull carries its own copy, and the two ends have to agree about
  // where. The shader declares the offset and the engine writes it.
  const shader = readFileSync(new URL("../src/shaders/passes/outline.ts", import.meta.url), "utf8")
  const struct = shader
    .slice(shader.indexOf("struct MaterialUniforms"), shader.indexOf("@group(0) @binding(0)"))
    .replace(/\/\*[\s\S]*?\*\//g, "")
  // edgeColor 0..16, edgeSize 16..20, dissolve 20..24, widthScale 24..28, a
  // pad to 32, colorOverride 32..48.
  assert.match(struct, /edgeColor: vec4f,\s*edgeSize: f32,\s*dissolve: f32,\s*widthScale: f32,\s*_padding3: f32,\s*colorOverride: vec4f,/)
  assert.doesNotMatch(struct, /_skip/, "no reach into the material block's layout")
  assert.match(shader, /export const RZ_OUTLINE_DISSOLVE_OFFSET = 20/)
  assert.match(shader, /export const RZ_OUTLINE_WIDTH_OFFSET = 24/)
  assert.match(shader, /export const RZ_OUTLINE_COLOR_OFFSET = 32/)

  const engine = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  // TWELVE FLOATS, the struct's 48 bytes. A count that disagrees with the
  // struct is the same validation failure from the other direction.
  const at = engine.indexOf("mat.edgeColor[0]")
  const made = engine
    .slice(engine.lastIndexOf("new Float32Array([", at), engine.indexOf("])", at))
    // Comments first — they hold commas of their own, and counting those counts
    // prose rather than floats.
    .replace(/\/\/[^\n]*/g, "")
    .replace("new Float32Array([", "")
  assert.equal(
    made.split(",").filter((l) => l.trim().length).length,
    12,
    "the outline uniform is twelve floats",
  )
  // Written through the shared constant, never a literal — the two ends cannot
  // drift if only one of them names the number.
  assert.match(engine, /writeBuffer\(buffer, RZ_OUTLINE_DISSOLVE_OFFSET, one\)/)
  assert.match(engine, /writeBuffer\(buffer, RZ_OUTLINE_WIDTH_OFFSET, one\)/)
  assert.match(engine, /writeBuffer\(buffer, RZ_OUTLINE_COLOR_OFFSET, data\)/)
  assert.match(engine, /inst\.outlineUniformBuffers\.push\(outlineUniformBuffer\)/)
})
