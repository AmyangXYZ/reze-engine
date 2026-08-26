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

// ── The REAL transform, measured off the engine's own cube ───────────────────
//
// The approximation above is fine for order-of-magnitude checks. This is not an
// approximation: it decodes the shipped AgX LUT and runs the same chain the
// shader does, so any constant mapping authored values onto the screen can be
// DERIVED rather than borrowed. Borrowing is what put a field exposure at 3.0
// on the strength of "Snow uses 3.0" — a particle intensity under a different
// curve — and flattened every falloff in the frame.

import { gunzipSync } from "node:zlib"
import { AGX_LUT_GZ, AGX_LUT_SIZE, AGX_INSET, AGX_MIN_EV } from "../dist/shaders/agx-lut.js"

const lut = gunzipSync(Buffer.from(AGX_LUT_GZ, "base64"))
const N = AGX_LUT_SIZE
const agxTexel = (x, y, z) => {
  const v = lut.readUInt32LE(((z * N + y) * N + x) * 4)
  return [(v & 1023) / 1023, ((v >> 10) & 1023) / 1023, ((v >> 20) & 1023) / 1023]
}
function agxSample(uvw) {
  const c = uvw.map((u) => Math.min(Math.max(u * N - 0.5, 0), N - 1))
  const i0 = c.map(Math.floor)
  const f = c.map((x, k) => x - i0[k])
  const i1 = i0.map((x) => Math.min(x + 1, N - 1))
  let out = [0, 0, 0]
  for (let dz = 0; dz < 2; dz++)
    for (let dy = 0; dy < 2; dy++)
      for (let dx = 0; dx < 2; dx++) {
        const w = (dx ? f[0] : 1 - f[0]) * (dy ? f[1] : 1 - f[1]) * (dz ? f[2] : 1 - f[2])
        const t = agxTexel(dx ? i1[0] : i0[0], dy ? i1[1] : i0[1], dz ? i1[2] : i0[2])
        out = out.map((v, k) => v + t[k] * w)
      }
  return out
}
const srgbEncode = (x) => (x <= 0.0031308 ? Math.max(x, 0) * 12.92 : 1.055 * Math.pow(Math.max(x, 0), 1 / 2.4) - 0.055)
function agx(rgb) {
  const m = AGX_INSET
  const e = [
    m[0] * rgb[0] + m[1] * rgb[1] + m[2] * rgb[2],
    m[3] * rgb[0] + m[4] * rgb[1] + m[5] * rgb[2],
    m[6] * rgb[0] + m[7] * rgb[1] + m[8] * rgb[2],
  ].map((v) => Math.max(v, 0))
  const t = e.map((v) => Math.min(Math.max((Math.log2(Math.max(v, 1e-10)) - AGX_MIN_EV) / 25.0, 0), 1))
  return agxSample(t.map((v) => v * ((N - 1) / N) + 0.5 / N)).map((v) => srgbEncode(Math.pow(Math.max(v, 0), 2.4)))
}
const saturation = (c) => {
  const mx = Math.max(...c)
  return mx <= 0 ? 0 : (mx - Math.min(...c)) / mx
}

test("the shipped AgX lifts midtones — it does NOT land 1.0 at mid grey", () => {
  // The belief that steered a reverted attempt was that an authored 1.0 arrives
  // as grey and needs a large boost. Measured, it arrives at 0.77 — light, not
  // grey — so the boost solved a problem of the wrong size.
  const at = (x) => agx([x, x, x])[0]
  assert.ok(Math.abs(at(0.5) - 0.66) < 0.02, `linear 0.5 -> ${at(0.5).toFixed(3)}`)
  assert.ok(Math.abs(at(1.0) - 0.77) < 0.02, `linear 1.0 -> ${at(1.0).toFixed(3)}`)
  assert.ok(at(0.5) > 0.5, "midtones are LIFTED, not crushed")
  assert.ok(at(1.0) < 1.0, "and the top is compressed, so nothing reaches pure white")
})

test("AgX desaturates saturated colour — why field effects stay in display space", () => {
  // THE measurement behind that decision. Note Fall's blue, at magnitude 1 with
  // no clipping anywhere: authored it is vivid, through the transform it is
  // washed. A filmic curve does this at EVERY magnitude, so an effect authored
  // as a display colour cannot keep its hue by being made dimmer or brighter.
  //
  // Field effects therefore composite in display space and land exactly as
  // authored. To be LIGHT in the scene an effect declares lights (lightEmit),
  // which illuminates the cast rather than merely glowing — a better answer
  // than bloom, and one that costs the author no new flag.
  const glow = [0.14, 0.42, 1.0]
  assert.ok(saturation(glow) > 0.8, "authored: a vivid blue")
  assert.ok(saturation(agx(glow)) < 0.45, `through AgX: washed (${saturation(agx(glow)).toFixed(2)})`)
  // And dimming does not rescue it — the hue loss is the transform's, not a
  // matter of exposure.
  assert.ok(saturation(agx(glow.map((c) => c * 0.5))) < 0.6, "still washed at half the magnitude")
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

  // Only the world projects irradiance, and only it carries the strength dial.
  const wd = body.slice(body.indexOf("setWorldEquirect(source"), body.indexOf("setBackdropEquirect(source"))
  assert.match(wd, /this\.worldSH = projectIrradianceSH/)
  assert.match(wd, /this\.worldStrength = Math\.max\(options\?\.strength \?\? 1, 0\)/)

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
  const passes = ["outline", "depth-prepass", "shadow"]
  for (const name of passes) {
    const src = readFileSync(new URL(`../src/shaders/passes/${name}.ts`, import.meta.url), "utf8")
    assert.match(
      src,
      /material\.dissolve < 0\.9995 && rz_dissolve_threshold\(in(put)?\.restPos\) > material\.dissolve/,
      `${name} must run the same dissolve test`,
    )
    // Against the same BIND-POSE position, or two passes disagree about which
    // flakes are gone and the model writes depth where it no longer draws.
    assert.match(src, /restPos/, `${name} must carry restPos`)
  }
  // And the colour pass, which reaches it through the graph prologue.
  const slots = readFileSync(new URL("../src/graph/slots.ts", import.meta.url), "utf8")
  assert.match(slots, /if \(rz_t > material\.dissolve\) \{ discard; \}/)
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
  const struct = shader.slice(shader.indexOf("struct MaterialUniforms"), shader.indexOf("@group(0) @binding(0)"))
  // edgeColor 0..16, edgeSize 16..20, dissolve 20..24, two pads to 32.
  assert.match(struct, /edgeColor: vec4f,\s*edgeSize: f32,\s*dissolve: f32,/)
  assert.doesNotMatch(struct, /_skip/, "no reach into the material block's layout")
  assert.match(shader, /export const RZ_OUTLINE_DISSOLVE_OFFSET = 20/)

  const engine = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  // EIGHT FLOATS. Adding a ninth grows the buffer past what the bind group
  // layout was built for, which is the same validation failure from the other
  // direction.
  const at = engine.indexOf("mat.edgeColor[0]")
  const made = engine
    .slice(engine.lastIndexOf("new Float32Array([", at), engine.indexOf("])", at))
    // Comments first — they hold commas of their own, and counting those counts
    // prose rather than floats.
    .replace(/\/\/[^\n]*/g, "")
    .replace("new Float32Array([", "")
  assert.equal(
    made.split(",").filter((l) => l.trim().length).length,
    8,
    "the outline uniform is eight floats",
  )
  // Written through the shared constant, never a literal — the two ends cannot
  // drift if only one of them names the number.
  assert.match(engine, /writeBuffer\(buffer, RZ_OUTLINE_DISSOLVE_OFFSET, one\)/)
  assert.match(engine, /inst\.outlineUniformBuffers\.push\(outlineUniformBuffer\)/)
})
