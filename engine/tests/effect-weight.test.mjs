// Weight: the one dial that means the same thing for every effect ever written.
//
// It is applied by ENGINE-GENERATED code at each mount's single output site,
// which is the whole reason these are source tests rather than render tests.
// An effect honouring its own weight would be an effect that could forget to,
// and a scheduler cannot be built on a promise every author has to keep — so
// what needs pinning is that the multiply is emitted into all four modules,
// whatever the author wrote.
//
// WHERE the multiply lands is not cosmetic either. Each mount blends
// differently, and weight has to scale the term its own target actually reads:
//
//   field (both layers)   colour x src-alpha   -> scale alpha
//   particles             premultiplied here   -> scale alpha, colour follows
//   trails                colour x src-alpha   -> scale alpha, into BOTH targets
//   lights                summed into the buffer -> scale the emitted colour
//
// Scaling the wrong term fades as the square, or does not fade at all. The
// tests below name the term for each.

import { test } from "node:test"
import assert from "node:assert/strict"
import { buildFieldShader } from "../dist/shaders/passes/composite.js"
import { buildParticleRenderShader } from "../dist/shaders/passes/particles.js"
import { buildTrailShader } from "../dist/shaders/passes/trails.js"
import { buildLightEmitShader } from "../dist/shaders/lights.js"
import { EFFECT_SCENE_API } from "../dist/shaders/passes/composite.js"
import { anchorAliasWgsl } from "../dist/shaders/anchor-table.js"

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

/** An author who has never heard of weight. Nothing below is opted into. */
const effect = (hasBackground, hasForeground) => ({
  wgsl:
    "fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f { return vec4f(1.0); }\n" +
    "fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f { return vec4f(1.0); }",
  paramsDecl: "",
  hasBackground,
  hasForeground,
  gridSize: 0,
})

test("the field's weight rides the clock buffer's spare component", () => {
  const src = buildFieldShader(effect(true, true))
  // One buffer, not two: a frame that uploaded the clock and not the weight
  // would draw an effect at last frame's opacity, and the two are written by
  // the same queue call precisely so that cannot happen.
  assert.match(src, /@binding\(22\) var<uniform> _rzFieldClock: vec4f/)
  assert.match(src, /out\.bg\.a \*= _rzFieldClock\.y;/)
  assert.match(src, /out\.fg\.a \*= _rzFieldClock\.y;/)
})

test("a field effect with one mount still scales it", () => {
  const bg = buildFieldShader(effect(true, false))
  assert.match(bg, /out\.bg\.a \*= _rzFieldClock\.y;/)
  const fg = buildFieldShader(effect(false, true))
  assert.match(fg, /out\.fg\.a \*= _rzFieldClock\.y;/)
})

test("weight scales the field's ALPHA, never its colour", () => {
  // Both field blends multiply the fragment's colour by src-alpha, so alpha
  // alone is the linear fade. Scaling colour too would fade as the square —
  // an effect that vanishes halfway through its own fade.
  const src = buildFieldShader(effect(true, true))
  assert.doesNotMatch(src, /out\.bg\.rgb \*= _rzFieldClock\.y/)
  assert.doesNotMatch(src, /out\.bg \*= _rzFieldClock\.y/)
})

test("particles carry weight in their uniform and apply it before the discard", () => {
  const src = buildParticleRenderShader({ wgsl: "", count: 64, blend: "alpha", bloom: false }, cast)
  assert.match(src, /weight: f32,/)
  assert.match(src, /c\.a \*= pu\.weight;/)
  // BEFORE the discard, so weight 0 kills every fragment rather than shading it
  // and multiplying the result out to nothing.
  assert.ok(
    src.indexOf("c.a *= pu.weight;") < src.indexOf("if (c.a <= 0.0) { discard; }"),
    "weight must be applied before the alpha discard",
  )
})

test("the particle shade result is mutable, or the multiply would not compile", () => {
  // It was `let c = particleShade(...)`. WGSL lets are immutable, so this is a
  // compile error rather than a silently ignored write — but a compile error
  // inside a generated module reads as the author's fault.
  const src = buildParticleRenderShader({ wgsl: "", count: 64, blend: "alpha", bloom: false }, cast)
  assert.match(src, /var c = particleShade\(p, in\.uv\);/)
})

test("both particle blends get the same multiply", () => {
  for (const blend of ["alpha", "additive"]) {
    const src = buildParticleRenderShader({ wgsl: "", count: 64, blend, bloom: false }, cast)
    assert.match(src, /c\.a \*= pu\.weight;/, `${blend} particles`)
  }
})

test("trails scale both targets by the effect's weight, not the strand's", () => {
  const src = buildTrailShader({ wgsl: "", slots: 2, ribbonSlots: [0, 1], blend: "additive", bloom: false }, cast)
  assert.match(src, /weight: f32,/)
  assert.match(src, /let a = c\.a \* tu\.weight;/)
  // The colour target AND the aux mask. The composite divides HDR colour by
  // that coverage to un-premultiply, so a mask left at full while the colour
  // faded is a band that brightens as the effect disappears.
  assert.match(src, /o\.color = vec4f\(c\.rgb, a\);/)
  assert.match(src, /o\.aux = vec4f\([^)]*, a\);/)
  // in.weight is the ribbon's own taper and belongs to the author — untouched.
  assert.match(src, /trailShade\(in\.uv\.x, in\.uv\.y, in\.age, in\.weight, i32\(in\.slot\)\)/)
})

test("lights scale their emitted colour, at the same site that sanitises it", () => {
  const emit = `#lights 2
fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  l.pos = vec3f(0.0);
  l.color = vec3f(1.0);
  l.intensity = 3.0;
  l.radius = 20.0;
  return l;
}`
  const src = buildLightEmitShader(emit, EFFECT_SCENE_API + anchorAliasWgsl([0]), { trailCount: 2 })
  // Colour, because this buffer is summed by every material that reads it —
  // there is no alpha here to scale instead.
  assert.match(src, /let c = select\(vec3f\(0\.0\), max\(l\.color \* l\.intensity, vec3f\(0\.0\)\), finite\) \* _rzLightU\.w;/)
  // Radius is zeroed at weight 0 and untouched above it: a dimming lamp keeps
  // its reach, and one that is off can be culled by distance like a slot that
  // was never filled.
  assert.match(src, /_rzLightsOut\[b \+ 3u\] = select\(0\.0, max\(l\.radius, 0\.0\), finite && _rzLightU\.w > 0\.0\);/)
})

// The engine's own frame path, read as source. These cannot be run here — they
// need a device — and they are exactly the rules a render test would be too
// coarse to catch, so they are pinned against the file.
import { readFileSync } from "node:fs"
const ENGINE = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
const body = ENGINE.replace(/\/\/[^\n]*/g, "").replace(/\/\*[\s\S]*?\*\//g, "")

test("the field pass runs on what is MOUNTED and draws what has weight", () => {
  // Two predicates. The pass must run whenever fieldPairUsed says the composite
  // is reading this layer's target, or a layer whose effects all faded to zero
  // would keep showing the last frame drawn into it — the composite's bind
  // group is built at install, not per frame.
  assert.match(body, /const mounted = this\.effects\.filter\(\(e\) => e\.fieldPipeline && e\.fieldBindGroups\)/)
  assert.match(body, /const drawn = mounted\.filter\(\(e\) => e\.weight > 0\)/)
  // fieldPairUsed is the binding's predicate and must stay weight-blind for the
  // same reason. If this ever gains a weight term, the composite bind group has
  // to be rebuilt whenever a weight crosses zero.
  assert.doesNotMatch(
    body,
    /fieldPairUsed\(layer: number\): boolean \{\s*return this\.effects\.some\(\(e\) => [^)]*weight/,
  )
})

test("a weight of zero draws nothing at any mount", () => {
  assert.match(body, /if \(!p \|\| e\.weight === 0\) continue/, "particle draw")
  assert.match(body, /this\.effects\.filter\(\(e\) => e\.trails && e\.weight > 0\)/, "trail draw")
  // LIGHTS ARE THE EXCEPTION, and the test says so rather than leaving the
  // absence to be read as an oversight: effects write their own slots into a
  // shared buffer nobody clears, so a skipped dispatch leaves the last frame's
  // lights lit. The shader zeroes them instead — colour by the weight, radius
  // at zero exactly.
  assert.doesNotMatch(body, /if \(!l \|\| l\.data\[2\] === 0 \|\| e\.weight === 0\)/, "light dispatch")
})

test("the particle SIMULATION is not gated on weight", () => {
  // Deliberate: an effect frozen while faded out would resume from the state it
  // left rather than the one it would have reached, so fading it back in would
  // rewind it. The draw is what stops.
  const from = body.indexOf("private stepParticles")
  const step = body.slice(from, body.indexOf("private renderParticles", from))
  assert.doesNotMatch(step, /e\.weight === 0/)
  assert.match(step, /p\.data\[4\] = e\.weight/)
})

test("setEffectInfluence clamps, and setEffectTime moves the epoch", () => {
  assert.match(body, /fx\.influence = Math\.min\(1, Math\.max\(0, influence\)\)/)
  // The epoch is the single origin every mount derives its clock from — field,
  // particles, ribbons, lightEmit and the grid's frame counter — so moving it
  // moves all of them and none can be left a frame behind.
  assert.match(body, /fx\.epochScene = this\.sceneClock - time/)
})
