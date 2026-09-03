// The media plane's geometry contract, checked against the SOURCE. Run: npm test.
//
// addPlane builds a Model by hand rather than parsing one, so the invariants the
// PMX loader normally guarantees are this function's own responsibility — and
// every one of them is silent when broken. A wrong winding renders nothing at
// all, an unflipped V renders the picture upside down, and an unskinned vertex
// collapses the card to a point. None of those throw.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const engine = readFileSync(join(here, "../src/engine.ts"), "utf8")
const body = engine.slice(engine.indexOf("  async addPlane("), engine.indexOf("  /** True while a stage is in the scene."))

test("addPlane exists and builds its own Model", () => {
  assert.ok(body.length > 0, "addPlane not found")
  assert.match(body, /new Model\(/, "a plane must construct a Model, not parse one")
})

test("the quad's V runs opposite its Y", () => {
  // A picture's rows run down from its top-left; a UV runs up from the
  // bottom-left. The bottom two corners therefore carry v=1 and the top two
  // v=0, and getting it backwards flips every card in the scene.
  const rows = [...body.matchAll(/^\s*(-?hw),\s*(-?hh),\s*0\.0,\s*0\.0,\s*0\.0,\s*1\.0,\s*([01]\.0),\s*([01]\.0),\s*$/gm)]
  assert.equal(rows.length, 4, "expected four quad vertices")
  for (const [, , y, , v] of rows) {
    const top = y === "hh"
    assert.equal(v, top ? "0.0" : "1.0", `vertex at y=${y} should carry v=${top ? 0 : 1}`)
  }
})

test("every vertex is fully weighted to the one bone", () => {
  // Model requires a skeleton and skins every vertex through it. A vertex whose
  // weights sum to zero is transformed by nothing and lands at the origin, which
  // collapses the card rather than erroring.
  assert.match(body, /weights\[i \* 4\] = 255/, "each vertex needs full weight on bone 0")
  assert.match(body, /bones: \[\{ name: "全ての親"/, "a plane still needs one bone to skin to")
})

test("double-sided adds a reversed pair rather than dropping the cull", () => {
  // Winding is what the scene pass culls by; there is no per-material cull flag
  // for this to reach for, so the back faces have to be real triangles.
  assert.match(body, /doubleSided \? \[0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2\] : \[0, 1, 2, 0, 2, 3\]/)
})

test("the texture is answered from memory, never fetched", () => {
  // The picture is already in hand as bytes. Going through the network reader
  // would try to fetch "plane://name" and fail.
  assert.match(body, /const reader: AssetReader = \{ readBinary: async \(\) => image \}/)
  assert.doesNotMatch(body, /createFetchAssetReader/, "a plane's texture never comes off the network")
})

test("the texture entry is relative, and the model path is its directory", () => {
  // The loader joins the texture entry onto the model path's DIRECTORY, exactly
  // as it does for a PMX. A scheme-shaped path composed into nonsense and every
  // card rendered with the untextured fallback — silently, because a missing
  // texture only warns.
  assert.match(body, /const texturePath = `plane\/\$\{name\}`/, "the model path must be a plain directory + name")
  assert.match(body, /\[\{ path: name, name \}\]/, "the texture entry must be relative to it")
})

test("a plane is added on its own path, not the stage's", () => {
  // A plane and a stage skip the same machinery and mean different things.
  // Routing one through the other is how adding a picture suppressed the floor.
  assert.match(body, /this\.addModel\(model, texturePath, name, reader, \{ plane: true,/)
  assert.doesNotMatch(body, /this\.addStage\(/, "a plane is not a stage")
})

test("a plane does not stand in for the floor", () => {
  // hasStage() suppresses the built-in ground and the far shadow cascade. A
  // stage brings its own floor and should; a card is scenery standing IN the
  // scene, and adding one must not delete the ground under everything.
  const hasStage = engine.slice(engine.indexOf("  hasStage(): boolean {"), engine.indexOf("  hasStage(): boolean {") + 200)
  assert.match(hasStage, /inst\.isStage/, "the floor question is about stages")
  assert.doesNotMatch(hasStage, /isPlane/, "a plane must not answer the floor question")
})

test("a plane skips everything a stage skips", () => {
  // Neither performs, so neither wants physics, IK, the cast buffer or the
  // camera clock. These are separate call sites and each one forgotten is its
  // own silent bug: a card solving IK, or claiming to be the subject an effect
  // follows.
  for (const [what, re] of [
    // The SUBJECT LIST, which is the one that was wrong: this used to be spelled
    // as the pattern below it and matched the focus loop instead, so a card sat
    // in the cast for as long as the test claimed it did not.
    ["the cast buffer", /if \(n >= MAX_EFFECT_SUBJECTS \|\| inst\.isStage \|\| inst\.isPlane\) return/],
    ["the focus target", /inst\.isStage \|\| inst\.isPlane\) continue\n      const model = inst\.model/],
    ["the camera clock", /if \(inst\.isStage \|\| inst\.isPlane\) continue\n      const p = inst\.model\.getAnimationProgress\(\)/],
    ["physics", /!isStage && !isPlane && rbs\.length > 0/],
    ["IK", /inst\.isStage \|\| inst\.isPlane \? false : this\.ikEnabled/],
  ]) {
    assert.match(engine, re, `a plane must skip ${what}`)
  }
})

test("a card is unlit, and casts no shadow", () => {
  // Both are silent when wrong. Ungrouped, a plane takes the neutral Principled
  // base and the scene's sun starts dimming artwork that has no sides; and
  // edgeFlag bit 0x04 is what castsShadow reads, so a gradient card would throw
  // a hard rectangle across the stage.
  assert.match(body, /graph: UNLIT_GRAPH/, "a card must not be shaded like a surface")
  assert.match(body, /this\.applyStyleGroups\(key, \[/, "the look must be a GROUP, so it can be swapped")
  assert.match(body, /edgeFlag: 0,/, "0 is both no outline and no shadow")
  const casts = engine.slice(engine.indexOf("const castsShadow ="), engine.indexOf("const castsShadow =") + 60)
  assert.match(casts, /edgeFlag & 0x04/, "if this bit changes, addPlane's edgeFlag comment is wrong")
})

test("a moving card writes into the texture it already has", () => {
  // Per FRAME, so it must not reallocate: a new texture means a new bind group
  // for the material behind it, every frame. The size check is what keeps that
  // true — a card is a fixed rectangle of texels, and the alternative to
  // refusing a mismatch is silently stretching one.
  const fn = engine.slice(engine.indexOf("  setPlaneFrame("), engine.indexOf("  setPlaneFrame(") + 900)
  assert.match(fn, /if \(tex\.width !== width \|\| tex\.height !== height\) return false/)
  assert.match(fn, /copyExternalImageToTexture/)
  assert.doesNotMatch(fn, /createTexture|createBindGroup/, "a frame update must not reallocate")
  // A moving card is allocated with ONE level so the guard below never fires:
  // rebuilding a pyramid per frame is a pass per level per card, which is most
  // of what a video plane used to cost.
  assert.match(engine, /inst\.dynamicTexture \? 1 : Math\.floor\(Math\.log2/, "a rewritten texture carries no mip chain")
})

test("removing a card forgets its texture", () => {
  // The texture cache frees it on removal; a stale entry would hand a destroyed
  // texture to the next frame push.
  const rm = engine.slice(engine.indexOf("  removeModel(name: string): void {"), engine.indexOf("  removeModel(name: string): void {") + 400)
  assert.match(rm, /this\.planeTextures\.delete\(name\)/)
})

test("a card draws in the opaque phase, before the ground", () => {
  // The scene pass runs opaque -> ground -> transparent, and the ground writes
  // depth at every opacity. A card in the transparent bucket is therefore
  // rejected by an INVISIBLE floor — which is what putting a picture into a
  // scene with the shadow catcher on used to look like.
  assert.match(engine, /const type: DrawCallType = inst\.isPlane \? "opaque"/)
})

