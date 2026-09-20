// Which models an effect is on. Run: npm test.
//
// The filter lives in the INDEX SPACE the shaders read, not in the shaders: an
// effect aimed at one dancer counts one subject and finds her at index 0. That is
// what lets every shipped built-in be aimed without a line changing, and it is
// also what makes these assertions worth pinning — the remap is invisible from
// any single file, and a mount that reads the cast through an unfiltered index is
// an effect that quietly applies to everybody again.
//
// Source patterns rather than a live device: every one of these is a GPU-side
// contract between two files that must agree, which is exactly what no runtime
// check reaches.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"

const read = (p) => readFileSync(new URL(p, import.meta.url), "utf8")
const engine = read("../src/engine.ts")
const castApi = read("../src/shaders/cast-api.ts")
const composite = read("../src/shaders/passes/composite.ts")
const particles = read("../src/shaders/passes/particles.ts")
const trails = read("../src/shaders/passes/trails.ts")
const grid = read("../src/shaders/passes/grid.ts")
const lights = read("../src/shaders/lights.ts")
const castDistance = read("../src/shaders/passes/cast-distance.ts")

test("every cast accessor goes through the remap", () => {
  // One of them reading the raw index is one place an effect sees a model it was
  // not aimed at — and the anchors and trails are addressed per subject, so the
  // wrong index there is another character's wrist.
  for (const fn of ["rzSubjectId", "rzSubject", "rzAnchor", "rzTrailCount", "rzTrail"]) {
    const at = castApi.indexOf(`fn ${fn}(`)
    assert.ok(at > 0, `${fn} exists`)
    const body = castApi.slice(at, castApi.indexOf("\n}", at))
    assert.match(body, /_rzSubjectSlot\(/, `${fn} maps its local index to a cast slot`)
  }
})

test("the count is the effect's own, and the slot walk agrees with it", () => {
  const count = castApi.slice(castApi.indexOf("fn rzSubjectCount()"), castApi.indexOf("fn _rzSubjectSlot("))
  assert.match(count, /_rzCastLive\(\)/, "bounded by the live cast")
  assert.match(count, /_rzSubjectMask\(\)/, "and narrowed by the mask")
  const slot = castApi.slice(castApi.indexOf("fn _rzSubjectSlot("), castApi.indexOf("/** Which model this is"))
  assert.match(slot, /_rzCastLive\(\)/)
  assert.match(slot, /_rzSubjectMask\(\)/)
  assert.match(slot, /return -1;/, "past its own count is invalid, not slot 0")
})

test("every mount supplies the two names the cast API needs", () => {
  // A mount that forgets one does not compile, which is the good failure. A mount
  // that supplies a CONSTANT compiles and silently stops following the target,
  // which is why these are pinned to the uniforms they read.
  assert.match(composite, /fn _rzCastLive\(\) -> i32 \{ return i32\(viewU\[10\]\.w\); \}/, "field/grid/lights read the view uniform")
  assert.match(composite, /subjectMaskApi\("u32\(_rzFieldClock\.z\)"\)/, "a field mount reads its own clock block")
  assert.match(composite, /subjectMaskApi\(CAST_MASK_ALL\)/, "the base composite carries no effect, so it sees the whole cast")
  assert.match(particles, /fn _rzCastLive\(\) -> i32 \{\n\s+var n = 0;/, "particles scan the buffer")
  assert.match(particles, /subjectMaskApi\("u32\(pu\.subjects\)"\)/)
  assert.match(trails, /fn _rzCastLive\(\) -> i32 \{\n\s+var n = 0;/, "trails scan it too")
  assert.match(trails, /subjectMaskApi\("u32\(tu\.mask\)"\)/)
  assert.match(grid, /subjectMaskApi\("u32\(su\.mask\)"\)/)
  assert.match(lights, /subjectMaskApi\("u32\(_rzLightU\[1\]\.x\)"\)/)
})

test("the engine writes the mask into every one of those uniforms", () => {
  assert.match(engine, /this\.fieldClockScratch\[2\] = e\.subjectMask/, "field")
  assert.match(engine, /p\.data\[5\] = e\.subjectMask/, "particles")
  assert.match(engine, /t\.data\[3\] = e\.subjectMask/, "trails")
  assert.match(engine, /grid\.data\[4\] = e\.subjectMask/, "grid")
  assert.match(engine, /l\.data\[4\] = e\.subjectMask/, "lights")
  // Both blocks grew a second vec4 to hold it. A buffer left at 16 bytes reads
  // the mask as garbage — or as zero, which is an effect on nobody.
  const gridBuf = engine.slice(engine.indexOf('label: "grid uniforms"'), engine.indexOf('label: "grid uniforms"') + 300)
  assert.match(gridBuf, /size: 32,/)
  const lightBuf = engine.slice(engine.indexOf("const data = new Float32Array(8)"), engine.indexOf('label: "light emit uniform"'))
  assert.ok(lightBuf.length > 0, "the light emit uniform is two vec4s")
})

test("a mask is resolved where the cast slots are decided", () => {
  // Names, not slots: the cast is every visible character in load order, so
  // hiding one moves everybody after them up. A mask computed anywhere else is a
  // mask that can be one frame — or one hidden model — out of date.
  const at = engine.indexOf("this.castSlotOf.clear()")
  const write = engine.slice(at, engine.indexOf("this.updateSubjectMasks()", at) + 40)
  assert.match(write, /this\.castSlotOf\.set\(inst\.name, n\)/, "the slot is recorded as it is assigned")
  assert.match(write, /this\.updateSubjectMasks\(\)/, "and every effect's mask follows in the same pass")
  const masks = engine.slice(engine.indexOf("private updateSubjectMasks()"), engine.indexOf("One character's slice of the effect API"))
  assert.match(masks, /let mask = 0xf/, "aimed at nobody in particular is the whole cast")
  assert.match(masks, /this\.castSlotOf\.get\(name\)/)
  assert.match(masks, /e\.subjectCount = n/, "and the CPU-side count the ribbons are sized by")
})

test("ribbons are instanced by the effect's own subjects", () => {
  // The trail shader decodes [ribbon][subject][segment] out of tu.subjects. Size
  // the draw by the scene's cast instead and a ribbon aimed at one dancer draws
  // four, three of them reading another character's path.
  const draw = engine.slice(engine.indexOf("private drawTrails("), engine.indexOf("private drawTrails(") + 2000)
  assert.match(draw, /const live = Math\.max\(1, e\.subjectCount\)/)
  assert.match(draw, /t\.data\[1\] = live/)
  assert.match(draw, /pass\.draw\(6, t\.slots \* live \*/)
})

test("the distance field is built per target set, not filtered afterwards", () => {
  // "How far is the nearest cast pixel" is a different number for a smaller cast,
  // not a subset of the same one — the model left out WAS the answer, and the
  // distance to the one kept is unknown. Filtering it would bite a hole in an
  // aura wherever another character passed nearer.
  assert.match(castDistance, /@binding\(3\) var<uniform> _rzSeedU: vec4f;/, "the seed pass takes a mask")
  const seeds = castDistance.slice(castDistance.indexOf("fn rzSeeds("), castDistance.indexOf("fn rzSeeds(") + 400)
  assert.match(seeds, /let mask = u32\(_rzSeedU\.x\);/)
  assert.match(seeds, /if \(\(mask & \(1u << u32\(i\)\)\) == 0u\) \{ continue; \}/, "an unmasked subject does not seed")
  assert.match(engine, /for \(const v of this\.castDistanceVariants\) \{\n\s+const seed = encoder\.beginRenderPass/, "one chain per variant")
  assert.match(engine, /castSubjectKey\(v\.subjects\) === key/, "effects on the same models share one flood")
  assert.match(engine, /\{ binding: 26, resource: this\.castDistViewFor\(owner\) \}/, "and each effect binds its own")
})

test("aiming an effect is a setter, not a reinstall", () => {
  const setter = engine.slice(engine.indexOf("setEffectSubjects(index: number"), engine.indexOf("getEffectSubjects(index: number"))
  assert.match(setter, /fx\.subjects = normalizeSubjects\(models\)/)
  assert.match(setter, /if \(fx\.readsCastDistance\) this\.syncCastDistanceVariants\(\)/, "only the fields have to be rebuilt")
  assert.doesNotMatch(setter, /compile|createShaderModule/, "nothing recompiles: the mask is a uniform")
  // An empty list is the whole cast, not nobody: an effect on nobody is a dark
  // effect with no sign of why, and influence 0 already says that on purpose.
  const norm = engine.slice(engine.indexOf("function normalizeSubjects("), engine.indexOf("const EFFECT_PARAMS_BINDING = 7"))
  assert.match(norm, /if \(!list \|\| list\.length === 0\) return null/)
})

test("the install says whether aiming means anything", () => {
  // Rain falls on the scene and a glitch is on the lens. A host that offers the
  // control anyway offers one that does nothing.
  assert.match(engine, /readsCast: \/\\brz\(\?:Subject\|SubjectCount\|SubjectId\|Anchor\|Trail\|TrailCount\|CastDistance\)\\s\*\\\(\/\.test\(wgsl\) \|\|\n\s+anchors\.length > 0,/)
  assert.match(engine, /readsCast: built\.instance\.readsCast,/, "reported off the instance, not a second parse")
})
