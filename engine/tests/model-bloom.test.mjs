// A model left out of the bloom. Run: npm test.
//
// The bloom prefilter multiplies every pixel by the aux mask's .r. A model's
// surfaces and outlines write its ObjectLight's bloom.x there, 1 unless the host
// turned it off, so one character can stop glowing while the scene keeps its
// bloom.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"

const src = (p) => readFileSync(new URL(`../src/${p}`, import.meta.url), "utf8")
const engine = src("engine.ts")

test("every model starts in the bloom", () => {
  assert.match(engine, /const objectLight = new Float32Array\(100\)\n\s+objectLight\[96\] = 1\n/)
})

test("setModelBloom writes the slot the shaders read", () => {
  const setter = engine.slice(engine.indexOf("setModelBloom(name: string, on: boolean)"))
  assert.match(setter.slice(0, 400), /inst\.objectLight\[96\] = on \? 1 : 0/)
  assert.match(setter.slice(0, 400), /writeBuffer\(inst\.lightBuffer, 96 \* 4, inst\.objectLight, 96, 4\)/)
  assert.match(src("shaders/materials/common.ts"), /jelly: array<vec4f, 12>, bloom: vec4f \}/, "bloom follows the 48 jelly words")
})

test("surfaces and outlines write the model's bloom as the mask", () => {
  const slots = src("graph/slots.ts")
  assert.equal(slots.match(/out\.mask = vec4f\(objectLight\.bloom\.x, 1\.0, _rzSubsurface, out\.color\.a\);/g)?.length, 2)
  assert.doesNotMatch(slots, /out\.mask = vec4f\(1\.0,/)
  const outline = src("shaders/passes/outline.ts")
  assert.match(outline, /@group\(1\) @binding\(1\) var<uniform> objectLight: ObjectLight;/)
  assert.match(outline, /out\.mask = vec4f\(objectLight\.bloom\.x, 1\.0, 0\.0, out\.color\.a\);/)
})

test("one ObjectLight declaration for every pass", () => {
  for (const p of ["shaders/materials/common.ts", "shaders/passes/shadow.ts", "shaders/passes/outline.ts"]) {
    assert.doesNotMatch(src(p), /^struct ObjectLight \{/m, p)
  }
})
