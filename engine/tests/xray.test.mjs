// The see-through window (#xray): a rectangle held in front of the lens. Behind
// it the materials the host marks on the characters it is aimed at — their
// clothes — are not drawn, so what is under them shows. Run: npm test.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { parseDirectives } from "../dist/shaders/directives.js"

const src = (p) => readFileSync(new URL(`../src/${p}`, import.meta.url), "utf8")

test("#xray names its four dials", () => {
  const { directives, errors } = parseDirectives(`#param float X 0 -0.5 0.5
#param float Y 0 -0.5 0.5
#param float W 0.3 0 1
#param float H 0.4 0 1
#xray X Y W H`)
  assert.deepEqual(errors, [])
  assert.deepEqual(directives.xray, { x: "X", y: "Y", width: "W", height: "H" })
})

test("#xray refuses dials that are missing or not numbers", () => {
  const { errors } = parseDirectives(`#param float X 0
#param vec3 Y 0 0 0
#param float W 0.3
#xray X Y W H`)
  assert.equal(errors.length, 2)
  assert.match(errors.join("\n"), /Y has to be a #param float/)
  assert.match(errors.join("\n"), /H has to be a #param float/)
})

test("every pass that draws her runs the window's test, the shadow excepted", () => {
  // The colour pass cuts the marked materials, the depth prepass agrees with it,
  // and the outline cuts their lines and keeps hers.
  assert.match(src("graph/slots.ts"), /if \(rz_xray_cut\(input\.position, material\.materialId\)\) \{ discard; \}/)
  assert.match(src("shaders/passes/depth-prepass.ts"), /if \(rz_xray_cut\(in\.position, material\.materialId\)\) \{ discard; \}/)
  assert.match(src("shaders/passes/outline.ts"), /if \(rz_xray_cut\(input\.position, material\.xray\.x\)\) \{ discard; \}/)
  // What the window reveals is lit, not left in the shadow of the clothes it cut.
  assert.match(src("graph/slots.ts"), /let shadow = select\(sampleShadow\(input\.worldPos, n\), 1\.0, rz_xray_inside\(input\.position\)\);/)
  // A window on the lens changes nothing about what casts a shadow.
  assert.doesNotMatch(src("shaders/passes/shadow.ts"), /rz_xray/)
})

test("the window and its material mask sit after bloom in the object block", () => {
  const engine = src("engine.ts")
  assert.match(engine, /inst\.objectLight\.set\(rect \?\? new Float32Array\(16\), 100\)/)
  assert.match(engine, /new Uint32Array\(inst\.objectLight\.buffer, 116 \* 4, 64\)/)
  assert.match(src("shaders/materials/common.ts"), /bloom: vec4f, xray: array<vec4f, 4>, xrayMask: array<vec4u, 16> \}/)
})
