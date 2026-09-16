// The floor's height. Run: npm test.
//
// A camera motion authored for a taller or shorter character fits by moving the
// model and the floor together, so the floor needs a Y. Everything it touches is
// GPU-side — a vertex buffer, a uniform slot, a reflection plane — so these are
// source pins: the parts that must agree, checked where they are written.
//
// What must agree, and why each one bit when it was hardcoded to zero:
//   - all FOUR corners of the quad, or the plane is a wedge;
//   - the draw call's bounds, which is what anything reading the scene's extent
//     believes about where the floor is;
//   - the mirror plane, because a reflection across y = 0 slides off a floor
//     that is not there;
//   - the shader's own copy, which mirrors the eye across that plane.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"

const engine = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
const ground = readFileSync(new URL("../src/shaders/passes/ground.ts", import.meta.url), "utf8")

test("addGround takes a height and defaults it to MMD's floor", () => {
  // The option, in addGround's own block: sliced rather than matched across,
  // because the doc comment between them is longer than any lookahead worth
  // writing and a character count is a pin that breaks on an edited comment.
  // ...and the END is searched FROM the block, not from the file: an earlier
  // method closes an object parameter too, so a search from zero lands before
  // this block even starts and slices nothing at all.
  const from = engine.indexOf("addGround(options?: {")
  const options = engine.slice(from, engine.indexOf("}): void {", from))
  assert.match(options, /\n    y\?: number\n/, "the option exists")
  assert.match(engine, /const opts = \{\n      width: 160,\n      height: 160,\n      y: 0,\n/, "0 is the default")
  assert.match(engine, /this\.createGroundGeometry\(opts\.width, opts\.height, opts\.y\)/)
  assert.match(engine, /private createGroundGeometry\(width: number = 100, height: number = 100, y: number = 0\)/)
})

test("all four corners of the quad sit at that height", () => {
  const from = engine.indexOf("private createGroundGeometry")
  const body = engine.slice(from, engine.indexOf("const indices", from))
  const corners = body.match(/\n\s+-?halfWidth,\n\s+y,\n\s+-?halfHeight, \/\/ position/g) ?? []
  assert.equal(corners.length, 4, "a quad with a corner left at 0 is a wedge, not a floor")
  assert.doesNotMatch(body, /\n\s+-?halfWidth,\n\s+0,\n\s+-?halfHeight, \/\/ position/, "no corner still pinned to zero")
})

test("the bounds and the mirror plane follow the floor", () => {
  // The bounds are what the scene believes about where the floor is.
  assert.match(engine, /bounds: new Float32Array\(\[\n\s+-opts\.width \/ 2,\n\s+opts\.y,/)
  assert.match(engine, /opts\.width \/ 2,\n\s+opts\.y,\n\s+opts\.height \/ 2,\n\s+\]\)/)
  // (0, 1, 0, -y): the floor's own plane, up, offset to where it was put.
  assert.match(engine, /private groundY = 0/)
  assert.match(engine, /this\.groundY = opts\.y/)
  assert.match(engine, /this\.mirrorPlane\.set\(Engine\.GROUND_PLANE\)\n\s+this\.mirrorPlane\[3\] = -this\.groundY/)
})

test("the shader reflects across the floor's plane, not across zero", () => {
  // gb[19] was padding (_mb2); it carries the height now.
  assert.match(engine, /gb\[19\] = y/)
  assert.match(ground, /shadowSoftness: f32, groundY: f32,/)
  assert.doesNotMatch(ground, /_mb2/, "the slot is named for what it holds")
  // Mirroring a POINT across y = h is 2h - y; the forward vector is a direction
  // and still flips alone.
  assert.match(ground, /let eyeM = vec3f\(camera\.viewPos\.x, 2\.0 \* material\.groundY - camera\.viewPos\.y, camera\.viewPos\.z\);/)
  assert.match(ground, /let fwdM = vec3f\(fwd\.x, -fwd\.y, fwd\.z\);/)
  // Height above the plane, measured from the plane.
  assert.match(ground, /let height = max\(material\.groundY - \(eyeM\.y \+ dir\.y \* t\), 0\.0\);/)
})

test("the floor's height reaches the uniform the shader reads", () => {
  // The writer takes it and destructures it — a field added to the type but not
  // pulled out of `opts` compiles and writes undefined into the buffer.
  assert.match(engine, /private createShadowGroundResources\(opts: \{\n\s+diffuseColor: Vec3\n\s+y: number\n/)
  assert.match(engine, /const \{\n\s+diffuseColor,\n\s+y,\n/)
})
