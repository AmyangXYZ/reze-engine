// The ground's depth contract, checked against the SOURCE. Run: npm test.
//
// The ground is the only surface in the scene whose VISIBILITY is a material
// uniform rather than a draw decision: `opacity` fades the surface while the
// shadow-catcher term stays, so a scene sets ground opacity to 0 and still has
// a floor. Everything that locates something on the floor by reading drawn
// depth — a foreground effect, DoF autofocus — depends on the plane writing
// depth at every opacity, including 0.
//
// That belief was once written down as its opposite. Footprints placed its
// marks at the drawn surface, they vanished on a scene with ground opacity 0,
// and the conclusion recorded in its source was that an invisible ground
// "writes no depth". It does write depth, and did then: the pipeline sets
// depthWriteEnabled and the shader has no discard, so a fragment that shades to
// alpha 0 still lands in the depth buffer. The test below is what that
// paragraph should have been — the property stated where a change to it fails,
// rather than asserted in a comment on an effect that no longer reads depth.
//
// Source rather than dist, and rather than a GPU: there is no device here, and
// what is being pinned is what the code SAYS to build.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const engine = readFileSync(join(here, "../src/engine.ts"), "utf8")
const shader = readFileSync(join(here, "../src/shaders/passes/ground.ts"), "utf8")

/** The object literal starting at `from`, by brace matching. */
function objectAt(src, from) {
  let depth = 0
  let i = src.indexOf("{", from)
  const start = i
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++
    else if (src[i] === "}") depth--
    if (depth === 0) return src.slice(start, i + 1)
  }
  throw new Error("unbalanced braces")
}

test("the ground pipeline writes depth", () => {
  const at = engine.indexOf('label: "ground shadow pipeline"')
  assert.ok(at > 0, "ground shadow pipeline not found — it was renamed, and this test went blind with it")
  const desc = objectAt(engine, engine.lastIndexOf("createRenderPipeline", at))
  assert.match(
    desc,
    /depthWriteEnabled:\s*true/,
    "the ground must write depth at every opacity: a scene with an invisible floor still has a floor, " +
      "and anything locating a point on it by drawn depth reads the far plane without this",
  )
})

test("the ground shader never discards", () => {
  // The radial fade returns early on a fully faded pixel, which is a shading
  // shortcut and NOT a discard — the fragment still writes depth. A discard
  // here would be the same bug as depthWriteEnabled: false, arrived at from the
  // other side, and it would be invisible in the pipeline descriptor above.
  assert.doesNotMatch(
    shader,
    /\bdiscard\b/,
    "a discarded fragment writes no depth, which would punch holes in the floor for everything that " +
      "reads depth to find it",
  )
})

test("nothing skips the ground draw on opacity", () => {
  const at = engine.indexOf("private renderGround(")
  assert.ok(at > 0, "renderGround not found")
  const body = engine.slice(at, engine.indexOf("\n  }", at))
  // Suppression is a STAGE decision (a stage brings its own floor and the two
  // z-fight). Opacity is a material uniform and must never reach this path.
  assert.doesNotMatch(body, /opacity/i, "renderGround must not gate the draw on opacity — see the pipeline test above")
})
