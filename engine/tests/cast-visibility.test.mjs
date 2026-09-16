// Who the cast is, and whose cloth keeps moving. Run: npm test.
//
// Two rules a scene with an alternate skin depends on, both GPU-side, so both
// pinned where they are written:
//
//   1. A HIDDEN model is not a subject. A costume's twin sits loaded and
//      invisible; with it in the cast it takes subject 0 by load order, and
//      Teleportation spawns its motes off a body nobody can see.
//   2. A hidden model's cloth keeps simulating when the host asks. The body of a
//      hidden model animates regardless — the skirt is what falls behind, and a
//      skirt caught up at the reveal snaps in front of the audience.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"

const engine = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")

test("the cast is who is on stage", () => {
  // The subject write and the name lookup must agree, or an effect draws off one
  // body while the dissolve takes another apart.
  const writes = engine.match(
    /if \([^)]*n >= MAX_EFFECT_SUBJECTS \|\| inst\.isStage \|\| inst\.isPlane \|\| inst\.isProp \|\| !inst\.model\.visible\)/g,
  ) ?? []
  assert.equal(writes.length, 2, "both the cast write and castSubjectName skip hidden models")
  assert.doesNotMatch(
    engine,
    /n >= MAX_EFFECT_SUBJECTS \|\| inst\.isStage \|\| inst\.isPlane \|\| inst\.isProp\) return/,
    "no subject filter still lets a hidden model through",
  )
})

test("cloth keeps moving while hidden only when asked", () => {
  assert.match(engine, /simulateWhileHidden: boolean/, "the instance carries the flag")
  assert.match(engine, /const inst: ModelInstance = \{\n\s+name,\n\s+model,\n\s+simulateWhileHidden: false,/, "off by default")
  assert.match(
    engine,
    /if \(inst\.physics && this\.physicsEnabled && \(inst\.model\.visible \|\| inst\.simulateWhileHidden\)\) \{/,
    "the physics gate widens to the flag",
  )
  const setter = engine.slice(engine.indexOf("setModelPhysicsWhileHidden(modelName"))
  assert.match(setter.slice(0, 260), /inst\.simulateWhileHidden = on/, "the setter writes it")
})
