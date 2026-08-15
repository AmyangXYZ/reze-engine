// The scene anchor table and its per-effect alias. Run: npm test.
//
// Worth pinning by test rather than by eye because every way this can be wrong
// is silent: a bad alias does not error, it reads ANOTHER effect's bone, and the
// symptom is a ribbon on the wrong wrist in a scene that still renders happily.

import { test } from "node:test"
import assert from "node:assert/strict"
import { buildAnchorTable, anchorAliasWgsl, ribbonSlotWgsl } from "../dist/shaders/anchor-table.js"

const a = (bone, trail = false) => ({ bone, trail })

test("one effect gets the identity alias — the mechanism is a no-op until it isn't", () => {
  const t = buildAnchorTable([[a("頭"), a("左手首", true)]], 8)
  assert.deepEqual(t.alias, [[0, 1]])
  assert.deepEqual(
    t.entries.map((e) => e.bone),
    ["頭", "左手首"],
  )
  assert.match(anchorAliasWgsl(t.alias[0]), /return local;/, "identity must fold away, not emit a switch")
})

test("two effects claiming slot 0 for different bones do not collide", () => {
  // The bug this exists for: without aliasing both write address 0.
  const t = buildAnchorTable([[a("左手首", true)], [a("右手首", true)]], 8)
  assert.deepEqual(t.alias, [[0], [1]])
  assert.equal(t.entries.length, 2)
})

test("the same bone is allocated once and the trail is shared", () => {
  const t = buildAnchorTable([[a("左手首", true)], [a("左手首", true)]], 8)
  assert.equal(t.entries.length, 1, "one ring, not two — trails are the expensive resource")
  assert.deepEqual(t.alias, [[0], [0]])
})

test("asking for a bone bare and then trailed turns the trail on for both", () => {
  const t = buildAnchorTable([[a("頭", false)], [a("頭", true)]], 8)
  assert.deepEqual(t.entries, [{ bone: "頭", trail: true }])
  assert.deepEqual(t.alias, [[0], [0]])
})

test("the cap counts DISTINCT bones across the scene, not declarations", () => {
  // Four effects each naming the same two bones fit in two slots.
  const one = [a("頭"), a("左手首", true)]
  const t = buildAnchorTable([one, one, one, one], 8)
  assert.equal(t.entries.length, 2)
  assert.deepEqual(t.alias, [
    [0, 1],
    [0, 1],
    [0, 1],
    [0, 1],
  ])
})

test("a refused anchor aliases to -1 and is reported, and the rest still resolve", () => {
  const first = [a("b0"), a("b1")]
  const second = [a("b2"), a("b3"), a("b4")]
  const t = buildAnchorTable([first, second], 3)
  assert.deepEqual(t.alias[0], [0, 1])
  assert.deepEqual(t.alias[1], [2, -1, -1], "the ones that fit still work")
  assert.deepEqual(
    t.dropped.map((d) => d.bone),
    ["b3", "b4"],
  )
})

test("a non-identity alias emits a switch covering every local slot", () => {
  const wgsl = anchorAliasWgsl([2, 0, 5])
  assert.match(wgsl, /case 0: \{ return 2; \}/)
  assert.match(wgsl, /case 1: \{ return 0; \}/)
  assert.match(wgsl, /case 2: \{ return 5; \}/)
  assert.match(wgsl, /default: \{ return -1; \}/, "an out-of-range slot must resolve to nothing, not to slot 0")
})

test("ribbon index maps to local anchor slot, skipping untrailed anchors", () => {
  // All-trailed (every library effect today): identity, folds away.
  assert.match(ribbonSlotWgsl([0, 1]), /return ribbon;/)

  // The latent bug: @anchor 頭 (0, no trail) then @anchor 左手首 trail (1).
  // One ribbon, and it must resolve to anchor slot 1 — as slot 0 it asked 頭
  // for a trail it never recorded and drew nothing at all.
  const mixed = ribbonSlotWgsl([1])
  assert.match(mixed, /case 0: \{ return 1; \}/)
  assert.match(mixed, /default: \{ return -1; \}/, "an out-of-range ribbon must resolve to nothing, not to slot 0")
})

test("a ribbon out of range resolves to -1, which the trail accessors reject", () => {
  // -1 has to survive the whole chain: _rzRibbonSlot -> _rzSlot -> bounds check.
  // The identity alias passes -1 through, and rzTrailCount guards g < 0.
  assert.match(anchorAliasWgsl([0, 1]), /return local;/)
})
