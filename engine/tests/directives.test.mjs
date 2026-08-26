// One parser for everything an effect declares. Run: npm test.
//
// The old spelling put these in comments and anchored every parser to
// end-of-line, so a note on the same line silently unmade the directive: three
// shipped effects ran at half resolution and a fourth stopped being additive,
// with nothing failing and nothing reported. These tests pin the two properties
// that make that impossible now — a trailing note is ACCEPTED, and anything
// unrecognised is an ERROR rather than a shrug.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, readdirSync } from "node:fs"
import { dirname, join } from "node:path"
import { fileURLToPath } from "node:url"
import { parseDirectives, stripDirectives } from "../dist/shaders/directives.js"

const ok = (src) => {
  const r = parseDirectives(src)
  assert.deepEqual(r.errors, [], "expected no errors")
  return r.directives
}

test("a trailing note does not unmake the directive", () => {
  // The exact line that cost the Lyrics effect its resolution.
  const d = ok("#halfres — glyph edges are sub-pixel detail\n")
  assert.equal(d.fieldLayer, 1)
  assert.equal(ok("#anchor 左手首 trail — her sword hand\n").anchors[0].bone, "左手首")
  assert.equal(ok("#anchor 左手首 trail // the one that swings\n").anchors[0].trail, true)
})

test("full resolution is what an author gets for saying nothing", () => {
  // The default has to be the answer that cannot silently ruin an effect;
  // claiming to be cheap is a claim only the author can make.
  assert.equal(ok("fn background() {}\n").fieldLayer, 0)
  assert.equal(ok("#halfres\n").fieldLayer, 1)
})

test("an unknown directive is an error, not a shrug", () => {
  // The property `// @` could never have: a comment starting with @ is a
  // legitimate thing to write, so it could only ever be warned about.
  const r = parseDirectives("#fulres\n")
  assert.equal(r.errors.length, 1)
  assert.match(r.errors[0], /line 1/)
  assert.match(r.errors[0], /#fulres is not a directive/)
})

test("wrong arity is named, with its line", () => {
  for (const [src, re] of [
    ["\n\n#lights\n", /line 3.*takes 1 argument/],
    ["#layer over\n", /#layer takes "additive"/],
    ["#anchor\n", /a bone name/],
    ["#param float\n", /WGSL identifier/],
    ["#param float D\n", /needs a default/],
    ["#param float D 1 0\n", /both a min and a max/],
    ["#param color C red\n", /like #3b82f6/],
    ["#param vec3 W 0 1\n", /three numbers/],
    ["#param bool B 1\n", /float, color or vec3/],
  ]) {
    const r = parseDirectives(src)
    assert.equal(r.errors.length, 1, `expected one error for ${JSON.stringify(src)}`)
    assert.match(r.errors[0], re)
  }
})

test("params parse with and without a range", () => {
  const d = ok("#param float DENSITY 1.15 0 4\n#param color TINT #3b82f6\n#param vec3 WIND 0 1 0\n")
  assert.deepEqual(d.params, [
    { name: "DENSITY", kind: "float", value: 1.15, min: 0, max: 4 },
    { name: "TINT", kind: "color", value: "#3b82f6" },
    { name: "WIND", kind: "vec3", value: [0, 1, 0] },
  ])
  assert.deepEqual(ok("#param float G 2\n").params[0], { name: "G", kind: "float", value: 2 })
})

test("a name declared twice is caught", () => {
  // An author editing one line and forgetting the other; last-wins is a coin
  // toss they never see resolved.
  const r = parseDirectives("#param float D 1\n#param float D 2\n")
  assert.match(r.errors[0], /declared twice/)
})

test("stripping preserves line numbers", () => {
  // Diagnostics are rebased to the author's source, so a directive must leave
  // an empty line rather than closing the gap.
  const src = "#halfres\nfn a() {}\n#bloom\nfn b() {}\n"
  const out = stripDirectives(src)
  assert.equal(out.split("\n").length, src.split("\n").length)
  assert.equal(out.split("\n")[1], "fn a() {}")
  assert.equal(out.split("\n")[3], "fn b() {}")
  assert.doesNotMatch(out, /#/)
})

test("every directive the engine knows round-trips", () => {
  const d = ok(
    [
      "#halfres",
      "#layer additive",
      "#blend additive",
      "#particles 7000",
      "#lights 3",
      "#grid 256",
      "#bloom",
      "#dissolve",
      "#anchor 頭",
      "#anchor 左手首 trail",
    ].join("\n"),
  )
  assert.equal(d.fieldLayer, 1)
  assert.equal(d.additiveLayer, true)
  assert.equal(d.particleBlend, "additive")
  assert.equal(d.particles, 7000)
  assert.equal(d.lights, 3)
  assert.equal(d.grid, 256)
  assert.equal(d.bloom, true)
  assert.equal(d.dissolve, true)
  assert.deepEqual(d.anchors, [
    { bone: "頭", trail: false },
    { bone: "左手首", trail: true },
  ])
})

test("every directive the shipped effects use is one the parser knows", () => {
  // The vocabulary and the library have to agree, and an unknown directive is
  // now an ERROR — so a fixture using one the parser dropped is a scene that
  // refuses to install, not a property quietly missing.
  const dir = join(dirname(fileURLToPath(import.meta.url)), "fixtures")
  for (const f of readdirSync(dir).filter((n) => n.endsWith(".wgsl"))) {
    const r = parseDirectives(readFileSync(join(dir, f), "utf8"))
    assert.deepEqual(r.errors, [], `${f} declares something the parser rejects`)
  }
})

test("an effect can declare how long ONE firing lasts", () => {
  // A hit has an arc and its length is a fact about it, the way a video clip's
  // length is a fact about the file. Declaring it is what lets a host PLACE the
  // effect rather than make someone construct it.
  assert.equal(parseDirectives("#duration 3.5").directives.duration, 3.5)
  // Undeclared means ambient — stars, fog, rain. Not a length of zero: a
  // condition the scene is in, which a host spans across the whole scene.
  assert.equal(parseDirectives("// nothing").directives.duration, 0)
  // Prose mentioning one must not declare anything, the #anchor rule.
  assert.equal(parseDirectives("// runs #duration 4 or so").directives.duration, 0)
  // Zero and negative are rejected rather than silently taken: a strip of no
  // length plays nothing, and an author who typed it meant something else.
  assert.equal(parseDirectives("#duration 0").errors.length, 1)
  assert.equal(parseDirectives("#duration -2").errors.length, 1)
  assert.equal(parseDirectives("#duration").errors.length, 1)
  // A trailing note, the rule every directive follows.
  assert.equal(parseDirectives("#duration 2.5 — the whole flare").directives.duration, 2.5)
})

test("an install reports the effect's own declarations back", () => {
  // The host builds its controls and places its strip from what the INSTALL
  // returned, not from its own second parse of the same source — two readers of
  // one declaration is how they come to disagree, which is the trap the `// @`
  // era cost this project once already. So `params` and `duration` are siblings
  // on EffectResult, and neither may be the one you have to go and re-derive.
  const src = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  const at = src.indexOf("export type EffectResult")
  const type = src.slice(at, src.indexOf("\n}", at))
  assert.match(type, /params: EffectParamDecl\[\]/)
  assert.match(type, /duration: number/)
  // Every early return has to carry them too, or a failed install is a
  // different shape from a successful one and a caller has to test for it.
  assert.doesNotMatch(src, /mounts: noMounts, params: \[\] \}/)
})
