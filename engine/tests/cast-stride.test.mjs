// The cast buffer's subject stride, read the same way everywhere. Run: npm test.
//
// Each subject is EFFECT_SUBJECT_VEC4S vec4s — root, hip+id, bounds, gaze — and a
// shader that walks the subjects with a written-out number reads subject 0 right
// and every other subject from the wrong slots. That is exactly how outlines,
// ribbons and particles came to follow only the first character once the gaze
// slot made the stride four: subject 1's "id" was its dissolve, 1.0, which no
// pixel carries. So no shader may spell the stride as a literal.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join, relative } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const shaders = join(here, "../src/shaders")

function sources(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) return sources(path)
    return name.endsWith(".ts") ? [path] : []
  })
}

test("no shader indexes the cast's subjects with a literal stride", () => {
  const offenders = []
  for (const path of sources(shaders)) {
    const text = readFileSync(path, "utf8")
    text.split("\n").forEach((line, i) => {
      if (/_rzCast\[\s*i\s*\*\s*\d+\s*[+\]]/.test(line)) offenders.push(`${relative(shaders, path)}:${i + 1}: ${line.trim()}`)
    })
  }
  assert.deepEqual(offenders, [], "index subjects by EFFECT_SUBJECT_VEC4S, never a number")
})

test("the subject id and the subject count read the slots the engine writes", async () => {
  const { EFFECT_SUBJECT_VEC4S } = await import("../dist/shaders/cast-layout.js")
  const { CAST_API } = await import("../dist/shaders/cast-api.js")
  const stride = EFFECT_SUBJECT_VEC4S
  // writeCastEntry puts the object id in the hip vec4's w, and the bounds (w =
  // radius, always > 0 for a present subject) one slot further on.
  assert.match(CAST_API, new RegExp(`_rzCast\\[i \\* ${stride} \\+ 1\\]\\.w`), "rzSubjectId reads the hip vec4's w")
  const engine = readFileSync(join(here, "../src/engine.ts"), "utf8")
  assert.match(engine, /cd\[b \+ 7\] = inst\.objectId/, "the engine writes the id at slot 1, w")
  assert.match(engine, /cd\[b \+ 11\] = height \* 0\.75/, "and the bounds radius at slot 2, w")
})
