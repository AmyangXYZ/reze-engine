// Bind group / layout agreement, checked against the SOURCE. Run: npm test.
//
// These exist because both halves of this contract are invisible to TypeScript
// and to every other test here: a bind group whose bindings disagree with its
// layout compiles perfectly and fails at pipeline or encode time, usually
// blaming a pass that had nothing to do with it. Adding the score interface
// broke both halves in one afternoon — a duplicated binding index in the field
// group, and a stray layout entry that landed on the morph compute layout
// because the line it was matched against was not unique. The morph pass was
// what the console named; the morph pass was not what was wrong.
//
// Source rather than dist, because this is about what the code SAYS. A regex
// cannot see entries built by a spread or a loop, so anything assembled that
// way is skipped by name rather than guessed at — see SPREAD_BUILT.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const src = readFileSync(join(dirname(fileURLToPath(import.meta.url)), "../src/engine.ts"), "utf8")

/** Layouts whose bind groups assemble entries from a spread or a loop, so the
 *  literal bindings in the call are only part of the set. Checking coverage for
 *  these would report a mismatch that is not one. */
const SPREAD_BUILT = new Set(["mainPerMaterialBindGroupLayout"])

/** Every `createBindGroup({...})` / `createBindGroupLayout({...})` block, by
 *  brace matching — the objects nest, so a regex alone cannot find the end. */
function blocks(pattern) {
  const out = []
  for (const m of src.matchAll(pattern)) {
    let depth = 0
    let i = m.index + m[0].length - 1
    for (;;) {
      if (src[i] === "{") depth++
      else if (src[i] === "}") depth--
      if (depth === 0) break
      i++
    }
    out.push({ line: src.slice(0, m.index).split("\n").length, text: src.slice(m.index, i) })
  }
  return out
}

const bindingsIn = (text) => [...text.matchAll(/binding:\s*(\d+)/g)].map((m) => Number(m[1]))

test("no bind group or layout declares the same binding twice", () => {
  for (const [kind, pattern] of [
    ["bind group", /createBindGroup\(\{/g],
    ["layout", /createBindGroupLayout\(\{/g],
  ]) {
    for (const { line, text } of blocks(pattern)) {
      const nums = bindingsIn(text)
      const dupes = [...new Set(nums.filter((n) => nums.filter((x) => x === n).length > 1))]
      assert.deepEqual(dupes, [], `${kind} at engine.ts:${line} repeats binding(s) ${dupes.join(", ")}`)
    }
  }
})

test("every bind group covers exactly its layout's bindings", () => {
  const layouts = new Map()
  for (const { text } of blocks(/this\.\w+\s*=\s*this\.device\.createBindGroupLayout\(\{/g)) {
    const name = /this\.(\w+)\s*=/.exec(text)[1]
    layouts.set(name, [...new Set(bindingsIn(text))].sort((a, b) => a - b))
  }
  assert.ok(layouts.size > 10, "found suspiciously few named layouts — did the parse break?")

  for (const { line, text } of blocks(/createBindGroup\(\{/g)) {
    const ref = /layout:\s*this\.(\w+)/.exec(text)
    if (!ref || !layouts.has(ref[1]) || SPREAD_BUILT.has(ref[1])) continue
    const want = layouts.get(ref[1])
    const have = [...new Set(bindingsIn(text))].sort((a, b) => a - b)
    assert.deepEqual(
      have,
      want,
      `bind group at engine.ts:${line} against ${ref[1]}: layout wants [${want}], group provides [${have}]`,
    )
  }
})

/** The class method containing `needle`, by brace matching from its signature. */
function methodContaining(needle) {
  const at = src.indexOf(needle)
  assert.ok(at > 0, `not found in engine.ts: ${needle}`)
  // The nearest method signature above it — a line at class-body indent that
  // opens a brace. Nested closures are indented deeper and never match.
  let start = -1
  for (const m of src.matchAll(/^ {2}(?:private |protected |public )?(?:async )?[A-Za-z_]\w*\([^\n]*\{$/gm)) {
    if (m.index < at) start = m.index
    else break
  }
  assert.ok(start >= 0, "no enclosing method found")
  let depth = 0
  let i = src.indexOf("{", start)
  for (; i < src.length; i++) {
    if (src[i] === "{") depth++
    else if (src[i] === "}") depth--
    if (depth === 0) break
  }
  return src.slice(start, i)
}

// A buffer bound before it is created reads as `{ buffer: undefined }`, and
// WebGPU rejects the whole bind group with "Required member is undefined" —
// naming the descriptor, never the field. TypeScript cannot see it either: the
// definite-assignment `!` on these fields is what makes the class compile at
// all, and it silences exactly this.
//
// Sound only WITHIN one function, which is why this is scoped rather than
// file-wide: across methods, file order says nothing about call order, and
// seven fields legitimately appear bound above where they are assigned. Inside
// a single body, statement order IS execution order.
//
// This is a real bug, shipped 2026-08-16: the lights buffer was created after
// the per-frame bind group that binds it, and the app failed to start.
test("a bind group's buffers are created before the bind group that binds them", () => {
  for (const marker of ['label: "main per-frame bind group"', 'label: "composite bind group"']) {
    const body = methodContaining(marker)
    for (const use of body.matchAll(/\{\s*buffer:\s*this\.([A-Za-z0-9_]+)\s*\}/g)) {
      const assign = body.indexOf(`this.${use[1]} = `)
      // Assigned in ANOTHER method: this test cannot order those, and says so
      // rather than guessing.
      if (assign < 0) continue
      assert.ok(
        assign < use.index,
        `this.${use[1]} is bound before it is assigned, inside the method holding ${marker} — ` +
          `the bind group will see undefined and WebGPU will reject the whole descriptor`,
      )
    }
  }
})
