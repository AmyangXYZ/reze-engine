// Every registry entry emits a call. Does the function it calls actually exist?
//
// The registry and the shader library are separate files that agree only by
// convention, and TypeScript cannot check across that seam — a typo'd function
// name compiles fine and fails on the GPU, at the moment a user applies the
// graph. This walks every node type and checks the seam directly.

import { test } from "node:test"
import assert from "node:assert/strict"
import { NODE_REGISTRY } from "../dist/graph/registry.js"
import { NODES_WGSL } from "../dist/shaders/materials/nodes.js"
import { COMMON_BINDINGS_WGSL } from "../dist/shaders/materials/common.js"

const LIB = NODES_WGSL + COMMON_BINDINGS_WGSL
// Functions AND structs: a struct is callable in WGSL as its own constructor,
// which is how the principled node passes its inputs.
const defined = new Set([
  ...[...LIB.matchAll(/\bfn\s+([A-Za-z_]\w*)\s*\(/g)].map((m) => m[1]),
  ...[...LIB.matchAll(/\bstruct\s+([A-Za-z_]\w*)\s*\{/g)].map((m) => m[1]),
])

// WGSL builtins and constructors a node may legitimately emit directly.
const BUILTIN = new Set([
  "abs", "acos", "asin", "atan", "atan2", "ceil", "clamp", "cos", "cross", "degrees", "distance",
  "dot", "exp", "exp2", "floor", "fract", "inverseSqrt", "length", "log", "log2", "max", "min",
  "mix", "normalize", "pow", "radians", "reflect", "refract", "round", "saturate", "select",
  "sign", "sin", "smoothstep", "sqrt", "step", "tan", "trunc", "textureSample", "textureSampleLevel",
  "vec2f", "vec3f", "vec4f", "f32", "i32", "u32", "mat3x3", "select",
])

/** Ask each node to emit, with an argument per socket, and read back the calls. */
function callsOf(spec) {
  if (!spec.emit) return []
  const args = Object.fromEntries(Object.keys(spec.inputs).map((k) => [k, `ARG_${k}`]))
  let src
  try {
    src = spec.emit(args)
  } catch (e) {
    return [{ broken: String(e) }]
  }
  return [...String(src).matchAll(/\b([A-Za-z_]\w*)\s*\(/g)].map((m) => m[1])
}

test("every emitted function exists in the shader library", () => {
  const missing = []
  for (const [type, spec] of Object.entries(NODE_REGISTRY)) {
    for (const fn of callsOf(spec)) {
      if (fn && fn.broken) { missing.push(`${type}: emit threw — ${fn.broken}`); continue }
      if (!defined.has(fn) && !BUILTIN.has(fn)) missing.push(`${type} → ${fn}()`)
    }
  }
  assert.deepEqual(missing, [], `node types calling functions that do not exist:\n  ${missing.join("\n  ")}`)
})

test("every emitted call consumes only sockets the node declares", () => {
  // Catches the other half of the seam: an emit referencing a socket that was
  // renamed out from under it silently produces `undefined` in the WGSL.
  const bad = []
  for (const [type, spec] of Object.entries(NODE_REGISTRY)) {
    if (!spec.emit) continue
    const args = Object.fromEntries(Object.keys(spec.inputs).map((k) => [k, `ARG_${k}`]))
    const src = String(spec.emit(args))
    if (src.includes("undefined")) bad.push(`${type}: ${src}`)
  }
  assert.deepEqual(bad, [], `emits referencing sockets that do not exist:\n  ${bad.join("\n  ")}`)
})

test("outputSelect only names declared outputs", () => {
  const bad = []
  for (const [type, spec] of Object.entries(NODE_REGISTRY)) {
    for (const key of Object.keys(spec.outputSelect ?? {})) {
      if (!(key in spec.outputs)) bad.push(`${type}: outputSelect.${key} has no matching output`)
    }
  }
  assert.deepEqual(bad, [])
})
