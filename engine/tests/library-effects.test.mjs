// Real published effects, assembled through the real builders. Run: npm test.
//
// The gap this closes: every other test here checks engine code against engine
// code, so a change to the WGSL surface AUTHORS write against passes everything
// and then fails in a browser. That happened twice in one week — most recently
// RZ_TRAIL_SLOTS, which the engine itself never references, so renaming it read
// as a tidy-up right up until Hand Ribbon stopped installing.
//
// The fixtures in ./fixtures are verbatim copies of built-ins from the design
// app's content/effects.json, chosen to cover every mount path:
//   hand-ribbon      particles + trails  (and the only user of RZ_TRAIL_SLOTS)
//   snow             particles
//   summoning-circle foreground
//   dry-ice          foreground + sim
// They are COPIES, so refreshing them is a deliberate act — an effect edited in
// the app cannot silently change what this asserts.
//
// The check is resolution, not compilation: no GPU here, so this asserts that
// every rz*/RZ_* name an effect reaches for is DEFINED in each module its source
// is spliced into. That is exactly the failure class above, and it is the half a
// type checker can never see.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, readdirSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"
import { buildFieldShader, parseEffectAnchors } from "../dist/shaders/passes/composite.js"
import { buildParticleComputeShader, buildParticleRenderShader, parseParticleCount } from "../dist/shaders/passes/particles.js"
import { buildTrailShader } from "../dist/shaders/passes/trails.js"
import { buildAnchorTable } from "../dist/shaders/anchor-table.js"

const here = dirname(fileURLToPath(import.meta.url))
const load = (f) => readFileSync(join(here, "fixtures", f), "utf8")
const FIXTURES = readdirSync(join(here, "fixtures")).filter((f) => f.endsWith(".wgsl"))

/** Names the engine provides that an effect may call or read. Anything an
 *  effect references and no module defines is the bug this test exists for. */
function definedIn(code) {
  const fns = [...code.matchAll(/\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(/g)].map((m) => m[1])
  const consts = [...code.matchAll(/\b(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)/g)].map((m) => m[1])
  const structs = [...code.matchAll(/\bstruct\s+([A-Za-z_][A-Za-z0-9_]*)/g)].map((m) => m[1])
  return new Set([...fns, ...consts, ...structs])
}

/** Engine-surface names the effect reaches for: rz*/ /* prefixed, or RZ_ constants. */
function referenced(wgsl) {
  const body = wgsl.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "")
  return new Set([...body.matchAll(/\b(rz[A-Za-z0-9_]*|RZ_[A-Z_0-9]+)\b/g)].map((m) => m[1]))
}

const CAST = {
  subjects: 4,
  samples: 128,
  base: 12,
  trailBase: 108,
  slots: 8,
  reversedZ: false,
}

/** Every module this effect's source is spliced into, as the engine would. */
function modulesFor(wgsl) {
  const anchors = parseEffectAnchors(wgsl, 8)
  const alias = buildAnchorTable([anchors], 8).alias[0]
  const trailed = anchors.map((a, i) => (a.trail ? i : -1)).filter((i) => i >= 0)
  const cast = { ...CAST, alias, trailCount: trailed.length }
  const effect = {
    wgsl,
    paramsDecl: "",
    hasBackground: /\bfn\s+background\s*\(/.test(wgsl),
    hasForeground: /\bfn\s+foreground\s*\(/.test(wgsl),
    simSize: /\bfn\s+simStep\s*\(/.test(wgsl) ? 256 : 0,
  }
  const out = []
  if (effect.hasBackground || effect.hasForeground) {
    // FIELD only. The composite is static — "the user's code compiles in the
    // field module alone, and the composite only decides whether to sample it"
    // — so it is deliberately not in this list. Asserting against it claimed
    // Dry Ice was broken because composite has no rzSim*, which is correct and
    // irrelevant: composite never sees the effect's source.
    out.push(["field", buildFieldShader(effect)])
  }
  if (/\bfn\s+particleInit\s*\(/.test(wgsl)) {
    const src = { wgsl, count: parseParticleCount(wgsl) || 64, blend: "alpha", bloom: false }
    out.push(
      ["particle compute", buildParticleComputeShader(src, cast)],
      ["particle render", buildParticleRenderShader(src, cast)],
    )
  }
  if (/\bfn\s+trailWidth\s*\(/.test(wgsl)) {
    const src = { wgsl, slots: trailed.length, ribbonSlots: trailed, blend: "additive", bloom: false }
    out.push(["trail", buildTrailShader(src, cast)])
  }
  return out
}

for (const file of FIXTURES) {
  const wgsl = load(file)

  test(`${file}: compiles into at least one module`, () => {
    const mods = modulesFor(wgsl)
    assert.ok(mods.length > 0, "no mount detected — the fixture or the entry-point probes drifted")
  })

  test(`${file}: every engine name it uses is defined in every module it reaches`, () => {
    const wants = referenced(wgsl)
    for (const [name, code] of modulesFor(wgsl)) {
      const has = definedIn(code)
      const missing = [...wants].filter((n) => !has.has(n))
      assert.deepEqual(
        missing,
        [],
        `${file} in the ${name} module references ${missing.join(", ")} — an effect that installs today would stop`,
      )
    }
  })
}
