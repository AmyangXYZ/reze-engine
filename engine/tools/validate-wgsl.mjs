#!/usr/bin/env node
// Compile every emitted shader on a REAL WebGPU device, headlessly.
//
//   node --import ./tests/register.mjs tools/validate-wgsl.mjs
//
// The npm-test suite reads emitted STRINGS — it can name-check and layout-check
// but it cannot compile WGSL, so a type error in a rewritten sampler sails
// through 341 green tests and fails in the user's browser. This tool is the
// missing half: it assembles the same modules the engine would, hands them to
// headless Chrome's WebGPU device, and prints real compiler diagnostics.
//
// Deliberately NOT part of `npm test`: it needs a Chrome binary and a GPU, and
// the suite stays hermetic. Run it after any shader-touching change; CI can
// gate on it where a GPU runner exists.

import { readFileSync, readdirSync, writeFileSync, mkdtempSync } from "node:fs"
import { execFileSync } from "node:child_process"
import { tmpdir } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const dist = join(here, "..", "dist")

const { COMMON_MATERIAL_PRELUDE_WGSL } = await import(`${dist}/shaders/materials/common.js`)
const { compileGraph } = await import(`${dist}/graph/compile.js`)
const { DEFAULT_GRAPH } = await import(`${dist}/graph/presets/default.js`)
const { groundShaderWgsl } = await import(`${dist}/shaders/passes/ground.js`)
const { SHADOW_DEPTH_SHADER_WGSL } = await import(`${dist}/shaders/passes/shadow.js`)
const { setMrtIds } = await import(`${dist}/shaders/passes/scene-contract.js`)
const {
  COMPOSITE_SHADER_WGSL,
  buildCompositeShader,
  buildFieldShader,
  parseEffectAnchors,
  EFFECT_SCENE_API,
} = await import(`${dist}/shaders/passes/composite.js`)
const { buildSimShader, gridEntryPoint, GRID_MAX } = await import(`${dist}/shaders/passes/grid.js`)
const { parseDirectives, stripDirectives } = await import(`${dist}/shaders/directives.js`)
const {
  buildParticleComputeShader,
  buildParticleRenderShader,
  particleEntryPoints,
} = await import(`${dist}/shaders/passes/particles.js`)
const { buildTrailShader, trailEntryPoints } = await import(`${dist}/shaders/passes/trails.js`)
const { buildLightEmitShader, hasLightEmit, MAX_LIGHTS } = await import(`${dist}/shaders/lights.js`)
const { buildAnchorTable, anchorAliasWgsl } = await import(`${dist}/shaders/anchor-table.js`)
const { REFLECTION_DEBUG_WGSL } = await import(`${dist}/reflection.js`)
const { OVERLAY_SHADER_WGSL } = await import(`${dist}/shaders/passes/overlay.js`)

/** What the effect declared, clamped to what the engine will build. Directives
 *  are one parse now: the per-tag parsers this used to call are gone. */
const declared = (wgsl) => parseDirectives(wgsl).directives

/** name → wgsl. Everything here must compile clean on a real device. */
const shaders = {}

// ── The scene pass's own shaders, in both id-attachment states ──
for (const ids of [false, true]) {
  setMrtIds(ids)
  const tag = ids ? "ids-on" : "ids-off"
  shaders[`ground (${tag})`] = groundShaderWgsl()
  const def = compileGraph(DEFAULT_GRAPH, { renderClass: "auto", alphaMode: "opaque" })
  if (!def.ok) throw new Error(`DEFAULT_GRAPH failed to compile: ${JSON.stringify(def.diagnostics)}`)
  shaders[`material default graph (${tag})`] = def.wgsl
}
setMrtIds(false)

shaders["shadow depth"] = SHADOW_DEPTH_SHADER_WGSL
shaders["reflection debug"] = REFLECTION_DEBUG_WGSL
shaders["composite base"] = COMPOSITE_SHADER_WGSL
shaders["editor overlay"] = OVERLAY_SHADER_WGSL

// A dummy fragment over the shared prelude, CALLING sampleShadow and the lights
// loop — so the prelude compiles as hand-written presets use it, not only as
// the graph generator wraps it.
shaders["material prelude + minimal fs"] =
  COMMON_MATERIAL_PRELUDE_WGSL +
  `
@fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
  let n = safe_normal(input.normal);
  let s = sampleShadow(input.worldPos, n);
  let lamps = rzLightsDiffuse(input.worldPos, n);
  return vec4f(vec3f(s) + lamps, 1.0);
}
`

// ── Every fixture effect, through every module its mounts reach ──
const fixtures = readdirSync(join(here, "..", "tests", "fixtures")).filter((f) => f.endsWith(".wgsl"))
for (const file of fixtures) {
  const source = readFileSync(join(here, "..", "tests", "fixtures", file), "utf8")
  // The directives come off before anything compiles: `#lights 3` is a line the
  // engine reads and WGSL cannot, so a module built from the raw file fails at
  // the first one with "invalid character found". Every builder below takes the
  // stripped source and every parser takes the raw text, which is the split the
  // engine itself makes.
  const wgsl = stripDirectives(source)
  const name = file.replace(/\.wgsl$/, "")
  const anchors = parseEffectAnchors(source, 8)
  const table = buildAnchorTable([anchors], 8)
  const alias = table.alias[0]
  const trailed = anchors.map((a, i) => (a.trail ? i : -1)).filter((i) => i >= 0)
  const cast = {
    subjects: 4,
    samples: 128,
    base: 12,
    trailBase: 12 + 8 * 4 * 3,
    slots: 8,
    trailCount: trailed.length,
    alias,
    reversedZ: false,
  }
  const hasBackground = /\bfn\s+background\s*\(/.test(wgsl)
  const hasForeground = /\bfn\s+foreground\s*\(/.test(wgsl)
  if (hasBackground || hasForeground) {
    shaders[`${name}: field`] = buildFieldShader({
      wgsl,
      paramsDecl: "",
      hasBackground,
      hasForeground,
      gridSize: Math.min(declared(source).grid, GRID_MAX),
      alias,
      trailCount: trailed.length,
    })
    shaders[`${name}: composite`] = buildCompositeShader({
      wgsl: "",
      paramsDecl: "",
      hasBackground,
      hasForeground,
      gridSize: 0,
      trailCount: 0,
    })
  }
  if (gridEntryPoint(wgsl)) {
    shaders[`${name}: grid step`] = buildSimShader(wgsl, Math.min(declared(source).grid, GRID_MAX) || 256, cast)
  }
  const pep = particleEntryPoints(wgsl)
  if (pep.init && pep.step && pep.shade) {
    const src = {
      wgsl,
      count: Math.min(declared(source).particles, 4096) || 64,
      blend: declared(source).particleBlend,
      bloom: declared(source).bloom,
    }
    shaders[`${name}: particle sim`] = buildParticleComputeShader(src, cast)
    shaders[`${name}: particle draw`] = buildParticleRenderShader(src, cast)
  }
  const tep = trailEntryPoints(wgsl)
  if (tep.width && tep.shade) {
    shaders[`${name}: trail`] = buildTrailShader(
      { wgsl, slots: trailed.length, ribbonSlots: trailed, blend: "additive", bloom: true },
      cast,
    )
  }
  if (hasLightEmit(wgsl) && Math.min(declared(source).lights, MAX_LIGHTS) > 0) {
    shaders[`${name}: light emit`] = buildLightEmitShader(wgsl, EFFECT_SCENE_API + anchorAliasWgsl(alias), cast)
  }
}

// ── Hand them to a real device ──
const work = mkdtempSync(join(tmpdir(), "wgsl-validate-"))
// Base64-inlined rather than fetched: file:// fetch is CORS-blocked in
// headless Chrome whatever the flags say, and escaping WGSL into a script tag
// is its own bug farm.
const payload = Buffer.from(JSON.stringify(shaders), "utf8").toString("base64")
writeFileSync(
  join(work, "validate.html"),
  `<!doctype html><script type="module">
const shaders = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob("${payload}"), (c) => c.charCodeAt(0))))
const adapter = await navigator.gpu?.requestAdapter()
if (!adapter) { console.log("WGSL-VALIDATE FATAL: no adapter"); }
else {
  const device = await adapter.requestDevice()
  window.onerror = (m) => console.log("WGSL-VALIDATE FATAL: " + m)
  for (const [name, code] of Object.entries(shaders)) {
    try {
      if (typeof code !== "string") { console.log("WGSL-VALIDATE ERROR: " + name + " builder returned " + typeof code); continue }
      const mod = device.createShaderModule({ code })
      const info = await mod.getCompilationInfo()
      const errors = info.messages.filter((m) => m.type === "error")
      if (errors.length === 0) console.log("WGSL-VALIDATE OK: " + name)
      for (const e of errors) console.log("WGSL-VALIDATE ERROR: " + name + " @" + e.lineNum + ":" + e.linePos + " " + e.message)
    } catch (e) {
      console.log("WGSL-VALIDATE ERROR: " + name + " threw: " + e.message)
    }
  }
  console.log("WGSL-VALIDATE DONE " + Object.keys(shaders).length)
}
window.close()
</script></html>`,
)

const chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
// No --virtual-time-budget: it fast-forwards timers and exits while REAL async
// GPU work (getCompilationInfo crosses to the GPU process) is still pending,
// truncating the run at a random shader. The page window.close()es itself when
// done; the exec timeout is the only backstop.
let out = ""
try {
  out = execFileSync(
    "bash",
    ["-c", `"${chrome}" --headless=new --enable-unsafe-webgpu --enable-logging=stderr --v=0 --no-sandbox "file://${join(work, "validate.html")}" 2>&1`],
    { encoding: "utf8", timeout: 90_000 },
  )
} catch (e) {
  out = (e.stdout ?? "") + (e.stderr ?? "")
}

const lines = out.split("\n").filter((l) => l.includes("WGSL-VALIDATE"))
const errors = lines.filter((l) => l.includes("WGSL-VALIDATE ERROR") || l.includes("FATAL"))
const oks = lines.filter((l) => l.includes("WGSL-VALIDATE OK"))
const done = lines.some((l) => l.includes("WGSL-VALIDATE DONE"))

for (const l of errors) console.error(l.slice(l.indexOf("WGSL-VALIDATE")))
console.log(`${oks.length}/${Object.keys(shaders).length} shaders compile clean on a real device${done ? "" : "  (WARNING: run did not finish)"}`)
process.exit(errors.length || !done ? 1 : 0)
