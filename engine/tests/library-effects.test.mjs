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

/**
 * WGSL's reserved words — names the grammar refuses even though nothing uses
 * them. Verbatim from the spec's "Reserved Words" table.
 *
 * These are the second failure class this file exists for, and the one that
 * resolution checking cannot see: `let from = ...` references no engine name at
 * all, so the check above passes it, and the browser then reports a SYNTAX
 * error at the line that USES the variable — which reads as a problem with
 * whatever that line was doing. It has cost two debugging sessions: `meta` in
 * the cull shader, `from` in a note-scan helper.
 */
const RESERVED = new Set(`NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await
become binding_array cast catch class co_await co_return co_yield coherent column_major common compile
compile_fragment concept const_cast consteval constexpr constinit crate debugger decltype delete demote
demote_to_helper do dynamic_cast enum explicit export extends extern external fallthrough filter final finally
friend from fxgroup get goto groupshared highp impl implements import inline instanceof interface layout lowp
macro macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept noinline
nointerpolation non_coherent noncoherent noperspective null nullptr of operator package packoffset partition
pass patch pixelfragment precise precision premerge priv protected pub public readonly ref regardless register
reinterpret_cast require resource restrict self set shared sizeof smooth snorm static static_assert static_cast
std subroutine super target template this thread_local throw trait try type typedef typeid typename typeof
union unless unorm unsafe unsized use using varying virtual volatile wgsl where while writeonly yield`.split(/\s+/))

/** WGSL's own vocabulary: keywords, types, builtin functions, address spaces,
 *  access modes and the texture formats. Everything the language supplies, so
 *  that what is left over after removing it is genuinely undefined. */
const WGSL_VOCAB = new Set(`alias break case const const_assert continue continuing default diagnostic discard else
enable false fn for if let loop override requires return struct switch true var while bitcast
bool f16 f32 i32 u32 vec2 vec3 vec4 vec2i vec3i vec4i vec2u vec3u vec4u vec2f vec3f vec4f vec2h vec3h vec4h
mat2x2 mat2x3 mat2x4 mat3x2 mat3x3 mat3x4 mat4x2 mat4x3 mat4x4 mat2x2f mat2x3f mat2x4f mat3x2f mat3x3f mat3x4f
mat4x2f mat4x3f mat4x4f array atomic ptr sampler sampler_comparison
texture_1d texture_2d texture_2d_array texture_3d texture_cube texture_cube_array texture_multisampled_2d
texture_depth_multisampled_2d texture_external texture_storage_1d texture_storage_2d texture_storage_2d_array
texture_storage_3d texture_depth_2d texture_depth_2d_array texture_depth_cube texture_depth_cube_array
function private workgroup uniform storage handle read write read_write
all any select arrayLength abs acos acosh asin asinh atan atanh atan2 ceil clamp cos cosh countLeadingZeros
countOneBits countTrailingZeros cross degrees determinant distance dot dot4U8Packed dot4I8Packed exp exp2
extractBits faceForward firstLeadingBit firstTrailingBit floor fma fract frexp insertBits inverseSqrt ldexp
length log log2 max min mix modf normalize pow quantizeToF16 radians reflect refract reverseBits round saturate
sign sin sinh smoothstep sqrt step tan tanh transpose trunc
dpdx dpdxCoarse dpdxFine dpdy dpdyCoarse dpdyFine fwidth fwidthCoarse fwidthFine
textureDimensions textureGather textureGatherCompare textureLoad textureNumLayers textureNumLevels
textureNumSamples textureSample textureSampleBias textureSampleCompare textureSampleCompareLevel
textureSampleGrad textureSampleLevel textureSampleBaseClampToEdge textureStore
atomicLoad atomicStore atomicAdd atomicSub atomicMax atomicMin atomicAnd atomicOr atomicXor atomicExchange
atomicCompareExchangeWeak storageBarrier textureBarrier workgroupBarrier workgroupUniformLoad
pack4x8snorm pack4x8unorm pack4xI8 pack4xU8 pack4xI8Clamp pack4xU8Clamp pack2x16snorm pack2x16unorm pack2x16float
unpack4x8snorm unpack4x8unorm unpack4xI8 unpack4xU8 unpack2x16snorm unpack2x16unorm unpack2x16float
rgba8unorm rgba8snorm rgba8uint rgba8sint rgba16uint rgba16sint rgba16float r32uint r32sint r32float rg32uint
rg32sint rg32float rgba32uint rgba32sint rgba32float bgra8unorm rg11b10ufloat`.split(/\s+/))

/** Strip comments, attributes and member/swizzle accesses — everything that
 *  looks like an identifier without being a reference to one. */
function codeOnly(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, " ")
    .replace(/\/\/[^\n]*/g, " ")
    .replace(/@\w+\s*\([^)]*\)/g, " ") // @builtin(position), @group(0)
    .replace(/@\w+/g, " ") // @vertex, @const
    .replace(/\.\s*[A-Za-z_]\w*/g, " ") // .rgb, .x, member access
}

/** Every name a module DECLARES: functions and their parameters, locals,
 *  module constants, structs and their members. */
function declaredIn(src) {
  const code = codeOnly(src)
  const names = [
    ...code.matchAll(/\b(?:fn|struct|alias)\s+([A-Za-z_]\w*)/g),
    ...code.matchAll(/\b(?:let|var|const|override)(?:\s*<[^>]*>)?\s+([A-Za-z_]\w*)/g),
  ].map((m) => m[1])
  // Parameters and struct members: `name : type`, inside parens or braces.
  for (const m of code.matchAll(/[({,;]\s*([A-Za-z_]\w*)\s*:/g)) names.push(m[1])
  return new Set(names)
}

for (const file of FIXTURES) {
  const wgsl = load(file)

  test(`${file}: every name it uses resolves to something`, () => {
    // The third failure class, and the one neither check below can see: a
    // plain local that is used and never declared. It references no engine
    // name, so the resolution test passes it; it is not reserved, so the
    // keyword test passes it. It cost a round-trip when an edit to the line
    // glow dropped `let ly = lineY(...)` and left the use behind.
    //
    // Checked against the ASSEMBLED module, so engine bindings and helpers
    // count as declared and only genuinely dangling names survive.
    const used = new Set([...codeOnly(wgsl).matchAll(/\b([A-Za-z_]\w*)\b/g)].map((m) => m[1]))
    for (const [name, code] of modulesFor(wgsl)) {
      const known = declaredIn(code)
      const dangling = [...used].filter((n) => !known.has(n) && !WGSL_VOCAB.has(n) && !/^\d/.test(n))
      assert.deepEqual(dangling, [], `${file} in the ${name} module uses undeclared ${dangling.join(", ")}`)
    }
  })

  test(`${file}: declares nothing named after a WGSL reserved word`, () => {
    const body = wgsl.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/[^\n]*/g, "")
    const declared = [
      ...body.matchAll(/\b(?:const|let|var|fn|struct)\s+([A-Za-z_][A-Za-z0-9_]*)/g),
      // Function parameters, which are declarations the pattern above misses.
      ...body.matchAll(/[(,]\s*([A-Za-z_][A-Za-z0-9_]*)\s*:/g),
    ].map((m) => m[1])
    const bad = [...new Set(declared.filter((n) => RESERVED.has(n)))]
    assert.deepEqual(bad, [], `${file} declares ${bad.join(", ")} — reserved in WGSL, so the effect will not install`)
  })

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
