// The scene pass's attachment agreement, checked against the SOURCE. Run: npm test.
//
// Sibling of bindings.test.mjs, and it exists for the same reason: both halves
// of this contract are invisible to TypeScript. A pipeline whose target count
// disagrees with its shader's outputs compiles perfectly and fails at pipeline
// creation, and a shader writing @location(2) with only two targets declared is
// a WGSL file that is valid on its own.
//
// Two directions, because they fail differently and the spec treats them
// differently (see scene-contract.ts):
//   target with no matching output   -> legal at writeMask 0 (gpuweb#1918)
//   output with no matching target   -> ungoverned, disputed (gpuweb#5341)
// So the second is what this file refuses to let happen.
//
// Source rather than dist for the pipelines, because what is being checked is
// what the code SAYS to build; dist for the contract module, because what is
// being checked there is what it RETURNS.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"
import { sceneTargets, sceneColorFormats, mrtIdsEnabled, setMrtIds, SCENE_ID_FORMAT } from "../dist/shaders/passes/scene-contract.js"
import * as materials from "../dist/shaders/materials/common.js"
import * as ground from "../dist/shaders/passes/ground.js"
import * as outline from "../dist/shaders/passes/outline.js"
import * as particles from "../dist/shaders/passes/particles.js"
import * as trails from "../dist/shaders/passes/trails.js"

const here = dirname(fileURLToPath(import.meta.url))
const read = (p) => readFileSync(join(here, p), "utf8")
const engine = read("../src/engine.ts")
const groundSrc = read("../src/shaders/passes/ground.ts")

/** The formats the engine settles on at init. rgba16float is the fallback; the
 *  rg11b10ufloat path differs only in the colour format, and every assertion
 *  here is about count, order and blend rather than which float it is. */
const FORMATS = { hdr: "rgba16float", aux: "rg8unorm" }

const CLASSES = ["material", "ground", "outline", "particle", "particle-additive", "trail", "depth-prepass"]

/** Classes whose shaders write an id, and so declare a third fragment output. */
const WRITES_ID = ["material", "ground"]

/** How many attachments the pass carries right now. */
const ATTACHMENTS = () => (mrtIdsEnabled() ? 3 : 2)

test("every render class declares the pass's attachments, in order", () => {
  for (const cls of CLASSES) {
    const t = sceneTargets(cls, FORMATS)
    assert.equal(t.length, ATTACHMENTS(), `${cls} declares ${t.length} targets`)
    assert.equal(t[0].format, FORMATS.hdr, `${cls} location 0 must be the HDR colour attachment`)
    assert.equal(t[1].format, FORMATS.aux, `${cls} location 1 must be the aux attachment`)
  }
})

// The id attachment, exercised whatever the flag is set to right now: flipping
// it must not be the moment anyone finds out what it does. setMrtIds is
// restored in a finally, so the rest of the file sees the state it expected.
test("with ids on, every class carries the id target and only two write it", () => {
  const was = mrtIdsEnabled()
  try {
    setMrtIds(true)
    for (const cls of CLASSES) {
      const t = sceneTargets(cls, FORMATS)
      assert.equal(t.length, 3, `${cls} must carry the id target — every pipeline in a pass shares its attachments`)
      assert.equal(t[2].format, SCENE_ID_FORMAT)
      assert.equal(t[2].blend, undefined, "a uint target takes no blend; an averaged id is not an id")
      const writes = WRITES_ID.includes(cls)
      assert.equal(
        t[2].writeMask,
        writes ? 0xf : 0,
        `${cls} ${writes ? "writes ids and must not be masked off" : "writes no id and must be masked off — an " +
          "output with no target is the ungoverned direction, a target with no output is legal at writeMask 0"}`,
      )
    }
    assert.deepEqual(sceneColorFormats(FORMATS), [FORMATS.hdr, FORMATS.aux, SCENE_ID_FORMAT])

    // And the two that write it say so, as vec2u — the type the format needs.
    for (const [what, wgsl] of [
      ["ground", ground.groundShaderWgsl()],
      ["materials", materials.commonFsOutWgsl()],
    ]) {
      assert.match(wgsl, /@location\(2\) id: vec2u,/, `${what} must declare the id output when ids are on`)
      // @interpolate is legal only on a vertex output or a fragment input, so
      // it must never appear on this one — the plan called for it, and it would
      // not have compiled.
      assert.doesNotMatch(wgsl, /@interpolate\([a-z]+\)\s+id:/, `${what} must not put @interpolate on a fragment output`)
    }
    assert.match(ground.groundShaderWgsl(), /out\.id = vec2u\(/, "ground must actually assign its id")
  } finally {
    setMrtIds(was)
  }
})

test("with ids off, nothing carries an id target and no shader writes one", () => {
  const was = mrtIdsEnabled()
  try {
    setMrtIds(false)
    for (const cls of CLASSES) assert.equal(sceneTargets(cls, FORMATS).length, 2, `${cls} must not carry an id target`)
    assert.deepEqual(sceneColorFormats(FORMATS), [FORMATS.hdr, FORMATS.aux])
    // Scoped to the OUTPUT STRUCT, not the whole file. A bare /@location\(2\)/
    // also matches ground's vertex attribute `@location(2) uv` — vertex inputs
    // and fragment outputs number independently, and conflating them makes this
    // assertion fail on a shader that is entirely correct.
    assert.deepEqual(outputLocations(ground.groundShaderWgsl(), "FSOut"), [0, 1], "ground writes an output with no target")
    assert.deepEqual(outputLocations(materials.commonFsOutWgsl(), "FSOut"), [0, 1], "materials write an output with no target")
  } finally {
    setMrtIds(was)
  }
})

test("a class either blends or is explicitly write-masked off, never neither", () => {
  // A target with no blend and no writeMask is the silent case: it writes, and
  // it replaces. Nothing in this pass wants that, so saying nothing must not be
  // how a class ends up with it.
  for (const cls of CLASSES) {
    for (const [i, t] of sceneTargets(cls, FORMATS).entries()) {
      const decided = t.blend !== undefined || t.writeMask === 0
      assert.ok(decided, `${cls} target ${i} neither blends nor masks — it would replace the attachment`)
    }
  }
})

test("the depth prepass writes no colour at all", () => {
  // Its whole job is depth, after the fabric's colour blended. If it ever wrote
  // colour it would paint the fabric a second time, opaque.
  for (const t of sceneTargets("depth-prepass", FORMATS)) {
    assert.equal(t.writeMask, 0)
    assert.equal(t.blend, undefined, "a write-masked target must not also claim a blend")
  }
})

test("additive classes leave the alpha channel to the geometry", () => {
  // Light adds; it does not claim coverage it never occluded. Both additive
  // classes must therefore take nothing from src on the alpha channel.
  for (const cls of ["particle-additive", "trail"]) {
    const [color] = sceneTargets(cls, FORMATS)
    assert.equal(color.blend.alpha.srcFactor, "zero", `${cls} colour target must not add to alpha`)
    assert.equal(color.blend.alpha.dstFactor, "one", `${cls} colour target must keep the alpha it found`)
  }
})

/** Every `sceneTargetsFor("<class>", ...)` the engine asks for. */
function requestedClasses() {
  return [...engine.matchAll(/sceneTargetsFor\(\s*(?:[^,]*\?\s*)?"([a-z-]+)"(?:\s*:\s*"([a-z-]+)")?/g)]
    .flatMap((m) => [m[1], m[2]])
    .filter(Boolean)
}

test("every class the engine asks for is a class the contract defines", () => {
  const asked = [...new Set(requestedClasses())]
  assert.ok(asked.length > 0, "no sceneTargetsFor call found — the engine stopped asking, or this regex went blind")
  for (const cls of asked) {
    assert.ok(CLASSES.includes(cls), `engine.ts asks for "${cls}", which scene-contract does not define`)
  }
})

test("nothing names the aux format except the attachment and the contract", () => {
  // The aux format is what makes an attachment list a SCENE-pass list — the HDR
  // format alone is not, since the bloom pyramid renders single-target into it.
  // So naming it is the tell: outside the places below, a mention of it is a
  // pipeline, bundle or pass describing the scene's attachments for itself,
  // which is the duplication the module removed.
  //
  // Checked this way rather than by looking for inline target literals, because
  // the code this replaced did NOT inline them — it built `maskBlend` and
  // `layerMaskTarget` as consts first, and a guard against literals would have
  // watched it happen. Every one of those spellings names the format.
  const ALLOWED = [
    /private static readonly BLOOM_MASK_FORMAT/, // the declaration
    /return \{ hdr: this\.hdrFormat, aux: Engine\.BLOOM_MASK_FORMAT \}/, // the one feed into the contract
    /format: Engine\.BLOOM_MASK_FORMAT,\n\s*usage:/, // creating the attachment itself
  ]
  // Each line judged at ITS OWN offset. Looking the line up with indexOf finds
  // the first copy of its text, and `format: Engine.BLOOM_MASK_FORMAT,` appears
  // several times — every copy would then be judged by the first one's context,
  // so the whole set passes or fails together for the wrong reason.
  const stray = []
  let offset = 0
  for (const line of engine.split("\n")) {
    if (line.includes("BLOOM_MASK_FORMAT")) {
      const window = engine.slice(offset, offset + line.length + 80)
      if (!ALLOWED.some((re) => re.test(window))) stray.push(`${line.trim()}  (line ${engine.slice(0, offset).split("\n").length})`)
    }
    offset += line.length + 1
  }
  assert.deepEqual(
    stray,
    [],
    `these restate the scene pass's attachments instead of asking scene-contract:\n  ${stray.join("\n  ")}`,
  )
})

test("the bundle encoder is handed the same attachment list as the pipelines", () => {
  // A bundle declares the formats it will be replayed into and is rejected
  // against a pass that does not match. It is recorded once and replayed, so a
  // disagreement surfaces at replay naming the bundle, not the attachment that
  // moved. Same source, so it cannot drift.
  assert.match(
    engine,
    /colorFormats: sceneColorFormats\(this\.sceneFormats\)/,
    "recordBundles must take its formats from scene-contract",
  )
})

// ── The engine half: the attachment, the probe, and the ids themselves ──
//
// Source-level, because none of it can run without a device. What is being
// pinned is that the pieces move together — an attachment created but not
// attached, or ids written to a buffer nothing reads, are each silent.

test("the id attachment is created and attached under the same condition", () => {
  assert.match(engine, /if \(mrtIdsEnabled\(\)\) \{\s*\n\s*this\.idTexture = this\.device\.createTexture\(/, "the id texture must be created only when ids are on")
  assert.match(engine, /colorAttachments: idAttachment\s*\n?\s*\? \[colorAttachment, maskAttachment, idAttachment\]/, "the scene pass must attach the id texture when it exists")
})

test("the id attachment is multisampled and never resolved", () => {
  const at = engine.indexOf('label: "object id"')
  assert.ok(at > 0, "the id texture was renamed and this test went blind with it")
  const desc = engine.slice(at, engine.indexOf("})", at))
  assert.match(desc, /sampleCount: Engine\.MULTISAMPLE_COUNT/, "it shares the pass's sample count or it cannot join it")
  assert.match(desc, /format: SCENE_ID_FORMAT/)
  // The attachment must have no resolveTarget: resolving averages, and the
  // average of two ids names something that was never drawn.
  const attachAt = engine.indexOf("view: this.idView")
  // A fixed window, not up to the next "}" — clearValue is itself an object, so
  // brace-hunting stops inside it and never reaches loadOp/storeOp.
  const attach = engine.slice(attachAt, attachAt + 300)
  assert.doesNotMatch(attach, /resolveTarget/, "an averaged id is not an id — this attachment must not resolve")
  assert.match(attach, /loadOp: "clear"/, "id 0 is the reserved nothing; a stale id is worse than none")
  assert.match(attach, /storeOp: "store"/, "an attachment nothing can read afterwards is pure cost")
})

test("the probe pops its error scope exactly once, on both paths", () => {
  const at = engine.indexOf("private async probeMultisampledIds()")
  assert.ok(at > 0, "probeMultisampledIds not found")
  const body = engine.slice(at, engine.indexOf("\n  }", at))
  assert.equal((body.match(/pushErrorScope/g) ?? []).length, 1)
  assert.equal((body.match(/popErrorScope/g) ?? []).length, 1)
  // Outside the try. A scope left pushed swallows the next error anywhere in
  // the device, and that error would then be attributed to nothing.
  assert.match(body, /\}\n\s*\/\/[^\n]*\n(?:\s*\/\/[^\n]*\n)*\s*const err = await this\.device\.popErrorScope\(\)/)
})

test("the ids ride the material uniform's existing padding", () => {
  // Slots 13 and 14 of a 16-float block that was already this size. If these
  // moved to a buffer of their own, the indirect-draw path would need a new
  // binding and this comment would be the only warning.
  assert.match(engine, /data\[13\] = materialId/)
  assert.match(engine, /data\[14\] = objectId/)
  assert.match(engine, /const data = new Float32Array\(16\)/, "the block must not have grown")
})

test("a morphing material keeps its id", () => {
  // The morph path rebuilds the whole uniform block from `base` and writes it
  // back, so a base built without ids would blank a material's identity for as
  // long as it morphed — visible only to whatever reads ids, and only sometimes.
  assert.match(
    engine,
    /const base = this\.materialUniformData\(mat, sphereMode, headBoneIndex, materialId, modelId\)/,
    "the morph base must carry the ids too",
  )
})

test("the ground's ids collide with no model's", () => {
  // Both engine counters are 1-based, so the bottom of the range is taken.
  assert.match(groundSrc, /GROUND_MATERIAL_ID = 0xffff/)
  assert.match(groundSrc, /GROUND_OBJECT_ID = 0xffff/)
  // Minted ONCE, on the instance. The pick pass and the id attachment have to
  // name the same object by the same number, and this used to be derived twice.
  assert.match(engine, /objectId: this\.modelInstances\.size \+ 1/, "model ids are 1-based — 0 stays 'nothing'")
  assert.match(engine, /const modelId = inst\.objectId/, "the pick path must READ the id, not re-derive it")
})

test("the debug view refuses to draw when there is nothing to show", () => {
  // Turning it on with no id attachment would clear the screen to black, and
  // black is what "every id is zero" looks like — the one wrong answer this
  // instrument must never give, since it exists to be believed.
  const at = engine.indexOf("setIdDebug(on: boolean)")
  assert.ok(at > 0, "setIdDebug not found")
  const body = engine.slice(at, engine.indexOf("\n  }", at))
  assert.match(body, /if \(on && !this\.idView\) return false/, "it must refuse rather than draw a misleading frame")
})

test("the debug bind group is dropped when the attachment is recreated", () => {
  // It holds a view of a texture the resize destroys. Keeping it would sample
  // freed memory the first time the window changed size with the view open.
  const at = engine.indexOf("this.idTexture?.destroy()")
  const body = engine.slice(at, at + 500)
  assert.match(body, /this\.idDebugBindGroup = null/, "the stale bind group must be dropped alongside the texture")
})

/** Fragment-output locations a shader writes, from its output struct. */
function outputLocations(wgsl, structName) {
  const at = wgsl.indexOf(`struct ${structName}`)
  if (at < 0) return null
  const body = wgsl.slice(at, wgsl.indexOf("}", at))
  return [...body.matchAll(/@location\((\d+)\)/g)].map((m) => Number(m[1])).sort()
}

// The EMITTED shader, not the file: materials and ground get their output
// struct from scene-contract now, so it does not appear as text in either
// source. What the GPU is handed is the honest thing to check anyway.
const CAST = { subjects: 4, samples: 128, base: 12, trailBase: 108, slots: 8, reversedZ: false, alias: [0], trailCount: 1 }

const SHADERS = [
  ["materials (shared prelude)", () => materials.commonFsOutWgsl(), "FSOut", "material"],
  ["ground", () => ground.groundShaderWgsl(), "FSOut", "ground"],
  ["outline", () => outline.OUTLINE_SHADER_WGSL, "FSOut", "outline"],
  [
    "particles",
    () =>
      particles.buildParticleRenderShader(
        { wgsl: "fn particleInit(id: u32, seed: f32) -> Particle { var p: Particle; return p; }", count: 64, blend: "alpha", bloom: false },
        CAST,
      ),
    "FSOut",
    "particle",
  ],
  [
    "trails",
    () =>
      trails.buildTrailShader(
        { wgsl: "fn trailWidth(u: f32, age: f32) -> f32 { return 1.0; }", slots: 1, ribbonSlots: [0], blend: "additive", bloom: true },
        CAST,
      ),
    "TrailFSOut",
    "trail",
  ],
]

for (const [label, emit, structName, cls] of SHADERS) {
  test(`${label}: every @location it writes has a target`, () => {
    const locations = outputLocations(emit(), structName)
    assert.ok(locations, `struct ${structName} not found in the emitted ${label} shader`)
    const targets = sceneTargets(cls, FORMATS)
    for (const loc of locations) {
      assert.ok(
        loc < targets.length,
        `${label} writes @location(${loc}) but the ${cls} class declares ${targets.length} targets — ` +
          `a fragment output with no target is the direction the spec does NOT govern (gpuweb#5341)`,
      )
    }
    // Contiguous from 0: a gap means a location nothing fills, which is the
    // same hazard wearing a different hat.
    assert.deepEqual(locations, [...locations.keys()], `${label} output locations are not 0..n`)
  })
}

// ── The id CONSUMER: what the whole MRT phase was built for ──

import { buildFieldShader } from "../dist/shaders/passes/composite.js"

const fieldWith = (ids) =>
  buildFieldShader({
    wgsl: "fn foreground(r: vec3f, uv: vec2f, t: f32, d: f32) -> vec4f { return vec4f(0.0); }",
    paramsDecl: "",
    hasBackground: false,
    hasForeground: true,
    gridSize: 0,
    ids,
  })

test("an effect can read the id buffer, and compare it against a subject", () => {
  // Reading an id says nothing without something to compare it to, so both
  // halves have to exist: the pixel's id, and the id of a character.
  const src = fieldWith(true)
  assert.match(src, /fn rzObjectAt\(uv: vec2f\) -> u32/)
  assert.match(src, /fn rzMaterialAt\(uv: vec2f\) -> u32/)
  assert.match(src, /fn rzSubjectId\(i: i32\) -> u32/)
  // Multisampled and unresolved, read at sample 0 — the linearDepth rule. An
  // averaged id belongs to nothing.
  assert.match(src, /var _rzIdTex: texture_multisampled_2d<u32>/)
  // Sample 0 explicitly. The naive [^)]* could not span the nested clamp() the
  // load actually contains, so match the call and its sample index separately.
  assert.match(src, /textureLoad\(_rzIdTex,/)
  assert.match(src, /, 0\)\.y;/, "the object id is the y channel, read at sample 0")
  assert.match(src, /, 0\)\.x;/, "the material id is the x channel, read at sample 0")
})

test("with ids off the accessors still resolve, and answer nothing", () => {
  // An effect that masks by id must not fail to COMPILE on a device that could
  // not give us the attachment — it should simply mask nothing. The binding is
  // absent (no fallback texture exists to bind), so only the functions remain.
  const src = fieldWith(false)
  assert.match(src, /fn rzObjectAt\(uv: vec2f\) -> u32 \{ return 0u; \}/)
  assert.match(src, /fn rzMaterialAt\(uv: vec2f\) -> u32 \{ return 0u; \}/)
  assert.doesNotMatch(src, /_rzIdTex/, "no binding may be declared when the attachment does not exist")
})

test("the subject id and the pick id are the same number", () => {
  // The cast carries objectId in the centre vec4's w; the pick pass reads the
  // same field off the instance. Two derivations of one id are two that drift,
  // and then a mask selects a different model than a click does.
  assert.match(engine, /cd\[b \+ 7\] = inst\.objectId/, "the cast must carry the object id")
  assert.match(fieldWith(true), /return u32\(_rzCast\[i \* 3 \+ 1\]\.w\)/, "rzSubjectId must read that same slot")
})
