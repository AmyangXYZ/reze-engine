/**
 * The scene pass's attachment contract, in one place.
 *
 * Everything drawn INSIDE the scene pass — models, ground, outline hulls,
 * particles, ribbons, and the transparent depth prepass — shares one set of
 * attachments, and therefore one set of formats, blends and write masks. That
 * agreement used to be restated at every pipeline that joins the pass and in
 * every shader that writes to it, which is why adding an attachment was
 * dangerous rather than tedious: a class left behind does not fail loudly, it
 * fails as a validation error naming a pipeline that was never edited.
 *
 * So the contract is DATA here, and the pipelines ask for it by render class.
 * Adding the id attachment (MRT) becomes one edit in this file plus a per-class
 * decision about whether that class writes it — which is the shape of the
 * question, and now the shape of the code.
 *
 * WHAT IS NOT HERE. Two passes look like they belong and do not:
 *   - the FIELD pass has its own pair of rgba16float targets, its own blend and
 *     no MSAA. It is a different pass with a different contract.
 *   - the gizmo and selection-edge pipelines draw to the SWAPCHAIN in their own
 *     unmultisampled pass, so the presentation format is the whole of their
 *     contract. The plan listed gizmo as a scene render class; the code says
 *     otherwise, and the code is right.
 */

/** The scene pass's attachments, as the engine has them at init. Passed in
 *  rather than imported: hdr is chosen per device (rg11b10ufloat where it is
 *  available and blendable, rgba16float otherwise). */
export type SceneFormats = {
  /** The HDR colour attachment, @location(0). */
  hdr: GPUTextureFormat
  /** The aux attachment, @location(1) — (bloom mask, coverage). */
  aux: GPUTextureFormat
}

/**
 * The id attachment's format: (material index, object index), one u16 each.
 *
 * Two 16-bit channels rather than one 32-bit: 32-bit formats are not
 * multisamplable, and this attachment is multisampled with the rest of the
 * pass. Uint targets take no blend at all per spec, which is exactly right —
 * an averaged id is not an id.
 */
export const SCENE_ID_FORMAT: GPUTextureFormat = "rg16uint"

/**
 * Whether the scene pass carries the id attachment.
 *
 * Runtime rather than a compile-time constant, and mutable, because it is not
 * only a decision — it is a CAPABILITY. Multisampled rg16uint has to be probed
 * on the device (see the engine's init), and a device that cannot do it must
 * leave this off. That forces the shaders to be assembled after the probe,
 * which is why the two shader modules that gain an output stopped being
 * module-level constants: a string baked at import cannot know what the device
 * said.
 *
 * Set ONCE at init, before any pipeline or shader module is built. Nothing
 * reads it per frame.
 */
let mrtIds = false

/** Called by the engine at init, after probing the device. */
export function setMrtIds(on: boolean): void {
  mrtIds = on
}

export function mrtIdsEnabled(): boolean {
  return mrtIds
}

/**
 * What is being drawn, which is the only thing that varies.
 *
 * The classes differ ONLY in blend and write mask; formats are the pass's, not
 * the draw's. A class here is a thing with a reason, not a pipeline name — two
 * pipelines that blend the same way share one.
 */
type SceneRenderClass =
  /** Models, opaque and transparent. Straight alpha over. */
  | "material"
  /** The shadow-catcher floor. Blends PREMULTIPLIED, not like a material: its
   *  coverage is a lit surface plus a colourless shadow layer, so it weights
   *  its own colour before the blend sees it. */
  | "ground"
  /** Backface-expanded hulls. Also a material blend — it is geometry. */
  | "outline"
  /** Particles and ribbons in their default, non-additive mode. */
  | "particle"
  /** Particles declaring `#blend additive` — LIGHT rather than matter, so
   *  colour sums and alpha is left alone: a glow must not claim coverage it
   *  never occluded. The aux target sums with it, which is what lets an
   *  additive effect reach the bloom gate at all. */
  | "particle-additive"
  /** Ribbons. Additive colour like the above, but premultiplied by the
   *  fragment's own alpha on the way in (src-alpha, one) rather than added
   *  whole, and an ordinary alpha-over aux so a ribbon's mask does not
   *  saturate along every overlap. */
  | "trail"
  /** The transparent depth prepass: it exists to write DEPTH after the fabric's
   *  colour blended, so an outline drawn later is occluded behind it. It must
   *  therefore write no colour at all — the targets exist only to make the
   *  pipeline compatible with the pass it joins. */
  | "depth-prepass"

const ALPHA_OVER: GPUBlendState = {
  color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
}

/** Colour sums; alpha is not touched (src zero, dst one). */
const ADD_KEEP_ALPHA: GPUBlendState = {
  color: { srcFactor: "one", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
}

/** Both channels sum. rg8unorm clamps at 1, which is the saturation alpha-over
 *  would have reached anyway. */
const ADD_BOTH: GPUBlendState = {
  color: { srcFactor: "one", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
}

/**
 * OVER for something that arrives ALREADY premultiplied: take the source whole
 * and let it displace the destination by its own coverage.
 *
 * The ground needs this and nothing else does. Every other class writes a
 * straight colour and an alpha, and the src-alpha factor premultiplies it once
 * on the way in. The ground cannot: its coverage is the SUM of a lit surface
 * and a colourless shadow-catcher layer, so it has to weight its own colour by
 * the surface's share before it gets here. Handed to the src-alpha blend, that
 * weighting happened a second time.
 */
const PREMULTIPLIED_OVER: GPUBlendState = {
  color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
}

/** Additive, premultiplied by the fragment's alpha as it writes. */
const ADD_PREMULTIPLIED: GPUBlendState = {
  color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
}

/**
 * Which classes write a MEANINGFUL id.
 *
 * Not the same question as which shaders declare the output — every scene-pass
 * shader declares all three now (see sceneFsOutWgsl). This set decides only
 * whose id survives: writeMask 0xf here, writeMask 0 for everyone else, so the
 * others compute an id nothing stores. Leaving the target off those pipelines
 * is not available: every pipeline in a pass must agree with the pass's
 * attachments.
 *
 * The ground is in because a mark placed by id needs the floor to have one.
 * Transparent fabric writes ids too — dissolving a dress needs the dress's own
 * pixels, and last write wins there, which is a documented choice rather than a
 * consequence. Outline hulls, particles and ribbons are OUT: they are not
 * things you would ever address by id, and a hull would overwrite the id of the
 * body it traces.
 */
const WRITES_ID = new Set<SceneRenderClass>(["material", "ground"])

/** The blends each class writes its two attachments with. */
const BLENDS: Record<Exclude<SceneRenderClass, "depth-prepass">, [GPUBlendState, GPUBlendState]> = {
  material: [ALPHA_OVER, ALPHA_OVER],
  // PREMULTIPLIED colour, alone among the classes — see the blend's own note.
  // The aux is ordinary alpha-over: the ground writes its mask unweighted, like
  // everything else, and coverage is what the blend applies.
  ground: [PREMULTIPLIED_OVER, ALPHA_OVER],
  outline: [ALPHA_OVER, ALPHA_OVER],
  particle: [ALPHA_OVER, ALPHA_OVER],
  "particle-additive": [ADD_KEEP_ALPHA, ADD_BOTH],
  trail: [ADD_PREMULTIPLIED, ALPHA_OVER],
}

/**
 * The colour targets a scene-pass pipeline of this class declares, in
 * attachment order.
 *
 * Fresh objects every call, deliberately: a caller that mutated a shared
 * descriptor would change every pipeline built after it, and the ones built
 * before would keep the old value — a difference that only shows up as one
 * pipeline blending unlike its neighbours.
 */
export function sceneTargets(cls: SceneRenderClass, formats: SceneFormats): GPUColorTargetState[] {
  const targets: GPUColorTargetState[] =
    cls === "depth-prepass"
      ? // Format only, and writeMask 0 — the pipeline writes no colour. It still
        // DECLARES every output, which is the part that used to be missing; see
        // the note on sceneFsOutWgsl about why writeMask 0 is not a licence to
        // leave the output off.
        [
          { format: formats.hdr, writeMask: 0 },
          { format: formats.aux, writeMask: 0 },
        ]
      : (() => {
          const [color, aux] = BLENDS[cls]
          return [
            { format: formats.hdr, blend: { color: { ...color.color }, alpha: { ...color.alpha } } },
            { format: formats.aux, blend: { color: { ...aux.color }, alpha: { ...aux.alpha } } },
          ]
        })()
  if (mrtIds) targets.push({ format: SCENE_ID_FORMAT, writeMask: WRITES_ID.has(cls) ? 0xf : 0 })
  return targets
}

/**
 * The pass's colour attachments, in order — for the things that describe the
 * pass itself rather than a draw within it.
 *
 * A render bundle declares the formats it will be replayed into and is rejected
 * against a pass that does not match, so the bundle encoder has to move in
 * lockstep with the targets above. It restated the list independently until
 * this existed, which made it the one consumer an MRT change would have missed
 * — a bundle is recorded once and replayed, so the failure would have arrived
 * at replay, naming the bundle rather than the attachment that changed.
 */
export function sceneColorFormats(formats: SceneFormats): GPUTextureFormat[] {
  return mrtIds ? [formats.hdr, formats.aux, SCENE_ID_FORMAT] : [formats.hdr, formats.aux]
}

/**
 * The fragment-output struct a scene-pass shader returns.
 *
 * Emitted rather than written out per file so that the attachment list has one
 * author. The struct and the targets above have to agree on count and order,
 * and they now disagree in one file rather than in five.
 *
 * EVERY scene-pass shader takes this — outline, particles, ribbons and the
 * depth prepass included, none of which write a meaningful id (and the prepass
 * no meaningful colour either). They did not, once. The reasoning was that a
 * target at writeMask 0 needs no matching output: true of Dawn, and the reading
 * of gpuweb#1918 this file used to assert. It is not a reading every browser
 * shares, and the failure mode when a browser disagrees is the worst kind —
 * createRenderPipeline does not throw, it returns a pipeline that is already
 * invalid, and the pass that binds it is dropped whole. Missing geometry, clean
 * console, and a bug that reproduces on one vendor's browser only.
 *
 * So the rule is now the strict one, and it costs nothing to hold: declare
 * every output the pass has attachments for, and let writeMask decide what is
 * kept. A shader with no id to write pads it — see sceneIdPadWgsl.
 */
export function sceneFsOutWgsl(opts?: { name?: string; aux?: string }): string {
  const name = opts?.name ?? "FSOut"
  const aux = opts?.aux ?? "mask"
  // No @interpolate here, deliberately. The plan called for
  // `@location(2) @interpolate(flat)`, and that attribute is only legal on a
  // vertex OUTPUT or a fragment INPUT — a fragment output is neither, so it
  // would not compile. It is also unnecessary: the id is read from the per-draw
  // uniform, not carried across the triangle as a varying, so there is no
  // interpolation to suppress. (A varying carrying it WOULD need flat, since an
  // integer varying must be.)
  //
  // vec2u for rg16uint: the output type must be compatible with the format, and
  // uint targets take no blend, which is what makes last-write-wins the rule.
  const id = mrtIds ? `  @location(2) id: vec2u,\n` : ""
  return `struct ${name} {
  @location(0) color: vec4f,
  @location(1) ${aux}: vec4f,
${id}};
`
}

/**
 * The line a fragment shader assigns its id with, or nothing when ids are off.
 *
 * Emitted rather than written into each shader for the same reason as the
 * struct: with ids off there must be no assignment either, and a shader cannot
 * ask the device what it supports.
 */
export function sceneIdWriteWgsl(out: string, material: string, object: string): string {
  return mrtIds ? `  ${out}.id = vec2u(${material}, ${object});\n` : ""
}

/**
 * Just the id FIELD, for a shader that keeps its own struct.
 *
 * Particles and ribbons declare their outputs by hand because the comments on
 * their aux field are about their own blend and belong beside it — taking the
 * whole struct from sceneFsOutWgsl would move that prose away from what it
 * explains. They still need the id output when the pass has the attachment, so
 * they splice this in and pad it with sceneIdPadWgsl.
 */
export function sceneIdFieldWgsl(): string {
  return mrtIds ? `  @location(2) id: vec2u,\n` : ""
}

/**
 * The id assignment for a shader that has no id to give.
 *
 * The counterpart to the strict rule on sceneFsOutWgsl: outline hulls,
 * particles, ribbons and the depth prepass all declare the output because the
 * pass has the attachment, and all of them take writeMask 0, so what they
 * assign is never stored. Zero, which is the reserved "nothing" id the pass
 * clears to — so if one of these ever did reach the target, it would read as
 * absence rather than as a plausible wrong answer pointing at material 0.
 *
 * A separate function from sceneIdWriteWgsl rather than a call to it with "0u"
 * twice, because the two say different things: that one records an identity,
 * this one satisfies a declaration.
 */
export function sceneIdPadWgsl(out: string): string {
  return mrtIds ? `  ${out}.id = vec2u(0u, 0u);\n` : ""
}
