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
export type SceneRenderClass =
  /** Models, opaque and transparent. Straight alpha over. */
  | "material"
  /** The shadow-catcher floor. Blends exactly as a material does. */
  | "ground"
  /** Backface-expanded hulls. Also a material blend — it is geometry. */
  | "outline"
  /** Particles and ribbons in their default, non-additive mode. */
  | "particle"
  /** Particles declaring `// @blend additive` — LIGHT rather than matter, so
   *  colour sums and alpha is left alone: a glow must not claim coverage it
   *  never occluded. The aux target sums with it, which is what lets an
   *  additive effect reach the bloom gate at all. */
  | "particle-additive"
  /** Ribbons. Additive colour like the above, but premultiplied by the
   *  fragment's own alpha on the way in (src-alpha, one) rather than added
   *  whole, and an ordinary alpha-over aux so a ribbon's mask does not
   *  saturate along every overlap. */
  | "trail"
  /** The field layer blitted into the scene — a background before any geometry,
   *  a foreground after all of it. PREMULTIPLIED over, because the field targets
   *  have been premultiplied since N effects had to compose into one
   *  associatively; the material blend would scale rgb by alpha a second time
   *  and every layer would arrive dark. */
  | "field-blit"
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

/** Straight OVER for something already premultiplied: take the source whole,
 *  and let it displace the destination by its own coverage. */
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
 * Which classes actually WRITE an id, and so gain a fragment output for it.
 *
 * Everything else keeps its shader exactly as it is and takes the id target at
 * writeMask 0 — legal, specified, and free. The alternative (leaving the target
 * off those pipelines) is not available: every pipeline in a pass must agree
 * with the pass's attachments.
 *
 * The ground is in because a mark placed by id needs the floor to have one.
 * Transparent fabric writes ids too — dissolving a dress needs the dress's own
 * pixels, and last write wins there, which is a documented choice rather than a
 * consequence. Outline hulls, particles and ribbons are OUT: they are not
 * things you would ever address by id, and a hull would overwrite the id of the
 * body it traces.
 */
const WRITES_ID = new Set<SceneRenderClass>(["material", "ground"])
// NOT the field blit: a fullscreen layer has no object to name, and writing one
// would stamp the whole screen with a single id and destroy the buffer for
// everyone reading it.

/** The blends each class writes its two attachments with. */
const BLENDS: Record<Exclude<SceneRenderClass, "depth-prepass">, [GPUBlendState, GPUBlendState]> = {
  material: [ALPHA_OVER, ALPHA_OVER],
  ground: [ALPHA_OVER, ALPHA_OVER],
  outline: [ALPHA_OVER, ALPHA_OVER],
  particle: [ALPHA_OVER, ALPHA_OVER],
  "particle-additive": [ADD_KEEP_ALPHA, ADD_BOTH],
  trail: [ADD_PREMULTIPLIED, ALPHA_OVER],
  // Colour arrives premultiplied; the aux does not — the blit writes its mask
  // unweighted and this blend applies coverage, the convention the materials
  // already follow.
  "field-blit": [PREMULTIPLIED_OVER, ALPHA_OVER],
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
      ? // Format only, and writeMask 0. Note the asymmetry this leans on, which
        // is the same one the id target leans on below: a target the shader has
        // no output for is legal at writeMask 0 (gpuweb#1918), while an output
        // with no target is NOT governed (gpuweb#5341). This is the specified
        // direction, and it is the only one this file ever uses.
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
 * Only the shaders that will GAIN an output take this today — the materials
 * (hand-written and graph-generated alike, through COMMON_FS_OUT_WGSL) and the
 * ground. Outline, particles and ribbons keep their own declarations on
 * purpose: they never write the id, so they would take an `id: false` argument
 * forever, and their structs carry comments about their own blend that belong
 * where they are.
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
