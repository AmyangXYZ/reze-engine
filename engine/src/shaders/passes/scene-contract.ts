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

/** Additive, premultiplied by the fragment's alpha as it writes. */
const ADD_PREMULTIPLIED: GPUBlendState = {
  color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
  alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
}

/** The blends each class writes its two attachments with. */
const BLENDS: Record<Exclude<SceneRenderClass, "depth-prepass">, [GPUBlendState, GPUBlendState]> = {
  material: [ALPHA_OVER, ALPHA_OVER],
  ground: [ALPHA_OVER, ALPHA_OVER],
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
  if (cls === "depth-prepass") {
    // Format only, and writeMask 0. Note the asymmetry this leans on, which is
    // the same one MRT will: a target the shader has no output for is legal at
    // writeMask 0 (gpuweb#1918), while an output with no target is not
    // governed. This direction is the specified one.
    return [{ format: formats.hdr, writeMask: 0 }, { format: formats.aux, writeMask: 0 }]
  }
  const [color, aux] = BLENDS[cls]
  return [
    { format: formats.hdr, blend: { color: { ...color.color }, alpha: { ...color.alpha } } },
    { format: formats.aux, blend: { color: { ...aux.color }, alpha: { ...aux.alpha } } },
  ]
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
  return `struct ${name} {
  @location(0) color: vec4f,
  @location(1) ${aux}: vec4f,
};
`
}
