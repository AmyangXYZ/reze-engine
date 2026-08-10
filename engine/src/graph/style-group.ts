// Style groups — the user-facing unit of material styling. A group binds a set of
// materials to a shader graph (pure shading) plus the engine-owned pass-integration axes
// (renderClass + alphaMode). Each material belongs to at most one group; grouped
// materials render via the group's compiled-graph pipeline, ungrouped ones fall back to
// the hand-written preset path. See docs/style-groups-spec.md.

import type { Diagnostic, ShaderGraph } from "./schema"
import type { AlphaMode, RenderClass } from "./render-class"
import type { StyleSlot } from "./compile"

export type StyleGroup = {
  /** Stable, unique within a model; /^[a-z0-9_-]+$/. The engine keys installs by this. */
  id: string
  /** Human display name ("Skirt", "Visor"); host-owned, round-tripped, never interpreted
   *  by the engine. Defaults to `id`. */
  label?: string
  /** Material names in this group; each material is in AT MOST one group. */
  materials: string[]
  /** Pure shading (ShaderGraph has no `slot` — integration lives here on the group). */
  graph: ShaderGraph
  /** Pass-integration class: stencil/cull/draw-order. Default "auto". */
  renderClass?: RenderClass
  /** Alpha-handling axis, orthogonal to renderClass. Default "opaque". */
  alphaMode?: AlphaMode
  /**
   * Extra image maps for this group's shading, up to four.
   *
   * A PMX material carries exactly ONE image, and a Blender-authored look is
   * usually built on more than that — a lightmap encoding shadow thresholds and
   * specular masks in its channels, a ramp, a detail map. Those belong to the
   * LOOK rather than to the model, so they ride on the group: the same graph
   * applied to another character brings its own maps with it.
   *
   * Read by the `tex_image/0`…`tex_image/3` nodes, in order. A slot left empty
   * (or a graph reading past the end) samples 1×1 white, so a missing map is a
   * no-op rather than a failure.
   */
  images?: (GroupImage | GroupImageSource | null)[]
}

export type GroupImageSource = ImageBitmap | HTMLImageElement | HTMLCanvasElement

/**
 * A group map, with the one thing the engine cannot infer from the pixels:
 * whether they are colour.
 *
 * A PMX texture is always colour, so material textures are uploaded sRGB
 * unconditionally. A group's maps are not — half of them are data whose channels
 * encode thresholds, masks and distances, and running those through an sRGB
 * decode bends every value. Blender asks the same question per image, and a
 * ported look carries whatever answer it was authored with.
 *
 * A bare source reads as data (`srgb: false`), because a map that is silently
 * mis-decoded as colour is the harder failure to see: it renders, just wrong.
 */
/**
 * `premultiplied` — whether the upload should multiply RGB by alpha.
 *
 * Blender premultiplies a straight-alpha image for rendering, so a transparent
 * pixel contributes nothing. A highlight texture is white everywhere with its
 * shape only in alpha, and without this it screens to a white card instead of
 * resolving to its shape. Left off for a channel-packed map, whose alpha is a
 * separate signal and would scale colour by something unrelated.
 *
 * It has to happen HERE rather than in the host's decode: copyExternalImageToTexture
 * converts to the destination's tagged alpha mode, so a premultiplied ImageBitmap
 * copied into an untagged texture is un-premultiplied again on the way in.
 */
export type GroupImage = { source: GroupImageSource; srgb?: boolean; premultiplied?: boolean }

export type GroupDiagnostic = { groupId: string; diagnostics: Diagnostic[]; ok: boolean }

export type ApplyStyleGroupsResult = {
  /** Every group compiled + swapped. */
  ok: boolean
  groups: GroupDiagnostic[]
  /** Names in a group that no model material matches (warning). */
  unknownMaterials: string[]
  /** Materials claimed by >1 group; the last group in array order wins (logged). */
  conflicts: string[]
}

export type ApplyStyleGroupResult = { ok: boolean; diagnostics: Diagnostic[]; slotMap: StyleSlot[] }
