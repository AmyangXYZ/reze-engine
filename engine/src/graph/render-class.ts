// Render-class + alpha-mode: the small, closed, engine-owned vocabularies that carry
// pass integration for a style group. A group's shader graph is pure shading; these select
// how the engine wires that shading into the passes. Not user-extensible — this is where
// rendering correctness lives (stencil, cull, draw order, alpha test). See
// docs/style-groups-spec.md §5.

/** Pass-integration class. Selects stencil/cull/draw-order behavior. */
export type RenderClass = "auto" | "eye" | "hair"

/** Alpha-handling axis, orthogonal to RenderClass. "hashed" = Wyman & McGuire object-space
 *  hashed alpha test (stockings); "opaque" = the standard near-zero threshold discard. */
export type AlphaMode = "opaque" | "hashed"

/** Descriptive manifest for hosts (reze-design) so the render-class picker is data-driven
 *  instead of hardcoding strings. The effect implementations stay engine-side; this only
 *  describes them. `pairsWith` = the effect needs a counterpart class present to show. */
export type RenderClassInfo = {
  id: RenderClass
  label: string
  description: string
  pairsWith?: RenderClass
}

export const RENDER_CLASSES: readonly RenderClassInfo[] = [
  { id: "auto", label: "Standard", description: "Opaque or transparent, derived from material alpha." },
  { id: "eye", label: "Eye", description: "Stamps the see-through stencil; visible through hair.", pairsWith: "hair" },
  {
    id: "hair",
    label: "Hair",
    description: "Reads the eye stencil so eyes show through the silhouette.",
    pairsWith: "eye",
  },
]
