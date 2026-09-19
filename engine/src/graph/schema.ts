// ShaderGraph — a JSON-serializable graph of shading nodes, compiled to WGSL by compile.ts.
// Engine-space (LH, Y-up): porting a Blender graph converts coordinates at authoring
// time (Blender Normal Z → engine Y), the compiler never sees Blender conventions.
// See docs/graph-compiler-spec.md.

export type SocketValue = number | [number, number, number] | [number, number, number, number]

export type GraphNode = {
  /** Unique within the graph, /^[a-z0-9_]+$/ — becomes the WGSL variable suffix. */
  id: string
  /** Node registry key, e.g. "hue_sat", "math/power", "mix/blend". */
  type: string
  /** Literal defaults for unlinked input sockets. */
  inputs?: Record<string, SocketValue>
  /** Editor-only metadata (React Flow position). Ignored by the compiler, round-tripped
   *  by serialization so engine-side tooling never strips an editor's layout. */
  ui?: { position?: { x: number; y: number } }
}

export type GraphLink = {
  from: { node: string; socket: string }
  to: { node: string; socket: string }
}

/** Adjust-tier slider: overrides one unlinked literal input with a StyleUniforms read. */
export type ExposedParam = {
  /** Stable key — slot assignment and serialization identity. */
  id: string
  label: string
  target: { node: string; socket: string }
  kind: "float" | "color"
  min?: number
  max?: number
  default: SocketValue
}

export type ShaderGraph = {
  version: 1
  name: string
  nodes: GraphNode[]
  links: GraphLink[]
  /** Must resolve to a color (vec3f) or float (auto-splatted) socket. */
  output: { node: string; socket: string }
  /**
   * How opaque the fragment is, when the graph wants to say.
   *
   * Absent, alpha is MMD's own: the material's alpha times its texture's, one
   * number per material. That is right for a dress and wrong for anything whose
   * transparency depends on where you stand — glass and water are the same
   * surface at 5% looking straight down and near-mirror at a grazing angle, and
   * no constant is both. A game's own water shader writes
   * `(1 - NdotV) * fresnel` per pixel; this is the socket that lets a graph do
   * the same.
   *
   * It REPLACES the constant rather than scaling it, because a graph that
   * bothers to compute opacity is claiming the whole answer. A graph that still
   * wants the texture's holes multiplies in `texture.alpha` itself.
   *
   * A material wearing a group whose graph declares this always draws in the
   * transparent phase: the phase is otherwise decided at load from the texture's
   * alpha, which knows nothing about a curve the graph invents.
   */
  opacity?: { node: string; socket: string }
  params?: ExposedParam[]
  /** Soft, host-only hints for library filtering and smart default-group / render-class
   *  matching (e.g. ["hair"]). Ignored by the compiler; round-tripped. A graph is pure
   *  shading — pass integration lives on the style group's renderClass, not here. */
  tags?: string[]
}

export type Diagnostic = {
  severity: "error" | "warning"
  nodeId?: string
  message: string
}

export const MAX_NODES = 64
export const MAX_PARAMS = 16
