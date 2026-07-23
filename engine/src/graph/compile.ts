// Graph → WGSL compiler: validate → prune → toposort → peephole → emit.
// Deterministic by construction (topo ties broken by node id, literals formatted by
// shortest round-trip) so the same graph JSON always yields byte-identical WGSL.
// See docs/graph-compiler-spec.md.

import { MAX_NODES, MAX_PARAMS } from "./schema"
import type { Diagnostic, ExposedParam, GraphNode, SocketValue, ShaderGraph } from "./schema"
import { NODE_REGISTRY, canConvert, convert, fmtValue, literalFits } from "./registry"
import type { NodeSpec, SockT } from "./registry"
import { assembleModule } from "./slots"
import type { AlphaMode, RenderClass } from "./render-class"

export type CompileOptions = {
  /** Fold exposed params to their defaults as consts (no StyleUniforms binding).
   *  Used by golden tests and by hosts that don't bind the style buffer. */
  inlineParams?: boolean
  /** Override the graph output with any node's socket — Blender's node-preview
   *  (Ctrl+Shift+Click viewer) workflow for the editor. */
  previewNode?: { node: string; socket: string }
  /** Pass-integration class the fs() shell is assembled for (stencil/cull/gate). The
   *  graph is pure shading; the group supplies this. Default "auto". */
  renderClass?: RenderClass
  /** Alpha-handling axis, orthogonal to renderClass. Default "opaque". */
  alphaMode?: AlphaMode
}

/** UBO slot for one exposed param: write `value` at style.p[vec4Index] (+ component). */
export type StyleSlot = {
  id: string
  kind: "float" | "color"
  vec4Index: number
  component?: "x" | "y" | "z" | "w"
  expr: string
}

export type CompileResult = {
  ok: boolean
  /** Full WGSL module (empty string when !ok). */
  wgsl: string
  /** Just the generated node section — snapshot tests and editor debugging. */
  fsBody: string
  slotMap: StyleSlot[]
  diagnostics: Diagnostic[]
  prunedNodes: string[]
}

const ID_RE = /^[a-z0-9_]+$/

const err = (message: string, nodeId?: string): Diagnostic => ({ severity: "error", nodeId, message })
const warn = (message: string, nodeId?: string): Diagnostic => ({ severity: "warning", nodeId, message })

// ─── Param slot assignment ──────────────────────────────────────────
// Floats pack 4-per-vec4 in declaration order; colors take .rgb of a fresh vec4.
// Depends only on the params array, never on graph topology — a topology edit
// doesn't move slider slots.

const COMPONENTS = ["x", "y", "z", "w"] as const

export function assignStyleSlots(params: ExposedParam[]): StyleSlot[] {
  const slots: StyleSlot[] = []
  let vec4Index = 0
  let comp = 0
  for (const p of params) {
    if (p.kind === "color") {
      if (comp > 0) {
        vec4Index++
        comp = 0
      }
      slots.push({ id: p.id, kind: "color", vec4Index, expr: `style.p[${vec4Index}].rgb` })
      vec4Index++
    } else {
      const c = COMPONENTS[comp]
      slots.push({ id: p.id, kind: "float", vec4Index, component: c, expr: `style.p[${vec4Index}].${c}` })
      comp++
      if (comp === 4) {
        vec4Index++
        comp = 0
      }
    }
  }
  return slots
}

// ─── Validation ─────────────────────────────────────────────────────

export function validateGraph(graph: ShaderGraph, opts: CompileOptions = {}): Diagnostic[] {
  const d: Diagnostic[] = []
  const nodes = new Map<string, GraphNode>()

  if (graph.version !== 1) d.push(err(`unsupported graph version: ${graph.version}`))
  if (graph.nodes.length > MAX_NODES) d.push(err(`graph has ${graph.nodes.length} nodes (max ${MAX_NODES})`))

  for (const node of graph.nodes) {
    if (!ID_RE.test(node.id)) d.push(err(`invalid node id "${node.id}" (want /^[a-z0-9_]+$/)`, node.id))
    if (nodes.has(node.id)) d.push(err(`duplicate node id "${node.id}"`, node.id))
    nodes.set(node.id, node)
    const spec = NODE_REGISTRY[node.type]
    if (!spec) {
      d.push(err(`unknown node type "${node.type}"`, node.id))
      continue
    }
    for (const [socket, value] of Object.entries(node.inputs ?? {})) {
      const input = spec.inputs[socket]
      if (!input) d.push(err(`node "${node.id}" (${node.type}) has no input socket "${socket}"`, node.id))
      else if (!literalFits(value, input.type))
        d.push(err(`literal for "${node.id}.${socket}" doesn't fit ${input.type} socket`, node.id))
    }
  }

  const linkedInputs = new Set<string>()
  for (const link of graph.links) {
    const fromNode = nodes.get(link.from.node)
    const toNode = nodes.get(link.to.node)
    if (!fromNode) {
      d.push(err(`link source node "${link.from.node}" doesn't exist`))
      continue
    }
    if (!toNode) {
      d.push(err(`link target node "${link.to.node}" doesn't exist`))
      continue
    }
    const fromSpec = NODE_REGISTRY[fromNode.type]
    const toSpec = NODE_REGISTRY[toNode.type]
    if (!fromSpec || !toSpec) continue // unknown type already reported
    const fromT = fromSpec.outputs[link.from.socket]
    if (!fromT) {
      d.push(err(`node "${link.from.node}" (${fromNode.type}) has no output socket "${link.from.socket}"`, link.from.node))
      continue
    }
    const toInput = toSpec.inputs[link.to.socket]
    if (!toInput) {
      d.push(err(`node "${link.to.node}" (${toNode.type}) has no input socket "${link.to.socket}"`, link.to.node))
      continue
    }
    if (toInput.type === "vec4") {
      d.push(err(`ramp stop colors must be literals — cannot link into "${link.to.node}.${link.to.socket}"`, link.to.node))
      continue
    }
    if (!canConvert(fromT, toInput.type))
      d.push(
        err(
          `type mismatch: "${link.from.node}.${link.from.socket}" (${fromT}) → "${link.to.node}.${link.to.socket}" (${toInput.type})`,
          link.to.node,
        ),
      )
    const key = `${link.to.node}.${link.to.socket}`
    if (linkedInputs.has(key)) d.push(err(`input "${key}" has more than one incoming link`, link.to.node))
    linkedInputs.add(key)
  }

  const params = graph.params ?? []
  if (params.length > MAX_PARAMS) d.push(err(`graph exposes ${params.length} params (max ${MAX_PARAMS})`))
  const paramIds = new Set<string>()
  const paramTargets = new Set<string>()
  for (const p of params) {
    if (paramIds.has(p.id)) d.push(err(`duplicate param id "${p.id}"`))
    paramIds.add(p.id)
    const node = nodes.get(p.target.node)
    if (!node) {
      d.push(err(`param "${p.id}" targets missing node "${p.target.node}"`))
      continue
    }
    const spec = NODE_REGISTRY[node.type]
    const input = spec?.inputs[p.target.socket]
    if (!input) {
      d.push(err(`param "${p.id}" targets missing socket "${p.target.node}.${p.target.socket}"`, p.target.node))
      continue
    }
    const key = `${p.target.node}.${p.target.socket}`
    if (linkedInputs.has(key)) d.push(err(`param "${p.id}" targets linked socket "${key}" — sliders only override literals`, p.target.node))
    if (paramTargets.has(key)) d.push(err(`params target socket "${key}" more than once`, p.target.node))
    paramTargets.add(key)
    const wantKind: SockT = p.kind === "color" ? "color" : "float"
    if (input.type !== wantKind)
      d.push(err(`param "${p.id}" is ${p.kind} but socket "${key}" is ${input.type}`, p.target.node))
    if (!literalFits(p.default, input.type)) d.push(err(`param "${p.id}" default doesn't fit ${input.type}`, p.target.node))
  }

  const out = opts.previewNode ?? graph.output
  const outNode = nodes.get(out.node)
  const outSpec = outNode ? NODE_REGISTRY[outNode.type] : undefined
  const outT = outSpec?.outputs[out.socket]
  if (!outT) d.push(err(`output "${out.node}.${out.socket}" doesn't resolve`, out.node))
  else if (outT === "vec4") d.push(err(`output "${out.node}.${out.socket}" must be color or float`, out.node))

  return d
}

// ─── Compile ────────────────────────────────────────────────────────

export function compileGraph(graph: ShaderGraph, opts: CompileOptions = {}): CompileResult {
  const fail = (diagnostics: Diagnostic[]): CompileResult => ({
    ok: false,
    wgsl: "",
    fsBody: "",
    slotMap: assignStyleSlots(graph.params ?? []),
    diagnostics,
    prunedNodes: [],
  })

  const diagnostics = validateGraph(graph, opts)
  if (diagnostics.some((x) => x.severity === "error")) return fail(diagnostics)

  const nodes = new Map(graph.nodes.map((node) => [node.id, node]))
  const spec = (node: GraphNode): NodeSpec => NODE_REGISTRY[node.type]

  // incoming[nodeId.socket] = producer; outgoing edges for toposort
  const incoming = new Map<string, { node: string; socket: string }>()
  const dependsOn = new Map<string, Set<string>>()
  for (const id of nodes.keys()) dependsOn.set(id, new Set())
  for (const link of graph.links) {
    incoming.set(`${link.to.node}.${link.to.socket}`, link.from)
    dependsOn.get(link.to.node)!.add(link.from.node)
  }

  // Prune: reverse-DFS from the (possibly preview-overridden) output.
  const out = opts.previewNode ?? graph.output
  const reachable = new Set<string>()
  const stack = [out.node]
  while (stack.length) {
    const id = stack.pop()!
    if (reachable.has(id)) continue
    reachable.add(id)
    for (const dep of dependsOn.get(id)!) stack.push(dep)
  }
  const prunedNodes = [...nodes.keys()].filter((id) => !reachable.has(id)).sort()

  // requiresLink applies to live nodes only.
  for (const id of reachable) {
    const node = nodes.get(id)!
    for (const [socket, input] of Object.entries(spec(node).inputs)) {
      if (input.requiresLink && !incoming.has(`${id}.${socket}`) && node.inputs?.[socket] === undefined)
        diagnostics.push(err(`input "${id}.${socket}" must be linked`, id))
    }
  }

  // Exposed params → style slots. Slot assignment is topology-independent; a param
  // whose target got pruned keeps its slot but has no effect (warning).
  const params = graph.params ?? []
  const slotMap = assignStyleSlots(params)
  const paramBySocket = new Map<string, { param: ExposedParam; slot: StyleSlot }>()
  params.forEach((param, i) => {
    if (!reachable.has(param.target.node))
      diagnostics.push(warn(`param "${param.id}" targets pruned node "${param.target.node}" — slider has no effect`))
    paramBySocket.set(`${param.target.node}.${param.target.socket}`, { param, slot: slotMap[i] })
  })
  if (diagnostics.some((x) => x.severity === "error")) return fail(diagnostics)

  // Toposort (Kahn) over reachable nodes; ready set kept sorted for determinism.
  const remaining = new Map<string, Set<string>>()
  for (const id of reachable) {
    const deps = new Set([...dependsOn.get(id)!].filter((dep) => reachable.has(dep)))
    remaining.set(id, deps)
  }
  const order: string[] = []
  const ready = [...remaining.entries()]
    .filter(([, deps]) => deps.size === 0)
    .map(([id]) => id)
    .sort()
  while (ready.length) {
    const id = ready.shift()!
    order.push(id)
    for (const [other, deps] of remaining) {
      if (deps.delete(id) && deps.size === 0) {
        ready.push(other)
        ready.sort()
      }
    }
    remaining.delete(id)
  }
  if (remaining.size) {
    diagnostics.push(err(`cycle through nodes: ${[...remaining.keys()].sort().join(" → ")}`))
    return fail(diagnostics)
  }

  // ── Emit ──
  const varName = (id: string) => `n_${id}`
  const usesStyle = { current: false }

  type Resolved = { expr: string; literal?: SocketValue }

  // Expression for a producer's output socket, converted to the consumer's type.
  const outputExpr = (from: { node: string; socket: string }, wantT: SockT): string => {
    const fromNode = nodes.get(from.node)!
    const fromSpec = spec(fromNode)
    const fromT = fromSpec.outputs[from.socket]
    const base = fromSpec.contextOutputs
      ? fromSpec.contextOutputs[from.socket]
      : varName(from.node) + (fromSpec.outputSelect?.[from.socket] ?? "")
    return convert(fromT, wantT, base)
  }

  const resolveInput = (node: GraphNode, socket: string, wantT: SockT): Resolved => {
    const key = `${node.id}.${socket}`
    const link = incoming.get(key)
    if (link) return { expr: outputExpr(link, wantT) }
    const exposed = paramBySocket.get(key)
    if (exposed && !opts.inlineParams) {
      usesStyle.current = true
      const raw = exposed.slot.expr
      return { expr: exposed.param.kind === "float" ? convert("float", wantT, raw) : convert("color", wantT, raw) }
    }
    const inputSpec = spec(node).inputs[socket]
    const value = exposed?.param.default ?? node.inputs?.[socket] ?? inputSpec.default
    if (value === undefined) {
      if (inputSpec.contextDefault) return { expr: inputSpec.contextDefault }
      throw new Error(`unresolved input ${key}`) // guarded by requiresLink validation
    }
    return { expr: fmtValue(value, wantT), literal: value }
  }

  // Peephole canonicalizations — fire only on plain literals (unexposed, unlinked),
  // reproducing the hand-written Safari specializations.
  const emitExpr = (node: GraphNode, args: Record<string, Resolved>): string => {
    const raw: Record<string, string> = {}
    for (const [k, v] of Object.entries(args)) raw[k] = v.expr
    if (node.type === "hue_sat" && args.hue.literal === 0.5)
      return `hue_sat_id(${raw.saturation}, ${raw.value}, ${raw.fac}, ${raw.color})`
    if (node.type.startsWith("mix/") && node.type !== "mix/add_emit") {
      if (args.fac.literal === 0) return raw.a
      if (node.type === "mix/blend" && args.fac.literal === 1) return raw.b
    }
    if (
      node.type === "tex_noise" &&
      args.detail.literal === 2 &&
      args.roughness.literal === 0.5 &&
      args.distortion.literal === 0
    )
      return `tex_noise_d2(${raw.vector}, ${raw.scale})`
    return spec(node).emit!(raw)
  }

  const lines: string[] = []
  for (const id of order) {
    const node = nodes.get(id)!
    const s = spec(node)
    if (s.contextOutputs) continue
    const args: Record<string, Resolved> = {}
    for (const [socket, input] of Object.entries(s.inputs)) args[socket] = resolveInput(node, socket, input.type)
    lines.push(`  let ${varName(id)} = ${emitExpr(node, args)}; // @node:${id}`)
  }

  lines.push(`  let final_color = ${outputExpr(out, "color")}; // @node:${out.node}`)

  const fsBody = lines.join("\n")
  const wgsl = assembleModule(opts.renderClass ?? "auto", opts.alphaMode ?? "opaque", fsBody, usesStyle.current)
  return { ok: true, wgsl, fsBody, slotMap, diagnostics, prunedNodes }
}
