# Material Graph → WGSL Compiler — Spec v0.1

Engine-side foundation for Reze Design's style system. A material style is a JSON node
graph; the compiler validates it, topologically orders it, and emits a WGSL fragment
shader that concatenates with the existing shared blocks (`NODES_WGSL` +
`COMMON_MATERIAL_PRELUDE_WGSL`) exactly like today's hand-written presets. The
existing presets are the compiler's ground truth: the hair graph in this document must
render pixel-identical to `engine/src/shaders/materials/hair.ts`.

## 1. Goals / non-goals

**Goals**

- Graph JSON schema + validation (headless, no UI deps — the React Flow editor in
  reze.design serializes to/from this format).
- Deterministic compile: same graph → byte-identical WGSL (cacheable, diffable,
  testable).
- Two-tier edits: *exposed* params live in a uniform buffer (instant slider writes, no
  recompile); everything else folds to WGSL `const` (full constant-folding for the
  common case). Topology changes trigger an async recompile (~100 ms budget).
- Async pipeline lifecycle with fallback-on-error: a bad graph never blanks the frame —
  the previous pipeline keeps rendering and the compile error surfaces as diagnostics.
- Preset serialization: preset = graph + default param values + slider metadata.

**Non-goals (v0.1)**

- No new WGSL node functions. The node registry is exactly what `NODES_WGSL` already
  exports. New nodes are added by hand to `nodes.ts` first, then registered.
- No vertex-stage graphs. `COMMON_VS_WGSL` (skinning) is fixed.
- No graph control over pipeline state (blend, stencil, depth, the hair
  `IS_OVER_EYES` over-eyes variant, eye stencil stamp). Those stay per-preset-slot
  template decisions in `engine.ts` (§7).
- No Blender `.blend`/JSON import. The schema is **engine-space** (LH, Y-up). Porting
  a Blender graph is an authoring-time conversion (Normal Z→Y etc., see
  coord-systems notes), not a compiler concern.

## 2. Graph JSON schema

```ts
export type StyleGraph = {
  version: 1                      // schema version, bump on breaking change
  name: string                    // preset display name
  slot: MaterialPreset            // which preset slot this style targets ("hair", "face", …)
  nodes: GraphNode[]
  links: GraphLink[]
  output: { node: string; socket: string }   // must resolve to a color (vec3f) value
  params?: ExposedParam[]         // adjust-tier sliders
}

export type GraphNode = {
  id: string                      // unique within graph, /^[a-z0-9_]+$/
  type: NodeType                  // registry key, §3
  inputs?: Record<string, SocketValue>  // unlinked input sockets → literal defaults
}

export type SocketValue = number | [number, number, number] | [number, number, number, number]

export type GraphLink = {
  from: { node: string; socket: string }
  to:   { node: string; socket: string }
}

export type ExposedParam = {
  id: string                      // stable key for serialization + UBO slot assignment
  label: string                   // slider display name
  target: { node: string; socket: string }   // which literal input it overrides
  kind: "float" | "color"
  min?: number; max?: number      // float sliders
  default: SocketValue
}
```

Rules enforced by `validateGraph()`:

- Node ids unique; every link endpoint resolves to an existing node + socket with
  compatible (or implicitly convertible, §4) types.
- Graph is a DAG (cycle → error naming the cycle path).
- `output` resolves to `color`/`vec3f` (or `float`, auto-splatted).
- Each input socket has ≤ 1 incoming link; unlinked sockets take the node's `inputs`
  literal, else the registry default.
- ≤ 64 nodes, ≤ 16 exposed params (UBO budget, §5).
- `ExposedParam.target` must point at an *unlinked* input socket (you can't slider a
  computed value).

## 3. Node registry

One entry per Blender-equivalent node already in `NODES_WGSL`. Registry entry shape:

```ts
type NodeSpec = {
  fn: string                        // WGSL function name in NODES_WGSL
  inputs: Record<string, SockT>     // ordered — must match WGSL param order
  output: SockT
  argOrder: string[]                // socket → WGSL arg position
}
type SockT = "float" | "color" | "vector"
```

| NodeType | WGSL fn | Inputs (socket: type) | Out |
|---|---|---|---|
| `hue_sat` | `hue_sat` | hue, saturation, value, fac: float; color | color |
| `bright_contrast` | `bright_contrast` | color; bright, contrast: float | color |
| `invert` | `invert` / `invert_f` | fac: float; color | color |
| `ramp_constant` | `ramp_constant` | fac: float; pos0, pos1: float; color0, color1: vec4 | color |
| `ramp_constant_aa` | `ramp_constant_edge_aa` | fac, edge: float; color0, color1: vec4 | color |
| `ramp_linear` | `ramp_linear` | same as ramp_constant | color |
| `ramp_cardinal` | `ramp_cardinal` | same | color |
| `math` (op: add/multiply/power/greater_than) | `math_*` | a, b: float | float |
| `mix` (blend/overlay/multiply/lighten/linear_light) | `mix_*` | fac: float; a, b: color | color |
| `fresnel` | `fresnel` | ior: float | float |
| `layer_weight` (fresnel/facing) | `layer_weight_*` | blend: float | float |
| `shader_to_rgb_diffuse` | `shader_to_rgb_diffuse` | — (context-fed) | float |
| `bump` | `bump_lh` | strength, height: float | vector |
| `normal_map` | `normal_map` | strength: float; color | vector |
| `tex_noise` | `tex_noise` / `tex_noise_d2` | vector; scale, detail, roughness, distortion: float | float |
| `tex_gradient` | `tex_gradient_linear` | vector | float |
| `tex_voronoi` (f1/color) | `tex_voronoi_*` | vector; scale: float | float / color |
| `mapping` | `mapping_point` | vector; loc, rot, scl: vector | vector |
| `separate_xyz` | (inline swizzle) | vector | float ×3 (x/y/z sockets) |
| `vect_cross` | `vect_math_cross` | a, b: vector | vector |
| `principled` | `eval_principled` | base: color; metallic, specular, roughness, spec_clamp, sheen, sheen_tint: float | color |
| `mix_shader` | (inline `mix`) | fac: float; a, b: color | color |
| `value` / `rgb` | (literal) | — | float / color |

**Context inputs** — nodes whose WGSL signatures take `n, l, v, sun, amb, shadow,
worldPos, restPos, uv, tex_color, tex_alpha`: the compiler injects these from the
template's local variables (§6); they are *not* graph sockets. A single `geometry`
node exposes what graphs may tap directly: `normal` (post-`safe_normal`), `view`,
`rest_pos`, `world_pos`, `uv`. `texture` node exposes `color`/`alpha` of the bound
PMX diffuse texture (samplers/bindings are fixed by the shared layout — graphs never
declare bindings).

**Enum-parameterized nodes** (`math.op`, `mix.blend_type`, `ramp` interpolation,
`layer_weight.mode`): the enum is part of the node `type` string
(`"math/power"`, `"mix/blend"`) so it is unambiguously topology, never a slider.

## 4. Implicit socket conversions (Blender-faithful)

| From → To | Emitted WGSL |
|---|---|
| color → float | `color_to_value(x)` (BT.601 — matches Blender's node.cc grayscale) |
| float → color | `vec3f(x)` |
| vec4 (ramp out) → color | `.rgb` |
| float → vec4 stop color | `vec4f(vec3f(x), 1.0)` |
| vector ↔ color | bit-identical pass-through (`vec3f`) |

Ramps expose three output sockets: `color` (`.rgb`), `alpha` (`.a`), and `fac_out`
(`.r`) — the last matching how a grayscale ramp feeds a scalar consumer in the hand
ports (`ramp_008.r` in hair). Emitting `fac_out` as `.r` avoids routing it through
the BT.601 color→float conversion, which would change the value.

## 5. Two-tier params: consts vs `StyleUniforms`

- **Unexposed literal** → emitted as a WGSL `const` (or inlined literal argument).
  Full constant folding; zero runtime cost vs today's hand-written shaders.
- **Exposed param** → compiler assigns a slot in a new per-material uniform buffer:

```wgsl
// group(2) binding(4) — appended to the shared material bind group layout.
// Fixed 16-vec4f block (256 B) so ONE layout still serves every material pipeline;
// non-graph presets bind a shared zero buffer.
struct StyleUniforms { p: array<vec4f, 16> };
@group(2) @binding(4) var<uniform> style: StyleUniforms;
```

  Slot packing: floats pack 4-per-vec4 (`style.p[2].y`), colors take `.rgb` of one
  vec4. Slot map `{paramId → offset}` is returned by the compile so the engine's
  `setStyleParam(model, materialName, paramId, value)` is a plain
  `queue.writeBuffer` — the instant tier, no pipeline touch.
- Changing a param's *exposed/unexposed* status or any link/node → topology change →
  recompile (async tier).

## 6. Compilation algorithm

```
compileGraph(graph) → { wgsl: string, slotMap: Map<paramId, offset>, diagnostics }
```

1. **Validate** (§2 rules + registry socket typecheck with §4 conversions).
2. **Prune** nodes not reachable (reverse-DFS) from `output` — authoring deletions
   don't cost fragment work. Log pruned ids in diagnostics.
3. **Toposort** (Kahn, deterministic: ties broken by node id) → emission order.
4. **Emit** one `let n_<id> = <fn>(<args>);` per node, SSA-style, into the fs()
   template below. Multi-consumer outputs are naturally shared (one `let`, N uses) —
   this reproduces the hand-written sharing (e.g. hair's `hue_sat_002` used twice).
5. **Peephole canonicalizations** — required to hit today's optimized shaders:
   - `hue_sat` with literal `hue == 0.5` → `hue_sat_id(sat, val, fac, color)`
     (Safari perf: skips HSV roundtrip). Only when hue is *unexposed*.
   - `tex_noise` with literal `detail == 2, roughness == 0.5, distortion == 0` →
     `tex_noise_d2(p, scale)` (Safari can't unroll the runtime-octave loop).
   - `mix/blend` with literal fac 0 or 1 → operand passthrough.
   These fire only on unexposed (const) values — exposing the param disables the
   specialization for that node, accepted cost.
6. **Assemble module**:

```
NODES_WGSL + COMMON_MATERIAL_PRELUDE_WGSL + styleUniformsBlock + slotTemplate(fsBody)
```

The **slot template** is the per-`MaterialPreset`-slot boilerplate today's hand
shaders repeat — the graph only computes `final_color: vec3f`:

```wgsl
@fragment fn fs(input: VertexOutput) -> FSOut {
  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  let alpha = material.alpha * tex_s.a;        // MMD alpha semantics
  if (alpha < 0.001) { discard; }
  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);
  let tex_color = tex_s.rgb;

  // ── generated node lets ──
  ${emittedNodes}
  let final_color = ${outputExpr};
  // ────────────────────────

  ${slotEpilogue}   // per-slot: hair adds IS_OVER_EYES alpha, writes FSOut + mask
}
```

Slot epilogues (hair over-eyes `override`, face/eye stencil interactions, stockings
alpha-hash dither) are engine-owned strings keyed by `graph.slot` — graphs restyle a
slot's shading, not its pass integration.

## 7. Async pipeline lifecycle

On topology change:

1. `device.pushErrorScope("validation")` → `createShaderModule` →
   `getCompilationInfo()` — WGSL errors mapped back to node ids via `// @node:<id>`
   line markers emitted with each `let`.
2. `createRenderPipelineAsync` (never the sync variant — no jank on the render loop).
3. Success → atomically swap the slot's pipeline for that model+material; destroy the
   old one *next frame* (it may be referenced by the in-flight encoder).
4. Failure → keep previous pipeline rendering, return diagnostics
   `{ nodeId?, message }[]` to the caller (reze.design shows them on the node).
5. Generation counter per material-slot: a compile that finishes after a newer edit's
   compile is discarded (stale-write guard).

## 8. Worked example — hair (the ground-truth graph)

`M_Hair` (仿深空之眼渲染预设v1.0) as a `StyleGraph`. Node ids mirror the Blender
names in `hair.ts` comments. This graph, compiled, must pixel-match
`HAIR_SHADER_WGSL` — it is golden-test #1.

```jsonc
{
  "version": 1, "name": "Deep Space Hair", "slot": "hair",
  "nodes": [
    { "id": "tex",           "type": "texture" },
    { "id": "geo",           "type": "geometry" },
    { "id": "hs_shadow",     "type": "hue_sat",  "inputs": { "hue": 0.5, "saturation": 1.2, "value": 0.5, "fac": 1.0 } },
    { "id": "hs_002",        "type": "hue_sat",  "inputs": { "hue": 0.48, "saturation": 1.2, "value": 0.7, "fac": 1.0 } },
    { "id": "hs_001",        "type": "hue_sat",  "inputs": { "hue": 0.5, "saturation": 1.5, "value": 1.0, "fac": 1.0 } },
    { "id": "str",           "type": "shader_to_rgb_diffuse" },
    { "id": "ramp_008",      "type": "ramp_constant", "inputs": { "pos0": 0.0, "color0": [0,0,0,1], "pos1": 0.2966, "color1": [1,1,1,1] } },
    { "id": "mix_004",       "type": "mix/blend" },
    { "id": "bc",            "type": "bright_contrast", "inputs": { "bright": 0.1, "contrast": 0.2 } },
    { "id": "sep_n",         "type": "separate_xyz" },          // Blender Z ⇒ engine Y (authoring-time conversion)
    { "id": "bevel_clamp",   "type": "math/clamp01" },          // §9: needs registry add, see note
    { "id": "mix_003",       "type": "mix/blend" },
    { "id": "fres",          "type": "fresnel",      "inputs": { "ior": 1.45 } },
    { "id": "lw",            "type": "layer_weight/fresnel", "inputs": { "blend": 0.61 } },
    { "id": "rim_mul",       "type": "math/multiply" },
    { "id": "rim_pow",       "type": "math/power",   "inputs": { "b": 0.6300000548362732 } },
    { "id": "mix_shader_002","type": "mix_shader",   "inputs": { "b": [0.1673291176557541, 0.1673291176557541, 0.1673291176557541] } },
    { "id": "gate",          "type": "math/greater_than", "inputs": { "b": 0.15000000596046448 } },
    { "id": "gate_scale",    "type": "math/multiply", "inputs": { "b": 0.1 } },
    { "id": "npr_add",       "type": "mix/add_emit" },          // §9 note
    { "id": "principled",    "type": "principled",   "inputs": { "metallic": 0, "specular": 1.0, "roughness": 0.3, "spec_clamp": 10.0, "sheen": 0, "sheen_tint": 0 } },
    { "id": "mix_shader_001","type": "mix_shader",   "inputs": { "fac": 0.2 } }
  ],
  "links": [
    { "from": {"node":"tex","socket":"color"},        "to": {"node":"hs_shadow","socket":"color"} },
    { "from": {"node":"hs_shadow","socket":"color"},  "to": {"node":"hs_002","socket":"color"} },
    { "from": {"node":"tex","socket":"color"},        "to": {"node":"hs_001","socket":"color"} },
    { "from": {"node":"str","socket":"value"},        "to": {"node":"ramp_008","socket":"fac"} },
    { "from": {"node":"ramp_008","socket":"fac_out"}, "to": {"node":"mix_004","socket":"fac"} },
    { "from": {"node":"hs_002","socket":"color"},     "to": {"node":"mix_004","socket":"a"} },
    { "from": {"node":"hs_001","socket":"color"},     "to": {"node":"mix_004","socket":"b"} },
    { "from": {"node":"mix_004","socket":"color"},    "to": {"node":"bc","socket":"color"} },
    { "from": {"node":"geo","socket":"normal"},       "to": {"node":"sep_n","socket":"vector"} },
    { "from": {"node":"sep_n","socket":"y"},          "to": {"node":"bevel_clamp","socket":"a"} },
    { "from": {"node":"bevel_clamp","socket":"value"},"to": {"node":"mix_003","socket":"fac"} },
    { "from": {"node":"bc","socket":"color"},         "to": {"node":"mix_003","socket":"a"} },
    { "from": {"node":"hs_002","socket":"color"},     "to": {"node":"mix_003","socket":"b"} },
    { "from": {"node":"fres","socket":"value"},       "to": {"node":"rim_mul","socket":"a"} },
    { "from": {"node":"lw","socket":"value"},         "to": {"node":"rim_mul","socket":"b"} },
    { "from": {"node":"rim_mul","socket":"value"},    "to": {"node":"rim_pow","socket":"a"} },
    { "from": {"node":"mix_003","socket":"color"},    "to": {"node":"mix_shader_002","socket":"a"} },
    { "from": {"node":"rim_pow","socket":"value"},    "to": {"node":"mix_shader_002","socket":"fac"} },
    { "from": {"node":"tex","socket":"color"},        "to": {"node":"gate","socket":"a"} },   // color→float ⇒ color_to_value()
    { "from": {"node":"gate","socket":"value"},       "to": {"node":"gate_scale","socket":"a"} },
    { "from": {"node":"mix_shader_002","socket":"color"}, "to": {"node":"npr_add","socket":"a"} },
    { "from": {"node":"gate_scale","socket":"value"}, "to": {"node":"npr_add","socket":"b"} },
    { "from": {"node":"bc","socket":"color"},         "to": {"node":"principled","socket":"base"} },
    { "from": {"node":"npr_add","socket":"color"},    "to": {"node":"mix_shader_001","socket":"a"} },
    { "from": {"node":"principled","socket":"color"}, "to": {"node":"mix_shader_001","socket":"b"} }
  ],
  "output": { "node": "mix_shader_001", "socket": "color" },
  "params": [
    { "id": "npr_mix",  "label": "Realism",     "target": {"node":"mix_shader_001","socket":"fac"}, "kind": "float", "min": 0, "max": 1,   "default": 0.2 },
    { "id": "rim",      "label": "Rim Power",   "target": {"node":"rim_pow","socket":"b"},          "kind": "float", "min": 0.2, "max": 2, "default": 0.63 },
    { "id": "shad_pos", "label": "Shadow Edge", "target": {"node":"ramp_008","socket":"pos1"},      "kind": "float", "min": 0.05, "max": 0.6, "default": 0.2966 },
    { "id": "rough",    "label": "Gloss",       "target": {"node":"principled","socket":"roughness"},"kind": "float", "min": 0.05, "max": 0.8, "default": 0.3 }
  ]
}
```

Expected emission (body excerpt; `hs_shadow`/`hs_001` hit the `hue_sat_id` peephole
because their hue is a literal 0.5; `hs_002` does not):

```wgsl
let n_hs_shadow = hue_sat_id(1.2, 0.5, 1.0, tex_color);
let n_hs_002 = hue_sat(0.48, 1.2, 0.7, 1.0, n_hs_shadow);
let n_hs_001 = hue_sat_id(1.5, 1.0, 1.0, tex_color);
let n_str = shader_to_rgb_diffuse(n, l, sun, amb, shadow);
let n_ramp_008 = ramp_constant(n_str, 0.0, vec4f(0,0,0,1), style.p[0].z, vec4f(1,1,1,1)).r;
let n_mix_004 = mix_blend(n_ramp_008, n_hs_002, n_hs_001);
...
let final_color = mix(n_npr_add, n_principled, style.p[0].x);
```

Line-for-line this is `hair.ts:49–81` with hand names swapped for `n_<id>` and the
four exposed literals swapped for `style.p[...]` reads — the pixel-golden test
(params at defaults) proves equivalence.

**Known intentional divergences from the Blender original** (already made in
`hair.ts`, the graph inherits them by construction): the noise→bump subtree on
Principled.Normal is *omitted from the graph* (imperceptible at 0.2 mix weight —
authoring decision, not a compiler transform); the bevel node is approximated by
`clamp(n.y, 0, 1)`.

## 9. Registry gaps the hair graph surfaces — RESOLVED (implemented in registry.ts)

1. `math/clamp01` — `saturate(a)` (WGSL builtin, no nodes.ts change needed).
2. `mix/add_emit` — `a + vec3f(b)` (emission-add of a scalar-scaled gate). Blender's
   original is an Emission shader + Add Shader; this is the ShaderToRGB-era stand-in
   the hand port already uses (`npr_stack = mix_shader_002 + gate_emit`).
3. `mix_shader` with a color literal on one side (hair's `HAIR_MIX_BG`) — plain
   `mix(a, b, fac)`, registry-only, no WGSL add.

## 10. Testing

- **Golden pixel tests** (the contract): compile each shipped graph (hair first, then
  the other 7 as they're transcribed), render the reference model frame offscreen,
  compare against the hand-written pipeline's readback. Tolerance 0 — same module
  structure should produce identical codegen through Tint/Naga. If a driver defies
  that, fall back to per-channel ≤ 1/255 on ≥ 99.9 % of pixels.
- **Emission snapshot tests** (fast, no GPU): `compileGraph(hairGraph).wgsl` snapshot
  — catches codegen churn in CI without a WebGPU device.
- **Validation tests**: cycle, dangling link, type mismatch, param-on-linked-socket,
  node-count overflow — each yields a diagnostic naming the offending node id.
- **Lifecycle test**: submit a graph with a deliberate WGSL-level error (e.g. NaN-able
  construct rejected by validation layer) → previous pipeline still renders,
  diagnostics point at the node.

## 11. Implementation layout

```
engine/src/graph/
  schema.ts      // StyleGraph types + limits (validateGraph lives in compile.ts)
  registry.ts    // NodeSpec table (§3) + conversion rules (§4) + literal formatting
  compile.ts     // validate/prune/toposort/peephole/emit → { wgsl, fsBody, slotMap, diagnostics }
  slots.ts       // per-MaterialPreset slot templates + epilogues (§6) + assembleModule
  presets/hair.ts     // §8 graph — first preset-as-data. Typed TS object, pure JSON
                      // (stringify round-trips); external presets load the same shape.
engine/tests/graph.test.mjs   // snapshot + validation + lifecycle tests (npm test;
                              // register.mjs adds ".js" so node can import dist/)
engine.ts        // TODO: StyleUniforms binding(4) in material BGL, zero-buffer fallback,
                 //   createRenderPipelineAsync swap path (§7), setStyleParam()
```

Migration: hand-written `hair.ts` stays until its golden test passes, then the hair
slot flips to the compiled path; repeat per preset. `mmd_classic` never migrates
(it's PMX-data-driven, not a node graph).

## 12. Editor interop & Blender version policy

**React Flow (reze.design / test page)**: the engine schema stays pure; each node may
carry `ui: { position }` which the compiler ignores and serialization round-trips.
The React Flow adapter is a bijection — node `id` ↔ RF node id, socket names ↔
`sourceHandle`/`targetHandle`, links ↔ edges. Never persist React Flow's own
serialization (viewport/selection junk, library-version coupling).

**Node preview debugging**: `compileGraph(graph, { previewNode: {node, socket} })`
overrides the output with any socket (float auto-splats to color) and prunes the
rest — Blender's Ctrl+Shift+Click viewer workflow, for side-by-side comparison
against the source Blender project.

**Blender version**: Reze node semantics are frozen **Blender 3.6 legacy-EEVEE** —
that's what `NODES_WGSL`/`eval_principled` port, and what the MMD preset templates
(仿深空之眼 etc.) were authored in. The nodes we support are semantically unchanged
through Blender 5; the two real divergences (Principled socket rework in 4.0,
EEVEE Next in 4.2) are handled as a socket-name mapping table in the porting guide,
not in the engine.
