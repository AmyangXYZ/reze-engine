# Style Groups — Engine API Contract v0.2

Supersedes the fixed 9-slot style-override system (0.18) with **user-defined style
groups**: an unlimited set of `{ materials, graph, renderClass }` bundles per model.
The user decides which materials belong to which group and which node graph each group
uses; the engine owns only a tiny, closed `renderClass` vocabulary that carries pass
integration (stencil, cull, draw order). See `docs/graph-compiler-spec.md` for the
graph→WGSL compiler this builds on.

## 1. Why

`MaterialPreset` welded two unrelated things into one fixed enum: **shading** (the node
graph / look) and **pass integration** (transparent vs opaque, the eye stencil stamp,
the hair-over-eyes second pass, cull, draw order). Artists group materials by their own
logic and want arbitrary looks per subset — a weapon, a visor, an emblem, two distinct
cloth looks — none of which fit 9 fixed roles. Style groups decouple the two:

- **User owns**: which materials are in which group, and which graph each group uses.
  Unlimited groups, arbitrary grouping.
- **Engine owns**: the `renderClass` vocabulary and its stencil/draw-order contracts —
  because that is where rendering correctness lives and it must not be user-inventable.

**Overlay-first (v1).** Three levels, kept distinct so "compiled vs hand-shader" is never
ambiguous:

1. **Groups always render compiled graphs** — that is what a group *is*. This includes
   auto-created default groups (§8): if a material is in a group, its pixels come from the
   group's compiled-graph pipeline.
2. **The ungrouped base path stays hand-shaders** in all of v1 — a material in no group
   renders via the existing preset shaders, untouched. Groups override only what they claim.
3. **Auto-grouping is opt-in per model load** (a flag, default **off**, §8). With it off, a
   freshly loaded model has zero groups and is byte-identical to 0.18. With it on (what
   reze-design passes), matched materials render compiled graphs immediately — a documented
   choice, accepting minor pre-golden drift on the compiled path, not an accident.

The later **primary phase** (§12 phase 5) is a *separate* step: retiring the hand-shader
ungrouped path entirely so groups are the only path. That one is gated on the pixel-golden
harness. v1 does not touch it — the base path stays hand-shaders regardless of how many
groups exist.

## 2. Types

```ts
// Small closed vocabulary — pass integration only. Not user-extensible.
export type RenderClass = "auto" | "eye" | "hair"

export type StyleGroup = {
  id: string                 // stable, unique within a model; /^[a-z0-9_-]+$/. Engine keys by this.
  label?: string             // human display name ("Skirt", "Visor"); host-owned, round-tripped,
                             //   never interpreted by the engine. Defaults to `id` if omitted.
  materials: string[]        // material names; each material is in AT MOST one group
  graph: StyleGraph          // pure shading (see §7 — StyleGraph drops `slot`)
  renderClass?: RenderClass  // stencil/cull/draw-order axis. Default "auto"
  alphaMode?: AlphaMode      // alpha-handling axis, orthogonal to renderClass. Default "opaque"
                             //   ("hashed" = Wyman hashed alpha test, e.g. stockings)
}

// StyleGraph change (from graph-compiler-spec §2):
//   - REMOVE  `slot: MaterialPreset`   (a graph is pure shading now)
//   - ADD     `tags?: string[]`        (soft hint for library filtering + smart
//                                        default-group / render-class matching)

// UI manifest — descriptive, so reze-design's group picker is data-driven instead of
// hardcoding strings. The effect implementations stay engine-side; this only describes.
export type RenderClassInfo = {
  id: RenderClass
  label: string
  description: string
  pairsWith?: RenderClass    // effect needs a counterpart class present (hair↔eye)
}
export const RENDER_CLASSES: RenderClassInfo[]
```

**Known limitation (non-blocking).** Materials are identified by name in `materials:
string[]`, so a PMX with duplicate material names is ambiguous — the same as 0.18's
`setMaterialPresets`, so consistent and fine to leave. An index-based identity
(`{ index }` or `name#n`) is a clean future add if it ever bites.

`RENDER_CLASSES` ships with:

| id | label | description | pairsWith |
|---|---|---|---|
| `auto` | Standard | Opaque or transparent, derived from material alpha. | — |
| `eye` | Eye | Stamps the see-through stencil; visible through hair. | `hair` |
| `hair` | Hair | Reads the eye stencil so eyes show through the silhouette. | `eye` |

## 3. Public API

All group state is **per model** (materials belong to models). Async methods compile
and swap pipelines off the render loop; a failed compile keeps the previous pipeline.

**State ownership.** The **host is the single source of truth** for group definitions and
pushes them down with `applyStyleGroups` (or `upsert`/`remove`). The engine holds the
current set only to drive rendering — it is not a second authoritative store the host must
reconcile against. `getStyleGroups` exists for **bootstrap/read** (e.g. reading the
auto-created defaults after load to seed the host's own store, §8), not for round-tripping
host edits back through the engine. Once the host owns a set, it keeps owning it; the
engine never mutates group definitions behind the host's back (auto-grouping only ever
*creates* the initial set, before the host has one).

```ts
// Full-set replace — the authoritative form. Diff against current groups, compile new/
// changed ones, swap atomically, drop removed ones. Returns per-group diagnostics.
applyStyleGroups(modelName: string, groups: StyleGroup[]): Promise<ApplyStyleGroupsResult>

// Incremental — add or replace a single group by id (compile + swap just that one).
upsertStyleGroup(modelName: string, group: StyleGroup): Promise<ApplyStyleGroupResult>

// Remove one group by id; its materials fall back to the ungrouped (hand-shader) path.
removeStyleGroup(modelName: string, groupId: string): void

// Instant adjust-tier: write one exposed slider on a group's applied graph. No recompile.
setStyleParam(modelName: string, groupId: string, paramId: string, value: number | [number, number, number]): boolean

// Read current groups (including auto-created defaults) for editor round-trip.
getStyleGroups(modelName: string): StyleGroup[]

// Clear all groups — every material returns to the hand-shader path.
resetStyleGroups(modelName: string): void

// Auto-create default groups from PMX material-name hints (§8). Called automatically on
// model load; exposed so a host can re-derive defaults after clearing.
autoStyleGroups(modelName: string): Promise<ApplyStyleGroupsResult>
```

```ts
export type GroupDiagnostic = { groupId: string; diagnostics: Diagnostic[]; ok: boolean }
export type ApplyStyleGroupsResult = {
  ok: boolean                    // every group compiled + swapped
  groups: GroupDiagnostic[]
  unknownMaterials: string[]     // names in a group that no model material matches
  conflicts: string[]            // materials claimed by >1 group (later group wins; logged)
}
export type ApplyStyleGroupResult = { ok: boolean; diagnostics: Diagnostic[]; slotMap: StyleSlot[] }
```

**Validation before compile** (whole-set invariants the per-graph validator can't see):

- A material appears in ≤1 group. Duplicates → reported in `conflicts`; last group in
  array order wins (deterministic), the material is removed from earlier groups.
- Group ids unique and well-formed; unknown group id on `setStyleParam`/`removeStyleGroup`
  is a no-op returning false/void.
- `materials` names that match no model material → `unknownMaterials` (warning, not fatal).

The 0.18 slot API (`applyStyleGraph`, `setStyleParam(slot,…)`, `resetStyleSlot`) is
**removed** in favor of these — see §9.

## 4. Overlay semantics

Each of a model's draw calls resolves to a group or to nothing:

```
drawCall.groupId = group whose `materials` contains drawCall.materialName, else null
```

- **Grouped** (`groupId != null`): renders with that group's compiled graph pipeline;
  its material bind group binding(4) points at the group's uniform buffer (§6); its
  render behavior is the group's `renderClass` (§5).
- **Ungrouped** (`groupId == null`): renders exactly as today — `pipelineForPreset(draw.preset)`
  over the hand-written preset shaders, binding(4) → the shared zero buffer.

So groups are a strict overlay: they never remove capability, only override the
materials they claim. A model with zero groups renders identically to 0.18.

## 5. Render-class contract

`renderClass` selects the template (prelude/epilogue/module decls) the graph body is
assembled into (§7) **and** the pipeline state + draw scheduling the engine applies.
It is the *only* engine-owned, non-user-extensible part of a group.

### `auto` (default, ~95% of groups)
- Standard depth (`less-equal`, depth-write on), standard blend, `cull: none`.
- Opaque vs transparent stays **per material** (`materialAlpha < 1.0` → transparent),
  driving which draw bucket/order the material lands in — a single `auto` group may hold
  both, and its one pipeline serves both (blend is src-alpha; α=1 opaque reduces to
  replace). No stencil, no extra pass.

### `eye`
- Pipeline: `cull: front`, small negative depth bias, `stencilFront/Back: replace` with
  `STENCIL_EYE_VALUE`, writeMask `0xff`.
- Template prelude adds the rear-view gate (open-shell head occlusion via the 頭 bone;
  built-in, not in the graph — see `EYE_TEMPLATE`).

### `hair`
- Pipeline (primary): `stencil compare: not-equal` `STENCIL_EYE_VALUE` (skip eye-stamped
  pixels), writeMask 0.
- Engine also compiles a **second `over-eyes` variant** from the same module
  (`IS_OVER_EYES=1`, `depthWrite off`, `stencil compare: equal`) and re-draws every
  hair-class draw with it after the opaque bucket — the 25%-alpha see-through pass.

### Draw-order + stencil contract (engine-owned)
Generalizes today's `presetRank` and `drawHairOverEyes`:

```
within each bucket (opaque, then transparent):
  rank(auto)=0, rank(eye)=1, rank(hair)=2       // eye stamps before hair reads
after the opaque bucket, before transparent:
  re-draw every hair-class draw with its over-eyes pipeline
```

**Graceful degradation.** The effect fires only when both classes are present:
- No `eye`-class group → nothing stamps → `hair` primary (`not-equal`) passes everywhere
  (draws normally), over-eyes (`equal`) matches nothing (draws nothing). Correct for a
  hairless-eyeless prop or a model whose eyes weren't classed.
- Multiple `eye` or `hair` groups are fine — all eye-class groups stamp the same value,
  all hair-class groups test it. A twin-tail model with two hair groups gets the effect
  on both. The engine never requires exactly one of anything.

## 6. Pipeline & uniform-buffer lifecycle

**Per group, on compile:**
1. `compileGraph(group.graph)` → `{ wgsl, slotMap, diagnostics }`. WGSL is assembled
   with the group's `renderClass` template (§7).
2. `pushErrorScope` → `createShaderModule` → `getCompilationInfo()`; WGSL errors mapped
   back to node ids via the `// @node:<id>` markers.
3. `createRenderPipelineAsync` with the render-class pipeline state (§5). For `hair`,
   also build the `over-eyes` variant.
4. On success, atomically install `{ pipeline, overEyesPipeline?, slotMap, renderClass }`
   keyed by `(modelName, groupId)`; on failure, keep the previous install and return
   diagnostics.

**Generation guard.** A per-`(model, group)` counter; a compile that finishes after a
newer `applyStyleGroups`/`upsert`/`remove` touched the same group is discarded (stale
write). Old pipelines are GC'd after the swap (referenced by at most the in-flight
encoder).

**Uniform buffers.** One 256-byte `StyleUniforms` buffer per group (replacing the 9
fixed per-slot buffers). Allocated on group create, destroyed on group remove. A
material's bind group binding(4) is (re)bound to its group's buffer whenever its group
assignment or the group's buffer changes — the `baseBindGroupEntries` rebind mechanism
already added for `setMaterialPresets` generalizes directly. Ungrouped materials keep
the shared `zeroStyleBuffer`.

`setStyleParam(model, groupId, paramId, value)` looks up the group's `slotMap`, resolves
the param's `vec4` offset, and `queue.writeBuffer`s — no pipeline touch, instant.

## 7. Compile integration

`StyleGraph` drops `slot`; the group supplies `renderClass` separately, so shading and
integration compose independently (a metal-look graph in an `eye`-class group = a glowing
metal eye that still reads through hair). Two changes to the compiler layer:

- `assembleModule(renderClass, fsBody, includeStyleUniforms)` — keyed on `RenderClass`,
  not the old `MaterialPreset`. The current `SLOT_TEMPLATES` map (`hair`, `stockings`,
  `eye`) becomes `RENDER_CLASS_TEMPLATES` keyed on `RenderClass`. **`stockings`' hashed
  alpha is a shading/alpha concern, not a stencil one** — it moves out of the render-class
  set: either folded into the graph (a hashed-alpha discard node) or carried as a separate
  orthogonal `alphaMode` axis in a later revision. v1 render-class templates are exactly
  `{ auto (default), eye, hair }`.
- `validateGraph` no longer requires/reads `slot`; `graph.tags` is ignored by the compiler
  (host-only metadata, round-tripped).

## 8. Auto-default groups

Reproduces today's material resolution as groups, so casual users get editable looks with
zero interaction:

1. Resolve each material through **`resolvePreset(name, inst.materialPresets)`** — the
   host's `setMaterialPresets` map first, then `PRESET_NAME_HINTS` — to a preset label.
   (Using `resolvePreset`, not raw hints, is what makes the CJK-named demo group correctly
   once the host has pushed its curated map; §9.)
2. Bucket materials by label into one group each; group id = the label
   (`"hair"`, `"eye"`, …); `label` = a display name for that bucket; `graph` = the exported
   preset graph for that label (`HAIR_GRAPH`, `FACE_GRAPH`, …); `renderClass` = `eye` for
   the eye bucket, `hair` for the hair bucket, `auto` for the rest.
3. Materials resolving to `mmd_classic` (no map entry, no hint) → left **ungrouped**
   (hand-shader / `mmd_classic` path), not forced into a group.

Because the graphs are snapshot-identical to the hand shaders, an auto-grouped model
renders the same as an ungrouped one — modulo the compiled vs hand-written path (which the
pixel-golden harness certifies before the primary phase, §12).

**Opt-in + ordering.** Auto-grouping is controlled by a load flag,
`loadModel(name, { path, autoStyleGroups?: boolean })`, **default `false`** (so a bare load
stays byte-identical to 0.18). When `true`, `loadModel`'s promise **resolves only after
auto-grouping completes, including the async graph compiles** — so `getStyleGroups(model)`
is fully populated the instant the load resolves, and the first rendered frame already shows
compiled-graph looks (no hand-shader flash). A host that wants control over ordering (e.g.
call `setMaterialPresets` between load and grouping) leaves the flag off and calls
`autoStyleGroups(model)` itself; that promise carries the same after-compile guarantee.

## 9. Migration from the 0.18 slot API

Removed (days-old, small userbase, we control the only consumer):

| 0.18 (slot) | 0.19 (group) |
|---|---|
| `applyStyleGraph(graph)` (graph.slot picks slot) | `upsertStyleGroup(model, { id, materials, graph, renderClass })` |
| `setStyleParam(slot, id, v)` | `setStyleParam(model, groupId, id, v)` |
| `resetStyleSlot(slot)` | `removeStyleGroup(model, groupId)` |
| per-slot `styleBuffers`/`styleOverrides`/`styleGenerations` maps | per-`(model, groupId)` maps |
| `SLOT_TEMPLATES` keyed by `MaterialPreset` | `RENDER_CLASS_TEMPLATES` keyed by `RenderClass` |
| `StyleGraph.slot` | removed; `renderClass` on the group + optional `StyleGraph.tags` |

**Kept, unchanged:** `MaterialPreset`, `resolvePreset`, the hand-written preset pipelines,
and **`setMaterialPresets(model, map)`**. They serve two live roles: (1) the ungrouped
fallback path still needs material→preset resolution, and (2) `autoStyleGroups` resolves
each material through `resolvePreset(name, inst.materialPresets)` — **map first, then name
hints** — so a host's curated `setMaterialPresets` map is exactly what drives correct
auto-grouping on models the built-in hints can't read (§8, e.g. the CJK-named demo). So the
demo path is: `setMaterialPresets(curated map)` → `autoStyleGroups` yields correct groups;
no separate hand-built `applyStyleGroups` needed (though pushing one explicitly is equally
valid).

## 10. Data flow (one `applyStyleGroups` call)

```
applyStyleGroups(model, groups)
  → validate whole-set (dupes/conflicts/unknown mats)          §3
  → diff vs current groups → { added, changed, removed }
  → for each removed:  destroy uniform buffer, drop install, rebind its materials
                       back to the ungrouped path (binding4 → zeroStyleBuffer)
  → for each added/changed (in parallel):
        compileGraph(graph) → assembleModule(renderClass,…)    §6,§7
        createShaderModule + getCompilationInfo                (node-id diagnostics)
        createRenderPipelineAsync (+ over-eyes if hair)
        install keyed by (model, groupId), bump generation
        allocate/keep 256B uniform buffer, write param defaults
  → walk model draw calls: assign groupId from material→group map,
        rebind binding(4) → group buffer, tag renderClass
  → re-sort draw calls by (bucket, renderClass rank)           §5
  → return per-group diagnostics
```

Per frame, `drawMaterials` picks `install.pipeline` for grouped calls (else
`pipelineForPreset`), and the over-eyes pass iterates hair-class grouped calls.

## 11. reze-design integration

- **Import, don't copy.** Factory library = the engine's exported graphs
  (`HAIR_GRAPH`, `FACE_GRAPH`, …) shown read-only; editing forks into reze-design's own
  storage. `.graph.json` export/import stays the user-portability format.
- **The group is the object the UI is built around**; `renderClass` is an auto-assigned
  property surfaced only in an advanced per-group setting. Casual flow: load → auto-groups
  appear named + classed → pick looks. Pro flow: create/split/rename groups, assign looks,
  occasionally set render-class.
- **`RENDER_CLASSES`** drives the render-class picker (labels, descriptions, and
  `pairsWith` → "you have a hair group but no eye group; the see-through effect won't
  show").
- **`NODE_REGISTRY`** already drives the node palette. Same data-driven pattern.

**Live material preview falls out for free.** Previewing a graph on a sphere (or any
geometry) is just a one-material group on a second `Engine` instance —
`loadModel("sphere", …)` + `applyStyleGroups("sphere", [{ id: "p", materials: ["sphere"], graph }])`
— no dedicated preview API needed as long as a host can spin up a second live WebGPU
context. A separate preview/thumbnail API is only warranted for **baked, context-free
previews** (library card thumbnails, an SEO/OG image for a shared scene) that must render
*without* a live canvas — offscreen render → PNG readback. That is a distinct engine
capability (offscreen target + `copyTextureToBuffer` + PNG encode); worth considering in
the same cycle so groups and baked previews land together, but out of scope for this
contract. (Per-*node* output preview on the loaded model is already covered by
`compileGraph`'s `previewNode` option — unrelated to material thumbnails.)

## 12. Implementation phasing

1. **Types + compile plumbing** — `RenderClass`, `RENDER_CLASS_TEMPLATES`,
   `assembleModule(renderClass,…)`, drop `StyleGraph.slot` / add `tags`, move stockings
   hashed-alpha off the render-class axis. Snapshot tests updated (graphs unchanged
   except the field rename).
2. **Group runtime** — per-`(model,group)` installs + uniform buffers, `applyStyleGroups` /
   `upsert` / `remove` / `setStyleParam` / `getStyleGroups`, draw-call group assignment +
   binding(4) rebind, render-class draw-order generalization of `presetRank` /
   `drawHairOverEyes`. Remove the 0.18 slot API.
3. **Auto-defaults** — `autoStyleGroups` via `resolvePreset` (map + hints), gated by the
   `loadModel` opt-in flag (default off); `loadModel`'s promise awaits grouping+compile.
   Verify an auto-grouped model matches an ungrouped one.
4. **reze-design refactor** — group panel over `applyStyleGroups`, `RENDER_CLASSES`
   picker, factory-import + fork-on-edit.
5. **(Later) Primary phase** — gated on the pixel-golden harness: default groups adopt
   compiled graphs, hand-written preset pipelines retire.
```
