# Reze Engine

[![npm](https://img.shields.io/npm/v/reze-engine)](https://www.npmjs.com/package/reze-engine)

**Zero-runtime-dependency** WebGPU engine for real-time MMD/PMX rendering — renderer, animation, IK, and physics, all in TypeScript.

![screenshot](./screenshot.png)

```bash
npm install reze-engine
```

## Features

- **Anime-style rendering** — toon-ramp NPR over a Principled GGX BSDF, mixed per material
- **Shader-graph materials** — every look is a Blender-style node graph compiled to WGSL; style groups bind any materials to any graph, fully customizable
- **HDR pipeline** — bloom, Filmic tone mapping, 4× MSAA, Apple-TBDR-friendly targets
- **In-house TS physics** — sequential-impulse rigid bodies for PMX rigs, zero dependencies
- **VMD animation** — MMD IK, morphs on a GPU compute path, and VMD export
- **Interactive editing** — GPU picking, transform gizmo, bone/material selection
- **Camera** — orbit, bone-follow, or a driven MMD camera VMD; ground + PCF shadows, multi-model scenes

See [Physics](#physics) and [Rendering](#rendering) for the internals.

## Used by

- [Reze Design](https://reze.design) (web-native scene composer & shader-graph styling)
- [Reze Studio](https://reze.studio) (MMD animation editor)
- [MiKaPo](https://mikapo.vercel.app) (motion capture)
- [Popo](https://popo.love) (LLM-generated poses)
- [MPL](https://mmd-mpl.vercel.app) (motion language)
- [Mixamo-MMD](https://mixamo-mmd.vercel.app) (FBX→VMD retarget)

## Quick start

```javascript
import { Engine } from "reze-engine"

const engine = new Engine(canvas)
await engine.init()

const model = await engine.loadModel("reze", "/models/reze/reze.pmx")
await engine.autoStyleGroups("reze")

await model.loadVmd("idle", "/animations/idle.vmd")
model.show("idle")
model.play()

engine.addGround()
engine.runRenderLoop()
```

## Codebase map

```
engine/src/
  engine.ts          Engine: device/context, render loop, all passes & pipelines,
                     per-model GPU resources, picking, gizmo   (entry point)
  model.ts           Model: skeleton, 4-bone skinning, morphs (CPU + GPU compute),
                     animation state, drives IK; per-frame update()
  animation.ts       AnimationClip, VMD bezier interpolation, playback/priority
  ik-solver.ts       MMD-style CCD IK (angle limits, solve-axis specialization)
  camera.ts          Orbit camera (alpha/beta/radius), bone-follow, mouse/touch
  math.ts            Vec3 / Quat / Mat4 (zero-alloc *Into variants for hot paths)
  pmx-loader.ts      PMX parser: mesh, bones, morphs, rigid bodies, joints
  vmd-loader.ts      VMD motion parser  ·  vmd-writer.ts  VMD export (Shift-JIS)
  asset-reader.ts    URL + local-folder asset resolution  ·  folder-upload.ts
  tga-loader.ts      TGA decoder (formats createImageBitmap can't read)
  index.ts           public exports

  graph/             shader-graph → WGSL compiler — materials as data
    schema.ts          ShaderGraph / StyleGroup / param types + validation
    registry.ts        node registry (Blender node → WGSL) + socket conversions
    compile.ts         validate → prune → toposort → peephole → emit
    render-class.ts    RenderClass / AlphaMode + the RENDER_CLASSES manifest
    slots.ts           per-render-class fs() shell (stencil/alpha) around the graph body
    presets/           the 9 built-in shader graphs (hair, face, eye, cloth, …)

  physics/           in-house rigid-body solver (~2.5k lines)
    physics.ts         RezePhysics: bone↔body sync, fixed-step accumulator + interpolation
    solver.ts          sequential-impulse PGS (joint + contact rows)
    contact.ts         narrowphase (analytical sphere/box/capsule pairs) + contact pool
    constraint.ts      6DOF spring joints   ·   world.ts  broadphase + step
    body.ts            SoA rigid-body store  ·  types.ts

  shaders/
    materials/       nodes.ts (Blender-node WGSL library the graph compiler emits into) +
                     common.ts (bindings, skinning VS, fs() shell)
    passes/          shadow, morph (GPU vertex-morph compute), bloom, composite (Filmic),
                     outline, selection, gizmo, pick, ground, mipmap
```

## API

One WebGPU **Engine** per page (singleton after `init()`). Models load by URL **or** from a user-selected folder ([below](#local-folder-uploads-browser)).

### Engine

```javascript
engine.init()
engine.loadModel(name, path)                 // or ({ files, pmxFile? }) for folder upload
engine.getModel(name) / getModelNames() / removeModel(name)
engine.setModelTransform(name, { position?, rotation?, scale?, visible? }) / getModelTransform(name)  // place a stage, scale or hide a model (scale is uniform)

engine.autoStyleGroups(name, overrides?)     // default style groups by material name
engine.applyStyleGroups(name, groups) / upsertStyleGroup / removeStyleGroup / getStyleGroups
engine.setMaterialVisible(name, material, visible) / toggleMaterialVisible / isMaterialVisible

engine.setIKEnabled(enabled)
engine.setPhysicsEnabled(enabled)
engine.resetPhysics()                        // re-pose bodies from animation + zero velocities (call if physics explodes)

engine.setCameraFollow(model, bone?, offset?) / setCameraFollow(null)
engine.setCameraTarget(vec3) / setCameraDistance(d) / setCameraAlpha(a) / setCameraBeta(b)

engine.loadCameraVmd(url) / loadCameraVmdFromBuffer(buffer)   // MMD camera track (dedicated file or a VMD's camera block) drives target/rotation/distance/fov — default-on once loaded
engine.setCameraVmdEnabled(on) / isCameraVmdEnabled() / hasCameraVmd() / clearCameraVmd()   // toggle the shot; off falls back to orbit / follow-bone

engine.setWorld({ color?, strength? }) / setSun({ color?, strength?, direction? })   // runtime lighting
engine.addGround(options?)
engine.runRenderLoop(callback?) / stopRenderLoop()
engine.getStats()                            // fps + smoothness metrics (frameTimeMax, fps1PercentLow, jitter)
engine.dispose()
```

**Options** — Blender-style scene config: `world` = environment lighting, `sun` = directional lamp (`direction` points from sun into the scene), `camera` = framing (`fov` in radians). Callbacks: `onRaycast`, `onGizmoDrag`. The shadow map is cast from `sun.direction` — the same vector the shader lights with — so shading and cast shadows stay coupled.

### Model

```javascript
await model.loadVmd(name, url) / model.loadClip(name, clip)
model.show(name)
model.play(name, { priority?, loop? })       // priority: higher wins when clips compete (0 = default)
model.pause() / stop() / seek(time)
model.getAnimationProgress()                 // { current, duration (s), playing, paused, looping, … }
model.exportVmd(name)                        // → ArrayBuffer (Shift-JIS bone/morph names)

model.rotateBones({ 首: quat }, ms?) / moveBones({ センター: vec3 }, ms?)
model.setMorphWeight(name, weight, ms?)
model.resetAllBones() / resetAllMorphs()
model.getBoneWorldPosition(name)
```

`AnimationClip` holds keyframes only (bone/morph tracks keyed by `frame`, plus `frameCount`); time advances at fixed `FPS` (exported, default 30).

### Local folder uploads (browser)

Feed a `<input type="file" webkitdirectory>` `FileList` (or drag/drop) into the engine; textures resolve relative to the chosen PMX inside that tree.

> **Gotcha:** copy `input.files` into an array **before** `input.value = ""` — the `FileList` is live and clearing the input empties it.

`parsePmxFolderInput(fileList)` returns a tagged result; for `single` you get `{ files, pmxFile }` directly, for `multiple` show a picker over `pmxRelativePaths` and resolve with `pmxFileAtRelativePath(files, path)`. Then:

```javascript
const picked = parsePmxFolderInput(e.target.files)
e.target.value = ""
if (picked.status === "single") await engine.loadModel("m", { files: picked.files, pmxFile: picked.pmxFile })
```

VMD and other assets still load by URL when the path starts with `/` or `http(s):`; relative paths resolve against the PMX directory.

### Interactive pose editing

Double-click picks a bone or material (per-triangle dominant-joint from the GPU pick, so one handler serves both modes); a local-axis transform gizmo drags it. **The engine only reports — it never writes the skeleton itself**, so the host chooses the write policy.

```typescript
engine.setSelectedBone(modelName | null, boneName | null)       // shows the gizmo
engine.setSelectedMaterial(modelName | null, materialName | null) // selection outline

onRaycast: (modelName, material, bone, screenX, screenY) => { ... } // modelName "" = missed

type GizmoDragEvent = {
  boneName: string; boneIndex: number; kind: "rotate" | "translate"
  localRotation: Quat; localTranslation: Vec3   // target absolute local transform
  phase?: "start" | "end"                       // undefined during drag moves
}
```

The gizmo consumes mouse input inside its bounding sphere so drags never fight camera orbit. Apply the reported transform — runtime override (below) or keyframe edit into a clip you re-`loadClip`:

```javascript
onGizmoDrag: (e) => {
  const model = engine.getModel(e.modelName)
  if (!model) return
  if (e.phase === "start") {
    model.pause()
    model.setClipApplySuspended(true)
    return
  } // stop re-sampling wiping the edit
  if (e.phase === "end") return
  if (e.kind === "rotate")
    model.rotateBones({ [e.boneName]: e.localRotation }, 0) // 0 = instant write
  else model.setBoneLocalTranslation(e.boneIndex, e.localTranslation)
}
// play()/seek() auto-clear the suspend flag (edit is lost — runtime-override semantic).
```

Note the asymmetry: rotation goes through `rotateBones(…, 0)`, but translation uses `setBoneLocalTranslation(idx, v)` — `moveBones` converts VMD-relative→local, and the gizmo output is already local.

## Shader graphs & style groups

Materials are styled by **shader graphs** — plain JSON (`ShaderGraph`) validated and compiled to WGSL at runtime. Node semantics are frozen **Blender 3.6 legacy-EEVEE**, so community Blender NPR presets port by transcription. Nine graphs ship built-in (`FACE_GRAPH`, `HAIR_GRAPH`, `BODY_GRAPH`, `EYE_GRAPH`, `METAL_GRAPH`, `STOCKINGS_GRAPH`, `CLOTH_SMOOTH_GRAPH`, `CLOTH_ROUGH_GRAPH`, `DEFAULT_GRAPH` — the neutral base) as a starter library; you can also author or import your own.

A **style group** binds `{ materials, graph, renderClass?, alphaMode? }` — a set of materials, the graph that shades them, and the engine's small pass-integration vocabulary (`renderClass`: `auto`/`eye`/`hair` for stencil/cull/draw-order; `alphaMode`: `opaque`/`hashed`). **Groups are user-defined and unlimited** — any materials, any graph. A graph is pure shading; `renderClass` carries the built-in effects (hair's over-eyes stencil, the eye see-through stamp), so any graph in an `eye`/`hair` group inherits them. **Every group needs a valid graph**; a material in no group renders the **neutral default** (`DEFAULT_GRAPH`).

Two ways to make groups:

```javascript
import { HAIR_GRAPH, compileGraph } from "reze-engine"

// 1. autoStyleGroups — one default group per matched category (its shipped graph). Easy path.
await engine.autoStyleGroups("reze")

// 2. applyStyleGroups — arbitrary groups: any id, any materials, any graph.
await engine.applyStyleGroups("reze", [
  { id: "hair", materials: ["髪", "前髪"], graph: HAIR_GRAPH, renderClass: "hair" },
  { id: "visor", materials: ["visor", "hud"], graph: myCustomGraph }, // your own graph
])
engine.setStyleParam("reze", "hair", "rim", 0.8) // exposed param → instant uniform write
engine.removeStyleGroup("reze", "hair") // its materials drop to the neutral default

// Headless (no GPU needed):
const { ok, wgsl, diagnostics } = compileGraph(HAIR_GRAPH, { renderClass: "hair" })
```

**How `autoStyleGroups(model, overrides?)` resolves** — it assigns each material a *style category*, then buckets materials by category into one group each:

1. **`overrides` first** — an explicit `{ category: [materialNames] }` map (the arg). Use it for the names the built-in hints can't read.
2. **Then built-in name hints** — a case-insensitive **substring** match of the material name against per-category JP/CN/EN keyword lists, ordered **most-specific-first** so families don't collide (`靴下`/`stocking` resolves to `stockings` before `靴`/`shoes` would hit `cloth_smooth`). This covers standard-named models with no overrides at all.
3. **No match → ungrouped** — the material renders the neutral default. "Unmatched" is a real, intended outcome, not a catch-all bucket.

The **built-in name hints**, checked top-to-bottom (first match wins), with the graph and pass-integration each category carries:

| Category | Graph · render-class / alpha | Matches a name containing (case-insensitive substring) |
| --- | --- | --- |
| `stockings` | `STOCKINGS_GRAPH` · `hashed` alpha | 靴下 · ソックス · タイツ · ニーソ · 袜 · stocking · socks · tights |
| `eye` | `EYE_GRAPH` · `eye` | 白目 · 目影 · 二重 · 睫 · まつげ · まゆ · 眉 · 目 · 瞳 · 眼 · eye · iris · pupil · lash · brow |
| `face` | `FACE_GRAPH` | 顔 · 颜 · 脸 · かお · face · 舌 · tongue · 牙 · 牙齿 · 歯 · teeth · tooth · 口腔 · 口内 · mouth · 嘴 · 歯茎 · gums |
| `hair` | `HAIR_GRAPH` · `hair` | 前髪 · 後髪 · 髪 · 髮 · 头发 · 頭髪 · もみあげ · アホ毛 · ヘア · hair · ahoge · bang |
| `body` | `BODY_GRAPH` | 肌 · 皮肤 · skin |
| `metal` | `METAL_GRAPH` | 金属 · メタル · metal |
| `cloth_smooth` | `CLOTH_SMOOTH_GRAPH` | 服 · 衣 · 裙 · 裤 · スカート · ワンピ · リボン · 袖 · 靴 · 鞋 · 帽 · 体 · 飾 · 饰 · 尾 · skirt · dress · ribbon · sleeve · shoes · boot · hat · cloth · accessor |

`cloth_rough` and `default` have **no** name hints — a material reaches them only via an explicit `overrides` entry. The group `id` is the category name, so re-running `autoStyleGroups` is idempotent; its promise resolves after every graph compiles, so `getStyleGroups(model)` is ready the moment it resolves — seed your own store from it, then edit with `applyStyleGroups`.

Validation catches material conflicts, type mismatches, cycles, and bad links with node-level diagnostics; a failed compile keeps the previous pipeline rendering (fallback-on-error).

## Physics

In-house sequential-impulse rigid-body solver for PMX rigs (sphere / box / capsule colliders, 6DOF spring joints), ~2.5k lines of TypeScript, no external dependency, at quality comparable to Bullet's defaults. A fixed-timestep accumulator runs at a constant **60 Hz** (≤6 substeps/frame) so spring impulse, damping, and integration stay deterministic; dynamic bodies are **render-interpolated** between substeps to stay smooth when the display rate ≠ 60 Hz.

**Per substep:** `predict velocities → broad + narrowphase → solve constraints (10 iters) → split-impulse position correction → integrate`.

- **Solver** — projected Gauss-Seidel, joint rows + contact rows in one loop. Joints are 6DOF springs (3 linear + 3 angular) with stop-ERP limit correction and per-axis stiffness×error impulse. Linear rows pivot on each body's own joint-frame origin (Spring2-style, not Bullet 2.7x's shared mid-anchor), so a violated joint pulls itself back together instead of degenerating into torque and "breaking".
- **Angular limits** — hybrid: small violations (< 0.5 rad, the resting-cloth regime) use per-axis Euler stop rows, which converge cleanly and keep resting cloth still; larger violations switch to a single geodesic row toward the Euler-clamped target rotation, because per-axis Euler rows chase phantom errors near the ±90° singularity and pump energy. Ranged stops are unilateral (accumulated impulse clamped to the corrective sign) so a limit pushes back into range but never brakes natural recovery; locked axes stay bilateral equality joints. Spring rows stay per-axis, with stiffness clamped to the `k·dt² ≤ ¼` stability bound.
- **Narrowphase** — analytical sphere-sphere / -capsule / -box and capsule-capsule / -box. Capsule-capsule emits multiple contacts along near-parallel axes so cloth can't pivot around a single closest point.
- **Speculative contacts** (`margin 0.04`) fire at near-touch with a push-only clamp — inert until real overlap, but they stop fast bodies tunneling thin surfaces in one substep.
- **Split-impulse correction** resolves penetration by a mass-weighted translation *outside* the velocity solver, so joint pulls can't fight separation.
- **Kinematic advancement** — bone-driven bodies move toward the frame's bone pose incrementally per substep, with velocities derived over the fixed step, so the solver never sees more than one 60 Hz step of anchor motion regardless of render dt.
- **Discontinuity guards** — a bone-pose jump beyond continuous motion (timeline scrub, long stall) rigidly carries each dynamic body along with its kinematic root's transform delta and zeroes momentum instead of dragging cloth across the gap; correction velocities are clamped (120 u/s linear, 30 rad/s angular), per-step travel is capped, and any body that still goes non-finite is restored to its previous substep pose.
- Sleeping is off (cloth must always react); resting bodies bleed micro-velocity via per-PMX damping.

Engine surface is just `setPhysicsEnabled` / `resetPhysics` — all tuning (mass, damping, friction, restitution, joint stiffness/limits, collision groups) lives on the PMX rig.

## Rendering

Each built-in shader graph mixes an NPR stack with a Principled-style BSDF, so characters keep a flat illustrated look while highlights and reflections stay grounded. A graph compiles to a fragment shader following one 7-stage layout (node primitives from `nodes.ts`, the fs() shell from `common.ts`):

```
(A) setup → (B) texture + alpha → (C) NPR stack → (D) optional bump
→ (E) Principled BSDF → (F) NPR↔PBR mix → (G) FSOut
```

`default` uses only A/B/E/G; the NPR graphs add C (and sometimes D), with F choosing how NPR-leaning the result is.

- **PBR core** (`eval_principled`) — GGX + Schlick Fresnel, Walter–Smith G1, Fdez-Agüera 2019 multi-scatter, Karis split-sum DFG LUT, Heitz 2016 LTC direct-spec, optional sheen.
- **NPR toolbox** — toon ramps (constant / fwidth-AA'd), HSV warm-shadow / cool-light remaps, fresnel + layer-weight rims, value-noise bump, Voronoi metallic sparkle, BT.601-gated emission.


| Built-in graph | Notes                                                                   |
| -------------- | ----------------------------------------------------------------------- |
| `default`      | Plain Principled, metallic 0, rough 0.5                                 |
| `eye`          | Plain + post-eval emission ×1.5                                         |
| `face`         | Toon + warm rim + dual-fresnel rim + bright-tex gate, noise bump        |
| `body`         | Toon + warm rim + fresnel + facing rim, noise bump                      |
| `hair`         | Toon + fresnel + bevel + bright-tex gate, 20% PBR                       |
| `cloth_smooth` | Toon + bevel + emission overlay (×18)                                   |
| `cloth_rough`  | Same NPR, live noise bump, rough 0.82                                   |
| `metal`        | Toon + emission overlay (×8), Voronoi base, metallic 1                  |
| `stockings`    | Gradient × facing mask + HSV emission (×5), sheen 0.7, **alpha-hashed** |


**Post & output.** Directional shadow map (2048², depth32float, PCF) → HDR main pass at 4× MSAA (`rg11b10ufloat` color + `rg8unorm` aux MRT for bloom mask + alpha; fits Apple-Silicon TBDR tile memory so MSAA resolves in-tile, `rgba16float` fallback) → bloom mip pyramid → Filmic tone map (Blender 3.6 "Filmic / Medium High Contrast" LUT) → inverted-hull outline.

- **Alpha-hashed transparency** (`stockings`) — Wyman & McGuire 2017 derivative-aware stochastic discard in object space, so self-overlapping meshes resolve under MSAA with opaque depth writes and the dither doesn't swim.
- **See-through hair over eyes** — stencil-gated extra pass: the eye stamps `EYE_VALUE`, main hair skips it, an extra pass matches it and blends hair at 25% in linear HDR so eyes stay readable.

## Tutorial

[How to Render an Anime Character with WebGPU](https://reze.one/tutorial)