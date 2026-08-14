import { Camera } from "./camera"
import { decodeDds, isDds } from "./dds-loader"
import { Mat4, Quat, Vec3 } from "./math"
import { decodePsd, isPsd } from "./psd-loader"
import { Model, MATERIAL_MORPH_MULTIPLY, type Material } from "./model"
import { MORPH_COMPUTE_WGSL } from "./shaders/passes/morph"
import { CULL_COMPUTE_WGSL } from "./shaders/passes/cull"
import { decodeTga } from "./tga-loader"
import { VMDLoader } from "./vmd-loader"
import { CameraAnimation } from "./camera-animation"
import { PmxLoader } from "./pmx-loader"
import { RezePhysics } from "./physics"
import type { WindOptions } from "./physics/world"
import {
  createFetchAssetReader,
  createFileMapAssetReader,
  deriveBasePathFromPmxPath,
  fileListToMap,
  findFirstPmxFileInList,
  joinAssetPath,
  normalizeAssetPath,
  type AssetReader,
} from "./asset-reader"
import { BRDF_LUT_SIZE, BRDF_LUT_BAKE_WGSL } from "./shaders/dfg_lut"
import { LTC_MAG_LUT_SIZE, LTC_MAG_LUT_DATA } from "./shaders/ltc_mag_lut"
import { SHADOW_DEPTH_SHADER_WGSL } from "./shaders/passes/shadow"
import { GROUND_SHADOW_SHADER_WGSL } from "./shaders/passes/ground"
import { OUTLINE_SHADER_WGSL } from "./shaders/passes/outline"
import { TRANSPARENT_DEPTH_PREPASS_WGSL } from "./shaders/passes/depth-prepass"
import { SELECTION_MASK_SHADER_WGSL, SELECTION_EDGE_SHADER_WGSL } from "./shaders/passes/selection"
import { GIZMO_SHADER_WGSL } from "./shaders/passes/gizmo"
import {
  BLOOM_BLIT_SHADER_WGSL,
  BLOOM_DOWNSAMPLE_SHADER_WGSL,
  BLOOM_UPSAMPLE_SHADER_WGSL,
} from "./shaders/passes/bloom"
import { AGX_LUT_GZ, AGX_LUT_SIZE } from "./shaders/agx-lut"
import {
  buildCompositeShader,
  buildFieldShader,
  parseEffectAnchors,
  EFFECT_ANCHORS,
  EFFECT_SUBJECTS,
  EFFECT_TRAIL_BASE,
  EFFECT_TRAIL_SAMPLES,
} from "./shaders/passes/composite"
import {
  buildParticleComputeShader,
  buildParticleRenderShader,
  parseParticleBlend,
  parseParticleBloom,
  parseParticleCount,
  particleEntryPoints,
  PARTICLE_STRIDE,
} from "./shaders/passes/particles"
import {
  SIM_FORMAT,
  SIM_MAX,
  buildSimShader,
  parseSimSize,
  simEntryPoint,
} from "./shaders/passes/sim"
import { buildTrailShader, trailEntryPoints, TRAIL_SUBDIVISIONS } from "./shaders/passes/trails"
import { PICK_SHADER_WGSL } from "./shaders/passes/pick"
import { MIPMAP_BLIT_SHADER_WGSL } from "./shaders/passes/mipmap"
import { compileGraph, type CompileOptions, type StyleSlot } from "./graph/compile"
import type { Diagnostic, ShaderGraph } from "./graph/schema"
import type { AlphaMode, RenderClass } from "./graph/render-class"
import type {
  ApplyStyleGroupResult,
  ApplyStyleGroupsResult,
  GroupDiagnostic,
  StyleGroup,
} from "./graph/style-group"
import { DEFAULT_GRAPH } from "./graph/presets/default"
import { FACE_GRAPH } from "./graph/presets/face"
import { HAIR_GRAPH } from "./graph/presets/hair"
import { BODY_GRAPH } from "./graph/presets/body"
import { EYE_GRAPH } from "./graph/presets/eye"
import { STOCKINGS_GRAPH } from "./graph/presets/stockings"
import { METAL_GRAPH } from "./graph/presets/metal"
import { CLOTH_SMOOTH_GRAPH } from "./graph/presets/cloth_smooth"
import { CLOTH_ROUGH_GRAPH } from "./graph/presets/cloth_rough"

// Material preset dispatch. Consumers supply a MaterialPresetMap assigning material names
// to presets; unmapped materials fall back to "default" (Principled BSDF).
export type MaterialPreset =
  | "default"
  | "face"
  | "hair"
  | "body"
  | "eye"
  | "stockings"
  | "metal"
  | "cloth_smooth"
  | "cloth_rough"

export type MaterialPresetMap = Partial<Record<MaterialPreset, string[]>>

// Substring hints mapping common PMX material names (JP/CN/EN) to a style category,
// tried when a material isn't in the caller's explicit override map. Ordered: more
// specific families first (靴下 must hit stockings before 靴 hits cloth). A material
// matching nothing resolves to null — it stays ungrouped (neutral default).
const PRESET_NAME_HINTS: Array<[MaterialPreset, string[]]> = [
  ["stockings", ["靴下", "ソックス", "タイツ", "ニーソ", "袜", "stocking", "socks", "tights"]],
  [
    "eye",
    ["白目", "目影", "二重", "睫", "まつげ", "まゆ", "眉", "目", "瞳", "眼", "eye", "iris", "pupil", "lash", "brow"],
  ],
  // face also catches mouth-interior parts (tongue / teeth / gums / oral cavity), which
  // share the face material family. Bare 口 is omitted — it collides with 袖口 (cuff).
  [
    "face",
    ["顔", "颜", "顏", "脸", "臉", "かお", "face", "舌", "tongue", "牙", "牙齿", "齿", "歯", "teeth", "tooth", "口腔", "口内", "mouth", "嘴", "唇", "歯茎", "gums"],
  ],
  // Simplified 发 is listed as compounds, never bare: it also writes 发光 (glow),
  // and hair carries a renderClass, so a chance hit puts an emissive panel in the
  // hair pass. Same reasoning as bare 口 being omitted from face above.
  [
    "hair",
    [
      "前髪", "後髪", "髪", "髮", "頭髪", "もみあげ", "アホ毛", "ヘア",
      "头发", "前发", "后发", "长发", "短发", "发丝", "刘海", "辫", "马尾",
      "hair", "ahoge", "bang",
    ],
  ],
  ["body", ["肌", "皮肤", "skin"]],
  ["metal", ["金属", "メタル", "metal", "earring", "耳环", "耳環"]],
  [
    "cloth_smooth",
    [
      "服",
      "衣",
      "裙",
      "裤",
      "スカート",
      "ワンピ",
      "リボン",
      "袖",
      "靴",
      "鞋",
      "帽",
      "体",
      "飾",
      "饰",
      "尾",
      "套", // 外套 (coat), 手套 (gloves)
      "腿", // 腿环 (leg ring/garter) and other leg-wear accessories
      "带", // straps and bands: 头带/发带/背带/腰带
      "绳", // ropes: 背绳/腰绳
      "纱", // gauze/veils: 头纱
      "巾", // kerchiefs/scarves: 头巾/领巾/围巾
      "布", // cloth panels: 肩布/腰布
      "背球", // back ornament sphere
      "腰花", // waist flower
      "花蕊", // flower pistil ornament
      "skirt",
      "dress",
      "ribbon",
      "sleeve",
      "shoes",
      "shirt",
      "short", // shorts
      "boot",
      "hat",
      "cloth",
      "accessor",
      "trigger",
    ],
  ],
]

// Resolve a material name to a style category (override map first, then name hints), or
// null if nothing matches — a null-resolving material stays ungrouped (neutral default).
function resolvePreset(materialName: string, map: MaterialPresetMap | undefined): MaterialPreset | null {
  if (map) {
    for (const [preset, names] of Object.entries(map)) {
      if (names && names.includes(materialName)) return preset as MaterialPreset
    }
  }
  const lower = materialName.toLowerCase()
  for (const [preset, hints] of PRESET_NAME_HINTS) {
    for (const hint of hints) {
      if (lower.includes(hint)) return preset
    }
  }
  return null
}

// Default-group recipe per style category: the shipped graph + its natural pass-integration
// (renderClass, alphaMode). This is the auto-default-groups mapping — the same
// category→integration knowledge the old fixed slots encoded, now producing editable groups.
const PRESET_GROUP_INFO: Partial<Record<MaterialPreset, { graph: ShaderGraph; renderClass: RenderClass; alphaMode: AlphaMode }>> = {
  default: { graph: DEFAULT_GRAPH, renderClass: "auto", alphaMode: "opaque" },
  face: { graph: FACE_GRAPH, renderClass: "auto", alphaMode: "opaque" },
  hair: { graph: HAIR_GRAPH, renderClass: "hair", alphaMode: "opaque" },
  body: { graph: BODY_GRAPH, renderClass: "auto", alphaMode: "opaque" },
  eye: { graph: EYE_GRAPH, renderClass: "eye", alphaMode: "opaque" },
  stockings: { graph: STOCKINGS_GRAPH, renderClass: "auto", alphaMode: "hashed" },
  metal: { graph: METAL_GRAPH, renderClass: "auto", alphaMode: "opaque" },
  cloth_smooth: { graph: CLOTH_SMOOTH_GRAPH, renderClass: "auto", alphaMode: "opaque" },
  cloth_rough: { graph: CLOTH_ROUGH_GRAPH, renderClass: "auto", alphaMode: "opaque" },
}

// Map a WGSL compile-error line back to the graph node whose `let` produced it —
// the compiler tags every generated line with a trailing `// @node:<id>` marker.
function nodeIdForWgslLine(wgsl: string, lineNum: number): string | undefined {
  const lines = wgsl.split("\n")
  for (let i = Math.min(lineNum - 1, lines.length - 1); i >= 0; i--) {
    const m = lines[i].match(/\/\/ @node:([a-z0-9_]+)/)
    if (m) return m[1]
  }
  return undefined
}

// A compiled + installed style group on a model: the swapped pipeline(s), the group's
// StyleUniforms buffer, the slider→UBO map setStyleParam consults, and the resolved
// render-class (draw-order + over-eyes participation). Keyed per-model by group id.
type GroupInstall = {
  group: StyleGroup
  renderClass: RenderClass
  alphaMode: AlphaMode
  pipeline: GPURenderPipeline
  /** Depth-write-off twin — dormant, kept for a future OIT path. */
  pipelineNoDepthWrite: GPURenderPipeline
  /** hair render-class only: the stencil-matched IS_OVER_EYES=true variant. */
  overEyesPipeline?: GPURenderPipeline
  uniformBuffer: GPUBuffer
  /** The group's own image maps, uploaded once per apply and owned here — the
   *  install destroys them when it is replaced, so a re-apply cannot leak. */
  images?: (GPUTexture | null)[]
  slotMap: StyleSlot[]
  /** Serialized (graph + renderClass + alphaMode) — lets applyStyleGroups skip recompiling
   *  an unchanged group. */
  signature: string
}

export type RaycastCallback = (
  modelName: string,
  material: string | null,
  bone: string | null,
  screenX: number,
  screenY: number,
) => void

/** Select a folder (webkitdirectory) and pass FileList or File[]; pmxFile picks which .pmx when several exist. */
export type LoadModelFromFilesOptions = {
  files: FileList | File[]
  pmxFile?: File
}

// Blender-style scene config. World = environment lighting (ambient);
// Sun = the single directional lamp; Camera = view framing.
export type WorldOptions = {
  /** Linear scene-referred color of the World Background (Blender: World > Surface > Color). */
  color?: Vec3
  /** Multiplier on world color (Blender: World > Surface > Strength). */
  strength?: number
}

/** A model's scene placement — root offset baked into skinning + visibility. Serializable
 *  into a scene descriptor via getModelTransform. */
/** How many character positions an effect can read (viewU[11..14]). Defined
 *  beside the shader that reads them — the layout arithmetic has to agree. */
const MAX_EFFECT_SUBJECTS = EFFECT_SUBJECTS
/** Where a character IS, for an effect that follows them. センター carries a
 *  motion's root movement — walking, jumping — where the model transform only
 *  carries where the model was placed; 全ての親 is the fallback for a model that
 *  animates the true root instead. */
const SUBJECT_BONES = ["センター", "全ての親"]

/** How many bones one effect may name. Eight is already a lot for one file, and
 *  this is a MINIMUM: raising it breaks nothing, because effects read through
 *  rzAnchor() rather than indexing the buffer. Lowering it would. */
const MAX_EFFECT_ANCHORS = EFFECT_ANCHORS

/** Only for the bounding sphere's height. */
const HEAD_BONE = "頭"

/** Path samples kept per trailed anchor. ~2.1s at the sampling rate below, which
 *  is a long ribbon — a dancer's arm draws most of a circle in that time.
 *
 *  A MINIMUM, like every cap here, and raising it is why that matters: effects
 *  read through rzTrail and loop to rzTrailCount, so this went 64 → 128 without
 *  touching a single published effect. Lowering it is the direction that breaks. */
const TRAIL_SAMPLES = EFFECT_TRAIL_SAMPLES
/** Sampled on the SCENE clock at a fixed rate, so a path is identical in the
 *  editor, in an export and in a re-export, and its spacing does not change with
 *  the display's refresh. */
const TRAIL_HZ = 60
const TRAIL_DT = 1 / TRAIL_HZ

/** vec4 slots: four subjects × 3, then anchors × four subjects × 3, then the
 *  trails — slot-major, four subjects each, TRAIL_SAMPLES apiece. */
const CAST_SUBJECT_VEC4S = MAX_EFFECT_SUBJECTS * 3
const CAST_ANCHOR_VEC4S = MAX_EFFECT_ANCHORS * MAX_EFFECT_SUBJECTS * 3
const CAST_TRAIL_BASE = EFFECT_TRAIL_BASE
const CAST_VEC4S = CAST_TRAIL_BASE + MAX_EFFECT_ANCHORS * MAX_EFFECT_SUBJECTS * TRAIL_SAMPLES

export type ModelTransform = {
  position: Vec3
  rotation: Quat
  /** Uniform scale (default 1). */
  scale: number
  visible: boolean
}

export type SunOptions = {
  /** Linear color of the sun lamp (Blender: Light > Color). */
  color?: Vec3
  /** Lamp power in Blender units (Blender: Light > Strength). */
  strength?: number
  /** Direction sunlight travels (points FROM sun TO scene, Blender: -light.rotation.Z). */
  direction?: Vec3
}

/** An effect param: number → f32, vector-like → vec3f (see setEffect).
 *  Structural {x,y,z} rather than the Vec3 class so JSON-derived values (a
 *  shared scene document's params) pass straight in. */
export type EffectParamValue = number | { x: number; y: number; z: number }
export type EffectResult = {
  ok: boolean
  /** Compile/validation errors, line:col relative to the USER's WGSL. */
  diagnostics: string[]
  /** Which mounts the WGSL declared — `fn background` / `fn foreground`. Both
   *  false only on a failed compile, since defining neither IS the failure. */
  mounts: { background: boolean; foreground: boolean }
}

export type CameraOptions = {
  /** Orbit distance from target. */
  distance?: number
  /** World-space orbit center. */
  target?: Vec3
  /** Vertical field of view in radians. */
  fov?: number
}

/** EEVEE Bloom panel (3D Viewport > Render > Bloom). Fields map 1:1 to Blender's UI. */
export type BloomOptions = {
  enabled: boolean
  threshold: number
  knee: number
  radius: number
  color: Vec3
  intensity: number
  clamp: number
}

export const DEFAULT_BLOOM_OPTIONS: BloomOptions = {
  enabled: true,
  threshold: 0.5,
  knee: 0.5,
  radius: 4.0,
  color: new Vec3(1.0, 0.7247558832168579, 0.6487361788749695),
  intensity: 0.05,
  clamp: 0.0,
}

/** Camera depth of field — a bokeh gather in the composite pass. Costs nothing
 *  while disabled: the scene pass discards its depth buffer and the composite
 *  branch never runs. Enabled, the pass stores depth and the gather reads it. */
export type DepthOfFieldOptions = {
  enabled: boolean
  /** "auto" focuses the first visible character each frame — the camera-space
   *  depth span of its bones sets both distance and a floor on range — so a
   *  dancer stays sharp without anyone touching a slider. "manual" uses
   *  focusDistance/focusRange as given. */
  focusMode: "auto" | "manual"
  /** Camera-space distance to the focus plane (MMD units). */
  focusDistance: number
  /** Depth band that stays perfectly sharp, centered on focusDistance. In auto
   *  mode this is a minimum — the band never cuts into the subject. */
  focusRange: number
  /** Blur strength scale; 1 is a natural lens, higher is dreamier. */
  aperture: number
  /** Largest blur circle, in device pixels. */
  maxBlurRadius: number
  /** Bokeh polygon blade count (3–12); 6 is the classic hexagon. */
  bladeCount: number
  /** Gather tap count: 8 / 16 / 24. */
  quality: "performance" | "balanced" | "cinematic"
}

export const DEFAULT_DEPTH_OF_FIELD_OPTIONS: DepthOfFieldOptions = {
  enabled: false,
  focusMode: "auto",
  focusDistance: 25,
  focusRange: 2,
  aperture: 1,
  maxBlurRadius: 18,
  bladeCount: 6,
  quality: "balanced",
}

/** Blender Color Management / View (rendering.txt: Filmic, exposure, gamma). `look` is reserved for future curve tweaks. */
export type ViewTransformOptions = {
  /** Stops applied before Filmic: `linear *= 2^exposure`. */
  exposure: number
  /** After Filmic, display gamma (`pow(rgb, 1/gamma)`). */
  gamma: number
  /**
   * Which display transform the frame is formed with.
   *
   * "standard" is Blender's Standard: the sRGB encoding and nothing else, which
   * is what NPR and anime work uses — the colours the graph computes are the
   * colours that land, with no film curve reinterpreting them. Both of the
   * reference Wuthering Waves projects render this way.
   *
   * "filmic" is Blender 3.6's Filmic, Medium High Contrast, baked as a LUT.
   */
  transform: "agx" | "filmic" | "standard"
}

// Matches the reference Blender project: Filmic view, Medium High Contrast look,
// exposure 0.3, gamma 1.0, sRGB display, no curves.
export const DEFAULT_VIEW_TRANSFORM: ViewTransformOptions = {
  exposure: 0.6,
  gamma: 1.0,
  transform: "filmic",
}

/** Color grading applied to the tonemapped scene (ASC CDL — see grade() in
 *  composite.ts). The three tonal controls are expressed as COLORS with
 *  mid-gray (0.5, 0.5, 0.5) as neutral: the direction from neutral is the hue
 *  you push toward, and the distance from neutral is the amount — so no
 *  separate strength slider is needed. Display-space sRGB, since grading runs
 *  after the view transform. */
export type ColorGradingOptions = {
  /** Lifts/tints the dark end (CDL offset). */
  shadows: Vec3
  /** Bends the midtones (CDL power) — brighter above neutral, darker below. */
  midtones: Vec3
  /** Scales/tints the bright end (CDL slope). */
  highlights: Vec3
  /** Contrast about the 0.5 display pivot. 1 = neutral. */
  contrast: number
  /** 1 = neutral, 0 = grayscale, >1 = punchier. */
  saturation: number
}

const NEUTRAL_GRADE_CHANNEL = 0.5
export const DEFAULT_COLOR_GRADING: ColorGradingOptions = {
  shadows: new Vec3(NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL),
  midtones: new Vec3(NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL),
  highlights: new Vec3(NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL),
  contrast: 1,
  saturation: 1,
}

export type GizmoDragKind = "rotate" | "translate"

export interface GizmoDragEvent {
  modelName: string
  boneName: string
  boneIndex: number
  kind: GizmoDragKind
  /** Computed target local rotation (for "rotate") / target local translation (for "translate"). */
  localRotation: Quat
  localTranslation: Vec3
  /** Drag start (mousedown) or end (mouseup). Undefined during drag moves. */
  phase?: "start" | "end"
}

/**
 * Gizmo drag callback. The engine does NOT write to the skeleton on its own —
 * it only computes the target local rotation / translation for the dragged bone
 * and fires this callback. The host decides how to apply it (e.g. call
 * `model.setBoneLocalRotation(boneIndex, localRotation)` for a runtime-only
 * edit, call `rotateBones({ [boneName]: localRotation }, 0)` for a tweened
 * write, or mutate an animation clip keyframe and re-seek).
 *
 * Fires once with phase="start" on mousedown, on every mousemove (no phase),
 * and once with phase="end" on mouseup.
 */
export type GizmoDragCallback = (event: GizmoDragEvent) => void

export type EngineOptions = {
  world?: WorldOptions
  sun?: SunOptions
  camera?: CameraOptions
  /** Canvas background (display-space sRGB 0–1), composited under the scene after
   *  tonemapping. Omit/null = transparent canvas (see setBackgroundColor). */
  background?: Vec3 | null
  /** Initial EEVEE-style bloom; tune at runtime with `setBloomOptions`. */
  bloom?: Partial<BloomOptions>
  /** View transform (exposure/gamma) applied in composite before/after Filmic. */
  view?: Partial<ViewTransformOptions>
  onRaycast?: RaycastCallback
  /** See {@link GizmoDragCallback}. */
  onGizmoDrag?: GizmoDragCallback
}

export const DEFAULT_ENGINE_OPTIONS = {
  world: { color: new Vec3(0.4014, 0.4944, 0.647), strength: 0.3 },
  sun: { color: new Vec3(1.0, 1.0, 1.0), strength: 2.0, direction: new Vec3(-0.0873, -0.3844, 0.919) },
  camera: { distance: 26.6, target: new Vec3(0, 12.5, 0), fov: Math.PI / 4 },
  onRaycast: undefined,
}

export interface EngineStats {
  fps: number // derived from mean frame interval — bounded by the real refresh rate
  frameTime: number // ms — mean frame interval (vsync-to-vsync), not CPU work time
  frameTimeMax: number // ms — worst frame interval in the window (hitch / stutter indicator)
  fps1PercentLow: number // "1% low" fps = 1000 / 99th-percentile frame interval
  jitter: number // ms — stddev of frame intervals (pacing evenness; high = janky at any mean fps)
  cpuAnimMs: number // ms/frame (EMA) — model updates: blending, IK, world matrices
  cpuPhysicsMs: number // ms/frame (EMA) — physics stepping across all instances
  cpuRenderMs: number // ms/frame (EMA) — the rest of the render thread: uniforms, encoding, submit
}

type DrawCallType = "opaque" | "transparent" | "ground" | "opaque-outline" | "transparent-outline"

interface DrawCall {
  type: DrawCallType
  count: number
  firstIndex: number
  bindGroup: GPUBindGroup
  materialName: string
  // Style group this material belongs to, or null (ungrouped → neutral base pipeline).
  // Outline/ground draw calls are never grouped and leave this null.
  groupId: string | null
  // Bindings 0–3 kept so the bind group can be rebuilt when the material's group changes
  // (binding 4 must follow the group's style buffer, or the zero buffer when ungrouped).
  // Present only for material draw calls (opaque/transparent) — the grouping walk skips
  // any draw call without it.
  baseBindGroupEntries?: GPUBindGroupEntry[]
  /** Material draws only: false = excluded from the shadow map (PMX cast-shadow
   *  flag off). Sheer texels are cut per fragment by the shadow pass's alpha test. */
  castsShadow?: boolean
  /** Edge-flagged materials: interleaved inverted-hull outline drawn right after
   *  this material with the outline pipeline. Shares this call's index range;
   *  own bind group (edge uniforms + diffuse texture for the alpha test). */
  outline?: { bindGroup: GPUBindGroup }
  /** MODEL-SPACE AABB over this material's index range, computed at load:
   *  [minX, minY, minZ, maxX, maxY, maxZ]. Usable for culling only while the
   *  owning model is rigid (see ModelInstance.rigid) — animation moves vertices
   *  out of it, which is why a skinned model culls per model instead. */
  bounds: Float32Array
  /** Slot in the cull metadata and indirect-argument buffers. Reassigned every
   *  time the flat draw list is rebuilt (model added/removed, draws re-sorted).
   *  -1 until the list is built. */
  cullIndex: number
}

/** One draw's place in the flat, scene-wide cull list. */
interface CullEntry {
  inst: ModelInstance
  draw: DrawCall
}

/** What getCullDiagnostics() reports: the GPU's answer, an independent CPU
 *  answer over the same source data, and every draw where they disagree. */
export interface CullDiagnostics {
  drawCount: number
  modelCount: number
  /** Draws the GPU compute left with instanceCount = 1. */
  cameraVisibleGpu: number
  shadowVisibleGpu: number
  /** The same test run on the CPU from the same bounds and frusta. */
  cameraVisibleCpu: number
  shadowVisibleCpu: number
  /** How each model was bounded this frame — the split that decides whether a
   *  stage culls per material or per model. */
  rigidModels: number
  skinnedModels: number
  mismatches: {
    model: string
    material: string
    pass: "camera" | "shadow"
    gpu: boolean
    cpu: boolean
  }[]
  /** What each model was actually tested against. Without this a report saying
   *  "everything visible" is unreadable — you cannot tell a working cull looking
   *  at geometry that genuinely fills the screen from a cull that never rejects
   *  anything. The radius against the camera distance answers it directly. */
  models: {
    name: string
    rigid: boolean
    visible: boolean
    draws: number
    /** How many of this model's draws survived the camera frustum. */
    cameraVisible: number
    /** How many survived the light frustum AND carry the PMX cast-shadow flag.
     *  Split from `casters` so a zero here is readable: no casters at all means
     *  the author turned shadows off, while casters with zero survivors means
     *  the light-frustum test rejected them. */
    shadowVisible: number
    /** Draws with the PMX cast-shadow flag set (bit 0x04), before any culling. */
    casters: number
    /** Sphere path only: the world centre and radius tested against. Null for a
     *  rigid model, whose draws each carry their own box instead. */
    sphere: [number, number, number, number] | null
  }[]
  /** Where the camera was, so the numbers above can be read against it. */
  camera: { eye: [number, number, number]; target: [number, number, number] }
}

interface PickDrawCall {
  count: number
  firstIndex: number
  bindGroup: GPUBindGroup
}

interface ModelInstance {
  name: string
  model: Model
  basePath: string
  assetReader: AssetReader
  gpuBuffers: GPUBuffer[]
  textureCacheKeys: string[]
  vertexBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  jointsBuffer: GPUBuffer
  weightsBuffer: GPUBuffer
  skinMatrixBuffer: GPUBuffer
  drawCalls: DrawCall[]
  shadowDrawCalls: DrawCall[]
  shadowBindGroup: GPUBindGroup
  mainPerInstanceBindGroup: GPUBindGroup
  pickPerInstanceBindGroup: GPUBindGroup
  pickDrawCalls: PickDrawCall[]
  /** Environment geometry added via addStage — no physics, no IK, and it
   *  suppresses the built-in ground. See addStage for why each of those. */
  isStage: boolean
  /** A pose pass ran since the last skin-matrix upload. Always true for cast
   *  members; false for an idle stage, which is the point. */
  skinMatricesDirty: boolean
  hiddenMaterials: Set<string>
  /** Materials a material morph has driven to zero alpha. Kept apart from
   *  hiddenMaterials so a morph switching a part off never clobbers the user's
   *  own visibility toggle, and vice versa. */
  morphHiddenMaterials: Set<string>
  /** Material-morph targets, or null when the model has no type-8 morphs. */
  materialMorphTargets: MaterialMorphTarget[] | null
  /** The same targets by PMX material index, so a named offset is one lookup. */
  materialMorphByIndex: Map<number, MaterialMorphTarget> | null
  physics: RezePhysics | null
  vertexBufferNeedsUpdate: boolean
  gpuMorph: GpuMorph | null
  // Style groups applied to this model: group id → compiled install.
  styleGroups: Map<string, GroupInstall>
  // Material name → group id (each material in ≤1 group). Drives draw-call assignment.
  materialToGroup: Map<string, string>
  // Per-group compile generation — an async compile finishing after a newer edit/remove
  // on the same id is discarded (stale-write guard).
  styleGroupGen: Map<string, number>
  // ── Cull bounds ──
  /** Slot in the per-model cull buffer, assigned when the draw list is rebuilt. */
  cullModelIndex: number
  /** Every bone shares one skin matrix (within tolerance) and no vertex morph can
   *  move a vertex out of its material's box — so the model is a rigid transform
   *  of its bind pose and its per-material AABBs are live. True for a stage, and
   *  for any character still in bind pose. Re-evaluated only on the frames the
   *  skin matrices are re-uploaded, which is never for an idle stage. */
  rigid: boolean
  /** The shared skin matrix, when rigid: model space → world, INCLUDING the
   *  scene placement (setModelTransform bakes the root into skinning). */
  rigidXform: Float32Array
  /** Bound on how far skinning can carry a vertex from the bone that drives it:
   *  max over vertices of max over influencing joints of |v − bindPos(joint)|,
   *  plus the largest single vertex-morph displacement. An AABB over the model's
   *  POSED bone positions grown by this contains every skinned vertex, because a
   *  skinned position is a convex combination of rigid images of v, each within
   *  that distance of its bone's posed position. */
  skinMargin: number
}

/**
 * One material a type-8 morph can reach, with the uniform block as it loaded.
 *
 * Material morphs are re-derived from base every time a weight changes rather
 * than accumulated, because weights go down as well as up and a running total
 * drifts. The buffer is already COPY_DST, so this is a writeBuffer, not a
 * rebuild.
 */
interface MaterialMorphTarget {
  /** Index into the PMX material array — what MaterialMorphOffset points at. */
  pmxIndex: number
  materialName: string
  buffer: GPUBuffer
  /** The 16-float MaterialUniforms block as createMaterialUniformBuffer wrote it. */
  base: Float32Array
  /** Scratch for the morphed block, so the per-change pass allocates nothing. */
  work: Float32Array
  /** What was last uploaded. `applyMorphs` marks weights dirty on every frame of
   *  any clip carrying morph tracks — i.e. every character with a face VMD — so
   *  without this the pass would re-upload byte-identical material blocks
   *  forever on behalf of a switch that never moves. */
  last: Float32Array
}

// Per-model GPU vertex-morph compute state. Present only for models with vertex morphs.
interface GpuMorph {
  bindGroup: GPUBindGroup
  weightsBuffer: GPUBuffer
  weightsData: Float32Array // staging copy uploaded when weights change
  workgroups: number
  dispatchNeeded: boolean
}

// ── Sheer-material detection ──────────────────────────────────────────────────
// PMX carries no "translucent" flag: a see-through veil usually has diffuse
// alpha 1.0 and does its transparency entirely in the TEXTURE's alpha channel.
// Classifying by material alpha alone put such cloth in the opaque bucket,
// where it draws in PMX order with depth writes — anything the engine draws
// after it (the hair render-class draws LAST for the eye-stencil effect) got
// depth-rejected behind the veil, so you saw the body through it but not the
// hair. These helpers measure a material's real coverage by sampling the
// texture's alpha at the material's own triangle CENTROIDS — centroids, not
// vertices, because hair-card corners sit in transparent texture margins and
// vertex sampling would misclassify hair (which must stay opaque-bucket for
// stencil interplay and shadows).

/** Downsampled alpha plane of a decoded texture (≤128², nearest-sampled). */
function buildAlphaSampler(
  source: ImageBitmap | null,
  rgba: Uint8Array | null,
  width: number,
  height: number,
): { a: Uint8ClampedArray; w: number; h: number } | null {
  try {
    const w = Math.max(1, Math.min(128, width))
    const h = Math.max(1, Math.min(128, height))
    const canvas = new OffscreenCanvas(w, h)
    const cx = canvas.getContext("2d", { willReadFrequently: true })
    if (!cx) return null
    if (source) {
      cx.drawImage(source, 0, 0, w, h)
    } else if (rgba) {
      const tmp = new OffscreenCanvas(width, height)
      const tcx = tmp.getContext("2d")
      if (!tcx) return null
      tcx.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0)
      cx.drawImage(tmp, 0, 0, w, h)
    } else {
      return null
    }
    const img = cx.getImageData(0, 0, w, h).data
    const a = new Uint8ClampedArray(w * h)
    for (let i = 0; i < w * h; i++) a[i] = img[i * 4 + 3]
    return { a, w, h }
  } catch {
    return null
  }
}

/** Texture-alpha statistics over ≤400 of the material's triangle centroids:
 *  `avg` (0..1) and `translucentFrac` — the fraction of samples that are
 *  neither fully opaque nor fully cut out (alpha in ~0.03..0.97). Together
 *   Bucketing itself is binary (babylon-mmd parity): ANY translucent coverage
 *  routes to the alpha-blend bucket. `avg` below this threshold additionally
 *  marks a material as fully sheer (a veil). */
const SHEER_ALPHA_THRESHOLD = 0.7
function materialAlphaStats(
  verts: Float32Array,
  indices: Uint32Array,
  firstIndex: number,
  count: number,
  sampler: { a: Uint8ClampedArray; w: number; h: number } | null | undefined,
): { avg: number; translucentFrac: number } {
  if (!sampler) return { avg: 1, translucentFrac: 0 }
  const triCount = Math.floor(count / 3)
  if (triCount === 0) return { avg: 1, translucentFrac: 0 }
  const step = Math.max(1, Math.floor(triCount / 400))
  let sum = 0
  let translucent = 0
  let n = 0
  for (let t = 0; t < triCount; t += step) {
    const i0 = indices[firstIndex + t * 3]
    const i1 = indices[firstIndex + t * 3 + 1]
    const i2 = indices[firstIndex + t * 3 + 2]
    const u = (verts[i0 * 8 + 6] + verts[i1 * 8 + 6] + verts[i2 * 8 + 6]) / 3
    const v = (verts[i0 * 8 + 7] + verts[i1 * 8 + 7] + verts[i2 * 8 + 7]) / 3
    // Wrap (MMD UVs may tile), then nearest-sample the downsampled plane.
    const x = Math.min(sampler.w - 1, Math.max(0, Math.floor((u - Math.floor(u)) * sampler.w)))
    const y = Math.min(sampler.h - 1, Math.max(0, Math.floor((v - Math.floor(v)) * sampler.h)))
    const a = sampler.a[y * sampler.w + x]
    sum += a
    if (a > 8 && a < 247) translucent++
    n++
  }
  if (n === 0) return { avg: 1, translucentFrac: 0 }
  return { avg: sum / n / 255, translucentFrac: translucent / n }
}

// ── Cull bounds ───────────────────────────────────────────────────────────────

/** World units added to every side of a material's model-space AABB.
 *
 *  Three things reach past the vertices the box was measured from, and one
 *  number covers all of them because all three are small next to an MMD model's
 *  ~20-unit height:
 *   · the inverted-hull outline, which shares the material's index range and
 *     extrudes along the normal;
 *   · the tolerance RIGID_BONE_EPS allows on "every bone shares one matrix",
 *     which lets a vertex sit up to about a hundredth of a unit off where the
 *     box says it is;
 *   · fp32 rounding through the skinning multiply.
 *  A tenth of a unit is roughly a fingernail on a character and invisible on a
 *  stage, so nothing is lost by being generous here — a box too small drops
 *  geometry that should have drawn, which is the only failure that shows. */
const CULL_BOUNDS_SLACK = 0.1

/** How far two bones' skin matrices may differ and still count as "the same
 *  transform". A stage at bind pose computes world × inverseBind per bone
 *  numerically, so the products are identity only to fp32 — bit equality would
 *  reject every stage there is. Linear terms are unitless; the translation term
 *  is in world units and carries the looser bound because it accumulates the
 *  bind position's own magnitude. */
const RIGID_LINEAR_EPS = 1e-4
const RIGID_TRANSLATION_EPS = 1e-2

/** Model-space AABB over one material's index range, as
 *  [minX, minY, minZ, maxX, maxY, maxZ]. Walks the material's own indices, so a
 *  200-material stage costs one pass over its index buffer in total.
 *
 *  Empty ranges collapse to a zero box, which the frustum test then rejects from
 *  every direction — correct, because there is nothing to draw. */
function materialBounds(verts: Float32Array, indices: Uint32Array, firstIndex: number, count: number): Float32Array {
  const b = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity])
  for (let i = 0; i < count; i++) {
    const v = indices[firstIndex + i] * 8 // VERTEX_STRIDE — position at +0
    const x = verts[v]
    const y = verts[v + 1]
    const z = verts[v + 2]
    if (x < b[0]) b[0] = x
    if (y < b[1]) b[1] = y
    if (z < b[2]) b[2] = z
    if (x > b[3]) b[3] = x
    if (y > b[4]) b[4] = y
    if (z > b[5]) b[5] = z
  }
  if (b[0] > b[3]) b.fill(0)
  return b
}

/** Bind-pose world position of every bone, from the inverse-bind matrices:
 *  for invBind = [R | t] (column-major), the bind position is −Rᵀt. Read from
 *  the matrices rather than from Bone.bindTranslation because that field is
 *  parent-relative, and this needs model space. */
function boneBindPositions(invBind: Float32Array, boneCount: number): Float32Array {
  const out = new Float32Array(boneCount * 3)
  for (let i = 0; i < boneCount; i++) {
    const o = i * 16
    const tx = invBind[o + 12]
    const ty = invBind[o + 13]
    const tz = invBind[o + 14]
    out[i * 3] = -(invBind[o] * tx + invBind[o + 1] * ty + invBind[o + 2] * tz)
    out[i * 3 + 1] = -(invBind[o + 4] * tx + invBind[o + 5] * ty + invBind[o + 6] * tz)
    out[i * 3 + 2] = -(invBind[o + 8] * tx + invBind[o + 9] * ty + invBind[o + 10] * tz)
  }
  return out
}

/**
 * How far skinning can carry a vertex away from the bones that drive it.
 *
 * A skinned position is p = Σ wᵢ·(Mᵢv), and each Mᵢ is rigid (MMD bones do not
 * scale), so Mᵢv = pᵢ + Rᵢ(v − bᵢ) where pᵢ is bone i's posed position and bᵢ
 * its bind position. That splits p into a convex combination of the pᵢ — which
 * is inside their AABB — plus Σ wᵢ·Rᵢ(v − bᵢ), whose length is at most
 * Σ wᵢ·|v − bᵢ|. So an AABB over the posed bone positions, grown by the largest
 * such WEIGHTED SUM over all vertices, contains the whole mesh in any pose. The
 * per-model sphere is therefore derived, not guessed.
 *
 * The weighting is what makes it usable, and it took measuring real models to
 * see why. Bounding by max|v − bᵢ| over the influencing bones is also valid, and
 * on five MMD models it produced spheres 1.5–2.3× the model's own radius —
 * mostly air, and a character that never culls. The cause is always the same: a
 * stray 1/255 weight tying some vertex to a control bone parked at the origin
 * (全ての親, 操作中心, a glasses bone). Such a weight moves the vertex by
 * millimetres and must be charged for millimetres, which the weighted form does
 * and the max form does not. Weighted, the same models land at 1.35–1.45×.
 *
 * Weights are renormalized here exactly as the vertex shader renormalizes them,
 * including its fallback to joint 0 at full weight when they sum to zero — a
 * bound has to describe the vertex the shader will actually place.
 */
function computeSkinMargin(
  verts: Float32Array,
  joints: Uint16Array,
  weights: Uint8Array,
  bindPos: Float32Array,
  boneCount: number,
): number {
  const vertexCount = Math.floor(verts.length / 8)
  let worst = 0
  for (let v = 0; v < vertexCount; v++) {
    const p = v * 8
    const x = verts[p]
    const y = verts[p + 1]
    const z = verts[p + 2]
    const j = v * 4
    const sum = weights[j] + weights[j + 1] + weights[j + 2] + weights[j + 3]
    let reach = 0
    for (let k = 0; k < 4; k++) {
      const w = sum > 0 ? weights[j + k] / sum : k === 0 ? 1 : 0
      if (w === 0) continue
      const b = joints[j + k]
      if (b >= boneCount) continue
      const dx = x - bindPos[b * 3]
      const dy = y - bindPos[b * 3 + 1]
      const dz = z - bindPos[b * 3 + 2]
      reach += w * Math.sqrt(dx * dx + dy * dy + dz * dz)
    }
    if (reach > worst) worst = reach
  }
  return worst
}

/** Reused by writeCullSphere — one sphere centre per model per frame is not
 *  worth an allocation. */
const cullScratchVec = new Vec3(0, 0, 0)

/**
 * Does every bone share one skin matrix, within tolerance?
 *
 * That is exactly the condition under which per-material model-space AABBs are
 * live: the whole mesh is one rigid transform of its bind pose, and the shared
 * matrix IS the transform to apply. Stated this way it needs no bind-pose
 * reasoning and no "is this a stage" flag — a stage passes, and so does a
 * character that has not been given a motion yet, which then culls per material
 * for free.
 *
 * Bone 0 is the reference. The early exit is what makes this cheap on a
 * character: the first animated bone disagrees, so the common case reads about
 * sixteen floats and stops.
 */
function skinMatricesAgree(m: Float32Array, boneCount: number): boolean {
  for (let i = 1; i < boneCount; i++) {
    const o = i * 16
    for (let k = 0; k < 16; k++) {
      // Indices 12–15 are the translation column, in world units; the rest are
      // the unitless linear part.
      const eps = k >= 12 ? RIGID_TRANSLATION_EPS : RIGID_LINEAR_EPS
      if (Math.abs(m[o + k] - m[k]) > eps) return false
    }
  }
  return true
}

/**
 * Six inward frustum planes from a column-major view-projection, normalized,
 * written as vec4(nx, ny, nz, d) at `at`. Inside is `dot(n, p) + d >= 0`.
 *
 * Gribb–Hartmann. The near plane is row 2 ALONE, not row3 + row2 as the OpenGL
 * form in most references has it, and that is the one line to be careful about
 * here — Mat4.perspectiveInto writes the OpenGL matrix, whose z lands in
 * [-1, 1], while WebGPU clips at z >= 0. The two together put the real near
 * plane at twice the camera's nominal near (verify: with near 0.1, ndc z crosses
 * zero at 0.2), and row 2 is the plane that sits there. So this extracts the
 * boundary the RASTERIZER enforces rather than the one the matrix was written
 * for, which is the only one culling may agree with.
 */
function writeFrustumPlanes(vp: Float32Array, out: Float32Array, at: number): void {
  // row_i of the matrix, from column-major storage.
  const r = (i: number, c: number) => vp[c * 4 + i]
  const set = (slot: number, x: number, y: number, z: number, w: number) => {
    const len = Math.hypot(x, y, z) || 1
    const o = at + slot * 4
    out[o] = x / len
    out[o + 1] = y / len
    out[o + 2] = z / len
    out[o + 3] = w / len
  }
  set(0, r(3, 0) + r(0, 0), r(3, 1) + r(0, 1), r(3, 2) + r(0, 2), r(3, 3) + r(0, 3)) // left
  set(1, r(3, 0) - r(0, 0), r(3, 1) - r(0, 1), r(3, 2) - r(0, 2), r(3, 3) - r(0, 3)) // right
  set(2, r(3, 0) + r(1, 0), r(3, 1) + r(1, 1), r(3, 2) + r(1, 2), r(3, 3) + r(1, 3)) // bottom
  set(3, r(3, 0) - r(1, 0), r(3, 1) - r(1, 1), r(3, 2) - r(1, 2), r(3, 3) - r(1, 3)) // top
  set(4, r(2, 0), r(2, 1), r(2, 2), r(2, 3)) // near — z >= 0, not z >= -w
  set(5, r(3, 0) - r(2, 0), r(3, 1) - r(2, 1), r(3, 2) - r(2, 2), r(3, 3) - r(2, 3)) // far
}

/** CPU mirror of the compute's AABB test — the projected-extent form, so a box
 *  is rejected only when every corner is behind one plane. */
function aabbInsideFrustum(
  planes: Float32Array,
  base: number,
  cx: number,
  cy: number,
  cz: number,
  ex: number,
  ey: number,
  ez: number,
): boolean {
  for (let i = 0; i < 6; i++) {
    const o = base + i * 4
    const nx = planes[o]
    const ny = planes[o + 1]
    const nz = planes[o + 2]
    const d = nx * cx + ny * cy + nz * cz + planes[o + 3]
    const reach = Math.abs(nx) * ex + Math.abs(ny) * ey + Math.abs(nz) * ez
    if (d + reach < 0) return false
  }
  return true
}

/** CPU mirror of the compute's sphere test. */
function sphereInsideFrustum(
  planes: Float32Array,
  base: number,
  x: number,
  y: number,
  z: number,
  r: number,
): boolean {
  for (let i = 0; i < 6; i++) {
    const o = base + i * 4
    if (planes[o] * x + planes[o + 1] * y + planes[o + 2] * z + planes[o + 3] + r < 0) return false
  }
  return true
}

/** The largest single vertex-morph displacement in a model. Charged to both
 *  bound kinds as slack. One morph, not the sum of all of them: several at full
 *  weight could in principle stack past it, but face morphs are millimetres on a
 *  twenty-unit model and summing fifty of them would inflate every box in the
 *  scene to pay for a case that does not occur. */
function vertexMorphReach(model: Model): number {
  let worstSq = 0
  for (const morph of model.getMorphing().morphs) {
    if (morph.type !== 1) continue
    for (const off of morph.vertexOffsets) {
      const [ox, oy, oz] = off.positionOffset
      const d = ox * ox + oy * oy + oz * oz
      if (d > worstSq) worstSq = d
    }
  }
  return Math.sqrt(worstSq)
}

/** Tried in order when a PMX names a texture without an extension. */
const TEXTURE_EXTENSION_GUESSES = [".png", ".jpg", ".jpeg", ".bmp", ".tga", ".dds", ".spa", ".sph"]

export class Engine {
  private static instance: Engine | null = null

  static getInstance(): Engine {
    if (!Engine.instance) {
      throw new Error("Engine not ready: create Engine, await init(), then load models via engine.loadModel().")
    }
    return Engine.instance
  }

  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  // No `!`: the constructor assigns it, so the type is the guarantee. Every other
  // `!` field here is genuinely absent until init() — this one no longer is.
  private camera: Camera
  private cameraUniformBuffer!: GPUBuffer
  private cameraMatrixData = new Float32Array(36)
  // Blender-style scene config groups (resolved from EngineOptions)
  private world!: { color: Vec3; strength: number }
  private sun!: { color: Vec3; strength: number; direction: Vec3 }
  private cameraConfig!: { distance: number; target: Vec3; fov: number }
  private lightUniformBuffer!: GPUBuffer
  private lightData = new Float32Array(64)
  private lightCount = 0
  private resizeObserver: ResizeObserver | null = null
  private resizePending = false
  private depthTexture!: GPUTexture
  // The one base shading model: ungrouped materials render this (compiled DEFAULT_GRAPH).
  // Grouped materials use their group's own compiled pipeline.
  private neutralPipeline!: GPURenderPipeline
  private neutralPipelineNoDepthWrite!: GPURenderPipeline
  private transparentDepthPrepassPipeline!: GPURenderPipeline
  // ── Style group runtime ──
  // Shared 256 B zero StyleUniforms buffer (group(2) binding(4)) bound by every ungrouped
  // material; grouped materials rebind to their group's own buffer (per-model, in the
  // ModelInstance's styleGroups map). See docs/style-groups-spec.md §6.
  private zeroStyleBuffer!: GPUBuffer
  // Stashed at createPipelines so group pipelines can be compiled later.
  private mainPipelineLayout!: GPUPipelineLayout
  private sceneTargets!: GPUColorTargetState[]
  private fullVertexBufferLayouts!: GPUVertexBufferLayout[]
  // 1×64 vertical ramp for shared-toon materials: lit (top) → soft shadow
  // tone (bottom). Stand-in for MMD's toon01–10.bmp, which we can't ship.
  private defaultToonRampTexture!: GPUTexture
  private groundShadowPipeline!: GPURenderPipeline
  private groundShadowBindGroupLayout!: GPUBindGroupLayout
  private outlinePipeline!: GPURenderPipeline
  private selectedMaterial: { modelName: string; materialName: string } | null = null
  private selectionMaskTexture?: GPUTexture
  private selectionMaskView?: GPUTextureView
  private selectionMaskPipeline!: GPURenderPipeline
  private selectionMaskPassDescriptor!: GPURenderPassDescriptor
  private selectionEdgePipeline!: GPURenderPipeline
  private selectionEdgeBindGroupLayout!: GPUBindGroupLayout
  private selectionEdgeBindGroup?: GPUBindGroup
  private selectionEdgeUniformBuffer!: GPUBuffer
  private selectionEdgePassDescriptor!: GPURenderPassDescriptor
  private selectionSampler!: GPUSampler

  // ─── Transform gizmo ───────────────────────────────────────────────
  private selectedBone: { modelName: string; boneName: string; boneIndex: number } | null = null
  private gizmoVertexBuffer!: GPUBuffer
  private gizmoTransformBuffer!: GPUBuffer
  private gizmoPipeline!: GPURenderPipeline
  private gizmoBindGroup0!: GPUBindGroup
  private gizmoColorBindGroups: GPUBindGroup[] = []
  private gizmoPassDescriptor!: GPURenderPassDescriptor
  private static readonly GIZMO_RING_SEGMENTS = 96
  private static readonly GIZMO_RING_RADIUS = 0.8
  // Axis visible length (relative to gizmo size). Extends past ring radius so
  // the "arrow stub" sticking out of the ring is a comfortable click target.
  private static readonly GIZMO_AXIS_LENGTH = 1.25
  // Draw ranges derived from GIZMO_RING_SEGMENTS at init (setupGizmo) so the
  // segment-count constant is the single source of truth. Axes: 3 × 6 = 18
  // verts; each ring: SEG × 6 verts.
  private gizmoDraws!: { first: number; count: number; color: number }[]
  private static readonly GIZMO_WORLD_SIZE = 1.5
  private static readonly GIZMO_THICKNESS_PX = 15.0
  private static readonly GIZMO_PICK_THRESHOLD_PX = 17.0

  // Drag state — set on mousedown if the pointer is over a gizmo handle; cleared
  // on mouseup. While non-null, the camera is locked and mousemove/up are routed
  // to the drag handler. All vectors/quats stored are in world / local frames as
  // indicated; we snapshot "initial" values on drag start so the drag is driven
  // by mouse-delta relative to the click point (not cumulative frame-to-frame).
  private gizmoDrag: {
    kind: "axis" | "ring"
    axis: 0 | 1 | 2 // local-axis index: 0 = X, 1 = Y, 2 = Z (bone-local)
    bonePos: Vec3 // gizmo world origin at drag start
    worldAxis: Vec3 // snapshot of the local axis rotated into world at drag start
    // Ring drag: in-plane basis vectors (world) perpendicular to worldAxis.
    basisU: Vec3
    basisV: Vec3
    initialLocalRot: Quat
    initialLocalTrans: Vec3
    parentWorldRot: Quat // parent bone's world rotation (identity if no parent)
    parentWorldRotInv: Quat
    initialAngle: number
    initialAxisParam: number
  } | null = null
  private mainPerFrameBindGroupLayout!: GPUBindGroupLayout
  private mainPerInstanceBindGroupLayout!: GPUBindGroupLayout
  private mainPerMaterialBindGroupLayout!: GPUBindGroupLayout
  private outlinePerFrameBindGroupLayout!: GPUBindGroupLayout
  private outlinePerMaterialBindGroupLayout!: GPUBindGroupLayout
  private perFrameBindGroup!: GPUBindGroup
  private outlinePerFrameBindGroup!: GPUBindGroup
  private multisampleTexture!: GPUTexture
  private hdrResolveTexture!: GPUTexture
  private static readonly MULTISAMPLE_COUNT = 4
  // HDR intermediate format. rg11b10ufloat when the adapter exposes the
  // `rg11b10ufloat-renderable` feature (Chrome + Safari on Apple Silicon both
  // do), else fall back to rgba16float.
  //
  // Why it matters — Apple TBDR tile memory: rgba16float is 8 bytes/texel, so
  // 4× MSAA is 32 bytes/texel and does not fit Apple Silicon's tile memory at
  // useful tile sizes. The driver then stores the full MSAA buffer to system
  // memory every frame and resolves from there — ~300 MB/frame of extra
  // bandwidth at 1920×1200 DPR=2, which is the dominant frame-pacing hit on
  // Safari (visibly: shrinking the window made Safari smooth; Chrome was
  // always smooth because Dawn apparently amortizes it). rg11b10ufloat at
  // 4 bytes/texel → 16 bytes/texel at 4× MSAA → fits tile memory like
  // rgba8unorm does, resolves in-tile, no system-memory round-trip. No alpha
  // channel (the HDR path never needed one — alpha blending reads src.a from
  // the fragment shader and treats missing dst.a as 1, so the blend math is
  // unchanged).
  private hdrFormat: GPUTextureFormat = "rgba16float"
  /** Stencil value stamped by eye draws so hair can stencil-test against it and
   *  alpha-blend a second pass over eye silhouette pixels (see-through-hair effect). */
  private static readonly STENCIL_EYE_VALUE = 1
  /** Aux MRT alongside HDR color. Two channels:
   *   .r — bloom mask (1 = model geometry, 0 = ground; sampled by bloom blit to gate prefilter).
   *   .g — accumulated alpha (the channel that used to live in hdr.a before the HDR format
   *        switched to rg11b10ufloat, which has no alpha). Sampled by composite/bloom to
   *        un-premultiply color for tonemap and to produce the canvas-drawable alpha used by
   *        the premultiplied alphaMode compositor (so the page background still shows through
   *        cleared / edge-faded regions like before).
   *  rg8unorm at 4× MSAA is 8 bytes/texel — still fits Apple TBDR tile memory comfortably. */
  private static readonly BLOOM_MASK_FORMAT: GPUTextureFormat = "rg8unorm"
  private multisampleMaskTexture!: GPUTexture
  private maskResolveTexture!: GPUTexture
  private maskResolveView!: GPUTextureView
  /**
   * The installed effect's particle system, or null when it declared none.
   *
   * A fixed pool: the count is chosen at install and the slots recycle, so there
   * is no allocation and no spawn-rate bookkeeping in the hot path. Dead slots
   * cost a degenerate quad the rasteriser rejects, which is cheaper than the
   * prefix sum and readback a compacted draw list would need every frame.
   */
  private particles: {
    count: number
    buffer: GPUBuffer
    uniform: GPUBuffer
    data: Float32Array
    counts: Uint32Array
    compute: GPUComputePipeline
    computeLayout: GPUBindGroupLayout
    computeBind: GPUBindGroup
    render: GPURenderPipeline
    renderLayout: GPUBindGroupLayout
    renderBind: GPUBindGroup
    rebind: () => { computeBind: GPUBindGroup; renderBind: GPUBindGroup }
  } | null = null
  /** Ceiling for `// @particles`. Past this an author is asking for a stall. */
  private static readonly MAX_PARTICLES = 65536
  private particleFrame = 0
  /**
   * The installed effect's persistent grid, or null when it declared none.
   *
   * Two textures, not one, and read/write alternate between them every frame:
   * a shader cannot coherently read and write the same texture, so this is not
   * an optimisation but the only correct shape. `parity` says which one holds
   * the CURRENT grid — the one everything else samples.
   */
  private sim: {
    size: number
    textures: [GPUTexture, GPUTexture]
    /** Sampled views, for reading. */
    read: [GPUTextureView, GPUTextureView]
    pipeline: GPUComputePipeline
    layout: GPUBindGroupLayout
    /** Bind groups per parity: binds[i] reads textures[i], writes the other. */
    binds: [GPUBindGroup, GPUBindGroup]
    uniform: GPUBuffer
    data: Float32Array
    parity: number
    frame: number
  } | null = null
  private simSampler!: GPUSampler
  private simFallbackView!: GPUTextureView
  /**
   * The installed effect's ribbons, or null when it declared none.
   *
   * No buffer of its own: it reads the very same path history the field-based
   * ribbon read through rzTrail, so a trail costs one draw and nothing recorded.
   */
  private trails: {
    instances: number
    uniform: GPUBuffer
    data: Float32Array
    pipeline: GPURenderPipeline
    layout: GPUBindGroupLayout
    bind: GPUBindGroup
  } | null = null
  /** The ribbons' own offscreen target — max-blended, composited after tone map. */
  private trailLayerTexture: GPUTexture | null = null
  private trailLayerView: GPUTextureView | null = null
  /** 1×1 transparent stand-in so the composite layout binds with no trails installed. */
  private trailFallbackView!: GPUTextureView
  /** The field layer: user background/foreground mounts at half resolution. */
  private fieldBgTexture: GPUTexture | null = null
  private fieldBgView: GPUTextureView | null = null
  private fieldFgTexture: GPUTexture | null = null
  private fieldFgView: GPUTextureView | null = null
  private fieldUniformBuffer!: GPUBuffer
  /** 2 = half resolution (the default); 1 = full, for effects that declare
   *  `// @fullres` because they draw sub-pixel detail no upsample can carry. */
  private fieldScale = 2
  private fieldFullW = 0
  private fieldFullH = 0
  private fieldPipeline: GPURenderPipeline | null = null
  private fieldBindGroupLayout!: GPUBindGroupLayout
  private fieldBindGroups: [GPUBindGroup, GPUBindGroup] | null = null
  private fieldPipelineLayout!: GPUPipelineLayout
  /**
   * The audio analysis buffer every effect module binds: header
   * [frames, bands, secondsPerFrame, audioTime], then [level, band0..bandN-1]
   * per frame. Precomputed by the host for the whole track — never a live
   * analyser, which would render silence during an export. Falls back to four
   * zeroes (frames = 0) so layouts always bind.
   */
  private audioBuffer!: GPUBuffer
  private audioFallbackBuffer!: GPUBuffer
  private audioTimeScratch = new Float32Array(2)
  private renderPassDescriptor!: GPURenderPassDescriptor
  private compositePassDescriptor!: GPURenderPassDescriptor
  // Two specialized composite pipelines via WGSL pipeline-override constants.
  // Identity variant skips the gamma pow entirely at shader-compile time —
  // Safari's Metal backend won't fold pow(x, 1) to identity.
  private compositePipelineIdentity!: GPURenderPipeline
  private compositePipelineGamma!: GPURenderPipeline
  private morphComputePipeline!: GPUComputePipeline
  private morphComputeBindGroupLayout!: GPUBindGroupLayout
  // ── GPU frustum cull (see shaders/passes/cull.ts) ──
  // The compute runs every frame and writes indirect draw arguments. Nothing
  // consumes them yet: the draw path still issues direct draws, and this
  // increment exists so the culling DATA can be validated against a working app
  // before the draw path changes. setCullApply(true) gates the direct draws on
  // the CPU mirror of the same test, which is how a wrong bound is made visible.
  /** Null once the pipeline has failed to compile — the pass then does nothing
   *  rather than invalidating every command buffer it touches. */
  private cullPipeline: GPUComputePipeline | null = null
  private cullBindGroupLayout!: GPUBindGroupLayout
  private cullBindGroup: GPUBindGroup | null = null
  /** Every material draw in the scene, flat, in the order the passes walk them.
   *  A draw's position here IS its slot in every cull buffer. */
  private cullDraws: CullEntry[] = []
  private cullModels: ModelInstance[] = []
  /** Structure changed — model added or removed, draws re-sorted. NOT set by
   *  animation, physics or camera movement, which is the whole point. */
  private cullListDirty = true
  private cullMetaBuffer: GPUBuffer | null = null
  private cullModelBuffer: GPUBuffer | null = null
  private cullCameraArgs: GPUBuffer | null = null
  private cullShadowArgs: GPUBuffer | null = null
  private cullFrustaBuffer: GPUBuffer | null = null
  private cullFrustaBytes = new ArrayBuffer(208)
  private cullFrustaF32 = new Float32Array(this.cullFrustaBytes)
  private cullFrustaU32 = new Uint32Array(this.cullFrustaBytes)
  /** CPU-side mirrors of what was uploaded, so the reference test reads exactly
   *  the same numbers the compute did. */
  private cullMetaBytes = new ArrayBuffer(0)
  private cullMetaF32 = new Float32Array(0)
  private cullMetaU32 = new Uint32Array(0)
  private cullModelData = new Float32Array(0)
  private cullModelFlags = new Uint32Array(0)
  /** Per draw: bit0 = passes the camera frustum, bit1 = passes the light frustum
   *  and casts. Filled by the CPU reference; only computed when something asks. */
  private cullReference = new Uint8Array(0)
  private cullReferenceFrame = -1
  private cullApply = false
  private cullFrame = 0
  private cullScratchVp = new Float32Array(16)
  private cullReadback: { camera: GPUBuffer; shadow: GPUBuffer; bytes: number } | null = null
  private cullReadbackInFlight = false
  private compositeBindGroupLayout!: GPUBindGroupLayout
  private compositeBindGroup!: GPUBindGroup
  private depthOfField: DepthOfFieldOptions = { ...DEFAULT_DEPTH_OF_FIELD_OPTIONS }
  private dofUniformBuffer!: GPUBuffer
  private dofUniformData = new Float32Array(12)
  private dofFocusScratch = new Vec3(0, 0, 0)
  /** Depth-only view of the scene's MSAA depth buffer, read by the DoF gather. */
  private depthReadView: GPUTextureView | null = null
  private compositeUniformBuffer!: GPUBuffer
  // [exposure, invGamma, _, _,  bloomTint.x, bloomTint.y, bloomTint.z, bloomIntensity]
  // 11 × vec4f — see the viewU comment in composite.ts. The last one is the
  // camera's world position, which is what lets a foreground effect turn the
  // depth it is handed into a PLACE (bgWorldPos) rather than a distance.
  private readonly compositeUniformData = new Float32Array(60)
  /** Composite background (display-space sRGB 0–1) — null = transparent canvas. */
  private backgroundColor: Vec3 | null = null
  // 360 backdrop (equirectangular skybox, sampled by view ray in composite).
  private backdropEquirectTexture: GPUTexture | null = null
  private backdropEquirectView: GPUTextureView | null = null
  private fallbackEquirectTexture!: GPUTexture
  private fallbackEquirectView!: GPUTextureView
  // The scene's user WGSL effect (setEffect). ONE per scene, mounted under the
  // scene, over it, or both — whichever of background()/foreground() the code
  // defines. The composite pipelines are REBUILT with the user code injected;
  // params live in their own uniform buffer so setEffectParam is a write, not a
  // recompile (the same instant tier as setStyleParam).
  private effect: {
    wgsl: string
    paramLayout: Map<string, { offset: number; comps: 1 | 3 }>
    paramsBuffer: GPUBuffer | null
    paramsData: Float32Array<ArrayBuffer>
    /** Mounted under the scene. */
    hasBackground: boolean
    /** Mounted over the finished frame — and the reason the scene pass has to
     *  STORE its depth, which it otherwise discards into tile memory. */
    hasForeground: boolean
    /** Bones the source asked for, in declaration order — rzAnchor's slots. Only
     *  these are resolved and uploaded, so a file that names none costs nothing. */
    anchors: { bone: string; trail: boolean }[]
  } | null = null
  /** The cast, as the effect API sees it. Written per frame while an effect is
   *  installed, and only up to what that effect actually declared. */
  private castBuffer!: GPUBuffer
  private castData!: Float32Array<ArrayBuffer>
  /** Last frame's anchor world positions, for velocity. Keyed model id → slot. */
  private anchorPrev = new Map<string, Float32Array>()
  private castLastMs = 0
  /** Recent path per trailed anchor, keyed "model\0slot". Newest first, so the
   *  shader's index 0 is now — written by unshifting rather than by tracking a
   *  head, because 64 is short and the alternative is an index the GPU side
   *  would also have to know about. */
  private anchorTrail = new Map<string, { pos: number[]; t: number[] }>()
  /** Scene seconds, advanced by the frame delta — NOT wall time, so an offline
   *  export samples the same path the editor showed. */
  private sceneClock = 0
  private trailAccum = 0
  /** Trail samples owed this frame, computed once so every trail on every
   *  character samples in lockstep and their paths stay comparable. */
  private trailDue = 0
  private agxLutTexture: GPUTexture | null = null
  private agxFallbackTexture!: GPUTexture
  /** Bound at composite binding 7 when no effect (or a param-less one) is set. */
  private bgParamsDummyBuffer!: GPUBuffer
  private compositePipelineLayout!: GPUPipelineLayout
  /** time=0 origin for the active effect — reset each setEffect. */
  /** Scene-clock reading when the current effect was installed. The effect's
   *  `time` is measured from here — see where it is written. */
  private effectEpochScene = 0
  private compositeBloomView: GPUTextureView | null = null

  // EEVEE-style bloom pyramid (mirrors Blender 3.6 effect_bloom_frag.glsl):
  //   blit (HDR → half-res, 4-tap Karis + soft threshold/knee)
  //   N-1 downsamples (13-tap Jimenez/COD box filter, 5 group averages)
  //   N-1 upsamples (9-tap tent, additively combined with corresponding downsample mip)
  //   composite adds bloomUp mip 0 × (color × intensity) to HDR before Filmic.
  // Matches EEVEE energy: tint/intensity applied at composite, not prefilter.
  private bloomSampler!: GPUSampler
  private bloomBlitUniformBuffer!: GPUBuffer
  private bloomUpsampleUniformBuffer!: GPUBuffer
  private readonly bloomBlitUniformData = new Float32Array(4)
  private readonly bloomUpsampleUniformData = new Float32Array(4)
  private bloomBlitPipeline!: GPURenderPipeline
  private bloomDownsamplePipeline!: GPURenderPipeline
  private bloomUpsamplePipeline!: GPURenderPipeline
  private bloomBlitBindGroupLayout!: GPUBindGroupLayout
  private bloomDownsampleBindGroupLayout!: GPUBindGroupLayout
  private bloomUpsampleBindGroupLayout!: GPUBindGroupLayout
  private bloomDownTexture!: GPUTexture
  private bloomUpTexture!: GPUTexture
  private bloomMipCount = 0
  private bloomDownMipViews: GPUTextureView[] = []
  private bloomUpMipViews: GPUTextureView[] = []
  private bloomBlitBindGroup!: GPUBindGroup
  private bloomDownsampleBindGroups: GPUBindGroup[] = []
  private bloomUpsampleBindGroups: GPUBindGroup[] = []
  /** Single-attachment pass; colorAttachments[0].view set per bloom step. */
  private bloomPassDescriptor!: GPURenderPassDescriptor
  private static readonly BLOOM_MAX_LEVELS = 5

  // Ground properties (shadow only)
  private groundVertexBuffer?: GPUBuffer
  private groundIndexBuffer?: GPUBuffer
  private hasGround = false
  private shadowMapTexture!: GPUTexture
  private shadowMapDepthView!: GPUTextureView
  private brdfLutTexture!: GPUTexture
  private brdfLutView!: GPUTextureView
  private filmicLutTexture!: GPUTexture
  private filmicLutView!: GPUTextureView
  // Width of the baked Filmic tone LUT (composite.ts FILMIC_LUT_W must match).
  private static readonly FILMIC_LUT_WIDTH = 256
  // 4096² over the 64-unit light box ≈ 64 texels/world-unit — crisp contact
  // shadows on the ground catcher (2048 read visibly blurry). ~64 MB depth,
  // acceptable for WebGPU-class hardware; deliberately NOT user-configurable.
  private static readonly SHADOW_MAP_SIZE = 4096
  private shadowDepthPipeline!: GPURenderPipeline
  private shadowLightVPBuffer!: GPUBuffer
  private shadowLightVPMatrix = new Float32Array(16)
  private groundShadowBindGroup?: GPUBindGroup
  private shadowComparisonSampler!: GPUSampler
  private groundShadowMaterialBuffer?: GPUBuffer
  private groundDrawCall: DrawCall | null = null

  private onRaycast?: RaycastCallback
  private onGizmoDrag?: GizmoDragCallback
  private lastTouchTime = 0
  private readonly DOUBLE_TAP_DELAY = 300
  // GPU picking
  private pickPipeline!: GPURenderPipeline
  private pickPerFrameBindGroupLayout!: GPUBindGroupLayout
  private pickPerInstanceBindGroupLayout!: GPUBindGroupLayout
  private pickPerMaterialBindGroupLayout!: GPUBindGroupLayout
  private pickPerFrameBindGroup!: GPUBindGroup
  private pickTexture!: GPUTexture
  private pickDepthTexture!: GPUTexture
  private pickReadbackBuffer!: GPUBuffer
  private pendingPick: { x: number; y: number } | null = null

  private modelInstances = new Map<string, ModelInstance>()
  private materialSampler!: GPUSampler
  private fallbackMaterialTexture!: GPUTexture
  private textureCache = new Map<string, GPUTexture>()
  // Downsampled CPU alpha channel per texture (≤128², ~16KB) — kept so materials
  // can be classified as SHEER at load by sampling alpha at their own UVs (the
  // GPU texture can't be read back cheaply, and PMX diffuse alpha is usually 1.0
  // even for see-through cloth: the translucency lives in the texture).
  private textureAlphaCache = new Map<string, { a: Uint8ClampedArray; w: number; h: number } | null>()
  private mipBlitPipeline: GPURenderPipeline | null = null
  private mipBlitSampler: GPUSampler | null = null
  private _nextDefaultModelId = 0

  // IK and physics enabled at engine level (same for all models)
  private ikEnabled = true
  private physicsEnabled = true
  // World-wide, not per-model: MMD treats gravity and wind as properties of the
  // scene, and a model loaded later must arrive into the same air as the rest.
  private gravity = new Vec3(0, -98, 0)
  private wind: WindOptions | null = null
  // GPU vertex-morph path. Set false BEFORE loadModel to fall back to the CPU path (A/B).
  private useGpuMorphs = true

  // VMD camera track (a dedicated camera VMD). When loaded + enabled it drives the shot,
  // sampled off the animated model's clock so it stays synced to the dance.
  private cameraAnimation: CameraAnimation | null = null

  // Camera target binding (Babylon/Three style: camera follows model)
  private cameraTargetModel: Model | null = null
  private cameraTargetBoneName = "全ての親"
  private cameraFollowSmoothing = 0
  private cameraFollowSeeded = false
  private readonly cameraFollowPos = new Vec3(0, 0, 0)
  private cameraTargetOffset: Vec3 = new Vec3(0, 0, 0)

  private lastFrameTime = performance.now()
  // Smoothness metrics are computed over a ring buffer of true frame intervals
  // (vsync-to-vsync), recomputed at STATS_REFRESH_MS so the readout doesn't flicker.
  private static readonly STATS_WINDOW = 120
  private static readonly STATS_REFRESH_MS = 500
  private frameIntervals = new Float32Array(Engine.STATS_WINDOW)
  private frameIntervalWrite = 0
  private frameIntervalFilled = 0
  private lastStatsCompute = performance.now()
  private stats: EngineStats = {
    fps: 0,
    frameTime: 0,
    frameTimeMax: 0,
    fps1PercentLow: 0,
    cpuAnimMs: 0,
    cpuPhysicsMs: 0,
    cpuRenderMs: 0,
    jitter: 0,
  }
  private animationFrameId: number | null = null
  private renderLoopCallback: (() => void) | null = null
  private bloomSettings!: BloomOptions
  private viewTransform!: ViewTransformOptions

  constructor(canvas: HTMLCanvasElement, options?: EngineOptions) {
    this.canvas = canvas
    const d = DEFAULT_ENGINE_OPTIONS
    this.world = {
      color: options?.world?.color ?? d.world.color,
      strength: options?.world?.strength ?? d.world.strength,
    }
    this.sun = {
      color: options?.sun?.color ?? d.sun.color,
      strength: options?.sun?.strength ?? d.sun.strength,
      direction: options?.sun?.direction ?? d.sun.direction,
    }
    this.cameraConfig = {
      distance: options?.camera?.distance ?? d.camera.distance,
      target: options?.camera?.target ?? d.camera.target,
      fov: options?.camera?.fov ?? d.camera.fov,
    }
    // Built HERE and not in setupCamera, because a host holds the Engine before
    // init() resolves — the reference is assigned, then init is awaited — and it
    // reads the camera in that window. isCameraVmdEnabled() on a camera that did
    // not exist yet threw "Cannot read properties of undefined (reading
    // 'vmdDriven')", which surfaces as the whole page failing to load. The Camera
    // is pure math, so nothing about it needed the device; only its aspect and
    // its input listeners do, and those still wait for a sized canvas.
    this.camera = new Camera(
      Math.PI,
      Math.PI / 2.5,
      this.cameraConfig.distance,
      this.cameraConfig.target,
      this.cameraConfig.fov,
    )
    this.onRaycast = options?.onRaycast
    this.onGizmoDrag = options?.onGizmoDrag
    this.bloomSettings = Engine.mergeBloomDefaults(options?.bloom)
    this.viewTransform = Engine.mergeViewTransformDefaults(options?.view)
    const bg = options?.background
    this.backgroundColor = bg ? new Vec3(bg.x, bg.y, bg.z) : null
  }

  /** Merge partial bloom with EEVEE defaults (same as constructor). */
  static mergeBloomDefaults(partial?: Partial<BloomOptions>): BloomOptions {
    const d = DEFAULT_BLOOM_OPTIONS
    const c = partial?.color
    return {
      enabled: partial?.enabled ?? d.enabled,
      threshold: partial?.threshold ?? d.threshold,
      knee: partial?.knee ?? d.knee,
      radius: partial?.radius ?? d.radius,
      color: c ? new Vec3(c.x, c.y, c.z) : new Vec3(d.color.x, d.color.y, d.color.z),
      intensity: partial?.intensity ?? d.intensity,
      clamp: partial?.clamp ?? d.clamp,
    }
  }

  static mergeViewTransformDefaults(partial?: Partial<ViewTransformOptions>): ViewTransformOptions {
    const d = DEFAULT_VIEW_TRANSFORM
    return {
      exposure: partial?.exposure ?? d.exposure,
      gamma: partial?.gamma ?? d.gamma,
      transform: partial?.transform ?? d.transform,
    }
  }

  /** Current bloom settings (Blender names; tint is a copied `Vec3`). */
  getBloomOptions(): BloomOptions {
    const b = this.bloomSettings
    return {
      enabled: b.enabled,
      threshold: b.threshold,
      knee: b.knee,
      radius: b.radius,
      color: new Vec3(b.color.x, b.color.y, b.color.z),
      intensity: b.intensity,
      clamp: b.clamp,
    }
  }

  getViewTransformOptions(): ViewTransformOptions {
    const v = this.viewTransform
    return { exposure: v.exposure, gamma: v.gamma, transform: v.transform }
  }

  private colorGrading: ColorGradingOptions = {
    shadows: new Vec3(NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL),
    midtones: new Vec3(NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL),
    highlights: new Vec3(NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL, NEUTRAL_GRADE_CHANNEL),
    contrast: DEFAULT_COLOR_GRADING.contrast,
    saturation: DEFAULT_COLOR_GRADING.saturation,
  }

  /**
   * Color-grade the tonemapped scene (ASC CDL slope/offset/power + saturation).
   * The background layer is deliberately left ungraded — see the call site in
   * composite.ts. Uniforms-only: no pipeline rebuild, safe to call per frame
   * (e.g. from a slider drag).
   */
  setColorGrading(patch: Partial<ColorGradingOptions>): void {
    const g = this.colorGrading
    if (patch.shadows) g.shadows = new Vec3(patch.shadows.x, patch.shadows.y, patch.shadows.z)
    if (patch.midtones) g.midtones = new Vec3(patch.midtones.x, patch.midtones.y, patch.midtones.z)
    if (patch.highlights) g.highlights = new Vec3(patch.highlights.x, patch.highlights.y, patch.highlights.z)
    if (patch.contrast !== undefined) g.contrast = patch.contrast
    if (patch.saturation !== undefined) g.saturation = patch.saturation
    if (this.device && this.compositeUniformBuffer) this.writeCompositeViewUniforms()
  }

  /** Current grade (for serialization into a scene descriptor). */
  getColorGrading(): ColorGradingOptions {
    const g = this.colorGrading
    return {
      shadows: new Vec3(g.shadows.x, g.shadows.y, g.shadows.z),
      midtones: new Vec3(g.midtones.x, g.midtones.y, g.midtones.z),
      highlights: new Vec3(g.highlights.x, g.highlights.y, g.highlights.z),
      contrast: g.contrast,
      saturation: g.saturation,
    }
  }

  setViewTransformOptions(patch: Partial<ViewTransformOptions>): void {
    const v = this.viewTransform
    if (patch.exposure !== undefined) v.exposure = patch.exposure
    if (patch.gamma !== undefined) v.gamma = patch.gamma
    if (patch.transform !== undefined) v.transform = patch.transform
    if (this.device && this.compositeUniformBuffer) {
      this.writeCompositeViewUniforms()
    }
  }

  private writeCompositeViewUniforms(): void {
    const v = this.viewTransform
    const b = this.bloomSettings
    const effIntensity = b.enabled ? b.intensity : 0.0
    const u = this.compositeUniformData
    u[0] = v.exposure
    // Store 1/gamma so the shader avoids a per-pixel divide. Safari's Metal
    // compiler doesn't fold `pow(x, 1/g)` into identity when g=1, so also emit
    // a uniform branch that skips the pow entirely in the common case.
    u[1] = 1.0 / Math.max(v.gamma, 1e-4)
    u[2] = 0.0
    u[3] = 0.0
    u[4] = b.color.x
    u[5] = b.color.y
    u[6] = b.color.z
    u[7] = effIntensity
    // Background composited UNDER the scene in display space (post-tonemap), so it
    // matches a CSS color of the same value exactly. Mode (u[11]): 0 = transparent
    // (DOM shows), 1 = solid color, 2 = 360 equirect (sampled by view ray; the
    // camera basis at u[12..23] is refreshed per frame by updateCameraUniforms).
    const bg = this.backgroundColor
    u[8] = bg?.x ?? 0
    u[9] = bg?.y ?? 0
    u[10] = bg?.z ?? 0
    // Base-layer mode only. A user effect is a separate LAYER over whichever
    // base is active, and needs no flag of its own: the composite pipeline is
    // rebuilt per effect, so the compiled variant IS the flag.
    u[11] = this.backdropEquirectView ? 2 : bg ? 1 : 0
    // Which display transform forms the frame (see viewTransform in composite.ts).
    u[25] = v.transform === "agx" ? 2 : v.transform === "standard" ? 1 : 0
    u[26] = this.canvas.width
    u[27] = this.canvas.height
    // ── Grade (viewU[7..9]) ── The UI's three tonal COLORS map to ASC CDL here,
    // on the CPU, so the shader only ever sees slope/offset/power. Mid-gray is
    // neutral in all three; the signed distance from it is the amount.
    const g = this.colorGrading
    const off = (c: number) => (c - NEUTRAL_GRADE_CHANNEL) * 0.5 // ±0.25 lift
    // power < 1 brightens, so midtones ABOVE neutral must lower the exponent.
    const pow_ = (c: number) => Math.max(0.05, 1 - (c - NEUTRAL_GRADE_CHANNEL) * 1.5)
    const slope = (c: number) => Math.max(0, 1 + (c - NEUTRAL_GRADE_CHANNEL) * 1.5)
    u[28] = off(g.shadows.x)
    u[29] = off(g.shadows.y)
    u[30] = off(g.shadows.z)
    u[31] = g.contrast
    u[32] = pow_(g.midtones.x)
    u[33] = pow_(g.midtones.y)
    u[34] = pow_(g.midtones.z)
    u[35] = g.saturation
    u[36] = slope(g.highlights.x)
    u[37] = slope(g.highlights.y)
    u[38] = slope(g.highlights.z)
    // Neutral grade → flag off, so the default pipeline pays nothing per pixel.
    const neutral =
      u[28] === 0 && u[29] === 0 && u[30] === 0 &&
      u[32] === 1 && u[33] === 1 && u[34] === 1 &&
      u[36] === 1 && u[37] === 1 && u[38] === 1 &&
      g.contrast === 1 && g.saturation === 1
    u[39] = neutral ? 0 : 1
    this.device.queue.writeBuffer(this.compositeUniformBuffer, 0, u)
  }

  /**
   * Set the canvas background color (display-space sRGB, 0–1 per channel — the
   * same value a CSS background of that color shows, applied after tonemapping).
   * Pass null for a transparent canvas (the page/DOM shows through — e.g. when a
   * backdrop image layer sits behind the canvas). Applies on the next frame.
   * A 360 backdrop (setBackdropEquirect) takes precedence while set.
   */
  setBackgroundColor(color: Vec3 | null): void {
    this.backgroundColor = color ? new Vec3(color.x, color.y, color.z) : null
    if (this.device && this.compositeUniformBuffer) this.writeCompositeViewUniforms()
  }

  /** Debug/diagnostic: skip every inverted-hull outline draw. */
  // OFF by default — the product aesthetic. Modern high-detail models read
  // better without hulls (babylon-mmd's own demos disable its outline renderer
  // too), and no hull pass means no depth-tie edge cases against near-coplanar
  // cloth. The full MMD-faithful machinery (interleaved per-material hulls,
  // texture-alpha-modulated rims) stays in place behind setOutlineEnabled(true).
  private outlineEnabled = false
  setOutlineEnabled(on: boolean): void {
    this.outlineEnabled = on
  }

  private rebuildCompositeBindGroup(): void {
    if (!this.device || !this.hdrResolveTexture || !this.compositeBloomView || !this.depthReadView) return
    if (!this.castBuffer) return
    this.compositeBindGroup = this.device.createBindGroup({
      label: "composite bind group",
      layout: this.compositeBindGroupLayout,
      entries: [
        { binding: 0, resource: this.hdrResolveTexture.createView() },
        { binding: 1, resource: this.compositeBloomView },
        { binding: 2, resource: this.bloomSampler },
        { binding: 3, resource: { buffer: this.compositeUniformBuffer } },
        { binding: 4, resource: this.maskResolveView },
        { binding: 5, resource: this.filmicLutView },
        { binding: 6, resource: this.backdropEquirectView ?? this.fallbackEquirectView },
        { binding: 7, resource: { buffer: this.effect?.paramsBuffer ?? this.bgParamsDummyBuffer } },
        { binding: 8, resource: this.depthReadView },
        { binding: 9, resource: { buffer: this.dofUniformBuffer } },
        { binding: 10, resource: (this.agxLutTexture ?? this.agxFallbackTexture).createView({ dimension: "3d" }) },
        { binding: 11, resource: { buffer: this.castBuffer } },
        { binding: 12, resource: this.trails && this.trailLayerView ? this.trailLayerView : this.trailFallbackView },
        { binding: 13, resource: { buffer: this.audioBuffer } },
        { binding: 15, resource: this.fieldBgView ?? this.trailFallbackView },
        { binding: 16, resource: this.fieldFgView ?? this.trailFallbackView },
      ],
    })
    this.rebuildFieldBindGroup()
  }

  private createFieldTargets(): void {
    if (!this.device || this.fieldFullW === 0) return
    const w = Math.max(1, Math.ceil(this.fieldFullW / this.fieldScale))
    const h = Math.max(1, Math.ceil(this.fieldFullH / this.fieldScale))
    this.fieldBgTexture?.destroy()
    this.fieldFgTexture?.destroy()
    this.fieldBgTexture = this.device.createTexture({
      label: "field layer (background)",
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.fieldFgTexture = this.device.createTexture({
      label: "field layer (foreground)",
      size: [w, h],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.fieldBgView = this.fieldBgTexture.createView()
    this.fieldFgView = this.fieldFgTexture.createView()
    this.device.queue.writeBuffer(this.fieldUniformBuffer, 0, new Float32Array([w, h, this.fieldFullW, this.fieldFullH]))
  }

  /**
   * ONE PER GRID PARITY.
   *
   * The sim alternates which texture holds the current grid, so the field pass
   * needs a bind group for each — built once here rather than rebuilt every
   * frame, which is what a single group would force and is pure waste for a
   * change that only ever toggles between two known states.
   */
  private rebuildFieldBindGroup(): void {
    if (!this.device || !this.depthReadView || !this.fieldUniformBuffer) return
    // Captured, so the null guard above survives into the closure.
    const depth = this.depthReadView
    const build = (grid: GPUTextureView) =>
      this.device.createBindGroup({
        label: "field layer bind group",
        layout: this.fieldBindGroupLayout,
        entries: [
          { binding: 3, resource: { buffer: this.compositeUniformBuffer } },
          { binding: 7, resource: { buffer: this.effect?.paramsBuffer ?? this.bgParamsDummyBuffer } },
          { binding: 8, resource: depth },
          { binding: 9, resource: { buffer: this.dofUniformBuffer } },
          { binding: 11, resource: { buffer: this.castBuffer } },
          { binding: 13, resource: { buffer: this.audioBuffer } },
          { binding: 14, resource: { buffer: this.fieldUniformBuffer } },
          { binding: 17, resource: grid },
          { binding: 18, resource: this.simSampler },
        ],
      })
    const sim = this.sim
    this.fieldBindGroups = sim
      ? [build(sim.read[0]), build(sim.read[1])]
      : [build(this.simFallbackView), build(this.simFallbackView)]
  }

  /**
   * Set a 360° backdrop from an equirectangular (2:1) image — a PhotoDome-style
   * skybox at infinity, sampled per-pixel by view direction so it follows the
   * camera. Display-only: composited in display space behind the scene, it never
   * affects lighting, bloom, or tonemapping. Pass null to remove (the background
   * color, or transparency, takes over again).
   */
  setBackdropEquirect(source: ImageBitmap | HTMLImageElement | HTMLCanvasElement | null): void {
    this.backdropEquirectTexture?.destroy()
    this.backdropEquirectTexture = null
    this.backdropEquirectView = null
    if (source && this.device) {
      let width = Math.max(1, "naturalWidth" in source ? source.naturalWidth : source.width)
      let height = Math.max(1, "naturalHeight" in source ? source.naturalHeight : source.height)
      let upload: ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas = source
      // Panoramas routinely exceed maxTextureDimension2D (e.g. 10000×5000 vs the
      // default 8192) — quietly downscale to fit rather than surfacing an error.
      const limit = this.device.limits.maxTextureDimension2D
      if (width > limit || height > limit) {
        const scale = Math.min(limit / width, limit / height)
        const w = Math.max(1, Math.floor(width * scale))
        const h = Math.max(1, Math.floor(height * scale))
        const canvas = new OffscreenCanvas(w, h)
        const cx = canvas.getContext("2d")!
        cx.imageSmoothingQuality = "high"
        cx.drawImage(source, 0, 0, w, h)
        upload = canvas
        width = w
        height = h
      }
      const tex = this.device.createTexture({
        label: "backdrop equirect",
        size: [width, height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.device.queue.copyExternalImageToTexture({ source: upload }, { texture: tex }, [width, height])
      this.backdropEquirectTexture = tex
      this.backdropEquirectView = tex.createView()
    }
    this.rebuildCompositeBindGroup()
    if (this.device && this.compositeUniformBuffer) this.writeCompositeViewUniforms()
  }

  private makeCompositePipeline(module: GPUShaderModule, applyGamma: boolean, label: string): GPURenderPipeline {
    return this.device.createRenderPipeline({
      label,
      layout: this.compositePipelineLayout,
      vertex: { module, entryPoint: "vs" },
      fragment: {
        module,
        entryPoint: "fs",
        constants: { APPLY_GAMMA: applyGamma ? 1 : 0 },
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    })
  }

  /**
   * Install the scene's WGSL effect (shadertoy-style), rendered per-pixel in the
   * composite pass. ONE effect per scene, and the code says where it mounts by
   * which of these it defines — either, or both in one file:
   *
   *     fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f
   *     fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f
   *
   * `background` is a LAYER between the base background and the scene,
   * over-composited onto whichever base is active (solid color, 360 equirect, or
   * transparency) — its alpha lets the base show through, so a starfield is
   * stars over the user's background color. `foreground` composites over the
   * finished frame instead, which is where rain, snow, petals and fog live, and
   * is handed `depth`: the camera-space distance in metres of whatever the scene
   * drew at that pixel (the far plane where it drew nothing). Compare a
   * particle's own distance against it and the model occludes it; fog just reads
   * it, since fog's alpha IS a function of distance.
   *
   * `ray` is the pixel's normalized world-space view direction (LH, +Z forward —
   * what the skybox samples by), `uv` is 0..1 bottom-left origin, `time` is
   * seconds since apply, and `bgResolution()` gives the canvas size. Return sRGB
   * + alpha; alpha is the only "how much does this replace" control there is.
   * Declared `params` arrive as `params.<name>` (number → f32, Vec3 → vec3f),
   * shared by both mounts, and are later tweaked without recompiling via
   * setEffectParam.
   *
   * Both mounts are display-space: neither affects lighting, bloom or
   * tonemapping, and both are captured by offline export. A foreground makes the
   * scene pass STORE its depth buffer (it otherwise discards it into tile
   * memory) for as long as one is installed.
   *
   * Compiles off the hot path (async pipelines): on failure the previous effect
   * is KEPT and diagnostics are returned with line numbers relative to the
   * user's WGSL. Pass null to remove the effect.
   */
  async setEffect(wgsl: string | null, params?: Record<string, EffectParamValue>): Promise<EffectResult> {
    const noMounts = { background: false, foreground: false }
    if (!this.device) return { ok: false, diagnostics: ["setEffect requires init() to have run"], mounts: noMounts }

    if (wgsl === null) {
      this.effect?.paramsBuffer?.destroy()
      this.effect = null
      this.releaseParticles()
      this.releaseTrails()
      this.releaseSim()
      this.fieldPipeline = null
      const module = this.device.createShaderModule({ label: "composite shader", code: buildCompositeShader(null) })
      this.compositePipelineIdentity = this.makeCompositePipeline(module, false, "composite pipeline (gamma=1)")
      this.compositePipelineGamma = this.makeCompositePipeline(module, true, "composite pipeline (gamma!=1)")
      this.rebuildCompositeBindGroup()
      this.writeCompositeViewUniforms()
      return { ok: true, diagnostics: [], mounts: noMounts }
    }

    // ── Which mounts did the author ask for? A declaration, not a setting: the
    // entry points present in the source are the ones compiled in. Matching the
    // `fn` keyword is enough to be safe against a `foreground` LOCAL or a call
    // to one — those never follow `fn`.
    const hasBackground = /\bfn\s+background\s*\(/.test(wgsl)
    const hasForeground = /\bfn\s+foreground\s*\(/.test(wgsl)
    // Particles are a THIRD mount, declared the same way — by the functions the
    // source defines. All three are required together: a pool with no shader to
    // draw it, or a draw with nothing spawning into it, is a silent blank rather
    // than an error, which is the worst way for an effect to fail.
    const pe = particleEntryPoints(wgsl)
    const wantsParticles = pe.init || pe.step || pe.shade
    const te = trailEntryPoints(wgsl)
    const wantsTrails = te.width || te.shade
    if (wantsTrails && !(te.width && te.shade)) {
      return {
        ok: false,
        diagnostics: [
          `a ribbon effect needs both fn trailWidth(u: f32, age: f32) -> f32 and ` +
            `fn trailShade(u: f32, v: f32, age: f32, weight: f32, slot: i32) -> vec4f`,
        ],
        mounts: noMounts,
      }
    }
    if (wantsParticles && !(pe.init && pe.step && pe.shade)) {
      const missing = [
        pe.init ? null : "fn particleInit(id: u32, seed: f32) -> Particle",
        pe.step ? null : "fn particleStep(p: Particle, dt: f32) -> Particle",
        pe.shade ? null : "fn particleShade(p: Particle, uv: vec2f) -> vec4f",
      ].filter(Boolean)
      return { ok: false, diagnostics: [`a particle effect also needs ${missing.join(" and ")}`], mounts: noMounts }
    }
    // One file, one kind — for now.
    //
    // The two kinds compile into different modules: field functions belong to the
    // composite pass, particle functions to the particle pair. A file holding both
    // would have to be spliced into both, and each module would then need the
    // OTHER's scaffolding (the Particle struct in the composite; the composite's
    // uniforms in the particle stages) for the dead half to compile — several
    // declarations that exist only so unused code type-checks, and a handful of
    // accessors that would silently return zero on the wrong side. Splitting into
    // two effects costs the author nothing once a scene can hold a list, and this
    // says so plainly instead of failing with "unresolved type Particle" from a
    // pass they did not know they were compiling into.
    if ((wantsParticles || wantsTrails) && (hasBackground || hasForeground)) {
      return {
        ok: false,
        diagnostics: [
          "an effect declares field mounts (background/foreground) or particles, not both — " +
            "split them into two effects",
        ],
        mounts: noMounts,
      }
    }
    if (!hasBackground && !hasForeground && !wantsParticles && !wantsTrails) {
      return {
        ok: false,
        diagnostics: [
          "an effect must define fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f, " +
            "fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f, " +
            "the particle trio (particleInit/particleStep/particleShade), " +
            "or the ribbon pair (trailWidth/trailShade)",
        ],
        mounts: noMounts,
      }
    }
    const mounts = { background: hasBackground, foreground: hasForeground }

    // ── Which bones did the author ask for? Same idea as the mounts above: a
    // declaration in the source, not a setting somewhere else. Only what is
    // named here gets resolved and uploaded, so naming none costs nothing and
    // naming eight costs eight — rather than every rig's 500 bones costing
    // everybody. Past the cap the extras are dropped rather than silently
    // shifting every slot after them.
    const anchors = parseEffectAnchors(wgsl, MAX_EFFECT_ANCHORS)

    // ── Params: codegen a WGSL struct and mirror its uniform layout on the CPU.
    // Fields are emitted in declaration order; offsets follow WGSL's natural
    // uniform rules (f32 align 4, vec3f align 16 size 12), computed identically
    // on both sides so no reordering is needed.
    const entries = Object.entries(params ?? {})
    const layout = new Map<string, { offset: number; comps: 1 | 3 }>()
    const fields: string[] = []
    let cursor = 0
    for (const [name, value] of entries) {
      if (!/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
        return { ok: false, diagnostics: [`invalid param name "${name}" (must be a WGSL identifier)`], mounts }
      }
      const isVec = typeof value !== "number"
      const align = isVec ? 16 : 4
      const offset = Math.ceil(cursor / align) * align
      layout.set(name, { offset: offset / 4, comps: isVec ? 3 : 1 })
      fields.push(`  ${name}: ${isVec ? "vec3f" : "f32"},`)
      cursor = offset + (isVec ? 12 : 4)
    }
    const paramsData = new Float32Array(Math.max(4, Math.ceil(cursor / 16) * 4))
    for (const [name, value] of entries) {
      const slot = layout.get(name)!
      if (typeof value === "number") paramsData[slot.offset] = value
      else {
        paramsData[slot.offset] = value.x
        paramsData[slot.offset + 1] = value.y
        paramsData[slot.offset + 2] = value.z
      }
    }
    const paramsDecl = entries.length
      ? `struct EffectParams {\n${fields.join("\n")}\n}\n@group(0) @binding(7) var<uniform> params: EffectParams;\n`
      : ""

    // ── Compile with validation captured, not thrown at the console. Line
    // numbers in diagnostics are rebased to the USER's source.
    // The composite is STATIC: user field code compiles in its own half-res
    // module (buildFieldShader), so a bad effect can no longer produce errors at
    // line numbers in a shader the author never wrote — and installing one no
    // longer recompiles the composite's tone-mapping half at all.
    const simSize = simEntryPoint(wgsl) ? parseSimSize(wgsl, SIM_MAX) || 256 : 0
    const fieldEffect =
      hasBackground || hasForeground ? { wgsl, paramsDecl, hasBackground, hasForeground, simSize } : null
    const source = buildCompositeShader(fieldEffect)
    this.device.pushErrorScope("validation")
    const module = this.device.createShaderModule({ label: "composite shader (effect)", code: source })
    const scopeErr = await this.device.popErrorScope()
    if (scopeErr) return { ok: false, diagnostics: [scopeErr.message], mounts }

    let fieldPipeline: GPURenderPipeline | null = null
    if (fieldEffect) {
      const fieldSource = buildFieldShader(fieldEffect)
      const userLineOffset = fieldSource.slice(0, fieldSource.indexOf(wgsl)).split("\n").length - 1
      this.device.pushErrorScope("validation")
      const fieldModule = this.device.createShaderModule({ label: "field shader (effect)", code: fieldSource })
      const info = await fieldModule.getCompilationInfo()
      const fieldScopeErr = await this.device.popErrorScope()
      const diagnostics = info.messages
        .filter((m) => m.type === "error")
        .map((m) => `${Math.max(0, m.lineNum - userLineOffset)}:${m.linePos} ${m.message}`)
      if (diagnostics.length === 0 && fieldScopeErr) diagnostics.push(fieldScopeErr.message)
      if (diagnostics.length > 0) return { ok: false, diagnostics, mounts }
      try {
        fieldPipeline = await this.device.createRenderPipelineAsync({
          label: "field layer pipeline",
          layout: this.fieldPipelineLayout,
          vertex: { module: fieldModule, entryPoint: "fieldVs" },
          fragment: {
            module: fieldModule,
            entryPoint: "fieldFs",
            targets: [{ format: "rgba16float" }, { format: "rgba16float" }],
          },
          primitive: { topology: "triangle-list" },
          multisample: { count: 1 },
        })
      } catch (e) {
        return { ok: false, diagnostics: [e instanceof Error ? e.message : String(e)], mounts }
      }
    }
    let identity: GPURenderPipeline
    let gamma: GPURenderPipeline
    try {
      const make = (applyGamma: boolean, label: string) =>
        this.device.createRenderPipelineAsync({
          label,
          layout: this.compositePipelineLayout,
          vertex: { module, entryPoint: "vs" },
          fragment: {
            module,
            entryPoint: "fs",
            constants: { APPLY_GAMMA: applyGamma ? 1 : 0 },
            targets: [{ format: this.presentationFormat }],
          },
          primitive: { topology: "triangle-list" },
        })
      ;[identity, gamma] = await Promise.all([
        make(false, "composite pipeline (effect, gamma=1)"),
        make(true, "composite pipeline (effect, gamma!=1)"),
      ])
    } catch (e) {
      return { ok: false, diagnostics: [e instanceof Error ? e.message : String(e)], mounts }
    }

    // Built BEFORE the swap: a particle stage that fails to compile has to leave
    // the previously installed effect running, exactly as a bad composite does.
    let particles: NonNullable<Engine["particles"]> | null = null
    if (wantsParticles) {
      const built = await this.buildParticles(wgsl, anchors.filter((a) => a.trail).length)
      if (!built.ok) return { ok: false, diagnostics: built.diagnostics, mounts }
      particles = built.state
    }
    let sim: NonNullable<Engine["sim"]> | null = null
    if (simEntryPoint(wgsl)) {
      const built = await this.buildSim(wgsl, anchors.filter((a) => a.trail).length)
      if (!built.ok) return { ok: false, diagnostics: built.diagnostics, mounts }
      sim = built.state
    }
    let trails: NonNullable<Engine["trails"]> | null = null
    if (wantsTrails) {
      // Only anchors that asked for `trail` have a path to draw; a ribbon on a
      // bone recorded without one would read zeroes and paint a line to the origin.
      const trailSlots = anchors.filter((a) => a.trail).length
      if (trailSlots === 0) {
        return {
          ok: false,
          diagnostics: ["a ribbon effect needs at least one // @anchor <bone> trail"],
          mounts,
        }
      }
      const built = await this.buildTrails(wgsl, trailSlots)
      if (!built.ok) return { ok: false, diagnostics: built.diagnostics, mounts }
      trails = built.state
    }

    // ── Swap — only now does the old effect (and its params buffer) go away.
    this.effect?.paramsBuffer?.destroy()
    this.releaseParticles()
    this.releaseTrails()
    this.releaseSim()
    this.particles = particles
    this.trails = trails
    this.sim = sim
    this.fieldPipeline = fieldPipeline
    // `// @fullres`: an effect that draws SUB-PIXEL detail — hairline curves,
    // scanlines — declares it and pays full price; everything soft stays at
    // half. The field shader reads its size from fieldU, so nothing else moves.
    const wantScale = /^\s*\/\/\s*@fullres\s*$/m.test(wgsl) ? 1 : 2
    if (wantScale !== this.fieldScale) {
      this.fieldScale = wantScale
      this.createFieldTargets()
    }
    let paramsBuffer: GPUBuffer | null = null
    if (entries.length) {
      paramsBuffer = this.device.createBuffer({
        label: "effect params",
        size: paramsData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(paramsBuffer, 0, paramsData)
    }
    this.effect = { wgsl, paramLayout: layout, paramsBuffer, paramsData, hasBackground, hasForeground, anchors }
    // Velocities and paths restart from rest rather than continuing from
    // whatever the last effect's slot 0 happened to be — the slots mean
    // something different now, and a trail would otherwise draw a line from the
    // old bone to the new one across the whole scene.
    this.anchorPrev.clear()
    this.anchorTrail.clear()
    this.compositePipelineIdentity = identity
    this.compositePipelineGamma = gamma
    this.effectEpochScene = this.sceneClock
    this.rebuildCompositeBindGroup()
    this.writeCompositeViewUniforms()
    return { ok: true, diagnostics: [], mounts }
  }

  /**
   * Compile an effect's particle stages and allocate its pool.
   *
   * Two modules, not one: the compute and render stages bind the same buffer
   * with different access (read_write vs read), and a single module would have
   * to pick one. Compiling them separately also means an author's helper names
   * live in their own compilation unit, which is what lets two effects both
   * define `hash21` without meeting.
   */
  private async buildParticles(
    wgsl: string,
    trailSlots: number,
  ): Promise<{ ok: true; state: NonNullable<Engine["particles"]> } | { ok: false; diagnostics: string[] }> {
    // No pragma means "some": an author who wrote the trio clearly wants
    // particles, and failing over a missing comment would be pedantry.
    const count = parseParticleCount(wgsl, Engine.MAX_PARTICLES) || 1024
    const src = { wgsl, count, blend: parseParticleBlend(wgsl), bloom: parseParticleBloom(wgsl) }
    // Sparks want to spawn where a trail is, so the particle stages see the same
    // cast buffer the trail draw reads.
    const cast = {
      subjects: MAX_EFFECT_SUBJECTS,
      samples: TRAIL_SAMPLES,
      base: MAX_EFFECT_SUBJECTS * 3,
      trailBase: CAST_TRAIL_BASE,
      slots: trailSlots,
    }

    const compile = async (code: string, label: string): Promise<GPUShaderModule | string[]> => {
      const offset = code.slice(0, code.indexOf(wgsl)).split("\n").length - 1
      this.device.pushErrorScope("validation")
      const module = this.device.createShaderModule({ label, code })
      const info = await module.getCompilationInfo()
      const scopeErr = await this.device.popErrorScope()
      const diagnostics = info.messages
        .filter((m) => m.type === "error")
        .map((m) => `${Math.max(0, m.lineNum - offset)}:${m.linePos} ${m.message}`)
      if (diagnostics.length === 0 && scopeErr) diagnostics.push(scopeErr.message)
      return diagnostics.length ? diagnostics : module
    }

    const computeModule = await compile(buildParticleComputeShader(src, cast), "particle compute")
    if (Array.isArray(computeModule)) return { ok: false, diagnostics: computeModule }
    const renderModule = await compile(buildParticleRenderShader(src, cast), "particle render")
    if (Array.isArray(renderModule)) return { ok: false, diagnostics: renderModule }

    const buffer = this.device.createBuffer({
      label: "particle pool",
      size: count * PARTICLE_STRIDE,
      usage: GPUBufferUsage.STORAGE,
    })
    const uniform = this.device.createBuffer({
      label: "particle uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const uniformBytes = new ArrayBuffer(16)
    const uniformView = { floats: new Float32Array(uniformBytes), uints: new Uint32Array(uniformBytes) }

    // Visibility is per LAYOUT, not shared: a read_write storage buffer may not be
    // visible to the vertex stage at all (WebGPU forbids it — a vertex shader
    // that could write memory has no defined ordering against the rasteriser).
    // Declaring one set of flags for both layouts is what made the pipeline
    // layout invalid, and the error surfaces later and unhelpfully as "invalid
    // due to a previous error".
    const layoutFor = (storage: GPUBufferBindingType, visibility: number) =>
      this.device.createBindGroupLayout({
        entries: [
          { binding: 0, visibility, buffer: { type: storage } },
          { binding: 1, visibility, buffer: { type: "uniform" } },
          { binding: 2, visibility, buffer: { type: "uniform" } },
          { binding: 3, visibility, buffer: { type: "read-only-storage" } },
          { binding: 4, visibility, buffer: { type: "read-only-storage" } },
        ],
      })
    const bindFor = (layout: GPUBindGroupLayout) =>
      this.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: uniform } },
          { binding: 2, resource: { buffer: this.cameraUniformBuffer } },
          { binding: 3, resource: { buffer: this.castBuffer } },
          { binding: 4, resource: { buffer: this.audioBuffer } },
        ],
      })

    const computeLayout = layoutFor("storage", GPUShaderStage.COMPUTE)
    const renderLayout = layoutFor("read-only-storage", GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT)

    // Additive keeps the destination and adds to it; the alpha channel is left
    // alone (dst factor one, src zero) so a glow does not also claim coverage
    // it never occluded.
    // Additive effects need the MASK to sum like the colour does — see the
    // fragment shaders' mask comment. rg8unorm clamps the sum at 1, which is the
    // saturation alpha-over would reach anyway.
    const maskTarget: GPUColorTargetState =
      src.blend === "additive"
        ? {
            format: Engine.BLOOM_MASK_FORMAT,
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
            },
          }
        : this.sceneTargets[1]
    const colorTarget: GPUColorTargetState =
      src.blend === "additive"
        ? {
            format: this.hdrFormat,
            blend: {
              color: { srcFactor: "one", dstFactor: "one", operation: "add" },
              alpha: { srcFactor: "zero", dstFactor: "one", operation: "add" },
            },
          }
        : this.sceneTargets[0]

    this.device.pushErrorScope("validation")
    try {
      const compute = await this.device.createComputePipelineAsync({
        label: "particle compute pipeline",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [computeLayout] }),
        compute: { module: computeModule, entryPoint: "main" },
      })
      const render = await this.device.createRenderPipelineAsync({
        label: "particle render pipeline",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [renderLayout] }),
        vertex: { module: renderModule, entryPoint: "vs" },
        fragment: { module: renderModule, entryPoint: "fs", targets: [colorTarget, maskTarget] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        // Tested but not WRITTEN: particles are transparent, so writing depth
        // would make whichever quad drew first occlude the ones behind it.
        depthStencil: { format: "depth24plus-stencil8", depthWriteEnabled: false, depthCompare: "less-equal" },
        multisample: { count: Engine.MULTISAMPLE_COUNT },
      })
      const scoped = await this.device.popErrorScope()
      if (scoped) {
        buffer.destroy()
        uniform.destroy()
        return { ok: false, diagnostics: [scoped.message] }
      }
      return {
        ok: true,
        state: {
          count,
          buffer,
          uniform,
          // One 16-byte block, two views: time/dt are floats and count/frame are
          // integers, and writing them through separate arrays would upload two
          // different buffers with the same name.
          data: uniformView.floats,
          counts: uniformView.uints,
          compute,
          computeLayout,
          computeBind: bindFor(computeLayout),
          render,
          renderLayout,
          renderBind: bindFor(renderLayout),
          rebind: () => ({ computeBind: bindFor(computeLayout), renderBind: bindFor(renderLayout) }),
        },
      }
    } catch (e) {
      await this.device.popErrorScope()
      buffer.destroy()
      uniform.destroy()
      return { ok: false, diagnostics: [e instanceof Error ? e.message : String(e)] }
    }
  }

  /**
   * Step the pool, before the scene pass.
   *
   * Outside the render pass because a compute dispatch cannot be encoded inside
   * one — and it has to precede the draw that reads the same buffer, or the
   * quads render last frame's positions.
   */
  private stepParticles(encoder: GPUCommandEncoder, deltaTime: number): void {
    const p = this.particles
    if (!p) return
    p.data[0] = this.sceneClock - this.effectEpochScene
    // Clamped: a backgrounded tab returns with a delta of whole seconds, and an
    // unclamped step flings every particle out of the scene in one frame.
    p.data[1] = Math.min(0.1, Math.max(0, deltaTime))
    p.counts[2] = p.count
    p.counts[3] = this.particleFrame++
    this.device.queue.writeBuffer(p.uniform, 0, p.data.buffer as ArrayBuffer)
    const cp = encoder.beginComputePass({ label: "particles" })
    cp.setPipeline(p.compute)
    cp.setBindGroup(0, p.computeBind)
    cp.dispatchWorkgroups(Math.ceil(p.count / 64))
    cp.end()
  }

  /** Draw the pool. Inside the scene pass, so it is depth-tested and pre-bloom. */
  private renderParticles(pass: GPURenderPassEncoder): void {
    const p = this.particles
    if (!p) return
    pass.setPipeline(p.render)
    pass.setBindGroup(0, p.renderBind)
    pass.draw(6, p.count)
  }

  /**
   * Compile an effect's ribbon stage.
   *
   * One instance per (slot, subject, segment), so a scene with several dancers
   * and several declared bones is still one draw and nothing is computed per
   * frame on the CPU.
   */
  private async buildTrails(
    wgsl: string,
    slots: number,
  ): Promise<{ ok: true; state: NonNullable<Engine["trails"]> } | { ok: false; diagnostics: string[] }> {
    const src = { wgsl, slots, blend: parseParticleBlend(wgsl), bloom: parseParticleBloom(wgsl) }
    const code = buildTrailShader(src, {
      subjects: MAX_EFFECT_SUBJECTS,
      samples: TRAIL_SAMPLES,
      base: MAX_EFFECT_SUBJECTS * 3,
      trailBase: CAST_TRAIL_BASE,
    })
    const offset = code.slice(0, code.indexOf(wgsl)).split("\n").length - 1
    this.device.pushErrorScope("validation")
    const module = this.device.createShaderModule({ label: "trail shader", code })
    const info = await module.getCompilationInfo()
    const scopeErr = await this.device.popErrorScope()
    const diagnostics = info.messages
      .filter((m) => m.type === "error")
      .map((m) => `${Math.max(0, m.lineNum - offset)}:${m.linePos} ${m.message}`)
    if (diagnostics.length === 0 && scopeErr) diagnostics.push(scopeErr.message)
    if (diagnostics.length) return { ok: false, diagnostics }

    const uniform = this.device.createBuffer({
      label: "trail uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const layout = this.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        // The scene's depth, for the fragment's manual occlusion test.
        {
          binding: 3,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "depth", viewDimension: "2d", multisampled: true },
        },
        // The audio analysis, for rzAudio* in width and shade alike.
        { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
      ],
    })
    // ONE target: the ribbons' own layer, blended with MAX in both channels.
    // Max is the original's core-takes-the-max rule as a blend mode — parallel
    // strands of a circling hand meet as max and cannot double into bright
    // dashes, which every additive variant of this pipeline drew. The layer is
    // composited over the frame after tone mapping (see composite.ts), which is
    // where the fullscreen ribbon always ran.
    const layerTarget: GPUColorTargetState = {
      format: "rgba16float",
      blend: {
        color: { srcFactor: "one", dstFactor: "one", operation: "max" },
        alpha: { srcFactor: "one", dstFactor: "one", operation: "max" },
      },
    }
    this.device.pushErrorScope("validation")
    try {
      const pipeline = await this.device.createRenderPipelineAsync({
        label: "trail pipeline",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [layerTarget] },
        primitive: { topology: "triangle-list", cullMode: "none" },
        // No depth attachment and no MSAA: the layer is a lone colour target,
        // and occlusion happens in the fragment against the scene's own depth.
        multisample: { count: 1 },
      })
      const scoped = await this.device.popErrorScope()
      if (scoped) {
        uniform.destroy()
        return { ok: false, diagnostics: [scoped.message] }
      }
      return {
        ok: true,
        state: {
          instances: slots * MAX_EFFECT_SUBJECTS * (TRAIL_SAMPLES - 1) * TRAIL_SUBDIVISIONS,
          uniform,
          data: new Float32Array(4),
          pipeline,
          layout,
          bind: this.device.createBindGroup({
            layout,
            entries: [
              { binding: 0, resource: { buffer: this.castBuffer } },
              { binding: 1, resource: { buffer: uniform } },
              { binding: 2, resource: { buffer: this.cameraUniformBuffer } },
              { binding: 3, resource: this.depthReadView! },
              { binding: 4, resource: { buffer: this.audioBuffer } },
            ],
          }),
        },
      }
    } catch (e) {
      await this.device.popErrorScope()
      uniform.destroy()
      return { ok: false, diagnostics: [e instanceof Error ? e.message : String(e)] }
    }
  }

  private releaseTrails(): void {
    this.trails?.uniform.destroy()
    this.trails = null
  }

  /** Draw the ribbons into their own layer — cleared, max-blended, and
   *  composited over the frame after tone mapping. */
  private renderTrailLayer(encoder: GPUCommandEncoder): void {
    const t = this.trails
    if (!t || !this.trailLayerView) return
    t.data[0] = this.sceneClock - this.effectEpochScene
    this.device.queue.writeBuffer(t.uniform, 0, t.data.buffer as ArrayBuffer)
    const pass = encoder.beginRenderPass({
      label: "trail layer",
      colorAttachments: [
        { view: this.trailLayerView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      ],
    })
    pass.setPipeline(t.pipeline)
    pass.setBindGroup(0, t.bind)
    pass.draw(6, t.instances)
    pass.end()
  }

  /** The user's field mounts, drawn at half resolution for the composite to
   *  upsample. Runs the whole quad — uniform control flow, so effects may use
   *  derivatives freely, which the old inline path had to forbid. */
  private renderFieldPass(encoder: GPUCommandEncoder): void {
    if (!this.fieldPipeline || !this.fieldBgView || !this.fieldFgView || !this.fieldBindGroups) return
    const pass = encoder.beginRenderPass({
      label: "field layer",
      colorAttachments: [
        { view: this.fieldBgView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        { view: this.fieldFgView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
      ],
    })
    pass.setPipeline(this.fieldPipeline)
    // The grid the sim just WROTE, which after its parity flip is the one at
    // `parity` — an effect reads this frame's simulation, not last frame's.
    pass.setBindGroup(0, this.fieldBindGroups[this.sim?.parity ?? 0])
    pass.draw(3)
    pass.end()
  }

  /** The trail bind group holds the depth view, which a resize recreates. */
  private rebindTrails(): void {
    const t = this.trails
    if (!t || !this.depthReadView) return
    t.bind = this.device.createBindGroup({
      layout: t.layout,
      entries: [
        { binding: 0, resource: { buffer: this.castBuffer } },
        { binding: 1, resource: { buffer: t.uniform } },
        { binding: 2, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 3, resource: this.depthReadView },
        { binding: 4, resource: { buffer: this.audioBuffer } },
      ],
    })
  }

  private releaseParticles(): void {
    this.particles?.buffer.destroy()
    this.particles?.uniform.destroy()
    this.particles = null
  }

  /**
   * Compile and allocate the effect's persistent grid.
   *
   * The textures are created ZEROED, which is the contract a kernel is written
   * against: rzSimFrame() is 0 on the first step and every value it reads is
   * zero, so seeding is just "if frame is 0, return the initial state".
   */
  private async buildSim(
    wgsl: string,
    trailSlots: number,
  ): Promise<{ ok: true; state: NonNullable<Engine["sim"]> } | { ok: false; diagnostics: string[] }> {
    const size = parseSimSize(wgsl, SIM_MAX) || 256
    const cast = {
      subjects: MAX_EFFECT_SUBJECTS,
      samples: TRAIL_SAMPLES,
      base: MAX_EFFECT_SUBJECTS * 3,
      trailBase: CAST_TRAIL_BASE,
      slots: trailSlots,
    }
    const code = buildSimShader(wgsl, size, cast)
    const offset = code.slice(0, code.indexOf(wgsl)).split("\n").length - 1
    this.device.pushErrorScope("validation")
    const module = this.device.createShaderModule({ label: "sim step", code })
    const info = await module.getCompilationInfo()
    const scopeErr = await this.device.popErrorScope()
    const diagnostics = info.messages
      .filter((m) => m.type === "error")
      .map((m) => `${Math.max(0, m.lineNum - offset)}:${m.linePos} ${m.message}`)
    if (diagnostics.length === 0 && scopeErr) diagnostics.push(scopeErr.message)
    if (diagnostics.length) return { ok: false, diagnostics }

    const layout = this.device.createBindGroupLayout({
      label: "sim bind layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: "filtering" } },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: "write-only", format: SIM_FORMAT, viewDimension: "2d" },
        },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    })

    const make = (n: number) =>
      this.device.createTexture({
        label: `sim grid ${n}`,
        size: [size, size],
        format: SIM_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      })
    const textures: [GPUTexture, GPUTexture] = [make(0), make(1)]
    const read: [GPUTextureView, GPUTextureView] = [textures[0].createView(), textures[1].createView()]
    const uniform = this.device.createBuffer({
      label: "sim uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.device.pushErrorScope("validation")
    try {
      const pipeline = await this.device.createComputePipelineAsync({
        label: "sim step pipeline",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module, entryPoint: "main" },
      })
      const scoped = await this.device.popErrorScope()
      if (scoped) {
        textures[0].destroy()
        textures[1].destroy()
        uniform.destroy()
        return { ok: false, diagnostics: [scoped.message] }
      }
      // One per parity: binds[i] READS textures[i] and WRITES the other.
      const bindFor = (i: number) =>
        this.device.createBindGroup({
          layout,
          entries: [
            { binding: 0, resource: { buffer: uniform } },
            { binding: 1, resource: read[i] },
            { binding: 2, resource: this.simSampler },
            { binding: 3, resource: textures[1 - i].createView() },
            { binding: 4, resource: { buffer: this.castBuffer } },
            { binding: 5, resource: { buffer: this.audioBuffer } },
            { binding: 6, resource: { buffer: this.compositeUniformBuffer } },
          ],
        })
      return {
        ok: true,
        state: {
          size,
          textures,
          read,
          pipeline,
          layout,
          binds: [bindFor(0), bindFor(1)],
          uniform,
          data: new Float32Array(4),
          parity: 0,
          frame: 0,
        },
      }
    } catch (e) {
      await this.device.popErrorScope()
      textures[0].destroy()
      textures[1].destroy()
      uniform.destroy()
      return { ok: false, diagnostics: [e instanceof Error ? e.message : String(e)] }
    }
  }

  private releaseSim(): void {
    this.sim?.textures[0].destroy()
    this.sim?.textures[1].destroy()
    this.sim?.uniform.destroy()
    this.sim = null
  }

  /**
   * Step the grid, before anything reads it.
   *
   * Outside the render pass, like the particle step and for the same reason —
   * and before the field pass, or an effect samples a grid one frame stale.
   */
  private stepSim(encoder: GPUCommandEncoder, deltaTime: number): void {
    const sim = this.sim
    if (!sim) return
    sim.data[0] = this.sceneClock - this.effectEpochScene
    // Clamped like the particle step: a backgrounded tab returns with a delta of
    // whole seconds, and one unclamped step of an advection kernel throws the
    // whole grid off its own edge.
    sim.data[1] = Math.min(0.1, Math.max(0, deltaTime))
    sim.data[2] = sim.size
    sim.data[3] = sim.frame++
    this.device.queue.writeBuffer(sim.uniform, 0, sim.data.buffer as ArrayBuffer)
    const cp = encoder.beginComputePass({ label: "sim" })
    cp.setPipeline(sim.pipeline)
    cp.setBindGroup(0, sim.binds[sim.parity])
    const groups = Math.ceil(sim.size / 8)
    cp.dispatchWorkgroups(groups, groups)
    cp.end()
    // The freshly written texture is now the current one.
    sim.parity = 1 - sim.parity
  }

  /** Which mounts the installed effect declared. Both false when none is set. */
  getEffectMounts(): { background: boolean; foreground: boolean } {
    return { background: this.effect?.hasBackground ?? false, foreground: this.effect?.hasForeground ?? false }
  }

  /** Write one effect param (declared at setEffect) — a uniform write, no
   *  recompile; the instant tier, like setStyleParam. */
  setEffectParam(name: string, value: EffectParamValue): void {
    const fx = this.effect
    if (!fx || !fx.paramsBuffer) return
    const slot = fx.paramLayout.get(name)
    if (!slot) return
    if (typeof value === "number") fx.paramsData[slot.offset] = value
    else {
      fx.paramsData[slot.offset] = value.x
      fx.paramsData[slot.offset + 1] = value.y
      fx.paramsData[slot.offset + 2] = value.z
    }
    this.device.queue.writeBuffer(fx.paramsBuffer, 0, fx.paramsData)
  }

  /** Patch bloom; GPU uniforms update immediately if `init()` has run. */
  /** Camera depth of field (see DepthOfFieldOptions). Free while disabled —
   *  the scene pass only stores its depth buffer on frames the gather reads. */
  setDepthOfField(patch: Partial<DepthOfFieldOptions>): void {
    this.depthOfField = { ...this.depthOfField, ...patch }
    if (!this.device || !this.dofUniformBuffer) return
    if (this.depthOfField.enabled) {
      this.writeDepthOfFieldUniforms()
    } else {
      // One last write so the shader's uniform branch reads a clean zero.
      this.dofUniformData[0] = 0
      this.device.queue.writeBuffer(this.dofUniformBuffer, 0, this.dofUniformData)
    }
  }

  getDepthOfField(): DepthOfFieldOptions {
    return { ...this.depthOfField }
  }

  /** Auto-focus target: the camera-space depth span of the first visible
   *  character's bones. Focus sits at the span's midpoint; the range covers the
   *  span plus a margin for what bones don't reach (shoes, hair, cloth). */
  private getModelBodyFocus(): { distance: number; range: number } | null {
    if (!this.camera) return null
    const view = this.camera.getViewMatrix().values
    for (const inst of this.modelInstances.values()) {
      if (!inst.model.visible || inst.isStage) continue
      const model = inst.model
      const matrices = model.getWorldMatrices()
      if (matrices.length === 0) continue
      const scale = model.scale
      const local = this.dofFocusScratch
      let minDepth = Infinity
      let maxDepth = -Infinity
      for (const matrix of matrices) {
        const values = matrix.values
        // Bone matrices are model-space; apply the same root transform the
        // renderer bakes into skinning, then take camera-space z.
        local.setXYZ(values[12] * scale, values[13] * scale, values[14] * scale)
        Quat.rotateVecInto(model.rotation, local, local)
        const x = model.position.x + local.x
        const y = model.position.y + local.y
        const z = model.position.z + local.z
        const depth = view[2] * x + view[6] * y + view[10] * z + view[14]
        if (!Number.isFinite(depth) || depth <= this.camera.near) continue
        minDepth = Math.min(minDepth, depth)
        maxDepth = Math.max(maxDepth, depth)
      }
      if (Number.isFinite(minDepth) && Number.isFinite(maxDepth)) {
        const span = Math.max(0, maxDepth - minDepth)
        const meshMargin = Math.max(2.0, span * 0.15)
        return { distance: (minDepth + maxDepth) * 0.5, range: Math.max(2.0, span + meshMargin) }
      }
    }
    return null
  }

  private writeDepthOfFieldUniforms(): void {
    if (!this.device || !this.dofUniformBuffer) return
    const d = this.depthOfField
    const u = this.dofUniformData
    // `d.enabled &&`, because a foreground effect also drives this write (for
    // projA/projB alone) and auto-focus walks every visible character's bones —
    // work nothing would read with the gather switched off.
    const auto = d.enabled && d.focusMode === "auto" ? this.getModelBodyFocus() : null
    u[0] = d.enabled ? 1 : 0
    u[1] = auto?.distance ?? Math.max(d.focusDistance, 0.05)
    // In auto mode the authored range is a floor — the sharp band never cuts
    // into the character's own depth span.
    u[2] = Math.max(d.focusRange, auto?.range ?? 0.02, 0.02)
    u[3] = Math.max(d.aperture, 0)
    u[4] = Math.max(d.maxBlurRadius, 0)
    u[5] = Math.min(12, Math.max(3, Math.round(d.bladeCount)))
    u[6] = d.quality === "performance" ? 8 : d.quality === "cinematic" ? 24 : 16
    u[7] = 1 // anamorphic ratio, reserved (the shader clamps ≥ 0.25)
    // viewZ = projB / (z − projA), the inverse of perspectiveInto's z mapping.
    // near/far track the camera radius, so these refresh every enabled frame.
    const near = this.camera.near
    const far = this.camera.far
    u[8] = (far + near) / (far - near)
    u[9] = (-2 * near * far) / (far - near)
    this.device.queue.writeBuffer(this.dofUniformBuffer, 0, u)
  }

  setBloomOptions(patch: Partial<BloomOptions>): void {
    const b = this.bloomSettings
    if (patch.enabled !== undefined) b.enabled = patch.enabled
    if (patch.threshold !== undefined) b.threshold = patch.threshold
    if (patch.knee !== undefined) b.knee = patch.knee
    if (patch.radius !== undefined) b.radius = patch.radius
    if (patch.color !== undefined) {
      b.color.x = patch.color.x
      b.color.y = patch.color.y
      b.color.z = patch.color.z
    }
    if (patch.intensity !== undefined) b.intensity = patch.intensity
    if (patch.clamp !== undefined) b.clamp = patch.clamp
    if (this.device && this.bloomBlitUniformBuffer) {
      this.writeBloomUniforms()
      this.writeCompositeViewUniforms()
    }
  }

  // EEVEE prefilter uniforms (blit stage) + upsample sample scale. Intensity/tint live in composite.
  private writeBloomUniforms(): void {
    const b = this.bloomSettings
    const bu = this.bloomBlitUniformData
    // EEVEE prefilter: threshold, knee_half, clamp (0 → disabled), _unused
    // Blender halves the knee before passing to the shader (eevee_bloom.c: knee * 0.5f).
    // The blit shader's quadratic soft-knee curve uses knee_half as the offset from threshold,
    // so the soft ramp spans [threshold - knee/2 .. threshold + knee/2] — NOT [threshold - knee .. threshold + knee].
    bu[0] = b.threshold
    bu[1] = b.knee * 0.5
    bu[2] = b.clamp
    bu[3] = 0.0
    this.device.queue.writeBuffer(this.bloomBlitUniformBuffer, 0, bu)
    const us = this.bloomUpsampleUniformData
    // Blender: bloom.radius directly controls the tent-filter sample scale in texel units.
    us[0] = Math.max(0.5, b.radius)
    us[1] = 0
    us[2] = 0
    us[3] = 0
    this.device.queue.writeBuffer(this.bloomUpsampleUniformBuffer, 0, us)
  }

  // Step 1: Get WebGPU device and context
  async init() {
    const adapter = await navigator.gpu?.requestAdapter()
    if (!adapter) throw new Error("WebGPU is not supported in this browser.")
    const wantFeature: GPUFeatureName = "rg11b10ufloat-renderable"
    const hasRg11b10 = adapter.features.has(wantFeature)
    const device = await adapter.requestDevice({
      requiredFeatures: hasRg11b10 ? [wantFeature] : [],
    })
    if (!device) {
      throw new Error("WebGPU is not supported in this browser.")
    }
    this.device = device
    if (hasRg11b10) this.hdrFormat = "rg11b10ufloat"

    const context = this.canvas.getContext("webgpu")
    if (!context) {
      throw new Error("Failed to get WebGPU context.")
    }
    this.context = context

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat()

    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
      alphaMode: "premultiplied",
    })

    this.setupCamera()
    this.setupLighting()
    this.createPipelines()
    this.setupResize()
    Engine.instance = this
  }

  // One-shot bake of EEVEE's combined BRDF LUT — DFG (bsdf_lut_frag.glsl) packed
  // with ltc_mag_ggx (eevee_lut.c) into a single 64×64 rgba8unorm texture:
  //   .rg = split-sum DFG   → F_brdf_*_scatter
  //   .ba = LTC magnitude   → ltc_brdf_scale_from_lut
  // One texture fetch per fragment replaces the previous 2–3 taps. rgba8unorm
  // (vs rgba16float) halves sample bandwidth; DFG/LTC values fit [0,1] cleanly.
  private bakeBrdfLut() {
    if (BRDF_LUT_SIZE !== LTC_MAG_LUT_SIZE) {
      throw new Error("BRDF LUT bake requires DFG size == LTC size (both 64).")
    }

    // Temp rg16float LTC source — loaded 1:1 by the bake fragment shader, then dropped.
    const ltcTemp = this.device.createTexture({
      label: "LTC mag LUT (bake input)",
      size: [LTC_MAG_LUT_SIZE, LTC_MAG_LUT_SIZE],
      format: "rg16float",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    })
    const n = LTC_MAG_LUT_DATA.length
    const half = new Uint16Array(n)
    const f32 = new Float32Array(1)
    const u32 = new Uint32Array(f32.buffer)
    for (let i = 0; i < n; i++) {
      f32[0] = LTC_MAG_LUT_DATA[i]
      const x = u32[0]
      const sign = (x >>> 16) & 0x8000
      let exp = ((x >>> 23) & 0xff) - 127 + 15
      const mant = x & 0x7fffff
      if (exp <= 0) {
        half[i] = sign
      } else if (exp >= 31) {
        half[i] = sign | 0x7c00
      } else {
        half[i] = sign | (exp << 10) | (mant >>> 13)
      }
    }
    this.device.queue.writeTexture(
      { texture: ltcTemp },
      half,
      { bytesPerRow: LTC_MAG_LUT_SIZE * 4, rowsPerImage: LTC_MAG_LUT_SIZE },
      { width: LTC_MAG_LUT_SIZE, height: LTC_MAG_LUT_SIZE, depthOrArrayLayers: 1 },
    )

    this.brdfLutTexture = this.device.createTexture({
      label: "BRDF LUT (DFG + LTC packed)",
      size: [BRDF_LUT_SIZE, BRDF_LUT_SIZE],
      format: "rgba8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.brdfLutView = this.brdfLutTexture.createView()

    const module = this.device.createShaderModule({ label: "BRDF LUT bake", code: BRDF_LUT_BAKE_WGSL })
    const pipeline = this.device.createRenderPipeline({
      label: "BRDF LUT bake pipeline",
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] },
      primitive: { topology: "triangle-list" },
    })

    const bakeBindGroup = this.device.createBindGroup({
      label: "BRDF LUT bake bind group",
      layout: pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: ltcTemp.createView() }],
    })

    const enc = this.device.createCommandEncoder({ label: "BRDF LUT bake encoder" })
    const pass = enc.beginRenderPass({
      colorAttachments: [
        { view: this.brdfLutView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    })
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bakeBindGroup)
    pass.draw(3, 1, 0, 0)
    pass.end()
    this.device.queue.submit([enc.finish()])

    ltcTemp.destroy()
  }

  // Bake the Blender 3.6 Filmic MHC tone curve into a WIDTH×1 r16float LUT sampled by the
  // composite pass. The 14 anchors are the same as the old inline array; we fit a monotone
  // cubic (Fritsch–Carlson) through them so the curve is C1-continuous (no Mach banding in
  // smooth gradients) while still passing through every anchor (look preserved) and staying
  // monotone (no tonemap overshoot/ringing). Domain is uniform in log2 space: anchor k sits
  // at t=k, k=0..13 (t = log2(linear)+10). See composite.ts::filmic for the sampling map.
  /**
   * Decompress and upload Blender's AgX cube.
   *
   * Deliberately off the critical path: it is 723 KB once inflated, and a frame
   * rendered before it lands should show the scene under whatever transform is
   * already there rather than wait. Until then binding 10 holds a 1×1×1 stand-in,
   * which is only ever sampled if someone selects AgX in that window.
   */
  private async loadAgxLut(): Promise<void> {
    try {
      const packed = Uint8Array.from(atob(AGX_LUT_GZ), (ch) => ch.charCodeAt(0))
      const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream("gzip"))
      const bytes = new Uint8Array(await new Response(stream).arrayBuffer())
      const n = AGX_LUT_SIZE
      if (bytes.byteLength !== n * n * n * 4) throw new Error(`AgX LUT is ${bytes.byteLength} bytes, expected ${n ** 3 * 4}`)
      const tex = this.device.createTexture({
        label: "AgX 57³ LUT",
        size: [n, n, n],
        dimension: "3d",
        format: "rgb10a2unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      // .cube order is red fastest, which is exactly a 3D texture's own layout.
      this.device.queue.writeTexture({ texture: tex }, bytes, { bytesPerRow: n * 4, rowsPerImage: n }, [n, n, n])
      this.agxLutTexture = tex
      this.rebuildCompositeBindGroup()
    } catch {
      // A missing LUT costs AgX, not the renderer — the other transforms stand.
    }
  }

  private bakeFilmicLut() {
    const anchors = [
      0.0028, 0.0068, 0.0151, 0.0313, 0.061, 0.112, 0.192, 0.306, 0.459, 0.631, 0.82, 0.907, 0.962, 0.989,
    ]
    const n = anchors.length
    // Secant slopes (unit spacing, so d_k = y_{k+1} - y_k).
    const d = new Array<number>(n - 1)
    for (let k = 0; k < n - 1; k++) d[k] = anchors[k + 1] - anchors[k]
    // Endpoint + interior tangents.
    const m = new Array<number>(n)
    m[0] = d[0]
    m[n - 1] = d[n - 2]
    for (let k = 1; k < n - 1; k++) m[k] = (d[k - 1] + d[k]) * 0.5
    // Fritsch–Carlson monotonicity clamp.
    for (let k = 0; k < n - 1; k++) {
      if (d[k] === 0) {
        m[k] = 0
        m[k + 1] = 0
        continue
      }
      const a = m[k] / d[k]
      const b = m[k + 1] / d[k]
      const s = a * a + b * b
      if (s > 9) {
        const tau = 3 / Math.sqrt(s)
        m[k] = tau * a * d[k]
        m[k + 1] = tau * b * d[k]
      }
    }
    const W = Engine.FILMIC_LUT_WIDTH
    const values = new Float32Array(W)
    for (let j = 0; j < W; j++) {
      const t = ((n - 1) * j) / (W - 1) // t ∈ [0, n-1]
      const k = Math.min(Math.floor(t), n - 2)
      const s = t - k // local param in [0,1], unit-spaced segment
      const s2 = s * s
      const s3 = s2 * s
      // Hermite basis (h=1).
      const h00 = 2 * s3 - 3 * s2 + 1
      const h10 = s3 - 2 * s2 + s
      const h01 = -2 * s3 + 3 * s2
      const h11 = s3 - s2
      values[j] = h00 * anchors[k] + h10 * m[k] + h01 * anchors[k + 1] + h11 * m[k + 1]
    }

    // f32 → f16 bits (same conversion as bakeBrdfLut).
    const half = new Uint16Array(W)
    const f32 = new Float32Array(1)
    const u32 = new Uint32Array(f32.buffer)
    for (let j = 0; j < W; j++) {
      f32[0] = values[j]
      const x = u32[0]
      const sign = (x >>> 16) & 0x8000
      const exp = ((x >>> 23) & 0xff) - 127 + 15
      const mant = x & 0x7fffff
      if (exp <= 0) {
        half[j] = sign
      } else if (exp >= 31) {
        half[j] = sign | 0x7c00
      } else {
        half[j] = sign | (exp << 10) | (mant >>> 13)
      }
    }

    this.filmicLutTexture = this.device.createTexture({
      label: "Filmic tone LUT",
      size: [W, 1],
      format: "r16float",
      usage: GPUTextureUsage.COPY_DST | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.filmicLutView = this.filmicLutTexture.createView()
    this.device.queue.writeTexture(
      { texture: this.filmicLutTexture },
      half,
      { bytesPerRow: W * 2, rowsPerImage: 1 },
      { width: W, height: 1, depthOrArrayLayers: 1 },
    )
  }

  private createRenderPipeline(config: {
    label: string
    layout: GPUPipelineLayout
    shaderModule: GPUShaderModule
    vertexBuffers: GPUVertexBufferLayout[]
    fragmentTarget?: GPUColorTargetState
    fragmentTargets?: GPUColorTargetState[]
    fragmentEntryPoint?: string
    cullMode?: GPUCullMode
    depthStencil?: GPUDepthStencilState
    multisample?: GPUMultisampleState
  }): GPURenderPipeline {
    const targets = config.fragmentTargets ?? (config.fragmentTarget ? [config.fragmentTarget] : undefined)
    return this.device.createRenderPipeline({
      label: config.label,
      layout: config.layout,
      vertex: {
        module: config.shaderModule,
        buffers: config.vertexBuffers,
      },
      fragment: targets
        ? {
            module: config.shaderModule,
            entryPoint: config.fragmentEntryPoint,
            targets,
          }
        : undefined,
      primitive: { cullMode: config.cullMode ?? "none" },
      depthStencil: config.depthStencil,
      multisample: config.multisample ?? { count: Engine.MULTISAMPLE_COUNT },
    })
  }

  private createPipelines() {
    this.materialSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    })

    this.trailFallbackView = this.device
      .createTexture({
        label: "trail layer fallback (1x1 transparent)",
        size: [1, 1],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING,
      })
      .createView()

    // CLAMPED, not repeated. A grid holds a bounded patch of world — a pool of
    // fog, a stretch of water — and a kernel that reads past its edge means to
    // ask what is just outside, not to wrap around to the far side of it.
    this.simSampler = this.device.createSampler({
      label: "sim grid sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    })
    this.simFallbackView = this.device
      .createTexture({
        label: "sim grid fallback (1x1 zero)",
        size: [1, 1],
        format: SIM_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING,
      })
      .createView()

    this.audioFallbackBuffer = this.device.createBuffer({
      label: "audio analysis fallback (silence)",
      size: 32,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.audioBuffer = this.audioFallbackBuffer

    this.fieldUniformBuffer = this.device.createBuffer({
      label: "field layer uniforms (half size, full size)",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // The field pass's own layout: the subset of the composite's bindings the
    // user's code can statically reach, WITHOUT the field textures themselves —
    // a pass may not sample its own attachments, and WebGPU counts every
    // resource in a bound group whether the shader reads it or not.
    this.fieldBindGroupLayout = this.device.createBindGroupLayout({
      label: "field layer bind layout",
      entries: [
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        {
          binding: 8,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "depth", viewDimension: "2d", multisampled: true },
        },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 11, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 13, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 14, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 17, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 18, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    })
    this.fieldPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.fieldBindGroupLayout],
    })

    this.fallbackMaterialTexture = this.device.createTexture({
      label: "fallback material texture (1x1 white)",
      size: [1, 1],
      format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.device.queue.writeTexture(
      { texture: this.fallbackMaterialTexture },
      new Uint8Array([255, 255, 255, 255]),
      { bytesPerRow: 4 },
      [1, 1],
    )

    // Generic shared-toon ramp: lit white down to a soft cool shadow tone with
    // a tight terminator around the midpoint, approximating MMD's toon ramps.
    const TOON_H = 64
    const toonData = new Uint8Array(TOON_H * 4)
    for (let y = 0; y < TOON_H; y++) {
      const v = y / (TOON_H - 1)
      // smoothstep terminator centered at 0.55, width ~0.1
      const t = Math.min(1, Math.max(0, (v - 0.5) / 0.1))
      const s = t * t * (3 - 2 * t)
      toonData[y * 4 + 0] = Math.round(255 - s * (255 - 196))
      toonData[y * 4 + 1] = Math.round(255 - s * (255 - 186))
      toonData[y * 4 + 2] = Math.round(255 - s * (255 - 205))
      toonData[y * 4 + 3] = 255
    }
    this.defaultToonRampTexture = this.device.createTexture({
      label: "default toon ramp (1x64)",
      size: [1, TOON_H],
      format: "rgba8unorm-srgb",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.device.queue.writeTexture({ texture: this.defaultToonRampTexture }, toonData, { bytesPerRow: 4 }, [1, TOON_H])

    // Shared vertex buffer layouts
    const fullVertexBuffers: GPUVertexBufferLayout[] = [
      {
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
          { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat },
          { shaderLocation: 2, offset: 6 * 4, format: "float32x2" as GPUVertexFormat },
        ],
      },
      {
        arrayStride: 4 * 2,
        attributes: [{ shaderLocation: 3, offset: 0, format: "uint16x4" as GPUVertexFormat }],
      },
      {
        arrayStride: 4,
        attributes: [{ shaderLocation: 4, offset: 0, format: "unorm8x4" as GPUVertexFormat }],
      },
    ]

    const outlineVertexBuffers: GPUVertexBufferLayout[] = [
      {
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
          { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat },
          // uv — the outline FS alpha-tests the diffuse texture (babylon-mmd parity)
          { shaderLocation: 2, offset: 6 * 4, format: "float32x2" as GPUVertexFormat },
        ],
      },
      {
        arrayStride: 4 * 2,
        attributes: [{ shaderLocation: 3, offset: 0, format: "uint16x4" as GPUVertexFormat }],
      },
      {
        arrayStride: 4,
        attributes: [{ shaderLocation: 4, offset: 0, format: "unorm8x4" as GPUVertexFormat }],
      },
    ]

    // Internal scene passes render into the HDR offscreen target; only the final
    // composite pass writes the swapchain. Tonemap moved to composite so bloom
    // (added next) can run on linear HDR.
    const standardBlend: GPUColorTargetState = {
      format: this.hdrFormat,
      blend: {
        color: {
          srcFactor: "src-alpha",
          dstFactor: "one-minus-src-alpha",
          operation: "add",
        },
        alpha: {
          srcFactor: "one",
          dstFactor: "one-minus-src-alpha",
          operation: "add",
        },
      },
    }

    // Aux target carrying (bloom mask, alpha). Src-alpha blend so the .g channel
    // accumulates proper alpha-over (same semantic the old rgba16f hdr.a had).
    // Materials write vec2f(mask, 1.0); ground writes vec2f(0.0, 1.0). With src.a
    // coming from the fragment color.a, the blend equation produces
    //   out.g = 1·src.a + dst.g·(1-src.a)  →  premultiplied over operator on alpha.
    // .r gets weighted by src.a too, which is fine: opaque pixels (α=1) give full
    // mask, partially translucent fragments dilute mask proportionally — acceptable
    // for the bloom-gate use.
    const maskBlend: GPUColorTargetState = {
      format: Engine.BLOOM_MASK_FORMAT,
      blend: {
        color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
        alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      },
    }
    const sceneTargets: GPUColorTargetState[] = [standardBlend, maskBlend]
    this.sceneTargets = sceneTargets
    this.fullVertexBufferLayouts = fullVertexBuffers

    // group 0: per-frame (camera + light + sampler + shadow) — bound once per pass
    this.mainPerFrameBindGroupLayout = this.device.createBindGroupLayout({
      label: "main per-frame bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    })
    // group 1: per-instance (skinMats) — bound once per model
    this.mainPerInstanceBindGroupLayout = this.device.createBindGroupLayout({
      label: "main per-instance bind group layout",
      // FRAGMENT visibility: the eye shader reads the 頭 bone's skinning
      // matrix for its rear-view gate.
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "read-only-storage" },
        },
      ],
    })
    // group 2: per-material (textures + material uniforms) — bound per draw call.
    // Toon + sphere texture slots (bindings 2/3) are reserved for future sphere/toon graph
    // nodes; graphs that don't read them just bind the 1×1 white fallback.
    this.mainPerMaterialBindGroupLayout = this.device.createBindGroupLayout({
      label: "main per-material bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        // StyleUniforms for compiled graph shaders (adjust-tier sliders). Hand-written
        // presets simply don't declare it — a layout may carry bindings a shader ignores.
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        // Style-group image maps. A PMX material carries one image; a
        // Blender-authored look needs a lightmap or ramp beside it, and those
        // belong to the GROUP rather than to the model's own material data.
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    })

    // Shared zero StyleUniforms buffer — bound by every ungrouped material; grouped
    // materials rebind binding(4) to their group's own buffer (per model, per group).
    this.zeroStyleBuffer = this.device.createBuffer({
      label: "style uniforms (zero)",
      size: 256,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const mainPipelineLayout = this.device.createPipelineLayout({
      label: "main pipeline layout",
      bindGroupLayouts: [
        this.mainPerFrameBindGroupLayout,
        this.mainPerInstanceBindGroupLayout,
        this.mainPerMaterialBindGroupLayout,
      ],
    })
    this.mainPipelineLayout = mainPipelineLayout

    // perFrameBindGroup is created after shadow resources below

    // Ungrouped materials render this neutral base — the compiled DEFAULT_GRAPH (diffuse
    // texture x material color -> Principled BSDF). Grouped materials use their group's
    // compiled pipeline instead. This is the single base shading model; the per-preset
    // hand shaders are retired in favor of graphs.
    const neutral = compileGraph(DEFAULT_GRAPH, { renderClass: "auto", alphaMode: "opaque" })
    if (!neutral.ok) throw new Error("failed to compile the neutral default graph")
    const neutralModule = this.device.createShaderModule({ label: "neutral base (default graph)", code: neutral.wgsl })
    this.neutralPipeline = this.createRenderPipeline({
      label: "neutral base pipeline",
      layout: mainPipelineLayout,
      shaderModule: neutralModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })
    // Depth-write-off twin for transparent-bucket draws (see pipelineForDrawCall).
    this.neutralPipelineNoDepthWrite = this.createRenderPipeline({
      label: "neutral base pipeline (no depth write)",
      layout: mainPipelineLayout,
      shaderModule: neutralModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: false,
        depthCompare: "less-equal",
      },
    })
    // Depth-only prepass for transparent draws (see depth-prepass.ts): writes the
    // fabric's depth AFTER its color blended, so outlines drawn later are
    // occluded behind it. Color targets kept for pass compatibility, writeMask 0.
    const prepassModule = this.device.createShaderModule({
      label: "transparent depth prepass",
      code: TRANSPARENT_DEPTH_PREPASS_WGSL,
    })
    this.transparentDepthPrepassPipeline = this.device.createRenderPipeline({
      label: "transparent depth prepass",
      layout: mainPipelineLayout,
      vertex: { module: prepassModule, entryPoint: "vs", buffers: fullVertexBuffers as GPUVertexBufferLayout[] },
      fragment: {
        module: prepassModule,
        entryPoint: "fs",
        targets: sceneTargets.map((t) => ({ format: (t as GPUColorTargetState).format, writeMask: 0 })),
      },
      primitive: { cullMode: "none" },
      multisample: { count: Engine.MULTISAMPLE_COUNT },
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    this.shadowLightVPBuffer = this.device.createBuffer({
      size: 64,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const shadowBindGroupLayout = this.device.createBindGroupLayout({
      label: "shadow depth bind layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    })
    const shadowShader = this.device.createShaderModule({
      label: "shadow depth",
      code: SHADOW_DEPTH_SHADER_WGSL,
    })
    this.shadowDepthPipeline = this.device.createRenderPipeline({
      label: "shadow depth pipeline",
      // Group 1 is the main pass's per-material layout so each shadow draw can
      // rebind the draw call's existing material bind group for the alpha test.
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [shadowBindGroupLayout, this.mainPerMaterialBindGroupLayout],
      }),
      vertex: { module: shadowShader, entryPoint: "vs", buffers: fullVertexBuffers as GPUVertexBufferLayout[] },
      fragment: { module: shadowShader, entryPoint: "fs", targets: [] },
      primitive: { cullMode: "none" },
      depthStencil: {
        format: "depth32float",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        depthBias: 2,
        depthBiasSlopeScale: 1.5,
        depthBiasClamp: 0,
      },
    })
    this.shadowComparisonSampler = this.device.createSampler({
      compare: "less",
      magFilter: "linear",
      minFilter: "linear",
    })
    this.shadowMapTexture = this.device.createTexture({
      label: "shadow map",
      size: [Engine.SHADOW_MAP_SIZE, Engine.SHADOW_MAP_SIZE],
      format: "depth32float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.shadowMapDepthView = this.shadowMapTexture.createView()

    // One-shot bake of Blender EEVEE's combined BRDF LUT (DFG + LTC packed rgba8unorm).
    this.bakeBrdfLut()
    this.agxFallbackTexture = this.device.createTexture({
      label: "AgX LUT fallback",
      size: [1, 1, 1],
      dimension: "3d",
      format: "rgb10a2unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    void this.loadAgxLut()
    this.bakeFilmicLut()

    // Now that shadow resources exist, create the main per-frame bind group
    this.perFrameBindGroup = this.device.createBindGroup({
      label: "main per-frame bind group",
      layout: this.mainPerFrameBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.lightUniformBuffer } },
        { binding: 2, resource: this.materialSampler },
        { binding: 3, resource: this.shadowMapDepthView },
        { binding: 4, resource: this.shadowComparisonSampler },
        { binding: 5, resource: { buffer: this.shadowLightVPBuffer } },
        { binding: 9, resource: this.brdfLutView },
      ],
    })

    this.groundShadowBindGroupLayout = this.device.createBindGroupLayout({
      label: "ground shadow layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "comparison" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    })
    const groundShadowShader = this.device.createShaderModule({
      label: "ground shadow",
      code: GROUND_SHADOW_SHADER_WGSL,
    })
    this.groundShadowPipeline = this.createRenderPipeline({
      label: "ground shadow pipeline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.groundShadowBindGroupLayout] }),
      shaderModule: groundShadowShader,
      // Slot 0 only — the ground has no skinning, and declaring the full
      // 3-slot layout while renderGround binds one buffer is a WebGPU
      // validation error that invalidates the whole command buffer.
      vertexBuffers: [fullVertexBuffers[0]],
      fragmentTargets: sceneTargets,
      cullMode: "back",
      depthStencil: { format: "depth24plus-stencil8", depthWriteEnabled: true, depthCompare: "less-equal" },
    })

    // Outline: group 0 = per-frame (camera), group 1 = per-instance (skinMats), group 2 = per-material (edge uniforms)
    this.outlinePerFrameBindGroupLayout = this.device.createBindGroupLayout({
      label: "outline per-frame bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
      ],
    })
    // Outline per-instance reuses mainPerInstanceBindGroupLayout (same skinMats binding)
    this.outlinePerMaterialBindGroupLayout = this.device.createBindGroupLayout({
      label: "outline per-material bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    })

    const outlinePipelineLayout = this.device.createPipelineLayout({
      label: "outline pipeline layout",
      bindGroupLayouts: [
        this.outlinePerFrameBindGroupLayout,
        this.mainPerInstanceBindGroupLayout,
        this.outlinePerMaterialBindGroupLayout,
      ],
    })

    this.outlinePerFrameBindGroup = this.device.createBindGroup({
      label: "outline per-frame bind group",
      layout: this.outlinePerFrameBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: this.materialSampler },
      ],
    })

    const outlineShaderModule = this.device.createShaderModule({
      label: "outline shaders",
      code: OUTLINE_SHADER_WGSL,
    })

    this.outlinePipeline = this.createRenderPipeline({
      label: "outline pipeline",
      layout: outlinePipelineLayout,
      shaderModule: outlineShaderModule,
      vertexBuffers: outlineVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "back",
      depthStencil: {
        format: "depth24plus-stencil8",
        // babylon-mmd draws outlines WITH depth write (its _afterRenderingMesh
        // forces setDepthWrite(true)); the constant bias below still makes
        // hulls lose depth ties against their own near-coplanar geometry.
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        // CONFIRMED fix (bisected live via setOutlineEnabled): hull fragments
        // carry their surface's exact depth, so against this model's paired
        // near-coplanar skirt layers the hulls WON depth ties in patches —
        // the black shapes on the dress. A small constant bias makes hulls lose
        // every tie; silhouette rims compare against the far background and are
        // unaffected. No slope term — slope explodes at silhouettes and would
        // erase the rims themselves (previous regression).
        depthBias: 4,
        depthBiasSlopeScale: 0,
        depthBiasClamp: 0,
        // Skip fragments where the eye stamped stencil=EYE_VALUE. Those pixels are owned by
        // the see-through-hair blend (hair-over-eyes), so letting the outline's near-black
        // edge color overwrite them would re-introduce the dark almond we just killed.
        stencilFront: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0,
      },
    })

    // ─── Selection overlay (screen-space edge-detect on a per-material mask) ───
    // Reuses outline camera + main skinMats bind group layouts. No group 2 (no per-mat uniform).
    const selectionMaskPipelineLayout = this.device.createPipelineLayout({
      label: "selection mask pipeline layout",
      bindGroupLayouts: [this.outlinePerFrameBindGroupLayout, this.mainPerInstanceBindGroupLayout],
    })
    const selectionMaskShaderModule = this.device.createShaderModule({
      label: "selection mask shader",
      code: SELECTION_MASK_SHADER_WGSL,
    })
    this.selectionMaskPipeline = this.device.createRenderPipeline({
      label: "selection mask pipeline",
      layout: selectionMaskPipelineLayout,
      vertex: { module: selectionMaskShaderModule, entryPoint: "vs", buffers: outlineVertexBuffers },
      fragment: {
        module: selectionMaskShaderModule,
        entryPoint: "fs",
        targets: [{ format: "r8unorm" }],
      },
      primitive: { cullMode: "none" },
      // Single-sample, no depth (depth-always via not attaching a depth buffer at all).
      multisample: { count: 1 },
    })

    this.selectionEdgeBindGroupLayout = this.device.createBindGroupLayout({
      label: "selection edge bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    })
    const selectionEdgePipelineLayout = this.device.createPipelineLayout({
      label: "selection edge pipeline layout",
      bindGroupLayouts: [this.selectionEdgeBindGroupLayout],
    })
    const selectionEdgeShaderModule = this.device.createShaderModule({
      label: "selection edge shader",
      code: SELECTION_EDGE_SHADER_WGSL,
    })
    this.selectionEdgePipeline = this.device.createRenderPipeline({
      label: "selection edge pipeline",
      layout: selectionEdgePipelineLayout,
      vertex: { module: selectionEdgeShaderModule, entryPoint: "vs" },
      fragment: {
        module: selectionEdgeShaderModule,
        entryPoint: "fs",
        targets: [
          {
            format: this.presentationFormat,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
      multisample: { count: 1 },
    })
    this.selectionSampler = this.device.createSampler({
      label: "selection sampler",
      magFilter: "linear",
      minFilter: "linear",
    })
    this.selectionEdgeUniformBuffer = this.device.createBuffer({
      label: "selection edge uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // thickness (pixels), + 3 floats padding
    this.device.queue.writeBuffer(this.selectionEdgeUniformBuffer, 0, new Float32Array([5.0, 0, 0, 0]))

    // ─── Transform gizmo (3 axes + 3 rings) ─────────────────────────
    this.setupGizmo()

    // ─── Bloom (EEVEE 3.6 pyramid): blit(Karis prefilter) → 13-tap downsamples → 9-tap tent upsamples ───
    // Mirrors source/blender/draw/engines/eevee/shaders/effect_bloom_frag.glsl.
    // Firefly suppression lives in the blit (Karis luminance-weighted 4-tap average). A single-pass
    // Gaussian cannot reproduce this — hot pixels dominate and produce the sparkle halo.
    this.bloomSampler = this.device.createSampler({
      label: "bloom sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    })
    this.bloomBlitUniformBuffer = this.device.createBuffer({
      label: "bloom blit uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.bloomUpsampleUniformBuffer = this.device.createBuffer({
      label: "bloom upsample uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.bloomBlitBindGroupLayout = this.device.createBindGroupLayout({
      label: "bloom blit layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
      ],
    })
    this.bloomDownsampleBindGroupLayout = this.device.createBindGroupLayout({
      label: "bloom downsample layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    })
    this.bloomUpsampleBindGroupLayout = this.device.createBindGroupLayout({
      label: "bloom upsample layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // coarser-mip accumulator
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // matching downsample mip (base add)
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    })

    const bloomBlitShader = this.device.createShaderModule({
      label: "bloom blit (Karis prefilter)",
      code: BLOOM_BLIT_SHADER_WGSL,
    })

    const bloomDownsampleShader = this.device.createShaderModule({
      label: "bloom downsample 13-tap",
      code: BLOOM_DOWNSAMPLE_SHADER_WGSL,
    })

    const bloomUpsampleShader = this.device.createShaderModule({
      label: "bloom upsample 9-tap tent",
      code: BLOOM_UPSAMPLE_SHADER_WGSL,
    })

    const bloomBlitLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bloomBlitBindGroupLayout] })
    const bloomDownLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.bloomDownsampleBindGroupLayout],
    })
    const bloomUpLayout = this.device.createPipelineLayout({ bindGroupLayouts: [this.bloomUpsampleBindGroupLayout] })

    this.bloomBlitPipeline = this.device.createRenderPipeline({
      label: "bloom blit pipeline",
      layout: bloomBlitLayout,
      vertex: { module: bloomBlitShader, entryPoint: "vs" },
      fragment: { module: bloomBlitShader, entryPoint: "fs", targets: [{ format: this.hdrFormat }] },
      primitive: { topology: "triangle-list" },
    })
    this.bloomDownsamplePipeline = this.device.createRenderPipeline({
      label: "bloom downsample pipeline",
      layout: bloomDownLayout,
      vertex: { module: bloomDownsampleShader, entryPoint: "vs" },
      fragment: { module: bloomDownsampleShader, entryPoint: "fs", targets: [{ format: this.hdrFormat }] },
      primitive: { topology: "triangle-list" },
    })
    this.bloomUpsamplePipeline = this.device.createRenderPipeline({
      label: "bloom upsample pipeline",
      layout: bloomUpLayout,
      vertex: { module: bloomUpsampleShader, entryPoint: "vs" },
      fragment: { module: bloomUpsampleShader, entryPoint: "fs", targets: [{ format: this.hdrFormat }] },
      primitive: { topology: "triangle-list" },
    })

    // ─── Composite: HDR + bloom → Filmic → swapchain (premultiplied) ───
    // Bloom color/intensity applied HERE (pyramid is pure energy; tint belongs to the combine step,
    // mirroring EEVEE where bloom color/intensity are combine-stage params, not prefilter).
    this.compositeUniformBuffer = this.device.createBuffer({
      label: "composite view uniforms",
      // 11 × vec4f: (exposure, invGamma, _, _) · (bloom tint, intensity) ·
      // (bg rgb, mode) · camera right/up/forward basis for the 360 skybox ray ·
      // (time, _, canvas width, canvas height) for user effects · three grade
      // vectors (CDL offset+contrast, power+saturation, slope+flag) · camera
      // world position, for an effect placing itself in the scene · four
      // character positions, for one that wants to respond to the cast.
      size: 240,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.dofUniformBuffer = this.device.createBuffer({
      label: "depth of field uniforms",
      // 3 × vec4f — see the dofU comment in composite.ts.
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.bgParamsDummyBuffer = this.device.createBuffer({
      label: "bg effect params (dummy)",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // Allocated at full size once rather than grown: it is ~7KB, the bind group
    // would otherwise be rebuilt whenever an effect declared a different number
    // of bones, and only the declared prefix is ever written.
    this.castData = new Float32Array(CAST_VEC4S * 4)
    this.castBuffer = this.device.createBuffer({
      label: "effect cast data",
      size: this.castData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.compositeBindGroupLayout = this.device.createBindGroupLayout({
      label: "composite bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float" } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        // Aux mask/alpha texture — composite reads .g to reconstruct the alpha that
        // used to live in the HDR target before the rg11b10ufloat switch.
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        // Filmic tone LUT (r16float, filterable) — sampled with the binding-2 sampler.
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        // 360 backdrop equirect (PhotoDome-style skybox) — 1×1 fallback when unset.
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        // User background-effect params — dummy buffer when no effect is set. The
        // layout is explicit, so the base shader legally ignores the binding.
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        // The scene pass's MSAA depth, depth-only aspect — read by the DoF
        // gather. Contents are undefined while DoF is off (the scene pass
        // discards depth then), and the shader never reads it then either.
        {
          binding: 8,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: "depth", viewDimension: "2d", multisampled: true },
        },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        // AgX's 57³ cube. Decompressed and uploaded off the critical path, so a
        // 1×1×1 stand-in keeps the bind group valid until it arrives.
        { binding: 10, visibility: GPUShaderStage.FRAGMENT, texture: { viewDimension: "3d" } },
        // The cast, for rzSubject/rzAnchor. Always bound so the base shader's
        // layout matches; the base shader simply never reads it.
        { binding: 11, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // The trail layer. Bound to a transparent 1×1 when no ribbon effect is
        // installed, so the base shader's layout always matches.
        { binding: 12, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        // The audio analysis, for rzAudio*. Silence fallback when the scene has
        // no track.
        { binding: 13, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // The field layer's two halves. Fallback-bound when no field effect runs.
        { binding: 15, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 16, visibility: GPUShaderStage.FRAGMENT, texture: {} },
      ],
    })
    this.fallbackEquirectTexture = this.device.createTexture({
      label: "equirect fallback",
      size: [1, 1],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.fallbackEquirectView = this.fallbackEquirectTexture.createView()

    this.compositePipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.compositeBindGroupLayout],
    })
    const compositeShader = this.device.createShaderModule({
      label: "composite shader",
      code: buildCompositeShader(null),
    })
    this.compositePipelineIdentity = this.makeCompositePipeline(compositeShader, false, "composite pipeline (gamma=1)")
    this.compositePipelineGamma = this.makeCompositePipeline(compositeShader, true, "composite pipeline (gamma!=1)")

    // GPU vertex-morph compute pipeline (shared by all models; per-model bind groups).
    // Bindings: 0-4 read-only storage (base pos, CSR rowStart/colMorph/colOffset, weights),
    // 5 read-write storage (vertex buffer), 6 uniform (params).
    const roStorage = { type: "read-only-storage" as const }
    this.morphComputeBindGroupLayout = this.device.createBindGroupLayout({
      label: "morph compute bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: roStorage },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: roStorage },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: roStorage },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: roStorage },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: roStorage },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      ],
    })
    this.morphComputePipeline = this.device.createComputePipeline({
      label: "morph compute pipeline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.morphComputeBindGroupLayout] }),
      compute: {
        module: this.device.createShaderModule({ label: "morph compute shader", code: MORPH_COMPUTE_WGSL }),
        entryPoint: "cs",
      },
    })

    // GPU frustum cull. One pipeline for the whole scene; the bind group is rebuilt
    // with the buffers whenever the draw list changes. See shaders/passes/cull.ts.
    this.cullBindGroupLayout = this.device.createBindGroupLayout({
      label: "cull bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: roStorage },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: roStorage },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    })
    // Scoped, unlike the morph pipeline beside it, because this one is OPTIONAL:
    // nothing renders from it. An invalid compute pipeline poisons the command
    // encoder it is set on, so a WGSL slip here would take the whole frame down —
    // every pass, every model, an unrelated-looking cascade of style-group and
    // effect failures. Catching it turns that into "culling is off".
    this.device.pushErrorScope("validation")
    this.cullPipeline = this.device.createComputePipeline({
      label: "cull pipeline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.cullBindGroupLayout] }),
      compute: {
        module: this.device.createShaderModule({ label: "cull compute shader", code: CULL_COMPUTE_WGSL }),
        entryPoint: "cs",
      },
    })
    void this.device.popErrorScope().then((err) => {
      if (!err) return
      console.error(`[cull] pipeline failed to compile — frustum culling disabled:\n${err.message}`)
      this.cullPipeline = null
    })

    this.bloomPassDescriptor = {
      label: "bloom pass",
      colorAttachments: [
        {
          view: undefined as unknown as GPUTextureView,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    } as GPURenderPassDescriptor

    const pickShaderModule = this.device.createShaderModule({
      label: "pick shader",
      code: PICK_SHADER_WGSL,
    })

    this.pickPerFrameBindGroupLayout = this.device.createBindGroupLayout({
      label: "pick per-frame layout",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
    })
    this.pickPerInstanceBindGroupLayout = this.device.createBindGroupLayout({
      label: "pick per-instance layout",
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }],
    })
    this.pickPerMaterialBindGroupLayout = this.device.createBindGroupLayout({
      label: "pick per-material layout",
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
    })

    const pickPipelineLayout = this.device.createPipelineLayout({
      label: "pick pipeline layout",
      bindGroupLayouts: [
        this.pickPerFrameBindGroupLayout,
        this.pickPerInstanceBindGroupLayout,
        this.pickPerMaterialBindGroupLayout,
      ],
    })

    this.pickPerFrameBindGroup = this.device.createBindGroup({
      label: "pick per-frame bind group",
      layout: this.pickPerFrameBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
    })

    this.pickPipeline = this.device.createRenderPipeline({
      label: "pick pipeline",
      layout: pickPipelineLayout,
      vertex: { module: pickShaderModule, buffers: fullVertexBuffers },
      fragment: {
        module: pickShaderModule,
        targets: [{ format: "rgba8unorm" }],
      },
      primitive: { cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    this.pickReadbackBuffer = this.device.createBuffer({
      label: "pick readback",
      size: 256,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })
  }

  // Step 3: Setup canvas resize handling.
  // The observer only flags the resize; render() applies it at the top of the next
  // frame. Resizing inside the RO callback (post-layout) clears the canvas buffer
  // after the frame's rAF draw already ran, so during continuous drags (resizable
  // panels) every paint showed a stale-aspect or cleared buffer — one frame behind,
  // reading as laggy/stretchy. Flag-and-apply keeps resize + redraw in one frame.
  private setupResize() {
    this.resizeObserver = new ResizeObserver(() => {
      this.resizePending = true
    })
    this.resizeObserver.observe(this.canvas)
    this.handleResize()

    // Setup raycasting double-click handler for desktop
    if (this.onRaycast) {
      this.canvas.addEventListener("dblclick", this.handleCanvasDoubleClick)
      this.canvas.addEventListener("touchend", this.handleCanvasTouch)
    }

    // Gizmo drag. mousedown registered in capture phase so we can consume the
    // event via stopImmediatePropagation before the camera's mousedown handler
    // runs (both listen on the canvas). move/up on window so drag tracks even
    // if the cursor leaves the canvas.
    this.canvas.addEventListener("mousedown", this.handleGizmoMouseDown, { capture: true })
    window.addEventListener("mousemove", this.handleGizmoMouseMove)
    window.addEventListener("mouseup", this.handleGizmoMouseUp)
  }

  /** When set, render resolution is pinned to this size instead of tracking the
   *  canvas's CSS size × devicePixelRatio (see setRenderSize). */
  private fixedRenderSize: { width: number; height: number } | null = null

  /**
   * Pin the render resolution (canvas backing store + every render target) to an
   * explicit size, decoupled from the canvas's CSS layout size — for offline
   * rendering at arbitrary resolution (video export). On screen the browser scales
   * the buffer to the layout box, so the canvas may display letterboxed/stretched
   * while pinned; hosts typically cover it with an export overlay. Pass null to
   * return to CSS-size × devicePixelRatio tracking. Applies immediately (targets
   * rebuild before this returns), so the next render() is at the new size.
   */
  setRenderSize(width: number, height: number): void
  setRenderSize(size: null): void
  setRenderSize(widthOrNull: number | null, height?: number): void {
    this.fixedRenderSize =
      widthOrNull === null
        ? null
        : { width: Math.max(1, Math.floor(widthOrNull)), height: Math.max(1, Math.floor(height ?? 1)) }
    this.resizePending = false
    this.handleResize()
  }

  private handleResize() {
    // Fixed override (offline/video rendering) wins; otherwise track CSS size × dpr.
    const dpr = window.devicePixelRatio || 1
    const width = this.fixedRenderSize ? this.fixedRenderSize.width : Math.floor(this.canvas.clientWidth * dpr)
    const height = this.fixedRenderSize ? this.fixedRenderSize.height : Math.floor(this.canvas.clientHeight * dpr)

    if (!this.multisampleTexture || this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width
      this.canvas.height = height
      // bgResolution() reads the canvas size from the composite uniforms —
      // refresh on resize or effects aspect-correct against the stale size.
      if (this.compositeUniformBuffer) this.writeCompositeViewUniforms()

      this.multisampleTexture = this.device.createTexture({
        label: "multisample HDR render target",
        size: [width, height],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })

      this.hdrResolveTexture = this.device.createTexture({
        label: "HDR resolve target",
        size: [width, height],
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })

      // rgba16float explicitly, NOT hdrFormat: the composite reads this layer's
      // ALPHA to composite it over the frame, and an rg11b10 hdr fallback has no
      // alpha channel to read.
      this.trailLayerTexture?.destroy()
      this.trailLayerTexture = this.device.createTexture({
        label: "trail layer",
        size: [width, height],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.trailLayerView = this.trailLayerTexture.createView()

      // The field layer — half resolution by default, full for @fullres effects.
      this.fieldFullW = width
      this.fieldFullH = height
      this.createFieldTargets()

      // Bloom-mask MRT attachments — same dims + MSAA as HDR so they share the render pass.
      // MS buffer gets resolved into maskResolveTexture, which the bloom blit pass samples.
      this.multisampleMaskTexture = this.device.createTexture({
        label: "multisample bloom mask",
        size: [width, height],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: Engine.BLOOM_MASK_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.maskResolveTexture = this.device.createTexture({
        label: "bloom mask resolve",
        size: [width, height],
        format: Engine.BLOOM_MASK_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.maskResolveView = this.maskResolveTexture.createView()

      // Bloom pyramid: mip 0 is half-res, each subsequent mip halves again.
      // Mip count chosen so the coarsest mip is ≥4 px on the short side, capped at BLOOM_MAX_LEVELS.
      const bw = Math.max(1, Math.floor(width / 2))
      const bh = Math.max(1, Math.floor(height / 2))
      const shortSide = Math.max(1, Math.min(bw, bh))
      this.bloomMipCount = Math.max(1, Math.min(Engine.BLOOM_MAX_LEVELS, Math.floor(Math.log2(shortSide)) - 1))
      this.bloomDownTexture = this.device.createTexture({
        label: "bloom down pyramid",
        size: [bw, bh],
        mipLevelCount: this.bloomMipCount,
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.bloomUpTexture = this.device.createTexture({
        label: "bloom up pyramid",
        size: [bw, bh],
        mipLevelCount: Math.max(1, this.bloomMipCount - 1),
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.bloomDownMipViews = []
      for (let i = 0; i < this.bloomMipCount; i++) {
        this.bloomDownMipViews.push(this.bloomDownTexture.createView({ baseMipLevel: i, mipLevelCount: 1 }))
      }
      this.bloomUpMipViews = []
      const upLevels = Math.max(1, this.bloomMipCount - 1)
      for (let i = 0; i < upLevels; i++) {
        this.bloomUpMipViews.push(this.bloomUpTexture.createView({ baseMipLevel: i, mipLevelCount: 1 }))
      }

      this.depthTexture = this.device.createTexture({
        label: "depth texture",
        size: [width, height],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: "depth24plus-stencil8",
        // TEXTURE_BINDING for the DoF gather — a usage flag, not a copy; the
        // zero-cost-when-off story lives in depthStoreOp, not here.
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })

      const depthTextureView = this.depthTexture.createView()
      this.depthReadView = this.depthTexture.createView({ aspect: "depth-only" })
      this.rebindTrails()

      // storeOp="discard" on MSAA views keeps per-sample data in Apple TBDR tile memory —
      // only the resolveTarget (hdrResolveTexture / maskResolveView) gets written to RAM.
      // With storeOp="store" Safari's Metal backend spills the full MS buffer every frame
      // (rgba16f × 4 samples on a 4K canvas ≈ 256 MB/frame of dead bandwidth).
      const colorAttachment: GPURenderPassColorAttachment = {
        view: this.multisampleTexture.createView(),
        resolveTarget: this.hdrResolveTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "discard",
      }

      const maskAttachment: GPURenderPassColorAttachment = {
        view: this.multisampleMaskTexture.createView(),
        resolveTarget: this.maskResolveView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "discard",
      }

      this.renderPassDescriptor = {
        label: "renderPass",
        colorAttachments: [colorAttachment, maskAttachment],
        depthStencilAttachment: {
          view: depthTextureView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          // Main-pass depth is not sampled later (shadow uses its own map, composite is depthless).
          depthStoreOp: "discard",
          stencilClearValue: 0,
          stencilLoadOp: "clear",
          stencilStoreOp: "discard",
        },
      }

      // Composite pass descriptor (color attachment view patched per-frame to current swapchain).
      this.compositePassDescriptor = {
        label: "composite pass",
        colorAttachments: [
          {
            view: undefined as unknown as GPUTextureView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      }

      // Selection mask: single-channel canvas-res texture. Depth-always (no depth attachment).
      this.selectionMaskTexture = this.device.createTexture({
        label: "selection mask",
        size: [width, height],
        format: "r8unorm",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.selectionMaskView = this.selectionMaskTexture.createView()
      this.selectionMaskPassDescriptor = {
        label: "selection mask pass",
        colorAttachments: [
          {
            view: this.selectionMaskView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          },
        ],
      }
      this.selectionEdgeBindGroup = this.device.createBindGroup({
        label: "selection edge bind group",
        layout: this.selectionEdgeBindGroupLayout,
        entries: [
          { binding: 0, resource: this.selectionMaskView },
          { binding: 1, resource: this.selectionSampler },
          { binding: 2, resource: { buffer: this.selectionEdgeUniformBuffer } },
        ],
      })
      // Edge pass draws on top of the composite output — load-store on swapchain.
      this.selectionEdgePassDescriptor = {
        label: "selection edge pass",
        colorAttachments: [
          {
            view: undefined as unknown as GPUTextureView,
            loadOp: "load",
            storeOp: "store",
          },
        ],
      }

      this.writeBloomUniforms()

      if (this.compositeBindGroupLayout && this.bloomBlitBindGroupLayout) {
        // Blit: reads HDR resolve texture (full-res), writes bloomDown mip 0.
        this.bloomBlitBindGroup = this.device.createBindGroup({
          label: "bloom blit bind group",
          layout: this.bloomBlitBindGroupLayout,
          entries: [
            { binding: 0, resource: this.hdrResolveTexture.createView() },
            { binding: 1, resource: { buffer: this.bloomBlitUniformBuffer } },
            { binding: 2, resource: this.maskResolveView },
          ],
        })
        // Downsample[i] reads bloomDown mip (i-1), writes bloomDown mip i. i ∈ [1..N-1].
        this.bloomDownsampleBindGroups = []
        for (let i = 1; i < this.bloomMipCount; i++) {
          this.bloomDownsampleBindGroups.push(
            this.device.createBindGroup({
              label: `bloom downsample ${i}`,
              layout: this.bloomDownsampleBindGroupLayout,
              entries: [
                { binding: 0, resource: this.bloomDownMipViews[i - 1] },
                { binding: 1, resource: this.bloomSampler },
              ],
            }),
          )
        }
        // Upsample[i] writes bloomUp mip i. Coarsest step reads bloomDown[N-1] (no prior up yet);
        // subsequent steps read bloomUp[i+1]. Both read bloomDown[i] as the base (additive combine).
        this.bloomUpsampleBindGroups = []
        const topIdx = this.bloomMipCount - 2
        for (let i = topIdx; i >= 0; i--) {
          const srcView = i === topIdx ? this.bloomDownMipViews[this.bloomMipCount - 1] : this.bloomUpMipViews[i + 1]
          this.bloomUpsampleBindGroups.push(
            this.device.createBindGroup({
              label: `bloom upsample ${i}`,
              layout: this.bloomUpsampleBindGroupLayout,
              entries: [
                { binding: 0, resource: srcView },
                { binding: 1, resource: this.bloomDownMipViews[i] },
                { binding: 2, resource: this.bloomSampler },
                { binding: 3, resource: { buffer: this.bloomUpsampleUniformBuffer } },
              ],
            }),
          )
        }
        // Composite reads bloomUp mip 0 (full pyramid collapsed); fallback to bloomDown mip 0 if no upsample level.
        this.compositeBloomView = this.bloomMipCount > 1 ? this.bloomUpMipViews[0] : this.bloomDownMipViews[0]
        this.rebuildCompositeBindGroup()
      }

      this.writeCompositeViewUniforms()

      this.camera.aspect = width / height

      if (this.onRaycast) {
        this.pickTexture = this.device.createTexture({
          label: "pick render target",
          size: [width, height],
          format: "rgba8unorm",
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC,
        })
        this.pickDepthTexture = this.device.createTexture({
          label: "pick depth",
          size: [width, height],
          format: "depth24plus",
          usage: GPUTextureUsage.RENDER_ATTACHMENT,
        })
      }
    }
  }

  // Builds the gizmo pipeline, its shared transform bind group, 3 per-color bind
  // groups (R/G/B), and the packed triangle-list vertex buffer. Each original
  // line segment is expanded to 6 verts (2 triangles) carrying (pos, dir, side)
  // so the VS can extrude to a uniform pixel-width ribbon.
  private setupGizmo() {
    const SEG = Engine.GIZMO_RING_SEGMENTS
    const R = Engine.GIZMO_RING_RADIUS
    const ringVerts = SEG * 6
    this.gizmoDraws = [
      { first: 0, count: 6, color: 0 }, // X axis
      { first: 6, count: 6, color: 1 }, // Y axis
      { first: 12, count: 6, color: 2 }, // Z axis
      { first: 18, count: ringVerts, color: 0 }, // X ring (YZ plane)
      { first: 18 + ringVerts, count: ringVerts, color: 1 }, // Y ring (XZ plane)
      { first: 18 + 2 * ringVerts, count: ringVerts, color: 2 }, // Z ring (XY plane)
    ]
    const verts: number[] = []
    // Per-vertex layout: pos(3), segDir(3), side(1), axisT(1) = 8 floats.
    // axisT encodes "parameter along the axis" for axis verts (0 at center, 1
    // at tip). Ring verts use -1 as a "not an axis" flag the FS uses to skip
    // the dash + fade treatment.
    const pushSeg = (p0: [number, number, number], p1: [number, number, number], t0: number, t1: number) => {
      const d = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]]
      const dn = [-d[0], -d[1], -d[2]]
      verts.push(p0[0], p0[1], p0[2], d[0], d[1], d[2], -1, t0)
      verts.push(p0[0], p0[1], p0[2], d[0], d[1], d[2], 1, t0)
      verts.push(p1[0], p1[1], p1[2], dn[0], dn[1], dn[2], -1, t1)
      verts.push(p0[0], p0[1], p0[2], d[0], d[1], d[2], 1, t0)
      verts.push(p1[0], p1[1], p1[2], dn[0], dn[1], dn[2], 1, t1)
      verts.push(p1[0], p1[1], p1[2], dn[0], dn[1], dn[2], -1, t1)
    }
    // Axes (open). t = 0 at center → 1 at tip. FS dashes + dims the inside-ring part.
    const L = Engine.GIZMO_AXIS_LENGTH
    pushSeg([0, 0, 0], [L, 0, 0], 0, 1)
    pushSeg([0, 0, 0], [0, L, 0], 0, 1)
    pushSeg([0, 0, 0], [0, 0, L], 0, 1)
    // Rings (closed). t = -1 signals "not an axis".
    for (let plane = 0; plane < 3; plane++) {
      for (let i = 0; i < SEG; i++) {
        const t0 = (i / SEG) * Math.PI * 2
        const t1 = ((i + 1) / SEG) * Math.PI * 2
        const c0 = Math.cos(t0) * R,
          s0 = Math.sin(t0) * R
        const c1 = Math.cos(t1) * R,
          s1 = Math.sin(t1) * R
        if (plane === 0) pushSeg([0, c0, s0], [0, c1, s1], -1, -1)
        else if (plane === 1) pushSeg([s0, 0, c0], [s1, 0, c1], -1, -1)
        else pushSeg([c0, s0, 0], [c1, s1, 0], -1, -1)
      }
    }
    const geom = new Float32Array(verts)
    this.gizmoVertexBuffer = this.device.createBuffer({
      label: "gizmo vertex buffer",
      size: geom.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.gizmoVertexBuffer, 0, geom)

    // Shared transform+viewport+thickness uniform. Rewritten per frame.
    this.gizmoTransformBuffer = this.device.createBuffer({
      label: "gizmo transform",
      size: 80, // mat4 (64) + vec2 viewport (8) + thickness f32 (4) + pad (4)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    const bg0Layout = this.device.createBindGroupLayout({
      label: "gizmo group 0 layout (camera + transform)",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    })
    const bg1Layout = this.device.createBindGroupLayout({
      label: "gizmo group 1 layout (color)",
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }],
    })
    const pipelineLayout = this.device.createPipelineLayout({
      label: "gizmo pipeline layout",
      bindGroupLayouts: [bg0Layout, bg1Layout],
    })
    const shader = this.device.createShaderModule({ label: "gizmo shader", code: GIZMO_SHADER_WGSL })
    this.gizmoPipeline = this.device.createRenderPipeline({
      label: "gizmo pipeline",
      layout: pipelineLayout,
      vertex: {
        module: shader,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: 8 * 4, // pos(3) + segDir(3) + side(1) + axisT(1) = 8 floats
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat }, // position
              { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat }, // segDir
              { shaderLocation: 2, offset: 6 * 4, format: "float32" as GPUVertexFormat }, // side
              { shaderLocation: 3, offset: 7 * 4, format: "float32" as GPUVertexFormat }, // axisT
            ],
          },
        ],
      },
      fragment: {
        module: shader,
        entryPoint: "fs",
        targets: [
          {
            format: this.presentationFormat,
            blend: {
              color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      multisample: { count: 1 },
    })

    this.gizmoBindGroup0 = this.device.createBindGroup({
      label: "gizmo bind group 0",
      layout: bg0Layout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.gizmoTransformBuffer } },
      ],
    })

    // Vivid game-UI palette. FS applies an edge-to-center alpha falloff so these
    // full-saturation colors stay readable without feeling flat. Pipeline writes
    // straight to the LDR swapchain (no tonemap), so values > 1 clamp.
    const colors = [
      new Float32Array([1.0, 0.24, 0.38, 1.0]), // X: warm red, slight pink
      new Float32Array([0.35, 0.95, 0.52, 1.0]), // Y: emerald
      new Float32Array([0.33, 0.62, 1.0, 1.0]), // Z: azure
    ]
    this.gizmoColorBindGroups = []
    for (let i = 0; i < 3; i++) {
      const buf = this.device.createBuffer({
        label: `gizmo color ${i}`,
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(buf, 0, colors[i])
      this.gizmoColorBindGroups.push(
        this.device.createBindGroup({
          label: `gizmo color bg ${i}`,
          layout: bg1Layout,
          entries: [{ binding: 0, resource: { buffer: buf } }],
        }),
      )
    }

    // Gizmo pass — depth-less, loads the swapchain so it composites on top.
    this.gizmoPassDescriptor = {
      label: "gizmo pass",
      colorAttachments: [
        {
          view: undefined as unknown as GPUTextureView,
          loadOp: "load",
          storeOp: "store",
        },
      ],
    }
  }

  // Step 4: Create camera and uniform buffer
  private setupCamera() {
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 40 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // The camera came up with the engine (see the constructor). What waits for
    // init is only what needs a device and a sized canvas.
    this.camera.aspect = this.canvas.width / this.canvas.height
    this.camera.attachControl(this.canvas)
  }

  /** Set static camera look-at / orbit center. Clears any model follow binding. */
  setCameraTarget(v: Vec3): void
  /** Bind camera orbit center to a model's bone (Souls-style follow cam). Pass null to unbind. */
  setCameraTarget(model: Model | null, boneName: string, offset?: Vec3): void
  setCameraTarget(modelOrVec: Model | Vec3 | null, boneName?: string, offset?: Vec3): void {
    if (modelOrVec === null) {
      this.cameraTargetModel = null
      return
    }
    if ("x" in modelOrVec && "y" in modelOrVec && "z" in modelOrVec) {
      this.cameraTargetModel = null
      this.camera.target.x = modelOrVec.x
      this.camera.target.y = modelOrVec.y
      this.camera.target.z = modelOrVec.z
      return
    }
    this.cameraTargetModel = modelOrVec
    this.cameraTargetBoneName = boneName ?? ""
    this.cameraTargetOffset.x = offset?.x ?? 0
    this.cameraTargetOffset.y = offset?.y ?? 0
    this.cameraTargetOffset.z = offset?.z ?? 0
  }

  /** Souls-style follow cam: orbit center tracks a model bone each frame. Shorthand for setCameraTarget(model, boneName, offset). */
  setCameraFollow(model: Model | null, boneName?: string, offset?: Vec3, smoothing?: number): void {
    if (model === null) {
      this.cameraTargetModel = null
      return
    }
    this.cameraTargetModel = model
    this.cameraTargetBoneName = boneName ?? "全ての親"
    this.cameraTargetOffset.x = offset?.x ?? 0
    this.cameraTargetOffset.y = offset?.y ?? 0
    this.cameraTargetOffset.z = offset?.z ?? 0
    // Handheld feel: seconds for the camera to close ~63% of the gap to the
    // bone (exponential, frame-rate independent). 0 = rigid instant follow.
    this.cameraFollowSmoothing = Math.max(0, smoothing ?? 0)
    this.cameraFollowSeeded = false
  }

  // ── VMD camera track ──
  // A dedicated camera VMD (target / rotation / distance / fov animated). Motion VMDs loaded
  // via model.loadVmd never touch the camera — the camera shot is opt-in through here.

  /** Load a camera VMD (dedicated camera file, or any VMD's camera block) and drive the shot
   *  from it. Default-on once a non-empty track loads; toggle with setCameraVmdEnabled. */
  async loadCameraVmd(url: string): Promise<void> {
    const frames = await VMDLoader.loadCamera(url)
    this.cameraAnimation = frames.length ? new CameraAnimation(frames) : null
    this.camera.setVmdDriven(this.cameraAnimation !== null)
  }

  /** Load a camera VMD from an already-fetched buffer (e.g. a File the user dropped). */
  loadCameraVmdFromBuffer(buffer: ArrayBuffer): void {
    const frames = VMDLoader.loadCameraFromBuffer(buffer)
    this.cameraAnimation = frames.length ? new CameraAnimation(frames) : null
    this.camera.setVmdDriven(this.cameraAnimation !== null)
  }

  /** Turn the loaded camera VMD on/off (falls back to orbit when off). No-op if none loaded. */
  setCameraVmdEnabled(enabled: boolean): void {
    this.camera.setVmdDriven(enabled && this.cameraAnimation !== null)
    if (!enabled && this.cameraTargetModel) {
      // Follow resumes with a clean snap to bone + configured offset — one
      // predictable cut to the scene's framing, no easing from the shot.
      this.cameraFollowSeeded = false
    }
  }

  /** True while the orbit target is riding a model bone (setCameraFollow). */
  isCameraFollowing(): boolean {
    return this.cameraTargetModel !== null
  }

  /** True while the loaded camera VMD is actively driving the shot. */
  isCameraVmdEnabled(): boolean {
    return this.camera.vmdDriven
  }

  /** True if a (non-empty) camera VMD is loaded, regardless of enabled state. */
  hasCameraVmd(): boolean {
    return this.cameraAnimation !== null
  }

  /** Seconds the loaded camera VMD runs for — its last keyframe — or 0 with none
   *  loaded. A timeline cannot draw a lane to scale without it, and the camera's
   *  length is its own: it does not have to match any model's clip. */
  getCameraVmdDuration(): number {
    return this.cameraAnimation?.duration ?? 0
  }

  /**
   * Install a track's precomputed analysis for the rzAudio* effect functions:
   * `data` is frames × (2 + bands) floats — loudness, bass onset, then the band
   * magnitudes, all 0..1 — sampled by the clock given to setAudioTime. Null
   * clears back to silence.
   *
   * Precomputed for the WHOLE track, never fed live from an analyser: an export
   * steps the engine frame by frame rather than playing in real time, so live
   * analysis would render silence into the exported video.
   */
  setAudioData(data: Float32Array | null, bandsPerFrame: number, secondsPerFrame: number): void {
    if (this.audioBuffer !== this.audioFallbackBuffer) this.audioBuffer.destroy()
    if (!data || data.length === 0) {
      this.audioBuffer = this.audioFallbackBuffer
    } else {
      const frames = Math.floor(data.length / (bandsPerFrame + 2))
      const payload = new Float32Array(8 + data.length)
      payload[0] = frames
      payload[1] = bandsPerFrame
      payload[2] = secondsPerFrame
      payload.set(data, 8)
      this.audioBuffer = this.device.createBuffer({
        label: "audio analysis",
        size: payload.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(this.audioBuffer, 0, payload)
    }
    // Every consumer holds the buffer by reference in a bind group; all of them
    // re-bind so audio arriving after an effect (or before one) both work.
    this.rebuildCompositeBindGroup()
    this.rebindTrails()
    if (this.particles) {
      const b = this.particles.rebind()
      this.particles.computeBind = b.computeBind
      this.particles.renderBind = b.renderBind
    }
  }

  /**
   * Where the track is NOW, in seconds — written by whoever owns playback: the
   * editor's audio clock, the viewer's, or the export loop with its exact
   * per-frame time. A 4-byte header write, cheap enough for every frame.
   */
  setAudioTime(seconds: number, playing = true): void {
    if (this.audioBuffer === this.audioFallbackBuffer) return
    this.audioTimeScratch[0] = seconds
    this.audioTimeScratch[1] = playing ? 1 : 0
    this.device.queue.writeBuffer(this.audioBuffer, 12, this.audioTimeScratch)
  }

  /** Every camera keyframe's frame index — what a timeline draws as its cuts.
   *  Empty when no camera VMD is loaded. */
  getCameraVmdKeyframes(): number[] {
    return this.cameraAnimation?.keyframeIndices() ?? []
  }

  /** Drop the loaded camera VMD and return to orbit control. */
  clearCameraVmd(): void {
    this.cameraAnimation = null
    this.camera.setVmdDriven(false)
  }

  // Clock the camera VMD runs on: the first model with an active clip (playing or scrubbed),
  // so a static stage in the scene never freezes the shot at frame 0. Falls back to the first
  // model, then to 0 (empty scene).
  private cameraClockTime(): number {
    let fallback: number | null = null
    for (const inst of this.modelInstances.values()) {
      // Stages are skipped outright. Scenery carries no motion, and it is added
      // BEFORE the cast — it paints while the models stream in behind it — so it
      // is first in insertion order and was seeding this clock with its own
      // permanent zero. In a scene with a stage, a camera VMD therefore sampled
      // frame 0 forever and the shot never moved.
      if (inst.isStage) continue
      const p = inst.model.getAnimationProgress()
      if (p.playing || p.paused) return p.current
      // Otherwise the first cast member that actually HAS a clip: one still at
      // bind pose must not claim the clock from one holding the motion.
      if (fallback === null && p.duration > 0) fallback = p.current
    }
    return fallback ?? 0
  }

  /** Current orbit eye position (spherical coords resolved to a point). */
  getCameraPosition(): Vec3 {
    return this.camera.getPosition()
  }

  getCameraDistance(): number {
    return this.camera.radius
  }
  setCameraDistance(d: number): void {
    this.camera.radius = d
  }
  getCameraAlpha(): number {
    return this.camera.alpha
  }
  setCameraAlpha(a: number): void {
    this.camera.alpha = a
  }
  getCameraBeta(): number {
    return this.camera.beta
  }
  setCameraBeta(b: number): void {
    this.camera.beta = b
  }
  /** Vertical field of view in radians (default π/4). While a camera VMD
   *  drives the view it animates fov itself; the orbit value set here is
   *  restored when the VMD releases the camera. */
  getCameraFov(): number {
    return this.camera.fov
  }
  setCameraFov(fov: number): void {
    this.camera.fov = fov
  }

  // Step 5: Create lighting buffers
  private setupLighting() {
    this.lightUniformBuffer = this.device.createBuffer({
      label: "light uniforms",
      size: 64 * 4, // ambientColor vec4f (4) + 4 lights * 2 vec4f each (32) = 36 f32 padded to 64
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.lightData.fill(0)
    this.lightCount = 0
    this.writeWorld()
    this.writeSun(0)
  }

  /**
   * Write world ambient. For a uniform-radiance world, hemispherical irradiance
   * is E = π·L and a Lambertian BRDF reflects (albedo/π)·E = albedo·L, so the
   * shader's ambient uniform is just `world.color × world.strength` — no /π.
   */
  private writeWorld() {
    const s = this.world.strength
    this.lightData[0] = this.world.color.x * s
    this.lightData[1] = this.world.color.y * s
    this.lightData[2] = this.world.color.z * s
    this.lightData[3] = 0
    this.updateLightBuffer()
  }

  /** Write sun lamp into light slot `index` (0..3). Layout mirrors the WGSL struct. */
  private writeSun(index: number) {
    if (index < 0 || index >= 4) return
    const normalized = this.sun.direction.normalize()
    const base = 4 + index * 8 // 8 floats per light (direction vec4, color vec4)
    this.lightData[base] = normalized.x
    this.lightData[base + 1] = normalized.y
    this.lightData[base + 2] = normalized.z
    this.lightData[base + 3] = 0
    this.lightData[base + 4] = this.sun.color.x
    this.lightData[base + 5] = this.sun.color.y
    this.lightData[base + 6] = this.sun.color.z
    this.lightData[base + 7] = this.sun.strength
    if (index >= this.lightCount) this.lightCount = index + 1
    this.updateLightBuffer()
  }

  /** Update the world environment (Blender: World Background). Ambient recomputes immediately. */
  setWorld(options: WorldOptions): void {
    if (options.color) this.world.color = options.color
    if (options.strength !== undefined) this.world.strength = options.strength
    this.writeWorld()
  }

  /** Update the sun lamp (Blender: Light > Sun). Direction change marks shadow VP dirty. */
  setSun(options: SunOptions): void {
    if (options.color) this.sun.color = options.color
    if (options.strength !== undefined) this.sun.strength = options.strength
    if (options.direction) {
      this.sun.direction = options.direction
      this.shadowLightVPDirty = true
    }
    this.writeSun(0)
  }

  getWorld(): Readonly<{ color: Vec3; strength: number }> {
    return this.world
  }
  getSun(): Readonly<{ color: Vec3; strength: number; direction: Vec3 }> {
    return this.sun
  }

  addGround(options?: {
    width?: number
    height?: number
    diffuseColor?: Vec3
    fadeStart?: number
    fadeEnd?: number
    shadowStrength?: number
    gridSpacing?: number
    gridLineWidth?: number
    gridLineOpacity?: number
    gridLineColor?: Vec3
    noiseStrength?: number
    /** Whole-ground opacity, 0–1 (multiplies the radial edge fade). Default 1. */
    opacity?: number
  }): void {
    const opts = {
      width: 160,
      height: 160,
      diffuseColor: new Vec3(0.9, 0.1, 1.0),
      fadeStart: 10.0,
      fadeEnd: 80.0,
      shadowStrength: 1.0,
      gridSpacing: 4.2,
      gridLineWidth: 0.012,
      gridLineOpacity: 0.4,
      gridLineColor: new Vec3(0.85, 0.85, 0.85),
      noiseStrength: 0.05,
      opacity: 1.0,
      ...options,
    }
    this.createGroundGeometry(opts.width, opts.height)
    this.createShadowGroundResources(opts)
    this.hasGround = true
    this.groundDrawCall = {
      type: "ground",
      count: 6,
      firstIndex: 0,
      bindGroup: this.groundShadowBindGroup!,
      materialName: "Ground",
      groupId: null,
      // The ground belongs to no model instance, so it is not in the cull list —
      // cullIndex -1 leaves renderGround unconditional. Its box is filled in
      // anyway rather than left a lie for whoever reads this next.
      bounds: new Float32Array([-opts.width / 2, 0, -opts.height / 2, opts.width / 2, 0, opts.height / 2]),
      cullIndex: -1,
    }
  }

  private updateLightBuffer() {
    this.device.queue.writeBuffer(this.lightUniformBuffer, 0, this.lightData)
  }

  getStats(): EngineStats {
    return { ...this.stats }
  }

  // The render loop runs at display rate, always. A frame-rate cap used to be
  // offered here for high-refresh displays, on the reasoning that VMD content
  // is 30fps and running the whole pipeline at 240 buys nothing. Nothing ever
  // called it — every product in the family chases native refresh, because
  // dropping frames a display can show is the one thing that reads as cheap.
  // Physics already decouples: it steps at a fixed 60Hz behind an accumulator
  // and the drawn pose is interpolated between substeps, so a 240Hz display
  // costs four interpolations per simulation step, not four simulations.
  runRenderLoop(callback?: () => void) {
    this.renderLoopCallback = callback || null

    const loop = () => {
      this.animationFrameId = requestAnimationFrame(loop)
      this.render()
      if (this.renderLoopCallback) {
        this.renderLoopCallback()
      }
    }

    this.animationFrameId = requestAnimationFrame(loop)
  }

  stopRenderLoop() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
    this.renderLoopCallback = null
  }

  dispose() {
    this.stopRenderLoop()
    this.forEachInstance((inst) => inst.model.stop())
    if (Engine.instance === this) Engine.instance = null
    if (this.camera) this.camera.detachControl()

    // Remove raycasting event listeners
    if (this.onRaycast) {
      this.canvas.removeEventListener("dblclick", this.handleCanvasDoubleClick)
      this.canvas.removeEventListener("touchend", this.handleCanvasTouch)
    }

    // Remove gizmo drag listeners
    this.canvas.removeEventListener("mousedown", this.handleGizmoMouseDown, { capture: true })
    window.removeEventListener("mousemove", this.handleGizmoMouseMove)
    window.removeEventListener("mouseup", this.handleGizmoMouseUp)

    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }

    // Style group runtime: per-group uniform buffers (pipelines are GC'd; buffers need
    // explicit destroy). Per-model group buffers are torn down in removeModel; the shared
    // zero buffer is engine-owned.
    this.forEachInstance((inst) => {
      for (const install of inst.styleGroups.values()) this.destroyInstall(install)
      inst.styleGroups.clear()
    })
    this.zeroStyleBuffer?.destroy()
    this.releaseCullBuffers()
    this.cullFrustaBuffer?.destroy()
    this.cullFrustaBuffer = null
  }

  async loadModel(path: string): Promise<Model>
  async loadModel(name: string, path: string): Promise<Model>
  async loadModel(name: string, options: LoadModelFromFilesOptions): Promise<Model>
  async loadModel(nameOrPath: string, pathOrOptions?: string | LoadModelFromFilesOptions): Promise<Model> {
    if (pathOrOptions !== undefined && typeof pathOrOptions === "object" && "files" in pathOrOptions) {
      const name = nameOrPath
      const pmxFile = pathOrOptions.pmxFile ?? findFirstPmxFileInList(pathOrOptions.files)
      if (!pmxFile) throw new Error("No .pmx file found in the selected folder")
      const map = fileListToMap(pathOrOptions.files)
      // `||`, not `??`: flat-picked files carry webkitRelativePath === "" (see
      // fileListToMap) — `""` must fall through to the filename.
      const pmxKey = normalizeAssetPath(
        (pmxFile as File & { webkitRelativePath?: string }).webkitRelativePath || pmxFile.name,
      )
      const reader = createFileMapAssetReader(map)
      const model = await PmxLoader.loadFromReader(reader, pmxKey)
      model.setName(name)
      await this.addModel(model, pmxKey, name, reader)
      return model
    }

    const pmxPath = pathOrOptions === undefined ? nameOrPath : pathOrOptions
    const name = pathOrOptions === undefined ? "model_" + this._nextDefaultModelId++ : nameOrPath
    const model = await PmxLoader.load(pmxPath)
    model.setName(name)
    await this.addModel(model, pmxPath, name)
    return model
  }

  /** loadModel's folder/zip path for a stage. Shares the whole prelude — only
   *  what the PMX becomes differs. */
  async loadStage(
    name: string,
    options: LoadModelFromFilesOptions & { transform?: Partial<ModelTransform> },
  ): Promise<Model> {
    const { model, pmxKey, reader } = await this.openPmxFromFiles(name, options)
    await this.addStage(model, pmxKey, { name, transform: options.transform, assetReader: reader })
    return model
  }

  /** Read a PMX out of a picked folder / expanded zip. Shared by loadModel and
   *  loadStage so the file-map and path handling exist in exactly one place. */
  private async openPmxFromFiles(
    name: string,
    options: LoadModelFromFilesOptions,
  ): Promise<{ model: Model; pmxKey: string; reader: AssetReader }> {
    const pmxFile = options.pmxFile ?? findFirstPmxFileInList(options.files)
    if (!pmxFile) throw new Error("No .pmx file found in the selected folder")
    const map = fileListToMap(options.files)
    // `||`, not `??`: flat-picked files carry webkitRelativePath === "" (see
    // fileListToMap) — `""` must fall through to the filename.
    const pmxKey = normalizeAssetPath(
      (pmxFile as File & { webkitRelativePath?: string }).webkitRelativePath || pmxFile.name,
    )
    const reader = createFileMapAssetReader(map)
    const model = await PmxLoader.loadFromReader(reader, pmxKey)
    model.setName(name)
    return { model, pmxKey, reader }
  }

  async addModel(
    model: Model,
    pmxPath: string,
    name?: string,
    assetReader?: AssetReader,
    options?: { stage?: boolean },
  ): Promise<string> {
    const requested = name ?? model.name
    let key = requested
    let n = 1
    while (this.modelInstances.has(key)) {
      key = `${requested}_${n++}`
    }
    const reader = assetReader ?? createFetchAssetReader()
    const basePath = deriveBasePathFromPmxPath(pmxPath)
    model.setAssetContext(reader, basePath)
    await this.setupModelInstance(key, model, basePath, reader, options?.stage ?? false)
    return key
  }

  /**
   * Add a PMX as the scene's environment rather than as a character.
   *
   * A stage is the same geometry and the same materials — style groups and
   * shader graphs work on it unchanged, which is the whole reason pure-PMX
   * stages are worth supporting — but it is not a performer:
   *
   *  - no physics. A stage's rigidbodies are set dressing for MMD's solver and
   *    cost a full simulation island for scenery that never moves.
   *  - no IK. Nothing drives a stage's chains, and solving them every frame is
   *    pure waste on what is usually the heaviest mesh in the scene.
   *  - no per-frame pose work while it is idle: with no clip and no morph
   *    change there is nothing to recompute, so update is skipped entirely.
   *  - it owns the floor. See groundIsSuppressed — the built-in ground plane
   *    and a stage's own floor both sit at y=0 and z-fight.
   *
   * Bone and material morphs still apply, because that is how a stage's doors,
   * lifts and colour switches are rigged.
   */
  async addStage(
    model: Model,
    pmxPath: string,
    options?: { name?: string; transform?: Partial<ModelTransform>; assetReader?: AssetReader },
  ): Promise<string> {
    const key = await this.addModel(model, pmxPath, options?.name, options?.assetReader, { stage: true })
    if (options?.transform) this.setModelTransform(key, options.transform)
    return key
  }

  /** True while a stage is in the scene, which is when the built-in ground plane
   *  must not draw. */
  groundIsSuppressed(): boolean {
    for (const inst of this.modelInstances.values()) if (inst.isStage) return true
    return false
  }

  removeModel(name: string): void {
    const inst = this.modelInstances.get(name)
    if (!inst) return
    inst.model.stop()
    for (const path of inst.textureCacheKeys) {
      const tex = this.textureCache.get(path)
      if (!tex) continue
      // The texture cache is shared across models — destroy only when no OTHER
      // live instance references the key, else the survivor submits with a
      // destroyed texture.
      let shared = false
      for (const other of this.modelInstances.values()) {
        if (other !== inst && other.textureCacheKeys.includes(path)) {
          shared = true
          break
        }
      }
      if (!shared) {
        tex.destroy()
        this.textureCache.delete(path)
        this.textureAlphaCache.delete(path)
      }
    }
    for (const buf of inst.gpuBuffers) {
      buf.destroy()
    }
    // Per-group StyleUniforms buffers aren't in gpuBuffers (allocated post-load).
    for (const install of inst.styleGroups.values()) this.destroyInstall(install)
    this.modelInstances.delete(name)
    this.cullListDirty = true
  }

  getModelNames(): string[] {
    return Array.from(this.modelInstances.keys())
  }

  getModel(name: string): Model | null {
    return this.modelInstances.get(name)?.model ?? null
  }

  /**
   * Place a model in the scene — position, rotation, uniform scale, visibility. The
   * transform is a root offset baked into skinning (moves the whole rig), so it composes
   * with animation. Use it to sit a `stage.pmx` with a character, or to fit/hide either.
   * Scale is **uniform** (normals renormalize in-shader). Don't scale a physics-driven
   * character — its colliders won't scale; scale stages (which are typically physics-free).
   */
  setModelTransform(name: string, transform: Partial<ModelTransform>): void {
    const inst = this.modelInstances.get(name)
    const model = inst?.model
    if (!inst || !model) return
    if (transform.position) model.setPosition(transform.position)
    if (transform.rotation) model.setRotation(transform.rotation)
    if (transform.scale !== undefined) model.setScale(transform.scale)
    if (transform.visible !== undefined) model.setVisible(transform.visible)
    // The root transform is baked into the skin matrices, so moving a model is a
    // reason to re-upload them even though no pose pass ran. A cast member gets
    // one every frame anyway; an idle stage would otherwise never see the change
    // — which is exactly the case this API exists to serve.
    inst.skinMatricesDirty = true
  }

  /** Read a model's scene transform (for serialization into a scene descriptor). */
  getModelTransform(name: string): ModelTransform | null {
    const model = this.modelInstances.get(name)?.model
    if (!model) return null
    const p = model.position
    return {
      position: new Vec3(p.x, p.y, p.z),
      rotation: model.rotation.clone(),
      scale: model.scale,
      visible: model.visible,
    }
  }

  markVertexBufferDirty(modelNameOrModel?: string | Model): void {
    if (modelNameOrModel === undefined) return
    if (typeof modelNameOrModel === "string") {
      const inst = this.modelInstances.get(modelNameOrModel)
      if (inst) inst.vertexBufferNeedsUpdate = true
      return
    }
    for (const inst of this.modelInstances.values()) {
      if (inst.model === modelNameOrModel) {
        inst.vertexBufferNeedsUpdate = true
        return
      }
    }
  }

  setSelectedMaterial(modelName: string | null, materialName: string | null): void {
    this.selectedMaterial = modelName && materialName ? { modelName, materialName } : null
  }

  setSelectedBone(modelName: string | null, boneName: string | null): void {
    if (!modelName || !boneName) {
      this.selectedBone = null
      return
    }
    const inst = this.modelInstances.get(modelName)
    if (!inst) {
      this.selectedBone = null
      return
    }
    const boneIndex = inst.model.getSkeleton().bones.findIndex((b) => b.name === boneName)
    this.selectedBone = boneIndex >= 0 ? { modelName, boneName, boneIndex } : null
  }

  // Build a material's bind group with binding(4) pointing at a given StyleUniforms buffer
  // (the group's buffer when grouped, or the shared zero buffer when ungrouped).
  /** A group's uniform buffer and its maps have the same lifetime — freeing one
   *  without the other is how a re-apply leaks GPU memory a frame at a time. */
  private destroyInstall(install: GroupInstall): void {
    install.uniformBuffer.destroy()
    for (const tex of install.images ?? []) tex?.destroy()
  }

  /** Upload a group's image maps. Sources are decoded images the host already
   *  holds; the engine never fetches, matching how models and motions arrive. */
  private uploadGroupImages(group: StyleGroup): (GPUTexture | null)[] | undefined {
    if (!group.images?.length) return undefined
    return group.images.slice(0, 4).map((entry) => {
      if (!entry) return null
      const wrapped = "source" in entry
      const src = wrapped ? entry.source : entry
      const width = Math.max(1, "naturalWidth" in src ? src.naturalWidth : src.width)
      const height = Math.max(1, "naturalHeight" in src ? src.naturalHeight : src.height)
      const tex = this.device.createTexture({
        label: `group map: ${group.id}`,
        size: [width, height],
        // Colour maps decode to linear on sample, the way material textures do;
        // data maps must not, or every threshold packed in their channels moves.
        format: wrapped && entry.srgb ? "rgba8unorm-srgb" : "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.device.queue.copyExternalImageToTexture(
        { source: src },
        { texture: tex, premultipliedAlpha: wrapped && entry.premultiplied === true },
        [width, height],
      )
      return tex
    })
  }

  private createMaterialBindGroup(
    label: string,
    baseEntries: GPUBindGroupEntry[],
    styleBuffer: GPUBuffer,
    groupImages?: (GPUTexture | null)[],
  ): GPUBindGroup {
    // Every material bind group in the engine is built here, which is why the
    // group's maps are threaded through this one function rather than patched in
    // at each call site — an unset slot reads white, never stale.
    const slots: GPUBindGroupEntry[] = []
    for (let i = 0; i < 4; i++) {
      const tex = groupImages?.[i] ?? this.fallbackMaterialTexture
      slots.push({ binding: 5 + i, resource: tex.createView() })
    }
    return this.device.createBindGroup({
      label,
      layout: this.mainPerMaterialBindGroupLayout,
      entries: [...baseEntries, { binding: 4, resource: { buffer: styleBuffer } }, ...slots],
    })
  }

  setMaterialVisible(modelName: string, materialName: string, visible: boolean): void {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return
    if (visible) inst.hiddenMaterials.delete(materialName)
    else inst.hiddenMaterials.add(materialName)
  }

  toggleMaterialVisible(modelName: string, materialName: string): void {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return
    if (inst.hiddenMaterials.has(materialName)) inst.hiddenMaterials.delete(materialName)
    else inst.hiddenMaterials.add(materialName)
  }

  isMaterialVisible(modelName: string, materialName: string): boolean {
    const inst = this.modelInstances.get(modelName)
    return inst ? !inst.hiddenMaterials.has(materialName) : false
  }

  // Toggle the GPU vertex-morph path. Only affects models loaded afterwards.
  setGpuMorphsEnabled(enabled: boolean): void {
    this.useGpuMorphs = enabled
  }

  /**
   * Engine-wide IK switch. Off suppresses every chain regardless of what any
   * motion says — for hosts that pose the skeleton themselves and want their
   * own rotations left alone. On (the default) hands the decision to the clip,
   * which carries per-chain state from the VMD it came from.
   */
  setIKEnabled(enabled: boolean): void {
    this.ikEnabled = enabled
  }

  getIKEnabled(): boolean {
    return this.ikEnabled
  }

  setPhysicsEnabled(enabled: boolean): void {
    this.physicsEnabled = enabled
  }

  getPhysicsEnabled(): boolean {
    return this.physicsEnabled
  }

  /**
   * Scene gravity, applied to every model's cloth and hair. The default is
   * (0, -98, 0) — MMD's own scale, where a character stands about 20 units
   * tall. Lower magnitudes float; tilting it sideways hangs everything on a
   * slant, which is the cheap way to fake a strong draught.
   */
  setGravity(gravity: Vec3): void {
    this.gravity = new Vec3(gravity.x, gravity.y, gravity.z)
    this.forEachInstance((inst) => inst.physics?.setGravity(this.gravity))
  }

  getGravity(): Vec3 {
    return new Vec3(this.gravity.x, this.gravity.y, this.gravity.z)
  }

  /**
   * Air movement across the scene — null is still air.
   *
   * Applied as an acceleration alongside gravity, so `strength` is in the same
   * units: against the default gravity of 98, a strength of 10-30 reads as a
   * breeze through hair and a skirt without lifting them off the body. Gusting
   * is driven by simulated time rather than wall time, so an exported take
   * gusts exactly as the preview did.
   */
  setWind(wind: WindOptions | null): void {
    this.wind = wind ? { ...wind, direction: new Vec3(wind.direction.x, wind.direction.y, wind.direction.z) } : null
    this.forEachInstance((inst) => inst.physics?.setWind(this.wind))
  }

  getWind(): WindOptions | null {
    return this.wind ? { ...this.wind, direction: new Vec3(this.wind.direction.x, this.wind.direction.y, this.wind.direction.z) } : null
  }

  resetPhysics(): void {
    this.forEachInstance((inst) => {
      if (!inst.physics) return
      // Re-pose bones from animation at dt=0 so we don't snap bodies to
      // whatever exploded state the last physics step wrote into dynamic bones.
      inst.model.update(0, this.ikEnabled)
      inst.physics.reset(inst.model.getWorldMatrices())
      inst.vertexBufferNeedsUpdate = true
    })
  }

  private forEachInstance(fn: (inst: ModelInstance) => void): void {
    for (const inst of this.modelInstances.values()) fn(inst)
  }

  // CPU frame-time breakdown (EMA-smoothed into getStats): where a frame's
  // milliseconds actually go — animation/IK/blending vs physics vs everything
  // else on the render thread. The first question of any perf report.
  private cpuAnimMs = 0
  private cpuPhysicsMs = 0
  private cpuRenderMs = 0
  private frameAnimMsRaw = 0
  private framePhysicsMsRaw = 0

  private updateInstances(deltaTime: number): void {
    let animMs = 0
    let physicsMs = 0
    this.forEachInstance((inst) => {
      const tAnim = performance.now()
      // A stage never solves IK — nothing drives its chains — and skips the pose
      // pass entirely while it is idle. Morph changes still come through, since
      // that is the one thing a stage's controls do move.
      const stageIdle = inst.isStage && inst.model.isIdle()
      let verticesChanged = false
      if (!stageIdle) {
        verticesChanged = inst.model.update(deltaTime, inst.isStage ? false : this.ikEnabled)
        inst.skinMatricesDirty = true
      }
      animMs += performance.now() - tAnim
      // Material morphs ride the same weight change as vertex morphs but land in
      // uniform buffers, so they consume their own flag — a model whose only
      // morphs are material morphs never enters the GPU vertex path below.
      if (inst.materialMorphTargets && inst.model.consumeAuxMorphDirty()) {
        this.applyMaterialMorphs(inst)
      }
      if (inst.gpuMorph) {
        // GPU path: on a weight change, upload effective weights (thresholding tiny values
        // to 0 to match the CPU skip) and flag the compute dispatch for this frame.
        if (inst.model.consumeMorphWeightsDirty()) {
          const eff = inst.model.getEffectiveMorphWeights()
          const wd = inst.gpuMorph.weightsData
          const n = Math.min(wd.length, eff.length)
          for (let i = 0; i < n; i++) {
            const w = eff[i]
            wd[i] = w < 0.0001 ? 0 : w
          }
          this.device.queue.writeBuffer(inst.gpuMorph.weightsBuffer, 0, wd as ArrayBufferView<ArrayBuffer>)
          inst.gpuMorph.dispatchNeeded = true
        }
      } else if (verticesChanged) {
        inst.vertexBufferNeedsUpdate = true
      }
      // Hidden models keep animating (cheap, and a reveal must not pop a stale
      // pose) but skip cloth simulation entirely — a roster of resident
      // alternate skins would otherwise pay full physics for invisible cloth.
      // Hosts that reveal after long hiding reset physics anyway (resetPhysics).
      if (inst.physics && this.physicsEnabled && inst.model.visible) {
        const tPhys = performance.now()
        inst.physics.step(deltaTime, inst.model.getWorldMatrices(), inst.model.getBoneInverseBindMatrices())
        physicsMs += performance.now() - tPhys
      }
      if (inst.vertexBufferNeedsUpdate) this.updateVertexBuffer(inst)
    })
    this.frameAnimMsRaw = animMs
    this.framePhysicsMsRaw = physicsMs
    const EMA = 0.1
    this.cpuAnimMs += (animMs - this.cpuAnimMs) * EMA
    this.cpuPhysicsMs += (physicsMs - this.cpuPhysicsMs) * EMA
  }

  private updateVertexBuffer(inst: ModelInstance): void {
    // GPU-morph models never CPU-upload the vertex buffer after load — the compute pass
    // owns the position slots. Ignore any stray dirty flag (e.g. from markVertexBufferDirty).
    if (inst.gpuMorph) {
      inst.vertexBufferNeedsUpdate = false
      return
    }
    const vertices = inst.model.getVertices()
    if (!vertices?.length) return
    // Vertex morphs touch only a subset of verts (typically the face), so upload just the
    // changed [minVert, maxVert] slice when the model can report one; null = full upload.
    const range = inst.model.consumeVertexUploadRange()
    if (range) {
      const STRIDE = 8 // floats per vertex (pos3 + normal3 + uv2)
      const firstFloat = range.minVert * STRIDE
      const floatLen = (range.maxVert - range.minVert + 1) * STRIDE
      const byteOffset = firstFloat * 4
      this.device.queue.writeBuffer(
        inst.vertexBuffer,
        byteOffset,
        vertices.buffer,
        vertices.byteOffset + byteOffset,
        floatLen * 4,
      )
    } else {
      this.device.queue.writeBuffer(inst.vertexBuffer, 0, vertices)
    }
    inst.vertexBufferNeedsUpdate = false
  }

  // One compute pass covering every model whose morph weights changed this frame.
  private dispatchMorphCompute(encoder: GPUCommandEncoder): void {
    let pass: GPUComputePassEncoder | null = null
    for (const inst of this.modelInstances.values()) {
      const gm = inst.gpuMorph
      if (!gm || !gm.dispatchNeeded) continue
      if (!pass) {
        pass = encoder.beginComputePass({ label: "morph compute" })
        pass.setPipeline(this.morphComputePipeline)
      }
      pass.setBindGroup(0, gm.bindGroup)
      pass.dispatchWorkgroups(gm.workgroups)
      gm.dispatchNeeded = false
    }
    if (pass) pass.end()
  }

  // ── GPU frustum cull ────────────────────────────────────────────────────────
  //
  // Sizes, once, so the arithmetic below is readable: a DrawMeta is 32 bytes
  // (vec3 lo + u32 model + vec3 hi + u32 flags), a ModelRec is 96 (mat4 + vec4
  // sphere + u32 flags + padding), and an indirect drawIndexed record is 5 u32.
  private static readonly CULL_META_BYTES = 32
  private static readonly CULL_MODEL_FLOATS = 24
  private static readonly CULL_ARG_WORDS = 5
  private static readonly CULL_DRAW_CASTS_SHADOW = 1
  private static readonly CULL_MODEL_VISIBLE = 1
  private static readonly CULL_MODEL_RIGID = 2

  /**
   * Flatten the scene's material draws into the order every cull buffer is
   * indexed by, and upload everything that only changes with STRUCTURE: the
   * per-draw metadata and the constant words of the indirect arguments.
   *
   * Runs on model add/remove and on any re-sort of a model's draws (a style
   * group assignment re-ranks them). Animation, physics and camera movement all
   * leave it alone — that is the same invalidation set the render bundles will
   * want, tested here first where being wrong is cheap.
   */
  private rebuildCullList(): void {
    this.cullListDirty = false
    this.cullDraws = []
    this.cullModels = []
    for (const inst of this.modelInstances.values()) {
      inst.cullModelIndex = this.cullModels.length
      this.cullModels.push(inst)
      for (const draw of inst.drawCalls) {
        draw.cullIndex = this.cullDraws.length
        this.cullDraws.push({ inst, draw })
      }
    }

    const drawCount = this.cullDraws.length
    const modelCount = this.cullModels.length
    this.releaseCullBuffers()
    if (drawCount === 0 || modelCount === 0) return

    this.cullMetaBytes = new ArrayBuffer(drawCount * Engine.CULL_META_BYTES)
    this.cullMetaF32 = new Float32Array(this.cullMetaBytes)
    this.cullMetaU32 = new Uint32Array(this.cullMetaBytes)
    const args = new Uint32Array(drawCount * Engine.CULL_ARG_WORDS)
    for (let i = 0; i < drawCount; i++) {
      const { inst, draw } = this.cullDraws[i]
      const f = i * 8
      const b = draw.bounds
      this.cullMetaF32[f] = b[0]
      this.cullMetaF32[f + 1] = b[1]
      this.cullMetaF32[f + 2] = b[2]
      this.cullMetaU32[f + 3] = inst.cullModelIndex
      this.cullMetaF32[f + 4] = b[3]
      this.cullMetaF32[f + 5] = b[4]
      this.cullMetaF32[f + 6] = b[5]
      this.cullMetaU32[f + 7] = draw.castsShadow === true ? Engine.CULL_DRAW_CASTS_SHADOW : 0
      // Everything but instanceCount is structural, so the compute never writes
      // it — one fewer store per draw per frame, and the args stay readable in a
      // capture as "this is the draw, that is whether it survived".
      const a = i * Engine.CULL_ARG_WORDS
      args[a] = draw.count
      args[a + 1] = 0
      args[a + 2] = draw.firstIndex
      args[a + 3] = 0
      args[a + 4] = 0
    }

    const modelBytes = modelCount * Engine.CULL_MODEL_FLOATS * 4
    const modelBuffer = new ArrayBuffer(modelBytes)
    this.cullModelData = new Float32Array(modelBuffer)
    this.cullModelFlags = new Uint32Array(modelBuffer)
    this.cullReference = new Uint8Array(drawCount)
    this.cullReferenceFrame = -1

    this.cullMetaBuffer = this.device.createBuffer({
      label: "cull draw metadata",
      size: this.cullMetaBytes.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.cullMetaBuffer, 0, this.cullMetaBytes)

    this.cullModelBuffer = this.device.createBuffer({
      label: "cull model records",
      size: modelBytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    const argUsage =
      GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    this.cullCameraArgs = this.device.createBuffer({
      label: "cull camera indirect args",
      size: args.byteLength,
      usage: argUsage,
    })
    this.cullShadowArgs = this.device.createBuffer({
      label: "cull shadow indirect args",
      size: args.byteLength,
      usage: argUsage,
    })
    this.device.queue.writeBuffer(this.cullCameraArgs, 0, args)
    this.device.queue.writeBuffer(this.cullShadowArgs, 0, args)

    if (!this.cullFrustaBuffer) {
      this.cullFrustaBuffer = this.device.createBuffer({
        label: "cull frusta",
        size: this.cullFrustaBytes.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
    }

    this.cullBindGroup = this.device.createBindGroup({
      label: "cull bind group",
      layout: this.cullBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cullMetaBuffer } },
        { binding: 1, resource: { buffer: this.cullModelBuffer } },
        { binding: 2, resource: { buffer: this.cullFrustaBuffer } },
        { binding: 3, resource: { buffer: this.cullCameraArgs } },
        { binding: 4, resource: { buffer: this.cullShadowArgs } },
      ],
    })
  }

  private releaseCullBuffers(): void {
    this.cullMetaBuffer?.destroy()
    this.cullModelBuffer?.destroy()
    this.cullCameraArgs?.destroy()
    this.cullShadowArgs?.destroy()
    this.cullMetaBuffer = null
    this.cullModelBuffer = null
    this.cullCameraArgs = null
    this.cullShadowArgs = null
    this.cullBindGroup = null
    this.cullReadback?.camera.destroy()
    this.cullReadback?.shadow.destroy()
    this.cullReadback = null
  }

  /**
   * One record per model: the rigid transform its per-material boxes live under,
   * or the world sphere that bounds it in any pose. Written every frame, because
   * this is the part animation moves — a few hundred bytes per model.
   */
  private writeCullModels(): void {
    const data = this.cullModelData
    const flags = this.cullModelFlags
    for (let i = 0; i < this.cullModels.length; i++) {
      const inst = this.cullModels[i]
      const o = i * Engine.CULL_MODEL_FLOATS
      let f = inst.model.visible ? Engine.CULL_MODEL_VISIBLE : 0
      if (inst.rigid) {
        data.set(inst.rigidXform, o)
        f |= Engine.CULL_MODEL_RIGID
        // The sphere is not read for a rigid model; leave it zeroed rather than
        // walking a stage's bones every frame to fill a field nothing consumes.
        data[o + 16] = 0
        data[o + 17] = 0
        data[o + 18] = 0
        data[o + 19] = 0
      } else {
        this.writeCullSphere(inst, data, o + 16)
      }
      flags[o + 20] = f
    }
    if (this.cullModelBuffer) this.device.queue.writeBuffer(this.cullModelBuffer, 0, data.buffer as ArrayBuffer)
  }

  /**
   * The world sphere for a skinned model: an AABB over its POSED bone positions,
   * grown by the model's skin margin, then carried through the scene placement.
   *
   * Bone positions come from the pose that already ran this frame, so a jump, a
   * run across the stage or a physics-driven skirt are all inside it by
   * construction — none of which a bind-pose box would have contained. See
   * ModelInstance.skinMargin for why growing by that one number is a bound and
   * not an estimate.
   */
  private writeCullSphere(inst: ModelInstance, out: Float32Array, at: number): void {
    const m = inst.model
    const bones = m.getWorldMatrices()
    if (bones.length === 0) {
      // Nothing to bound it with. A radius nothing can cull is the honest answer:
      // a model that never culls costs vertex work, one wrongly culled vanishes.
      out[at] = m.position.x
      out[at + 1] = m.position.y
      out[at + 2] = m.position.z
      out[at + 3] = 1e9
      return
    }
    let minX = Infinity
    let minY = Infinity
    let minZ = Infinity
    let maxX = -Infinity
    let maxY = -Infinity
    let maxZ = -Infinity
    for (let i = 0; i < bones.length; i++) {
      const v = bones[i].values
      const x = v[12]
      const y = v[13]
      const z = v[14]
      if (x < minX) minX = x
      if (y < minY) minY = y
      if (z < minZ) minZ = z
      if (x > maxX) maxX = x
      if (y > maxY) maxY = y
      if (z > maxZ) maxZ = z
    }
    const hx = (maxX - minX) * 0.5
    const hy = (maxY - minY) * 0.5
    const hz = (maxZ - minZ) * 0.5
    const centre = cullScratchVec
    centre.setXYZ(minX + hx, minY + hy, minZ + hz)
    const radius = Math.sqrt(hx * hx + hy * hy + hz * hz) + inst.skinMargin + CULL_BOUNDS_SLACK
    // Model space → world: the same composition setModelTransform bakes into the
    // skin matrices, applied to one point instead of every bone.
    centre.setXYZ(centre.x * m.scale, centre.y * m.scale, centre.z * m.scale)
    Quat.rotateVecInto(m.rotation, centre, centre)
    out[at] = centre.x + m.position.x
    out[at + 1] = centre.y + m.position.y
    out[at + 2] = centre.z + m.position.z
    out[at + 3] = radius * m.scale
  }

  /**
   * The camera's six frustum planes then the sun's, normalized, as inward
   * half-spaces (`dot(n, p) + d >= 0` is inside).
   *
   * Extracted from the combined view-projection rather than rebuilt from the
   * camera's own parameters, so a VMD-driven shot, an orbit and the shadow
   * volume's ortho box all go through one code path and none of them can
   * disagree with what the vertex shader actually projects.
   *
   * Culling shadow casters to the light's frustum looks like it should lose
   * casters that stand outside the volume and throw shade into it. It cannot,
   * because the shadow map is a single fixed 64×64 ortho box and the rasterizer
   * already clips to exactly these six planes — anything this rejects was
   * contributing nothing. That equivalence is a property of the one-cascade
   * setup and stops holding the day a second cascade arrives.
   */
  private writeCullFrusta(): void {
    if (!this.cullFrustaBuffer) return
    // cameraMatrixData holds view at 0 and projection at 16 — already written
    // this frame by updateCameraUniforms.
    Mat4.multiplyArrays(this.cameraMatrixData, 16, this.cameraMatrixData, 0, this.cullScratchVp, 0)
    writeFrustumPlanes(this.cullScratchVp, this.cullFrustaF32, 0)
    writeFrustumPlanes(this.shadowLightVPMatrix, this.cullFrustaF32, 24)
    this.cullFrustaU32[48] = this.cullDraws.length
    this.device.queue.writeBuffer(this.cullFrustaBuffer, 0, this.cullFrustaBytes)
  }

  /**
   * Cull every material draw against the camera and the sun, writing
   * `instanceCount` into two indirect-argument buffers.
   *
   * Nothing draws from those buffers yet. This increment stands the pass up and
   * leaves the draw path issuing direct draws, so the bounds and the frusta can
   * be checked against a scene that is definitely rendering correctly — see
   * getCullDiagnostics and setCullApply.
   */
  private dispatchCull(encoder: GPUCommandEncoder): void {
    if (this.cullListDirty) this.rebuildCullList()
    if (!this.cullBindGroup || this.cullDraws.length === 0) return
    // The CPU half runs even with a dead pipeline: it is what setCullApply gates
    // on, and a stale mirror would gate draws against last-known frusta.
    this.writeCullModels()
    this.writeCullFrusta()
    this.cullFrame++
    if (!this.cullPipeline) return
    const pass = encoder.beginComputePass({ label: "cull" })
    pass.setPipeline(this.cullPipeline)
    pass.setBindGroup(0, this.cullBindGroup)
    pass.dispatchWorkgroups(Math.ceil(this.cullDraws.length / 64))
    pass.end()
  }

  /**
   * The same test the compute runs, on the CPU, from the same uploaded numbers.
   *
   * Deliberately a second implementation rather than a shared one: two
   * independent readings of the same data is what makes agreement evidence. It
   * reads the mirrors written by writeCullModels/writeCullFrusta, so it answers
   * for the frame that was last dispatched, and caches per frame because both
   * the debug draw gate and the diagnostics want it.
   */
  private cullReferencePass(): Uint8Array {
    if (this.cullReferenceFrame === this.cullFrame) return this.cullReference
    this.cullReferenceFrame = this.cullFrame
    const out = this.cullReference
    const planes = this.cullFrustaF32
    for (let i = 0; i < this.cullDraws.length; i++) {
      const f = i * 8
      const mi = this.cullMetaU32[f + 3]
      const o = mi * Engine.CULL_MODEL_FLOATS
      const mf = this.cullModelFlags[o + 20]
      let bits = 0
      if ((mf & Engine.CULL_MODEL_VISIBLE) !== 0) {
        let inCamera: boolean
        let inLight: boolean
        if ((mf & Engine.CULL_MODEL_RIGID) !== 0) {
          const cx = (this.cullMetaF32[f] + this.cullMetaF32[f + 4]) * 0.5
          const cy = (this.cullMetaF32[f + 1] + this.cullMetaF32[f + 5]) * 0.5
          const cz = (this.cullMetaF32[f + 2] + this.cullMetaF32[f + 6]) * 0.5
          const ex = (this.cullMetaF32[f + 4] - this.cullMetaF32[f]) * 0.5
          const ey = (this.cullMetaF32[f + 5] - this.cullMetaF32[f + 1]) * 0.5
          const ez = (this.cullMetaF32[f + 6] - this.cullMetaF32[f + 2]) * 0.5
          const m = this.cullModelData
          const wx = m[o] * cx + m[o + 4] * cy + m[o + 8] * cz + m[o + 12]
          const wy = m[o + 1] * cx + m[o + 5] * cy + m[o + 9] * cz + m[o + 13]
          const wz = m[o + 2] * cx + m[o + 6] * cy + m[o + 10] * cz + m[o + 14]
          const gx = Math.abs(m[o]) * ex + Math.abs(m[o + 4]) * ey + Math.abs(m[o + 8]) * ez
          const gy = Math.abs(m[o + 1]) * ex + Math.abs(m[o + 5]) * ey + Math.abs(m[o + 9]) * ez
          const gz = Math.abs(m[o + 2]) * ex + Math.abs(m[o + 6]) * ey + Math.abs(m[o + 10]) * ez
          inCamera = aabbInsideFrustum(planes, 0, wx, wy, wz, gx, gy, gz)
          inLight = aabbInsideFrustum(planes, 24, wx, wy, wz, gx, gy, gz)
        } else {
          const m = this.cullModelData
          const sx = m[o + 16]
          const sy = m[o + 17]
          const sz = m[o + 18]
          const sr = m[o + 19]
          inCamera = sphereInsideFrustum(planes, 0, sx, sy, sz, sr)
          inLight = sphereInsideFrustum(planes, 24, sx, sy, sz, sr)
        }
        if (inCamera) bits |= 1
        if (inLight && (this.cullMetaU32[f + 7] & Engine.CULL_DRAW_CASTS_SHADOW) !== 0) bits |= 2
      }
      out[i] = bits
    }
    return out
  }

  /** The debug gate on the direct draws. Off (the default) it is one boolean
   *  read per draw and the scene renders exactly as it did before this pass
   *  existed. */
  private cullPasses(draw: DrawCall, shadowPass: boolean): boolean {
    if (!this.cullApply) return true
    if (draw.cullIndex < 0 || draw.cullIndex >= this.cullReference.length) return true
    return (this.cullReferencePass()[draw.cullIndex] & (shadowPass ? 2 : 1)) !== 0
  }

  /**
   * Skip draws the cull rejected, using the CPU mirror of the test.
   *
   * The CPU mirror rather than the GPU result on purpose: reading the compute's
   * output back would land one or two frames late and every camera move would
   * pop, which looks exactly like a culling bug and would waste the check. This
   * way anything that vanishes wrongly is a wrong BOUND, which is the thing
   * being validated. Development aid — it costs a per-frame pass over the draw
   * list, and the real path is the indirect draws.
   */
  setCullApply(on: boolean): void {
    this.cullApply = on
  }

  /**
   * Read the GPU's culling decisions back and diff them against the CPU
   * reference over the same frame's uploaded data.
   *
   * A clean report means the compute, its buffer layouts and the plane
   * extraction all agree with a second implementation. It does NOT prove the
   * bounds contain the geometry — that is what setCullApply(true) and looking at
   * the scene is for.
   */
  async getCullDiagnostics(): Promise<CullDiagnostics> {
    const drawCount = this.cullDraws.length
    const report: CullDiagnostics = {
      drawCount,
      modelCount: this.cullModels.length,
      cameraVisibleGpu: 0,
      shadowVisibleGpu: 0,
      cameraVisibleCpu: 0,
      shadowVisibleCpu: 0,
      rigidModels: 0,
      skinnedModels: 0,
      mismatches: [],
      models: [],
      camera: {
        eye: [this.camera.getEyePosition().x, this.camera.getEyePosition().y, this.camera.getEyePosition().z],
        target: [this.camera.target.x, this.camera.target.y, this.camera.target.z],
      },
    }
    for (const inst of this.cullModels) {
      if (inst.rigid) report.rigidModels++
      else report.skinnedModels++
      const o = inst.cullModelIndex * Engine.CULL_MODEL_FLOATS
      report.models.push({
        name: inst.name,
        rigid: inst.rigid,
        visible: inst.model.visible,
        draws: inst.drawCalls.length,
        cameraVisible: 0,
        shadowVisible: 0,
        casters: inst.shadowDrawCalls.length,
        sphere: inst.rigid
          ? null
          : [
              this.cullModelData[o + 16],
              this.cullModelData[o + 17],
              this.cullModelData[o + 18],
              this.cullModelData[o + 19],
            ],
      })
    }
    if (drawCount === 0 || !this.cullCameraArgs || !this.cullShadowArgs) return report

    const reference = this.cullReferencePass()
    for (let i = 0; i < drawCount; i++) {
      if ((reference[i] & 1) !== 0) {
        report.cameraVisibleCpu++
        report.models[this.cullDraws[i].inst.cullModelIndex].cameraVisible++
      }
      if ((reference[i] & 2) !== 0) {
        report.shadowVisibleCpu++
        report.models[this.cullDraws[i].inst.cullModelIndex].shadowVisible++
      }
    }

    // Before touching the staging buffers, not after: a second call arriving
    // mid-readback would otherwise destroy and replace the very buffers the
    // first one is mapping. mapAsync on an already-mapped buffer throws, and
    // this is a console API somebody will double-call — so the second caller
    // gets the CPU half and no exception.
    if (this.cullReadbackInFlight) return report
    this.cullReadbackInFlight = true
    try {
      const bytes = drawCount * Engine.CULL_ARG_WORDS * 4
      if (!this.cullReadback || this.cullReadback.bytes !== bytes) {
        this.cullReadback?.camera.destroy()
        this.cullReadback?.shadow.destroy()
        const mk = (label: string) =>
          this.device.createBuffer({ label, size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })
        this.cullReadback = { camera: mk("cull readback (camera)"), shadow: mk("cull readback (shadow)"), bytes }
      }
      await this.readCullArgs(
        this.cullReadback,
        [this.cullCameraArgs, this.cullShadowArgs],
        drawCount,
        reference,
        report,
      )
    } finally {
      this.cullReadbackInFlight = false
    }
    return report
  }

  /** The GPU half of getCullDiagnostics: copy both argument buffers back and
   *  diff them against the reference. */
  private async readCullArgs(
    rb: { camera: GPUBuffer; shadow: GPUBuffer; bytes: number },
    src: [GPUBuffer, GPUBuffer],
    drawCount: number,
    reference: Uint8Array,
    report: CullDiagnostics,
  ): Promise<void> {
    const bytes = rb.bytes
    const encoder = this.device.createCommandEncoder({ label: "cull readback" })
    encoder.copyBufferToBuffer(src[0], 0, rb.camera, 0, bytes)
    encoder.copyBufferToBuffer(src[1], 0, rb.shadow, 0, bytes)
    this.device.queue.submit([encoder.finish()])
    await Promise.all([rb.camera.mapAsync(GPUMapMode.READ), rb.shadow.mapAsync(GPUMapMode.READ)])
    const cameraArgs = new Uint32Array(rb.camera.getMappedRange().slice(0))
    const shadowArgs = new Uint32Array(rb.shadow.getMappedRange().slice(0))
    rb.camera.unmap()
    rb.shadow.unmap()

    for (let i = 0; i < drawCount; i++) {
      const gpuCamera = cameraArgs[i * Engine.CULL_ARG_WORDS + 1] !== 0
      const gpuShadow = shadowArgs[i * Engine.CULL_ARG_WORDS + 1] !== 0
      if (gpuCamera) report.cameraVisibleGpu++
      if (gpuShadow) report.shadowVisibleGpu++
      const cpuCamera = (reference[i] & 1) !== 0
      const cpuShadow = (reference[i] & 2) !== 0
      if (gpuCamera === cpuCamera && gpuShadow === cpuShadow) continue
      const { inst, draw } = this.cullDraws[i]
      if (gpuCamera !== cpuCamera)
        report.mismatches.push({
          model: inst.name,
          material: draw.materialName,
          pass: "camera",
          gpu: gpuCamera,
          cpu: cpuCamera,
        })
      if (gpuShadow !== cpuShadow)
        report.mismatches.push({
          model: inst.name,
          material: draw.materialName,
          pass: "shadow",
          gpu: gpuShadow,
          cpu: cpuShadow,
        })
    }
  }

  private async setupModelInstance(
    name: string,
    model: Model,
    basePath: string,
    assetReader: AssetReader,
    isStage = false,
  ): Promise<void> {
    const vertices = model.getVertices()
    const skinning = model.getSkinning()
    const skeleton = model.getSkeleton()
    const boneCount = skeleton.bones.length
    const matrixSize = boneCount * 16 * 4

    const vertexBuffer = this.device.createBuffer({
      label: `${name}: vertex buffer`,
      size: vertices.byteLength,
      // STORAGE so the morph compute pass can write morphed positions in place.
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
    })
    this.device.queue.writeBuffer(vertexBuffer, 0, vertices)

    const jointsBuffer = this.device.createBuffer({
      label: `${name}: joints buffer`,
      size: skinning.joints.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(
      jointsBuffer,
      0,
      skinning.joints.buffer,
      skinning.joints.byteOffset,
      skinning.joints.byteLength,
    )

    const weightsBuffer = this.device.createBuffer({
      label: `${name}: weights buffer`,
      size: skinning.weights.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(
      weightsBuffer,
      0,
      skinning.weights.buffer,
      skinning.weights.byteOffset,
      skinning.weights.byteLength,
    )

    const skinMatrixBuffer = this.device.createBuffer({
      label: `${name}: skin matrices`,
      size: Math.max(256, matrixSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })

    const indices = model.getIndices()
    if (!indices) throw new Error("Model has no index buffer")
    const indexBuffer = this.device.createBuffer({
      label: `${name}: index buffer`,
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(indexBuffer, 0, indices)

    const rbs = model.getRigidbodies()
    // A stage never simulates, so its bodies are never built — constructing the
    // solver for the heaviest mesh in the scene and dropping it afterwards was
    // both wasted work and an invariant maintained in the wrong place.
    const physics = !isStage && rbs.length > 0 ? new RezePhysics(rbs, model.getJoints()) : null
    // Adopt the scene's air, or a model added mid-session would fall under
    // different gravity from the ones already on stage.
    if (physics) {
      physics.setGravity(this.gravity)
      if (this.wind) physics.setWind(this.wind)
    }

    const shadowBindGroup = this.device.createBindGroup({
      label: `${name}: shadow bind`,
      layout: this.shadowDepthPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.shadowLightVPBuffer } },
        { binding: 1, resource: { buffer: skinMatrixBuffer } },
        { binding: 2, resource: this.materialSampler },
      ],
    })

    const mainPerInstanceBindGroup = this.device.createBindGroup({
      label: `${name}: main per-instance bind group`,
      layout: this.mainPerInstanceBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: skinMatrixBuffer } }],
    })

    const pickPerInstanceBindGroup = this.device.createBindGroup({
      label: `${name}: pick per-instance bind group`,
      layout: this.pickPerInstanceBindGroupLayout,
      entries: [{ binding: 0, resource: { buffer: skinMatrixBuffer } }],
    })

    const gpuBuffers: GPUBuffer[] = [vertexBuffer, indexBuffer, jointsBuffer, weightsBuffer, skinMatrixBuffer]

    const gpuMorph = this.createGpuMorph(name, model, vertexBuffer, gpuBuffers)

    // Cull bounds. The margin is the skinning reach plus the largest single
    // vertex-morph displacement — a face morph is millimetres against a reach of
    // whole units, so charging one morph rather than the sum of all of them keeps
    // the bound honest without inflating it.
    const bindPositions = boneBindPositions(skeleton.inverseBindMatrices, boneCount)
    const skinMargin =
      computeSkinMargin(vertices, skinning.joints, skinning.weights, bindPositions, boneCount) +
      vertexMorphReach(model)

    const inst: ModelInstance = {
      name,
      model,
      basePath,
      assetReader,
      gpuBuffers,
      textureCacheKeys: [],
      vertexBuffer,
      indexBuffer,
      jointsBuffer,
      weightsBuffer,
      skinMatrixBuffer,
      drawCalls: [],
      shadowDrawCalls: [],
      shadowBindGroup,
      mainPerInstanceBindGroup,
      pickPerInstanceBindGroup,
      pickDrawCalls: [],
      isStage,
      // Seeded true: the bind pose has to reach the GPU once before any frame.
      skinMatricesDirty: true,
      hiddenMaterials: new Set(),
      morphHiddenMaterials: new Set(),
      materialMorphTargets: null,
      materialMorphByIndex: null,
      physics,
      vertexBufferNeedsUpdate: false,
      gpuMorph,
      styleGroups: new Map(),
      materialToGroup: new Map(),
      styleGroupGen: new Map(),
      cullModelIndex: 0,
      // Seeded false: the first skin-matrix upload decides it, and until then the
      // sphere path is the safe answer (it never culls something it should not).
      rigid: false,
      rigidXform: new Float32Array(16),
      skinMargin,
    }
    await this.setupMaterialsForInstance(inst)
    this.modelInstances.set(name, inst)
    this.cullListDirty = true
  }

  // Build the per-model GPU vertex-morph state. Returns null (and leaves the model on the
  // CPU morph path) when the model has no vertex morphs. Created buffers are pushed into
  // gpuBuffers so they're released with the instance.
  private createGpuMorph(
    name: string,
    model: Model,
    vertexBuffer: GPUBuffer,
    gpuBuffers: GPUBuffer[],
  ): GpuMorph | null {
    if (!this.useGpuMorphs) return null
    const data = model.buildMorphComputeData()
    if (!data) return null

    const RO = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
    const mkStorage = (label: string, arr: Float32Array | Uint32Array): GPUBuffer => {
      const buf = this.device.createBuffer({
        label: `${name}: morph ${label}`,
        size: Math.max(arr.byteLength, 4),
        usage: RO,
      })
      this.device.queue.writeBuffer(buf, 0, arr as ArrayBufferView<ArrayBuffer>)
      gpuBuffers.push(buf)
      return buf
    }

    const baseBuf = mkStorage("basePositions", data.basePositions)
    const rowBuf = mkStorage("rowStart", data.rowStart)
    const colMorphBuf = mkStorage("colMorph", data.colMorph)
    const colOffsetBuf = mkStorage("colOffset", data.colOffset)

    // Weights are zero-initialized by WebGPU; the first weight change uploads real values.
    const weightsBuffer = this.device.createBuffer({
      label: `${name}: morph weights`,
      size: Math.max(data.morphCount * 4, 4),
      usage: RO,
    })
    gpuBuffers.push(weightsBuffer)

    const paramsBuffer = this.device.createBuffer({
      label: `${name}: morph params`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(paramsBuffer, 0, new Uint32Array([data.vertexCount, 0, 0, 0]))
    gpuBuffers.push(paramsBuffer)

    const bindGroup = this.device.createBindGroup({
      label: `${name}: morph compute bind group`,
      layout: this.morphComputeBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: baseBuf } },
        { binding: 1, resource: { buffer: rowBuf } },
        { binding: 2, resource: { buffer: colMorphBuf } },
        { binding: 3, resource: { buffer: colOffsetBuf } },
        { binding: 4, resource: { buffer: weightsBuffer } },
        { binding: 5, resource: { buffer: vertexBuffer } },
        { binding: 6, resource: { buffer: paramsBuffer } },
      ],
    })

    model.enableGpuMorphs()

    return {
      bindGroup,
      weightsBuffer,
      weightsData: new Float32Array(data.morphCount),
      workgroups: Math.ceil(data.vertexCount / 64),
      dispatchNeeded: false, // vertex buffer already holds base; dispatch on first weight change
    }
  }

  private createGroundGeometry(width: number = 100, height: number = 100) {
    const halfWidth = width / 2
    const halfHeight = height / 2

    const vertices = new Float32Array([
      // Bottom-left
      -halfWidth,
      0,
      -halfHeight, // position
      0,
      1,
      0, // normal (up)
      0,
      0, // uv

      // Bottom-right
      halfWidth,
      0,
      -halfHeight, // position
      0,
      1,
      0, // normal (up)
      1,
      0, // uv

      // Top-right
      halfWidth,
      0,
      halfHeight, // position
      0,
      1,
      0, // normal (up)
      1,
      1, // uv

      // Top-left
      -halfWidth,
      0,
      halfHeight, // position
      0,
      1,
      0, // normal (up)
      0,
      1, // uv
    ])

    // Create indices for two triangles
    const indices = new Uint16Array([
      0,
      1,
      2, // First triangle
      0,
      2,
      3, // Second triangle
    ])

    // Create vertex buffer
    this.groundVertexBuffer = this.device.createBuffer({
      label: "ground vertex buffer",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.groundVertexBuffer, 0, vertices)

    this.groundIndexBuffer = this.device.createBuffer({
      label: "ground index buffer",
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.groundIndexBuffer, 0, indices)
  }

  private createShadowGroundResources(opts: {
    diffuseColor: Vec3
    fadeStart: number
    fadeEnd: number
    shadowStrength: number
    gridSpacing: number
    gridLineWidth: number
    gridLineOpacity: number
    gridLineColor: Vec3
    noiseStrength: number
    opacity: number
  }) {
    const {
      diffuseColor,
      fadeStart,
      fadeEnd,
      shadowStrength,
      gridSpacing,
      gridLineWidth,
      gridLineOpacity,
      gridLineColor,
      noiseStrength,
      opacity,
    } = opts
    // Shadow map is already created in setupPipelines()
    const gb = new Float32Array(16)
    gb[0] = diffuseColor.x
    gb[1] = diffuseColor.y
    gb[2] = diffuseColor.z
    gb[3] = fadeStart
    gb[4] = fadeEnd
    gb[5] = shadowStrength
    gb[6] = 1 / Engine.SHADOW_MAP_SIZE
    gb[7] = gridSpacing
    gb[8] = gridLineWidth
    gb[9] = gridLineOpacity
    gb[10] = noiseStrength
    gb[11] = opacity
    gb[12] = gridLineColor.x
    gb[13] = gridLineColor.y
    gb[14] = gridLineColor.z
    gb[15] = 0
    this.groundShadowMaterialBuffer = this.device.createBuffer({
      size: gb.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.groundShadowMaterialBuffer, 0, gb)
    this.groundShadowBindGroup = this.device.createBindGroup({
      label: "ground shadow bind",
      layout: this.groundShadowBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.lightUniformBuffer } },
        { binding: 2, resource: this.shadowMapDepthView },
        { binding: 3, resource: this.shadowComparisonSampler },
        { binding: 4, resource: { buffer: this.groundShadowMaterialBuffer } },
        { binding: 5, resource: { buffer: this.shadowLightVPBuffer } },
      ],
    })
  }


  // Shadow is cast from the visible sun direction — same vector the shader lights with.
  /** Whether the shadow map needs clearing — see the shadow pass in `render`.
   *
   *  Starts true so the very first frame runs the pass even with an empty scene. The
   *  depth texture is created once and WebGPU zero-fills it, and depth 0.0 is the
   *  nearest possible occluder: leave it uncleared and every ground pixel inside the
   *  light frustum tests as shadowed, painting a hard-edged patch the shape of the
   *  frustum onto an otherwise empty floor. */
  private shadowMapPopulated = true
  private shadowLightVPDirty = true
  // Last shadow-volume center, to skip recomputes while nothing moves.
  private readonly shadowCenter = new Vec3(0, 11, 0)

  private updateShadowLightVP() {
    // The 64×64-unit volume follows the camera target so a character carried far
    // from the origin by code-driven root motion stays inside the lit frustum.
    const t = this.camera.target
    const moved =
      Math.abs(t.x - this.shadowCenter.x) > 1e-3 ||
      Math.abs(t.y - this.shadowCenter.y) > 1e-3 ||
      Math.abs(t.z - this.shadowCenter.z) > 1e-3
    if (!this.shadowLightVPDirty && !moved) return
    this.shadowLightVPDirty = false
    this.shadowCenter.setXYZ(t.x, t.y, t.z)

    const dir = new Vec3(this.sun.direction.x, this.sun.direction.y, this.sun.direction.z)
    dir.normalize()
    const up = Math.abs(dir.y) > 0.99 ? new Vec3(0, 0, -1) : new Vec3(0, 1, 0)

    // Snap the center to shadow-map texels in the light's right/up plane so the
    // moving volume doesn't shimmer the shadow edges while running.
    const right = Vec3.crossInto(up, dir, new Vec3(0, 0, 0)).normalize()
    const upv = Vec3.crossInto(dir, right, new Vec3(0, 0, 0))
    const texel = 64 / Engine.SHADOW_MAP_SIZE
    const tr = Math.round(t.dot(right) / texel) * texel
    const tu = Math.round(t.dot(upv) / texel) * texel
    const td = t.dot(dir)
    const target = new Vec3(
      right.x * tr + upv.x * tu + dir.x * td,
      right.y * tr + upv.y * tu + dir.y * td,
      right.z * tr + upv.z * tu + dir.z * td
    )

    const eye = new Vec3(target.x - dir.x * 72, target.y - dir.y * 72, target.z - dir.z * 72)
    const view = Mat4.lookAt(eye, target, up)
    const proj = Mat4.orthographicLh(-32, 32, -32, 32, 1, 140)
    const vp = proj.multiply(view)
    this.shadowLightVPMatrix.set(vp.values)
    this.device.queue.writeBuffer(this.shadowLightVPBuffer, 0, this.shadowLightVPMatrix)
  }

  private async setupMaterialsForInstance(inst: ModelInstance): Promise<void> {
    const model = inst.model
    const materials = model.getMaterials()
    if (materials.length === 0) throw new Error("Model has no materials")
    const textures = model.getTextures()
    const prefix = `${inst.name}: `
    // 1-based so that (0,0) = clear color = "no hit"
    const modelId = this.modelInstances.size + 1

    const texLogicalPath = (texIndex: number): string | null =>
      texIndex < 0 || texIndex >= textures.length
        ? null
        : joinAssetPath(inst.basePath, normalizeAssetPath(textures[texIndex].path))
    const loadTextureByIndex = async (texIndex: number): Promise<GPUTexture | null> => {
      const logicalPath = texLogicalPath(texIndex)
      return logicalPath ? this.createTextureFromLogicalPath(inst, logicalPath) : null
    }
    // Mesh data for sheerness sampling (8 floats/vertex; uv at +6). See
    // materialIsSheer — classification happens per material below.
    const meshVertices = model.getVertices()
    const meshIndices = model.getIndices()

    // 頭 bone index for the eye shader's rear-view gate (-1 when absent).
    const headBoneIndex = model.getSkeleton().bones.findIndex((b) => b.name === "頭")

    // Materials a type-8 morph can reach. -1 in an offset means "all of them",
    // so the presence of ANY material morph makes every material a target.
    const morphedMaterials = new Set<number>()
    for (const morph of model.getMorphing().morphs) {
      if (morph.type !== 8 || !morph.materialOffsets) continue
      for (const off of morph.materialOffsets) {
        if (off.materialIndex < 0) for (let i = 0; i < materials.length; i++) morphedMaterials.add(i)
        else morphedMaterials.add(off.materialIndex)
      }
    }
    const morphTargets: MaterialMorphTarget[] = []
    // Cull slack charged to every material box below.
    const morphReach = vertexMorphReach(model)

    let currentIndexOffset = 0
    let materialId = 0
    // The PMX index, which is what a material morph points at — distinct from
    // materialId, which only counts materials that produced a draw.
    let pmxMaterialIndex = -1
    for (const mat of materials) {
      pmxMaterialIndex++
      const indexCount = mat.vertexCount
      if (indexCount === 0) continue
      materialId++

      let diffuseTexture = await loadTextureByIndex(mat.diffuseTextureIndex)
      if (!diffuseTexture) {
        console.warn(`${prefix}material "${mat.name}" has no loadable diffuse texture — using fallback`)
        diffuseTexture = this.fallbackMaterialTexture
      }

      const materialAlpha = mat.diffuse[3]
      const diffusePath = texLogicalPath(mat.diffuseTextureIndex)
      const alphaSampler = diffusePath ? this.textureAlphaCache.get(diffusePath) : null
      const stats = materialAlphaStats(meshVertices, meshIndices, currentIndexOffset, indexCount, alphaSampler)
      // babylon-mmd parity (its default DepthWriteAlphaBlendingWithEvaluation
      // method): the bucket decision is BINARY. A material with ANY translucent
      // texels on its geometry is alpha-blend — drawn in PMX author order with
      // depth write ON (forceDepthWrite); everything else is opaque. The old
      // avg/frac tier system left mostly-opaque lace (translucentFrac 0.09) in
      // the opaque bucket while its sibling panels went transparent, breaking
      // the author's compositing order — the gray fold patches. The 2% floor
      // only guards against centroid-sampling noise on genuinely solid cloth.
      const sheer = stats.avg < SHEER_ALPHA_THRESHOLD
      const isTransparent = materialAlpha < 1.0 - 0.001 || sheer || stats.translucentFrac > 0.02
      // Shadow casting: the PMX author's own flag (bit 0x04, cast self-shadow) —
      // exactly what MMD honors. Sheerness is handled per texel by the shadow
      // pass's alpha test, not by a per-material veto: a threshold on avg alpha
      // misclassified fully-worn opaque dresses (avg 0.69) as veils and stripped
      // their shadows, while any lower cliff would strand the next model.
      const castsShadow = (mat.edgeFlag & 0x04) !== 0

      // Sphere map (sph=1 multiply / spa=2 add). Mode 3 (sub-texture UV) is
      // rare and not implemented — treated as none, like a failed load.
      let sphereMode = mat.sphereMode === 1 || mat.sphereMode === 2 ? mat.sphereMode : 0
      let sphereTexture: GPUTexture | null = null
      if (sphereMode !== 0) {
        sphereTexture = await loadTextureByIndex(mat.sphereTextureIndex)
        if (!sphereTexture) sphereMode = 0
      }

      // Toon ramp: model-local file, or the generic ramp for the shared
      // toon01–10 set. No toon → white (no ramp modulation), MMD behavior.
      let toonTexture: GPUTexture | null = null
      if (mat.sharedToon) {
        toonTexture = this.defaultToonRampTexture
      } else if (mat.toonTextureIndex >= 0) {
        toonTexture = await loadTextureByIndex(mat.toonTextureIndex)
      }

      const materialUniformBuffer = this.createMaterialUniformBuffer(prefix + mat.name, mat, sphereMode, headBoneIndex)
      inst.gpuBuffers.push(materialUniformBuffer)
      if (morphedMaterials.has(pmxMaterialIndex)) {
        const base = this.materialUniformData(mat, sphereMode, headBoneIndex)
        morphTargets.push({
          pmxIndex: pmxMaterialIndex,
          materialName: mat.name,
          buffer: materialUniformBuffer,
          base,
          work: new Float32Array(base.length),
          // Seeded from base: that is what createMaterialUniformBuffer already
          // uploaded, so an unmorphed material never writes a first time.
          last: Float32Array.from(base),
        })
      }

      const textureView = diffuseTexture.createView()
      const baseBindGroupEntries: GPUBindGroupEntry[] = [
        { binding: 0, resource: textureView },
        { binding: 1, resource: { buffer: materialUniformBuffer } },
        { binding: 2, resource: (toonTexture ?? this.fallbackMaterialTexture).createView() },
        { binding: 3, resource: (sphereTexture ?? this.fallbackMaterialTexture).createView() },
      ]
      // Ungrouped at load — binding(4) = zero buffer, neutral base pipeline. autoStyleGroups
      // / applyStyleGroups rebind grouped materials to their group's buffer + pipeline.
      const bindGroup = this.createMaterialBindGroup(
        `${prefix}material: ${mat.name}`,
        baseBindGroupEntries,
        this.zeroStyleBuffer,
      )

      // Inverted-hull outline for EVERY edge-flagged material (PMX bit 0x10) —
      // the outline FS alpha-tests the diffuse texture, so sheer fabric masks
      // its own hull where it is see-through instead of us skipping it here.
      // Drawn interleaved right after this material's color draw (babylon-mmd's
      // per-mesh afterRender outline stage) — see drawMaterials.
      // Stages get no outline hulls. The inverted hull is a SECOND full draw of
      // the material's geometry, and stage PMX routinely set the edge flag across
      // every material — on the heaviest mesh in the scene that doubles the
      // geometry submitted per frame to draw cartoon outlines around
      // architecture, which is not the look anyone is after.
      let outline: DrawCall["outline"]
      if (!inst.isStage && (mat.edgeFlag & 0x10) !== 0 && mat.edgeSize > 0) {
        const materialUniformData = new Float32Array([
          mat.edgeColor[0],
          mat.edgeColor[1],
          mat.edgeColor[2],
          mat.edgeColor[3],
          mat.edgeSize,
          0,
          0,
          0,
        ])
        const outlineUniformBuffer = this.createUniformBuffer(`${prefix}outline: ${mat.name}`, materialUniformData)
        inst.gpuBuffers.push(outlineUniformBuffer)
        const outlineBindGroup = this.device.createBindGroup({
          label: `${prefix}outline: ${mat.name}`,
          layout: this.outlinePerMaterialBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: outlineUniformBuffer } },
            { binding: 1, resource: textureView },
          ],
        })
        outline = { bindGroup: outlineBindGroup }
      }

      // Model-space AABB for the cull compute, grown by the three things that can
      // put geometry outside the vertices it was measured from: a vertex morph,
      // the inverted-hull outline sharing this index range, and the tolerance the
      // rigid test allows on the skin matrices. See CULL_BOUNDS_SLACK.
      const bounds = materialBounds(meshVertices, meshIndices, currentIndexOffset, indexCount)
      const grow = morphReach + CULL_BOUNDS_SLACK
      bounds[0] -= grow
      bounds[1] -= grow
      bounds[2] -= grow
      bounds[3] += grow
      bounds[4] += grow
      bounds[5] += grow

      const type: DrawCallType = isTransparent ? "transparent" : "opaque"
      inst.drawCalls.push({
        type,
        count: indexCount,
        firstIndex: currentIndexOffset,
        bindGroup,
        materialName: mat.name,
        groupId: null,
        baseBindGroupEntries,
        castsShadow,
        outline,
        bounds,
        cullIndex: -1,
      })

      if (this.onRaycast) {
        const pickIdData = new Float32Array([modelId, materialId, 0, 0])
        const pickIdBuffer = this.createUniformBuffer(`${prefix}pick: ${mat.name}`, pickIdData)
        inst.gpuBuffers.push(pickIdBuffer)
        const pickBindGroup = this.device.createBindGroup({
          label: `${prefix}pick: ${mat.name}`,
          layout: this.pickPerMaterialBindGroupLayout,
          entries: [{ binding: 0, resource: { buffer: pickIdBuffer } }],
        })
        inst.pickDrawCalls.push({ count: indexCount, firstIndex: currentIndexOffset, bindGroup: pickBindGroup })
      }

      currentIndexOffset += indexCount
    }

    // Sort so the opaque bucket is emitted in the order the stencil-based see-through-hair
    // effect requires: {non-hair, non-eye} → {eye} → {hair}. Eye writes stencil=EYE_VALUE;
    // hair stencil-tests "not equal" and skips eye pixels; the follow-up hairOverEyes pass
    // re-fills them alpha-blended. sortDrawCalls also (re)builds shadowDrawCalls. All draws
    // are ungrouped at setup, so the rank comes from the preset; applyStyleGroups re-sorts
    // by render-class when groups are assigned. Array.sort is stable → PMX order preserved
    // within a bucket.
    this.sortDrawCalls(inst)

    inst.materialMorphTargets = morphTargets.length > 0 ? morphTargets : null
    inst.materialMorphByIndex = inst.materialMorphTargets
      ? new Map(morphTargets.map((t) => [t.pmxIndex, t]))
      : null
    // Seed from the current weights: a scene can open with a switch already on.
    if (inst.materialMorphTargets) this.applyMaterialMorphs(inst)
  }

  /** Matches the WGSL MaterialUniforms struct in common.ts — 64 bytes
   *  (diffuse+alpha | ambient+shininess | specular+sphereMode | headIdx+pad). */
  private materialUniformData(mat: Material, sphereMode: number, headBoneIndex: number): Float32Array {
    const data = new Float32Array(16)
    data[0] = mat.diffuse[0]
    data[1] = mat.diffuse[1]
    data[2] = mat.diffuse[2]
    data[3] = mat.diffuse[3]
    data[4] = mat.ambient[0]
    data[5] = mat.ambient[1]
    data[6] = mat.ambient[2]
    data[7] = mat.shininess
    data[8] = mat.specular[0]
    data[9] = mat.specular[1]
    data[10] = mat.specular[2]
    data[11] = sphereMode
    data[12] = headBoneIndex
    return data
  }

  private createMaterialUniformBuffer(
    label: string,
    mat: Material,
    sphereMode: number,
    headBoneIndex: number,
  ): GPUBuffer {
    return this.createUniformBuffer(
      `material uniform: ${label}`,
      this.materialUniformData(mat, sphereMode, headBoneIndex),
    )
  }

  /**
   * Re-derive every morph-targeted material's uniform block from base and push
   * the ones that moved.
   *
   * Blend maths follow MMD (and babylon-mmd's _applyMaterialMorph): multiply
   * lerps from base toward base*morph, add offsets from base. Weight 0 must
   * therefore land exactly on base, which is why this recomputes rather than
   * accumulates.
   *
   * A material driven to zero alpha is dropped from the draw instead of being
   * written through: the opaque/transparent bucket is decided at load from the
   * PMX alpha, so an opaque draw cannot become see-through by uniform alone.
   * Full-off is the switch stage artists actually ship (帽子消失 and friends);
   * a partial fade on a material that loaded opaque still will not blend.
   */
  private applyMaterialMorphs(inst: ModelInstance): void {
    const targets = inst.materialMorphTargets
    if (!targets) return
    const morphs = inst.model.getMorphing().morphs
    const weights = inst.model.getEffectiveMorphWeights()

    for (const target of targets) {
      target.work.set(target.base)
    }

    for (let i = 0; i < morphs.length; i++) {
      const w = weights[i]
      if (w < 0.0001) continue
      const morph = morphs[i]
      if (morph.type !== 8 || !morph.materialOffsets) continue
      for (const off of morph.materialOffsets) {
        // A named material resolves in one lookup. Only the -1 wildcard walks
        // every target — and once any offset uses it, every material in the
        // model is a target, so scanning per offset would be quadratic on the
        // large stages this is meant to serve.
        const hit = off.materialIndex >= 0 ? inst.materialMorphByIndex?.get(off.materialIndex) : undefined
        const affected = off.materialIndex >= 0 ? (hit ? [hit] : []) : targets
        for (const target of affected) {
          const d = target.work
          if (off.offsetType === MATERIAL_MORPH_MULTIPLY) {
            d[0] += (d[0] * off.diffuse[0] - d[0]) * w
            d[1] += (d[1] * off.diffuse[1] - d[1]) * w
            d[2] += (d[2] * off.diffuse[2] - d[2]) * w
            d[3] += (d[3] * off.diffuse[3] - d[3]) * w
            d[4] += (d[4] * off.ambient[0] - d[4]) * w
            d[5] += (d[5] * off.ambient[1] - d[5]) * w
            d[6] += (d[6] * off.ambient[2] - d[6]) * w
            d[7] += (d[7] * off.shininess - d[7]) * w
            d[8] += (d[8] * off.specular[0] - d[8]) * w
            d[9] += (d[9] * off.specular[1] - d[9]) * w
            d[10] += (d[10] * off.specular[2] - d[10]) * w
          } else {
            d[0] += off.diffuse[0] * w
            d[1] += off.diffuse[1] * w
            d[2] += off.diffuse[2] * w
            d[3] += off.diffuse[3] * w
            d[4] += off.ambient[0] * w
            d[5] += off.ambient[1] * w
            d[6] += off.ambient[2] * w
            d[7] += off.shininess * w
            d[8] += off.specular[0] * w
            d[9] += off.specular[1] * w
            d[10] += off.specular[2] * w
          }
        }
      }
    }

    inst.morphHiddenMaterials.clear()
    for (const target of targets) {
      const d = target.work
      // Alpha is the switch; clamp the rest so a stacked multiply cannot send a
      // colour negative and light the material from the inside.
      for (let k = 0; k < 11; k++) if (d[k] < 0) d[k] = 0
      if (d[3] < 0.0001) inst.morphHiddenMaterials.add(target.materialName)
      let changed = false
      for (let k = 0; k < 11; k++) {
        if (d[k] !== target.last[k]) {
          changed = true
          break
        }
      }
      if (!changed) continue
      target.last.set(d)
      this.device.queue.writeBuffer(target.buffer, 0, d as ArrayBufferView<ArrayBuffer>)
    }
  }

  private createUniformBuffer(label: string, data: Float32Array | Uint32Array): GPUBuffer {
    const buffer = this.device.createBuffer({
      label,
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(buffer, 0, data as ArrayBufferView<ArrayBuffer>)
    return buffer
  }

  private shouldRenderDrawCall(inst: ModelInstance, drawCall: DrawCall): boolean {
    return !inst.hiddenMaterials.has(drawCall.materialName) && !inst.morphHiddenMaterials.has(drawCall.materialName)
  }

  private async createTextureFromLogicalPath(inst: ModelInstance, logicalPath: string): Promise<GPUTexture | null> {
    const cacheKey = logicalPath
    const cached = this.textureCache.get(cacheKey)
    if (cached) {
      // Record the reference on THIS instance too — the cache is engine-global
      // (two PMX in one folder share texture paths), and removeModel decides
      // destruction by who still references a key. Without this, a cache-hit
      // borrower kept rendering a texture the creator's removal destroyed
      // ("Destroyed texture used in a submit").
      if (!inst.textureCacheKeys.includes(cacheKey)) inst.textureCacheKeys.push(cacheKey)
      return cached
    }

    // PMX texture tables are hand-maintained, and they routinely carry entries
    // that are not files. Two kinds show up constantly: a bare directory
    // ("Textures", "spa\\"), which is a leftover placeholder pointing at nothing,
    // and a name whose extension was dropped — where the texture is sitting right
    // there on disk one suffix longer, and the material renders white for want of
    // it. The first is answered by staying quiet, the second by trying.
    let buffer: ArrayBuffer | null = null
    let readError: unknown = null
    try {
      buffer = await inst.assetReader.readBinary(logicalPath)
    } catch (e) {
      readError = e
    }
    if (!buffer) {
      const base = logicalPath.split(/[\\/]/).pop() ?? ""
      // No basename at all: the entry named a directory. Nothing was ever meant
      // to load, so this is not a failure worth a line in anyone's console.
      if (!base) return null
      if (!base.includes(".")) {
        for (const ext of TEXTURE_EXTENSION_GUESSES) {
          try {
            buffer = await inst.assetReader.readBinary(`${logicalPath}${ext}`)
            break
          } catch {
            // keep trying — the list is short and only runs for a broken entry
          }
        }
      }
      if (!buffer) {
        console.warn(`[reze] texture read failed: ${logicalPath}`, readError instanceof Error ? readError.message : readError)
        return null
      }
    }

    // Decode to either an ImageBitmap (web-native formats) or raw RGBA (TGA, DDS, PSD).
    //
    // DDS and PSD are recognised by their MAGIC rather than their extension, because
    // the extension lies often enough to matter — a converted stage's .tga is
    // sometimes a DDS, and a repacked texture folder is full of .png that never
    // stopped being Photoshop files. TGA has no magic to key on, so .tga skips
    // straight to its decoder (createImageBitmap can't read it) and every other
    // extension tries the browser first, then falls back to TGA in case a
    // .spa/.sph/etc. is TGA underneath. Every failure is logged and soft — this
    // never throws to the caller; the material just gets the white texture.
    let source: ImageBitmap | null = null
    let rgba: Uint8Array | null = null
    let width: number
    let height: number

    const cpuDecoder = isDds(buffer) ? decodeDds : isPsd(buffer) ? decodePsd : null
    const isTga = logicalPath.toLowerCase().endsWith(".tga")
    if (!isTga && !cpuDecoder) {
      try {
        source = await createImageBitmap(new Blob([buffer]), { premultiplyAlpha: "none", colorSpaceConversion: "none" })
      } catch {
        source = null // not a browser-native image — try the CPU decoders below
      }
    }

    if (source) {
      width = source.width
      height = source.height
    } else {
      try {
        const img = (cpuDecoder ?? decodeTga)(buffer)
        rgba = img.rgba
        width = img.width
        height = img.height
      } catch (e) {
        console.warn(
          `[reze] texture decode failed (unsupported format?): ${logicalPath}`,
          e instanceof Error ? e.message : e,
        )
        return null
      }
    }

    // CPU alpha sampler for sheerness classification (see textureAlphaCache).
    // Canvas 2D premultiplies RGB on readback, but the ALPHA channel is exact.
    this.textureAlphaCache.set(cacheKey, buildAlphaSampler(source, rgba, width, height))

    const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1
    const texture = this.device.createTexture({
      label: `texture: ${cacheKey}`,
      size: [width, height],
      format: "rgba8unorm-srgb",
      mipLevelCount,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    if (source) {
      this.device.queue.copyExternalImageToTexture({ source }, { texture }, [width, height])
    } else {
      this.device.queue.writeTexture(
        { texture },
        rgba! as ArrayBufferView<ArrayBuffer>,
        { bytesPerRow: width * 4, rowsPerImage: height },
        [width, height],
      )
    }

    if (mipLevelCount > 1) this.generateMipmaps(texture, mipLevelCount)

    this.textureCache.set(cacheKey, texture)
    inst.textureCacheKeys.push(cacheKey)
    return texture
  }

  // Bilinear box-filter downsample per level. Reads srgb view (hardware linearizes on sample,
  // re-encodes on write), so intensities are filtered in linear space — matching EEVEE/Blender.
  private generateMipmaps(texture: GPUTexture, mipLevelCount: number) {
    if (!this.mipBlitPipeline || !this.mipBlitSampler) {
      this.mipBlitSampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "clamp-to-edge",
        addressModeV: "clamp-to-edge",
      })
      const module = this.device.createShaderModule({
        label: "mipmap blit",
        code: MIPMAP_BLIT_SHADER_WGSL,
      })
      this.mipBlitPipeline = this.device.createRenderPipeline({
        label: "mipmap blit pipeline",
        layout: "auto",
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format: "rgba8unorm-srgb" }] },
        primitive: { topology: "triangle-list" },
      })
    }

    const encoder = this.device.createCommandEncoder({ label: "mipgen" })
    for (let level = 1; level < mipLevelCount; level++) {
      const srcView = texture.createView({ baseMipLevel: level - 1, mipLevelCount: 1 })
      const dstView = texture.createView({ baseMipLevel: level, mipLevelCount: 1 })
      const bindGroup = this.device.createBindGroup({
        layout: this.mipBlitPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: srcView },
          { binding: 1, resource: this.mipBlitSampler },
        ],
      })
      const pass = encoder.beginRenderPass({
        colorAttachments: [
          { view: dstView, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        ],
      })
      pass.setPipeline(this.mipBlitPipeline)
      pass.setBindGroup(0, bindGroup)
      pass.draw(3)
      pass.end()
    }
    this.device.queue.submit([encoder.finish()])
  }

  private renderGround(pass: GPURenderPassEncoder) {
    // A stage brings its own floor. Both sit at y=0, so drawing the built-in
    // plane underneath produces z-fighting across the whole scene — enforced
    // here rather than left to callers, who cannot see the conflict coming.
    // hasGround is left alone: remove the stage and the ground comes back.
    if (this.groundIsSuppressed()) return
    if (!this.hasGround || !this.groundVertexBuffer || !this.groundIndexBuffer || !this.groundDrawCall) return
    pass.setPipeline(this.groundShadowPipeline)
    pass.setVertexBuffer(0, this.groundVertexBuffer)
    pass.setIndexBuffer(this.groundIndexBuffer, "uint16")
    pass.setBindGroup(0, this.groundDrawCall.bindGroup)
    pass.drawIndexed(this.groundDrawCall.count, 1, this.groundDrawCall.firstIndex, 0, 0)
  }

  private handleCanvasDoubleClick = (event: MouseEvent) => {
    if (!this.onRaycast || this.modelInstances.size === 0) return
    const rect = this.canvas.getBoundingClientRect()
    this.performRaycast(event.clientX - rect.left, event.clientY - rect.top)
  }

  private handleCanvasTouch = (event: TouchEvent) => {
    if (!this.onRaycast || this.modelInstances.size === 0) return

    // Prevent default to avoid triggering mouse events
    event.preventDefault()

    // Get the first touch
    const touch = event.changedTouches[0]
    if (!touch) return

    const currentTime = Date.now()
    const timeDiff = currentTime - this.lastTouchTime

    // Check for double-tap (within delay threshold)
    if (timeDiff < this.DOUBLE_TAP_DELAY) {
      const rect = this.canvas.getBoundingClientRect()
      const x = touch.clientX - rect.left
      const y = touch.clientY - rect.top

      this.performRaycast(x, y)
      // Reset last touch time to prevent triple-tap triggering double-tap
      this.lastTouchTime = 0
    } else {
      // Single tap - update last touch time for potential double-tap
      this.lastTouchTime = currentTime
    }
  }

  private performRaycast(screenX: number, screenY: number) {
    if (!this.onRaycast || this.modelInstances.size === 0) {
      this.onRaycast?.("", null, null, screenX, screenY)
      return
    }
    const dpr = window.devicePixelRatio || 1
    this.pendingPick = { x: Math.floor(screenX * dpr), y: Math.floor(screenY * dpr) }
  }

  private renderSelectionPasses(encoder: GPUCommandEncoder, swapchainView: GPUTextureView): void {
    if (!this.selectedMaterial || !this.selectionEdgeBindGroup) return
    const inst = this.modelInstances.get(this.selectedMaterial.modelName)
    if (!inst) return
    const target = this.selectedMaterial.materialName
    const draw = inst.drawCalls.find(
      (d) => (d.type === "opaque" || d.type === "transparent") && d.materialName === target,
    )
    if (!draw || !this.shouldRenderDrawCall(inst, draw)) return

    // Mask pass: fill the selected material's projected footprint with 1.0. Depth-always
    // (no depth attachment) so the outline traces complete boundaries even when the
    // material is partially occluded — matches Blender selection-through behaviour.
    const mpass = encoder.beginRenderPass(this.selectionMaskPassDescriptor)
    mpass.setPipeline(this.selectionMaskPipeline)
    mpass.setBindGroup(0, this.outlinePerFrameBindGroup)
    mpass.setBindGroup(1, inst.mainPerInstanceBindGroup)
    mpass.setVertexBuffer(0, inst.vertexBuffer)
    mpass.setVertexBuffer(1, inst.jointsBuffer)
    mpass.setVertexBuffer(2, inst.weightsBuffer)
    mpass.setIndexBuffer(inst.indexBuffer, "uint32")
    mpass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    mpass.end()

    // Edge pass: screen-space edge detect on the mask, alpha-blended over swapchain.
    const edgeAttachment = (this.selectionEdgePassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
    edgeAttachment.view = swapchainView
    const epass = encoder.beginRenderPass(this.selectionEdgePassDescriptor)
    epass.setPipeline(this.selectionEdgePipeline)
    epass.setBindGroup(0, this.selectionEdgeBindGroup)
    epass.draw(3)
    epass.end()
  }

  // Writes gizmo transform = T(bonePos) · R(boneWorldRot) · S(GIZMO_WORLD_SIZE),
  // then runs 6 triangle-list draws (3 axes + 3 rings). Local-axes mode: rotation
  // aligns rings with the bone's current world orientation, so clicking a ring
  // rotates around that bone's natural axis.
  private renderGizmoPass(encoder: GPUCommandEncoder, swapchainView: GPUTextureView): void {
    if (!this.selectedBone || !this.camera) return
    const inst = this.modelInstances.get(this.selectedBone.modelName)
    if (!inst) return
    const worldMats = inst.model.getWorldMatrices()
    if (this.selectedBone.boneIndex >= worldMats.length) return

    const boneMat = worldMats[this.selectedBone.boneIndex]
    const bonePos = boneMat.getPosition()
    const q = boneMat.toQuat().normalize() // world rotation
    const s = Engine.GIZMO_WORLD_SIZE

    // Column-major mat4: rotation columns × scale, then translation in col 3.
    const xx = q.x * q.x,
      yy = q.y * q.y,
      zz = q.z * q.z
    const xy = q.x * q.y,
      xz = q.x * q.z,
      yz = q.y * q.z
    const wx = q.w * q.x,
      wy = q.w * q.y,
      wz = q.w * q.z
    const u = new Float32Array(20)
    u[0] = s * (1 - 2 * (yy + zz))
    u[1] = s * 2 * (xy + wz)
    u[2] = s * 2 * (xz - wy)
    u[3] = 0
    u[4] = s * 2 * (xy - wz)
    u[5] = s * (1 - 2 * (xx + zz))
    u[6] = s * 2 * (yz + wx)
    u[7] = 0
    u[8] = s * 2 * (xz + wy)
    u[9] = s * 2 * (yz - wx)
    u[10] = s * (1 - 2 * (xx + yy))
    u[11] = 0
    u[12] = bonePos.x
    u[13] = bonePos.y
    u[14] = bonePos.z
    u[15] = 1
    u[16] = this.canvas.width
    u[17] = this.canvas.height
    u[18] = Engine.GIZMO_THICKNESS_PX
    u[19] = 0
    this.device.queue.writeBuffer(this.gizmoTransformBuffer, 0, u)

    const att = (this.gizmoPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
    att.view = swapchainView
    const pass = encoder.beginRenderPass(this.gizmoPassDescriptor)
    pass.setPipeline(this.gizmoPipeline)
    pass.setBindGroup(0, this.gizmoBindGroup0)
    pass.setVertexBuffer(0, this.gizmoVertexBuffer)
    for (const d of this.gizmoDraws) {
      pass.setBindGroup(1, this.gizmoColorBindGroups[d.color])
      pass.draw(d.count, 1, d.first, 0)
    }
    pass.end()
  }

  // ──────────────────────────────────────────────────────────────────
  // Gizmo drag — hit test + input handlers + rotation/translation math
  // ──────────────────────────────────────────────────────────────────

  private rotateVec3ByQuat(v: Vec3, q: Quat): Vec3 {
    // Standard rodrigues-via-quat formulation. Cheaper than q * v * q_conj.
    const tx = 2 * (q.y * v.z - q.z * v.y)
    const ty = 2 * (q.z * v.x - q.x * v.z)
    const tz = 2 * (q.x * v.y - q.y * v.x)
    return new Vec3(
      v.x + q.w * tx + (q.y * tz - q.z * ty),
      v.y + q.w * ty + (q.z * tx - q.x * tz),
      v.z + q.w * tz + (q.x * ty - q.y * tx),
    )
  }

  private unproject(invVP: Mat4, ndcX: number, ndcY: number, ndcZ: number): Vec3 | null {
    const m = invVP.values
    const x = m[0] * ndcX + m[4] * ndcY + m[8] * ndcZ + m[12]
    const y = m[1] * ndcX + m[5] * ndcY + m[9] * ndcZ + m[13]
    const z = m[2] * ndcX + m[6] * ndcY + m[10] * ndcZ + m[14]
    const w = m[3] * ndcX + m[7] * ndcY + m[11] * ndcZ + m[15]
    if (Math.abs(w) < 1e-9) return null
    return new Vec3(x / w, y / w, z / w)
  }

  // World-space ray from camera through a canvas pixel. Uses WebGPU's NDC z ∈ [0,1].
  private buildMouseRay(px: number, py: number): { origin: Vec3; dir: Vec3 } | null {
    if (!this.camera) return null
    const width = this.canvas.clientWidth
    const height = this.canvas.clientHeight
    if (width <= 0 || height <= 0) return null
    const ndcX = (px / width) * 2 - 1
    const ndcY = -((py / height) * 2 - 1)
    const view = this.camera.getViewMatrix()
    const proj = this.camera.getProjectionMatrix()
    const invVP = proj.multiply(view).inverse()
    const near = this.unproject(invVP, ndcX, ndcY, 0)
    const far = this.unproject(invVP, ndcX, ndcY, 1)
    if (!near || !far) return null
    return { origin: near, dir: far.subtract(near).normalize() }
  }

  // Finds the closest gizmo handle to the mouse ray, within `worldThreshold`.
  // `worldAxes[i]` is the i-th local axis rotated into world by bone world rotation.
  private hitTestGizmo(
    ray: { origin: Vec3; dir: Vec3 },
    bonePos: Vec3,
    gizmoSize: number,
    worldThreshold: number,
    worldAxes: [Vec3, Vec3, Vec3],
  ): { kind: "axis" | "ring"; axis: 0 | 1 | 2 } | null {
    let bestKind: "axis" | "ring" | null = null
    let bestAxis: 0 | 1 | 2 = 0
    let bestDist = worldThreshold

    // Axes only hit on their OUTER portion (past the ring radius). Inside the
    // ring the axis line passes through the plane of the perpendicular ring
    // (e.g. X-axis passes through the interior of the Y ring), so including the
    // full axis produced ring-vs-axis ties and constant misclicks. Axis extends
    // to AXIS_LENGTH, so the hit zone is roughly half the visible axis length —
    // easy to grab while leaving the ring's interior unambiguous.
    const axisHitStart = gizmoSize * (Engine.GIZMO_RING_RADIUS + 0.05)
    const axisHitEnd = gizmoSize * Engine.GIZMO_AXIS_LENGTH
    for (let i = 0; i < 3; i++) {
      const segA = bonePos.add(worldAxes[i].scale(axisHitStart))
      const segB = bonePos.add(worldAxes[i].scale(axisHitEnd))
      const d = this.distSegmentRay(segA, segB, ray.origin, ray.dir)
      if (d < bestDist) {
        bestDist = d
        bestKind = "axis"
        bestAxis = i as 0 | 1 | 2
      }
    }

    const ringR = gizmoSize * Engine.GIZMO_RING_RADIUS
    for (let i = 0; i < 3; i++) {
      const n = worldAxes[i]
      const denom = ray.dir.dot(n)
      if (Math.abs(denom) < 1e-6) continue
      const t = bonePos.subtract(ray.origin).dot(n) / denom
      if (t < 0) continue
      const hit = ray.origin.add(ray.dir.scale(t))
      const rel = hit.subtract(bonePos)
      const radial = rel.subtract(n.scale(rel.dot(n)))
      const radius = radial.length()
      const d = Math.abs(radius - ringR)
      if (d < bestDist) {
        bestDist = d
        bestKind = "ring"
        bestAxis = i as 0 | 1 | 2
      }
    }

    return bestKind ? { kind: bestKind, axis: bestAxis } : null
  }

  // Shortest distance between segment [A, B] and ray (origin, dir-unit).
  private distSegmentRay(A: Vec3, B: Vec3, rayO: Vec3, rayD: Vec3): number {
    const u = B.subtract(A) // segment direction (not normalized)
    const w = A.subtract(rayO)
    const a = u.dot(u)
    const b = u.dot(rayD)
    const d = u.dot(w)
    const e = rayD.dot(w)
    const denom = a - b * b // since |rayD|=1
    let sc: number, tc: number
    if (Math.abs(denom) < 1e-9) {
      sc = 0
      tc = e
    } else {
      sc = (b * e - d) / denom
      tc = (a * e - b * d) / denom
    }
    sc = Math.max(0, Math.min(1, sc))
    if (tc < 0) tc = 0
    const ps = new Vec3(A.x + sc * u.x, A.y + sc * u.y, A.z + sc * u.z)
    const pr = new Vec3(rayO.x + tc * rayD.x, rayO.y + tc * rayD.y, rayO.z + tc * rayD.z)
    return ps.subtract(pr).length()
  }

  // Line-line closest point: returns the parameter t on line (A, dir) where the
  // closest approach to the ray is. Used by axis-translation drag so frame N
  // reads a signed delta vs the mouse-down snapshot.
  private closestParamOnAxisLine(A: Vec3, dir: Vec3, rayO: Vec3, rayD: Vec3): number {
    const w = A.subtract(rayO)
    const b = dir.dot(rayD)
    const d = dir.dot(w)
    const e = rayD.dot(w)
    const denom = 1 - b * b // |dir|=|rayD|=1
    if (Math.abs(denom) < 1e-9) return -d // lines parallel
    return (b * e - d) / denom
  }

  // Ray-vs-plane (point bonePos, normal n). Returns the hit point or null.
  private rayPlane(rayO: Vec3, rayD: Vec3, bonePos: Vec3, n: Vec3): Vec3 | null {
    const denom = rayD.dot(n)
    if (Math.abs(denom) < 1e-6) return null
    const t = bonePos.subtract(rayO).dot(n) / denom
    if (t < 0) return null
    return rayO.add(rayD.scale(t))
  }

  // 2D angle of `hit` around `bonePos` in a plane spanned by (u, v). Basis vectors
  // are snapshotted at drag start so the angle frame is stable even if the bone
  // (and gizmo visual) rotates during the drag.
  private angleInRingPlane(hit: Vec3, bonePos: Vec3, u: Vec3, v: Vec3): number {
    const rel = hit.subtract(bonePos)
    return Math.atan2(rel.dot(v), rel.dot(u))
  }

  private handleGizmoMouseDown = (e: MouseEvent) => {
    if (!this.selectedBone || !this.camera || !this.device || e.button !== 0) return
    const inst = this.modelInstances.get(this.selectedBone.modelName)
    if (!inst) return
    const worldMats = inst.model.getWorldMatrices()
    const boneMat = worldMats[this.selectedBone.boneIndex]
    if (!boneMat) return
    const bonePos = boneMat.getPosition()
    const boneWorldRot = boneMat.toQuat().normalize()

    const rect = this.canvas.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const ray = this.buildMouseRay(px, py)
    if (!ray) return

    const gizmoSize = Engine.GIZMO_WORLD_SIZE

    // Bounding-sphere check: if the mouse ray passes inside an imaginary sphere
    // around the gizmo, ALWAYS consume the event — so the user never accidentally
    // orbits the camera while trying to click near a handle. Outside the sphere,
    // let the camera handler take over as normal.
    const sphereR = gizmoSize * Engine.GIZMO_AXIS_LENGTH * 1.05
    const f = ray.origin.subtract(bonePos)
    const fd = f.dot(ray.dir)
    const rayInsideSphere = fd * fd - (f.dot(f) - sphereR * sphereR) >= 0
    if (!rayInsideSphere) return

    // We're inside the gizmo's claim area — the event is ours regardless of hit.
    e.stopImmediatePropagation()
    e.preventDefault()

    // Pick threshold stays pixel-based — clicking should feel the same at any zoom.
    const camPos = this.camera.getEyePosition()
    const dist = Math.max(0.01, bonePos.subtract(camPos).length())
    const worldPerPixel = (dist * Math.tan(this.camera.fov * 0.5) * 2) / Math.max(1, this.canvas.clientHeight)
    const worldThreshold = Engine.GIZMO_PICK_THRESHOLD_PX * worldPerPixel

    // World-rotated local axes (where the visible gizmo arms actually point).
    const worldAxes: [Vec3, Vec3, Vec3] = [
      this.rotateVec3ByQuat(new Vec3(1, 0, 0), boneWorldRot),
      this.rotateVec3ByQuat(new Vec3(0, 1, 0), boneWorldRot),
      this.rotateVec3ByQuat(new Vec3(0, 0, 1), boneWorldRot),
    ]

    const hit = this.hitTestGizmo(ray, bonePos, gizmoSize, worldThreshold, worldAxes)
    if (!hit) return // Inside sphere but didn't hit a handle — event consumed, no drag.

    this.camera.setInputLocked(true)

    const parentIdx = inst.model.getSkeleton().bones[this.selectedBone.boneIndex].parentIndex
    const parentWorldRot =
      parentIdx >= 0 && parentIdx < worldMats.length ? worldMats[parentIdx].toQuat().normalize() : Quat.identity()
    const parentWorldRotInv = parentWorldRot.clone().conjugate()

    const worldAxis = worldAxes[hit.axis]
    // In-plane basis for the ring: u/v are the OTHER two world-rotated axes.
    //   X ring (normal X) → (u=Y, v=Z); Y ring → (u=Z, v=X); Z ring → (u=X, v=Y)
    const basisU = hit.axis === 0 ? worldAxes[1] : hit.axis === 1 ? worldAxes[2] : worldAxes[0]
    const basisV = hit.axis === 0 ? worldAxes[2] : hit.axis === 1 ? worldAxes[0] : worldAxes[1]

    let initialAngle = 0
    let initialAxisParam = 0
    if (hit.kind === "ring") {
      const p = this.rayPlane(ray.origin, ray.dir, bonePos, worldAxis)
      if (p) initialAngle = this.angleInRingPlane(p, bonePos, basisU, basisV)
    } else {
      initialAxisParam = this.closestParamOnAxisLine(bonePos, worldAxis, ray.origin, ray.dir)
    }

    const initialLocalRot = inst.model.getBoneLocalRotation(this.selectedBone.boneIndex).clone()
    const initTrans = inst.model.getBoneLocalTranslation(this.selectedBone.boneIndex)
    const initialLocalTrans = new Vec3(initTrans.x, initTrans.y, initTrans.z)

    this.gizmoDrag = {
      kind: hit.kind,
      axis: hit.axis,
      bonePos,
      worldAxis,
      basisU,
      basisV,
      initialLocalRot,
      initialLocalTrans,
      parentWorldRot,
      parentWorldRotInv,
      initialAngle,
      initialAxisParam,
    }

    if (this.onGizmoDrag) {
      this.onGizmoDrag({
        modelName: this.selectedBone.modelName,
        boneName: this.selectedBone.boneName,
        boneIndex: this.selectedBone.boneIndex,
        kind: hit.kind === "ring" ? "rotate" : "translate",
        localRotation: initialLocalRot.clone(),
        localTranslation: new Vec3(initialLocalTrans.x, initialLocalTrans.y, initialLocalTrans.z),
        phase: "start",
      })
    }
  }

  private handleGizmoMouseMove = (e: MouseEvent) => {
    const drag = this.gizmoDrag
    if (!drag || !this.selectedBone || !this.camera) return
    const inst = this.modelInstances.get(this.selectedBone.modelName)
    if (!inst) return

    const rect = this.canvas.getBoundingClientRect()
    const px = e.clientX - rect.left
    const py = e.clientY - rect.top
    const ray = this.buildMouseRay(px, py)
    if (!ray) return

    // Compute the target local rotation / translation. The engine never writes
    // to the skeleton itself — we hand the result to the host callback and let
    // it decide (runtime write, tween, clip keyframe edit, …).
    let nextRot = drag.initialLocalRot
    let nextTrans = drag.initialLocalTrans
    if (drag.kind === "ring") {
      const p = this.rayPlane(ray.origin, ray.dir, drag.bonePos, drag.worldAxis)
      if (!p) return
      const currentAngle = this.angleInRingPlane(p, drag.bonePos, drag.basisU, drag.basisV)
      const deltaAngle = currentAngle - drag.initialAngle
      const qWorld = Quat.fromAxisAngle(drag.worldAxis, deltaAngle)
      // L_new = P_inv · Q_world · P · L_initial
      const lNew = drag.parentWorldRotInv.multiply(qWorld).multiply(drag.parentWorldRot).multiply(drag.initialLocalRot)
      lNew.normalize()
      nextRot = lNew
    } else {
      const tNow = this.closestParamOnAxisLine(drag.bonePos, drag.worldAxis, ray.origin, ray.dir)
      const deltaParam = tNow - drag.initialAxisParam
      const worldDelta = drag.worldAxis.scale(deltaParam)
      const localDelta = this.rotateVec3ByQuat(worldDelta, drag.parentWorldRotInv)
      nextTrans = new Vec3(
        drag.initialLocalTrans.x + localDelta.x,
        drag.initialLocalTrans.y + localDelta.y,
        drag.initialLocalTrans.z + localDelta.z,
      )
    }

    this.onGizmoDrag?.({
      modelName: this.selectedBone.modelName,
      boneName: this.selectedBone.boneName,
      boneIndex: this.selectedBone.boneIndex,
      kind: drag.kind === "ring" ? "rotate" : "translate",
      localRotation: nextRot,
      localTranslation: nextTrans,
    })
  }

  private handleGizmoMouseUp = () => {
    const drag = this.gizmoDrag
    if (!drag) return
    if (this.onGizmoDrag && this.selectedBone) {
      const inst = this.modelInstances.get(this.selectedBone.modelName)
      if (inst) {
        const finalRot = inst.model.getBoneLocalRotation(this.selectedBone.boneIndex).clone()
        const t = inst.model.getBoneLocalTranslation(this.selectedBone.boneIndex)
        const finalTrans = new Vec3(t.x, t.y, t.z)
        this.onGizmoDrag({
          modelName: this.selectedBone.modelName,
          boneName: this.selectedBone.boneName,
          boneIndex: this.selectedBone.boneIndex,
          kind: drag.kind === "ring" ? "rotate" : "translate",
          localRotation: finalRot,
          localTranslation: finalTrans,
          phase: "end",
        })
      }
    }
    this.gizmoDrag = null
    this.camera?.setInputLocked(false)
  }

  private renderPickPass(encoder: GPUCommandEncoder): void {
    if (!this.pendingPick || !this.pickTexture || !this.pickDepthTexture) return

    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.pickTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.pickDepthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    })

    pass.setPipeline(this.pickPipeline)
    pass.setBindGroup(0, this.pickPerFrameBindGroup)

    this.forEachInstance((inst) => {
      if (!inst.model.visible) return // hidden models aren't pickable
      pass.setVertexBuffer(0, inst.vertexBuffer)
      pass.setVertexBuffer(1, inst.jointsBuffer)
      pass.setVertexBuffer(2, inst.weightsBuffer)
      pass.setIndexBuffer(inst.indexBuffer, "uint32")
      pass.setBindGroup(1, inst.pickPerInstanceBindGroup)
      for (const draw of inst.pickDrawCalls) {
        pass.setBindGroup(2, draw.bindGroup)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    })

    pass.end()

    // Copy the single pixel under cursor to readback buffer
    const px = Math.min(this.pendingPick.x, this.pickTexture.width - 1)
    const py = Math.min(this.pendingPick.y, this.pickTexture.height - 1)
    encoder.copyTextureToBuffer(
      { texture: this.pickTexture, origin: { x: Math.max(0, px), y: Math.max(0, py) } },
      { buffer: this.pickReadbackBuffer, bytesPerRow: 256 },
      { width: 1, height: 1 },
    )
  }

  private async resolvePickResult(screenX: number, screenY: number): Promise<void> {
    if (!this.onRaycast) return
    await this.pickReadbackBuffer.mapAsync(GPUMapMode.READ)
    const data = new Uint8Array(this.pickReadbackBuffer.getMappedRange())
    const modelId = data[0]
    const materialId = data[1]
    const boneId = data[2]
    this.pickReadbackBuffer.unmap()

    if (modelId === 0) {
      this.onRaycast("", null, null, screenX, screenY)
      return
    }

    // Find model by 1-based index
    let idx = 1
    let hitModel = ""
    for (const [name] of this.modelInstances) {
      if (idx === modelId) {
        hitModel = name
        break
      }
      idx++
    }

    let hitMaterial: string | null = null
    let hitBone: string | null = null
    if (hitModel) {
      const inst = this.modelInstances.get(hitModel)
      if (inst) {
        // Find material by 1-based index (skipping zero-vertex materials)
        const materials = inst.model.getMaterials()
        let matIdx = 0
        for (const mat of materials) {
          if (mat.vertexCount === 0) continue
          matIdx++
          if (matIdx === materialId) {
            hitMaterial = mat.name
            break
          }
        }
        // Bone index is 0-based (matches joints0 attribute values fed to pick shader).
        const bones = inst.model.getSkeleton().bones
        if (boneId < bones.length) hitBone = bones[boneId].name
      }
    }

    this.onRaycast(hitModel, hitMaterial, hitBone, screenX, screenY)
  }

  render() {
    if (!this.multisampleTexture || !this.camera || !this.device) return

    const currentTime = performance.now()
    const deltaTime = this.lastFrameTime > 0 ? (currentTime - this.lastFrameTime) / 1000 : 0.016
    this.lastFrameTime = currentTime
    this.renderWithDelta(deltaTime)
  }

  /**
   * Render one frame advancing every clock — animation, physics, tweens, and the
   * camera VMD — by exactly `deltaSeconds`, independent of wall time. This is the
   * offline-rendering primitive (video export): call it N times with 1/fps and the
   * result is deterministic whether the machine renders faster or slower than
   * realtime. Also resets the realtime clock so a later render() (returning to the
   * live loop) doesn't see the export's wall-clock gap as one giant delta.
   */
  renderFrame(deltaSeconds: number) {
    if (!this.multisampleTexture || !this.camera || !this.device) return
    this.lastFrameTime = performance.now()
    this.renderWithDelta(deltaSeconds)
  }

  private renderWithDelta(deltaTime: number) {
    // The scene clock, and the only clock a trail may sample on: renderFrame()
    // drives offline export with an exact per-frame delta, so a path recorded
    // against this is reproducible where one recorded against wall time is not.
    this.sceneClock += deltaTime
    this.trailAccum += deltaTime
    this.trailDue = Math.floor(this.trailAccum / TRAIL_DT)
    this.trailAccum -= this.trailDue * TRAIL_DT
    const tFrame = performance.now()
    this.frameAnimMsRaw = 0
    this.framePhysicsMsRaw = 0
    if (this.resizePending) {
      this.resizePending = false
      this.handleResize()
    }

    const hasModels = this.modelInstances.size > 0
    if (hasModels) {
      this.updateInstances(deltaTime)
      this.updateSkinMatrices()
      // Update camera target from bound model. Bone world matrices are model-space,
      // so compose the scene placement (setModelTransform) — otherwise the camera
      // ignores a moved/rotated model (code-driven root motion). Bone not found →
      // follow the model root itself.
      if (this.cameraTargetModel) {
        const m = this.cameraTargetModel
        const pos = m.getBoneWorldPosition(this.cameraTargetBoneName)
        let px = m.position.x
        let py = m.position.y
        let pz = m.position.z
        if (pos) {
          const s = m.scale
          pos.setXYZ(pos.x * s, pos.y * s, pos.z * s)
          Quat.rotateVecInto(m.rotation, pos, pos)
          px += pos.x
          py += pos.y
          pz += pos.z
        }
        px += this.cameraTargetOffset.x
        py += this.cameraTargetOffset.y
        pz += this.cameraTargetOffset.z
        const tau = this.cameraFollowSmoothing
        if (tau > 0 && this.cameraFollowSeeded) {
          // Exponential lag toward the bone: the handheld-camera feel. The
          // orbit pivot trails the target and eases in, never snapping.
          const k = 1 - Math.exp(-deltaTime / tau)
          const f = this.cameraFollowPos
          f.x += (px - f.x) * k
          f.y += (py - f.y) * k
          f.z += (pz - f.z) * k
          this.camera.target.x = f.x
          this.camera.target.y = f.y
          this.camera.target.z = f.z
        } else {
          this.cameraFollowPos.setXYZ(px, py, pz)
          this.cameraFollowSeeded = true
          this.camera.target.x = px
          this.camera.target.y = py
          this.camera.target.z = pz
        }
      }
    }

    // Drive the shot from the camera VMD (synced to the animated model's clock).
    if (this.camera.vmdDriven && this.cameraAnimation) {
      const pose = this.cameraAnimation.sample(this.cameraClockTime())
      if (pose) this.camera.setVmdPose(pose)
    }

    this.updateCameraUniforms()
    this.updateShadowLightVP()

    // Depth of field's entire disabled cost is this branch: depth stays in
    // TBDR tile memory (discard) unless something in the composite reads it this
    // frame. Two things can — the DoF gather, and the depth handed to a
    // foreground effect — and either one makes the pass store it.
    //
    // The uniform refresh is shared for the same reason: linearDepth() inverts
    // the z-buffer with projA/projB out of dofU[2], which track the camera's
    // near/far and so must be rewritten every frame either reader is live. A
    // foreground with a stale pair would read metres from the wrong frustum. The
    // write leaves dofU[0].x at 0 while DoF is off, so refreshing it does not
    // switch the gather on.
    const dofOn = this.depthOfField.enabled
    const depthRead = dofOn || (this.effect?.hasForeground ?? false) || this.trails !== null
    this.renderPassDescriptor.depthStencilAttachment!.depthStoreOp = depthRead ? "store" : "discard"
    if (depthRead) this.writeDepthOfFieldUniforms()

    const encoder = this.device.createCommandEncoder()

    // GPU vertex morphs: write morphed positions into vertex buffers before any pass reads
    // them. WebGPU inserts the storage→vertex barrier between this pass and the render passes.
    if (hasModels) this.dispatchMorphCompute(encoder)

    // Frustum cull into indirect arguments. After the camera and shadow matrices
    // are settled, before the passes that will eventually draw from them.
    if (hasModels) this.dispatchCull(encoder)

    // Runs one more time after the last model goes: this pass owns the shadow map's
    // only `depthLoadOp: "clear"`, so skipping it outright leaves the texture holding
    // the final frame's depth — and the ground, which draws on `hasGround` alone,
    // keeps PCF-sampling a character that is no longer in the scene. One clearing
    // pass on the transition to empty, then it stops.
    if (hasModels || this.shadowMapPopulated) {
      const sp = encoder.beginRenderPass({
        colorAttachments: [],
        depthStencilAttachment: {
          view: this.shadowMapDepthView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      })
      sp.setPipeline(this.shadowDepthPipeline)
      this.forEachInstance((inst) => {
        if (inst.model.visible) this.drawInstanceShadow(sp, inst)
      })
      sp.end()
      this.shadowMapPopulated = hasModels
    }

    // Before the particles and before the field pass: both may read the grid,
    // and a grid stepped after them is one frame stale in everything that used it.
    this.stepSim(encoder, deltaTime)
    this.stepParticles(encoder, deltaTime)

    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    // Phase order: opaque models → ground → transparent fabric.
    // The ground shader is the most expensive full-coverage draw in the frame
    // (9-tap PCF on the 4096² shadow map per pixel), so it draws AFTER the
    // opaque phase to get early-z rejected behind the body — drawing it first
    // shaded every covered pixel and measurably dropped Safari fps. It still
    // draws BEFORE the transparent phase so sheer fabric blends over the floor
    // instead of over the background with the floor depth-rejected behind it.
    if (hasModels)
      this.forEachInstance((inst) => {
        if (inst.model.visible) this.renderModelOpaquePhase(pass, inst)
      })
    if (this.hasGround) this.renderGround(pass)
    if (hasModels)
      this.forEachInstance((inst) => {
        if (inst.model.visible) this.renderModelTransparentPhase(pass, inst)
      })
    // Last in the pass: depth-tested against everything drawn above, so a
    // particle behind the character is simply hidden, and still inside the HDR
    // target so an `@bloom` effect reaches the pyramid below.
    this.renderParticles(pass)
    pass.end()

    // Ribbons draw AFTER the scene pass ends, so its depth is resolved for
    // their manual occlusion test — and before the composite that reads them.
    this.renderTrailLayer(encoder)
    // The field mounts, likewise: after the scene so foregrounds can read its
    // depth, before the composite that samples both layers.
    this.renderFieldPass(encoder)

    // Bloom pyramid (EEVEE 3.6):
    //   1. Blit: HDR → bloomDown[0] (Karis prefilter, half-res)
    //   2. Downsample: bloomDown[0] → bloomDown[1] → … → bloomDown[N-1] (13-tap)
    //   3. Upsample (top-down): bloomUp[N-2] = tent(bloomDown[N-1]) + bloomDown[N-2],
    //      then bloomUp[i] = tent(bloomUp[i+1]) + bloomDown[i] until i=0 (9-tap tent)
    //   Composite reads bloomUp[0] and adds tint * intensity * bloom before Filmic.
    if (this.bloomBlitBindGroup && this.compositeBindGroup && this.bloomMipCount > 0) {
      const bloomAtt = this.bloomPassDescriptor.colorAttachments as GPURenderPassColorAttachment[]

      // 1. Blit
      bloomAtt[0].view = this.bloomDownMipViews[0]
      const pBlit = encoder.beginRenderPass(this.bloomPassDescriptor)
      pBlit.setPipeline(this.bloomBlitPipeline)
      pBlit.setBindGroup(0, this.bloomBlitBindGroup)
      pBlit.draw(3)
      pBlit.end()

      // 2. Downsample chain
      for (let i = 1; i < this.bloomMipCount; i++) {
        bloomAtt[0].view = this.bloomDownMipViews[i]
        const p = encoder.beginRenderPass(this.bloomPassDescriptor)
        p.setPipeline(this.bloomDownsamplePipeline)
        p.setBindGroup(0, this.bloomDownsampleBindGroups[i - 1])
        p.draw(3)
        p.end()
      }

      // 3. Upsample chain (coarsest to finest; bindGroups[0] is the coarsest step)
      const upSteps = this.bloomUpsampleBindGroups.length
      const topIdx = this.bloomMipCount - 2
      for (let k = 0; k < upSteps; k++) {
        const levelIdx = topIdx - k // writes bloomUp[levelIdx]
        bloomAtt[0].view = this.bloomUpMipViews[levelIdx]
        const p = encoder.beginRenderPass(this.bloomPassDescriptor)
        p.setPipeline(this.bloomUpsamplePipeline)
        p.setBindGroup(0, this.bloomUpsampleBindGroups[k])
        p.draw(3)
        p.end()
      }
    }

    // Composite: HDR + bloom → Filmic tonemap → swapchain.
    const swapchainView = this.context.getCurrentTexture().createView()
    const compositeAttachment = (this.compositePassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
    compositeAttachment.view = swapchainView
    const cpass = encoder.beginRenderPass(this.compositePassDescriptor)
    const compositePipeline =
      this.viewTransform.gamma === 1.0 ? this.compositePipelineIdentity : this.compositePipelineGamma
    cpass.setPipeline(compositePipeline)
    cpass.setBindGroup(0, this.compositeBindGroup)
    cpass.draw(3)
    cpass.end()

    if (this.selectedMaterial && hasModels) this.renderSelectionPasses(encoder, swapchainView)
    if (this.selectedBone && hasModels) this.renderGizmoPass(encoder, swapchainView)

    const pick = this.pendingPick
    if (pick && hasModels) this.renderPickPass(encoder)

    this.device.queue.submit([encoder.finish()])

    // Everything this frame that wasn't animation or physics: uniforms, encoding, submit.
    const renderOnly = performance.now() - tFrame - this.frameAnimMsRaw - this.framePhysicsMsRaw
    this.cpuRenderMs += (renderOnly - this.cpuRenderMs) * 0.1

    if (pick) {
      this.pendingPick = null
      const dpr = window.devicePixelRatio || 1
      this.resolvePickResult(pick.x / dpr, pick.y / dpr)
    }

    // Feed the true vsync-to-vsync interval (deltaTime, computed at frame start), not the
    // CPU time spent in render() — that's what actually reflects perceived smoothness.
    this.updateStats(deltaTime * 1000)
  }

  private drawInstanceShadow(sp: GPURenderPassEncoder, inst: ModelInstance): void {
    sp.setBindGroup(0, inst.shadowBindGroup)
    sp.setVertexBuffer(0, inst.vertexBuffer)
    sp.setVertexBuffer(1, inst.jointsBuffer)
    sp.setVertexBuffer(2, inst.weightsBuffer)
    sp.setIndexBuffer(inst.indexBuffer, "uint32")
    for (const draw of inst.shadowDrawCalls) {
      if (!this.shouldRenderDrawCall(inst, draw)) continue
      if (!this.cullPasses(draw, true)) continue
      sp.setBindGroup(1, draw.bindGroup)
      sp.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    }
  }

  // ─── Style group API ──────────────────────────────────────────────
  // Two-tier edits: applyStyleGroups/upsert = topology (async compile + pipeline swap,
  // fallback-on-error, per-group stale guard); setStyleParam = adjust (instant uniform
  // write). Overlay-first: grouped materials render via their group's compiled graph;
  // ungrouped ones keep the hand-written preset path. See docs/style-groups-spec.md.

  /** Read a model's current style groups (including auto-created defaults) for editor
   *  round-trip. The host owns group state; this is bootstrap/read, not a second store. */
  getStyleGroups(modelName: string): StyleGroup[] {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return []
    return [...inst.styleGroups.values()].map((g) => g.group)
  }

  /**
   * Create default style groups from each material's resolved style category — `overrides`
   * (material name → category) first, then the built-in JP/CN/EN name hints. So a
   * standard-named model auto-groups with no overrides; a custom-named one passes a map for
   * the materials the hints miss. Unmatched materials (no override, no hint) stay ungrouped
   * (neutral default). The category picks the default graph + render-class + alpha-mode.
   * Resolves after grouping AND the async compiles, so `getStyleGroups` is then populated.
   */
  async autoStyleGroups(modelName: string, overrides?: MaterialPresetMap): Promise<ApplyStyleGroupsResult> {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return { ok: false, groups: [], unknownMaterials: [], conflicts: [] }

    const buckets = new Map<MaterialPreset, string[]>()
    for (const dc of inst.drawCalls) {
      if (!dc.baseBindGroupEntries) continue // material draw calls only (skip outlines)
      const preset = resolvePreset(dc.materialName, overrides)
      if (!preset) continue // unmatched → stays ungrouped (neutral default)
      const arr = buckets.get(preset) ?? []
      if (!arr.includes(dc.materialName)) arr.push(dc.materialName)
      buckets.set(preset, arr)
    }

    const groups: StyleGroup[] = []
    for (const [preset, materials] of buckets) {
      const info = PRESET_GROUP_INFO[preset]
      if (!info) continue
      groups.push({
        id: preset,
        label: info.graph.name,
        materials,
        graph: info.graph,
        renderClass: info.renderClass,
        alphaMode: info.alphaMode,
      })
    }
    return this.applyStyleGroups(modelName, groups)
  }

  /**
   * Replace a model's full style-group set. Unchanged groups (same graph + renderClass +
   * alphaMode) keep their pipeline; new/changed ones compile and swap; removed ones are
   * torn down and their materials revert to the ungrouped hand-shader path. A group whose
   * compile fails is not installed and its materials stay ungrouped (fallback-on-error).
   */
  async applyStyleGroups(modelName: string, groups: StyleGroup[]): Promise<ApplyStyleGroupsResult> {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return { ok: false, groups: [], unknownMaterials: [], conflicts: [] }

    // Whole-set validation: material claims (last group in array order wins), unknowns.
    const modelMaterials = new Set(inst.drawCalls.map((d) => d.materialName))
    const claimed = new Map<string, string>()
    const conflicts = new Set<string>()
    const unknownMaterials = new Set<string>()
    for (const g of groups) {
      for (const m of g.materials) {
        if (!modelMaterials.has(m)) unknownMaterials.add(m)
        if (claimed.has(m)) conflicts.add(m)
        claimed.set(m, g.id)
      }
    }

    // Tear down installs no longer present.
    const nextIds = new Set(groups.map((g) => g.id))
    for (const [id, install] of inst.styleGroups) {
      if (!nextIds.has(id)) {
        inst.styleGroupGen.set(id, (inst.styleGroupGen.get(id) ?? 0) + 1)
        this.destroyInstall(install)
        inst.styleGroups.delete(id)
      }
    }

    const groupResults: GroupDiagnostic[] = []
    for (const g of groups) {
      const r = await this.compileAndInstallGroup(inst, g)
      groupResults.push({ groupId: g.id, diagnostics: r.diagnostics, ok: r.ok })
    }

    this.assignDrawCallGroups(inst, claimed)
    return {
      ok: groupResults.every((r) => r.ok),
      groups: groupResults,
      unknownMaterials: [...unknownMaterials],
      conflicts: [...conflicts],
    }
  }

  /** Add or replace a single style group by id. `opts` may carry a `previewNode` for the
   *  editor's node-output preview workflow. */
  async upsertStyleGroup(modelName: string, group: StyleGroup, opts?: CompileOptions): Promise<ApplyStyleGroupResult> {
    const inst = this.modelInstances.get(modelName)
    if (!inst)
      return { ok: false, diagnostics: [{ severity: "error", message: `unknown model "${modelName}"` }], slotMap: [] }
    const r = await this.compileAndInstallGroup(inst, group, opts)
    this.assignDrawCallGroups(inst, this.currentClaims(inst))
    return r
  }

  /** Remove a style group; its materials revert to the ungrouped hand-shader path. */
  removeStyleGroup(modelName: string, groupId: string): void {
    const inst = this.modelInstances.get(modelName)
    const install = inst?.styleGroups.get(groupId)
    if (!inst || !install) return
    inst.styleGroupGen.set(groupId, (inst.styleGroupGen.get(groupId) ?? 0) + 1) // discard in-flight compile
    this.destroyInstall(install)
    inst.styleGroups.delete(groupId)
    this.assignDrawCallGroups(inst, this.currentClaims(inst))
  }

  /** Clear all style groups on a model — every material returns to the hand-shader path. */
  resetStyleGroups(modelName: string): void {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return
    for (const [id, install] of inst.styleGroups) {
      inst.styleGroupGen.set(id, (inst.styleGroupGen.get(id) ?? 0) + 1)
      this.destroyInstall(install)
    }
    inst.styleGroups.clear()
    this.assignDrawCallGroups(inst, new Map())
  }

  /** Instant adjust-tier write: set one exposed slider on a group's applied graph. */
  setStyleParam(
    modelName: string,
    groupId: string,
    paramId: string,
    value: number | [number, number, number],
  ): boolean {
    const install = this.modelInstances.get(modelName)?.styleGroups.get(groupId)
    const styleSlot = install?.slotMap.find((s) => s.id === paramId)
    if (!install || !styleSlot) return false
    if (styleSlot.kind === "float") {
      if (typeof value !== "number") return false
      const offset = styleSlot.vec4Index * 16 + ["x", "y", "z", "w"].indexOf(styleSlot.component!) * 4
      this.device.queue.writeBuffer(install.uniformBuffer, offset, new Float32Array([value]))
    } else {
      if (typeof value === "number") return false
      this.device.queue.writeBuffer(install.uniformBuffer, styleSlot.vec4Index * 16, new Float32Array(value))
    }
    return true
  }

  // Materials claimed by the model's currently-installed groups (for upsert/remove paths;
  // applyStyleGroups derives claims from its input array instead).
  private currentClaims(inst: ModelInstance): Map<string, string> {
    const claimed = new Map<string, string>()
    for (const install of inst.styleGroups.values())
      for (const m of install.group.materials) claimed.set(m, install.group.id)
    return claimed
  }

  // Compile a group's graph → WGSL → pipeline(s), install keyed by group id. Reuses the
  // install (pipeline + uniform buffer) when the graph/integration is byte-unchanged.
  private async compileAndInstallGroup(
    inst: ModelInstance,
    group: StyleGroup,
    opts?: CompileOptions,
  ): Promise<ApplyStyleGroupResult> {
    const renderClass = group.renderClass ?? "auto"
    const alphaMode = group.alphaMode ?? "opaque"
    const signature = JSON.stringify({ g: group.graph, rc: renderClass, am: alphaMode, o: opts?.previewNode ?? null })
    const existing = inst.styleGroups.get(group.id)
    if (existing && existing.signature === signature) {
      existing.group = group // refresh def (label/materials) without recompiling
      return { ok: true, diagnostics: [], slotMap: existing.slotMap }
    }

    const result = compileGraph(group.graph, { ...opts, renderClass, alphaMode })
    if (!result.ok) return { ok: false, diagnostics: result.diagnostics, slotMap: result.slotMap }

    const generation = (inst.styleGroupGen.get(group.id) ?? 0) + 1
    inst.styleGroupGen.set(group.id, generation)

    this.device.pushErrorScope("validation")
    const module = this.device.createShaderModule({ label: `style group: ${group.id} (${renderClass})`, code: result.wgsl })
    const info = await module.getCompilationInfo()
    const scopeError = await this.device.popErrorScope()
    const diagnostics = [...result.diagnostics]
    for (const msg of info.messages) {
      if (msg.type !== "error") continue
      diagnostics.push({ severity: "error", nodeId: nodeIdForWgslLine(result.wgsl, msg.lineNum), message: `WGSL: ${msg.message}` })
    }
    if (diagnostics.some((d) => d.severity === "error") || scopeError) {
      if (scopeError && !diagnostics.some((d) => d.severity === "error"))
        diagnostics.push({ severity: "error", message: `WGSL: ${scopeError.message}` })
      return { ok: false, diagnostics, slotMap: result.slotMap }
    }

    let pipeline: GPURenderPipeline
    let pipelineNoDepthWrite: GPURenderPipeline
    let overEyesPipeline: GPURenderPipeline | undefined
    try {
      pipeline = await this.createRenderClassPipeline(renderClass, module, false)
      // Dormant OIT twin — kept for a future order-independent-transparency path.
      pipelineNoDepthWrite = await this.createRenderClassPipeline(renderClass, module, false, false)
      if (renderClass === "hair") overEyesPipeline = await this.createRenderClassPipeline(renderClass, module, true)
    } catch (e) {
      diagnostics.push({ severity: "error", message: `pipeline creation failed: ${(e as Error).message}` })
      return { ok: false, diagnostics, slotMap: result.slotMap }
    }

    // Stale guard: a newer compile/remove for this id happened while we awaited.
    if (inst.styleGroupGen.get(group.id) !== generation) {
      diagnostics.push({ severity: "warning", message: "superseded by a newer edit — result discarded" })
      return { ok: false, diagnostics, slotMap: result.slotMap }
    }

    const uniformBuffer =
      existing?.uniformBuffer ??
      this.device.createBuffer({
        label: `style uniforms: ${group.id}`,
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })

    // The outgoing install's maps go with it — a re-apply that changes images
    // would otherwise strand the old textures for the life of the model.
    const previousImages = inst.styleGroups.get(group.id)?.images
    inst.styleGroups.set(group.id, {
      group,
      renderClass,
      alphaMode,
      pipeline,
      pipelineNoDepthWrite,
      overEyesPipeline,
      uniformBuffer,
      images: this.uploadGroupImages(group),
      slotMap: result.slotMap,
      signature,
    })
    // Rebind this group's draw calls before the old textures go.
    //
    // assignDrawCallGroups only rebuilds a bind group when a material CHANGES
    // group, which is the wrong test here: swapping the graph on a group a
    // material already belongs to leaves its id alone while replacing the maps
    // underneath it. The draw call then kept a bind group holding the outgoing
    // textures — destroyed on the next line — so a graph swap either sampled the
    // old maps, or the fallback white where the previous graph had none (which
    // reads as a blown-out white material through any screen or add), or tripped
    // a validation error on a destroyed texture. Only a reload cleared it.
    const install = inst.styleGroups.get(group.id)
    for (const dc of inst.drawCalls) {
      if (!dc.baseBindGroupEntries || dc.groupId !== group.id) continue
      dc.bindGroup = this.createMaterialBindGroup(
        `material: ${dc.materialName}`,
        dc.baseBindGroupEntries,
        uniformBuffer,
        install?.images,
      )
    }
    for (const tex of previousImages ?? []) tex?.destroy()
    this.writeGroupDefaults(uniformBuffer, group, result.slotMap)
    return { ok: true, diagnostics, slotMap: result.slotMap }
  }

  // Rebind each material draw call to its (successfully-installed) group's uniform buffer,
  // or the zero buffer when ungrouped, then re-sort by render-class draw order.
  private assignDrawCallGroups(inst: ModelInstance, claimed: Map<string, string>): void {
    inst.materialToGroup.clear()
    for (const dc of inst.drawCalls) {
      if (!dc.baseBindGroupEntries) continue // outlines/ground are never grouped
      const wantId = claimed.get(dc.materialName)
      const install = wantId ? inst.styleGroups.get(wantId) : undefined
      const groupId = install ? wantId! : null
      if (groupId) inst.materialToGroup.set(dc.materialName, groupId)
      if (dc.groupId === groupId) continue
      dc.groupId = groupId
      dc.bindGroup = this.createMaterialBindGroup(
        `material: ${dc.materialName}`,
        dc.baseBindGroupEntries,
        install ? install.uniformBuffer : this.zeroStyleBuffer,
        install?.images,
      )
    }
    this.sortDrawCalls(inst)
  }

  private writeGroupDefaults(buffer: GPUBuffer, group: StyleGroup, slotMap: StyleSlot[]): void {
    const data = new Float32Array(64) // 16 vec4f
    for (const styleSlot of slotMap) {
      const param = group.graph.params?.find((p) => p.id === styleSlot.id)
      if (!param) continue
      const base = styleSlot.vec4Index * 4
      if (styleSlot.kind === "float" && typeof param.default === "number") {
        data[base + ["x", "y", "z", "w"].indexOf(styleSlot.component!)] = param.default
      } else if (styleSlot.kind === "color" && typeof param.default !== "number") {
        data.set(param.default.slice(0, 3), base)
      }
    }
    this.device.queue.writeBuffer(buffer, 0, data)
  }

  // Draw-order rank within a bucket: eye stamps before hair reads. Purely from the group's
  // render-class — ungrouped materials are neutral (rank 0, no stencil interplay).
  private drawCallRank(inst: ModelInstance, dc: DrawCall): number {
    const rc = dc.groupId ? (inst.styleGroups.get(dc.groupId)?.renderClass ?? "auto") : "auto"
    return rc === "hair" ? 2 : rc === "eye" ? 1 : 0
  }

  private sortDrawCalls(inst: ModelInstance): void {
    const typeOrder: Record<DrawCallType, number> = {
      opaque: 0,
      "opaque-outline": 1,
      transparent: 2,
      "transparent-outline": 3,
      ground: 4,
    }
    inst.drawCalls.sort(
      (a, b) => typeOrder[a.type] - typeOrder[b.type] || this.drawCallRank(inst, a) - this.drawCallRank(inst, b),
    )
    inst.shadowDrawCalls = inst.drawCalls.filter(
      (d) => (d.type === "opaque" || d.type === "transparent") && d.castsShadow === true,
    )
    // The sort reorders drawCalls, and a draw's position in that array is its
    // slot in every cull buffer.
    this.cullListDirty = true
  }

  /**
   * Render-class pipeline state. A group's compiled graph swaps the fragment shading; the
   * render-class owns pass integration (stencil interplay, depth bias, cull). auto = plain;
   * eye = stamp + front cull + bias; hair = stencil-test (+ the over-eyes variant).
   */
  private createRenderClassPipeline(
    renderClass: RenderClass,
    module: GPUShaderModule,
    overEyes: boolean,
    depthWrite = true,
  ): Promise<GPURenderPipeline> {
    const base = {
      label: `style ${renderClass}${overEyes ? " (over eyes)" : ""}`,
      layout: this.mainPipelineLayout,
      vertex: { module, buffers: this.fullVertexBufferLayouts },
      primitive: { cullMode: (renderClass === "eye" ? "front" : "none") as GPUCullMode },
      multisample: { count: Engine.MULTISAMPLE_COUNT },
    }
    const plainDepth: GPUDepthStencilState = {
      format: "depth24plus-stencil8",
      depthWriteEnabled: depthWrite,
      depthCompare: "less-equal",
    }
    let depthStencil: GPUDepthStencilState = plainDepth
    let constants: Record<string, number> | undefined
    if (renderClass === "hair" && !overEyes) {
      depthStencil = {
        ...plainDepth,
        stencilFront: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0,
      }
    } else if (renderClass === "hair" && overEyes) {
      constants = { IS_OVER_EYES: 1 }
      depthStencil = {
        format: "depth24plus-stencil8",
        depthWriteEnabled: false,
        depthCompare: "less-equal",
        stencilFront: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0,
      }
    } else if (renderClass === "eye") {
      depthStencil = {
        ...plainDepth,
        depthBias: -0.00005,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
        stencilFront: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "replace" },
        stencilBack: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "replace" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0xff,
      }
    }
    return this.device.createRenderPipelineAsync({
      ...base,
      fragment: { module, constants, targets: this.sceneTargets },
      depthStencil,
    })
  }

  // Pipeline for a material draw call: its group's compiled pipeline when grouped, else
  // the neutral base (ungrouped materials render the default graph). Transparent-bucket
  // draws use the SAME depth-write-on pipeline — babylon-mmd's forceDepthWrite
  // blending (see renderModelTransparentPhase for the trade-off record).
  private pipelineForDrawCall(inst: ModelInstance, dc: DrawCall): GPURenderPipeline {
    if (dc.groupId) {
      const install = inst.styleGroups.get(dc.groupId)
      if (install) return install.pipeline
    }
    return this.neutralPipeline
  }

  /**
   * Draw every material of a given type (`opaque` or `transparent`) using the main
   * pipeline(s), and — babylon-mmd's per-mesh outline stage — each edge-flagged
   * material's inverted hull IMMEDIATELY after its color draw. Interleaving is what
   * makes outlines compose like MMD: every material drawn later in the author's
   * order covers earlier hulls, and each hull sits over everything drawn before it.
   */
  private drawMaterials(pass: GPURenderPassEncoder, inst: ModelInstance, type: "opaque" | "transparent"): void {
    let currentPipeline: GPURenderPipeline | null = null
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== type || !this.shouldRenderDrawCall(inst, draw)) continue
      if (!this.cullPasses(draw, false)) continue
      if (!bound) {
        pass.setBindGroup(0, this.perFrameBindGroup)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      const pipeline = this.pipelineForDrawCall(inst, draw)
      if (pipeline !== currentPipeline) {
        pass.setPipeline(pipeline)
        currentPipeline = pipeline
      }
      pass.setBindGroup(2, draw.bindGroup)
      pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      if (draw.outline && this.outlineEnabled) {
        // Same index range; own pipeline + groups 0/2. Group 1 (skinMats) is
        // layout-identical between the main and outline pipelines and stays
        // bound. Restore group 0 afterwards and force a pipeline re-set.
        pass.setPipeline(this.outlinePipeline)
        pass.setBindGroup(0, this.outlinePerFrameBindGroup)
        pass.setBindGroup(2, draw.outline.bindGroup)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
        pass.setBindGroup(0, this.perFrameBindGroup)
        currentPipeline = null
      }
    }
  }

  /**
   * Main-pass render sequence for one model instance — babylon-mmd parity:
   * opaque bucket, the hair-over-eyes stencil pass, then alpha-blend materials
   * in PMX author order with depth write ON (forceDepthWrite). Outlines are not
   * a separate phase: drawMaterials draws each edge-flagged material's hull
   * right after the material itself, like MMD's per-mesh outline stage.
   */
  private setModelDrawState(pass: GPURenderPassEncoder, inst: ModelInstance): void {
    pass.setVertexBuffer(0, inst.vertexBuffer)
    pass.setVertexBuffer(1, inst.jointsBuffer)
    pass.setVertexBuffer(2, inst.weightsBuffer)
    pass.setIndexBuffer(inst.indexBuffer, "uint32")
    // Single stencil-reference set covers eye (write), hair (read not-equal),
    // and hairOverEyes (read equal). Non-stencil pipelines ignore the value.
    pass.setStencilReference(Engine.STENCIL_EYE_VALUE)
  }

  private renderModelOpaquePhase(pass: GPURenderPassEncoder, inst: ModelInstance): void {
    this.setModelDrawState(pass, inst)
    this.drawMaterials(pass, inst, "opaque")
    this.drawHairOverEyes(pass, inst)
  }

  private renderModelTransparentPhase(pass: GPURenderPassEncoder, inst: ModelInstance): void {
    this.setModelDrawState(pass, inst)
    // Transparent: babylon-mmd's forceDepthWrite blending — PMX author order
    // with depth write ON. The accepted trade-off after trying every variant:
    //   · depth-write ON (this): a fold hides its far side; rare view-dependent
    //     double-blend seams at some angles. MMD's own known behavior.
    //   · nearest-surface prepass: view-independent, but punched see-through
    //     holes to whatever sat far behind a fold.
    //   · depth-write OFF layering: every overlap visible everywhere — MORE
    //     gray patches and texture artifacts in practice.
    this.drawMaterials(pass, inst, "transparent")
  }

  /** Depth-only re-draw of transparent-bucket materials (see depth-prepass.ts).
   *  Dormant — kept for a future order-independent-transparency path. */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected drawTransparentDepthPrepass(pass: GPURenderPassEncoder, inst: ModelInstance): void {
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== "transparent" || !this.shouldRenderDrawCall(inst, draw)) continue
      if (!bound) {
        pass.setPipeline(this.transparentDepthPrepassPipeline)
        pass.setBindGroup(0, this.perFrameBindGroup)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      pass.setBindGroup(2, draw.bindGroup)
      pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    }
  }

  /**
   * Second hair pass for the see-through-hair effect. Re-draws every hair-class grouped
   * opaque draw with its compiled over-eyes pipeline — stencil-matched to `EYE_VALUE`,
   * `IS_OVER_EYES=true` (25% alpha), depth-write off. Ungrouped materials are neutral and
   * never participate.
   */
  private drawHairOverEyes(pass: GPURenderPassEncoder, inst: ModelInstance): void {
    let bound = false
    let currentPipeline: GPURenderPipeline | null = null
    for (const draw of inst.drawCalls) {
      if (draw.type !== "opaque" || !this.shouldRenderDrawCall(inst, draw)) continue
      if (!this.cullPasses(draw, false)) continue
      const overEyes = this.overEyesPipelineFor(inst, draw)
      if (!overEyes) continue
      if (!bound) {
        pass.setBindGroup(0, this.perFrameBindGroup)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      if (overEyes !== currentPipeline) {
        pass.setPipeline(overEyes)
        currentPipeline = overEyes
      }
      pass.setBindGroup(2, draw.bindGroup)
      pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    }
  }

  // The over-eyes pipeline for a hair-class grouped draw call, or null. Ungrouped
  // materials are neutral — no see-through pass.
  private overEyesPipelineFor(inst: ModelInstance, dc: DrawCall): GPURenderPipeline | null {
    if (!dc.groupId) return null
    const install = inst.styleGroups.get(dc.groupId)
    return install?.renderClass === "hair" ? (install.overEyesPipeline ?? null) : null
  }

  private updateCameraUniforms() {
    const viewMatrix = this.camera.getViewMatrix()
    const projectionMatrix = this.camera.getProjectionMatrix()
    const cameraPos = this.camera.getEyePosition()
    this.cameraMatrixData.set(viewMatrix.values, 0)
    this.cameraMatrixData.set(projectionMatrix.values, 16)
    this.cameraMatrixData[32] = cameraPos.x
    this.cameraMatrixData[33] = cameraPos.y
    this.cameraMatrixData[34] = cameraPos.z
    // Spare float after viewPos: render-target height in device px — the outline
    // shader derives the full viewport (width via projection aspect) for its
    // babylon-mmd constant-pixel edge extrusion.
    this.cameraMatrixData[35] = this.canvas.height
    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)

    // 360 backdrop: the composite reconstructs each pixel's view ray from the
    // camera basis — refresh it every frame the skybox is active. The view matrix
    // is LEFT-HANDED (+Z forward, see Mat4.lookAtInto), so the world-space
    // right/up/FORWARD vectors are rows 0/1/2 of its rotation block directly
    // (column-major storage: row i = values[i], values[i+4], values[i+8]).
    if ((this.backdropEquirectView || this.effect) && this.compositeUniformBuffer) {
      const v = viewMatrix.values
      const u = this.compositeUniformData
      const tanHalf = Math.tan((this.camera.fov ?? Math.PI / 4) / 2)
      const aspect = this.canvas.width / Math.max(1, this.canvas.height)
      u[12] = v[0]
      u[13] = v[4]
      u[14] = v[8]
      u[15] = tanHalf * aspect
      u[16] = v[1]
      u[17] = v[5]
      u[18] = v[9]
      u[19] = tanHalf
      u[20] = v[2]
      u[21] = v[6]
      u[22] = v[10]
      u[23] = 0
      // Effect clock + canvas size (viewU[6]) — written on the same refresh.
      // The effect clock, on the SCENE's time rather than the wall's.
      //
      // renderFrame() drives offline export as fast as the encoder will take
      // frames, so wall time races ahead of the video's own time — a rain effect
      // fell at the wrong rate in the export, a twinkle blinked at the wrong
      // speed, and none of it matched what the editor had shown. Measured
      // against the accumulated frame delta, an effect animates identically in
      // the editor, in an export, and in a re-export, which is the same rule the
      // trails already followed.
      u[24] = this.sceneClock - this.effectEpochScene
      u[26] = this.canvas.width
      u[27] = this.canvas.height
      // Camera world position (viewU[10]) — the other half of bgWorldPos. It
      // rides this refresh rather than writeCompositeViewUniforms because it
      // changes every frame the camera does, exactly like the basis above.
      u[40] = cameraPos.x
      u[41] = cameraPos.y
      u[42] = cameraPos.z
      // Character positions (viewU[11..14]), count in viewU[10].w. Stages are
      // excluded: an effect asking where the cast is means the characters, and a
      // stage's origin is wherever its author put it, which is not a place
      // anything is standing. Four is the cap because the uniform is small and a
      // scene with five characters is not the case this serves.
      let n = 0
      this.forEachInstance((inst) => {
        if (n >= MAX_EFFECT_SUBJECTS || inst.isStage) return
        const m = inst.model
        // The model transform is only where the model was PLACED. A motion moves
        // the character by animating bones, so an effect anchored to the
        // transform never follows anyone anywhere — it sits at the spawn point
        // while they walk out of it. Composed exactly as the follow camera
        // composes it, for the same reason: bone matrices are model-space.
        let px = m.position.x
        let py = m.position.y
        let pz = m.position.z
        for (const bone of SUBJECT_BONES) {
          const pos = m.getBoneWorldPosition(bone)
          if (!pos) continue
          const sc = m.scale
          pos.setXYZ(pos.x * sc, pos.y * sc, pos.z * sc)
          Quat.rotateVecInto(m.rotation, pos, pos)
          px += pos.x
          py += pos.y
          pz += pos.z
          break
        }
        u[44 + n * 4] = px
        u[45 + n * 4] = py
        u[46 + n * 4] = pz
        this.writeCastEntry(inst, n, px, py, pz)
        n++
      })
      u[43] = n
      this.device.queue.writeBuffer(this.compositeUniformBuffer, 0, u)
      // Only what an effect declared, and only while one is installed. A scene
      // with no effect writes nothing here at all.
      if (this.effect) {
        // Up to the last trailed slot, not the whole buffer: an effect with no
        // trails never uploads the 32KB it would otherwise pay for every frame.
        let lastTrail = -1
        for (let i = 0; i < this.effect.anchors.length; i++) if (this.effect.anchors[i].trail) lastTrail = i
        const used =
          lastTrail >= 0
            ? CAST_TRAIL_BASE + (lastTrail * MAX_EFFECT_SUBJECTS + MAX_EFFECT_SUBJECTS) * TRAIL_SAMPLES
            : CAST_SUBJECT_VEC4S + this.effect.anchors.length * MAX_EFFECT_SUBJECTS * 3
        this.device.queue.writeBuffer(this.castBuffer, 0, this.castData, 0, used * 4)
        this.castLastMs = performance.now()
      }
    }
  }

  /**
   * One character's slice of the effect API's view of the cast.
   *
   * `px/py/pz` is the hip point the caller just composed — passed in rather than
   * recomputed, since it is the same two bone lookups.
   *
   * Bone positions are model-space, so each is scaled, rotated and translated by
   * the model transform exactly as the hip point above was. Getting that wrong
   * does not look wrong on a model standing at the origin, which is precisely
   * how it would ship.
   */
  private writeCastEntry(inst: ModelInstance, n: number, px: number, py: number, pz: number): void {
    const effect = this.effect
    if (!effect) return
    const m = inst.model
    const cd = this.castData
    const toWorld = (v: Vec3): Vec3 => {
      v.setXYZ(v.x * m.scale, v.y * m.scale, v.z * m.scale)
      Quat.rotateVecInto(m.rotation, v, v)
      v.setXYZ(v.x + m.position.x, v.y + m.position.y, v.z + m.position.z)
      return v
    }

    // The floor under this character: where the model was PLACED.
    //
    // A foot bone was the obvious answer and the wrong one. 足ＩＫ sits at the
    // ANKLE, not on the sole — an ankle above the ground even standing still,
    // and further still in heels — so a floor derived from it lands a hand's
    // width up the leg, which is exactly where the first version of Footfalls
    // drew its marks. A PMX's origin is between the feet on the floor by
    // convention, and placing a character on a stage moves that origin with
    // them, so the placement already answers "what is the ground here".
    //
    // Deliberately NOT the animated height: a jump lifts the character, not the
    // floor, and a floor that follows a jump is not a floor.
    const floorY = m.position.y
    // Generous on purpose: this is for culling, and a sphere that is too small
    // clips the effect it was meant to bound. Height is hip-to-head doubled;
    // arm span is about height on a human, so half of it is the radius, and the
    // rest is margin for a motion that reaches.
    const head = m.getBoneWorldPosition(HEAD_BONE)
    const height = head ? Math.max(0.01, toWorld(head).y - floorY) : Math.max(0.01, (py - floorY) * 2)
    const b = n * 12
    cd[b] = px
    cd[b + 1] = floorY
    cd[b + 2] = pz
    cd[b + 3] = 1
    cd[b + 4] = px
    cd[b + 5] = py
    cd[b + 6] = pz
    cd[b + 8] = px
    cd[b + 9] = floorY + height * 0.5
    cd[b + 10] = pz
    cd[b + 11] = height * 0.75

    // Declared bones. Velocity is per model AND per slot, so two characters
    // wearing the same effect never inherit each other's motion.
    const anchors = effect.anchors
    if (anchors.length === 0) return
    let prev = this.anchorPrev.get(inst.name)
    const dtMs = Math.max(1, performance.now() - this.castLastMs)
    const invDt = 1000 / dtMs
    if (!prev || prev.length !== anchors.length * 3) {
      prev = new Float32Array(anchors.length * 3).fill(NaN)
      this.anchorPrev.set(inst.name, prev)
    }
    for (let s = 0; s < anchors.length; s++) {
      const a = CAST_SUBJECT_VEC4S * 4 + (s * MAX_EFFECT_SUBJECTS + n) * 12
      const pos = m.getBoneWorldPosition(anchors[s].bone)
      if (!pos) {
        cd[a + 3] = 0
        continue
      }
      toWorld(pos)
      const p = s * 3
      // NaN on the first frame a slot exists — a velocity out of nothing would
      // be a spike, and a trail or a spark reading it would fire on load.
      const vx = Number.isNaN(prev[p]) ? 0 : (pos.x - prev[p]) * invDt
      const vy = Number.isNaN(prev[p]) ? 0 : (pos.y - prev[p + 1]) * invDt
      const vz = Number.isNaN(prev[p]) ? 0 : (pos.z - prev[p + 2]) * invDt
      prev[p] = pos.x
      prev[p + 1] = pos.y
      prev[p + 2] = pos.z
      cd[a] = pos.x
      cd[a + 1] = pos.y
      cd[a + 2] = pos.z
      cd[a + 3] = 1
      cd[a + 4] = vx
      cd[a + 5] = vy
      cd[a + 6] = vz
      if (anchors[s].trail) this.writeTrail(inst.name, s, n, pos, cd, a)
      const fwd = m.getBoneWorldForward(anchors[s].bone)
      if (fwd) {
        Quat.rotateVecInto(m.rotation, fwd, fwd)
        cd[a + 8] = fwd.x
        cd[a + 9] = fwd.y
        cd[a + 10] = fwd.z
      }
    }
  }

  /**
   * One trailed anchor's recent path, sampled on the scene clock and written
   * newest-first.
   *
   * Newest-first is what lets a ribbon be drawn by walking the index upward and
   * fading on age, and it means the shader never needs to know where the ring's
   * head is. Sixty-four entries is short enough that unshifting beats the
   * bookkeeping an actual ring buffer would push onto the GPU side too.
   *
   * Sampling is gated on TRAIL_DT of SCENE time, so a 120Hz display and a 30fps
   * export record the same path at the same spacing. A frame that covers several
   * intervals emits several samples rather than one, or a fast hand would tear.
   */
  private writeTrail(model: string, slot: number, n: number, pos: Vec3, cd: Float32Array, anchorBase: number): void {
    const key = `${model}\u0000${slot}`
    let ring = this.anchorTrail.get(key)
    if (!ring) {
      ring = { pos: [], t: [] }
      this.anchorTrail.set(key, ring)
    }
    // A TELEPORT is not motion. A model popping from the origin to its place at
    // load, a scrub, a scene swap — the bone genuinely moves many units in one
    // frame, and a recorder that faithfully keeps both ends hands every reader a
    // path across the world: the ribbon drew it as a streak and the sparks
    // seeded a burst along it. Fifty units per second is far beyond any dance
    // (a hard flick peaks around twenty); past it, the history restarts here.
    if (ring.pos.length > 0) {
      const dx = pos.x - ring.pos[0]
      const dy = pos.y - ring.pos[1]
      const dz = pos.z - ring.pos[2]
      const dt = Math.max(1 / 120, this.sceneClock - ring.t[0])
      if (Math.hypot(dx, dy, dz) / dt > 50) {
        ring.pos.length = 0
        ring.t.length = 0
      }
    }
    if (this.trailDue > 0 || ring.pos.length === 0) {
      // ONE sample per frame, never one per due tick. A frame that spanned
      // several 60Hz ticks only knows where the bone is NOW, and unshifting that
      // position once per tick fabricated duplicate samples — same point, same
      // timestamp, up to four copies — precisely when the scene ran heavy. Every
      // duplicate pair kinked the spline, and each kink drew as a bright bar
      // across the ribbon: banding that appeared under load, was spaced once per
      // frame, and survived every renderer fix because the renderer was
      // faithfully drawing corrupted history. Coarser spacing under load is
      // honest — each sample carries its true timestamp, and the spline and the
      // central-difference weight exist to handle uneven spacing.
      ring.pos.unshift(pos.x, pos.y, pos.z)
      ring.t.unshift(this.sceneClock)
      if (ring.t.length > TRAIL_SAMPLES) {
        ring.t.length = TRAIL_SAMPLES
        ring.pos.length = TRAIL_SAMPLES * 3
      }
    }
    const count = ring.t.length
    // Age rather than a timestamp: the shader would otherwise need the scene
    // clock too, and there is only one place that has to know what time it is.
    const base = (CAST_TRAIL_BASE + (slot * MAX_EFFECT_SUBJECTS + n) * TRAIL_SAMPLES) * 4
    for (let i = 0; i < count; i++) {
      cd[base + i * 4] = ring.pos[i * 3]
      cd[base + i * 4 + 1] = ring.pos[i * 3 + 1]
      cd[base + i * 4 + 2] = ring.pos[i * 3 + 2]
      cd[base + i * 4 + 3] = this.sceneClock - ring.t[i]
    }
    // The count rides in the anchor's spare lane, so rzTrailCount is one read.
    cd[anchorBase + 11] = count
  }

  private updateSkinMatrices() {
    this.forEachInstance((inst) => {
      // Only a pose pass can change these, and an idle stage did not run one —
      // re-uploading bones×64 bytes for scenery that never moves is the one
      // per-frame cost a stage would otherwise still pay in full.
      if (!inst.skinMatricesDirty) return
      const skinMatrices = inst.model.getSkinMatrices()
      this.device.queue.writeBuffer(
        inst.skinMatrixBuffer,
        0,
        skinMatrices.buffer,
        skinMatrices.byteOffset,
        skinMatrices.byteLength,
      )
      inst.skinMatricesDirty = false
      // Re-decide how this model is bounded, here and only here: the skin
      // matrices are the one thing that can change the answer, and an idle stage
      // never reaches this line twice.
      const boneCount = Math.floor(skinMatrices.length / 16)
      inst.rigid = boneCount > 0 && skinMatricesAgree(skinMatrices, boneCount)
      if (inst.rigid) inst.rigidXform.set(skinMatrices.subarray(0, 16))
    })
  }

  // frameIntervalMs is the true vsync-to-vsync frame interval (render dt), NOT the CPU
  // time spent in render() — the latter misses GPU cost and pacing, so it can read fast
  // while the scene stutters. Metrics are derived from a ring buffer of these intervals.
  private updateStats(frameIntervalMs: number) {
    const w = Engine.STATS_WINDOW
    this.frameIntervals[this.frameIntervalWrite] = frameIntervalMs
    this.frameIntervalWrite = (this.frameIntervalWrite + 1) % w
    if (this.frameIntervalFilled < w) this.frameIntervalFilled++

    const now = performance.now()
    if (now - this.lastStatsCompute < Engine.STATS_REFRESH_MS) return
    this.lastStatsCompute = now

    const n = this.frameIntervalFilled
    if (n === 0) return

    let sum = 0
    let max = 0
    for (let i = 0; i < n; i++) {
      const v = this.frameIntervals[i]
      sum += v
      if (v > max) max = v
    }
    const mean = sum / n

    let varSum = 0
    for (let i = 0; i < n; i++) {
      const d = this.frameIntervals[i] - mean
      varSum += d * d
    }
    const stddev = Math.sqrt(varSum / n)

    // 99th-percentile frame interval → "1% low" fps. TypedArray.sort is numeric.
    const sorted = this.frameIntervals.slice(0, n).sort()
    const p99 = sorted[Math.min(n - 1, Math.floor(n * 0.99))]

    // fps from the MEAN interval is inherently bounded by the real refresh (a vsync-locked
    // interval can't average below the refresh period), so this never reads above the
    // monitor rate — fixing the old frame-count/window off-by-one (61 on 60Hz, 241 on 240Hz).
    this.stats.fps = mean > 0 ? Math.round(1000 / mean) : 0
    this.stats.frameTime = Math.round(mean * 100) / 100
    this.stats.frameTimeMax = Math.round(max * 100) / 100
    this.stats.fps1PercentLow = p99 > 0 ? Math.round(1000 / p99) : 0
    this.stats.jitter = Math.round(stddev * 100) / 100
    this.stats.cpuAnimMs = Math.round(this.cpuAnimMs * 100) / 100
    this.stats.cpuPhysicsMs = Math.round(this.cpuPhysicsMs * 100) / 100
    this.stats.cpuRenderMs = Math.round(this.cpuRenderMs * 100) / 100
  }
}
