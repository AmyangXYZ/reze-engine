import { Camera } from "./camera"
import { decodeDds, isDds } from "./dds-loader"
import { Mat4, Quat, Vec3 } from "./math"
import { decodePsd, isPsd } from "./psd-loader"
import { Model, MATERIAL_MORPH_MULTIPLY, type Material, type Skeleton } from "./model"
import { MORPH_COMPUTE_WGSL } from "./shaders/passes/morph"
import { CULL_COMPUTE_WGSL } from "./shaders/passes/cull"
import { buildAnchorTable, anchorAliasWgsl, EMPTY_ANCHOR_TABLE, type AnchorTable } from "./shaders/anchor-table"
import { MIDI_HEADER, MIDI_KEYS, MIDI_NOTES, MIDI_STRIDE } from "./shaders/midi-api"
import { decodeTga } from "./tga-loader"
import { VMDLoader, type CameraKeyframe } from "./vmd-loader"
import { VMDWriter } from "./vmd-writer"
import { CameraAnimation, type CameraPose } from "./camera-animation"
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
import { ID_DEBUG_SHADER_WGSL } from "./shaders/passes/id-debug"
import { paramChanged, sampleParamTrack, type ParamKey, type ParamValue } from "./param-track"
import { effectState, type EffectWindow } from "./effect-schedule"
import { SHADOW_CASCADES, buildShadowVP } from "./shadow-cascades"
import { REFLECTION_DEBUG_WGSL, buildMirrorCamera } from "./reflection"
import { packHalf, type HdrImage } from "./hdr"
import { evalIrradianceSH, projectIrradianceSH } from "./ibl"
import { LYRIC_ATLAS_MAX_H, LYRIC_ATLAS_MAX_W, LYRICS_FLOATS, lyricsApi, packLyrics, type LyricLine, type LyricRect } from "./shaders/lyrics-api"
import {
  sceneTargets as sceneTargetsFor,
  sceneColorFormats,
  setMrtIds,
  mrtIdsEnabled,
  SCENE_ID_FORMAT,
  type SceneFormats,
} from "./shaders/passes/scene-contract"
import {
  LIGHT_HEADER,
  LIGHT_STRIDE,
  LIGHTS_FLOATS,
  MAX_LIGHTS,
  buildLightEmitShader,
  hasLightEmit,
} from "./shaders/lights"
import { groundShaderWgsl, GROUND_NOISE_BAKE_WGSL, GROUND_NOISE_SIZE } from "./shaders/passes/ground"
import { outlineShaderWgsl, RZ_OUTLINE_DISSOLVE_OFFSET } from "./shaders/passes/outline"
import { transparentDepthPrepassWgsl } from "./shaders/passes/depth-prepass"
import { SELECTION_MASK_SHADER_WGSL, SELECTION_EDGE_SHADER_WGSL } from "./shaders/passes/selection"
import { GIZMO_SHADER_WGSL } from "./shaders/passes/gizmo"
import { OVERLAY_SHADER_WGSL, OVERLAY_COMPOSITE_SHADER_WGSL } from "./shaders/passes/overlay"
import { WIREFRAME_SHADER_WGSL } from "./shaders/passes/wireframe"
import {
  boneOverlay,
  boneMarkerPositions,
  buildOverlayShapes,
  jointOverlay,
  rigidbodyOverlay,
  writeOverlayInstance,
  OVERLAY_INSTANCE_FLOATS,
  OVERLAY_VERTEX_FLOATS,
  OVERLAY_SHAPES,
  OVERLAY_SOLID_SHAPES,
  DEFAULT_VERTEX_COLOR,
  OVERLAY_STYLE,
  type BoneOverlayOptions,
  type JointOverlayOptions,
  type OverlayGeometry,
  type OverlayPrimitive,
  type OverlayShape,
  type RigidbodyOverlayOptions,
} from "./overlay"
import {
  BLOOM_BLIT_SHADER_WGSL,
  BLOOM_DOWNSAMPLE_SHADER_WGSL,
  BLOOM_UPSAMPLE_SHADER_WGSL,
} from "./shaders/passes/bloom"
import { AGX_LUT_GZ, AGX_LUT_SIZE } from "./shaders/agx-lut"
import {
  buildCompositeShader,
  EFFECT_SCENE_API,
  buildFieldShader,
  EFFECT_ANCHORS,
  EFFECT_SUBJECTS,
  EFFECT_TRAIL_BASE,
  EFFECT_TRAIL_SAMPLES,
} from "./shaders/passes/composite"
import {
  buildParticleComputeShader,
  buildParticleRenderShader,
  particleEntryPoints,
  PARTICLE_STRIDE,
} from "./shaders/passes/particles"
import {
  SIM_FORMAT,
  GRID_MAX,
  buildSimShader,
  gridEntryPoint,
} from "./shaders/passes/grid"
import {
  buildCastResolveShader,
  buildCastSeedShader,
  buildCastStepShader,
  castDistanceUsed,
  CAST_COVERAGE_FORMAT,
  CAST_DIST_FORMAT,
  CAST_FIELD_DIV,
  CAST_SEED_FORMAT,
} from "./shaders/passes/cast-distance"
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
import { parseDirectives, stripDirectives, type EffectDirectives, type EffectParamDecl } from "./shaders/directives"
import { UNLIT_GRAPH } from "./graph/presets/unlit"
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

type RaycastCallback = (
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
type WorldOptions = {
  /** Linear scene-referred color of the World Background (Blender: World > Surface > Color). */
  color?: Vec3
  /** Multiplier on world color (Blender: World > Surface > Strength). */
  strength?: number
}

/**
 * One note in a score: when it sounds, for how long, at what pitch, how hard.
 *
 * Seconds and MIDI pitch (60 = middle C) rather than ticks and a tempo map — a
 * score is consumed against the scene clock, so anything the engine stores in
 * musical time would have to be resolved to seconds before every read.
 */
export interface MidiNote {
  /** Seconds from the start of the piece. */
  start: number
  /** Seconds the note sounds for. */
  duration: number
  /** MIDI pitch, 0–127. */
  pitch: number
  /** How hard it was struck, 0–1. Defaults to 1. */
  velocity?: number
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

/** How a model rides another — MMD's 外部親 (outside parent). See setModelParent. */
export type ModelAttachment = {
  /** Model key of the parent. */
  model: string
  /** Bone on the parent. A name the parent's rig lacks rides the parent's root. */
  bone: string
}

/** The attachment as the engine keeps it: the record plus the two matrices the
 *  per-frame placement needs, allocated once per attach rather than per frame. */
type Attachment = ModelAttachment & {
  /** Where the child's origin sits in the bone's space (position · rotation). */
  offsetMatrix: Float32Array
  /** The root the child is posed under this frame. Handed to Model.setRootParent
   *  BY REFERENCE and refilled every frame; see placeAttached. */
  rootMatrix: Float32Array
}

type SunOptions = {
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

/** A vector by shape rather than by class — Vec3 satisfies it, and so does a
 *  JSON object out of a scene document or a literal typed into a console. */
export type XYZ = { x: number; y: number; z: number }
export type EffectResult = {
  ok: boolean
  /** Compile/validation errors, line:col relative to the USER's WGSL. Also
   *  carries non-fatal warnings on an effect that DID install — a directive
   *  that parsed but will never fire, an anchor the scene had no slot for. So
   *  a non-empty list is not a failure; `ok` is. */
  diagnostics: string[]
  /** Which mounts the WGSL declared — `fn background` / `fn foreground`. Both
   *  false only on a failed compile, since defining neither IS the failure. */
  mounts: { background: boolean; foreground: boolean }
  /** The knobs this effect exposes, from its own `#param` lines — name, type,
   *  default and any range. A host builds controls from THIS rather than from
   *  a second parse of the source, so what the panel offers and what the shader
   *  reads cannot come apart. Empty when the effect declares none. */
  params: EffectParamDecl[]
  /** How long ONE firing lasts, seconds, from `#duration`. 0 = the effect
   *  declared none and is AMBIENT — a condition the scene is in rather than
   *  something that happens at a moment. A host places a hit at its own length
   *  and spans an ambient one, which is the same reason `params` is here: what
   *  the host does with an effect should come from the effect, not from a
   *  second parse that can drift from it. */
  duration: number
}

type CameraOptions = {
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

const DEFAULT_ENGINE_OPTIONS = {
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
  /** Times the draw list has been rebuilt since the engine started. It should
   *  climb only when scene STRUCTURE changes — a model loaded, a style group
   *  applied — and then stop. A number that keeps rising while nothing but the
   *  animation is moving means something is dirtying the list every frame, which
   *  is the failure mode that makes render bundles cost more than they save. */
  rebuilds: number
  /** Times the render bundles have been re-recorded. Held to the same standard
   *  as `rebuilds`, and for the sharper reason: a bundle exists to be replayed,
   *  so one re-recorded every frame is strictly worse than not having it. */
  bundleRecords: number
}

interface PickDrawCall {
  count: number
  firstIndex: number
  bindGroup: GPUBindGroup
}

/**
 * A repeating dissolve, in seconds within one cycle.
 *
 * Four moments rather than a duration and a delay: every one of them is a thing
 * you can see happen, and an author tuning this is watching for exactly those
 * four frames.
 */
export interface DissolveCycle {
  period: number
  /** She starts to come apart. */
  breakAt: number
  /** Fully gone. */
  hiddenAt: number
  /** She starts to come back. */
  backAt: number
  /** Whole again. */
  doneAt: number
}

interface ModelInstance {
  name: string
  /** This model's id in the id attachment — 1-based, so 0 stays "nothing".
   *  The pick pass has always minted it; the cast carries it now too, so an
   *  effect can compare what it reads out of the id buffer against a subject. */
  objectId: number
  /** How much of this model is still THERE: 1 whole, 0 gone. Written into every
   *  material's uniform (see setModelDissolve) and mirrored into the cast, so
   *  the material shell can take her apart and an effect can draw what is
   *  leaving — both from one number rather than two clocks that must agree. */
  dissolve: number
  /** Every material's uniform buffer, in draw order. Kept because a dissolve
   *  writes ONE float into each of them and needs no other reason to hold a
   *  block: the whole 16-float copy exists only for materials that morph. */
  materialUniformBuffers: GPUBuffer[]
  /** The outline hulls' own uniforms, for the materials that have them.
   *
   *  A SEPARATE LIST because they are a separate buffer: the hull pass binds 32
   *  bytes of edge data, not the material block, so a dissolve written into the
   *  material buffers never reached it and a dissolved character kept her
   *  outline. Kept here so that write has somewhere to go. */
  outlineUniformBuffers: GPUBuffer[]
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
  shadowBindGroups: GPUBindGroup[]
  mainPerInstanceBindGroup: GPUBindGroup
  pickPerInstanceBindGroup: GPUBindGroup
  pickDrawCalls: PickDrawCall[]
  /** Environment geometry added via addStage — no physics, no IK, and it
   *  suppresses the built-in ground. See addStage for why each of those. */
  isStage: boolean
  /**
   * A media plane: a flat card carrying a picture.
   *
   * Its OWN flag rather than a shade of isStage. The two overlap in what they
   * skip — neither performs, so neither wants physics, IK, the cast buffer or
   * the camera clock — but they disagree on the thing a stage exists for: a
   * stage IS the floor and suppresses the built-in ground, while a card is
   * scenery standing in the scene and must leave the floor alone. Folding a
   * plane into isStage would have made adding a title graphic delete the ground.
   */
  isPlane: boolean
  /**
   * A PROP: a PMX object a character holds or wears — a microphone, a fan, a
   * sword. The third answer beside stage and plane. It keeps what a cast member
   * has that scenery does not (physics, outlines, its own clip) and drops what
   * makes one a performer: no effect subject id, no seeding of the scene clock,
   * no bone picking. Like a card it leaves the floor alone. See addProp.
   */
  isProp: boolean
  /** Who this model hangs from, or null. Any model can: a prop by design, a
   *  card for a sign in her hand, a second character for a mascot on her
   *  shoulder. See setModelParent. */
  parent: Attachment | null
  /** This card's texture is rewritten every frame, so it is allocated with no
   *  mip chain — rebuilding one per frame is a pass per level per card, and is
   *  what a moving card was mostly costing. See setPlaneFrame. */
  dynamicTexture: boolean
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
  /** The mesh's unique edges as a line-list index buffer, built on first use.
   *  Deduplicated: an interior edge belongs to two triangles, so drawing the
   *  triangle list's edges directly would draw most of the mesh twice.
   *
   *  Keyed by the material the edges were cut from, "" for the whole mesh. A
   *  material is a consecutive index run, so scoping is a slice of the same
   *  build — and both stay cached, because narrowing to one material and
   *  widening back out is the loop somebody auditing a model is in. */
  wireEdges: Map<string, { buffer: GPUBuffer; count: number; bindGroup: GPUBindGroup } | null>
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

/**
 * A 2D context for the alpha readback, from whichever canvas this browser has.
 *
 * OffscreenCanvas's 2D context is not universal — Safari only gained it in
 * 16.4, and a worker-less fallback has to be a DOM canvas. This used to be an
 * unguarded `new OffscreenCanvas`, so a browser without it took the catch below
 * and every material on the model was classified opaque. That is a rendering
 * difference produced by a feature probe failing, which is the kind of thing
 * that must never be silent.
 */
function alphaReadbackContext(w: number, h: number): CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D | null {
  if (typeof OffscreenCanvas !== "undefined") {
    const cx = new OffscreenCanvas(w, h).getContext("2d", { willReadFrequently: true })
    if (cx) return cx
  }
  if (typeof document === "undefined") return null
  const el = document.createElement("canvas")
  el.width = w
  el.height = h
  return el.getContext("2d", { willReadFrequently: true })
}

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
    // Raw RGBA needs no canvas at all, and must not use one. It arrives from the
    // TGA/DDS/PSD decoders as exact, straight-alpha bytes; the old path pushed it
    // through putImageData → drawImage → getImageData, which is two premultiply
    // round-trips and a resample to learn what was already in hand. Box-filtered
    // straight off the array instead: same ≤128² plane, exact values, no canvas
    // to be unavailable and no alpha to lose.
    if (rgba) {
      const a = new Uint8ClampedArray(w * h)
      for (let y = 0; y < h; y++) {
        const y0 = Math.floor((y * height) / h)
        const y1 = Math.max(y0 + 1, Math.floor(((y + 1) * height) / h))
        for (let x = 0; x < w; x++) {
          const x0 = Math.floor((x * width) / w)
          const x1 = Math.max(x0 + 1, Math.floor(((x + 1) * width) / w))
          let sum = 0
          let n = 0
          for (let sy = y0; sy < y1; sy++) {
            for (let sx = x0; sx < x1; sx++) {
              sum += rgba[(sy * width + sx) * 4 + 3]
              n++
            }
          }
          a[y * w + x] = n > 0 ? sum / n : 255
        }
      }
      return { a, w, h }
    }
    if (!source) return null
    const cx = alphaReadbackContext(w, h)
    if (!cx) return null
    cx.drawImage(source, 0, 0, w, h)
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

/** One effect's GPU particle pool. Per effect: each declares its own count. */
/**
 * The binding number an effect's params take in each pass.
 *
 * 7 in the composite, the particle stages and the ribbon pass, which all stop
 * below it. The GRID reaches 8 on its own, so it takes the next one up rather
 * than everything else moving to accommodate one mount.
 */
const EFFECT_PARAMS_BINDING = 7
const EFFECT_PARAMS_BINDING_GRID = 9

/** How a mount receives the effect's declared dials: a generator for the struct
 *  at the binding that mount has free, and the buffer behind it. Null buffer
 *  means the effect declared none, and then neither the decl nor the binding is
 *  emitted — WGSL has no empty struct, and a bound buffer nothing reads is a
 *  layout mismatch. */
type EffectParamsBinding = {
  wgsl: (binding: number) => string
  buffer: GPUBuffer | null
}

interface EffectParticles {
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
  /** Same draw, the MIRRORED camera: how particles join the floor mirror. The
   *  billboards face whichever eye is bound, so one extra bind group is the
   *  whole cost. */
  mirrorRenderBind: GPUBindGroup
  rebind: () => { computeBind: GPUBindGroup; renderBind: GPUBindGroup; mirrorRenderBind: GPUBindGroup }
}

/**
 * One effect's persistent grid, or null when it declared none.
 *
 * Two textures, not one, and read/write alternate between them every frame: a
 * shader cannot coherently read and write the same texture, so this is not an
 * optimisation but the only correct shape. `parity` says which one holds the
 * CURRENT grid — the one everything else samples. Per effect, and the only
 * effect resource that does not scale: 9 MB at 768 squared, doubled for the
 * ping-pong, which is why setEffects gives the scene a budget.
 */
interface EffectGrid {
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
  /** The effect's declared dials, or null. Held HERE as well as on the instance
   *  because the rebind path rebuilds this bind group from the grid alone. */
  params: GPUBuffer | null
}

/**
 * One effect's ribbons, or null when it declared none.
 *
 * No buffer of its own: it reads the very same path history the field-based
 * ribbon read through rzTrail, so a trail costs one draw and nothing recorded.
 */
interface EffectTrails {
  /** Ribbons this effect declared — one per trailed anchor. The instance count
   *  is derived from it per draw, against the live subject count, rather than
   *  baked here against the four-subject cap. See drawTrails. */
  slots: number
  uniform: GPUBuffer
  data: Float32Array
  pipeline: GPURenderPipeline
  layout: GPUBindGroupLayout
  bind: GPUBindGroup
  /** The mirrored camera's view of the same ribbons — the floor mirror's. */
  mirrorBind: GPUBindGroup
}

/**
 * How one field draw lands on the ones before it: OVER, into a premultiplied
 * target. Shared by both field attachments so they cannot drift apart, which
 * would show as a foreground that layers differently from its own background.
 */
const FIELD_LAYER_BLEND: GPUBlendState = {
  color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" },
  alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
}

/**
 * `#layer additive` — for LIGHT rather than matter.
 *
 * Alpha-over is right for anything with mass: smoke, fog, a backdrop. It is
 * wrong for a glow, and visibly so the moment two of them cross — the later
 * bolt occludes the earlier one in proportion to its own brightness, when what
 * light does is get brighter. Unity and Unreal both ship exactly this split,
 * and the particle path here already has it as `#blend additive`.
 *
 * Colour still scales by the author's alpha, so alpha keeps meaning "how much
 * of this is here" and an effect fades out the way it always did. What changes
 * is that the destination is never scaled down: nothing behind an emissive
 * layer is removed by it. Alpha writes are dropped for the same reason — a glow
 * does not COVER the base colour, so it must not claim coverage the composite
 * would then use to hide it.
 */
const FIELD_LAYER_BLEND_ADDITIVE: GPUBlendState = {
  color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
  // COVERAGE ACCUMULATES TOO, and dropping it was a real bug rather than a
  // simplification. The canvas is premultiplied, which REQUIRES rgb <= alpha;
  // an additive layer that contributed colour and no alpha handed the
  // compositor an invalid pixel, and an invalid pixel loses its colour.
  //
  // Nothing showed it while the background was opaque, because the composite
  // forces the canvas opaque there and never consults this at all. Put anything
  // transparent behind it — a backdrop video, an alpha export — and every
  // additive effect vanished, surviving only where the SCENE happened to supply
  // the alpha its own pixels lacked. Which read as the effect being drawn on
  // the ground grid and nowhere else.
  //
  // Additive, to match the colour beside it: light that arrives adds, and what
  // arrives is what covers. srcRgb <= 1, so the colour's src-alpha * srcRgb is
  // always <= this one's srcAlpha, and the premultiplied invariant holds by
  // construction rather than by hoping an author stayed inside it.
  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" },
}

/**
 * One installed effect, whole.
 *
 * Everything here used to be a singleton field on the Engine, which is exactly
 * what made "one effect per scene" structural rather than a choice. Grouping it
 * per effect is the change; the plural comes free once nothing reaches for
 * `this.effect` any more.
 *
 * The `epochScene` is per effect on purpose. The grid clock is measured from it,
 * and an effect installed while another is already running would otherwise
 * begin mid-animation and never see `rzGridFrame() == 0` — its only chance to
 * seed a grid.
 */
interface EffectInstance {
  wgsl: string
  /** What this instance's source declared — kept so setEffectParam can refuse a
   *  name the effect never offered instead of writing nowhere. */
  paramDecls: EffectParamDecl[]
  /** One firing's length in seconds, from `#duration`. 0 = ambient. Reported
   *  back at install so a host can place the effect at its own length. */
  duration: number
  paramLayout: Map<string, { offset: number; comps: 1 | 3 }>
  paramsBuffer: GPUBuffer | null
  paramsData: Float32Array<ArrayBuffer>
  /** Mounted under the scene. */
  hasBackground: boolean
  /** Mounted over the finished frame — and the reason the scene pass has to
   *  STORE its depth, which it otherwise discards into tile memory. */
  hasForeground: boolean
  /** Does this source actually call rzObjectAt / rzMaterialAt?
   *
   *  The exact sibling of hasForeground above, for the exact same reason. The id
   *  attachment is the pass's most expensive STORE — rg16uint at the pass's
   *  sample count, around 33MB a frame at 1080p — and it is written out for
   *  every scene whether or not a single effect ever reads it. Declaring
   *  the attachment is what keeps the pipelines agreeing; STORING it is what
   *  costs, and only a reader can justify that.
   *
   *  Parsed once at install rather than tested per frame: the answer cannot
   *  change while an effect is installed, and the frame path should not be
   *  running regexes. */
  readsIds: boolean
  /** Does this source read the distance-to-cast field? Parsed at install for the
   *  same reason readsIds is, and it turns the whole flood on by itself. */
  readsCastDistance: boolean
  /** Bones this source asked for, in ITS OWN declaration order. The scene table
   *  maps these onto shared addresses; this list is what it is rebuilt from. */
  anchors: { bone: string; trail: boolean }[]
  /** Where this effect's own clock started, in scene seconds. */
  epochScene: number
  /**
   * The level this effect reaches, 0..1 — Blender's `influence`, and its
   * meaning: a strip's blends ramp toward THIS rather than toward 1, so a
   * permanently half-strength effect and a scheduled one are the same dial.
   */
  influence: number
  /** Its strips, in scene seconds — a LANE, so one effect can fire more than
   *  once. Null or empty = on for the whole scene, which is what applying an
   *  effect does until someone places it. */
  window: readonly EffectWindow[] | null
  /**
   * What the mounts actually read this frame: `influence` shaped by the strip.
   *
   * Applied by ENGINE-GENERATED code at each mount's one output site, never by
   * the author's — an effect that had to honour its own weight would be an
   * effect that could forget to, and a scheduler cannot be built on a promise
   * every author has to keep. At 0 the mount's draw is skipped outright, which
   * is what makes a scheduled effect cost nothing outside its window.
   */
  weight: number
  /** This effect's OWN clock, as a uniform the field shader reads. Per effect
   *  because the shared one (viewU[6].x) is measured from the first installed
   *  effect's epoch, so everything later started mid-stream. Null when the
   *  effect has no field mount to read it. */
  fieldClock: GPUBuffer | null
  /** The lightEmit mount: a compute stage that writes this effect's own slots
   *  in the shared lights buffer, once per light per frame. Null unless the
   *  source declares `#lights n` AND defines fn lightEmit. */
  lights: {
    pipeline: GPUComputePipeline
    bind: GPUBindGroup
    /** Kept so `bind` can be rebuilt when a shared buffer it names is replaced. */
    layout: GPUBindGroupLayout
    uniform: GPUBuffer
    data: Float32Array<ArrayBuffer>
    /** How many slots it asked for. Its base is assigned by the engine and can
     *  move, which is why it travels in the uniform rather than the shader. */
    count: number
    /** The effect's declared parameters, or null when it declares none. Kept
     *  for the same reason `layout` is: `bind` is rebuilt on a buffer swap. */
    params: GPUBuffer | null
  } | null
  /** The field mount's pipeline and its two parity bind groups, or null when the
   *  effect declares neither background nor foreground. */
  fieldPipeline: GPURenderPipeline | null
  fieldBindGroups: [GPUBindGroup, GPUBindGroup] | null
  /** Which resolution pair this effect draws into: 0 full, 1 half. Its own
   *  declaration, not the scene's — see Engine.FIELD_SCALES. */
  fieldLayer: number
  particles: EffectParticles | null
  grid: EffectGrid | null
  trails: EffectTrails | null
}

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
  // ambient vec4 (4) + 4 lights x 2 vec4 (32) + 9 irradiance-SH vec4s (36),
  // padded to 80. sh[0].w is the IBL flag: 0 = flat world colour, 1 = the sky.
  private lightData = new Float32Array(80)
  private lightCount = 0
  private resizeObserver: ResizeObserver | null = null
  private resizePending = false
  private depthTexture!: GPUTexture
  // The one base shading model: ungrouped materials render this (compiled DEFAULT_GRAPH).
  // Grouped materials use their group's own compiled pipeline.
  private neutralPipeline!: GPURenderPipeline
  private neutralPipelineNoDepthWrite!: GPURenderPipeline
  private depthPrepassPipeline!: GPURenderPipeline
  private solidPrepassPipeline!: GPURenderPipeline
  private hairPrimePipeline!: GPURenderPipeline
  // ── Style group runtime ──
  // Shared 256 B zero StyleUniforms buffer (group(2) binding(4)) bound by every ungrouped
  // material; grouped materials rebind to their group's own buffer (per-model, in the
  // ModelInstance's styleGroups map). See docs/style-groups-spec.md §6.
  private zeroStyleBuffer!: GPUBuffer
  // Stashed at createPipelines so group pipelines can be compiled later.
  private mainPipelineLayout!: GPUPipelineLayout
  private sceneTargets!: GPUColorTargetState[]
  /** The scene pass's attachment formats, settled at init once the device has
   *  said which HDR format it will blend. Every scene-pass pipeline asks
   *  scene-contract for its targets against these. */
  private get sceneFormats(): SceneFormats {
    return { hdr: this.hdrFormat, aux: Engine.BLOOM_MASK_FORMAT }
  }
  private fullVertexBufferLayouts!: GPUVertexBufferLayout[]
  // 1×64 vertical ramp for shared-toon materials: lit (top) → soft shadow
  // tone (bottom). Stand-in for MMD's toon01–10.bmp, which we can't ship.
  private defaultToonRampTexture!: GPUTexture
  private groundShadowPipeline!: GPURenderPipeline
  /** The soft-edge variant, built the first time a scene asks for one. Null while
   *  no scene has, which is most of them — a pipeline nobody draws with is still
   *  a shader compile at load. */
  private groundShadowSoftPipeline: GPURenderPipeline | null = null
  /** How the ground's own pipeline is chosen, kept beside the uniform that sets
   *  it so the draw does not have to read the buffer back. */
  private groundSoft = false
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

  // ─── Editor overlays (bones, rigidbodies, joints) ──────────────────
  // One instanced pass over unit wireframes. Static layers are whatever the
  // host handed setOverlay; the three live ones are rebuilt from the model
  // every frame, because a posed skeleton has moved by the next one.
  private overlayVertexBuffer!: GPUBuffer
  private overlayInstanceBuffer: GPUBuffer | null = null
  private overlayInstanceCapacity = 0
  private overlayPipeline!: GPURenderPipeline
  private overlaySolidPipeline!: GPURenderPipeline
  private overlayBindGroup!: GPUBindGroup
  private overlayGeometry!: OverlayGeometry
  private overlayPassDescriptor!: GPURenderPassDescriptor
  private overlayDepthTexture: GPUTexture | null = null
  private overlayMsaaTexture: GPUTexture | null = null
  private overlayResolveTexture: GPUTexture | null = null
  private overlayUniformBuffer!: GPUBuffer
  private overlayUniformData = new Float32Array(4)
  private overlayCompositePipeline!: GPURenderPipeline
  private overlayCompositeLayout!: GPUBindGroupLayout
  private overlayCompositeBindGroup: GPUBindGroup | null = null
  private overlayCompositePassDescriptor!: GPURenderPassDescriptor
  private overlayTargetSize: [number, number] = [0, 0]
  /** The overlay renders multisampled into its own layer; the scene's own depth
   *  is discarded before the composite (see the depthRead note in render), so it
   *  could not have shared either that or the single-sample swapchain. */
  private static readonly OVERLAY_SAMPLE_COUNT = 4
  /** Dash period in device pixels — dashes are geometry, so this is only the
   *  reference the dashedLine shape is cut against. */
  private static readonly OVERLAY_DASH_PERIOD_PX = 8.0
  private overlayLayers = new Map<string, OverlayPrimitive[]>()
  private overlayBones: { modelName: string; options: BoneOverlayOptions } | null = null
  private overlayBodies: { modelName: string; options: RigidbodyOverlayOptions } | null = null
  private overlayJoints: { modelName: string; options: JointOverlayOptions } | null = null
  private overlayVertices: { modelName: string; xray: boolean; material: string | null } | null = null
  private wireframePipeline!: GPURenderPipeline
  private wireframeDepthPipeline!: GPURenderPipeline
  private wireframeUniformBuffer!: GPUBuffer
  private wireframeBindGroup!: GPUBindGroup
  /** The seam pass draws in the same frame at a different stroke and alpha, and
   *  every queue write lands before the command buffer runs — writing the one
   *  buffer twice would give both draws the second value. */
  private wireframeSeamUniformBuffer!: GPUBuffer
  private wireframeSeamBindGroup!: GPUBindGroup
  /** Same reason: the hovered material draws its own stroke, self-occluding, in
   *  the same frame as the base mesh and the seams. */
  private wireframeHoverUniformBuffer!: GPUBuffer
  private wireframeHoverBindGroup!: GPUBindGroup
  private wireframeSkinLayout!: GPUBindGroupLayout
  private wireframeColorData = new Float32Array(8)
  /** Rebuilt every frame into these, grouped by shape so each shape is one draw. */
  private overlayByShape = new Map<OverlayShape, OverlayPrimitive[]>()
  private overlayScratch: OverlayPrimitive[] = []
  private bonePickScratch: Float32Array = new Float32Array(0)
  private overlayInstanceData = new Float32Array(0)

  // ─── Transform gizmo ───────────────────────────────────────────────
  private selectedBone: { modelName: string; boneName: string; boneIndex: number } | null = null
  /** The material a pointer is currently over, or null. Cheap and separate from
   *  setVertexOverlay on purpose — the same split setSelectedBone takes from
   *  setBoneOverlay — because this is written every frame the pointer moves and
   *  the overlay's own option object is not something to reconstruct that often. */
  private hoverMaterial: { modelName: string; materialName: string } | null = null
  /** The transform gizmo follows setSelectedBone, which is also what selects a
   *  bone to INSPECT. A model editor selects bones constantly and poses them
   *  rarely, so the two need separating: off leaves selection working and takes
   *  the handles away. */
  private gizmoEnabled = true
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
  /**
   * Shadow map depth format — 16-bit, deliberately.
   *
   * The maps are ORTHOGRAPHIC, so depth is linear across the box: 65,536 steps
   * over the near cascade's 140-unit range is 0.002 units per step, and every
   * bias in play dwarfs it — the samplers subtract 0.0035 ndc (~229 of these
   * steps) and the materials offset along the normal by 0.08 units (~37 steps)
   * before the compare ever runs. Quantisation cannot flip an answer the biases
   * have already moved that far, so the pixels are identical to depth32float's.
   *
   * What is NOT identical is the bandwidth, which is the term WebKit pays
   * hardest: every PCF tap is a hardware-bilinear compare reading four texels,
   * so nine taps read half the bytes at 2 B/texel — 72 B/pixel instead of 144
   * across every shadowed surface on screen — and the 4096² map's clear+store
   * each frame drops from 64 MB to 32.
   */
  private static readonly SHADOW_DEPTH_FORMAT: GPUTextureFormat = "depth16unorm"
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
  /**
   * Force the HDR format instead of taking the device's answer. Null = probe,
   * which is what ships.
   *
   * A diagnostic, and deliberately a coarse one. The choice above is the ONE
   * render-target difference between a Safari device and a desktop Chrome that
   * lacks the feature, which makes it the first thing to eliminate when
   * something renders correctly on one and not the other — and specifically when
   * the something involves alpha, because rg11b10ufloat is the path with no
   * alpha channel to carry it. Setting this to "rgba16float" on the device puts
   * Safari back on the desktop's path at the cost of the tile-memory win, so a
   * symptom that survives is not about the format and a symptom that vanishes
   * is.
   *
   * Static, like MRT_IDS: read once in init(), before any texture or pipeline
   * exists, so there is no such thing as changing it on a live engine.
   */
  static HDR_FORMAT_OVERRIDE: GPUTextureFormat | null = null
  /** Main-pass depth. Float when the adapter offers depth32float-stencil8, which
   *  is also what makes reversed-Z worth switching on. */
  private depthFormat: GPUTextureFormat = "depth24plus-stencil8"
  /** Near maps to 1 and far to 0. Set once at init and never toggled — every
   *  pipeline's compare function and both depth clears are chosen from it. */
  private reversedZ = false
  /** The compare a "draw what is in front" pipeline wants, either way round. */
  private get depthAhead(): GPUCompareFunction {
    return this.reversedZ ? "greater-equal" : "less-equal"
  }
  /** The value a cleared depth buffer holds: the FAR plane, whichever end that is. */
  private get depthClear(): number {
    return this.reversedZ ? 0 : 1
  }
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
  /**
   * The master switch for the id attachment. OFF — and off having been proven
   * to work, not off because it was never finished.
   *
   * ON, because there is finally something that reads it: rzObjectAt and
   * rzMaterialAt in the field module let an effect mask itself to one character
   * or one material. It was switched off in the meantime rather than left
   * running — rg16uint at the pass's sample count is around 33MB at 1080p,
   * cleared and stored every frame, and paying that for a buffer nobody read
   * would have handed back the same order of bandwidth the empty field-pass
   * clears had just saved.
   *
   * Verified through setIdDebug against a real scene: flat colour per material
   * with hard edges (so nothing interpolates or resolves them), the floor on
   * its reserved id, black exactly where nothing drew. Turning it back off is
   * this line, and the accessors then answer 0 rather than failing to compile.
   */
  private static readonly MRT_IDS = true
  /**
   * What fraction of its authored damping a chest rig's body keeps.
   *
   * The whole tuning surface for how long those rigs swing: lower rings
   * longer, 1 restores the authored value exactly. It does NOT change where
   * they hang at rest — that is the property that made damping the right knob
   * (see RezePhysics.setJiggleDamping). Judge it against the models that
   * motivated it; it is a starting point, not a measurement.
   */
  private static readonly JIGGLE_DAMPING_SCALE = 0.5
  /** The id attachment. Multisampled with the pass and NEVER resolved: an
   *  averaged id belongs to nothing, so consumers textureLoad sample 0. */
  private idTexture: GPUTexture | null = null
  private idView: GPUTextureView | null = null
  /** The id buffer drawn to the screen — see setIdDebug. */
  private idDebugPipeline: GPURenderPipeline | null = null
  private idDebugBindGroupLayout: GPUBindGroupLayout | null = null
  private idDebugBindGroup: GPUBindGroup | null = null
  private idDebug = false
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
  /** Ceiling for `#particles`. Past this an author is asking for a stall. */
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
  private simSampler!: GPUSampler
  private simFallbackView!: GPUTextureView
  /** 1×1 transparent stand-in, for every layer binding with nothing behind it. */
  private trailFallbackView!: GPUTextureView
  /**
   * The field layer: user background/foreground mounts, ONE TARGET PAIR PER
   * RESOLUTION. Index 0 is full, index 1 is half — coarsest last, so the
   * composite reads them full-over-half.
   *
   * `#fullres` used to be a property of the shared targets: one effect
   * declaring it promoted the pass for every effect installed, so a starfield
   * that upsamples perfectly paid four times the pixels because a keyboard
   * beside it needed crisp edges. Measured, that was the largest avoidable cost
   * in the frame — Footprints went from about 1.2ms to 4.5ms purely by being
   * dragged along.
   *
   * The price is that a resolution boundary is now a LAYER boundary. Within a
   * pair, effects blend in document order; across pairs the full-res layer
   * composites over the half-res one whatever the document said. Invisible for
   * the additive glows this is nearly always used for, and stated because it is
   * the one thing document order stopped deciding.
   */
  private static readonly FIELD_SCALES = [1, 2] as const
  /** Reused for the per-frame field-clock upload — one 16-byte write per
   *  drawing effect, and allocating a fresh array for each would be garbage
   *  every frame. */
  private fieldClockScratch = new Float32Array(4)
  /** Material parameters driven by the scene clock — see setStyleParamTrack. */
  /** Repeating dissolves, by model name — see setModelDissolveCycle. */
  private dissolveCycles = new Map<string, DissolveCycle>()
  private paramTracks = new Map<
    string,
    { modelName: string; groupId: string; paramId: string; keys: ParamKey[]; last: ParamValue | null }
  >()
  /**
   * The distance-to-cast field: seeds ping-ponged by a jump flood, then resolved.
   *
   * All null until some installed effect names rzCastDistance. Nothing is
   * allocated and no pass is encoded for a scene that never asks — the same
   * bargain the grid and the id buffer strike.
   */
  private castSeedTextures: (GPUTexture | null)[] = [null, null]
  private castSeedViews: (GPUTextureView | null)[] = [null, null]
  private castCoverageTexture: GPUTexture | null = null
  private castCoverageView: GPUTextureView | null = null
  private castDistTexture: GPUTexture | null = null
  private castDistView: GPUTextureView | null = null
  /** 1x1 holding half-float 65504, bound whenever the pass is not running: an
   *  effect keyed on distance then finds the cast unreachably far and draws
   *  nothing, rather than the accessor being a name that does not exist. */
  private castDistFallback: GPUTexture | null = null
  private castDistFallbackView: GPUTextureView | null = null
  private castSeedPipeline: GPURenderPipeline | null = null
  private castStepPipeline: GPURenderPipeline | null = null
  private castResolvePipeline: GPURenderPipeline | null = null
  private castSeedBindGroup: GPUBindGroup | null = null
  private castStepBindGroups: GPUBindGroup[] = []
  private castResolveBindGroup: GPUBindGroup | null = null
  private castStepStrideBuffers: GPUBuffer[] = []
  /** Does anything installed actually read the field? Set from the effect list. */
  private castDistanceWanted = false

  private fieldBgTextures: (GPUTexture | null)[] = [null, null]
  private fieldBgViews: (GPUTextureView | null)[] = [null, null]
  private fieldFgTextures: (GPUTexture | null)[] = [null, null]
  private fieldFgViews: (GPUTextureView | null)[] = [null, null]
  /** One per scale: the field shader reconstructs the full-res pixel it stands
   *  in for, so each pass needs its own (w, h, fullW, fullH). */
  private fieldUniformBuffers: GPUBuffer[] = []
  private fieldFullW = 0
  private fieldFullH = 0
  private fieldBindGroupLayout!: GPUBindGroupLayout
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
  // ── Score (note events) ──
  private midiBuffer!: GPUBuffer
  private midiFallbackBuffer!: GPUBuffer
  /** Fixed-size (LYRICS_FLOATS): setLyrics is a write, never a reallocation,
   *  so lyric data arriving after any effect reaches it with no re-binding. */
  private lyricsBuffer!: GPUBuffer
  /** The rasterised lines, for rzLyricText. Sized to the track that arrives —
   *  a 1×1 placeholder until one does, so a scene with no lyrics pays nothing
   *  and a song's text is stored at the resolution it is drawn at. */
  private lyricsTexture!: GPUTexture
  private lyricsTextureView!: GPUTextureView
  /** The notes as installed, kept CPU-side because the per-pitch key map is
   *  rebuilt from them every time the clock moves. */
  private midiNotes: MidiNote[] = []
  /** Header + key map, re-uploaded per clock write. */
  private midiLiveScratch = new Float32Array(MIDI_KEYS + 2)
  private midiRelease = 0.35
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
  private cullHiddenBuffer: GPUBuffer | null = null
  private cullHidden = new Uint32Array(0)
  private cullArgs = new Uint32Array(0)
  /** Draws and models the current buffers can hold. A rebuild that fits inside
   *  these rewrites contents and allocates nothing. */
  private cullCapacity = 0
  private cullModelCapacity = 0
  /** How many times the draw list has been rebuilt — a bundle-era regression
   *  guard, reported by getCullDiagnostics. Steady-state it must not climb. */
  private cullRebuilds = 0
  // ── Render bundles ──
  private opaqueBundle: GPURenderBundle | null = null
  private shadowBundles: GPURenderBundle[] = []
  /** Set by scene STRUCTURE only. Every frame of animation, every physics step
   *  and every camera move must leave this alone — re-recording constantly is
   *  worse than having no bundles at all. */
  private bundlesDirty = true
  /** Companion to cullRebuilds: how many times the bundles have been recorded.
   *  Steady-state it must not climb. */
  private bundleRecords = 0
  // ── GPU pass timings ──
  // The passes worth a number, in the order their queries are laid out. Kept
  // short on purpose: the point is to notice a restructure making a pass more
  // expensive, and a list long enough to need reading is one nobody reads.
  /** Every pass worth watching across a refactor. The query set, the resolve
   *  buffer and the readback are all sized from this, so adding one here is the
   *  whole change.
   *
   *  `field` earns its place now that a scene runs SEVERAL field effects at
   *  once: it is one pass with N draws, its resolution is a property of the
   *  shared targets rather than of any one effect — so a single `#fullres`
   *  effect quadruples the pixel count for all of them — and it is the pass the
   *  field restructure moves. Restructuring it while it was the only untimed
   *  pass in the frame would have meant reasoning about the cost instead of
   *  reading it. */
  /**
   * The passes worth a number, in the order the frame runs them.
   *
   * These ARE the boxes on the architecture figure, deliberately: a reading that
   * cannot be pointed at a component is a reading nobody acts on. Three were
   * missing and each is a real per-frame cost a report of "it feels slower"
   * could have been about — the morph compute, the mirror's second pass over the
   * whole cast, and the bloom pyramid, which is NINE render passes and was the
   * largest unmeasured thing in the frame.
   *
   * The per-effect computes (particles, grids, lights) are deliberately absent:
   * they are a loop of one pass per effect, so there is no single span to stamp
   * and a number attributed to the wrong one is worse than no number. They fall
   * into the "rest" the readout derives from the frame time.
   *
   * Adding one costs two query slots and nothing else; the query set is sized
   * from this array's length.
   */
  private static readonly TIMED_PASSES = [
    "cull",
    "morph",
    "shadow",
    "mirror",
    "scene",
    "field",
    "bloom",
    "composite",
    "overlay",
  ] as const
  private timestampQuerySet: GPUQuerySet | null = null
  private timestampResolve: GPUBuffer | null = null
  private timestampRead: GPUBuffer | null = null
  /** A map is in flight; the readback buffer cannot be written while it is. */
  private timestampBusy = false
  private gpuPassMs: Record<string, number> | null = null
  private cullCameraArgs: GPUBuffer | null = null
  private cullShadowArgs: GPUBuffer | null = null
  private cullMirrorArgs: GPUBuffer | null = null
  // ── The floor mirror (step 7C) ──
  // Half-res scene-contract attachments a mirrored draw renders into, plus the
  // mirror's own camera block. The plane is the ground plane: MMD floors live
  // at y = 0 by convention and addGround builds its quad there.
  private static readonly REFLECTION_PLANE_Y = 0
  private mirrorCameraData = new Float32Array(40)
  private mirrorCameraBuffer!: GPUBuffer
  // proj x mirrorView for the ground's projective sample and the cull planes,
  // then (projA, projB, 0, 0) — the depth-linearisation pair, read off the
  // SHARED projection the way dofU does, for the depth-proportional blur.
  private mirrorVPData = new Float32Array(20)
  private mirrorVPBuffer!: GPUBuffer
  private mirrorPerFrameBindGroup!: GPUBindGroup
  private mirrorColorMsTexture: GPUTexture | null = null
  private mirrorColorTexture: GPUTexture | null = null
  private mirrorColorView: GPUTextureView | null = null
  private mirrorMipCount = 1
  private mirrorMipViews: GPUTextureView[] = []
  private mirrorBlurBindGroups: GPUBindGroup[] | null = null
  private groundMirrorBlur = 0
  private mirrorMaskMsTexture: GPUTexture | null = null
  private mirrorIdMsTexture: GPUTexture | null = null
  private mirrorDepthTexture: GPUTexture | null = null
  private mirrorDepthReadView: GPUTextureView | null = null
  private mirrorPassDescriptor: GPURenderPassDescriptor | null = null
  private mirrorOpaqueBundle: GPURenderBundle | null = null
  private mirrorTransparentBundle: GPURenderBundle | null = null
  /** setGroundMirror lands in step 7D; the debug dial is what exercises C. */
  private groundMirror = 0
  private reflectionDebug = false
  private reflectionDebugPipeline: GPURenderPipeline | null = null
  private reflectionDebugBindGroupLayout: GPUBindGroupLayout | null = null
  private reflectionDebugBindGroup: GPUBindGroup | null = null
  private get reflectionActive(): boolean {
    return this.reflectionDebug || this.groundMirror > 0
  }
  private cullFrustaBuffer: GPUBuffer | null = null
  // 18 planes (camera, shadow, mirror) x 16 bytes, then the counts vec4u.
  private cullFrustaBytes = new ArrayBuffer(304)
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
  private cullEnabled = true
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
  /**
   * The HDRI WORLD — what lights the scene, and what you see when nothing else
   * is behind it.
   *
   * Separate from the backdrop because they answer different questions. An
   * HDRI is a measurement of light: it drives the ambient term through
   * `worldSH` whether or not it is the thing on screen. A 360 picture is
   * wallpaper: it is what you see and it lights nothing. They shared one slot
   * and so were mutually exclusive, which made "light her with a studio HDRI
   * and put a different sky behind her" impossible to say — the ordinary split
   * every renderer draws between a world and a film backdrop.
   */
  private worldEquirectTexture: GPUTexture | null = null
  private worldEquirectView: GPUTextureView | null = null
  private worldStrength = 1
  /** The installed HDRI's folded irradiance SH (27 floats), or null. */
  private worldSH: Float32Array | null = null
  private fallbackEquirectTexture!: GPUTexture
  private fallbackEquirectView!: GPUTextureView
  // The scene's user WGSL effect (setEffect). ONE per scene, mounted under the
  // scene, over it, or both — whichever of background()/foreground() the code
  // defines. The composite pipelines are REBUILT with the user code injected;
  // params live in their own uniform buffer so setEffectParam is a write, not a
  // recompile (the same instant tier as setStyleParam).
  /**
   * The scene's effects, in document order — the order they layer in.
   *
   * An array from here down even while setEffect installs exactly one, because
   * the plural is the whole point of this step and a singleton that has to be
   * "generalised later" is a singleton that shapes every call site against it.
   */
  /** Subjects the cast actually holds, set while it is filled. The ribbons size
   *  their instance count by this rather than by the four-subject cap. */
  private castSubjectCount = 0
  private effects: EffectInstance[] = []
  /** The first installed effect, for the many places that legitimately want
   *  "is anything installed" or the singleton API's one effect. */
  private get effect(): EffectInstance | null {
    return this.effects[0] ?? null
  }
  /** The cast, as the effect API sees it. Written per frame while an effect is
   *  installed, and only up to what that effect actually declared. */
  private castBuffer!: GPUBuffer
  /** The positional lights, as data — see shaders/lights.ts for the layout.
   *  Allocated once at full size and zero-filled, so "no lights" is a count of
   *  zero rather than an absent binding. */
  private lightsBuffer!: GPUBuffer
  private lightsData!: Float32Array<ArrayBuffer>
  /** Just the header, for rewriting the total without touching a record. */
  private lightHeader = new Float32Array(LIGHT_HEADER)
  /** How many of the slots belong to the DOCUMENT. Effects get what follows. */
  private docLightCount = 0
  private castData!: Float32Array<ArrayBuffer>
  /** Last frame's anchor world positions, for velocity. Keyed model id → slot. */
  private anchorPrev = new Map<string, Float32Array>()
  /**
   * The SCENE's bones: which ones are recorded into the cast buffer, and which
   * address each one holds. Built at install from every effect's requests, so
   * the buffer is written once per bone however many effects read it.
   *
   * Everything that touches the cast buffer iterates THIS rather than any one
   * effect's declarations — that is the whole change, and it is what makes N
   * effects a loop instead of a rewrite. With one effect installed the table is
   * exactly that effect's anchors in declaration order, which is why this lands
   * with no visible difference.
   */
  private anchorTable: AnchorTable = EMPTY_ANCHOR_TABLE
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
  /** The user's own "no ground" switch — see setGroundVisible. Distinct from
   *  hasGround, which records whether a ground was ever built. */
  private groundHidden = false
  private shadowMapTextures: GPUTexture[] = []
  private shadowMapDepthViews: GPUTextureView[] = []
  private brdfLutTexture!: GPUTexture
  private brdfLutView!: GPUTextureView
  private filmicLutTexture!: GPUTexture
  private filmicLutView!: GPUTextureView
  // Width of the baked Filmic tone LUT (composite.ts FILMIC_LUT_W must match).
  private static readonly FILMIC_LUT_WIDTH = 256
  private shadowDepthPipeline!: GPURenderPipeline
  private shadowLightVPBuffer!: GPUBuffer
  // The shadow PASS reads one cascade's matrix per pass, and a uniform binding
  // into the aggregate would need 256-byte alignment padding — two tiny buffers
  // are simpler than teaching every reader about a stride.
  private shadowCascadeVPBuffers: GPUBuffer[] = []
  // All cascades' view-projections, 16 floats each, inner to outer.
  private shadowLightVPMatrix = new Float32Array(16 * SHADOW_CASCADES.length)
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
  private physicsFloor = true
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

  /** Sensor grain: how much, and whether it moves. */
  private grain = { amount: 0, animated: true }

  /**
   * Film grain over the rendered scene, 0–1.
   *
   * A property of a SENSOR, so it belongs to the camera rather than to any one
   * subject, and it lands on what the engine drew and on nothing else — never on
   * a background image or a backdrop video, which arrived with grain of their
   * own and would be graded rather than matched by a second helping.
   *
   * `animated` false freezes it. A still photograph's grain does not move, and
   * noise crawling over a frozen picture makes the rendering look more alive
   * than the thing it is standing in.
   *
   * Costs one hash per pixel in a pass that already runs, and nothing at all at
   * zero — the branch is on a uniform.
   */
  setFilmGrain(amount: number, animated = true): void {
    this.grain.amount = Math.min(Math.max(amount, 0), 1)
    this.grain.animated = animated
    if (this.device && this.compositeUniformBuffer) this.writeCompositeViewUniforms()
  }
  getFilmGrain(): Readonly<{ amount: number; animated: boolean }> {
    return this.grain
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

  /**
   * Whether bloom will actually reach the frame this frame.
   *
   * The composite multiplies the pyramid by this same effective intensity, so a
   * zero here means every pass that BUILDS the pyramid is work whose result is
   * multiplied by nothing. That was the state of it: `enabled` reached exactly
   * one line — the intensity uniform below — and the nine render passes that
   * fill the pyramid ran regardless, on every frame, of every scene, whether or
   * not anyone had asked for bloom.
   *
   * Nine passes is the number that matters rather than the pixels: on a
   * tile-based GPU a render pass is a tile load and store whatever it draws, so
   * this is paid in full on Apple hardware and largely hidden on a desktop
   * immediate-mode one. It is the same asymmetry as the bundle bug — cheap where
   * it was written, expensive where it was reported.
   */
  private bloomContributes(): boolean {
    const b = this.bloomSettings
    return b.enabled && b.intensity > 0
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
    u[2] = this.grain.amount
    // The seed. Zero means STILL: a plate that is one photograph has grain that
    // does not move, and CG noise crawling over a frozen picture makes the CG
    // look more alive than the footage — the opposite of the point.
    u[3] = this.grain.animated ? Math.floor(this.sceneClock * 24) % 1024 : 0
    u[4] = b.color.x
    u[5] = b.color.y
    u[6] = b.color.z
    u[7] = effIntensity
    // Background composited UNDER the scene in display space (post-tonemap), so it
    // matches a CSS color of the same value exactly. Mode (u[11]): 0 = transparent
    // (DOM shows), 1 = solid color, 2 = LDR 360 equirect (display-space
    // wallpaper), 3 = HDR equirect (scene-linear radiance through the SAME
    // exposure and view transform as the scene — a sun rolls off like a sun).
    // The camera basis at u[12..23] is refreshed per frame.
    // In modes 2 and 3 the colour slot is dead, so mode 3 carries the world
    // STRENGTH in u[8] — Blender's world-strength dial, default 1.
    const bg = this.backgroundColor
    // THE BACKDROP WINS WHAT YOU SEE; the world lights regardless. With only a
    // world installed it is also the sky, which is what an HDRI alone has
    // always done.
    const showingWorld = this.backdropEquirectView === null && this.worldEquirectView !== null
    u[8] = showingWorld ? this.worldStrength : (bg?.x ?? 0)
    u[9] = bg?.y ?? 0
    u[10] = bg?.z ?? 0
    // Base-layer mode only. A user effect is a separate LAYER over whichever
    // base is active, and needs no flag of its own: the composite pipeline is
    // rebuilt per effect, so the compiled variant IS the flag.
    u[11] = this.backdropEquirectView ? 2 : showingWorld ? 3 : bg ? 1 : 0
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
    if (this.outlineEnabled === on) return
    this.outlineEnabled = on
    // Whether a hull is drawn is decided at record time, so this is one of the
    // few switches that genuinely has to re-record. It is a user toggle, not a
    // per-frame state, which is what makes that affordable.
    this.bundlesDirty = true
  }

  /**
   * Draw the id attachment instead of the scene.
   *
   * The only way to SEE whether ids are right: with no consumer, a correct id
   * buffer and a wrong one render the same frame. See id-debug.ts for what
   * correct looks like and what each failure looks like instead.
   *
   * Returns false when there is nothing to show — ids compiled out, or a device
   * that cannot multisample the format — rather than turning on and drawing
   * black, which would read as "the ids are all zero".
   */
  setIdDebug(on: boolean): boolean {
    if (on && !this.idView) return false
    this.idDebug = on
    return true
  }

  /** True when the id attachment exists on this device. */
  hasObjectIds(): boolean {
    return this.idView !== null
  }

  /** The pass that draws it, built lazily so a scene that never asks for the
   *  debug view never compiles it. */
  private ensureIdDebugPipeline(): boolean {
    if (!this.idView) return false
    if (!this.idDebugBindGroupLayout) {
      this.idDebugBindGroupLayout = this.device.createBindGroupLayout({
        label: "id debug bind group layout",
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            texture: { sampleType: "uint", viewDimension: "2d", multisampled: true },
          },
        ],
      })
    }
    if (!this.idDebugPipeline) {
      const module = this.device.createShaderModule({ label: "id debug", code: ID_DEBUG_SHADER_WGSL })
      this.idDebugPipeline = this.device.createRenderPipeline({
        label: "id debug pipeline",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.idDebugBindGroupLayout] }),
        vertex: { module, entryPoint: "vs" },
        // The swapchain, unmultisampled — this pass replaces the finished frame
        // rather than joining the scene pass.
        fragment: { module, entryPoint: "fs", targets: [{ format: this.presentationFormat }] },
        primitive: { topology: "triangle-list" },
      })
    }
    if (!this.idDebugBindGroup) {
      this.idDebugBindGroup = this.device.createBindGroup({
        label: "id debug bind group",
        layout: this.idDebugBindGroupLayout,
        entries: [{ binding: 0, resource: this.idView }],
      })
    }
    return true
  }

  private renderIdDebugPass(encoder: GPUCommandEncoder, swapchainView: GPUTextureView): void {
    if (!this.idDebug || !this.ensureIdDebugPipeline()) return
    const pass = encoder.beginRenderPass({
      label: "id debug",
      colorAttachments: [
        { view: swapchainView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    })
    pass.setPipeline(this.idDebugPipeline!)
    pass.setBindGroup(0, this.idDebugBindGroup!)
    pass.draw(3)
    pass.end()
  }

  /**
   * Show the floor mirror's target instead of the finished frame — the
   * instrument that makes the reflection pass checkable before the ground
   * consumes it. Dev surface, like setIdDebug beside it.
   */
  setReflectionDebug(on: boolean): void {
    this.reflectionDebug = on
  }

  /**
   * What is lighting the world seat right now — the dev-console answer to
   * "did the HDRI actually arrive". Flat mode reports the same colour for
   * every direction, which is what flat means.
   */
  getWorldLighting(): {
    source: "hdri" | "flat"
    strength: number
    up: [number, number, number]
    down: [number, number, number]
  } {
    const s = this.world.strength
    if (this.worldSH) {
      const at = (n: { x: number; y: number; z: number }) =>
        evalIrradianceSH(this.worldSH!, n).map((v) => Math.max(v * s, 0)) as [number, number, number]
      return { source: "hdri", strength: s, up: at({ x: 0, y: 1, z: 0 }), down: at({ x: 0, y: -1, z: 0 }) }
    }
    const c = this.world.color
    const flat: [number, number, number] = [c.x * s, c.y * s, c.z * s]
    return { source: "flat", strength: s, up: flat, down: flat }
  }

  /**
   * Switch the floor mirror without rebuilding the ground — the adjust-tier
   * sibling of addGround's own options. False when there is no ground.
   *
   * ON OR OFF, deliberately not a strength: the reflection is an independent
   * LAYER beneath the floor surface, and how much of it shows is the ground's
   * own opacity covering it. Blur 0 is a polished mirror; 1 samples the
   * softest level, scaled by how far the reflected geometry sits behind the
   * surface.
   */
  setGroundMirror(on: boolean, blur?: number): boolean {
    if (!this.groundShadowMaterialBuffer) return false
    this.groundMirror = on ? 1 : 0
    if (blur !== undefined) this.groundMirrorBlur = Math.min(Math.max(blur, 0), 1)
    this.device.queue.writeBuffer(
      this.groundShadowMaterialBuffer,
      15 * 4,
      new Float32Array([this.groundMirror, this.groundMirrorBlur]),
    )
    return true
  }

  private ensureReflectionDebugPipeline(): boolean {
    if (!this.mirrorColorView) return false
    if (!this.reflectionDebugBindGroupLayout) {
      this.reflectionDebugBindGroupLayout = this.device.createBindGroupLayout({
        label: "reflection debug bind group layout",
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        ],
      })
    }
    if (!this.reflectionDebugPipeline) {
      const module = this.device.createShaderModule({ label: "reflection debug", code: REFLECTION_DEBUG_WGSL })
      this.reflectionDebugPipeline = this.device.createRenderPipeline({
        label: "reflection debug pipeline",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.reflectionDebugBindGroupLayout] }),
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets: [{ format: this.presentationFormat }] },
        primitive: { topology: "triangle-list" },
      })
    }
    if (!this.reflectionDebugBindGroup) {
      this.reflectionDebugBindGroup = this.device.createBindGroup({
        label: "reflection debug bind group",
        layout: this.reflectionDebugBindGroupLayout,
        entries: [
          { binding: 0, resource: this.mirrorColorView },
          { binding: 1, resource: this.materialSampler },
        ],
      })
    }
    return true
  }

  private renderReflectionDebugPass(encoder: GPUCommandEncoder, swapchainView: GPUTextureView): void {
    if (!this.reflectionDebug || !this.ensureReflectionDebugPipeline()) return
    const pass = encoder.beginRenderPass({
      label: "reflection debug",
      colorAttachments: [
        { view: swapchainView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" },
      ],
    })
    pass.setPipeline(this.reflectionDebugPipeline!)
    pass.setBindGroup(0, this.reflectionDebugBindGroup!)
    pass.draw(3)
    pass.end()
  }

  /** Refold the live camera with the reflection — a copy and a handful of
   *  sign flips; cheap enough to run every frame a mirror is on. */
  private updateMirrorCamera(): void {
    buildMirrorCamera(this.cameraMatrixData, Engine.REFLECTION_PLANE_Y, this.mirrorCameraData)
    this.device.queue.writeBuffer(this.mirrorCameraBuffer, 0, this.mirrorCameraData)
    Mat4.multiplyArrays(this.cameraMatrixData, 16, this.mirrorCameraData, 0, this.mirrorVPData, 0)
    // projA/projB ARE m[10] and m[14] of the projection (the dofU discipline:
    // read them off the matrix, never re-derive from near/far). The mirror
    // shares the main projection, so the pair linearises its depth too.
    this.mirrorVPData[16] = this.cameraMatrixData[16 + 10]
    this.mirrorVPData[17] = this.cameraMatrixData[16 + 14]
    this.device.queue.writeBuffer(this.mirrorVPBuffer, 0, this.mirrorVPData)
  }

  /**
   * The scene, mirrored about the floor, into the half-res reflection target.
   *
   * Models only: no ground (the mirror IS the ground), no particles, trails or
   * field effects — the classic MMD stage-floor reflection is the cast, and
   * each of those layers would need its own mirrored variant to join. Runs
   * between emitLights (materials read the lights buffer) and the scene pass
   * (whose ground will sample the resolve).
   *
   * KNOWN LIMIT, deliberate: geometry BELOW the floor plane would reflect up
   * into the target — there is no oblique clip. MMD stages rarely have any;
   * the clip is the follow-up if one shows.
   */
  private renderMirrorPass(encoder: GPUCommandEncoder): void {
    if (!this.reflectionActive || !this.mirrorPassDescriptor) return
    if (!this.mirrorOpaqueBundle && !this.mirrorTransparentBundle) return
    // A mirror reflects the sky, not the void: clear to the scene's background
    // so the empty regions of the reflection read as backdrop instead of
    // black. Linearised, because the mirror lives in scene-linear HDR and the
    // stored colour is display sRGB — and honestly APPROXIMATE: the real
    // backdrop composites after the view transform, so the mirrored patch
    // rides through AgX/filmic and lands close, not identical. A transparent
    // background keeps the black clear; a 360 equirect gets the flat colour —
    // sampling the skybox along reflected rays is the recorded follow-up.
    const bg = this.backgroundColor
    const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4))
    const atts = this.mirrorPassDescriptor.colorAttachments as GPURenderPassColorAttachment[]
    atts[0].clearValue = bg ? { r: lin(bg.x), g: lin(bg.y), b: lin(bg.z), a: 1 } : { r: 0, g: 0, b: 0, a: 0 }
    // The descriptor is reused every frame, so the stamp is set on it rather
    // than passed — same as the scene pass, which is built once too.
    this.mirrorPassDescriptor.timestampWrites = this.stamps("mirror")
    const pass = encoder.beginRenderPass(this.mirrorPassDescriptor)
    pass.setStencilReference(Engine.STENCIL_EYE_VALUE)
    const bundles: GPURenderBundle[] = []
    if (this.mirrorOpaqueBundle) bundles.push(this.mirrorOpaqueBundle)
    if (this.mirrorTransparentBundle) bundles.push(this.mirrorTransparentBundle)
    pass.executeBundles(bundles)
    // Particles and ribbons are scene geometry, and a mirror that dropped them
    // showed a dancer whose hand ribbon cast no reflection. Field effects stay
    // out BY DESIGN: they are display-space overlays composited after the view
    // transform, with no world position to mirror. executeBundles reset the
    // pass state, so these draws bind everything themselves — which they do.
    this.renderParticles(pass, "mirror")
    this.drawTrails(pass, "mirror")
    pass.end()
    this.renderMirrorBlurChain(encoder)
  }

  /**
   * Fill the mirror's mip levels — the bloom pyramid's own 13-tap downsample,
   * one pass per level. Only when the blur dial is up: at zero the ground
   * samples level 0 exactly and the chain would be work nobody reads.
   */
  private renderMirrorBlurChain(encoder: GPUCommandEncoder): void {
    if (this.groundMirrorBlur <= 0 || this.mirrorMipCount < 2) return
    if (!this.mirrorBlurBindGroups) {
      const layout = this.bloomDownsamplePipeline.getBindGroupLayout(0)
      this.mirrorBlurBindGroups = []
      for (let i = 1; i < this.mirrorMipCount; i++) {
        this.mirrorBlurBindGroups.push(
          this.device.createBindGroup({
            label: `mirror blur ${i}`,
            layout,
            entries: [
              { binding: 0, resource: this.mirrorMipViews[i - 1] },
              { binding: 1, resource: this.bloomSampler },
            ],
          }),
        )
      }
    }
    for (let i = 1; i < this.mirrorMipCount; i++) {
      const p = encoder.beginRenderPass({
        label: `mirror blur ${i}`,
        colorAttachments: [{ view: this.mirrorMipViews[i], loadOp: "clear", storeOp: "store" }],
      })
      p.setPipeline(this.bloomDownsamplePipeline)
      p.setBindGroup(0, this.mirrorBlurBindGroups[i - 1])
      p.draw(3)
      p.end()
    }
  }

  /**
   * Can this device multisample the id format at the pass's sample count?
   *
   * Asked by creating one and catching the validation error, because there is
   * no capability flag for it — WebGPU guarantees multisampling for renderable
   * colour formats but implementations have differed on uint targets, and the
   * cost of finding out the hard way is a device-lost on someone's machine and
   * a black canvas.
   *
   * The scope is popped in a finally: leaving an error scope pushed swallows
   * the NEXT error in this device, wherever it happens, and that error would
   * then be attributed to nothing.
   */
  private async probeMultisampledIds(): Promise<boolean> {
    this.device.pushErrorScope("validation")
    let probe: GPUTexture | null = null
    try {
      probe = this.device.createTexture({
        label: "id attachment probe",
        size: [4, 4],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: SCENE_ID_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
    } catch {
      // A synchronous throw is the other way this can fail.
      probe = null
    }
    // Outside the try, and unconditional: the scope is pushed once and must be
    // popped once whichever way creation went.
    const err = await this.device.popErrorScope()
    probe?.destroy()
    return probe !== null && !err
  }

  /**
   * Record an uncaptured validation error, once per distinct message.
   *
   * Distinct, because the interesting property of these is WHICH ones happened,
   * not how many times — a pass that fails validation fails identically every
   * frame, so the second occurrence carries no information the first did not.
   * The count is kept anyway: "1×" and "94000×" distinguish a one-off at init
   * from something the render loop is doing, and that distinction is the first
   * question anyone reading the report will have.
   */
  private noteGpuError(message: string): void {
    const seen = this.gpuErrors.get(message)
    if (seen !== undefined) {
      this.gpuErrors.set(message, seen + 1)
      return
    }
    // The cap is on DISTINCT messages, so it is reached only by a device
    // disagreeing about many different things — at which point the first 32
    // have said what the device is, and the rest are noise.
    if (this.gpuErrors.size >= 32) return
    this.gpuErrors.set(message, 1)
    // First occurrence only, and console.error rather than a silent buffer: a
    // validation error means something did not draw, and a developer with the
    // console open should not have to know this report exists to find out.
    console.error(`[reze] WebGPU validation: ${message}`)
  }

  /** Distinct uncaptured validation messages → how many times each arrived. */
  private readonly gpuErrors = new Map<string, number>()

  /**
   * What this device actually gave us, and what it refused.
   *
   * The report exists because the three answers below are the ones that differ
   * between two browsers on the same machine, and a scene that renders wrong on
   * one of them is otherwise indistinguishable from a scene that is wrong. It is
   * meant to be read off a phone that cannot be attached to a debugger, which is
   * why it returns a value rather than logging: the host decides where to put it.
   */
  gpuReport(): {
    hdrFormat: GPUTextureFormat
    depthFormat: GPUTextureFormat
    reversedZ: boolean
    ids: boolean
    sampleCount: number
    presentationFormat: GPUTextureFormat
    features: string[]
    errors: { message: string; count: number }[]
  } {
    return {
      hdrFormat: this.hdrFormat,
      depthFormat: this.depthFormat,
      reversedZ: this.reversedZ,
      ids: mrtIdsEnabled(),
      sampleCount: Engine.MULTISAMPLE_COUNT,
      presentationFormat: this.presentationFormat,
      features: this.device ? [...this.device.features].sort() : [],
      errors: [...this.gpuErrors].map(([message, count]) => ({ message, count })),
    }
  }

  private rebuildCompositeBindGroup(): void {
    if (!this.device || !this.hdrResolveTexture || !this.compositeBloomView || !this.depthReadView) return
    if (!this.castBuffer) return
    // BEFORE the entries below, not after: they ask fieldPairUsed which effects
    // draw, and that answer includes whether an effect has its field bind group
    // yet. Building the composite first and the field groups second would bind
    // the fallback for a pair that then became drawable in the same call, and
    // the effect would render into a target the composite was not reading.
    this.rebuildFieldBindGroup()
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
        // Whichever equirect is SHOWING — the backdrop if there is one, the
        // world otherwise. The world's light does not come through here; it
        // rides worldSH into the material shells.
        { binding: 6, resource: this.backdropEquirectView ?? this.worldEquirectView ?? this.fallbackEquirectView },
        { binding: 7, resource: { buffer: this.effect?.paramsBuffer ?? this.bgParamsDummyBuffer } },
        { binding: 8, resource: this.depthReadView },
        { binding: 9, resource: { buffer: this.dofUniformBuffer } },
        { binding: 10, resource: (this.agxLutTexture ?? this.agxFallbackTexture).createView({ dimension: "3d" }) },
        { binding: 11, resource: { buffer: this.castBuffer } },
        { binding: 13, resource: { buffer: this.audioBuffer } },
        { binding: 19, resource: { buffer: this.midiBuffer } },
        { binding: 24, resource: { buffer: this.lyricsBuffer } },
        { binding: 15, resource: this.fieldLayerView(this.fieldBgViews[0], 0) },
        { binding: 16, resource: this.fieldLayerView(this.fieldFgViews[0], 0) },
        { binding: 20, resource: this.fieldLayerView(this.fieldBgViews[1], 1) },
        { binding: 21, resource: this.fieldLayerView(this.fieldFgViews[1], 1) },
      ],
    })
  }

  /**
   * Does any installed effect draw into field pair `layer`?
   *
   * Shared with renderFieldPass deliberately. The pass skips a pair nothing
   * draws into, so the composite must read the 1x1 fallback for that pair
   * rather than a target no one cleared. Two spellings of "empty" is two
   * spellings that eventually disagree, and the frame it disagreed on would
   * show last frame's effect after the effect was removed.
   *
   * It is only ever asked at bind-group build time, and setEffects rebuilds the
   * bind group after assigning this.effects — which is exactly the moment a
   * pair can change between empty and not.
   */
  private fieldPairUsed(layer: number): boolean {
    return this.effects.some((e) => e.fieldPipeline && e.fieldBindGroups && e.fieldLayer === layer)
  }

  /** One half of one field pair, as the composite should read it. */
  private fieldLayerView(view: GPUTextureView | null, layer: number): GPUTextureView {
    return this.fieldPairUsed(layer) && view ? view : this.trailFallbackView
  }

  /**
   * The distance field's textures, and the bind groups that walk the flood.
   *
   * Rebuilt with the field targets, because it is sized off the same swap chain
   * and a resize invalidates every view. Torn down entirely when nothing reads
   * it: the memory is two coordinate targets and a distance one at half res,
   * which is real, and a scene with no such effect should not hold it.
   */
  private createCastDistanceTargets(): void {
    for (const t of this.castSeedTextures) t?.destroy()
    this.castCoverageTexture?.destroy()
    this.castDistTexture?.destroy()
    for (const b of this.castStepStrideBuffers) b.destroy()
    this.castSeedTextures = [null, null]
    this.castSeedViews = [null, null]
    this.castCoverageTexture = null
    this.castCoverageView = null
    this.castDistTexture = null
    this.castDistView = null
    this.castStepStrideBuffers = []
    this.castStepBindGroups = []
    this.castSeedBindGroup = null
    this.castResolveBindGroup = null
    if (!this.device || !this.castDistanceWanted || this.fieldFullW === 0) return
    if (!this.castSeedPipeline || !this.castStepPipeline || !this.castResolvePipeline) return

    const w = Math.max(1, Math.ceil(this.fieldFullW / CAST_FIELD_DIV))
    const h = Math.max(1, Math.ceil(this.fieldFullH / CAST_FIELD_DIV))
    for (let i = 0; i < 2; i++) {
      this.castSeedTextures[i] = this.device.createTexture({
        label: `cast distance seeds ${i}`,
        size: [w, h],
        format: CAST_SEED_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.castSeedViews[i] = this.castSeedTextures[i]!.createView()
    }
    this.castCoverageTexture = this.device.createTexture({
      label: "cast coverage",
      size: [w, h],
      format: CAST_COVERAGE_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.castCoverageView = this.castCoverageTexture.createView()
    this.castDistTexture = this.device.createTexture({
      label: "cast distance",
      size: [w, h],
      format: CAST_DIST_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.castDistView = this.castDistTexture.createView()

    // The flood starts at half the longest side and halves to one. That is what
    // makes it exact everywhere rather than out to some radius: every seed gets
    // the chance to reach every texel it is nearest to.
    const strides: number[] = []
    for (let k = 1 << Math.ceil(Math.log2(Math.max(w, h))); k >= 1; k >>= 1) strides.push(k)

    this.castSeedBindGroup = this.device.createBindGroup({
      label: "cast distance seed",
      layout: this.castSeedPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.idView! },
        { binding: 1, resource: { buffer: this.castBuffer } },
      ],
    })
    strides.forEach((stride, i) => {
      const buf = this.device!.createBuffer({
        label: `cast distance stride ${stride}`,
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this.device!.queue.writeBuffer(buf, 0, new Float32Array([stride, 0, 0, 0]))
      this.castStepStrideBuffers.push(buf)
      // Pass i reads the texture pass i-1 wrote. The seed lands in 0, so an even
      // pass reads 0 and writes 1.
      this.castStepBindGroups.push(
        this.device!.createBindGroup({
          label: `cast distance step ${stride}`,
          layout: this.castStepPipeline!.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: this.castSeedViews[i % 2]! },
            { binding: 1, resource: { buffer: buf } },
          ],
        }),
      )
    })
    this.castResolveBindGroup = this.device.createBindGroup({
      label: "cast distance resolve",
      layout: this.castResolvePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.castSeedViews[strides.length % 2]! },
        { binding: 1, resource: this.castCoverageView! },
      ],
    })
  }

  /**
   * The flood, encoded once a frame before the field pass reads it.
   *
   * Seed, then one pass per halving of the stride, then resolve to a distance.
   * Nothing here depends on how far any effect intends to look — that is the
   * whole point of paying for it in passes rather than in per-pixel search.
   */
  private encodeCastDistance(encoder: GPUCommandEncoder): void {
    if (!this.castDistanceWanted || !this.castSeedBindGroup || !this.castResolveBindGroup) return
    const seed = encoder.beginRenderPass({
      label: "cast distance (seed)",
      colorAttachments: [
        { view: this.castSeedViews[0]!, loadOp: "clear", clearValue: { r: -1, g: -1, b: 0, a: 0 }, storeOp: "store" },
        { view: this.castCoverageView!, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" },
      ],
    })
    seed.setPipeline(this.castSeedPipeline!)
    seed.setBindGroup(0, this.castSeedBindGroup)
    seed.draw(3)
    seed.end()

    this.castStepBindGroups.forEach((group, i) => {
      const pass = encoder.beginRenderPass({
        label: "cast distance (flood)",
        colorAttachments: [{ view: this.castSeedViews[(i + 1) % 2]!, loadOp: "clear", clearValue: { r: -1, g: -1, b: 0, a: 0 }, storeOp: "store" }],
      })
      pass.setPipeline(this.castStepPipeline!)
      pass.setBindGroup(0, group)
      pass.draw(3)
      pass.end()
    })

    const resolve = encoder.beginRenderPass({
      label: "cast distance (resolve)",
      colorAttachments: [{ view: this.castDistView!, loadOp: "clear", clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: "store" }],
    })
    resolve.setPipeline(this.castResolvePipeline!)
    resolve.setBindGroup(0, this.castResolveBindGroup)
    resolve.draw(3)
    resolve.end()
  }

  private createFieldTargets(): void {
    if (!this.device || this.fieldFullW === 0) return
    // Same swap chain, same invalidation.
    this.createCastDistanceTargets()
    for (let i = 0; i < Engine.FIELD_SCALES.length; i++) {
      const scale = Engine.FIELD_SCALES[i]
      const w = Math.max(1, Math.ceil(this.fieldFullW / scale))
      const h = Math.max(1, Math.ceil(this.fieldFullH / scale))
      this.fieldBgTextures[i]?.destroy()
      this.fieldFgTextures[i]?.destroy()
      this.fieldBgTextures[i] = this.device.createTexture({
        label: `field layer ${scale === 1 ? "full" : "half"} (background)`,
        size: [w, h],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.fieldFgTextures[i] = this.device.createTexture({
        label: `field layer ${scale === 1 ? "full" : "half"} (foreground)`,
        size: [w, h],
        format: "rgba16float",
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.fieldBgViews[i] = this.fieldBgTextures[i]!.createView()
      this.fieldFgViews[i] = this.fieldFgTextures[i]!.createView()
      this.device.queue.writeBuffer(
        this.fieldUniformBuffers[i],
        0,
        new Float32Array([w, h, this.fieldFullW, this.fieldFullH]),
      )
    }
  }

  /**
   * ONE PER GRID PARITY.
   *
   * The grid alternates which texture holds the current grid, so the field pass
   * needs a bind group for each — built once here rather than rebuilt every
   * frame, which is what a single group would force and is pure waste for a
   * change that only ever toggles between two known states.
   */
  private rebuildFieldBindGroup(): void {
    if (!this.device || !this.depthReadView || !this.compositeBloomView || this.fieldUniformBuffers.length === 0) return
    // Captured, so the null guards above survive into the closure.
    const depth = this.depthReadView
    const bloom = this.compositeBloomView
    const build = (owner: EffectInstance, grid: GPUTextureView) =>
      this.device.createBindGroup({
        label: "field layer bind group",
        layout: this.fieldBindGroupLayout,
        entries: [
          { binding: 3, resource: { buffer: this.compositeUniformBuffer } },
          { binding: 7, resource: { buffer: owner.paramsBuffer ?? this.bgParamsDummyBuffer } },
          { binding: 8, resource: depth },
          { binding: 9, resource: { buffer: this.dofUniformBuffer } },
          { binding: 11, resource: { buffer: this.castBuffer } },
          { binding: 13, resource: { buffer: this.audioBuffer } },
          { binding: 19, resource: { buffer: this.midiBuffer } },
          { binding: 24, resource: { buffer: this.lyricsBuffer } },
          { binding: 25, resource: this.lyricsTextureView },
          // The size uniform for the pair THIS effect draws into.
          { binding: 14, resource: { buffer: this.fieldUniformBuffers[owner.fieldLayer] } },
          { binding: 22, resource: { buffer: owner.fieldClock ?? this.fieldUniformBuffers[owner.fieldLayer] } },
          ...(this.idView ? [{ binding: 23, resource: this.idView }] : []),
          { binding: 17, resource: grid },
          { binding: 18, resource: this.simSampler },
          { binding: 26, resource: this.castDistView ?? this.castDistFallbackView! },
          { binding: 27, resource: this.hdrResolveTexture.createView() },
          { binding: 28, resource: this.simSampler },
          { binding: 2, resource: this.bloomSampler },
          { binding: 5, resource: this.filmicLutView },
          { binding: 10, resource: (this.agxLutTexture ?? this.agxFallbackTexture).createView({ dimension: "3d" }) },
          { binding: 1, resource: bloom },
          { binding: 4, resource: this.maskResolveView },
        ],
      })
    // Per effect: the params buffer and the grid are both its own, so two
    // effects cannot share a bind group even when everything else matches.
    for (const e of this.effects) {
      e.fieldBindGroups = e.grid
        ? [build(e, e.grid.read[0]), build(e, e.grid.read[1])]
        : [build(e, this.simFallbackView), build(e, this.simFallbackView)]
    }
  }

  /**
   * The grid mount's bind group for one parity.
   *
   * A method rather than a closure at the creation site because the SET of
   * buffers in here is a contract with two parties: the grid is built once, and
   * rebuilt whenever a shared buffer it names is replaced (see
   * rebindSharedBuffers). Written twice, the rebuild silently keeps a binding
   * the creation grew — and a bind group that names a destroyed buffer does not
   * fail where it was written, it fails at the next submit.
   */
  private gridBindGroup(
    g: {
      layout: GPUBindGroupLayout
      uniform: GPUBuffer
      read: [GPUTextureView, GPUTextureView]
      textures: [GPUTexture, GPUTexture]
      params: GPUBuffer | null
    },
    i: number,
  ): GPUBindGroup {
    return this.device.createBindGroup({
      layout: g.layout,
      entries: [
        { binding: 0, resource: { buffer: g.uniform } },
        { binding: 1, resource: g.read[i] },
        { binding: 2, resource: this.simSampler },
        { binding: 3, resource: g.textures[1 - i].createView() },
        { binding: 4, resource: { buffer: this.castBuffer } },
        { binding: 5, resource: { buffer: this.audioBuffer } },
        { binding: 6, resource: { buffer: this.compositeUniformBuffer } },
        { binding: 7, resource: { buffer: this.midiBuffer } },
        { binding: 8, resource: { buffer: this.lyricsBuffer } },
        // The grid is the one pass whose own bindings reach 8, so its params sit
        // above them rather than everything else shifting for one mount.
        ...(g.params ? [{ binding: EFFECT_PARAMS_BINDING_GRID, resource: { buffer: g.params } }] : []),
      ],
    })
  }

  /**
   * Re-point EVERY bind group that names a shared scene buffer at the buffer
   * that is there NOW.
   *
   * setAudioData and setMidiNotes do not write their buffer, they REPLACE it:
   * the payload is a different length each time, so the old one is destroyed and
   * a new one takes its place. Every bind group built before that moment still
   * names the dead buffer, and a bind group is not re-read — it holds the
   * resource it was given. The failure is therefore not at the swap but one
   * frame later, as `[Buffer "score"] used in submit while destroyed`, with the
   * scene dead and nothing pointing at the setter that did it.
   *
   * Both setters used to rebind three of the six families that hold these
   * buffers — composite, ribbons, particles — and miss the field mount, the grid
   * mount and the light emitter. Which is to say it worked for every effect that
   * happened not to have a field, and a falling-note effect is exactly the kind
   * that does. So the list lives HERE, once, and both setters call it: the
   * question "who holds this buffer?" now has one place to be answered, and the
   * next binding added is added to a list that everything already consults.
   */
  private rebindSharedBuffers(): void {
    this.rebuildCompositeBindGroup()
    this.rebindTrails()
    this.rebuildFieldBindGroup()
    for (const e of this.effects) {
      if (e.particles) {
        const b = e.particles.rebind()
        e.particles.computeBind = b.computeBind
        e.particles.renderBind = b.renderBind
        e.particles.mirrorRenderBind = b.mirrorRenderBind
      }
      if (e.grid) e.grid.binds = [this.gridBindGroup(e.grid, 0), this.gridBindGroup(e.grid, 1)]
      if (e.lights) e.lights.bind = this.lightEmitBindGroup(e.lights.layout, e.lights.uniform, e.lights.params)
    }
  }

  /**
   * Set a 360° backdrop from an equirectangular (2:1) image — a PhotoDome-style
   * skybox at infinity, sampled per-pixel by view direction so it follows the
   * camera. Display-only: composited in display space behind the scene, it never
   * affects lighting, bloom, or tonemapping. Pass null to remove (the background
   * color, or transparency, takes over again).
   */
  /**
   * The HDRI world: what LIGHTS the scene.
   *
   * Its irradiance goes to the world seat as spherical harmonics, so it lights
   * whether or not it is the thing you see — and it IS the thing you see until
   * a backdrop is set, which is what an HDRI on its own has always done.
   *
   * `strength` is Blender's world-strength dial and is folded into the
   * coefficients, so what lights her is what you see.
   */
  setWorldEquirect(source: HdrImage | null, options?: { strength?: number }): void {
    this.worldEquirectTexture?.destroy()
    this.worldEquirectTexture = null
    this.worldEquirectView = null
    this.worldStrength = Math.max(options?.strength ?? 1, 0)
    const hadSH = this.worldSH !== null
    this.worldSH = null
    if (source && this.device) {
      // Scene-linear radiance in rgba16float. The composite treats it as light
      // rather than wallpaper (mode 3) — a sun in it rolls off like a sun,
      // through the same exposure and view transform as the scene.
      const tex = this.device.createTexture({
        label: "world equirect (HDR)",
        size: [source.width, source.height],
        format: "rgba16float",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      this.device.queue.writeTexture(
        { texture: tex },
        packHalf(source.data),
        { bytesPerRow: source.width * 8, rowsPerImage: source.height },
        [source.width, source.height],
      )
      this.worldEquirectTexture = tex
      this.worldEquirectView = tex.createView()
      // The sky lights the scene, not only backs it. The sun keeps the toon
      // ramp — this is the ambient term, exactly where the flat world colour
      // used to sit.
      this.worldSH = projectIrradianceSH({ ...source, data: source.data }, 4)
      if (this.worldStrength !== 1) {
        for (let i = 0; i < this.worldSH.length; i++) this.worldSH[i] *= this.worldStrength
      }
    }
    if (this.worldSH || hadSH) this.writeWorld()
    this.rebuildCompositeBindGroup()
    if (this.device && this.compositeUniformBuffer) this.writeCompositeViewUniforms()
  }

  /**
   * The 360 backdrop: what you SEE behind the scene.
   *
   * Wallpaper, and only wallpaper — it lights nothing. An HDRI belongs in
   * setWorldEquirect, which is why this no longer takes one: the two shared a
   * slot and were therefore mutually exclusive, and a picture that silently
   * changed the lighting because of its file format was a surprise nobody
   * asked for.
   *
   * Set alongside a world and this is what shows while the world goes on
   * lighting. Cleared, the world's own sky comes back.
   */
  setBackdropEquirect(source: ImageBitmap | HTMLImageElement | HTMLCanvasElement | null): void {
    this.backdropEquirectTexture?.destroy()
    this.backdropEquirectTexture = null
    this.backdropEquirectView = null
    if (source && this.device) {
      let width = Math.max(1, "naturalWidth" in source ? source.naturalWidth : source.width)
      let height = Math.max(1, "naturalHeight" in source ? source.naturalHeight : source.height)
      let upload: ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas = source
      // Panoramas routinely exceed maxTextureDimension2D (e.g. 10000x5000 vs the
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

  private async compileEffect(
    /** The author's file, directives included — parsed here and nowhere else. */
    authored: string,
    params: Record<string, EffectParamValue> | undefined,
    /** This effect's own declarations, already parsed by the caller — which had
     *  to read them anyway to build the scene table. */
    anchors: { bone: string; trail: boolean }[],
    /** Its row of that table: local slot → scene slot. */
    alias: number[],
  ): Promise<{ ok: true; instance: EffectInstance; warnings: string[] } | EffectResult> {
    const noMounts = { background: false, foreground: false }
    if (!this.device) return { ok: false, diagnostics: ["setEffect requires init() to have run"], mounts: noMounts, params: [], duration: 0 }

    // WHAT THE FILE DECLARES, read once. Everything below takes it from `d`
    // rather than running a regex of its own — eight parsers over one file was
    // eight chances to disagree about what it said, and they did.
    //
    // An unrecognised or malformed directive is an ERROR. `#` is not WGSL
    // syntax, so a line starting with one is unambiguously ours and there is
    // nothing to be lenient about; the old spelling lived in comments, where a
    // typo was indistinguishable from prose and could only ever be warned about.
    const parsed = parseDirectives(authored)
    if (parsed.errors.length) return { ok: false, diagnostics: parsed.errors, mounts: noMounts, params: [], duration: 0 }
    const d = parsed.directives
    // The compiler sees the file with its directive lines BLANKED, so every
    // diagnostic below still names the line the author is looking at.
    const wgsl = stripDirectives(authored)

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
      return { ok: false, diagnostics: [
          `a ribbon effect needs both fn trailWidth(u: f32, age: f32) -> f32 and ` +
            `fn trailShade(u: f32, v: f32, age: f32, weight: f32, slot: i32) -> vec4f`,
        ], mounts: noMounts, params: [], duration: 0 }
    }
    if (wantsParticles && !(pe.init && pe.step && pe.shade)) {
      const missing = [
        pe.init ? null : "fn particleInit(id: u32, seed: f32) -> Particle",
        pe.step ? null : "fn particleStep(p: Particle, dt: f32) -> Particle",
        pe.shade ? null : "fn particleShade(p: Particle, uv: vec2f) -> vec4f",
      ].filter(Boolean)
      return { ok: false, diagnostics: [`a particle effect also needs ${missing.join(" and ")}`], mounts: noMounts, params: [], duration: 0 }
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
      return { ok: false, diagnostics: [
          "an effect declares field mounts (background/foreground) or particles, not both — " +
            "split them into two effects",
        ], mounts: noMounts, params: [], duration: 0 }
    }
    // lightEmit counts as a mount on its own: a pure lighting rig draws nothing
    // and is still an effect — it is how a scene gets stage lights without also
    // getting geometry it did not ask for.
    if (!hasBackground && !hasForeground && !wantsParticles && !wantsTrails && !hasLightEmit(wgsl)) {
      return { ok: false, diagnostics: [
          "an effect must define fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f, " +
            "fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f, " +
            "the particle trio (particleInit/particleStep/particleShade), " +
            "the ribbon pair (trailWidth/trailShade), " +
            "or fn lightEmit(i: u32) -> RzLight with #lights <n>",
        ], mounts: noMounts, params: [], duration: 0 }
    }
    const mounts = { background: hasBackground, foreground: hasForeground }

    // ── Directives only some mounts honour ──
    //
    // #bloom sets the aux mask, and only the particle and ribbon modules write
    // that mask: they draw inside the scene pass, in HDR, while the bloom
    // pyramid can still see them. A field effect composites in DISPLAY space
    // after tone mapping, so there is nothing left to pick it up and the
    // directive does exactly nothing. It parsed silently either way, which is
    // the same author-surface lie the three guards above exist to kill — Note
    // Fall declared it for a glow that was its own falloff the whole time, and
    // finding that out cost a round trip.
    //
    // A WARNING, not an error. A published link is immutable, so a scene
    // pinning an effect that declares this has to keep installing; saying so is
    // all that was ever missing.
    const warnings: string[] = []
    if (d.bloom && !wantsParticles && !wantsTrails) {
      warnings.push(
        "#bloom does nothing here. A field effect (background/foreground) composites after tone " +
          "mapping, past the bloom pyramid — the directive applies to particles and ribbons, which draw " +
          "in HDR inside the scene pass. Make the effect's own falloff brighter instead.",
      )
    }

    // ── Which bones did the author ask for? Same idea as the mounts above: a
    // declaration in the source, not a setting somewhere else. Only what is
    // named here gets resolved and uploaded, so naming none costs nothing and
    // naming eight costs eight — rather than every rig's 500 bones costing
    // everybody. Past the cap the extras are dropped rather than silently
    // shifting every slot after them.

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
        return { ok: false, diagnostics: [`invalid param name "${name}" (must be a WGSL identifier)`], mounts, params: d.params, duration: d.duration }
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
    /**
     * The params block, at whatever binding the pass asking has free.
     *
     * A single hardcoded slot cannot serve every mount — the grid's layout
     * already reaches binding 8 — so the struct is generated per pass and each
     * one states the number it can spare. Empty when nothing is declared: WGSL
     * has no empty struct, and a binding nothing reads is a layout mismatch.
     */
    const paramsWgsl = (binding: number) =>
      entries.length
        ? `struct EffectParams {\n${fields.join("\n")}\n}\n@group(0) @binding(${binding}) var<uniform> params: EffectParams;\n`
        : ""
    const paramsDecl = paramsWgsl(EFFECT_PARAMS_BINDING)

    // ── Compile with validation captured, not thrown at the console. Line
    // numbers in diagnostics are rebased to the USER's source.
    // The composite is STATIC: user field code compiles in its own half-res
    // module (buildFieldShader), so a bad effect can no longer produce errors at
    // line numbers in a shader the author never wrote — and installing one no
    // longer recompiles the composite's tone-mapping half at all.
    const gridSize = gridEntryPoint(wgsl) ? Math.min(d.grid || 256, GRID_MAX) : 0
    // `alias` goes in: a field effect reads bones through _rzSlot exactly as a
    // particle one does, and it was the only module never handed the mapping.
    const fieldEffect =
      hasBackground || hasForeground ? { wgsl, paramsDecl, hasBackground, hasForeground, gridSize, alias, trailCount: anchors.filter((a) => a.trail).length } : null
    const source = buildCompositeShader(fieldEffect)
    this.device.pushErrorScope("validation")
    const module = this.device.createShaderModule({ label: "composite shader (effect)", code: source })
    const scopeErr = await this.device.popErrorScope()
    if (scopeErr) return { ok: false, diagnostics: [scopeErr.message], mounts, params: d.params, duration: d.duration }

    // Declared like every other mount property: by what the source says, not by
    // a setting somewhere else that an author cannot see from the file.
    const layerBlend = d.additiveLayer
      ? FIELD_LAYER_BLEND_ADDITIVE
      : FIELD_LAYER_BLEND
    let fieldPipeline: GPURenderPipeline | null = null
    if (fieldEffect) {
      const fieldSource = buildFieldShader({ ...fieldEffect, ids: mrtIdsEnabled() })
      const userLineOffset = fieldSource.slice(0, fieldSource.indexOf(wgsl)).split("\n").length - 1
      this.device.pushErrorScope("validation")
      const fieldModule = this.device.createShaderModule({ label: "field shader (effect)", code: fieldSource })
      const info = await fieldModule.getCompilationInfo()
      const fieldScopeErr = await this.device.popErrorScope()
      const diagnostics = info.messages
        .filter((m) => m.type === "error")
        .map((m) => `${Math.max(0, m.lineNum - userLineOffset)}:${m.linePos} ${m.message}`)
      if (diagnostics.length === 0 && fieldScopeErr) diagnostics.push(fieldScopeErr.message)
      if (diagnostics.length > 0) return { ok: false, diagnostics, mounts, params: d.params, duration: d.duration }
      try {
        fieldPipeline = await this.device.createRenderPipelineAsync({
          label: "field layer pipeline",
          layout: this.fieldPipelineLayout,
          vertex: { module: fieldModule, entryPoint: "fieldVs" },
          fragment: {
            module: fieldModule,
            entryPoint: "fieldFs",
            // OVER, accumulating PREMULTIPLIED colour: several effects draw into
            // these two targets in document order, and each must layer onto what
            // the earlier ones left rather than replace it. src-alpha on colour
            // premultiplies as it writes; alpha accumulates as one-over. An
            // author still returns STRAIGHT colour+alpha, exactly as before —
            // the premultiplication happens here, and the composite reads it
            // back knowing that. With one effect over a cleared target the
            // result is identical to the replace it used to do.
            targets: [
              { format: "rgba16float", blend: layerBlend },
              { format: "rgba16float", blend: layerBlend },
            ],
          },
          primitive: { topology: "triangle-list" },
          multisample: { count: 1 },
        })
      } catch (e) {
        return { ok: false, diagnostics: [e instanceof Error ? e.message : String(e)], mounts, params: d.params, duration: d.duration }
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
      return { ok: false, diagnostics: [e instanceof Error ? e.message : String(e)], mounts, params: d.params, duration: d.duration }
    }

    // Built BEFORE the swap: a particle stage that fails to compile has to leave
    // the previously installed effect running, exactly as a bad composite does.
    // A stage that fails after an earlier one succeeded would otherwise strand
    // the earlier one's buffers: the candidate is never installed, so no release
    // path ever sees them. Matters more with a list — one bad effect among
    // thirteen should cost nothing but itself.
    // DECLARED BEFORE abandon, and that is load-bearing.
    //
    // abandon closes over all three. Declaring them after it meant that a
    // failure in the FIRST stage — particles — called a function whose body
    // touches `grid`, which was still in its temporal dead zone: the throw
    // replaced the diagnostic with "Cannot access 'grid' before initialization"
    // and the author never saw why their shader was rejected. Only the first
    // stage could hit it, which is why it survived: a bad grid or a bad ribbon
    // reported correctly.
    let particles: EffectParticles | null = null
    let grid: EffectGrid | null = null
    let trails: EffectTrails | null = null

    // PARAMS ARE NOT A FIELD-MOUNT FEATURE, and used to be one by accident.
    //
    // `paramsDecl` was spliced into the composite alone, so `#param` worked for
    // background/foreground and produced "unresolved value 'params'" for every
    // particle, grid and ribbon effect — which is most of the ones anyone wants
    // a dial on. Rain's fall speed is the whole example.
    //
    // The buffer is created HERE rather than beside the instance it ends up on,
    // because the mount builders below need to bind it and they run first. That
    // makes `abandon` its owner: a mount that fails to compile must not leak it.
    let paramsBuffer: GPUBuffer | null = null
    if (entries.length) {
      paramsBuffer = this.device.createBuffer({
        label: "effect params",
        size: paramsData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(paramsBuffer, 0, paramsData)
    }
    /** What every mount splices and binds: a generator for the struct decl at
     *  the binding that mount has free, and the buffer behind it. The buffer is
     *  null when nothing is declared, and then no binding is added at all. */
    const paramsFor: EffectParamsBinding = { wgsl: paramsWgsl, buffer: paramsBuffer }

    const abandon = (diagnostics: string[]): EffectResult => {
      paramsBuffer?.destroy()
      particles?.buffer.destroy()
      particles?.uniform.destroy()
      grid?.textures[0].destroy()
      grid?.textures[1].destroy()
      grid?.uniform.destroy()
      trails?.uniform.destroy()
      return { ok: false, diagnostics, mounts, params: d.params, duration: d.duration }
    }
    if (wantsParticles) {
      const built = await this.buildParticles(wgsl, d, anchors, alias, paramsFor)
      if (!built.ok) return abandon(built.diagnostics)
      particles = built.state
    }
    if (gridEntryPoint(wgsl)) {
      const built = await this.buildSim(wgsl, d, anchors, alias, paramsFor)
      if (!built.ok) return abandon(built.diagnostics)
      grid = built.state
    }
    if (wantsTrails) {
      // Only anchors that asked for `trail` have a path to draw; a ribbon on a
      // bone recorded without one would read zeroes and paint a line to the origin.
      const trailSlots = anchors.filter((a) => a.trail).length
      if (trailSlots === 0) {
        return abandon(["a ribbon effect needs at least one #anchor <bone> trail"])
      }
      const built = await this.buildTrails(wgsl, d, anchors, alias, paramsFor)
      if (!built.ok) return abandon(built.diagnostics)
      trails = built.state
    }

    // ── The lightEmit mount ──
    //
    // Both halves or neither, the same rule the particle trio and the ribbon
    // pair follow: a count with no emitter allocates slots nobody writes (a
    // light stuck wherever the buffer last left it), and an emitter with no
    // count is a function nothing calls. Either alone is a silent blank, which
    // is the worst way for an effect to fail.
    let lights: EffectInstance["lights"] = null
    const declaredLights = Math.min(d.lights, MAX_LIGHTS)
    const emits = hasLightEmit(wgsl)
    if (declaredLights > 0 !== emits) {
      return abandon([
        emits
          ? "an effect defining fn lightEmit(i: u32) -> RzLight must also declare how many with #lights <n>"
          : "#lights <n> needs fn lightEmit(i: u32) -> RzLight to fill those slots",
      ])
    }
    if (declaredLights > 0) {
      const built = await this.buildLightEmit(wgsl, declaredLights, alias, anchors, paramsFor)
      if (!built.ok) return abandon(built.diagnostics)
      lights = built.state
    }

    // One per emitting-or-drawing field effect. 16 bytes, written per frame.
    const fieldClock = fieldPipeline
      ? this.device.createBuffer({
          label: "field clock",
          size: 16,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
      : null

    const instance: EffectInstance = {
        wgsl,
        paramDecls: d.params,
        duration: d.duration,
        paramLayout: layout,
        paramsBuffer,
        paramsData,
        hasBackground,
        hasForeground,
        // The author's OWN source, not the assembled module: the assembled one
        // always carries the accessors (as real readers or as the zero stubs),
        // so matching against it would report every effect as a reader and the
        // attachment would be stored exactly as often as before.
        readsIds: /\brz(?:ObjectAt|MaterialAt)\s*\(/.test(wgsl),
        readsCastDistance: castDistanceUsed(wgsl),
        anchors,
        // The effect's own clock starts now. Per effect so that one installed
        // later still gets a frame where rzGridFrame() is 0 and can seed.
        epochScene: this.sceneClock,
        // Fully on, unscheduled. An effect that is installed is showing;
        // scheduling it is something a caller does afterwards, and an install
        // that silently began at zero would look like a compile that failed.
        influence: 1,
        window: null,
        weight: 1,
        // Its OWN resolution, no longer the scene's: an effect that never asked
        // for full res is not promoted because a neighbour did.
        // FULL RESOLUTION UNLESS TOLD OTHERWISE.
        //
        // It was the other way round, and the default was the bug. An author
        // who has never heard of the flag writes an effect with an edge in it
        // and gets a soft one — nothing fails, nothing warns, because nothing
        // was declared to fail. Three shipped effects DID declare it and were
        // half-res anyway on a parsing technicality, which is the same bug
        // wearing a different hat: the safe answer has to be the one you get
        // for saying nothing.
        //
        // The cost is real and is why the half layer stays: `#halfres` is worth
        // about 3.7x on a full-screen effect (Footprints, measured, 1.2ms
        // against 4.5ms). It is the right call for a soft additive glow, which
        // upsamples invisibly — and it is now a claim an author makes about
        // their own effect rather than a fate that befalls one.
        fieldLayer: d.fieldLayer,
        fieldPipeline,
        fieldClock,
        // Filled by rebuildFieldBindGroup below, which needs the instance to
        // exist first — it binds this effect's own params buffer and grid.
        fieldBindGroups: null,
        particles,
        grid,
        trails,
        lights,
    }
    return { ok: true, instance, warnings }
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
  /**
   * Install a LIST of effects, in document order — the order they layer in.
   *
   * Each is compiled independently and a failure is contained: it is reported in
   * its own slot of the returned array and left out of the scene, while the rest
   * install. That is the style-group discipline, and it matters more here — with
   * four effects on screen, "one bad shader blanks the scene" is the first bug
   * report anyone would file.
   *
   * The bones are allocated ONCE for the whole list: two effects naming the same
   * wrist share one address and one recorded path, and the cap is eight distinct
   * bones across the scene rather than per file.
   *
   * Null or empty clears everything.
   */
  async setEffects(
    list: { wgsl: string; params?: Record<string, EffectParamValue> }[] | null,
  ): Promise<EffectResult[]> {
    const noMounts = { background: false, foreground: false }
    if (!this.device) return [{ ok: false, diagnostics: ["setEffects requires init() to have run"], mounts: noMounts, params: [], duration: 0 }]

    const requested = list ?? []
    if (requested.length === 0) {
      for (const e of this.effects) {
      e.paramsBuffer?.destroy()
      e.lights?.uniform.destroy()
      e.fieldClock?.destroy()
    }
      this.releaseParticles()
      this.releaseTrails()
      this.releaseGrid()
      this.effects = []
      this.allocateLightSlots()
      this.anchorTable = EMPTY_ANCHOR_TABLE
      this.clearTrailHistory()
      const module = this.device.createShaderModule({ label: "composite shader", code: buildCompositeShader(null) })
      this.compositePipelineIdentity = this.makeCompositePipeline(module, false, "composite pipeline (gamma=1)")
      this.compositePipelineGamma = this.makeCompositePipeline(module, true, "composite pipeline (gamma!=1)")
      this.rebuildCompositeBindGroup()
      this.writeCompositeViewUniforms()
      return []
    }

    // One table for the whole scene, built before anything compiles: an effect's
    // alias is its row, and a bone two effects both name is allocated once.
    // The table needs every effect's anchors before any of them compiles, so
    // this is the one place a source is read twice — compileEffect parses it
    // again for everything else. A malformed file yields no anchors here and
    // fails with its real diagnostics there, which is the right order: the
    // error names the line, not the table.
    const perEffectAnchors = requested.map((e) =>
      parseDirectives(e.wgsl).directives.anchors.slice(0, MAX_EFFECT_ANCHORS),
    )
    const table = buildAnchorTable(perEffectAnchors, MAX_EFFECT_ANCHORS)

    const results: EffectResult[] = []
    const instances: EffectInstance[] = []
    for (let i = 0; i < requested.length; i++) {
      const built = await this.compileEffect(
        requested[i].wgsl,
        requested[i].params,
        perEffectAnchors[i],
        table.alias[i],
      )
      if (!("instance" in built)) {
        // Contained: this one is out, the others carry on.
        results.push(built)
        continue
      }
      instances.push(built.instance)
      results.push({
        ok: true,
        params: built.instance.paramDecls,
        duration: built.instance.duration,
        // Installed, and still with something to say — a directive that parsed
        // but will never fire. Same channel as the dropped-anchor note below.
        diagnostics: built.warnings,
        mounts: { background: built.instance.hasBackground, foreground: built.instance.hasForeground },
      })
    }
    // An anchor the cap refused is worth saying out loud on the effect that
    // asked for it — its rzAnchor will read invalid, and silence would make that
    // look like a rig that spells the bone differently.
    for (const d of table.dropped) {
      const r = results[d.effect]
      if (r) r.diagnostics.push(`anchor "${d.bone}" dropped: the scene is already using all ${MAX_EFFECT_ANCHORS} slots`)
    }

    // ── Swap. Everything above either succeeded or was excluded, so the scene
    // that was running is only torn down now.
    for (const e of this.effects) {
      e.paramsBuffer?.destroy()
      e.lights?.uniform.destroy()
      e.fieldClock?.destroy()
    }
    this.releaseParticles()
    this.releaseTrails()
    this.releaseGrid()
    this.effects = instances
    // Turn the distance field on or off with the list that asked for it. Only
    // rebuilds when the answer CHANGES: the targets are half-res render
    // attachments and reallocating them per install would churn them for every
    // unrelated effect a scene adds.
    const wantsCastDistance = instances.some((e) => e.readsCastDistance)
    if (wantsCastDistance !== this.castDistanceWanted) {
      this.castDistanceWanted = wantsCastDistance
      this.createCastDistanceTargets()
      // The field bind groups name the distance texture, and it has just been
      // created or destroyed — they are rebuilt below with the new list anyway.
    }
    // The new list's emitters need their slots before the next frame reads them.
    this.allocateLightSlots()
    // A rig the cap refused is worth saying out loud on the effect that asked
    // for it — the dropped-anchor rule. Its lights are simply absent otherwise,
    // and absence reads as a broken shader rather than a full scene.
    {
      let ok = 0
      for (const r of results) {
        if (!r.ok) continue
        const l = instances[ok++]?.lights
        if (l && l.count > 0 && l.data[2] === 0) {
          r.diagnostics.push(
            `${l.count} light${l.count === 1 ? "" : "s"} dropped: the scene is already using all ${MAX_LIGHTS} slots`,
          )
        }
      }
    }
    this.anchorTable = table
    // Slots have been re-dealt; a recorded path is stale by ADDRESS, not by age.
    this.clearTrailHistory()

    // Scene-level, from the union: the composite decides only whether to SAMPLE
    // the field layers, so one effect with a background is enough to turn that
    // on for the frame.
    const hasBackground = instances.some((e) => e.hasBackground)
    const hasForeground = instances.some((e) => e.hasForeground)
    const compositeModule = this.device.createShaderModule({
      label: "composite shader (effects)",
      code: buildCompositeShader(
        hasBackground || hasForeground
          ? // No wgsl and so no trails: the composite hosts no effect source at
            // all, it only decides whether to sample the layer the field pass drew.
            { wgsl: "", paramsDecl: "", hasBackground, hasForeground, gridSize: 0, trailCount: 0 }
          : null,
      ),
    })
    this.compositePipelineIdentity = this.makeCompositePipeline(compositeModule, false, "composite pipeline (gamma=1)")
    this.compositePipelineGamma = this.makeCompositePipeline(compositeModule, true, "composite pipeline (gamma!=1)")

    // Nothing to promote any more: `#fullres` is per effect, read into
    // fieldLayer when the instance is built, and both target pairs exist for
    // the life of the surface. What used to be a scene-wide decision made here
    // is now each effect's own.
    this.rebuildFieldBindGroup()
    this.rebuildCompositeBindGroup()
    this.writeCompositeViewUniforms()
    return results
  }

  /**
   * Install ONE effect — the singleton API, kept because most scenes are one
   * effect and every existing caller uses it. A one-element setEffects.
   */
  async setEffect(wgsl: string | null, params?: Record<string, EffectParamValue>): Promise<EffectResult> {
    const noMounts = { background: false, foreground: false }
    if (wgsl === null) {
      await this.setEffects(null)
      return { ok: true, diagnostics: [], mounts: noMounts, params: [], duration: 0 }
    }
    const [result] = await this.setEffects([{ wgsl, params }])
    return result ?? { ok: false, diagnostics: ["effect failed to install"], mounts: noMounts, params: [], duration: 0 }
  }

  private async buildParticles(
    /** Already stripped of directives — see compileEffect. */
    wgsl: string,
    /** What the file declared. Read here rather than re-parsed: the source no
     *  longer carries the lines, and two readers is how they drift. */
    d: EffectDirectives,
    anchors: { bone: string; trail: boolean }[],
    /** This effect's local→scene slot map. Passed rather than read off the
     *  engine: the builders run BEFORE the swap, so this.anchorTable still
     *  describes the effect that is still on screen. */
    alias: number[],
    /** The effect's declared dials. Spliced into both stages and bound at 7,
     *  the same binding the composite gives them. */
    params: EffectParamsBinding,
  ): Promise<{ ok: true; state: EffectParticles } | { ok: false; diagnostics: string[] }> {
    // No pragma means "some": an author who wrote the trio clearly wants
    // particles, and failing over a missing comment would be pedantry.
    const count = Math.min(d.particles || 1024, Engine.MAX_PARTICLES)
    const src = { wgsl, count, blend: d.particleBlend, bloom: d.bloom, paramsDecl: params.wgsl(EFFECT_PARAMS_BINDING) }
    // Sparks want to spawn where a trail is, so the particle stages see the same
    // cast buffer the trail draw reads.
    const cast = {
      subjects: MAX_EFFECT_SUBJECTS,
      samples: TRAIL_SAMPLES,
      base: MAX_EFFECT_SUBJECTS * 3,
      trailBase: CAST_TRAIL_BASE,
      slots: MAX_EFFECT_ANCHORS,
      trailCount: anchors.filter((x) => x.trail).length,
      alias,
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
      // Two vec4-sized rows: (time, dt, count, frame) and (weight, _, _, _).
      // The first was exactly full, and weight has to live in the same buffer
      // as the clock or a frame could draw one without the other.
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const uniformBytes = new ArrayBuffer(32)
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
          // The score, for rzNote*/rzKey* — bound wherever audio is, so a spawn
          // rule and a background read the same instant.
          { binding: 5, visibility, buffer: { type: "read-only-storage" } },
          { binding: 6, visibility, buffer: { type: "read-only-storage" } },
          // Only when the effect declared any: WGSL has no empty struct, so a
          // param-less effect must not carry the decl OR the binding.
          ...(params.buffer ? [{ binding: 7, visibility, buffer: { type: "uniform" as const } }] : []),
        ],
      })
    const bindFor = (layout: GPUBindGroupLayout, camera: GPUBuffer) =>
      this.device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer } },
          { binding: 1, resource: { buffer: uniform } },
          { binding: 2, resource: { buffer: camera } },
          { binding: 3, resource: { buffer: this.castBuffer } },
          { binding: 4, resource: { buffer: this.audioBuffer } },
          { binding: 5, resource: { buffer: this.midiBuffer } },
          { binding: 6, resource: { buffer: this.lyricsBuffer } },
          ...(params.buffer ? [{ binding: 7, resource: { buffer: params.buffer } }] : []),
        ],
      })

    const computeLayout = layoutFor("storage", GPUShaderStage.COMPUTE)
    const renderLayout = layoutFor("read-only-storage", GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT)

    // Additive keeps the destination and adds to it, and leaves alpha alone, so
    // a glow does not claim coverage it never occluded. The MASK sums with it —
    // otherwise an additive effect could never reach the bloom gate. Both live
    // in scene-contract as the "particle-additive" class.
    const targets = sceneTargetsFor(src.blend === "additive" ? "particle-additive" : "particle", this.sceneFormats)

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
        fragment: { module: renderModule, entryPoint: "fs", targets },
        primitive: { topology: "triangle-list", cullMode: "none" },
        // Tested but not WRITTEN: particles are transparent, so writing depth
        // would make whichever quad drew first occlude the ones behind it.
        depthStencil: { format: this.depthFormat, depthWriteEnabled: false, depthCompare: this.depthAhead },
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
          computeBind: bindFor(computeLayout, this.cameraUniformBuffer),
          render,
          renderLayout,
          renderBind: bindFor(renderLayout, this.cameraUniformBuffer),
          mirrorRenderBind: bindFor(renderLayout, this.mirrorCameraBuffer),
          rebind: () => ({
            computeBind: bindFor(computeLayout, this.cameraUniformBuffer),
            renderBind: bindFor(renderLayout, this.cameraUniformBuffer),
            mirrorRenderBind: bindFor(renderLayout, this.mirrorCameraBuffer),
          }),
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
  /**
   * Compile an effect's lightEmit stage and give it a bind group.
   *
   * Its own layout rather than a shared one: this is the only place the lights
   * buffer is WRITABLE, and every other binding of it is read-only. Keeping the
   * writable view here means a material pipeline cannot accidentally acquire
   * one.
   */
  private async buildLightEmit(
    wgsl: string,
    count: number,
    alias: number[],
    anchors: { bone: string; trail: boolean }[],
    params: EffectParamsBinding,
  ): Promise<{ ok: true; state: NonNullable<EffectInstance["lights"]> } | { ok: false; diagnostics: string[] }> {
    const layout = this.device.createBindGroupLayout({
      label: "light emit layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ...(params.buffer
          ? [{ binding: EFFECT_PARAMS_BINDING, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" as const } }]
          : []),
      ],
    })
    const module = this.device.createShaderModule({
      label: "light emit",
      // The same scene API and the same per-effect anchor alias its drawing
      // half gets, so a lamp reads the cast exactly as the beam that paints it.
      code: buildLightEmitShader(
        wgsl,
        EFFECT_SCENE_API + anchorAliasWgsl(alias),
        { trailCount: anchors.filter((a) => a.trail).length },
        params.wgsl(EFFECT_PARAMS_BINDING),
      ),
    })
    const info = await module.getCompilationInfo()
    const diagnostics = info.messages.filter((m) => m.type === "error").map((m) => `${m.lineNum}:${m.linePos} ${m.message}`)
    if (diagnostics.length) return { ok: false, diagnostics }
    const data = new Float32Array(4)
    const uniform = this.device.createBuffer({
      label: "light emit uniform",
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.pushErrorScope("validation")
    const pipeline = await this.device.createComputePipelineAsync({
      label: "light emit pipeline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module, entryPoint: "lightEmitMain" },
    })
    const scoped = await this.device.popErrorScope()
    if (scoped) {
      uniform.destroy()
      return { ok: false, diagnostics: [scoped.message] }
    }
    const bind = this.lightEmitBindGroup(layout, uniform, params.buffer)
    // The layout travels with the state so the emitter can be rebound when a
    // shared buffer under it is replaced — see rebindSharedBuffers.
    return { ok: true, state: { pipeline, bind, layout, uniform, data, count, params: params.buffer } }
  }

  /** The light emitter's bind group. One author, for the reason gridBindGroup
   *  gives: it is built once and rebuilt on every shared-buffer swap. */
  private lightEmitBindGroup(layout: GPUBindGroupLayout, uniform: GPUBuffer, params: GPUBuffer | null): GPUBindGroup {
    return this.device.createBindGroup({
      label: "light emit bind",
      layout,
      entries: [
        { binding: 0, resource: { buffer: this.lightsBuffer } },
        { binding: 1, resource: { buffer: uniform } },
        { binding: 2, resource: { buffer: this.compositeUniformBuffer } },
        { binding: 3, resource: { buffer: this.castBuffer } },
        { binding: 4, resource: { buffer: this.audioBuffer } },
        { binding: 5, resource: { buffer: this.midiBuffer } },
        { binding: 6, resource: { buffer: this.lyricsBuffer } },
        ...(params ? [{ binding: EFFECT_PARAMS_BINDING, resource: { buffer: params } }] : []),
      ],
    })
  }

  /**
   * Hand every emitting effect its slot range, and write the total.
   *
   * Document lights sit FIRST, at slot 0, and effects follow in document order.
   * That ordering is the stable one: a scene's own lamps are the thing a person
   * placed and can see in a list, so they should not move because an effect was
   * installed ahead of them.
   *
   * Called whenever either producer changes. The base lands in each effect's
   * uniform, so nothing recompiles.
   */
  private allocateLightSlots(): void {
    let next = this.docLightCount
    for (const e of this.effects) {
      if (!e.lights) continue
      // Past the cap an effect gets NOTHING rather than a partial rig: half a
      // set of stage lights is a lighting design nobody authored.
      const fits = next + e.lights.count <= MAX_LIGHTS
      e.lights.data[1] = next
      e.lights.data[2] = fits ? e.lights.count : 0
      if (fits) next += e.lights.count
      this.device.queue.writeBuffer(e.lights.uniform, 0, e.lights.data.buffer as ArrayBuffer)
    }
    this.lightHeader[0] = Math.min(next, MAX_LIGHTS)
    this.device.queue.writeBuffer(this.lightsBuffer, 0, this.lightHeader)
  }

  /** Run every effect's lightEmit, before the pass that reads the result. */
  private emitLights(encoder: GPUCommandEncoder): void {
    for (const e of this.effects) {
      const l = e.lights
      // NOT skipped at weight 0, unlike every other mount. Each effect writes
      // its OWN slots in a shared buffer that is never cleared, so a skipped
      // dispatch leaves last frame's lights burning — the one place where not
      // running is the wrong answer. The shader zeroes them instead, and the
      // dispatch it costs is a single workgroup.
      if (!l || l.data[2] === 0) continue
      // The effect's OWN epoch — the same one its field, particle, ribbon and
      // grid halves now read. This was briefly conditional, to match a field
      // clock that was shared from the first installed effect; that clock is
      // per effect now, so every mount in one file agrees by construction.
      l.data[0] = this.sceneClock - e.epochScene
      l.data[3] = e.weight
      this.device.queue.writeBuffer(l.uniform, 0, l.data.buffer as ArrayBuffer)
      const cp = encoder.beginComputePass({ label: "light emit" })
      cp.setPipeline(l.pipeline)
      cp.setBindGroup(0, l.bind)
      cp.dispatchWorkgroups(Math.ceil(l.data[2] / 64))
      cp.end()
    }
  }

  private stepParticles(encoder: GPUCommandEncoder, deltaTime: number): void {
    for (const e of this.effects) {
      const p = e.particles
      if (!p) continue
      p.data[0] = this.sceneClock - e.epochScene
      // The SIMULATION runs at every weight, 0 included — only the draw stops.
      // A scheduled effect that froze while faded out would resume from the
      // state it left rather than the one it would have reached, so fading one
      // back in would rewind it.
      p.data[4] = e.weight
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
  }

  /** Draw the pool. Inside the scene pass, so it is depth-tested and pre-bloom. */
  private renderParticles(pass: GPURenderPassEncoder, view: "camera" | "mirror"): void {
    for (const e of this.effects) {
      const p = e.particles
      if (!p || e.weight === 0) continue
      pass.setPipeline(p.render)
      pass.setBindGroup(0, view === "mirror" ? p.mirrorRenderBind : p.renderBind)
      pass.draw(6, p.count)
    }
  }

  /**
   * Compile an effect's ribbon stage.
   *
   * One instance per (slot, subject, segment), so a scene with several dancers
   * and several declared bones is still one draw and nothing is computed per
   * frame on the CPU.
   */
  private async buildTrails(
    /** Already stripped of directives — see compileEffect. */
    wgsl: string,
    /** What the file declared. Read here rather than re-parsed: the source no
     *  longer carries the lines, and two readers is how they drift. */
    d: EffectDirectives,
    anchors: { bone: string; trail: boolean }[],
    /** This effect's local→scene slot map. Passed rather than read off the
     *  engine: the builders run BEFORE the swap, so this.anchorTable still
     *  describes the effect that is still on screen. */
    alias: number[],
    /** The effect's declared dials, spliced and bound at 7 as everywhere else. */
    params: EffectParamsBinding,
  ): Promise<{ ok: true; state: EffectTrails } | { ok: false; diagnostics: string[] }> {
    // `slots` here is how many RIBBONS to draw — one per trailed anchor — which
    // is a different number from the anchor ADDRESS SPACE the accessors index
    // by. Conflating the two is what made a trail declared after a bare anchor
    // read zeroes; they are now named apart and computed apart.
    // Which LOCAL anchor each ribbon belongs to. Identity for an all-trailed
    // file (every library effect today), and the reason a mixed one drew
    // nothing before: ribbon i was read as anchor slot i.
    const ribbonSlots = anchors.map((a, i) => (a.trail ? i : -1)).filter((i) => i >= 0)
    const slots = ribbonSlots.length
    const src = { wgsl, slots, ribbonSlots, blend: d.particleBlend, bloom: d.bloom, paramsDecl: params.wgsl(EFFECT_PARAMS_BINDING) }
    const code = buildTrailShader(src, {
      subjects: MAX_EFFECT_SUBJECTS,
      samples: TRAIL_SAMPLES,
      base: MAX_EFFECT_SUBJECTS * 3,
      trailBase: CAST_TRAIL_BASE,
      slots: MAX_EFFECT_ANCHORS,
      alias,
      reversedZ: this.reversedZ,
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
        // No depth binding, and binding 3 stays vacant rather than renumbering.
        // Ribbons draw inside the scene pass now, and sampling that pass's own
        // depth attachment from within it is a usage conflict WebGPU rejects.
        // The audio analysis, for rzAudio* in width and shade alike.
        { binding: 4, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // The score, for rzNote*/rzKey*.
        { binding: 5, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // The lyrics, for rzLyric*.
        { binding: 6, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        ...(params.buffer
          ? [{ binding: 7, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" as const } }]
          : []),
      ],
    })
    // TWO targets, the scene pass's own: HDR colour and the aux (bloom mask,
    // coverage). Ribbons draw INSIDE that pass now, so they are lit geometry
    // rather than a layer pasted over the finished frame — which is the whole
    // point: a layer composited after tone mapping can never bloom.
    //
    // Additive, where this used to be MAX. Max was right for a post-tonemap
    // layer; in HDR before bloom, overlapping light sums.
    const targets = sceneTargetsFor("trail", this.sceneFormats)
    this.device.pushErrorScope("validation")
    try {
      const pipeline = await this.device.createRenderPipelineAsync({
        label: "trail pipeline",
        layout: this.device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        vertex: { module, entryPoint: "vs" },
        fragment: { module, entryPoint: "fs", targets },
        primitive: { topology: "triangle-list", cullMode: "none" },
        // Depth TESTED, never written: a ribbon is occluded by the body it
        // circles, and must not occlude the fabric drawn after it.
        depthStencil: {
          format: this.depthFormat,
          depthWriteEnabled: false,
          depthCompare: this.reversedZ ? "greater" : "less",
        },
        multisample: { count: Engine.MULTISAMPLE_COUNT },
      })
      const scoped = await this.device.popErrorScope()
      if (scoped) {
        uniform.destroy()
        return { ok: false, diagnostics: [scoped.message] }
      }
      return {
        ok: true,
        state: {
          // Ribbons declared by this effect. The INSTANCE count is no longer
          // baked here — it follows the live subject count and is computed per
          // draw (see drawTrails).
          slots,
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
              { binding: 4, resource: { buffer: this.audioBuffer } },
              { binding: 5, resource: { buffer: this.midiBuffer } },
              { binding: 6, resource: { buffer: this.lyricsBuffer } },
              ...(params.buffer ? [{ binding: 7, resource: { buffer: params.buffer } }] : []),
            ],
          }),
          mirrorBind: this.device.createBindGroup({
            layout,
            entries: [
              { binding: 0, resource: { buffer: this.castBuffer } },
              { binding: 1, resource: { buffer: uniform } },
              { binding: 2, resource: { buffer: this.mirrorCameraBuffer } },
              { binding: 4, resource: { buffer: this.audioBuffer } },
              { binding: 5, resource: { buffer: this.midiBuffer } },
              { binding: 6, resource: { buffer: this.lyricsBuffer } },
              ...(params.buffer ? [{ binding: 7, resource: { buffer: params.buffer } }] : []),
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
    for (const e of this.effects) {
      e.trails?.uniform.destroy()
      e.trails = null
    }
  }

  /**
   * Ribbons, drawn INSIDE the scene pass — as geometry, in HDR, before bloom.
   *
   * They used to own a colour target and be pasted over the finished frame
   * after tone mapping, which is exactly why they could not bloom: nothing
   * composited post-tonemap can. Here they are lit like anything else in the
   * scene, depth-tested against the body they circle, and their emission
   * reaches the bloom prefilter through the aux mask they now write.
   *
   * Takes the pass rather than opening one: that IS the change.
   */
  private drawTrails(pass: GPURenderPassEncoder, view: "camera" | "mirror"): void {
    const drawn = this.effects.filter((e) => e.trails && e.weight > 0)
    if (drawn.length === 0) return
    for (const e of drawn) {
      const t = e.trails!
      // The clock upload happens once, on the camera draw: queue writes land
      // before the encoder submits, so both passes read the same value — the
      // mirror draw writing it again would only write it twice.
      // Instances follow the LIVE subject count, not MAX_EFFECT_SUBJECTS.
      //
      // This used to be baked at install as slots x 4 x (samples-1) x subs, so a
      // scene with ONE character issued four characters' worth of ribbon quads
      // and threw three quarters of them away as degenerate — every frame, at
      // every sample length. Vertex invocations with no fragments are cheap, not
      // free, and they scale with the sample count, which is what made a longer
      // trail expensive.
      //
      // The shader decodes [ribbon][subject][segment] with the same number out
      // of its uniform, so the two cannot drift: change one without the other
      // and ribbons land on the wrong subject rather than merely costing more.
      const live = Math.max(1, this.castSubjectCount)
      if (view === "camera") {
        t.data[0] = this.sceneClock - e.epochScene
        t.data[1] = live
        t.data[2] = e.weight
        this.device.queue.writeBuffer(t.uniform, 0, t.data.buffer as ArrayBuffer)
      }
      pass.setPipeline(t.pipeline)
      pass.setBindGroup(0, view === "mirror" ? t.mirrorBind : t.bind)
      pass.draw(6, t.slots * live * (TRAIL_SAMPLES - 1) * TRAIL_SUBDIVISIONS)
    }
  }

  /** The user's field mounts, drawn at half resolution for the composite to
   *  upsample. Runs the whole quad — uniform control flow, so effects may use
   *  derivatives freely, which the old inline path had to forbid. */
  private renderFieldPass(encoder: GPUCommandEncoder): void {
    // TWO PREDICATES, deliberately, and they are not interchangeable.
    //
    // MOUNTED decides whether the pass runs, and it must agree exactly with
    // fieldPairUsed — that is what the composite's bind group was built against,
    // at install, and it is not rebuilt per frame. A pass skipped under a
    // binding that still points at its target leaves the last frame it drew
    // sitting there, so an effect faded to nothing would freeze on screen
    // instead of disappearing.
    //
    // DRAWN decides what is drawn into it, and this is where weight is worth
    // something: a field mount is a full-screen quad however little of the frame
    // it ends up touching, so an effect that is scheduled off would otherwise
    // shade every pixel to multiply it out to nothing. The pass still clears —
    // which is what makes the layer transparent rather than stale — and shades
    // nothing.
    const mounted = this.effects.filter((e) => e.fieldPipeline && e.fieldBindGroups)
    if (mounted.length === 0) return
    const drawn = mounted.filter((e) => e.weight > 0)
    // Each effect's own clock, before the pass that reads it. Seconds since
    // THIS effect was installed — so an effect added to a running scene starts
    // at zero and can seed, rather than joining whatever the first one is up to.
    for (const e of drawn) {
      if (!e.fieldClock) continue
      this.fieldClockScratch[0] = this.sceneClock - e.epochScene
      this.fieldClockScratch[1] = e.weight
      this.device.queue.writeBuffer(e.fieldClock, 0, this.fieldClockScratch.buffer as ArrayBuffer)
    }
    // ONE PASS PER RESOLUTION, N draws each, in document order — a pair is
    // cleared once and each effect blends over what the earlier ones left.
    // Alpha is the layer, the same rule the composite already states, and it
    // keeps memory flat however many effects a scene installs.
    //
    // An EMPTY pair is skipped rather than cleared. It used to be cleared on the
    // grounds that the composite reads both pairs every frame and a stale one
    // would keep drawing a removed effect — true of the target, but the composite
    // does not read the target for an empty pair, it reads the 1x1 fallback
    // (fieldLayerView). Clearing and storing an empty full-res rgba16f pair is
    // two 16MB writes a frame to produce the transparent black the fallback
    // already is. Most scenes leave the full-res pair empty, since an effect only
    // lands there by declaring #fullres.
    // The distance field, before anything reads it. Returns immediately when no
    // installed effect names rzCastDistance, which is the common case.
    this.encodeCastDistance(encoder)

    let stamped = false
    for (let i = 0; i < Engine.FIELD_SCALES.length; i++) {
      const bg = this.fieldBgViews[i]
      const fg = this.fieldFgViews[i]
      if (!bg || !fg || !this.fieldPairUsed(i)) continue
      const pass = encoder.beginRenderPass({
        label: `field layer (${Engine.FIELD_SCALES[i] === 1 ? "full" : "half"})`,
        colorAttachments: [
          { view: bg, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
          { view: fg, clearValue: { r: 0, g: 0, b: 0, a: 0 }, loadOp: "clear", storeOp: "store" },
        ],
        // One query pair is reserved for "field", and it goes to the first pair
        // that actually runs — full res when something declared #fullres, half
        // otherwise. Pinning it to i === 0 would have measured a pass that, now
        // that empty pairs are skipped, usually does not happen.
        timestampWrites: stamped ? undefined : this.stamps("field"),
      })
      stamped = true
      for (const e of drawn) {
        if (e.fieldLayer !== i) continue
        pass.setPipeline(e.fieldPipeline!)
        // The grid this effect just wrote — after its parity flip, the one at
        // `parity`. Per effect, so two grids never read each other's frame.
        pass.setBindGroup(0, e.fieldBindGroups![e.grid?.parity ?? 0])
        pass.draw(3)
      }
      pass.end()
    }
  }

  /** The trail bind group holds the depth view, which a resize recreates. */
  private rebindTrails(): void {
    if (!this.depthReadView) return
    for (const e of this.effects) {
    const t = e.trails
    if (!t) continue
    t.bind = this.device.createBindGroup({
      layout: t.layout,
      entries: [
        { binding: 0, resource: { buffer: this.castBuffer } },
        { binding: 1, resource: { buffer: t.uniform } },
        { binding: 2, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 4, resource: { buffer: this.audioBuffer } },
        { binding: 5, resource: { buffer: this.midiBuffer } },
        { binding: 6, resource: { buffer: this.lyricsBuffer } },
      ],
    })
    t.mirrorBind = this.device.createBindGroup({
      layout: t.layout,
      entries: [
        { binding: 0, resource: { buffer: this.castBuffer } },
        { binding: 1, resource: { buffer: t.uniform } },
        { binding: 2, resource: { buffer: this.mirrorCameraBuffer } },
        { binding: 4, resource: { buffer: this.audioBuffer } },
        { binding: 5, resource: { buffer: this.midiBuffer } },
        { binding: 6, resource: { buffer: this.lyricsBuffer } },
      ],
    })
    }
  }

  private releaseParticles(): void {
    for (const e of this.effects) {
      e.particles?.buffer.destroy()
      e.particles?.uniform.destroy()
      e.particles = null
    }
  }

  /**
   * Compile and allocate the effect's persistent grid.
   *
   * The textures are created ZEROED, which is the contract a kernel is written
   * against: rzGridFrame() is 0 on the first step and every value it reads is
   * zero, so seeding is just "if frame is 0, return the initial state".
   */
  private async buildSim(
    /** Already stripped of directives — see compileEffect. */
    wgsl: string,
    /** What the file declared. Read here rather than re-parsed: the source no
     *  longer carries the lines, and two readers is how they drift. */
    d: EffectDirectives,
    anchors: { bone: string; trail: boolean }[],
    /** This effect's local→scene slot map. Passed rather than read off the
     *  engine: the builders run BEFORE the swap, so this.anchorTable still
     *  describes the effect that is still on screen. */
    alias: number[],
    /** The effect's declared dials, spliced and bound above the grid's own. */
    params: EffectParamsBinding,
  ): Promise<{ ok: true; state: EffectGrid } | { ok: false; diagnostics: string[] }> {
    const size = Math.min(d.grid || 256, GRID_MAX)
    const cast = {
      subjects: MAX_EFFECT_SUBJECTS,
      samples: TRAIL_SAMPLES,
      base: MAX_EFFECT_SUBJECTS * 3,
      trailBase: CAST_TRAIL_BASE,
      slots: MAX_EFFECT_ANCHORS,
      trailCount: anchors.filter((x) => x.trail).length,
      alias,
    }
    const code = buildSimShader(wgsl, size, cast, params.wgsl(EFFECT_PARAMS_BINDING_GRID))
    const offset = code.slice(0, code.indexOf(wgsl)).split("\n").length - 1
    this.device.pushErrorScope("validation")
    const module = this.device.createShaderModule({ label: "grid step", code })
    const info = await module.getCompilationInfo()
    const scopeErr = await this.device.popErrorScope()
    const diagnostics = info.messages
      .filter((m) => m.type === "error")
      .map((m) => `${Math.max(0, m.lineNum - offset)}:${m.linePos} ${m.message}`)
    if (diagnostics.length === 0 && scopeErr) diagnostics.push(scopeErr.message)
    if (diagnostics.length) return { ok: false, diagnostics }

    const layout = this.device.createBindGroupLayout({
      label: "grid bind layout",
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
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 8, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        ...(params.buffer
          ? [{ binding: EFFECT_PARAMS_BINDING_GRID, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" as const } }]
          : []),
      ],
    })

    const make = (n: number) =>
      this.device.createTexture({
        label: `grid grid ${n}`,
        size: [size, size],
        format: SIM_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
      })
    const textures: [GPUTexture, GPUTexture] = [make(0), make(1)]
    const read: [GPUTextureView, GPUTextureView] = [textures[0].createView(), textures[1].createView()]
    const uniform = this.device.createBuffer({
      label: "grid uniforms",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.device.pushErrorScope("validation")
    try {
      const pipeline = await this.device.createComputePipelineAsync({
        label: "grid step pipeline",
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
      const bindFor = (i: number) => this.gridBindGroup({ layout, uniform, read, textures, params: params.buffer }, i)
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
          params: params.buffer,
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

  private releaseGrid(): void {
    for (const e of this.effects) {
      e.grid?.textures[0].destroy()
      e.grid?.textures[1].destroy()
      e.grid?.uniform.destroy()
      e.grid = null
    }
  }

  /**
   * Step the grid, before anything reads it.
   *
   * Outside the render pass, like the particle step and for the same reason —
   * and before the field pass, or an effect samples a grid one frame stale.
   */
  private stepSim(encoder: GPUCommandEncoder, deltaTime: number): void {
    for (const e of this.effects) {
    const grid = e.grid
    if (!grid) continue
    grid.data[0] = this.sceneClock - e.epochScene
    // Clamped like the particle step: a backgrounded tab returns with a delta of
    // whole seconds, and one unclamped step of an advection kernel throws the
    // whole grid off its own edge.
    grid.data[1] = Math.min(0.1, Math.max(0, deltaTime))
    grid.data[2] = grid.size
    grid.data[3] = grid.frame++
    this.device.queue.writeBuffer(grid.uniform, 0, grid.data.buffer as ArrayBuffer)
    const cp = encoder.beginComputePass({ label: "grid" })
    cp.setPipeline(grid.pipeline)
    cp.setBindGroup(0, grid.binds[grid.parity])
    const groups = Math.ceil(grid.size / 8)
    cp.dispatchWorkgroups(groups, groups)
    cp.end()
    // The freshly written texture is now the current one.
    grid.parity = 1 - grid.parity
    }
  }

  /** Which mounts the installed effect declared. Both false when none is set. */
  getEffectMounts(): { background: boolean; foreground: boolean } {
    return { background: this.effect?.hasBackground ?? false, foreground: this.effect?.hasForeground ?? false }
  }

  /**
   * Set one parameter on one INSTANCE.
   *
   * By index, because the scene holds a list and the same effect may appear in
   * it twice with different values — which is the whole point of an instance
   * and was impossible while this addressed `this.effect`, a singular left over
   * from when a scene could wear exactly one.
   *
   * A write, not a recompile: parameters live in their own uniform buffer, so
   * dragging a slider costs a 16-byte upload rather than a shader build.
   */
  setEffectParam(index: number, name: string, value: EffectParamValue): void {
    const fx = this.effects[index]
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

  /**
   * How much of one instance is showing, 0..1.
   *
   * The third of the three things an instance has — parameters, weight, time —
   * and the one a scheduler drives. Weight is not a parameter: a parameter is
   * whatever the author decided to expose and means only what their source
   * makes it mean, while weight means the same thing for every effect ever
   * written, including one whose author never heard of it. That is why it is
   * applied by engine-generated code at each mount's output rather than handed
   * to the source as a uniform to respect.
   *
   * At 0 nothing is drawn: no field quad, no particle draw, no ribbon, no light
   * dispatch. A scheduled effect outside its window costs its simulation and
   * nothing else — and a particle effect keeps simulating on purpose, so that
   * fading one back in continues rather than rewinds.
   *
   * Instant, and free: a float in a uniform every mount already uploads once a
   * frame. Nothing recompiles, so this is safe to drive per frame from a
   * timeline.
   */
  setEffectInfluence(index: number, influence: number): void {
    const fx = this.effects[index]
    if (!fx) return
    // Clamped rather than trusted: above 1 the field's own clamp would swallow
    // it while an additive particle would happily keep getting brighter, so the
    // same number would mean two things.
    fx.influence = Math.min(1, Math.max(0, influence))
  }

  getEffectInfluence(index: number): number {
    return this.effects[index]?.influence ?? 0
  }

  /**
   * Schedule one instance: when it is alive, and how it enters and leaves.
   *
   * Null is the unscheduled case — on for the whole scene, on the scene's own
   * clock — and is what an effect starts as.
   *
   * The engine evaluates this every frame rather than taking a weight from a
   * caller, because every loop that renders would otherwise have to remember to
   * drive it. The offline export loop already carries a scar about exactly that
   * shape of bug. Evaluating where the scene clock advances means playback and
   * export cannot disagree, and neither can forget.
   *
   * A caller that wants to drive an effect from something OTHER than the scene
   * clock — an animation's progress, a skill firing — leaves this null and
   * writes setEffectInfluence and setEffectTime itself, per frame. Both paths
   * exist on purpose; this one is what a timeline wants.
   */
  setEffectSchedule(index: number, windows: readonly EffectWindow[] | null): void {
    const fx = this.effects[index]
    if (!fx) return
    fx.window = windows && windows.length ? windows : null
  }

  getEffectSchedule(index: number): readonly EffectWindow[] | null {
    return this.effects[index]?.window ?? null
  }

  /**
   * Every scheduled effect, at the current scene clock.
   *
   * Called once a frame, BEFORE anything reads a weight or a clock. An effect
   * with no window keeps whatever a caller last set, which is what makes the
   * manual path above work — evaluating it would fight the caller for the field
   * every frame.
   */
  private evaluateEffectSchedules(): void {
    // Read ONCE: it walks the cast, and every effect wants the same answer.
    const transport = this.transportTime()
    for (const fx of this.effects) {
      if (!fx.window || fx.window.length === 0) {
        fx.weight = fx.influence
        continue
      }
      const at = effectState(fx.window, fx.influence, transport)
      fx.weight = at.weight
      // Its own clock, expressed the way the mounts read it. Every mount
      // derives time from the epoch against sceneClock, so this one write moves
      // the field, the particles, the ribbons, lightEmit and the grid together
      // — and hands them the STRIP's local time while they keep running on the
      // smooth monotonic clock a particle integrator needs.
      fx.epochScene = this.sceneClock - at.time
    }
  }

  /**
   * Move one instance's own clock to a given second.
   *
   * Everything an effect can animate is derived from its epoch — the field
   * clock, the particle and ribbon clocks, lightEmit's time argument, the grid's
   * frame counter — so moving the epoch moves all of them together and there is
   * no mount that can be left reading last frame's time.
   *
   * This is what lets an effect be SCHEDULED rather than merely switched on: an
   * instance that enters at bar 33 is handed a time that starts at zero there,
   * so it plays its own opening instead of joining whatever the scene clock had
   * reached. Feeding it the transport's time instead gives the other reading —
   * an effect that runs in lockstep with the music — and both are one call.
   */
  setEffectTime(index: number, time: number): void {
    const fx = this.effects[index]
    if (!fx) return
    fx.epochScene = this.sceneClock - time
  }


  getEffectTime(index: number): number {
    const fx = this.effects[index]
    return fx ? this.sceneClock - fx.epochScene : 0
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
      // Neither a stage nor a plane is a performer, so neither is a subject an
      // effect can follow.
      if (!inst.model.visible || inst.isStage || inst.isPlane || inst.isProp) continue
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
    // viewZ = projB / (z − projA), the inverse of the projection's z mapping —
    // and projA/projB ARE m[10] and m[14], so read them off the matrix rather
    // than re-deriving them from near/far. Re-derivation is what would silently
    // rot the day the projection changed convention, which is exactly what just
    // happened: the old pair encoded the OpenGL mapping and would have inverted
    // a reversed-Z buffer into nonsense.
    const proj = this.camera.getProjectionMatrix().values
    u[8] = proj[10]
    u[9] = proj[14]
    // THE FAR PLANE ITSELF, written rather than left to be re-derived. An effect
    // that resamples has to know whether the scene was drawn at a given pixel,
    // and "nothing was drawn" reads back as exactly this distance. Recovering it
    // from the pair above means inverting a cleared depth, which is a sign trap
    // that differs by projection convention and fails silently — as an effect
    // that draws nothing at all, or one that paints the sky. The camera knows
    // the number; this hands it over.
    u[10] = this.camera.far
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
    // Float depth WITH stencil, which the eye/hair stencil interplay needs. It is
    // an optional WebGPU feature, so this is a request, not an assumption — and
    // it is the whole reason reversed-Z is worth doing: reversing a UNORM buffer
    // mirrors the precision curve without improving it, while reversing a float
    // one cancels 1/z crowding against float's own crowding near zero and buys
    // back the near plane a close-up camera needs.
    const wantDepth32: GPUFeatureName = "depth32float-stencil8"
    const hasDepth32 = adapter.features.has(wantDepth32)
    // GPU pass timings. Optional, and asked for on every device rather than
    // behind a debug flag: this is the regression guard for a restructure whose
    // whole claim is that it does not cost anything, and a guard nobody runs
    // guards nothing.
    const wantTimestamp: GPUFeatureName = "timestamp-query"
    const hasTimestamp = adapter.features.has(wantTimestamp)
    const device = await adapter.requestDevice({
      requiredFeatures: [
        ...(hasRg11b10 ? [wantFeature] : []),
        ...(hasDepth32 ? [wantDepth32] : []),
        ...(hasTimestamp ? [wantTimestamp] : []),
      ],
    })
    if (!device) {
      throw new Error("WebGPU is not supported in this browser.")
    }
    this.device = device
    // Every validation error this device ever raises, kept.
    //
    // WebGPU does not throw for a bad pipeline: createRenderPipeline hands back
    // an object that is already invalid, and the complaint arrives here instead
    // — or nowhere, if nobody is listening. Nobody was. That is why a device
    // that disagrees with this engine has, until now, had no way to say so: the
    // pipeline is built, setPipeline poisons the pass that uses it, and the
    // symptom reaches the user as geometry that is simply absent, with a clean
    // console. A browser is not obliged to agree with Dawn about what is legal,
    // and the two places this engine knowingly leans on Dawn's reading are both
    // in the scene pass (see scene-contract's writeMask-0 note).
    //
    // Bounded, and not on the console by default: a pass that fails validation
    // fails it again every frame, so an unbounded log is a memory leak with a
    // frame counter and an unconditional console.error is a browser tab that
    // stops responding. First N distinct messages, counted thereafter.
    device.addEventListener("uncapturederror", (e) => {
      const message = (e as GPUUncapturedErrorEvent).error.message
      this.noteGpuError(message)
    })
    if (hasRg11b10) this.hdrFormat = "rg11b10ufloat"
    // The override has the last word, including over a device that would have
    // been left on the fallback anyway — asking for the format you are already
    // getting is a no-op, not a contradiction. See HDR_FORMAT_OVERRIDE.
    if (Engine.HDR_FORMAT_OVERRIDE) {
      this.hdrFormat = Engine.HDR_FORMAT_OVERRIDE
      // Only when forced. The probed answer is the normal one and does not need
      // announcing on every boot; a forced one is a state someone set and will
      // want confirmed, and is the state they will forget they left on.
      console.info(`[reze] HDR target forced to ${this.hdrFormat}`)
    }
    // The id attachment, if this device will multisample a uint texture at the
    // pass's sample count. Probed by ASKING — creating one inside an error
    // scope — rather than by reading a feature flag, because there is no
    // feature to read: multisampled uint support is a limit of the
    // implementation, not an extension. A device that refuses leaves ids off
    // and every shader is assembled without the output, which is why this runs
    // before any pipeline or module is built.
    //
    // Gated by MRT_IDS as well, which is the master switch: the probe says
    // CAN, and that says SHOULD.
    setMrtIds(Engine.MRT_IDS && (await this.probeMultisampledIds()))
    if (hasTimestamp) {
      this.timestampQuerySet = device.createQuerySet({
        label: "pass timings",
        type: "timestamp",
        count: Engine.TIMED_PASSES.length * 2,
      })
      const bytes = Engine.TIMED_PASSES.length * 2 * 8 // one u64 per query
      this.timestampResolve = device.createBuffer({
        label: "pass timings (resolve)",
        size: bytes,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      })
      this.timestampRead = device.createBuffer({
        label: "pass timings (readback)",
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      })
    }
    if (hasDepth32) {
      this.depthFormat = "depth32float-stencil8"
      this.reversedZ = true
    }
    this.camera.reversedZ = this.reversedZ

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
  /** The frost tile the ground samples instead of evaluating fbm per pixel. */
  private groundNoiseTexture!: GPUTexture
  private groundNoiseView!: GPUTextureView

  /**
   * Bake the ground's frost noise once — the same fbm the shader used to run
   * per pixel, rendered to a seamless 1024² r8unorm tile at init.
   *
   * Why this exists is measured, not argued: on WebKit the ground's whole cost
   * was this evaluation (see the note at the sample site in ground.ts). The
   * bake is one fullscreen pass at init — under a millisecond, once — and the
   * per-pixel cost becomes a single level-0 texture read.
   */
  private bakeGroundNoise() {
    this.groundNoiseTexture = this.device.createTexture({
      label: "ground frost noise (baked)",
      size: [GROUND_NOISE_SIZE, GROUND_NOISE_SIZE],
      format: "r8unorm",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.groundNoiseView = this.groundNoiseTexture.createView()
    const module = this.device.createShaderModule({ label: "ground noise bake", code: GROUND_NOISE_BAKE_WGSL })
    const pipeline = this.device.createRenderPipeline({
      label: "ground noise bake",
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: "fs", targets: [{ format: "r8unorm" }] },
      primitive: { topology: "triangle-list" },
    })
    const encoder = this.device.createCommandEncoder({ label: "ground noise bake" })
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: this.groundNoiseView, loadOp: "clear", storeOp: "store" }],
    })
    pass.setPipeline(pipeline)
    pass.draw(3)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

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
      label: "grid grid sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    })
    this.simFallbackView = this.device
      .createTexture({
        label: "grid grid fallback (1x1 zero)",
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

    // Header + key map even when empty: every rzNote*/rzKey* accessor reads the
    // header first, so an effect written against a score still compiles and runs
    // in a scene that has none — it simply sees no notes.
    this.midiFallbackBuffer = this.device.createBuffer({
      label: "score fallback (no notes)",
      size: (MIDI_HEADER + MIDI_KEYS) * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.midiBuffer = this.midiFallbackBuffer

    // One per resolution — see FIELD_SCALES.
    this.fieldUniformBuffers = Engine.FIELD_SCALES.map((scale) =>
      this.device.createBuffer({
        label: `field layer uniforms (${scale === 1 ? "full" : "half"})`,
        size: 16,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    )
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
        // The score. 19 rather than a low number because both this layout and
        // the field layer's already speak for everything below it.
        { binding: 19, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 24, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // The lyric line atlas, for rzLyricText.
        { binding: 25, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 14, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        // This effect's own clock — see the field shader's note on why it is
        // not viewU[6].x.
        { binding: 22, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        // The id attachment, for rzObjectAt/rzMaterialAt. Declared only when it
        // exists: the alternative is a multisampled uint fallback texture whose
        // only job is to be bound, and the layout is built after the probe so
        // both halves agree by construction.
        ...(mrtIdsEnabled()
          ? [
              {
                binding: 23,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: "uint" as const, viewDimension: "2d" as const, multisampled: true },
              },
            ]
          : []),
        { binding: 17, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
        { binding: 18, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
        // Distance to the cast. Always in the layout — the accessor is always
        // compiled, and a 1x1 stands in when the flood is not running, so an
        // author never has to guard the name.
        { binding: 26, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
        // The finished scene, and a sampler of its own — see scene-tap.ts for
        // why it may not borrow the one at 18.
        { binding: 27, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 28, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        // THE VIEW TRANSFORM'S OWN RESOURCES. viewTransform() lives in the
        // header both this module and the composite share, but its lookups did
        // not: an effect calling it compiled and then failed at pipeline
        // creation with "binding doesn't exist", naming a binding no effect
        // author ever wrote. Sharing the code meant sharing these.
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 10, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "3d" } },
        // And the scene's COVERAGE and bloom, without which the tap cannot
        // reconstruct a pixel: the HDR target is premultiplied, so colour alone
        // reads a half-transparent ground as a dark opaque one.
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    })
    this.fieldPipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.fieldBindGroupLayout],
    })

    // ── The distance-to-cast field's three pipelines ──
    //
    // Built once, whether or not anything reads the field: they cost a shader
    // module each and nothing per frame, and building them lazily would put a
    // pipeline compile in the frame where an author first types the name.
    {
      const seedModule = this.device.createShaderModule({ label: "cast distance seed", code: buildCastSeedShader(Engine.MULTISAMPLE_COUNT) })
      const stepModule = this.device.createShaderModule({ label: "cast distance step", code: buildCastStepShader() })
      const resolveModule = this.device.createShaderModule({ label: "cast distance resolve", code: buildCastResolveShader() })
      this.castSeedPipeline = this.device.createRenderPipeline({
        label: "cast distance seed",
        layout: "auto",
        vertex: { module: seedModule, entryPoint: "vs" },
        fragment: {
          module: seedModule,
          entryPoint: "fs",
          targets: [{ format: CAST_SEED_FORMAT }, { format: CAST_COVERAGE_FORMAT }],
        },
        primitive: { topology: "triangle-list" },
      })
      this.castStepPipeline = this.device.createRenderPipeline({
        label: "cast distance step",
        layout: "auto",
        vertex: { module: stepModule, entryPoint: "vs" },
        fragment: { module: stepModule, entryPoint: "fs", targets: [{ format: CAST_SEED_FORMAT }] },
        primitive: { topology: "triangle-list" },
      })
      this.castResolvePipeline = this.device.createRenderPipeline({
        label: "cast distance resolve",
        layout: "auto",
        vertex: { module: resolveModule, entryPoint: "vs" },
        fragment: { module: resolveModule, entryPoint: "fs", targets: [{ format: CAST_DIST_FORMAT }] },
        primitive: { topology: "triangle-list" },
      })
      // 65504 is the largest half float. Bound wherever the field is not
      // running, so rzCastDistance answers "unreachably far" and an effect keyed
      // on it draws nothing at all.
      this.castDistFallback = this.device.createTexture({
        label: "cast distance fallback (1x1, far)",
        size: [1, 1],
        format: CAST_DIST_FORMAT,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      this.device.queue.writeTexture(
        { texture: this.castDistFallback },
        new Uint16Array([0x7bff]),
        { bytesPerRow: 2 },
        { width: 1, height: 1 },
      )
      this.castDistFallbackView = this.castDistFallback.createView()
    }

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
    //
    // Formats, blends and write masks now come from scene-contract.ts, which is
    // the one author of what this pass's attachments are. The aux target carries
    // (bloom mask, alpha) and blends alpha-over so its .g accumulates coverage:
    // materials write vec2f(mask, 1.0), ground writes vec2f(0.0, 1.0), and with
    // src.a coming from the fragment's own colour.a the equation gives
    //   out.g = 1·src.a + dst.g·(1-src.a)  →  the premultiplied over operator.
    // .r is weighted by src.a as well, which is right for a bloom gate: an
    // opaque pixel contributes its whole mask, a translucent one its share.
    const sceneTargets = sceneTargetsFor("material", this.sceneFormats)
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
        // The positional lights. Always bound, empty or not, so every material
        // pipeline shares one layout whether or not the scene has any.
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // The far cascade's shadow map. 8 stays free; 9 is the BRDF LUT.
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
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
        format: this.depthFormat,
        depthWriteEnabled: true,
        depthCompare: this.depthAhead,
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
        format: this.depthFormat,
        depthWriteEnabled: false,
        depthCompare: this.depthAhead,
      },
    })
    // Depth-only prepass for transparent draws (see depth-prepass.ts): writes the
    // fabric's depth AFTER its color blended, so outlines drawn later are
    // occluded behind it. Color targets kept for pass compatibility, writeMask 0.
    const prepassModule = this.device.createShaderModule({
      label: "transparent depth prepass",
      code: transparentDepthPrepassWgsl(),
    })
    const prepassDesc = {
      layout: mainPipelineLayout,
      vertex: { module: prepassModule, entryPoint: "vs", buffers: fullVertexBuffers as GPUVertexBufferLayout[] },
      primitive: { cullMode: "none" as GPUCullMode },
      multisample: { count: Engine.MULTISAMPLE_COUNT },
      depthStencil: {
        format: this.depthFormat,
        depthWriteEnabled: true,
        depthCompare: this.depthAhead,
      },
    }
    this.depthPrepassPipeline = this.device.createRenderPipeline({
      label: "opaque depth prepass",
      ...prepassDesc,
      fragment: {
        module: prepassModule,
        entryPoint: "fs",
        targets: sceneTargetsFor("depth-prepass", this.sceneFormats),
      },
    })
    // The SOLID prime: same module, cutoff forced to exactly 1.0. Only texels
    // whose blend ignores the destination may pre-claim depth in the
    // transparent phase — see the override's note in depth-prepass.ts.
    this.solidPrepassPipeline = this.device.createRenderPipeline({
      label: "transparent solid prepass",
      ...prepassDesc,
      fragment: {
        module: prepassModule,
        entryPoint: "fs",
        constants: { CUTOFF: 1.0 },
        targets: sceneTargetsFor("depth-prepass", this.sceneFormats),
      },
    })
    // The HAIR prime: solid texels only, and stencil-fenced off the eye
    // silhouette. It records after the non-hair opaque draws, so the eye has
    // already written its stencil — not-equal here is what keeps the primed
    // hair depth from ever claiming the pixels the see-through-hair pass needs
    // the eye to survive on. (Bundle draws use the PASS's stencil reference;
    // only pipeline/bind/vertex state resets across executeBundles.)
    this.hairPrimePipeline = this.device.createRenderPipeline({
      label: "hair depth prime",
      ...prepassDesc,
      depthStencil: {
        ...prepassDesc.depthStencil,
        stencilFront: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0,
      },
      fragment: {
        module: prepassModule,
        entryPoint: "fs",
        constants: { CUTOFF: 1.0 },
        targets: sceneTargetsFor("depth-prepass", this.sceneFormats),
      },
    })

    this.shadowLightVPBuffer = this.device.createBuffer({
      size: 64 * SHADOW_CASCADES.length,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.shadowCascadeVPBuffers = SHADOW_CASCADES.map((_, i) =>
      this.device.createBuffer({
        label: `shadow cascade ${i} view-projection`,
        size: 64,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      }),
    )
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
        format: Engine.SHADOW_DEPTH_FORMAT,
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        // The shadow map keeps the NON-reversed convention (orthographicLh maps
        // [0,1] with far = 1, and this pass never flipped) — so this bias must
        // NOT follow reversedZ. It is the camera-pass biases that flip.
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
    // One map per cascade, each at its own resolution — the near one crisp,
    // the far one wide. Same format so one pipeline records into both.
    this.shadowMapTextures = SHADOW_CASCADES.map((c, i) =>
      this.device.createTexture({
        label: `shadow map cascade ${i}`,
        size: [c.mapSize, c.mapSize],
        format: Engine.SHADOW_DEPTH_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
    )
    this.shadowMapDepthViews = this.shadowMapTextures.map((t) => t.createView())

    // One-shot bake of Blender EEVEE's combined BRDF LUT (DFG + LTC packed rgba8unorm).
    this.bakeBrdfLut()
    this.bakeGroundNoise()
    this.agxFallbackTexture = this.device.createTexture({
      label: "AgX LUT fallback",
      size: [1, 1, 1],
      dimension: "3d",
      format: "rgb10a2unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    void this.loadAgxLut()
    this.bakeFilmicLut()

    // BEFORE the bind group below, which binds it. Full size from the start:
    // every material pipeline binds this, so sizing it to the light count would
    // mean rebuilding bind groups whenever a scene gained a lamp. Zero-filled,
    // and float 0 is a count of 0.
    this.lightsData = new Float32Array(LIGHTS_FLOATS)
    this.lightsBuffer = this.device.createBuffer({
      label: "positional lights",
      size: this.lightsData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.lightsBuffer, 0, this.lightsData)
    // Zero-filled = zero lines, which every accessor answers gracefully.
    this.lyricsBuffer = this.device.createBuffer({
      label: "lyrics",
      size: LYRICS_FLOATS * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    // A placeholder until a track's lines arrive: one channel is all a glyph
    // mask is, and a zero texel reads as "no text" everywhere.
    this.lyricsTexture = this.device.createTexture({
      label: "lyric line atlas (placeholder)",
      size: [1, 1],
      format: "r8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    })
    this.lyricsTextureView = this.lyricsTexture.createView()

    // Now that shadow resources exist, create the main per-frame bind group
    this.perFrameBindGroup = this.device.createBindGroup({
      label: "main per-frame bind group",
      layout: this.mainPerFrameBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.lightUniformBuffer } },
        { binding: 2, resource: this.materialSampler },
        { binding: 3, resource: this.shadowMapDepthViews[0] },
        { binding: 4, resource: this.shadowComparisonSampler },
        { binding: 5, resource: { buffer: this.shadowLightVPBuffer } },
        { binding: 6, resource: { buffer: this.lightsBuffer } },
        { binding: 7, resource: this.shadowMapDepthViews[SHADOW_CASCADES.length - 1] },
        { binding: 9, resource: this.brdfLutView },
      ],
    })
    // The mirror's, identical but for the camera — same lights, same shadow
    // maps, because a reflection is the same scene lit the same way, seen from
    // a reflected eye.
    this.mirrorPerFrameBindGroup = this.device.createBindGroup({
      label: "mirror per-frame bind group",
      layout: this.mainPerFrameBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.mirrorCameraBuffer } },
        { binding: 1, resource: { buffer: this.lightUniformBuffer } },
        { binding: 2, resource: this.materialSampler },
        { binding: 3, resource: this.shadowMapDepthViews[0] },
        { binding: 4, resource: this.shadowComparisonSampler },
        { binding: 5, resource: { buffer: this.shadowLightVPBuffer } },
        { binding: 6, resource: { buffer: this.lightsBuffer } },
        { binding: 7, resource: this.shadowMapDepthViews[SHADOW_CASCADES.length - 1] },
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
        // Same lights the materials read. A lamp that lit the cast and not the
        // floor under her would read as a sticker.
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth" } },
        // The floor mirror: the mirror camera's view-projection, the reflection
        // resolve, an ordinary sampler beside the comparison one, and the
        // mirror pass's own depth for the depth-proportional blur.
        { binding: 8, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 9, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
        { binding: 10, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
        { binding: 11, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "depth", multisampled: true } },
        // The baked frost tile — see bakeGroundNoise. Sampled with binding 10's
        // repeat sampler, so it brings no sampler of its own.
        { binding: 12, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } },
      ],
    })
    this.groundShadowPipelineDesc = {
      label: "ground shadow pipeline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [this.groundShadowBindGroupLayout] }),
      // Slot 0 only — the ground has no skinning, and declaring the full
      // 3-slot layout while renderGround binds one buffer is a WebGPU
      // validation error that invalidates the whole command buffer.
      vertexBuffers: [fullVertexBuffers[0]],
      fragmentTargets: sceneTargetsFor("ground", this.sceneFormats),
      cullMode: "back",
      depthStencil: { format: this.depthFormat, depthWriteEnabled: true, depthCompare: this.depthAhead },
    }
    this.groundShadowPipeline = this.buildGroundPipeline(false)

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
      code: outlineShaderWgsl(),
    })

    this.outlinePipeline = this.createRenderPipeline({
      label: "outline pipeline",
      layout: outlinePipelineLayout,
      shaderModule: outlineShaderModule,
      vertexBuffers: outlineVertexBuffers,
      fragmentTargets: sceneTargetsFor("outline", this.sceneFormats),
      cullMode: "back",
      depthStencil: {
        format: this.depthFormat,
        // babylon-mmd draws outlines WITH depth write (its _afterRenderingMesh
        // forces setDepthWrite(true)); the constant bias below still makes
        // hulls lose depth ties against their own near-coplanar geometry.
        depthWriteEnabled: true,
        depthCompare: this.depthAhead,
        // CONFIRMED fix (bisected live via setOutlineEnabled): hull fragments
        // carry their surface's exact depth, so against this model's paired
        // near-coplanar skirt layers the hulls WON depth ties in patches —
        // the black shapes on the dress. A small constant bias makes hulls lose
        // every tie; silhouette rims compare against the far background and are
        // unaffected. No slope term — slope explodes at silhouettes and would
        // erase the rims themselves (previous regression).
        //
        // SIGNED BY CONVENTION: bias adds to the depth VALUE, and reversed-Z
        // inverts what a larger value means — +4 there makes hulls WIN the ties
        // this exists to lose, which is the dress regression back again. The
        // compare op flips via depthAhead; the bias has to flip by hand.
        depthBias: this.reversedZ ? -4 : 4,
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

    // ─── Editor overlays (instanced wireframe primitives) ────────────
    this.setupOverlay()

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
        // The audio analysis, for rzAudio*. Silence fallback when the scene has
        // no track.
        { binding: 13, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // The score. 19 rather than a low number because both this layout and
        // the field layer's already speak for everything below it.
        { binding: 19, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        { binding: 24, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "read-only-storage" } },
        // The field layer's two halves. Fallback-bound when no field effect runs.
        { binding: 15, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 16, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        // The half-resolution pair. Always bound, empty or not — see the
        // composite's own note on why both are read every frame.
        { binding: 20, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 21, visibility: GPUShaderStage.FRAGMENT, texture: {} },
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
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: roStorage },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
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
        depthCompare: this.depthAhead,
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
    // No device, nothing to size.
    //
    // Three callers reach this, and two of them can arrive before init() has a
    // device or after teardown has released one: setRenderSize is PUBLIC and
    // unordered with respect to init, and the ResizeObserver keeps firing across
    // a hot reload while the replaced engine is still mounted. Both landed on
    // `this.device.createTexture` and threw — which is why this only shows up
    // during development, and why 0.43 never saw it: setRenderSize did not exist
    // to be called early.
    //
    // Returning is correct rather than merely quiet. fixedRenderSize has already
    // been recorded by the time we get here, and init() ends with its own
    // handleResize — so the size asked for before the device existed is applied
    // in full the moment there is something to apply it to.
    if (!this.device) return
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

      // RELEASED BEFORE REPLACING. WebGPU does not free a texture when the last
      // JS reference drops — it waits for GC, which has no idea a 130 MB MSAA
      // target is riding on a small object. The id and mirror targets below
      // already did this; the seven main scene targets did not, so every resize
      // orphaned the whole set.
      //
      // A video export is what made it hurt: it resizes UP to the output size
      // and back DOWN on the way out, so one 4K render orphaned roughly a third
      // of a gigabyte, twice. A few exports in one session and the tab is
      // starved — slow, then slow to reload.
      //
      // destroy() is safe against work already submitted: the implementation
      // defers the free until the GPU is done. What it forbids is USING a
      // destroyed texture in a new command, and every view is rebuilt below.
      this.multisampleTexture?.destroy()
      this.multisampleTexture = this.device.createTexture({
        label: "multisample HDR render target",
        size: [width, height],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })

      this.hdrResolveTexture?.destroy()
      this.hdrResolveTexture = this.device.createTexture({
        label: "HDR resolve target",
        size: [width, height],
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })

      // The id attachment. Multisampled like the rest of the pass, and with NO
      // resolve texture beside it: resolving averages, and the average of two
      // ids is a third id naming something that was never drawn. Consumers read
      // sample 0 with textureLoad, the way linearDepth already does.
      this.idTexture?.destroy()
      this.idTexture = null
      this.idView = null
      // The debug bind group holds the OLD view. Dropped here so it is rebuilt
      // against the new one — keeping it would sample a destroyed texture at
      // the first resize with the debug view open.
      this.idDebugBindGroup = null
      if (mrtIdsEnabled()) {
        this.idTexture = this.device.createTexture({
          label: "object id",
          size: [width, height],
          sampleCount: Engine.MULTISAMPLE_COUNT,
          format: SCENE_ID_FORMAT,
          usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        })
        this.idView = this.idTexture.createView()
      }

      // The field layer — half resolution by default, full for #fullres effects.
      // AFTER the id attachment above: createFieldTargets rebuilds the cast
      // distance flood, whose seed pass binds idView. Built before it, the seed
      // group names the id texture the lines above have just destroyed, and
      // every frame that encodes the flood is rejected whole — a black canvas
      // for the first resize after a silhouette effect is installed, which is
      // what a video export always is.
      this.fieldFullW = width
      this.fieldFullH = height
      this.createFieldTargets()

      // Bloom-mask MRT attachments — same dims + MSAA as HDR so they share the render pass.
      // MS buffer gets resolved into maskResolveTexture, which the bloom blit pass samples.
      this.multisampleMaskTexture?.destroy()
      this.multisampleMaskTexture = this.device.createTexture({
        label: "multisample bloom mask",
        size: [width, height],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: Engine.BLOOM_MASK_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.maskResolveTexture?.destroy()
      this.maskResolveTexture = this.device.createTexture({
        label: "bloom mask resolve",
        size: [width, height],
        format: Engine.BLOOM_MASK_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.maskResolveView = this.maskResolveTexture.createView()

      // The floor mirror's targets — FULL resolution, the same attachment
      // contract and sample count as the scene pass, which is what lets the
      // mirror bundles reuse every scene pipeline unchanged. Half res was the
      // first cut and read soft at mirror 1 (user call, 2026-08-16); the blur
      // dial makes softness a CHOICE now, so the base target is sharp. Aux and
      // id are along for pipeline compatibility and discarded; only the HDR
      // colour resolves to something samplable.
      const mw = width
      const mh = height
      this.mirrorColorMsTexture?.destroy()
      this.mirrorColorTexture?.destroy()
      this.mirrorMaskMsTexture?.destroy()
      this.mirrorIdMsTexture?.destroy()
      this.mirrorDepthTexture?.destroy()
      this.reflectionDebugBindGroup = null
      this.mirrorColorMsTexture = this.device.createTexture({
        label: "mirror HDR (msaa)",
        size: [mw, mh],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
      // Mipped: the blur dial samples a higher level, and the chain below
      // fills the levels with the bloom pyramid's own 13-tap downsample. Mip 0
      // is the resolve target; the ground's view spans them all, and a blur of
      // exactly zero reads only level 0, which is why an unfilled chain is
      // safe for scenes that never touch the dial.
      this.mirrorMipCount = Math.max(1, Math.min(6, Math.floor(Math.log2(Math.min(mw, mh))) - 2))
      this.mirrorColorTexture = this.device.createTexture({
        label: "mirror HDR resolve",
        size: [mw, mh],
        mipLevelCount: this.mirrorMipCount,
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.mirrorColorView = this.mirrorColorTexture.createView()
      this.mirrorMipViews = []
      for (let i = 0; i < this.mirrorMipCount; i++) {
        this.mirrorMipViews.push(this.mirrorColorTexture.createView({ baseMipLevel: i, mipLevelCount: 1 }))
      }
      this.mirrorBlurBindGroups = null
      this.mirrorMaskMsTexture = this.device.createTexture({
        label: "mirror aux (msaa, discarded)",
        size: [mw, mh],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: Engine.BLOOM_MASK_FORMAT,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.mirrorIdMsTexture = mrtIdsEnabled()
        ? this.device.createTexture({
            label: "mirror id (msaa, discarded)",
            size: [mw, mh],
            sampleCount: Engine.MULTISAMPLE_COUNT,
            format: SCENE_ID_FORMAT,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
          })
        : null
      this.mirrorDepthTexture = this.device.createTexture({
        label: "mirror depth",
        size: [mw, mh],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: this.depthFormat,
        // TEXTURE_BINDING: the ground reads it back for depth-proportional
        // blur — how far behind the mirror surface the reflection sits.
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.mirrorDepthReadView = this.mirrorDepthTexture.createView({ aspect: "depth-only" })
      const mirrorColor: GPURenderPassColorAttachment = {
        view: this.mirrorColorMsTexture.createView(),
        resolveTarget: this.mirrorMipViews[0],
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "discard",
      }
      const mirrorMask: GPURenderPassColorAttachment = {
        view: this.mirrorMaskMsTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "discard",
      }
      const mirrorId: GPURenderPassColorAttachment | null = this.mirrorIdMsTexture
        ? {
            view: this.mirrorIdMsTexture.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "discard",
          }
        : null
      this.mirrorPassDescriptor = {
        label: "mirror pass",
        colorAttachments: mirrorId ? [mirrorColor, mirrorMask, mirrorId] : [mirrorColor, mirrorMask],
        depthStencilAttachment: {
          view: this.mirrorDepthTexture.createView(),
          depthClearValue: this.depthClear,
          depthLoadOp: "clear",
          // Stored, not discarded: the ground's blur reads it. Stencil stays
          // discarded — nothing reads stencil back.
          depthStoreOp: "store",
          stencilClearValue: 0,
          stencilLoadOp: "clear",
          stencilStoreOp: "discard",
        },
      }
      // The ground binds the reflection resolve; rebind it against the new one.
      this.buildGroundBindGroup()

      // Bloom pyramid: mip 0 is half-res, each subsequent mip halves again.
      // Mip count chosen so the coarsest mip is ≥4 px on the short side, capped at BLOOM_MAX_LEVELS.
      const bw = Math.max(1, Math.floor(width / 2))
      const bh = Math.max(1, Math.floor(height / 2))
      const shortSide = Math.max(1, Math.min(bw, bh))
      this.bloomMipCount = Math.max(1, Math.min(Engine.BLOOM_MAX_LEVELS, Math.floor(Math.log2(shortSide)) - 1))
      this.bloomDownTexture?.destroy()
      this.bloomDownTexture = this.device.createTexture({
        label: "bloom down pyramid",
        size: [bw, bh],
        mipLevelCount: this.bloomMipCount,
        format: this.hdrFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })
      this.bloomUpTexture?.destroy()
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

      this.depthTexture?.destroy()
      this.depthTexture = this.device.createTexture({
        label: "depth texture",
        size: [width, height],
        sampleCount: Engine.MULTISAMPLE_COUNT,
        format: this.depthFormat,
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

      // Cleared to 0, which is the reserved "nothing" id — so a pixel nothing
      // drew reports nothing rather than whatever the last frame left. Stored,
      // since the whole point is to be read after the pass.
      const idAttachment: GPURenderPassColorAttachment | null = this.idView
        ? {
            view: this.idView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          }
        : null

      this.renderPassDescriptor = {
        label: "renderPass",
        timestampWrites: this.stamps("scene"),
        colorAttachments: idAttachment
          ? [colorAttachment, maskAttachment, idAttachment]
          : [colorAttachment, maskAttachment],
        depthStencilAttachment: {
          view: depthTextureView,
          depthClearValue: this.depthClear,
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
        timestampWrites: this.stamps("composite"),
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

  // Builds the overlay pipeline and the one vertex buffer holding every unit
  // wireframe. The instance buffer is grown on demand in renderOverlayPass — a
  // scene with no overlays on never allocates one.
  private setupOverlay() {
    this.overlayGeometry = buildOverlayShapes()
    const verts = this.overlayGeometry.vertices
    this.overlayVertexBuffer = this.device.createBuffer({
      label: "overlay vertex buffer",
      size: verts.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.overlayVertexBuffer, 0, verts)

    this.overlayUniformBuffer = this.device.createBuffer({
      label: "overlay uniforms",
      size: 16, // vec2 viewport + dash period + pad
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const bgLayout = this.device.createBindGroupLayout({
      label: "overlay group 0 layout (camera + overlay)",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    })
    const shader = this.device.createShaderModule({ label: "overlay shader", code: OVERLAY_SHADER_WGSL })
    const overlayPipelineDescriptor = {
      label: "overlay pipeline",
      layout: this.device.createPipelineLayout({
        label: "overlay pipeline layout",
        bindGroupLayouts: [bgLayout],
      }),
      vertex: {
        module: shader,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: OVERLAY_VERTEX_FLOATS * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat }, // pos
              { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat }, // dir
              { shaderLocation: 2, offset: 6 * 4, format: "float32x2" as GPUVertexFormat }, // caps
              { shaderLocation: 3, offset: 8 * 4, format: "float32" as GPUVertexFormat }, // side
              { shaderLocation: 4, offset: 9 * 4, format: "float32" as GPUVertexFormat }, // t
              { shaderLocation: 5, offset: 10 * 4, format: "float32" as GPUVertexFormat }, // mode
            ],
          },
          {
            arrayStride: OVERLAY_INSTANCE_FLOATS * 4,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 6, offset: 0, format: "float32x4" as GPUVertexFormat }, // rotation
              { shaderLocation: 7, offset: 4 * 4, format: "float32x4" as GPUVertexFormat }, // position + extent
              { shaderLocation: 8, offset: 8 * 4, format: "float32x4" as GPUVertexFormat }, // scale + thickness
              { shaderLocation: 9, offset: 12 * 4, format: "float32x4" as GPUVertexFormat }, // color
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
            // Premultiplied: the FS already scaled rgb by alpha. See the shader.
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      // The rig ignores depth entirely. It shares this pass's buffer with the
      // wireframe's mesh prepass, and that prepass exists to hide the far side
      // of the BODY — not to hide the skeleton inside it. An editor wants the
      // rig in front of the mesh, which is what "always" says. The cost is that
      // the rig no longer sorts against itself; for line work a few pixels wide,
      // draw order reads the same.
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: false,
        depthCompare: "always",
      },
      multisample: { count: Engine.OVERLAY_SAMPLE_COUNT },
    } satisfies GPURenderPipelineDescriptor
    this.overlayPipeline = this.device.createRenderPipeline(overlayPipelineDescriptor)

    // The solid volumes: the same shader and layout, with no depth write and no
    // culling. A translucent body must not hide the rig behind it, and you have
    // to see its far wall for it to read as a volume rather than a silhouette.
    this.overlaySolidPipeline = this.device.createRenderPipeline({
      ...overlayPipelineDescriptor,
      label: "overlay solid pipeline",
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: "always" },
    })

    const compositeShader = this.device.createShaderModule({
      label: "overlay composite shader",
      code: OVERLAY_COMPOSITE_SHADER_WGSL,
    })
    this.overlayCompositeLayout = this.device.createBindGroupLayout({
      label: "overlay composite layout",
      entries: [{ binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float" } }],
    })
    this.overlayCompositePipeline = this.device.createRenderPipeline({
      label: "overlay composite pipeline",
      layout: this.device.createPipelineLayout({
        label: "overlay composite pipeline layout",
        bindGroupLayouts: [this.overlayCompositeLayout],
      }),
      vertex: { module: compositeShader, entryPoint: "vs" },
      fragment: {
        module: compositeShader,
        entryPoint: "fs",
        targets: [
          {
            format: this.presentationFormat,
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list" },
      multisample: { count: 1 },
    })
    this.overlayCompositePassDescriptor = {
      label: "overlay composite pass",
      colorAttachments: [
        { view: undefined as unknown as GPUTextureView, loadOp: "load", storeOp: "store" },
      ],
    }

    this.overlayBindGroup = this.device.createBindGroup({
      label: "overlay bind group",
      layout: bgLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.overlayUniformBuffer } },
      ],
    })

    // The mesh wireframe: the same line-list target, its own pipeline, because it
    // draws the model's OWN vertex buffer through the model's OWN skin matrices.
    // That is the whole reason it exists rather than emitting lines from the
    // loader's positions — those are bind pose, and a wireframe built from them
    // sits perfectly on a T-posed model and slides off every animated one.
    this.wireframeUniformBuffer = this.device.createBuffer({
      label: "wireframe color",
      size: 32, // vec4 colour + vec2 viewport + thickness + pad
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.wireframeSeamUniformBuffer = this.device.createBuffer({
      label: "wireframe color (material borders)",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.wireframeHoverUniformBuffer = this.device.createBuffer({
      label: "wireframe color (hovered material)",
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const wireBg0 = this.device.createBindGroupLayout({
      label: "wireframe group 0 layout (camera + wire)",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        // Both stages: the FS takes the colour, the VS takes the viewport and
        // the stroke width it extrudes each edge quad to.
        {
          binding: 1,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform" },
        },
      ],
    })
    // Spelled out rather than mapped over a range: tests/bindings.test.mjs reads
    // these statically to check every bind group covers its layout, and a loop
    // hides the bindings from it.
    this.wireframeSkinLayout = this.device.createBindGroupLayout({
      label: "wireframe group 1 layout (mesh + skin)",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } },
      ],
    })
    const wireShader = this.device.createShaderModule({ label: "wireframe shader", code: WIREFRAME_SHADER_WGSL })
    this.wireframePipeline = this.device.createRenderPipeline({
      label: "wireframe pipeline",
      layout: this.device.createPipelineLayout({
        label: "wireframe pipeline layout",
        bindGroupLayouts: [wireBg0, this.wireframeSkinLayout],
      }),
      // No vertex stream: an edge quad's corners come from two different model
      // vertices, so the mesh is read through storage instead.
      vertex: { module: wireShader, entryPoint: "vs" },
      fragment: {
        module: wireShader,
        entryPoint: "fs",
        targets: [
          {
            format: this.presentationFormat,
            blend: {
              color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
              alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            },
          },
        ],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      // Depth-TESTED but not written: the mesh is a haze the rig reads against,
      // so a bone behind a triangle must not be punched out by it.
      depthStencil: { format: "depth24plus", depthWriteEnabled: false, depthCompare: this.depthAhead },
      multisample: { count: Engine.OVERLAY_SAMPLE_COUNT },
    })
    // The mesh's own depth, so the wireframe can be occluded by the body it
    // belongs to. Occluded is the default everywhere — Blender's edit mode, Maya,
    // three's and Babylon's wireframe materials all depth-test, and X-ray is a
    // toggle beside them. Seeing both walls of a 30k-triangle body at once is
    // moire, not information.
    //
    // It writes depth and nothing else — but it still DECLARES the colour
    // target, at writeMask 0. A pipeline's attachment state has to match its
    // pass's, and a pass with a colour attachment will not take a pipeline that
    // has none. Same trick the scene's own depth prepass uses.
    //
    // Its own pass rather than the scene's, because the scene's depth is
    // multisampled and discarded before the composite.
    this.wireframeDepthPipeline = this.device.createRenderPipeline({
      label: "wireframe depth prepass pipeline",
      layout: this.device.createPipelineLayout({
        label: "wireframe depth prepass layout",
        bindGroupLayouts: [wireBg0, this.wireframeSkinLayout],
      }),
      vertex: {
        module: wireShader,
        entryPoint: "vsDepth",
        buffers: [
          { arrayStride: 8 * 4, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat }] },
          { arrayStride: 4 * 2, attributes: [{ shaderLocation: 1, offset: 0, format: "uint16x4" as GPUVertexFormat }] },
          { arrayStride: 4, attributes: [{ shaderLocation: 2, offset: 0, format: "unorm8x4" as GPUVertexFormat }] },
        ],
      },
      fragment: {
        module: wireShader,
        entryPoint: "fs",
        targets: [{ format: this.presentationFormat, writeMask: 0 }],
      },
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: this.depthAhead,
        // The wireframe lies exactly ON the surface this writes, so every edge
        // ties with its own triangles and loses wherever rounding goes the wrong
        // way — lines that break up and shift as the camera turns. Push the
        // solid mesh back so the edges win their own ties. The slope term is
        // what handles a surface seen at a grazing angle, where a pixel spans
        // far more depth than a constant bias can cover.
        //
        // SIGNED BY CONVENTION, as the outline hulls are: bias adds to the depth
        // VALUE, and reversed-Z inverts what a larger value means.
        depthBias: this.reversedZ ? -64 : 64,
        depthBiasSlopeScale: this.reversedZ ? -2 : 2,
        depthBiasClamp: 0,
      },
      multisample: { count: Engine.OVERLAY_SAMPLE_COUNT },
    })

    this.wireframeBindGroup = this.device.createBindGroup({
      label: "wireframe bind group",
      layout: wireBg0,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.wireframeUniformBuffer } },
      ],
    })
    this.wireframeSeamBindGroup = this.device.createBindGroup({
      label: "wireframe bind group (material borders)",
      layout: wireBg0,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.wireframeSeamUniformBuffer } },
      ],
    })
    this.wireframeHoverBindGroup = this.device.createBindGroup({
      label: "wireframe bind group (hovered material)",
      layout: wireBg0,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.wireframeHoverUniformBuffer } },
      ],
    })

    this.overlayPassDescriptor = {
      label: "overlay pass",
      timestampWrites: this.stamps("overlay"),
      colorAttachments: [
        {
          view: undefined as unknown as GPUTextureView,
          resolveTarget: undefined,
          // Transparent, because this layer is composited over the frame rather
          // than drawn into it. storeOp discard keeps the 4 samples in tile
          // memory on a TBDR part — only the resolve reaches RAM.
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "discard",
        },
      ],
      depthStencilAttachment: {
        view: undefined as unknown as GPUTextureView,
        depthClearValue: this.depthClear,
        depthLoadOp: "clear",
        depthStoreOp: "discard",
      },
    }
  }

  // Step 4: Create camera and uniform buffer
  private setupCamera() {
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 40 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    // The mirror's camera: same block, view folded with the reflection. Always
    // allocated — it is 160 bytes, and the bind group that binds it is built
    // once beside the main one rather than on the first frame a mirror turns on.
    this.mirrorCameraBuffer = this.device.createBuffer({
      label: "mirror camera uniforms",
      size: 40 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.mirrorVPBuffer = this.device.createBuffer({
      label: "mirror view-projection",
      size: 80,
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
    // panLocked tracks the BINDING, on every path that changes it — a target
    // set to a fixed point is pannable again, and forgetting it here is how the
    // pan stays dead after the follow was turned off. See Camera.panLocked.
    if (modelOrVec === null) {
      this.camera.panLocked = false
      this.cameraTargetModel = null
      return
    }
    if ("x" in modelOrVec && "y" in modelOrVec && "z" in modelOrVec) {
      this.camera.panLocked = false
      this.cameraTargetModel = null
      this.camera.target.x = modelOrVec.x
      this.camera.target.y = modelOrVec.y
      this.camera.target.z = modelOrVec.z
      return
    }
    this.camera.panLocked = true
    this.cameraTargetModel = modelOrVec
    this.cameraTargetBoneName = boneName ?? ""
    this.cameraTargetOffset.x = offset?.x ?? 0
    this.cameraTargetOffset.y = offset?.y ?? 0
    this.cameraTargetOffset.z = offset?.z ?? 0
  }

  /** Souls-style follow cam: orbit center tracks a model bone each frame. Shorthand for setCameraTarget(model, boneName, offset). */
  setCameraFollow(model: Model | null, boneName?: string, offset?: Vec3, smoothing?: number): void {
    // Panning sets the very thing the follow overwrites each frame, so it is
    // refused for as long as the shot is riding a bone. See Camera.panLocked.
    this.camera.panLocked = model !== null
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

  /** Whether a loaded camera track is allowed to drive (setCameraVmdEnabled).
   *  Held separately from `camera.vmdDriven` because that flag now answers to
   *  two sources, and a track switched off must stay off when the other one
   *  releases the camera. */
  private cameraVmdEnabled = true
  /** A pose pushed in from outside — see setCameraPose. Reapplied every frame,
   *  so it outranks the orbit AND a loaded track for as long as it is set. */
  private cameraPoseOverride: CameraPose | null = null

  /** The one place that decides who is holding the camera. An external pose
   *  wins; a track drives when it is loaded and enabled; otherwise orbit. */
  private refreshCameraDrive(): void {
    this.camera.setVmdDriven(
      this.cameraPoseOverride !== null || (this.cameraVmdEnabled && this.cameraAnimation !== null),
    )
  }

  /**
   * Aim the camera from outside — a solved match-move, a saved shot, a rig
   * driving the view from the host's own clock.
   *
   * The exact partner of `getCameraPose`, and the same five channels: the shot
   * as MMD states it, roll included. Orbit cannot express roll, so this is the
   * only way a tilted camera reaches the engine.
   *
   * Reapplied every frame while set, which makes it authoritative rather than
   * advisory — nothing the transport or a loaded track does moves it. Pass null
   * to release, and whatever was driving before takes the camera back.
   */
  setCameraPose(pose: CameraPose | null): void {
    if (pose) {
      // Copied, not held: a host reusing one object per frame is the normal
      // shape of a track, and storing the reference would make the value we
      // reapply depend on when the caller next touched theirs.
      this.cameraPoseOverride = {
        target: new Vec3(pose.target.x, pose.target.y, pose.target.z),
        rotation: new Vec3(pose.rotation.x, pose.rotation.y, pose.rotation.z),
        distance: pose.distance,
        fov: pose.fov,
      }
    } else {
      this.cameraPoseOverride = null
    }
    this.refreshCameraDrive()
    if (this.cameraPoseOverride) this.camera.setVmdPose(this.cameraPoseOverride)
  }

  /** The pose currently forced from outside, or null when nothing is. */
  getCameraPoseOverride(): CameraPose | null {
    return this.cameraPoseOverride
  }

  /** Load a camera VMD (dedicated camera file, or any VMD's camera block) and drive the shot
   *  from it. Default-on once a non-empty track loads; toggle with setCameraVmdEnabled. */
  async loadCameraVmd(url: string): Promise<void> {
    const frames = await VMDLoader.loadCamera(url)
    this.cameraAnimation = frames.length ? new CameraAnimation(frames) : null
    this.cameraVmdEnabled = true
    this.refreshCameraDrive()
  }

  /** Load a camera VMD from an already-fetched buffer (e.g. a File the user dropped). */
  loadCameraVmdFromBuffer(buffer: ArrayBuffer): void {
    const frames = VMDLoader.loadCameraFromBuffer(buffer)
    this.cameraAnimation = frames.length ? new CameraAnimation(frames) : null
    this.cameraVmdEnabled = true
    this.refreshCameraDrive()
  }

  /**
   * Drive the shot from camera keyframes built in JS — the camera's answer to
   * `Model.loadClip`.
   *
   * The two loadCameraVmd* methods take FILE BYTES, which is all a viewer ever
   * needs. An editor needs the other direction: hold the track as data, change
   * a keyframe, and see the result immediately. Going through the writer and
   * back through the parser for every edit would work and would be absurd.
   *
   * Empty (or an empty array) clears the track and returns the camera to orbit,
   * same as clearCameraVmd — a track with no keyframes cannot drive anything,
   * and silently keeping the previous one would be worse than saying so.
   */
  loadCameraClip(frames: CameraKeyframe[]): void {
    this.cameraAnimation = frames.length ? new CameraAnimation([...frames]) : null
    this.cameraVmdEnabled = true
    this.refreshCameraDrive()
  }

  /** The loaded camera track as editable keyframes, or [] with none loaded.
   *  Copies — mutating them does not reach the track being sampled. */
  getCameraClip(): CameraKeyframe[] {
    return this.cameraAnimation?.keyframes() ?? []
  }

  /** The loaded camera track as camera-VMD bytes. Throws with none loaded:
   *  writing an empty camera file is a mistake worth hearing about, not a
   *  30-byte header to hand someone as a download. */
  exportCameraVmd(): ArrayBuffer {
    const frames = this.cameraAnimation?.keyframes()
    if (!frames?.length) throw new Error("No camera track loaded")
    return new VMDWriter().writeCamera(frames)
  }

  /** Turn the loaded camera VMD on/off (falls back to orbit when off). No-op if none loaded. */
  setCameraVmdEnabled(enabled: boolean): void {
    this.cameraVmdEnabled = enabled
    this.refreshCameraDrive()
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
    // NOT YET, OR NEVER AGAIN. `device` is definite-assignment: it is undefined
    // until init() resolves and after dispose(), and TypeScript cannot see
    // either state. This setter is reached from an ASYNC callback — the audio
    // analysis, the score fetch, the lyric rasteriser all land whenever they
    // land — so on a hot reload the in-flight promise of the outgoing engine
    // resolves against the incoming one, which is holding a ref but has not
    // finished init. It threw "Cannot read properties of undefined (reading
    // 'createBuffer')" from a line whose only crime was being fast.
    //
    // Dropped rather than queued: every caller here re-pushes on the effect
    // that owns the asset, and that effect re-runs on the very reload that
    // caused this.
    if (!this.device) return
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
    this.rebindSharedBuffers()
  }

  /**
   * Install a score — note events — for the rzNote* and rzKey* effect functions.
   * Null clears it.
   *
   * The sibling of setAudioData, and deliberately a SEPARATE interface rather
   * than something derived from it: a spectrum cannot give back a discrete pitch
   * and onset, and that discreteness is the whole substance of a falling note.
   * A scene can hold both and read them together.
   *
   * `release` is how long a key keeps glowing after its note ends, in seconds.
   * It belongs here rather than in the effect because the key map it feeds is
   * computed on the CPU — see writeMidiClock.
   */
  /**
   * Install the track's lyric lines for the rzLyric* effect functions — the
   * timing of the words on the scene clock, and optionally the words
   * themselves: `atlas.source` is a canvas/bitmap of rasterised lines (the
   * host draws them — Canvas2D handles every script the platform does) with
   * `atlas.rects` saying where each line sits, in 0..1 [u0, vTop, u1, vBottom].
   * Null clears. A plain buffer write plus at most a texture copy: buffer and
   * atlas are both fixed-size, so nothing re-binds whenever lyrics arrive.
   */
  setLyrics(
    lines: LyricLine[] | null,
    atlas?: { source: GPUCopyExternalImageSource; width: number; height: number; rects: LyricRect[] },
  ): void {
    // NOT YET, OR NEVER AGAIN. `device` is definite-assignment: it is undefined
    // until init() resolves and after dispose(), and TypeScript cannot see
    // either state. This setter is reached from an ASYNC callback — the audio
    // analysis, the score fetch, the lyric rasteriser all land whenever they
    // land — so on a hot reload the in-flight promise of the outgoing engine
    // resolves against the incoming one, which is holding a ref but has not
    // finished init. It threw "Cannot read properties of undefined (reading
    // 'createBuffer')" from a line whose only crime was being fast.
    //
    // Dropped rather than queued: every caller here re-pushes on the effect
    // that owns the asset, and that effect re-runs on the very reload that
    // caused this.
    if (!this.device) return
    this.device.queue.writeBuffer(this.lyricsBuffer, 0, packLyrics(lines ?? [], atlas?.rects))
    if (!atlas) return
    const w = Math.min(atlas.width, LYRIC_ATLAS_MAX_W)
    const h = Math.min(atlas.height, LYRIC_ATLAS_MAX_H)
    if (this.lyricsTexture.width !== w || this.lyricsTexture.height !== h) {
      this.lyricsTexture.destroy()
      this.lyricsTexture = this.device.createTexture({
        label: "lyric line atlas",
        size: [w, h],
        format: "r8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.lyricsTextureView = this.lyricsTexture.createView()
      // The one re-bind lyrics can cause, and only when the atlas changes SIZE
      // — a new song, not a new line. The timing buffer never moves, so an
      // effect reading rzLyric* is never interrupted by lyrics arriving.
      this.rebuildFieldBindGroup()
    }
    // premultipliedAlpha: a 2D canvas IS premultiplied, and the default (false)
    // makes the copy UNpremultiply — which divides each texel by its own alpha
    // and turns every antialiased glyph edge into a hard one. Saying the
    // destination is premultiplied means no conversion, so the red channel
    // arrives as the coverage the rasteriser drew.
    this.device.queue.copyExternalImageToTexture(
      { source: atlas.source },
      { texture: this.lyricsTexture, premultipliedAlpha: true },
      [w, h],
    )
  }

  setMidiNotes(notes: MidiNote[] | null, release = 0.35): void {
    // NOT YET, OR NEVER AGAIN. `device` is definite-assignment: it is undefined
    // until init() resolves and after dispose(), and TypeScript cannot see
    // either state. This setter is reached from an ASYNC callback — the audio
    // analysis, the score fetch, the lyric rasteriser all land whenever they
    // land — so on a hot reload the in-flight promise of the outgoing engine
    // resolves against the incoming one, which is holding a ref but has not
    // finished init. It threw "Cannot read properties of undefined (reading
    // 'createBuffer')" from a line whose only crime was being fast.
    //
    // Dropped rather than queued: every caller here re-pushes on the effect
    // that owns the asset, and that effect re-runs on the very reload that
    // caused this.
    if (!this.device) return
    if (this.midiBuffer !== this.midiFallbackBuffer) this.midiBuffer.destroy()
    this.midiRelease = Math.max(0, release)
    // Sorted by onset. Nothing in the accessors requires it, but a caller
    // walking the list to spawn in time order is the obvious use and a score
    // arriving unsorted would make that silently wrong.
    this.midiNotes = notes ? [...notes].sort((a, b) => a.start - b.start) : []
    if (this.midiNotes.length === 0) {
      this.midiBuffer = this.midiFallbackBuffer
    } else {
      let lo = 127
      let hi = 0
      let end = 0
      for (const n of this.midiNotes) {
        if (n.pitch < lo) lo = n.pitch
        if (n.pitch > hi) hi = n.pitch
        end = Math.max(end, n.start + n.duration)
      }
      const payload = new Float32Array(MIDI_NOTES + this.midiNotes.length * MIDI_STRIDE)
      payload[0] = this.midiNotes.length
      payload[1] = lo
      payload[2] = hi
      payload[5] = end
      payload[6] = this.midiRelease
      for (let i = 0; i < this.midiNotes.length; i++) {
        const n = this.midiNotes[i]
        const o = MIDI_NOTES + i * MIDI_STRIDE
        payload[o] = n.start
        payload[o + 1] = n.duration
        payload[o + 2] = n.pitch
        payload[o + 3] = n.velocity ?? 1
      }
      this.midiBuffer = this.device.createBuffer({
        label: "score",
        size: payload.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(this.midiBuffer, 0, payload)
    }
    // Same reason as setAudioData: every consumer holds the buffer by reference,
    // so all of them re-bind and a score arriving before or after an effect both
    // work.
    this.rebindSharedBuffers()
  }

  /**
   * Move the score's clock, and rebuild the per-pitch key map from it.
   *
   * The map is why this is more than a header write. Falling notes index the
   * note list directly, but a keyboard glow asks the opposite question per
   * pixel — is anything sounding at THIS pitch — and answering that in the
   * shader would be a scan of the whole score per fragment. One O(notes) pass
   * here, once a frame, turns it into a single lookup.
   *
   * The pass is a full scan rather than a cursor because scrubbing exists: a
   * cursor is only correct while time moves forward, and a timeline drag is
   * exactly when a wrong answer is most visible. Ten thousand notes is ten
   * thousand comparisons, which is nothing beside the pose pass.
   */
  setMidiTime(seconds: number, playing = true): void {
    if (this.midiBuffer === this.midiFallbackBuffer) return
    const keys = this.midiLiveScratch
    keys.fill(0)
    keys[0] = seconds
    keys[1] = playing ? 1 : 0
    const release = Math.max(this.midiRelease, 1e-4)
    for (const n of this.midiNotes) {
      if (n.start > seconds) continue // sorted, but a later note may still be shorter
      const k = Math.round(n.pitch)
      if (k < 0 || k >= MIDI_KEYS) continue
      const since = seconds - (n.start + n.duration)
      // Held reads 1; released decays linearly over the release window. Max,
      // not sum: two notes on one key is still one key, lit once.
      const e = since <= 0 ? 1 : since >= release ? 0 : 1 - since / release
      if (e > keys[2 + k]) keys[2 + k] = e
    }
    // One write covering the clock (floats 3–4) and the key map (8 onward);
    // scratch holds them adjacently so it stays a single upload.
    this.device.queue.writeBuffer(this.midiBuffer, 3 * 4, keys.buffer as ArrayBuffer, 0, 2 * 4)
    this.device.queue.writeBuffer(this.midiBuffer, MIDI_HEADER * 4, keys.buffer as ArrayBuffer, 2 * 4, MIDI_KEYS * 4)
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
    this.refreshCameraDrive()
  }

  /**
   * THE TRANSPORT'S CLOCK — where the scene is in its own playback.
   *
   * The first model with an active clip (playing or scrubbed), so a static stage
   * never freezes it at frame 0. Falls back to the first model with a clip, then
   * to 0 for an empty scene.
   *
   * NOT `sceneClock`, and the difference is the whole reason this has a name.
   * `sceneClock` only ever accumulates delta — it is how long the engine has
   * been running, it does not move when you scrub, and it does not stop when you
   * pause. Anything that should line up with what the transport shows has to
   * read THIS. An effect scheduled to frame 100 against sceneClock fires once,
   * a hundred frames after the page loaded, and never again.
   *
   * Deterministic offline: the export loop advances model animation by an exact
   * per-frame delta, so this reproduces frame for frame.
   */
  private transportTime(): number {
    let fallback: number | null = null
    for (const inst of this.modelInstances.values()) {
      // Stages are skipped outright. Scenery carries no motion, and it is added
      // BEFORE the cast — it paints while the models stream in behind it — so it
      // is first in insertion order and was seeding this clock with its own
      // permanent zero. In a scene with a stage, a camera VMD therefore sampled
      // frame 0 forever and the shot never moved.
      if (inst.isStage || inst.isPlane || inst.isProp) continue
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

  /**
   * The live orbit, read in ONE call.
   *
   * A host that stores the shot has to be able to ask where the camera actually
   * IS, because a drag on the canvas moves this and nothing else — and a
   * document that never asks will happily write back the angle it last set,
   * discarding whatever the person just did with the mouse. Reading the four
   * separately invites a torn set across a frame boundary; this cannot tear.
   *
   * `target` is the orbit's own centre. While the engine is following a bone
   * that point rides the bone, so a caller storing a FOLLOW offset must keep its
   * own and take only the angles from here.
   */
  getCameraOrbit(): { alpha: number; beta: number; distance: number; target: Vec3 } {
    const c = this.camera
    return {
      alpha: c.alpha,
      beta: c.beta,
      distance: c.radius,
      target: new Vec3(c.target.x, c.target.y, c.target.z),
    }
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
  /**
   * Roll the orbiting shot, radians — the lean alpha and beta cannot state.
   *
   * Tips the up vector about the eye→target line, so the camera stays exactly
   * where it was and keeps looking at exactly what it looked at. Everything the
   * orbit does still works underneath it: following a bone, dragging, zooming.
   *
   * A camera VMD carries its own roll and ignores this while it drives.
   */
  setCameraRoll(r: number): void {
    this.camera.roll = r
  }
  getCameraRoll(): number {
    return this.camera.roll
  }
  /** Vertical field of view in radians (default π/4). While a camera VMD
   *  drives the view it animates fov itself; the orbit value set here is
   *  restored when the VMD releases the camera. */
  /**
   * The shot as MMD states it — target, euler rotation, distance, fov.
   *
   * For a host writing the camera out to something else: an AE composition, a
   * .vmd, a log. The same five channels whichever mode is driving, so the
   * caller never asks and never decomposes a view matrix to find out.
   */
  getCameraPose(): CameraPose {
    return this.camera.getPose()
  }

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
      size: 80 * 4, // ambient (4) + 4 lights x 2 vec4 (32) + irradiance SH 9 x vec4 (36), padded to 80
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
    // The sky's irradiance, when an HDRI world is installed — the world
    // STRENGTH dial keeps its meaning by scaling it, and the world COLOUR is
    // simply unread while the flag is up (Blender's own semantics: the image
    // replaces the colour, strength applies to either).
    for (let i = 0; i < 9; i++) {
      const b = 36 + i * 4
      if (this.worldSH) {
        this.lightData[b] = this.worldSH[i * 3] * s
        this.lightData[b + 1] = this.worldSH[i * 3 + 1] * s
        this.lightData[b + 2] = this.worldSH[i * 3 + 2] * s
      } else {
        this.lightData[b] = 0
        this.lightData[b + 1] = 0
        this.lightData[b + 2] = 0
      }
      this.lightData[b + 3] = 0
    }
    this.lightData[39] = this.worldSH ? 1 : 0
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
    /** Floor mirror, on or off — NOT a strength: the reflection is its own
     *  layer, and how much of it shows is the surface's own `opacity`
     *  covering it. Off (default) never renders the reflection pass at all. */
    mirror?: boolean
    /** Mirror softness, 0–1: 0 a polished mirror, 1 the softest blur level,
     *  scaled by how far the reflected geometry sits behind the surface. */
    mirrorBlur?: number
    /** How soft the received shadow's edge is, 0–1. 0 (default) is the sharp
     *  kernel this has always used, to the bit; 1 spreads the taps fourteen
     *  times as wide, which is the edge an overcast sky throws.
     *
     *  A property of the LIGHT, applied where the light is received: the sun
     *  in a scene is either a point source with a hard edge or a sky with
     *  none, and a floor that always answers "hard" can only match one of
     *  them. Above 0 the taps go from nine to sixteen, so leave it at 0 for
     *  scenes that want the sharp edge and pay nothing. */
    shadowSoftness?: number
  }): void {
    // NOT YET, OR NEVER AGAIN — same race setAudioData documents. This call is
    // deferred a frame by useSceneSync's own rAF batching, and a hot reload
    // that swaps in a new (uninitialized) engine between the schedule and the
    // callback lands this on a `device` that has not been assigned yet. The
    // effect that scheduled it re-fires once the new engine is ready, so
    // dropping this one loses nothing.
    if (!this.device) return
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
      mirror: false,
      mirrorBlur: 0,
      shadowSoftness: 0,
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

  /**
   * The scene's positional lights — an ADDITIVE layer over the sun, which stays
   * the key light and keeps the toon ramp to itself.
   *
   * Colour and intensity are multiplied here rather than stored apart: every
   * read is the product, and two numbers that are only ever multiplied are two
   * numbers that can disagree.
   *
   * Past MAX_LIGHTS the extras are DROPPED, not wrapped: the lights that fit
   * keep the meaning the caller gave them, which is the same rule the anchor
   * table follows. Passing none (or an empty list) turns the layer off and the
   * scene renders exactly as it did before lights existed.
   */
  setLights(
    /** Structural {x,y,z} rather than the Vec3 class, the same choice effect
     *  params make: a scene document's JSON passes straight in, and so does a
     *  literal typed into a console. Vec3 satisfies it either way. */
    lights: { position: XYZ; color: XYZ; intensity?: number; radius?: number }[] | null,
  ): void {
    const list = (lights ?? []).slice(0, MAX_LIGHTS)
    this.docLightCount = list.length
    // RECORDS only — the header belongs to allocateLightSlots, the one writer.
    // This used to zero-and-upload the header region too, which left two CPU
    // mirrors of the count (this array's, holding a transient zero, and
    // lightHeader's, holding the truth). Correct on the GPU by queue ordering,
    // and a trap on the CPU: the first future path that uploads lightsData
    // whole would silently switch every light off. Effects' slots are likewise
    // untouched — they are rewritten per frame by their own compute.
    for (let i = 0; i < list.length; i++) {
      const l = list[i]
      const b = LIGHT_HEADER + i * LIGHT_STRIDE
      this.lightsData[b] = l.position.x
      this.lightsData[b + 1] = l.position.y
      this.lightsData[b + 2] = l.position.z
      // A radius of zero would switch the light off through the window term,
      // which is a confusing way to spell "off" — default to a stage-sized
      // reach instead, and let 0 mean 0 only when it is asked for explicitly.
      this.lightsData[b + 3] = Math.max(l.radius ?? 10, 0)
      // Clamped at zero: the layer is ADDITIVE, and a negative channel would
      // darken what it lands on — same rule the emit stage enforces.
      const k = l.intensity ?? 1
      this.lightsData[b + 4] = Math.max(l.color.x * k, 0)
      this.lightsData[b + 5] = Math.max(l.color.y * k, 0)
      this.lightsData[b + 6] = Math.max(l.color.z * k, 0)
      // [b + 7] is `type`, reserved: every light is a point light today.
    }
    if (list.length) {
      this.device.queue.writeBuffer(
        this.lightsBuffer,
        LIGHT_HEADER * 4,
        this.lightsData.buffer as ArrayBuffer,
        LIGHT_HEADER * 4,
        list.length * LIGHT_STRIDE * 4,
      )
    }
    // The effects' bases move when the document's count does; the header —
    // count included — is written there.
    this.allocateLightSlots()
  }

  /** How many positional lights the scene is carrying. */
  getLightCount(): number {
    return this.lightHeader[0]
  }

  /** Guarded, unlike most private writers here, because its callers are not:
   *  setWorld/setSun are public and can be called before init() finishes
   *  assigning `device` — a scene-settings effect firing on mount races the
   *  engine's own async setup. The state write still lands immediately either
   *  way; only the GPU upload defers, and setupLighting's own writeWorld/
   *  writeSun calls during init pick up whatever was already set. */
  private updateLightBuffer() {
    if (!this.device || !this.lightUniformBuffer) return
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

    this.overlayDepthTexture?.destroy()
    this.overlayDepthTexture = null
    this.overlayMsaaTexture?.destroy()
    this.overlayMsaaTexture = null
    this.overlayResolveTexture?.destroy()
    this.overlayResolveTexture = null
    this.overlayInstanceBuffer?.destroy()
    this.overlayInstanceBuffer = null
    for (const inst of this.modelInstances.values()) {
      for (const edges of inst.wireEdges.values()) edges?.buffer.destroy()
      inst.wireEdges.clear()
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

  /** loadModel's folder/zip path for a prop. See addProp. */
  async loadProp(
    name: string,
    options: LoadModelFromFilesOptions & { transform?: Partial<ModelTransform> },
  ): Promise<Model> {
    const { model, pmxKey, reader } = await this.openPmxFromFiles(name, options)
    await this.addProp(model, pmxKey, { name, transform: options.transform, assetReader: reader })
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
    options?: { stage?: boolean; plane?: boolean; dynamic?: boolean; prop?: boolean },
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
    await this.setupModelInstance(
      key,
      model,
      basePath,
      reader,
      options?.stage ?? false,
      options?.plane ?? false,
      options?.dynamic ?? false,
      options?.prop ?? false,
    )
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

  /**
   * Add a PMX as a PROP: an object a character holds or wears rather than a
   * performer or the environment. A microphone, a fan, a sword, an umbrella.
   *
   * It keeps what makes a held thing look right — physics (the charm on a phone
   * strap swings), toon outlines, its own clip if it has one — and drops what
   * makes a model a cast member: no effect subject id, so a silhouette effect
   * still outlines HER and not the mic; no seeding of the scene clock; no bone
   * picking in the pose editor. Like a card it leaves the built-in ground
   * alone, which is the one thing a stage does that a prop must not. Usually
   * hung from a bone with setModelParent, though it can stand on its own.
   */
  async addProp(
    model: Model,
    pmxPath: string,
    options?: { name?: string; transform?: Partial<ModelTransform>; assetReader?: AssetReader },
  ): Promise<string> {
    const key = await this.addModel(model, pmxPath, options?.name, options?.assetReader, { prop: true })
    if (options?.transform) this.setModelTransform(key, options.transform)
    return key
  }

  /**
   * Put a picture in the scene as a flat card.
   *
   * The thing compositors arrange in a post tool's fake 3D space — Nuke calls
   * it a Card, After Effects a 3D layer, MMD 板ポリ — except the space here is
   * the real one. A card is occluded by anything in front of it, occludes what
   * is behind it, takes perspective when turned, and is caught by depth of
   * field like everything else, because it is ordinary geometry rather than a
   * layer composited afterwards.
   *
   * It is a MODEL, deliberately. Not a new kind of scene object with its own
   * list, its own persistence and its own selection: a card wants a position,
   * a rotation and a size, which is exactly what a model already has, and
   * everything built around models — the transform, the shadow settings, the
   * material editor, the asset bundle — works on it the day it exists. It is
   * not a STAGE, though: it skips the same machinery for the same reasons, and
   * leaves the floor alone. See ModelInstance.isPlane.
   *
   * @returns the model key, for setModelTransform and removeModel.
   */
  async addPlane(options: {
    /** The picture's own bytes, exactly as uploaded. The name's extension picks
     *  the decoder, so this never re-encodes anything. */
    image: ArrayBuffer
    /** File name — decides the decoder, names the model and keys its texture. */
    name: string
    /** World size of the card. The caller owns the aspect: it knows the
     *  picture's own proportions, and a card is free to disagree with them. */
    width: number
    height: number
    transform?: Partial<ModelTransform>
    /** Drawn from behind as well. Off by default — a card turned away from the
     *  camera vanishing is the same thing a sheet of paper does. */
    doubleSided?: boolean
    /** The picture will be replaced every frame (see setPlaneFrame). Allocates
     *  the texture without a mip chain, which is what makes that affordable. */
    dynamic?: boolean
  }): Promise<string> {
    const { image, name, width, height } = options
    const hw = Math.max(width, 1e-4) / 2
    const hh = Math.max(height, 1e-4) / 2

    // A quad on the XY plane, facing +Z, centred on its own origin — so a
    // rotation turns it about its middle and a position places its centre,
    // which is what a handle in the viewport implies.
    //
    // V IS FLIPPED, and this is the whole of it: a picture's rows run downward
    // from its top-left, a UV runs upward from the bottom-left, and a card that
    // renders its image upside down looks like a bug in everything else.
    // prettier-ignore
    const vertexData = new Float32Array([
      // x     y     z    nx   ny   nz   u    v
      -hw,  -hh,  0.0,  0.0, 0.0, 1.0,  0.0, 1.0,
       hw,  -hh,  0.0,  0.0, 0.0, 1.0,  1.0, 1.0,
       hw,   hh,  0.0,  0.0, 0.0, 1.0,  1.0, 0.0,
      -hw,   hh,  0.0,  0.0, 0.0, 1.0,  0.0, 0.0,
    ])
    // Two triangles, counter-clockwise seen from +Z. A second pair wound the
    // other way is how "visible from behind" is done here, rather than a
    // per-material cull flag the rest of the engine has no concept of.
    const indices = options.doubleSided ? [0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3]
    const indexData = new Uint32Array(indices)

    // The texture table's one entry. The path is a key, not a location — the
    // reader below answers it from memory, so nothing is fetched and nothing is
    // written to disk.
    //
    // A PLAIN RELATIVE NAME under a plain directory, because the loader treats
    // this exactly as it treats a PMX's: it takes the model path's directory
    // and JOINS the texture entry onto it. A scheme-looking path went through
    // that as `plane://` + `plane://name` and matched nothing, so every card
    // came out with the untextured fallback. `plane/<name>` joins to
    // `plane/<name>` and stays unique per card, which the engine-wide texture
    // cache needs it to be.
    const texturePath = `plane/${name}`
    const material: Material = {
      name,
      diffuse: [1, 1, 1, 1],
      specular: [0, 0, 0],
      ambient: [0, 0, 0],
      shininess: 0,
      diffuseTextureIndex: 0,
      normalTextureIndex: -1,
      sphereTextureIndex: -1,
      sphereMode: 0,
      toonTextureIndex: -1,
      sharedToon: false,
      // 0 CARRIES TWO DECISIONS, both wanted, and both silent if changed.
      //
      // No inverted-hull outline (bit 0x10): a card is not a character, and a
      // black rim around a light leak is the opposite of what it is for.
      //
      // AND NO SHADOW (bit 0x04, which is what castsShadow reads). A card is
      // usually light or artwork rather than an object, and a rectangle of hard
      // shadow thrown across the stage by a gradient reads as the renderer
      // being broken. Set geometry that SHOULD cast one is the rarer case, and
      // it can say so.
      edgeFlag: 0,
      edgeColor: [0, 0, 0, 1],
      edgeSize: 0,
      vertexCount: indices.length,
    }

    // ONE BONE, because Model requires one — it throws on an empty skeleton,
    // every vertex has to be skinned to something, and a card has nothing to
    // articulate. It is never posed; the model transform is what moves a card.
    const skeleton: Skeleton = {
      bones: [{ name: "全ての親", parentIndex: -1, bindTranslation: [0, 0, 0], children: [] }],
      inverseBindMatrices: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    }
    const vertexCount = 4
    const joints = new Uint16Array(vertexCount * 4)
    const weights = new Uint8Array(vertexCount * 4)
    for (let i = 0; i < vertexCount; i++) weights[i * 4] = 255

    const model = new Model(
      vertexData,
      indexData,
      [{ path: name, name }],
      [material],
      skeleton,
      { joints, weights },
      { morphs: [] },
    )

    // ITS OWN PATH, not addStage's. A plane and a stage skip the same machinery
    // and mean different things, and routing one through the other is how the
    // ground came to be suppressed by adding a picture.
    // The picture answers from memory, whatever it is asked for: a card has
    // exactly ONE texture, so there is nothing to disambiguate and no way for a
    // path to be wrong. Matching the string instead is what silently produced
    // untextured cards, because the loader composes that string itself.
    const reader: AssetReader = { readBinary: async () => image }
    const key = await this.addModel(model, texturePath, name, reader, { plane: true, dynamic: options.dynamic })

    // UNLIT, because a card is FOOTAGE and not a surface.
    //
    // Its pixels were finished somewhere else — a gradient painted in
    // Photoshop, a title, a rendered element — so its brightness is the artwork
    // rather than a response to anything. Shading it means the sun dimming one
    // side of a thing that has no sides, and the world colour tinting a picture
    // whose colour was the point. Left ungrouped it would take the neutral
    // Principled base, which is exactly that mistake.
    //
    // A group, not a hard-coded pipeline: a card used as SET geometry — a photo
    // of a wall, a poster standing in the room — genuinely does want the light,
    // and this is the same control every other material is changed through, so
    // that case is a graph swap rather than a feature request.
    await this.applyStyleGroups(key, [
      { id: "plane", label: "Plane", materials: [name], graph: UNLIT_GRAPH, alphaMode: "hashed" },
    ])

    if (options.transform) this.setModelTransform(key, options.transform)
    // Kept so a moving card can push frames into it. The cache is keyed by the
    // texture's logical path, which is derived rather than stored anywhere the
    // caller can see — and deriving it twice is how the two would drift.
    const tex = this.textureCache.get(texturePath)
    if (tex) this.planeTextures.set(key, tex)
    return key
  }

  /**
   * Replace what a card is showing, in place.
   *
   * For a moving card: a video element, a decoded frame, a canvas — anything
   * copyExternalImageToTexture accepts. Nothing is reallocated and no bind group
   * is rebuilt, so this is a per-frame call rather than a per-clip one; the
   * texture is written where it stands and the material keeps pointing at it.
   *
   * The frame must be the size the card was created at. A card is a fixed
   * rectangle of texels and resizing one mid-clip would mean rebuilding the
   * material behind it — so the caller allocates the card at its video's size
   * and this refuses anything else rather than stretching it silently.
   */
  setPlaneFrame(id: string, source: GPUCopyExternalImageSource, width: number, height: number): boolean {
    const tex = this.planeTextures.get(id)
    if (!tex || !this.device) return false
    if (tex.width !== width || tex.height !== height) return false
    this.device.queue.copyExternalImageToTexture({ source }, { texture: tex }, [width, height])
    // A moving card is allocated with one level precisely so this is never
    // reached: rebuilding a mip pyramid per frame is a pass per level per card.
    if (tex.mipLevelCount > 1) this.generateMipmaps(tex, tex.mipLevelCount)
    return true
  }

  /** True while a stage is in the scene. Two things turn on it: the built-in
   *  ground plane must not draw, and the far shadow cascade has nothing to
   *  cover without one (see the cascade loop). */
  hasStage(): boolean {
    for (const inst of this.modelInstances.values()) if (inst.isStage) return true
    return false
  }

  /** True while a stage is in the scene, which is when the built-in ground plane
   *  must not draw. */
  groundIsSuppressed(): boolean {
    return this.hasStage() || this.groundHidden
  }

  /**
   * Draw the built-in ground, or do not.
   *
   * Separate from opacity, which cannot express this: a ground at opacity 0
   * still WRITES DEPTH and still catches shadow — that is what makes it a
   * shadow catcher, and it is why an alpha export keeps its shadows. So a scene
   * that wants no floor at all cannot ask for one by turning the opacity down;
   * the plane is still there, still occluding, and anything reading scene depth
   * still finds a square where the floor is. A water surface deciding what lies
   * beneath it draws that square's edge across the pool.
   *
   * Suppression rather than a teardown, matching what a stage does to the same
   * plane: the ground keeps its colour, its size and its grid, so switching it
   * back restores the scene the user had rather than an engine default.
   */
  setGroundVisible(on: boolean): void {
    this.groundHidden = !on
  }

  /** Per cascade: does its map currently hold nothing but the cleared far plane?
   *  Set by the cascade loop, which skips a cascade that is unwanted and already
   *  cleared rather than re-clearing it every frame. */
  private shadowCascadeCleared: boolean[] = []

  removeModel(name: string): void {
    const inst = this.modelInstances.get(name)
    if (!inst) return
    // Before the texture cache below frees it: a stale entry here would hand a
    // destroyed texture to the next setPlaneFrame.
    this.planeTextures.delete(name)
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
    // Whatever hung from it stands on its own now, at identity — the same
    // place a detach leaves a model.
    for (const other of this.modelInstances.values()) {
      if (other.parent?.model === name) this.setModelParent(other.name, null)
    }
    this.cullListDirty = true
    this.bundlesDirty = true
    this.updateOrderDirty = true
  }

  getModelNames(): string[] {
    return Array.from(this.modelInstances.keys())
  }

  getModel(name: string): Model | null {
    return this.modelInstances.get(name)?.model ?? null
  }

  /**
   * Hang a model from a bone of another — MMD's 外部親 (outside parent).
   *
   * Every frame, after the parent has been posed and simulated, the child's
   * root bones are placed at that bone with `offset` composed on top, and only
   * then is the child posed itself. The placement enters through the child's
   * BONES rather than its model transform (Model.setRootParent): physics runs
   * in model space, so a root moved by the transform would have a charm on a
   * phone strap feel gravity swing with the hand, while a root moved by the
   * skeleton keeps down down. It also puts the child's own clip on top of the
   * ride, as MMD does — an umbrella that spins keeps spinning in the hand.
   *
   * While attached the child's position and rotation are held at identity and
   * setModelTransform ignores them; scale still applies, and is folded into
   * the placement so the offset stays in the parent's units. Detaching leaves
   * the model at identity until the host places it again.
   *
   * A bone the parent lacks rides the parent's root, which is what camera
   * follow does with an unknown name. Returns false for an unknown model, a
   * missing parent, or a model asked to ride itself.
   */
  setModelParent(
    name: string,
    parent: string | null,
    bone = "全ての親",
    offset?: { position?: Vec3; rotation?: Quat },
  ): boolean {
    const inst = this.modelInstances.get(name)
    if (!inst) return false
    if (parent === null) {
      if (inst.parent) {
        inst.parent = null
        inst.model.setRootParent(null)
        inst.skinMatricesDirty = true
        this.updateOrderDirty = true
      }
      return true
    }
    if (parent === name || !this.modelInstances.has(parent)) return false
    const p = offset?.position ?? new Vec3(0, 0, 0)
    const r = offset?.rotation ?? Quat.identity()
    const offsetMatrix = inst.parent?.offsetMatrix ?? new Float32Array(16)
    Mat4.fromPositionRotationScaleInto(p.x, p.y, p.z, r.x, r.y, r.z, r.w, 1, offsetMatrix)
    // Identity until the first frame fills it: a physics reset between now and
    // then re-poses the model, and a zero matrix would fold it to a point.
    const rootMatrix = inst.parent?.rootMatrix ?? new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
    inst.parent = { model: parent, bone, offsetMatrix, rootMatrix }
    inst.model.setPosition(new Vec3(0, 0, 0))
    inst.model.setRotation(Quat.identity())
    inst.model.setRootParent(rootMatrix)
    inst.skinMatricesDirty = true
    this.updateOrderDirty = true
    return true
  }

  /** What a model hangs from, or null. */
  getModelParent(name: string): ModelAttachment | null {
    const att = this.modelInstances.get(name)?.parent
    return att ? { model: att.model, bone: att.bone } : null
  }

  /**
   * The root an attached model is posed under this frame: the parent's
   * placement, its bone as posed and simulated, then the offset.
   *
   * The translation is divided by the child's own scale. The skin bake
   * multiplies the child's scale back on outside the skeleton, and a uniform
   * scale commutes with the rotation, so this is exactly what lands the child
   * at the bone in world units while its mesh still comes out scaled.
   */
  private placeAttached(inst: ModelInstance): void {
    const att = inst.parent!
    const parent = this.modelInstances.get(att.model)
    if (!parent) {
      this.setModelParent(inst.name, null)
      return
    }
    const out = att.rootMatrix
    const tmp = this.attachScratch
    const root = parent.model.getRootMatrix()
    const bone = parent.model.getBoneWorldMatrix(att.bone)
    if (bone) {
      Mat4.multiplyArrays(root, 0, bone, 0, tmp, 0)
      Mat4.multiplyArrays(tmp, 0, att.offsetMatrix, 0, out, 0)
    } else {
      Mat4.multiplyArrays(root, 0, att.offsetMatrix, 0, out, 0)
    }
    const s = inst.model.scale
    if (s > 0 && s !== 1) {
      const k = 1 / s
      out[12] *= k
      out[13] *= k
      out[14] *= k
    }
  }
  private readonly attachScratch = new Float32Array(16)

  /** Instances in pose order: a parent before every model hanging from it, so
   *  a child reads the bone as posed and simulated THIS frame. Insertion order
   *  otherwise. Rebuilt when a model is added, removed or re-parented. */
  private updateOrder: ModelInstance[] = []
  private updateOrderDirty = true
  private instancesInUpdateOrder(): ModelInstance[] {
    if (!this.updateOrderDirty) return this.updateOrder
    const placed = new Set<string>()
    const order: ModelInstance[] = []
    let pending = Array.from(this.modelInstances.values())
    while (pending.length > 0) {
      const rest: ModelInstance[] = []
      for (const inst of pending) {
        const p = inst.parent?.model
        if (p === undefined || placed.has(p) || !this.modelInstances.has(p)) {
          order.push(inst)
          placed.add(inst.name)
        } else rest.push(inst)
      }
      if (rest.length === pending.length) {
        // A cycle: nothing left can go first. They pose in insertion order and
        // each reads the other's previous frame, which is the best a cycle gets.
        console.warn(`[reze] attachment cycle: ${rest.map((r) => r.name).join(" → ")}`)
        order.push(...rest)
        break
      }
      pending = rest
    }
    this.updateOrder = order
    this.updateOrderDirty = false
    return order
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
    // An attached model is placed by its parent's bone; its own position and
    // rotation are held at identity so the ride is the whole placement (see
    // setModelParent). Scale and visibility are still its own.
    if (transform.position && !inst.parent) model.setPosition(transform.position)
    if (transform.rotation && !inst.parent) model.setRotation(transform.rotation)
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

  /** Show the transform gizmo on the selected bone. On by default. */
  setGizmoEnabled(on: boolean): void {
    this.gizmoEnabled = on
  }

  /** A pointer-driven preview of a pick, not a pick itself — see pickMaterial
   *  for the click that actually selects one. Cheap: a field write, nothing
   *  rebuilt, safe to call every frame the pointer is over the canvas. */
  setHoveredMaterial(modelName: string | null, materialName: string | null): void {
    this.hoverMaterial = modelName && materialName ? { modelName, materialName } : null
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

  // ─── Editor overlays ───────────────────────────────────────────────
  //
  // Two ways in. setOverlay takes a list and draws exactly that list, so a host
  // can paste one in, hand one to a test, or print one back out. The three live
  // layers name a model instead and are rebuilt from its pose every frame,
  // which is the only way a skeleton overlay can be right on an animated model.

  /**
   * Replace one named layer of overlay primitives. World space, drawn as given
   * until it is replaced. An empty list removes the layer.
   */
  setOverlay(layer: string, primitives: OverlayPrimitive[]): void {
    if (primitives.length === 0) this.overlayLayers.delete(layer)
    else this.overlayLayers.set(layer, primitives)
  }

  /** Drop one named layer, or every one. Live layers keep drawing. */
  clearOverlay(layer?: string): void {
    if (layer === undefined) this.overlayLayers.clear()
    else this.overlayLayers.delete(layer)
  }

  /**
   * The bone whose marker is nearest a point on the canvas, or null.
   *
   * On the CPU, and exact. A few hundred bones with known world positions is a
   * loop, not a render pass — and having the answer synchronously is what makes
   * cycling through overlapping bones possible at all. Only VERTICES justify GPU
   * picking, at tens of thousands.
   *
   * `x`/`y` are CSS pixels relative to the canvas, which is what a MouseEvent
   * gives once getBoundingClientRect is subtracted.
   *
   * It projects boneMarkerPositions, the same points the overlay draws markers
   * at, so the hit box cannot drift away from the circle you are aiming at.
   */
  pickBone(
    x: number,
    y: number,
    options: { radiusPx?: number; modelName?: string } = {},
  ): { modelName: string; boneName: string; boneIndex: number } | null {
    if (!this.camera) return null
    const width = this.canvas.clientWidth
    const height = this.canvas.clientHeight
    if (width <= 0 || height <= 0) return null
    const vp = this.camera.getProjectionMatrix().multiply(this.camera.getViewMatrix()).values

    let best: { modelName: string; boneName: string; boneIndex: number } | null = null
    let bestDist = options.radiusPx ?? 14
    let bestDepth = Infinity

    for (const inst of this.modelInstances.values()) {
      if (options.modelName !== undefined && inst.name !== options.modelName) continue
      if (inst.isStage || inst.isPlane || inst.isProp) continue
      const bones = inst.model.getSkeleton().bones
      this.bonePickScratch = boneMarkerPositions(inst.model, this.bonePickScratch)
      const pos = this.bonePickScratch
      for (let i = 0; i < bones.length; i++) {
        const px = pos[i * 3]
        const py = pos[i * 3 + 1]
        const pz = pos[i * 3 + 2]
        const cw = vp[3] * px + vp[7] * py + vp[11] * pz + vp[15]
        if (cw <= 1e-6) continue // behind the camera
        const cx = vp[0] * px + vp[4] * py + vp[8] * pz + vp[12]
        const cy = vp[1] * px + vp[5] * py + vp[9] * pz + vp[13]
        const sx = ((cx / cw) * 0.5 + 0.5) * width
        const sy = (1 - ((cy / cw) * 0.5 + 0.5)) * height
        const d = Math.hypot(sx - x, sy - y)
        if (d > bestDist) continue
        // Within a couple of pixels the two are the same click, and MMD stacks
        // control bones on one point — so the nearer bone takes it.
        if (d < bestDist - 2 || cw < bestDepth) {
          best = { modelName: inst.name, boneName: bones[i].name, boneIndex: i }
          bestDist = d
          bestDepth = cw
        }
      }
    }
    return best
  }

  /** Skinned positions for picking, grown on demand. One click's worth of work
   *  reused across clicks — a model's vertex count does not change. */
  private materialPickScratch: Float32Array | null = null

  /**
   * The material under a point on the canvas, or null for a miss.
   *
   * On the CPU, like pickBone, and for the same reason: a click (or a hover) is
   * rare and an answer you have synchronously is worth more than one that
   * arrives a frame later. Tens of thousands of triangles is a loop that costs
   * a few milliseconds ONCE, against a GPU id pass that costs an attachment and
   * a readback every frame whether anyone is pointing at the model or not.
   *
   * Skinned on the CPU with getSkinMatrices — the same matrices the vertex
   * shader uses — so the pick lands on the POSED mesh. Bind-pose geometry would
   * be right on a T-posed model and wrong on every animated one, which is
   * exactly when someone is clicking around a costume.
   *
   * Morph offsets are NOT applied: they move a face, never move it into another
   * material, and reading them back per click would cost more than the pick.
   *
   * `x`/`y` are CSS pixels relative to the canvas, as pickBone takes them.
   */
  pickMaterial(
    x: number,
    y: number,
    options: { modelName?: string } = {},
  ): { modelName: string; materialName: string; materialIndex: number } | null {
    if (!this.camera) return null
    const width = this.canvas.clientWidth
    const height = this.canvas.clientHeight
    if (width <= 0 || height <= 0) return null
    const vp = this.camera.getProjectionMatrix().multiply(this.camera.getViewMatrix()).values

    let best: { modelName: string; materialName: string; materialIndex: number } | null = null
    let bestDepth = Infinity

    for (const inst of this.modelInstances.values()) {
      if (options.modelName !== undefined && inst.name !== options.modelName) continue
      if (inst.isStage || inst.isPlane || inst.isProp) continue
      const model = inst.model
      const { positions } = model.getGeometry()
      const count = positions.length / 3
      const { joints, weights } = model.getSkinning()
      const skin = model.getSkinMatrices()

      // Project every vertex ONCE into screen x, y and clip w. The triangle
      // loop then reads three of these rather than re-skinning shared vertices
      // — a closed mesh uses each vertex about six times.
      if (!this.materialPickScratch || this.materialPickScratch.length !== count * 3) {
        this.materialPickScratch = new Float32Array(count * 3)
      }
      const proj = this.materialPickScratch
      for (let v = 0; v < count; v++) {
        const bx = positions[v * 3]
        const by = positions[v * 3 + 1]
        const bz = positions[v * 3 + 2]
        let px = 0
        let py = 0
        let pz = 0
        for (let k = 0; k < 4; k++) {
          const w = weights[v * 4 + k] / 255
          if (w === 0) continue
          const m = joints[v * 4 + k] * 16
          px += w * (skin[m] * bx + skin[m + 4] * by + skin[m + 8] * bz + skin[m + 12])
          py += w * (skin[m + 1] * bx + skin[m + 5] * by + skin[m + 9] * bz + skin[m + 13])
          pz += w * (skin[m + 2] * bx + skin[m + 6] * by + skin[m + 10] * bz + skin[m + 14])
        }
        const cw = vp[3] * px + vp[7] * py + vp[11] * pz + vp[15]
        proj[v * 3 + 2] = cw
        if (cw <= 1e-6) continue
        const cx = vp[0] * px + vp[4] * py + vp[8] * pz + vp[12]
        const cy = vp[1] * px + vp[5] * py + vp[9] * pz + vp[13]
        proj[v * 3] = ((cx / cw) * 0.5 + 0.5) * width
        proj[v * 3 + 1] = (1 - ((cy / cw) * 0.5 + 0.5)) * height
      }

      // Point-in-triangle in SCREEN space, nearest w wins. The same projection
      // pickBone uses, so the two agree about where things are, and it needs no
      // inverse view-projection to build a ray from.
      const indices = model.getIndices()
      const materials = model.getMaterials()
      let m = 0
      let matEnd = materials.length > 0 ? materials[0].vertexCount : indices.length
      for (let i = 0; i + 2 < indices.length; i += 3) {
        while (i >= matEnd && m + 1 < materials.length) {
          m++
          matEnd += materials[m].vertexCount
        }
        const a = indices[i] * 3
        const b = indices[i + 1] * 3
        const c = indices[i + 2] * 3
        if (proj[a + 2] <= 1e-6 || proj[b + 2] <= 1e-6 || proj[c + 2] <= 1e-6) continue
        const ax = proj[a]
        const ay = proj[a + 1]
        const bx = proj[b]
        const by = proj[b + 1]
        const cx2 = proj[c]
        const cy2 = proj[c + 1]
        // Barycentric sign test, both windings: PMX faces are one winding but a
        // double-sided material is legitimately seen from behind.
        const d1 = (x - bx) * (ay - by) - (ax - bx) * (y - by)
        const d2 = (x - cx2) * (by - cy2) - (bx - cx2) * (y - cy2)
        const d3 = (x - ax) * (cy2 - ay) - (cx2 - ax) * (y - ay)
        const neg = d1 < 0 || d2 < 0 || d3 < 0
        const pos = d1 > 0 || d2 > 0 || d3 > 0
        if (neg && pos) continue
        const depth = (proj[a + 2] + proj[b + 2] + proj[c + 2]) / 3
        if (depth >= bestDepth) continue
        bestDepth = depth
        best = { modelName: inst.name, materialName: materials[m].name, materialIndex: m }
      }
    }
    return best
  }

  /** Draw an octahedron per bone of `modelName`, rebuilt each frame. Null off. */
  setBoneOverlay(modelName: string | null, options: BoneOverlayOptions = {}): void {
    this.overlayBones = modelName ? { modelName, options } : null
  }

  /** Draw every rigidbody of `modelName` where the simulation has it, rebuilt
   *  each frame. Null off. */
  setRigidbodyOverlay(modelName: string | null, options: RigidbodyOverlayOptions = {}): void {
    this.overlayBodies = modelName ? { modelName, options } : null
  }

  /** Draw a cross per joint of `modelName` plus dashed lines to the bodies it
   *  holds together, rebuilt each frame. Null off. */
  setJointOverlay(modelName: string | null, options: JointOverlayOptions = {}): void {
    this.overlayJoints = modelName ? { modelName, options } : null
  }

  /**
   * Draw `modelName`'s mesh as a wireframe — its vertices and its topology.
   *
   * Skinned on the GPU from the model's own vertex buffer and skin matrices, so
   * it sits on the POSED mesh. The loader's CPU-side positions are bind pose: a
   * wireframe built from those looks right on a T-posed model and slides off
   * every animated one, which is exactly the state a user is in while looking at
   * weights.
   *
   * The edge list is deduplicated and built once, on the first frame this is on.
   *
   * `material` narrows the wireframe to one material's faces. The mesh still
   * writes depth in full, so the material reads as part of the body rather than
   * as a shell floating in front of it — which is the point of scoping it: you
   * are asking where this material's faces ARE, and an answer that ignores the
   * torso in front of them is not one.
   */
  setVertexOverlay(
    modelName: string | null,
    options: { xray?: boolean; material?: string | null } = {},
  ): void {
    this.overlayVertices = modelName
      ? { modelName, xray: options.xray ?? false, material: options.material ?? null }
      : null
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

  /**
   * Push a colour/shading edit straight into a material's own uniform buffer —
   * the same block createMaterialUniformBuffer wrote at load, offset for
   * offset. A single write per call, on the fields that actually moved, so
   * dragging one slider does not touch the other eleven.
   *
   * UNGROUPED materials only. A grouped material renders through its style
   * group's own compiled graph (setupPipelines' neutral/DEFAULT_GRAPH path is
   * what reads this buffer, and a grouped material never runs it) — the call
   * still writes the bytes, they are simply never sampled, which would look
   * like the edit silently failing. Callers check groupsByModel first and this
   * quietly no-ops rather than assume that check was made, since a document
   * edit landing to the WRONG channel (the group's own colour input) would be
   * a worse failure than one that does nothing.
   *
   * Structural fields — anything that changes which draw bucket a material is
   * in, or which texture it binds — are NOT here: alpha crossing the 1.0
   * opaque/transparent line, edge on/off, a texture swap, all need the draw
   * list or the bind group rebuilt, not a uniform write. Those get their own
   * call when something needs them.
   */
  setMaterialUniforms(
    modelName: string,
    materialName: string,
    patch: {
      diffuse?: readonly [number, number, number, number]
      specular?: readonly [number, number, number]
      specularPower?: number
      ambient?: readonly [number, number, number]
    },
  ): boolean {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return false
    const materials = inst.model.getMaterials()
    const index = materials.findIndex((m) => m.name === materialName)
    if (index < 0) return false
    const buffer = inst.materialUniformBuffers[index]
    if (!buffer) return false
    if (patch.diffuse) {
      this.device.queue.writeBuffer(buffer, 0, new Float32Array(patch.diffuse))
    }
    if (patch.ambient) {
      this.device.queue.writeBuffer(buffer, 16, new Float32Array(patch.ambient))
    }
    if (patch.specularPower !== undefined) {
      this.device.queue.writeBuffer(buffer, 28, new Float32Array([patch.specularPower]))
    }
    if (patch.specular) {
      this.device.queue.writeBuffer(buffer, 32, new Float32Array(patch.specular))
    }
    return true
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

  /**
   * Run the solver, or stop it.
   *
   * Turning it OFF snaps every body back onto its bone. Merely halting the step
   * leaves hair and skirts hanging wherever the simulation happened to be — a
   * pose nothing in the document describes, which is the opposite of what "off"
   * is asked for: you switch physics off to see what the RIG does, and a frozen
   * mid-swing is still the solver's answer, just a stale one.
   */
  setPhysicsEnabled(enabled: boolean): void {
    if (this.physicsEnabled === enabled) return
    this.physicsEnabled = enabled
    if (enabled) return
    for (const inst of this.modelInstances.values()) {
      if (!inst.physics) continue
      inst.physics.reset(inst.model.getWorldMatrices())
    }
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

  /**
   * Whether cloth and hair land on a floor at each figure's own feet.
   *
   * ON is what a standing character wants: the floor is at her model-space
   * y = 0, so a long skirt or hair reaching the ground rests on it instead of
   * passing through. That same plane travels with her, which is what makes this
   * a switch — lift her onto a stage, hang her in the air, carry her up with
   * root motion, and everything that should now fall past her feet piles up on
   * a surface nothing in the scene is standing on.
   */
  setPhysicsFloor(on: boolean): void {
    this.physicsFloor = on
    this.forEachInstance((inst) => inst.physics?.setFloor(on))
  }

  getPhysicsFloor(): boolean {
    return this.physicsFloor
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
    for (const inst of this.instancesInUpdateOrder()) {
      const tAnim = performance.now()
      // An attached model is placed from its parent's bone as posed and
      // simulated THIS frame — the order guarantees the parent came first —
      // and only then posed itself, so its clip and physics ride the placement.
      const attached = inst.parent !== null
      if (attached) this.placeAttached(inst)
      // A stage never solves IK — nothing drives its chains — and skips the pose
      // pass entirely while it is idle. Morph changes still come through, since
      // that is the one thing a stage's controls do move. A prop idles the same
      // way while it stands on its own; hung from a hand it moves every frame.
      const stageIdle = (inst.isStage || inst.isPlane || inst.isProp) && !attached && inst.model.isIdle()
      let verticesChanged = false
      if (!stageIdle) {
        verticesChanged = inst.model.update(deltaTime, inst.isStage || inst.isPlane ? false : this.ikEnabled)
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
        // The step published new world matrices for the simulated bones; the
        // bones that INHERIT from them are still wearing the animated pose.
        // Returns immediately unless this rig actually has such a bone.
        inst.model.applyPhysicsAppend()
        physicsMs += performance.now() - tPhys
      }
      if (inst.vertexBufferNeedsUpdate) this.updateVertexBuffer(inst)
    }
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
        pass = encoder.beginComputePass({ label: "morph compute", timestampWrites: this.stamps("morph") })
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
    // A grow reallocates the argument buffers, and a bundle holds the buffer it
    // recorded against by reference — a stale one would draw from freed memory.
    this.bundlesDirty = true
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
    if (drawCount === 0 || modelCount === 0) {
      this.releaseCullBuffers()
      return
    }
    this.cullRebuilds++

    // Reuse the buffers whenever they are big enough, and rewrite their contents
    // instead. This path is the COMMON one, not an optimisation for a rare case:
    // every style-group compile re-sorts a model's draws, which reorders the
    // list without changing its length, and those compiles land one after
    // another over the first frames after a scene loads. Reallocating five GPU
    // buffers and a bind group on each of them put the churn exactly where a
    // scene is least able to afford it — the frames the viewer is watching
    // appear.
    const fits = this.cullCapacity >= drawCount && this.cullModelCapacity >= modelCount && this.cullBindGroup !== null
    if (!fits) {
      this.releaseCullBuffers()
      this.cullCapacity = drawCount
      this.cullModelCapacity = modelCount
    }

    const cap = this.cullCapacity
    if (!fits) {
      this.cullMetaBytes = new ArrayBuffer(cap * Engine.CULL_META_BYTES)
      this.cullMetaF32 = new Float32Array(this.cullMetaBytes)
      this.cullMetaU32 = new Uint32Array(this.cullMetaBytes)
      this.cullArgs = new Uint32Array(cap * Engine.CULL_ARG_WORDS)
      this.cullHidden = new Uint32Array(cap)
      this.cullReference = new Uint8Array(cap)
      const modelBytes = new ArrayBuffer(this.cullModelCapacity * Engine.CULL_MODEL_FLOATS * 4)
      this.cullModelData = new Float32Array(modelBytes)
      this.cullModelFlags = new Uint32Array(modelBytes)
    }
    this.cullReferenceFrame = -1
    const args = this.cullArgs
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
      //
      // instanceCount seeds to 1, not 0: the passes draw from this buffer, so
      // the value it holds when the compute has NOT run is what renders. One is
      // the whole scene unculled; zero would be an empty screen, and every way
      // the compute can fail to run — a shader that would not compile, a frame
      // encoded before the first dispatch — would present as everything gone.
      const a = i * Engine.CULL_ARG_WORDS
      args[a] = draw.count
      args[a + 1] = 1
      args[a + 2] = draw.firstIndex
      args[a + 3] = 0
      args[a + 4] = 0
    }

    if (!fits) {
      const store = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      this.cullMetaBuffer = this.device.createBuffer({
        label: "cull draw metadata",
        size: this.cullMetaBytes.byteLength,
        usage: store,
      })
      this.cullModelBuffer = this.device.createBuffer({
        label: "cull model records",
        size: this.cullModelData.byteLength,
        usage: store,
      })
      this.cullHiddenBuffer = this.device.createBuffer({
        label: "cull per-draw hidden",
        size: Math.max(4, this.cullHidden.byteLength),
        usage: store,
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
      this.cullMirrorArgs = this.device.createBuffer({
        label: "cull mirror indirect args",
        size: args.byteLength,
        usage: argUsage,
      })
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
          { binding: 5, resource: { buffer: this.cullHiddenBuffer } },
          { binding: 6, resource: { buffer: this.cullMirrorArgs } },
        ],
      })
    }

    // Contents, every rebuild — the reuse above is about not reallocating, not
    // about skipping the upload: a re-sort changes which draw sits in which slot
    // without changing how many there are.
    this.device.queue.writeBuffer(this.cullMetaBuffer!, 0, this.cullMetaBytes)
    this.device.queue.writeBuffer(this.cullCameraArgs!, 0, args.buffer as ArrayBuffer)
    this.device.queue.writeBuffer(this.cullShadowArgs!, 0, args.buffer as ArrayBuffer)
    this.device.queue.writeBuffer(this.cullMirrorArgs!, 0, args.buffer as ArrayBuffer)
    // Seeded so the first frame is not a frame of everything hidden.
    this.writeCullHidden(true)
  }

  /**
   * Per-draw hidden state, uploaded only when it actually changes.
   *
   * The two sets behind it are cheap to read but the buffer is not worth
   * rewriting every frame: applyMaterialMorphs rebuilds morphHiddenMaterials on
   * every frame of any character carrying a face VMD, and almost every one of
   * those rebuilds produces the same answer. So compare, then upload.
   */
  private writeCullHidden(force = false): void {
    if (!this.cullHiddenBuffer || this.cullDraws.length === 0) return
    const out = this.cullHidden
    let changed = force
    for (let i = 0; i < this.cullDraws.length; i++) {
      const { inst, draw } = this.cullDraws[i]
      const v =
        inst.hiddenMaterials.has(draw.materialName) || inst.morphHiddenMaterials.has(draw.materialName) ? 1 : 0
      if (out[i] !== v) {
        out[i] = v
        changed = true
      }
    }
    if (changed) this.device.queue.writeBuffer(this.cullHiddenBuffer, 0, out.buffer as ArrayBuffer)
  }

  private releaseCullBuffers(): void {
    this.cullMetaBuffer?.destroy()
    this.cullModelBuffer?.destroy()
    this.cullHiddenBuffer?.destroy()
    this.cullCameraArgs?.destroy()
    this.cullShadowArgs?.destroy()
    this.cullMirrorArgs?.destroy()
    this.cullMetaBuffer = null
    this.cullModelBuffer = null
    this.cullHiddenBuffer = null
    this.cullCapacity = 0
    this.cullModelCapacity = 0
    this.cullCameraArgs = null
    this.cullShadowArgs = null
    this.cullMirrorArgs = null
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
    this.updateCasterSphere(data)
  }

  /**
   * One sphere containing every shadow caster in the scene, for the ground.
   *
   * The ground's PCF is the most expensive thing in the frame on a tile-based
   * GPU — nine hardware-bilinear comparisons per pixel on a full-coverage draw,
   * which is what 0.33.2 was about and what a second cascade quietly undid. But
   * the floor is vastly larger than the thing standing on it, and a pixel the
   * character cannot possibly shadow does not need to ask the shadow map: the
   * answer is lit, and nine taps is an expensive way to spell it.
   *
   * So the ground gets a bound and tests against it in ALU. This reuses the
   * spheres the cull already builds every frame — an AABB over POSED bone
   * positions grown by the skin margin, which its own note calls a bound rather
   * than an estimate, so a jump or a physics-driven skirt is inside it by
   * construction. Union, not per model: one sphere is one test, and the ground
   * shader must not loop over the cast.
   *
   * A RIGID caster (a stage) leaves its cull sphere zeroed deliberately — the
   * cull reads its boxes instead — so any rigid model disables this entirely by
   * setting radius to -1. Wrong here is a missing shadow, and a scene with a
   * stage keeps the taps rather than risk one.
   */
  private updateCasterSphere(data: Float32Array): void {
    const out = this.casterSphere
    out[3] = 0
    let cx = 0
    let cy = 0
    let cz = 0
    let r = 0
    let any = false
    for (let i = 0; i < this.cullModels.length; i++) {
      const inst = this.cullModels[i]
      if (!inst.model.visible || inst.shadowDrawCalls.length === 0) continue
      if (inst.rigid) {
        // No sphere to read. Bail out of the whole optimisation.
        out[3] = -1
        return
      }
      const o = i * Engine.CULL_MODEL_FLOATS + 16
      const x = data[o]
      const y = data[o + 1]
      const z = data[o + 2]
      const rad = data[o + 3]
      if (rad <= 0) continue
      if (!any) {
        cx = x
        cy = y
        cz = z
        r = rad
        any = true
        continue
      }
      // Union of two spheres, the standard construction: if one already contains
      // the other keep it, else grow along the line between the centres.
      const dx = x - cx
      const dy = y - cy
      const dz = z - cz
      const d = Math.hypot(dx, dy, dz)
      if (d + rad <= r) continue
      if (d + r <= rad) {
        cx = x
        cy = y
        cz = z
        r = rad
        continue
      }
      const nr = (d + r + rad) * 0.5
      const t = (nr - r) / d
      cx += dx * t
      cy += dy * t
      cz += dz * t
      r = nr
    }
    out[0] = cx
    out[1] = cy
    out[2] = cz
    out[3] = any ? r : 0
  }

  /** Every shadow caster in one sphere: (x, y, z, radius). radius 0 = nothing
   *  casts, -1 = do not use (a rigid caster has no sphere). See updateCasterSphere. */
  /** A card's own texture, by model key — see setPlaneFrame. */
  private planeTextures = new Map<string, GPUTexture>()
  private casterSphere = new Float32Array(4)

  /** The ground's uniform block, kept so the caster sphere can be refreshed in
   *  it every frame rather than rebuilding the buffer (addGround allocates). */
  private groundMaterialData: Float32Array | null = null

  /**
   * Push this frame's caster sphere into the ground's uniform.
   *
   * Four floats, one writeBuffer, and only while a ground exists. Rebuilding the
   * block the way addGround does would allocate a buffer and a bind group per
   * frame, which is the cost this is trying to remove rather than a way to pay
   * it somewhere else.
   */
  private writeGroundCasterSphere(): void {
    const gb = this.groundMaterialData
    if (!gb || !this.groundShadowMaterialBuffer) return
    if (gb[20] === this.casterSphere[0] && gb[21] === this.casterSphere[1] &&
        gb[22] === this.casterSphere[2] && gb[23] === this.casterSphere[3]) return
    gb.set(this.casterSphere, 20)
    this.device.queue.writeBuffer(this.groundShadowMaterialBuffer, 80, this.casterSphere as Float32Array<ArrayBuffer>)
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
   * casters that stand outside the volume and throw shade into it. It cannot:
   * the cull tests the OUTERMOST cascade's box, each cascade's rasterizer clips
   * to its own box, and every inner box lies inside the outer one (the
   * containment invariant in shadow-cascades.ts) — so anything rejected here
   * was contributing to no cascade at all.
   */
  private writeCullFrusta(): void {
    if (!this.cullFrustaBuffer) return
    // cameraMatrixData holds view at 0 and projection at 16 — already written
    // this frame by updateCameraUniforms.
    Mat4.multiplyArrays(this.cameraMatrixData, 16, this.cameraMatrixData, 0, this.cullScratchVp, 0)
    writeFrustumPlanes(this.cullScratchVp, this.cullFrustaF32, 0)
    // The OUTERMOST cascade: it contains every inner one (the containment
    // invariant in shadow-cascades.ts), so its six planes are the union and the
    // rasterizer clips each cascade to its own box — the argument that made
    // single-volume shadow culling exact, kept true for a list.
    writeFrustumPlanes(this.shadowLightVPMatrix.subarray(16 * (SHADOW_CASCADES.length - 1)), this.cullFrustaF32, 24)
    // The mirror pass sees the CAMERA frustum reflected about the floor plane:
    // a close-up of the floor shows a reflection whose owner is out of frame,
    // so the camera args would cull her out of her own mirror. When no
    // reflection is active the camera planes stand in, keeping the args sane
    // for bundles that never execute.
    if (this.reflectionActive) {
      // Computed by updateMirrorCamera, which ran before the cull this frame.
      writeFrustumPlanes(this.mirrorVPData, this.cullFrustaF32, 48)
    } else {
      this.cullFrustaF32.copyWithin(48, 0, 24)
    }
    this.cullFrustaU32[72] = this.cullDraws.length
    this.cullFrustaU32[73] = this.cullEnabled ? 1 : 0
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
  /**
   * Record the three bundles: the shadow pass, the opaque phase and the
   * transparent phase.
   *
   * The insight this rests on is that a character's DRAW COMMANDS are stable
   * frame to frame — same pipeline, same bind groups, same index ranges, with
   * only the contents of the skin-matrix buffer changing. So bundles apply to
   * the cast, not merely to scenery, and what invalidates one is scene
   * STRUCTURE. Animation, physics, camera movement, material morphs, a hidden
   * material and a hidden model all leave a bundle valid — the first three
   * because they touch buffers and not commands, the last two because their
   * switches live in the cull compute rather than in this encode loop.
   *
   * The ground and the particles are deliberately not in a bundle: the ground is
   * one draw that sits BETWEEN the two phases, and the particle count comes from
   * the CPU. Both are cheaper to leave direct than to invalidate around.
   */
  private recordBundles(): void {
    this.bundlesDirty = false
    const scene = {
      colorFormats: sceneColorFormats(this.sceneFormats),
      depthStencilFormat: this.depthFormat,
      sampleCount: Engine.MULTISAMPLE_COUNT,
    }
    if (this.modelInstances.size === 0) {
      this.opaqueBundle = null
      this.mirrorOpaqueBundle = null
      this.mirrorTransparentBundle = null
      this.shadowBundles = []
      return
    }

    const camView = this.sceneView("camera")
    const opaque = this.device.createRenderBundleEncoder({ label: "opaque phase", ...scene })
    this.forEachInstance((inst) => this.renderModelOpaquePhase(opaque, inst, camView))
    this.opaqueBundle = opaque.finish({ label: "opaque phase" })

    // NO camera transparent bundle. The camera pass draws that phase directly —
    // see the note at the executeBundles call for what recording one cost on
    // WebKit. Recording it anyway "in case" is not free and not harmless: it is
    // work on every rebuild, and a live bundle beside a direct draw of the same
    // phase is an invitation to execute it again.
    //
    // The MIRROR pair below keeps both bundles, and is allowed to: that pass
    // hands them to a single executeBundles with nothing direct in between,
    // which is the pattern that works.
    const mirrorView = this.sceneView("mirror")
    const mo = this.device.createRenderBundleEncoder({ label: "mirror opaque phase", ...scene })
    this.forEachInstance((inst) => this.renderModelOpaquePhase(mo, inst, mirrorView))
    this.mirrorOpaqueBundle = mo.finish({ label: "mirror opaque phase" })

    const mt = this.device.createRenderBundleEncoder({ label: "mirror transparent phase", ...scene })
    this.forEachInstance((inst) => this.renderModelTransparentPhase(mt, inst, mirrorView))
    this.mirrorTransparentBundle = mt.finish({ label: "mirror transparent phase" })

    // One bundle per cascade: the draws are identical — same pipeline, same
    // indirect args culled to the OUTERMOST volume — and only bind group 0
    // (which cascade's view-projection) differs. Each cascade's rasterizer
    // clips the shared list to its own box.
    this.shadowBundles = SHADOW_CASCADES.map((_, ci) => {
      const shadow = this.device.createRenderBundleEncoder({
        label: `shadow pass, cascade ${ci}`,
        colorFormats: [],
        depthStencilFormat: Engine.SHADOW_DEPTH_FORMAT,
      })
      shadow.setPipeline(this.shadowDepthPipeline)
      this.forEachInstance((inst) => this.drawInstanceShadow(shadow, inst, ci))
      return shadow.finish({ label: `shadow pass, cascade ${ci}` })
    })
    this.bundleRecords++
  }

  /** The two query slots for a pass, or undefined where the device has no
   *  timestamps — which every pass descriptor accepts as "do not measure". */
  private stamps(pass: (typeof Engine.TIMED_PASSES)[number]): GPURenderPassTimestampWrites | undefined {
    if (!this.timestampQuerySet) return undefined
    const i = Engine.TIMED_PASSES.indexOf(pass)
    return { querySet: this.timestampQuerySet, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 }
  }

  /**
   * Half a stamp, for a component that is several passes rather than one.
   *
   * Bloom is nine render passes — a prefilter blit, a downsample chain and an
   * upsample chain — and what anyone wants to know is what the PYRAMID cost, not
   * what its fourth mip cost. Both fields of GPURenderPassTimestampWrites are
   * optional, so the opening query goes on the first pass and the closing one on
   * the last, and the pair reads as one span across everything between.
   */
  private stampOpen(pass: (typeof Engine.TIMED_PASSES)[number]): GPURenderPassTimestampWrites | undefined {
    if (!this.timestampQuerySet) return undefined
    return { querySet: this.timestampQuerySet, beginningOfPassWriteIndex: Engine.TIMED_PASSES.indexOf(pass) * 2 }
  }

  private stampClose(pass: (typeof Engine.TIMED_PASSES)[number]): GPURenderPassTimestampWrites | undefined {
    if (!this.timestampQuerySet) return undefined
    return { querySet: this.timestampQuerySet, endOfPassWriteIndex: Engine.TIMED_PASSES.indexOf(pass) * 2 + 1 }
  }

  /**
   * Resolve this frame's timings and start a readback, at most one in flight.
   *
   * Deliberately not awaited anywhere in the frame: a timing that costs a stall
   * to collect would change the thing it is measuring. The numbers are therefore
   * a frame or two old, which is exactly right for what they are for — watching
   * a pass get more expensive across a refactor, not attributing one frame.
   */
  private resolveTimestamps(encoder: GPUCommandEncoder): void {
    const qs = this.timestampQuerySet
    if (!qs || !this.timestampResolve || !this.timestampRead) return
    // Nobody has asked. See getGpuTimings — the read is what enrols.
    if (!this.timestampsWanted) return
    const count = Engine.TIMED_PASSES.length * 2
    encoder.resolveQuerySet(qs, 0, count, this.timestampResolve, 0)
    if (this.timestampBusy) return
    encoder.copyBufferToBuffer(this.timestampResolve, 0, this.timestampRead, 0, count * 8)
    this.timestampBusy = true
    // After the submit this encoder belongs to, which is why the map is started
    // from a microtask rather than here.
    queueMicrotask(() => {
      const buf = this.timestampRead
      if (!buf) return
      buf
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          const t = new BigInt64Array(buf.getMappedRange().slice(0))
          buf.unmap()
          const out: Record<string, number> = {}
          for (let i = 0; i < Engine.TIMED_PASSES.length; i++) {
            // Nanoseconds, and a pass that did not run leaves its pair equal —
            // report 0 rather than a negative from an unwritten query.
            const ns = Number(t[i * 2 + 1] - t[i * 2])
            out[Engine.TIMED_PASSES[i]] = ns > 0 ? ns / 1e6 : 0
          }
          this.gpuPassMs = out
          this.timestampBusy = false
        })
        .catch(() => {
          // Device lost, or the buffer was destroyed under us. Stop reporting
          // rather than wedging the flag and never reading again.
          this.timestampBusy = false
        })
    })
  }

  /**
   * Milliseconds on the GPU per pass, or null where the device cannot measure.
   *
   * The regression guard for the draw-path work: these are the numbers that say
   * whether restructuring cost anything, which is the claim being made — not
   * whether it made the scene faster, which was never the goal.
   *
   * ASKING IS WHAT TURNS IT ON. The first call to this enrols the engine in the
   * per-frame readback; until then resolveTimestamps does nothing. That is why
   * the first call returns null even on a device that can measure — the answer
   * arrives a frame or two later, which is already true of these numbers and
   * documented on resolveTimestamps.
   *
   * The alternative was what this used to do: resolve the query set, copy it to
   * a staging buffer and map that buffer, every frame, on every device, for a
   * reader that in this codebase did not exist. A map is a synchronisation point
   * and the whole path is instrumentation — paying for it unasked is the same
   * mistake as shipping a debug flag, only invisible.
   */
  getGpuTimings(): Record<string, number> | null {
    this.timestampsWanted = true
    return this.gpuPassMs
  }

  /** Set by the first getGpuTimings() call. See it for why asking is the switch. */
  private timestampsWanted = false

  private dispatchCull(encoder: GPUCommandEncoder): void {
    if (this.cullListDirty) this.rebuildCullList()
    if (!this.cullBindGroup || this.cullDraws.length === 0) return
    // The CPU half runs even with a dead pipeline: it is what setCullApply gates
    // on, and a stale mirror would gate draws against last-known frusta.
    this.writeCullModels()
    this.writeCullFrusta()
    this.writeCullHidden()
    this.cullFrame++
    if (!this.cullPipeline) return
    const pass = encoder.beginComputePass({ label: "cull", timestampWrites: this.stamps("cull") })
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
      if (this.cullHidden[i] !== 0) {
        out[i] = 0
        continue
      }
      if (!this.cullEnabled) {
        // Mirror the compute's own bypass, or every draw would read as a
        // disagreement the moment culling is switched off for an A/B.
        out[i] = 3
        continue
      }
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

  /**
   * Issue one material draw, indirect when the cull owns it.
   *
   * The outline and the over-eyes pass share their material's slot rather than
   * getting their own: same index range, same bounds, so the same decision. That
   * also means an outline can never survive a material that was culled, which is
   * the only relationship between them that is ever correct.
   *
   * Falls back to a direct draw for anything outside the cull list — the ground,
   * and any draw whose slot has not been assigned yet.
   */
  private issueDraw(
    pass: GPURenderPassEncoder | GPURenderBundleEncoder,
    draw: DrawCall,
    kind: "camera" | "shadow" | "mirror",
  ): void {
    const args =
      kind === "shadow" ? this.cullShadowArgs : kind === "mirror" ? this.cullMirrorArgs : this.cullCameraArgs
    if (args && draw.cullIndex >= 0) {
      pass.drawIndexedIndirect(args, draw.cullIndex * Engine.CULL_ARG_WORDS * 4)
    } else {
      pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    }
  }

  /**
   * Turn frustum culling off without turning the pass off.
   *
   * The compute keeps running and keeps reporting; it just writes "visible" for
   * every draw. That is the A/B a missing-geometry report needs — if it is still
   * missing with culling off, the cull was not what removed it — and it costs
   * one uniform word rather than a rebuild.
   */
  setCullEnabled(on: boolean): void {
    this.cullEnabled = on
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
      rebuilds: this.cullRebuilds,
      bundleRecords: this.bundleRecords,
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
    isPlane = false,
    dynamicTexture = false,
    isProp = false,
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
      // STORAGE so the wireframe overlay can skin from it: its quads read two
      // different model vertices per corner, which no vertex stream can supply.
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
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
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE,
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
    const physics = !isStage && !isPlane && rbs.length > 0 ? new RezePhysics(rbs, model.getJoints()) : null
    // Which bones the simulation will overwrite, handed to the pose pipeline so
    // the append (付与) pass can consume the simulated result instead of the
    // animated one. Precomputed here, once, because the answer is topology —
    // see Model.setPhysicsDrivenBones for what it costs when a rig needs it and
    // why it costs nothing when none does.
    if (physics) {
      model.setPhysicsDrivenBones(physics.getPhysicsDrivenBones())
      // The bodies an inherited-from bone rides on are damped less than the
      // rest, so they swing longer WITHOUT hanging lower — see
      // RezePhysics.setJiggleDamping for why damping is the separable knob and
      // solver iterations are not.
      const appendSources = model.getAppendSourceBones()
      if (appendSources.length > 0) physics.setJiggleDamping(appendSources, Engine.JIGGLE_DAMPING_SCALE)
    }
    // Adopt the scene's air, or a model added mid-session would fall under
    // different gravity from the ones already on stage.
    if (physics) {
      physics.setGravity(this.gravity)
      if (this.wind) physics.setWind(this.wind)
      physics.setFloor(this.physicsFloor)
    }

    // One per cascade: the shadow VERTEX shader reads a single matrix, and
    // which one is the only thing that differs between the cascade passes.
    const shadowBindGroups = SHADOW_CASCADES.map((_, ci) =>
      this.device.createBindGroup({
        label: `${name}: shadow bind, cascade ${ci}`,
        layout: this.shadowDepthPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.shadowCascadeVPBuffers[ci] } },
          { binding: 1, resource: { buffer: skinMatrixBuffer } },
          { binding: 2, resource: this.materialSampler },
        ],
      }),
    )

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
      wireEdges: new Map(),
      drawCalls: [],
      shadowDrawCalls: [],
      shadowBindGroups,
      mainPerInstanceBindGroup,
      pickPerInstanceBindGroup,
      pickDrawCalls: [],
      isStage,
      isPlane,
      isProp,
      parent: null,
      dynamicTexture,
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
      objectId: this.modelInstances.size + 1,
      dissolve: 1,
      materialUniformBuffers: [],
      outlineUniformBuffers: [],
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
    this.bundlesDirty = true
    this.updateOrderDirty = true
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

  /** Everything about the ground's pipeline except which shadow variant it
   *  compiles, so the two are built from one description and cannot drift. */
  private groundShadowPipelineDesc!: Omit<Parameters<Engine["createRenderPipeline"]>[0], "shaderModule">

  private buildGroundPipeline(soft: boolean): GPURenderPipeline {
    return this.createRenderPipeline({
      ...this.groundShadowPipelineDesc,
      label: soft ? "ground shadow pipeline (soft)" : "ground shadow pipeline",
      shaderModule: this.device.createShaderModule({
        label: soft ? "ground shadow (soft)" : "ground shadow",
        code: groundShaderWgsl(soft),
      }),
    })
  }

  /** Built on the first frame that actually needs it. A shader compile costs
   *  load time, and the overwhelming majority of scenes never soften a shadow. */
  private ensureGroundSoftPipeline(): GPURenderPipeline {
    if (!this.groundShadowSoftPipeline) this.groundShadowSoftPipeline = this.buildGroundPipeline(true)
    return this.groundShadowSoftPipeline
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
    mirror: boolean
    mirrorBlur: number
    shadowSoftness: number
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
      mirror,
      mirrorBlur,
      shadowSoftness,
    } = opts
    // Shadow map is already created in setupPipelines()
    // 20 floats: 16 for the original block, then (mirrorBlur, pad, pad, pad)
    // keeping the uniform vec4-aligned.
    const gb = new Float32Array(24)
    this.groundMaterialData = gb
    gb[0] = diffuseColor.x
    gb[1] = diffuseColor.y
    gb[2] = diffuseColor.z
    gb[3] = fadeStart
    gb[4] = fadeEnd
    gb[5] = shadowStrength
    gb[6] = 1 / SHADOW_CASCADES[0].mapSize
    gb[7] = gridSpacing
    gb[8] = gridLineWidth
    gb[9] = gridLineOpacity
    gb[10] = noiseStrength
    gb[11] = opacity
    gb[12] = gridLineColor.x
    gb[13] = gridLineColor.y
    gb[14] = gridLineColor.z
    gb[15] = mirror ? 1 : 0
    this.groundMirror = gb[15]
    gb[16] = Math.min(Math.max(mirrorBlur, 0), 1)
    this.groundMirrorBlur = gb[16]
    // gb[18] — shadow edge softness. Was padding; the shader reads it as the
    // Vogel disk's radius, and 0 takes the sharp nine-tap path unchanged.
    gb[18] = Math.min(Math.max(shadowSoftness, 0), 1)
    // Which variant the draw picks. Zero is the sharp shader, which is the one
    // that existed before softness did.
    this.groundSoft = gb[18] > 0
    // gb[17] — does the FAR cascade hold anything?
    //
    // It holds something only when a stage is loaded; that is what it exists for
    // and the cascade loop already skips drawing into it otherwise, leaving it
    // cleared. A cleared depth map compares as "no occluder", so the ground's far
    // branch is nine comparison taps whose answer is known in advance.
    //
    // That branch runs wherever the NEAR cascade does not reach, and the near one
    // is a 64-unit box around the camera target — so on a floor receding to the
    // horizon it is most of the visible pixels, on the most expensive
    // full-coverage draw in the frame. Skipping it is free in the exact sense:
    // the shader takes vis = 1.0, which is what the taps would have returned.
    gb[17] = this.hasStage() ? 1 : 0
    // gb[20..23] — the caster sphere, refreshed every frame by
    // writeGroundCasterSphere. Zero here so a frame that renders before the
    // first cull (there is one) reads "nothing casts" and skips the taps, which
    // is true: no model has been posed yet.
    this.groundShadowMaterialBuffer = this.device.createBuffer({
      size: gb.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.groundShadowMaterialBuffer, 0, gb)
    this.buildGroundBindGroup()
  }

  /**
   * (Re)build the ground's bind group. Its own method because the RESIZE path
   * needs it too: the reflection resolve is recreated at every canvas size,
   * and a bind group holding the old view would sample a destroyed texture on
   * the first resized frame with a mirror on.
   */
  private buildGroundBindGroup(): void {
    if (!this.groundShadowMaterialBuffer) return
    this.groundShadowBindGroup = this.device.createBindGroup({
      label: "ground shadow bind",
      layout: this.groundShadowBindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.lightUniformBuffer } },
        { binding: 2, resource: this.shadowMapDepthViews[0] },
        { binding: 3, resource: this.shadowComparisonSampler },
        { binding: 4, resource: { buffer: this.groundShadowMaterialBuffer } },
        { binding: 5, resource: { buffer: this.shadowLightVPBuffer } },
        { binding: 6, resource: { buffer: this.lightsBuffer } },
        { binding: 7, resource: this.shadowMapDepthViews[SHADOW_CASCADES.length - 1] },
        { binding: 8, resource: { buffer: this.mirrorVPBuffer } },
        // Created in handleResize, which runs during init — before any ground
        // can exist to bind it.
        { binding: 9, resource: this.mirrorColorView! },
        { binding: 10, resource: this.materialSampler },
        { binding: 11, resource: this.mirrorDepthReadView! },
        { binding: 12, resource: this.groundNoiseView },
      ],
    })
    if (this.groundDrawCall) this.groundDrawCall.bindGroup = this.groundShadowBindGroup
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
    // The volumes follow the camera target so a character carried far from the
    // origin by code-driven root motion stays inside the lit frustum. The
    // volume MATH lives in shadow-cascades.ts, where it is testable without a
    // GPU; this method owns only the dirty-tracking and the upload.
    const t = this.camera.target
    const moved =
      Math.abs(t.x - this.shadowCenter.x) > 1e-3 ||
      Math.abs(t.y - this.shadowCenter.y) > 1e-3 ||
      Math.abs(t.z - this.shadowCenter.z) > 1e-3
    if (!this.shadowLightVPDirty && !moved) return
    this.shadowLightVPDirty = false
    this.shadowCenter.setXYZ(t.x, t.y, t.z)

    for (let i = 0; i < SHADOW_CASCADES.length; i++) {
      buildShadowVP(t, this.sun.direction, SHADOW_CASCADES[i], this.shadowLightVPMatrix, i * 16)
      this.device.queue.writeBuffer(this.shadowCascadeVPBuffers[i], 0, this.shadowLightVPMatrix, i * 16, 16)
    }
    this.device.queue.writeBuffer(this.shadowLightVPBuffer, 0, this.shadowLightVPMatrix)
  }

  private async setupMaterialsForInstance(inst: ModelInstance): Promise<void> {
    const model = inst.model
    const materials = model.getMaterials()
    if (materials.length === 0) throw new Error("Model has no materials")
    const textures = model.getTextures()
    const prefix = `${inst.name}: `
    // 1-based so that (0,0) = clear color = "no hit". Minted when the instance
    // was built and READ here rather than derived again: two derivations of one
    // id are two that can disagree, and the pick pass and the id attachment
    // have to name the same object by the same number.
    const modelId = inst.objectId

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

      const materialUniformBuffer = this.createMaterialUniformBuffer(
        prefix + mat.name,
        mat,
        sphereMode,
        headBoneIndex,
        materialId,
        modelId,
      )
      inst.gpuBuffers.push(materialUniformBuffer)
      inst.materialUniformBuffers.push(materialUniformBuffer)
      if (morphedMaterials.has(pmxMaterialIndex)) {
        const base = this.materialUniformData(mat, sphereMode, headBoneIndex, materialId, modelId)
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
          // How much of this material is still there — the hull has to go with
          // the surface it traces. Its OWN copy at its own offset: this buffer
          // is edge data, not the material block, and growing it to reach the
          // block's layout would be 32 bytes of padding per outlined material
          // to carry one float. See RZ_OUTLINE_DISSOLVE_OFFSET.
          1,
          0,
          0,
        ])
        const outlineUniformBuffer = this.createUniformBuffer(`${prefix}outline: ${mat.name}`, materialUniformData)
        inst.gpuBuffers.push(outlineUniformBuffer)
        inst.outlineUniformBuffers.push(outlineUniformBuffer)
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

      // A CARD IS ALWAYS OPAQUE-PHASE, whatever its alpha says.
      //
      // The scene pass runs opaque -> ground -> transparent, and the ground
      // writes depth at every opacity (effects locate the floor by it). Every
      // card qualifies as transparent — a cutout has translucent texels, and a
      // video card starts from a blank sheet that is nothing but — so cards
      // drew after the ground and an INVISIBLE floor rejected them. Turning the
      // ground down for the shadow catcher made pictures disappear into it.
      //
      // Not a workaround: alphaMode "hashed" is alpha-to-coverage, which is the
      // transparency technique built for this phase, and addPlane already sets
      // it. The cost is dithering on a large soft gradient, where MSAA has four
      // coverage levels to spend — a cutout edge, which is what a card usually
      // has, resolves exactly.
      const type: DrawCallType = inst.isPlane ? "opaque" : isTransparent ? "transparent" : "opaque"
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
  private materialUniformData(
    mat: Material,
    sphereMode: number,
    headBoneIndex: number,
    /** This draw's identity for the id attachment. Both 1-based, so 0 stays the
     *  reserved "nothing". NOT defaulted: the material-morph path rebuilds this
     *  whole block from a `base` copy and writes it back, so a base built
     *  without them would blank a material's id for as long as it morphed. */
    materialId: number,
    objectId: number,
  ): Float32Array {
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
    // 13 and 14 are the padding MaterialUniforms already carried, now named:
    // the ids ride the uniform the material binds anyway, so nothing new is
    // bound and the indirect-draw path is untouched.
    data[13] = materialId
    data[14] = objectId
    // 15 is the last of that padding: how much of this material is there. ONE,
    // not zero — the default has to be "whole", or every model would load
    // already gone.
    data[15] = 1
    return data
  }

  private createMaterialUniformBuffer(
    label: string,
    mat: Material,
    sphereMode: number,
    headBoneIndex: number,
    materialId: number,
    objectId: number,
  ): GPUBuffer {
    return this.createUniformBuffer(
      `material uniform: ${label}`,
      this.materialUniformData(mat, sphereMode, headBoneIndex, materialId, objectId),
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

  /** Whether a material is switched on — the user's own toggle, or a material
   *  morph having driven its alpha to zero.
   *
   *  The material passes no longer consult this: their draws are indirect, and
   *  the cull compute zeroes the instance count of anything hidden, which is
   *  what lets a render bundle survive a face VMD rewriting the morph-hidden set
   *  sixty times a second. It remains the answer for the passes that draw
   *  directly, where there is no argument buffer to zero. */
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
    const alphaPlane = buildAlphaSampler(source, rgba, width, height)
    // Loud, because the fallback is WRONG rather than merely absent: a material
    // with no alpha plane scores avg 1 / translucentFrac 0, which routes sheer
    // fabric into the OPAQUE bucket and changes what the frame looks like. A
    // readback that fails is therefore a rendering bug, not a missing nicety,
    // and it must not reach the user as "the dress looks different on my phone".
    if (!alphaPlane) {
      console.warn(
        `[reze] alpha readback failed for ${cacheKey} — this material will be classified OPAQUE, ` +
          `so sheer fabric will not blend. The canvas 2D readback is what failed.`,
      )
    }
    this.textureAlphaCache.set(cacheKey, alphaPlane)

    // NO MIPS FOR A MOVING CARD. The chain would have to be rebuilt on every
    // frame written into it — a full pyramid of render passes per video plane
    // per frame, which is most of what a moving card was costing. Level 0 is
    // the only level a card in frame reads anyway; the price is aliasing on one
    // shrunk far into the distance, which is the case a video card is least
    // often in.
    const mipLevelCount = inst.dynamicTexture ? 1 : Math.floor(Math.log2(Math.max(width, height))) + 1
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
    pass.setPipeline(this.groundSoft ? this.ensureGroundSoftPipeline() : this.groundShadowPipeline)
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

  // Unique edges of the mesh, as a line-list index buffer. Each interior edge is
  // shared by two triangles, so deduplicating halves both the buffer and the
  // draw. Built once per model, on the first frame its wireframe is asked for.
  /** The index run `material` owns, or the whole list when it is null. Materials
   *  are consecutive runs in declaration order, so the offset is a prefix sum —
   *  the same walk the draw list does. Returns null for a name the model does
   *  not have, which is what a stale selection looks like after a reload. */
  private materialIndexRange(inst: ModelInstance, material: string | null): [number, number] | null {
    const indices = inst.model.getIndices()
    if (!material) return [0, indices.length]
    let offset = 0
    for (const m of inst.model.getMaterials()) {
      if (m.name === material) return [offset, offset + m.vertexCount]
      offset += m.vertexCount
    }
    return null
  }

  /** A cache key no material can collide with — a PMX name is never empty and
   *  never contains a NUL. */
  private static readonly SEAM_KEY = "\u0000seams"

  /**
   * @param material one material's own edges, or null for the whole mesh
   * @param seams    every material's OUTLINE instead — the borders between them
   */
  private ensureEdgeBuffer(inst: ModelInstance, material: string | null, seams = false): boolean {
    const key = material ?? (seams ? Engine.SEAM_KEY : "")
    if (inst.wireEdges.has(key)) return inst.wireEdges.get(key) !== null
    const indices = inst.model.getIndices()
    const vertexCount = inst.model.getGeometry().positions.length / 3
    const edges: number[] = []

    if (seams && !material) {
      // Inside one material's run an interior edge belongs to two triangles and
      // a border edge to one, so counting uses within the run and keeping the
      // singles gives exactly that material's outline.
      //
      // Per run, never over the whole mesh: an edge two materials share is
      // interior to the model and a border to both, and only the per-run count
      // can tell those two cases apart.
      const used = new Map<number, number>()
      const seen = new Set<number>()
      let offset = 0
      for (const m of inst.model.getMaterials()) {
        const end = offset + m.vertexCount
        used.clear()
        const bump = (a: number, b: number) => {
          const k = (a < b ? a : b) * vertexCount + (a < b ? b : a)
          used.set(k, (used.get(k) ?? 0) + 1)
        }
        for (let i = offset; i + 2 < end; i += 3) {
          bump(indices[i], indices[i + 1])
          bump(indices[i + 1], indices[i + 2])
          bump(indices[i + 2], indices[i])
        }
        for (const [k, count] of used) {
          if (count !== 1 || seen.has(k)) continue
          seen.add(k)
          edges.push(Math.floor(k / vertexCount), k % vertexCount)
        }
        offset = end
      }
    } else {
      const range = this.materialIndexRange(inst, material)
      if (!range) return false
      const [start, end] = range
      const seen = new Set<number>()
      const add = (a: number, b: number) => {
        const lo = a < b ? a : b
        const hi = a < b ? b : a
        const k = lo * vertexCount + hi
        if (seen.has(k)) return
        seen.add(k)
        edges.push(lo, hi)
      }
      for (let i = start; i + 2 < end; i += 3) {
        add(indices[i], indices[i + 1])
        add(indices[i + 1], indices[i + 2])
        add(indices[i + 2], indices[i])
      }
    }
    if (edges.length === 0) {
      inst.wireEdges.set(key, null)
      return false
    }
    const data = new Uint32Array(edges)
    const buffer = this.device.createBuffer({
      label: `wireframe edges ${inst.name}${material ? ` / ${material}` : seams ? " / seams" : ""}`,
      size: data.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(buffer, 0, data)
    const bindGroup = this.device.createBindGroup({
      label: `wireframe mesh ${inst.name}${material ? ` / ${material}` : seams ? " / seams" : ""}`,
      layout: this.wireframeSkinLayout,
      entries: [
        { binding: 0, resource: { buffer: inst.skinMatrixBuffer } },
        { binding: 1, resource: { buffer: inst.vertexBuffer } },
        { binding: 2, resource: { buffer: inst.jointsBuffer } },
        { binding: 3, resource: { buffer: inst.weightsBuffer } },
        { binding: 4, resource: { buffer } },
      ],
    })
    inst.wireEdges.set(key, { buffer, count: edges.length, bindGroup })
    return true
  }

  private renderWireframe(pass: GPURenderPassEncoder): void {
    if (!this.overlayVertices) return
    const inst = this.overlayModel(this.overlayVertices.modelName)
    const { material, xray } = this.overlayVertices
    if (!inst || !this.ensureEdgeBuffer(inst, material)) return
    const edges = inst.wireEdges.get(material ?? "")
    if (!edges) return

    // The whole mesh gets its material BORDERS drawn over it — that is what
    // makes the view read as every material at once rather than as one body of
    // undifferentiated lines. A single material asked for by name is already
    // one material, so it needs no borders to separate it from anything.
    const seams =
      material === null && this.ensureEdgeBuffer(inst, null, true)
        ? (inst.wireEdges.get(Engine.SEAM_KEY) ?? null)
        : null

    // A pointer over a material previews EXACTLY what clicking it would pick —
    // the same self-occluding reveal, layered over the section-wide view rather
    // than replacing it, so the rest of the mesh stays legible while one
    // material calls attention to itself. Meaningless once something IS
    // picked, since only the picked material draws at all then.
    const hm = this.hoverMaterial
    const hoverName = material === null && hm?.modelName === inst.name ? hm.materialName : null
    const hoverRange = hoverName ? this.materialIndexRange(inst, hoverName) : null
    const hover = hoverRange && this.ensureEdgeBuffer(inst, hoverName) ? inst.wireEdges.get(hoverName!) : null

    this.wireframeColorData.set(DEFAULT_VERTEX_COLOR)
    this.wireframeColorData[4] = this.canvas.width
    this.wireframeColorData[5] = this.canvas.height
    this.wireframeColorData[6] = OVERLAY_STYLE.meshStrokePx
    // The triangulation steps back only when there is something drawn over it
    // to step back FROM.
    if (seams) this.wireframeColorData[3] = DEFAULT_VERTEX_COLOR[3] * OVERLAY_STYLE.meshAlpha
    this.device.queue.writeBuffer(this.wireframeUniformBuffer, 0, this.wireframeColorData)
    if (seams) {
      this.wireframeColorData[3] = DEFAULT_VERTEX_COLOR[3]
      this.wireframeColorData[6] = OVERLAY_STYLE.seamStrokePx
      this.device.queue.writeBuffer(this.wireframeSeamUniformBuffer, 0, this.wireframeColorData)
    }
    if (hover) {
      this.wireframeColorData[6] = OVERLAY_STYLE.hoverStrokePx
      this.device.queue.writeBuffer(this.wireframeHoverUniformBuffer, 0, this.wireframeColorData)
    }

    // Both bind groups, before the FIRST draw call below, regardless of which
    // branch runs first — the depth prepass reads the camera from group 0 and
    // the skin matrices from group 1 same as the edge pass does, and every
    // draw in this function needs both set to SOMETHING before it runs. Each
    // block below is free to swap either one out for its own draws.
    pass.setBindGroup(0, this.wireframeBindGroup)
    pass.setBindGroup(1, edges.bindGroup)

    const bindMesh = () => {
      pass.setPipeline(this.wireframeDepthPipeline)
      pass.setVertexBuffer(0, inst.vertexBuffer)
      pass.setVertexBuffer(1, inst.jointsBuffer)
      pass.setVertexBuffer(2, inst.weightsBuffer)
      pass.setIndexBuffer(inst.indexBuffer, "uint32")
    }

    // The hover preview writes and draws against its OWN depth first, while the
    // shared depth buffer is still empty — the same trick a pick uses, run
    // before the base mesh below gets a chance to occlude it. The base mesh's
    // depth write further down repeats the SAME geometry for these faces
    // (identical z), so it neither disturbs this nor needs to skip them.
    if (hover && hoverRange && !xray) {
      bindMesh()
      pass.drawIndexed(hoverRange[1] - hoverRange[0], 1, hoverRange[0])
      pass.setBindGroup(0, this.wireframeHoverBindGroup)
      pass.setBindGroup(1, hover.bindGroup)
      pass.setPipeline(this.wireframePipeline)
      pass.draw(6, hover.count / 2)
      pass.setBindGroup(0, this.wireframeBindGroup)
      pass.setBindGroup(1, edges.bindGroup)
    }

    // What writes depth is what is allowed to hide the wireframe, and that
    // differs between the two views.
    //
    // The whole mesh, so the far wall of a 30k-triangle body does not draw on
    // top of the near one — occluded is the default everywhere, Blender's edit
    // mode and Maya included, and seeing both walls at once is moire rather than
    // information.
    //
    // A PICKED material writes only its OWN faces. The question a pick asks is
    // where this material is, and half of it is usually under a coat; letting
    // the coat hide it does not answer that. Its own depth still goes in, so its
    // back faces stay hidden and it reads as an object instead of a haze —
    // which is the difference between this and turning x-ray on.
    const range = material !== null ? this.materialIndexRange(inst, material) : null
    if (!xray) {
      bindMesh()
      if (range) pass.drawIndexed(range[1] - range[0], 1, range[0])
      else pass.drawIndexed(inst.model.getIndices().length)
    }

    // Six vertices an edge, instanced.
    pass.setPipeline(this.wireframePipeline)
    pass.draw(6, edges.count / 2)

    if (seams) {
      pass.setBindGroup(0, this.wireframeSeamBindGroup)
      pass.setBindGroup(1, seams.bindGroup)
      pass.draw(6, seams.count / 2)
    }
  }

  /** The bone overlay's options with `selected` filled in from setSelectedBone,
   *  so clicking a bone highlights it without the host mirroring the state. An
   *  explicit `selected` in the options still wins. */
  private boneOverlayOptions(modelName: string): BoneOverlayOptions {
    const options = this.overlayBones?.options ?? {}
    if (options.selected !== undefined) return options
    const chosen = this.selectedBone?.modelName === modelName ? this.selectedBone.boneName : null
    this.boneOptionsScratch.selected = chosen
    this.boneOptionsScratch.include = options.include
    return this.boneOptionsScratch
  }
  private boneOptionsScratch: BoneOverlayOptions = {}

  private overlayActive(): boolean {
    return (
      this.overlayLayers.size > 0 ||
      this.overlayBones !== null ||
      this.overlayBodies !== null ||
      this.overlayJoints !== null ||
      this.overlayVertices !== null
    )
  }

  private overlayModel(name: string): ModelInstance | null {
    return this.modelInstances.get(name) ?? null
  }

  /** The primitives a live layer would draw right now. Same list the pass uses,
   *  so a host can show it as data, diff it, or hit-test it on the CPU. */
  getOverlayPrimitives(layer: "bones" | "rigidbodies" | "joints"): OverlayPrimitive[] {
    if (layer === "bones") {
      const inst = this.overlayBones ? this.overlayModel(this.overlayBones.modelName) : null
      return inst ? boneOverlay(inst.model, this.boneOverlayOptions(inst.name)) : []
    }
    if (layer === "rigidbodies") {
      const inst = this.overlayBodies ? this.overlayModel(this.overlayBodies.modelName) : null
      return inst ? rigidbodyOverlay(inst.model, inst.physics, this.overlayBodies!.options) : []
    }
    const inst = this.overlayJoints ? this.overlayModel(this.overlayJoints.modelName) : null
    return inst ? jointOverlay(inst.model, inst.physics, this.overlayJoints!.options) : []
  }

  // The overlay's own layer: a 4x multisampled colour target, its resolve, and a
  // matching depth. All three are allocated the first frame an overlay is
  // actually on, so a scene that never shows one never pays for any of it.
  private ensureOverlayTargets(width: number, height: number): void {
    if (this.overlayResolveTexture && this.overlayTargetSize[0] === width && this.overlayTargetSize[1] === height) {
      return
    }
    const samples = Engine.OVERLAY_SAMPLE_COUNT
    this.overlayDepthTexture?.destroy()
    this.overlayMsaaTexture?.destroy()
    this.overlayResolveTexture?.destroy()

    this.overlayMsaaTexture = this.device.createTexture({
      label: "overlay msaa",
      size: [width, height],
      sampleCount: samples,
      format: this.presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.overlayResolveTexture = this.device.createTexture({
      label: "overlay resolve",
      size: [width, height],
      format: this.presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.overlayDepthTexture = this.device.createTexture({
      label: "overlay depth",
      size: [width, height],
      sampleCount: samples,
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })
    this.overlayTargetSize = [width, height]

    const colorAtt = (this.overlayPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
    colorAtt.view = this.overlayMsaaTexture.createView()
    colorAtt.resolveTarget = this.overlayResolveTexture.createView()
    const depthAtt = this.overlayPassDescriptor.depthStencilAttachment as GPURenderPassDepthStencilAttachment
    depthAtt.view = this.overlayDepthTexture.createView()

    this.overlayCompositeBindGroup = this.device.createBindGroup({
      label: "overlay composite bind group",
      layout: this.overlayCompositeLayout,
      entries: [{ binding: 0, resource: this.overlayResolveTexture.createView() }],
    })
  }

  private ensureOverlayInstanceCapacity(count: number): void {
    if (this.overlayInstanceBuffer && this.overlayInstanceCapacity >= count) return
    // Grow in powers of two so a skirt gaining bodies one at a time does not
    // reallocate once per body.
    let capacity = Math.max(64, this.overlayInstanceCapacity || 64)
    while (capacity < count) capacity *= 2
    this.overlayInstanceBuffer?.destroy()
    this.overlayInstanceBuffer = this.device.createBuffer({
      label: "overlay instance buffer",
      size: capacity * OVERLAY_INSTANCE_FLOATS * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.overlayInstanceCapacity = capacity
    this.overlayInstanceData = new Float32Array(capacity * OVERLAY_INSTANCE_FLOATS)
  }

  // Collects every layer, groups it by shape so each shape is one instanced
  // draw, and runs them into the swapchain over the finished frame.
  private renderOverlayPass(encoder: GPUCommandEncoder, swapchainView: GPUTextureView): void {
    if (!this.camera) return

    const byShape = this.overlayByShape
    for (const shape of OVERLAY_SHAPES) {
      const list = byShape.get(shape)
      if (list) list.length = 0
    }
    let total = 0
    const collect = (primitives: readonly OverlayPrimitive[]) => {
      for (const primitive of primitives) {
        let list = byShape.get(primitive.shape)
        if (!list) {
          list = []
          byShape.set(primitive.shape, list)
        }
        list.push(primitive)
        total++
      }
    }

    for (const layer of this.overlayLayers.values()) collect(layer)
    if (this.overlayBones) {
      const inst = this.overlayModel(this.overlayBones.modelName)
      if (inst) collect(boneOverlay(inst.model, this.boneOverlayOptions(inst.name)))
    }
    if (this.overlayBodies) {
      const inst = this.overlayModel(this.overlayBodies.modelName)
      if (inst) collect(rigidbodyOverlay(inst.model, inst.physics, this.overlayBodies.options))
    }
    if (this.overlayJoints) {
      const inst = this.overlayModel(this.overlayJoints.modelName)
      if (inst) collect(jointOverlay(inst.model, inst.physics, this.overlayJoints.options))
    }
    if (total === 0 && !this.overlayVertices) return

    this.ensureOverlayInstanceCapacity(total)
    const data = this.overlayInstanceData
    const draws: { shape: OverlayShape; first: number; count: number }[] = []
    let written = 0
    for (const shape of OVERLAY_SHAPES) {
      const list = byShape.get(shape)
      if (!list || list.length === 0) continue
      draws.push({ shape, first: written, count: list.length })
      for (const primitive of list) {
        writeOverlayInstance(primitive, data, written * OVERLAY_INSTANCE_FLOATS)
        written++
      }
    }
    this.device.queue.writeBuffer(
      this.overlayInstanceBuffer!,
      0,
      data.buffer,
      data.byteOffset,
      written * OVERLAY_INSTANCE_FLOATS * 4,
    )

    const width = this.canvas.width
    const height = this.canvas.height
    this.ensureOverlayTargets(width, height)
    this.overlayUniformData[0] = width
    this.overlayUniformData[1] = height
    this.overlayUniformData[2] = Engine.OVERLAY_DASH_PERIOD_PX
    this.device.queue.writeBuffer(this.overlayUniformBuffer, 0, this.overlayUniformData)

    const pass = encoder.beginRenderPass(this.overlayPassDescriptor)
    // Under everything: the mesh is the haze the rig is read against.
    this.renderWireframe(pass)
    pass.setBindGroup(0, this.overlayBindGroup)
    pass.setVertexBuffer(0, this.overlayVertexBuffer)
    pass.setVertexBuffer(1, this.overlayInstanceBuffer!)
    // Volumes first and without depth writes, then the line work over them.
    for (const solid of [true, false]) {
      let bound = false
      for (const draw of draws) {
        if (OVERLAY_SOLID_SHAPES.has(draw.shape) !== solid) continue
        if (!bound) {
          pass.setPipeline(solid ? this.overlaySolidPipeline : this.overlayPipeline)
          bound = true
        }
        const range = this.overlayGeometry.ranges[draw.shape]
        pass.draw(range.count, draw.count, range.first, draw.first)
      }
    }
    pass.end()

    // The resolved layer over the finished frame, premultiplied.
    const compositeAtt = (this.overlayCompositePassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
    compositeAtt.view = swapchainView
    const composite = encoder.beginRenderPass(this.overlayCompositePassDescriptor)
    composite.setPipeline(this.overlayCompositePipeline)
    composite.setBindGroup(0, this.overlayCompositeBindGroup!)
    composite.draw(3)
    composite.end()
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
  /**
   * Where a point on the canvas lands on a horizontal plane.
   *
   * `px,py` are canvas-relative pixels, top-left origin — what a pointer event
   * gives you after subtracting the element's rect. Returns null when the ray
   * cannot reach the plane: parallel to it, or pointing the other way, which is
   * what a click on the sky above the horizon is.
   *
   * The one primitive a placement UI needs. Dragging a thing across the floor is
   * otherwise three sliders in world units, which asks someone to guess numbers
   * that have no visible relation to the picture they are looking at — and it
   * throws away the property that makes pointing work at all: under perspective,
   * moving something further away makes it smaller by exactly the right amount,
   * so position and size stop being two controls to tune against each other.
   */
  groundPointAt(px: number, py: number, planeY = 0): Vec3 | null {
    const ray = this.buildMouseRay(px, py)
    if (!ray) return null
    // Parallel to the plane: no intersection, and a huge one is not an answer.
    if (Math.abs(ray.dir.y) < 1e-6) return null
    const t = (planeY - ray.origin.y) / ray.dir.y
    // Behind the camera — the plane is there, but not in this shot.
    if (!(t > 0) || !isFinite(t)) return null
    return new Vec3(ray.origin.x + ray.dir.x * t, planeY, ray.origin.z + ray.dir.z * t)
  }

  /** Hand the pointer to something else — a placement drag, a gizmo, a host's own
   *  overlay — so the orbit does not also act on it. */
  setCameraInputLocked(locked: boolean): void {
    this.camera?.setInputLocked(locked)
  }

  private buildMouseRay(px: number, py: number): { origin: Vec3; dir: Vec3 } | null {
    if (!this.camera) return null
    const width = this.canvas.clientWidth
    const height = this.canvas.clientHeight
    if (width <= 0 || height <= 0 || this.canvas.width <= 0 || this.canvas.height <= 0) return null
    // THE PICTURE, NOT THE ELEMENT.
    //
    // The projection's aspect comes from the DRAWING BUFFER, while a pointer
    // arrives in the CSS box — and the two do not have to agree. The canvas is
    // laid out `object-contain`, so whenever they differ the rendered image sits
    // letterboxed inside the element with bars either side of it, and dividing
    // by the element's own size lands the ray somewhere the picture is not.
    // They disagree on every resize until the observer catches up, and
    // permanently wherever a host frames the canvas to a shape of its own.
    //
    // So: work out where the image actually sits, and take the ray from that.
    const bufAspect = this.canvas.width / this.canvas.height
    const boxAspect = width / height
    const imgW = bufAspect > boxAspect ? width : height * bufAspect
    const imgH = bufAspect > boxAspect ? width / bufAspect : height
    const ox = (width - imgW) / 2
    const oy = (height - imgH) / 2
    const ndcX = ((px - ox) / imgW) * 2 - 1
    const ndcY = -(((py - oy) / imgH) * 2 - 1)
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
    if (!this.gizmoEnabled || !this.selectedBone || !this.camera || !this.device || e.button !== 0) return
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
        depthClearValue: this.depthClear,
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

    // Who holds the shot this frame. An external pose is a statement about
    // where the camera IS, so it is reapplied rather than sampled — and it
    // outranks a loaded track, which is scene data.
    if (this.cameraPoseOverride) {
      this.camera.setVmdPose(this.cameraPoseOverride)
    } else if (this.camera.vmdDriven && this.cameraAnimation) {
      // Drive the shot from the camera VMD (synced to the animated model's clock).
      const pose = this.cameraAnimation.sample(this.transportTime())
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
    // ANY effect: one foreground mount anywhere in the scene, or one ribbon,
    // is enough to make the pass store its depth instead of discarding it.
    // Ribbons are NOT in this list, and removing them is the single largest
    // bandwidth saving in the frame on a tile-based GPU.
    //
    // They were, from when a ribbon was its own layer drawn after the scene and
    // depth-tested BY HAND against the stored buffer. That layer is gone —
    // ribbons draw inside this pass and the hardware depth test replaced what
    // they read it for (see trails.ts, "Binding 3 is GONE"). The clause outlived
    // the change by about twelve hours and then sat here.
    //
    // What it cost: this flag decides whether the pass STORES its depth or
    // discards it into tile memory, and the buffer is depth32float-stencil8 at
    // the pass's sample count — on a retina canvas that is a nine-figure number
    // of bytes written to RAM every frame, for a texture nothing then sampled.
    // Chrome hides it (an immediate-mode GPU has depth in memory regardless);
    // Apple's TBDR does not, which is exactly the reported shape: adding a hand
    // ribbon costs a lot of fps on Safari and almost nothing on Chrome.
    //
    // The two real readers are both in the composite and both have their own
    // flag above: linearDepth() feeds the DoF gather and the depth handed to a
    // foreground mount. Nothing else binds depthTex at all.
    const depthRead = dofOn || this.effects.some((e) => e.hasForeground)
    this.renderPassDescriptor.depthStencilAttachment!.depthStoreOp = depthRead ? "store" : "discard"
    if (depthRead) this.writeDepthOfFieldUniforms()

    // The id attachment, on exactly the same terms as the depth above it.
    //
    // It is the most expensive STORE in the pass — rg16uint at the pass's sample
    // count, ~33MB a frame at 1080p — and a uint target cannot be resolved, so
    // storing is the only way to get it out. It was stored unconditionally, for
    // every scene, whether or not anything read it. Nothing usually does: the
    // readers are rzObjectAt / rzMaterialAt in an effect that masks itself to one
    // character, and the id-buffer debug view.
    //
    // Discarding is not the same as removing. Every pipeline still declares the
    // attachment and the pass still carries it, so nothing is rebuilt and no
    // shader changes — the frame is bit-identical either way, because the only
    // difference is whether tile memory is written back to RAM after a pass
    // whose result no one is going to read.
    const idAtt = (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[2]
    if (idAtt) {
      // The flood seeds from the id attachment, so an effect that reads only the
      // DISTANCE still needs the ids kept — it just never names them itself.
      const idsRead = this.idDebug || this.effects.some((e) => e.readsIds || e.readsCastDistance)
      idAtt.storeOp = idsRead ? "store" : "discard"
    }

    const encoder = this.device.createCommandEncoder()

    // GPU vertex morphs: write morphed positions into vertex buffers before any pass reads
    // them. WebGPU inserts the storage→vertex barrier between this pass and the render passes.
    if (hasModels) this.dispatchMorphCompute(encoder)

    // Frustum cull into indirect arguments. After the camera and shadow matrices
    // are settled, before the passes that draw from them.
    if (this.reflectionActive) this.updateMirrorCamera()
    if (hasModels) this.dispatchCull(encoder)
    // After the cull, which is what recomputes the spheres it unions.
    this.writeGroundCasterSphere()

    // After the cull, because a rebuild there can reallocate the argument
    // buffers and a bundle captures the buffer it recorded against.
    if (this.bundlesDirty) this.recordBundles()

    // Runs one more time after the last model goes: this pass owns the shadow map's
    // only `depthLoadOp: "clear"`, so skipping it outright leaves the texture holding
    // the final frame's depth — and the ground, which draws on `hasGround` alone,
    // keeps PCF-sampling a character that is no longer in the scene. One clearing
    // pass on the transition to empty, then it stops.
    if (hasModels || this.shadowMapPopulated) {
      // The far cascade is the STAGE cascade, and it costs a full pass over the
      // whole cast every frame to say so. Its own spec explains what it is for —
      // "a set piece 100 units out still throws" — and a scene with no stage has
      // no set piece: every caster sits inside the near cascade's 64-unit box,
      // which follows the camera target, and the far map's only readers are
      // ground pixels beyond that box, where nothing is casting.
      //
      // So when no stage is loaded it is drawn ONCE, cleared, and then skipped —
      // the same shape as shadowMapPopulated above, and for the same reason. A
      // cleared depth map reads as "no occluder", which is the correct answer
      // here rather than a missing one. Load a stage and it comes straight back.
      //
      // 0.43 had ONE shadow map. This is half of what the second one costs.
      const stage = this.hasStage()
      for (let ci = 0; ci < SHADOW_CASCADES.length; ci++) {
        const wanted = ci === 0 || stage
        // Already cleared and still unwanted — nothing to do, and the map still
        // holds the far plane from the pass that cleared it.
        if (!wanted && this.shadowCascadeCleared[ci]) continue
        const sp = encoder.beginRenderPass({
          // One timestamp pair exists for "shadow"; the near cascade wears it.
          timestampWrites: ci === 0 ? this.stamps("shadow") : undefined,
          colorAttachments: [],
          depthStencilAttachment: {
            view: this.shadowMapDepthViews[ci],
            depthClearValue: 1.0,
            depthLoadOp: "clear",
            depthStoreOp: "store",
          },
        })
        // The per-model `visible` test that used to guard this is gone: it is a
        // per-frame boolean, and baking it into a bundle would make toggling a
        // model re-record. It lives in the cull compute now, which zeroes the
        // instance count of an invisible model's draws.
        if (wanted && this.shadowBundles[ci]) sp.executeBundles([this.shadowBundles[ci]])
        sp.end()
        this.shadowCascadeCleared[ci] = !wanted
      }
      this.shadowMapPopulated = hasModels
    }

    // Before the particles and before the field pass: both may read the grid,
    // and a grid stepped after them is one frame stale in everything that used it.
    // Material parameters on the scene clock, before anything reads their
    // uniforms this frame.
    this.evaluateDissolveCycles()
    this.evaluateParamTracks()
    // FIRST among the things that read an effect, because every one of them
    // reads what this writes: the sim's clock, the particle uniform's weight,
    // the light dispatch, the field draw. Evaluated here rather than by a
    // caller so that playback, the export loop and a warm-up pass cannot
    // disagree about when an effect is alive — none of them has to remember it.
    this.evaluateEffectSchedules()
    this.stepSim(encoder, deltaTime)
    this.stepParticles(encoder, deltaTime)
    // Before the scene pass, which READS the slots this writes. Same buffer,
    // two access modes, never in one pass.
    this.emitLights(encoder)
    this.renderMirrorPass(encoder)

    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    // Phase order: opaque models → ground → transparent fabric.
    // The ground shader is the most expensive full-coverage draw in the frame
    // (9-tap PCF on the 4096² shadow map per pixel), so it draws AFTER the
    // opaque phase to get early-z rejected behind the body — drawing it first
    // shaded every covered pixel and measurably dropped Safari fps. It still
    // draws BEFORE the transparent phase so sheer fabric blends over the floor
    // instead of over the background with the floor depth-rejected behind it.
    //
    // Pass state the bundles cannot carry, set once for both of them:
    // GPURenderBundleEncoder has no setStencilReference, and this is a constant
    // anyway — eye writes it, hair tests not-equal, hairOverEyes tests equal.
    pass.setStencilReference(Engine.STENCIL_EYE_VALUE)
    if (this.opaqueBundle) pass.executeBundles([this.opaqueBundle])
    // Re-asserted after the bundle, not merely set once before it.
    //
    // Stencil reference is pass state a bundle cannot carry — GPURenderBundleEncoder
    // has no setStencilReference — which is why it was hoisted above the bundle in
    // the first place. But "cannot carry" and "cannot disturb" are different
    // claims, and only the first is specified. Everything below this line that
    // stencil-tests (hair at not-equal, outline hulls at not-equal) reads a
    // reference of 0 instead of 1 if a replay resets it, and not-equal against 0
    // is FALSE for the cleared buffer — every such fragment silently rejected.
    // One redundant word against a whole class of invisible failure.
    pass.setStencilReference(Engine.STENCIL_EYE_VALUE)
    if (this.hasGround) this.renderGround(pass)
    // The transparent phase is drawn DIRECTLY, and must stay that way. It is the
    // one part of this pass that is not bundled, so the reason is worth keeping.
    //
    // It WAS a bundle, and on WebKit the entire transparent bucket vanished while
    // the opaque bucket and the ground rendered perfectly — sheer fabric simply
    // absent, with no validation error anywhere. It was not the fragments: with
    // alpha forced to 1 they still never appeared, the cull reported every draw
    // visible with its GPU and CPU halves agreeing, and a cast model's
    // transparent draws use the SAME pipeline, bind groups and depth state as its
    // opaque ones (pipelineForDrawCall, forceDepthWrite). Identical draws,
    // identical state, one bucket rendering.
    //
    // What differed was only how they reached the pass: the opaque bundle is the
    // FIRST executeBundles here, and the transparent one was the SECOND, issued
    // after direct commands (the ground). Legal, and correct on Dawn. Not
    // replayed on WebKit. The mirror pass is the counter-example that pins the
    // shape of it — it passes BOTH bundles to a single executeBundles with
    // nothing direct in between, and has never lost a draw.
    //
    // So the rule this pass now keeps: at most one executeBundles, and nothing
    // direct before it. Bundling this phase again means first moving the ground
    // into the opaque bundle so the two can go in one call, the way the mirror
    // does it. The saving that buys is CPU encode time over a handful of draws,
    // which was never this renderer's bottleneck.
    const camView = this.sceneView("camera")
    this.forEachInstance((inst) => this.renderModelTransparentPhase(pass, inst, camView))
    // Last in the pass: depth-tested against everything drawn above, so a
    // particle behind the character is simply hidden, and still inside the HDR
    // target so an `#bloom` effect reaches the pyramid below.
    this.renderParticles(pass, "camera")
    // Ribbons, in the same pass and after the particles: both are additive
    // light in HDR, and both reach the bloom pyramid because of it. This used
    // to run after pass.end() into a layer of its own, which is precisely what
    // kept ribbons out of bloom.
    this.drawTrails(pass, "camera")
    pass.end()
    // The field mounts, likewise: after the scene so foregrounds can read its
    // depth, before the composite that samples both layers.
    this.renderFieldPass(encoder)

    // Bloom pyramid (EEVEE 3.6):
    //   1. Blit: HDR → bloomDown[0] (Karis prefilter, half-res)
    //   2. Downsample: bloomDown[0] → bloomDown[1] → … → bloomDown[N-1] (13-tap)
    //   3. Upsample (top-down): bloomUp[N-2] = tent(bloomDown[N-1]) + bloomDown[N-2],
    //      then bloomUp[i] = tent(bloomUp[i+1]) + bloomDown[i] until i=0 (9-tap tent)
    //   Composite reads bloomUp[0] and adds tint * intensity * bloom before Filmic.
    // bloomContributes() gates the whole pyramid, not just its intensity. The
    // composite still SAMPLES bloomUp[0] unconditionally, which is safe and
    // deliberate: it scales what it reads by the same effective intensity, so a
    // stale or never-written pyramid is multiplied by zero. Skipping the build
    // is therefore invisible in the frame and nine render passes cheaper.
    if (this.bloomContributes() && this.bloomBlitBindGroup && this.compositeBindGroup && this.bloomMipCount > 0) {
      const bloomAtt = this.bloomPassDescriptor.colorAttachments as GPURenderPassColorAttachment[]

      // 1. Blit — opens the pyramid's timing span. See stampOpen: the nine
      // passes below read as ONE component, which is the only useful grain.
      bloomAtt[0].view = this.bloomDownMipViews[0]
      this.bloomPassDescriptor.timestampWrites = this.stampOpen("bloom")
      const pBlit = encoder.beginRenderPass(this.bloomPassDescriptor)
      pBlit.setPipeline(this.bloomBlitPipeline)
      pBlit.setBindGroup(0, this.bloomBlitBindGroup)
      pBlit.draw(3)
      pBlit.end()

      // 2. Downsample chain
      this.bloomPassDescriptor.timestampWrites = undefined
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
        // The LAST upsample closes the span opened on the blit.
        this.bloomPassDescriptor.timestampWrites = k === upSteps - 1 ? this.stampClose("bloom") : undefined
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

    // Over the finished frame, before the editor's own overlays: the whole
    // point is to see the id buffer instead of the scene.
    this.renderReflectionDebugPass(encoder, swapchainView)
    this.renderIdDebugPass(encoder, swapchainView)

    if (this.selectedMaterial && hasModels) this.renderSelectionPasses(encoder, swapchainView)
    // Under the gizmo: the handles you drag stay on top of the rig you are
    // reading them against.
    if (this.overlayActive()) this.renderOverlayPass(encoder, swapchainView)
    if (this.gizmoEnabled && this.selectedBone && hasModels) this.renderGizmoPass(encoder, swapchainView)

    const pick = this.pendingPick
    if (pick && hasModels) this.renderPickPass(encoder)

    // Last thing encoded: every timed pass has run by here.
    this.resolveTimestamps(encoder)

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

  private drawInstanceShadow(sp: GPURenderPassEncoder | GPURenderBundleEncoder, inst: ModelInstance, cascade: number): void {
    sp.setBindGroup(0, inst.shadowBindGroups[cascade])
    sp.setVertexBuffer(0, inst.vertexBuffer)
    sp.setVertexBuffer(1, inst.jointsBuffer)
    sp.setVertexBuffer(2, inst.weightsBuffer)
    sp.setIndexBuffer(inst.indexBuffer, "uint32")
    for (const draw of inst.shadowDrawCalls) {
      sp.setBindGroup(1, draw.bindGroup)
      this.issueDraw(sp, draw, "shadow")
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
  /**
   * Drive a material parameter from the SCENE CLOCK.
   *
   * The channel this writes into already existed — setStyleParam below puts a
   * value straight into the style uniform. What a track adds is WHEN: the value
   * is a pure function of scene time, so a scene describes a dissolve once and
   * playback, a re-open and an offline export stepped at another rate all
   * produce the same frames.
   *
   * Addressed BY NAME (model, group, param), not by id. Worth stating because
   * the id attachment landed alongside this and the two look related: ids
   * answer "which object is this PIXEL", which is a screen-space question, and
   * a track answers "what is this parameter NOW". Nothing here needs MRT.
   *
   * Null or empty clears the track and leaves the parameter wherever it was —
   * removing an animation is not the same as resetting a value, and guessing
   * which the caller meant would be worse than either.
   */
  setStyleParamTrack(modelName: string, groupId: string, paramId: string, keys: ParamKey[] | null): boolean {
    const id = `${modelName}\u0000${groupId}\u0000${paramId}`
    if (!keys || keys.length === 0) {
      this.paramTracks.delete(id)
      return true
    }
    // Refused rather than stored if the target does not exist: a track on a
    // parameter nobody has is silence, and silence is what makes an author
    // hunt through their document for a typo the engine could have named.
    const install = this.modelInstances.get(modelName)?.styleGroups.get(groupId)
    if (!install?.slotMap.find((s) => s.id === paramId)) return false
    // Sorted ONCE here so the per-frame sample can binary-search.
    this.paramTracks.set(id, {
      modelName,
      groupId,
      paramId,
      keys: [...keys].sort((a, b) => a.t - b.t),
      last: null,
    })
    return true
  }

  /** Forget what a group's tracks last wrote, so the next frame writes it again.
   *  Called whenever something else has written that uniform underneath them. */
  private invalidateParamTracks(modelName: string, groupId: string): void {
    for (const track of this.paramTracks.values()) {
      if (track.modelName === modelName && track.groupId === groupId) track.last = null
    }
  }

  /** Every track, at the current scene clock. Called once per frame, before the
   *  pass that reads the uniforms it writes. */
  private evaluateParamTracks(): void {
    if (this.paramTracks.size === 0) return
    for (const track of this.paramTracks.values()) {
      const v = sampleParamTrack(track.keys, this.sceneClock)
      // Most tracks are flat most of the time. Writing only on a CHANGE is what
      // keeps a still scene from spending a uniform write per parameter per
      // frame for values nobody moved.
      if (v === null || !paramChanged(v, track.last)) continue
      // Recorded only if the write LANDED. A group that is mid-recompile or
      // gone refuses it, and remembering a value that never reached the GPU
      // would mean never trying again — the track would go quiet permanently
      // instead of resuming when the group comes back.
      if (this.setStyleParam(track.modelName, track.groupId, track.paramId, v)) track.last = v
    }
  }

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

  /**
   * How much of a model is still THERE: 1 whole, 0 gone.
   *
   * The instant tier, like setStyleParam — one float per material, no recompile,
   * no pipeline touched. What it drives is a THRESHOLD, not an opacity: the
   * material shell throws away every flake whose object-space threshold has
   * passed, and lights the ones about to go. So a model at 0.5 is not
   * half-transparent, it is half GONE, which is the difference between a fade
   * and a disintegration.
   *
   * Written into the depth prepass's copy of the same test as well, so the
   * flakes stop claiming depth the moment they stop being drawn — see
   * DISSOLVE_WGSL for why that has to be one implementation.
   *
   * Mirrored into the cast, so an effect can read rzSubject(i).dissolve and draw
   * the sparks that leave her in step with the body they came off. That is the
   * whole reason this lives on the model rather than in an effect's uniform: the
   * material pass runs long before any effect, and only the engine sees both.
   */
  setModelDissolve(modelName: string, value: number): boolean {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return false
    const v = Math.min(1, Math.max(0, value))
    if (inst.dissolve === v) return true
    inst.dissolve = v
    // Offset 60: the sixteenth float of MaterialUniforms. One four-byte write
    // per material rather than the whole block — the block only exists as a
    // copy for materials that morph.
    const one = new Float32Array([v])
    for (const buffer of inst.materialUniformBuffers) {
      this.device.queue.writeBuffer(buffer, 60, one)
    }
    // And the hulls, which bind their own 32 bytes of edge data rather than the
    // material block — so this is a different buffer at a different offset, and
    // missing it left a dissolved character standing in her own outline.
    for (const buffer of inst.outlineUniformBuffers) {
      this.device.queue.writeBuffer(buffer, RZ_OUTLINE_DISSOLVE_OFFSET, one)
    }
    // The morph path rebuilds a material's block from its `base` copy and
    // uploads it whole, which would put the old value straight back. Patching
    // `base` is what keeps a face that is morphing while she dissolves from
    // coming back solid for those frames; `last` is cleared so the next
    // comparison genuinely re-uploads rather than deciding nothing moved.
    if (inst.materialMorphTargets) {
      for (const t of inst.materialMorphTargets) {
        t.base[15] = v
        t.last[15] = Number.NaN
      }
    }
    return true
  }

  /**
   * A repeating dissolve, on the scene clock.
   *
   * The alternative was a host calling setModelDissolve every frame, and it is
   * the wrong shape twice: an exported take stepped at another rate would land
   * on different values than the preview did, and the effect drawing the sparks
   * would be reading a number some other clock wrote. Here the engine samples it
   * where it samples everything else time-driven, so a take reproduces exactly
   * and rzSubject().dissolve is the same value the material shell used on that
   * very frame.
   *
   * The five numbers are seconds within one cycle: when she starts to go, when
   * she is fully gone, when she starts to come back, and when she is whole. The
   * gaps between them are the timing, and the hold between the middle two is how
   * long she is away.
   */
  setModelDissolveCycle(modelName: string, cycle: DissolveCycle | null): boolean {
    if (!this.modelInstances.has(modelName)) return false
    if (!cycle) {
      if (this.dissolveCycles.delete(modelName)) this.setModelDissolve(modelName, 1)
      return true
    }
    this.dissolveCycles.set(modelName, cycle)
    return true
  }

  /** Every dissolve cycle, at the current scene clock. Once per frame, before
   *  the cast is written and long before any effect reads it. */
  private evaluateDissolveCycles(): void {
    if (this.dissolveCycles.size === 0) return
    for (const [name, c] of this.dissolveCycles) {
      const period = Math.max(c.period, 1e-3)
      const t = this.sceneClock - Math.floor(this.sceneClock / period) * period
      let v = 1
      if (t >= c.breakAt && t < c.hiddenAt) {
        v = 1 - (t - c.breakAt) / Math.max(c.hiddenAt - c.breakAt, 1e-4)
      } else if (t >= c.hiddenAt && t < c.backAt) {
        v = 0
      } else if (t >= c.backAt && t < c.doneAt) {
        v = (t - c.backAt) / Math.max(c.doneAt - c.backAt, 1e-4)
      }
      this.setModelDissolve(name, v)
    }
  }

  /** What setModelDissolve last set, or 1 for a model that has never dissolved. */
  getModelDissolve(modelName: string): number {
    return this.modelInstances.get(modelName)?.dissolve ?? 1
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
      // The depth-write-off twin: stage transparency draws with it (see
      // pipelineForDrawCall), and a future OIT path would too.
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
    // Swapping bind groups without re-sorting is the one structural change that
    // does not pass through sortDrawCalls, so it has to say so itself. A bundle
    // holds the bind group it recorded, and the textures behind the outgoing one
    // are destroyed on the next line — the same failure the comment above
    // describes, one level further out.
    this.bundlesDirty = true
    for (const tex of previousImages ?? []) tex?.destroy()
    this.writeGroupDefaults(uniformBuffer, group, result.slotMap)
    // The defaults just overwrote whatever a track had driven into this buffer.
    // A track is only written when its value CHANGES, so a flat one would never
    // write again and the parameter would sit at its default until the next
    // key — silently, and only after an unrelated graph edit. Forgetting what
    // was last written makes the next frame restate it.
    this.invalidateParamTracks(inst.name, group.id)
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
    this.bundlesDirty = true
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
      format: this.depthFormat,
      depthWriteEnabled: depthWrite,
      depthCompare: this.depthAhead,
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
        format: this.depthFormat,
        depthWriteEnabled: false,
        depthCompare: this.depthAhead,
        stencilFront: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0,
      }
    } else if (renderClass === "eye") {
      depthStencil = {
        ...plainDepth,
        // No depth bias, and none was ever in effect: this carried
        // depthBias: -0.00005 for its whole life, and GPUDepthBias is an i32 —
        // WebIDL truncates -0.00005 to ZERO before the driver sees it. The
        // see-through-hair effect demonstrably works without a bias (that IS
        // the deployed look), via draw order + front-cull + the stencil stamp.
        // Removing the dead literal is behavior-identical; introducing a REAL
        // bias would change how eyes sit against the face on every published
        // scene, so it is deliberately not done here.
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
  // draws on the CAST use the SAME depth-write-on pipeline — babylon-mmd's
  // forceDepthWrite blending (see renderModelTransparentPhase for the trade-off
  // record). A STAGE's transparent draws are the exception: forceDepthWrite
  // exists for fabric self-layering, and a stage does not self-fold — what its
  // translucent shell's depth DID do was occlude every particle behind it, so
  // rain vanished the instant the camera crossed a glass dome or a curtain
  // (reported: binary vanish/recover with camera angle, stage loaded). Stage
  // transparency blends and leaves depth alone.
  private pipelineForDrawCall(inst: ModelInstance, dc: DrawCall): GPURenderPipeline {
    const stageGlass = inst.isStage && dc.type === "transparent"
    if (dc.groupId) {
      const install = inst.styleGroups.get(dc.groupId)
      if (install) return stageGlass ? install.pipelineNoDepthWrite : install.pipeline
    }
    return stageGlass ? this.neutralPipelineNoDepthWrite : this.neutralPipeline
  }

  /**
   * Draw every material of a given type (`opaque` or `transparent`) using the main
   * pipeline(s), and — babylon-mmd's per-mesh outline stage — each edge-flagged
   * material's inverted hull IMMEDIATELY after its color draw. Interleaving is what
   * makes outlines compose like MMD: every material drawn later in the author's
   * order covers earlier hulls, and each hull sits over everything drawn before it.
   */
  /** Is this draw's compiled class "hair"? Ungrouped draws never are — the
   *  neutral pipeline is the auto class. */
  private isHairDraw(inst: ModelInstance, dc: DrawCall): boolean {
    if (!dc.groupId) return false
    const install = inst.styleGroups.get(dc.groupId)
    return install?.renderClass === "hair"
  }

  private drawMaterials(
    pass: GPURenderPassEncoder | GPURenderBundleEncoder,
    inst: ModelInstance,
    type: "opaque" | "transparent",
    view: { perFrame: GPUBindGroup; args: "camera" | "mirror"; outlines: boolean },
    // The opaque phase walks its author order twice — non-hair, then hair — so
    // the hair depth prime can sit between the eye's stencil write and the hair
    // colour that must respect it. See renderModelOpaquePhase.
    only?: "hair" | "non-hair",
  ): void {
    let currentPipeline: GPURenderPipeline | null = null
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== type) continue
      if (only && (only === "hair") !== this.isHairDraw(inst, draw)) continue
      if (!bound) {
        pass.setBindGroup(0, view.perFrame)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      const pipeline = this.pipelineForDrawCall(inst, draw)
      if (pipeline !== currentPipeline) {
        pass.setPipeline(pipeline)
        currentPipeline = pipeline
      }
      pass.setBindGroup(2, draw.bindGroup)
      this.issueDraw(pass, draw, view.args)
      if (draw.outline && this.outlineEnabled && view.outlines) {
        // Same index range; own pipeline + groups 0/2. Group 1 (skinMats) is
        // layout-identical between the main and outline pipelines and stays
        // bound. Restore group 0 afterwards and force a pipeline re-set.
        pass.setPipeline(this.outlinePipeline)
        pass.setBindGroup(0, this.outlinePerFrameBindGroup)
        pass.setBindGroup(2, draw.outline.bindGroup)
        this.issueDraw(pass, draw, view.args)
        pass.setBindGroup(0, view.perFrame)
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
  private setModelDrawState(pass: GPURenderPassEncoder | GPURenderBundleEncoder, inst: ModelInstance): void {
    pass.setVertexBuffer(0, inst.vertexBuffer)
    pass.setVertexBuffer(1, inst.jointsBuffer)
    pass.setVertexBuffer(2, inst.weightsBuffer)
    pass.setIndexBuffer(inst.indexBuffer, "uint32")
    // The stencil reference used to be set here. It is pass state, not bundle
    // state — GPURenderBundleEncoder has no setStencilReference at all — so it
    // moved to the pass, which is where it always belonged: one constant covering
    // eye (write), hair (read not-equal) and hairOverEyes (read equal), set once
    // instead of once per model. Non-stencil pipelines ignore the value.
  }

  /**
   * Which eye the scene is being drawn FOR — the main camera or the floor
   * mirror. Threaded explicitly through the phase draws rather than read off
   * the engine, because both sets of bundles are recorded in one call and
   * ambient state at record time is how a mirror bundle ends up baked with the
   * main camera's bind group.
   */
  private sceneView(kind: "camera" | "mirror"): {
    perFrame: GPUBindGroup
    args: "camera" | "mirror"
    outlines: boolean
  } {
    return kind === "mirror"
      ? // No outlines in the mirror: the hull pipeline culls back faces, and a
        // reflection flips winding, so the hull would ink over the model.
        { perFrame: this.mirrorPerFrameBindGroup, args: "mirror", outlines: false }
      : { perFrame: this.perFrameBindGroup, args: "camera", outlines: true }
  }

  private renderModelOpaquePhase(
    pass: GPURenderPassEncoder | GPURenderBundleEncoder,
    inst: ModelInstance,
    view: { perFrame: GPUBindGroup; args: "camera" | "mirror"; outlines: boolean },
  ): void {
    this.setModelDrawState(pass, inst)
    // Depth first, colour second — the close-up fix, and the oldest one there
    // is. See drawOpaqueDepthPrepass.
    this.drawOpaqueDepthPrepass(pass, inst, view)
    // The opaque author order, in two walks with the hair prime between them.
    //
    // Hair could not join the plain prepass: primed hair depth would depth-
    // reject the eye before it writes the stencil the see-through-hair pass
    // needs. But the trick only needs the eye BEFORE hair, not before
    // everything — so the non-hair walk runs first (the eye writes stencil
    // against real face depth, exactly as it always did), the prime then lays
    // hair depth down stencil-fenced off the eye silhouette, and the hair walk
    // shades once per pixel instead of once per card.
    //
    // The one thing this reorders: hair now draws after any opaque material
    // authored later than it. A soft hair edge over such a material blends
    // over the material instead of over whatever the framebuffer held mid-
    // order — deterministic where it used to be accidental, and only at
    // sub-alpha edge texels over late-authored geometry.
    this.drawMaterials(pass, inst, "opaque", view, "non-hair")
    this.drawHairDepthPrime(pass, inst, view)
    this.drawMaterials(pass, inst, "opaque", view, "hair")
    this.drawHairOverEyes(pass, inst, view)
  }

  /** Depth-only prime of the hair's alpha-1 texels, stencil-fenced off the eye
   *  silhouette. See the note at its call site and hairPrimePipeline. */
  private drawHairDepthPrime(
    pass: GPURenderPassEncoder | GPURenderBundleEncoder,
    inst: ModelInstance,
    view: { perFrame: GPUBindGroup; args: "camera" | "mirror"; outlines: boolean },
  ): void {
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== "opaque" || !this.isHairDraw(inst, draw)) continue
      if (!bound) {
        pass.setPipeline(this.hairPrimePipeline)
        pass.setBindGroup(0, view.perFrame)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      pass.setBindGroup(2, draw.bindGroup)
      this.issueDraw(pass, draw, view.args)
    }
  }

  /**
   * Depth-only prime of the plain opaque draws, so each covered pixel SHADES
   * once instead of once per layer.
   *
   * The oldest fps complaint this engine has — zoom close and the frame drops,
   * in every material generation back to the earliest — was never the vertices
   * and never one shader's fault: with the fragment shaders flattened to a
   * constant the close-up ran smooth with identical geometry, overdraw and
   * MSAA. The cost is per-fragment shading TIMES how many times a pixel runs
   * it, and an MMD model at close-up is layers all the way down: cloth over
   * body, sleeves over cloth, hair over everything. Author-order drawing
   * shades every layer and then buries all but one.
   *
   * So the plain opaque draws lay their depth down first, through the same
   * depth-only pipeline the transparent bucket keeps for its dormant prepass —
   * same skinned vertex path (position marked @invariant in both modules, so
   * the colour pass lands on exactly these depths and its less-equal test
   * keeps the visible surface and rejects the buried ones), same alpha-0.5
   * cutout, writeMask 0 on every colour target. The pixels are identical by
   * construction: this pass writes no colour, and the colour pass draws
   * exactly what it always drew minus the fragments something opaque provably
   * covers.
   *
   * WHO IS IN. Only render-class "auto" with alpha-mode "opaque" — the body,
   * face and cloth materials that are the bulk of every model — plus every
   * ungrouped material (the neutral pipeline is that same class). WHO IS OUT,
   * each for a reason that would change pixels: EYE front-culls and gates on a
   * bone read, and pre-filled hair depth over the socket would depth-reject
   * the eye before it could write the stencil the see-through-hair pass needs
   * — which is also why HAIR stays out entirely. HASHED alpha (stockings)
   * discards by a position hash this pass does not run, so priming it would
   * punch its cutout into the depth buffer at the wrong texels. They all still
   * BENEFIT: their fragments early-z against the primed depth of whatever
   * plain opaque surface sits in front of them.
   */
  private drawOpaqueDepthPrepass(
    pass: GPURenderPassEncoder | GPURenderBundleEncoder,
    inst: ModelInstance,
    view: { perFrame: GPUBindGroup; args: "camera" | "mirror"; outlines: boolean },
  ): void {
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== "opaque") continue
      if (draw.groupId) {
        const install = inst.styleGroups.get(draw.groupId)
        if (install && !(install.renderClass === "auto" && install.alphaMode === "opaque")) continue
      }
      if (!bound) {
        pass.setPipeline(this.depthPrepassPipeline)
        pass.setBindGroup(0, view.perFrame)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      pass.setBindGroup(2, draw.bindGroup)
      this.issueDraw(pass, draw, view.args)
    }
  }

  /**
   * Depth-only prime of the transparent bucket's FULLY SOLID texels.
   *
   * The dress problem. A "transparent" MMD material is mostly weave at alpha
   * exactly 1 with sheer margins, and its layers draw in author order — so a
   * close-up skirt shades every buried panel and then covers the work. The
   * buried SHEER fragments must shade (their blend reads what is behind), but
   * at alpha 1 over-blending is plain replacement: the destination cannot
   * matter, so a fragment buried behind an alpha-1 texel contributes nothing.
   * Priming depth for exactly those texels (CUTOFF 1.0) rejects the buried
   * work and cannot move a pixel.
   *
   * A STAGE's transparent draws are excluded the way their colour path already
   * is: stage glass deliberately leaves depth alone so rain and particles
   * survive behind a dome (see pipelineForDrawCall), and a prime would put the
   * occlusion right back.
   */
  private drawTransparentSolidPrepass(
    pass: GPURenderPassEncoder | GPURenderBundleEncoder,
    inst: ModelInstance,
    view: { perFrame: GPUBindGroup; args: "camera" | "mirror"; outlines: boolean },
  ): void {
    if (inst.isStage) return
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== "transparent") continue
      if (!bound) {
        pass.setPipeline(this.solidPrepassPipeline)
        pass.setBindGroup(0, view.perFrame)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      pass.setBindGroup(2, draw.bindGroup)
      this.issueDraw(pass, draw, view.args)
    }
  }

  private renderModelTransparentPhase(
    pass: GPURenderPassEncoder | GPURenderBundleEncoder,
    inst: ModelInstance,
    view: { perFrame: GPUBindGroup; args: "camera" | "mirror"; outlines: boolean },
  ): void {
    // Draw state FIRST — each phase records into its own bundle encoder, and a
    // bundle starts with nothing bound.
    this.setModelDrawState(pass, inst)
    this.drawTransparentSolidPrepass(pass, inst, view)
    // Transparent: babylon-mmd's forceDepthWrite blending — PMX author order
    // with depth write ON. The accepted trade-off after trying every variant:
    //   · depth-write ON (this): a fold hides its far side; rare view-dependent
    //     double-blend seams at some angles. MMD's own known behavior.
    //   · nearest-surface prepass: view-independent, but punched see-through
    //     holes to whatever sat far behind a fold.
    //   · depth-write OFF layering: every overlap visible everywhere — MORE
    //     gray patches and texture artifacts in practice.
    this.drawMaterials(pass, inst, "transparent", view)
  }

  /** Depth-only re-draw of transparent-bucket materials (see depth-prepass.ts).
   *  Dormant — kept for a future order-independent-transparency path. */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  protected drawTransparentDepthPrepass(pass: GPURenderPassEncoder, inst: ModelInstance): void {
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== "transparent") continue
      if (!bound) {
        pass.setPipeline(this.depthPrepassPipeline)
        pass.setBindGroup(0, this.perFrameBindGroup)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      pass.setBindGroup(2, draw.bindGroup)
      this.issueDraw(pass, draw, "camera")
    }
  }

  /**
   * Second hair pass for the see-through-hair effect. Re-draws every hair-class grouped
   * opaque draw with its compiled over-eyes pipeline — stencil-matched to `EYE_VALUE`,
   * `IS_OVER_EYES=true` (25% alpha), depth-write off. Ungrouped materials are neutral and
   * never participate.
   */
  private drawHairOverEyes(
    pass: GPURenderPassEncoder | GPURenderBundleEncoder,
    inst: ModelInstance,
    view: { perFrame: GPUBindGroup; args: "camera" | "mirror"; outlines: boolean },
  ): void {
    let bound = false
    let currentPipeline: GPURenderPipeline | null = null
    for (const draw of inst.drawCalls) {
      if (draw.type !== "opaque") continue
      const overEyes = this.overEyesPipelineFor(inst, draw)
      if (!overEyes) continue
      if (!bound) {
        pass.setBindGroup(0, view.perFrame)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      if (overEyes !== currentPipeline) {
        pass.setPipeline(overEyes)
        currentPipeline = overEyes
      }
      pass.setBindGroup(2, draw.bindGroup)
      this.issueDraw(pass, draw, view.args)
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
      // The FIELD clock, and it is shared: viewU lives in one composite uniform
      // that every field draw reads, so rzTime() is the same for all of them.
      // Taken from the first effect, which keeps a single-effect scene exactly
      // as it was. KNOWN GAP for multi-effect: an effect installed later starts
      // its rzTime() mid-stream rather than at zero, so a one-shot intro
      // animation would be skipped. Periodic effects — nearly all of them — do
      // not care. Fixing it properly means a per-effect field uniform, which is
      // the field-pass restructure's business, not this increment's. The SIM
      // clock is already per effect, which is the one that actually breaks
      // things (rzGridFrame()==0 is a grid's only chance to seed).
      u[24] = this.sceneClock - (this.effects[0]?.epochScene ?? 0)
      // The grain's seed rides the same per-frame refresh, because it is the
      // only thing that makes it move — a seed written once by its setter is a
      // still pattern welded to the picture. On the SCENE clock like everything
      // else here, so an export reproduces the editor exactly rather than
      // scattering differently at whatever rate the encoder ran.
      u[3] = this.grain.animated ? Math.floor((this.sceneClock * 24) % 1024) : 0
      u[26] = this.canvas.width
      u[27] = this.canvas.height
      // Camera world position (viewU[10]) — the other half of bgWorldPos. It
      // rides this refresh rather than writeCompositeViewUniforms because it
      // changes every frame the camera does, exactly like the basis above.
      u[40] = cameraPos.x
      u[41] = cameraPos.y
      u[42] = cameraPos.z
      // Character positions (viewU[11..14]), count in viewU[10].w. Neither a
      // stage nor a plane is a performer: an effect asking where the cast is
      // means the characters, a stage's origin is wherever its author put it,
      // and a card is a picture with an id. Leaving a card in the list hands its
      // object id to every consumer of the cast — the distance field then seeds
      // the whole rectangle, so a silhouette effect drew a border around the
      // video behind her and none at all around her. Four is the cap because the
      // uniform is small and a scene with five characters is not the case this
      // serves.
      let n = 0
      this.forEachInstance((inst) => {
        if (n >= MAX_EFFECT_SUBJECTS || inst.isStage || inst.isPlane || inst.isProp) return
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
      // The same number the ribbons size their instance count by — see
      // drawTrails. Recorded rather than recomputed: this loop is the one place
      // that knows how many subjects the cast actually ended up holding.
      this.castSubjectCount = n
      this.device.queue.writeBuffer(this.compositeUniformBuffer, 0, u)
      // Only what an effect declared, and only while one is installed. A scene
      // with no effect writes nothing here at all.
      if (this.effect) {
        // Up to the last trailed slot, not the whole buffer: an effect with no
        // trails never uploads the 32KB it would otherwise pay for every frame.
        const scene = this.anchorTable.entries
        let lastTrail = -1
        for (let i = 0; i < scene.length; i++) if (scene[i].trail) lastTrail = i
        const used =
          lastTrail >= 0
            ? CAST_TRAIL_BASE + (lastTrail * MAX_EFFECT_SUBJECTS + MAX_EFFECT_SUBJECTS) * TRAIL_SAMPLES
            : CAST_SUBJECT_VEC4S + scene.length * MAX_EFFECT_SUBJECTS * 3
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
    // The root vec4's w, a constant 1 until now: how much of this subject is
    // still there. An effect that draws what is LEAVING her needs to know how
    // far along she is, and reading it here is what keeps the sparks in step
    // with the body without a second clock to agree with.
    cd[b + 3] = inst.dissolve
    cd[b + 4] = px
    cd[b + 5] = py
    cd[b + 6] = pz
    // The centre vec4's w, unused until now: this subject's OBJECT ID. It is
    // what makes the id attachment addressable from an effect — reading an id
    // out of the buffer is useless without something to compare it against, and
    // "the character I am following" is the comparison every masking effect
    // actually wants.
    cd[b + 7] = inst.objectId
    cd[b + 8] = px
    cd[b + 9] = floorY + height * 0.5
    cd[b + 10] = pz
    cd[b + 11] = height * 0.75

    // Declared bones. Velocity is per model AND per slot, so two characters
    // wearing the same effect never inherit each other's motion.
    // The SCENE's bones, deduplicated — not this effect's declarations. Two
    // effects naming the same wrist resolve and upload it once, and the slot
    // each reads is the one the table dealt them.
    const anchors = this.anchorTable.entries
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
  /**
   * Forget every recorded path, because the slots have been re-dealt.
   *
   * Called on every effect swap. The rings are keyed by (model, slot) and a slot
   * is an ADDRESS, so re-allocating the table can leave a left wrist's recorded
   * history sitting at the address a right wrist now occupies — a ribbon drawn
   * confidently along a path that belongs to another bone. Refilling from live
   * samples costs about two seconds of trail and cannot be wrong.
   */
  private clearTrailHistory(): void {
    this.anchorTrail.clear()
    this.anchorPrev.clear()
    // The GPU copy too: rzTrailCount reads the recorded count out of the cast
    // buffer, and an effect installed mid-frame would otherwise read the old
    // effect's counts before the next upload replaces them.
    this.castData.fill(0, CAST_SUBJECT_VEC4S * 4)
  }

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
