import { Camera } from "./camera"
import { Mat4, Quat, Vec3 } from "./math"
import { Model } from "./model"
import { PmxLoader } from "./pmx-loader"
import { RezePhysics } from "./physics"
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
import { DEFAULT_SHADER_WGSL } from "./shaders/materials/default"
import { FACE_SHADER_WGSL } from "./shaders/materials/face"
import { HAIR_SHADER_WGSL } from "./shaders/materials/hair"
import { CLOTH_SMOOTH_SHADER_WGSL } from "./shaders/materials/cloth_smooth"
import { CLOTH_ROUGH_SHADER_WGSL } from "./shaders/materials/cloth_rough"
import { METAL_SHADER_WGSL } from "./shaders/materials/metal"
import { BODY_SHADER_WGSL } from "./shaders/materials/body"
import { EYE_SHADER_WGSL } from "./shaders/materials/eye"
import { STOCKINGS_SHADER_WGSL } from "./shaders/materials/stockings"
import { BRDF_LUT_SIZE, BRDF_LUT_BAKE_WGSL } from "./shaders/dfg_lut"
import { LTC_MAG_LUT_SIZE, LTC_MAG_LUT_DATA } from "./shaders/ltc_mag_lut"
import { SHADOW_DEPTH_SHADER_WGSL } from "./shaders/passes/shadow"
import { GROUND_SHADOW_SHADER_WGSL } from "./shaders/passes/ground"
import { OUTLINE_SHADER_WGSL } from "./shaders/passes/outline"
import { SELECTION_MASK_SHADER_WGSL, SELECTION_EDGE_SHADER_WGSL } from "./shaders/passes/selection"
import { GIZMO_SHADER_WGSL } from "./shaders/passes/gizmo"
import {
  BLOOM_BLIT_SHADER_WGSL,
  BLOOM_DOWNSAMPLE_SHADER_WGSL,
  BLOOM_UPSAMPLE_SHADER_WGSL,
} from "./shaders/passes/bloom"
import { COMPOSITE_SHADER_WGSL } from "./shaders/passes/composite"
import { PICK_SHADER_WGSL } from "./shaders/passes/pick"
import { MIPMAP_BLIT_SHADER_WGSL } from "./shaders/passes/mipmap"

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

function resolvePreset(materialName: string, map: MaterialPresetMap | undefined): MaterialPreset {
  if (!map) return "default"
  for (const [preset, names] of Object.entries(map)) {
    if (names && names.includes(materialName)) return preset as MaterialPreset
  }
  return "default"
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

export type SunOptions = {
  /** Linear color of the sun lamp (Blender: Light > Color). */
  color?: Vec3
  /** Lamp power in Blender units (Blender: Light > Strength). */
  strength?: number
  /** Direction sunlight travels (points FROM sun TO scene, Blender: -light.rotation.Z). */
  direction?: Vec3
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

/** Blender Color Management / View (rendering.txt: Filmic, exposure, gamma). `look` is reserved for future curve tweaks. */
export type ViewTransformOptions = {
  /** Stops applied before Filmic: `linear *= 2^exposure`. */
  exposure: number
  /** After Filmic, display gamma (`pow(rgb, 1/gamma)`). */
  gamma: number
  look: "default" | "medium_high_contrast"
}

// Matches the reference Blender project: Filmic view, Medium High Contrast look,
// exposure 0.3, gamma 1.0, sRGB display, no curves.
export const DEFAULT_VIEW_TRANSFORM: ViewTransformOptions = {
  exposure: 0.6,
  gamma: 1.0,
  look: "medium_high_contrast",
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
}

type DrawCallType = "opaque" | "transparent" | "ground" | "opaque-outline" | "transparent-outline"

interface DrawCall {
  type: DrawCallType
  count: number
  firstIndex: number
  bindGroup: GPUBindGroup
  materialName: string
  preset: MaterialPreset
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
  hiddenMaterials: Set<string>
  materialPresets: MaterialPresetMap | undefined
  physics: RezePhysics | null
  vertexBufferNeedsUpdate: boolean
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
  private camera!: Camera
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
  private depthTexture!: GPUTexture
  private modelPipeline!: GPURenderPipeline
  private facePipeline!: GPURenderPipeline
  private hairPipeline!: GPURenderPipeline
  private clothSmoothPipeline!: GPURenderPipeline
  private clothRoughPipeline!: GPURenderPipeline
  private metalPipeline!: GPURenderPipeline
  private bodyPipeline!: GPURenderPipeline
  private eyePipeline!: GPURenderPipeline
  private hairOverEyesPipeline!: GPURenderPipeline
  private stockingsPipeline!: GPURenderPipeline
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
  private renderPassDescriptor!: GPURenderPassDescriptor
  private compositePassDescriptor!: GPURenderPassDescriptor
  // Two specialized composite pipelines via WGSL pipeline-override constants.
  // Identity variant skips the gamma pow entirely at shader-compile time —
  // Safari's Metal backend won't fold pow(x, 1) to identity.
  private compositePipelineIdentity!: GPURenderPipeline
  private compositePipelineGamma!: GPURenderPipeline
  private compositeBindGroupLayout!: GPUBindGroupLayout
  private compositeBindGroup!: GPUBindGroup
  private compositeUniformBuffer!: GPUBuffer
  // [exposure, invGamma, _, _,  bloomTint.x, bloomTint.y, bloomTint.z, bloomIntensity]
  private readonly compositeUniformData = new Float32Array(8)

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
  private static readonly SHADOW_MAP_SIZE = 2048
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
  private mipBlitPipeline: GPURenderPipeline | null = null
  private mipBlitSampler: GPUSampler | null = null
  private _nextDefaultModelId = 0

  // IK and physics enabled at engine level (same for all models)
  private ikEnabled = true
  private physicsEnabled = true

  // Camera target binding (Babylon/Three style: camera follows model)
  private cameraTargetModel: Model | null = null
  private cameraTargetBoneName = "全ての親"
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
    this.onRaycast = options?.onRaycast
    this.onGizmoDrag = options?.onGizmoDrag
    this.bloomSettings = Engine.mergeBloomDefaults(options?.bloom)
    this.viewTransform = Engine.mergeViewTransformDefaults(options?.view)
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
      look: partial?.look ?? d.look,
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
    return { exposure: v.exposure, gamma: v.gamma, look: v.look }
  }

  setViewTransformOptions(patch: Partial<ViewTransformOptions>): void {
    const v = this.viewTransform
    if (patch.exposure !== undefined) v.exposure = patch.exposure
    if (patch.gamma !== undefined) v.gamma = patch.gamma
    if (patch.look !== undefined) v.look = patch.look
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
    this.device.queue.writeBuffer(this.compositeUniformBuffer, 0, u)
  }

  /** Patch bloom; GPU uniforms update immediately if `init()` has run. */
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
      [1, 1]
    )

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

    const shaderModule = this.device.createShaderModule({
      label: "default model shader",
      code: DEFAULT_SHADER_WGSL,
    })

    const faceShaderModule = this.device.createShaderModule({
      label: "face NPR shader",
      code: FACE_SHADER_WGSL,
    })

    const hairShaderModule = this.device.createShaderModule({
      label: "hair NPR shader",
      code: HAIR_SHADER_WGSL,
    })

    const clothSmoothShaderModule = this.device.createShaderModule({
      label: "cloth smooth NPR shader",
      code: CLOTH_SMOOTH_SHADER_WGSL,
    })

    const clothRoughShaderModule = this.device.createShaderModule({
      label: "cloth rough NPR shader",
      code: CLOTH_ROUGH_SHADER_WGSL,
    })

    const metalShaderModule = this.device.createShaderModule({
      label: "metal NPR shader",
      code: METAL_SHADER_WGSL,
    })

    const bodyShaderModule = this.device.createShaderModule({
      label: "body NPR shader",
      code: BODY_SHADER_WGSL,
    })

    const eyeShaderModule = this.device.createShaderModule({
      label: "eye shader",
      code: EYE_SHADER_WGSL,
    })

    const stockingsShaderModule = this.device.createShaderModule({
      label: "stockings NPR shader",
      code: STOCKINGS_SHADER_WGSL,
    })

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
      entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }],
    })
    // group 2: per-material (texture + material uniforms) — bound per draw call
    this.mainPerMaterialBindGroupLayout = this.device.createBindGroupLayout({
      label: "main per-material bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    })

    const mainPipelineLayout = this.device.createPipelineLayout({
      label: "main pipeline layout",
      bindGroupLayouts: [
        this.mainPerFrameBindGroupLayout,
        this.mainPerInstanceBindGroupLayout,
        this.mainPerMaterialBindGroupLayout,
      ],
    })

    // perFrameBindGroup is created after shadow resources below

    this.modelPipeline = this.createRenderPipeline({
      label: "model pipeline",
      layout: mainPipelineLayout,
      shaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    this.facePipeline = this.createRenderPipeline({
      label: "face NPR pipeline",
      layout: mainPipelineLayout,
      shaderModule: faceShaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    // Hair opaque: stencil != EYE_VALUE so fragments on top of eyes are skipped entirely —
    // depth and color stay as the eye wrote them; the follow-up hairOverEyesPipeline then
    // draws those skipped fragments alpha-blended so the eye reads through the hair.
    this.hairPipeline = this.createRenderPipeline({
      label: "hair NPR pipeline",
      layout: mainPipelineLayout,
      shaderModule: hairShaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        stencilFront: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "not-equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0,
      },
    })

    // Hair-over-eyes: same shader with IS_OVER_EYES=true so alpha is scaled to 25% at compile time.
    // Only fragments where eye stencil == EYE_VALUE pass; depth test still culls fragments
    // that are further from camera than the eye, so hair behind the eye never shows through.
    // depthWriteEnabled=false keeps the eye's depth authoritative for everything drawn after.
    this.hairOverEyesPipeline = this.device.createRenderPipeline({
      label: "hair over eyes pipeline",
      layout: mainPipelineLayout,
      vertex: { module: hairShaderModule, buffers: fullVertexBuffers },
      fragment: {
        module: hairShaderModule,
        constants: { IS_OVER_EYES: 1 },
        targets: sceneTargets,
      },
      primitive: { cullMode: "none" },
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: false,
        depthCompare: "less-equal",
        stencilFront: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilBack: { compare: "equal", failOp: "keep", depthFailOp: "keep", passOp: "keep" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0,
      },
      multisample: { count: Engine.MULTISAMPLE_COUNT },
    })

    this.clothSmoothPipeline = this.createRenderPipeline({
      label: "cloth smooth NPR pipeline",
      layout: mainPipelineLayout,
      shaderModule: clothSmoothShaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    this.clothRoughPipeline = this.createRenderPipeline({
      label: "cloth rough NPR pipeline",
      layout: mainPipelineLayout,
      shaderModule: clothRoughShaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    this.metalPipeline = this.createRenderPipeline({
      label: "metal NPR pipeline",
      layout: mainPipelineLayout,
      shaderModule: metalShaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    this.bodyPipeline = this.createRenderPipeline({
      label: "body NPR pipeline",
      layout: mainPipelineLayout,
      shaderModule: bodyShaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    // Eye: stamps stencil = EYE_VALUE on every fragment it writes. Later hair passes read
    // this stamp to split into "draw normally (not over eye)" vs "draw alpha-blended".
    // cullMode="front" + small negative depthBias is the MMD post-alpha-eye trick: only the
    // back half of the eye sphere renders, it passes depth against the face (via bias) when
    // viewed from the front, and it gets culled when viewed from behind — so eye fragments
    // can't leak through the back of the head without needing a per-model skull occluder.
    this.eyePipeline = this.createRenderPipeline({
      label: "eye pipeline",
      layout: mainPipelineLayout,
      shaderModule: eyeShaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "front",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        depthBias: -0.00005,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
        stencilFront: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "replace" },
        stencilBack: { compare: "always", failOp: "keep", depthFailOp: "keep", passOp: "replace" },
        stencilReadMask: 0xff,
        stencilWriteMask: 0xff,
      },
    })

    this.stockingsPipeline = this.createRenderPipeline({
      label: "stockings NPR pipeline",
      layout: mainPipelineLayout,
      shaderModule: stockingsShaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "none",
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
      ],
    })
    const shadowShader = this.device.createShaderModule({
      label: "shadow depth",
      code: SHADOW_DEPTH_SHADER_WGSL,
    })
    this.shadowDepthPipeline = this.device.createRenderPipeline({
      label: "shadow depth pipeline",
      layout: this.device.createPipelineLayout({ bindGroupLayouts: [shadowBindGroupLayout] }),
      vertex: { module: shadowShader, entryPoint: "vs", buffers: fullVertexBuffers as GPUVertexBufferLayout[] },
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
      vertexBuffers: fullVertexBuffers,
      fragmentTargets: sceneTargets,
      cullMode: "back",
      depthStencil: { format: "depth24plus-stencil8", depthWriteEnabled: true, depthCompare: "less-equal" },
    })

    // Outline: group 0 = per-frame (camera), group 1 = per-instance (skinMats), group 2 = per-material (edge uniforms)
    this.outlinePerFrameBindGroupLayout = this.device.createBindGroupLayout({
      label: "outline per-frame bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
      ],
    })
    // Outline per-instance reuses mainPerInstanceBindGroupLayout (same skinMats binding)
    this.outlinePerMaterialBindGroupLayout = this.device.createBindGroupLayout({
      label: "outline per-material bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
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
      entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
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
        // Don’t write outline into depth buffer — stops z-fighting / black cracks vs body (MMD-style; body depth stays authoritative)
        depthWriteEnabled: false,
        depthCompare: "less-equal",
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
      size: 32,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
      ],
    })

    const compositeShader = this.device.createShaderModule({
      label: "composite shader",
      code: COMPOSITE_SHADER_WGSL,
    })

    const compositePipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [this.compositeBindGroupLayout],
    })
    const makeCompositePipeline = (applyGamma: boolean, label: string): GPURenderPipeline =>
      this.device.createRenderPipeline({
        label,
        layout: compositePipelineLayout,
        vertex: { module: compositeShader, entryPoint: "vs" },
        fragment: {
          module: compositeShader,
          entryPoint: "fs",
          constants: { APPLY_GAMMA: applyGamma ? 1 : 0 },
          targets: [{ format: this.presentationFormat }],
        },
        primitive: { topology: "triangle-list" },
      })
    this.compositePipelineIdentity = makeCompositePipeline(false, "composite pipeline (gamma=1)")
    this.compositePipelineGamma = makeCompositePipeline(true, "composite pipeline (gamma!=1)")

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

  // Step 3: Setup canvas resize handling
  private setupResize() {
    this.resizeObserver = new ResizeObserver(() => this.handleResize())
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

  private handleResize() {
    const displayWidth = this.canvas.clientWidth
    const displayHeight = this.canvas.clientHeight

    const dpr = window.devicePixelRatio || 1
    const width = Math.floor(displayWidth * dpr)
    const height = Math.floor(displayHeight * dpr)

    if (!this.multisampleTexture || this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width
      this.canvas.height = height

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
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })

      const depthTextureView = this.depthTexture.createView()

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
        const compositeBloomView = this.bloomMipCount > 1 ? this.bloomUpMipViews[0] : this.bloomDownMipViews[0]
        this.compositeBindGroup = this.device.createBindGroup({
          label: "composite bind group",
          layout: this.compositeBindGroupLayout,
          entries: [
            { binding: 0, resource: this.hdrResolveTexture.createView() },
            { binding: 1, resource: compositeBloomView },
            { binding: 2, resource: this.bloomSampler },
            { binding: 3, resource: { buffer: this.compositeUniformBuffer } },
            { binding: 4, resource: this.maskResolveView },
            { binding: 5, resource: this.filmicLutView },
          ],
        })
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
    const pushSeg = (
      p0: [number, number, number],
      p1: [number, number, number],
      t0: number,
      t1: number,
    ) => {
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
        const c0 = Math.cos(t0) * R, s0 = Math.sin(t0) * R
        const c1 = Math.cos(t1) * R, s1 = Math.sin(t1) * R
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

    this.camera = new Camera(
      Math.PI,
      Math.PI / 2.5,
      this.cameraConfig.distance,
      this.cameraConfig.target,
      this.cameraConfig.fov,
    )

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
  setCameraFollow(model: Model | null, boneName?: string, offset?: Vec3): void {
    if (model === null) {
      this.cameraTargetModel = null
      return
    }
    this.cameraTargetModel = model
    this.cameraTargetBoneName = boneName ?? "全ての親"
    this.cameraTargetOffset.x = offset?.x ?? 0
    this.cameraTargetOffset.y = offset?.y ?? 0
    this.cameraTargetOffset.z = offset?.z ?? 0
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
      preset: "cloth_rough",
    }
  }

  private updateLightBuffer() {
    this.device.queue.writeBuffer(this.lightUniformBuffer, 0, this.lightData)
  }

  getStats(): EngineStats {
    return { ...this.stats }
  }

  runRenderLoop(callback?: () => void) {
    this.renderLoopCallback = callback || null

    const loop = () => {
      this.render()

      if (this.renderLoopCallback) {
        this.renderLoopCallback()
      }

      this.animationFrameId = requestAnimationFrame(loop)
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
    this.forEachInstance((inst) => inst.model.stopAnimation())
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
      const pmxKey = normalizeAssetPath(
        (pmxFile as File & { webkitRelativePath?: string }).webkitRelativePath ?? pmxFile.name,
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

  async addModel(model: Model, pmxPath: string, name?: string, assetReader?: AssetReader): Promise<string> {
    const requested = name ?? model.name
    let key = requested
    let n = 1
    while (this.modelInstances.has(key)) {
      key = `${requested}_${n++}`
    }
    const reader = assetReader ?? createFetchAssetReader()
    const basePath = deriveBasePathFromPmxPath(pmxPath)
    model.setAssetContext(reader, basePath)
    await this.setupModelInstance(key, model, basePath, reader)
    return key
  }

  removeModel(name: string): void {
    const inst = this.modelInstances.get(name)
    if (!inst) return
    inst.model.stopAnimation()
    for (const path of inst.textureCacheKeys) {
      const tex = this.textureCache.get(path)
      if (tex) {
        tex.destroy()
        this.textureCache.delete(path)
      }
    }
    for (const buf of inst.gpuBuffers) {
      buf.destroy()
    }
    this.modelInstances.delete(name)
  }

  getModelNames(): string[] {
    return Array.from(this.modelInstances.keys())
  }

  getModel(name: string): Model | null {
    return this.modelInstances.get(name)?.model ?? null
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

  setMaterialPresets(modelName: string, presets: MaterialPresetMap): void {
    const inst = this.modelInstances.get(modelName)
    if (!inst) return
    inst.materialPresets = presets
    for (const dc of inst.drawCalls) {
      dc.preset = resolvePreset(dc.materialName, presets)
    }
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

  private updateInstances(deltaTime: number): void {
    this.forEachInstance((inst) => {
      const verticesChanged = inst.model.update(deltaTime, this.ikEnabled)
      if (verticesChanged) inst.vertexBufferNeedsUpdate = true
      if (inst.physics && this.physicsEnabled) {
        inst.physics.step(deltaTime, inst.model.getWorldMatrices(), inst.model.getBoneInverseBindMatrices())
      }
      if (inst.vertexBufferNeedsUpdate) this.updateVertexBuffer(inst)
    })
  }

  private updateVertexBuffer(inst: ModelInstance): void {
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

  private async setupModelInstance(
    name: string,
    model: Model,
    basePath: string,
    assetReader: AssetReader,
  ): Promise<void> {
    const vertices = model.getVertices()
    const skinning = model.getSkinning()
    const skeleton = model.getSkeleton()
    const boneCount = skeleton.bones.length
    const matrixSize = boneCount * 16 * 4

    const vertexBuffer = this.device.createBuffer({
      label: `${name}: vertex buffer`,
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
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
    const physics = rbs.length > 0 ? new RezePhysics(rbs, model.getJoints()) : null

    const shadowBindGroup = this.device.createBindGroup({
      label: `${name}: shadow bind`,
      layout: this.shadowDepthPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.shadowLightVPBuffer } },
        { binding: 1, resource: { buffer: skinMatrixBuffer } },
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
      hiddenMaterials: new Set(),
      materialPresets: undefined,
      physics,
      vertexBufferNeedsUpdate: false,
    }
    await this.setupMaterialsForInstance(inst)
    this.modelInstances.set(name, inst)
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
    gb[11] = 0
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
  private shadowLightVPDirty = true
  private updateShadowLightVP() {
    if (!this.shadowLightVPDirty) return
    this.shadowLightVPDirty = false
    const dir = new Vec3(this.sun.direction.x, this.sun.direction.y, this.sun.direction.z)
    dir.normalize()
    const target = new Vec3(0, 11, 0)
    const eye = new Vec3(target.x - dir.x * 72, target.y - dir.y * 72, target.z - dir.z * 72)
    const up = Math.abs(dir.y) > 0.99 ? new Vec3(0, 0, -1) : new Vec3(0, 1, 0)
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

    const loadTextureByIndex = async (texIndex: number): Promise<GPUTexture | null> => {
      if (texIndex < 0 || texIndex >= textures.length) return null
      const logicalPath = joinAssetPath(inst.basePath, normalizeAssetPath(textures[texIndex].path))
      return this.createTextureFromLogicalPath(inst, logicalPath)
    }

    let currentIndexOffset = 0
    let materialId = 0
    for (const mat of materials) {
      const indexCount = mat.vertexCount
      if (indexCount === 0) continue
      materialId++

      let diffuseTexture = await loadTextureByIndex(mat.diffuseTextureIndex)
      if (!diffuseTexture) {
        console.warn(`${prefix}material "${mat.name}" has no loadable diffuse texture — using fallback`)
        diffuseTexture = this.fallbackMaterialTexture
      }

      const materialAlpha = mat.diffuse[3]
      const isTransparent = materialAlpha < 1.0 - 0.001

      const materialUniformBuffer = this.createMaterialUniformBuffer(prefix + mat.name, materialAlpha, [
        mat.diffuse[0],
        mat.diffuse[1],
        mat.diffuse[2],
      ])
      inst.gpuBuffers.push(materialUniformBuffer)

      const textureView = diffuseTexture.createView()
      const bindGroup = this.device.createBindGroup({
        label: `${prefix}material: ${mat.name}`,
        layout: this.mainPerMaterialBindGroupLayout,
        entries: [
          { binding: 0, resource: textureView },
          { binding: 1, resource: { buffer: materialUniformBuffer } },
        ],
      })

      const type: DrawCallType = isTransparent ? "transparent" : "opaque"
      const preset = resolvePreset(mat.name, inst.materialPresets)
      inst.drawCalls.push({
        type,
        count: indexCount,
        firstIndex: currentIndexOffset,
        bindGroup,
        materialName: mat.name,
        preset,
      })

      if ((mat.edgeFlag & 0x10) !== 0 && mat.edgeSize > 0) {
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
          entries: [{ binding: 0, resource: { buffer: outlineUniformBuffer } }],
        })
        const outlineType: DrawCallType = isTransparent ? "transparent-outline" : "opaque-outline"
        inst.drawCalls.push({
          type: outlineType,
          count: indexCount,
          firstIndex: currentIndexOffset,
          bindGroup: outlineBindGroup,
          materialName: mat.name,
          preset,
        })
      }

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

    // Sort so the opaque bucket is emitted in the order the stencil-based
    // see-through-hair effect requires: {non-hair, non-eye} → {eye} → {hair}.
    // Eye writes stencil=EYE_VALUE; hair's pipeline stencil-tests "not equal" so
    // it skips eye pixels; a follow-up hairOverEyes pass (see renderOneModel)
    // re-fills those skipped pixels alpha-blended. Array.sort is stable in
    // ES2019+, so within a bucket the PMX material order is preserved.
    const typeOrder: Record<DrawCallType, number> = {
      opaque: 0,
      "opaque-outline": 1,
      transparent: 2,
      "transparent-outline": 3,
      ground: 4,
    }
    const presetRank = (p: MaterialPreset): number => (p === "hair" ? 2 : p === "eye" ? 1 : 0)
    inst.drawCalls.sort((a, b) => {
      const ta = typeOrder[a.type] - typeOrder[b.type]
      if (ta !== 0) return ta
      return presetRank(a.preset) - presetRank(b.preset)
    })

    for (const d of inst.drawCalls) {
      if (d.type === "opaque") inst.shadowDrawCalls.push(d)
    }
  }

  private createMaterialUniformBuffer(label: string, alpha: number, diffuseColor: [number, number, number]): GPUBuffer {
    // Matches WGSL `struct MaterialUniforms { diffuseColor: vec3f, alpha: f32 }` — 16 bytes.
    const data = new Float32Array(4)
    data[0] = diffuseColor[0]
    data[1] = diffuseColor[1]
    data[2] = diffuseColor[2]
    data[3] = alpha
    return this.createUniformBuffer(`material uniform: ${label}`, data)
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
    return !inst.hiddenMaterials.has(drawCall.materialName)
  }

  private async createTextureFromLogicalPath(inst: ModelInstance, logicalPath: string): Promise<GPUTexture | null> {
    const cacheKey = logicalPath
    const cached = this.textureCache.get(cacheKey)
    if (cached) {
      return cached
    }

    try {
      const buffer = await inst.assetReader.readBinary(logicalPath)
      const imageBitmap = await createImageBitmap(new Blob([buffer]), {
        premultiplyAlpha: "none",
        colorSpaceConversion: "none",
      })

      const mipLevelCount = Math.floor(Math.log2(Math.max(imageBitmap.width, imageBitmap.height))) + 1
      const texture = this.device.createTexture({
        label: `texture: ${cacheKey}`,
        size: [imageBitmap.width, imageBitmap.height],
        format: "rgba8unorm-srgb",
        mipLevelCount,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture }, [
        imageBitmap.width,
        imageBitmap.height,
      ])

      if (mipLevelCount > 1) this.generateMipmaps(texture, mipLevelCount)

      this.textureCache.set(cacheKey, texture)
      inst.textureCacheKeys.push(cacheKey)
      return texture
    } catch {
      return null
    }
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
    const xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z
    const xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z
    const wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z
    const u = new Float32Array(20)
    u[0] = s * (1 - 2 * (yy + zz)); u[1] = s * 2 * (xy + wz);     u[2] = s * 2 * (xz - wy);     u[3] = 0
    u[4] = s * 2 * (xy - wz);       u[5] = s * (1 - 2 * (xx + zz)); u[6] = s * 2 * (yz + wx);   u[7] = 0
    u[8] = s * 2 * (xz + wy);       u[9] = s * 2 * (yz - wx);     u[10] = s * (1 - 2 * (xx + yy)); u[11] = 0
    u[12] = bonePos.x; u[13] = bonePos.y; u[14] = bonePos.z; u[15] = 1
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
    const camPos = this.camera.getPosition()
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
      const lNew = drag.parentWorldRotInv
        .multiply(qWorld)
        .multiply(drag.parentWorldRot)
        .multiply(drag.initialLocalRot)
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

    const hasModels = this.modelInstances.size > 0
    if (hasModels) {
      this.updateInstances(deltaTime)
      this.updateSkinMatrices()
      // Update camera target from bound model (bone not found → 0,0,0 + offset)
      if (this.cameraTargetModel) {
        const pos = this.cameraTargetModel.getBoneWorldPosition(this.cameraTargetBoneName)
        const px = pos?.x ?? 0
        const py = pos?.y ?? 0
        const pz = pos?.z ?? 0
        this.camera.target.x = px + this.cameraTargetOffset.x
        this.camera.target.y = py + this.cameraTargetOffset.y
        this.camera.target.z = pz + this.cameraTargetOffset.z
      }
    }

    this.updateCameraUniforms()
    this.updateShadowLightVP()

    const encoder = this.device.createCommandEncoder()
    if (hasModels) {
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
      this.forEachInstance((inst) => this.drawInstanceShadow(sp, inst))
      sp.end()
    }

    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    if (hasModels) this.forEachInstance((inst) => this.renderOneModel(pass, inst))
    if (this.hasGround) this.renderGround(pass)
    pass.end()

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
      if (this.shouldRenderDrawCall(inst, draw)) sp.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    }
  }

  private pipelineForPreset(preset: MaterialPreset): GPURenderPipeline {
    if (preset === "face") return this.facePipeline
    if (preset === "hair") return this.hairPipeline
    if (preset === "cloth_smooth") return this.clothSmoothPipeline
    if (preset === "cloth_rough") return this.clothRoughPipeline
    if (preset === "metal") return this.metalPipeline
    if (preset === "body") return this.bodyPipeline
    if (preset === "eye") return this.eyePipeline
    if (preset === "stockings") return this.stockingsPipeline
    return this.modelPipeline
  }

  /**
   * Draw every material of a given type (`opaque` or `transparent`) using the main
   * pipeline(s). Binds the per-frame and per-instance groups once at the top of the
   * batch, then issues one draw per material. Early-outs if nothing to draw so we
   * don't waste bindings when a model has no transparents, etc.
   */
  private drawMaterials(pass: GPURenderPassEncoder, inst: ModelInstance, type: "opaque" | "transparent"): void {
    let currentPipeline: GPURenderPipeline | null = null
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== type || !this.shouldRenderDrawCall(inst, draw)) continue
      if (!bound) {
        pass.setBindGroup(0, this.perFrameBindGroup)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      const pipeline = this.pipelineForPreset(draw.preset)
      if (pipeline !== currentPipeline) {
        pass.setPipeline(pipeline)
        currentPipeline = pipeline
      }
      pass.setBindGroup(2, draw.bindGroup)
      pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    }
  }

  /**
   * Draw every outline of a given type (`opaque-outline` or `transparent-outline`).
   * Uses its own pipeline layout (group 0 = camera-only, group 2 = edge uniforms), so
   * every batch binds its own groups from scratch — the next drawMaterials call will
   * rebind group 0/1 correctly if needed.
   */
  private drawOutlines(pass: GPURenderPassEncoder, inst: ModelInstance, type: DrawCallType): void {
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== type || !this.shouldRenderDrawCall(inst, draw)) continue
      if (!bound) {
        pass.setPipeline(this.outlinePipeline)
        pass.setBindGroup(0, this.outlinePerFrameBindGroup)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      pass.setBindGroup(2, draw.bindGroup)
      pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    }
  }

  /**
   * Main-pass render sequence for one model instance:
   *   1) opaque bodies → 2) opaque outlines → 3) transparents → 4) transparent outlines.
   * Each batch binds the groups it needs, so switching between main and outline
   * pipelines is self-contained (no cross-batch dependencies).
   */
  private renderOneModel(pass: GPURenderPassEncoder, inst: ModelInstance): void {
    pass.setVertexBuffer(0, inst.vertexBuffer)
    pass.setVertexBuffer(1, inst.jointsBuffer)
    pass.setVertexBuffer(2, inst.weightsBuffer)
    pass.setIndexBuffer(inst.indexBuffer, "uint32")

    // Single stencil-reference set covers eye (write), hair (read not-equal),
    // and hairOverEyes (read equal). Non-stencil pipelines ignore the value.
    pass.setStencilReference(Engine.STENCIL_EYE_VALUE)

    this.drawMaterials(pass, inst, "opaque")
    this.drawOutlines(pass, inst, "opaque-outline")
    this.drawHairOverEyes(pass, inst)
    this.drawMaterials(pass, inst, "transparent")
    this.drawOutlines(pass, inst, "transparent-outline")
  }

  /**
   * Second hair pass for the see-through-hair effect. Re-draws every hair opaque
   * draw using `hairOverEyesPipeline` — which stencil-matches `EYE_VALUE` and runs
   * the hair shader with `IS_OVER_EYES=true` so alpha is scaled to 25%. depthWriteEnabled
   * is off, so the eye's depth stays authoritative for anything drawn after.
   */
  private drawHairOverEyes(pass: GPURenderPassEncoder, inst: ModelInstance): void {
    let bound = false
    for (const draw of inst.drawCalls) {
      if (draw.type !== "opaque" || draw.preset !== "hair" || !this.shouldRenderDrawCall(inst, draw)) continue
      if (!bound) {
        pass.setPipeline(this.hairOverEyesPipeline)
        pass.setBindGroup(0, this.perFrameBindGroup)
        pass.setBindGroup(1, inst.mainPerInstanceBindGroup)
        bound = true
      }
      pass.setBindGroup(2, draw.bindGroup)
      pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
    }
  }

  private updateCameraUniforms() {
    const viewMatrix = this.camera.getViewMatrix()
    const projectionMatrix = this.camera.getProjectionMatrix()
    const cameraPos = this.camera.getPosition()
    this.cameraMatrixData.set(viewMatrix.values, 0)
    this.cameraMatrixData.set(projectionMatrix.values, 16)
    this.cameraMatrixData[32] = cameraPos.x
    this.cameraMatrixData[33] = cameraPos.y
    this.cameraMatrixData[34] = cameraPos.z
    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)
  }

  private updateSkinMatrices() {
    this.forEachInstance((inst) => {
      const skinMatrices = inst.model.getSkinMatrices()
      this.device.queue.writeBuffer(
        inst.skinMatrixBuffer,
        0,
        skinMatrices.buffer,
        skinMatrices.byteOffset,
        skinMatrices.byteLength,
      )
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
  }
}
