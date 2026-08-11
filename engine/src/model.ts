import { Mat4, Quat, Vec3, easeInOut, scratchMat4Values, scratchQuat } from "./math"
import { Engine } from "./engine"
import { joinAssetPath, type AssetReader } from "./asset-reader"
import { Rigidbody, Joint } from "./physics"
import { IKSolverSystem } from "./ik-solver"
import { VMDLoader, type VMDKeyFrame } from "./vmd-loader"
import { VMDWriter } from "./vmd-writer"
import {
  AnimationClip,
  AnimationPlayOptions,
  AnimationProgress,
  AnimationState,
  BlendEntry,
  BoneInterpolation,
  BoneKeyframe,
  FPS,
  IkKeyframe,
  MorphKeyframe,
  interpolateControlPoints,
  rawInterpolationToBoneInterpolation,
} from "./animation"

const VERTEX_STRIDE = 8

// Animation-sampling scratch (applyPoseFromClip → convertVMDTranslationToLocal). These
// run sequentially and are not reentrant with the world-matrix scratch in math.ts, so
// plain module singletons are safe and let the per-bone sample path allocate nothing.
const _animSlerp = new Quat(0, 0, 0, 1)
const _animInterpT = new Vec3(0, 0, 0)
const _convOut = new Vec3(0, 0, 0)
const _convMat = new Float32Array(16)
// Blend-path scratch: per-entry sample target and the crossfade's two fixed entries.
const _blendQ = new Quat(0, 0, 0, 1)
const _blendT = new Vec3(0, 0, 0)

// Bone-morph scratch: the weighted rotation, slerped out of identity.
const _boneMorphQ = new Quat(0, 0, 0, 1)
const _boneMorphIdentity = new Quat(0, 0, 0, 1)
export interface ClipEventInfo {
  clip: string
  /** The registered event time, seconds. */
  time: number
  /** The clip's blend weight at the moment of firing. */
  weight: number
}

/** A clip's authored horizontal root path, lifted off センター by
 *  Model.extractRootMotion: per-key offsets from the FIRST key, raw clip
 *  units, clip space (rest facing -Z). `frames` are VMD frames (30fps). */
export interface RootMotionProfile {
  frames: number[]
  x: number[]
  z: number[]
}

const _fadeEntries: BlendEntry[] = [
  { name: "", time: 0, weight: 0 },
  { name: "", time: 0, weight: 0 },
]

export interface Texture {
  path: string
  name: string
}

export interface Material {
  name: string
  diffuse: [number, number, number, number]
  specular: [number, number, number]
  ambient: [number, number, number]
  shininess: number
  diffuseTextureIndex: number
  normalTextureIndex: number
  sphereTextureIndex: number
  sphereMode: number
  toonTextureIndex: number
  // True when toonTextureIndex refers to the shared toon set (toon01–10)
  // instead of the model's texture table.
  sharedToon: boolean
  edgeFlag: number
  edgeColor: [number, number, number, number]
  edgeSize: number
  vertexCount: number
}

export interface Bone {
  name: string
  parentIndex: number // -1 if no parent
  bindTranslation: [number, number, number]
  children: number[] // child bone indices (built on skeleton creation)
  appendParentIndex?: number // index of the bone to inherit from
  appendRatio?: number // 0..1
  appendRotate?: boolean
  appendMove?: boolean
  /** 軸制限: this bone rotates ONLY about this axis (the twist bones use it). */
  fixedAxis?: [number, number, number]
  /** 変形階層: MMD poses whole layers in order, not bones in array order.
   *  Parsed but not yet honoured — distinct from Model's `deformOrder`, which is
   *  this engine's parent-before-child traversal. */
  deformLayer?: number
  ikTargetIndex?: number // IK target bone index (if this bone is an IK effector)
  ikIteration?: number // IK iteration count
  ikLimitAngle?: number // IK rotation constraint (radians)
  ikLinks?: IKLink[] // IK chain links
}

// IK link with angle constraints
export interface IKLink {
  boneIndex: number
  hasLimit: boolean
  minAngle?: Vec3 // Minimum Euler angles (radians)
  maxAngle?: Vec3 // Maximum Euler angles (radians)
}

// IK solver definition
export interface IKSolver {
  index: number
  ikBoneIndex: number // Effector bone (the bone that should reach the target)
  targetBoneIndex: number // Target bone
  iterationCount: number
  limitAngle: number // Max rotation per iteration (radians)
  links: IKLink[] // Chain bones from effector to root
}

// IK chain info per bone (runtime state)
export interface IKChainInfo {
  ikRotation: Quat // Accumulated IK rotation
  localRotation: Quat // Cached local rotation before IK
}

export interface Skeleton {
  bones: Bone[]
  inverseBindMatrices: Float32Array // One inverse-bind matrix per bone (column-major mat4, 16 floats per bone)
}

export interface Skinning {
  joints: Uint16Array // length = vertexCount * 4, bone indices per vertex
  weights: Uint8Array // UNORM8, length = vertexCount * 4, sums ~ 255 per-vertex
}

// Vertex morph offset data
export interface VertexMorphOffset {
  vertexIndex: number
  positionOffset: [number, number, number]
}

// Group morph reference (for type 0)
export interface GroupMorphReference {
  morphIndex: number
  ratio: number
}

// Bone morph offset data (type 2). A stage's doors, platforms and hatches are
// posed with these — there is no VMD to drive them, so the weight IS the pose.
export interface BoneMorphOffset {
  boneIndex: number
  translation: [number, number, number]
  /** Rotation quaternion (x, y, z, w). */
  rotation: [number, number, number, number]
}

/** PMX material-morph blend mode. Multiply lerps toward base*morph; add offsets from base. */
export const MATERIAL_MORPH_MULTIPLY = 0
export const MATERIAL_MORPH_ADD = 1

// Material morph offset data (type 8). Stage artists ship colour and on/off
// switches this way: alpha to 0 hides a part, a diffuse tint recolours a set.
export interface MaterialMorphOffset {
  /** -1 targets EVERY material in the model, per the PMX spec. */
  materialIndex: number
  /** MATERIAL_MORPH_MULTIPLY | MATERIAL_MORPH_ADD */
  offsetType: number
  diffuse: [number, number, number, number]
  specular: [number, number, number]
  shininess: number
  ambient: [number, number, number]
  edgeColor: [number, number, number, number]
  edgeSize: number
  textureCoeff: [number, number, number, number]
  sphereCoeff: [number, number, number, number]
  toonCoeff: [number, number, number, number]
}

// UV morph offset data (types 3–7). Type 3 is the base UV channel; 4–7 are the
// additional UV channels, which this engine does not carry — kept so the panel
// can tell "unsupported" from "absent".
export interface UvMorphOffset {
  vertexIndex: number
  /** (u, v, z, w) — only u/v apply to the base channel. */
  offset: [number, number, number, number]
}

// Morph definition
export interface Morph {
  name: string
  /** 0=group, 1=vertex, 2=bone, 3–7=UV, 8=material, 9=flip, 10=impulse. */
  type: number
  vertexOffsets: VertexMorphOffset[] // Only for type 1 (vertex morph)
  groupReferences?: GroupMorphReference[] // Only for type 0 (group morph)
  boneOffsets?: BoneMorphOffset[] // Only for type 2
  materialOffsets?: MaterialMorphOffset[] // Only for type 8
  uvOffsets?: UvMorphOffset[] // Only for types 3–7
}

export interface Morphing {
  morphs: Morph[]
}

// CSR inversion of vertex-morph offsets for the GPU compute pass (built once at load).
export interface MorphComputeData {
  basePositions: Float32Array // vertexCount * 3
  rowStart: Uint32Array // vertexCount + 1 (prefix offsets into the entry arrays)
  colMorph: Uint32Array // entryCount (morph index per entry)
  colOffset: Float32Array // entryCount * 3 (xyz offset per entry)
  morphCount: number
  vertexCount: number
  entryCount: number
}

// Runtime skeleton pose state (updated each frame)
export interface SkeletonRuntime {
  nameIndex: Record<string, number> // Cached lookup: bone name -> bone index (built on initialization)
  localRotations: Quat[] // quat per bone
  localTranslations: Vec3[] // vec3 per bone
  worldMatrices: Mat4[] // mat4 per bone
  ikChainInfo?: IKChainInfo[] // IK chain info per bone (only for IK chain bones)
  ikSolvers?: IKSolver[] // All IK solvers in the model
}

// Runtime morph state
export interface MorphRuntime {
  nameIndex: Record<string, number> // Cached lookup: morph name -> morph index
  weights: Float32Array // One weight per morph (0.0 to 1.0)
}

// Tween state - combines rotation, translation, and morph tweens
// All tweens share the same time reference to avoid conflicts
interface TweenState {
  // Bone rotation tweens
  rotActive: Uint8Array // 0/1 per bone
  rotStartQuat: Quat[]
  rotTargetQuat: Quat[]
  rotStartTimeMs: Float32Array // one float per bone (ms)
  rotDurationMs: Float32Array // one float per bone (ms)

  // Bone translation tweens
  transActive: Uint8Array // 0/1 per bone
  transStartVec: Vec3[] // vec3 per bone (x,y,z)
  transTargetVec: Vec3[] // vec3 per bone (x,y,z)
  transStartTimeMs: Float32Array // one float per bone (ms)
  transDurationMs: Float32Array // one float per bone (ms)

  // Morph weight tweens
  morphActive: Uint8Array // 0/1 per morph
  morphStartWeight: Float32Array // one float per morph
  morphTargetWeight: Float32Array // one float per morph
  morphStartTimeMs: Float32Array // one float per morph (ms)
  morphDurationMs: Float32Array // one float per morph (ms)
}

export class Model {
  private _name: string = ""

  get name(): string {
    return this._name
  }

  setName(value: string): void {
    this._name = value
  }

  // Root transform public API. Instant setters — no tween baked in; wrap in
  // your own lerp if you need smoothing. Changes are applied on the next
  // getSkinMatrices() call (once per frame during rendering).
  get position(): Vec3 {
    return this._position
  }

  get rotation(): Quat {
    return this._rotation
  }

  /** Uniform scale (default 1). Used to fit a stage.pmx to the character. */
  get scale(): number {
    return this._scale
  }

  /** Whether this model renders. Hidden models skip the main, shadow, and pick passes;
   *  physics keeps running so they resume consistently. */
  get visible(): boolean {
    return this._visible
  }

  setPosition(position: Vec3): void {
    this._position.set(position)
    this.rootMatrixDirty = true
  }

  setRotation(rotation: Quat): void {
    this._rotation.set(rotation)
    this.rootMatrixDirty = true
  }

  setScale(scale: number): void {
    this._scale = scale
    this.rootMatrixDirty = true
  }

  setVisible(visible: boolean): void {
    this._visible = visible
  }

  private vertexData: Float32Array<ArrayBuffer>
  private baseVertexData: Float32Array<ArrayBuffer> // Original vertex data before morphing
  private vertexCount: number
  private indexData: Uint32Array<ArrayBuffer>

  // Morph state reused across frames (S1) + partial vertex-upload range tracking (S2, CPU path).
  private morphEffectiveWeights?: Float32Array
  private morphPrevMinVert = -1 // vertices touched by last applyMorphs (reset to base this pass)
  private morphPrevMaxVert = -1
  private morphPendingMinVert = -1 // accumulated range awaiting a GPU upload
  private morphPendingMaxVert = -1
  private morphUploadFull = true // first upload after load pushes the whole buffer once
  // GPU morph path: when enabled (engine set up the compute buffers), applyMorphs only
  // resolves effective weights and flags them dirty — the compute pass does the vertex work.
  private gpuMorphEnabled = false
  private morphWeightsDirty = false
  private textures: Texture[] = []
  private materials: Material[] = []
  // Static skeleton/skinning (not necessarily serialized yet)
  private skeleton: Skeleton
  private skinning: Skinning

  // Static morph data (from PMX)
  private morphing: Morphing

  // Physics data from PMX
  private rigidbodies: Rigidbody[] = []
  private joints: Joint[] = []

  // Non-fatal problems collected while parsing the PMX (see PmxLoader.warn).
  private loadWarnings: string[] = []

  // Runtime skeleton pose state (updated each frame)
  private runtimeSkeleton!: SkeletonRuntime

  // Runtime morph state
  private runtimeMorph!: MorphRuntime
  private morphsDirty: boolean = false // Flag indicating if morphs need to be applied

  // Bone morphs (type 2), flattened once at load so the per-frame pass is a
  // straight walk with no allocation. Empty for models without them, which is
  // most characters — a stage's doors and platforms are the common case.
  private boneMorphPlan: { morphIndex: number; boneIndex: number; translation: Vec3; rotation: Quat }[] = []
  // Bones any bone morph touches, plus their locals as they were BEFORE the last
  // application. A pose source rewrites the locals it owns every frame, but a
  // stage usually has no clip at all — nothing would reset these, and the offset
  // would compound frame after frame into a slow drift.
  private boneMorphBones: number[] = []
  private boneMorphRestoreR: Quat[] = []
  private boneMorphRestoreT: Vec3[] = []
  private boneMorphApplied = false
  /** A full pose pass has run, so the world matrices are real. See isIdle. */
  private posedOnce = false
  // Set whenever effective weights change, for material/UV consumers that live
  // outside this class. Vertex morphs use morphWeightsDirty (GPU) instead.
  private auxMorphDirty = true

  // Root transform — model's placement in world space, independent of bones.
  // Folded into skin matrices (see getSkinMatrices) so every pass (main VS,
  // shadow VS, any future skinned pass) sees it without per-shader plumbing.
  // Skip-when-identity flag avoids the extra mat mul per bone when unused.
  private _position: Vec3 = Vec3.zeros()
  private _rotation: Quat = Quat.identity()
  private _scale: number = 1
  private _visible: boolean = true
  private rootMatrixValues: Float32Array = new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
  private rootMatrixDirty: boolean = false
  private rootIsIdentity: boolean = true

  // Cached skin matrices array to avoid allocations in getSkinMatrices
  private skinMatricesArray?: Float32Array

  // Static bone traversal order (parents before children), precomputed once at load.
  // computeWorldMatrices replays this flat instead of recursing with a per-call
  // visited-array + closure. Order depends only on parentIndex (static), so this
  // reproduces the old recursion's finishing order exactly. See buildDeformOrder.
  private deformOrder!: Int32Array

  // Bind-pose absolute (world) position per bone, xyz packed. Static (bindTranslation
  // accumulated down the hierarchy). Precomputed once so convertVMDTranslationToLocal
  // stops re-deriving it recursively (with per-ancestor allocations) every frame.
  private bindWorldPos!: Float32Array

  private tweenState!: TweenState
  private tweenTimeMs: number = 0 // Time tracking for tweens (milliseconds)

  // Animation: state and multiple slots (idle, walk, attack, etc.); commit/rollback for action-game style
  private readonly animationState = new AnimationState()
  private boneTrackIndices: Map<string, number> = new Map()
  private morphTrackIndices: Map<string, number> = new Map()
  private lastAppliedClip: AnimationClip | null = null

  // One-shot action layer: plays a clip ONCE over whatever else drives the pose
  // (locomotion blend, a playing clip, a crossfade, or rest), with fade-in/out
  // envelopes — the background keeps advancing and is what the fade-out returns to.
  private oneShot: {
    name: string
    time: number
    duration: number
    fadeIn: number
    fadeOut: number
    cancelW: number
    cancelling: boolean
    onEnd: (() => void) | null
  } | null = null
  private readonly oneShotEntries: BlendEntry[] = []
  /** Time-triggered clip callbacks (footsteps, skill timing). Fired by every
   *  playback path when a clip's cursor crosses the event time with weight. */
  private readonly clipEvents = new Map<string, { time: number; minWeight: number; callback: (e: ClipEventInfo) => void }[]>()
  private readonly entryEventPrev = new WeakMap<BlendEntry, { name: string; time: number }>()

  // Blended pose: declarative N-clip mix (setBlendPose) or a running crossfade.
  // Cursor caches are per clip so sampling several clips in one frame doesn't
  // thrash the single-clip caches above; WeakMap so removed clips can collect.
  private blendEntries: BlendEntry[] | null = null
  private crossfade: { fromName: string | null; fromFrame: number; fromLoop: boolean; elapsed: number; duration: number } | null = null
  private readonly blendBoneCursors = new WeakMap<AnimationClip, Map<string, number>>()
  private readonly blendMorphCursors = new WeakMap<AnimationClip, Map<string, number>>()
  // Per-bone accumulators (lazy — sized to the skeleton on first blend). Generation
  // marks avoid a clear pass: a slot is live this frame only when gen matches.
  private blendRotAcc: Quat[] | null = null
  private blendTransAcc: Vec3[] | null = null
  private blendWeightAcc: Float32Array | null = null
  private blendBoneGen: Int32Array | null = null
  private blendMorphAcc: Float32Array | null = null
  private blendMorphGen: Int32Array | null = null
  private blendGenCounter = 0

  private assetReader: AssetReader | null = null
  private assetBasePath = ""

  /** Called by Engine when registering the model; enables loadVmd to resolve relative paths for folder uploads. */
  setAssetContext(reader: AssetReader, basePath: string): void {
    this.assetReader = reader
    this.assetBasePath = basePath
  }

  constructor(
    vertexData: Float32Array<ArrayBuffer>,
    indexData: Uint32Array<ArrayBuffer>,
    textures: Texture[],
    materials: Material[],
    skeleton: Skeleton,
    skinning: Skinning,
    morphing: Morphing,
    rigidbodies: Rigidbody[] = [],
    joints: Joint[] = [],
    loadWarnings: string[] = []
  ) {
    // Store base vertex data (original positions before morphing)
    this.baseVertexData = new Float32Array(vertexData)
    this.vertexData = vertexData
    this.vertexCount = vertexData.length / VERTEX_STRIDE
    this.indexData = indexData
    this.textures = textures
    this.materials = materials
    this.skeleton = skeleton
    this.skinning = skinning
    this.morphing = morphing
    this.rigidbodies = rigidbodies
    this.joints = joints
    this.loadWarnings = loadWarnings

    if (this.skeleton.bones.length == 0) {
      throw new Error("Model has no bones")
    }

    this.initializeRuntimeSkeleton()
    this.initializeRuntimeMorph()
    this.initializeTweenBuffers()
    this.applyMorphs()
  }

  /** 軸制限 bones and their normalised axes — see applyFixedAxes. */
  private fixedAxisBones: { index: number; x: number; y: number; z: number }[] = []

  private initializeRuntimeSkeleton(): void {
    const boneCount = this.skeleton.bones.length

    // Pre-allocate object arrays for skeletal pose
    const localRotations: Quat[] = new Array(boneCount)
    const localTranslations: Vec3[] = new Array(boneCount)
    const worldMatrices: Mat4[] = new Array(boneCount)
    for (let i = 0; i < boneCount; i++) {
      localRotations[i] = Quat.identity()
      localTranslations[i] = Vec3.zeros()
      worldMatrices[i] = Mat4.identity()
    }

    // 軸制限 bones, gathered once and normalised. A standard rig has four —
    // the two arm twists and the two wrist twists — so this is a four-element
    // walk per pose rather than a branch on every bone.
    this.fixedAxisBones = []
    for (let i = 0; i < boneCount; i++) {
      const a = this.skeleton.bones[i].fixedAxis
      if (!a) continue
      const len = Math.hypot(a[0], a[1], a[2])
      if (len < 1e-8) continue
      this.fixedAxisBones.push({ index: i, x: a[0] / len, y: a[1] / len, z: a[2] / len })
    }

    this.runtimeSkeleton = {
      localRotations,
      localTranslations,
      worldMatrices,
      nameIndex: this.skeleton.bones.reduce((acc, bone, index) => {
        acc[bone.name] = index
        return acc
      }, {} as Record<string, number>),
    }

    // Initialize IK runtime state
    this.initializeIKRuntime()
  }

  private initializeIKRuntime(): void {
    const boneCount = this.skeleton.bones.length
    const bones = this.skeleton.bones

    // Initialize IK chain info for all bones (will be populated for IK chain bones)
    const ikChainInfo: IKChainInfo[] = new Array(boneCount)
    for (let i = 0; i < boneCount; i++) {
      ikChainInfo[i] = {
        ikRotation: Quat.identity(),
        localRotation: Quat.identity(),
      }
    }

    // Build IK solvers from bone data
    const ikSolvers: IKSolver[] = []
    let solverIndex = 0

    for (let i = 0; i < boneCount; i++) {
      const bone = bones[i]
      if (bone.ikTargetIndex !== undefined && bone.ikLinks && bone.ikLinks.length > 0) {
        const solver: IKSolver = {
          index: solverIndex++,
          ikBoneIndex: i,
          targetBoneIndex: bone.ikTargetIndex,
          iterationCount: bone.ikIteration ?? 1,
          limitAngle: bone.ikLimitAngle ?? Math.PI,
          links: bone.ikLinks,
        }
        ikSolvers.push(solver)
      }
    }

    this.runtimeSkeleton.ikChainInfo = ikChainInfo
    this.runtimeSkeleton.ikSolvers = ikSolvers

    this.buildDeformOrder()
  }

  // Precompute the bone order that computeWorldMatrices iterates every frame: every
  // bone appears after its parent. This is the exact finishing order the previous
  // recursive computeWorld() produced (walk up the not-yet-emitted ancestor chain,
  // then emit it top-down; ties broken by ascending index) — so behavior is identical,
  // minus the per-frame closure + visited-array allocation and the recursion overhead.
  private buildDeformOrder(): void {
    const bones = this.skeleton.bones
    const n = bones.length
    const order = new Int32Array(n)
    const done = new Uint8Array(n)
    const stack: number[] = []
    let k = 0
    for (let i = 0; i < n; i++) {
      if (done[i]) continue
      stack.length = 0
      let cur = i
      // Collect the chain of not-yet-emitted ancestors up to the root (or a done one).
      while (cur >= 0 && cur < n && !done[cur]) {
        stack.push(cur)
        cur = bones[cur].parentIndex
      }
      // Emit from the topmost ancestor down so parents precede children.
      for (let s = stack.length - 1; s >= 0; s--) {
        const b = stack[s]
        done[b] = 1
        order[k++] = b
      }
    }
    this.deformOrder = order

    // Accumulate bind-pose world positions in the same parent-before-child order and
    // with the same add order as the old recursive computeBindPoseWorldPosition, so the
    // downstream arithmetic stays bit-identical.
    const bindWorld = new Float32Array(n * 3)
    for (let idx = 0; idx < n; idx++) {
      const i = order[idx]
      const bt = bones[i].bindTranslation
      const p = bones[i].parentIndex
      if (p >= 0 && p < n) {
        bindWorld[i * 3 + 0] = bindWorld[p * 3 + 0] + bt[0]
        bindWorld[i * 3 + 1] = bindWorld[p * 3 + 1] + bt[1]
        bindWorld[i * 3 + 2] = bindWorld[p * 3 + 2] + bt[2]
      } else {
        bindWorld[i * 3 + 0] = bt[0]
        bindWorld[i * 3 + 1] = bt[1]
        bindWorld[i * 3 + 2] = bt[2]
      }
    }
    this.bindWorldPos = bindWorld
  }

  private initializeTweenBuffers(): void {
    const boneCount = this.skeleton.bones.length
    const morphCount = this.morphing.morphs.length

    // Pre-allocate Quat and Vec3 arrays to avoid reallocation during tweens
    const rotStartQuat: Quat[] = new Array(boneCount)
    const rotTargetQuat: Quat[] = new Array(boneCount)
    const transStartVec: Vec3[] = new Array(boneCount)
    const transTargetVec: Vec3[] = new Array(boneCount)
    for (let i = 0; i < boneCount; i++) {
      rotStartQuat[i] = Quat.identity()
      rotTargetQuat[i] = Quat.identity()
      transStartVec[i] = Vec3.zeros()
      transTargetVec[i] = Vec3.zeros()
    }

    this.tweenState = {
      // Bone rotation tweens
      rotActive: new Uint8Array(boneCount),
      rotStartQuat,
      rotTargetQuat,
      rotStartTimeMs: new Float32Array(boneCount),
      rotDurationMs: new Float32Array(boneCount),

      // Bone translation tweens
      transActive: new Uint8Array(boneCount),
      transStartVec,
      transTargetVec,
      transStartTimeMs: new Float32Array(boneCount),
      transDurationMs: new Float32Array(boneCount),

      // Morph weight tweens
      morphActive: new Uint8Array(morphCount),
      morphStartWeight: new Float32Array(morphCount),
      morphTargetWeight: new Float32Array(morphCount),
      morphStartTimeMs: new Float32Array(morphCount),
      morphDurationMs: new Float32Array(morphCount),
    }
  }

  private initializeRuntimeMorph(): void {
    const morphCount = this.morphing.morphs.length
    this.runtimeMorph = {
      nameIndex: this.morphing.morphs.reduce((acc, morph, index) => {
        acc[morph.name] = index
        return acc
      }, {} as Record<string, number>),
      weights: new Float32Array(morphCount),
    }
    const boneCount = this.skeleton.bones.length
    this.boneMorphPlan = []
    for (let i = 0; i < morphCount; i++) {
      const morph = this.morphing.morphs[i]
      if (morph.type !== 2 || !morph.boneOffsets) continue
      for (const off of morph.boneOffsets) {
        if (off.boneIndex < 0 || off.boneIndex >= boneCount) continue
        this.boneMorphPlan.push({
          morphIndex: i,
          boneIndex: off.boneIndex,
          translation: new Vec3(off.translation[0], off.translation[1], off.translation[2]),
          rotation: new Quat(off.rotation[0], off.rotation[1], off.rotation[2], off.rotation[3]),
        })
      }
    }
    const touched = new Set(this.boneMorphPlan.map((e) => e.boneIndex))
    this.boneMorphBones = [...touched]
    this.boneMorphRestoreR = this.boneMorphBones.map(() => Quat.identity())
    this.boneMorphRestoreT = this.boneMorphBones.map(() => Vec3.zeros())
    this.boneMorphApplied = false
  }

  /**
   * Bone morphs (type 2) compose over whatever the pose sources produced, the
   * same way boneRotationOffsets do — they are an offset on the animated local
   * transform, not a replacement for it. Re-applied every frame because each
   * pose source rewrites the locals it touches.
   *
   * Stages are the reason this exists: a door or a lift is rigged as a bone
   * morph and there is no VMD anywhere that drives it.
   */
  /**
   * Take the last frame's bone-morph offsets back off, before any pose source
   * runs.
   *
   * This cannot live at the top of applyBoneMorphs, which is where it used to
   * be. A pose source writes only the bones its clip keys, and it writes them
   * from the clip — so restoring afterwards overwrote the freshly animated pose
   * with a snapshot taken a frame earlier, and the re-snapshot immediately
   * below then saved that same stale value again. The result was not a drift or
   * a lag: every bone any bone morph touched stayed pinned to the first frame
   * the model was posed at, permanently, and at ANY weight — the restore ran
   * over the touched set before the weight test, so a morph sitting at 0 froze
   * its bones just as hard as one at 1.
   *
   * It reads as "the arms don't animate" because arms are what these morphs
   * touch: character models routinely ship T-Pose / A-Pose / ShouderBlend /
   * ElbowBlend adjusters on 腕 and ひじ, and nothing else in the rig is a
   * comparably popular bone-morph target.
   */
  private undoBoneMorphs(): void {
    if (!this.boneMorphApplied) return
    for (let i = 0; i < this.boneMorphBones.length; i++) {
      const b = this.boneMorphBones[i]
      this.runtimeSkeleton.localRotations[b].set(this.boneMorphRestoreR[i])
      this.runtimeSkeleton.localTranslations[b].set(this.boneMorphRestoreT[i])
    }
    this.boneMorphApplied = false
  }

  private applyBoneMorphs(): void {
    if (this.boneMorphPlan.length === 0) return

    // Snapshot the pose as the sources left it, so undoBoneMorphs can hand these
    // bones back unmorphed at the top of the next frame. The two halves are a
    // pair: without the undo a stage — which has no clip to rewrite its bones —
    // would compound the same offset every frame.
    for (let i = 0; i < this.boneMorphBones.length; i++) {
      const b = this.boneMorphBones[i]
      this.boneMorphRestoreR[i].set(this.runtimeSkeleton.localRotations[b])
      this.boneMorphRestoreT[i].set(this.runtimeSkeleton.localTranslations[b])
    }
    this.boneMorphApplied = true

    const weights = this.getEffectiveMorphWeights()
    for (const entry of this.boneMorphPlan) {
      const w = weights[entry.morphIndex]
      if (w < 0.0001) continue
      const t = this.runtimeSkeleton.localTranslations[entry.boneIndex]
      t.x += entry.translation.x * w
      t.y += entry.translation.y * w
      t.z += entry.translation.z * w
      // Scale the rotation by weight the way MMD does — slerp out of identity,
      // then compose. Multiplying components would not stay a unit quaternion.
      const r = this.runtimeSkeleton.localRotations[entry.boneIndex]
      Quat.slerpInto(_boneMorphIdentity, entry.rotation, w, _boneMorphQ)
      Quat.multiplyInto(r, _boneMorphQ, r)
    }
  }

  /** True (once) when effective morph weights changed — the engine re-derives
   *  material-morph uniforms from it. Separate from the GPU vertex path's flag
   *  so both can consume the same change. */
  consumeAuxMorphDirty(): boolean {
    const d = this.auxMorphDirty
    this.auxMorphDirty = false
    return d
  }

  /**
   * Morph indices this model can actually act on, so a UI never offers a control
   * that moves nothing.
   *
   * Driven directly: vertex (1), bone (2), material (8). Excluded: UV (3–7),
   * which are parsed and kept but not yet applied; flip (9) and impulse (10),
   * which are PMX 2.1 and would need the rigidbody solver.
   *
   * A group morph (0) is only as alive as what it points at — one referencing
   * nothing but UV morphs is just as dead as the UV morphs themselves, so it is
   * resolved rather than assumed.
   */
  getSupportedMorphIndices(): number[] {
    const morphs = this.morphing.morphs
    const drivable = (t: number) => t === 1 || t === 2 || t === 8
    // Groups can reference groups, so walk with a seen-set rather than recursing.
    const resolves = (start: number): boolean => {
      const seen = new Set<number>()
      const stack = [start]
      while (stack.length > 0) {
        const i = stack.pop()!
        if (seen.has(i) || i < 0 || i >= morphs.length) continue
        seen.add(i)
        const m = morphs[i]
        if (drivable(m.type)) return true
        if (m.type === 0 && m.groupReferences) {
          for (const ref of m.groupReferences) stack.push(ref.morphIndex)
        }
      }
      return false
    }
    const out: number[] = []
    for (let i = 0; i < morphs.length; i++) {
      const t = morphs[i].type
      if (drivable(t) || (t === 0 && resolves(i))) out.push(i)
    }
    return out
  }

  // Tween update - processes all tweens together with a single time reference
  // This avoids conflicts and ensures consistent timing across all tween types
  // Returns true if morph weights changed (needed for vertex buffer updates)
  private updateTweens(): boolean {
    const state = this.tweenState
    const now = this.tweenTimeMs // Single time reference for all tweens
    let morphChanged = false

    // Update bone rotation tweens
    const rotations = this.runtimeSkeleton.localRotations
    const boneCount = this.skeleton.bones.length
    for (let i = 0; i < boneCount; i++) {
      if (state.rotActive[i] !== 1) continue

      const startMs = state.rotStartTimeMs[i]
      const durMs = Math.max(1, state.rotDurationMs[i])
      const t = Math.max(0, Math.min(1, (now - startMs) / durMs))
      const e = t // Linear interpolation

      const result = Quat.slerp(state.rotStartQuat[i], state.rotTargetQuat[i], e)
      rotations[i].set(result)

      if (t >= 1) {
        state.rotActive[i] = 0
      }
    }

    // Update bone translation tweens
    const translations = this.runtimeSkeleton.localTranslations
    for (let i = 0; i < boneCount; i++) {
      if (state.transActive[i] !== 1) continue

      const startMs = state.transStartTimeMs[i]
      const durMs = Math.max(1, state.transDurationMs[i])
      const t = Math.max(0, Math.min(1, (now - startMs) / durMs))
      const e = t // Linear interpolation

      const startVec = state.transStartVec[i]
      const targetVec = state.transTargetVec[i]
      translations[i].x = startVec.x + (targetVec.x - startVec.x) * e
      translations[i].y = startVec.y + (targetVec.y - startVec.y) * e
      translations[i].z = startVec.z + (targetVec.z - startVec.z) * e

      if (t >= 1) {
        state.transActive[i] = 0
      }
    }

    // Update morph weight tweens
    const weights = this.runtimeMorph.weights
    const morphCount = this.morphing.morphs.length
    for (let i = 0; i < morphCount; i++) {
      if (state.morphActive[i] !== 1) continue

      const startMs = state.morphStartTimeMs[i]
      const durMs = Math.max(1, state.morphDurationMs[i])
      const t = Math.max(0, Math.min(1, (now - startMs) / durMs))
      const e = t // Linear interpolation

      const oldWeight = weights[i]
      weights[i] = state.morphStartWeight[i] + (state.morphTargetWeight[i] - state.morphStartWeight[i]) * e

      // Check if weight actually changed (accounting for floating point precision)
      if (Math.abs(weights[i] - oldWeight) > 1e-6) {
        morphChanged = true
      }

      if (t >= 1) {
        weights[i] = state.morphTargetWeight[i]
        state.morphActive[i] = 0
        // Check if final weight is different from old weight
        if (Math.abs(weights[i] - oldWeight) > 1e-6) {
          morphChanged = true
        }
      }
    }

    return morphChanged
  }

  getVertices(): Float32Array<ArrayBuffer> {
    return this.vertexData
  }

  getTextures(): Texture[] {
    return this.textures
  }

  getMaterials(): Material[] {
    return this.materials
  }

  getIndices(): Uint32Array<ArrayBuffer> {
    return this.indexData
  }

  /**
   * Bind-pose geometry for design-layer consumers — the cast swatch's
   * area-weighted colour extraction is the first. Positions and UVs are fresh
   * copies deinterleaved from the RETAINED vertex data (base, pre-morph: the
   * caller wants the model's shape, not whatever morph is live this frame), so
   * callers may mutate them freely. Indices are the live buffer, same contract
   * as getIndices — read-only. Per-material triangle ranges come from
   * getMaterials(): each material's vertexCount is its consecutive index run.
   */
  getGeometry(): { positions: Float32Array; uvs: Float32Array; indices: Uint32Array } {
    const count = this.vertexCount
    const positions = new Float32Array(count * 3)
    const uvs = new Float32Array(count * 2)
    const src = this.baseVertexData
    for (let i = 0; i < count; i++) {
      const vi = i * 8
      positions[i * 3] = src[vi]
      positions[i * 3 + 1] = src[vi + 1]
      positions[i * 3 + 2] = src[vi + 2]
      uvs[i * 2] = src[vi + 6]
      uvs[i * 2 + 1] = src[vi + 7]
    }
    return { positions, uvs, indices: this.indexData }
  }

  getSkeleton(): Skeleton {
    return this.skeleton
  }

  // Direct bone local-transform accessors (used by interactive gizmo drag).
  // Readers return the live runtime state — callers that want a snapshot for
  // later comparison should `.clone()` the returned Quat / copy the Vec3.
  getBoneLocalRotation(boneIndex: number): Quat {
    return this.runtimeSkeleton.localRotations[boneIndex]
  }

  getBoneLocalTranslation(boneIndex: number): Vec3 {
    return this.runtimeSkeleton.localTranslations[boneIndex]
  }

  // Raw absolute-local translation write. NOT equivalent to
  // `moveBones({ name: v }, 0)` — moveBones treats the input as VMD-relative
  // (offset from bind pose) and runs convertVMDTranslationToLocal() over it.
  // Use this when you already have the final local translation (e.g. the
  // gizmo's computed target). For rotation, just use rotateBones(..., 0).
  setBoneLocalTranslation(boneIndex: number, v: Vec3): void {
    const t = this.runtimeSkeleton.localTranslations[boneIndex]
    t.x = v.x; t.y = v.y; t.z = v.z
    this.tweenState.transActive[boneIndex] = 0
  }

  // When true, update() skips applyPoseFromClip, so whatever was last written to
  // localRotations / localTranslations persists across frames. Used by gizmo drag
  // and other direct-manipulation flows to prevent the currently-shown clip from
  // overwriting manual edits each frame. Auto-cleared on play()/seek() so the user
  // gets back to normal playback without having to manage this flag explicitly.
  private clipApplySuspended = false
  setClipApplySuspended(suspended: boolean): void {
    this.clipApplySuspended = suspended
  }
  isClipApplySuspended(): boolean {
    return this.clipApplySuspended
  }

  // World bone origin (world matrix col3); unknown name → null
  getBoneWorldPosition(boneName: string): Vec3 | null {
    const idx = this.runtimeSkeleton.nameIndex[boneName]
    if (idx === undefined || idx < 0) return null
    return this.runtimeSkeleton.worldMatrices[idx].getPosition()
  }

  /**
   * A bone's forward axis, normalised — which way a foot points, where a head
   * looks. Model space, like getBoneWorldPosition: the caller composes the model
   * transform on top if it wants world space.
   *
   * Column 2 of the world matrix. Null for a name this rig does not have, which
   * is the ordinary case across rigs that spell bones differently.
   */
  getBoneWorldForward(boneName: string): Vec3 | null {
    const idx = this.runtimeSkeleton.nameIndex[boneName]
    if (idx === undefined || idx < 0) return null
    const m = this.runtimeSkeleton.worldMatrices[idx].values
    const len = Math.hypot(m[8], m[9], m[10])
    if (len < 1e-8) return null
    return new Vec3(m[8] / len, m[9] / len, m[10] / len)
  }

  getSkinning(): Skinning {
    return this.skinning
  }

  // True when the PMX carried a usable rigidbody section. False means the
  // model renders but has no physics — surface this in the UI instead of
  // letting it read as "physics silently broken".
  hasPhysicsData(): boolean {
    return this.rigidbodies.length > 0
  }

  // Non-fatal PMX parse problems (truncated sections, suspicious counts…).
  getLoadWarnings(): readonly string[] {
    return this.loadWarnings
  }

  /** Post-pose local rotation offsets by bone index (see setBoneRotationOffset). */
  private readonly boneRotationOffsets = new Map<number, Quat>()

  /** Compose a constant local rotation onto a bone AFTER every pose source (clip,
   *  blend, tween) each frame — the classic MMD "heel correction": pitch the 足首
   *  bones so a motion authored for flat shoes grounds a heeled model. Persists
   *  across play()/show(); pass null to clear. */
  setBoneRotationOffset(boneName: string, rotation: Quat | null): boolean {
    const idx = this.runtimeSkeleton.nameIndex[boneName]
    if (idx === undefined || idx < 0) return false
    if (rotation === null) this.boneRotationOffsets.delete(idx)
    else this.boneRotationOffsets.set(idx, rotation.clone())
    return true
  }

  getRigidbodies(): Rigidbody[] {
    return this.rigidbodies
  }

  getJoints(): Joint[] {
    return this.joints
  }

  getMorphing(): Morphing {
    return this.morphing
  }

  getMorphWeights(): Float32Array {
    return this.runtimeMorph.weights
  }

  // ------- Bone helpers (API) -------

  rotateBones(boneRotations: Record<string, Quat>, durationMs?: number): void {
    const state = this.tweenState
    // Clone and normalize to avoid mutating input
    Object.values(boneRotations).forEach((q) => q.normalize())
    const now = this.tweenTimeMs
    const dur = durationMs && durationMs > 0 ? durationMs : 0

    for (const [name, targetQuat] of Object.entries(boneRotations)) {
      const idx = this.runtimeSkeleton.nameIndex[name] ?? -1
      if (idx < 0 || idx >= this.skeleton.bones.length) continue

      const rotations = this.runtimeSkeleton.localRotations
      const targetNorm = targetQuat

      if (dur === 0) {
        rotations[idx].set(targetNorm)
        state.rotActive[idx] = 0
        continue
      }

      const currentRot = rotations[idx]
      let sx = currentRot.x
      let sy = currentRot.y
      let sz = currentRot.z
      let sw = currentRot.w

      if (state.rotActive[idx] === 1) {
        const startMs = state.rotStartTimeMs[idx]
        const prevDur = Math.max(1, state.rotDurationMs[idx])
        const t = Math.max(0, Math.min(1, (now - startMs) / prevDur))
        const e = t // Linear interpolation
        const result = Quat.slerp(state.rotStartQuat[idx], state.rotTargetQuat[idx], e)
        sx = result.x
        sy = result.y
        sz = result.z
        sw = result.w
      }

      state.rotStartQuat[idx].x = sx
      state.rotStartQuat[idx].y = sy
      state.rotStartQuat[idx].z = sz
      state.rotStartQuat[idx].w = sw
      state.rotTargetQuat[idx].set(targetNorm)
      state.rotStartTimeMs[idx] = now
      state.rotDurationMs[idx] = dur
      state.rotActive[idx] = 1
    }
  }

  // Move bones using VMD-style relative translations (relative to bind pose world position)
  // This is the default behavior for VMD animations
  moveBones(boneTranslations: Record<string, Vec3>, durationMs?: number): void {
    const state = this.tweenState
    const now = this.tweenTimeMs
    const dur = durationMs && durationMs > 0 ? durationMs : 0

    for (const [name, vmdRelativeTranslation] of Object.entries(boneTranslations)) {
      const idx = this.runtimeSkeleton.nameIndex[name] ?? -1
      if (idx < 0 || idx >= this.skeleton.bones.length) continue

      const translations = this.runtimeSkeleton.localTranslations

      // Convert VMD relative translation to local translation
      const localTranslation = this.convertVMDTranslationToLocal(idx, vmdRelativeTranslation)
      const [tx, ty, tz] = [localTranslation.x, localTranslation.y, localTranslation.z]

      if (dur === 0) {
        translations[idx].x = tx
        translations[idx].y = ty
        translations[idx].z = tz
        state.transActive[idx] = 0
        continue
      }

      const currentTrans = translations[idx]
      let sx = currentTrans.x
      let sy = currentTrans.y
      let sz = currentTrans.z

      if (state.transActive[idx] === 1) {
        const startMs = state.transStartTimeMs[idx]
        const prevDur = Math.max(1, state.transDurationMs[idx])
        const t = Math.max(0, Math.min(1, (now - startMs) / prevDur))
        const e = t // Linear interpolation
        const startVec = state.transStartVec[idx]
        const targetVec = state.transTargetVec[idx]
        sx = startVec.x + (targetVec.x - startVec.x) * e
        sy = startVec.y + (targetVec.y - startVec.y) * e
        sz = startVec.z + (targetVec.z - startVec.z) * e
      }

      state.transStartVec[idx].x = sx
      state.transStartVec[idx].y = sy
      state.transStartVec[idx].z = sz
      state.transTargetVec[idx].x = tx
      state.transTargetVec[idx].y = ty
      state.transTargetVec[idx].z = tz
      state.transStartTimeMs[idx] = now
      state.transDurationMs[idx] = dur
      state.transActive[idx] = 1
    }
  }

  // VMD translation (world delta from bind pose) → bone local space; optional rotation for animation vs IK
  // Returns a REUSED scratch Vec3 (_convOut) — callers must copy immediately (they do:
  // .set() / destructure). Zero allocation; result is bit-identical to the previous
  // recursive implementation (verified numerically).
  private convertVMDTranslationToLocal(boneIdx: number, vmdRelativeTranslation: Vec3, rotation?: Quat): Vec3 {
    const bone = this.skeleton.bones[boneIdx]
    const bindWorld = this.bindWorldPos
    const bt = bone.bindTranslation
    const p = bone.parentIndex

    // afterBindTranslation = (bindWorld[bone] + vmd − bindWorld[parent]) − bindTranslation.
    // (Algebraically this reduces to vmd, since bindWorld[bone] = bindWorld[parent] +
    // bindTranslation, but the explicit form keeps the result bit-identical to before.)
    const bi3 = boneIdx * 3
    const targetX = bindWorld[bi3 + 0] + vmdRelativeTranslation.x
    const targetY = bindWorld[bi3 + 1] + vmdRelativeTranslation.y
    const targetZ = bindWorld[bi3 + 2] + vmdRelativeTranslation.z
    const px = p >= 0 ? bindWorld[p * 3 + 0] : 0
    const py = p >= 0 ? bindWorld[p * 3 + 1] : 0
    const pz = p >= 0 ? bindWorld[p * 3 + 2] : 0
    const abx = targetX - px - bt[0]
    const aby = targetY - py - bt[1]
    const abz = targetZ - pz - bt[2]

    // Inverse rotation = conjugate + normalize (matches localRotation.clone().conjugate()
    // .normalize()), applied via a rotation matrix (matches Mat4.fromQuat). Uses animation
    // rotation when provided so IK-modified localRot doesn't perturb the conversion.
    const q = rotation ?? this.runtimeSkeleton.localRotations[boneIdx]
    const qlen = Math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    let ix: number, iy: number, iz: number, iw: number
    if (qlen === 0) {
      ix = 0; iy = 0; iz = 0; iw = 1
    } else {
      const inv = 1 / qlen
      ix = -q.x * inv; iy = -q.y * inv; iz = -q.z * inv; iw = q.w * inv
    }
    Mat4.fromQuatInto(ix, iy, iz, iw, _convMat, 0)
    const rm = _convMat
    _convOut.setXYZ(
      rm[0] * abx + rm[4] * aby + rm[8] * abz,
      rm[1] * abx + rm[5] * aby + rm[9] * abz,
      rm[2] * abx + rm[6] * aby + rm[10] * abz
    )
    return _convOut
  }

  getWorldMatrices(): Mat4[] {
    return this.runtimeSkeleton.worldMatrices
  }

  getBoneWorldMatrices(): Float32Array {
    // Convert Mat4[] to Float32Array for WebGPU compatibility
    const boneCount = this.skeleton.bones.length
    const worldMats = this.runtimeSkeleton.worldMatrices
    const result = new Float32Array(boneCount * 16)
    for (let i = 0; i < boneCount; i++) {
      result.set(worldMats[i].values, i * 16)
    }
    return result
  }

  getBoneInverseBindMatrices(): Float32Array {
    return this.skeleton.inverseBindMatrices
  }

  getSkinMatrices(): Float32Array {
    const boneCount = this.skeleton.bones.length
    const worldMats = this.runtimeSkeleton.worldMatrices
    const invBindMats = this.skeleton.inverseBindMatrices

    // Initialize cached array if needed or if bone count changed
    if (!this.skinMatricesArray || this.skinMatricesArray.length !== boneCount * 16) {
      this.skinMatricesArray = new Float32Array(boneCount * 16)
    }

    const skinMatrices = this.skinMatricesArray

    // Rebuild root matrix + cache identity-shortcut flag only when pos/rot changed.
    if (this.rootMatrixDirty) {
      const p = this._position, r = this._rotation, s = this._scale
      Mat4.fromPositionRotationScaleInto(p.x, p.y, p.z, r.x, r.y, r.z, r.w, s, this.rootMatrixValues)
      this.rootIsIdentity =
        p.x === 0 && p.y === 0 && p.z === 0 &&
        r.x === 0 && r.y === 0 && r.z === 0 && r.w === 1 && s === 1
      this.rootMatrixDirty = false
    }

    if (this.rootIsIdentity) {
      // skinMatrix = worldMatrix × inverseBindMatrix
      for (let i = 0; i < boneCount; i++) {
        const off = i * 16
        Mat4.multiplyArrays(worldMats[i].values, 0, invBindMats, off, skinMatrices, off)
      }
    } else {
      // skinMatrix = rootMatrix × worldMatrix × inverseBindMatrix
      // Two-mul path. scratchMat4Values[1] — [0] is owned by computeWorldMatrices.
      const rootVals = this.rootMatrixValues
      const tmp = scratchMat4Values[1]
      for (let i = 0; i < boneCount; i++) {
        const off = i * 16
        Mat4.multiplyArrays(rootVals, 0, worldMats[i].values, 0, tmp, 0)
        Mat4.multiplyArrays(tmp, 0, invBindMats, off, skinMatrices, off)
      }
    }

    return skinMatrices
  }

  setMorphWeight(name: string, weight: number, durationMs?: number): void {
    const idx = this.runtimeMorph.nameIndex[name] ?? -1
    if (idx < 0 || idx >= this.runtimeMorph.weights.length) return

    const clampedWeight = Math.max(0, Math.min(1, weight))
    const dur = durationMs && durationMs > 0 ? durationMs : 0

    if (dur === 0) {
      // Instant change
      this.runtimeMorph.weights[idx] = clampedWeight
      this.tweenState.morphActive[idx] = 0
      this.applyMorphs()
      // Vertex and material morphs are done by applyMorphs alone, but a bone
      // morph lands in the pose pass — and a model with no clip (every stage)
      // reports idle, so without this the pass never runs and the switch does
      // nothing. Costs one redundant applyMorphs on the next frame.
      if (this.boneMorphPlan.length > 0) this.morphsDirty = true
      try {
        Engine.getInstance().markVertexBufferDirty(this)
      } catch {
        // not registered yet
      }
      return
    }

    // Animated change
    const state = this.tweenState
    const now = this.tweenTimeMs

    // If already tweening, start from current interpolated value
    let startWeight = this.runtimeMorph.weights[idx]
    if (state.morphActive[idx] === 1) {
      const startMs = state.morphStartTimeMs[idx]
      const prevDur = Math.max(1, state.morphDurationMs[idx])
      const t = Math.max(0, Math.min(1, (now - startMs) / prevDur))
      const e = t // Linear interpolation
      startWeight = state.morphStartWeight[idx] + (state.morphTargetWeight[idx] - state.morphStartWeight[idx]) * e
    }

    state.morphStartWeight[idx] = startWeight
    state.morphTargetWeight[idx] = clampedWeight
    state.morphStartTimeMs[idx] = now
    state.morphDurationMs[idx] = dur
    state.morphActive[idx] = 1

    // Immediately apply morphs with current weight
    this.runtimeMorph.weights[idx] = startWeight
    this.applyMorphs()
  }

  private applyMorphs(): void {
    const vertexCount = this.vertexCount
    const morphCount = this.morphing.morphs.length
    const weights = this.runtimeMorph.weights

    // Effective weights (group-morph resolution + clamp). Both paths need these: the GPU
    // path uploads them for the compute pass; the CPU path applies them below. Reused buffer.
    if (!this.morphEffectiveWeights || this.morphEffectiveWeights.length !== morphCount) {
      this.morphEffectiveWeights = new Float32Array(morphCount)
    }
    const effectiveWeights = this.morphEffectiveWeights
    effectiveWeights.set(weights) // Start with direct weights

    // Apply group morphs: group morph weight * ratio affects referenced morphs
    for (let morphIdx = 0; morphIdx < morphCount; morphIdx++) {
      const morph = this.morphing.morphs[morphIdx]
      if (morph.type === 0 && morph.groupReferences) {
        const groupWeight = weights[morphIdx]
        if (groupWeight > 0.0001) {
          for (const ref of morph.groupReferences) {
            if (ref.morphIndex >= 0 && ref.morphIndex < morphCount) {
              effectiveWeights[ref.morphIndex] += groupWeight * ref.ratio
            }
          }
        }
      }
    }
    for (let i = 0; i < morphCount; i++) {
      effectiveWeights[i] = Math.max(0, Math.min(1, effectiveWeights[i]))
    }

    // Bone morphs read these every frame, but material/UV consumers live outside
    // this class and need telling. Set on BOTH paths — a model whose only morphs
    // are material morphs never enables the GPU vertex path at all.
    this.auxMorphDirty = true

    // GPU path: the compute pass applies the vertex offsets from these weights.
    if (this.gpuMorphEnabled) {
      this.morphWeightsDirty = true
      return
    }

    // ── CPU path ── Reset only the vertices morphed by the previous pass back to base
    // (targeted reset; vertexData never diverges from base outside that range).
    if (this.morphPrevMaxVert >= 0) {
      const s = this.morphPrevMinVert * VERTEX_STRIDE
      const e = (this.morphPrevMaxVert + 1) * VERTEX_STRIDE
      this.vertexData.set(this.baseVertexData.subarray(s, e), s)
    }

    // Apply vertex morphs, tracking the touched vertex-index range for partial upload.
    let curMinVert = -1
    let curMaxVert = -1
    for (let morphIdx = 0; morphIdx < morphCount; morphIdx++) {
      const effectiveWeight = effectiveWeights[morphIdx]
      if (effectiveWeight === 0 || effectiveWeight < 0.0001) continue

      const morph = this.morphing.morphs[morphIdx]
      if (morph.type !== 1) continue // Only process vertex morphs

      for (const vertexOffset of morph.vertexOffsets) {
        const vIdx = vertexOffset.vertexIndex
        if (vIdx < 0 || vIdx >= vertexCount) continue

        const offsetX = vertexOffset.positionOffset[0]
        const offsetY = vertexOffset.positionOffset[1]
        const offsetZ = vertexOffset.positionOffset[2]
        if (Math.abs(offsetX) < 0.0001 && Math.abs(offsetY) < 0.0001 && Math.abs(offsetZ) < 0.0001) {
          continue
        }

        const vertexIdx = vIdx * VERTEX_STRIDE
        this.vertexData[vertexIdx] += offsetX * effectiveWeight
        this.vertexData[vertexIdx + 1] += offsetY * effectiveWeight
        this.vertexData[vertexIdx + 2] += offsetZ * effectiveWeight

        if (curMinVert < 0 || vIdx < curMinVert) curMinVert = vIdx
        if (vIdx > curMaxVert) curMaxVert = vIdx
      }
    }

    let dirtyMin = curMinVert
    let dirtyMax = curMaxVert
    if (this.morphPrevMaxVert >= 0) {
      if (dirtyMin < 0 || this.morphPrevMinVert < dirtyMin) dirtyMin = this.morphPrevMinVert
      if (this.morphPrevMaxVert > dirtyMax) dirtyMax = this.morphPrevMaxVert
    }
    if (dirtyMin >= 0 && dirtyMax >= 0) {
      this.morphPendingMinVert =
        this.morphPendingMinVert < 0 ? dirtyMin : Math.min(this.morphPendingMinVert, dirtyMin)
      this.morphPendingMaxVert =
        this.morphPendingMaxVert < 0 ? dirtyMax : Math.max(this.morphPendingMaxVert, dirtyMax)
    }
    this.morphPrevMinVert = curMinVert
    this.morphPrevMaxVert = curMaxVert
  }

  // ── GPU morph path support ──
  // Called by the engine once it has created the compute buffers for this model; switches
  // applyMorphs to the weights-only branch.
  enableGpuMorphs(): void {
    this.gpuMorphEnabled = true
  }

  // True (once) when morph weights changed since the last check — the engine then uploads
  // getEffectiveMorphWeights() and dispatches the compute pass.
  consumeMorphWeightsDirty(): boolean {
    const d = this.morphWeightsDirty
    this.morphWeightsDirty = false
    return d
  }

  // Effective (group-resolved, clamped) morph weights for GPU upload. Ensures they're
  // computed at least once even before the first weight change.
  getEffectiveMorphWeights(): Float32Array {
    if (!this.morphEffectiveWeights) this.applyMorphs()
    return this.morphEffectiveWeights ?? new Float32Array(0)
  }

  // Build the CSR inversion of vertex-morph offsets for the GPU compute pass. Returns null
  // when the model has no vertex-morph offsets (no GPU path needed). Entries for each vertex
  // are emitted in ascending morph-index order, matching the CPU accumulation order.
  buildMorphComputeData(): MorphComputeData | null {
    const V = this.vertexCount
    const M = this.morphing.morphs.length
    const morphs = this.morphing.morphs
    const EPS = 0.0001

    const isLive = (o: VertexMorphOffset): boolean =>
      o.vertexIndex >= 0 &&
      o.vertexIndex < V &&
      (Math.abs(o.positionOffset[0]) >= EPS ||
        Math.abs(o.positionOffset[1]) >= EPS ||
        Math.abs(o.positionOffset[2]) >= EPS)

    const counts = new Uint32Array(V)
    for (let m = 0; m < M; m++) {
      const morph = morphs[m]
      if (morph.type !== 1) continue
      for (const o of morph.vertexOffsets) if (isLive(o)) counts[o.vertexIndex]++
    }

    const rowStart = new Uint32Array(V + 1)
    let acc = 0
    for (let v = 0; v < V; v++) {
      rowStart[v] = acc
      acc += counts[v]
    }
    rowStart[V] = acc
    const E = acc
    if (E === 0) return null

    const colMorph = new Uint32Array(E)
    const colOffset = new Float32Array(E * 3)
    const fill = new Uint32Array(V)
    for (let m = 0; m < M; m++) {
      const morph = morphs[m]
      if (morph.type !== 1) continue
      for (const o of morph.vertexOffsets) {
        if (!isLive(o)) continue
        const v = o.vertexIndex
        const p = rowStart[v] + fill[v]++
        colMorph[p] = m
        colOffset[p * 3] = o.positionOffset[0]
        colOffset[p * 3 + 1] = o.positionOffset[1]
        colOffset[p * 3 + 2] = o.positionOffset[2]
      }
    }

    const basePositions = new Float32Array(V * 3)
    for (let v = 0; v < V; v++) {
      const vi = v * VERTEX_STRIDE
      basePositions[v * 3] = this.baseVertexData[vi]
      basePositions[v * 3 + 1] = this.baseVertexData[vi + 1]
      basePositions[v * 3 + 2] = this.baseVertexData[vi + 2]
    }

    return { basePositions, rowStart, colMorph, colOffset, morphCount: M, vertexCount: V, entryCount: E }
  }

  // Consume the pending morph vertex-upload range for the engine. Returns null when a
  // full upload is needed (first time after load, or nothing tracked), else the inclusive
  // [minVert, maxVert] slice that changed. Resets pending state.
  consumeVertexUploadRange(): { minVert: number; maxVert: number } | null {
    if (this.morphUploadFull || this.morphPendingMaxVert < 0) {
      this.morphUploadFull = false
      this.morphPendingMinVert = -1
      this.morphPendingMaxVert = -1
      return null
    }
    const range = { minVert: this.morphPendingMinVert, maxVert: this.morphPendingMaxVert }
    this.morphPendingMinVert = -1
    this.morphPendingMaxVert = -1
    return range
  }

  private buildClipFromVmdKeyFrames(vmdKeyFrames: VMDKeyFrame[]): AnimationClip {
    const boneTracksByBone: Record<string, Array<{ frame: number; rotation: Quat; translation: Vec3; interpolation: BoneInterpolation }>> = {}
    for (const keyFrame of vmdKeyFrames) {
      for (const bf of keyFrame.boneFrames) {
        if (!boneTracksByBone[bf.boneName]) boneTracksByBone[bf.boneName] = []
        boneTracksByBone[bf.boneName].push({
          frame: bf.frame,
          rotation: bf.rotation,
          translation: bf.translation,
          interpolation: rawInterpolationToBoneInterpolation(bf.interpolation),
        })
      }
    }
    const boneTracks = new Map<string, BoneKeyframe[]>()
    for (const name in boneTracksByBone) {
      const keyframes = boneTracksByBone[name]
      const sorted = [...keyframes].sort((a, b) => a.frame - b.frame)
      boneTracks.set(
        name,
        sorted.map((kf) => ({
          boneName: name,
          frame: kf.frame,
          rotation: kf.rotation,
          translation: kf.translation,
          interpolation: kf.interpolation,
        }))
      )
    }
    const morphTracksByMorph: Record<string, Array<{ frame: number; weight: number }>> = {}
    for (const keyFrame of vmdKeyFrames) {
      for (const mf of keyFrame.morphFrames) {
        if (!morphTracksByMorph[mf.morphName]) morphTracksByMorph[mf.morphName] = []
        morphTracksByMorph[mf.morphName].push({ frame: mf.frame, weight: mf.weight })
      }
    }
    const morphTracks = new Map<string, MorphKeyframe[]>()
    for (const name in morphTracksByMorph) {
      const keyframes = morphTracksByMorph[name]
      const sorted = [...keyframes].sort((a, b) => a.frame - b.frame)
      morphTracks.set(
        name,
        sorted.map((kf) => ({
          morphName: name,
          frame: kf.frame,
          weight: kf.weight,
        }))
      )
    }
    let maxFrame = 0
    for (const frames of boneTracks.values()) {
      if (frames.length > 0) maxFrame = Math.max(maxFrame, frames[frames.length - 1].frame)
    }
    for (const frames of morphTracks.values()) {
      if (frames.length > 0) maxFrame = Math.max(maxFrame, frames[frames.length - 1].frame)
    }
    return { boneTracks, morphTracks, frameCount: maxFrame }
  }

  loadVmd(name: string, urlOrRelative: string): Promise<void> {
    const loadBuffer = (): Promise<ArrayBuffer> => {
      const u = urlOrRelative.trim()
      const useSiteFetch =
        u.startsWith("http://") ||
        u.startsWith("https://") ||
        u.startsWith("/") ||
        u.startsWith("blob:") ||
        u.startsWith("data:")
      if (useSiteFetch) {
        return fetch(u).then((r) => {
          if (!r.ok) throw new Error(`Failed to fetch VMD ${u}: ${r.status}`)
          return r.arrayBuffer()
        })
      }
      if (this.assetReader) {
        return this.assetReader.readBinary(joinAssetPath(this.assetBasePath, u))
      }
      return fetch(u).then((r) => {
        if (!r.ok) throw new Error(`Failed to fetch VMD ${u}: ${r.status}`)
        return r.arrayBuffer()
      })
    }
    return loadBuffer().then((buf) => {
      const vmdKeyFrames = VMDLoader.loadFromBuffer(buf)
      const clip = this.buildClipFromVmdKeyFrames(vmdKeyFrames)
      // The IK block lives past every other section, so it is read separately
      // rather than threaded through the keyframe grouping.
      const ikFrames = VMDLoader.loadIkFromBuffer(buf)
      if (ikFrames.length > 0) {
        const ikTracks = new Map<string, IkKeyframe[]>()
        for (const record of ikFrames) {
          for (const state of record.states) {
            const track = ikTracks.get(state.boneName)
            if (track) track.push({ frame: record.frame, enabled: state.enabled })
            else ikTracks.set(state.boneName, [{ frame: record.frame, enabled: state.enabled }])
          }
        }
        for (const track of ikTracks.values()) track.sort((a, b) => a.frame - b.frame)
        clip.ikTracks = ikTracks
      }
      this.animationState.loadAnimation(name, clip)
    })
  }

  loadClip(name: string, clip: AnimationClip): void {
    this.animationState.loadAnimation(name, clip)
  }

  resetAllBones(): void {
    for (let boneIdx = 0; boneIdx < this.skeleton.bones.length; boneIdx++) {
      const localRot = this.runtimeSkeleton.localRotations[boneIdx]
      const localTrans = this.runtimeSkeleton.localTranslations[boneIdx]

      localRot.set(Quat.identity())
      localTrans.set(Vec3.zeros())
    }
    this.computeWorldMatrices()
  }

  resetAllMorphs(): void {
    for (let morphIdx = 0; morphIdx < this.morphing.morphs.length; morphIdx++) {
      const morphName = this.morphing.morphs[morphIdx].name
      this.setMorphWeight(morphName, 0)
    }
    this.morphsDirty = true
    this.applyMorphs()
  }

  getClip(name: string): AnimationClip | null {
    return this.animationState.getAnimationClip(name)
  }

  /** Lift a clip's authored horizontal root path off センター so the host can
   *  drive the MODEL ROOT along it — game-style root motion. The clip keeps
   *  its vertical bob but its horizontal センター flattens to `rest` (default
   *  0,0): a displaced first frame — common in per-pose game exports — would
   *  otherwise ride the whole clip as a constant and visibly slide off during
   *  the exit crossfade. Pass the pack's NEUTRAL standing offset (the idle
   *  clip's first センター key) as `rest` so the flattened pose blends into
   *  the surrounding states without even a micro-slide. The removed path
   *  returns RELATIVE TO THE FIRST KEY, raw clip units, clip space (rest
   *  facing -Z). Leg-IK target position tracks lose the same path so feet
   *  keep oscillating around the body. Call once per clip, after
   *  loadVmd/loadClip; null if the clip or its センター track doesn't exist. */
  extractRootMotion(name: string, rest?: { x?: number; z?: number }): RootMotionProfile | null {
    const clip = this.animationState.getAnimationClip(name)
    const track = clip?.boneTracks.get("センター")
    if (!clip || !track || track.length === 0) return null
    const restX = rest?.x ?? 0
    const restZ = rest?.z ?? 0
    const frames: number[] = []
    const x: number[] = []
    const z: number[] = []
    const pathX: number[] = [] // horizontal removed (rel. rest), for the IK tracks
    const pathZ: number[] = []
    const x0 = track[0].translation.x
    const z0 = track[0].translation.z
    for (const kf of track) {
      frames.push(kf.frame)
      x.push(kf.translation.x - x0)
      z.push(kf.translation.z - z0)
      pathX.push(kf.translation.x - restX)
      pathZ.push(kf.translation.z - restZ)
      kf.translation.x = restX
      kf.translation.z = restZ
    }
    for (const [bone, kfs] of clip.boneTracks) {
      if (!bone.includes("ＩＫ")) continue
      for (const kf of kfs) {
        kf.translation.x -= Model.sampleTrack(frames, pathX, kf.frame)
        kf.translation.z -= Model.sampleTrack(frames, pathZ, kf.frame)
      }
    }
    return { frames, x, z }
  }

  /** Linear sample of a keyed scalar track at fractional frame f. */
  private static sampleTrack(frames: number[], values: number[], f: number): number {
    const n = frames.length
    if (f <= frames[0]) return values[0]
    if (f >= frames[n - 1]) return values[n - 1]
    let i = 1
    while (frames[i] < f) i++
    const t = (f - frames[i - 1]) / (frames[i] - frames[i - 1])
    return values[i - 1] + (values[i] - values[i - 1]) * t
  }

  exportVmd(name: string): ArrayBuffer {
    const clip = this.animationState.getAnimationClip(name)
    if (!clip) throw new Error(`Animation clip "${name}" not found`)
    return new VMDWriter().write(clip)
  }

  play(): void
  play(name: string): boolean
  play(name: string, options?: AnimationPlayOptions): boolean
  play(name?: string, options?: AnimationPlayOptions): void | boolean {
    this.clipApplySuspended = false
    if (name === undefined) {
      // Resume: a paused crossfade continues from where it held.
      this.animationState.play()
      return
    }
    this.blendEntries = null
    this.crossfade = null
    this.oneShot = null
    this.resetAllBones()
    this.resetAllMorphs()
    return this.animationState.play(name, options)
  }

  show(name: string): void {
    this.blendEntries = null
    this.crossfade = null
    this.oneShot = null
    this.resetAllBones()
    this.resetAllMorphs()
    this.animationState.show(name)
  }

  /** Drive the pose from N weighted clips; the caller owns every clock (see BlendEntry).
   *  The entries array is held by reference and read each update, so a per-frame driver
   *  can mutate it in place without re-calling. Cleared by play(name)/show()/stop()/
   *  clearAnimation()/crossfadeTo(). The single-clip player keeps ticking underneath
   *  but stops writing the pose while a blend is set. */
  setBlendPose(entries: BlendEntry[]): void {
    this.blendEntries = entries
    this.crossfade = null
    this.clipApplySuspended = false
  }

  clearBlendPose(): void {
    this.blendEntries = null
  }

  /** Play a clip ONCE over whatever currently drives the pose — the locomotion blend,
   *  a playing clip, a crossfade, or rest. The background keeps advancing underneath;
   *  fade-in ramps the one-shot over it and fade-out returns to whatever the
   *  background is doing by then. Replaces any active one-shot immediately. */
  playOneShot(name: string, options?: { fadeIn?: number; fadeOut?: number; onEnd?: () => void }): boolean {
    const clip = this.animationState.getAnimationClip(name)
    if (!clip || clip.frameCount <= 0 || !Number.isFinite(clip.frameCount)) return false
    const duration = clip.frameCount / FPS
    const fadeIn = Math.max(0, options?.fadeIn ?? 0.15)
    const fadeOut = Math.max(0, Math.min(options?.fadeOut ?? 0.25, duration))
    this.oneShot = { name, time: 0, duration, fadeIn, fadeOut, cancelW: 1, cancelling: false, onEnd: options?.onEnd ?? null }
    this.clipApplySuspended = false
    return true
  }

  /** Fade the active one-shot out early over `fadeOut` seconds (default 0.2). */
  cancelOneShot(fadeOut = 0.2): void {
    if (!this.oneShot) return
    if (fadeOut <= 0) {
      this.oneShot = null
      return
    }
    this.oneShot.cancelling = true
    this.oneShot.fadeOut = fadeOut
  }

  /** Name of the active one-shot, or null. */
  getOneShot(): string | null {
    return this.oneShot?.name ?? null
  }

  /** Fire `callback` whenever `clip`'s playback crosses `time` (seconds) with at
   *  least `minWeight` influence (default 0.1) — on any path: blend entries,
   *  one-shots, crossfades, or plain play. Loop wraps fire correctly; hard cursor
   *  jumps may occasionally fire or skip (events are for sfx-grade timing, not
   *  logic). Returns an unsubscribe function. */
  addClipEvent(clip: string, time: number, callback: (e: ClipEventInfo) => void, options?: { minWeight?: number }): () => void {
    let list = this.clipEvents.get(clip)
    if (!list) {
      list = []
      this.clipEvents.set(clip, list)
    }
    const def = { time, minWeight: options?.minWeight ?? 0.1, callback }
    list.push(def)
    return () => {
      const i = list.indexOf(def)
      if (i >= 0) list.splice(i, 1)
    }
  }

  private fireClipEvents(name: string, prev: number, now: number, weight: number): void {
    const evs = this.clipEvents.get(name)
    if (!evs || evs.length === 0) return
    const wrapped = now < prev
    for (let i = 0; i < evs.length; i++) {
      const ev = evs[i]
      if (weight < ev.minWeight) continue
      const hit = wrapped ? ev.time > prev || ev.time <= now : ev.time > prev && ev.time <= now
      if (hit) ev.callback({ clip: name, time: ev.time, weight })
    }
  }

  /** Per-entry cursor memory for event crossing detection; keyed on entry object
   *  identity (controllers reuse their arrays, so this stays warm). */
  private trackEntryEvents(e: BlendEntry): void {
    const prev = this.entryEventPrev.get(e)
    if (prev === undefined) {
      this.entryEventPrev.set(e, { name: e.name, time: e.time })
      return
    }
    if (prev.name === e.name && prev.time !== e.time) this.fireClipEvents(e.name, prev.time, e.time, e.weight)
    prev.name = e.name
    prev.time = e.time
  }

  /** Fade from the currently playing clip (or from the rest pose when nothing plays)
   *  into `name` over `seconds`. The target starts at frame 0 and becomes the current
   *  clip immediately — progress, looping and the camera clock report the target for
   *  the whole fade. Bones only the outgoing clip animates ease back to rest. */
  crossfadeTo(name: string, seconds: number, options?: { loop?: boolean }): boolean {
    if (!this.animationState.hasAnimation(name)) return false
    this.blendEntries = null
    this.oneShot = null
    this.clipApplySuspended = false

    const fromName = this.animationState.getCurrentAnimation()
    const fromFrame = this.animationState.getCurrentFrame()
    const fromLoop = this.animationState.getProgress().looping
    this.animationState.forcePlay(name, options?.loop ?? false)

    // Fading a clip into itself has no second pose to hold — just restart.
    if (seconds <= 0 || fromName === name) {
      this.crossfade = null
      return true
    }
    this.crossfade = { fromName, fromFrame, fromLoop, elapsed: 0, duration: seconds }
    return true
  }

  pause(): void {
    this.animationState.pause()
  }

  stop(): void {
    this.blendEntries = null
    this.crossfade = null
    this.oneShot = null
    this.animationState.stop()
  }

  /** Deactivate the current clip entirely (stop + forget). Unlike stop(), the
   *  pose is no longer re-applied each frame afterwards — follow with
   *  resetAllBones()/resetAllMorphs() to return to the bind pose. */
  clearAnimation(): void {
    this.blendEntries = null
    this.crossfade = null
    this.oneShot = null
    this.animationState.clear()
  }

  // Seek by absolute timeline seconds, not frame index.
  seek(seconds: number): void {
    this.clipApplySuspended = false
    this.crossfade = null // a timeline jump mid-fade snaps to the target clip
    this.animationState.seek(seconds)
  }

  getAnimationProgress(): AnimationProgress {
    const p = this.animationState.getProgress()
    return {
      current: p.current,
      duration: p.duration,
      percentage: p.percentage,
      animationName: p.animationName,
      looping: p.looping,
      playing: p.playing,
      paused: p.paused,
    }
  }

  private static upperBound<T extends { frame: number }>(frame: number, keyFrames: T[], startIdx: number = 0): number {
    let left = startIdx,
      right = keyFrames.length
    while (left < right) {
      const mid = Math.floor((left + right) / 2)
      if (keyFrames[mid].frame <= frame) left = mid + 1
      else right = mid
    }
    return left
  }

  private findKeyframeIndex<T extends { frame: number }>(frame: number, keyFrames: T[], cachedIdx: number): number {
    if (keyFrames.length === 0) return -1

    if (cachedIdx >= 0 && cachedIdx < keyFrames.length) {
      const currentFrame = keyFrames[cachedIdx].frame
      const nextFrame = cachedIdx + 1 < keyFrames.length ? keyFrames[cachedIdx + 1].frame : Infinity
      if (frame >= currentFrame && frame < nextFrame) {
        return cachedIdx
      }
    }
    const idx = Model.upperBound(frame, keyFrames, 0) - 1
    return idx
  }

  /**
   * IK bones whose chains are switched OFF at the current frame.
   *
   * VMD carries this per chain and over time — a motion legitimately disables
   * foot IK for a lift and restores it on landing — so it belongs to the clip,
   * not to a global switch on the engine.
   */
  private ikDisabled = new Set<number>()

  /** Step the IK state to `frame`. VMD IK keys are steps: a state holds until
   *  the next one changes it, so this is a lookup, not an interpolation. */
  private applyIkFromClip(clip: AnimationClip, frame: number): void {
    if (!clip.ikTracks || clip.ikTracks.size === 0) {
      if (this.ikDisabled.size > 0) this.ikDisabled.clear()
      return
    }
    this.ikDisabled.clear()
    for (const [boneName, keys] of clip.ikTracks) {
      const index = this.runtimeSkeleton.nameIndex[boneName]
      if (index === undefined || index < 0 || keys.length === 0) continue
      let state = keys[0].frame <= frame ? keys[0].enabled : true
      for (let i = 1; i < keys.length && keys[i].frame <= frame; i++) state = keys[i].enabled
      if (!state) this.ikDisabled.add(index)
    }
  }

  /** Sample one bone track at `frame` into outRot + outVmdTrans. The translation is
   *  VMD-space (bind-relative), NOT yet converted to bone-local — callers convert with
   *  convertVMDTranslationToLocal once they know the final rotation. The cursor cache
   *  belongs to the caller (single-clip path vs per-clip blend caches). Returns false
   *  for an empty track. */
  private sampleBoneTrackInto(
    boneName: string,
    keyFrames: BoneKeyframe[],
    frame: number,
    cursors: Map<string, number>,
    outRot: Quat,
    outVmdTrans: Vec3
  ): boolean {
    if (keyFrames.length === 0) return false

    const cachedIdx = cursors.get(boneName) ?? -1
    const clampedFrame = Math.max(keyFrames[0].frame, Math.min(keyFrames[keyFrames.length - 1].frame, frame))
    const idx = this.findKeyframeIndex(clampedFrame, keyFrames, cachedIdx)
    if (idx < 0) return false
    cursors.set(boneName, idx)

    const frameA = keyFrames[idx]
    const frameB = keyFrames[idx + 1]

    if (!frameB) {
      outRot.set(frameA.rotation)
      outVmdTrans.set(frameA.translation)
    } else {
      const frameDelta = frameB.frame - frameA.frame
      const gradient = frameDelta > 0 ? (clampedFrame - frameA.frame) / frameDelta : 0
      const interp = frameB.interpolation

      const rotT = interpolateControlPoints(interp.rotation, gradient)
      Quat.slerpInto(frameA.rotation, frameB.rotation, rotT, outRot)

      const txWeight = interpolateControlPoints(interp.translationX, gradient)
      const tyWeight = interpolateControlPoints(interp.translationY, gradient)
      const tzWeight = interpolateControlPoints(interp.translationZ, gradient)

      outVmdTrans.setXYZ(
        frameA.translation.x + (frameB.translation.x - frameA.translation.x) * txWeight,
        frameA.translation.y + (frameB.translation.y - frameA.translation.y) * tyWeight,
        frameA.translation.z + (frameB.translation.z - frameA.translation.z) * tzWeight
      )
    }
    return true
  }

  /** Sample one morph track at `frame` (linear weight lerp). Returns NaN for an empty track. */
  private sampleMorphTrack(morphName: string, keyFrames: MorphKeyframe[], frame: number, cursors: Map<string, number>): number {
    if (keyFrames.length === 0) return NaN

    const cachedIdx = cursors.get(morphName) ?? -1
    const clampedFrame = Math.max(keyFrames[0].frame, Math.min(keyFrames[keyFrames.length - 1].frame, frame))
    const idx = this.findKeyframeIndex(clampedFrame, keyFrames, cachedIdx)
    if (idx < 0) return NaN
    cursors.set(morphName, idx)

    const frameA = keyFrames[idx]
    const frameB = keyFrames[idx + 1]
    return frameB
      ? frameA.weight +
      (frameB.weight - frameA.weight) *
      (frameB.frame > frameA.frame ? (clampedFrame - frameA.frame) / (frameB.frame - frameA.frame) : 0)
      : frameA.weight
  }

  /**
   * Hold 軸制限 bones to their own axis.
   *
   * A twist bone (腕捩 / 手捩) exists to rotate about the length of the limb and
   * nothing else, and PMX says so by giving it a fixed axis. A VMD still keys it
   * with a full quaternion, so without the constraint the bone bends as well as
   * twists — and because the elbow is parented to 腕捩 and 腕捩1..3 inherit a
   * fraction of it, that error is carried into the whole forearm and multiplied
   * three ways down the arm. It is the most visible thing on the model.
   *
   * Projecting the quaternion's vector part onto the axis and renormalising is
   * what keeps the twist and drops everything else; the same operation MMD and
   * every faithful runtime performs.
   *
   * Applied to the STORED local rotation rather than inside a matrix path,
   * because the append children read that array directly: constrain it once and
   * the bone, its inheritors and a hand-posed gizmo all agree.
   */
  private applyFixedAxes(): void {
    for (const a of this.fixedAxisBones) {
      const q = this.runtimeSkeleton.localRotations[a.index]
      const dot = q.x * a.x + q.y * a.y + q.z * a.z
      const x = a.x * dot
      const y = a.y * dot
      const z = a.z * dot
      const len = Math.sqrt(x * x + y * y + z * z + q.w * q.w)
      if (len > 1e-8) {
        const inv = 1 / len
        q.setXYZW(x * inv, y * inv, z * inv, q.w * inv)
      }
    }
  }

  private applyPoseFromClip(clip: AnimationClip | null, frame: number): void {
    if (!clip) return
    this.applyIkFromClip(clip, frame)
    if (clip !== this.lastAppliedClip) {
      this.boneTrackIndices.clear()
      this.morphTrackIndices.clear()
      this.lastAppliedClip = clip
    }

    for (const [boneName, keyFrames] of clip.boneTracks.entries()) {
      if (!this.sampleBoneTrackInto(boneName, keyFrames, frame, this.boneTrackIndices, _animSlerp, _animInterpT)) continue

      const boneIdx = this.runtimeSkeleton.nameIndex[boneName]
      if (boneIdx === undefined) continue

      const localTranslation = this.convertVMDTranslationToLocal(boneIdx, _animInterpT, _animSlerp)
      this.runtimeSkeleton.localRotations[boneIdx].set(_animSlerp)
      this.runtimeSkeleton.localTranslations[boneIdx].set(localTranslation)
    }
    this.applyFixedAxes()

    for (const [morphName, keyFrames] of clip.morphTracks.entries()) {
      const weight = this.sampleMorphTrack(morphName, keyFrames, frame, this.morphTrackIndices)
      if (Number.isNaN(weight)) continue

      const morphIdx = this.runtimeMorph.nameIndex[morphName]
      if (morphIdx === undefined) continue

      this.runtimeMorph.weights[morphIdx] = weight
      this.morphsDirty = true // Mark as dirty when animation sets morph weights
    }
  }

  /** Weighted N-clip pose blend. Rotations accumulate by hemisphere-aligned weighted
   *  sum then normalize (nlerp generalized to N inputs); VMD-space translations lerp.
   *  A bone missing from a clip contributes that clip's share of the REST pose, so a
   *  weight sum below 1 fades toward rest — that is also how crossfading from "no
   *  clip" works. IK on/off state follows the highest-weight entry. */
  private applyBlendedPose(entries: BlendEntry[]): void {
    // Resolve entries; find the total weight and the dominant entry (drives IK state).
    let total = 0
    let live = 0
    let domClip: AnimationClip | null = null
    let domFrame = 0
    let domWeight = 0
    for (let i = 0; i < entries.length; i++) {
      const e = entries[i]
      if (!(e.weight > 1e-6)) continue
      const clip = this.animationState.getAnimationClip(e.name)
      if (!clip) continue
      total += e.weight
      live++
      if (e.weight > domWeight) {
        domWeight = e.weight
        domClip = clip
        domFrame = e.time * FPS
      }
    }
    if (live === 0 || domClip === null) return
    // A sum above 1 normalizes; below 1 stays — the remainder is the rest pose's share.
    const norm = total > 1 ? 1 / total : 1

    this.applyIkFromClip(domClip, domFrame)

    if (this.blendRotAcc === null) {
      const boneCount = this.runtimeSkeleton.localRotations.length
      this.blendRotAcc = Array.from({ length: boneCount }, () => new Quat(0, 0, 0, 0))
      this.blendTransAcc = Array.from({ length: boneCount }, () => new Vec3(0, 0, 0))
      this.blendWeightAcc = new Float32Array(boneCount)
      this.blendBoneGen = new Int32Array(boneCount).fill(-1)
      this.blendMorphAcc = new Float32Array(this.runtimeMorph.weights.length)
      this.blendMorphGen = new Int32Array(this.runtimeMorph.weights.length).fill(-1)
    }
    const rotAcc = this.blendRotAcc
    const transAcc = this.blendTransAcc!
    const weightAcc = this.blendWeightAcc!
    const boneGen = this.blendBoneGen!
    const morphAcc = this.blendMorphAcc!
    const morphGen = this.blendMorphGen!
    const gen = ++this.blendGenCounter

    for (let i = 0; i < entries.length; i++) {
      const e = entries[i]
      if (!(e.weight > 1e-6)) continue
      const clip = this.animationState.getAnimationClip(e.name)
      if (!clip) continue
      const w = e.weight * norm
      const frame = e.time * FPS
      if (this.clipEvents.size > 0) this.trackEntryEvents(e)

      let cursors = this.blendBoneCursors.get(clip)
      if (!cursors) {
        cursors = new Map()
        this.blendBoneCursors.set(clip, cursors)
      }
      for (const [boneName, keyFrames] of clip.boneTracks.entries()) {
        if (!this.sampleBoneTrackInto(boneName, keyFrames, frame, cursors, _blendQ, _blendT)) continue
        const boneIdx = this.runtimeSkeleton.nameIndex[boneName]
        if (boneIdx === undefined) continue

        const r = rotAcc[boneIdx]
        const t = transAcc[boneIdx]
        if (boneGen[boneIdx] !== gen) {
          boneGen[boneIdx] = gen
          r.setXYZW(_blendQ.x * w, _blendQ.y * w, _blendQ.z * w, _blendQ.w * w)
          t.setXYZ(_blendT.x * w, _blendT.y * w, _blendT.z * w)
          weightAcc[boneIdx] = w
        } else {
          // Hemisphere-align this sample against the accumulated sum before adding.
          const d = r.x * _blendQ.x + r.y * _blendQ.y + r.z * _blendQ.z + r.w * _blendQ.w
          const s = d < 0 ? -w : w
          r.x += _blendQ.x * s
          r.y += _blendQ.y * s
          r.z += _blendQ.z * s
          r.w += _blendQ.w * s
          t.x += _blendT.x * w
          t.y += _blendT.y * w
          t.z += _blendT.z * w
          weightAcc[boneIdx] += w
        }
      }

      let morphCursors = this.blendMorphCursors.get(clip)
      if (!morphCursors) {
        morphCursors = new Map()
        this.blendMorphCursors.set(clip, morphCursors)
      }
      for (const [morphName, keyFrames] of clip.morphTracks.entries()) {
        const weight = this.sampleMorphTrack(morphName, keyFrames, frame, morphCursors)
        if (Number.isNaN(weight)) continue
        const morphIdx = this.runtimeMorph.nameIndex[morphName]
        if (morphIdx === undefined) continue
        if (morphGen[morphIdx] !== gen) {
          morphGen[morphIdx] = gen
          morphAcc[morphIdx] = weight * w
        } else {
          morphAcc[morphIdx] += weight * w
        }
      }
    }

    // Finalize touched bones: fold the rest-pose remainder in, normalize, convert.
    const boneCount = rotAcc.length
    for (let i = 0; i < boneCount; i++) {
      if (boneGen[i] !== gen) continue
      const r = rotAcc[i]
      const wSum = weightAcc[i]
      if (wSum < 1) {
        // Rest local rotation is identity (0,0,0,1); translation contribution is zero.
        const rest = 1 - wSum
        r.w += r.w < 0 ? -rest : rest
      }
      const len = Math.sqrt(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w)
      if (len > 1e-8) {
        const inv = 1 / len
        _blendQ.setXYZW(r.x * inv, r.y * inv, r.z * inv, r.w * inv)
      } else {
        _blendQ.setIdentity()
      }
      const localTranslation = this.convertVMDTranslationToLocal(i, transAcc[i], _blendQ)
      this.runtimeSkeleton.localRotations[i].set(_blendQ)
      this.runtimeSkeleton.localTranslations[i].set(localTranslation)
    }
    this.applyFixedAxes()

    const morphCount = morphAcc.length
    for (let i = 0; i < morphCount; i++) {
      if (morphGen[i] !== gen) continue
      this.runtimeMorph.weights[i] = morphAcc[i]
      this.morphsDirty = true
    }
  }

  /** One one-shot step: weight envelope from fade-in/out (easeInOut-shaped), the
   *  background entries scaled by 1-w underneath, the one-shot clip at w on top. */
  private applyOneShot(deltaTime: number): void {
    const os = this.oneShot
    if (os === null) return
    os.time += deltaTime

    // Envelope: min of the fade-in ramp and the fade-out ramp (or the cancel ramp).
    const wIn = os.fadeIn > 0 ? Math.min(1, os.time / os.fadeIn) : 1
    let wOut = 1
    if (os.cancelling) {
      os.cancelW -= deltaTime / os.fadeOut
      wOut = Math.max(0, os.cancelW)
    } else if (os.fadeOut > 0) {
      wOut = Math.max(0, Math.min(1, (os.duration - os.time) / os.fadeOut))
    }
    const w = easeInOut(Math.min(wIn, wOut))

    // Background entries at (1 - w): the still-advancing blend, the current clip, or
    // nothing (rest fill). Copied into a pooled array — never scale caller-owned weights.
    const pool = this.oneShotEntries
    let n = 0
    const put = (name: string, time: number, weight: number) => {
      if (pool.length <= n) pool.push({ name: "", time: 0, weight: 0 })
      const e = pool[n++]
      e.name = name
      e.time = time
      e.weight = weight
    }
    const bg = 1 - w
    if (bg > 1e-6) {
      if (this.blendEntries !== null && this.blendEntries.length > 0) {
        for (const e of this.blendEntries) put(e.name, e.time, e.weight * bg)
      } else {
        const clip = this.animationState.getCurrentClip()
        const name = this.animationState.getCurrentAnimation()
        if (clip !== null && name !== null && name !== os.name) {
          put(name, this.animationState.getCurrentFrame() / FPS, bg)
        }
        // else: rest pose fills the remainder inside applyBlendedPose
      }
    }
    put(os.name, Math.min(os.time, os.duration), w)
    for (let i = n; i < pool.length; i++) pool[i].weight = 0
    this.applyBlendedPose(pool)

    if (os.time >= os.duration || (os.cancelling && os.cancelW <= 0)) {
      const onEnd = os.onEnd
      this.oneShot = null
      onEnd?.()
    }
  }

  /** One crossfade step: advance the outgoing clock (the target's clock lives in
   *  animationState, already ticked by update), shape the weight with easeInOut,
   *  and hand both to the blend sampler. Holds in place while paused. */
  private applyCrossfade(deltaTime: number): void {
    const fade = this.crossfade
    if (fade === null) return
    const playing = this.animationState.getProgress().playing

    if (playing) {
      fade.elapsed += deltaTime
      if (fade.fromName !== null) {
        const fromClip = this.animationState.getAnimationClip(fade.fromName)
        if (fromClip && fromClip.frameCount > 0 && Number.isFinite(fromClip.frameCount)) {
          fade.fromFrame += deltaTime * FPS
          if (fade.fromLoop) {
            while (fade.fromFrame >= fromClip.frameCount) fade.fromFrame -= fromClip.frameCount
          } else if (fade.fromFrame > fromClip.frameCount) {
            fade.fromFrame = fromClip.frameCount
          }
        } else {
          fade.fromName = null // outgoing clip was removed mid-fade: fade from rest
        }
      }
    }

    const t = fade.duration > 0 ? Math.min(1, fade.elapsed / fade.duration) : 1
    const w = easeInOut(t)
    const toName = this.animationState.getCurrentAnimation()

    _fadeEntries[0].name = toName ?? ""
    _fadeEntries[0].time = this.animationState.getCurrentFrame() / FPS
    _fadeEntries[0].weight = toName !== null ? w : 0
    _fadeEntries[1].name = fade.fromName ?? ""
    _fadeEntries[1].time = fade.fromFrame / FPS
    _fadeEntries[1].weight = fade.fromName !== null ? 1 - w : 0
    this.applyBlendedPose(_fadeEntries)

    if (t >= 1) this.crossfade = null
  }

  // Returns true when morphs changed (vertex buffer may need upload). `ikEnabled`
  // is the host's runtime switch (engine-wide); the clip decides which chains
  // within that. A host driving bones directly — motion capture writing FK
  // rotations every frame with no clip playing — turns it off wholesale, because
  // there is no motion present to carry the per-chain answer.
  /**
   * Nothing can have moved this frame: no clip, no blend, no live tween, no
   * morph weight change.
   *
   * Environment geometry is in this state almost every frame, so the engine
   * skips the whole pose pass — sampling, world matrices, and the skin-matrix
   * upload — for a stage that reports idle. A stage is usually the heaviest mesh
   * in the scene and the one that never moves; paying a full pose pass for it
   * every frame is the thing worth not doing.
   */
  isIdle(): boolean {
    // Never idle before the first pose pass. The constructor leaves the world
    // matrices identity, so skin = world × inverseBind collapses every vertex
    // into bone-local space — the mesh piles up at the origin. A cast member is
    // saved by running update() on frame 1 regardless; a stage that reported
    // idle immediately would render as a heap and never recover.
    if (!this.posedOnce) return false
    return (
      !this.morphsDirty &&
      this.oneShot === null &&
      this.crossfade === null &&
      (this.blendEntries === null || this.blendEntries.length === 0) &&
      this.animationState.getCurrentClip() === null &&
      !this.hasActiveTweens()
    )
  }

  /** Any live rotation / translation / morph tween. */
  private hasActiveTweens(): boolean {
    const s = this.tweenState
    for (let i = 0; i < s.rotActive.length; i++) if (s.rotActive[i] === 1) return true
    for (let i = 0; i < s.transActive.length; i++) if (s.transActive[i] === 1) return true
    for (let i = 0; i < s.morphActive.length; i++) if (s.morphActive[i] === 1) return true
    return false
  }

  update(deltaTime: number, ikEnabled = true): boolean {
    // Update tween time (in milliseconds)
    this.tweenTimeMs += deltaTime * 1000

    // Update all active tweens (rotations, translations, morphs)
    const tweensChangedMorphs = this.updateTweens()

    const evWatch = this.clipEvents.size > 0 ? this.animationState.getCurrentAnimation() : null
    const evPrevFrame = evWatch !== null ? this.animationState.getCurrentFrame() : 0
    this.animationState.update(deltaTime)

    // Hand the pose sources their bones unmorphed. A bone morph is an offset on
    // top of the pose, so last frame's offset has to come off before this
    // frame's pose goes on — taking it off afterwards is what froze every
    // morph-touched bone at its first posed frame.
    this.undoBoneMorphs()

    if (!this.clipApplySuspended) {
      if (this.oneShot !== null) {
        this.applyOneShot(deltaTime)
      } else if (this.blendEntries !== null && this.blendEntries.length > 0) {
        this.applyBlendedPose(this.blendEntries)
      } else if (this.crossfade !== null) {
        this.applyCrossfade(deltaTime)
      } else {
        const clip = this.animationState.getCurrentClip()
        if (clip !== null) {
          this.applyPoseFromClip(clip, this.animationState.getCurrentFrame())
          if (evWatch !== null && evWatch === this.animationState.getCurrentAnimation()) {
            this.fireClipEvents(evWatch, evPrevFrame / FPS, this.animationState.getCurrentFrame() / FPS, 1)
          }
        }
      }
    }

    // Constant per-bone offsets compose after every pose source, before the
    // world/IK passes (ankle offsets survive IK — the solver moves thigh/knee).
    for (const [idx, offset] of this.boneRotationOffsets) {
      const r = this.runtimeSkeleton.localRotations[idx]
      Quat.multiplyInto(r, offset, r)
    }

    // Apply morphs if tweens changed morphs or animation changed morphs
    const verticesChanged = this.morphsDirty || tweensChangedMorphs
    if (this.morphsDirty || tweensChangedMorphs) {
      this.applyMorphs()
      this.morphsDirty = false
    }

    // After the pose sources and the constant offsets, before the world pass —
    // bone morphs must survive into the world matrices IK then reads.
    this.applyBoneMorphs()

    // Compute world matrices (needed for IK solving to read bone positions)
    this.computeWorldMatrices()
    this.posedOnce = true

    // Solve IK chains (modifies localRotations with final IK rotations). Chains
    // the clip switched off are skipped inside.
    if (ikEnabled) {
      this.solveIKChains()
      // Recompute world matrices with final IK rotations applied to localRotations
      this.computeWorldMatrices()
    }

    return verticesChanged
  }

  private solveIKChains(): void {
    const ikSolvers = this.runtimeSkeleton.ikSolvers
    if (!ikSolvers || ikSolvers.length === 0) return

    const ikChainInfo = this.runtimeSkeleton.ikChainInfo
    if (!ikChainInfo) return

    // Solve each IK solver sequentially, ensuring consistent state between solvers
    let firstSolver = true
    for (const solver of ikSolvers) {
      // Switched off by the motion for this frame — leave the chain on whatever
      // the FK track put there.
      if (this.ikDisabled.has(solver.ikBoneIndex)) continue
      // Each solver must see the effects of previous solvers on localRotations, so
      // recompute world matrices between solvers. The first solver is skipped: the
      // caller (update) already computed them and nothing has changed localRotations yet.
      if (!firstSolver) this.computeWorldMatrices()
      firstSolver = false

      // Clear computed set for this solver's pass
      this.ikComputedSet.clear()

      // Solve this IK chain
      // Pass callback that uses model's world matrix computation (handles append correctly)
      IKSolverSystem.solve(
        [solver], // Solve one at a time
        this.skeleton.bones,
        this.runtimeSkeleton.localRotations,
        this.runtimeSkeleton.localTranslations,
        this.runtimeSkeleton.worldMatrices,
        ikChainInfo,
        (boneIndex, applyIK) => {
          // Clear computed set for each bone update to allow recomputation in same iteration
          this.ikComputedSet.delete(boneIndex)
          this.computeSingleBoneWorldMatrix(boneIndex, applyIK)
        }
      )
    }
  }

  // Cached set to track which bones are being computed in current IK pass (to avoid infinite recursion)
  private ikComputedSet: Set<number> = new Set()

  // Add this new method to compute a single bone's world matrix
  // Recursively ensures parents are computed first to avoid using stale parent matrices
  private computeSingleBoneWorldMatrix(boneIndex: number, applyIK: boolean): void {
    const bones = this.skeleton.bones
    const localRot = this.runtimeSkeleton.localRotations
    const localTrans = this.runtimeSkeleton.localTranslations
    const worldMats = this.runtimeSkeleton.worldMatrices
    const ikChainInfo = this.runtimeSkeleton.ikChainInfo

    const b = bones[boneIndex]

    // Prevent infinite recursion: if this bone is already being computed in this call chain, skip
    if (this.ikComputedSet.has(boneIndex)) {
      return
    }

    // Mark this bone as being computed to prevent infinite recursion
    this.ikComputedSet.add(boneIndex)

    // Recursively compute parent first if it exists (ensures parent matrix is up-to-date)
    if (b.parentIndex >= 0) {
      this.computeSingleBoneWorldMatrix(b.parentIndex, applyIK)
    }

    // Get base rotation
    const baseRot = localRot[boneIndex]
    let fx = baseRot.x, fy = baseRot.y, fz = baseRot.z, fw = baseRot.w

    // Apply IK rotation if requested: finalRot = ik * base, then normalize
    if (applyIK && ikChainInfo) {
      const chainInfo = ikChainInfo[boneIndex]
      if (chainInfo?.ikRotation) {
        const ik = chainInfo.ikRotation
        const nx = ik.w * fx + ik.x * fw + ik.y * fz - ik.z * fy
        const ny = ik.w * fy - ik.x * fz + ik.y * fw + ik.z * fx
        const nz = ik.w * fz + ik.x * fy - ik.y * fx + ik.z * fw
        const nw = ik.w * fw - ik.x * fx - ik.y * fy - ik.z * fz
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
        const inv = len > 0 ? 1 / len : 0
        fx = nx * inv; fy = ny * inv; fz = nz * inv; fw = nw * inv
      }
    }

    let addLocalTx = 0, addLocalTy = 0, addLocalTz = 0

    // Handle append transformations (same logic as computeWorldMatrices)
    const appendParentIdx = b.appendParentIndex
    // `|| appendMove`: a move-only append (no rotation) was skipped altogether,
    // because the guard demanded appendRotate before either branch could run.
    const hasAppend = (b.appendRotate || b.appendMove) &&
      appendParentIdx !== undefined &&
      appendParentIdx >= 0 &&
      appendParentIdx < bones.length

    if (hasAppend) {
      const ratio = b.appendRatio === undefined ? 1 : Math.max(-1, Math.min(1, b.appendRatio))
      const hasRatio = Math.abs(ratio) > 1e-6

      if (hasRatio) {
        if (b.appendRotate) {
          // Recurse first (may touch scratch); all scratch use below happens after it unwinds
          if (appendParentIdx >= 0) {
            this.computeSingleBoneWorldMatrix(appendParentIdx, applyIK)
          }

          const appendRot = localRot[appendParentIdx]
          let ax = appendRot.x, ay = appendRot.y, az = appendRot.z
          const aw = appendRot.w
          const absRatio = ratio < 0 ? -ratio : ratio
          if (ratio < 0) { ax = -ax; ay = -ay; az = -az }

          // slerp(identity, appendQuat, absRatio) into scratchQuat[1]
          scratchQuat[0].setXYZW(ax, ay, az, aw)
          scratchQuat[2].setIdentity()
          Quat.slerpInto(scratchQuat[2], scratchQuat[0], absRatio, scratchQuat[1])

          // MMD composes the append on the RIGHT: own animated rotation first, then
          // the inherited one (saba's `r = r * appendRotate`). We had it on the
          // left, which is a different rotation whenever a bone carries BOTH its
          // own key and an append — the twist bones, if a motion keys them.
          const sx = scratchQuat[1].x, sy = scratchQuat[1].y, sz = scratchQuat[1].z, sw = scratchQuat[1].w
          const nx = fw * sx + fx * sw + fy * sz - fz * sy
          const ny = fw * sy - fx * sz + fy * sw + fz * sx
          const nz = fw * sz + fx * sy - fy * sx + fz * sw
          const nw = fw * sw - fx * sx - fy * sy - fz * sz
          fx = nx; fy = ny; fz = nz; fw = nw
        }

        if (b.appendMove) {
          const appendTrans = localTrans[appendParentIdx]
          addLocalTx = appendTrans.x * ratio
          addLocalTy = appendTrans.y * ratio
          addLocalTz = appendTrans.z * ratio
        }
      }
    }

    const boneTrans = localTrans[boneIndex]
    const localTx = boneTrans.x + addLocalTx
    const localTy = boneTrans.y + addLocalTy
    const localTz = boneTrans.z + addLocalTz

    // Fused local transform: T_bind · R(finalRot) · T_local → scratchMat4Values[0]
    const localMVals = scratchMat4Values[0]
    Mat4.localTransformInto(
      b.bindTranslation[0], b.bindTranslation[1], b.bindTranslation[2],
      fx, fy, fz, fw,
      localTx, localTy, localTz,
      localMVals
    )

    const worldMat = worldMats[boneIndex]
    if (b.parentIndex >= 0) {
      const parentMat = worldMats[b.parentIndex]
      Mat4.multiplyArrays(parentMat.values, 0, localMVals, 0, worldMat.values, 0)
    } else {
      worldMat.values.set(localMVals)
    }
  }

  computeWorldMatrices(): void {
    const bones = this.skeleton.bones
    const localRot = this.runtimeSkeleton.localRotations
    const localTrans = this.runtimeSkeleton.localTranslations
    const worldMats = this.runtimeSkeleton.worldMatrices
    const boneCount = bones.length

    if (boneCount === 0) return

    // Flat traversal in precomputed order: every bone's parent is already done, so no
    // per-bone visited check, no recursion, and no per-call allocation. Same per-bone
    // math as before. Scratch slots are safe to reuse since there's no reentrancy now.
    const order = this.deformOrder
    for (let k = 0; k < boneCount; k++) {
      const i = order[k]
      const b = bones[i]

      const boneRot = localRot[i]
      let fx = boneRot.x, fy = boneRot.y, fz = boneRot.z, fw = boneRot.w
      let addLocalTx = 0, addLocalTy = 0, addLocalTz = 0

      const appendParentIdx = b.appendParentIndex
      const hasAppend =
        (b.appendRotate || b.appendMove) && appendParentIdx !== undefined && appendParentIdx >= 0 && appendParentIdx < boneCount

      if (hasAppend) {
        const ratio = b.appendRatio === undefined ? 1 : Math.max(-1, Math.min(1, b.appendRatio))
        const hasRatio = Math.abs(ratio) > 1e-6

        if (hasRatio) {
          if (b.appendRotate) {
            const appendRot = localRot[appendParentIdx]
            let ax = appendRot.x, ay = appendRot.y, az = appendRot.z
            const aw = appendRot.w
            const absRatio = ratio < 0 ? -ratio : ratio
            if (ratio < 0) { ax = -ax; ay = -ay; az = -az }

            scratchQuat[0].setXYZW(ax, ay, az, aw)
            scratchQuat[2].setIdentity()
            Quat.slerpInto(scratchQuat[2], scratchQuat[0], absRatio, scratchQuat[1])

            // MMD composes the append on the RIGHT: own animated rotation first, then
            // the inherited one (saba's `r = r * appendRotate`). We had it on the
            // left, which is a different rotation whenever a bone carries BOTH its
            // own key and an append — the twist bones, if a motion keys them.
            const sx = scratchQuat[1].x, sy = scratchQuat[1].y, sz = scratchQuat[1].z, sw = scratchQuat[1].w
            const nx = fw * sx + fx * sw + fy * sz - fz * sy
            const ny = fw * sy - fx * sz + fy * sw + fz * sx
            const nz = fw * sz + fx * sy - fy * sx + fz * sw
            const nw = fw * sw - fx * sx - fy * sy - fz * sz
            fx = nx; fy = ny; fz = nz; fw = nw
          }

          if (b.appendMove) {
            const appendTrans = localTrans[appendParentIdx]
            const appendRatio = b.appendRatio ?? 1
            addLocalTx = appendTrans.x * appendRatio
            addLocalTy = appendTrans.y * appendRatio
            addLocalTz = appendTrans.z * appendRatio
          }
        }
      }

      const boneTrans = localTrans[i]
      const localTx = boneTrans.x + addLocalTx
      const localTy = boneTrans.y + addLocalTy
      const localTz = boneTrans.z + addLocalTz

      const localMVals = scratchMat4Values[0]
      Mat4.localTransformInto(
        b.bindTranslation[0], b.bindTranslation[1], b.bindTranslation[2],
        fx, fy, fz, fw,
        localTx, localTy, localTz,
        localMVals
      )

      const worldMat = worldMats[i]
      if (b.parentIndex >= 0) {
        const parentMat = worldMats[b.parentIndex]
        Mat4.multiplyArrays(parentMat.values, 0, localMVals, 0, worldMat.values, 0)
      } else {
        worldMat.values.set(localMVals)
      }
    }
  }
}