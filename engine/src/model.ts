import { Mat4, Quat, Vec3, scratchMat4Values, scratchQuat } from "./math"
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
  BoneInterpolation,
  BoneKeyframe,
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

// Morph definition
export interface Morph {
  name: string
  type: number // 0=group, 1=vertex, 2=bone, 3=UV, 8=material
  vertexOffsets: VertexMorphOffset[] // Only for type 1 (vertex morph)
  groupReferences?: GroupMorphReference[] // Only for type 0 (group morph)
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
      this.animationState.play()
      return
    }
    this.resetAllBones()
    this.resetAllMorphs()
    return this.animationState.play(name, options)
  }

  show(name: string): void {
    this.resetAllBones()
    this.resetAllMorphs()
    this.animationState.show(name)
  }

  // @deprecated Use model.play()
  playAnimation(): void {
    this.animationState.play()
  }

  pause(): void {
    this.animationState.pause()
  }

  // @deprecated Use model.pause()
  pauseAnimation(): void {
    this.animationState.pause()
  }

  stop(): void {
    this.animationState.stop()
  }

  // @deprecated Use model.stop()
  stopAnimation(): void {
    this.animationState.stop()
  }

  /** Deactivate the current clip entirely (stop + forget). Unlike stop(), the
   *  pose is no longer re-applied each frame afterwards — follow with
   *  resetAllBones()/resetAllMorphs() to return to the bind pose. */
  clearAnimation(): void {
    this.animationState.clear()
  }

  // Seek by absolute timeline seconds, not frame index.
  seek(seconds: number): void {
    this.clipApplySuspended = false
    this.animationState.seek(seconds)
  }

  // @deprecated Use model.seek()
  seekAnimation(seconds: number): void {
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

  private applyPoseFromClip(clip: AnimationClip | null, frame: number): void {
    if (!clip) return
    this.applyIkFromClip(clip, frame)
    if (clip !== this.lastAppliedClip) {
      this.boneTrackIndices.clear()
      this.morphTrackIndices.clear()
      this.lastAppliedClip = clip
    }

    for (const [boneName, keyFrames] of clip.boneTracks.entries()) {
      if (keyFrames.length === 0) continue

      const cachedIdx = this.boneTrackIndices.get(boneName) ?? -1
      const clampedFrame = Math.max(keyFrames[0].frame, Math.min(keyFrames[keyFrames.length - 1].frame, frame))
      const idx = this.findKeyframeIndex(clampedFrame, keyFrames, cachedIdx)

      if (idx < 0) continue

      this.boneTrackIndices.set(boneName, idx)

      const frameA = keyFrames[idx]
      const frameB = keyFrames[idx + 1]

      const boneIdx = this.runtimeSkeleton.nameIndex[boneName]
      if (boneIdx === undefined) continue

      const localRot = this.runtimeSkeleton.localRotations[boneIdx]
      const localTrans = this.runtimeSkeleton.localTranslations[boneIdx]

      if (!frameB) {
        const frameRotation = frameA.rotation
        localRot.set(frameRotation)
        const localTranslation = this.convertVMDTranslationToLocal(boneIdx, frameA.translation, frameRotation)
        localTrans.set(localTranslation)
      } else {
        const frameDelta = frameB.frame - frameA.frame
        const gradient = frameDelta > 0 ? (clampedFrame - frameA.frame) / frameDelta : 0
        const interp = frameB.interpolation

        const rotT = interpolateControlPoints(interp.rotation, gradient)
        const rotation = Quat.slerpInto(frameA.rotation, frameB.rotation, rotT, _animSlerp)

        const txWeight = interpolateControlPoints(interp.translationX, gradient)
        const tyWeight = interpolateControlPoints(interp.translationY, gradient)
        const tzWeight = interpolateControlPoints(interp.translationZ, gradient)

        const interpolatedVMDTranslation = _animInterpT.setXYZ(
          frameA.translation.x + (frameB.translation.x - frameA.translation.x) * txWeight,
          frameA.translation.y + (frameB.translation.y - frameA.translation.y) * tyWeight,
          frameA.translation.z + (frameB.translation.z - frameA.translation.z) * tzWeight
        )

        const localTranslation = this.convertVMDTranslationToLocal(boneIdx, interpolatedVMDTranslation, rotation)

        localRot.set(rotation)
        localTrans.set(localTranslation)
      }
    }

    for (const [morphName, keyFrames] of clip.morphTracks.entries()) {
      if (keyFrames.length === 0) continue

      const cachedIdx = this.morphTrackIndices.get(morphName) ?? -1
      const clampedFrame = Math.max(keyFrames[0].frame, Math.min(keyFrames[keyFrames.length - 1].frame, frame))
      const idx = this.findKeyframeIndex(clampedFrame, keyFrames, cachedIdx)

      if (idx < 0) continue

      this.morphTrackIndices.set(morphName, idx)

      const frameA = keyFrames[idx]
      const frameB = keyFrames[idx + 1]

      const morphIdx = this.runtimeMorph.nameIndex[morphName]
      if (morphIdx === undefined) continue

      const weight = frameB
        ? frameA.weight +
        (frameB.weight - frameA.weight) *
        (keyFrames[idx + 1].frame > keyFrames[idx].frame
          ? (clampedFrame - keyFrames[idx].frame) / (keyFrames[idx + 1].frame - keyFrames[idx].frame)
          : 0)
        : frameA.weight

      this.runtimeMorph.weights[morphIdx] = weight
      this.morphsDirty = true // Mark as dirty when animation sets morph weights
    }
  }

  // Returns true when morphs changed (vertex buffer may need upload). `ikEnabled`
  // is the host's runtime switch (engine-wide); the clip decides which chains
  // within that. A host driving bones directly — motion capture writing FK
  // rotations every frame with no clip playing — turns it off wholesale, because
  // there is no motion present to carry the per-chain answer.
  update(deltaTime: number, ikEnabled = true): boolean {
    // Update tween time (in milliseconds)
    this.tweenTimeMs += deltaTime * 1000

    // Update all active tweens (rotations, translations, morphs)
    const tweensChangedMorphs = this.updateTweens()

    this.animationState.update(deltaTime)
    const clip = this.animationState.getCurrentClip()
    const frame = this.animationState.getCurrentFrame()
    if (clip !== null && !this.clipApplySuspended) {
      this.applyPoseFromClip(clip, frame)
    }

    // Apply morphs if tweens changed morphs or animation changed morphs
    const verticesChanged = this.morphsDirty || tweensChangedMorphs
    if (this.morphsDirty || tweensChangedMorphs) {
      this.applyMorphs()
      this.morphsDirty = false
    }

    // Compute world matrices (needed for IK solving to read bone positions)
    this.computeWorldMatrices()

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
    const hasAppend = b.appendRotate &&
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

          // finalRot = slerpResult * finalRot (rotation composition as quat mul)
          const sx = scratchQuat[1].x, sy = scratchQuat[1].y, sz = scratchQuat[1].z, sw = scratchQuat[1].w
          const nx = sw * fx + sx * fw + sy * fz - sz * fy
          const ny = sw * fy - sx * fz + sy * fw + sz * fx
          const nz = sw * fz + sx * fy - sy * fx + sz * fw
          const nw = sw * fw - sx * fx - sy * fy - sz * fz
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
        b.appendRotate && appendParentIdx !== undefined && appendParentIdx >= 0 && appendParentIdx < boneCount

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

            // finalRot = slerpResult * finalRot (quat mul)
            const sx = scratchQuat[1].x, sy = scratchQuat[1].y, sz = scratchQuat[1].z, sw = scratchQuat[1].w
            const nx = sw * fx + sx * fw + sy * fz - sz * fy
            const ny = sw * fy - sx * fz + sy * fw + sz * fx
            const nz = sw * fz + sx * fy - sy * fx + sz * fw
            const nw = sw * fw - sx * fx - sy * fy - sz * fz
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