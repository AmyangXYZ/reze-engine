// RZM (Reze Model) Format - WebGPU-native model format
// Phase 2: Geometry + Textures + Materials

const RZM_MAGIC = 0x455a4552 // "REZE" in little-endian
const RZM_VERSION = 2 // Updated version for texture support
const VERTEX_STRIDE = 8 // floats per vertex: position(3) + normal(3) + uv(2) = 8

export interface RzmTexture {
  path: string
  name: string
}

export interface RzmMaterial {
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
  edgeFlag: number
  vertexCount: number
}

export interface RzmBone {
  name: string
  parentIndex: number // -1 if no parent
  bindTranslation: [number, number, number]
  children: number[] // child bone indices (built on skeleton creation)
  // Optional PMX append/inherit transform
  appendParentIndex?: number // index of the bone to inherit from
  appendRatio?: number // 0..1
  appendRotate?: boolean
  appendMove?: boolean
}

export interface RzmSkeleton {
  bones: RzmBone[]
  inverseBindMatrices: Float32Array // One inverse-bind matrix per bone (column-major mat4, 16 floats per bone)
}

export interface RzmSkinning {
  joints: Uint16Array // length = vertexCount * 4, bone indices per vertex
  weights: Uint8Array // UNORM8, length = vertexCount * 4, sums ~ 255 per-vertex
}

// Runtime skeleton pose state (updated each frame)
export interface RzmSkeletonRuntime {
  nameIndex: Record<string, number> // Cached lookup: bone name -> bone index (built on initialization)
  localRotations: Float32Array // quat per bone (x,y,z,w) length = boneCount*4
  localTranslations: Float32Array // vec3 per bone length = boneCount*3
  worldMatrices: Float32Array // mat4 per bone length = boneCount*16
  skinMatrices: Float32Array // mat4 per bone length = boneCount*16
}

// VRM-like collision sphere attached to a bone
export interface CollisionSphere {
  boneIndex: number // Bone this sphere is attached to
  radius: number // Sphere radius in cm
  offset: [number, number, number] // Offset from bone position (in bone's local space, converted to world)
}

// Collision group - collection of spheres for body parts (chest, hips, etc.)
export interface CollisionGroup {
  name: string // Optional name for debugging
  spheres: CollisionSphere[]
}

// Spring bone physics runtime state
export interface RzmSpringPhysics {
  chains: SpringBoneChain[] // Spring bone chain definitions
  collisionGroups: CollisionGroup[] // Body collision groups for spring bone collision
  currentPositions?: Float32Array // vec3 per bone in chains (current position)
  prevPositions?: Float32Array // vec3 per bone in chains (previous position)
  initialized: boolean // Whether spring positions have been initialized
  timeAccumulator: number // Accumulate time between frames for fixed timestep physics
}

// VRM-like spring bone chain
export interface SpringBoneChain {
  // Root bone (anchor point - this bone is not affected by physics)
  rootBoneIndex: number
  // Chain of bones to apply spring physics to (ordered from root to tip)
  boneIndices: number[] // Array of bone indices in the chain
  // Spring parameters (applied to each bone in the chain)
  stiffness: number // Spring stiffness (higher = stiffer, typical range: 0.01-1.0)
  dragForce: number // Damping factor (higher = less oscillation, typical range: 0.1-0.9)
  hitRadius: number // Sphere collision radius for this chain (in cm, typically 0.5-5.0)
  // Optional center point for rotation (relative to root bone)
  center?: [number, number, number]
}

// Rotation tween state per bone
interface RotationTweenState {
  active: Uint8Array // 0/1 per bone
  startQuat: Float32Array // quat per bone (x,y,z,w)
  targetQuat: Float32Array // quat per bone (x,y,z,w)
  startTimeMs: Float32Array // one float per bone (ms)
  durationMs: Float32Array // one float per bone (ms)
}

import { Mat4, Vec3, Quat } from "./math"

export class RzmModel {
  private vertexData: Float32Array<ArrayBuffer>
  private vertexCount: number
  private indexData: Uint32Array<ArrayBuffer>
  private indexCount: number
  private textures: RzmTexture[] = []
  private materials: RzmMaterial[] = []
  // Static skeleton/skinning (not necessarily serialized yet)
  private skeleton: RzmSkeleton
  private skinning: RzmSkinning

  // Runtime skeleton pose state (updated each frame)
  private runtimeSkeleton: RzmSkeletonRuntime

  // Spring bone physics runtime state
  private springPhysics: RzmSpringPhysics

  constructor(
    vertexData: Float32Array<ArrayBuffer>,
    indexData: Uint32Array<ArrayBuffer>,
    textures: RzmTexture[],
    materials: RzmMaterial[],
    skeleton: RzmSkeleton,
    skinning: RzmSkinning
  ) {
    this.vertexData = vertexData
    this.vertexCount = vertexData.length / VERTEX_STRIDE
    this.indexData = indexData
    this.indexCount = indexData.length
    this.textures = textures
    this.materials = materials
    this.skeleton = skeleton
    this.skinning = skinning

    // Initialize runtime skeleton pose state
    const boneCount = skeleton.bones.length
    this.runtimeSkeleton = {
      localRotations: new Float32Array(boneCount * 4),
      localTranslations: new Float32Array(boneCount * 3),
      worldMatrices: new Float32Array(boneCount * 16),
      skinMatrices: new Float32Array(boneCount * 16),
      nameIndex: {}, // Will be populated by buildBoneLookups()
    }

    // Initialize spring physics state
    this.springPhysics = {
      chains: [],
      collisionGroups: [],
      initialized: false,
      timeAccumulator: 0,
    }

    if (this.skeleton.bones.length > 0) {
      this.buildBoneLookups()
      this.initializeRuntimePose()
      this.initializeRotTweenBuffers()
      this.autoDetectSpringBones()
      this.setupChestSpringBones()
    }
  }

  // Build caches for O(1) bone lookups and populate children arrays
  // Called during model initialization
  private buildBoneLookups(): void {
    const nameToIndex: Record<string, number> = {}

    // Initialize children arrays for all bones and build name index
    for (let i = 0; i < this.skeleton.bones.length; i++) {
      this.skeleton.bones[i].children = []
      nameToIndex[this.skeleton.bones[i].name] = i
    }

    // Build parent->children relationships
    for (let i = 0; i < this.skeleton.bones.length; i++) {
      const bone = this.skeleton.bones[i]
      const parentIdx = bone.parentIndex
      if (parentIdx >= 0 && parentIdx < this.skeleton.bones.length) {
        this.skeleton.bones[parentIdx].children.push(i)
      }
    }

    this.runtimeSkeleton.nameIndex = nameToIndex
  }

  // Rotation tween state (runtime only, initialized during construction)
  private rotTweenState!: RotationTweenState

  private initializeRotTweenBuffers(): void {
    const n = this.skeleton.bones.length
    this.rotTweenState = {
      active: new Uint8Array(n),
      startQuat: new Float32Array(n * 4),
      targetQuat: new Float32Array(n * 4),
      startTimeMs: new Float32Array(n),
      durationMs: new Float32Array(n),
    }
  }

  private static easeInOut(t: number): number {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
  }

  private static slerp(
    aX: number,
    aY: number,
    aZ: number,
    aW: number,
    bX: number,
    bY: number,
    bZ: number,
    bW: number,
    t: number
  ): [number, number, number, number] {
    // Robust basic slerp with shortest path
    let cos = aX * bX + aY * bY + aZ * bZ + aW * bW
    let bx = bX,
      by = bY,
      bz = bZ,
      bw = bW
    if (cos < 0) {
      cos = -cos
      bx = -bx
      by = -by
      bz = -bz
      bw = -bw
    }
    if (cos > 0.9995) {
      // Linear fallback
      const x = aX + t * (bx - aX)
      const y = aY + t * (by - aY)
      const z = aZ + t * (bz - aZ)
      const w = aW + t * (bw - aW)
      const invLen = 1 / Math.hypot(x, y, z, w)
      return [x * invLen, y * invLen, z * invLen, w * invLen]
    }
    const theta0 = Math.acos(cos)
    const sinTheta0 = Math.sin(theta0)
    const theta = theta0 * t
    const s0 = Math.sin(theta0 - theta) / sinTheta0
    const s1 = Math.sin(theta) / sinTheta0
    const x = s0 * aX + s1 * bx
    const y = s0 * aY + s1 * by
    const z = s0 * aZ + s1 * bz
    const w = s0 * aW + s1 * bw
    return [x, y, z, w]
  }

  // Extract unit quaternion (x,y,z,w) from a column-major rotation matrix (upper-left 3x3 of mat4)
  private static mat3ToQuat(m: Float32Array): [number, number, number, number] {
    const m00 = m[0],
      m01 = m[4],
      m02 = m[8]
    const m10 = m[1],
      m11 = m[5],
      m12 = m[9]
    const m20 = m[2],
      m21 = m[6],
      m22 = m[10]
    const trace = m00 + m11 + m22
    let x = 0,
      y = 0,
      z = 0,
      w = 1
    if (trace > 0) {
      const s = Math.sqrt(trace + 1.0) * 2 // s = 4w
      w = 0.25 * s
      x = (m21 - m12) / s
      y = (m02 - m20) / s
      z = (m10 - m01) / s
    } else if (m00 > m11 && m00 > m22) {
      const s = Math.sqrt(1.0 + m00 - m11 - m22) * 2 // s = 4x
      w = (m21 - m12) / s
      x = 0.25 * s
      y = (m01 + m10) / s
      z = (m02 + m20) / s
    } else if (m11 > m22) {
      const s = Math.sqrt(1.0 + m11 - m00 - m22) * 2 // s = 4y
      w = (m02 - m20) / s
      x = (m01 + m10) / s
      y = 0.25 * s
      z = (m12 + m21) / s
    } else {
      const s = Math.sqrt(1.0 + m22 - m00 - m11) * 2 // s = 4z
      w = (m10 - m01) / s
      x = (m02 + m20) / s
      y = (m12 + m21) / s
      z = 0.25 * s
    }
    // Normalize to be safe
    const invLen = 1 / Math.hypot(x, y, z, w)
    return [x * invLen, y * invLen, z * invLen, w * invLen]
  }

  private updateRotationTweens(): void {
    const state = this.rotTweenState
    const now = performance.now()
    const n = this.skeleton.bones.length

    for (let i = 0; i < n; i++) {
      if (state.active[i] !== 1) continue
      const startMs = state.startTimeMs[i]
      const durMs = Math.max(1, state.durationMs[i])
      const t = Math.max(0, Math.min(1, (now - startMs) / durMs))
      const e = RzmModel.easeInOut(t)
      const qi = i * 4
      const a0 = state.startQuat[qi]
      const a1 = state.startQuat[qi + 1]
      const a2 = state.startQuat[qi + 2]
      const a3 = state.startQuat[qi + 3]
      const b0 = state.targetQuat[qi]
      const b1 = state.targetQuat[qi + 1]
      const b2 = state.targetQuat[qi + 2]
      const b3 = state.targetQuat[qi + 3]
      const [x, y, z, w] = RzmModel.slerp(a0, a1, a2, a3, b0, b1, b2, b3, e)
      this.runtimeSkeleton.localRotations[qi] = x
      this.runtimeSkeleton.localRotations[qi + 1] = y
      this.runtimeSkeleton.localRotations[qi + 2] = z
      this.runtimeSkeleton.localRotations[qi + 3] = w
      if (t >= 1) {
        state.active[i] = 0
      }
    }
  }

  // Load RZM model from URL
  static async load(url: string): Promise<RzmModel> {
    const response = await fetch(url)
    const buffer = await response.arrayBuffer()
    return RzmModel.parse(buffer)
  }

  // Parse RZM model from ArrayBuffer
  static parse(buffer: ArrayBuffer): RzmModel {
    const view = new DataView(buffer)

    // Read header
    const magic = view.getUint32(0, true)
    const version = view.getUint32(4, true)
    const vertexCount = view.getUint32(12, true)
    const indexCount = view.getUint32(16, true)

    // Validate
    if (magic !== RZM_MAGIC) {
      throw new Error(`Invalid RZM file: magic mismatch`)
    }
    if (version !== RZM_VERSION) {
      throw new Error(`Unsupported RZM version: ${version}`)
    }

    // Read vertex data (starts after 64-byte header)
    const vertexDataLength = vertexCount * VERTEX_STRIDE
    const sourceVertexData = new Float32Array(buffer, 64, vertexDataLength)
    const vertexData = new Float32Array(sourceVertexData) // Copy to new array

    // Read index data (starts after vertex data)
    let indexData: Uint32Array<ArrayBuffer>
    if (indexCount > 0) {
      const indexOffset = 64 + vertexDataLength * 4 // 4 bytes per float
      const sourceIndexData = new Uint32Array(buffer, indexOffset, indexCount)
      indexData = new Uint32Array(sourceIndexData) // Copy to new array
    } else {
      indexData = new Uint32Array(0)
    }

    // Create minimal skeleton and skinning for static model
    const skeleton: RzmSkeleton = {
      bones: [],
      inverseBindMatrices: new Float32Array(0),
    }
    const skinning: RzmSkinning = {
      joints: new Uint16Array(vertexCount * 4),
      weights: new Uint8Array(vertexCount * 4),
    }
    // Initialize weights to single bone (index 0, weight 255) per vertex
    for (let i = 0; i < vertexCount; i++) {
      skinning.joints[i * 4] = 0
      skinning.weights[i * 4] = 255
    }
    return new RzmModel(vertexData, indexData, [], [], skeleton, skinning)
  }

  // Get interleaved vertex data for GPU upload
  // Format: [x,y,z, nx,ny,nz, u,v, x,y,z, nx,ny,nz, u,v, ...]
  getVertices(): Float32Array<ArrayBuffer> {
    return this.vertexData
  }

  // Get texture information
  getTextures(): RzmTexture[] {
    return this.textures
  }

  // Get material information
  getMaterials(): RzmMaterial[] {
    return this.materials
  }

  // Get diffuse texture path for a material
  getDiffuseTexturePath(materialIndex: number): string | null {
    if (materialIndex < 0 || materialIndex >= this.materials.length) return null
    const material = this.materials[materialIndex]
    if (material.diffuseTextureIndex < 0 || material.diffuseTextureIndex >= this.textures.length) return null
    return this.textures[material.diffuseTextureIndex].path
  }

  // Get vertex count
  getVertexCount(): number {
    return this.vertexCount
  }

  // Get index data for GPU upload
  getIndices(): Uint32Array<ArrayBuffer> {
    return this.indexData
  }

  // Get index count
  getIndexCount(): number {
    return this.indexCount
  }

  // Create RZM model from position-only data
  // Generates dummy normals and UVs
  static fromPositions(positions: Float32Array): RzmModel {
    const vertexCount = positions.length / 3
    const vertexData = new Float32Array(vertexCount * VERTEX_STRIDE)

    for (let i = 0; i < vertexCount; i++) {
      const posIdx = i * 3
      const vertIdx = i * VERTEX_STRIDE

      // Position
      vertexData[vertIdx + 0] = positions[posIdx + 0]
      vertexData[vertIdx + 1] = positions[posIdx + 1]
      vertexData[vertIdx + 2] = positions[posIdx + 2]

      // Normal (dummy, pointing up)
      vertexData[vertIdx + 3] = 0
      vertexData[vertIdx + 4] = 1
      vertexData[vertIdx + 5] = 0

      // UV (dummy)
      vertexData[vertIdx + 6] = 0
      vertexData[vertIdx + 7] = 0
    }

    // Create minimal skeleton and skinning for static model
    const skeleton: RzmSkeleton = {
      bones: [],
      inverseBindMatrices: new Float32Array(0),
    }
    const skinning: RzmSkinning = {
      joints: new Uint16Array(vertexCount * 4),
      weights: new Uint8Array(vertexCount * 4),
    }
    // Initialize weights to single bone (index 0, weight 255) per vertex
    for (let i = 0; i < vertexCount; i++) {
      skinning.joints[i * 4] = 0
      skinning.weights[i * 4] = 255
    }
    // Create sequential indices (0, 1, 2, ...)
    const indexData = new Uint32Array(vertexCount)
    for (let i = 0; i < vertexCount; i++) {
      indexData[i] = i
    }
    return new RzmModel(vertexData, indexData, [], [], skeleton, skinning)
  }

  // Accessors for skeleton/skinning
  getSkeleton(): RzmSkeleton {
    return this.skeleton
  }

  getSkinning(): RzmSkinning {
    return this.skinning
  }

  getSkinMatrices(): Float32Array {
    return this.runtimeSkeleton.skinMatrices
  }

  // Initialize runtime pose buffers (called once during construction)
  private initializeRuntimePose(): void {
    // Initialize rotations to identity (0,0,0,1). Translations default to zero
    const boneCount = this.skeleton.bones.length
    for (let i = 0; i < boneCount; i++) {
      const qi = i * 4
      if (this.runtimeSkeleton.localRotations[qi + 3] === 0) {
        this.runtimeSkeleton.localRotations[qi] = 0
        this.runtimeSkeleton.localRotations[qi + 1] = 0
        this.runtimeSkeleton.localRotations[qi + 2] = 0
        this.runtimeSkeleton.localRotations[qi + 3] = 1
      }
    }
  }

  // ------- Bone helpers (public API) -------
  getBoneCount(): number {
    return this.skeleton.bones.length
  }

  getBoneNames(): string[] {
    return this.skeleton.bones.map((b) => b.name)
  }

  getBoneIndexByName(name: string): number {
    return this.runtimeSkeleton.nameIndex[name] ?? -1
  }

  getBoneName(index: number): string | undefined {
    if (index < 0 || index >= this.skeleton.bones.length) return undefined
    return this.skeleton.bones[index].name
  }

  getBoneLocal(index: number): { rotation: Quat; translation: Vec3 } | undefined {
    if (index < 0 || index >= this.skeleton.bones.length) return undefined
    const qi = index * 4
    const ti = index * 3
    return {
      rotation: new Quat(
        this.runtimeSkeleton.localRotations[qi],
        this.runtimeSkeleton.localRotations[qi + 1],
        this.runtimeSkeleton.localRotations[qi + 2],
        this.runtimeSkeleton.localRotations[qi + 3]
      ),
      translation: new Vec3(
        this.runtimeSkeleton.localTranslations[ti],
        this.runtimeSkeleton.localTranslations[ti + 1],
        this.runtimeSkeleton.localTranslations[ti + 2]
      ),
    }
  }

  // Batched rotation with optional interpolation and retargeting.
  rotateBones(indicesOrNames: Array<number | string>, quats: Quat[], durationMs?: number): void {
    const state = this.rotTweenState
    quats = quats.map((q) => q.normalize())

    // Resolve indices
    const indices: number[] = new Array(indicesOrNames.length)
    for (let i = 0; i < indicesOrNames.length; i++) {
      const v = indicesOrNames[i]
      const idx = typeof v === "number" ? v : this.getBoneIndexByName(v)
      if (idx < 0 || idx >= this.skeleton.bones.length) continue
      indices[i] = idx
    }

    const now = performance.now()
    const dur = durationMs && durationMs > 0 ? durationMs : 0

    for (let i = 0; i < indices.length; i++) {
      const bi = indices[i]
      if (bi === undefined) continue
      const qi = bi * 4
      const [tx, ty, tz, tw] = quats[i].toArray()

      if (dur === 0) {
        // Immediate set, cancel any tween
        this.runtimeSkeleton.localRotations[qi] = tx
        this.runtimeSkeleton.localRotations[qi + 1] = ty
        this.runtimeSkeleton.localRotations[qi + 2] = tz
        this.runtimeSkeleton.localRotations[qi + 3] = tw
        state.active[bi] = 0
        continue
      }

      // Retarget: if active, compute current pose as new start; else start from current local
      let sx = this.runtimeSkeleton.localRotations[qi]
      let sy = this.runtimeSkeleton.localRotations[qi + 1]
      let sz = this.runtimeSkeleton.localRotations[qi + 2]
      let sw = this.runtimeSkeleton.localRotations[qi + 3]

      if (state.active[bi] === 1) {
        const startMs = state.startTimeMs[bi]
        const prevDur = Math.max(1, state.durationMs[bi])
        const t = Math.max(0, Math.min(1, (now - startMs) / prevDur))
        const e = RzmModel.easeInOut(t)
        const a0 = state.startQuat[qi]
        const a1 = state.startQuat[qi + 1]
        const a2 = state.startQuat[qi + 2]
        const a3 = state.startQuat[qi + 3]
        const b0 = state.targetQuat[qi]
        const b1 = state.targetQuat[qi + 1]
        const b2 = state.targetQuat[qi + 2]
        const b3 = state.targetQuat[qi + 3]
        const cur = RzmModel.slerp(a0, a1, a2, a3, b0, b1, b2, b3, e)
        sx = cur[0]
        sy = cur[1]
        sz = cur[2]
        sw = cur[3]
      }

      // Write start and target, mark active
      state.startQuat[qi] = sx
      state.startQuat[qi + 1] = sy
      state.startQuat[qi + 2] = sz
      state.startQuat[qi + 3] = sw
      state.targetQuat[qi] = tx
      state.targetQuat[qi + 1] = ty
      state.targetQuat[qi + 2] = tz
      state.targetQuat[qi + 3] = tw
      state.startTimeMs[bi] = now
      state.durationMs[bi] = dur
      state.active[bi] = 1
    }
  }

  setBoneRotation(indexOrName: number | string, quat: Quat): void {
    const index = typeof indexOrName === "number" ? indexOrName : this.getBoneIndexByName(indexOrName)
    if (index < 0 || index >= this.skeleton.bones.length) return
    const qi = index * 4
    this.runtimeSkeleton.localRotations[qi] = quat.x
    this.runtimeSkeleton.localRotations[qi + 1] = quat.y
    this.runtimeSkeleton.localRotations[qi + 2] = quat.z
    this.runtimeSkeleton.localRotations[qi + 3] = quat.w
  }

  setBoneTranslation(indexOrName: number | string, t: Vec3): void {
    const index = typeof indexOrName === "number" ? indexOrName : this.getBoneIndexByName(indexOrName)
    if (index < 0 || index >= this.skeleton.bones.length) return
    const ti = index * 3
    this.runtimeSkeleton.localTranslations[ti] = t.x
    this.runtimeSkeleton.localTranslations[ti + 1] = t.y
    this.runtimeSkeleton.localTranslations[ti + 2] = t.z
  }

  resetBone(index: number): void {
    const qi = index * 4
    const ti = index * 3
    this.runtimeSkeleton.localRotations[qi] = 0
    this.runtimeSkeleton.localRotations[qi + 1] = 0
    this.runtimeSkeleton.localRotations[qi + 2] = 0
    this.runtimeSkeleton.localRotations[qi + 3] = 1
    this.runtimeSkeleton.localTranslations[ti] = 0
    this.runtimeSkeleton.localTranslations[ti + 1] = 0
    this.runtimeSkeleton.localTranslations[ti + 2] = 0
  }

  resetAllBones(): void {
    const count = this.getBoneCount()
    for (let i = 0; i < count; i++) this.resetBone(i)
  }

  // --- Spring Bone Physics (VRM-like chains) ---

  // Add a spring bone chain
  addSpringBoneChain(chain: SpringBoneChain): void {
    // Validate bone indices
    const boneCount = this.skeleton.bones.length
    if (chain.rootBoneIndex < 0 || chain.rootBoneIndex >= boneCount) {
      console.warn(`[RZM] Invalid spring chain root bone index: ${chain.rootBoneIndex}`)
      return
    }
    for (const boneIdx of chain.boneIndices) {
      if (boneIdx < 0 || boneIdx >= boneCount) {
        console.warn(`[RZM] Invalid spring chain bone index: ${boneIdx}`)
        return
      }
    }
    if (chain.boneIndices.length === 0) {
      console.warn(`[RZM] Spring chain must have at least one bone`)
      return
    }
    this.springPhysics.chains.push(chain)
    this.springPhysics.initialized = false // Will reinitialize on next update
  }

  // Get all spring chains
  getSpringChains(): SpringBoneChain[] {
    return [...this.springPhysics.chains]
  }

  // Clear all spring chains
  clearSpringBones(): void {
    this.springPhysics.chains = []
    this.springPhysics.currentPositions = undefined
    this.springPhysics.prevPositions = undefined
    this.springPhysics.initialized = false
  }

  // Add a collision group (body colliders)
  addCollisionGroup(group: CollisionGroup): void {
    // Validate bone indices
    for (const sphere of group.spheres) {
      if (sphere.boneIndex < 0 || sphere.boneIndex >= this.skeleton.bones.length) {
        console.warn(`[RZM] Invalid collision sphere bone index: ${sphere.boneIndex}`)
        return
      }
      if (sphere.radius <= 0) {
        console.warn(`[RZM] Invalid collision sphere radius: ${sphere.radius}`)
        return
      }
    }
    this.springPhysics.collisionGroups.push(group)
  }

  // Get all collision groups
  getCollisionGroups(): CollisionGroup[] {
    return [...this.springPhysics.collisionGroups]
  }

  // Clear all collision groups
  clearCollisionGroups(): void {
    this.springPhysics.collisionGroups = []
  }

  // Set up collision groups from PMX rigidbodies
  // Converts PMX rigidbody data into collision spheres for spring bone collision
  setupCollisionGroupsFromRigidbodies(
    rigidbodies: Array<{
      boneIndex: number
      radius: number
      group: number
      collisionMask: number
      position: [number, number, number]
    }>
  ): void {
    if (rigidbodies.length === 0) return

    // Group rigidbodies by their group number for better organization
    const groupsByCollisionGroup = new Map<
      number,
      Array<{ boneIndex: number; radius: number; position: [number, number, number] }>
    >()

    for (const rb of rigidbodies) {
      // Only use rigidbodies with valid bone indices
      if (rb.boneIndex < 0 || rb.boneIndex >= this.skeleton.bones.length) {
        console.warn(`[RZM] Skipping rigidbody with invalid bone index: ${rb.boneIndex}`)
        continue
      }

      if (!groupsByCollisionGroup.has(rb.group)) {
        groupsByCollisionGroup.set(rb.group, [])
      }

      groupsByCollisionGroup.get(rb.group)!.push({
        boneIndex: rb.boneIndex,
        radius: rb.radius,
        position: rb.position,
      })
    }

    // Create collision groups from PMX rigidbody data
    for (const [groupNum, spheres] of groupsByCollisionGroup.entries()) {
      const collisionSpheres: CollisionSphere[] = spheres.map((s) => ({
        boneIndex: s.boneIndex,
        radius: s.radius,
        offset: s.position, // PMX position is already in bone's local space
      }))

      this.addCollisionGroup({
        name: `pmx_group_${groupNum}`,
        spheres: collisionSpheres,
      })
    }

    if (this.springPhysics.collisionGroups.length > 0) {
      const groupNames = this.springPhysics.collisionGroups.map((g) => g.name).join(", ")
      const totalSpheres = this.springPhysics.collisionGroups.reduce((sum, g) => sum + g.spheres.length, 0)
      console.log(`[RZM] Collision groups from PMX rigidbodies: ${groupNames} (${totalSpheres} spheres total)`)
    } else {
      console.warn(`[RZM] No collision groups created from ${rigidbodies.length} rigidbodies`)
    }
  }

  // Automatically detect and group bones by pattern matching
  // Uses the first number in the name as the varying index and the rest as a pattern key
  // Creates spring bone chains for each group (e.g., hf_1_1, hf_2_1, hf_3_1 form one chain)
  private autoDetectSpringBones(): void {
    const boneNames = this.getBoneNames()
    const bones = this.skeleton.bones

    // Map: patternKey -> array of { boneIndex, firstNumber }
    // patternKey has first number replaced with {} (e.g., "hf_{}_1")
    const prefixGroups = new Map<string, Array<{ boneIndex: number; suffix: number }>>()

    // Parse each bone name: find first number and create pattern key
    for (let i = 0; i < boneNames.length; i++) {
      const boneName = boneNames[i]

      // Smart matching: use the first number as the varying index
      // Pattern examples:
      // - "hf_1_1", "hf_2_1", "hf_3_1" -> key: "hf_{}_1", index: 1/2/3 (group together)
      // - "hf_1_0", "hf_2_0" -> key: "hf_{}_0", index: 1/2 (separate group)
      // - "bda1", "bda2" -> key: "bda{}", index: 1/2

      // Find the first number in the bone name
      const firstNumberMatch = boneName.match(/(\d+)/)
      if (firstNumberMatch) {
        const firstNumber = parseInt(firstNumberMatch[1], 10)
        const firstNumberStart = firstNumberMatch.index!
        const firstNumberEnd = firstNumberStart + firstNumberMatch[1].length

        // Create pattern key by replacing first number with {}
        // This allows bones like hf_1_1, hf_2_1, hf_3_1 to share the same pattern key "hf_{}_1"
        const patternKey = boneName.slice(0, firstNumberStart) + "{}" + boneName.slice(firstNumberEnd)

        // Check if pattern key contains only ASCII characters (English, not Japanese)
        const isEnglishOnly = /^[\x00-\x7F]*$/.test(patternKey)

        // Only group if first number is valid, pattern key is English-only, and bone is not static
        if (!isNaN(firstNumber) && patternKey.length > 0 && isEnglishOnly && !this.isStaticBone(i)) {
          if (!prefixGroups.has(patternKey)) {
            prefixGroups.set(patternKey, [])
          }
          // Store first number as "suffix" for sorting (though it's actually the varying index)
          prefixGroups.get(patternKey)!.push({ boneIndex: i, suffix: firstNumber })
        }
      }
    }

    // Filter groups: only keep groups with at least 2 bones (need chain)
    const validGroups: Array<{ prefix: string; bones: Array<{ boneIndex: number; suffix: number }> }> = []

    for (const [patternKey, bones] of prefixGroups.entries()) {
      if (bones.length >= 2) {
        // Sort by first number (stored as "suffix" in the data structure)
        // This ensures bones are ordered correctly in the chain (e.g., hf_1_1 -> hf_2_1 -> hf_3_1)
        bones.sort((a, b) => a.suffix - b.suffix)
        validGroups.push({ prefix: patternKey, bones })
      }
    }

    if (validGroups.length === 0) {
      return
    }

    // Log all group pattern keys in one line
    const patternKeys = validGroups.map((g) => g.prefix).join(", ")
    console.log(`[RZM] Spring bone groups: ${patternKeys}`)

    // Create spring bone chains for each group
    for (const group of validGroups) {
      const boneIndices = group.bones.map((b) => b.boneIndex)

      // Find root bone: parent of first bone in chain, or use first bone itself if no parent
      const firstBoneIndex = boneIndices[0]
      let rootBoneIndex = firstBoneIndex

      if (firstBoneIndex < bones.length) {
        const firstBone = bones[firstBoneIndex]
        if (firstBone.parentIndex >= 0 && firstBone.parentIndex < bones.length) {
          rootBoneIndex = firstBone.parentIndex
        }
      }

      // Add spring chain with VRM-like parameters
      this.addSpringBoneChain({
        rootBoneIndex,
        boneIndices,
        stiffness: 0.5,
        dragForce: 0.5,
        hitRadius: 0.3, // Increased for better collision detection
      })
    }
  }

  // Bones that should never be affected by spring physics (static/kinematic bones)
  private static readonly STATIC_BONE_NAMES = ["全ての親", "センター", "グルーブ", "腰", "上半身", "上半身2"]

  // Check if a bone should be excluded from spring physics
  private isStaticBone(boneIndex: number): boolean {
    if (boneIndex < 0 || boneIndex >= this.skeleton.bones.length) return false
    const boneName = this.getBoneName(boneIndex)
    return boneName ? RzmModel.STATIC_BONE_NAMES.includes(boneName) : false
  }

  // Set up chest spring bone chains manually
  // Chest bones hierarchy: 上2 (root) -> 変形 -> 胸 -> 胸先
  private setupChestSpringBones(): void {
    const chestBoneNames = {
      left: {
        top2: "左胸上2",
        transform: "左胸変形",
        chest: "左胸",
        tip: "左胸先",
      },
      right: {
        top2: "右胸上2",
        transform: "右胸変形",
        chest: "右胸",
        tip: "右胸先",
      },
    }

    // Helper to get bone index, returning -1 if not found
    const getIdx = (name: string): number => {
      const idx = this.getBoneIndexByName(name)
      if (idx < 0) {
        console.warn(`[RZM] Chest bone not found: ${name}`)
      }
      return idx
    }

    // Helper to build chain from tip to root by following parent chain
    const buildChain = (tipBoneIdx: number, rootBoneName: string): number[] => {
      if (tipBoneIdx < 0) return []
      const chain: number[] = []
      let currentIdx = tipBoneIdx

      // Follow parent chain until we reach the root bone
      while (currentIdx >= 0) {
        chain.unshift(currentIdx) // Add to beginning to maintain root->tip order
        const bone = this.skeleton.bones[currentIdx]
        if (bone.name === rootBoneName) {
          break // Reached the root
        }
        if (bone.parentIndex < 0 || bone.parentIndex >= this.skeleton.bones.length) {
          break // No parent, stop
        }
        currentIdx = bone.parentIndex
      }

      return chain
    }

    // Build left chest chain: tip -> root (will be reversed to root->tip)
    const leftTipIdx = getIdx(chestBoneNames.left.tip)
    const leftChain = buildChain(leftTipIdx, chestBoneNames.left.top2)
    if (leftChain.length >= 2) {
      const rootIdx = this.getBoneIndexByName(chestBoneNames.left.top2)
      if (rootIdx < 0) {
        console.warn(`[RZM] Left chest root bone not found: ${chestBoneNames.left.top2}`)
      } else {
        // Use 上2 as the anchor (root), exclude it from boneIndices
        // Only the children bones (変形, 胸, 胸先) are affected by physics
        // Also exclude any static bones to prevent dragging parent chain
        const chainBones = leftChain.filter((idx) => idx !== rootIdx && !this.isStaticBone(idx))
        if (chainBones.length > 0) {
          this.addSpringBoneChain({
            rootBoneIndex: rootIdx, // 上2 is the anchor
            boneIndices: chainBones, // Only children bones affected by physics
            stiffness: 0.5,
            dragForce: 0.5,
            hitRadius: 1.5, // Increased for better collision detection with body
          })
          console.log(
            `[RZM] Left chest chain (root=${this.getBoneName(rootIdx)}): ${chainBones
              .map((i) => this.getBoneName(i))
              .join(" -> ")}`
          )
        }
      }
    }

    // Build right chest chain
    const rightTipIdx = getIdx(chestBoneNames.right.tip)
    const rightChain = buildChain(rightTipIdx, chestBoneNames.right.top2)
    if (rightChain.length >= 2) {
      const rootIdx = this.getBoneIndexByName(chestBoneNames.right.top2)
      if (rootIdx < 0) {
        console.warn(`[RZM] Right chest root bone not found: ${chestBoneNames.right.top2}`)
      } else {
        // Use 上2 as the anchor (root), exclude it from boneIndices
        // Only the children bones (変形, 胸, 胸先) are affected by physics
        // Also exclude any static bones to prevent dragging parent chain
        const chainBones = rightChain.filter((idx) => idx !== rootIdx && !this.isStaticBone(idx))
        if (chainBones.length > 0) {
          this.addSpringBoneChain({
            rootBoneIndex: rootIdx, // 上2 is the anchor
            boneIndices: chainBones, // Only children bones affected by physics
            stiffness: 0.5,
            dragForce: 0.5,
            hitRadius: 0.3, // Increased for better collision detection with body
          })
          console.log(
            `[RZM] Right chest chain (root=${this.getBoneName(rootIdx)}): ${chainBones
              .map((i) => this.getBoneName(i))
              .join(" -> ")}`
          )
        }
      }
    }
  }

  // Helper: get bone's bind translation as Vec3
  private getBindTranslation(boneIndex: number): Vec3 {
    const bone = this.skeleton!.bones[boneIndex]
    return new Vec3(bone.bindTranslation[0], bone.bindTranslation[1], bone.bindTranslation[2])
  }

  // Helper: extract world position from bone matrix
  private getBoneWorldPosition(boneIndex: number): Vec3 {
    const matIdx = boneIndex * 16
    return new Vec3(
      this.runtimeSkeleton.worldMatrices![matIdx + 12],
      this.runtimeSkeleton.worldMatrices![matIdx + 13],
      this.runtimeSkeleton.worldMatrices![matIdx + 14]
    )
  }

  // Helper: get parent's world transform (position and rotation)
  private getParentWorldTransform(parentBoneIdx: number): { pos: Vec3; quat: Quat } {
    const parentMatIdx = parentBoneIdx * 16
    const parentM = new Mat4(
      new Float32Array(
        this.runtimeSkeleton.worldMatrices!.buffer,
        this.runtimeSkeleton.worldMatrices!.byteOffset + parentMatIdx * 4,
        16
      )
    )
    const pos = new Vec3(
      this.runtimeSkeleton.worldMatrices![parentMatIdx + 12],
      this.runtimeSkeleton.worldMatrices![parentMatIdx + 13],
      this.runtimeSkeleton.worldMatrices![parentMatIdx + 14]
    )
    const [px, py, pz, pw] = RzmModel.mat3ToQuat(parentM.values)
    const quat = new Quat(px, py, pz, pw).normalize()
    return { pos, quat }
  }

  // Helper: compute anchor position for spring bone
  private computeAnchorPosition(springBoneIdx: number): { anchor: Vec3; parentQuat: Quat | null } {
    const springBone = this.skeleton!.bones[springBoneIdx]

    if (springBone.parentIndex >= 0) {
      const { pos: parentPos, quat: parentQuat } = this.getParentWorldTransform(springBone.parentIndex)
      const bindTransLocal = this.getBindTranslation(springBoneIdx)
      const bindTransWorld = parentQuat.rotate(bindTransLocal)
      return { anchor: parentPos.add(bindTransWorld), parentQuat }
    }

    // Root bone
    const bindTrans = this.getBindTranslation(springBoneIdx)
    return { anchor: bindTrans, parentQuat: null }
  }

  // Helper: compute anchor position for first bone in chain (anchored to root bone)
  private computeAnchorPositionForChainBone(
    boneIdx: number,
    rootBoneIdx: number
  ): { anchor: Vec3; parentQuat: Quat | null } {
    if (rootBoneIdx >= 0 && rootBoneIdx < this.skeleton.bones.length) {
      const { pos: rootPos, quat: rootQuat } = this.getParentWorldTransform(rootBoneIdx)
      const bindTransLocal = this.getBindTranslation(boneIdx)
      const bindTransWorld = rootQuat.rotate(bindTransLocal)
      return { anchor: rootPos.add(bindTransWorld), parentQuat: rootQuat }
    }
    // Fallback to regular anchor computation
    return this.computeAnchorPosition(boneIdx)
  }

  // Evaluate spring bone physics using proper VRM-style Verlet integration with chains
  evaluateSpringBones(deltaTime: number): void {
    if (!this.springPhysics.chains || this.springPhysics.chains.length === 0) return

    // Count total bones across all chains and create mapping
    let totalBones = 0
    const chainBoneOffsets: number[] = [] // Offset in position array for each chain
    const boneToChainMap = new Map<number, { chainIdx: number; boneIdx: number }>() // bone index -> {chain, position in chain}

    for (let chainIdx = 0; chainIdx < this.springPhysics.chains.length; chainIdx++) {
      const chain = this.springPhysics.chains[chainIdx]
      chainBoneOffsets.push(totalBones)
      for (let boneIdx = 0; boneIdx < chain.boneIndices.length; boneIdx++) {
        boneToChainMap.set(chain.boneIndices[boneIdx], { chainIdx, boneIdx })
        totalBones++
      }
    }

    if (totalBones === 0) return

    // Initialize spring positions if needed (only once)
    if (!this.springPhysics.initialized || !this.springPhysics.currentPositions || !this.springPhysics.prevPositions) {
      // Evaluate pose with identity rotations to get bind pose world matrices
      this.evaluatePose()

      // Initialize position arrays for all bones in all chains
      this.springPhysics.currentPositions = new Float32Array(totalBones * 3)
      this.springPhysics.prevPositions = new Float32Array(totalBones * 3)

      let globalBoneIdx = 0
      for (let chainIdx = 0; chainIdx < this.springPhysics.chains.length; chainIdx++) {
        const chain = this.springPhysics.chains[chainIdx]

        // Initialize positions for each bone in chain from bind pose
        for (let boneIdx = 0; boneIdx < chain.boneIndices.length; boneIdx++) {
          const boneIndex = chain.boneIndices[boneIdx]

          // Compute anchor position (same method as in evaluation loop)
          let anchorPos: Vec3
          let parentWorldQuat: Quat | null = null

          if (boneIdx === 0) {
            // First bone in chain: anchor to root bone
            if (chain.rootBoneIndex >= 0) {
              const { anchor, parentQuat } = this.computeAnchorPositionForChainBone(boneIndex, chain.rootBoneIndex)
              anchorPos = anchor
              parentWorldQuat = parentQuat
            } else {
              const { anchor, parentQuat } = this.computeAnchorPosition(boneIndex)
              anchorPos = anchor
              parentWorldQuat = parentQuat
            }
          } else {
            // Subsequent bones: anchor to previous bone's head position
            const prevBoneIndex = chain.boneIndices[boneIdx - 1]
            const prevBonePos = this.getBoneWorldPosition(prevBoneIndex)
            const prevBoneMatIdx = prevBoneIndex * 16
            const prevBoneMat = new Mat4(
              new Float32Array(
                this.runtimeSkeleton.worldMatrices.buffer,
                this.runtimeSkeleton.worldMatrices.byteOffset + prevBoneMatIdx * 4,
                16
              )
            )
            const [px, py, pz, pw] = RzmModel.mat3ToQuat(prevBoneMat.values)
            parentWorldQuat = new Quat(px, py, pz, pw).normalize()
            const bindTrans = this.getBindTranslation(boneIndex)
            const bindTransWorld = parentWorldQuat.rotate(bindTrans)
            anchorPos = prevBonePos.add(bindTransWorld)
          }

          // Initialize tail position: extend from anchor in bind direction
          const bindTrans = this.getBindTranslation(boneIndex)
          let restDir: Vec3
          if (parentWorldQuat !== null) {
            restDir = parentWorldQuat.rotate(bindTrans).normalize()
          } else {
            restDir = bindTrans.normalize()
          }
          const restLen = bindTrans.length()
          const tailPos = anchorPos.add(restDir.scale(restLen))

          const posIdx = globalBoneIdx * 3
          this.springPhysics.currentPositions[posIdx] = tailPos.x
          this.springPhysics.currentPositions[posIdx + 1] = tailPos.y
          this.springPhysics.currentPositions[posIdx + 2] = tailPos.z
          this.springPhysics.prevPositions[posIdx] = tailPos.x
          this.springPhysics.prevPositions[posIdx + 1] = tailPos.y
          this.springPhysics.prevPositions[posIdx + 2] = tailPos.z

          globalBoneIdx++
        }
      }
      this.springPhysics.initialized = true
    }

    // Helper: recompute world matrix after updating rotation
    const recomputeWorldMatrix = (boneIndex: number): Mat4 => {
      const bones = this.skeleton!.bones
      const bone = bones[boneIndex]
      const qi = boneIndex * 4
      const ti = boneIndex * 3

      const rotateM = Mat4.fromQuat(
        this.runtimeSkeleton.localRotations![qi],
        this.runtimeSkeleton.localRotations![qi + 1],
        this.runtimeSkeleton.localRotations![qi + 2],
        this.runtimeSkeleton.localRotations![qi + 3]
      )
      const translateBind = Mat4.identity().translateInPlace(
        bone.bindTranslation[0],
        bone.bindTranslation[1],
        bone.bindTranslation[2]
      )
      const translateLocal = Mat4.identity().translateInPlace(
        this.runtimeSkeleton.localTranslations![ti],
        this.runtimeSkeleton.localTranslations![ti + 1],
        this.runtimeSkeleton.localTranslations![ti + 2]
      )
      const localM = translateBind.multiply(rotateM).multiply(translateLocal)

      if (bone.parentIndex >= 0 && bone.parentIndex < bones.length) {
        const parentMatIdx = bone.parentIndex * 16
        const parentM = new Mat4(
          new Float32Array(
            this.runtimeSkeleton.worldMatrices!.buffer,
            this.runtimeSkeleton.worldMatrices!.byteOffset + parentMatIdx * 4,
            16
          )
        )
        return parentM.multiply(localM)
      }
      return localM
    }

    // Use fixed timestep substepping for truly frame-rate independent physics
    // At 60Hz: deltaTime ≈ 16.67ms, we run 1 step of 16.67ms
    // At 240Hz: deltaTime ≈ 4.17ms, we accumulate and run steps of 16.67ms every 4 frames
    // This ensures the same amount of physics simulation time per real second
    const fixedTimeStep = 1.0 / 60.0 // 60Hz physics timestep (16.67ms)
    const maxDeltaTime = 0.1 // Clamp to prevent huge jumps from frame drops

    // Clamp deltaTime to prevent huge first-frame jumps
    const clampedDeltaTime = Math.max(0, Math.min(deltaTime, maxDeltaTime))

    // Add this frame's time to accumulator
    this.springPhysics.timeAccumulator = this.springPhysics.timeAccumulator + clampedDeltaTime

    // Run physics steps until we've consumed all accumulated time
    // This ensures physics runs at a fixed 60Hz rate regardless of display refresh rate
    let stepCount = 0
    const maxSteps = 6 // Safety limit: max 6 steps per frame (100ms max)
    while (this.springPhysics.timeAccumulator >= fixedTimeStep && stepCount < maxSteps) {
      const dt = fixedTimeStep
      const dt2 = dt * dt
      this.evaluateSpringBonesStep(dt, dt2, chainBoneOffsets, recomputeWorldMatrix)
      this.springPhysics.timeAccumulator -= fixedTimeStep
      stepCount++
    }

    // Keep only fractional remainder (prevents unbounded accumulation if frames are very fast)
    if (this.springPhysics.timeAccumulator > fixedTimeStep * 2) {
      // If accumulator grows too large (e.g., from frame drops), reset it to prevent lag spikes
      this.springPhysics.timeAccumulator = fixedTimeStep
    }
  }

  private evaluateSpringBonesStep(
    dt: number,
    dt2: number,
    chainBoneOffsets: number[],
    recomputeWorldMatrix: (boneIndex: number) => Mat4
  ): void {
    if (!this.springPhysics.chains || this.springPhysics.chains.length === 0) return

    // Pre-compute collision sphere positions once (body bones don't move during spring bone physics)
    // This avoids repeated matrix calculations in the collision loop
    const collisionSpherePositions: Array<{ pos: Vec3; radius: number }> = []
    if (this.springPhysics.collisionGroups.length > 0) {
      for (const group of this.springPhysics.collisionGroups) {
        for (const sphere of group.spheres) {
          const boneMatIdx = sphere.boneIndex * 16

          // Extract rotation quaternion from world matrix
          const [qx, qy, qz, qw] = RzmModel.mat3ToQuat(
            new Float32Array(
              this.runtimeSkeleton.worldMatrices.buffer,
              this.runtimeSkeleton.worldMatrices.byteOffset + boneMatIdx * 4,
              16
            )
          )
          const rotQuat = new Quat(qx, qy, qz, qw).normalize()
          const offsetLocal = new Vec3(sphere.offset[0], sphere.offset[1], sphere.offset[2])
          const offsetWorld = rotQuat.rotate(offsetLocal)

          // Sphere position in world space = bone position + rotated offset
          const spherePos = new Vec3(
            this.runtimeSkeleton.worldMatrices[boneMatIdx + 12],
            this.runtimeSkeleton.worldMatrices[boneMatIdx + 13],
            this.runtimeSkeleton.worldMatrices[boneMatIdx + 14]
          ).add(offsetWorld)

          collisionSpherePositions.push({ pos: spherePos, radius: sphere.radius })
        }
      }
    }

    // Process each chain
    for (let chainIdx = 0; chainIdx < this.springPhysics.chains.length; chainIdx++) {
      const chain = this.springPhysics.chains[chainIdx]
      const offset = chainBoneOffsets[chainIdx]
      const gravityDir = new Vec3(0, -1, 0) // Gravity always points downward

      // Process each bone in the chain sequentially
      for (let boneIdx = 0; boneIdx < chain.boneIndices.length; boneIdx++) {
        const boneIndex = chain.boneIndices[boneIdx]
        const posIdx = (offset + boneIdx) * 3

        // Get current and previous positions
        const currentPos = new Vec3(
          this.springPhysics.currentPositions![posIdx],
          this.springPhysics.currentPositions![posIdx + 1],
          this.springPhysics.currentPositions![posIdx + 2]
        )
        const prevPos = new Vec3(
          this.springPhysics.prevPositions![posIdx],
          this.springPhysics.prevPositions![posIdx + 1],
          this.springPhysics.prevPositions![posIdx + 2]
        )

        // Compute anchor position using the same method as before
        // For chain bones: if first bone, use root bone; otherwise use previous bone in chain
        let anchorPos: Vec3
        let parentWorldQuat: Quat | null = null

        if (boneIdx === 0) {
          // First bone in chain: anchor to root bone
          if (chain.rootBoneIndex >= 0) {
            const { anchor, parentQuat } = this.computeAnchorPositionForChainBone(boneIndex, chain.rootBoneIndex)
            anchorPos = anchor
            parentWorldQuat = parentQuat
          } else {
            // Fallback to bone's own parent
            const { anchor, parentQuat } = this.computeAnchorPosition(boneIndex)
            anchorPos = anchor
            parentWorldQuat = parentQuat
          }
        } else {
          // Subsequent bones: anchor to previous bone using its world matrix position (more stable)
          const prevBoneIdx = boneIdx - 1
          const prevBoneIndex = chain.boneIndices[prevBoneIdx]

          // Get previous bone's world position and rotation from its updated world matrix
          // This uses the smoothed, stable position from the bone's world transform
          const prevBoneMatIdx = prevBoneIndex * 16
          const prevBoneWorldPos = new Vec3(
            this.runtimeSkeleton.worldMatrices![prevBoneMatIdx + 12],
            this.runtimeSkeleton.worldMatrices![prevBoneMatIdx + 13],
            this.runtimeSkeleton.worldMatrices![prevBoneMatIdx + 14]
          )

          // Get previous bone's world rotation from its matrix
          const prevBoneMat = new Mat4(
            new Float32Array(
              this.runtimeSkeleton.worldMatrices!.buffer,
              this.runtimeSkeleton.worldMatrices!.byteOffset + prevBoneMatIdx * 4,
              16
            )
          )
          const [px, py, pz, pw] = RzmModel.mat3ToQuat(prevBoneMat.values)
          parentWorldQuat = new Quat(px, py, pz, pw).normalize()

          // Compute anchor: previous bone's world position + this bone's bind translation rotated by previous bone's world rotation
          const bindTrans = this.getBindTranslation(boneIndex)
          const bindTransWorld = parentWorldQuat.rotate(bindTrans)
          anchorPos = prevBoneWorldPos.add(bindTransWorld)
        }

        // Verlet integration: compute velocity
        const velocity = currentPos.subtract(prevPos)

        // Calculate direction and distance from anchor
        const toCurrent = currentPos.subtract(anchorPos)
        const currentDist = toCurrent.length()
        const currentDir = currentDist > 0.001 ? toCurrent.normalize() : new Vec3(0, 0, 1)

        // Rest length from bind translation
        const bindTrans = this.getBindTranslation(boneIndex)
        const restLen = bindTrans.length()

        // Accumulate forces
        // Units: positions in cm, time in seconds, forces in cm/s² (acceleration)
        let force = new Vec3(0, 0, 0)

        // 1. Gravity
        // Gravity magnitude: 980.0 cm/s²
        // At 60fps (dt=1/60), gravity contributes: 980 * (1/60)² ≈ 0.27 cm/frame
        const gravity = 980.0
        force = force.add(gravityDir.scale(gravity))

        // 2. Stiffness force: restore orientation toward rest pose
        // stiffnessScale is chosen to be ~5x gravity magnitude (5000 vs 980)
        // This ensures stiffness force dominates when there's significant angular error,
        // while gravity provides constant downward pull
        // Formula: stiffnessForce = directionError * stiffness * stiffnessScale
        // where directionError is normalized (magnitude 0-2), stiffness is 0-1
        let restDir: Vec3
        if (parentWorldQuat !== null) {
          restDir = parentWorldQuat.rotate(bindTrans).normalize()
        } else {
          restDir = bindTrans.normalize()
        }

        const directionError = restDir.subtract(currentDir)
        const stiffnessScale = 5000.0
        const stiffnessForce = directionError.scale(chain.stiffness * stiffnessScale)
        force = force.add(stiffnessForce)

        // 3. Length constraint
        // Length constraint scale is ~0.5x stiffnessScale to be less aggressive
        // Prevents overstretching while allowing stiffness to handle orientation
        const distanceError = currentDist - restLen
        if (Math.abs(distanceError) > 0.01) {
          const lengthForce = currentDir.scale(-distanceError * 500.0)
          force = force.add(lengthForce)
        }

        // Verlet integration: newPos = currentPos + velocity*(1-drag) + force*dt²
        // Since forces are in cm/s² and dt² is (seconds)², force*dt² gives displacement in cm
        let newPos = currentPos.add(velocity.scale(1.0 - chain.dragForce)).add(force.scale(dt2))

        // Constraint: clamp distance from anchor
        const finalDir = newPos.subtract(anchorPos)
        const finalDist = finalDir.length()
        if (finalDist > 0.001) {
          const maxDist = restLen * 2.5
          const minDist = restLen * 0.3
          const clampedDist = Math.max(minDist, Math.min(maxDist, finalDist))
          newPos = anchorPos.add(finalDir.normalize().scale(clampedDist))
        }

        // Collision detection with other spring bone chains using hitRadius
        if (chain.hitRadius > 0 && this.springPhysics.chains.length > 1) {
          for (let otherChainIdx = 0; otherChainIdx < this.springPhysics.chains.length; otherChainIdx++) {
            const otherChain = this.springPhysics.chains[otherChainIdx]

            // Skip same chain and chains with no hitRadius
            if (otherChainIdx === chainIdx || otherChain.hitRadius <= 0) continue

            const otherOffset = chainBoneOffsets[otherChainIdx]

            // Check collision with each bone in the other chain
            for (let otherBoneIdx = 0; otherBoneIdx < otherChain.boneIndices.length; otherBoneIdx++) {
              const otherPosIdx = (otherOffset + otherBoneIdx) * 3
              if (!this.springPhysics.currentPositions || otherPosIdx + 2 >= this.springPhysics.currentPositions.length)
                continue

              const otherPos = new Vec3(
                this.springPhysics.currentPositions[otherPosIdx],
                this.springPhysics.currentPositions[otherPosIdx + 1],
                this.springPhysics.currentPositions[otherPosIdx + 2]
              )

              // Sphere-sphere collision
              const toOther = newPos.subtract(otherPos)
              const distToOther = toOther.length()
              const combinedRadius = chain.hitRadius + otherChain.hitRadius

              if (distToOther < combinedRadius && distToOther > 0.001) {
                const collisionDir = toOther.normalize()
                const pushDistance = combinedRadius - distToOther
                // Push away from the other spring bone
                newPos = newPos.add(collisionDir.scale(pushDistance))
              }
            }
          }
        }

        // Collision detection with body collision groups (VRM-style)
        // Use pre-computed sphere positions for performance
        if (chain.hitRadius > 0 && collisionSpherePositions.length > 0) {
          for (const sphereData of collisionSpherePositions) {
            // Sphere-sphere collision between spring bone and body collision sphere
            const toSphere = newPos.subtract(sphereData.pos)
            const distToSphere = toSphere.length()
            const combinedRadius = chain.hitRadius + sphereData.radius

            if (distToSphere < combinedRadius && distToSphere > 0.001) {
              const collisionDir = toSphere.normalize()
              const pushDistance = combinedRadius - distToSphere
              // Push spring bone away from body collider
              newPos = newPos.add(collisionDir.scale(pushDistance))
            }
          }
        }

        // Update Verlet state BEFORE smoothing (to maintain proper velocity)
        if (this.springPhysics.prevPositions) {
          this.springPhysics.prevPositions[posIdx] = currentPos.x
          this.springPhysics.prevPositions[posIdx + 1] = currentPos.y
          this.springPhysics.prevPositions[posIdx + 2] = currentPos.z
        }

        // Exponential smoothing to reduce high-frequency jitter (applied after Verlet update)
        // smoothingFactor is independent of physics - it's a post-processing filter
        // 0.5 means 50% new position, 50% old position (balanced responsiveness vs stability)
        // Higher values (0.7+) = more responsive but potentially jittery
        // Lower values (0.3-) = more stable but potentially laggy
        const smoothingFactor = 0.5
        const smoothedPos = currentPos.scale(1.0 - smoothingFactor).add(newPos.scale(smoothingFactor))
        newPos = smoothedPos

        if (this.springPhysics.currentPositions) {
          this.springPhysics.currentPositions[posIdx] = newPos.x
          this.springPhysics.currentPositions[posIdx + 1] = newPos.y
          this.springPhysics.currentPositions[posIdx + 2] = newPos.z
        }

        // Update bone rotation to point from anchor toward newPos (like before)
        if (parentWorldQuat !== null) {
          const targetDirWorld = newPos.subtract(anchorPos).normalize()

          // Get bind direction in parent's local space
          const bindDirLocal = bindTrans.normalize()

          // Transform target direction to parent's local space
          const invParentQuat = parentWorldQuat.conjugate().normalize()
          const targetDirLocal = invParentQuat.rotate(targetDirWorld)

          // Compute rotation from bind direction to target direction
          const localRotation = Quat.fromTo(bindDirLocal, targetDirLocal)

          // Update bone's local rotation
          const boneQi = boneIndex * 4
          this.runtimeSkeleton.localRotations[boneQi] = localRotation.x
          this.runtimeSkeleton.localRotations[boneQi + 1] = localRotation.y
          this.runtimeSkeleton.localRotations[boneQi + 2] = localRotation.z
          this.runtimeSkeleton.localRotations[boneQi + 3] = localRotation.w
        }

        // Update world matrix for next bone in chain
        const boneMatIdx = boneIndex * 16
        const boneWorldM = recomputeWorldMatrix(boneIndex)
        this.runtimeSkeleton.worldMatrices.set(boneWorldM.values, boneMatIdx)
      }
    }
  }

  getBoneWorldMatrix(index: number): Float32Array | undefined {
    this.evaluatePose()
    const start = index * 16
    return this.runtimeSkeleton.worldMatrices.slice(start, start + 16)
  }

  // Evaluate world and skin matrices from local TR and bind
  // If deltaTime is provided, also updates spring bone physics
  evaluatePose(deltaTime?: number): void {
    // Advance rotation tweens before composing matrices
    this.updateRotationTweens()

    // Compute world matrices BEFORE spring bone physics (needed for collision detection)
    // This gives us up-to-date matrices for body collider bones
    this.computeWorldMatrices()

    // Evaluate spring bones (if deltaTime provided) - uses the world matrices we just computed
    if (deltaTime !== undefined) {
      this.evaluateSpringBones(deltaTime)
      // Spring bones update rotations, so recompute matrices after
      this.computeWorldMatrices()
    }

    // Compute skin matrices from world matrices
    const bones = this.skeleton.bones
    const invBind = this.skeleton.inverseBindMatrices
    const worldBuf = this.runtimeSkeleton.worldMatrices
    const skinBuf = this.runtimeSkeleton.skinMatrices

    // Compute skin matrices from world and inverse bind matrices
    for (let i = 0; i < bones.length; i++) {
      const worldSeg = worldBuf.subarray(i * 16, i * 16 + 16)
      const invSeg = invBind.subarray(i * 16, i * 16 + 16)
      const skinSeg = skinBuf.subarray(i * 16, i * 16 + 16)
      const skinM = new Mat4(worldSeg).multiply(new Mat4(invSeg))
      skinSeg.set(skinM.values)
    }
  }

  // Compute world matrices from local rotations and translations
  private computeWorldMatrices(): void {
    const bones = this.skeleton.bones
    const localRot = this.runtimeSkeleton.localRotations
    const worldBuf = this.runtimeSkeleton.worldMatrices

    // compute recursively to respect parent-before-child regardless of file order
    const computed: boolean[] = new Array(bones.length).fill(false)

    const computeWorld = (i: number): void => {
      if (computed[i]) return
      if (bones[i].parentIndex >= bones.length) {
        console.warn(`[RZM] bone ${i} parent out of range: ${bones[i].parentIndex}`)
      }
      const qi = i * 4
      // Start from bone's local rotation
      let rotateM = Mat4.fromQuat(localRot[qi], localRot[qi + 1], localRot[qi + 2], localRot[qi + 3])
      // Accumulated local translation from append move
      let addLocalTx = 0
      let addLocalTy = 0
      let addLocalTz = 0

      // Apply PMX append/inherit rotation with decoded ratio
      const b = bones[i]
      if (
        b.appendRotate &&
        b.appendParentIndex !== undefined &&
        b.appendParentIndex >= 0 &&
        b.appendParentIndex < bones.length &&
        (b.appendRatio === undefined || Math.abs(b.appendRatio) > 1e-6)
      ) {
        const ap = b.appendParentIndex
        // Use append parent's LOCAL rotation/translation with ratio, as per PMX
        const apQi = ap * 4
        const apTi = ap * 3
        let ratio = b.appendRatio === undefined ? 1 : b.appendRatio
        ratio = Math.max(-1, Math.min(1, ratio))
        // Rotation append
        if (b.appendRotate) {
          let ax = localRot[apQi]
          let ay = localRot[apQi + 1]
          let az = localRot[apQi + 2]
          const aw = localRot[apQi + 3]
          if (ratio < 0) {
            // inverse (conjugate) for negative ratios
            ax = -ax
            ay = -ay
            az = -az
            ratio = -ratio
          }
          const [rx, ry, rz, rw] = RzmModel.slerp(0, 0, 0, 1, ax, ay, az, aw, ratio)
          const ratioM = Mat4.fromQuat(rx, ry, rz, rw)
          rotateM = ratioM.multiply(rotateM)
        }
        // Move append
        if (b.appendMove && (b.appendRatio === undefined || Math.abs(b.appendRatio) > 1e-6)) {
          const apTx = this.runtimeSkeleton.localTranslations![apTi]
          const apTy = this.runtimeSkeleton.localTranslations![apTi + 1]
          const apTz = this.runtimeSkeleton.localTranslations![apTi + 2]
          addLocalTx += apTx * (b.appendRatio ?? 1)
          addLocalTy += apTy * (b.appendRatio ?? 1)
          addLocalTz += apTz * (b.appendRatio ?? 1)
        }
      }
      const translateBind = Mat4.identity().translateInPlace(
        bones[i].bindTranslation[0],
        bones[i].bindTranslation[1],
        bones[i].bindTranslation[2]
      )
      // Apply additional local translation due to appendMove
      const translateLocal = Mat4.identity().translateInPlace(addLocalTx, addLocalTy, addLocalTz)
      // Local: move from parent to joint, then rotate around joint, then local translation
      const localM = translateBind.multiply(rotateM).multiply(translateLocal)

      const worldSeg = worldBuf.subarray(i * 16, i * 16 + 16)
      let worldM: Mat4
      if (bones[i].parentIndex >= 0) {
        const p = bones[i].parentIndex
        if (!computed[p]) computeWorld(p)
        const parentSeg = worldBuf.subarray(p * 16, p * 16 + 16)
        worldM = new Mat4(parentSeg).multiply(localM)
      } else {
        worldM = localM
      }
      worldSeg.set(worldM.values)
      computed[i] = true
    }

    for (let i = 0; i < bones.length; i++) computeWorld(i)
  }

  // Export to RZM binary format
  toArrayBuffer(): ArrayBuffer {
    const headerSize = 64
    const vertexDataSize = this.vertexData.byteLength
    const indexDataSize = this.indexData ? this.indexData.byteLength : 0
    const totalSize = headerSize + vertexDataSize + indexDataSize

    const buffer = new ArrayBuffer(totalSize)
    const view = new DataView(buffer)

    // Write header
    view.setUint32(0, RZM_MAGIC, true) // magic
    view.setUint32(4, RZM_VERSION, true) // version
    view.setUint32(8, 0, true) // flags
    view.setUint32(12, this.vertexCount, true) // vertexCount
    view.setUint32(16, this.indexCount, true) // indexCount
    view.setUint32(20, this.materials.length, true) // materialCount
    view.setUint32(24, this.skeleton ? this.skeleton.bones.length : 0, true) // boneCount
    // Rest of header is padding

    // Write vertex data
    new Float32Array(buffer, 64, this.vertexData.length).set(this.vertexData)

    // Write index data if present
    if (this.indexData) {
      const indexOffset = 64 + vertexDataSize
      new Uint32Array(buffer, indexOffset, this.indexData.length).set(this.indexData)
    }

    return buffer
  }
}
