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
  // One inverse-bind matrix per bone (column-major mat4, 16 floats per bone)
  inverseBindMatrices: Float32Array
  // Cached lookup for efficient bone access (built on creation)
  nameIndex: Record<string, number> // bone name -> bone index
}

export interface RzmSkinning {
  joints: Uint16Array // length = vertexCount * 4, bone indices per vertex
  weights: Uint8Array // UNORM8, length = vertexCount * 4, sums ~ 255 per-vertex
}

export interface SpringBone {
  // Bone indices
  parentBoneIndex: number // Bone that acts as the anchor/root of the spring
  childBoneIndex: number // Bone that will be affected by spring physics
  // Spring parameters
  stiffness: number // Spring stiffness (higher = stiffer, typical range: 0.01-1.0)
  damping: number // Damping factor (higher = less oscillation, typical range: 0.1-0.9)
  restLength?: number // Rest length of spring (auto-calculated if not provided)
  // Collision/limits (optional)
  collideRadius?: number // Sphere collision radius
  // Gravity (optional)
  gravityScale?: number // Multiplier for gravity effect
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

  // Runtime (non-serialized) pose data (initialized during construction)
  private boneLocalRotations!: Float32Array // quat per bone (x,y,z,w) length = boneCount*4
  private boneLocalTranslations!: Float32Array // vec3 per bone length = boneCount*3
  private boneWorldMatrices!: Float32Array // mat4 per bone length = boneCount*16
  private boneSkinMatrices!: Float32Array // mat4 per bone length = boneCount*16

  // Spring bone physics
  private springBones: SpringBone[] = []
  // Verlet integration state: current and previous tail positions in world space
  private springCurrentTails?: Float32Array // vec3 per spring bone (current tail position)
  private springPrevTails?: Float32Array // vec3 per spring bone (previous tail position)
  private springInitialized = false

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
    if (this.skeleton.bones.length > 0) {
      this.buildBoneLookups()
      this.initializeRuntimePose()
      this.initializeRotTweenBuffers()
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

    this.skeleton.nameIndex = nameToIndex
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
      this.boneLocalRotations[qi] = x
      this.boneLocalRotations[qi + 1] = y
      this.boneLocalRotations[qi + 2] = z
      this.boneLocalRotations[qi + 3] = w
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
      nameIndex: {},
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
      nameIndex: {},
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
    return this.boneSkinMatrices
  }

  // Initialize runtime pose buffers (called once during construction)
  private initializeRuntimePose(): void {
    const boneCount = this.skeleton.bones.length
    this.boneLocalRotations = new Float32Array(boneCount * 4)
    this.boneLocalTranslations = new Float32Array(boneCount * 3)
    this.boneWorldMatrices = new Float32Array(boneCount * 16)
    this.boneSkinMatrices = new Float32Array(boneCount * 16)
    // Initialize rotations to identity (0,0,0,1). Translations default to zero
    for (let i = 0; i < boneCount; i++) {
      const qi = i * 4
      if (this.boneLocalRotations[qi + 3] === 0) {
        this.boneLocalRotations[qi] = 0
        this.boneLocalRotations[qi + 1] = 0
        this.boneLocalRotations[qi + 2] = 0
        this.boneLocalRotations[qi + 3] = 1
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
    return this.skeleton.nameIndex[name] ?? -1
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
        this.boneLocalRotations[qi],
        this.boneLocalRotations[qi + 1],
        this.boneLocalRotations[qi + 2],
        this.boneLocalRotations[qi + 3]
      ),
      translation: new Vec3(
        this.boneLocalTranslations[ti],
        this.boneLocalTranslations[ti + 1],
        this.boneLocalTranslations[ti + 2]
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
        this.boneLocalRotations[qi] = tx
        this.boneLocalRotations[qi + 1] = ty
        this.boneLocalRotations[qi + 2] = tz
        this.boneLocalRotations[qi + 3] = tw
        state.active[bi] = 0
        continue
      }

      // Retarget: if active, compute current pose as new start; else start from current local
      let sx = this.boneLocalRotations[qi]
      let sy = this.boneLocalRotations[qi + 1]
      let sz = this.boneLocalRotations[qi + 2]
      let sw = this.boneLocalRotations[qi + 3]

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
    this.boneLocalRotations[qi] = quat.x
    this.boneLocalRotations[qi + 1] = quat.y
    this.boneLocalRotations[qi + 2] = quat.z
    this.boneLocalRotations[qi + 3] = quat.w
  }

  setBoneTranslation(indexOrName: number | string, t: Vec3): void {
    const index = typeof indexOrName === "number" ? indexOrName : this.getBoneIndexByName(indexOrName)
    if (index < 0 || index >= this.skeleton.bones.length) return
    const ti = index * 3
    this.boneLocalTranslations[ti] = t.x
    this.boneLocalTranslations[ti + 1] = t.y
    this.boneLocalTranslations[ti + 2] = t.z
  }

  resetBone(index: number): void {
    const qi = index * 4
    const ti = index * 3
    this.boneLocalRotations[qi] = 0
    this.boneLocalRotations[qi + 1] = 0
    this.boneLocalRotations[qi + 2] = 0
    this.boneLocalRotations[qi + 3] = 1
    this.boneLocalTranslations[ti] = 0
    this.boneLocalTranslations[ti + 1] = 0
    this.boneLocalTranslations[ti + 2] = 0
  }

  resetAllBones(): void {
    const count = this.getBoneCount()
    for (let i = 0; i < count; i++) this.resetBone(i)
  }

  // --- Spring Bone Physics ---

  // Add a spring bone constraint
  addSpringBone(springBone: SpringBone): void {
    // Validate bone indices
    const boneCount = this.skeleton.bones.length
    if (
      springBone.parentBoneIndex < 0 ||
      springBone.parentBoneIndex >= boneCount ||
      springBone.childBoneIndex < 0 ||
      springBone.childBoneIndex >= boneCount
    ) {
      console.warn(
        `[RZM] Invalid spring bone indices: parent=${springBone.parentBoneIndex}, child=${springBone.childBoneIndex}`
      )
      return
    }
    this.springBones.push(springBone)
    this.springInitialized = false // Will reinitialize on next update
  }

  // Get all spring bones
  getSpringBones(): SpringBone[] {
    return [...this.springBones]
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
      this.boneWorldMatrices![matIdx + 12],
      this.boneWorldMatrices![matIdx + 13],
      this.boneWorldMatrices![matIdx + 14]
    )
  }

  // Helper: get parent's world transform (position and rotation)
  private getParentWorldTransform(parentBoneIdx: number): { pos: Vec3; quat: Quat } {
    const parentMatIdx = parentBoneIdx * 16
    const parentM = new Mat4(
      new Float32Array(this.boneWorldMatrices!.buffer, this.boneWorldMatrices!.byteOffset + parentMatIdx * 4, 16)
    )
    const pos = new Vec3(
      this.boneWorldMatrices![parentMatIdx + 12],
      this.boneWorldMatrices![parentMatIdx + 13],
      this.boneWorldMatrices![parentMatIdx + 14]
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

  // Evaluate spring bone physics using proper VRM-style Verlet integration
  evaluateSpringBones(deltaTime: number): void {
    if (!this.springBones || this.springBones.length === 0) return

    const springCount = this.springBones.length
    const dt = Math.max(0.0001, Math.min(deltaTime, 0.033))
    const dt2 = dt * dt

    // Initialize spring tail positions if needed
    if (!this.springInitialized || !this.springCurrentTails || !this.springPrevTails) {
      // Evaluate pose with identity rotations to get bind pose world matrices
      this.evaluatePose()

      // Now initialize spring tail positions from the bind pose
      this.springCurrentTails = new Float32Array(springCount * 3)
      this.springPrevTails = new Float32Array(springCount * 3)

      for (let i = 0; i < springCount; i++) {
        const spring = this.springBones[i]
        const springBoneIdx = spring.childBoneIndex
        const springBone = this.skeleton.bones[springBoneIdx]
        const idx = i * 3

        // Calculate rest length from bind translation
        const bindTrans = this.getBindTranslation(springBoneIdx)
        const restLen = bindTrans.length()
        spring.restLength = restLen > 0.001 ? restLen : 0.1

        // Initialize tail position from bind pose (using world matrices computed with identity rotations)
        const childBoneIdx = springBone.children.length > 0 ? springBone.children[0] : -1
        if (childBoneIdx >= 0) {
          // Use child bone's position from bind pose world matrix
          const childPos = this.getBoneWorldPosition(childBoneIdx)
          this.springCurrentTails[idx] = childPos.x
          this.springCurrentTails[idx + 1] = childPos.y
          this.springCurrentTails[idx + 2] = childPos.z
        } else {
          // No child - extend in bind direction from anchor
          const { anchor: anchorPos } = this.computeAnchorPosition(springBoneIdx)
          const bindDirWorld = bindTrans.normalize() // In bind pose, direction is just the bind translation direction
          const tailPos = anchorPos.add(bindDirWorld.scale(spring.restLength))
          this.springCurrentTails[idx] = tailPos.x
          this.springCurrentTails[idx + 1] = tailPos.y
          this.springCurrentTails[idx + 2] = tailPos.z
        }

        // Initialize previous tail to current (no initial velocity)
        this.springPrevTails[idx] = this.springCurrentTails[idx]
        this.springPrevTails[idx + 1] = this.springCurrentTails[idx + 1]
        this.springPrevTails[idx + 2] = this.springCurrentTails[idx + 2]
      }
      this.springInitialized = true
    }

    // Sort springs by dependency order
    const childToSpringMap = new Map<number, number>()
    for (let i = 0; i < springCount; i++) {
      childToSpringMap.set(this.springBones[i].childBoneIndex, i)
    }

    const sortedIndices = Array.from({ length: springCount }, (_, i) => i)
    sortedIndices.sort((a, b) => {
      const springA = this.springBones[a]
      const springB = this.springBones[b]
      const parentOfA = childToSpringMap.get(springA.parentBoneIndex)
      const parentOfB = childToSpringMap.get(springB.parentBoneIndex)
      if (parentOfA === b) return 1
      if (parentOfB === a) return -1
      return 0
    })

    // Helper: recompute world matrix after updating rotation
    const recomputeWorldMatrix = (boneIndex: number): Mat4 => {
      const bones = this.skeleton!.bones
      const bone = bones[boneIndex]
      const qi = boneIndex * 4
      const ti = boneIndex * 3

      const rotateM = Mat4.fromQuat(
        this.boneLocalRotations![qi],
        this.boneLocalRotations![qi + 1],
        this.boneLocalRotations![qi + 2],
        this.boneLocalRotations![qi + 3]
      )
      const translateBind = Mat4.identity().translateInPlace(
        bone.bindTranslation[0],
        bone.bindTranslation[1],
        bone.bindTranslation[2]
      )
      const translateLocal = Mat4.identity().translateInPlace(
        this.boneLocalTranslations![ti],
        this.boneLocalTranslations![ti + 1],
        this.boneLocalTranslations![ti + 2]
      )
      const localM = translateBind.multiply(rotateM).multiply(translateLocal)

      if (bone.parentIndex >= 0 && bone.parentIndex < bones.length) {
        const parentMatIdx = bone.parentIndex * 16
        const parentM = new Mat4(
          new Float32Array(this.boneWorldMatrices!.buffer, this.boneWorldMatrices!.byteOffset + parentMatIdx * 4, 16)
        )
        return parentM.multiply(localM)
      }
      return localM
    }

    // Process each spring bone in dependency order
    for (const i of sortedIndices) {
      const spring = this.springBones[i]
      const springBoneIdx = spring.childBoneIndex
      const idx = i * 3

      // Compute anchor position (where spring attaches to parent)
      const { anchor: anchorPos, parentQuat: parentWorldQuat } = this.computeAnchorPosition(springBoneIdx)

      // Get current and previous tail positions for Verlet integration
      const currentTail = new Vec3(
        this.springCurrentTails[idx],
        this.springCurrentTails[idx + 1],
        this.springCurrentTails[idx + 2]
      )
      const prevTail = new Vec3(this.springPrevTails[idx], this.springPrevTails[idx + 1], this.springPrevTails[idx + 2])

      // Verlet integration: compute velocity
      const velocity = currentTail.subtract(prevTail)
      const damping = spring.damping || 0.5

      // Calculate current direction from anchor to current tail (before applying forces)
      const toCurrentTail = currentTail.subtract(anchorPos)
      const currentDist = toCurrentTail.length()
      const currentDir = currentDist > 0.001 ? toCurrentTail.normalize() : new Vec3(0, 0, 1)

      // Accumulate forces (all in acceleration units: cm/s²)
      let force = new Vec3(0, 0, 0)

      // 1. Gravity (world-space down)
      if (spring.gravityScale && spring.gravityScale > 0) {
        const gravity = spring.gravityScale * 980.0 // cm/s²
        force = force.add(new Vec3(0, -1, 0).scale(gravity))
      }

      // 2. Stiffness force: restore orientation toward rest pose
      const bindTrans = this.getBindTranslation(springBoneIdx)
      let restDir: Vec3
      if (parentWorldQuat !== null) {
        restDir = parentWorldQuat.rotate(bindTrans).normalize()
      } else {
        restDir = bindTrans.normalize()
      }

      const stiffness = spring.stiffness || 0.01
      const directionError = restDir.subtract(currentDir)

      // Scale stiffness force to match gravity magnitude
      // Stiffness of 1.0 should approximately balance gravity
      const stiffnessScale = 5000.0 // Tuning factor (experiment with 3000-10000)
      const stiffnessForce = directionError.scale(stiffness * stiffnessScale)
      force = force.add(stiffnessForce)

      // 3. Length constraint (only if stretched/compressed significantly)
      const restLen = spring.restLength || 0.1
      const distanceError = currentDist - restLen
      if (Math.abs(distanceError) > 0.01) {
        // Soft constraint - restore length gently
        const lengthForce = currentDir.scale(-distanceError * 500.0) // Moderate spring constant
        force = force.add(lengthForce)
      }

      // Apply Verlet: newPos = currentPos + velocity*(1-drag) + force*dt²
      let newTail = currentTail.add(velocity.scale(1.0 - damping)).add(force.scale(dt2))

      // Constraint: clamp distance to prevent extreme stretching/compression
      const finalDir = newTail.subtract(anchorPos)
      const finalDist = finalDir.length()
      if (finalDist > 0.001) {
        const maxDist = restLen * 2.5
        const minDist = restLen * 0.3
        const clampedDist = Math.max(minDist, Math.min(maxDist, finalDist))
        newTail = anchorPos.add(finalDir.normalize().scale(clampedDist))
      }

      // Exponential smoothing to reduce high-frequency jitter
      const smoothingFactor = 0.5 // 50% new, 50% old for smooth convergence
      const smoothedTail = currentTail.scale(1.0 - smoothingFactor).add(newTail.scale(smoothingFactor))
      newTail = smoothedTail

      // Update Verlet integration state
      this.springPrevTails[idx] = currentTail.x
      this.springPrevTails[idx + 1] = currentTail.y
      this.springPrevTails[idx + 2] = currentTail.z
      this.springCurrentTails[idx] = newTail.x
      this.springCurrentTails[idx + 1] = newTail.y
      this.springCurrentTails[idx + 2] = newTail.z

      // Update bone rotation to point from anchor toward newTail
      if (parentWorldQuat !== null) {
        const targetDirWorld = newTail.subtract(anchorPos).normalize()

        // Get bind direction in parent's local space
        const bindDirLocal = this.getBindTranslation(springBoneIdx).normalize()

        // Transform target direction to parent's local space
        const invParentQuat = parentWorldQuat.conjugate().normalize()
        const targetDirLocal = invParentQuat.rotate(targetDirWorld)

        // Compute rotation from bind direction to target direction
        const localRotation = Quat.fromTo(bindDirLocal, targetDirLocal)

        // Update bone's local rotation
        const springBoneQi = springBoneIdx * 4
        this.boneLocalRotations[springBoneQi] = localRotation.x
        this.boneLocalRotations[springBoneQi + 1] = localRotation.y
        this.boneLocalRotations[springBoneQi + 2] = localRotation.z
        this.boneLocalRotations[springBoneQi + 3] = localRotation.w

        // Update world matrix for next spring in chain
        const springBoneMatIdx = springBoneIdx * 16
        const springWorldM = recomputeWorldMatrix(springBoneIdx)
        this.boneWorldMatrices.set(springWorldM.values, springBoneMatIdx)
      }
    }
  }

  // Clear all spring bones
  clearSpringBones(): void {
    this.springBones = []
    this.springCurrentTails = undefined
    this.springPrevTails = undefined
    this.springInitialized = false
  }

  getBoneWorldMatrix(index: number): Float32Array | undefined {
    this.evaluatePose()
    const start = index * 16
    return this.boneWorldMatrices.slice(start, start + 16)
  }

  // Evaluate world and skin matrices from local TR and bind
  // If deltaTime is provided, also updates spring bone physics
  evaluatePose(deltaTime?: number): void {
    // Evaluate spring bones before evaluating pose (if deltaTime provided)
    if (deltaTime !== undefined) {
      this.evaluateSpringBones(deltaTime)
    }

    // Advance rotation tweens before composing matrices
    this.updateRotationTweens()
    const bones = this.skeleton.bones
    const invBind = this.skeleton.inverseBindMatrices
    const localRot = this.boneLocalRotations
    const worldBuf = this.boneWorldMatrices
    const skinBuf = this.boneSkinMatrices

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
          const apTx = this.boneLocalTranslations![apTi]
          const apTy = this.boneLocalTranslations![apTi + 1]
          const apTz = this.boneLocalTranslations![apTi + 2]
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

      const skinSeg = skinBuf.subarray(i * 16, i * 16 + 16)
      const invSeg = invBind.subarray(i * 16, i * 16 + 16)
      const skinM = new Mat4(worldSeg).multiply(new Mat4(invSeg))
      skinSeg.set(skinM.values)
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
