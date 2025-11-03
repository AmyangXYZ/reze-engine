const VERTEX_STRIDE = 8 // floats per vertex: position(3) + normal(3) + uv(2) = 8

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
  edgeFlag: number
  vertexCount: number
}

export interface Bone {
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

export interface Skeleton {
  bones: Bone[]
  inverseBindMatrices: Float32Array // One inverse-bind matrix per bone (column-major mat4, 16 floats per bone)
}

export interface Skinning {
  joints: Uint16Array // length = vertexCount * 4, bone indices per vertex
  weights: Uint8Array // UNORM8, length = vertexCount * 4, sums ~ 255 per-vertex
}

// Runtime skeleton pose state (updated each frame)
export interface SkeletonRuntime {
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
import { Rigidbody, Joint } from "./physics"

export class Model {
  private vertexData: Float32Array<ArrayBuffer>
  private vertexCount: number
  private indexData: Uint32Array<ArrayBuffer>
  private indexCount: number
  private textures: Texture[] = []
  private materials: Material[] = []
  // Static skeleton/skinning (not necessarily serialized yet)
  private skeleton: Skeleton
  private skinning: Skinning

  // Physics data from PMX
  private rigidbodies: Rigidbody[] = []
  private joints: Joint[] = []

  // Runtime skeleton pose state (updated each frame)
  private runtimeSkeleton: SkeletonRuntime

  // Spring bone physics runtime state
  private springPhysics: RzmSpringPhysics

  constructor(
    vertexData: Float32Array<ArrayBuffer>,
    indexData: Uint32Array<ArrayBuffer>,
    textures: Texture[],
    materials: Material[],
    skeleton: Skeleton,
    skinning: Skinning,
    rigidbodies: Rigidbody[] = [],
    joints: Joint[] = []
  ) {
    this.vertexData = vertexData
    this.vertexCount = vertexData.length / VERTEX_STRIDE
    this.indexData = indexData
    this.indexCount = indexData.length
    this.textures = textures
    this.materials = materials
    this.skeleton = skeleton
    this.skinning = skinning
    this.rigidbodies = rigidbodies
    this.joints = joints

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

  private updateRotationTweens(): void {
    const state = this.rotTweenState
    const now = performance.now()
    const n = this.skeleton.bones.length

    for (let i = 0; i < n; i++) {
      if (state.active[i] !== 1) continue
      const startMs = state.startTimeMs[i]
      const durMs = Math.max(1, state.durationMs[i])
      const t = Math.max(0, Math.min(1, (now - startMs) / durMs))
      const e = Model.easeInOut(t)
      const qi = i * 4
      const a0 = state.startQuat[qi]
      const a1 = state.startQuat[qi + 1]
      const a2 = state.startQuat[qi + 2]
      const a3 = state.startQuat[qi + 3]
      const b0 = state.targetQuat[qi]
      const b1 = state.targetQuat[qi + 1]
      const b2 = state.targetQuat[qi + 2]
      const b3 = state.targetQuat[qi + 3]
      const [x, y, z, w] = Model.slerp(a0, a1, a2, a3, b0, b1, b2, b3, e)
      this.runtimeSkeleton.localRotations[qi] = x
      this.runtimeSkeleton.localRotations[qi + 1] = y
      this.runtimeSkeleton.localRotations[qi + 2] = z
      this.runtimeSkeleton.localRotations[qi + 3] = w
      if (t >= 1) {
        state.active[i] = 0
      }
    }
  }

  // Get interleaved vertex data for GPU upload
  // Format: [x,y,z, nx,ny,nz, u,v, x,y,z, nx,ny,nz, u,v, ...]
  getVertices(): Float32Array<ArrayBuffer> {
    return this.vertexData
  }

  // Get texture information
  getTextures(): Texture[] {
    return this.textures
  }

  // Get material information
  getMaterials(): Material[] {
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

  // Accessors for skeleton/skinning
  getSkeleton(): Skeleton {
    return this.skeleton
  }

  getSkinning(): Skinning {
    return this.skinning
  }

  // Accessors for physics data
  getRigidbodies(): Rigidbody[] {
    return this.rigidbodies
  }

  getJoints(): Joint[] {
    return this.joints
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
        const e = Model.easeInOut(t)
        const a0 = state.startQuat[qi]
        const a1 = state.startQuat[qi + 1]
        const a2 = state.startQuat[qi + 2]
        const a3 = state.startQuat[qi + 3]
        const b0 = state.targetQuat[qi]
        const b1 = state.targetQuat[qi + 1]
        const b2 = state.targetQuat[qi + 2]
        const b3 = state.targetQuat[qi + 3]
        const cur = Model.slerp(a0, a1, a2, a3, b0, b1, b2, b3, e)
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

  getBoneWorldMatrix(index: number): Float32Array | undefined {
    this.evaluatePose()
    const start = index * 16
    return this.runtimeSkeleton.worldMatrices.slice(start, start + 16)
  }

  getBoneWorldMatrices(): Float32Array {
    this.evaluatePose()
    return this.runtimeSkeleton.worldMatrices
  }

  getBoneWorldPosition(index: number): Vec3 {
    this.evaluatePose()
    const matIdx = index * 16
    return new Vec3(
      this.runtimeSkeleton.worldMatrices[matIdx + 12],
      this.runtimeSkeleton.worldMatrices[matIdx + 13],
      this.runtimeSkeleton.worldMatrices[matIdx + 14]
    )
  }

  getBoneWorldRotation(index: number): Quat {
    this.evaluatePose()
    const start = index * 16
    const matrix = new Mat4(this.runtimeSkeleton.worldMatrices.subarray(start, start + 16))
    return matrix.toQuat()
  }

  getBoneInverseBindMatrix(index: number): Mat4 | undefined {
    if (index < 0 || index >= this.skeleton.bones.length) return undefined
    const start = index * 16
    return new Mat4(this.skeleton.inverseBindMatrices.subarray(start, start + 16))
  }

  getLocalRotations(): Float32Array {
    return this.runtimeSkeleton.localRotations
  }

  // Evaluate world and skin matrices from local TR and bind
  // If deltaTime is provided, also updates spring bone physics
  evaluatePose(deltaTime?: number): void {
    if (!deltaTime && false) {
      return
    }
    // Advance rotation tweens before composing matrices
    this.updateRotationTweens()

    // Compute world matrices BEFORE spring bone physics (needed for collision detection)
    // This gives us up-to-date matrices for body collider bones
    this.computeWorldMatrices()

    // Compute skin matrices from world matrices
    this.updateSkinMatrices()
  }

  // Update skin matrices from current world matrices
  // Called after physics modifies world matrices in-place to update skinning
  updateSkinMatrices(): void {
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
          const [rx, ry, rz, rw] = Model.slerp(0, 0, 0, 1, ax, ay, az, aw, ratio)
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
}
