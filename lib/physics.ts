import { Quat, Vec3, Mat4 } from "./math"

export enum RigidbodyShape {
  Sphere = 0,
  Box = 1,
  Capsule = 2,
}

export enum RigidbodyType {
  Static = 0,
  Dynamic = 1,
  Kinematic = 2,
}

export interface Rigidbody {
  name: string
  englishName: string
  boneIndex: number
  group: number
  collisionMask: number
  shape: RigidbodyShape
  size: Vec3
  position: Vec3
  rotation: Quat
  mass: number
  linearDamping: number
  angularDamping: number
  restitution: number
  friction: number
  type: RigidbodyType
  initialTransform: Mat4 // Transform matrix in bind pose world space
  // Runtime physics state (initialized lazily for dynamic rigidbodies)
  linearVelocity?: Vec3 // Linear velocity in cm/s
  angularVelocity?: Vec3 // Angular velocity in rad/s
}

export interface Joint {
  name: string
  englishName: string
  type: number
  rigidbodyIndexA: number
  rigidbodyIndexB: number
  position: Vec3
  rotation: Quat
  positionMin: Vec3
  positionMax: Vec3
  rotationMin: Quat
  rotationMax: Quat
  springPosition: Vec3
  springRotation: Quat
  initialTransform: Mat4 // Transform matrix in bind pose world space (position + rotation)
}

// Spring bone chain (matches reference code exactly)
export interface SpringBoneChain {
  rootBoneIndex: number // Anchor point - this bone is not affected by physics
  boneIndices: number[] // Chain of bones to apply spring physics to (ordered from root to tip)
  stiffness: number // Spring stiffness (higher = stiffer, typical range: 0.01-1.0)
  dragForce: number // Damping factor (higher = less oscillation, typical range: 0.1-0.9)
}

export interface Bone {
  name: string
  parentIndex: number
  bindTranslation: [number, number, number]
  children: number[]
}

export class Physics {
  private rigidbodies: Rigidbody[]
  private joints: Joint[] = []
  // Spring bone chains (auto-detected, works directly on bones)
  private springBoneChains: SpringBoneChain[] = []
  // Spring bone runtime state (matches reference code)
  private springBoneInitialized = false
  private springBoneCurrentPositions?: Float32Array // vec3 per bone in chains
  private springBonePrevPositions?: Float32Array // vec3 per bone in chains
  private springBoneTimeAccumulator = 0
  // Gravity acceleration vector (cm/s²) - Match spring bone reference: 980.0 cm/s²
  private gravity: Vec3 = new Vec3(0, -980, 0)
  // Track if dynamic rigidbodies have been initialized from bone state
  private dynamicRigidbodiesInitialized = false
  // Previous positions for Verlet-style joint solving (for stability)
  private jointPrevPositions: Map<number, Vec3> = new Map() // rigidbody index -> previous position
  // Cached spring bone indices set (for collision exclusion)
  private cachedSpringBoneIndices?: Set<number>
  // Cached collision candidates (Static/Kinematic rigidbodies not in spring bone chains)
  private collisionCandidates?: Rigidbody[]

  constructor(rigidbodies: Rigidbody[], joints: Joint[] = [], bones?: Bone[]) {
    this.rigidbodies = rigidbodies

    // Auto-detect spring bone chains from bone names (like reference code)
    if (bones && bones.length > 0) {
      this.springBoneChains = this.autoDetectSpringBoneChains(bones)
      console.log(`[Physics] Auto-detected ${this.springBoneChains.length} spring bone chains`)
    }

    // Store regular joints if provided
    this.joints = joints

    // Initialize velocities for dynamic rigidbodies
    for (const rb of this.rigidbodies) {
      if (rb.type === RigidbodyType.Dynamic) {
        if (!rb.linearVelocity) {
          rb.linearVelocity = new Vec3(0, 0, 0)
        }
        if (!rb.angularVelocity) {
          rb.angularVelocity = new Vec3(0, 0, 0)
        }
      }
    }
  }

  // Auto-detect spring bone chains from bone names (exactly like reference code)
  private autoDetectSpringBoneChains(bones: Bone[]): SpringBoneChain[] {
    const chains: SpringBoneChain[] = []
    const boneNames = bones.map((b) => b.name)

    // Pattern matching: group bones by pattern (like autoDetectSpringBones)
    const prefixGroups = new Map<string, Array<{ boneIndex: number; suffix: number }>>()

    for (let i = 0; i < boneNames.length; i++) {
      const boneName = boneNames[i]

      // Find the first number in the bone name
      const firstNumberMatch = boneName.match(/(\d+)/)
      if (firstNumberMatch) {
        const firstNumber = parseInt(firstNumberMatch[1], 10)
        const firstNumberStart = firstNumberMatch.index!
        const firstNumberEnd = firstNumberStart + firstNumberMatch[1].length

        // Create pattern key by replacing first number with {}
        const patternKey = boneName.slice(0, firstNumberStart) + "{}" + boneName.slice(firstNumberEnd)

        // Check if pattern key contains only ASCII characters (English, not Japanese)
        const isEnglishOnly = /^[\x00-\x7F]*$/.test(patternKey)

        // Only group if first number is valid and pattern key is English-only
        if (!isNaN(firstNumber) && patternKey.length > 0 && isEnglishOnly) {
          if (!prefixGroups.has(patternKey)) {
            prefixGroups.set(patternKey, [])
          }
          prefixGroups.get(patternKey)!.push({ boneIndex: i, suffix: firstNumber })
        }
      }
    }

    // Filter groups: only keep groups with at least 2 bones (need chain)
    const validGroups: Array<{ prefix: string; bones: Array<{ boneIndex: number; suffix: number }> }> = []

    for (const [patternKey, boneList] of prefixGroups.entries()) {
      if (boneList.length >= 2) {
        // Sort by first number (ensures bones are ordered correctly in the chain)
        boneList.sort((a, b) => a.suffix - b.suffix)
        validGroups.push({ prefix: patternKey, bones: boneList })
      }
    }

    if (validGroups.length === 0) {
      return chains
    }

    // Log all group pattern keys
    const patternKeys = validGroups.map((g) => g.prefix).join(", ")
    console.log(`[Physics] Spring bone groups: ${patternKeys}`)

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

      // Add spring chain with VRM-like parameters (matches reference)
      chains.push({
        rootBoneIndex,
        boneIndices,
        stiffness: 0.5,
        dragForce: 0.5,
      })
    }

    return chains
  }

  // Set gravity acceleration (default: -200 cm/s² on Y axis for smoother animation)
  // Use -980 for MMD-style physics if you want faster, more realistic falling
  setGravity(gravity: Vec3): void {
    this.gravity = gravity
  }

  getGravity(): Vec3 {
    return this.gravity
  }

  getRigidbodies(): Rigidbody[] {
    return this.rigidbodies
  }

  getJoints(): Joint[] {
    return this.joints
  }

  // Main physics step: applies spring bone physics directly to bones (like reference code)
  // Modifies boneWorldMatrices in-place for spring bone chains
  // Also updates localRotations so evaluatePose() doesn't overwrite physics changes
  step(
    dt: number,
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number,
    bones?: Bone[],
    localRotations?: Float32Array
  ): void {
    // Sync Static/Kinematic rigidbodies from bones (for visualization)
    this.syncFromBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)

    // Apply spring bone physics directly to bones (like reference code)
    if (this.springBoneChains.length > 0 && bones) {
      this.evaluateSpringBones(dt, boneWorldMatrices, boneInverseBindMatrices, boneCount, bones, localRotations)

      // After spring bones update bone matrices, sync corresponding rigidbodies
      // This allows spring bones to drive rigidbodies (for visualization and collision)
      this.syncRigidbodiesFromSpringBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)
    }

    // Legacy rigidbody physics (keep for compatibility, but spring bones take priority)
    if (this.rigidbodies.length > 0 && this.springBoneChains.length === 0) {
      // Initialize dynamic rigidbodies from current bone state (only once, before first simulation)
      if (!this.dynamicRigidbodiesInitialized) {
        this.syncDynamicRigidbodiesFromBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)
        this.dynamicRigidbodiesInitialized = true
      }

      // Simulate dynamic rigidbodies (integration)
      // Limit dt to prevent huge jumps on first frame or when framerate spikes
      const clampedDt = Math.min(dt, 0.017) // Cap at ~60fps minimum
      this.simulate()
      // Solve joint constraints (springs, limits)
      this.solve(clampedDt)
      // Update bone world matrices in-place for dynamic rigidbodies
      this.applyDynamicRigidbodiesToBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)
    }
  }

  // Sync Static/Kinematic rigidbodies to follow bone transforms
  private syncFromBones(
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number
  ): void {
    for (const rb of this.rigidbodies) {
      // Static and Kinematic both follow bones in MMD/PMX
      // Static: pure bone following, no physics
      // Kinematic: bone-driven but can interact with dynamic objects
      if (
        (rb.type === RigidbodyType.Static || rb.type === RigidbodyType.Kinematic) &&
        rb.boneIndex >= 0 &&
        rb.boneIndex < boneCount
      ) {
        const boneIdx = rb.boneIndex
        const worldMatIdx = boneIdx * 16
        const invBindIdx = boneIdx * 16

        // Get bone world matrix and inverse bind matrix
        const worldMat = new Mat4(boneWorldMatrices.subarray(worldMatIdx, worldMatIdx + 16))
        const invBindMat = new Mat4(boneInverseBindMatrices.subarray(invBindIdx, invBindIdx + 16))

        // Compute offset transform: worldMatrix * inverseBindMatrix (bind pose to current pose)
        const offsetMat = worldMat.multiply(invBindMat)

        // Transform initial transform to current world space
        const currentTransform = offsetMat.multiply(rb.initialTransform)

        // Extract position and rotation from current transform
        rb.position = currentTransform.getPosition()
        rb.rotation = currentTransform.toQuat()
      }
    }
  }

  // Sync rigidbodies that correspond to bones in spring bone chains
  // This allows spring bones to drive rigidbodies (for all types: Static, Kinematic, Dynamic)
  private syncRigidbodiesFromSpringBones(
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number
  ): void {
    // Build a set of bone indices that are affected by spring bones
    const springBoneIndices = new Set<number>()

    for (const chain of this.springBoneChains) {
      // Include root bone (anchor point) - though it's not directly affected, it may have rigidbodies
      if (chain.rootBoneIndex >= 0 && chain.rootBoneIndex < boneCount) {
        springBoneIndices.add(chain.rootBoneIndex)
      }
      // Include all bones in the chain
      for (const boneIdx of chain.boneIndices) {
        if (boneIdx >= 0 && boneIdx < boneCount) {
          springBoneIndices.add(boneIdx)
        }
      }
    }

    // Sync all rigidbodies that are attached to spring bone-affected bones
    for (const rb of this.rigidbodies) {
      if (rb.boneIndex >= 0 && rb.boneIndex < boneCount && springBoneIndices.has(rb.boneIndex)) {
        // Sync rigidbody from bone transform (same logic as syncFromBones)
        const boneIdx = rb.boneIndex
        const worldMatIdx = boneIdx * 16
        const invBindIdx = boneIdx * 16

        // Get bone world matrix and inverse bind matrix
        const worldMat = new Mat4(boneWorldMatrices.subarray(worldMatIdx, worldMatIdx + 16))
        const invBindMat = new Mat4(boneInverseBindMatrices.subarray(invBindIdx, invBindIdx + 16))

        // Compute offset transform: worldMatrix * inverseBindMatrix (bind pose to current pose)
        const offsetMat = worldMat.multiply(invBindMat)

        // Transform initial transform to current world space
        const currentTransform = offsetMat.multiply(rb.initialTransform)

        // Extract position and rotation from current transform
        rb.position = currentTransform.getPosition()
        rb.rotation = currentTransform.toQuat()
      }
    }
  }

  // Initialize dynamic rigidbodies from current bone state (called once before first simulation)
  // This ensures they start from the current bone pose, not the bind pose
  private syncDynamicRigidbodiesFromBones(
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number
  ): void {
    for (const rb of this.rigidbodies) {
      if (rb.type === RigidbodyType.Dynamic && rb.boneIndex >= 0 && rb.boneIndex < boneCount) {
        const boneIdx = rb.boneIndex
        const worldMatIdx = boneIdx * 16
        const invBindIdx = boneIdx * 16

        // Get bone world matrix and inverse bind matrix
        const worldMat = new Mat4(boneWorldMatrices.subarray(worldMatIdx, worldMatIdx + 16))
        const invBindMat = new Mat4(boneInverseBindMatrices.subarray(invBindIdx, invBindIdx + 16))

        // Compute offset transform: worldMatrix * inverseBindMatrix (bind pose to current pose)
        const offsetMat = worldMat.multiply(invBindMat)

        // Transform initial transform to current world space
        const currentTransform = offsetMat.multiply(rb.initialTransform)

        // Extract position and rotation from current transform
        // This initializes dynamic rigidbody to current bone pose
        rb.position = currentTransform.getPosition()
        rb.rotation = currentTransform.toQuat()
      }
    }
  }

  // Simulate dynamic rigidbodies - simplified for spring bone-like behavior
  // Gravity will be applied together with spring forces in solveJoints (Verlet-style)
  private simulate(): void {
    // Just ensure velocities are initialized - actual integration happens in solveJoints
    for (const rb of this.rigidbodies) {
      if (rb.type !== RigidbodyType.Dynamic) continue
      if (!rb.linearVelocity) {
        rb.linearVelocity = new Vec3(0, 0, 0)
      }
    }
  }

  // Solve joint constraints (legacy rigidbody-based joints - only used if no spring bones)
  private solve(dt: number): void {
    if (!dt) return
    // Legacy joint solving removed - spring bones handle physics directly
  }

  // Apply dynamic rigidbody world transforms to bone world matrices in-place
  // Directly modifies boneWorldMatrices without converting to local space
  private applyDynamicRigidbodiesToBones(
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number
  ): void {
    for (const rb of this.rigidbodies) {
      // Only dynamic rigidbodies drive bones (Static/Kinematic follow bones)
      if (rb.type === RigidbodyType.Dynamic && rb.boneIndex >= 0 && rb.boneIndex < boneCount) {
        const boneIdx = rb.boneIndex
        const worldMatIdx = boneIdx * 16

        // Reconstruct bone world transform from rigidbody world transform
        // From syncFromBones: rbWorld = boneWorld * invBind * rbInitial
        // So: boneWorld = rbWorld * inv(rbInitial) * bindMatrix
        const rbWorldMat = Mat4.fromPositionRotation(rb.position, rb.rotation)
        const invRbInitial = rb.initialTransform.inverse()
        const bindMat = boneInverseBindMatrices.subarray(boneIdx * 16, boneIdx * 16 + 16)
        const bindMatrix = new Mat4(new Float32Array(bindMat)).inverse() // inverseBind -> bind
        const boneWorldMat = rbWorldMat.multiply(invRbInitial).multiply(bindMatrix)

        // Write bone world matrix directly to the array in-place
        boneWorldMatrices.set(boneWorldMat.values, worldMatIdx)
      }
    }
  }

  // Extract unit quaternion (x,y,z,w) from a column-major rotation matrix (upper-left 3x3 of mat4)
  // Matches reference code exactly
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

  // Helper: get bone's bind translation as Vec3
  private getBindTranslation(boneIndex: number, bones: Bone[]): Vec3 {
    const bone = bones[boneIndex]
    return new Vec3(bone.bindTranslation[0], bone.bindTranslation[1], bone.bindTranslation[2])
  }

  // Helper: extract world position from bone matrix
  private getBoneWorldPosition(boneIndex: number, boneWorldMatrices: Float32Array): Vec3 {
    const matIdx = boneIndex * 16
    return new Vec3(boneWorldMatrices[matIdx + 12], boneWorldMatrices[matIdx + 13], boneWorldMatrices[matIdx + 14])
  }

  // Helper: get parent's world transform (position and rotation)
  private getParentWorldTransform(parentBoneIdx: number, boneWorldMatrices: Float32Array): { pos: Vec3; quat: Quat } {
    const parentMatIdx = parentBoneIdx * 16
    const parentM = new Mat4(
      new Float32Array(boneWorldMatrices.buffer, boneWorldMatrices.byteOffset + parentMatIdx * 4, 16)
    )
    const pos = new Vec3(
      boneWorldMatrices[parentMatIdx + 12],
      boneWorldMatrices[parentMatIdx + 13],
      boneWorldMatrices[parentMatIdx + 14]
    )
    const [px, py, pz, pw] = Physics.mat3ToQuat(parentM.values)
    const quat = new Quat(px, py, pz, pw).normalize()
    return { pos, quat }
  }

  // Helper: compute anchor position for spring bone
  private computeAnchorPosition(
    springBoneIdx: number,
    boneWorldMatrices: Float32Array,
    bones: Bone[]
  ): { anchor: Vec3; parentQuat: Quat | null } {
    const springBone = bones[springBoneIdx]

    if (springBone.parentIndex >= 0) {
      const { pos: parentPos, quat: parentQuat } = this.getParentWorldTransform(
        springBone.parentIndex,
        boneWorldMatrices
      )
      const bindTransLocal = this.getBindTranslation(springBoneIdx, bones)
      const bindTransWorld = parentQuat.rotate(bindTransLocal)
      return { anchor: parentPos.add(bindTransWorld), parentQuat }
    }

    // Root bone
    const bindTrans = this.getBindTranslation(springBoneIdx, bones)
    return { anchor: bindTrans, parentQuat: null }
  }

  // Helper: compute anchor position for first bone in chain (anchored to root bone)
  private computeAnchorPositionForChainBone(
    boneIdx: number,
    rootBoneIdx: number,
    boneWorldMatrices: Float32Array,
    bones: Bone[]
  ): { anchor: Vec3; parentQuat: Quat | null } {
    if (rootBoneIdx >= 0 && rootBoneIdx < bones.length) {
      const { pos: rootPos, quat: rootQuat } = this.getParentWorldTransform(rootBoneIdx, boneWorldMatrices)
      const bindTransLocal = this.getBindTranslation(boneIdx, bones)
      const bindTransWorld = rootQuat.rotate(bindTransLocal)
      return { anchor: rootPos.add(bindTransWorld), parentQuat: rootQuat }
    }
    // Fallback to regular anchor computation
    return this.computeAnchorPosition(boneIdx, boneWorldMatrices, bones)
  }

  // Evaluate spring bone physics using proper VRM-style Verlet integration with chains
  // Works directly on bone world matrices (like reference code)
  // Also updates localRotations so evaluatePose() doesn't overwrite physics changes
  private evaluateSpringBones(
    deltaTime: number,
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number,
    bones: Bone[],
    localRotations?: Float32Array
  ): void {
    if (!this.springBoneChains || this.springBoneChains.length === 0) return

    // Count total bones across all chains and create mapping
    let totalBones = 0
    const chainBoneOffsets: number[] = [] // Offset in position array for each chain

    for (let chainIdx = 0; chainIdx < this.springBoneChains.length; chainIdx++) {
      const chain = this.springBoneChains[chainIdx]
      chainBoneOffsets.push(totalBones)
      for (let boneIdx = 0; boneIdx < chain.boneIndices.length; boneIdx++) {
        totalBones++
      }
    }

    if (totalBones === 0) return

    // Initialize spring positions if needed (only once)
    if (!this.springBoneInitialized || !this.springBoneCurrentPositions || !this.springBonePrevPositions) {
      // Initialize position arrays for all bones in all chains
      this.springBoneCurrentPositions = new Float32Array(totalBones * 3)
      this.springBonePrevPositions = new Float32Array(totalBones * 3)

      let globalBoneIdx = 0
      for (let chainIdx = 0; chainIdx < this.springBoneChains.length; chainIdx++) {
        const chain = this.springBoneChains[chainIdx]

        // Initialize positions for each bone in chain from bind pose
        for (let boneIdx = 0; boneIdx < chain.boneIndices.length; boneIdx++) {
          const boneIndex = chain.boneIndices[boneIdx]

          // Compute anchor position (same method as in evaluation loop)
          let anchorPos: Vec3
          let parentWorldQuat: Quat | null = null

          if (boneIdx === 0) {
            // First bone in chain: anchor to root bone
            if (chain.rootBoneIndex >= 0) {
              const { anchor, parentQuat } = this.computeAnchorPositionForChainBone(
                boneIndex,
                chain.rootBoneIndex,
                boneWorldMatrices,
                bones
              )
              anchorPos = anchor
              parentWorldQuat = parentQuat
            } else {
              const { anchor, parentQuat } = this.computeAnchorPosition(boneIndex, boneWorldMatrices, bones)
              anchorPos = anchor
              parentWorldQuat = parentQuat
            }
          } else {
            // Subsequent bones: anchor to previous bone's head position
            const prevBoneIndex = chain.boneIndices[boneIdx - 1]
            const prevBonePos = this.getBoneWorldPosition(prevBoneIndex, boneWorldMatrices)

            const prevBoneMatIdx = prevBoneIndex * 16
            const prevBoneMat = new Mat4(
              new Float32Array(boneWorldMatrices.buffer, boneWorldMatrices.byteOffset + prevBoneMatIdx * 4, 16)
            )
            const [px, py, pz, pw] = Physics.mat3ToQuat(prevBoneMat.values)
            parentWorldQuat = new Quat(px, py, pz, pw).normalize()

            const bindTrans = this.getBindTranslation(boneIndex, bones)
            const bindTransWorld = parentWorldQuat.rotate(bindTrans)
            anchorPos = prevBonePos.add(bindTransWorld)
          }

          // Initialize tail position: extend from anchor in bind direction
          const bindTrans = this.getBindTranslation(boneIndex, bones)
          let restDir: Vec3
          if (parentWorldQuat !== null) {
            restDir = parentWorldQuat.rotate(bindTrans).normalize()
          } else {
            restDir = bindTrans.normalize()
          }
          const restLen = bindTrans.length()
          const tailPos = anchorPos.add(restDir.scale(restLen))

          const posIdx = globalBoneIdx * 3
          this.springBoneCurrentPositions[posIdx] = tailPos.x
          this.springBoneCurrentPositions[posIdx + 1] = tailPos.y
          this.springBoneCurrentPositions[posIdx + 2] = tailPos.z
          this.springBonePrevPositions[posIdx] = tailPos.x
          this.springBonePrevPositions[posIdx + 1] = tailPos.y
          this.springBonePrevPositions[posIdx + 2] = tailPos.z

          globalBoneIdx++
        }
      }
      this.springBoneInitialized = true
    }

    // Use fixed timestep substepping for truly frame-rate independent physics
    const fixedTimeStep = 1.0 / 60.0 // 60Hz physics timestep (16.67ms)
    const maxDeltaTime = 0.1 // Clamp to prevent huge jumps from frame drops

    // Clamp deltaTime to prevent huge first-frame jumps
    const clampedDeltaTime = Math.max(0, Math.min(deltaTime, maxDeltaTime))

    // Add this frame's time to accumulator
    this.springBoneTimeAccumulator = this.springBoneTimeAccumulator + clampedDeltaTime

    // Run physics steps until we've consumed all accumulated time
    let stepCount = 0
    const maxSteps = 6 // Safety limit: max 6 steps per frame (100ms max)
    while (this.springBoneTimeAccumulator >= fixedTimeStep && stepCount < maxSteps) {
      const dt = fixedTimeStep
      const dt2 = dt * dt
      this.evaluateSpringBonesStep(
        dt,
        dt2,
        chainBoneOffsets,
        boneWorldMatrices,
        boneInverseBindMatrices,
        boneCount,
        bones,
        localRotations
      )
      this.springBoneTimeAccumulator -= fixedTimeStep
      stepCount++
    }

    // Keep only fractional remainder (prevents unbounded accumulation if frames are very fast)
    if (this.springBoneTimeAccumulator > fixedTimeStep * 2) {
      this.springBoneTimeAccumulator = fixedTimeStep
    }
  }

  // Build and cache spring bone indices set and collision candidates
  // Called once per step to avoid rebuilding on every collision check
  private updateCollisionCandidates(): void {
    // Build spring bone indices set (only if not cached or chains changed)
    if (!this.cachedSpringBoneIndices) {
      this.cachedSpringBoneIndices = new Set<number>()
      for (const chain of this.springBoneChains) {
        if (chain.rootBoneIndex >= 0) this.cachedSpringBoneIndices.add(chain.rootBoneIndex)
        for (const idx of chain.boneIndices) this.cachedSpringBoneIndices.add(idx)
      }
    }

    // Pre-filter collision candidates (Static/Kinematic rigidbodies not in spring bone chains)
    this.collisionCandidates = this.rigidbodies.filter(
      (rb) =>
        (rb.type === RigidbodyType.Static || rb.type === RigidbodyType.Kinematic) &&
        !this.cachedSpringBoneIndices!.has(rb.boneIndex)
    )
  }

  // Resolve collision between spring bone position and rigidbodies
  // Standard physics collision resolution: push point out along contact normal
  // Returns the corrected position after collision response
  // Optimized: assumes rigidbodies are already synced and collision candidates are cached
  private resolveSpringBoneCollision(position: Vec3): Vec3 {
    // Use cached collision candidates (must be updated before calling this)
    const candidates = this.collisionCandidates
    if (!candidates || candidates.length === 0) return position

    let correctedPos = position.clone()
    const maxIterations = 4
    const separationMargin = 0.1 // Small margin to prevent re-collision (cm)

    // Iterative collision resolution (standard physics: resolve deepest penetration first)
    for (let iter = 0; iter < maxIterations; iter++) {
      let deepestPenetration = 0
      let separationNormal = new Vec3(0, 0, 0)

      // Find deepest collision (only check pre-filtered candidates)
      for (const rb of candidates) {
        const contact = this.computePointRigidbodyContact(correctedPos, rb)
        if (contact.penetrating && contact.penetration > deepestPenetration) {
          deepestPenetration = contact.penetration
          separationNormal = contact.normal
        }
      }

      // Apply separation along normal
      if (deepestPenetration > 0) {
        const separationDistance = deepestPenetration + separationMargin
        correctedPos = correctedPos.add(separationNormal.scale(separationDistance))
      } else {
        break // No collisions, done
      }
    }

    return correctedPos
  }

  // Compute contact information between point and rigidbody
  // Standard physics: returns penetration depth and separation normal
  // Normal points from rigidbody surface toward point (direction to push point out)
  private computePointRigidbodyContact(
    point: Vec3,
    rb: Rigidbody
  ): {
    penetrating: boolean
    normal: Vec3
    penetration: number
  } {
    switch (rb.shape) {
      case RigidbodyShape.Sphere:
        return this.computePointSphereContact(point, rb.position, rb.size.x)
      case RigidbodyShape.Box:
        return this.computePointBoxContact(point, rb.position, rb.rotation, rb.size)
      case RigidbodyShape.Capsule:
        return this.computePointCapsuleContact(point, rb.position, rb.rotation, rb.size.x, rb.size.y)
      default:
        return { penetrating: false, normal: new Vec3(0, 0, 0), penetration: 0 }
    }
  }

  // Standard sphere contact: penetration = radius - distance, normal = point - center
  private computePointSphereContact(
    point: Vec3,
    center: Vec3,
    radius: number
  ): {
    penetrating: boolean
    normal: Vec3
    penetration: number
  } {
    const toPoint = point.subtract(center)
    const distance = toPoint.length()

    if (distance < radius && radius > 0) {
      const penetration = radius - distance
      // Normal: from center to point (direction to push point out)
      const normal = distance > 1e-6 ? toPoint.normalize() : new Vec3(0, 1, 0)
      return { penetrating: true, normal, penetration }
    }

    return { penetrating: false, normal: new Vec3(0, 0, 0), penetration: 0 }
  }

  // Standard AABB contact: find closest face, penetration = distance inside
  private computePointBoxContact(
    point: Vec3,
    center: Vec3,
    rotation: Quat,
    size: Vec3
  ): {
    penetrating: boolean
    normal: Vec3
    penetration: number
  } {
    // Transform to local space
    const invRotation = rotation.conjugate().normalize()
    const localPoint = invRotation.rotate(point.subtract(center))
    const halfSize = size.scale(0.5)

    // Compute penetration for each axis
    const penX = halfSize.x - Math.abs(localPoint.x)
    const penY = halfSize.y - Math.abs(localPoint.y)
    const penZ = halfSize.z - Math.abs(localPoint.z)

    // Point is inside if all penetrations are positive
    if (penX > 0 && penY > 0 && penZ > 0) {
      // Find axis with smallest penetration (closest face)
      const minPen = Math.min(penX, penY, penZ)
      let localNormal: Vec3

      if (minPen === penX) {
        localNormal = localPoint.x >= 0 ? new Vec3(1, 0, 0) : new Vec3(-1, 0, 0)
      } else if (minPen === penY) {
        localNormal = localPoint.y >= 0 ? new Vec3(0, 1, 0) : new Vec3(0, -1, 0)
      } else {
        localNormal = localPoint.z >= 0 ? new Vec3(0, 0, 1) : new Vec3(0, 0, -1)
      }

      // Transform normal to world space
      const worldNormal = rotation.rotate(localNormal).normalize()
      return { penetrating: true, normal: worldNormal, penetration: minPen }
    }

    return { penetrating: false, normal: new Vec3(0, 0, 0), penetration: 0 }
  }

  // Standard capsule contact: distance to line segment, penetration = radius - distance
  private computePointCapsuleContact(
    point: Vec3,
    center: Vec3,
    rotation: Quat,
    radius: number,
    height: number
  ): {
    penetrating: boolean
    normal: Vec3
    penetration: number
  } {
    // Transform to local space
    const invRotation = rotation.conjugate().normalize()
    const localPoint = invRotation.rotate(point.subtract(center))
    const halfHeight = height * 0.5

    // Clamp to capsule axis (Y-aligned)
    const clampedY = Math.max(-halfHeight, Math.min(halfHeight, localPoint.y))
    const closestOnAxis = new Vec3(0, clampedY, 0)

    // Vector from closest axis point to point
    const toPoint = localPoint.subtract(closestOnAxis)
    const distance = toPoint.length()

    if (distance < radius && radius > 0) {
      const penetration = radius - distance
      // Normal: from axis to point (direction to push point out)
      const localNormal = distance > 1e-6 ? toPoint.normalize() : new Vec3(0, 1, 0)
      const worldNormal = rotation.rotate(localNormal).normalize()
      return { penetrating: true, normal: worldNormal, penetration }
    }

    return { penetrating: false, normal: new Vec3(0, 0, 0), penetration: 0 }
  }

  private evaluateSpringBonesStep(
    dt: number,
    dt2: number,
    chainBoneOffsets: number[],
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number,
    bones: Bone[],
    localRotations?: Float32Array
  ): void {
    if (!this.springBoneChains || this.springBoneChains.length === 0) return

    // Sync rigidbodies once per step (not per collision check)
    this.syncFromBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)
    // Update collision candidates once per step
    this.updateCollisionCandidates()

    // Process each chain
    for (let chainIdx = 0; chainIdx < this.springBoneChains.length; chainIdx++) {
      const chain = this.springBoneChains[chainIdx]
      const offset = chainBoneOffsets[chainIdx]
      const gravityDir = new Vec3(0, -1, 0) // Gravity always points downward

      // Process each bone in the chain sequentially
      for (let boneIdx = 0; boneIdx < chain.boneIndices.length; boneIdx++) {
        const boneIndex = chain.boneIndices[boneIdx]
        const posIdx = (offset + boneIdx) * 3

        // Get current and previous positions
        const currentPos = new Vec3(
          this.springBoneCurrentPositions![posIdx],
          this.springBoneCurrentPositions![posIdx + 1],
          this.springBoneCurrentPositions![posIdx + 2]
        )
        const prevPos = new Vec3(
          this.springBonePrevPositions![posIdx],
          this.springBonePrevPositions![posIdx + 1],
          this.springBonePrevPositions![posIdx + 2]
        )

        // Compute anchor position using the same method as before
        let anchorPos: Vec3
        let parentWorldQuat: Quat | null = null

        if (boneIdx === 0) {
          // First bone in chain: anchor to root bone
          if (chain.rootBoneIndex >= 0) {
            const { anchor, parentQuat } = this.computeAnchorPositionForChainBone(
              boneIndex,
              chain.rootBoneIndex,
              boneWorldMatrices,
              bones
            )
            anchorPos = anchor
            parentWorldQuat = parentQuat
          } else {
            const { anchor, parentQuat } = this.computeAnchorPosition(boneIndex, boneWorldMatrices, bones)
            anchorPos = anchor
            parentWorldQuat = parentQuat
          }
        } else {
          // Subsequent bones: anchor to previous bone using its world matrix position
          const prevBoneIdx = boneIdx - 1
          const prevBoneIndex = chain.boneIndices[prevBoneIdx]

          const prevBoneMatIdx = prevBoneIndex * 16
          const prevBoneWorldPos = new Vec3(
            boneWorldMatrices[prevBoneMatIdx + 12],
            boneWorldMatrices[prevBoneMatIdx + 13],
            boneWorldMatrices[prevBoneMatIdx + 14]
          )

          const prevBoneMat = new Mat4(
            new Float32Array(boneWorldMatrices.buffer, boneWorldMatrices.byteOffset + prevBoneMatIdx * 4, 16)
          )
          const [px, py, pz, pw] = Physics.mat3ToQuat(prevBoneMat.values)
          parentWorldQuat = new Quat(px, py, pz, pw).normalize()

          const bindTrans = this.getBindTranslation(boneIndex, bones)
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
        const bindTrans = this.getBindTranslation(boneIndex, bones)
        const restLen = bindTrans.length()

        // Accumulate forces (matches reference code exactly)
        let force = new Vec3(0, 0, 0)

        // 1. Gravity
        const gravity = 980.0
        force = force.add(gravityDir.scale(gravity))

        // 2. Stiffness force: restore orientation toward rest pose
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
        const distanceError = currentDist - restLen
        if (Math.abs(distanceError) > 0.01) {
          const lengthForce = currentDir.scale(-distanceError * 500.0)
          force = force.add(lengthForce)
        }

        // Verlet integration: newPos = currentPos + velocity*(1-drag) + force*dt²
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

        // Collision detection: resolve penetration with rigidbodies
        // (rigidbodies already synced at start of step)
        newPos = this.resolveSpringBoneCollision(newPos)

        // Update Verlet state BEFORE smoothing (to maintain proper velocity)
        if (this.springBonePrevPositions) {
          this.springBonePrevPositions[posIdx] = currentPos.x
          this.springBonePrevPositions[posIdx + 1] = currentPos.y
          this.springBonePrevPositions[posIdx + 2] = currentPos.z
        }

        // Exponential smoothing to reduce high-frequency jitter
        const smoothingFactor = 0.5
        const smoothedPos = currentPos.scale(1.0 - smoothingFactor).add(newPos.scale(smoothingFactor))
        newPos = smoothedPos

        if (this.springBoneCurrentPositions) {
          this.springBoneCurrentPositions[posIdx] = newPos.x
          this.springBoneCurrentPositions[posIdx + 1] = newPos.y
          this.springBoneCurrentPositions[posIdx + 2] = newPos.z
        }

        // Update bone rotation to point from anchor toward newPos
        // We update the WORLD matrix directly, but we also need to ensure it persists
        // by not being overwritten by evaluatePose() next frame
        if (parentWorldQuat !== null) {
          const targetDirWorld = newPos.subtract(anchorPos).normalize()

          const bindDirLocal = bindTrans.normalize()

          const invParentQuat = parentWorldQuat.conjugate().normalize()
          const targetDirLocal = invParentQuat.rotate(targetDirWorld)

          const localRotation = Quat.fromTo(bindDirLocal, targetDirLocal)

          // Compute world matrix from local rotation
          const boneMatIdx = boneIndex * 16
          const rotateM = Mat4.fromQuat(localRotation.x, localRotation.y, localRotation.z, localRotation.w)
          const translateBind = Mat4.identity().translateInPlace(
            bones[boneIndex].bindTranslation[0],
            bones[boneIndex].bindTranslation[1],
            bones[boneIndex].bindTranslation[2]
          )
          const localM = translateBind.multiply(rotateM)

          if (bones[boneIndex].parentIndex >= 0 && bones[boneIndex].parentIndex < bones.length) {
            const parentMatIdx = bones[boneIndex].parentIndex * 16
            const parentM = new Mat4(
              new Float32Array(boneWorldMatrices.buffer, boneWorldMatrices.byteOffset + parentMatIdx * 4, 16)
            )
            const boneWorldM = parentM.multiply(localM)
            boneWorldMatrices.set(boneWorldM.values, boneMatIdx)
          } else {
            boneWorldMatrices.set(localM.values, boneMatIdx)
          }

          // CRITICAL: Also update local rotation so evaluatePose() doesn't overwrite this
          // Extract local rotation from the computed world matrix
          if (localRotations) {
            const qi = boneIndex * 4
            localRotations[qi] = localRotation.x
            localRotations[qi + 1] = localRotation.y
            localRotations[qi + 2] = localRotation.z
            localRotations[qi + 3] = localRotation.w
          }
        }
      }
    }
  }
}
