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

export class Physics {
  private rigidbodies: Rigidbody[]
  private joints: Joint[]
  // Gravity acceleration vector (cm/s²) - Start with lower value for smoother animation
  // Can be increased to -980 for MMD-style physics if needed
  private gravity: Vec3 = new Vec3(0, -0.0, 0)
  // Track if dynamic rigidbodies have been initialized from bone state
  private dynamicRigidbodiesInitialized = false

  constructor(rigidbodies: Rigidbody[], joints: Joint[] = []) {
    this.rigidbodies = rigidbodies
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

  // Main physics step: syncs bones to rigidbodies, simulates dynamics, solves constraints
  // Modifies boneWorldMatrices in-place for dynamic rigidbodies that drive bones
  step(dt: number, boneWorldMatrices: Float32Array, boneInverseBindMatrices: Float32Array, boneCount: number): void {
    // Sync Static/Kinematic rigidbodies from bones
    this.syncFromBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)

    // Initialize dynamic rigidbodies from current bone state (only once, before first simulation)
    if (!this.dynamicRigidbodiesInitialized) {
      this.syncDynamicRigidbodiesFromBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)
      this.dynamicRigidbodiesInitialized = true
    }

    // Simulate dynamic rigidbodies (integration)
    // Limit dt to prevent huge jumps on first frame or when framerate spikes
    const clampedDt = Math.min(dt, 0.017) // Cap at ~60fps minimum
    this.simulate(clampedDt)
    // Solve joint constraints (springs, limits)
    this.solve(clampedDt)
    // Update bone world matrices in-place for dynamic rigidbodies
    this.applyDynamicRigidbodiesToBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)
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

  // Simulate dynamic rigidbodies (velocity integration with gravity)
  private simulate(dt: number): void {
    for (const rb of this.rigidbodies) {
      if (rb.type !== RigidbodyType.Dynamic) continue

      // Ensure velocity is initialized
      if (!rb.linearVelocity) {
        rb.linearVelocity = new Vec3(0, 0, 0)
      }

      // Apply gravity acceleration: v = v + g * dt
      rb.linearVelocity = rb.linearVelocity.add(this.gravity.scale(dt))

      // Apply linear damping: v = v * (1 - damping * dt)
      const dampingFactor = Math.max(0, 1 - rb.linearDamping * dt)
      rb.linearVelocity = rb.linearVelocity.scale(dampingFactor)

      // Integrate position: x = x + v * dt
      rb.position = rb.position.add(rb.linearVelocity.scale(dt))

      // TODO: Apply angular velocity integration for rotation
      // For now, rotation is not affected by physics
    }
  }

  // Solve joint constraints (springs, limits)
  private solve(_dt: number): void {
    if (!_dt) return
    // TODO: Apply spring forces and enforce joint limits
    // Iterative constraint solving for stable joints
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
}
