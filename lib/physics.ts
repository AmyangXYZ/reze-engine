import { Quat, Vec3, Mat4 } from "./math"
import { loadAmmo } from "./ammo-loader"
import type { AmmoInstance } from "@fred3d/ammo"

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
  // Original shape position/rotation from PMX (bind pose world space, used for joint frame calculation)
  shapePosition: Vec3 // Original shapePosition from PMX (bind pose world space)
  shapeRotation: Vec3 // Original shapeRotation from PMX (bind pose world space, Euler angles xyz)
  mass: number
  linearDamping: number
  angularDamping: number
  restitution: number
  friction: number
  type: RigidbodyType
  // Body offset matrix (like reference code): relates rigidbody shape transform to bone bind space
  // Computed during initialization: bodyOffsetMatrix = nodeWorldMatrix * boneInverseBindMatrix
  // where nodeWorldMatrix is the shape transform in bone bind space
  bodyOffsetMatrixInverse: Mat4 // Inverse of bodyOffsetMatrix, used to sync rigidbody to bone
}

export interface Joint {
  name: string
  englishName: string
  type: number
  rigidbodyIndexA: number
  rigidbodyIndexB: number
  position: Vec3
  rotation: Vec3 // Euler angles in radians (not quaternion)
  positionMin: Vec3
  positionMax: Vec3
  rotationMin: Vec3 // Euler angles in radians (not quaternion)
  rotationMax: Vec3 // Euler angles in radians (not quaternion)
  springPosition: Vec3
  springRotation: Vec3 // Spring stiffness values (not quaternion)
}

export class Physics {
  private rigidbodies: Rigidbody[]
  private joints: Joint[]
  // Gravity acceleration vector (cm/s²) - Default to MMD-style gravity
  private gravity: Vec3 = new Vec3(0, -980, 0)
  // Track if Ammo has been initialized
  private ammoInitialized = false
  private ammoPromise: Promise<AmmoInstance> | null = null
  // Ammo physics objects
  private ammo: AmmoInstance | null = null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private dynamicsWorld: any = null // btDiscreteDynamicsWorld
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ammoRigidbodies: any[] = [] // btRigidBody instances
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ammoConstraints: any[] = [] // btTypedConstraint instances
  // Track if rigidbodies have been initialized (bodyOffsetMatrixInverse computed and Ammo bodies positioned)
  private rigidbodiesInitialized = false
  // Track if joints have been created (delayed until after rigidbodies are positioned)
  private jointsCreated = false
  // Track if this is the first frame (needed to reposition bodies before creating joints)
  private firstFrame = true

  constructor(rigidbodies: Rigidbody[], joints: Joint[] = []) {
    this.rigidbodies = rigidbodies
    this.joints = joints
    // Start loading Ammo asynchronously
    this.initAmmo()
  }

  private async initAmmo(): Promise<void> {
    if (this.ammoInitialized || this.ammoPromise) return
    this.ammoPromise = loadAmmo()
    try {
      this.ammo = await this.ammoPromise
      this.createAmmoWorld()
      this.ammoInitialized = true
    } catch (error) {
      console.error("[Physics] Failed to initialize Ammo:", error)
      this.ammoPromise = null
    }
  }

  // Set gravity acceleration (default: -980 cm/s² on Y axis for MMD-style physics)
  setGravity(gravity: Vec3): void {
    this.gravity = gravity
    if (this.dynamicsWorld && this.ammo) {
      const Ammo = this.ammo
      const gravityVec = new Ammo.btVector3(gravity.x, gravity.y, gravity.z)
      this.dynamicsWorld.setGravity(gravityVec)
      Ammo.destroy(gravityVec)
    }
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

  // Create Ammo physics world and rigidbodies
  private createAmmoWorld(): void {
    if (!this.ammo) return

    const Ammo = this.ammo

    // Create collision configuration
    const collisionConfiguration = new Ammo.btDefaultCollisionConfiguration()
    const dispatcher = new Ammo.btCollisionDispatcher(collisionConfiguration)
    const overlappingPairCache = new Ammo.btDbvtBroadphase()
    const solver = new Ammo.btSequentialImpulseConstraintSolver()

    // Create dynamics world
    this.dynamicsWorld = new Ammo.btDiscreteDynamicsWorld(
      dispatcher,
      overlappingPairCache,
      solver,
      collisionConfiguration
    )

    // Set gravity
    const gravityVec = new Ammo.btVector3(this.gravity.x, this.gravity.y, this.gravity.z)
    this.dynamicsWorld.setGravity(gravityVec)
    Ammo.destroy(gravityVec)

    // Create rigidbodies
    this.createAmmoRigidbodies()

    // Don't create joints yet - wait until rigidbodies are positioned
    // Joints will be created after first sync from bones
  }

  // Create Ammo rigidbodies from parsed rigidbodies
  private createAmmoRigidbodies(): void {
    if (!this.ammo || !this.dynamicsWorld) return

    const Ammo = this.ammo
    this.ammoRigidbodies = []

    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]

      // Create collision shape based on type
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let shape: any = null
      const size = rb.size

      switch (rb.shape) {
        case RigidbodyShape.Sphere:
          // Use the largest dimension as radius
          const radius = Math.max(size.x, size.y, size.z)
          shape = new Ammo.btSphereShape(radius)
          break
        case RigidbodyShape.Box:
          // btBoxShape expects half extents
          // PMX size values are already in the correct format
          const halfExtents = new Ammo.btVector3(size.x, size.y, size.z)
          shape = new Ammo.btBoxShape(halfExtents)
          Ammo.destroy(halfExtents)
          break
        case RigidbodyShape.Capsule:
          // Capsule: radius = max(x, z), height = y
          const capsuleRadius = Math.max(size.x, size.z)
          const capsuleHeight = size.y
          shape = new Ammo.btCapsuleShape(capsuleRadius, capsuleHeight)
          break
        default:
          // Default to box
          const defaultHalfExtents = new Ammo.btVector3(size.x, size.y, size.z)
          shape = new Ammo.btBoxShape(defaultHalfExtents)
          Ammo.destroy(defaultHalfExtents)
          break
      }

      // Create transform at shape's world position (bind pose) from PMX
      // shapePosition and shapeRotation are already in bind pose world space
      // This is critical - bodies must start at correct position to avoid explosions when joints are created
      const transform = new Ammo.btTransform()
      transform.setIdentity()

      // Use shape position/rotation from PMX (bind pose world space)
      const shapePos = new Ammo.btVector3(rb.shapePosition.x, rb.shapePosition.y, rb.shapePosition.z)
      transform.setOrigin(shapePos)
      Ammo.destroy(shapePos)

      // Convert Euler angles to quaternion
      const shapeRotQuat = Quat.fromEuler(rb.shapeRotation.x, rb.shapeRotation.y, rb.shapeRotation.z)
      const quat = new Ammo.btQuaternion(shapeRotQuat.x, shapeRotQuat.y, shapeRotQuat.z, shapeRotQuat.w)
      transform.setRotation(quat)
      Ammo.destroy(quat)

      // Determine motion state and mass
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let motionState: any = null
      let mass = 0
      let isDynamic = false

      if (rb.type === RigidbodyType.Dynamic) {
        mass = rb.mass
        isDynamic = true
        motionState = new Ammo.btDefaultMotionState(transform)
      } else if (rb.type === RigidbodyType.Kinematic) {
        mass = 0 // Kinematic objects have infinite mass
        motionState = new Ammo.btDefaultMotionState(transform)
      } else {
        // Static
        mass = 0
        motionState = new Ammo.btDefaultMotionState(transform)
      }

      // Calculate local inertia
      const localInertia = new Ammo.btVector3(0, 0, 0)
      if (isDynamic && mass > 0) {
        shape.calculateLocalInertia(mass, localInertia)
      }

      // Create rigid body construction info
      const rbInfo = new Ammo.btRigidBodyConstructionInfo(mass, motionState, shape, localInertia)
      rbInfo.set_m_restitution(rb.restitution)
      rbInfo.set_m_friction(rb.friction)
      rbInfo.set_m_linearDamping(rb.linearDamping)
      rbInfo.set_m_angularDamping(rb.angularDamping)

      // Create rigid body
      const body = new Ammo.btRigidBody(rbInfo)

      // Set sleeping thresholds (disable sleeping like reference code)
      body.setSleepingThresholds(0.0, 0.0)

      // Set collision flags
      // CF_STATIC_OBJECT = 1, CF_KINEMATIC_OBJECT = 2, CF_NO_CONTACT_RESPONSE = 4
      // DISABLE_DEACTIVATION = 4 (for activation state)
      // CRITICAL: Static (FollowBone) should be kinematic, not static!
      // PMX type 0 = FollowBone → must follow bones, so it's kinematic
      if (rb.type === RigidbodyType.Static) {
        // FollowBone bodies must follow bones - treat as kinematic
        body.setCollisionFlags(body.getCollisionFlags() | 2) // CF_KINEMATIC_OBJECT
        body.setActivationState(4) // DISABLE_DEACTIVATION
      } else if (rb.type === RigidbodyType.Kinematic) {
        body.setCollisionFlags(body.getCollisionFlags() | 2) // CF_KINEMATIC_OBJECT
        body.setActivationState(4) // DISABLE_DEACTIVATION
      }

      // Set collision groups and masks
      // PMX stores group as an index (0-15), convert to bit flag like reference code
      // collisionMask is already a bitmask in PMX
      const collisionGroup = 1 << rb.group
      body.getBroadphaseProxy().set_m_collisionFilterGroup(collisionGroup)
      body.getBroadphaseProxy().set_m_collisionFilterMask(rb.collisionMask)

      // Set CF_NO_CONTACT_RESPONSE if collision mask is 0 (like reference code)
      if (rb.collisionMask === 0) {
        body.setCollisionFlags(body.getCollisionFlags() | 4) // CF_NO_CONTACT_RESPONSE
      }

      // Add to world
      this.dynamicsWorld.addRigidBody(body)

      // Store reference
      this.ammoRigidbodies.push(body)

      // Cleanup
      Ammo.destroy(rbInfo)
      Ammo.destroy(localInertia)
    }
  }

  // Create Ammo constraints/joints from parsed joints
  private createAmmoJoints(): void {
    if (!this.ammo || !this.dynamicsWorld || this.ammoRigidbodies.length === 0) return

    const Ammo = this.ammo
    this.ammoConstraints = []

    for (const joint of this.joints) {
      const rbIndexA = joint.rigidbodyIndexA
      const rbIndexB = joint.rigidbodyIndexB

      // Validate indices
      if (
        rbIndexA < 0 ||
        rbIndexA >= this.ammoRigidbodies.length ||
        rbIndexB < 0 ||
        rbIndexB >= this.ammoRigidbodies.length
      ) {
        console.warn(`[Physics] Invalid joint indices: ${rbIndexA}, ${rbIndexB}`)
        continue
      }

      const bodyA = this.ammoRigidbodies[rbIndexA]
      const bodyB = this.ammoRigidbodies[rbIndexB]

      // Validate bodies exist
      if (!bodyA || !bodyB) {
        console.warn(`[Physics] Body not found for joint ${joint.name}: bodyA=${!!bodyA}, bodyB=${!!bodyB}`)
        continue
      }

      // CRITICAL: Compute joint frames using ACTUAL current body positions
      // Bodies have been repositioned to current bone poses, so we must use their current transforms
      // Joint frames must be computed relative to where bodies actually are, not bind pose

      // Get the actual current body transforms (after positionBodiesFromBones)
      const bodyATransform = bodyA.getWorldTransform()
      const bodyBTransform = bodyB.getWorldTransform()

      // Read current body positions and rotations
      const bodyAOrigin = bodyATransform.getOrigin()
      const bodyARotQuat = bodyATransform.getRotation()
      const bodyAPos = new Vec3(bodyAOrigin.x(), bodyAOrigin.y(), bodyAOrigin.z())
      const bodyARot = new Quat(bodyARotQuat.x(), bodyARotQuat.y(), bodyARotQuat.z(), bodyARotQuat.w())
      const bodyAMat = Mat4.fromPositionRotation(bodyAPos, bodyARot)

      const bodyBOrigin = bodyBTransform.getOrigin()
      const bodyBRotQuat = bodyBTransform.getRotation()
      const bodyBPos = new Vec3(bodyBOrigin.x(), bodyBOrigin.y(), bodyBOrigin.z())
      const bodyBRot = new Quat(bodyBRotQuat.x(), bodyBRotQuat.y(), bodyBRotQuat.z(), bodyBRotQuat.w())
      const bodyBMat = Mat4.fromPositionRotation(bodyBPos, bodyBRot)

      // Joint transform in world space (from PMX, bind pose)
      const scalingFactor = 1.0 // Assume 1.0 for now (can be adjusted if model is scaled)
      const jointRotQuat = Quat.fromEuler(joint.rotation.x, joint.rotation.y, joint.rotation.z)
      const jointPos = new Vec3(
        joint.position.x * scalingFactor,
        joint.position.y * scalingFactor,
        joint.position.z * scalingFactor
      )
      const jointTransform = Mat4.fromPositionRotation(jointPos, jointRotQuat)

      // Compute frames in body-local space
      // frameInA = inverse(bodyAWorld) × jointWorld
      // This transforms the joint world position to body A's local space
      const frameInAMat = bodyAMat.inverse().multiply(jointTransform)
      const framePosA = frameInAMat.getPosition()
      const frameRotA = frameInAMat.toQuat()

      // frameInB = inverse(bodyBWorld) × jointWorld
      // This transforms the joint world position to body B's local space
      const frameInBMat = bodyBMat.inverse().multiply(jointTransform)
      const framePosB = frameInBMat.getPosition()
      const frameRotB = frameInBMat.toQuat()

      // Create constraint frames for Ammo
      const frameInA = new Ammo.btTransform()
      frameInA.setIdentity()
      const pivotInA = new Ammo.btVector3(framePosA.x, framePosA.y, framePosA.z)
      frameInA.setOrigin(pivotInA)
      const quatA = new Ammo.btQuaternion(frameRotA.x, frameRotA.y, frameRotA.z, frameRotA.w)
      frameInA.setRotation(quatA)

      const frameInB = new Ammo.btTransform()
      frameInB.setIdentity()
      const pivotInB = new Ammo.btVector3(framePosB.x, framePosB.y, framePosB.z)
      frameInB.setOrigin(pivotInB)
      const quatB = new Ammo.btQuaternion(frameRotB.x, frameRotB.y, frameRotB.z, frameRotB.w)
      frameInB.setRotation(quatB)

      // Use btGeneric6DofSpringConstraint for spring support
      // useLinearReferenceFrameA = true (matches reference code)
      const constraint = new Ammo.btGeneric6DofSpringConstraint(bodyA, bodyB, frameInA, frameInB, true)

      // CRITICAL: Disable offset for constraint frame for MMD compatibility
      // The reference code sets m_useOffsetForConstraintFrame = false via heap manipulation
      // This is needed to match MMD physics behavior (Bullet 2.75 style)
      // We'll try to access this via Ammo API if available, otherwise we may need heap manipulation
      // For now, the constraint should work, but MMD compatibility may require this setting

      // Set ERP parameter (like reference code)
      for (let i = 0; i < 6; ++i) {
        constraint.setParam(2 /* BT_CONSTRAINT_STOP_ERP */, 0.475, i)
      }

      // Set linear limits (position constraints) - NOT scaled in reference code
      const lowerLinear = new Ammo.btVector3(joint.positionMin.x, joint.positionMin.y, joint.positionMin.z)
      const upperLinear = new Ammo.btVector3(joint.positionMax.x, joint.positionMax.y, joint.positionMax.z)
      constraint.setLinearLowerLimit(lowerLinear)
      constraint.setLinearUpperLimit(upperLinear)

      // Set angular limits (rotation constraints)
      // rotationMin/Max are already Vec3 (Euler angles in radians)
      const normalizeAngle = (angle: number): number => {
        const pi = Math.PI
        const twoPi = 2 * pi
        angle = angle % twoPi
        if (angle < -pi) {
          angle += twoPi
        } else if (angle > pi) {
          angle -= twoPi
        }
        return angle
      }

      const lowerAngular = new Ammo.btVector3(
        normalizeAngle(joint.rotationMin.x),
        normalizeAngle(joint.rotationMin.y),
        normalizeAngle(joint.rotationMin.z)
      )
      const upperAngular = new Ammo.btVector3(
        normalizeAngle(joint.rotationMax.x),
        normalizeAngle(joint.rotationMax.y),
        normalizeAngle(joint.rotationMax.z)
      )
      constraint.setAngularLowerLimit(lowerAngular)
      constraint.setAngularUpperLimit(upperAngular)

      // Set spring parameters (stiffness only - reference code doesn't use additional damping)
      // Linear springs: only enable if stiffness is non-zero (like reference code)
      if (joint.springPosition.x !== 0) {
        constraint.setStiffness(0, joint.springPosition.x)
        constraint.enableSpring(0, true)
      } else {
        constraint.enableSpring(0, false)
      }
      if (joint.springPosition.y !== 0) {
        constraint.setStiffness(1, joint.springPosition.y)
        constraint.enableSpring(1, true)
      } else {
        constraint.enableSpring(1, false)
      }
      if (joint.springPosition.z !== 0) {
        constraint.setStiffness(2, joint.springPosition.z)
        constraint.enableSpring(2, true)
      } else {
        constraint.enableSpring(2, false)
      }

      // Angular springs: always enable (like reference code)
      // springRotation is now Vec3 (stiffness coefficients)
      constraint.setStiffness(3, joint.springRotation.x)
      constraint.enableSpring(3, true)
      constraint.setStiffness(4, joint.springRotation.y)
      constraint.enableSpring(4, true)
      constraint.setStiffness(5, joint.springRotation.z)
      constraint.enableSpring(5, true)

      // Add constraint to world
      // In Bullet: addConstraint(constraint, disableCollisionsBetweenLinkedBodies)
      // Reference code uses collision: true, meaning collisions are NOT disabled
      // So disableCollisionsBetweenLinkedBodies should be false
      this.dynamicsWorld.addConstraint(constraint, false)

      // Store reference
      this.ammoConstraints.push(constraint)
      // Cleanup temporary objects
      Ammo.destroy(pivotInA)
      Ammo.destroy(pivotInB)
      Ammo.destroy(quatA)
      Ammo.destroy(quatB)
      Ammo.destroy(lowerLinear)
      Ammo.destroy(upperLinear)
      Ammo.destroy(lowerAngular)
      Ammo.destroy(upperAngular)
    }
  }

  // Main physics step: syncs bones to rigidbodies, simulates dynamics, solves constraints
  // Modifies boneWorldMatrices in-place for dynamic rigidbodies that drive bones
  step(dt: number, boneWorldMatrices: Float32Array, boneInverseBindMatrices: Float32Array, boneCount: number): void {
    // Wait for Ammo to initialize
    if (!this.ammoInitialized || !this.ammo || !this.dynamicsWorld) {
      return
    }

    // On first frame: Compute offsets, reposition bodies based on CURRENT bone transforms, then create joints
    if (this.firstFrame) {
      // Compute bodyOffsetMatrix if not done
      if (!this.rigidbodiesInitialized) {
        this.computeBodyOffsets(boneInverseBindMatrices, boneCount)
        this.rigidbodiesInitialized = true
      }

      // CRITICAL: Position bodies based on CURRENT bone poses (not bind pose!)
      // This ensures bodies are at correct positions before joints are created
      this.positionBodiesFromBones(boneWorldMatrices, boneCount)

      // NOW create joints with bodies at correct positions
      if (!this.jointsCreated) {
        this.createAmmoJoints()
        this.jointsCreated = true
      }

      // Refresh broadphase after creating joints
      if (this.dynamicsWorld.stepSimulation) {
        this.dynamicsWorld.stepSimulation(0, 0, 0)
      }

      this.firstFrame = false
    }

    // Normal simulation continues...
    // Step order matches MMD physics reference:
    // 1. Sync Static/Kinematic rigidbodies from bones (FollowBone bodies)
    this.syncFromBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)

    // 2. Step Ammo physics simulation
    // Limit dt to prevent huge jumps on first frame or when framerate spikes
    // Reference uses fixedTimeStep = 1/60 for MMD compatibility
    const clampedDt = Math.min(dt, 0.017) // Cap at ~60fps minimum
    this.stepAmmoPhysics(clampedDt)

    // 3. Update bone world matrices in-place for dynamic rigidbodies (Physics bodies)
    this.applyAmmoRigidbodiesToBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)
  }

  // Compute bodyOffsetMatrixInverse for all rigidbodies
  // Called once during initialization to compute the offset between body shape and bone bind space
  // Reference: node.computeBodyOffsetMatrix(bone.linkedBone.getAbsoluteInverseBindMatrix())
  // The reference uses the parsed position/rotation directly to create the node transform,
  // then computes: bodyOffsetMatrix = nodeTransform * boneInverseBindMatrix
  private computeBodyOffsets(boneInverseBindMatrices: Float32Array, boneCount: number): void {
    if (!this.ammo || !this.dynamicsWorld) return

    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]
      if (rb.boneIndex >= 0 && rb.boneIndex < boneCount) {
        const boneIdx = rb.boneIndex
        const invBindIdx = boneIdx * 16

        // Get bone inverse bind matrix
        const invBindMat = new Mat4(boneInverseBindMatrices.subarray(invBindIdx, invBindIdx + 16))

        // Compute shape transform in BONE-LOCAL space (shapeLocal)
        // shapeLocal = boneInverseBind × shapeWorldBind
        // This represents the shape offset relative to the bone in bind pose
        const shapeRotQuat = Quat.fromEuler(rb.shapeRotation.x, rb.shapeRotation.y, rb.shapeRotation.z)
        const shapeWorldBind = Mat4.fromPositionRotation(rb.shapePosition, shapeRotQuat)

        // CRITICAL: Reverse the multiplication order!
        // shapeLocal = boneInverseBind × shapeWorldBind (not shapeWorldBind × boneInverseBind)
        // This gives us the shape transform in bone-local space
        const bodyOffsetMatrix = invBindMat.multiply(shapeWorldBind)
        rb.bodyOffsetMatrixInverse = bodyOffsetMatrix.inverse()
      } else {
        // No bone linked - bodyOffsetMatrix is identity (body is independent)
        rb.bodyOffsetMatrixInverse = Mat4.identity()
      }
    }
  }

  // Position bodies based on CURRENT bone transforms (called on first frame only)
  // This ensures bodies are at correct positions before joints are created
  // Reference: MmdAmmoPhysicsModel.initialize() - positions bodies using current bone world matrices
  private positionBodiesFromBones(boneWorldMatrices: Float32Array, boneCount: number): void {
    if (!this.ammo || !this.dynamicsWorld) return

    const Ammo = this.ammo

    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]
      const ammoBody = this.ammoRigidbodies[i]
      if (!ammoBody || rb.boneIndex < 0 || rb.boneIndex >= boneCount) continue

      // Note: Static (FollowBone) bodies ARE repositioned here - they need to follow bones
      // They will be synced every frame in syncFromBones()

      const boneIdx = rb.boneIndex
      const worldMatIdx = boneIdx * 16

      // Get bone world matrix (CURRENT frame, not bind pose)
      const boneWorldMat = new Mat4(boneWorldMatrices.subarray(worldMatIdx, worldMatIdx + 16))

      // Compute node world matrix: boneWorldMatrix * bodyOffsetMatrix
      // CRITICAL: Reverse the multiplication order!
      // nodeWorld = boneWorld × shapeLocal (not shapeLocal × boneWorld)
      // bodyOffsetMatrix = shapeLocal (shape transform in bone-local space)
      const bodyOffsetMatrix = rb.bodyOffsetMatrixInverse.inverse()
      const nodeWorldMatrix = boneWorldMat.multiply(bodyOffsetMatrix)

      // CRITICAL: Apply model world matrix if model has root transform
      // Reference code does: nodeWorldMatrix.multiplyToRef(modelWorldMatrix, nodeWorldMatrix)
      // For now, assume model has no root transform (identity matrix)
      // If your model has scaling/rotation on the root, you MUST apply it here
      // const modelWorldMatrix = Mat4.identity() // TODO: Get actual model transform if needed
      // nodeWorldMatrix = modelWorldMatrix.multiply(nodeWorldMatrix)

      // Extract position and rotation
      const worldPos = nodeWorldMatrix.getPosition()
      const worldRot = nodeWorldMatrix.toQuat()

      // Update Ammo body transform
      const transform = new Ammo.btTransform()
      const pos = new Ammo.btVector3(worldPos.x, worldPos.y, worldPos.z)
      const quat = new Ammo.btQuaternion(worldRot.x, worldRot.y, worldRot.z, worldRot.w)

      transform.setOrigin(pos)
      transform.setRotation(quat)

      // Set body transform to current position
      // Reference: For kinematic bodies (Static/FollowBone and Kinematic), make kinematic once and set transform
      // For dynamic bodies, also set initial transform
      // Note: Static bodies are already set as kinematic in createAmmoRigidbodies(), but ensure it here too
      if (rb.type === RigidbodyType.Static || rb.type === RigidbodyType.Kinematic) {
        // Ensure kinematic flags are set (already set in createAmmoRigidbodies, but double-check)
        ammoBody.setCollisionFlags(ammoBody.getCollisionFlags() | 2) // CF_KINEMATIC_OBJECT
        ammoBody.setActivationState(4) // DISABLE_DEACTIVATION
      }

      ammoBody.setWorldTransform(transform)
      ammoBody.getMotionState().setWorldTransform(transform)

      // Clear velocities
      const zeroVec = new Ammo.btVector3(0, 0, 0)
      ammoBody.setLinearVelocity(zeroVec)
      ammoBody.setAngularVelocity(zeroVec)

      Ammo.destroy(pos)
      Ammo.destroy(quat)
      Ammo.destroy(zeroVec)
      Ammo.destroy(transform)
    }
  }

  // Sync Static (FollowBone) and Kinematic rigidbodies to follow bone transforms (called every frame)
  // Static bodies are FollowBone mode - they MUST follow bones, so treat them as kinematic
  // Dynamic bodies are physics-driven and don't need syncing from bones
  private syncFromBones(
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number
  ): void {
    if (!this.ammo || !this.dynamicsWorld) return

    const Ammo = this.ammo

    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]
      const ammoBody = this.ammoRigidbodies[i]
      if (!ammoBody) continue

      // Sync BOTH Static (FollowBone) AND Kinematic bodies - they both follow bones
      // Static = FollowBone (type 0) → must follow bones
      // Kinematic = PhysicsWithBone (type 2) → also follows bones
      if (
        (rb.type === RigidbodyType.Static || rb.type === RigidbodyType.Kinematic) &&
        rb.boneIndex >= 0 &&
        rb.boneIndex < boneCount
      ) {
        const boneIdx = rb.boneIndex
        const worldMatIdx = boneIdx * 16

        // Get bone world matrix
        const boneWorldMat = new Mat4(boneWorldMatrices.subarray(worldMatIdx, worldMatIdx + 16))

        // Compute node world matrix: boneWorldMatrix * bodyOffsetMatrix
        // CRITICAL: Reverse the multiplication order!
        // nodeWorld = boneWorld × shapeLocal (not shapeLocal × boneWorld)
        const bodyOffsetMatrix = rb.bodyOffsetMatrixInverse.inverse()
        const nodeWorldMatrix = boneWorldMat.multiply(bodyOffsetMatrix)

        // Extract position and rotation
        const worldPos = nodeWorldMatrix.getPosition()
        const worldRot = nodeWorldMatrix.toQuat()

        // Update Ammo body
        const transform = new Ammo.btTransform()
        const pos = new Ammo.btVector3(worldPos.x, worldPos.y, worldPos.z)
        const quat = new Ammo.btQuaternion(worldRot.x, worldRot.y, worldRot.z, worldRot.w)

        transform.setOrigin(pos)
        transform.setRotation(quat)

        ammoBody.setWorldTransform(transform)
        ammoBody.getMotionState().setWorldTransform(transform)

        // Clear velocities for kinematic
        const zeroVec = new Ammo.btVector3(0, 0, 0)
        ammoBody.setLinearVelocity(zeroVec)
        ammoBody.setAngularVelocity(zeroVec)

        Ammo.destroy(pos)
        Ammo.destroy(quat)
        Ammo.destroy(zeroVec)
        Ammo.destroy(transform)
      }
    }
  }

  // Step Ammo physics simulation
  // Reference code: "120 steps per second is recommended for better reproduction of MMD physics"
  // The stepSimulation signature is: stepSimulation(timeStep, maxSubSteps, fixedTimeStep)
  // - timeStep: total time to simulate (delta time from frame) - this is dt (variable)
  // - maxSubSteps: maximum number of substeps (Bullet will use as many as needed up to this)
  // - fixedTimeStep: size of each physics substep (1/120 for 120 steps per second = better MMD physics)
  //
  // Bullet internally divides dt into fixed substeps of size fixedTimeStep
  // So if dt = 0.016 (60fps) and fixedTimeStep = 1/120, Bullet will do 2 substeps
  private stepAmmoPhysics(dt: number): void {
    if (!this.ammo || !this.dynamicsWorld) return

    // Use 120 steps per second (1/120) for better MMD physics reproduction
    // Reference: "120 steps per second is recommended for better reproduction of MMD physics"
    const fixedTimeStep = 1 / 120 // 120 steps per second (better than 1/60 for MMD)
    const maxSubSteps = 10 // Maximum substeps (should be enough for 60fps -> 120 substeps)

    // dt is the variable delta time from the frame
    // Bullet will divide dt into fixed substeps of size fixedTimeStep
    // Example: dt=0.016 (60fps) with fixedTimeStep=1/120 will do 2 substeps
    this.dynamicsWorld.stepSimulation(dt, maxSubSteps, fixedTimeStep)
  }

  // Apply dynamic rigidbody world transforms to bone world matrices in-place
  // Matches reference code: bodyOffsetMatrixInverse * nodeWorldMatrix -> boneWorldMatrix
  // Reads directly from Ammo bodies, not from stored rigidbody state
  private applyAmmoRigidbodiesToBones(
    boneWorldMatrices: Float32Array,
    boneInverseBindMatrices: Float32Array,
    boneCount: number
  ): void {
    if (!this.ammo || !this.dynamicsWorld) return

    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]
      const ammoBody = this.ammoRigidbodies[i]
      if (!ammoBody) continue

      // Only dynamic rigidbodies drive bones (Static/Kinematic follow bones)
      if (rb.type === RigidbodyType.Dynamic && rb.boneIndex >= 0 && rb.boneIndex < boneCount) {
        const boneIdx = rb.boneIndex
        const worldMatIdx = boneIdx * 16

        // Read node world transform directly from Ammo body
        const transform = ammoBody.getWorldTransform()
        const origin = transform.getOrigin()
        const rotation = transform.getRotation()

        // Reference code: bodyOffsetMatrixInverse.multiplyToArray(nodeWorldMatrix, boneWorldMatrix, 0)
        // nodeWorldMatrix = Matrix.Compose(node.scaling, node.rotationQuaternion, node.position)
        // We use identity scaling (1,1,1) like reference
        const nodePos = new Vec3(origin.x(), origin.y(), origin.z())
        const nodeRot = new Quat(rotation.x(), rotation.y(), rotation.z(), rotation.w())
        const nodeWorldMatrix = Mat4.fromPositionRotation(nodePos, nodeRot)

        // Apply bodyOffsetMatrixInverse to get bone world matrix
        // CRITICAL: Reverse the multiplication order!
        // We have: nodeWorld = boneWorld × shapeLocal
        // Solving for boneWorld: boneWorld = nodeWorld × shapeLocal^-1
        // So: boneWorld = nodeWorld × bodyOffsetMatrixInverse (not bodyOffsetMatrixInverse × nodeWorld)
        const boneWorldMat = nodeWorldMatrix.multiply(rb.bodyOffsetMatrixInverse)

        // Validate the resulting matrix before applying (sanity check)
        const values = boneWorldMat.values
        if (!isNaN(values[0]) && !isNaN(values[15]) && Math.abs(values[0]) < 1e6 && Math.abs(values[15]) < 1e6) {
          // Write bone world matrix directly to the array in-place
          boneWorldMatrices.set(values, worldMatIdx)
        } else {
          console.warn(`[Physics] Invalid bone world matrix for rigidbody ${i} (${rb.name}), skipping update`)
        }
      }
    }
  }
}
