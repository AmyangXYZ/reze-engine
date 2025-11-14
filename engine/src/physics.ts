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
  shapePosition: Vec3 // Bind pose world space position from PMX
  shapeRotation: Vec3 // Bind pose world space rotation (Euler angles) from PMX
  mass: number
  linearDamping: number
  angularDamping: number
  restitution: number
  friction: number
  type: RigidbodyType
  bodyOffsetMatrixInverse: Mat4 // Inverse of body offset matrix, used to sync rigidbody to bone
  bodyOffsetMatrix?: Mat4 // Cached non-inverse for performance (computed once during initialization)
}

export interface Joint {
  name: string
  englishName: string
  type: number
  rigidbodyIndexA: number
  rigidbodyIndexB: number
  position: Vec3
  rotation: Vec3 // Euler angles in radians
  positionMin: Vec3
  positionMax: Vec3
  rotationMin: Vec3 // Euler angles in radians
  rotationMax: Vec3 // Euler angles in radians
  springPosition: Vec3
  springRotation: Vec3 // Spring stiffness values
}

export class Physics {
  private rigidbodies: Rigidbody[]
  private joints: Joint[]
  private gravity: Vec3 = new Vec3(0, -98, 0) // Gravity acceleration (cm/s²), MMD-style default
  private ammoInitialized = false
  private ammoPromise: Promise<AmmoInstance> | null = null
  private ammo: AmmoInstance | null = null
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private dynamicsWorld: any = null // btDiscreteDynamicsWorld
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ammoRigidbodies: any[] = [] // btRigidBody instances
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private ammoConstraints: any[] = [] // btTypedConstraint instances
  private rigidbodiesInitialized = false // bodyOffsetMatrixInverse computed and bodies positioned
  private jointsCreated = false // Joints delayed until after rigidbodies are positioned
  private firstFrame = true // Needed to reposition bodies before creating joints
  private forceDisableOffsetForConstraintFrame = true // MMD compatibility (Bullet 2.75 behavior)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private zeroVector: any = null // Cached zero vector for velocity clearing

  constructor(rigidbodies: Rigidbody[], joints: Joint[] = []) {
    this.rigidbodies = rigidbodies
    this.joints = joints
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

  getRigidbodyTransforms(): Array<{ position: Vec3; rotation: Quat }> {
    const transforms: Array<{ position: Vec3; rotation: Quat }> = []

    if (!this.ammo || !this.ammoRigidbodies.length) {
      for (let i = 0; i < this.rigidbodies.length; i++) {
        transforms.push({
          position: new Vec3(
            this.rigidbodies[i].shapePosition.x,
            this.rigidbodies[i].shapePosition.y,
            this.rigidbodies[i].shapePosition.z
          ),
          rotation: Quat.fromEuler(
            this.rigidbodies[i].shapeRotation.x,
            this.rigidbodies[i].shapeRotation.y,
            this.rigidbodies[i].shapeRotation.z
          ),
        })
      }
      return transforms
    }

    for (let i = 0; i < this.ammoRigidbodies.length; i++) {
      const ammoBody = this.ammoRigidbodies[i]
      if (!ammoBody) {
        const rb = this.rigidbodies[i]
        transforms.push({
          position: new Vec3(rb.shapePosition.x, rb.shapePosition.y, rb.shapePosition.z),
          rotation: Quat.fromEuler(rb.shapeRotation.x, rb.shapeRotation.y, rb.shapeRotation.z),
        })
        continue
      }

      const transform = ammoBody.getWorldTransform()
      const origin = transform.getOrigin()
      const rotQuat = transform.getRotation()

      transforms.push({
        position: new Vec3(origin.x(), origin.y(), origin.z()),
        rotation: new Quat(rotQuat.x(), rotQuat.y(), rotQuat.z(), rotQuat.w()),
      })
    }

    return transforms
  }

  private createAmmoWorld(): void {
    if (!this.ammo) return

    const Ammo = this.ammo

    const collisionConfiguration = new Ammo.btDefaultCollisionConfiguration()
    const dispatcher = new Ammo.btCollisionDispatcher(collisionConfiguration)
    const overlappingPairCache = new Ammo.btDbvtBroadphase()
    const solver = new Ammo.btSequentialImpulseConstraintSolver()

    this.dynamicsWorld = new Ammo.btDiscreteDynamicsWorld(
      dispatcher,
      overlappingPairCache,
      solver,
      collisionConfiguration
    )

    const gravityVec = new Ammo.btVector3(this.gravity.x, this.gravity.y, this.gravity.z)
    this.dynamicsWorld.setGravity(gravityVec)
    Ammo.destroy(gravityVec)

    this.createAmmoRigidbodies()
  }

  private createAmmoRigidbodies(): void {
    if (!this.ammo || !this.dynamicsWorld) return

    const Ammo = this.ammo
    this.ammoRigidbodies = []

    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      let shape: any = null
      const size = rb.size

      switch (rb.shape) {
        case RigidbodyShape.Sphere:
          const radius = size.x
          shape = new Ammo.btSphereShape(radius)
          break
        case RigidbodyShape.Box:
          const sizeVector = new Ammo.btVector3(size.x, size.y, size.z)
          shape = new Ammo.btBoxShape(sizeVector)
          Ammo.destroy(sizeVector)
          break
        case RigidbodyShape.Capsule:
          const capsuleRadius = size.x
          const capsuleHalfHeight = size.y
          shape = new Ammo.btCapsuleShape(capsuleRadius, capsuleHalfHeight)
          break
        default:
          const defaultHalfExtents = new Ammo.btVector3(size.x / 2, size.y / 2, size.z / 2)
          shape = new Ammo.btBoxShape(defaultHalfExtents)
          Ammo.destroy(defaultHalfExtents)
          break
      }

      // Bodies must start at correct position to avoid explosions when joints are created
      const transform = new Ammo.btTransform()
      transform.setIdentity()

      const shapePos = new Ammo.btVector3(rb.shapePosition.x, rb.shapePosition.y, rb.shapePosition.z)
      transform.setOrigin(shapePos)
      Ammo.destroy(shapePos)

      const shapeRotQuat = Quat.fromEuler(rb.shapeRotation.x, rb.shapeRotation.y, rb.shapeRotation.z)
      const quat = new Ammo.btQuaternion(shapeRotQuat.x, shapeRotQuat.y, shapeRotQuat.z, shapeRotQuat.w)
      transform.setRotation(quat)
      Ammo.destroy(quat)

      // All types use the same motionState constructor
      const motionState = new Ammo.btDefaultMotionState(transform)
      const mass = rb.type === RigidbodyType.Dynamic ? rb.mass : 0
      const isDynamic = rb.type === RigidbodyType.Dynamic

      const localInertia = new Ammo.btVector3(0, 0, 0)
      if (isDynamic && mass > 0) {
        shape.calculateLocalInertia(mass, localInertia)
      }

      const rbInfo = new Ammo.btRigidBodyConstructionInfo(mass, motionState, shape, localInertia)
      rbInfo.set_m_restitution(rb.restitution)
      rbInfo.set_m_friction(rb.friction)
      rbInfo.set_m_linearDamping(rb.linearDamping)
      rbInfo.set_m_angularDamping(rb.angularDamping)

      const body = new Ammo.btRigidBody(rbInfo)

      body.setSleepingThresholds(0.0, 0.0)

      // Static (FollowBone) should be kinematic, not static - must follow bones
      if (rb.type === RigidbodyType.Static || rb.type === RigidbodyType.Kinematic) {
        body.setCollisionFlags(body.getCollisionFlags() | 2) // CF_KINEMATIC_OBJECT
        body.setActivationState(4) // DISABLE_DEACTIVATION
      }

      const collisionGroup = 1 << rb.group
      const collisionMask = rb.collisionMask

      const isZeroVolume =
        (rb.shape === RigidbodyShape.Sphere && rb.size.x === 0) ||
        (rb.shape === RigidbodyShape.Box && (rb.size.x === 0 || rb.size.y === 0 || rb.size.z === 0)) ||
        (rb.shape === RigidbodyShape.Capsule && (rb.size.x === 0 || rb.size.y === 0))

      if (collisionMask === 0 || isZeroVolume) {
        body.setCollisionFlags(body.getCollisionFlags() | 4) // CF_NO_CONTACT_RESPONSE
      }

      this.dynamicsWorld.addRigidBody(body, collisionGroup, collisionMask)

      this.ammoRigidbodies.push(body)

      Ammo.destroy(rbInfo)
      Ammo.destroy(localInertia)
    }
  }

  private createAmmoJoints(): void {
    if (!this.ammo || !this.dynamicsWorld || this.ammoRigidbodies.length === 0) return

    const Ammo = this.ammo
    this.ammoConstraints = []

    for (const joint of this.joints) {
      const rbIndexA = joint.rigidbodyIndexA
      const rbIndexB = joint.rigidbodyIndexB

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

      if (!bodyA || !bodyB) {
        console.warn(`[Physics] Body not found for joint ${joint.name}: bodyA=${!!bodyA}, bodyB=${!!bodyB}`)
        continue
      }

      // Compute joint frames using actual current body positions (after repositioning)
      const bodyATransform = bodyA.getWorldTransform()
      const bodyBTransform = bodyB.getWorldTransform()

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

      const scalingFactor = 1.0
      const jointRotQuat = Quat.fromEuler(joint.rotation.x, joint.rotation.y, joint.rotation.z)
      const jointPos = new Vec3(
        joint.position.x * scalingFactor,
        joint.position.y * scalingFactor,
        joint.position.z * scalingFactor
      )
      const jointTransform = Mat4.fromPositionRotation(jointPos, jointRotQuat)

      // Transform joint world position to body A's local space
      const frameInAMat = bodyAMat.inverse().multiply(jointTransform)
      const framePosA = frameInAMat.getPosition()
      const frameRotA = frameInAMat.toQuat()

      // Transform joint world position to body B's local space
      const frameInBMat = bodyBMat.inverse().multiply(jointTransform)
      const framePosB = frameInBMat.getPosition()
      const frameRotB = frameInBMat.toQuat()

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

      const useLinearReferenceFrameA = true
      const constraint = new Ammo.btGeneric6DofSpringConstraint(
        bodyA,
        bodyB,
        frameInA,
        frameInB,
        useLinearReferenceFrameA
      )

      // Disable offset for constraint frame for MMD compatibility (Bullet 2.75 behavior)
      if (this.forceDisableOffsetForConstraintFrame) {
        let jointPtr: number | undefined
        if (typeof Ammo.getPointer === "function") {
          jointPtr = Ammo.getPointer(constraint)
        } else {
          const constraintWithPtr = constraint as { ptr?: number }
          jointPtr = constraintWithPtr.ptr
        }

        if (jointPtr !== undefined && Ammo.HEAP8) {
          const heap8 = Ammo.HEAP8 as Uint8Array
          // jointPtr + 1300 = m_useLinearReferenceFrameA, jointPtr + 1301 = m_useOffsetForConstraintFrame
          if (heap8[jointPtr + 1300] === (useLinearReferenceFrameA ? 1 : 0) && heap8[jointPtr + 1301] === 1) {
            heap8[jointPtr + 1301] = 0
          }
        }
      }

      for (let i = 0; i < 6; ++i) {
        constraint.setParam(2, 0.475, i) // BT_CONSTRAINT_STOP_ERP
      }

      const lowerLinear = new Ammo.btVector3(joint.positionMin.x, joint.positionMin.y, joint.positionMin.z)
      const upperLinear = new Ammo.btVector3(joint.positionMax.x, joint.positionMax.y, joint.positionMax.z)
      constraint.setLinearLowerLimit(lowerLinear)
      constraint.setLinearUpperLimit(upperLinear)

      const lowerAngular = new Ammo.btVector3(
        this.normalizeAngle(joint.rotationMin.x),
        this.normalizeAngle(joint.rotationMin.y),
        this.normalizeAngle(joint.rotationMin.z)
      )
      const upperAngular = new Ammo.btVector3(
        this.normalizeAngle(joint.rotationMax.x),
        this.normalizeAngle(joint.rotationMax.y),
        this.normalizeAngle(joint.rotationMax.z)
      )
      constraint.setAngularLowerLimit(lowerAngular)
      constraint.setAngularUpperLimit(upperAngular)

      // Linear springs: only enable if stiffness is non-zero
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

      // Angular springs: always enable
      constraint.setStiffness(3, joint.springRotation.x)
      constraint.enableSpring(3, true)
      constraint.setStiffness(4, joint.springRotation.y)
      constraint.enableSpring(4, true)
      constraint.setStiffness(5, joint.springRotation.z)
      constraint.enableSpring(5, true)

      this.dynamicsWorld.addConstraint(constraint, false)

      this.ammoConstraints.push(constraint)
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

  // Normalize angle to [-π, π] range
  private normalizeAngle(angle: number): number {
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

  // Reset physics state (reposition bodies, clear velocities)
  // Following babylon-mmd pattern: initialize all rigid body positions from current bone poses
  // Call this when starting a new animation to prevent physics instability from sudden pose changes
  reset(boneWorldMatrices: Float32Array, boneInverseBindMatrices: Float32Array): void {
    if (!this.ammoInitialized || !this.ammo || !this.dynamicsWorld) {
      return
    }

    const boneCount = boneWorldMatrices.length / 16
    const Ammo = this.ammo

    // Ensure body offsets are computed
    if (!this.rigidbodiesInitialized) {
      this.computeBodyOffsets(boneInverseBindMatrices, boneCount)
      this.rigidbodiesInitialized = true
    }

    // Reposition ALL rigid bodies from current bone poses (like babylon-mmd initialize)
    // This ensures all bodies are correctly positioned before physics starts
    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]
      const ammoBody = this.ammoRigidbodies[i]
      if (!ammoBody || rb.boneIndex < 0 || rb.boneIndex >= boneCount) continue

      const boneIdx = rb.boneIndex
      const worldMatIdx = boneIdx * 16

      // Get bone world matrix
      const boneWorldMat = new Mat4(boneWorldMatrices.subarray(worldMatIdx, worldMatIdx + 16))

      // Compute body world matrix: bodyWorld = boneWorld × bodyOffsetMatrix
      // (like babylon-mmd: bodyWorldMatrix = bodyOffsetMatrix.multiplyToRef(bodyWorldMatrix))
      const bodyOffsetMatrix = rb.bodyOffsetMatrix || rb.bodyOffsetMatrixInverse.inverse()
      const bodyWorldMatrix = boneWorldMat.multiply(bodyOffsetMatrix)

      const worldPos = bodyWorldMatrix.getPosition()
      const worldRot = bodyWorldMatrix.toQuat()

      // Set transform matrix
      const transform = new Ammo.btTransform()
      const pos = new Ammo.btVector3(worldPos.x, worldPos.y, worldPos.z)
      const quat = new Ammo.btQuaternion(worldRot.x, worldRot.y, worldRot.z, worldRot.w)

      transform.setOrigin(pos)
      transform.setRotation(quat)

      ammoBody.setWorldTransform(transform)
      ammoBody.getMotionState().setWorldTransform(transform)

      // Clear velocities for all rigidbodies
      if (!this.zeroVector) {
        this.zeroVector = new Ammo.btVector3(0, 0, 0)
      }
      ammoBody.setLinearVelocity(this.zeroVector)
      ammoBody.setAngularVelocity(this.zeroVector)

      // Explicitly activate dynamic rigidbodies after reset (wake them up)
      // This is critical for dress pieces and other dynamic bodies to prevent teleporting
      if (rb.type === RigidbodyType.Dynamic) {
        ammoBody.activate(true) // Wake up the body
      }

      Ammo.destroy(pos)
      Ammo.destroy(quat)
    }

    // Step simulation once to stabilize (like babylon-mmd)
    if (this.dynamicsWorld.stepSimulation) {
      this.dynamicsWorld.stepSimulation(0, 0, 0)
    }
  }

  // Syncs bones to rigidbodies, simulates dynamics, solves constraints
  // Modifies boneWorldMatrices in-place for dynamic rigidbodies that drive bones
  step(dt: number, boneWorldMatrices: Float32Array, boneInverseBindMatrices: Float32Array): void {
    // Wait for Ammo to initialize
    if (!this.ammoInitialized || !this.ammo || !this.dynamicsWorld) {
      return
    }

    const boneCount = boneWorldMatrices.length / 16

    if (this.firstFrame) {
      if (!this.rigidbodiesInitialized) {
        this.computeBodyOffsets(boneInverseBindMatrices, boneCount)
        this.rigidbodiesInitialized = true
      }

      // Position bodies based on current bone poses (not bind pose) before creating joints
      this.positionBodiesFromBones(boneWorldMatrices, boneCount)

      if (!this.jointsCreated) {
        this.createAmmoJoints()
        this.jointsCreated = true
      }

      if (this.dynamicsWorld.stepSimulation) {
        this.dynamicsWorld.stepSimulation(0, 0, 0)
      }

      this.firstFrame = false
    }

    // Step order: 1) Sync Static/Kinematic from bones, 2) Step physics, 3) Apply dynamic to bones
    this.syncFromBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)

    this.stepAmmoPhysics(dt)

    this.applyAmmoRigidbodiesToBones(boneWorldMatrices, boneInverseBindMatrices, boneCount)
  }

  // Compute bodyOffsetMatrixInverse for all rigidbodies (called once during initialization)
  private computeBodyOffsets(boneInverseBindMatrices: Float32Array, boneCount: number): void {
    if (!this.ammo || !this.dynamicsWorld) return

    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]
      if (rb.boneIndex >= 0 && rb.boneIndex < boneCount) {
        const boneIdx = rb.boneIndex
        const invBindIdx = boneIdx * 16

        const invBindMat = new Mat4(boneInverseBindMatrices.subarray(invBindIdx, invBindIdx + 16))

        // Compute shape transform in bone-local space: shapeLocal = boneInverseBind × shapeWorldBind
        const shapeRotQuat = Quat.fromEuler(rb.shapeRotation.x, rb.shapeRotation.y, rb.shapeRotation.z)
        const shapeWorldBind = Mat4.fromPositionRotation(rb.shapePosition, shapeRotQuat)

        // shapeLocal = boneInverseBind × shapeWorldBind (not shapeWorldBind × boneInverseBind)
        const bodyOffsetMatrix = invBindMat.multiply(shapeWorldBind)
        rb.bodyOffsetMatrixInverse = bodyOffsetMatrix.inverse()
        rb.bodyOffsetMatrix = bodyOffsetMatrix // Cache non-inverse to avoid expensive inverse() calls
      } else {
        rb.bodyOffsetMatrixInverse = Mat4.identity()
        rb.bodyOffsetMatrix = Mat4.identity() // Cache non-inverse
      }
    }
  }

  // Position bodies based on current bone transforms (called on first frame only)
  private positionBodiesFromBones(boneWorldMatrices: Float32Array, boneCount: number): void {
    if (!this.ammo || !this.dynamicsWorld) return

    const Ammo = this.ammo

    for (let i = 0; i < this.rigidbodies.length; i++) {
      const rb = this.rigidbodies[i]
      const ammoBody = this.ammoRigidbodies[i]
      if (!ammoBody || rb.boneIndex < 0 || rb.boneIndex >= boneCount) continue

      const boneIdx = rb.boneIndex
      const worldMatIdx = boneIdx * 16

      const boneWorldMat = new Mat4(boneWorldMatrices.subarray(worldMatIdx, worldMatIdx + 16))

      // nodeWorld = boneWorld × shapeLocal (not shapeLocal × boneWorld)
      const bodyOffsetMatrix = rb.bodyOffsetMatrix || rb.bodyOffsetMatrixInverse.inverse()
      const nodeWorldMatrix = boneWorldMat.multiply(bodyOffsetMatrix)

      const worldPos = nodeWorldMatrix.getPosition()
      const worldRot = nodeWorldMatrix.toQuat()

      const transform = new Ammo.btTransform()
      const pos = new Ammo.btVector3(worldPos.x, worldPos.y, worldPos.z)
      const quat = new Ammo.btQuaternion(worldRot.x, worldRot.y, worldRot.z, worldRot.w)

      transform.setOrigin(pos)
      transform.setRotation(quat)

      if (rb.type === RigidbodyType.Static || rb.type === RigidbodyType.Kinematic) {
        ammoBody.setCollisionFlags(ammoBody.getCollisionFlags() | 2) // CF_KINEMATIC_OBJECT
        ammoBody.setActivationState(4) // DISABLE_DEACTIVATION
      }

      ammoBody.setWorldTransform(transform)
      ammoBody.getMotionState().setWorldTransform(transform)

      if (!this.zeroVector) {
        this.zeroVector = new Ammo.btVector3(0, 0, 0)
      }
      ammoBody.setLinearVelocity(this.zeroVector)
      ammoBody.setAngularVelocity(this.zeroVector)

      Ammo.destroy(pos)
      Ammo.destroy(quat)
      Ammo.destroy(transform)
    }
  }

  // Sync Static (FollowBone) and Kinematic rigidbodies to follow bone transforms
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

      // Sync both Static (FollowBone) and Kinematic bodies - they both follow bones
      if (
        (rb.type === RigidbodyType.Static || rb.type === RigidbodyType.Kinematic) &&
        rb.boneIndex >= 0 &&
        rb.boneIndex < boneCount
      ) {
        const boneIdx = rb.boneIndex
        const worldMatIdx = boneIdx * 16

        const boneWorldMat = new Mat4(boneWorldMatrices.subarray(worldMatIdx, worldMatIdx + 16))

        // nodeWorld = boneWorld × shapeLocal (not shapeLocal × boneWorld)
        const bodyOffsetMatrix = rb.bodyOffsetMatrix || rb.bodyOffsetMatrixInverse.inverse()
        const nodeWorldMatrix = boneWorldMat.multiply(bodyOffsetMatrix)

        const worldPos = nodeWorldMatrix.getPosition()
        const worldRot = nodeWorldMatrix.toQuat()

        const transform = new Ammo.btTransform()
        const pos = new Ammo.btVector3(worldPos.x, worldPos.y, worldPos.z)
        const quat = new Ammo.btQuaternion(worldRot.x, worldRot.y, worldRot.z, worldRot.w)

        transform.setOrigin(pos)
        transform.setRotation(quat)

        ammoBody.setWorldTransform(transform)
        ammoBody.getMotionState().setWorldTransform(transform)

        if (!this.zeroVector) {
          this.zeroVector = new Ammo.btVector3(0, 0, 0)
        }
        ammoBody.setLinearVelocity(this.zeroVector)
        ammoBody.setAngularVelocity(this.zeroVector)

        Ammo.destroy(pos)
        Ammo.destroy(quat)
        Ammo.destroy(transform)
      }
    }
  }

  // Step Ammo physics simulation
  private stepAmmoPhysics(dt: number): void {
    if (!this.ammo || !this.dynamicsWorld) return

    const fixedTimeStep = 1 / 75
    const maxSubSteps = 10

    this.dynamicsWorld.stepSimulation(dt, maxSubSteps, fixedTimeStep)
  }

  // Apply dynamic rigidbody world transforms to bone world matrices in-place
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

        const transform = ammoBody.getWorldTransform()
        const origin = transform.getOrigin()
        const rotation = transform.getRotation()

        const nodePos = new Vec3(origin.x(), origin.y(), origin.z())
        const nodeRot = new Quat(rotation.x(), rotation.y(), rotation.z(), rotation.w())
        const nodeWorldMatrix = Mat4.fromPositionRotation(nodePos, nodeRot)

        // boneWorld = nodeWorld × bodyOffsetMatrixInverse (not bodyOffsetMatrixInverse × nodeWorld)
        const boneWorldMat = nodeWorldMatrix.multiply(rb.bodyOffsetMatrixInverse)

        const values = boneWorldMat.values
        if (!isNaN(values[0]) && !isNaN(values[15]) && Math.abs(values[0]) < 1e6 && Math.abs(values[15]) < 1e6) {
          boneWorldMatrices.set(values, worldMatIdx)
        } else {
          console.warn(`[Physics] Invalid bone world matrix for rigidbody ${i} (${rb.name}), skipping update`)
        }
      }
    }
  }
}
