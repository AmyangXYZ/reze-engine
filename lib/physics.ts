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

  constructor(rigidbodies: Rigidbody[], joints: Joint[] = []) {
    this.rigidbodies = rigidbodies
    this.joints = joints
  }

  getRigidbodies(): Rigidbody[] {
    return this.rigidbodies
  }

  getJoints(): Joint[] {
    return this.joints
  }

  // Update rigidbody transforms from bone animations
  // Static/Kinematic: computed from bones (these are the constraints/collision geometry for solver)
  // Dynamic: not updated here; solver will compute and update rb.position/rb.rotation
  // Joints: position/rotation are world space; typically no updates needed as they bind to rigidbodies
  //         via rigidbodyIndexA/B indices, and solver uses the connected rigidbodies' current transforms
  update(boneWorldMatrices: Float32Array, boneInverseBindMatrices: Float32Array, boneCount: number): void {
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
}
