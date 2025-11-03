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
}

export class Physics {
  private definitions: Rigidbody[]

  constructor(rigidbodies: Rigidbody[]) {
    this.definitions = rigidbodies
  }

  getDefinitions(): Rigidbody[] {
    return this.definitions
  }

  update(boneWorldMatrices: Float32Array, boneInverseBindMatrices: Float32Array, boneCount: number): void {
    for (const rb of this.definitions) {
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
