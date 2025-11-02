import { Quat, Vec3 } from "./math"

export interface Rigidbody {
  boneIndex: number
  radius: number
  group: number
  collisionMask: number
  position: Vec3
}

export interface Joint {
  name: string
  englishName: string
  type: number // Joint type (uint8)
  rigidbodyIndexA: number // Index of first rigidbody (-1 for none)
  rigidbodyIndexB: number // Index of second rigidbody (-1 for none)
  position: Vec3 // Position (world space)
  rotation: Quat // Rotation (Euler angles)
  positionMin: Vec3 // Position constraint minimum
  positionMax: Vec3 // Position constraint maximum
  rotationMin: Quat // Rotation constraint minimum (Euler)
  rotationMax: Quat // Rotation constraint maximum (Euler)
  springPosition: Vec3 // Spring position parameters
  springRotation: Quat // Spring rotation parameters
}

export class Physics {
  constructor() {}
}
