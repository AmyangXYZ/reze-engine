import { Quat, Vec3 } from "./math"

export enum RigidbodyShape {
  Sphere = 0,
  Box = 1,
  Capsule = 2,
}

export enum RigidbodyType {
  Static = 0, // Follows bone transform, no physics
  Dynamic = 1, // Full physics simulation
  Kinematic = 2, // Follows bone but can be moved by physics
}

export interface Rigidbody {
  name: string
  englishName: string
  boneIndex: number
  group: number
  collisionMask: number
  shape: RigidbodyShape
  size: Vec3 // Size parameters (PMX stores as HALF-EXTENTS, confirmed by reference code):
  //   - Sphere: x=radius (half-extent)
  //   - Box: PMX stores as (width/2, height/2, depth/2) = (x, y, z)
  //   - Capsule: x=radius (half-extent), y=height (full height)
  position: Vec3 // Position relative to bone or world space
  rotation: Vec3 // Rotation in Euler angles (radians)
  mass: number
  linearDamping: number
  angularDamping: number
  restitution: number // Bounciness (0-1)
  friction: number
  type: RigidbodyType
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
