import { Vec3, Mat4 } from "../math"

export enum RigidbodyShape {
  Sphere = 0,
  Box = 1,
  Capsule = 2,
}

// PMX physics mode: 0 = follow bone, 1 = physics, 2 = physics + bone-position
// alignment. Mode 2 is DYNAMIC — the loader maps it to Dynamic with
// `aligned: true` (bone gets rotation only, body position re-pins to the
// animated bone each frame). Mapping the raw byte 1:1 onto this enum froze
// mode-2 bodies (most modern cloth rigs, and every 胸_回転 breast body).
export enum RigidbodyType {
  Static = 0, // follows bone (anchor)
  Dynamic = 1,
  // The loader no longer emits this — it maps mode 2 to Dynamic, which is the
  // fix above. The member stays because the physics step still names it when
  // asking "does this body follow its bone", and a host may set it directly.
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
  // PMX mode 2: dynamic body whose bone takes rotation only; the body's
  // position re-pins to the animated bone each frame.
  aligned?: boolean
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

