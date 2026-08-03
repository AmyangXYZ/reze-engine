import { Mat4, Quat } from "../math"
import { RigidbodyType, RigidbodyShape, type Rigidbody } from "./types"

// SoA storage for all rigid bodies. Per-body state, constants, bone-coupling
// matrices, and a per-step AABB.
export class RigidBodyStore {
  readonly count: number

  readonly positions: Float32Array // 3*N
  readonly orientations: Float32Array // 4*N (xyzw)
  readonly linearVelocities: Float32Array // 3*N
  readonly angularVelocities: Float32Array // 3*N

  readonly invMass: Float32Array // N (0 for static / kinematic)
  // Full anisotropic inertia. Local diagonal (body frame, Bullet's shape
  // formulas) plus the per-substep world tensor I⁻¹ = R·diag·Rᵀ (9 floats
  // row-major, symmetric). The old scalar approximation deposited constraint
  // impulses into rotational modes at the wrong rates on elongated capsules
  // (5:1 skirt/hair bodies are ~25× anisotropic), leaving cloth with several
  // times the kinetic energy real Bullet retains — visible as perpetual boil.
  readonly invInertiaLocal: Float32Array // 3*N
  readonly invInertiaWorld: Float32Array // 9*N
  readonly linearDamping: Float32Array
  readonly angularDamping: Float32Array
  readonly type: Uint8Array
  // PMX mode-2 bodies: dynamic, but the bone takes rotation only and the
  // body position re-pins to the animated bone each frame.
  readonly aligned: Uint8Array
  readonly boneIndex: Int32Array
  readonly friction: Float32Array
  readonly restitution: Float32Array

  // PMX has 16 collision groups. `collisionGroup[i]` is a single-bit set;
  // `willCollideMask[i]` is the 16-bit set of groups body i collides with.
  readonly collisionGroup: Uint16Array
  readonly willCollideMask: Uint16Array

  readonly shape: Uint8Array
  readonly size: Float32Array // 3*N (semantics depend on shape)

  readonly aabbMin: Float32Array // 3*N
  readonly aabbMax: Float32Array // 3*N

  // bodyOffsetMatrix[i] = boneInverseBind · shapeWorldBind.
  // bodyWorld = boneWorld · bodyOffsetMatrix; boneWorld = bodyWorld · bodyOffsetInverse.
  readonly bodyOffsetMatrix: Float32Array // 16*N column-major
  readonly bodyOffsetInverse: Float32Array // 16*N column-major
  private boneOffsetsReady = false

  // Flat list of (i, j) pairs that survive the static-static + group/mask
  // filter. None of those inputs change after construction, so building this
  // once collapses 60k pair tests/step (349 bodies) down to a few thousand.
  // Built lazily on first access.
  private collisionPairs: Uint16Array | null = null

  /** Index of the built-in floor body (see RezePhysics constructor), -1 if none.
   *  Excluded from the pair list; findContacts gives it a dedicated plane pass. */
  groundIndex = -1

  constructor(rigidbodies: Rigidbody[]) {
    const N = rigidbodies.length
    this.count = N

    this.positions = new Float32Array(N * 3)
    this.orientations = new Float32Array(N * 4)
    this.linearVelocities = new Float32Array(N * 3)
    this.angularVelocities = new Float32Array(N * 3)
    this.invMass = new Float32Array(N)
    this.invInertiaLocal = new Float32Array(N * 3)
    this.invInertiaWorld = new Float32Array(N * 9)
    this.linearDamping = new Float32Array(N)
    this.angularDamping = new Float32Array(N)
    this.type = new Uint8Array(N)
    this.aligned = new Uint8Array(N)
    this.boneIndex = new Int32Array(N)
    this.bodyOffsetMatrix = new Float32Array(N * 16)
    this.bodyOffsetInverse = new Float32Array(N * 16)
    this.friction = new Float32Array(N)
    this.restitution = new Float32Array(N)
    this.collisionGroup = new Uint16Array(N)
    this.willCollideMask = new Uint16Array(N)
    this.shape = new Uint8Array(N)
    this.size = new Float32Array(N * 3)
    this.aabbMin = new Float32Array(N * 3)
    this.aabbMax = new Float32Array(N * 3)

    for (let i = 0; i < N; i++) {
      const rb = rigidbodies[i]
      const i3 = i * 3
      const i4 = i * 4

      this.positions[i3 + 0] = rb.shapePosition.x
      this.positions[i3 + 1] = rb.shapePosition.y
      this.positions[i3 + 2] = rb.shapePosition.z

      const q = Quat.fromEuler(rb.shapeRotation.x, rb.shapeRotation.y, rb.shapeRotation.z)
      this.orientations[i4 + 0] = q.x
      this.orientations[i4 + 1] = q.y
      this.orientations[i4 + 2] = q.z
      this.orientations[i4 + 3] = q.w

      const dynamic = rb.type === RigidbodyType.Dynamic && rb.mass > 0
      this.invMass[i] = dynamic ? 1 / rb.mass : 0
      if (dynamic) computeLocalInvInertia(rb, this.invInertiaLocal, i * 3)
      this.linearDamping[i] = rb.linearDamping
      this.angularDamping[i] = rb.angularDamping
      this.type[i] = rb.type
      this.aligned[i] = rb.aligned ? 1 : 0
      this.boneIndex[i] = rb.boneIndex
      this.friction[i] = rb.friction
      this.restitution[i] = rb.restitution
      this.collisionGroup[i] = 1 << (rb.group & 0xf)
      this.willCollideMask[i] = rb.collisionMask & 0xffff
      this.shape[i] = rb.shape
      this.size[i * 3 + 0] = rb.size.x
      this.size[i * 3 + 1] = rb.size.y
      this.size[i * 3 + 2] = rb.size.z
    }
  }

  // Refresh I⁻¹_world = R·diag(invInertiaLocal)·Rᵀ for every dynamic body.
  // Called once per substep before constraint setup (orientations are
  // constant during a solve).
  updateInvInertiaWorld(): void {
    const N = this.count
    const ori = this.orientations
    const local = this.invInertiaLocal
    const W = this.invInertiaWorld
    const invMass = this.invMass

    for (let i = 0; i < N; i++) {
      if (invMass[i] <= 0) continue
      const i3 = i * 3
      const i4 = i * 4
      const i9 = i * 9
      const qx = ori[i4 + 0], qy = ori[i4 + 1], qz = ori[i4 + 2], qw = ori[i4 + 3]
      const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz
      const xx = qx * x2, yy = qy * y2, zz = qz * z2
      const xy = qx * y2, xz = qx * z2, yz = qy * z2
      const wx = qw * x2, wy = qw * y2, wz = qw * z2
      // R columns (column-major rotation matrix)
      const r00 = 1 - (yy + zz), r01 = xy - wz, r02 = xz + wy
      const r10 = xy + wz, r11 = 1 - (xx + zz), r12 = yz - wx
      const r20 = xz - wy, r21 = yz + wx, r22 = 1 - (xx + yy)
      const d0 = local[i3 + 0], d1 = local[i3 + 1], d2 = local[i3 + 2]
      // W = R·diag·Rᵀ (symmetric)
      const a0 = r00 * d0, a1 = r01 * d1, a2 = r02 * d2
      const b0 = r10 * d0, b1 = r11 * d1, b2 = r12 * d2
      const c0 = r20 * d0, c1 = r21 * d1, c2 = r22 * d2
      const w00 = a0 * r00 + a1 * r01 + a2 * r02
      const w01 = a0 * r10 + a1 * r11 + a2 * r12
      const w02 = a0 * r20 + a1 * r21 + a2 * r22
      const w11 = b0 * r10 + b1 * r11 + b2 * r12
      const w12 = b0 * r20 + b1 * r21 + b2 * r22
      const w22 = c0 * r20 + c1 * r21 + c2 * r22
      W[i9 + 0] = w00; W[i9 + 1] = w01; W[i9 + 2] = w02
      W[i9 + 3] = w01; W[i9 + 4] = w11; W[i9 + 5] = w12
      W[i9 + 6] = w02; W[i9 + 7] = w12; W[i9 + 8] = w22
    }
  }

  // World-space AABBs for every body. Inflated by margin so contacts stay
  // paired across small velocity jitter without recomputing per iteration.
  updateAabbs(margin = 0.5): void {
    const N = this.count
    const pos = this.positions
    const ori = this.orientations
    const shapes = this.shape
    const sz = this.size
    const minA = this.aabbMin
    const maxA = this.aabbMax

    for (let i = 0; i < N; i++) {
      const i3 = i * 3
      const i4 = i * 4
      const px = pos[i3 + 0],
        py = pos[i3 + 1],
        pz = pos[i3 + 2]
      let hx = 0,
        hy = 0,
        hz = 0

      switch (shapes[i]) {
        case RigidbodyShape.Sphere: {
          const r = sz[i3 + 0]
          hx = hy = hz = r
          break
        }
        case RigidbodyShape.Box: {
          // OBB AABB: half-extents projected by |R|·size.
          const qx = ori[i4 + 0],
            qy = ori[i4 + 1],
            qz = ori[i4 + 2],
            qw = ori[i4 + 3]
          const x2 = qx + qx,
            y2 = qy + qy,
            z2 = qz + qz
          const xx = qx * x2,
            yy = qy * y2,
            zz = qz * z2
          const xy = qx * y2,
            xz = qx * z2,
            yz = qy * z2
          const wx = qw * x2,
            wy = qw * y2,
            wz = qw * z2
          const m00 = Math.abs(1 - (yy + zz)),
            m01 = Math.abs(xy + wz),
            m02 = Math.abs(xz - wy)
          const m10 = Math.abs(xy - wz),
            m11 = Math.abs(1 - (xx + zz)),
            m12 = Math.abs(yz + wx)
          const m20 = Math.abs(xz + wy),
            m21 = Math.abs(yz - wx),
            m22 = Math.abs(1 - (xx + yy))
          const sx = sz[i3 + 0],
            sy = sz[i3 + 1],
            szz = sz[i3 + 2]
          hx = m00 * sx + m01 * sy + m02 * szz
          hy = m10 * sx + m11 * sy + m12 * szz
          hz = m20 * sx + m21 * sy + m22 * szz
          break
        }
        case RigidbodyShape.Capsule: {
          // After rotation, cap offsets are ±halfH · R·ŷ, so AABB half-
          // extents = |R·ŷ|·halfH + radius.
          const r = sz[i3 + 0]
          const halfH = sz[i3 + 1] * 0.5
          const qx = ori[i4 + 0],
            qy = ori[i4 + 1],
            qz = ori[i4 + 2],
            qw = ori[i4 + 3]
          // R · (0,1,0) = (2(xy − wz), 1 − 2(xx + zz), 2(yz + wx))
          const rx = 2 * (qx * qy - qw * qz)
          const ry = 1 - 2 * (qx * qx + qz * qz)
          const rz = 2 * (qy * qz + qw * qx)
          hx = Math.abs(rx) * halfH + r
          hy = Math.abs(ry) * halfH + r
          hz = Math.abs(rz) * halfH + r
          break
        }
      }

      minA[i3 + 0] = px - hx - margin
      minA[i3 + 1] = py - hy - margin
      minA[i3 + 2] = pz - hz - margin
      maxA[i3 + 0] = px + hx + margin
      maxA[i3 + 1] = py + hy + margin
      maxA[i3 + 2] = pz + hz + margin
    }
  }

  // Compute bone-coupling matrices once, on the first step. Bodies with
  // boneIndex < 0 get identity offsets.
  computeBoneOffsets(boneInverseBindMatrices: Float32Array): void {
    const N = this.count
    const offsets = this.bodyOffsetMatrix
    const inverses = this.bodyOffsetInverse
    const ori = this.orientations
    const pos = this.positions
    const boneIdx = this.boneIndex
    const totalBones = boneInverseBindMatrices.length / 16

    const shapeWorldBind = _scratchA
    const offsetMat = _scratchB

    for (let i = 0; i < N; i++) {
      const dst = i * 16
      const b = boneIdx[i]

      if (b < 0 || b >= totalBones) {
        identity16(offsets, dst)
        identity16(inverses, dst)
        continue
      }

      // shapeWorldBind = T(shapePosition) · R(shapeRotation)
      const i3 = i * 3
      const i4 = i * 4
      Mat4.fromPositionRotationInto(
        pos[i3 + 0],
        pos[i3 + 1],
        pos[i3 + 2],
        ori[i4 + 0],
        ori[i4 + 1],
        ori[i4 + 2],
        ori[i4 + 3],
        shapeWorldBind,
      )

      // bodyOffset = boneInverseBind × shapeWorldBind
      Mat4.multiplyArrays(boneInverseBindMatrices, b * 16, shapeWorldBind, 0, offsetMat, 0)

      // Copy into offsets[dst] and invert into inverses[dst].
      offsets.set(offsetMat, dst)
      const inverseTmp = _scratchC
      const ok = Mat4.inverseInto(offsetMat, inverseTmp)
      if (ok) {
        inverses.set(inverseTmp, dst)
      } else {
        identity16(inverses, dst)
      }
    }

    this.boneOffsetsReady = true
  }

  isBoneOffsetsReady(): boolean {
    return this.boneOffsetsReady
  }

  // Pair-filter inputs (invMass, group, mask) are immutable post-construction,
  // so build the candidate-pair list once and reuse every step.
  getCollisionPairs(): Uint16Array {
    if (this.collisionPairs !== null) return this.collisionPairs
    const N = this.count
    const invMass = this.invMass
    const group = this.collisionGroup
    const mask = this.willCollideMask
    const buf: number[] = []
    for (let i = 0; i < N; i++) {
      const gi = group[i]
      const mi = mask[i]
      const dynA = invMass[i] > 0
      for (let j = i + 1; j < N; j++) {
        if (!dynA && invMass[j] === 0) continue
        if ((mi & group[j]) === 0 || (mask[j] & gi) === 0) continue
        buf.push(i, j)
      }
    }
    this.collisionPairs = new Uint16Array(buf)
    return this.collisionPairs
  }
}

const _scratchA = new Float32Array(16)
const _scratchB = new Float32Array(16)
const _scratchC = new Float32Array(16)

// Diagonal local inverse inertia, matching Bullet's calculateLocalInertia
// per shape (so behavior tracks Ammo-based MMD engines):
//   Sphere:  I = (2/5)·m·r² on every axis
//   Box:     Ix = m/12·(ly²+lz²), … with l = full extents (2·half)
//   Capsule: Bullet's bounding-box approximation of the Y-axis capsule
//            (lx = lz = 2r, ly = h + 2r)
function computeLocalInvInertia(rb: Rigidbody, out: Float32Array, o: number): void {
  const m = rb.mass
  if (m <= 0) return
  let Ix: number, Iy: number, Iz: number
  switch (rb.shape) {
    case RigidbodyShape.Sphere: {
      const I = 0.4 * m * rb.size.x * rb.size.x
      Ix = I; Iy = I; Iz = I
      break
    }
    case RigidbodyShape.Box: {
      const lx2 = 4 * rb.size.x * rb.size.x
      const ly2 = 4 * rb.size.y * rb.size.y
      const lz2 = 4 * rb.size.z * rb.size.z
      Ix = (m / 12) * (ly2 + lz2)
      Iy = (m / 12) * (lx2 + lz2)
      Iz = (m / 12) * (lx2 + ly2)
      break
    }
    case RigidbodyShape.Capsule: {
      const lx = 2 * rb.size.x
      const ly = rb.size.y + 2 * rb.size.x
      const lx2 = lx * lx
      const ly2 = ly * ly
      Ix = (m / 12) * (ly2 + lx2)
      Iy = (m / 12) * (lx2 + lx2)
      Iz = (m / 12) * (lx2 + ly2)
      break
    }
    default: {
      Ix = m; Iy = m; Iz = m
    }
  }
  out[o + 0] = Ix > 0 ? 1 / Ix : 0
  out[o + 1] = Iy > 0 ? 1 / Iy : 0
  out[o + 2] = Iz > 0 ? 1 / Iz : 0
}

function identity16(out: Float32Array, offset: number): void {
  out[offset + 0] = 1
  out[offset + 1] = 0
  out[offset + 2] = 0
  out[offset + 3] = 0
  out[offset + 4] = 0
  out[offset + 5] = 1
  out[offset + 6] = 0
  out[offset + 7] = 0
  out[offset + 8] = 0
  out[offset + 9] = 0
  out[offset + 10] = 1
  out[offset + 11] = 0
  out[offset + 12] = 0
  out[offset + 13] = 0
  out[offset + 14] = 0
  out[offset + 15] = 1
}
