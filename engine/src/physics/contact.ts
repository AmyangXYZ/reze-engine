// Narrowphase contact generation for sphere/box/capsule pairs.
//
// Contact convention: `normal` points from body A toward body B, so a
// positive normal impulse pushes B away from A. `rA` / `rB` are world-space
// lever arms from each CG to the contact point. Depth is positive when
// shapes overlap, ≤ 0 for speculative contacts inside the margin band.
// Box-box not implemented (PMX rigs rarely use it and it needs SAT + clipping).

import { RigidbodyShape } from "./types"
import type { RigidBodyStore } from "./body"

// Speculative contact range. Depth is reported relative to the un-inflated
// surface, so values 0 ≥ depth ≥ −CONTACT_MARGIN cover the "near touch but
// not overlapping yet" case. The push-only impulse clamp keeps these inert
// until actual overlap, but they prevent fast bodies from crossing a thin
// surface in one substep without ever generating a contact.
export const CONTACT_MARGIN = 0.04

export interface Contact {
  bodyA: number
  bodyB: number
  // Lever arms (world-space) from each body's CG to the contact point.
  rAx: number
  rAy: number
  rAz: number
  rBx: number
  rBy: number
  rBz: number
  // Unit normal pointing A → B.
  nx: number
  ny: number
  nz: number
  depth: number
  friction: number
  restitution: number
  // SI-row state, written by the solver each iter.
  appliedNormalImpulse: number
  appliedFrictionImpulse1: number
  appliedFrictionImpulse2: number

  // Per-substep cache. Written by the solver's setup pass, read by iter.
  // Normal row:
  cAxN: number; cAyN: number; cAzN: number   // rA × n
  cBxN: number; cByN: number; cBzN: number   // rB × n
  jacInvN: number
  bounceVel: number   // restitution reference, captured at setup from initial relVelN
  // Friction tangent 1:
  t1x: number; t1y: number; t1z: number
  cAxT1: number; cAyT1: number; cAzT1: number
  cBxT1: number; cByT1: number; cBzT1: number
  jacInvT1: number
  // Friction tangent 2:
  t2x: number; t2y: number; t2z: number
  cAxT2: number; cAyT2: number; cAzT2: number
  cBxT2: number; cByT2: number; cBzT2: number
  jacInvT2: number
}

function makeContact(): Contact {
  return {
    bodyA: 0, bodyB: 0,
    rAx: 0, rAy: 0, rAz: 0,
    rBx: 0, rBy: 0, rBz: 0,
    nx: 0, ny: 0, nz: 0,
    depth: 0,
    friction: 0,
    restitution: 0,
    appliedNormalImpulse: 0,
    appliedFrictionImpulse1: 0,
    appliedFrictionImpulse2: 0,
    cAxN: 0, cAyN: 0, cAzN: 0,
    cBxN: 0, cByN: 0, cBzN: 0,
    jacInvN: 0,
    bounceVel: 0,
    t1x: 0, t1y: 0, t1z: 0,
    cAxT1: 0, cAyT1: 0, cAzT1: 0,
    cBxT1: 0, cByT1: 0, cBzT1: 0,
    jacInvT1: 0,
    t2x: 0, t2y: 0, t2z: 0,
    cAxT2: 0, cAyT2: 0, cAzT2: 0,
    cBxT2: 0, cByT2: 0, cBzT2: 0,
    jacInvT2: 0,
  }
}

// Pool of reusable Contact objects.
export class ContactPool {
  private pool: Contact[] = []
  count = 0

  acquire(): Contact {
    if (this.count < this.pool.length) {
      const c = this.pool[this.count]
      c.appliedNormalImpulse = 0
      c.appliedFrictionImpulse1 = 0
      c.appliedFrictionImpulse2 = 0
      this.count++
      return c
    }
    const c = makeContact()
    this.pool.push(c)
    this.count++
    return c
  }

  reset(): void {
    this.count = 0
  }
  get(i: number): Contact {
    return this.pool[i]
  }
}

// Geometric mean for friction, arithmetic for restitution.
function combineMaterials(store: RigidBodyStore, a: number, b: number, out: Contact): void {
  out.friction = Math.sqrt(store.friction[a] * store.friction[b])
  out.restitution = (store.restitution[a] + store.restitution[b]) * 0.5
}

// --- AABB overlap (broadphase reuses this) ---------------------------------
export function aabbOverlap(store: RigidBodyStore, a: number, b: number): boolean {
  const a3 = a * 3,
    b3 = b * 3
  const minA = store.aabbMin,
    maxA = store.aabbMax
  return (
    minA[a3 + 0] <= maxA[b3 + 0] &&
    maxA[a3 + 0] >= minA[b3 + 0] &&
    minA[a3 + 1] <= maxA[b3 + 1] &&
    maxA[a3 + 1] >= minA[b3 + 1] &&
    minA[a3 + 2] <= maxA[b3 + 2] &&
    maxA[a3 + 2] >= minA[b3 + 2]
  )
}

// --- Sphere–sphere ---------------------------------------------------------
function detectSphereSphere(store: RigidBodyStore, a: number, b: number, pool: ContactPool): void {
  const ai = a * 3,
    bi = b * 3
  const pos = store.positions,
    sz = store.size
  const dx = pos[bi + 0] - pos[ai + 0]
  const dy = pos[bi + 1] - pos[ai + 1]
  const dz = pos[bi + 2] - pos[ai + 2]
  const rA = sz[ai + 0]
  const rB = sz[bi + 0]
  const sumR = rA + rB
  const sumExt = sumR + CONTACT_MARGIN
  const d2 = dx * dx + dy * dy + dz * dz
  if (d2 > sumExt * sumExt) return
  const d = Math.sqrt(d2)
  let nx: number, ny: number, nz: number
  if (d > 1e-6) {
    nx = dx / d
    ny = dy / d
    nz = dz / d
  } else {
    nx = 0
    ny = 1
    nz = 0
  } // arbitrary axis when fully co-located
  const c = pool.acquire()
  c.bodyA = a
  c.bodyB = b
  c.nx = nx
  c.ny = ny
  c.nz = nz
  c.depth = sumR - d // signed: > 0 overlapping, ≤ 0 within margin
  c.rAx = nx * rA
  c.rAy = ny * rA
  c.rAz = nz * rA
  c.rBx = -nx * rB
  c.rBy = -ny * rB
  c.rBz = -nz * rB
  combineMaterials(store, a, b, c)
}

// --- Sphere–capsule helper -------------------------------------------------
// Returns closest point on capsule's line segment (centered at cBody, axis=R·ŷ,
// half-height halfH) to the sphere center sx,sy,sz. Out is (cx,cy,cz).
function closestPointOnCapsuleSegment(
  cx: number,
  cy: number,
  cz: number,
  ax: number,
  ay: number,
  az: number,
  halfH: number,
  sx: number,
  sy: number,
  sz: number,
  out: Float32Array,
): void {
  const dx = sx - cx,
    dy = sy - cy,
    dz = sz - cz
  let t = dx * ax + dy * ay + dz * az
  if (t > halfH) t = halfH
  else if (t < -halfH) t = -halfH
  out[0] = cx + ax * t
  out[1] = cy + ay * t
  out[2] = cz + az * t
}

const _capPoint = new Float32Array(3)
const _capPointB = new Float32Array(3)

function capsuleAxis(store: RigidBodyStore, i: number, out: Float32Array): void {
  const i4 = i * 4
  const qx = store.orientations[i4 + 0]
  const qy = store.orientations[i4 + 1]
  const qz = store.orientations[i4 + 2]
  const qw = store.orientations[i4 + 3]
  // R · (0,1,0)
  out[0] = 2 * (qx * qy - qw * qz)
  out[1] = 1 - 2 * (qx * qx + qz * qz)
  out[2] = 2 * (qy * qz + qw * qx)
}

// --- Sphere–capsule (sphere = a, capsule = b) ------------------------------
function detectSphereCapsule(store: RigidBodyStore, a: number, b: number, pool: ContactPool): void {
  const pos = store.positions,
    sz = store.size
  const ai = a * 3,
    bi = b * 3
  const sx = pos[ai + 0],
    sy = pos[ai + 1],
    sz_ = pos[ai + 2]
  const cx = pos[bi + 0],
    cy = pos[bi + 1],
    cz = pos[bi + 2]
  const rA = sz[ai + 0]
  const rB = sz[bi + 0]
  const halfH = sz[bi + 1] * 0.5
  const axis = _capPoint
  capsuleAxis(store, b, axis)
  const closest = _capPointB
  closestPointOnCapsuleSegment(cx, cy, cz, axis[0], axis[1], axis[2], halfH, sx, sy, sz_, closest)
  const dx = closest[0] - sx
  const dy = closest[1] - sy
  const dz = closest[2] - sz_
  const sumR = rA + rB
  const sumExt = sumR + CONTACT_MARGIN
  const d2 = dx * dx + dy * dy + dz * dz
  if (d2 > sumExt * sumExt) return
  const d = Math.sqrt(d2)
  let nx: number, ny: number, nz: number
  if (d > 1e-6) {
    nx = dx / d
    ny = dy / d
    nz = dz / d
  } else {
    nx = 0
    ny = 1
    nz = 0
  }
  const c = pool.acquire()
  c.bodyA = a
  c.bodyB = b
  c.nx = nx
  c.ny = ny
  c.nz = nz
  c.depth = sumR - d
  // Contact point on A's surface: sphere center + n * rA. Lever arm rA = that
  // offset since A's CG = sphere center.
  c.rAx = nx * rA
  c.rAy = ny * rA
  c.rAz = nz * rA
  // Contact point on B's surface: closest_on_segment − n * rB, lever from B's CG.
  c.rBx = closest[0] - nx * rB - cx
  c.rBy = closest[1] - ny * rB - cy
  c.rBz = closest[2] - nz * rB - cz
  combineMaterials(store, a, b, c)
}

// --- Capsule–capsule -------------------------------------------------------
const _cpA = new Float32Array(3)
const _cpB = new Float32Array(3)

// Closest pair on two segments. Adapted from Real-Time Collision Detection §5.1.9.
function closestPointsTwoSegments(
  p1x: number,
  p1y: number,
  p1z: number,
  q1x: number,
  q1y: number,
  q1z: number,
  p2x: number,
  p2y: number,
  p2z: number,
  q2x: number,
  q2y: number,
  q2z: number,
  outA: Float32Array,
  outB: Float32Array,
): void {
  const d1x = q1x - p1x,
    d1y = q1y - p1y,
    d1z = q1z - p1z
  const d2x = q2x - p2x,
    d2y = q2y - p2y,
    d2z = q2z - p2z
  const rx = p1x - p2x,
    ry = p1y - p2y,
    rz = p1z - p2z
  const a = d1x * d1x + d1y * d1y + d1z * d1z
  const e = d2x * d2x + d2y * d2y + d2z * d2z
  const f = d2x * rx + d2y * ry + d2z * rz
  let s = 0,
    t = 0
  const EPS = 1e-8
  if (a <= EPS && e <= EPS) {
    outA[0] = p1x
    outA[1] = p1y
    outA[2] = p1z
    outB[0] = p2x
    outB[1] = p2y
    outB[2] = p2z
    return
  }
  if (a <= EPS) {
    s = 0
    t = clamp01(f / e)
  } else {
    const c = d1x * rx + d1y * ry + d1z * rz
    if (e <= EPS) {
      t = 0
      s = clamp01(-c / a)
    } else {
      const b = d1x * d2x + d1y * d2y + d1z * d2z
      const denom = a * e - b * b
      if (denom !== 0) s = clamp01((b * f - c * e) / denom)
      t = (b * s + f) / e
      if (t < 0) {
        t = 0
        s = clamp01(-c / a)
      } else if (t > 1) {
        t = 1
        s = clamp01((b - c) / a)
      }
    }
  }
  outA[0] = p1x + d1x * s
  outA[1] = p1y + d1y * s
  outA[2] = p1z + d1z * s
  outB[0] = p2x + d2x * t
  outB[1] = p2y + d2y * t
  outB[2] = p2z + d2z * t
}

function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x
}

// Closest point on segment p1→q1 to a free point (sx,sy,sz). Out gets the
// projected point clamped to the segment.
function closestPointOnSegment(
  p1x: number,
  p1y: number,
  p1z: number,
  q1x: number,
  q1y: number,
  q1z: number,
  sx: number,
  sy: number,
  sz: number,
  out: Float32Array,
): void {
  const dx = q1x - p1x,
    dy = q1y - p1y,
    dz = q1z - p1z
  const segLen2 = dx * dx + dy * dy + dz * dz
  let t = 0
  if (segLen2 > 1e-8) {
    t = ((sx - p1x) * dx + (sy - p1y) * dy + (sz - p1z) * dz) / segLen2
    if (t < 0) t = 0
    else if (t > 1) t = 1
  }
  out[0] = p1x + dx * t
  out[1] = p1y + dy * t
  out[2] = p1z + dz * t
}

// Emit one capsule-vs-capsule contact given a pair of points (pA on A's
// segment, pB on B's segment). Skips silently if outside speculative range.
function emitCapsuleContact(
  store: RigidBodyStore,
  a: number,
  b: number,
  pool: ContactPool,
  pAx: number,
  pAy: number,
  pAz: number,
  pBx: number,
  pBy: number,
  pBz: number,
  rA: number,
  rB: number,
  sumR: number,
  sumExt: number,
  cAx: number,
  cAy: number,
  cAz: number,
  cBx: number,
  cBy: number,
  cBz: number,
): void {
  const dx = pBx - pAx,
    dy = pBy - pAy,
    dz = pBz - pAz
  const d2 = dx * dx + dy * dy + dz * dz
  if (d2 > sumExt * sumExt) return
  const d = Math.sqrt(d2)
  let nx: number, ny: number, nz: number
  if (d > 1e-6) {
    nx = dx / d
    ny = dy / d
    nz = dz / d
  } else {
    nx = 0
    ny = 1
    nz = 0
  }
  const c = pool.acquire()
  c.bodyA = a
  c.bodyB = b
  c.nx = nx
  c.ny = ny
  c.nz = nz
  c.depth = sumR - d
  c.rAx = pAx + nx * rA - cAx
  c.rAy = pAy + ny * rA - cAy
  c.rAz = pAz + nz * rA - cAz
  c.rBx = pBx - nx * rB - cBx
  c.rBy = pBy - ny * rB - cBy
  c.rBz = pBz - nz * rB - cBz
  combineMaterials(store, a, b, c)
}

function detectCapsuleCapsule(store: RigidBodyStore, a: number, b: number, pool: ContactPool): void {
  const pos = store.positions,
    sz = store.size
  const ai = a * 3,
    bi = b * 3
  const cAx = pos[ai + 0],
    cAy = pos[ai + 1],
    cAz = pos[ai + 2]
  const cBx = pos[bi + 0],
    cBy = pos[bi + 1],
    cBz = pos[bi + 2]
  const rA = sz[ai + 0],
    hA = sz[ai + 1] * 0.5
  const rB = sz[bi + 0],
    hB = sz[bi + 1] * 0.5
  const aAx = _capPoint
  const aBx = _capPointB
  capsuleAxis(store, a, aAx)
  capsuleAxis(store, b, aBx)
  const p1x = cAx - aAx[0] * hA,
    p1y = cAy - aAx[1] * hA,
    p1z = cAz - aAx[2] * hA
  const q1x = cAx + aAx[0] * hA,
    q1y = cAy + aAx[1] * hA,
    q1z = cAz + aAx[2] * hA
  const p2x = cBx - aBx[0] * hB,
    p2y = cBy - aBx[1] * hB,
    p2z = cBz - aBx[2] * hB
  const q2x = cBx + aBx[0] * hB,
    q2y = cBy + aBx[1] * hB,
    q2z = cBz + aBx[2] * hB

  const sumR = rA + rB
  const sumExt = sumR + CONTACT_MARGIN

  // Primary contact: closest-pair on the two segments.
  closestPointsTwoSegments(p1x, p1y, p1z, q1x, q1y, q1z, p2x, p2y, p2z, q2x, q2y, q2z, _cpA, _cpB)
  emitCapsuleContact(
    store,
    a,
    b,
    pool,
    _cpA[0],
    _cpA[1],
    _cpA[2],
    _cpB[0],
    _cpB[1],
    _cpB[2],
    rA,
    rB,
    sumR,
    sumExt,
    cAx,
    cAy,
    cAz,
    cBx,
    cBy,
    cBz,
  )

  // For nearly-parallel axes the closest-pair algorithm is degenerate
  // (denom = a·e − b² ≈ 0) and returns one arbitrary point. Sampling A's
  // endpoints adds two contacts that pin both rotation and length-wise push.
  const cosA = Math.abs(aAx[0] * aBx[0] + aAx[1] * aBx[1] + aAx[2] * aBx[2])
  if (cosA > 0.9) {
    closestPointOnSegment(p2x, p2y, p2z, q2x, q2y, q2z, p1x, p1y, p1z, _cpB)
    emitCapsuleContact(
      store,
      a,
      b,
      pool,
      p1x,
      p1y,
      p1z,
      _cpB[0],
      _cpB[1],
      _cpB[2],
      rA,
      rB,
      sumR,
      sumExt,
      cAx,
      cAy,
      cAz,
      cBx,
      cBy,
      cBz,
    )
    closestPointOnSegment(p2x, p2y, p2z, q2x, q2y, q2z, q1x, q1y, q1z, _cpB)
    emitCapsuleContact(
      store,
      a,
      b,
      pool,
      q1x,
      q1y,
      q1z,
      _cpB[0],
      _cpB[1],
      _cpB[2],
      rA,
      rB,
      sumR,
      sumExt,
      cAx,
      cAy,
      cAz,
      cBx,
      cBy,
      cBz,
    )
  }
}

// --- Sphere–box (sphere = a, box = b) --------------------------------------
const _localPt = new Float32Array(3)

// 3×3 row-major rotation matrix for body i (xx = 2·qx·qx etc.).
const _rot = new Float32Array(9)
function loadBodyRot(store: RigidBodyStore, i: number): void {
  const i4 = i * 4
  const qx = store.orientations[i4 + 0]
  const qy = store.orientations[i4 + 1]
  const qz = store.orientations[i4 + 2]
  const qw = store.orientations[i4 + 3]
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
  _rot[0] = 1 - (yy + zz)
  _rot[1] = xy - wz
  _rot[2] = xz + wy
  _rot[3] = xy + wz
  _rot[4] = 1 - (xx + zz)
  _rot[5] = yz - wx
  _rot[6] = xz - wy
  _rot[7] = yz + wx
  _rot[8] = 1 - (xx + yy)
}

// Transform world point into body i's local frame: v_local = R^T · (p − bodyPos).
function worldToBodyLocal(
  store: RigidBodyStore,
  i: number,
  px: number,
  py: number,
  pz: number,
  out: Float32Array,
): void {
  const i3 = i * 3
  const dx = px - store.positions[i3 + 0]
  const dy = py - store.positions[i3 + 1]
  const dz = pz - store.positions[i3 + 2]
  loadBodyRot(store, i)
  // R^T · v = (col k of R) · v.
  out[0] = _rot[0] * dx + _rot[3] * dy + _rot[6] * dz
  out[1] = _rot[1] * dx + _rot[4] * dy + _rot[7] * dz
  out[2] = _rot[2] * dx + _rot[5] * dy + _rot[8] * dz
}

// Rotate a body-local direction into world space: v_world = R · v_local.
function bodyLocalToWorldDir(
  store: RigidBodyStore,
  i: number,
  lx: number,
  ly: number,
  lz: number,
  out: Float32Array,
): void {
  loadBodyRot(store, i)
  out[0] = _rot[0] * lx + _rot[1] * ly + _rot[2] * lz
  out[1] = _rot[3] * lx + _rot[4] * ly + _rot[5] * lz
  out[2] = _rot[6] * lx + _rot[7] * ly + _rot[8] * lz
}

function detectSphereBox(store: RigidBodyStore, a: number, b: number, pool: ContactPool): void {
  const ai = a * 3,
    bi = b * 3
  const sx = store.positions[ai + 0]
  const sy = store.positions[ai + 1]
  const sz_ = store.positions[ai + 2]
  const rA = store.size[ai + 0]
  const hx = store.size[bi + 0]
  const hy = store.size[bi + 1]
  const hz = store.size[bi + 2]

  // Sphere center in box-local frame.
  worldToBodyLocal(store, b, sx, sy, sz_, _localPt)
  const lx = _localPt[0],
    ly = _localPt[1],
    lz = _localPt[2]

  // Closest point on box (clamp to half-extents).
  let qx = lx,
    qy = ly,
    qz = lz
  if (qx > hx) qx = hx
  else if (qx < -hx) qx = -hx
  if (qy > hy) qy = hy
  else if (qy < -hy) qy = -hy
  if (qz > hz) qz = hz
  else if (qz < -hz) qz = -hz

  let dx = lx - qx,
    dy = ly - qy,
    dz = lz - qz
  let d2 = dx * dx + dy * dy + dz * dz

  let nLocalX: number, nLocalY: number, nLocalZ: number
  let depth: number

  const rExt = rA + CONTACT_MARGIN
  if (d2 > rExt * rExt) return // outside speculative range

  if (d2 > 1e-12) {
    const d = Math.sqrt(d2)
    nLocalX = dx / d
    nLocalY = dy / d
    nLocalZ = dz / d
    depth = rA - d // signed: > 0 overlapping, ≤ 0 within margin
  } else {
    // Sphere center inside box — pick shortest axis to escape.
    const px = hx - Math.abs(lx),
      py = hy - Math.abs(ly),
      pz = hz - Math.abs(lz)
    if (px < py && px < pz) {
      nLocalX = lx > 0 ? 1 : -1
      nLocalY = 0
      nLocalZ = 0
      depth = rA + px
      qx = lx > 0 ? hx : -hx
      qy = ly
      qz = lz
    } else if (py < pz) {
      nLocalX = 0
      nLocalY = ly > 0 ? 1 : -1
      nLocalZ = 0
      depth = rA + py
      qx = lx
      qy = ly > 0 ? hy : -hy
      qz = lz
    } else {
      nLocalX = 0
      nLocalY = 0
      nLocalZ = lz > 0 ? 1 : -1
      depth = rA + pz
      qx = lx
      qy = ly
      qz = lz > 0 ? hz : -hz
    }
  }

  // Rotate local normal back to world. Convention: normal points A→B, but we
  // computed n = (lx − qx) which goes "from box surface toward sphere center"
  // (= B → A). Flip sign.
  const out = _capPoint
  bodyLocalToWorldDir(store, b, -nLocalX, -nLocalY, -nLocalZ, out)
  const nx = out[0],
    ny = out[1],
    nz = out[2]

  // Box's contact point in world: rotate (qx,qy,qz) and translate by box pos.
  const bp = _capPointB
  bodyLocalToWorldDir(store, b, qx, qy, qz, bp)
  const bpx = bp[0] + store.positions[bi + 0]
  const bpy = bp[1] + store.positions[bi + 1]
  const bpz = bp[2] + store.positions[bi + 2]

  const c = pool.acquire()
  c.bodyA = a
  c.bodyB = b
  c.nx = nx
  c.ny = ny
  c.nz = nz
  c.depth = depth
  c.rAx = nx * rA
  c.rAy = ny * rA
  c.rAz = nz * rA
  c.rBx = bpx - store.positions[bi + 0]
  c.rBy = bpy - store.positions[bi + 1]
  c.rBz = bpz - store.positions[bi + 2]
  combineMaterials(store, a, b, c)
}

// --- Capsule–box -----------------------------------------------------------
// Walk the capsule's segment (in box-local space) toward the box, sample
// sphere-box at the converged parameter plus both endpoints. Endpoint
// samples catch caps grazing a face when the closest-point parameter sits
// at one end of the segment.
function detectCapsuleBox(store: RigidBodyStore, a: number, b: number, pool: ContactPool): void {
  const pos = store.positions,
    sz = store.size
  const ai = a * 3,
    bi = b * 3
  const cx = pos[ai + 0],
    cy = pos[ai + 1],
    cz = pos[ai + 2]
  const rA = sz[ai + 0]
  const hA = sz[ai + 1] * 0.5
  const ax = _capPoint
  capsuleAxis(store, a, ax)

  // Endpoints in world space.
  const p1wx = cx - ax[0] * hA,
    p1wy = cy - ax[1] * hA,
    p1wz = cz - ax[2] * hA
  const p2wx = cx + ax[0] * hA,
    p2wy = cy + ax[1] * hA,
    p2wz = cz + ax[2] * hA

  // Endpoints in box-local space.
  worldToBodyLocal(store, b, p1wx, p1wy, p1wz, _localPt)
  const p1lx = _localPt[0],
    p1ly = _localPt[1],
    p1lz = _localPt[2]
  worldToBodyLocal(store, b, p2wx, p2wy, p2wz, _localPt)
  const p2lx = _localPt[0],
    p2ly = _localPt[1],
    p2lz = _localPt[2]

  const hx = sz[bi + 0],
    hy = sz[bi + 1],
    hz = sz[bi + 2]

  // Closest point on segment to box (in box-local). Iterate a few times to
  // converge — clamp each component, recompute t, repeat. Two passes is
  // enough for our use case (capsule modestly larger than box).
  let t = 0.5
  for (let iter = 0; iter < 4; iter++) {
    const px = p1lx + (p2lx - p1lx) * t
    const py = p1ly + (p2ly - p1ly) * t
    const pz = p1lz + (p2lz - p1lz) * t
    let qx = px,
      qy = py,
      qz = pz
    if (qx > hx) qx = hx
    else if (qx < -hx) qx = -hx
    if (qy > hy) qy = hy
    else if (qy < -hy) qy = -hy
    if (qz > hz) qz = hz
    else if (qz < -hz) qz = -hz
    // Project clamped point back onto the segment to refine t.
    const dx = p2lx - p1lx,
      dy = p2ly - p1ly,
      dz = p2lz - p1lz
    const segLen2 = dx * dx + dy * dy + dz * dz
    if (segLen2 < 1e-8) break
    t = ((qx - p1lx) * dx + (qy - p1ly) * dy + (qz - p1lz) * dz) / segLen2
    if (t < 0) {
      t = 0
      break
    }
    if (t > 1) {
      t = 1
      break
    }
  }

  // Sample at the converged t plus both endpoints — endpoints catch capsule
  // caps grazing the box surface where the closest-point loop sits at one
  // segment end.
  let bestDepth = -Infinity
  let bestNX = 0,
    bestNY = 0,
    bestNZ = 0
  let bestRAX = 0,
    bestRAY = 0,
    bestRAZ = 0
  let bestRBX = 0,
    bestRBY = 0,
    bestRBZ = 0
  let found = false

  const samples = [t, 0, 1]
  for (const s of samples) {
    const sx = p1wx + (p2wx - p1wx) * s
    const sy = p1wy + (p2wy - p1wy) * s
    const sz_ = p1wz + (p2wz - p1wz) * s
    worldToBodyLocal(store, b, sx, sy, sz_, _localPt)
    const lx = _localPt[0],
      ly = _localPt[1],
      lz = _localPt[2]
    let qx = lx,
      qy = ly,
      qz = lz
    if (qx > hx) qx = hx
    else if (qx < -hx) qx = -hx
    if (qy > hy) qy = hy
    else if (qy < -hy) qy = -hy
    if (qz > hz) qz = hz
    else if (qz < -hz) qz = -hz
    const dx = lx - qx,
      dy = ly - qy,
      dz = lz - qz
    const d2 = dx * dx + dy * dy + dz * dz
    const rExt = rA + CONTACT_MARGIN
    if (d2 > rExt * rExt) continue
    let nLocalX = 0,
      nLocalY = 0,
      nLocalZ = 0
    let depth: number
    if (d2 > 1e-12) {
      const d = Math.sqrt(d2)
      nLocalX = dx / d
      nLocalY = dy / d
      nLocalZ = dz / d
      depth = rA - d // signed: > 0 overlapping, ≤ 0 within margin
    } else {
      const px = hx - Math.abs(lx),
        py = hy - Math.abs(ly),
        pz = hz - Math.abs(lz)
      if (px < py && px < pz) {
        nLocalX = lx > 0 ? 1 : -1
        depth = rA + px
        qx = lx > 0 ? hx : -hx
        qy = ly
        qz = lz
      } else if (py < pz) {
        nLocalY = ly > 0 ? 1 : -1
        depth = rA + py
        qx = lx
        qy = ly > 0 ? hy : -hy
        qz = lz
      } else {
        nLocalZ = lz > 0 ? 1 : -1
        depth = rA + pz
        qx = lx
        qy = ly
        qz = lz > 0 ? hz : -hz
      }
    }
    if (depth <= bestDepth) continue
    bestDepth = depth
    found = true
    const dirOut = _localPt
    bodyLocalToWorldDir(store, b, -nLocalX, -nLocalY, -nLocalZ, dirOut)
    bestNX = dirOut[0]
    bestNY = dirOut[1]
    bestNZ = dirOut[2]
    const bpOut = _localPt
    bodyLocalToWorldDir(store, b, qx, qy, qz, bpOut)
    const bpx = bpOut[0] + pos[bi + 0]
    const bpy = bpOut[1] + pos[bi + 1]
    const bpz = bpOut[2] + pos[bi + 2]
    bestRAX = sx + bestNX * rA - cx
    bestRAY = sy + bestNY * rA - cy
    bestRAZ = sz_ + bestNZ * rA - cz
    bestRBX = bpx - pos[bi + 0]
    bestRBY = bpy - pos[bi + 1]
    bestRBZ = bpz - pos[bi + 2]
  }

  if (!found) return
  const c = pool.acquire()
  c.bodyA = a
  c.bodyB = b
  c.nx = bestNX
  c.ny = bestNY
  c.nz = bestNZ
  c.depth = bestDepth
  c.rAx = bestRAX
  c.rAy = bestRAY
  c.rAz = bestRAZ
  c.rBx = bestRBX
  c.rBy = bestRBY
  c.rBz = bestRBZ
  combineMaterials(store, a, b, c)
}

// Dispatch a pair to the matching narrowphase. Caller has already done
// broadphase + group/mask filtering. Some shape pairs (sphere-A capsule-B
// etc.) reuse a canonical implementation via swap + flipLastNormal.
export function generateContacts(store: RigidBodyStore, a: number, b: number, pool: ContactPool): void {
  const sA = store.shape[a]
  const sB = store.shape[b]
  if (sA === RigidbodyShape.Sphere && sB === RigidbodyShape.Sphere) {
    detectSphereSphere(store, a, b, pool)
    return
  }
  if (sA === RigidbodyShape.Sphere && sB === RigidbodyShape.Capsule) {
    detectSphereCapsule(store, a, b, pool)
    return
  }
  if (sA === RigidbodyShape.Capsule && sB === RigidbodyShape.Sphere) {
    // Only flip a contact this call actually produced. The detector returns
    // without emitting whenever the shapes are out of range, and flipping then
    // reverses the normal of whatever unrelated contact happens to sit at the
    // end of the pool — pushing those two bodies together instead of apart.
    const before = pool.count
    detectSphereCapsule(store, b, a, pool)
    if (pool.count > before) flipLastNormal(pool)
    return
  }
  if (sA === RigidbodyShape.Capsule && sB === RigidbodyShape.Capsule) {
    detectCapsuleCapsule(store, a, b, pool)
    return
  }
  if (sA === RigidbodyShape.Sphere && sB === RigidbodyShape.Box) {
    detectSphereBox(store, a, b, pool)
    return
  }
  if (sA === RigidbodyShape.Box && sB === RigidbodyShape.Sphere) {
    const before = pool.count
    detectSphereBox(store, b, a, pool)
    if (pool.count > before) flipLastNormal(pool)
    return
  }
  if (sA === RigidbodyShape.Capsule && sB === RigidbodyShape.Box) {
    detectCapsuleBox(store, a, b, pool)
    return
  }
  if (sA === RigidbodyShape.Box && sB === RigidbodyShape.Capsule) {
    const before = pool.count
    detectCapsuleBox(store, b, a, pool)
    if (pool.count > before) flipLastNormal(pool)
    return
  }
  // Box-box left unimplemented.
}

// After a swapped detect* call, the last contact's normal points the wrong
// way and lever arms are mismatched. Flip and re-anchor.
function flipLastNormal(pool: ContactPool): void {
  if (pool.count === 0) return
  const c = pool.get(pool.count - 1)
  const ta = c.bodyA
  c.bodyA = c.bodyB
  c.bodyB = ta
  const trAx = c.rAx,
    trAy = c.rAy,
    trAz = c.rAz
  c.rAx = c.rBx
  c.rAy = c.rBy
  c.rAz = c.rBz
  c.rBx = trAx
  c.rBy = trAy
  c.rBz = trAz
  c.nx = -c.nx
  c.ny = -c.ny
  c.nz = -c.nz
}

// Iterate the prebuilt candidate-pair list and AABB-test each pair. The
// static-static and group/mask filters were applied once at construction —
// see RigidBodyStore.getCollisionPairs. SAP / dynamic AABB tree pay off
// above ~500 bodies; below that this flat sweep wins on cache locality.
export function findContacts(store: RigidBodyStore, pool: ContactPool): void {
  store.updateAabbs()
  const pairs = store.getCollisionPairs()
  for (let p = 0; p < pairs.length; p += 2) {
    const i = pairs[p]
    const j = pairs[p + 1]
    if (!aabbOverlap(store, i, j)) continue
    generateContacts(store, i, j, pool)
  }
  // Built-in floor: a plane pass against every dynamic body. Cheaper and more
  // complete than routing the huge ground box through the pair machinery —
  // one y-test per body, and box hems collide too (there is no generic
  // box-box narrowphase to lean on).
  const g = store.groundIndex
  if (g >= 0) {
    const minA = store.aabbMin
    for (let i = 0; i < store.count; i++) {
      if (store.invMass[i] <= 0) continue
      if (minA[i * 3 + 1] > CONTACT_MARGIN) continue
      detectFloor(store, i, g, pool)
    }
  }
}

// The floor's top face is the model-space plane y = 0; its body (`g`) exists so
// contact rows have a static B side. Normal convention matches the detectors:
// A→B, so pointing DOWN into the floor; depth > 0 = penetrating.
function detectFloor(store: RigidBodyStore, a: number, g: number, pool: ContactPool): void {
  const ai = a * 3
  const pos = store.positions
  const gx = pos[g * 3 + 0]
  const gy = pos[g * 3 + 1]
  const gz = pos[g * 3 + 2]
  const emit = (px: number, py: number, pz: number, depth: number) => {
    const c = pool.acquire()
    c.bodyA = a
    c.bodyB = g
    c.nx = 0
    c.ny = -1
    c.nz = 0
    c.depth = depth
    c.rAx = px - pos[ai + 0]
    c.rAy = py - pos[ai + 1]
    c.rAz = pz - pos[ai + 2]
    c.rBx = px - gx
    c.rBy = py - gy
    c.rBz = pz - gz
    combineMaterials(store, a, g, c)
  }
  const cx = pos[ai + 0]
  const cy = pos[ai + 1]
  const cz = pos[ai + 2]
  switch (store.shape[a]) {
    case RigidbodyShape.Sphere: {
      const r = store.size[ai + 0]
      const low = cy - r
      if (low <= CONTACT_MARGIN) emit(cx, low, cz, -low)
      break
    }
    case RigidbodyShape.Capsule: {
      const r = store.size[ai + 0]
      const h = store.size[ai + 1] * 0.5
      const ax = _capPoint
      capsuleAxis(store, a, ax)
      for (const sgn of [-1, 1]) {
        const ex = cx + ax[0] * h * sgn
        const ey = cy + ax[1] * h * sgn
        const ez = cz + ax[2] * h * sgn
        const low = ey - r
        if (low <= CONTACT_MARGIN) emit(ex, low, ez, -low)
      }
      break
    }
    case RigidbodyShape.Box: {
      const hx = store.size[ai + 0]
      const hy = store.size[ai + 1]
      const hz = store.size[ai + 2]
      const out = _capPointB
      for (let k = 0; k < 8; k++) {
        const lx = k & 1 ? hx : -hx
        const ly = k & 2 ? hy : -hy
        const lz = k & 4 ? hz : -hz
        bodyLocalToWorldDir(store, a, lx, ly, lz, out)
        const wy = cy + out[1]
        if (wy <= CONTACT_MARGIN) emit(cx + out[0], wy, cz + out[2], -wy)
      }
      break
    }
  }
}
