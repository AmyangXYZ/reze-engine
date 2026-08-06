// Narrowphase contact generation for sphere/box/capsule pairs.
//
// Contact convention: `normal` points from body A toward body B, so a
// positive normal impulse pushes B away from A. `rA` / `rB` are world-space
// lever arms from each CG to the contact point. Depth is positive when
// shapes overlap, ≤ 0 for speculative contacts inside the margin band.
// Box-box is SAT + face clipping (see detectBoxBox) — MMD dress rigs are built
// from box panels, and it is the majority of collidable pairs on those models.

import { RigidbodyShape } from "./types"
import type { RigidBodyStore } from "./body"

// Speculative contact range. Depth is reported relative to the un-inflated
// surface, so values 0 ≥ depth ≥ −CONTACT_MARGIN cover the "near touch but
// not overlapping yet" case. They exist so a fast body cannot cross a thin
// surface in one substep without ever generating a contact.
//
// What keeps them inert until the body would actually arrive is the solver's
// `allowedApproachVel` (gap / dt), NOT the push-only impulse clamp — the clamp
// only forbids a negative (pulling) impulse and does nothing to stop a large
// positive one from stopping a body dead in mid-air. This comment used to
// claim otherwise, and the bug it hid was worth 88% of speculative rows firing
// on a dress rig. See setupContactRow.
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
  /** Per-contact relaxation gain, 1/max(rows on A, rows on B). See CONTACT_SOR_MIN. */
  sorGain: number
  // Approach speed this row is allowed to leave alone: gap / dt for a
  // speculative row, 0 once the shapes actually touch. See setupContactRow.
  allowedApproachVel: number
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
    sorGain: 1,
    allowedApproachVel: 0,
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
// sphere-box at the converged parameter plus both endpoints, and keep the
// DEEPEST sample. Endpoint samples catch caps grazing a face when the
// closest-point parameter sits at one end of the segment.
//
// Deliberately one contact, not a manifold along the touching segment. A line
// of contacts here does stop a panel pivoting into a leg, and it was tried:
// penetration fell, but idle jitter on a 688-body rig rose because every extra
// contact is another shove from the position-correction pass, which the joint
// springs hand straight back. Cloth that buzzes reads worse than cloth that
// clips, so this keeps a little 穿模 in exchange for stillness. Box-box is a
// different case — those panels had NO collision at all, so it is pure gain.
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

// --- Box–box ---------------------------------------------------------------
// SAT over the 15 candidate axes, then face clipping for a multi-point
// manifold. Everything below happens in A's local frame: B's centre and axes
// are transformed in once, so the 15 tests and the clipping all read as plain
// vector maths instead of repeated quaternion work.
//
// Why this exists at all: MMD dress rigs are built from flat box PANELS, and
// panel-against-panel is the collision that keeps skirt layers out of each
// other. Measured across seven shipped models, box-box is 45–70% of every
// collidable pair on models whose riggers left skirt self-collision enabled —
// all of it silently dropped before this. MMD's own physics is Bullet, which
// dispatches box-box through btBoxBoxDetector (ODE's dBoxBox: this same
// 15-axis SAT plus face clipping), so rigs are authored assuming it works.
//
// The pair count is not the frame cost: the AABB pass upstream filters first,
// and in a rest pose only ~220 of 诗蔻蒂's 60k box-box candidates reach here.

// Scratch, module-level so a frame of narrowphase allocates nothing.
const _bbBax = new Float32Array(9) // B's axes in A's frame, row j = axis j
const _bbC = new Float32Array(3) // B's centre in A's frame
const _bbAxis = new Float32Array(3) // best separating axis, A's frame
const _bbClip = new Float32Array(24) // clip buffer: up to 8 points
const _bbClip2 = new Float32Array(24)
const _bbDepth = new Float32Array(8)
const _bbTmp = new Float32Array(3)
// A's axes in A's own frame are the identity; kept as a constant so the
// reference/incident selection can treat both boxes through one code path
// without allocating a basis per call.
const _bbIdent = new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1])
const _bbRefH = new Float32Array(3)
const _bbIncH = new Float32Array(3)
const _bbPA = new Float32Array(3)
const _bbHA = new Float32Array(3)
const _bbHB = new Float32Array(3)

// A face axis has to lose by a real margin before an edge axis wins. Near
// ties are common between two flat panels lying against each other, and an
// edge axis there yields one point where a face yields four — the manifold
// would flicker between them frame to frame and the panel would rock.
const EDGE_AXIS_BIAS = 1.05

function detectBoxBox(store: RigidBodyStore, a: number, b: number, pool: ContactPool): void {
  const ai = a * 3,
    bi = b * 3
  const sz = store.size
  const hAx = sz[ai + 0], hAy = sz[ai + 1], hAz = sz[ai + 2]
  const hBx = sz[bi + 0], hBy = sz[bi + 1], hBz = sz[bi + 2]

  // B's centre, in A's frame. The body index here is A, not B — transforming
  // B's own centre by B's own transform yields the origin every time, which
  // makes every SAT distance zero and every pair maximally overlapping.
  worldToBodyLocal(store, a, store.positions[bi + 0], store.positions[bi + 1], store.positions[bi + 2], _bbC)
  const cx = _bbC[0], cy = _bbC[1], cz = _bbC[2]

  // B's three axes, in A's frame: RAᵀ · RB. loadBodyRot writes one body at a
  // time, so read B's columns out before loading A over the top of them.
  loadBodyRot(store, b)
  const b00 = _rot[0], b01 = _rot[1], b02 = _rot[2]
  const b10 = _rot[3], b11 = _rot[4], b12 = _rot[5]
  const b20 = _rot[6], b21 = _rot[7], b22 = _rot[8]
  loadBodyRot(store, a)
  const a00 = _rot[0], a01 = _rot[1], a02 = _rot[2]
  const a10 = _rot[3], a11 = _rot[4], a12 = _rot[5]
  const a20 = _rot[6], a21 = _rot[7], a22 = _rot[8]
  // Column j of RB is B's axis j in world; RAᵀ · that is it in A's frame.
  for (let j = 0; j < 3; j++) {
    const wx = j === 0 ? b00 : j === 1 ? b01 : b02
    const wy = j === 0 ? b10 : j === 1 ? b11 : b12
    const wz = j === 0 ? b20 : j === 1 ? b21 : b22
    _bbBax[j * 3 + 0] = a00 * wx + a10 * wy + a20 * wz
    _bbBax[j * 3 + 1] = a01 * wx + a11 * wy + a21 * wz
    _bbBax[j * 3 + 2] = a02 * wx + a12 * wy + a22 * wz
  }

  // --- SAT. Track the axis of MINIMUM overlap; that is the shallowest way out
  //     and therefore the contact normal.
  let bestOverlap = Infinity
  let bestAxis = -1 // 0-2 = A's faces, 3-5 = B's faces, 6-14 = edge crosses
  let bestNx = 0, bestNy = 0, bestNz = 0

  // `scale` lets the edge axes be judged slightly harder — see EDGE_AXIS_BIAS.
  const test = (nx: number, ny: number, nz: number, id: number, scale: number): boolean => {
    const len2 = nx * nx + ny * ny + nz * nz
    // Degenerate cross product: the two edges are parallel, so this axis adds
    // nothing that the face axes have not already covered.
    if (len2 < 1e-12) return true
    const inv = 1 / Math.sqrt(len2)
    const ux = nx * inv, uy = ny * inv, uz = nz * inv
    const projA = hAx * Math.abs(ux) + hAy * Math.abs(uy) + hAz * Math.abs(uz)
    const projB =
      hBx * Math.abs(ux * _bbBax[0] + uy * _bbBax[1] + uz * _bbBax[2]) +
      hBy * Math.abs(ux * _bbBax[3] + uy * _bbBax[4] + uz * _bbBax[5]) +
      hBz * Math.abs(ux * _bbBax[6] + uy * _bbBax[7] + uz * _bbBax[8])
    const dist = Math.abs(ux * cx + uy * cy + uz * cz)
    const overlap = projA + projB - dist
    // A gap wider than the speculative band: no contact, and no need to test
    // the rest — one separating axis is proof.
    if (overlap < -CONTACT_MARGIN) return false
    if (overlap * scale < bestOverlap) {
      bestOverlap = overlap * scale
      bestAxis = id
      bestNx = ux; bestNy = uy; bestNz = uz
    }
    return true
  }

  if (!test(1, 0, 0, 0, 1)) return
  if (!test(0, 1, 0, 1, 1)) return
  if (!test(0, 0, 1, 2, 1)) return
  for (let j = 0; j < 3; j++) {
    if (!test(_bbBax[j * 3 + 0], _bbBax[j * 3 + 1], _bbBax[j * 3 + 2], 3 + j, 1)) return
  }
  for (let i = 0; i < 3; i++) {
    const axi = i === 0 ? 1 : 0, ayi = i === 1 ? 1 : 0, azi = i === 2 ? 1 : 0
    for (let j = 0; j < 3; j++) {
      const bx = _bbBax[j * 3 + 0], by = _bbBax[j * 3 + 1], bz = _bbBax[j * 3 + 2]
      if (!test(ayi * bz - azi * by, azi * bx - axi * bz, axi * by - ayi * bx, 6 + i * 3 + j, EDGE_AXIS_BIAS)) return
    }
  }
  if (bestAxis < 0) return

  // Orient the normal A → B, matching the contact convention.
  if (bestNx * cx + bestNy * cy + bestNz * cz < 0) {
    bestNx = -bestNx; bestNy = -bestNy; bestNz = -bestNz
  }
  _bbAxis[0] = bestNx; _bbAxis[1] = bestNy; _bbAxis[2] = bestNz

  if (bestAxis >= 6) {
    emitBoxEdgeContact(store, a, b, bestOverlap / EDGE_AXIS_BIAS, (bestAxis - 6) / 3 | 0, (bestAxis - 6) % 3,
      hAx, hAy, hAz, hBx, hBy, hBz, cx, cy, cz, pool)
    return
  }
  emitBoxFaceManifold(store, a, b, bestAxis, hAx, hAy, hAz, hBx, hBy, hBz, cx, cy, cz, pool)
}

// Write one contact from a point given in A's LOCAL frame, with the manifold's
// shared world normal. Both lever arms come from the same world point: the
// clipped point sits on the incident face, within CONTACT_MARGIN of the
// reference face, so splitting them would be false precision.
function emitBoxContact(
  store: RigidBodyStore,
  a: number,
  b: number,
  lx: number, ly: number, lz: number,
  depth: number,
  pool: ContactPool,
): void {
  const ai = a * 3, bi = b * 3
  bodyLocalToWorldDir(store, a, lx, ly, lz, _bbTmp)
  const wx = _bbTmp[0] + store.positions[ai + 0]
  const wy = _bbTmp[1] + store.positions[ai + 1]
  const wz = _bbTmp[2] + store.positions[ai + 2]
  bodyLocalToWorldDir(store, a, _bbAxis[0], _bbAxis[1], _bbAxis[2], _bbTmp)
  const c = pool.acquire()
  c.bodyA = a
  c.bodyB = b
  c.nx = _bbTmp[0]; c.ny = _bbTmp[1]; c.nz = _bbTmp[2]
  c.depth = depth
  c.rAx = wx - store.positions[ai + 0]
  c.rAy = wy - store.positions[ai + 1]
  c.rAz = wz - store.positions[ai + 2]
  c.rBx = wx - store.positions[bi + 0]
  c.rBy = wy - store.positions[bi + 1]
  c.rBz = wz - store.positions[bi + 2]
  combineMaterials(store, a, b, c)
}

// Clip a polygon against the plane dot(p, t) ≤ offset (Sutherland–Hodgman).
// Points are xyz triples packed into `src`; returns the new count.
function clipPolyByPlane(
  src: Float32Array, n: number,
  tx: number, ty: number, tz: number, offset: number,
  dst: Float32Array,
): number {
  let out = 0
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n
    const px = src[i * 3], py = src[i * 3 + 1], pz = src[i * 3 + 2]
    const qx = src[j * 3], qy = src[j * 3 + 1], qz = src[j * 3 + 2]
    const dp = px * tx + py * ty + pz * tz - offset
    const dq = qx * tx + qy * ty + qz * tz - offset
    if (dp <= 0) {
      dst[out * 3] = px; dst[out * 3 + 1] = py; dst[out * 3 + 2] = pz
      out++
    }
    // Sign change: the edge crosses the plane, so the crossing point joins the
    // polygon. Guard the divide — a denominator this small means the edge lies
    // in the plane, and both endpoints are already handled by the tests above.
    if ((dp < 0 && dq > 0) || (dp > 0 && dq < 0)) {
      const den = dp - dq
      if (Math.abs(den) > 1e-12 && out < 8) {
        const s = dp / den
        dst[out * 3] = px + (qx - px) * s
        dst[out * 3 + 1] = py + (qy - py) * s
        dst[out * 3 + 2] = pz + (qz - pz) * s
        out++
      }
    }
    if (out >= 8) break
  }
  return out
}

// Face-vs-face: clip the incident face against the reference face's four side
// planes, then keep whatever is at or below the reference plane. This is what
// yields a multi-point manifold — the reason a flat panel resting on another
// stops pivoting about a single point.
function emitBoxFaceManifold(
  store: RigidBodyStore,
  a: number, b: number,
  bestAxis: number,
  hAx: number, hAy: number, hAz: number,
  hBx: number, hBy: number, hBz: number,
  cx: number, cy: number, cz: number,
  pool: ContactPool,
): void {
  const refIsA = bestAxis < 3
  const refAxisIdx = refIsA ? bestAxis : bestAxis - 3
  // Reference basis, half extents and centre — all in A's frame. When A is the
  // reference its axes ARE the frame, hence the identity rows.
  const refAx = refIsA ? _bbIdent : _bbBax
  const incAx = refIsA ? _bbBax : _bbIdent
  const refH = _bbRefH, incH = _bbIncH
  refH[0] = refIsA ? hAx : hBx; refH[1] = refIsA ? hAy : hBy; refH[2] = refIsA ? hAz : hBz
  incH[0] = refIsA ? hBx : hAx; incH[1] = refIsA ? hBy : hAy; incH[2] = refIsA ? hBz : hAz
  const refCx = refIsA ? 0 : cx, refCy = refIsA ? 0 : cy, refCz = refIsA ? 0 : cz
  const incCx = refIsA ? cx : 0, incCy = refIsA ? cy : 0, incCz = refIsA ? cz : 0

  // Outward normal of the reference face, pointing at the incident box.
  // _bbAxis runs A → B, so B-as-reference faces the other way.
  const sgn = refIsA ? 1 : -1
  const nx = _bbAxis[0] * sgn, ny = _bbAxis[1] * sgn, nz = _bbAxis[2] * sgn

  // Incident face: the one whose outward normal is most opposed to n.
  let incIdx = 0, incDot = Infinity, incSign = 1
  for (let k = 0; k < 3; k++) {
    const d = incAx[k * 3] * nx + incAx[k * 3 + 1] * ny + incAx[k * 3 + 2] * nz
    const s = d > 0 ? -1 : 1
    const v = d * s
    if (v < incDot) { incDot = v; incIdx = k; incSign = s }
  }

  // Its four corners, from the face centre along the two remaining axes.
  const u = (incIdx + 1) % 3, v = (incIdx + 2) % 3
  const fx = incCx + incAx[incIdx * 3] * incSign * incH[incIdx]
  const fy = incCy + incAx[incIdx * 3 + 1] * incSign * incH[incIdx]
  const fz = incCz + incAx[incIdx * 3 + 2] * incSign * incH[incIdx]
  let n0 = 0
  for (let iu = 0; iu < 2; iu++) {
    const su = iu === 0 ? 1 : -1
    for (let iv = 0; iv < 2; iv++) {
      const sv = iv === 0 ? 1 : -1
      // Wound consistently (++, +−, −−, −+) so the clip walks a real quad.
      const s2 = su === 1 ? sv : -sv
      _bbClip[n0 * 3] = fx + incAx[u * 3] * incH[u] * su + incAx[v * 3] * incH[v] * s2
      _bbClip[n0 * 3 + 1] = fy + incAx[u * 3 + 1] * incH[u] * su + incAx[v * 3 + 1] * incH[v] * s2
      _bbClip[n0 * 3 + 2] = fz + incAx[u * 3 + 2] * incH[u] * su + incAx[v * 3 + 2] * incH[v] * s2
      n0++
    }
  }

  // Clip against the reference face's four side planes.
  const ru = (refAxisIdx + 1) % 3, rv = (refAxisIdx + 2) % 3
  let src = _bbClip, dst = _bbClip2, cnt = n0
  for (let plane = 0; plane < 4; plane++) {
    const ax = plane < 2 ? ru : rv
    const sgn2 = plane % 2 === 0 ? 1 : -1
    const tx = refAx[ax * 3] * sgn2, ty = refAx[ax * 3 + 1] * sgn2, tz = refAx[ax * 3 + 2] * sgn2
    const offset = refCx * tx + refCy * ty + refCz * tz + refH[ax]
    cnt = clipPolyByPlane(src, cnt, tx, ty, tz, offset, dst)
    const t = src; src = dst; dst = t
    if (cnt === 0) return
  }

  // Keep what is at or below the reference face plane.
  const planeD = (refCx + nx * refH[refAxisIdx]) * nx + (refCy + ny * refH[refAxisIdx]) * ny +
    (refCz + nz * refH[refAxisIdx]) * nz
  let kept = 0
  for (let i = 0; i < cnt; i++) {
    const sep = src[i * 3] * nx + src[i * 3 + 1] * ny + src[i * 3 + 2] * nz - planeD
    if (sep > CONTACT_MARGIN) continue
    src[kept * 3] = src[i * 3]
    src[kept * 3 + 1] = src[i * 3 + 1]
    src[kept * 3 + 2] = src[i * 3 + 2]
    _bbDepth[kept] = -sep
    kept++
  }
  if (kept === 0) return

  // Cap the manifold at Bullet's four. Clipping a quad by four planes can
  // reach eight points, and every extra one is another solver row for a
  // patch the deepest four already describe. Deepest-first so the points
  // that matter survive the cut.
  if (kept > 4) {
    for (let i = 1; i < kept; i++) {
      const d = _bbDepth[i]
      const px = src[i * 3], py = src[i * 3 + 1], pz = src[i * 3 + 2]
      let j = i - 1
      while (j >= 0 && _bbDepth[j] < d) {
        _bbDepth[j + 1] = _bbDepth[j]
        src[(j + 1) * 3] = src[j * 3]
        src[(j + 1) * 3 + 1] = src[j * 3 + 1]
        src[(j + 1) * 3 + 2] = src[j * 3 + 2]
        j--
      }
      _bbDepth[j + 1] = d
      src[(j + 1) * 3] = px; src[(j + 1) * 3 + 1] = py; src[(j + 1) * 3 + 2] = pz
    }
    kept = 4
  }
  for (let i = 0; i < kept; i++) {
    emitBoxContact(store, a, b, src[i * 3], src[i * 3 + 1], src[i * 3 + 2], _bbDepth[i], pool)
  }
}

// Edge-vs-edge: one point, at the midpoint of the closest approach between the
// two supporting edges. Single-point is correct here — two crossed edges touch
// at a point, unlike two faces.
function emitBoxEdgeContact(
  store: RigidBodyStore,
  a: number, b: number,
  depth: number,
  i: number, j: number,
  hAx: number, hAy: number, hAz: number,
  hBx: number, hBy: number, hBz: number,
  cx: number, cy: number, cz: number,
  pool: ContactPool,
): void {
  const hA = _bbHA, hB = _bbHB
  hA[0] = hAx; hA[1] = hAy; hA[2] = hAz
  hB[0] = hBx; hB[1] = hBy; hB[2] = hBz
  const nx = _bbAxis[0], ny = _bbAxis[1], nz = _bbAxis[2]

  // A's supporting edge: offset along the two axes that are NOT the edge
  // direction, each toward B.
  const pA = _bbPA
  pA[0] = 0; pA[1] = 0; pA[2] = 0
  for (let k = 0; k < 3; k++) {
    if (k === i) continue
    const d = k === 0 ? nx : k === 1 ? ny : nz
    pA[k] = hA[k] * (d >= 0 ? 1 : -1)
  }
  const dAx = i === 0 ? 1 : 0, dAy = i === 1 ? 1 : 0, dAz = i === 2 ? 1 : 0

  // B's, offset the other way — its edge faces back toward A.
  let pBx = cx, pBy = cy, pBz = cz
  for (let k = 0; k < 3; k++) {
    if (k === j) continue
    const ax = _bbBax[k * 3], ay = _bbBax[k * 3 + 1], az = _bbBax[k * 3 + 2]
    const s = ax * nx + ay * ny + az * nz >= 0 ? -1 : 1
    pBx += ax * hB[k] * s; pBy += ay * hB[k] * s; pBz += az * hB[k] * s
  }
  const dBx = _bbBax[j * 3], dBy = _bbBax[j * 3 + 1], dBz = _bbBax[j * 3 + 2]

  closestPointsTwoSegments(
    pA[0] - dAx * hA[i], pA[1] - dAy * hA[i], pA[2] - dAz * hA[i],
    pA[0] + dAx * hA[i], pA[1] + dAy * hA[i], pA[2] + dAz * hA[i],
    pBx - dBx * hB[j], pBy - dBy * hB[j], pBz - dBz * hB[j],
    pBx + dBx * hB[j], pBy + dBy * hB[j], pBz + dBz * hB[j],
    _cpA, _cpB,
  )
  emitBoxContact(store, a, b,
    (_cpA[0] + _cpB[0]) * 0.5, (_cpA[1] + _cpB[1]) * 0.5, (_cpA[2] + _cpB[2]) * 0.5,
    depth, pool)
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
    flipNormalsFrom(pool, before)
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
    flipNormalsFrom(pool, before)
    return
  }
  if (sA === RigidbodyShape.Capsule && sB === RigidbodyShape.Box) {
    detectCapsuleBox(store, a, b, pool)
    return
  }
  if (sA === RigidbodyShape.Box && sB === RigidbodyShape.Capsule) {
    const before = pool.count
    detectCapsuleBox(store, b, a, pool)
    flipNormalsFrom(pool, before)
    return
  }
  if (sA === RigidbodyShape.Box && sB === RigidbodyShape.Box) {
    detectBoxBox(store, a, b, pool)
  }
}

// After a swapped detect* call, the produced contacts' normals point the wrong
// way and lever arms are mismatched. Flip and re-anchor EVERY contact the call
// emitted, not just the last: capsule-box now returns up to three, and flipping
// one of them would leave the others pulling the pair together instead of
// pushing it apart.
function flipNormalsFrom(pool: ContactPool, from: number): void {
  for (let i = from; i < pool.count; i++) flipOneNormal(pool.get(i))
}

function flipOneNormal(c: Contact): void {
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
