// 6DOF spring + contact constraint solver. Sequential-impulse projected
// Gauss-Seidel: per axis, target a relative velocity (limit correction +
// spring), apply the impulse needed to reach it. Friction is two Coulomb
// rows per contact, normal is push-only.
//
// Two passes per substep:
//   1. SETUP — for each constraint and contact, compute every quantity that
//      doesn't depend on lv/av (world axes, lever arms, Jacobian denominators,
//      target velocities, friction tangent bases, restitution reference).
//      These are constant during solve since pos/ori/inertia don't change.
//   2. ITERATE — `iterations` passes that read the cache and apply impulses
//      based on the current lv/av. ~2× faster than recomputing per iter.

import { Mat4 } from "../math"
import type { RigidBodyStore } from "./body"
import type { SixDofSpringConstraint } from "./constraint"
import { STOP_ERP } from "./constraint"
import type { Contact, ContactPool } from "./contact"

const BOUNCE_THRESHOLD = 2.0

// Ceilings on limit-correction velocity. In normal operation limit errors are
// tiny; a large error only appears after a discontinuity (teleport, stall,
// deep penetration), and feeding err·ERP/dt to the solver unclamped then
// injects explosion-scale impulses into the chain.
const MAX_LINEAR_CORRECTION_VEL = 120 // units/s
const MAX_ANGULAR_CORRECTION_VEL = 30 // rad/s

// Bullet's limit-motor softness defaults (0.7 translational, 0.5 rotational):
// scale each iteration's limit impulse so the stop engages progressively
// instead of as a hard velocity snap.
const LIMIT_SOFTNESS_LINEAR = 0.7
const LIMIT_SOFTNESS_ANGULAR = 0.5
// Angular limit violations below this switch to per-axis euler rows; above
// it, the single geodesic row takes over (see setupConstraint).
const GEODESIC_THRESHOLD = 0.5 // rad

// Module-level scratch (no per-iter allocations).
const _TA = new Float32Array(16)
const _TB = new Float32Array(16)
const _bodyMatA = new Float32Array(16)
const _bodyMatB = new Float32Array(16)
const _angDiffScratch = new Float32Array(3)
const _quatScratchA = new Float32Array(4)
const _quatScratchB = new Float32Array(4)

export function solveConstraints(
  store: RigidBodyStore,
  constraints: SixDofSpringConstraint[],
  contacts: ContactPool,
  dt: number,
  iterations: number,
): void {
  if (dt <= 0) return
  if (constraints.length === 0 && contacts.count === 0) return

  const invDt = 1 / dt
  const lv = store.linearVelocities
  const av = store.angularVelocities
  const invMass = store.invMass
  const invInertia = store.invInertia

  for (let c = 0; c < constraints.length; c++) {
    setupConstraint(constraints[c], store, dt, invDt)
  }
  for (let ci = 0; ci < contacts.count; ci++) {
    setupContactRow(contacts.get(ci), lv, av, invMass, invInertia)
  }

  for (let iter = 0; iter < iterations; iter++) {
    for (let c = 0; c < constraints.length; c++) {
      iterateConstraint(constraints[c], lv, av, invMass, invInertia)
    }
    for (let ci = 0; ci < contacts.count; ci++) {
      iterateContactRow(contacts.get(ci), lv, av, invMass, invInertia)
    }
  }
}

// SETUP: compute everything that doesn't depend on velocities. Caller
// guarantees pos/ori don't change between this and the iter loop.
function setupConstraint(
  con: SixDofSpringConstraint,
  store: RigidBodyStore,
  dt: number,
  invDt: number,
): void {
  const a = con.bodyA
  const b = con.bodyB
  const imA = store.invMass[a]
  const imB = store.invMass[b]
  const iiA = store.invInertia[a]
  const iiB = store.invInertia[b]

  con.cacheSkip = imA === 0 && imB === 0
  if (con.cacheSkip) return

  buildBodyMat(store, a, _bodyMatA)
  buildBodyMat(store, b, _bodyMatB)
  Mat4.multiplyArrays(_bodyMatA, 0, con.frameA, 0, _TA, 0)
  Mat4.multiplyArrays(_bodyMatB, 0, con.frameB, 0, _TB, 0)

  // Per-body pivots at each body's own joint-frame origin (Spring2-style).
  // Bullet 2.7x's shared mass-weighted anchor (m_AnchorPos) degenerates when
  // the joint is violated by a large distance: the midpoint sits far from
  // both bodies, the lever arms grow with the separation, the Jacobian
  // denominator blows up as err²·invInertia, and the row applies torque
  // instead of closing velocity — the joint "breaks" and the error runs
  // away. Per-body pivots keep the levers bounded by the frame offsets, so
  // the row stays effective no matter how large the violation is.
  const pos = store.positions
  const ai = a * 3
  const bi = b * 3
  const rAx = _TA[12] - pos[ai + 0]
  const rAy = _TA[13] - pos[ai + 1]
  const rAz = _TA[14] - pos[ai + 2]
  const rBx = _TB[12] - pos[bi + 0]
  const rBy = _TB[13] - pos[bi + 1]
  const rBz = _TB[14] - pos[bi + 2]
  const lA = con.cacheLeverA
  const lB = con.cacheLeverB
  lA[0] = rAx; lA[1] = rAy; lA[2] = rAz
  lB[0] = rBx; lB[1] = rBy; lB[2] = rBz

  // linearDiff = TA.basis^T · (TB.origin − TA.origin); axes = TA columns 0/1/2.
  const dxw = _TB[12] - _TA[12]
  const dyw = _TB[13] - _TA[13]
  const dzw = _TB[14] - _TA[14]
  const linDiff0 = _TA[0] * dxw + _TA[1] * dyw + _TA[2] * dzw
  const linDiff1 = _TA[4] * dxw + _TA[5] * dyw + _TA[6] * dzw
  const linDiff2 = _TA[8] * dxw + _TA[9] * dyw + _TA[10] * dzw

  const axes = con.cacheLinAxes
  const cA = con.cacheLinCrossA
  const cB = con.cacheLinCrossB
  const jac = con.cacheLinJacInv
  const tgt = con.cacheLinTargetVel
  const act = con.cacheLinActive

  for (let i = 0; i < 3; i++) {
    const o = i * 3
    const axx = i === 0 ? _TA[0] : i === 1 ? _TA[4] : _TA[8]
    const axy = i === 0 ? _TA[1] : i === 1 ? _TA[5] : _TA[9]
    const axz = i === 0 ? _TA[2] : i === 1 ? _TA[6] : _TA[10]
    axes[o + 0] = axx
    axes[o + 1] = axy
    axes[o + 2] = axz

    const cAx = rAy * axz - rAz * axy
    const cAy = rAz * axx - rAx * axz
    const cAz = rAx * axy - rAy * axx
    const cBx = rBy * axz - rBz * axy
    const cBy = rBz * axx - rBx * axz
    const cBz = rBx * axy - rBy * axx
    cA[o + 0] = cAx; cA[o + 1] = cAy; cA[o + 2] = cAz
    cB[o + 0] = cBx; cB[o + 1] = cBy; cB[o + 2] = cBz

    const denom = imA + imB +
      (cAx * cAx + cAy * cAy + cAz * cAz) * iiA +
      (cBx * cBx + cBy * cBy + cBz * cBz) * iiB
    jac[i] = denom > 0 ? 1 / denom : 0

    const lo = con.linearMin[i]
    const hi = con.linearMax[i]
    const curr = i === 0 ? linDiff0 : i === 1 ? linDiff1 : linDiff2
    // active: 1 = bilateral equality (locked axis — a joint, always on),
    //         2 = unilateral stop (ranged axis in violation).
    let target = 0
    let active = 0
    if (lo <= hi) {
      let err = 0
      if (curr < lo) err = curr - lo
      else if (curr > hi) err = curr - hi
      if (lo === hi) active = 1
      else if (err !== 0) active = 2
      if (err !== 0) {
        target = -err * STOP_ERP * invDt
        if (target > MAX_LINEAR_CORRECTION_VEL) target = MAX_LINEAR_CORRECTION_VEL
        else if (target < -MAX_LINEAR_CORRECTION_VEL) target = -MAX_LINEAR_CORRECTION_VEL
      }
    }
    tgt[i] = target
    act[i] = denom > 0 ? active : 0
    con.cacheLinLimitImp[i] = 0
    if (con.springEnabled[i] && denom > 0) {
      // Clamp k to the deadbeat limit: an explicit spring with k·dt² > 1
      // overshoots equilibrium every step and pumps energy — the classic
      // pre-Spring2 Bullet 6dof instability this port inherited. (¼ margin
      // for Gauss-Seidel coupling in chains.)
      const k = Math.min(con.springStiffness[i], 0.25 * invDt * invDt)
      const serr = curr - con.equilibriumPoint[i]
      con.cacheLinSpringTarget[i] = -k * serr * dt
      con.cacheLinSpringActive[i] = 1
    } else {
      con.cacheLinSpringActive[i] = 0
    }
  }

  // Angular: TA^T · TB → Euler XYZ; axes from TA.col2 × TB.col0.
  const r00 = _TA[0]*_TB[0] + _TA[1]*_TB[1] + _TA[2]*_TB[2]
  const r01 = _TA[0]*_TB[4] + _TA[1]*_TB[5] + _TA[2]*_TB[6]
  const r10 = _TA[4]*_TB[0] + _TA[5]*_TB[1] + _TA[6]*_TB[2]
  const r11 = _TA[4]*_TB[4] + _TA[5]*_TB[5] + _TA[6]*_TB[6]
  const r20 = _TA[8]*_TB[0] + _TA[9]*_TB[1] + _TA[10]*_TB[2]
  const r21 = _TA[8]*_TB[4] + _TA[9]*_TB[5] + _TA[10]*_TB[6]
  const r22 = _TA[8]*_TB[8] + _TA[9]*_TB[9] + _TA[10]*_TB[10]
  matrixToEulerXYZ(r00, r01, r10, r11, r20, r21, r22, _angDiffScratch)

  const a2x = _TA[8],  a2y = _TA[9],  a2z = _TA[10]
  const b0x = _TB[0],  b0y = _TB[1],  b0z = _TB[2]
  let yx = a2y * b0z - a2z * b0y
  let yy = a2z * b0x - a2x * b0z
  let yz = a2x * b0y - a2y * b0x
  let l = Math.hypot(yx, yy, yz)
  if (l > 1e-8) { const inv = 1/l; yx*=inv; yy*=inv; yz*=inv }
  let xx = yy * a2z - yz * a2y
  let xy = yz * a2x - yx * a2z
  let xz = yx * a2y - yy * a2x
  l = Math.hypot(xx, xy, xz)
  if (l > 1e-8) { const inv = 1/l; xx*=inv; xy*=inv; xz*=inv }
  let zx = b0y * yz - b0z * yy
  let zy = b0z * yx - b0x * yz
  let zz = b0x * yy - b0y * yx
  l = Math.hypot(zx, zy, zz)
  if (l > 1e-8) { const inv = 1/l; zx*=inv; zy*=inv; zz*=inv }

  const angAxes = con.cacheAngAxes
  angAxes[0] = xx; angAxes[1] = xy; angAxes[2] = xz
  angAxes[3] = yx; angAxes[4] = yy; angAxes[5] = yz
  angAxes[6] = zx; angAxes[7] = zy; angAxes[8] = zz

  const angDenom = iiA + iiB
  con.cacheAngJacInv = angDenom > 0 ? 1 / angDenom : 0

  // Per-axis rows carry only the springs. Sign flip vs linear:
  // d(angDiff)/dt = −(ω_B − ω_A)·ax.
  const angTgt = con.cacheAngTargetVel
  const angAct = con.cacheAngActive
  for (let i = 0; i < 3; i++) {
    const idx = i + 3
    if (con.springEnabled[idx] && angDenom > 0) {
      // Same deadbeat clamp as the linear springs.
      const k = Math.min(con.springStiffness[idx], 0.25 * invDt * invDt)
      const serr = _angDiffScratch[i] - con.equilibriumPoint[idx]
      angTgt[i] = k * serr * dt
      angAct[i] = 1
    } else {
      angTgt[i] = 0
      angAct[i] = 0
    }
  }

  // Angular limit handling is hybrid. Small violations (the resting-cloth
  // regime) use per-axis euler rows — they converge cleanly and keep resting
  // cloth dead still. Large violations switch to a single geodesic row toward
  // the euler-clamped target: per-axis euler rows (the Bullet-2.7x approach
  // this port used) become geometrically inconsistent for large errors — near
  // the asin singularity they chase phantom errors and pump angular velocity
  // into the chain instead of converging.
  con.cacheAngLimActive = 0
  if (angDenom > 0) {
    const ex = _angDiffScratch[0], ey = _angDiffScratch[1], ez = _angDiffScratch[2]
    // Free axes (min > max) follow the current angle, i.e. no correction.
    let tx = ex, ty = ey, tz = ez
    if (con.angularMin[0] <= con.angularMax[0]) tx = ex < con.angularMin[0] ? con.angularMin[0] : ex > con.angularMax[0] ? con.angularMax[0] : ex
    if (con.angularMin[1] <= con.angularMax[1]) ty = ey < con.angularMin[1] ? con.angularMin[1] : ey > con.angularMax[1] ? con.angularMax[1] : ey
    if (con.angularMin[2] <= con.angularMax[2]) tz = ez < con.angularMin[2] ? con.angularMin[2] : ez > con.angularMax[2] ? con.angularMax[2] : ez
    const errX = ex - tx, errY = ey - ty, errZ = ez - tz
    const maxErr = Math.max(Math.abs(errX), Math.abs(errY), Math.abs(errZ))
    if (maxErr > 0 && maxErr < GEODESIC_THRESHOLD) {
      // Per-axis euler limit rows, folded into the per-axis row set (the
      // spring impulse clamp must not bound a limit row — lift it).
      for (let i = 0; i < 3; i++) {
        const err = i === 0 ? errX : i === 1 ? errY : errZ
        if (err === 0) continue
        let target = err * STOP_ERP * invDt
        if (target > MAX_ANGULAR_CORRECTION_VEL) target = MAX_ANGULAR_CORRECTION_VEL
        else if (target < -MAX_ANGULAR_CORRECTION_VEL) target = -MAX_ANGULAR_CORRECTION_VEL
        angTgt[i] += target
        angAct[i] = 1
      }
    } else if (maxErr > 0) {
      // Bilateral (equality) if any violated axis is locked — a locked axis
      // is a joint, not a stop. Unilateral otherwise.
      const bilateral =
        (tx !== ex && con.angularMin[0] === con.angularMax[0]) ||
        (ty !== ey && con.angularMin[1] === con.angularMax[1]) ||
        (tz !== ez && con.angularMin[2] === con.angularMax[2])
      // The decomposition above satisfies R_rel^T = Rx(x)·Ry(y)·Rz(z), so
      // u = qx·qy·qz is conj(q_rel) and the error rotation (current →
      // clamped target, expressed in TA's frame) is q_E = conj(u_t) ⊗ u.
      eulerXYZQuatInto(ex, ey, ez, _quatScratchA)
      eulerXYZQuatInto(tx, ty, tz, _quatScratchB)
      const ux = _quatScratchA[0], uy = _quatScratchA[1], uz = _quatScratchA[2], uw = _quatScratchA[3]
      const vx = _quatScratchB[0], vy = _quatScratchB[1], vz = _quatScratchB[2], vw = _quatScratchB[3]
      // q_E = conj(v) ⊗ u
      let qex = vw * ux - vx * uw - vy * uz + vz * uy
      let qey = vw * uy + vx * uz - vy * uw - vz * ux
      let qez = vw * uz - vx * uy + vy * ux - vz * uw
      let qew = vw * uw + vx * ux + vy * uy + vz * uz
      if (qew < 0) { qex = -qex; qey = -qey; qez = -qez; qew = -qew }
      const sinHalf = Math.sqrt(qex * qex + qey * qey + qez * qez)
      if (sinHalf > 1e-6) {
        const angle = 2 * Math.atan2(sinHalf, qew)
        const invS = 1 / sinHalf
        const axx = qex * invS, axy = qey * invS, axz = qez * invS
        // Axis lives in TA's frame; TA's basis columns map it to world.
        const lim = con.cacheAngLimAxis
        lim[0] = _TA[0] * axx + _TA[4] * axy + _TA[8] * axz
        lim[1] = _TA[1] * axx + _TA[5] * axy + _TA[9] * axz
        lim[2] = _TA[2] * axx + _TA[6] * axy + _TA[10] * axz
        let target = angle * STOP_ERP * invDt
        if (target > MAX_ANGULAR_CORRECTION_VEL) target = MAX_ANGULAR_CORRECTION_VEL
        con.cacheAngLimTarget = target
        con.cacheAngLimActive = bilateral ? 1 : 2
      }
    }
  }
  con.cacheAngLimImp = 0
}

// ITER: read cache, compute relVel from current lv/av, apply impulse.
function iterateConstraint(
  con: SixDofSpringConstraint,
  lv: Float32Array,
  av: Float32Array,
  invMass: Float32Array,
  invInertia: Float32Array,
): void {
  if (con.cacheSkip) return
  const a = con.bodyA
  const b = con.bodyB
  const ai = a * 3
  const bi = b * 3
  const imA = invMass[a]
  const imB = invMass[b]
  const iiA = invInertia[a]
  const iiB = invInertia[b]

  // Linear axes — relVel at the offset point: v_pivot = v_CG + ω × r.
  const lA = con.cacheLeverA
  const lB = con.cacheLeverB
  const rAx = lA[0], rAy = lA[1], rAz = lA[2]
  const rBx = lB[0], rBy = lB[1], rBz = lB[2]
  const axes = con.cacheLinAxes
  const cA = con.cacheLinCrossA
  const cB = con.cacheLinCrossB
  const jac = con.cacheLinJacInv
  const tgt = con.cacheLinTargetVel
  const act = con.cacheLinActive

  const vAx = lv[ai + 0] + av[ai + 1] * rAz - av[ai + 2] * rAy
  const vAy = lv[ai + 1] + av[ai + 2] * rAx - av[ai + 0] * rAz
  const vAz = lv[ai + 2] + av[ai + 0] * rAy - av[ai + 1] * rAx
  const vBx = lv[bi + 0] + av[bi + 1] * rBz - av[bi + 2] * rBy
  const vBy = lv[bi + 1] + av[bi + 2] * rBx - av[bi + 0] * rBz
  const vBz = lv[bi + 2] + av[bi + 0] * rBy - av[bi + 1] * rBx
  const dvx = vBx - vAx
  const dvy = vBy - vAy
  const dvz = vBz - vAz

  const sprAct = con.cacheLinSpringActive
  const sprTgt = con.cacheLinSpringTarget
  const limImp = con.cacheLinLimitImp
  for (let i = 0; i < 3; i++) {
    if (!act[i] && !sprAct[i]) continue
    const o = i * 3
    const axx = axes[o + 0], axy = axes[o + 1], axz = axes[o + 2]
    const relVel = dvx * axx + dvy * axy + dvz * axz
    let j = 0

    // Limit row. Locked axes (act 1) are bilateral equality joints; ranged
    // axes in violation (act 2) are unilateral stops — accumulated impulse
    // clamped to the corrective sign, so the stop pushes back into range but
    // never pulls deeper or brakes natural recovery (a bilateral stop acts
    // as a motor and pumps energy into swinging cloth).
    if (act[i]) {
      const target = tgt[i]
      let dImp = LIMIT_SOFTNESS_LINEAR * (target - relVel) * jac[i]
      if (act[i] === 2) {
        const old = limImp[i]
        let next = old + dImp
        if (target > 0 ? next < 0 : next > 0) next = 0
        dImp = next - old
        limImp[i] = next
      }
      j += dImp
    }

    // Spring row: velocity-target drive; the deadbeat k-clamp at setup
    // bounds its aggression.
    if (sprAct[i]) {
      j += (sprTgt[i] - relVel) * jac[i]
    }

    if (j === 0) continue
    if (imA > 0) {
      lv[ai + 0] -= j * imA * axx
      lv[ai + 1] -= j * imA * axy
      lv[ai + 2] -= j * imA * axz
      av[ai + 0] -= j * iiA * cA[o + 0]
      av[ai + 1] -= j * iiA * cA[o + 1]
      av[ai + 2] -= j * iiA * cA[o + 2]
    }
    if (imB > 0) {
      lv[bi + 0] += j * imB * axx
      lv[bi + 1] += j * imB * axy
      lv[bi + 2] += j * imB * axz
      av[bi + 0] += j * iiB * cB[o + 0]
      av[bi + 1] += j * iiB * cB[o + 1]
      av[bi + 2] += j * iiB * cB[o + 2]
    }
  }

  // Angular axes — relAv = ω_B − ω_A.
  const angJacInv = con.cacheAngJacInv
  if (angJacInv === 0) return
  const angAxes = con.cacheAngAxes
  const angTgt = con.cacheAngTargetVel
  const angAct = con.cacheAngActive
  const dax = av[bi + 0] - av[ai + 0]
  const day = av[bi + 1] - av[ai + 1]
  const daz = av[bi + 2] - av[ai + 2]
  for (let i = 0; i < 3; i++) {
    if (!angAct[i]) continue
    const o = i * 3
    const axx = angAxes[o + 0], axy = angAxes[o + 1], axz = angAxes[o + 2]
    const relAv = dax * axx + day * axy + daz * axz
    // Spring drive (deadbeat-clamped at setup) plus, for small violations,
    // the folded per-axis limit correction.
    const j = (angTgt[i] - relAv) * angJacInv
    if (j === 0) continue
    if (iiA > 0) {
      av[ai + 0] -= j * iiA * axx
      av[ai + 1] -= j * iiA * axy
      av[ai + 2] -= j * iiA * axz
    }
    if (iiB > 0) {
      av[bi + 0] += j * iiB * axx
      av[bi + 1] += j * iiB * axy
      av[bi + 2] += j * iiB * axz
    }
  }

  // Geodesic limit row: drive (ω_B − ω_A)·axis toward the correction target.
  // Unilateral — the accumulated impulse can only push toward the legal
  // region (target is always ≥ 0 along the corrective axis).
  if (con.cacheAngLimActive) {
    const lim = con.cacheAngLimAxis
    const nx = lim[0], ny = lim[1], nz = lim[2]
    // Re-read relAv — the spring rows above may have changed av.
    const relAv =
      (av[bi + 0] - av[ai + 0]) * nx +
      (av[bi + 1] - av[ai + 1]) * ny +
      (av[bi + 2] - av[ai + 2]) * nz
    let j = LIMIT_SOFTNESS_ANGULAR * (con.cacheAngLimTarget - relAv) * angJacInv
    if (con.cacheAngLimActive === 2) {
      const old = con.cacheAngLimImp
      let next = old + j
      if (next < 0) next = 0
      j = next - old
      con.cacheAngLimImp = next
    }
    if (j !== 0) {
      if (iiA > 0) {
        av[ai + 0] -= j * iiA * nx
        av[ai + 1] -= j * iiA * ny
        av[ai + 2] -= j * iiA * nz
      }
      if (iiB > 0) {
        av[bi + 0] += j * iiB * nx
        av[bi + 1] += j * iiB * ny
        av[bi + 2] += j * iiB * nz
      }
    }
  }
}

// SETUP: pre-compute Jacobians, friction basis, and the bounce reference
// from the *initial* closing velocity (Bullet's pattern — captures restitution
// before iter 1 zeroes out the approach).
function setupContactRow(
  c: Contact,
  lv: Float32Array,
  av: Float32Array,
  invMass: Float32Array,
  invInertia: Float32Array,
): void {
  const ai = c.bodyA * 3
  const bi = c.bodyB * 3
  const imA = invMass[c.bodyA]
  const imB = invMass[c.bodyB]
  const iiA = invInertia[c.bodyA]
  const iiB = invInertia[c.bodyB]
  const rAx = c.rAx, rAy = c.rAy, rAz = c.rAz
  const rBx = c.rBx, rBy = c.rBy, rBz = c.rBz
  const nx = c.nx, ny = c.ny, nz = c.nz

  // Normal Jacobian.
  const cAxN = rAy * nz - rAz * ny
  const cAyN = rAz * nx - rAx * nz
  const cAzN = rAx * ny - rAy * nx
  const cBxN = rBy * nz - rBz * ny
  const cByN = rBz * nx - rBx * nz
  const cBzN = rBx * ny - rBy * nx
  const denomN = imA + imB +
    (cAxN * cAxN + cAyN * cAyN + cAzN * cAzN) * iiA +
    (cBxN * cBxN + cByN * cByN + cBzN * cBzN) * iiB
  c.cAxN = cAxN; c.cAyN = cAyN; c.cAzN = cAzN
  c.cBxN = cBxN; c.cByN = cByN; c.cBzN = cBzN
  c.jacInvN = denomN > 0 ? 1 / denomN : 0

  // Restitution reference, captured from initial relVelN.
  const vAx = lv[ai + 0] + av[ai + 1] * rAz - av[ai + 2] * rAy
  const vAy = lv[ai + 1] + av[ai + 2] * rAx - av[ai + 0] * rAz
  const vAz = lv[ai + 2] + av[ai + 0] * rAy - av[ai + 1] * rAx
  const vBx = lv[bi + 0] + av[bi + 1] * rBz - av[bi + 2] * rBy
  const vBy = lv[bi + 1] + av[bi + 2] * rBx - av[bi + 0] * rBz
  const vBz = lv[bi + 2] + av[bi + 0] * rBy - av[bi + 1] * rBx
  const relVelN0 = (vBx - vAx) * nx + (vBy - vAy) * ny + (vBz - vAz) * nz
  c.bounceVel = c.restitution > 0 && relVelN0 < -BOUNCE_THRESHOLD
    ? -c.restitution * relVelN0
    : 0

  // Friction tangent basis. Pick the axis least aligned with n.
  let t1x: number, t1y: number, t1z: number
  if (Math.abs(nx) < 0.7071) { t1x = 0; t1y = -nz; t1z = ny }
  else { t1x = nz; t1y = 0; t1z = -nx }
  const tl = Math.hypot(t1x, t1y, t1z)
  if (tl > 1e-8) {
    const tInv = 1 / tl
    t1x *= tInv; t1y *= tInv; t1z *= tInv
  } else {
    c.jacInvT1 = 0; c.jacInvT2 = 0
    return
  }
  const t2x = ny * t1z - nz * t1y
  const t2y = nz * t1x - nx * t1z
  const t2z = nx * t1y - ny * t1x
  c.t1x = t1x; c.t1y = t1y; c.t1z = t1z
  c.t2x = t2x; c.t2y = t2y; c.t2z = t2z

  // Friction Jacobians.
  const cAxT1 = rAy * t1z - rAz * t1y
  const cAyT1 = rAz * t1x - rAx * t1z
  const cAzT1 = rAx * t1y - rAy * t1x
  const cBxT1 = rBy * t1z - rBz * t1y
  const cByT1 = rBz * t1x - rBx * t1z
  const cBzT1 = rBx * t1y - rBy * t1x
  const denomT1 = imA + imB +
    (cAxT1 * cAxT1 + cAyT1 * cAyT1 + cAzT1 * cAzT1) * iiA +
    (cBxT1 * cBxT1 + cByT1 * cByT1 + cBzT1 * cBzT1) * iiB
  c.cAxT1 = cAxT1; c.cAyT1 = cAyT1; c.cAzT1 = cAzT1
  c.cBxT1 = cBxT1; c.cByT1 = cByT1; c.cBzT1 = cBzT1
  c.jacInvT1 = denomT1 > 0 ? 1 / denomT1 : 0

  const cAxT2 = rAy * t2z - rAz * t2y
  const cAyT2 = rAz * t2x - rAx * t2z
  const cAzT2 = rAx * t2y - rAy * t2x
  const cBxT2 = rBy * t2z - rBz * t2y
  const cByT2 = rBz * t2x - rBx * t2z
  const cBzT2 = rBx * t2y - rBy * t2x
  const denomT2 = imA + imB +
    (cAxT2 * cAxT2 + cAyT2 * cAyT2 + cAzT2 * cAzT2) * iiA +
    (cBxT2 * cBxT2 + cByT2 * cByT2 + cBzT2 * cBzT2) * iiB
  c.cAxT2 = cAxT2; c.cAyT2 = cAyT2; c.cAzT2 = cAzT2
  c.cBxT2 = cBxT2; c.cByT2 = cByT2; c.cBzT2 = cBzT2
  c.jacInvT2 = denomT2 > 0 ? 1 / denomT2 : 0
}

// ITER: one push-only normal row + two Coulomb friction rows. Friction
// bound depends on the *current* applied normal impulse, so it tightens
// as the normal row converges.
function iterateContactRow(
  c: Contact,
  lv: Float32Array,
  av: Float32Array,
  invMass: Float32Array,
  invInertia: Float32Array,
): void {
  const imA = invMass[c.bodyA]
  const imB = invMass[c.bodyB]
  if (imA === 0 && imB === 0) return
  const iiA = invInertia[c.bodyA]
  const iiB = invInertia[c.bodyB]
  const ai = c.bodyA * 3, bi = c.bodyB * 3
  const rAx = c.rAx, rAy = c.rAy, rAz = c.rAz
  const rBx = c.rBx, rBy = c.rBy, rBz = c.rBz

  const vAx = lv[ai + 0] + av[ai + 1] * rAz - av[ai + 2] * rAy
  const vAy = lv[ai + 1] + av[ai + 2] * rAx - av[ai + 0] * rAz
  const vAz = lv[ai + 2] + av[ai + 0] * rAy - av[ai + 1] * rAx
  const vBx = lv[bi + 0] + av[bi + 1] * rBz - av[bi + 2] * rBy
  const vBy = lv[bi + 1] + av[bi + 2] * rBx - av[bi + 0] * rBz
  const vBz = lv[bi + 2] + av[bi + 0] * rBy - av[bi + 1] * rBx
  const dvx = vBx - vAx
  const dvy = vBy - vAy
  const dvz = vBz - vAz

  // Normal row.
  const jacInvN = c.jacInvN
  if (jacInvN > 0) {
    const nx = c.nx, ny = c.ny, nz = c.nz
    const relVelN = dvx * nx + dvy * ny + dvz * nz
    let dImpN = (c.bounceVel - relVelN) * jacInvN
    const oldN = c.appliedNormalImpulse
    let newN = oldN + dImpN
    if (newN < 0) { newN = 0; dImpN = -oldN }
    c.appliedNormalImpulse = newN
    if (dImpN !== 0) {
      const cAxN = c.cAxN, cAyN = c.cAyN, cAzN = c.cAzN
      const cBxN = c.cBxN, cByN = c.cByN, cBzN = c.cBzN
      if (imA > 0) {
        lv[ai + 0] -= dImpN * imA * nx
        lv[ai + 1] -= dImpN * imA * ny
        lv[ai + 2] -= dImpN * imA * nz
        av[ai + 0] -= dImpN * iiA * cAxN
        av[ai + 1] -= dImpN * iiA * cAyN
        av[ai + 2] -= dImpN * iiA * cAzN
      }
      if (imB > 0) {
        lv[bi + 0] += dImpN * imB * nx
        lv[bi + 1] += dImpN * imB * ny
        lv[bi + 2] += dImpN * imB * nz
        av[bi + 0] += dImpN * iiB * cBxN
        av[bi + 1] += dImpN * iiB * cByN
        av[bi + 2] += dImpN * iiB * cBzN
      }
    }
  }

  // Friction. Bound = ±μ · current normal impulse.
  const muNormal = c.friction * c.appliedNormalImpulse
  if (muNormal <= 0) return

  // Re-read dv after the normal impulse possibly changed lv/av.
  const vAx2 = lv[ai + 0] + av[ai + 1] * rAz - av[ai + 2] * rAy
  const vAy2 = lv[ai + 1] + av[ai + 2] * rAx - av[ai + 0] * rAz
  const vAz2 = lv[ai + 2] + av[ai + 0] * rAy - av[ai + 1] * rAx
  const vBx2 = lv[bi + 0] + av[bi + 1] * rBz - av[bi + 2] * rBy
  const vBy2 = lv[bi + 1] + av[bi + 2] * rBx - av[bi + 0] * rBz
  const vBz2 = lv[bi + 2] + av[bi + 0] * rBy - av[bi + 1] * rBx
  const dvx2 = vBx2 - vAx2
  const dvy2 = vBy2 - vAy2
  const dvz2 = vBz2 - vAz2

  applyFrictionTangent(
    c, ai, bi, dvx2, dvy2, dvz2,
    c.t1x, c.t1y, c.t1z,
    c.cAxT1, c.cAyT1, c.cAzT1, c.cBxT1, c.cByT1, c.cBzT1,
    c.jacInvT1, muNormal, imA, imB, iiA, iiB, lv, av, 1,
  )
  applyFrictionTangent(
    c, ai, bi, dvx2, dvy2, dvz2,
    c.t2x, c.t2y, c.t2z,
    c.cAxT2, c.cAyT2, c.cAzT2, c.cBxT2, c.cByT2, c.cBzT2,
    c.jacInvT2, muNormal, imA, imB, iiA, iiB, lv, av, 2,
  )
}

function applyFrictionTangent(
  c: Contact,
  ai: number, bi: number,
  dvx: number, dvy: number, dvz: number,
  tx: number, ty: number, tz: number,
  cAx: number, cAy: number, cAz: number,
  cBx: number, cBy: number, cBz: number,
  jacInv: number, muNormal: number,
  imA: number, imB: number, iiA: number, iiB: number,
  lv: Float32Array, av: Float32Array,
  slot: 1 | 2,
): void {
  if (jacInv <= 0) return
  const relVel = dvx * tx + dvy * ty + dvz * tz
  let dImp = -relVel * jacInv
  const old = slot === 1 ? c.appliedFrictionImpulse1 : c.appliedFrictionImpulse2
  let next = old + dImp
  if (next < -muNormal) { next = -muNormal; dImp = next - old }
  else if (next > muNormal) { next = muNormal; dImp = next - old }
  if (slot === 1) c.appliedFrictionImpulse1 = next
  else c.appliedFrictionImpulse2 = next

  if (dImp === 0) return
  if (imA > 0) {
    lv[ai + 0] -= dImp * imA * tx
    lv[ai + 1] -= dImp * imA * ty
    lv[ai + 2] -= dImp * imA * tz
    av[ai + 0] -= dImp * iiA * cAx
    av[ai + 1] -= dImp * iiA * cAy
    av[ai + 2] -= dImp * iiA * cAz
  }
  if (imB > 0) {
    lv[bi + 0] += dImp * imB * tx
    lv[bi + 1] += dImp * imB * ty
    lv[bi + 2] += dImp * imB * tz
    av[bi + 0] += dImp * iiB * cBx
    av[bi + 1] += dImp * iiB * cBy
    av[bi + 2] += dImp * iiB * cBz
  }
}

function buildBodyMat(store: RigidBodyStore, i: number, out: Float32Array): void {
  const i3 = i * 3, i4 = i * 4
  Mat4.fromPositionRotationInto(
    store.positions[i3 + 0], store.positions[i3 + 1], store.positions[i3 + 2],
    store.orientations[i4 + 0], store.orientations[i4 + 1], store.orientations[i4 + 2], store.orientations[i4 + 3],
    out,
  )
}

// Quaternion of qx(x) ⊗ qy(y) ⊗ qz(z) (three.js 'XYZ' order).
function eulerXYZQuatInto(x: number, y: number, z: number, out: Float32Array): void {
  const sx = Math.sin(x * 0.5), cx = Math.cos(x * 0.5)
  const sy = Math.sin(y * 0.5), cy = Math.cos(y * 0.5)
  const sz = Math.sin(z * 0.5), cz = Math.cos(z * 0.5)
  out[0] = sx * cy * cz + cx * sy * sz
  out[1] = cx * sy * cz - sx * cy * sz
  out[2] = cx * cy * sz + sx * sy * cz
  out[3] = cx * cy * cz - sx * sy * sz
}

// Euler XYZ from a 3×3 rotation matrix (row-major elements).
function matrixToEulerXYZ(
  r00: number, r01: number,
  r10: number, r11: number,
  r20: number, r21: number, r22: number,
  out: Float32Array,
): void {
  if (r20 < 1) {
    if (r20 > -1) {
      out[0] = Math.atan2(-r21, r22)
      out[1] = Math.asin(r20)
      out[2] = Math.atan2(-r10, r00)
    } else {
      out[0] = -Math.atan2(r01, r11)
      out[1] = -Math.PI * 0.5
      out[2] = 0
    }
  } else {
    out[0] = Math.atan2(r01, r11)
    out[1] = Math.PI * 0.5
    out[2] = 0
  }
}
