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

// Bullet spring-motor velocity factor: target velocity = (fps/numIterations)
// × spring force, with the applied impulse clamped to the real spring force
// k·|err|·dt. The big target makes the row deliver its full physical force
// every substep (stiff, MMD-authored hold); the clamp keeps it from ever
// exceeding it (no energy pumping). Matches btGeneric6DofSpringConstraint
// with springDamping=1 and the default 10 solver iterations.
const SPRING_SOLVER_ITERATIONS = 10
// Slack on the spring impulse clamp: MMD rigs are authored against Bullet's
// spring motors, which deliver several times the naive k·err·dt per substep
// through their solver-row impulse bounds. Bounded slack keeps the hold
// stiff (breasts/hair rest near the authored pose) while still capping the
// energy any spring can inject per substep (the slow-boil guard).
const SPRING_FORCE_SLACK = 4

// ERP scale for loop-closing constraints (see buildConstraints: joints that
// close a cycle in the joint graph, e.g. the horizontal ring welds of
// cross-linked skirt lattices). A loop over-determines positions — when
// contacts push the lattice, the ring's errors cannot all reach zero, and
// full-rate corrections chase each other around the cycle as violent
// chatter. Loop edges keep shape at a fraction of the correction rate while
// the spanning-tree chains stay stiff.
const LOOP_ERP_SCALE = 1.0
// Loop-edge LOCKED axes are converted to force-clamped springs instead of
// weld rows: an equality row on a cycle fights the other cycle edges at any
// ERP (the velocity system is over-determined too). A spring bounded by its
// real force k·|err|·dt holds the ring's shape elastically without fighting.
const LOOP_SPRING_K = 900
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

  // World-space inverse inertia tensors for this substep's poses.
  store.updateInvInertiaWorld()
  const W = store.invInertiaWorld

  for (let c = 0; c < constraints.length; c++) {
    setupConstraint(constraints[c], store, dt, invDt)
  }
  for (let ci = 0; ci < contacts.count; ci++) {
    setupContactRow(contacts.get(ci), lv, av, invMass, W)
  }

  for (let iter = 0; iter < iterations; iter++) {
    for (let c = 0; c < constraints.length; c++) {
      iterateConstraint(constraints[c], lv, av, invMass)
    }
    for (let ci = 0; ci < contacts.count; ci++) {
      iterateContactRow(contacts.get(ci), lv, av, invMass)
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
  const W = store.invInertiaWorld
  const a9 = a * 9
  const b9 = b * 9

  con.cacheSkip = imA === 0 && imB === 0
  if (con.cacheSkip) return

  const erpScale = con.isLoop ? LOOP_ERP_SCALE : 1.0

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
    // Tensor-multiplied lever crosses: cache I⁻¹·(r×ax) for application;
    // denominator = (r×ax)ᵀ·I⁻¹·(r×ax).
    const wAx = W[a9 + 0] * cAx + W[a9 + 1] * cAy + W[a9 + 2] * cAz
    const wAy = W[a9 + 3] * cAx + W[a9 + 4] * cAy + W[a9 + 5] * cAz
    const wAz = W[a9 + 6] * cAx + W[a9 + 7] * cAy + W[a9 + 8] * cAz
    const wBx = W[b9 + 0] * cBx + W[b9 + 1] * cBy + W[b9 + 2] * cBz
    const wBy = W[b9 + 3] * cBx + W[b9 + 4] * cBy + W[b9 + 5] * cBz
    const wBz = W[b9 + 6] * cBx + W[b9 + 7] * cBy + W[b9 + 8] * cBz
    cA[o + 0] = wAx; cA[o + 1] = wAy; cA[o + 2] = wAz
    cB[o + 0] = wBx; cB[o + 1] = wBy; cB[o + 2] = wBz

    const denom = imA + imB +
      (cAx * wAx + cAy * wAy + cAz * wAz) +
      (cBx * wBx + cBy * wBy + cBz * wBz)
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
        target = -err * STOP_ERP * erpScale * invDt
        if (target > MAX_LINEAR_CORRECTION_VEL) target = MAX_LINEAR_CORRECTION_VEL
        else if (target < -MAX_LINEAR_CORRECTION_VEL) target = -MAX_LINEAR_CORRECTION_VEL
      }
    }
    tgt[i] = target
    act[i] = denom > 0 ? active : 0
    con.cacheLinLimitImp[i] = 0
    // A spring on a locked axis is redundant — the bilateral limit row
    // already welds the DOF, and driving it twice overshoots every
    // iteration (PMX rigs routinely put k=100000 springs on locked axes,
    // which turned welded weight-bodies into energy pumps).
    if (con.springEnabled[i] && denom > 0 && lo !== hi) {
      // Clamp k to the deadbeat limit: an explicit spring with k·dt² > 1
      // overshoots equilibrium every step and pumps energy — the classic
      // pre-Spring2 Bullet 6dof instability this port inherited.
      const k = Math.min(con.springStiffness[i], invDt * invDt)
      const serr = curr - con.equilibriumPoint[i]
      con.cacheLinSpringTarget[i] = -k * serr * dt
      // Boil guard: per-substep spring impulse bounded by (a slack over) the
      // physical spring force — unbounded spring drives pump energy through
      // the euler-axis mis-scale and slowly boil resting cloth.
      con.cacheLinSpringMaxImp[i] = SPRING_FORCE_SLACK * Math.abs(k * serr) * dt
      con.cacheLinSpringImp[i] = 0
      con.cacheLinSpringActive[i] = 1
    } else if (!(lo === hi && con.isLoop)) {
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

  // Per-axis angular Jacobians with the full tensors: cache I⁻¹·axis per
  // body plus 1/(axᵀ(I⁻¹A+I⁻¹B)ax) per axis.
  const angJac = con.cacheAngJacInv
  const angWAs = con.cacheAngWA
  const angWBs = con.cacheAngWB
  for (let i = 0; i < 3; i++) {
    const o = i * 3
    const axx = angAxes[o + 0], axy = angAxes[o + 1], axz = angAxes[o + 2]
    const wAx = W[a9 + 0] * axx + W[a9 + 1] * axy + W[a9 + 2] * axz
    const wAy = W[a9 + 3] * axx + W[a9 + 4] * axy + W[a9 + 5] * axz
    const wAz = W[a9 + 6] * axx + W[a9 + 7] * axy + W[a9 + 8] * axz
    const wBx = W[b9 + 0] * axx + W[b9 + 1] * axy + W[b9 + 2] * axz
    const wBy = W[b9 + 3] * axx + W[b9 + 4] * axy + W[b9 + 5] * axz
    const wBz = W[b9 + 6] * axx + W[b9 + 7] * axy + W[b9 + 8] * axz
    angWAs[o + 0] = wAx; angWAs[o + 1] = wAy; angWAs[o + 2] = wAz
    angWBs[o + 0] = wBx; angWBs[o + 1] = wBy; angWBs[o + 2] = wBz
    const denom = axx * (wAx + wBx) + axy * (wAy + wBy) + axz * (wAz + wBz)
    angJac[i] = denom > 0 ? 1 / denom : 0
  }

  // Per-axis rows carry only the springs. Sign flip vs linear:
  // d(angDiff)/dt = −(ω_B − ω_A)·ax.
  const angTgt = con.cacheAngTargetVel
  const angAct = con.cacheAngActive
  for (let i = 0; i < 3; i++) {
    const idx = i + 3
    // Springs on locked axes are skipped — the limit row welds those, and
    // double-driving a DOF overshoots every iteration (see the linear loop).
    if (con.springEnabled[idx] && angJac[i] > 0 && con.angularMin[i] !== con.angularMax[i]) {
      const k = Math.min(con.springStiffness[idx], invDt * invDt)
      const serr = _angDiffScratch[i] - con.equilibriumPoint[idx]
      angTgt[i] = k * serr * dt
      con.cacheAngSpringMaxImp[i] = SPRING_FORCE_SLACK * Math.abs(k * serr) * dt
      angAct[i] = 1
    } else {
      angTgt[i] = 0
      angAct[i] = 0
    }
    con.cacheAngSpringImp[i] = 0
  }

  // Angular limit handling is hybrid. Small violations (the resting-cloth
  // regime) use per-axis euler rows — they converge cleanly and keep resting
  // cloth dead still. Large violations switch to a single geodesic row toward
  // the euler-clamped target: per-axis euler rows (the Bullet-2.7x approach
  // this port used) become geometrically inconsistent for large errors — near
  // the asin singularity they chase phantom errors and pump angular velocity
  // into the chain instead of converging.
  con.cacheAngLimActive = 0
  con.cacheAngPAActive[0] = 0
  con.cacheAngPAActive[1] = 0
  con.cacheAngPAActive[2] = 0
  if (imA > 0 || imB > 0) {
    const ex = _angDiffScratch[0], ey = _angDiffScratch[1], ez = _angDiffScratch[2]
    // Free axes (min > max) follow the current angle, i.e. no correction.
    let tx = ex, ty = ey, tz = ez
    if (con.angularMin[0] <= con.angularMax[0]) tx = ex < con.angularMin[0] ? con.angularMin[0] : ex > con.angularMax[0] ? con.angularMax[0] : ex
    if (con.angularMin[1] <= con.angularMax[1]) ty = ey < con.angularMin[1] ? con.angularMin[1] : ey > con.angularMax[1] ? con.angularMax[1] : ey
    if (con.angularMin[2] <= con.angularMax[2]) tz = ez < con.angularMin[2] ? con.angularMin[2] : ez > con.angularMax[2] ? con.angularMax[2] : ez
    const errX = ex - tx, errY = ey - ty, errZ = ez - tz
    const maxErr = Math.max(Math.abs(errX), Math.abs(errY), Math.abs(errZ))
    if (maxErr > 0 && maxErr < GEODESIC_THRESHOLD) {
      // Per-axis euler limit rows. Locked axes are bilateral joints; ranged
      // axes are unilateral stops (sign-clamped accumulation) — a bilateral
      // row on a ranged axis brakes natural recovery every substep and
      // pumps energy into swinging cloth, and the pump grows WITH solver
      // convergence (more iterations enforce the brake harder).
      for (let i = 0; i < 3; i++) {
        const err = i === 0 ? errX : i === 1 ? errY : errZ
        con.cacheAngPAImp[i] = 0
        if (err === 0) {
          con.cacheAngPAActive[i] = 0
          continue
        }
        let target = err * STOP_ERP * erpScale * invDt
        if (target > MAX_ANGULAR_CORRECTION_VEL) target = MAX_ANGULAR_CORRECTION_VEL
        else if (target < -MAX_ANGULAR_CORRECTION_VEL) target = -MAX_ANGULAR_CORRECTION_VEL
        con.cacheAngPATarget[i] = target
        con.cacheAngPAActive[i] = con.angularMin[i] === con.angularMax[i] ? 1 : 2
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
        const gWA = con.cacheAngLimWA
        const gWB = con.cacheAngLimWB
        gWA[0] = W[a9 + 0] * lim[0] + W[a9 + 1] * lim[1] + W[a9 + 2] * lim[2]
        gWA[1] = W[a9 + 3] * lim[0] + W[a9 + 4] * lim[1] + W[a9 + 5] * lim[2]
        gWA[2] = W[a9 + 6] * lim[0] + W[a9 + 7] * lim[1] + W[a9 + 8] * lim[2]
        gWB[0] = W[b9 + 0] * lim[0] + W[b9 + 1] * lim[1] + W[b9 + 2] * lim[2]
        gWB[1] = W[b9 + 3] * lim[0] + W[b9 + 4] * lim[1] + W[b9 + 5] * lim[2]
        gWB[2] = W[b9 + 6] * lim[0] + W[b9 + 7] * lim[1] + W[b9 + 8] * lim[2]
        const gDenom = lim[0] * (gWA[0] + gWB[0]) + lim[1] * (gWA[1] + gWB[1]) + lim[2] * (gWA[2] + gWB[2])
        con.cacheAngLimJacInv = gDenom > 0 ? 1 / gDenom : 0
        let target = angle * STOP_ERP * erpScale * invDt
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
): void {
  if (con.cacheSkip) return
  const a = con.bodyA
  const b = con.bodyB
  const ai = a * 3
  const bi = b * 3
  const imA = invMass[a]
  const imB = invMass[b]

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
  const sprMax = con.cacheLinSpringMaxImp
  const sprImp = con.cacheLinSpringImp
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
    // bounds its aggression, and maxImp bounds loop-edge springs by their
    // real force. relVel is refreshed with the limit impulse applied just
    // above (j·denom = j / jac) — driving the spring off the stale value
    // double-corrects the DOF and overshoots every iteration.
    if (sprAct[i]) {
      const relVelNow = j !== 0 ? relVel + j / jac[i] : relVel
      let dImp = (sprTgt[i] - relVelNow) * jac[i]
      const max = sprMax[i]
      if (max !== Infinity) {
        const old = sprImp[i]
        let next = old + dImp
        if (next > max) next = max
        else if (next < -max) next = -max
        dImp = next - old
        sprImp[i] = next
      }
      j += dImp
    }

    if (j === 0) continue
    if (imA > 0) {
      lv[ai + 0] -= j * imA * axx
      lv[ai + 1] -= j * imA * axy
      lv[ai + 2] -= j * imA * axz
      av[ai + 0] -= j * cA[o + 0]
      av[ai + 1] -= j * cA[o + 1]
      av[ai + 2] -= j * cA[o + 2]
    }
    if (imB > 0) {
      lv[bi + 0] += j * imB * axx
      lv[bi + 1] += j * imB * axy
      lv[bi + 2] += j * imB * axz
      av[bi + 0] += j * cB[o + 0]
      av[bi + 1] += j * cB[o + 1]
      av[bi + 2] += j * cB[o + 2]
    }
  }

  // Angular axes — relAv = ω_B − ω_A.
  const angAxes = con.cacheAngAxes
  const angJac = con.cacheAngJacInv
  const angWAs = con.cacheAngWA
  const angWBs = con.cacheAngWB
  const angTgt = con.cacheAngTargetVel
  const angAct = con.cacheAngActive
  const dax = av[bi + 0] - av[ai + 0]
  const day = av[bi + 1] - av[ai + 1]
  const daz = av[bi + 2] - av[ai + 2]
  const angSprMax = con.cacheAngSpringMaxImp
  const angSprImp = con.cacheAngSpringImp
  for (let i = 0; i < 3; i++) {
    if (!angAct[i]) continue
    const o = i * 3
    const axx = angAxes[o + 0], axy = angAxes[o + 1], axz = angAxes[o + 2]
    const relAv = dax * axx + day * axy + daz * axz
    let j = (angTgt[i] - relAv) * angJac[i]
    {
      const max = angSprMax[i]
      const old = angSprImp[i]
      let next = old + j
      if (next > max) next = max
      else if (next < -max) next = -max
      j = next - old
      angSprImp[i] = next
    }
    if (j === 0) continue
    if (imA > 0) {
      av[ai + 0] -= j * angWAs[o + 0]
      av[ai + 1] -= j * angWAs[o + 1]
      av[ai + 2] -= j * angWAs[o + 2]
    }
    if (imB > 0) {
      av[bi + 0] += j * angWBs[o + 0]
      av[bi + 1] += j * angWBs[o + 1]
      av[bi + 2] += j * angWBs[o + 2]
    }
  }

  // Per-axis angular limit rows (small-violation regime), on the derived
  // euler axes. Sign convention matches the springs: positive target reduces
  // positive error via d(angDiff)/dt = −(ω_B − ω_A)·ax.
  const paAct = con.cacheAngPAActive
  if (paAct[0] || paAct[1] || paAct[2]) {
    const paTgt = con.cacheAngPATarget
    const paImp = con.cacheAngPAImp
    for (let i = 0; i < 3; i++) {
      if (!paAct[i]) continue
      const o = i * 3
      const axx = angAxes[o + 0], axy = angAxes[o + 1], axz = angAxes[o + 2]
      const relAv =
        (av[bi + 0] - av[ai + 0]) * axx +
        (av[bi + 1] - av[ai + 1]) * axy +
        (av[bi + 2] - av[ai + 2]) * axz
      const target = paTgt[i]
      // Locked axes (act 1) are welds — full gain, like the 0.16.3 fold;
      // softness only tempers the unilateral stops.
      const soft = paAct[i] === 2 ? LIMIT_SOFTNESS_ANGULAR : 1.0
      let j = soft * (target - relAv) * angJac[i]
      if (paAct[i] === 2) {
        const old = paImp[i]
        let next = old + j
        if (target > 0 ? next < 0 : next > 0) next = 0
        j = next - old
        paImp[i] = next
      }
      if (j === 0) continue
      if (imA > 0) {
        av[ai + 0] -= j * angWAs[o + 0]
        av[ai + 1] -= j * angWAs[o + 1]
        av[ai + 2] -= j * angWAs[o + 2]
      }
      if (imB > 0) {
        av[bi + 0] += j * angWBs[o + 0]
        av[bi + 1] += j * angWBs[o + 1]
        av[bi + 2] += j * angWBs[o + 2]
      }
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
    let j = LIMIT_SOFTNESS_ANGULAR * (con.cacheAngLimTarget - relAv) * con.cacheAngLimJacInv
    if (con.cacheAngLimActive === 2) {
      const old = con.cacheAngLimImp
      let next = old + j
      if (next < 0) next = 0
      j = next - old
      con.cacheAngLimImp = next
    }
    if (j !== 0) {
      const gWA = con.cacheAngLimWA
      const gWB = con.cacheAngLimWB
      if (imA > 0) {
        av[ai + 0] -= j * gWA[0]
        av[ai + 1] -= j * gWA[1]
        av[ai + 2] -= j * gWA[2]
      }
      if (imB > 0) {
        av[bi + 0] += j * gWB[0]
        av[bi + 1] += j * gWB[1]
        av[bi + 2] += j * gWB[2]
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
  W: Float32Array,
): void {
  const ai = c.bodyA * 3
  const bi = c.bodyB * 3
  const a9 = c.bodyA * 9
  const b9 = c.bodyB * 9
  const imA = invMass[c.bodyA]
  const imB = invMass[c.bodyB]
  const rAx = c.rAx, rAy = c.rAy, rAz = c.rAz
  const rBx = c.rBx, rBy = c.rBy, rBz = c.rBz
  const nx = c.nx, ny = c.ny, nz = c.nz

  // Normal Jacobian. Cached vectors are tensor-multiplied I⁻¹·(r×n).
  const cAxN = rAy * nz - rAz * ny
  const cAyN = rAz * nx - rAx * nz
  const cAzN = rAx * ny - rAy * nx
  const cBxN = rBy * nz - rBz * ny
  const cByN = rBz * nx - rBx * nz
  const cBzN = rBx * ny - rBy * nx
  const wAxN = W[a9 + 0] * cAxN + W[a9 + 1] * cAyN + W[a9 + 2] * cAzN
  const wAyN = W[a9 + 3] * cAxN + W[a9 + 4] * cAyN + W[a9 + 5] * cAzN
  const wAzN = W[a9 + 6] * cAxN + W[a9 + 7] * cAyN + W[a9 + 8] * cAzN
  const wBxN = W[b9 + 0] * cBxN + W[b9 + 1] * cByN + W[b9 + 2] * cBzN
  const wByN = W[b9 + 3] * cBxN + W[b9 + 4] * cByN + W[b9 + 5] * cBzN
  const wBzN = W[b9 + 6] * cBxN + W[b9 + 7] * cByN + W[b9 + 8] * cBzN
  const denomN = imA + imB +
    (cAxN * wAxN + cAyN * wAyN + cAzN * wAzN) +
    (cBxN * wBxN + cByN * wByN + cBzN * wBzN)
  c.cAxN = wAxN; c.cAyN = wAyN; c.cAzN = wAzN
  c.cBxN = wBxN; c.cByN = wByN; c.cBzN = wBzN
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
  const wAxT1 = W[a9 + 0] * cAxT1 + W[a9 + 1] * cAyT1 + W[a9 + 2] * cAzT1
  const wAyT1 = W[a9 + 3] * cAxT1 + W[a9 + 4] * cAyT1 + W[a9 + 5] * cAzT1
  const wAzT1 = W[a9 + 6] * cAxT1 + W[a9 + 7] * cAyT1 + W[a9 + 8] * cAzT1
  const wBxT1 = W[b9 + 0] * cBxT1 + W[b9 + 1] * cByT1 + W[b9 + 2] * cBzT1
  const wByT1 = W[b9 + 3] * cBxT1 + W[b9 + 4] * cByT1 + W[b9 + 5] * cBzT1
  const wBzT1 = W[b9 + 6] * cBxT1 + W[b9 + 7] * cByT1 + W[b9 + 8] * cBzT1
  const denomT1 = imA + imB +
    (cAxT1 * wAxT1 + cAyT1 * wAyT1 + cAzT1 * wAzT1) +
    (cBxT1 * wBxT1 + cByT1 * wByT1 + cBzT1 * wBzT1)
  c.cAxT1 = wAxT1; c.cAyT1 = wAyT1; c.cAzT1 = wAzT1
  c.cBxT1 = wBxT1; c.cByT1 = wByT1; c.cBzT1 = wBzT1
  c.jacInvT1 = denomT1 > 0 ? 1 / denomT1 : 0

  const cAxT2 = rAy * t2z - rAz * t2y
  const cAyT2 = rAz * t2x - rAx * t2z
  const cAzT2 = rAx * t2y - rAy * t2x
  const cBxT2 = rBy * t2z - rBz * t2y
  const cByT2 = rBz * t2x - rBx * t2z
  const cBzT2 = rBx * t2y - rBy * t2x
  const wAxT2 = W[a9 + 0] * cAxT2 + W[a9 + 1] * cAyT2 + W[a9 + 2] * cAzT2
  const wAyT2 = W[a9 + 3] * cAxT2 + W[a9 + 4] * cAyT2 + W[a9 + 5] * cAzT2
  const wAzT2 = W[a9 + 6] * cAxT2 + W[a9 + 7] * cAyT2 + W[a9 + 8] * cAzT2
  const wBxT2 = W[b9 + 0] * cBxT2 + W[b9 + 1] * cByT2 + W[b9 + 2] * cBzT2
  const wByT2 = W[b9 + 3] * cBxT2 + W[b9 + 4] * cByT2 + W[b9 + 5] * cBzT2
  const wBzT2 = W[b9 + 6] * cBxT2 + W[b9 + 7] * cByT2 + W[b9 + 8] * cBzT2
  const denomT2 = imA + imB +
    (cAxT2 * wAxT2 + cAyT2 * wAyT2 + cAzT2 * wAzT2) +
    (cBxT2 * wBxT2 + cByT2 * wByT2 + cBzT2 * wBzT2)
  c.cAxT2 = wAxT2; c.cAyT2 = wAyT2; c.cAzT2 = wAzT2
  c.cBxT2 = wBxT2; c.cByT2 = wByT2; c.cBzT2 = wBzT2
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
): void {
  const imA = invMass[c.bodyA]
  const imB = invMass[c.bodyB]
  if (imA === 0 && imB === 0) return
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
        av[ai + 0] -= dImpN * cAxN
        av[ai + 1] -= dImpN * cAyN
        av[ai + 2] -= dImpN * cAzN
      }
      if (imB > 0) {
        lv[bi + 0] += dImpN * imB * nx
        lv[bi + 1] += dImpN * imB * ny
        lv[bi + 2] += dImpN * imB * nz
        av[bi + 0] += dImpN * cBxN
        av[bi + 1] += dImpN * cByN
        av[bi + 2] += dImpN * cBzN
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
    c.jacInvT1, muNormal, imA, imB, lv, av, 1,
  )
  applyFrictionTangent(
    c, ai, bi, dvx2, dvy2, dvz2,
    c.t2x, c.t2y, c.t2z,
    c.cAxT2, c.cAyT2, c.cAzT2, c.cBxT2, c.cByT2, c.cBzT2,
    c.jacInvT2, muNormal, imA, imB, lv, av, 2,
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
  imA: number, imB: number,
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
    av[ai + 0] -= dImp * cAx
    av[ai + 1] -= dImp * cAy
    av[ai + 2] -= dImp * cAz
  }
  if (imB > 0) {
    lv[bi + 0] += dImp * imB * tx
    lv[bi + 1] += dImp * imB * ty
    lv[bi + 2] += dImp * imB * tz
    av[bi + 0] += dImp * cBx
    av[bi + 1] += dImp * cBy
    av[bi + 2] += dImp * cBz
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
