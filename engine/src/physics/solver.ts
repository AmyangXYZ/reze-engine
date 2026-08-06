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

// Successive over-relaxation factor on CONTACT rows only (joints untouched).
// Each contact row independently drives the relative velocity at its own point
// to zero; when a dress panel carries up to 43 of them at once, they all
// correct the same motion and the body is over-braked, differently every
// substep. Scaling each row's step damps that without changing the fixed
// point — the accumulated impulse still converges to the same answer, just
// approached rather than overshot.
//
// The gain is PER CONTACT, scaled by how contended its two bodies are, not a
// flat constant. A flat factor was measured first and is wrong: it also slows
// the well-conditioned rows, and a body resting on a SINGLE contact can then
// no longer cancel its approach velocity within the iteration budget, so it
// creeps forever instead of settling (托特 peak speed at 15 s: 0.11 at gain
// 1.0, 0.46 at 0.5, 0.72 at 0.3 — a permanent limit cycle on a one-contact
// body). Over-constraint is a local property, so the remedy has to be local.
//
// gain = 1 / max(rows on A, rows on B), floored. One contact → 1.0, exact and
// settling preserved; a dress panel sharing 20 rows → 0.05, damped. This is
// the standard mass-splitting blend toward Jacobi in the contended cluster.
const CONTACT_SOR_MIN = 0.12

// Per-body contact-row counts for the gain above; grown on demand, refilled
// each substep. Module-level so the solve allocates nothing.
let _rowCount = new Int32Array(0)

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

// Spring rows are IMPLICIT spring-dampers (Spring2/ODE-style soft
// constraints): each axis solves
//   relVel⁺ + (k/γ)·err + s·λ = 0,   γ = c + h·k,   s = 1/(h·γ)
// which is the backward-Euler update of that axis's spring-damper —
// unconditionally stable for ANY authored k (no deadbeat clamp, no force
// clamp). The previous clamped velocity-drive could inject velocity far from
// equilibrium but not absorb it near equilibrium (its clamp shrank with the
// error), so resting chains rang forever — the "static dress slowly boils"
// bug. c is derived per row from a fixed damping ratio against the row's
// effective mass: c = 2ζ√(k·m_eff).
const SPRING_DAMPING_ZETA = 0.7

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
  cache: SolverCache,
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
    setupConstraint(constraints[c], c, cache, store, dt, invDt)
  }
  // How many contact rows each body carries this substep — the per-contact
  // relaxation gain below is a function of the more contended of its two
  // bodies. Counting is O(contacts), done once, before any setup.
  if (_rowCount.length < store.count) _rowCount = new Int32Array(store.count)
  else _rowCount.fill(0, 0, store.count)
  // Only DYNAMIC bodies are counted. A static body — above all the ground —
  // collects a row from every body resting on it, and letting that inflate the
  // count crushes the gain of each of those contacts to the floor, so nothing
  // resting on the ground can build enough impulse and it creeps instead of
  // settling. A body with invMass 0 cannot be over-braked in the first place.
  for (let ci = 0; ci < contacts.count; ci++) {
    const c = contacts.get(ci)
    if (invMass[c.bodyA] > 0) _rowCount[c.bodyA]++
    if (invMass[c.bodyB] > 0) _rowCount[c.bodyB]++
  }
  for (let ci = 0; ci < contacts.count; ci++) {
    setupContactRow(contacts.get(ci), lv, av, invMass, W, invDt)
  }

  for (let iter = 0; iter < iterations; iter++) {
    for (let c = 0; c < constraints.length; c++) {
      iterateConstraint(c, cache, lv, av, invMass)
    }
    for (let ci = 0; ci < contacts.count; ci++) {
      iterateContactRow(contacts.get(ci), lv, av, invMass)
    }
  }
}

// ── Flat solver cache (SoA) ──────────────────────────────────────────────────
// One typed-array block instead of a dozen small Float32Arrays per constraint:
// the 10-iteration hot loop walks memory linearly off a single base pointer
// instead of pointer-chasing ~1000 scattered objects. Layout below mirrors the
// old cache fields one-to-one; the math is untouched and bit-identical.
const F_STRIDE = 108
const LEVER_A = 0 // 3
const LEVER_B = 3 // 3
const LIN_AXES = 6 // 9
const LIN_CA = 15 // 9
const LIN_CB = 24 // 9
const LIN_JAC = 33 // 3
const LIN_TGT = 36 // 3
const LIN_LIMIT_IMP = 39 // 3
const LIN_SPR_TGT = 42 // 3
const LIN_SPR_MAX = 45 // 3
const LIN_SPR_IMP = 48 // 3
const ANG_AXES = 51 // 9
const ANG_TGT = 60 // 3
const ANG_JAC = 63 // 3
const ANG_SPR_MAX = 66 // 3
const ANG_SPR_IMP = 69 // 3
const ANG_WA = 72 // 9
const ANG_WB = 81 // 9
const ANG_LIM_AXIS = 90 // 3
const ANG_LIM_WA = 93 // 3
const ANG_LIM_WB = 96 // 3
const ANG_PA_TGT = 99 // 3
const ANG_PA_IMP = 102 // 3
const I_STRIDE = 16
const I_BODY_A = 0
const I_BODY_B = 1
const I_SKIP = 2
const I_LIN_ACT = 3 // 3
const I_LIN_SPR_ACT = 6 // 3
const I_ANG_ACT = 9 // 3
const I_ANG_PA_ACT = 12 // 3
const I_ANG_LIM_ACT = 15

export class SolverCache {
  readonly F: Float32Array
  readonly I: Int32Array
  /** f64 lane for the geodesic row's scalars — they were plain number fields,
   *  and demoting them to f32 would break bit-identical results. */
  readonly D: Float64Array
  constructor(constraints: SixDofSpringConstraint[]) {
    const n = constraints.length
    this.F = new Float32Array(n * F_STRIDE)
    this.I = new Int32Array(n * I_STRIDE)
    this.D = new Float64Array(n * 3)
    for (let i = 0; i < n; i++) {
      this.I[i * I_STRIDE + I_BODY_A] = constraints[i].bodyA
      this.I[i * I_STRIDE + I_BODY_B] = constraints[i].bodyB
    }
  }
}

// SETUP: compute everything that doesn't depend on velocities. Caller
// guarantees pos/ori don't change between this and the iter loop.
function setupConstraint(
  con: SixDofSpringConstraint,
  ci: number,
  cache: SolverCache,
  store: RigidBodyStore,
  dt: number,
  invDt: number,
): void {
  const F = cache.F
  const I = cache.I
  const D = cache.D
  const base = ci * F_STRIDE
  const ib = ci * I_STRIDE
  const db = ci * 3
  const a = con.bodyA
  const b = con.bodyB
  const imA = store.invMass[a]
  const imB = store.invMass[b]
  const W = store.invInertiaWorld
  const a9 = a * 9
  const b9 = b * 9

  const skip = imA === 0 && imB === 0
  I[ib + I_SKIP] = skip ? 1 : 0
  if (skip) return

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
  const lA = base + LEVER_A
  const lB = base + LEVER_B
  F[lA + 0] = rAx; F[lA + 1] = rAy; F[lA + 2] = rAz
  F[lB + 0] = rBx; F[lB + 1] = rBy; F[lB + 2] = rBz

  // linearDiff = TA.basis^T · (TB.origin − TA.origin); axes = TA columns 0/1/2.
  const dxw = _TB[12] - _TA[12]
  const dyw = _TB[13] - _TA[13]
  const dzw = _TB[14] - _TA[14]
  const linDiff0 = _TA[0] * dxw + _TA[1] * dyw + _TA[2] * dzw
  const linDiff1 = _TA[4] * dxw + _TA[5] * dyw + _TA[6] * dzw
  const linDiff2 = _TA[8] * dxw + _TA[9] * dyw + _TA[10] * dzw

  const axes = base + LIN_AXES
  const cA = base + LIN_CA
  const cB = base + LIN_CB
  const jac = base + LIN_JAC
  const tgt = base + LIN_TGT
  const act = ib + I_LIN_ACT

  for (let i = 0; i < 3; i++) {
    const o = i * 3
    const axx = i === 0 ? _TA[0] : i === 1 ? _TA[4] : _TA[8]
    const axy = i === 0 ? _TA[1] : i === 1 ? _TA[5] : _TA[9]
    const axz = i === 0 ? _TA[2] : i === 1 ? _TA[6] : _TA[10]
    F[axes + o + 0] = axx
    F[axes + o + 1] = axy
    F[axes + o + 2] = axz

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
    F[cA + o + 0] = wAx; F[cA + o + 1] = wAy; F[cA + o + 2] = wAz
    F[cB + o + 0] = wBx; F[cB + o + 1] = wBy; F[cB + o + 2] = wBz

    const denom = imA + imB +
      (cAx * wAx + cAy * wAy + cAz * wAz) +
      (cBx * wBx + cBy * wBy + cBz * wBz)
    F[jac + i] = denom > 0 ? 1 / denom : 0

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
    F[tgt + i] = target
    I[act + i] = denom > 0 ? active : 0
    F[base + LIN_LIMIT_IMP + i] = 0
    // A spring on a locked axis is redundant — the bilateral limit row
    // already welds the DOF, and driving it twice overshoots every
    // iteration (PMX rigs routinely put k=100000 springs on locked axes,
    // which turned welded weight-bodies into energy pumps).
    if (con.springEnabled[i] && denom > 0 && lo !== hi) {
      // Implicit spring-damper (see SPRING_DAMPING_ZETA). Stored per axis:
      // cacheLinSpringTarget = −(k/γ)·err (the bias, in velocity units) and
      // cacheLinSpringMaxImp = s (the CFM softness — NOT a clamp anymore).
      const k = con.springStiffness[i]
      const serr = curr - con.equilibriumPoint[i]
      const meff = 1 / denom
      const c = 2 * SPRING_DAMPING_ZETA * Math.sqrt(k * meff)
      const gamma = c + dt * k
      F[base + LIN_SPR_TGT + i] = -(k / gamma) * serr
      F[base + LIN_SPR_MAX + i] = 1 / (dt * gamma)
      F[base + LIN_SPR_IMP + i] = 0
      I[ib + I_LIN_SPR_ACT + i] = 1
    } else if (!(lo === hi && con.isLoop)) {
      I[ib + I_LIN_SPR_ACT + i] = 0
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

  const angAxes = base + ANG_AXES
  F[angAxes + 0] = xx; F[angAxes + 1] = xy; F[angAxes + 2] = xz
  F[angAxes + 3] = yx; F[angAxes + 4] = yy; F[angAxes + 5] = yz
  F[angAxes + 6] = zx; F[angAxes + 7] = zy; F[angAxes + 8] = zz

  // Per-axis angular Jacobians with the full tensors: cache I⁻¹·axis per
  // body plus 1/(axᵀ(I⁻¹A+I⁻¹B)ax) per axis.
  const angJac = base + ANG_JAC
  const angWAs = base + ANG_WA
  const angWBs = base + ANG_WB
  for (let i = 0; i < 3; i++) {
    const o = i * 3
    const axx = F[angAxes + o + 0], axy = F[angAxes + o + 1], axz = F[angAxes + o + 2]
    const wAx = W[a9 + 0] * axx + W[a9 + 1] * axy + W[a9 + 2] * axz
    const wAy = W[a9 + 3] * axx + W[a9 + 4] * axy + W[a9 + 5] * axz
    const wAz = W[a9 + 6] * axx + W[a9 + 7] * axy + W[a9 + 8] * axz
    const wBx = W[b9 + 0] * axx + W[b9 + 1] * axy + W[b9 + 2] * axz
    const wBy = W[b9 + 3] * axx + W[b9 + 4] * axy + W[b9 + 5] * axz
    const wBz = W[b9 + 6] * axx + W[b9 + 7] * axy + W[b9 + 8] * axz
    F[angWAs + o + 0] = wAx; F[angWAs + o + 1] = wAy; F[angWAs + o + 2] = wAz
    F[angWBs + o + 0] = wBx; F[angWBs + o + 1] = wBy; F[angWBs + o + 2] = wBz
    const denom = axx * (wAx + wBx) + axy * (wAy + wBy) + axz * (wAz + wBz)
    F[angJac + i] = denom > 0 ? 1 / denom : 0
  }

  // Per-axis rows carry only the springs. Sign flip vs linear:
  // d(angDiff)/dt = −(ω_B − ω_A)·ax.
  const angTgt = base + ANG_TGT
  const angAct = ib + I_ANG_ACT
  for (let i = 0; i < 3; i++) {
    const idx = i + 3
    // Springs on locked axes are skipped — the limit row welds those, and
    // double-driving a DOF overshoots every iteration (see the linear loop).
    if (con.springEnabled[idx] && F[angJac + i] > 0 && con.angularMin[i] !== con.angularMax[i]) {
      // Implicit spring-damper, angular flavor. F[angJac + i] = 1/denom IS the
      // row's effective inertia. Sign flip vs linear: d(err)/dt = −relAv, so
      // the bias is positive for positive error. cacheAngSpringMaxImp holds
      // the CFM softness s (not a clamp).
      const k = con.springStiffness[idx]
      const serr = _angDiffScratch[i] - con.equilibriumPoint[idx]
      const c = 2 * SPRING_DAMPING_ZETA * Math.sqrt(k * F[angJac + i])
      const gamma = c + dt * k
      F[angTgt + i] = (k / gamma) * serr
      F[base + ANG_SPR_MAX + i] = 1 / (dt * gamma)
      I[angAct + i] = 1
    } else {
      F[angTgt + i] = 0
      I[angAct + i] = 0
    }
    F[base + ANG_SPR_IMP + i] = 0
  }

  // Angular limit handling is hybrid. Small violations (the resting-cloth
  // regime) use per-axis euler rows — they converge cleanly and keep resting
  // cloth dead still. Large violations switch to a single geodesic row toward
  // the euler-clamped target: per-axis euler rows (the Bullet-2.7x approach
  // this port used) become geometrically inconsistent for large errors — near
  // the asin singularity they chase phantom errors and pump angular velocity
  // into the chain instead of converging.
  I[ib + I_ANG_LIM_ACT] = 0
  I[ib + I_ANG_PA_ACT + 0] = 0
  I[ib + I_ANG_PA_ACT + 1] = 0
  I[ib + I_ANG_PA_ACT + 2] = 0
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
        F[base + ANG_PA_IMP + i] = 0
        if (err === 0) {
          I[ib + I_ANG_PA_ACT + i] = 0
          continue
        }
        let target = err * STOP_ERP * erpScale * invDt
        if (target > MAX_ANGULAR_CORRECTION_VEL) target = MAX_ANGULAR_CORRECTION_VEL
        else if (target < -MAX_ANGULAR_CORRECTION_VEL) target = -MAX_ANGULAR_CORRECTION_VEL
        F[base + ANG_PA_TGT + i] = target
        I[ib + I_ANG_PA_ACT + i] = con.angularMin[i] === con.angularMax[i] ? 1 : 2
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
        const lim = base + ANG_LIM_AXIS
        F[lim + 0] = _TA[0] * axx + _TA[4] * axy + _TA[8] * axz
        F[lim + 1] = _TA[1] * axx + _TA[5] * axy + _TA[9] * axz
        F[lim + 2] = _TA[2] * axx + _TA[6] * axy + _TA[10] * axz
        const gWA = base + ANG_LIM_WA
        const gWB = base + ANG_LIM_WB
        F[gWA + 0] = W[a9 + 0] * F[lim + 0] + W[a9 + 1] * F[lim + 1] + W[a9 + 2] * F[lim + 2]
        F[gWA + 1] = W[a9 + 3] * F[lim + 0] + W[a9 + 4] * F[lim + 1] + W[a9 + 5] * F[lim + 2]
        F[gWA + 2] = W[a9 + 6] * F[lim + 0] + W[a9 + 7] * F[lim + 1] + W[a9 + 8] * F[lim + 2]
        F[gWB + 0] = W[b9 + 0] * F[lim + 0] + W[b9 + 1] * F[lim + 1] + W[b9 + 2] * F[lim + 2]
        F[gWB + 1] = W[b9 + 3] * F[lim + 0] + W[b9 + 4] * F[lim + 1] + W[b9 + 5] * F[lim + 2]
        F[gWB + 2] = W[b9 + 6] * F[lim + 0] + W[b9 + 7] * F[lim + 1] + W[b9 + 8] * F[lim + 2]
        const gDenom = F[lim + 0] * (F[gWA + 0] + F[gWB + 0]) + F[lim + 1] * (F[gWA + 1] + F[gWB + 1]) + F[lim + 2] * (F[gWA + 2] + F[gWB + 2])
        D[db + 0] = gDenom > 0 ? 1 / gDenom : 0
        let target = angle * STOP_ERP * erpScale * invDt
        if (target > MAX_ANGULAR_CORRECTION_VEL) target = MAX_ANGULAR_CORRECTION_VEL
        D[db + 1] = target
        I[ib + I_ANG_LIM_ACT] = bilateral ? 1 : 2
      }
    }
  }
  D[db + 2] = 0
}

// ITER: read cache, compute relVel from current lv/av, apply impulse.
function iterateConstraint(
  ci: number,
  cache: SolverCache,
  lv: Float32Array,
  av: Float32Array,
  invMass: Float32Array,
): void {
  const F = cache.F
  const I = cache.I
  const D = cache.D
  const base = ci * F_STRIDE
  const ib = ci * I_STRIDE
  const db = ci * 3
  if (I[ib + I_SKIP] === 1) return
  const a = I[ib + I_BODY_A]
  const b = I[ib + I_BODY_B]
  const ai = a * 3
  const bi = b * 3
  const imA = invMass[a]
  const imB = invMass[b]

  // Linear axes — relVel at the offset point: v_pivot = v_CG + ω × r.
  const lA = base + LEVER_A
  const lB = base + LEVER_B
  const rAx = F[lA + 0], rAy = F[lA + 1], rAz = F[lA + 2]
  const rBx = F[lB + 0], rBy = F[lB + 1], rBz = F[lB + 2]
  const axes = base + LIN_AXES
  const cA = base + LIN_CA
  const cB = base + LIN_CB
  const jac = base + LIN_JAC
  const tgt = base + LIN_TGT
  const act = ib + I_LIN_ACT

  const vAx = lv[ai + 0] + av[ai + 1] * rAz - av[ai + 2] * rAy
  const vAy = lv[ai + 1] + av[ai + 2] * rAx - av[ai + 0] * rAz
  const vAz = lv[ai + 2] + av[ai + 0] * rAy - av[ai + 1] * rAx
  const vBx = lv[bi + 0] + av[bi + 1] * rBz - av[bi + 2] * rBy
  const vBy = lv[bi + 1] + av[bi + 2] * rBx - av[bi + 0] * rBz
  const vBz = lv[bi + 2] + av[bi + 0] * rBy - av[bi + 1] * rBx
  const dvx = vBx - vAx
  const dvy = vBy - vAy
  const dvz = vBz - vAz

  const sprAct = ib + I_LIN_SPR_ACT
  const sprTgt = base + LIN_SPR_TGT
  const sprMax = base + LIN_SPR_MAX
  const sprImp = base + LIN_SPR_IMP
  const limImp = base + LIN_LIMIT_IMP
  for (let i = 0; i < 3; i++) {
    if (!I[act + i] && !I[sprAct + i]) continue
    const o = i * 3
    const axx = F[axes + o + 0], axy = F[axes + o + 1], axz = F[axes + o + 2]
    const relVel = dvx * axx + dvy * axy + dvz * axz
    let j = 0

    // Limit row. Locked axes (act 1) are bilateral equality joints; ranged
    // axes in violation (act 2) are unilateral stops — accumulated impulse
    // clamped to the corrective sign, so the stop pushes back into range but
    // never pulls deeper or brakes natural recovery (a bilateral stop acts
    // as a motor and pumps energy into swinging cloth).
    if (I[act + i]) {
      const target = F[tgt + i]
      let dImp = LIMIT_SOFTNESS_LINEAR * (target - relVel) * F[jac + i]
      if (I[act + i] === 2) {
        const old = F[limImp + i]
        let next = old + dImp
        if (target > 0 ? next < 0 : next > 0) next = 0
        dImp = next - old
        F[limImp + i] = next
      }
      j += dImp
    }

    // Implicit spring-damper row (soft constraint, see setup): CFM-softened
    // with accumulated λ, dissipative by construction. relVel is refreshed
    // with the limit impulse applied just above (j·denom = j / jac) — driving
    // the spring off the stale value double-corrects the DOF.
    if (I[sprAct + i]) {
      const relVelNow = j !== 0 ? relVel + j / F[jac + i] : relVel
      const s = F[sprMax + i] // CFM softness
      const dImp = (F[sprTgt + i] - relVelNow - s * F[sprImp + i]) / (1 / F[jac + i] + s)
      F[sprImp + i] += dImp
      j += dImp
    }

    if (j === 0) continue
    if (imA > 0) {
      lv[ai + 0] -= j * imA * axx
      lv[ai + 1] -= j * imA * axy
      lv[ai + 2] -= j * imA * axz
      av[ai + 0] -= j * F[cA + o + 0]
      av[ai + 1] -= j * F[cA + o + 1]
      av[ai + 2] -= j * F[cA + o + 2]
    }
    if (imB > 0) {
      lv[bi + 0] += j * imB * axx
      lv[bi + 1] += j * imB * axy
      lv[bi + 2] += j * imB * axz
      av[bi + 0] += j * F[cB + o + 0]
      av[bi + 1] += j * F[cB + o + 1]
      av[bi + 2] += j * F[cB + o + 2]
    }
  }

  // Angular axes — relAv = ω_B − ω_A.
  const angAxes = base + ANG_AXES
  const angJac = base + ANG_JAC
  const angWAs = base + ANG_WA
  const angWBs = base + ANG_WB
  const angTgt = base + ANG_TGT
  const angAct = ib + I_ANG_ACT
  const dax = av[bi + 0] - av[ai + 0]
  const day = av[bi + 1] - av[ai + 1]
  const daz = av[bi + 2] - av[ai + 2]
  const angSprMax = base + ANG_SPR_MAX
  const angSprImp = base + ANG_SPR_IMP
  for (let i = 0; i < 3; i++) {
    if (!I[angAct + i]) continue
    const o = i * 3
    const axx = F[angAxes + o + 0], axy = F[angAxes + o + 1], axz = F[angAxes + o + 2]
    const relAv = dax * axx + day * axy + daz * axz
    // Implicit spring-damper row (soft constraint, see setup).
    const s = F[angSprMax + i] // CFM softness
    const j = (F[angTgt + i] - relAv - s * F[angSprImp + i]) / (1 / F[angJac + i] + s)
    F[angSprImp + i] += j
    if (j === 0) continue
    if (imA > 0) {
      av[ai + 0] -= j * F[angWAs + o + 0]
      av[ai + 1] -= j * F[angWAs + o + 1]
      av[ai + 2] -= j * F[angWAs + o + 2]
    }
    if (imB > 0) {
      av[bi + 0] += j * F[angWBs + o + 0]
      av[bi + 1] += j * F[angWBs + o + 1]
      av[bi + 2] += j * F[angWBs + o + 2]
    }
  }

  // Per-axis angular limit rows (small-violation regime), on the derived
  // euler axes. Sign convention matches the springs: positive target reduces
  // positive error via d(angDiff)/dt = −(ω_B − ω_A)·ax.
  const paAct = ib + I_ANG_PA_ACT
  if (I[paAct + 0] || I[paAct + 1] || I[paAct + 2]) {
    const paTgt = base + ANG_PA_TGT
    const paImp = base + ANG_PA_IMP
    for (let i = 0; i < 3; i++) {
      if (!I[paAct + i]) continue
      const o = i * 3
      const axx = F[angAxes + o + 0], axy = F[angAxes + o + 1], axz = F[angAxes + o + 2]
      const relAv =
        (av[bi + 0] - av[ai + 0]) * axx +
        (av[bi + 1] - av[ai + 1]) * axy +
        (av[bi + 2] - av[ai + 2]) * axz
      const target = F[paTgt + i]
      // Locked axes (act 1) are welds — full gain, like the 0.16.3 fold;
      // softness only tempers the unilateral stops.
      const soft = I[paAct + i] === 2 ? LIMIT_SOFTNESS_ANGULAR : 1.0
      let j = soft * (target - relAv) * F[angJac + i]
      if (I[paAct + i] === 2) {
        const old = F[paImp + i]
        let next = old + j
        if (target > 0 ? next < 0 : next > 0) next = 0
        j = next - old
        F[paImp + i] = next
      }
      if (j === 0) continue
      if (imA > 0) {
        av[ai + 0] -= j * F[angWAs + o + 0]
        av[ai + 1] -= j * F[angWAs + o + 1]
        av[ai + 2] -= j * F[angWAs + o + 2]
      }
      if (imB > 0) {
        av[bi + 0] += j * F[angWBs + o + 0]
        av[bi + 1] += j * F[angWBs + o + 1]
        av[bi + 2] += j * F[angWBs + o + 2]
      }
    }
  }

  // Geodesic limit row: drive (ω_B − ω_A)·axis toward the correction target.
  // Unilateral — the accumulated impulse can only push toward the legal
  // region (target is always ≥ 0 along the corrective axis).
  if (I[ib + I_ANG_LIM_ACT] !== 0) {
    const lim = base + ANG_LIM_AXIS
    const nx = F[lim + 0], ny = F[lim + 1], nz = F[lim + 2]
    // Re-read relAv — the spring rows above may have changed av.
    const relAv =
      (av[bi + 0] - av[ai + 0]) * nx +
      (av[bi + 1] - av[ai + 1]) * ny +
      (av[bi + 2] - av[ai + 2]) * nz
    let j = LIMIT_SOFTNESS_ANGULAR * (D[db + 1] - relAv) * D[db + 0]
    if (I[ib + I_ANG_LIM_ACT] === 2) {
      const old = D[db + 2]
      let next = old + j
      if (next < 0) next = 0
      j = next - old
      D[db + 2] = next
    }
    if (j !== 0) {
      const gWA = base + ANG_LIM_WA
      const gWB = base + ANG_LIM_WB
      if (imA > 0) {
        av[ai + 0] -= j * F[gWA + 0]
        av[ai + 1] -= j * F[gWA + 1]
        av[ai + 2] -= j * F[gWA + 2]
      }
      if (imB > 0) {
        av[bi + 0] += j * F[gWB + 0]
        av[bi + 1] += j * F[gWB + 1]
        av[bi + 2] += j * F[gWB + 2]
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
  invDt: number,
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

  // Speculative rows (depth < 0 — the shapes are inside the margin band but
  // NOT touching) must not brake a body that hasn't arrived yet. Their whole
  // job is to stop it crossing the surface within this substep, so the
  // approach speed they leave alone is exactly the one that closes the
  // remaining gap in dt; only the excess above that is cancelled.
  //
  // Without this the row targets relVelN = 0 like a touching contact and
  // stops approaching bodies dead up to CONTACT_MARGIN away from anything.
  // The push-only clamp does NOT prevent that — it only forbids a negative
  // (pulling) impulse, not a large positive one on a body in mid-air. On a
  // dress rig half of all contact rows are speculative and 88% of them fire,
  // which is the field of invisible brakes the cloth was shaking against.
  c.allowedApproachVel = c.depth < 0 ? -c.depth * invDt : 0

  // Relaxation gain, from the more contended of the two bodies (see
  // CONTACT_SOR_MIN). A lone contact keeps gain 1.0 and stays exact.
  const contended = _rowCount[c.bodyA] > _rowCount[c.bodyB] ? _rowCount[c.bodyA] : _rowCount[c.bodyB]
  const gain = contended > 1 ? 1 / contended : 1
  c.sorGain = gain < CONTACT_SOR_MIN ? CONTACT_SOR_MIN : gain

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
    let dImpN = (c.bounceVel - c.allowedApproachVel - relVelN) * jacInvN * c.sorGain
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
  let dImp = -relVel * jacInv * c.sorGain
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
