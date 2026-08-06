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
import type { ManifoldCache } from "./manifold"

const BOUNCE_THRESHOLD = 2.0

// Contact error-reduction parameter — btContactSolverInfo::m_erp in Bullet
// 2.75, the build MMD's own physics runs, at its stock 0.2. PMX rigs are
// authored against exactly this response.
//
// Bullet folds penetration recovery into the CONTACT VELOCITY ROW rather than
// translating positions (m_splitImpulse defaults to false in 2.75):
//
//   penetration     = cp.getDistance() + m_linearSlop      // < 0 when overlapping
//   positionalError = -penetration * m_erp / m_timeStep
//   velocityError   = restitution - rel_vel
//   m_rhs           = (positionalError + velocityError) * m_jacDiagABInv
//
// Our `depth` is the negation of Bullet's distance, so positionalError is just
// depth·ERP/dt — one expression covering both cases. Penetrating (depth > 0) it
// pushes apart; separated (depth < 0, a speculative row) it ALLOWS approach at
// the rate that closes the gap, which is what keeps those rows inert until the
// body would really arrive.
const CONTACT_ERP = 0.2

// btContactSolverInfo's SOLVER_RANDMIZE_ORDER. Gauss-Seidel is order-biased —
// rows solved first win — which on a cross-linked lattice shows up as a lean or
// a period-2 chatter. Bullet ships the flag off; it is here to be measured.
const RANDOMIZE_ORDER = true
// btSequentialImpulseConstraintSolver::btRand2, so the shuffle is reproducible
// and a take replays identically.
let _seed = 0
function rand2(): number {
  _seed = (1664525 * _seed + 1013904223) & 0xffffffff
  return _seed >>> 0
}
let _order = new Int32Array(0)

// btContactSolverInfo::m_warmstartingFactor. Seeded impulses are scaled by
// this so a stale cache decays instead of compounding.
const WARMSTARTING_FACTOR = 0.85
// btContactSolverInfo::m_restingContactRestitutionThreshold — past this many
// substeps a contact is "resting" and stops bouncing.
const RESTING_CONTACT_AGE = 2

// btContactSolverInfo::m_linearSlop is 0 in 2.75 — no allowance, the row aims
// at exactly touching. Kept named so the divergence is visible if it ever moves.
const LINEAR_SLOP = 0

// btContactSolverInfo::m_splitImpulse. False in 2.75 (only the demos turn it
// on), but TRUE by default from Bullet 2.8x onward — which is what babylon-mmd
// runs today. MMD is closed-source so its own setting cannot be read, but it is
// widely reported to enable the flag, and our own measurements agree
// independently: turning it on lowered the velocity-reversal rate on every rig
// tested (伊邪那美 0.118 → 0.098, 诗蔻蒂 0.098 → 0.074, 托特 0.054 → 0.028).
// With it on, the penetration term is moved out of the real
// velocity row into a pseudo-velocity ("push") channel that is solved
// separately and integrated straight into the transform, so recovering overlap
// never adds momentum the joint springs can hand back.
const SPLIT_IMPULSE = true
// btContactSolverInfo::m_splitImpulsePenetrationThreshold. Bullet's sign: only
// contacts DEEPER than this go to the split channel; shallower ones keep the
// bias in the real row. Our depth is the negation, hence the flipped compare.
const SPLIT_PENETRATION_THRESHOLD = 0.02

// Pseudo-velocity accumulators for the split channel, zeroed each substep.
let _pushLin = new Float32Array(0)
let _pushAng = new Float32Array(0)

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
// btGeneric6DofSpringConstraint drives a spring as a MOTOR, not as a
// spring-damper: stiffness becomes a velocity target with a force cap.
//
//   velFactor      = fps · damping / numIterations
//   targetVelocity = velFactor · force          (force = err · k)
//   maxMotorForce  = |force| / fps
//
// m_springDamping defaults to 1.0 and PMX carries no damping field, so every
// MMD spring runs at exactly that — which is what these rigs were authored
// against. This replaces an implicit spring-damper with an invented damping
// ratio (ζ = 0.7); ported in isolation the motor form measured 0.0015 mean
// reversal at jerk 0.5, the calmest joint-side result recorded.
const SPRING_DAMPING = 1.0
// btConstraintInfo2::erp, from infoGlobal.m_erp — used only by getMotorFactor.
const INFO_ERP = 0.2

// btTypedConstraint::getMotorFactor. Scales a motor down as it approaches the
// limit it is driving toward, so a stiff spring cannot drive straight through
// its own stop and then be fought by the limit row. That fight is a candidate
// explanation for the period-2 joint chatter seen on bodies carrying NO
// contacts at all (qh_15_5: accumulated joint impulse alternating 8.4 / 14.7).
function getMotorFactor(pos: number, lowLim: number, uppLim: number, vel: number, timeFact: number): number {
  if (lowLim > uppLim) return 1
  if (lowLim === uppLim) return 0
  const deltaMax = vel / timeFact
  if (deltaMax < 0) {
    if (pos >= lowLim && pos < lowLim - deltaMax) return (lowLim - pos) / deltaMax
    return pos < lowLim ? 0 : 1
  }
  if (deltaMax > 0) {
    if (pos <= uppLim && pos > uppLim - deltaMax) return (uppLim - pos) / deltaMax
    return pos > uppLim ? 0 : 1
  }
  return 0
}

// ERP scale for loop-closing constraints (see buildConstraints: joints that
// close a cycle in the joint graph, e.g. the horizontal ring welds of
// cross-linked skirt lattices). A loop over-determines positions — when
// contacts push the lattice, the ring's errors cannot all reach zero, and
// full-rate corrections chase each other around the cycle as violent
// chatter. Loop edges keep shape at a fraction of the correction rate while
// the spanning-tree chains stay stiff.
const LOOP_ERP_SCALE = 1.0
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
  manifolds: ManifoldCache | null = null,
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
    setupConstraint(constraints[c], c, cache, store, dt, invDt, iterations)
  }
  for (let ci = 0; ci < contacts.count; ci++) {
    setupContactRow(contacts.get(ci), lv, av, invMass, W, invDt)
  }
  // Warm start: seed each row from what the same point converged to last
  // substep and APPLY it before iterating, exactly as Bullet does at setup.
  if (manifolds !== null) {
    for (let ci = 0; ci < contacts.count; ci++) {
      warmStartContact(contacts.get(ci), store, manifolds, lv, av, invMass)
    }
  }

  // Bullet 2.75 solves each iteration in three separate passes, in this order:
  // all joint rows, then all contact normal rows, then all friction rows. The
  // friction bound is read from the contact's accumulated impulse AS IT STANDS
  // after the normal pass, so friction tightens as the normal converges. Doing
  // friction inline per contact (our previous shape) bounds it against a normal
  // impulse that half the manifold has not contributed to yet.
  if (RANDOMIZE_ORDER) {
    // Reset per substep. Bullet keeps the seed on the solver instance; ours is
    // module-level, so without this two worlds in one process share a stream
    // and the same scene stops replaying identically — which the determinism
    // test catches. Re-seeding costs nothing: the point is breaking the order
    // bias BETWEEN iterations, not being unpredictable across frames.
    _seed = 0
    if (_order.length < contacts.count) _order = new Int32Array(contacts.count)
    for (let i = 0; i < contacts.count; i++) _order[i] = i
  }
  for (let iter = 0; iter < iterations; iter++) {
    if (RANDOMIZE_ORDER) {
      // Bullet reshuffles between iterations, not once per substep.
      for (let i = 1; i < contacts.count; i++) {
        const j = rand2() % (i + 1)
        const tmp = _order[i]; _order[i] = _order[j]; _order[j] = tmp
      }
    }
    for (let c = 0; c < constraints.length; c++) {
      iterateConstraint(c, cache, lv, av, invMass)
    }
    for (let ci = 0; ci < contacts.count; ci++) {
      iterateContactNormal(contacts.get(RANDOMIZE_ORDER ? _order[ci] : ci), lv, av, invMass)
    }
    for (let ci = 0; ci < contacts.count; ci++) {
      iterateContactFriction(contacts.get(ci), lv, av, invMass)
    }
  }

  // Split-impulse pass: its OWN full set of iterations over the penetration
  // channel only, exactly as Bullet runs it after the main loop.
  if (SPLIT_IMPULSE) {
    const n3 = store.count * 3
    if (_pushLin.length < n3) { _pushLin = new Float32Array(n3); _pushAng = new Float32Array(n3) }
    else { _pushLin.fill(0, 0, n3); _pushAng.fill(0, 0, n3) }
    let any = false
    for (let ci = 0; ci < contacts.count; ci++) if (contacts.get(ci).rhsPenetration !== 0) { any = true; break }
    if (any) {
      if (RANDOMIZE_ORDER) {
    // Reset per substep. Bullet keeps the seed on the solver instance; ours is
    // module-level, so without this two worlds in one process share a stream
    // and the same scene stops replaying identically — which the determinism
    // test catches. Re-seeding costs nothing: the point is breaking the order
    // bias BETWEEN iterations, not being unpredictable across frames.
    _seed = 0
    if (_order.length < contacts.count) _order = new Int32Array(contacts.count)
    for (let i = 0; i < contacts.count; i++) _order[i] = i
  }
  for (let iter = 0; iter < iterations; iter++) {
    if (RANDOMIZE_ORDER) {
      // Bullet reshuffles between iterations, not once per substep.
      for (let i = 1; i < contacts.count; i++) {
        const j = rand2() % (i + 1)
        const tmp = _order[i]; _order[i] = _order[j]; _order[j] = tmp
      }
    }
        for (let ci = 0; ci < contacts.count; ci++) {
          resolveSplitPenetration(contacts.get(ci), invMass)
        }
      }
    }
  }
}

// Bullet's resolveSplitPenetrationImpulse: same Jacobian, but reading and
// writing the pseudo-velocity channel, with its own push-only accumulator.
function resolveSplitPenetration(c: Contact, invMass: Float32Array): void {
  if (c.rhsPenetration === 0) return
  const jacInvN = c.jacInvN
  if (jacInvN <= 0) return
  const imA = invMass[c.bodyA]
  const imB = invMass[c.bodyB]
  if (imA === 0 && imB === 0) return
  const ai = c.bodyA * 3, bi = c.bodyB * 3
  const nx = c.nx, ny = c.ny, nz = c.nz
  const rAx = c.rAx, rAy = c.rAy, rAz = c.rAz
  const rBx = c.rBx, rBy = c.rBy, rBz = c.rBz

  const vAx = _pushLin[ai + 0] + _pushAng[ai + 1] * rAz - _pushAng[ai + 2] * rAy
  const vAy = _pushLin[ai + 1] + _pushAng[ai + 2] * rAx - _pushAng[ai + 0] * rAz
  const vAz = _pushLin[ai + 2] + _pushAng[ai + 0] * rAy - _pushAng[ai + 1] * rAx
  const vBx = _pushLin[bi + 0] + _pushAng[bi + 1] * rBz - _pushAng[bi + 2] * rBy
  const vBy = _pushLin[bi + 1] + _pushAng[bi + 2] * rBx - _pushAng[bi + 0] * rBz
  const vBz = _pushLin[bi + 2] + _pushAng[bi + 0] * rBy - _pushAng[bi + 1] * rBx
  const relVel = (vBx - vAx) * nx + (vBy - vAy) * ny + (vBz - vAz) * nz

  let d = (c.rhsPenetration - relVel) * jacInvN
  const old = c.appliedPushImpulse
  let sum = old + d
  if (sum < 0) { sum = 0; d = -old }
  c.appliedPushImpulse = sum
  if (d === 0) return
  if (imA > 0) {
    _pushLin[ai + 0] -= d * imA * nx
    _pushLin[ai + 1] -= d * imA * ny
    _pushLin[ai + 2] -= d * imA * nz
    _pushAng[ai + 0] -= d * c.cAxN
    _pushAng[ai + 1] -= d * c.cAyN
    _pushAng[ai + 2] -= d * c.cAzN
  }
  if (imB > 0) {
    _pushLin[bi + 0] += d * imB * nx
    _pushLin[bi + 1] += d * imB * ny
    _pushLin[bi + 2] += d * imB * nz
    _pushAng[bi + 0] += d * c.cBxN
    _pushAng[bi + 1] += d * c.cByN
    _pushAng[bi + 2] += d * c.cBzN
  }
}

/** Integrate the accumulated push/turn velocity straight into the transform —
 *  btSolverBody::writebackVelocity(timeStep). Never touches real momentum. */
export function applySplitImpulsePush(store: RigidBodyStore, dt: number): void {
  if (!SPLIT_IMPULSE || _pushLin.length === 0) return
  const pos = store.positions
  const ori = store.orientations
  const invMass = store.invMass
  for (let i = 0; i < store.count; i++) {
    if (invMass[i] <= 0) continue
    const i3 = i * 3, i4 = i * 4
    const px = _pushLin[i3 + 0], py = _pushLin[i3 + 1], pz = _pushLin[i3 + 2]
    const wx = _pushAng[i3 + 0], wy = _pushAng[i3 + 1], wz = _pushAng[i3 + 2]
    if (px === 0 && py === 0 && pz === 0 && wx === 0 && wy === 0 && wz === 0) continue
    pos[i3 + 0] += px * dt
    pos[i3 + 1] += py * dt
    pos[i3 + 2] += pz * dt
    if (wx !== 0 || wy !== 0 || wz !== 0) {
      const qx = ori[i4 + 0], qy = ori[i4 + 1], qz = ori[i4 + 2], qw = ori[i4 + 3]
      const dx = qw * wx + wy * qz - wz * qy
      const dy = qw * wy + wz * qx - wx * qz
      const dz = qw * wz + wx * qy - wy * qx
      const dw = -(wx * qx + wy * qy + wz * qz)
      const h = 0.5 * dt
      let nx2 = qx + dx * h, ny2 = qy + dy * h, nz2 = qz + dz * h, nw2 = qw + dw * h
      const l2 = nx2 * nx2 + ny2 * ny2 + nz2 * nz2 + nw2 * nw2
      if (l2 > 0) {
        const inv = 1 / Math.sqrt(l2)
        ori[i4 + 0] = nx2 * inv; ori[i4 + 1] = ny2 * inv
        ori[i4 + 2] = nz2 * inv; ori[i4 + 3] = nw2 * inv
      }
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
  iterations: number,
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
      // Bullet motor form. Our relVel is (vB − vA)·axis where Bullet's is the
      // negation, so the target carries the opposite sign to Bullet's — which
      // is the same sign the old spring-damper produced, so behaviour keeps its
      // direction and only its magnitude law changes.
      const k = con.springStiffness[i]
      const serr = curr - con.equilibriumPoint[i]
      const force = serr * k
      const velFactor = (invDt * SPRING_DAMPING) / iterations
      const target = -velFactor * force
      // getMotorFactor's `vel` is Bullet's tag_vel: −targetVelocity for a
      // linear axis, which in our sign convention is the target itself.
      const motFact = getMotorFactor(curr, lo, hi, target, invDt * INFO_ERP)
      F[base + LIN_SPR_TGT + i] = motFact * target
      F[base + LIN_SPR_MAX + i] = Math.abs(force) * dt
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
      // Bullet motor form, angular flavour. Bullet's angular force is −err·k
      // and its rel_vel is the negation of our relAv, so the two flips cancel
      // and our target stays positive for positive error, as before.
      const k = con.springStiffness[idx]
      const serr = _angDiffScratch[i] - con.equilibriumPoint[idx]
      const force = -serr * k
      const velFactor = (invDt * SPRING_DAMPING) / iterations
      const target = -velFactor * force
      // tag_vel for a rotational axis is Bullet's targetVelocity, i.e. −ours.
      const motFact = getMotorFactor(_angDiffScratch[i], con.angularMin[i], con.angularMax[i], -target, invDt * INFO_ERP)
      F[angTgt + i] = motFact * target
      F[base + ANG_SPR_MAX + i] = Math.abs(force) * dt
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
      // Motor row: drive toward the target, accumulated impulse clamped to the
      // spring's real force over the step (Bullet's m_lowerLimit/m_upperLimit).
      let dImp = (F[sprTgt + i] - relVelNow) * F[jac + i]
      const maxF = F[sprMax + i]
      const oldS = F[sprImp + i]
      let nextS = oldS + dImp
      if (nextS < -maxF) nextS = -maxF
      else if (nextS > maxF) nextS = maxF
      dImp = nextS - oldS
      F[sprImp + i] = nextS
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
    let j = (F[angTgt + i] - relAv) * F[angJac + i]
    const maxF = F[angSprMax + i]
    const oldS = F[angSprImp + i]
    let nextS = oldS + j
    if (nextS < -maxF) nextS = -maxF
    else if (nextS > maxF) nextS = maxF
    j = nextS - oldS
    F[angSprImp + i] = nextS
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
  const dvx = vBx - vAx
  const dvy = vBy - vAy
  const dvz = vBz - vAz
  const relVelN0 = dvx * nx + dvy * ny + dvz * nz
  // Restitution is switched off once a point has been in contact for more than
  // a couple of substeps (m_restingContactRestitutionThreshold), or resting
  // cloth keeps being handed a little bounce forever.
  c.bounceVel = c.restitution > 0 && relVelN0 < -BOUNCE_THRESHOLD && c.age <= RESTING_CONTACT_AGE
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
  const positionalError = (c.depth - LINEAR_SLOP) * CONTACT_ERP * invDt
  if (SPLIT_IMPULSE && c.depth > SPLIT_PENETRATION_THRESHOLD) {
    c.biasVel = 0
    c.rhsPenetration = positionalError
  } else {
    c.biasVel = positionalError
    c.rhsPenetration = 0
  }

  // Friction direction, Bullet 2.75 style. The stock solverMode is
  // SOLVER_USE_WARMSTARTING | SOLVER_SIMD — SOLVER_USE_2_FRICTION_DIRECTIONS is
  // NOT set, so there is exactly ONE friction row, aligned with the direction
  // the contact is actually sliding:
  //
  //   m_lateralFrictionDir1 = vel - normalWorldOnB * rel_vel
  //
  // with vel = vA − vB. Our dv is vB − vA, so that direction is the negation of
  // dv's tangential part; sign is irrelevant to a row with symmetric ±μN bounds.
  // Below SIMD_EPSILON of sliding Bullet falls back to btPlaneSpace1, ported
  // exactly. Direction is recomputed every substep — friction-direction caching
  // is off in the stock mode.
  let t1x = dvx - nx * relVelN0
  let t1y = dvy - ny * relVelN0
  let t1z = dvz - nz * relVelN0
  let t2x: number, t2y: number, t2z: number
  const lat2 = t1x * t1x + t1y * t1y + t1z * t1z
  if (lat2 > 1.192092896e-07) {
    const inv = 1 / Math.sqrt(lat2)
    t1x *= inv; t1y *= inv; t1z *= inv
    // Second tangent is unused (single-direction mode) but kept orthonormal so
    // the cached Jacobian slots stay well-defined.
    t2x = ny * t1z - nz * t1y
    t2y = nz * t1x - nx * t1z
    t2z = nx * t1y - ny * t1x
  } else {
    // btPlaneSpace1, verbatim.
    if (Math.abs(nz) > 0.7071067811865475) {
      const a = ny * ny + nz * nz
      const k = 1 / Math.sqrt(a)
      t1x = 0; t1y = -nz * k; t1z = ny * k
      t2x = a * k; t2y = -nx * t1z; t2z = nx * t1y
    } else {
      const a = nx * nx + ny * ny
      const k = 1 / Math.sqrt(a)
      t1x = -ny * k; t1y = nx * k; t1z = 0
      t2x = -nz * t1y; t2y = nz * t1x; t2z = a * k
    }
  }
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

// ITER, normal row only — Bullet resolveSingleConstraintRowLowerLimit:
// deltaImpulse = (rhs - relVel) * jacDiagABInv, accumulated impulse clamped at
// the lower limit of 0 (push-only). `rhs` here is restitution + Baumgarte bias.
function iterateContactNormal(
  c: Contact,
  lv: Float32Array,
  av: Float32Array,
  invMass: Float32Array,
): void {
  const jacInvN = c.jacInvN
  if (jacInvN <= 0) return
  const imA = invMass[c.bodyA]
  const imB = invMass[c.bodyB]
  if (imA === 0 && imB === 0) return
  const ai = c.bodyA * 3, bi = c.bodyB * 3
  const rAx = c.rAx, rAy = c.rAy, rAz = c.rAz
  const rBx = c.rBx, rBy = c.rBy, rBz = c.rBz
  const nx = c.nx, ny = c.ny, nz = c.nz

  const vAx = lv[ai + 0] + av[ai + 1] * rAz - av[ai + 2] * rAy
  const vAy = lv[ai + 1] + av[ai + 2] * rAx - av[ai + 0] * rAz
  const vAz = lv[ai + 2] + av[ai + 0] * rAy - av[ai + 1] * rAx
  const vBx = lv[bi + 0] + av[bi + 1] * rBz - av[bi + 2] * rBy
  const vBy = lv[bi + 1] + av[bi + 2] * rBx - av[bi + 0] * rBz
  const vBz = lv[bi + 2] + av[bi + 0] * rBy - av[bi + 1] * rBx
  const relVelN = (vBx - vAx) * nx + (vBy - vAy) * ny + (vBz - vAz) * nz

  let dImpN = (c.bounceVel + c.biasVel - relVelN) * jacInvN
  const oldN = c.appliedNormalImpulse
  let newN = oldN + dImpN
  if (newN < 0) { newN = 0; dImpN = -oldN }
  c.appliedNormalImpulse = newN
  if (dImpN === 0) return
  if (imA > 0) {
    lv[ai + 0] -= dImpN * imA * nx
    lv[ai + 1] -= dImpN * imA * ny
    lv[ai + 2] -= dImpN * imA * nz
    av[ai + 0] -= dImpN * c.cAxN
    av[ai + 1] -= dImpN * c.cAyN
    av[ai + 2] -= dImpN * c.cAzN
  }
  if (imB > 0) {
    lv[bi + 0] += dImpN * imB * nx
    lv[bi + 1] += dImpN * imB * ny
    lv[bi + 2] += dImpN * imB * nz
    av[bi + 0] += dImpN * c.cBxN
    av[bi + 1] += dImpN * c.cByN
    av[bi + 2] += dImpN * c.cBzN
  }
}

// ITER, friction pass. One Coulomb row (stock Bullet has
// SOLVER_USE_2_FRICTION_DIRECTIONS off), bounded by ±friction · the normal
// impulse accumulated so far. Bullet skips the row entirely while that is
// still zero.
function iterateContactFriction(
  c: Contact,
  lv: Float32Array,
  av: Float32Array,
  invMass: Float32Array,
): void {
  const muNormal = c.friction * c.appliedNormalImpulse
  if (muNormal <= 0) return
  if (c.jacInvT1 <= 0) return
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
  const dvx = vBx - vAx, dvy = vBy - vAy, dvz = vBz - vAz

  const relVel = dvx * c.t1x + dvy * c.t1y + dvz * c.t1z
  let dImp = -relVel * c.jacInvT1
  const old = c.appliedFrictionImpulse1
  let next = old + dImp
  if (next < -muNormal) next = -muNormal
  else if (next > muNormal) next = muNormal
  dImp = next - old
  c.appliedFrictionImpulse1 = next
  if (dImp === 0) return
  if (imA > 0) {
    lv[ai + 0] -= dImp * imA * c.t1x
    lv[ai + 1] -= dImp * imA * c.t1y
    lv[ai + 2] -= dImp * imA * c.t1z
    av[ai + 0] -= dImp * c.cAxT1
    av[ai + 1] -= dImp * c.cAyT1
    av[ai + 2] -= dImp * c.cAzT1
  }
  if (imB > 0) {
    lv[bi + 0] += dImp * imB * c.t1x
    lv[bi + 1] += dImp * imB * c.t1y
    lv[bi + 2] += dImp * imB * c.t1z
    av[bi + 0] += dImp * c.cBxT1
    av[bi + 1] += dImp * c.cByT1
    av[bi + 2] += dImp * c.cBzT1
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
//
// ⚠ TRANSPOSED RELATIVE TO BULLET. Bullet's matrixToEulerXYZ (btGeneric6DofConstraint.cpp)
// reads asin(r02) and atan2(-r12, r22); this reads asin(r20) and atan2(-r21, r22).
// Every angular DOF here is therefore the NEGATION of Bullet's, and the code
// compensates by driving angular rows with +k where Bullet uses −k, and by
// treating a low violation as Bullet's high one. The two flips cancel, so
// nothing is wrong — but they must be flipped TOGETHER.
//
// Porting Bullet's sign without also swapping the violation enum makes every
// limit drive its joint further out of range: measured mean velocity-reversal
// 0.59 with jerk 1912 (an explosion), improving to 0.40 and then 0.137 as each
// half was corrected. If you ever port more Bullet joint code, either flip both
// or change this function to Bullet's convention and flip every consumer.
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


// Seed a row from the persistent manifold and apply that impulse up front —
// Bullet's setup does this inline (m_appliedImpulse = cp.m_appliedImpulse *
// m_warmstartingFactor, then applyImpulse). Contact points are matched in body
// A's local frame so the identity survives the bodies moving.
function warmStartContact(
  c: Contact,
  store: RigidBodyStore,
  manifolds: ManifoldCache,
  lv: Float32Array,
  av: Float32Array,
  invMass: Float32Array,
): void {
  const imA = invMass[c.bodyA]
  const imB = invMass[c.bodyB]
  if (imA === 0 && imB === 0) return
  worldToLocal(store, c.bodyA, c.rAx, c.rAy, c.rAz, _mfA)
  const prev = manifolds.find(c.bodyA, c.bodyB, _mfA[0], _mfA[1], _mfA[2])
  if (prev === null) {
    c.age = 0
    return
  }
  c.age = prev.age + 1
  const n = prev.normalImpulse * WARMSTARTING_FACTOR
  const f1 = prev.frictionImpulse1 * WARMSTARTING_FACTOR
  const f2 = prev.frictionImpulse2 * WARMSTARTING_FACTOR
  c.appliedNormalImpulse = n
  c.appliedFrictionImpulse1 = f1
  c.appliedFrictionImpulse2 = f2
  applyContactImpulse(c, lv, av, imA, imB, c.nx, c.ny, c.nz, c.cAxN, c.cAyN, c.cAzN, c.cBxN, c.cByN, c.cBzN, n)
  if (c.jacInvT1 > 0)
    applyContactImpulse(c, lv, av, imA, imB, c.t1x, c.t1y, c.t1z, c.cAxT1, c.cAyT1, c.cAzT1, c.cBxT1, c.cByT1, c.cBzT1, f1)
  if (c.jacInvT2 > 0)
    applyContactImpulse(c, lv, av, imA, imB, c.t2x, c.t2y, c.t2z, c.cAxT2, c.cAyT2, c.cAzT2, c.cBxT2, c.cByT2, c.cBzT2, f2)
}

function applyContactImpulse(
  c: Contact, lv: Float32Array, av: Float32Array,
  imA: number, imB: number,
  dx: number, dy: number, dz: number,
  cAx: number, cAy: number, cAz: number,
  cBx: number, cBy: number, cBz: number,
  j: number,
): void {
  if (j === 0) return
  const ai = c.bodyA * 3, bi = c.bodyB * 3
  if (imA > 0) {
    lv[ai + 0] -= j * imA * dx; lv[ai + 1] -= j * imA * dy; lv[ai + 2] -= j * imA * dz
    av[ai + 0] -= j * cAx; av[ai + 1] -= j * cAy; av[ai + 2] -= j * cAz
  }
  if (imB > 0) {
    lv[bi + 0] += j * imB * dx; lv[bi + 1] += j * imB * dy; lv[bi + 2] += j * imB * dz
    av[bi + 0] += j * cBx; av[bi + 1] += j * cBy; av[bi + 2] += j * cBz
  }
}

/** Store every solved row back into the manifold for the next substep. */
export function saveContactImpulses(
  store: RigidBodyStore,
  contacts: ContactPool,
  manifolds: ManifoldCache,
): void {
  for (let ci = 0; ci < contacts.count; ci++) {
    const c = contacts.get(ci)
    worldToLocal(store, c.bodyA, c.rAx, c.rAy, c.rAz, _mfA)
    worldToLocal(store, c.bodyB, c.rBx, c.rBy, c.rBz, _mfB)
    manifolds.store(
      c.bodyA, c.bodyB,
      _mfA[0], _mfA[1], _mfA[2],
      _mfB[0], _mfB[1], _mfB[2],
      c.appliedNormalImpulse, c.appliedFrictionImpulse1, c.appliedFrictionImpulse2,
      c.age,
    )
  }
  manifolds.endStep()
}

const _mfA = new Float32Array(3)
const _mfB = new Float32Array(3)

// Rotate a world-space lever arm into the body's local frame (R^T · v).
function worldToLocal(store: RigidBodyStore, i: number, x: number, y: number, z: number, out: Float32Array): void {
  const i4 = i * 4
  const qx = store.orientations[i4 + 0], qy = store.orientations[i4 + 1]
  const qz = store.orientations[i4 + 2], qw = store.orientations[i4 + 3]
  // v' = conj(q) * v * q
  const tx = 2 * (qy * z - qz * y)
  const ty = 2 * (qz * x - qx * z)
  const tz = 2 * (qx * y - qy * x)
  out[0] = x - qw * tx + (qy * tz - qz * ty)
  out[1] = y - qw * ty + (qz * tx - qx * tz)
  out[2] = z - qw * tz + (qx * ty - qy * tx)
}
