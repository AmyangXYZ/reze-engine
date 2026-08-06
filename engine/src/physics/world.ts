import { Vec3 } from "../math"
import type { RigidBodyStore } from "./body"
import { RigidbodyType } from "./types"
import type { SixDofSpringConstraint } from "./constraint"
import { solveConstraints, saveContactImpulses, applySplitImpulsePush, type SolverCache } from "./solver"
import { ManifoldCache } from "./manifold"
import { findContacts, type ContactPool } from "./contact"

// World step: predict velocities → collide → solve → position correction →
// integrate. Static and kinematic bodies are skipped during predict and
// integrate; the parent class syncs them from bones around the step. The
// solver pass runs on all bodies — kinematic ones have invMass = 0 and
// act as anchors.
/**
 * Air movement across the whole world.
 *
 * Applied as an acceleration alongside gravity, which is the same shape MMD's
 * own wind plugins take and keeps it free: gravity and wind are summed once per
 * substep, so the per-body predict loop is untouched. Terminal velocity comes
 * out of the existing per-body damping rather than a drag model — PMX authors
 * already tune that damping to get the hang they want, and a second, hidden
 * drag term would fight it.
 */
export interface WindOptions {
  /** Direction the air travels. Normalised on assignment; a zero vector is off. */
  direction: Vec3
  /** Acceleration along it, in gravity's units — the built-in gravity is 98. */
  strength: number
  /** Gust depth, clamped to 0–1. 0 is a steady breeze; 1 swings between still
   *  and double. It is a clamp rather than a free scalar because past 1 the
   *  swing goes negative and the wind blows backwards on every other beat,
   *  which is never what a caller meant by "more turbulent". */
  turbulence?: number
  /** Gusts per second. */
  frequency?: number
}

export class World {
  readonly gravity: Vec3
  solverIterations = 10

  /** Per-pair contact impulse history behind warm starting. */
  private manifolds = new ManifoldCache()

  // Per-body damping factors pow(1−damping, dt), cached because damping and
  // the fixed dt never change — two Math.pow per body per substep otherwise.
  private dampCacheDt = -1
  private linDampFactor: Float32Array | null = null
  private angDampFactor: Float32Array | null = null

  // Wind, resolved to a direction × strength at construction of the options so
  // the step loop only multiplies by a gust scalar.
  private windX = 0
  private windY = 0
  private windZ = 0
  private windTurbulence = 0
  private windFrequency = 0.35
  /** Advances with simulated time, not wall time, so a scrubbed or exported
   *  take gusts identically to a live one. */
  private windClock = 0

  constructor(gravity: Vec3) {
    this.gravity = new Vec3(gravity.x, gravity.y, gravity.z)
  }

  setGravity(g: Vec3): void {
    this.gravity.x = g.x
    this.gravity.y = g.y
    this.gravity.z = g.z
  }

  setWind(wind: WindOptions | null): void {
    // Shape first, and unconditionally: bailing out early on a zero strength
    // used to leave turbulence and frequency holding whatever a previous call
    // set, so raising the strength again resurrected settings the caller had
    // since replaced.
    this.windTurbulence = Math.min(1, Math.max(0, wind?.turbulence ?? 0))
    this.windFrequency = Math.max(0, wind?.frequency ?? 0.35)

    const d = wind?.direction
    const len = d ? Math.hypot(d.x, d.y, d.z) : 0
    if (!wind || !d || len < 1e-9 || wind.strength === 0) {
      this.windX = this.windY = this.windZ = 0
      return
    }
    const s = wind.strength / len
    this.windX = d.x * s
    this.windY = d.y * s
    this.windZ = d.z * s
  }

  getWind(): WindOptions | null {
    const strength = Math.hypot(this.windX, this.windY, this.windZ)
    if (strength === 0) return null
    return {
      direction: new Vec3(this.windX / strength, this.windY / strength, this.windZ / strength),
      strength,
      turbulence: this.windTurbulence,
      frequency: this.windFrequency,
    }
  }

  step(
    store: RigidBodyStore,
    constraints: SixDofSpringConstraint[],
    cache: SolverCache,
    contacts: ContactPool,
    dt: number,
  ): void {
    if (dt <= 0) return

    const N = store.count
    const types = store.type
    const lv = store.linearVelocities
    const av = store.angularVelocities
    const pos = store.positions
    const ori = store.orientations
    const ldamp = store.linearDamping
    const adamp = store.angularDamping
    const invMass = store.invMass

    // Gravity and wind are the same kind of term, so they are summed here and
    // the predict loop below never learns wind exists.
    let gx = this.gravity.x
    let gy = this.gravity.y
    let gz = this.gravity.z
    if (this.windX !== 0 || this.windY !== 0 || this.windZ !== 0) {
      this.windClock += dt
      let gust = 1
      if (this.windTurbulence > 0 && this.windFrequency > 0) {
        // Two incommensurate sines: a single one is a metronome, and real gusts
        // do not repeat on a bar line. Stays within 1 ± turbulence.
        const t = this.windClock * this.windFrequency * Math.PI * 2
        gust = 1 + this.windTurbulence * 0.5 * (Math.sin(t) + Math.sin(t * 0.37 + 1.3))
      }
      gx += this.windX * gust
      gy += this.windY * gust
      gz += this.windZ * gust
    }

    // 1. Predict — gravity + damping. The pow form (vs the linear
    //    1−damping·dt approximation) stays stable at high PMX damping
    //    values like 0.99. Factors are cached (damping and dt are constant).
    if (this.dampCacheDt !== dt || !this.linDampFactor || this.linDampFactor.length !== N) {
      this.dampCacheDt = dt
      this.linDampFactor = new Float32Array(N)
      this.angDampFactor = new Float32Array(N)
      for (let i = 0; i < N; i++) {
        this.linDampFactor[i] = Math.pow(Math.max(0, 1 - ldamp[i]), dt)
        this.angDampFactor[i] = Math.pow(Math.max(0, 1 - adamp[i]), dt)
      }
    }
    const linDamp = this.linDampFactor
    const angDamp = this.angDampFactor!
    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic || invMass[i] <= 0) continue
      const i3 = i * 3
      lv[i3 + 0] += gx * dt
      lv[i3 + 1] += gy * dt
      lv[i3 + 2] += gz * dt
      const ld = linDamp[i]
      const ad = angDamp[i]
      lv[i3 + 0] *= ld; lv[i3 + 1] *= ld; lv[i3 + 2] *= ld
      av[i3 + 0] *= ad; av[i3 + 1] *= ad; av[i3 + 2] *= ad
    }

    // 2. Collide.
    contacts.reset()
    findContacts(store, contacts)

    // 3. Solve joint + contact constraints (velocity-only).
    if (constraints.length > 0 || contacts.count > 0) {
      solveConstraints(store, constraints, cache, contacts, dt, this.solverIterations, this.manifolds)
      applySplitImpulsePush(store, dt)
      saveContactImpulses(store, contacts, this.manifolds)
    } else {
      this.manifolds.clear()
    }

    // 4. Penetration recovery happens in the contact velocity row (Baumgarte,
    //    CONTACT_ERP), not here. Bullet 2.75 — the build MMD's physics runs —
    //    defaults m_splitImpulse to false and folds positionalError straight
    //    into m_rhs, so a separate position-translation pass is a divergence
    //    from what PMX rigs are authored against. It also could not be made to
    //    behave: every contact translated the body by its own full share, so a
    //    panel carrying dozens of rows was shoved out dozens of times.
    // 5. Integrate. Cap angular velocity at π/2 per step — a high-impulse
    //    contact spike on a low-inertia body would otherwise spin past π
    //    in one step and trash the quaternion integration. Linear velocity
    //    is capped at 5 units per step for the same reason: an explosion
    //    guard far above what legit cloth motion ever reaches.
    const MAX_ANGVEL_DT = Math.PI * 0.5
    const MAX_LINVEL_DT = 5
    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic || invMass[i] <= 0) continue
      const i3 = i * 3
      const i4 = i * 4

      const vx = lv[i3 + 0], vy = lv[i3 + 1], vz = lv[i3 + 2]
      const vmag = Math.sqrt(vx * vx + vy * vy + vz * vz)
      if (vmag * dt > MAX_LINVEL_DT) {
        const scale = MAX_LINVEL_DT / (vmag * dt)
        lv[i3 + 0] = vx * scale
        lv[i3 + 1] = vy * scale
        lv[i3 + 2] = vz * scale
      }

      pos[i3 + 0] += lv[i3 + 0] * dt
      pos[i3 + 1] += lv[i3 + 1] * dt
      pos[i3 + 2] += lv[i3 + 2] * dt

      let wx = av[i3 + 0]
      let wy = av[i3 + 1]
      let wz = av[i3 + 2]
      const wmag = Math.sqrt(wx * wx + wy * wy + wz * wz)
      if (wmag * dt > MAX_ANGVEL_DT) {
        const scale = MAX_ANGVEL_DT / (wmag * dt)
        wx *= scale; wy *= scale; wz *= scale
        av[i3 + 0] = wx; av[i3 + 1] = wy; av[i3 + 2] = wz
      }
      if (wx !== 0 || wy !== 0 || wz !== 0) {
        const qx = ori[i4 + 0]
        const qy = ori[i4 + 1]
        const qz = ori[i4 + 2]
        const qw = ori[i4 + 3]

        const dx = qw * wx + wy * qz - wz * qy
        const dy = qw * wy + wz * qx - wx * qz
        const dz = qw * wz + wx * qy - wy * qx
        const dw = -(wx * qx + wy * qy + wz * qz)

        const half = 0.5 * dt
        const nx = qx + dx * half
        const ny = qy + dy * half
        const nz = qz + dz * half
        const nw = qw + dw * half

        const len2 = nx * nx + ny * ny + nz * nz + nw * nw
        if (len2 > 0) {
          const inv = 1 / Math.sqrt(len2)
          ori[i4 + 0] = nx * inv
          ori[i4 + 1] = ny * inv
          ori[i4 + 2] = nz * inv
          ori[i4 + 3] = nw * inv
        }
      }
    }
  }
}
