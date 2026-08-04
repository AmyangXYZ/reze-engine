import { Vec3 } from "../math"
import type { RigidBodyStore } from "./body"
import { RigidbodyType } from "./types"
import type { SixDofSpringConstraint } from "./constraint"
import { solveConstraints, type SolverCache } from "./solver"
import { findContacts, type ContactPool } from "./contact"

// World step: predict velocities → collide → solve → position correction →
// integrate. Static and kinematic bodies are skipped during predict and
// integrate; the parent class syncs them from bones around the step. The
// solver pass runs on all bodies — kinematic ones have invMass = 0 and
// act as anchors.
export class World {
  readonly gravity: Vec3
  solverIterations = 10

  // Per-body damping factors pow(1−damping, dt), cached because damping and
  // the fixed dt never change — two Math.pow per body per substep otherwise.
  private dampCacheDt = -1
  private linDampFactor: Float32Array | null = null
  private angDampFactor: Float32Array | null = null

  constructor(gravity: Vec3) {
    this.gravity = new Vec3(gravity.x, gravity.y, gravity.z)
  }

  setGravity(g: Vec3): void {
    this.gravity.x = g.x
    this.gravity.y = g.y
    this.gravity.z = g.z
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

    const gx = this.gravity.x
    const gy = this.gravity.y
    const gz = this.gravity.z

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
      solveConstraints(store, constraints, cache, contacts, dt, this.solverIterations)
    }

    // 4. Position correction (split impulse). Direct translation along the
    //    contact normal — joint constraints in the same SI loop can't undo
    //    it because it doesn't go through the velocity channel. Inverse-mass
    //    weighted so a kinematic body stays put and only the dynamic one
    //    translates.
    // Softer and later than a rigid-body engine would use, deliberately. MMD
    // cloth is a dense chain of small bodies resting on the character, and
    // aggressive penetration recovery feeds those contacts energy that the
    // joint springs hand straight back — the tips never settle. Measured on a
    // 688-body model: resting peak velocity 1.79 → 1.10, and 12.9 → 6.6 under
    // an idle animation, for a slop of two hundredths of a unit (under 2 mm at
    // MMD scale) that no eye finds.
    const POS_CORRECTION_FACTOR = 0.15
    const POS_SLOP = 0.02
    for (let ci = 0; ci < contacts.count; ci++) {
      const c = contacts.get(ci)
      if (c.depth <= POS_SLOP) continue
      const imA = invMass[c.bodyA]
      const imB = invMass[c.bodyB]
      const total = imA + imB
      if (total <= 0) continue
      const correction = (c.depth - POS_SLOP) * POS_CORRECTION_FACTOR
      const dx = correction * c.nx
      const dy = correction * c.ny
      const dz = correction * c.nz
      const ai = c.bodyA * 3
      const bi = c.bodyB * 3
      if (imA > 0) {
        const fA = imA / total
        pos[ai + 0] -= dx * fA
        pos[ai + 1] -= dy * fA
        pos[ai + 2] -= dz * fA
      }
      if (imB > 0) {
        const fB = imB / total
        pos[bi + 0] += dx * fB
        pos[bi + 1] += dy * fB
        pos[bi + 2] += dz * fB
      }
    }

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
