import { Vec3, Quat, Mat4 } from "../math"
import type { Rigidbody, Joint } from "./types"
import { RigidbodyType } from "./types"
import { RigidBodyStore } from "./body"
import { World } from "./world"
import { buildConstraints, type SixDofSpringConstraint } from "./constraint"
import { ContactPool } from "./contact"

const _bodyMat = new Float32Array(16)
const _boneMat = new Float32Array(16)
const _scratchQuat = new Quat(0, 0, 0, 1)

// Static / kinematic bodies follow their bone via boneWorld × bodyOffset;
// dynamic bodies integrate under gravity + constraints and write their pose
// back via bodyWorld × bodyOffsetInverse.
export class RezePhysics {
  private rigidbodies: Rigidbody[]
  private joints: Joint[]
  private store: RigidBodyStore
  private world: World
  private constraints: SixDofSpringConstraint[]
  private contacts: ContactPool
  private firstFrame = true
  private timeAccum = 0
  private readonly fixedTimeStep = 1 / 60
  private readonly maxSubSteps = 6

  // Fixed-timestep render interpolation ("Fix Your Timestep"): the dynamic body pose is
  // rendered as lerp(prev, curr, alpha) between the last two completed substeps, where
  // alpha = leftover accumulator / fixedTimeStep. Removes the temporal aliasing that shows
  // as hair/cloth judder when the render rate doesn't line up with the 60Hz physics step.
  private prevPositions: Float32Array
  private prevOrientations: Float32Array

  constructor(rigidbodies: Rigidbody[], joints: Joint[] = []) {
    this.rigidbodies = rigidbodies
    this.joints = joints
    this.store = new RigidBodyStore(rigidbodies)
    this.world = new World(new Vec3(0, -98, 0))
    this.constraints = buildConstraints(rigidbodies, joints)
    this.contacts = new ContactPool()
    this.prevPositions = new Float32Array(this.store.count * 3)
    this.prevOrientations = new Float32Array(this.store.count * 4)
  }

  // Snapshot the current body pose as the interpolation "previous" state.
  private savePrevState(): void {
    this.prevPositions.set(this.store.positions)
    this.prevOrientations.set(this.store.orientations)
  }

  setGravity(gravity: Vec3): void {
    this.world.setGravity(gravity)
  }
  getGravity(): Vec3 {
    return this.world.gravity
  }
  getRigidbodies(): Rigidbody[] {
    return this.rigidbodies
  }
  getJoints(): Joint[] {
    return this.joints
  }
  getStore(): RigidBodyStore {
    return this.store
  }

  getRigidbodyTransforms(): Array<{ position: Vec3; rotation: Quat }> {
    const out: Array<{ position: Vec3; rotation: Quat }> = []
    const pos = this.store.positions
    const ori = this.store.orientations
    for (let i = 0; i < this.store.count; i++) {
      const i3 = i * 3
      const i4 = i * 4
      out.push({
        position: new Vec3(pos[i3 + 0], pos[i3 + 1], pos[i3 + 2]),
        rotation: new Quat(ori[i4 + 0], ori[i4 + 1], ori[i4 + 2], ori[i4 + 3]),
      })
    }
    return out
  }

  // Snap dynamic bodies back to their bone-driven pose, zero velocities.
  // Used when the simulation diverged or the user scrubbed the timeline.
  reset(boneWorldMatrices: Mat4[]): void {
    if (this.firstFrame) return
    this.snapBodiesToBones(boneWorldMatrices)
    this.savePrevState() // prev == curr after a snap, so no interpolation jump
    this.timeAccum = 0
  }

  step(dt: number, boneWorldMatrices: Mat4[], boneInverseBindMatrices: Float32Array): void {
    if (this.firstFrame) {
      this.store.computeBoneOffsets(boneInverseBindMatrices)
      // Start at current bone pose, not the PMX bind pose, so animations
      // that skip frame 0 don't pop bodies on first step.
      this.snapBodiesToBones(boneWorldMatrices)
      this.savePrevState()
      this.firstFrame = false
    }

    // Sync once per render frame; kinematic targets don't change between
    // substeps. Render dt is used to derive kinematic velocities from the
    // bone-pose delta so joints feel the kinematic motion.
    this.syncFromBones(boneWorldMatrices, dt)

    // Fixed-timestep substeps. The maxSubSteps cap prevents runaway after
    // a long stall (tab backgrounded, etc.). Snapshot the pose before each step so
    // after the loop prevState is one substep behind the live (current) state.
    this.timeAccum += dt
    let sub = 0
    while (this.timeAccum >= this.fixedTimeStep && sub < this.maxSubSteps) {
      this.savePrevState()
      this.world.step(this.store, this.constraints, this.contacts, this.fixedTimeStep)
      this.timeAccum -= this.fixedTimeStep
      sub++
    }
    if (sub === this.maxSubSteps) this.timeAccum = 0

    // Fraction into the next (not-yet-taken) step; always in [0, 1).
    const alpha = this.fixedTimeStep > 0 ? this.timeAccum / this.fixedTimeStep : 0
    this.applyDynamicsToBones(boneWorldMatrices, alpha)
  }

  // Snap all bone-bound bodies to boneWorld × bodyOffset, zero velocities.
  private snapBodiesToBones(boneWorldMatrices: Mat4[]): void {
    const N = this.store.count
    const offsets = this.store.bodyOffsetMatrix
    const positions = this.store.positions
    const orientations = this.store.orientations
    const lv = this.store.linearVelocities
    const av = this.store.angularVelocities
    const boneIdx = this.store.boneIndex

    for (let i = 0; i < N; i++) {
      const b = boneIdx[i]
      if (b < 0 || b >= boneWorldMatrices.length) continue

      Mat4.multiplyArrays(boneWorldMatrices[b].values, 0, offsets, i * 16, _bodyMat, 0)

      const i3 = i * 3
      const i4 = i * 4
      positions[i3 + 0] = _bodyMat[12]
      positions[i3 + 1] = _bodyMat[13]
      positions[i3 + 2] = _bodyMat[14]
      Mat4.toQuatFromArrayInto(_bodyMat, 0, _scratchQuat)
      orientations[i4 + 0] = _scratchQuat.x
      orientations[i4 + 1] = _scratchQuat.y
      orientations[i4 + 2] = _scratchQuat.z
      orientations[i4 + 3] = _scratchQuat.w

      lv[i3 + 0] = 0
      lv[i3 + 1] = 0
      lv[i3 + 2] = 0
      av[i3 + 0] = 0
      av[i3 + 1] = 0
      av[i3 + 2] = 0
    }
  }

  // Pull Static / Kinematic bodies to their bones and derive velocities
  // from the bone-pose delta — joints attached to fast limbs need to see
  // the kinematic motion, not just the position jump, or dependent cloth
  // bodies lag behind quick movement.
  private syncFromBones(boneWorldMatrices: Mat4[], dt: number): void {
    const N = this.store.count
    const offsets = this.store.bodyOffsetMatrix
    const positions = this.store.positions
    const orientations = this.store.orientations
    const lv = this.store.linearVelocities
    const av = this.store.angularVelocities
    const types = this.store.type
    const boneIdx = this.store.boneIndex
    const invDt = dt > 0 ? 1 / dt : 0

    for (let i = 0; i < N; i++) {
      const t = types[i]
      if (t !== RigidbodyType.Static && t !== RigidbodyType.Kinematic) continue
      const b = boneIdx[i]
      if (b < 0 || b >= boneWorldMatrices.length) continue

      Mat4.multiplyArrays(boneWorldMatrices[b].values, 0, offsets, i * 16, _bodyMat, 0)

      const i3 = i * 3
      const i4 = i * 4

      // Save previous transform for the velocity diff. invDt = 0 (first
      // frame / reset) skips the diff and zeros velocities.
      const oldPx = positions[i3 + 0],
        oldPy = positions[i3 + 1],
        oldPz = positions[i3 + 2]
      const oldOx = orientations[i4 + 0],
        oldOy = orientations[i4 + 1]
      const oldOz = orientations[i4 + 2],
        oldOw = orientations[i4 + 3]

      positions[i3 + 0] = _bodyMat[12]
      positions[i3 + 1] = _bodyMat[13]
      positions[i3 + 2] = _bodyMat[14]
      Mat4.toQuatFromArrayInto(_bodyMat, 0, _scratchQuat)
      const newOx = _scratchQuat.x,
        newOy = _scratchQuat.y
      const newOz = _scratchQuat.z,
        newOw = _scratchQuat.w
      orientations[i4 + 0] = newOx
      orientations[i4 + 1] = newOy
      orientations[i4 + 2] = newOz
      orientations[i4 + 3] = newOw

      if (invDt === 0) {
        lv[i3 + 0] = 0
        lv[i3 + 1] = 0
        lv[i3 + 2] = 0
        av[i3 + 0] = 0
        av[i3 + 1] = 0
        av[i3 + 2] = 0
      } else {
        lv[i3 + 0] = (_bodyMat[12] - oldPx) * invDt
        lv[i3 + 1] = (_bodyMat[13] - oldPy) * invDt
        lv[i3 + 2] = (_bodyMat[14] - oldPz) * invDt

        // ω ≈ 2 · qDiff.xyz / dt with qDiff = qNew · conj(qOld). Shortest-
        // arc sign keeps qDiff and −qDiff (same rotation) from doubling ω.
        const cox = -oldOx,
          coy = -oldOy,
          coz = -oldOz,
          cow = oldOw
        const dx = newOw * cox + newOx * cow + newOy * coz - newOz * coy
        const dy = newOw * coy - newOx * coz + newOy * cow + newOz * cox
        const dz = newOw * coz + newOx * coy - newOy * cox + newOz * cow
        const dw = newOw * cow - newOx * cox - newOy * coy - newOz * coz
        const sign = dw < 0 ? -1 : 1
        av[i3 + 0] = 2 * sign * dx * invDt
        av[i3 + 1] = 2 * sign * dy * invDt
        av[i3 + 2] = 2 * sign * dz * invDt
      }
    }
  }

  // Dynamic bodies write their transform back to the bone matrix:
  //   boneWorld = bodyWorld × bodyOffsetInverse.
  // The body pose is the render-interpolated pose between the previous and current
  // substep states (alpha = fraction into the next step), which removes fixed-step judder.
  private applyDynamicsToBones(boneWorldMatrices: Mat4[], alpha: number): void {
    const N = this.store.count
    const inv = this.store.bodyOffsetInverse
    const positions = this.store.positions
    const orientations = this.store.orientations
    const prevPos = this.prevPositions
    const prevOri = this.prevOrientations
    const types = this.store.type
    const boneIdx = this.store.boneIndex
    const oneMinus = 1 - alpha

    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic) continue
      const b = boneIdx[i]
      if (b < 0 || b >= boneWorldMatrices.length) continue

      const i3 = i * 3
      const i4 = i * 4

      // Position: straight lerp.
      const px = prevPos[i3 + 0] * oneMinus + positions[i3 + 0] * alpha
      const py = prevPos[i3 + 1] * oneMinus + positions[i3 + 1] * alpha
      const pz = prevPos[i3 + 2] * oneMinus + positions[i3 + 2] * alpha

      // Orientation: shortest-arc nlerp (bodies rotate little per fixed step, so nlerp
      // tracks slerp closely and avoids the trig).
      const ax = prevOri[i4 + 0], ay = prevOri[i4 + 1], az = prevOri[i4 + 2], aw = prevOri[i4 + 3]
      let bx = orientations[i4 + 0], by = orientations[i4 + 1], bz = orientations[i4 + 2], bw = orientations[i4 + 3]
      if (ax * bx + ay * by + az * bz + aw * bw < 0) {
        bx = -bx; by = -by; bz = -bz; bw = -bw
      }
      let qx = ax * oneMinus + bx * alpha
      let qy = ay * oneMinus + by * alpha
      let qz = az * oneMinus + bz * alpha
      let qw = aw * oneMinus + bw * alpha
      const ql = Math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
      if (ql > 0) {
        const invL = 1 / ql
        qx *= invL; qy *= invL; qz *= invL; qw *= invL
      } else {
        qx = 0; qy = 0; qz = 0; qw = 1
      }

      Mat4.fromPositionRotationInto(px, py, pz, qx, qy, qz, qw, _bodyMat)
      Mat4.multiplyArrays(_bodyMat, 0, inv, i * 16, _boneMat, 0)

      // Sanity gate against NaN / extreme values — silently drop the update.
      if (Number.isFinite(_boneMat[0]) && Math.abs(_boneMat[0]) < 1e6) {
        boneWorldMatrices[b].values.set(_boneMat)
      }
    }
  }
}
