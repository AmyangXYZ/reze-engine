import { Vec3, Quat, Mat4 } from "../math"
import type { Rigidbody, Joint } from "./types"
import { RigidbodyType, RigidbodyShape } from "./types"
import { RigidBodyStore } from "./body"
import { World } from "./world"
import { buildConstraints, type SixDofSpringConstraint } from "./constraint"
import { SolverCache } from "./solver"
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
  private solverCache: SolverCache
  private contacts: ContactPool
  private firstFrame = true
  private timeAccum = 0
  private readonly fixedTimeStep = 1 / 60
  private readonly maxSubSteps = 6
  /** EMA of one world.step's CPU cost; drives catch-up load shedding. */
  private stepCostEmaMs = 0.5
  /** Max ms a single frame may spend catching up physics before shedding. */
  private readonly stepBudgetMs = 8

  // Fixed-timestep render interpolation ("Fix Your Timestep"): the dynamic body pose is
  // rendered as lerp(prev, curr, alpha) between the last two completed substeps, where
  // alpha = leftover accumulator / fixedTimeStep. Removes the temporal aliasing that shows
  // as hair/cloth judder when the render rate doesn't line up with the 60Hz physics step.
  private prevPositions: Float32Array
  private prevOrientations: Float32Array

  // Per-frame kinematic body targets (boneWorld × bodyOffset). Kinematic
  // bodies advance toward these incrementally per substep instead of jumping
  // to the final pose before substep 1. Velocities come from the target
  // trajectory (frame-to-frame target delta over render dt) — deriving them
  // from the per-substep advancement instead ripples with the accumulator
  // phase at render rates above 60 Hz and visibly excites cloth chains.
  private kinTargetPos: Float32Array
  private kinTargetOri: Float32Array
  private kinTargetVel: Float32Array
  private kinTargetAngVel: Float32Array

  // For each dynamic body, the kinematic body its constraint chain roots at
  // (-1 if unreachable). On a teleport, dynamic bodies are carried rigidly by
  // their root's transform delta so cloth keeps its pose relative to the
  // character instead of being dragged across the jump.
  private kinRoot: Int32Array
  // Kinematic bodies whose target jumped discontinuously this frame.
  private teleportFlags: Uint8Array
  // PMX mode-2 bodies jointed *directly* to a bone-follow body (breasts,
  // chain roots). Only these get MMD's bone-position alignment: their bone's
  // animated matrix is trustworthy, whereas deeper chain bones' animated
  // matrices don't include the physics deflection of their parents, so
  // pinning those would freeze the chain.
  private alignPinned: Uint8Array
  // Debug counter: frames on which a teleport (scrub/jump discontinuity)
  // was detected and settled.
  teleportCount = 0

  constructor(rigidbodies: Rigidbody[], joints: Joint[] = []) {
    this.rigidbodies = rigidbodies
    this.joints = joints
    // The floor is part of every world: a huge static box whose top face is
    // model-space y = 0, so hair and hems rest on the ground instead of
    // clipping through when a pose reaches it. Boneless (every bone-sync loop
    // skips boneIndex < 0); collides with EVERY dynamic body regardless of
    // group masks via findContacts' dedicated plane pass (spheres, capsules
    // and boxes — box hems included).
    const ground: Rigidbody = {
      name: "__ground__",
      englishName: "__ground__",
      boneIndex: -1,
      group: 0,
      collisionMask: 0xffff,
      shape: RigidbodyShape.Box,
      size: new Vec3(500, 1, 500),
      shapePosition: new Vec3(0, -1, 0),
      shapeRotation: new Vec3(0, 0, 0),
      mass: 0,
      linearDamping: 0,
      angularDamping: 0,
      restitution: 0,
      friction: 0.6,
      type: RigidbodyType.Static,
      bodyOffsetMatrixInverse: Mat4.identity(),
    }
    this.store = new RigidBodyStore([...rigidbodies, ground])
    const gi = this.store.count - 1
    // Mask 0 keeps the floor OUT of the generic pair list — findContacts runs a
    // dedicated plane pass against every dynamic body (all shapes, boxes too).
    this.store.collisionGroup[gi] = 0
    this.store.willCollideMask[gi] = 0
    this.store.groundIndex = gi
    this.world = new World(new Vec3(0, -98, 0))
    this.constraints = buildConstraints(rigidbodies, joints)
    this.solverCache = new SolverCache(this.constraints)
    this.contacts = new ContactPool()
    this.prevPositions = new Float32Array(this.store.count * 3)
    this.prevOrientations = new Float32Array(this.store.count * 4)
    this.kinTargetPos = new Float32Array(this.store.count * 3)
    this.kinTargetOri = new Float32Array(this.store.count * 4)
    this.kinTargetVel = new Float32Array(this.store.count * 3)
    this.kinTargetAngVel = new Float32Array(this.store.count * 3)
    this.kinRoot = this.buildKinematicRoots()
    this.teleportFlags = new Uint8Array(this.store.count)

    this.alignPinned = new Uint8Array(this.store.count)
    const types = this.store.type
    for (const c of this.constraints) {
      const tA = types[c.bodyA]
      const tB = types[c.bodyB]
      const aFollows = tA === RigidbodyType.Static || tA === RigidbodyType.Kinematic
      const bFollows = tB === RigidbodyType.Static || tB === RigidbodyType.Kinematic
      if (aFollows && this.store.aligned[c.bodyB]) this.alignPinned[c.bodyB] = 1
      if (bFollows && this.store.aligned[c.bodyA]) this.alignPinned[c.bodyA] = 1
    }
  }

  // BFS over the joint graph from every kinematic body, assigning each
  // reachable dynamic body the kinematic body it (transitively) hangs off.
  private buildKinematicRoots(): Int32Array {
    const N = this.store.count
    const root = new Int32Array(N).fill(-1)
    const types = this.store.type
    const adj: number[][] = Array.from({ length: N }, () => [])
    for (const c of this.constraints) {
      adj[c.bodyA].push(c.bodyB)
      adj[c.bodyB].push(c.bodyA)
    }
    const queue: number[] = []
    for (let i = 0; i < N; i++) {
      if (types[i] === RigidbodyType.Static || types[i] === RigidbodyType.Kinematic) {
        root[i] = i
        queue.push(i)
      }
    }
    for (let h = 0; h < queue.length; h++) {
      const i = queue[h]
      for (const j of adj[i]) {
        if (root[j] !== -1) continue
        root[j] = root[i]
        queue.push(j)
      }
    }
    return root
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
    // Reseed kinematic targets from the snapped pose — otherwise the next
    // step derives the target-trajectory velocity against the pre-reset
    // targets and feeds one frame of enormous anchor velocity to the solver.
    this.kinTargetPos.set(this.store.positions)
    this.kinTargetOri.set(this.store.orientations)
    this.kinTargetVel.fill(0)
    this.kinTargetAngVel.fill(0)
    this.timeAccum = 0
  }

  step(dt: number, boneWorldMatrices: Mat4[], boneInverseBindMatrices: Float32Array): void {
    if (this.firstFrame) {
      this.store.computeBoneOffsets(boneInverseBindMatrices)
      // Start at current bone pose, not the PMX bind pose, so animations
      // that skip frame 0 don't pop bodies on first step.
      this.snapBodiesToBones(boneWorldMatrices)
      this.savePrevState()
      // Seed kinematic targets so the first frame's target-delta velocity
      // is zero instead of a jump from the origin.
      this.kinTargetPos.set(this.store.positions)
      this.kinTargetOri.set(this.store.orientations)
      this.firstFrame = false
    }

    // Compute this frame's kinematic targets from the current bone pose. A
    // target jump beyond what continuous motion can produce (timeline scrub)
    // is a teleport, handled per kinematic root: rigidly carry that root's
    // dynamic chains across the jump with zeroed momentum, snap the root,
    // and keep simulating — dragging cloth through a discontinuity at the
    // raw derived velocity is what used to explode the solver. Chains under
    // unaffected roots keep their momentum untouched.
    if (this.computeKinematicTargets(boneWorldMatrices, dt)) {
      this.teleportCount++
      this.carryDynamicThroughTeleport()
      this.snapKinematicToTargets(true)
      this.savePrevState() // prev == curr so interpolation doesn't streak
    }

    // Fixed-timestep substeps. The maxSubSteps cap prevents runaway after
    // a long stall (tab backgrounded, etc.). Snapshot the pose before each step so
    // after the loop prevState is one substep behind the live (current) state.
    // Kinematic bodies split the remaining gap evenly across this frame's
    // substeps (f = 1/substeps-left) and land exactly on the frame's bone
    // pose by the last one: at 60 Hz render that reproduces the classic
    // sync-once-per-frame behavior exactly (no fractional lag trembling
    // against the rendered mesh), while at lower rates the per-substep
    // constraint error stays bounded to one fixed step of bone motion.
    this.timeAccum += dt
    let nSub = Math.floor(this.timeAccum / this.fixedTimeStep)
    if (nSub > this.maxSubSteps) nSub = this.maxSubSteps
    // Load shedding: the maxSubSteps cap alone still allows a stable DEGRADED state
    // on weak/throttled devices — one long frame schedules catch-up substeps, which
    // lengthen the next frame, which schedules more, and the cap becomes the steady
    // state (6× step cost every frame; iOS "starts smooth, then stays slow").
    // Catching up must never cost more than a frame-budget slice: shed the backlog
    // instead (physics runs fractionally slow-motion for a beat; framerate holds).
    const affordable = Math.max(1, Math.floor(this.stepBudgetMs / Math.max(0.05, this.stepCostEmaMs)))
    if (nSub > affordable) nSub = affordable
    for (let k = 0; k < nSub; k++) {
      this.savePrevState()
      this.advanceKinematicToTargets(1 / (nSub - k))
      const tStep = performance.now()
      this.world.step(this.store, this.constraints, this.solverCache, this.contacts, this.fixedTimeStep)
      this.stepCostEmaMs += (performance.now() - tStep - this.stepCostEmaMs) * 0.2
      this.restoreNonFiniteBodies()
      this.timeAccum -= this.fixedTimeStep
    }
    if (this.timeAccum >= this.fixedTimeStep) {
      // Substep budget exhausted mid-catchup: drop the remaining time and
      // snap kinematic bodies the rest of the way (zero velocity) so they
      // don't start next frame lagging behind their bones.
      this.timeAccum = 0
      this.snapKinematicToTargets(false)
    }

    // MMD mode-2 bone alignment for pinned (depth-1) bodies: position
    // re-pins to the animated bone, rotation stays simulated.
    this.alignPinnedBodiesToBones(boneWorldMatrices)

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

  // Fill kinTargetPos/kinTargetOri = boneWorld × bodyOffset for every bone-
  // bound Static/Kinematic body. Returns true if any target is discontinuous
  // with the current body pose — farther than continuous motion can carry it
  // in one render frame, or rotated more than 90°.
  private computeKinematicTargets(boneWorldMatrices: Mat4[], dt: number): boolean {
    const N = this.store.count
    const offsets = this.store.bodyOffsetMatrix
    const positions = this.store.positions
    const orientations = this.store.orientations
    const types = this.store.type
    const boneIdx = this.store.boneIndex
    const tp = this.kinTargetPos
    const to = this.kinTargetOri

    // 250 units/s scaled by frame time, floored at 4 units: far above the
    // fastest limb motion (a false positive freezes that chain's momentum
    // for a frame, which reads as stutter), far below any real scrub jump.
    const maxJump = Math.max(4, 250 * dt)
    const maxJumpSq = maxJump * maxJump
    const flags = this.teleportFlags
    let teleport = false

    const tv = this.kinTargetVel
    const tav = this.kinTargetAngVel
    const invDt = dt > 0 ? 1 / dt : 0

    for (let i = 0; i < N; i++) {
      flags[i] = 0
      const t = types[i]
      if (t !== RigidbodyType.Static && t !== RigidbodyType.Kinematic) continue
      const b = boneIdx[i]
      if (b < 0 || b >= boneWorldMatrices.length) continue

      Mat4.multiplyArrays(boneWorldMatrices[b].values, 0, offsets, i * 16, _bodyMat, 0)

      const i3 = i * 3
      const i4 = i * 4
      // Previous frame's target — the reference for the trajectory velocity.
      const oldTx = tp[i3 + 0], oldTy = tp[i3 + 1], oldTz = tp[i3 + 2]
      const oldOx = to[i4 + 0], oldOy = to[i4 + 1], oldOz = to[i4 + 2], oldOw = to[i4 + 3]

      tp[i3 + 0] = _bodyMat[12]
      tp[i3 + 1] = _bodyMat[13]
      tp[i3 + 2] = _bodyMat[14]
      Mat4.toQuatFromArrayInto(_bodyMat, 0, _scratchQuat)
      const nOx = _scratchQuat.x, nOy = _scratchQuat.y, nOz = _scratchQuat.z, nOw = _scratchQuat.w
      to[i4 + 0] = nOx
      to[i4 + 1] = nOy
      to[i4 + 2] = nOz
      to[i4 + 3] = nOw

      const dx = tp[i3 + 0] - positions[i3 + 0]
      const dy = tp[i3 + 1] - positions[i3 + 1]
      const dz = tp[i3 + 2] - positions[i3 + 2]
      // |q·q'| = cos(θ/2); below cos(45°) the body turned more than 90°.
      const dot =
        to[i4 + 0] * orientations[i4 + 0] +
        to[i4 + 1] * orientations[i4 + 1] +
        to[i4 + 2] * orientations[i4 + 2] +
        to[i4 + 3] * orientations[i4 + 3]
      if (dx * dx + dy * dy + dz * dz > maxJumpSq || Math.abs(dot) < 0.7071) {
        flags[i] = 1
        teleport = true
        tv[i3 + 0] = 0; tv[i3 + 1] = 0; tv[i3 + 2] = 0
        tav[i3 + 0] = 0; tav[i3 + 1] = 0; tav[i3 + 2] = 0
        continue
      }

      tv[i3 + 0] = (tp[i3 + 0] - oldTx) * invDt
      tv[i3 + 1] = (tp[i3 + 1] - oldTy) * invDt
      tv[i3 + 2] = (tp[i3 + 2] - oldTz) * invDt
      // ω ≈ 2 · qDiff.xyz / dt with qDiff = qNew · conj(qOld). Shortest-arc
      // sign keeps qDiff and −qDiff (same rotation) from doubling ω.
      const cox = -oldOx, coy = -oldOy, coz = -oldOz, cow = oldOw
      const qdx = nOw * cox + nOx * cow + nOy * coz - nOz * coy
      const qdy = nOw * coy - nOx * coz + nOy * cow + nOz * cox
      const qdz = nOw * coz + nOx * coy - nOy * cox + nOz * cow
      const qdw = nOw * cow - nOx * cox - nOy * coy - nOz * coz
      const sign = qdw < 0 ? -1 : 1
      tav[i3 + 0] = 2 * sign * qdx * invDt
      tav[i3 + 1] = 2 * sign * qdy * invDt
      tav[i3 + 2] = 2 * sign * qdz * invDt
    }
    return teleport
  }

  // Move kinematic bodies fraction f of the way to the frame target with the
  // target-trajectory velocity, so joints see continuous anchor motion (and a
  // smooth velocity signal) instead of a frame-sized jump on substep 1.
  private advanceKinematicToTargets(f: number): void {
    const N = this.store.count
    const positions = this.store.positions
    const orientations = this.store.orientations
    const lv = this.store.linearVelocities
    const av = this.store.angularVelocities
    const types = this.store.type
    const boneIdx = this.store.boneIndex
    const tp = this.kinTargetPos
    const to = this.kinTargetOri
    const tv = this.kinTargetVel
    const tav = this.kinTargetAngVel

    for (let i = 0; i < N; i++) {
      const t = types[i]
      if (t !== RigidbodyType.Static && t !== RigidbodyType.Kinematic) continue
      const b = boneIdx[i]
      if (b < 0) continue

      const i3 = i * 3
      const i4 = i * 4
      positions[i3 + 0] += (tp[i3 + 0] - positions[i3 + 0]) * f
      positions[i3 + 1] += (tp[i3 + 1] - positions[i3 + 1]) * f
      positions[i3 + 2] += (tp[i3 + 2] - positions[i3 + 2]) * f
      lv[i3 + 0] = tv[i3 + 0]
      lv[i3 + 1] = tv[i3 + 1]
      lv[i3 + 2] = tv[i3 + 2]
      av[i3 + 0] = tav[i3 + 0]
      av[i3 + 1] = tav[i3 + 1]
      av[i3 + 2] = tav[i3 + 2]

      // Shortest-arc nlerp toward the target orientation.
      const oldOx = orientations[i4 + 0],
        oldOy = orientations[i4 + 1]
      const oldOz = orientations[i4 + 2],
        oldOw = orientations[i4 + 3]
      let tx = to[i4 + 0],
        ty = to[i4 + 1],
        tz = to[i4 + 2],
        tw = to[i4 + 3]
      if (oldOx * tx + oldOy * ty + oldOz * tz + oldOw * tw < 0) {
        tx = -tx; ty = -ty; tz = -tz; tw = -tw
      }
      let newOx = oldOx + (tx - oldOx) * f
      let newOy = oldOy + (ty - oldOy) * f
      let newOz = oldOz + (tz - oldOz) * f
      let newOw = oldOw + (tw - oldOw) * f
      const len2 = newOx * newOx + newOy * newOy + newOz * newOz + newOw * newOw
      if (len2 > 1e-12) {
        const inv = 1 / Math.sqrt(len2)
        newOx *= inv; newOy *= inv; newOz *= inv; newOw *= inv
      } else {
        newOx = tx; newOy = ty; newOz = tz; newOw = tw
      }
      orientations[i4 + 0] = newOx
      orientations[i4 + 1] = newOy
      orientations[i4 + 2] = newOz
      orientations[i4 + 3] = newOw
    }
  }

  // Snap kinematic bodies straight to the frame target with zero velocity.
  // Used on teleports (onlyFlagged) and when the substep budget runs out
  // mid-catchup (all).
  private snapKinematicToTargets(onlyFlagged: boolean): void {
    const N = this.store.count
    const positions = this.store.positions
    const orientations = this.store.orientations
    const lv = this.store.linearVelocities
    const av = this.store.angularVelocities
    const types = this.store.type
    const boneIdx = this.store.boneIndex
    const tp = this.kinTargetPos
    const to = this.kinTargetOri
    const flags = this.teleportFlags

    for (let i = 0; i < N; i++) {
      const t = types[i]
      if (t !== RigidbodyType.Static && t !== RigidbodyType.Kinematic) continue
      if (boneIdx[i] < 0) continue
      if (onlyFlagged && !flags[i]) continue
      const i3 = i * 3
      const i4 = i * 4
      positions[i3 + 0] = tp[i3 + 0]
      positions[i3 + 1] = tp[i3 + 1]
      positions[i3 + 2] = tp[i3 + 2]
      orientations[i4 + 0] = to[i4 + 0]
      orientations[i4 + 1] = to[i4 + 1]
      orientations[i4 + 2] = to[i4 + 2]
      orientations[i4 + 3] = to[i4 + 3]
      lv[i3 + 0] = 0; lv[i3 + 1] = 0; lv[i3 + 2] = 0
      av[i3 + 0] = 0; av[i3 + 1] = 0; av[i3 + 2] = 0
    }
  }

  // Rigidly carry each dynamic body whose kinematic root teleported along
  // with that root's current→target transform delta (velocity zeroed),
  // preserving the cloth pose relative to the character across the jump.
  // Must run before snapKinematicToTargets (it reads the pre-snap pose).
  private carryDynamicThroughTeleport(): void {
    const N = this.store.count
    const positions = this.store.positions
    const orientations = this.store.orientations
    const lv = this.store.linearVelocities
    const av = this.store.angularVelocities
    const types = this.store.type
    const tp = this.kinTargetPos
    const to = this.kinTargetOri
    const root = this.kinRoot
    const flags = this.teleportFlags

    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic) continue
      const k = root[i]
      if (k < 0 || !flags[k] || this.store.boneIndex[k] < 0) continue
      const k3 = k * 3
      const k4 = k * 4
      // Root delta rotation R = qTarget · conj(qCurrent).
      const cx = -orientations[k4 + 0],
        cy = -orientations[k4 + 1],
        cz = -orientations[k4 + 2],
        cw = orientations[k4 + 3]
      const txq = to[k4 + 0],
        tyq = to[k4 + 1],
        tzq = to[k4 + 2],
        twq = to[k4 + 3]
      const rx = twq * cx + txq * cw + tyq * cz - tzq * cy
      const ry = twq * cy - txq * cz + tyq * cw + tzq * cx
      const rz = twq * cz + txq * cy - tyq * cx + tzq * cw
      const rw = twq * cw - txq * cx - tyq * cy - tzq * cz

      const i3 = i * 3
      const i4 = i * 4
      // Position: rotate the offset from the root by R, re-anchor at target.
      const ox = positions[i3 + 0] - positions[k3 + 0]
      const oy = positions[i3 + 1] - positions[k3 + 1]
      const oz = positions[i3 + 2] - positions[k3 + 2]
      // v' = v + 2·rw·(r × v) + 2·(r × (r × v))
      const c1x = ry * oz - rz * oy
      const c1y = rz * ox - rx * oz
      const c1z = rx * oy - ry * ox
      const c2x = ry * c1z - rz * c1y
      const c2y = rz * c1x - rx * c1z
      const c2z = rx * c1y - ry * c1x
      positions[i3 + 0] = tp[k3 + 0] + ox + 2 * (rw * c1x + c2x)
      positions[i3 + 1] = tp[k3 + 1] + oy + 2 * (rw * c1y + c2y)
      positions[i3 + 2] = tp[k3 + 2] + oz + 2 * (rw * c1z + c2z)

      // Orientation: q' = R · q, renormalized.
      const qx = orientations[i4 + 0],
        qy = orientations[i4 + 1],
        qz = orientations[i4 + 2],
        qw = orientations[i4 + 3]
      let nx = rw * qx + rx * qw + ry * qz - rz * qy
      let ny = rw * qy - rx * qz + ry * qw + rz * qx
      let nz = rw * qz + rx * qy - ry * qx + rz * qw
      let nw = rw * qw - rx * qx - ry * qy - rz * qz
      const len2 = nx * nx + ny * ny + nz * nz + nw * nw
      if (len2 > 1e-12) {
        const inv = 1 / Math.sqrt(len2)
        nx *= inv; ny *= inv; nz *= inv; nw *= inv
        orientations[i4 + 0] = nx
        orientations[i4 + 1] = ny
        orientations[i4 + 2] = nz
        orientations[i4 + 3] = nw
      }

      // Momentum doesn't carry across a discontinuity.
      lv[i3 + 0] = 0; lv[i3 + 1] = 0; lv[i3 + 2] = 0
      av[i3 + 0] = 0; av[i3 + 1] = 0; av[i3 + 2] = 0
    }
  }

  // Overwrite pinned mode-2 bodies' positions with boneWorld × bodyOffset
  // from the animated bone pose, keeping simulated orientation and
  // velocities. prevPositions follows so interpolation doesn't streak.
  private alignPinnedBodiesToBones(boneWorldMatrices: Mat4[]): void {
    const N = this.store.count
    const pinned = this.alignPinned
    const boneIdx = this.store.boneIndex
    const offsets = this.store.bodyOffsetMatrix
    const positions = this.store.positions
    const prevPos = this.prevPositions

    for (let i = 0; i < N; i++) {
      if (!pinned[i]) continue
      const b = boneIdx[i]
      if (b < 0 || b >= boneWorldMatrices.length) continue
      Mat4.multiplyArrays(boneWorldMatrices[b].values, 0, offsets, i * 16, _bodyMat, 0)
      const i3 = i * 3
      positions[i3 + 0] = _bodyMat[12]
      positions[i3 + 1] = _bodyMat[13]
      positions[i3 + 2] = _bodyMat[14]
      prevPos[i3 + 0] = _bodyMat[12]
      prevPos[i3 + 1] = _bodyMat[13]
      prevPos[i3 + 2] = _bodyMat[14]
    }
  }

  // Backstop: if a dynamic body's state went non-finite despite the velocity
  // caps, restore its previous-substep pose with zero velocity instead of
  // letting NaNs spread through constraints and contacts. Runs after every
  // substep so prevState is always a finite pose.
  private restoreNonFiniteBodies(): void {
    const N = this.store.count
    const positions = this.store.positions
    const orientations = this.store.orientations
    const lv = this.store.linearVelocities
    const av = this.store.angularVelocities
    const types = this.store.type
    const prevPos = this.prevPositions
    const prevOri = this.prevOrientations

    for (let i = 0; i < N; i++) {
      if (types[i] !== RigidbodyType.Dynamic) continue
      const i3 = i * 3
      const i4 = i * 4
      // NaN/Inf propagates through the sum, so one check covers all 13 slots.
      const s =
        positions[i3 + 0] + positions[i3 + 1] + positions[i3 + 2] +
        orientations[i4 + 0] + orientations[i4 + 1] + orientations[i4 + 2] + orientations[i4 + 3] +
        lv[i3 + 0] + lv[i3 + 1] + lv[i3 + 2] +
        av[i3 + 0] + av[i3 + 1] + av[i3 + 2]
      if (Number.isFinite(s)) continue
      positions[i3 + 0] = prevPos[i3 + 0]
      positions[i3 + 1] = prevPos[i3 + 1]
      positions[i3 + 2] = prevPos[i3 + 2]
      orientations[i4 + 0] = prevOri[i4 + 0]
      orientations[i4 + 1] = prevOri[i4 + 1]
      orientations[i4 + 2] = prevOri[i4 + 2]
      orientations[i4 + 3] = prevOri[i4 + 3]
      lv[i3 + 0] = 0; lv[i3 + 1] = 0; lv[i3 + 2] = 0
      av[i3 + 0] = 0; av[i3 + 1] = 0; av[i3 + 2] = 0
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
        const v = boneWorldMatrices[b].values
        if (this.alignPinned[i]) {
          // Pinned mode-2: the bone keeps its animated position, physics
          // drives rotation only.
          v[0] = _boneMat[0]; v[1] = _boneMat[1]; v[2] = _boneMat[2]
          v[4] = _boneMat[4]; v[5] = _boneMat[5]; v[6] = _boneMat[6]
          v[8] = _boneMat[8]; v[9] = _boneMat[9]; v[10] = _boneMat[10]
        } else {
          v.set(_boneMat)
        }
      }
    }
  }
}
