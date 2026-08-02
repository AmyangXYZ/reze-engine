// Game-style locomotion over the blend primitive: idle ↔ run ↔ sprint mixed by a
// smoothed speed level, yaw eased toward the input heading, root motion integrated
// in code (the clips are in-place). Zero renderer coupling — the controller only
// talks to Model.setBlendPose and returns the root transform for the host to apply
// via engine.setModelTransform.

import { Model } from "./model"
import { FPS, type BlendEntry } from "./animation"
import { Quat, Vec3 } from "./math"

export interface LocomotionClips {
  /** Clip names previously loaded on the model (loadVmd/loadClip). */
  idle: string
  run: string
  sprint?: string
}

export interface LocomotionOptions {
  /** Ground speed in MMD units/second at full run (default 67 — measured from the Unity
   *  locomotion pack's root motion before the in-place strip; ≈5.4 m/s at MMD scale).
   *  Match this to the clips or the feet slide. */
  runSpeed?: number
  /** Ground speed at full sprint (default 92, measured the same way; ≈7.4 m/s). */
  sprintSpeed?: number
  /** Response rate of the idle↔run↔sprint blend, in speed-levels/second (default 5). */
  speedResponse?: number
  /** Yaw easing rate toward the input heading, 1/second (default 10). */
  turnResponse?: number
  /** Heading error (radians) beyond which the character pivots strictly in place —
   *  no translation until the body is back within this cone (default PI/4). */
  turnInPlaceThreshold?: number
  /** Tank-mode steering rate, radians/second (default 2.5 ≈ 143°/s). */
  steerRate?: number
  /** Backpedal speed as a fraction of run speed in tank mode (default 0.5). */
  backpedalScale?: number
  /** World yaw the model faces at rotation 0. MMD models rest facing -Z, so facing a
   *  heading of `yaw` needs rotationY = yaw + PI — the default. */
  yawOffset?: number
}

/** The integrated root transform for this frame. `position` and `rotation` are
 *  REUSED instances owned by the controller — apply them immediately (they feed
 *  straight into engine.setModelTransform), don't store them. */
export interface LocomotionPose {
  position: Vec3
  /** Heading in radians: 0 = +Z, increasing toward +X. */
  yaw: number
  /** rotationY quat including yawOffset, ready for setModelTransform. */
  rotation: Quat
  /** Smoothed speed level: 0 idle, 1 run, 2 sprint. */
  speedLevel: number
}

const TWO_PI = Math.PI * 2

function wrapAngle(a: number): number {
  while (a > Math.PI) a -= TWO_PI
  while (a < -Math.PI) a += TWO_PI
  return a
}

export class LocomotionController {
  private readonly model: Model
  private readonly clips: LocomotionClips
  private readonly runSpeed: number
  private readonly sprintSpeed: number
  private readonly speedResponse: number
  private readonly turnResponse: number
  private readonly cosTurnThreshold: number
  private readonly yawOffset: number

  private inputX = 0
  private inputY = 0
  private inputSprint = false
  // Tank mode: forward/steer relative to the CURRENT facing, camera never involved.
  private tankMode = false
  private inputForward = 0
  private inputSteer = 0
  private readonly steerRate: number
  private readonly backpedalScale: number

  private speedLevel = 0
  private yaw = 0
  // Movement direction = the INPUT heading, not the body yaw. The body turns
  // cosmetically toward it; translating along the (sweeping) body yaw instead
  // would nudge the character through the forward arc on every L↔R reversal.
  private dirX = 0
  private dirZ = 1
  private readonly position = new Vec3(0, 0, 0)
  private readonly rotation = new Quat(0, 0, 0, 1)
  private idleTime = 0
  private gaitPhase = 0 // 0..1, shared by run and sprint so legs stay aligned across the blend

  private readonly entries: BlendEntry[]
  private readonly pose: LocomotionPose

  constructor(model: Model, clips: LocomotionClips, options?: LocomotionOptions) {
    this.model = model
    this.clips = clips
    this.runSpeed = options?.runSpeed ?? 67
    this.sprintSpeed = options?.sprintSpeed ?? 92
    this.speedResponse = options?.speedResponse ?? 5
    this.turnResponse = options?.turnResponse ?? 10
    this.cosTurnThreshold = Math.cos(options?.turnInPlaceThreshold ?? Math.PI / 4)
    this.steerRate = options?.steerRate ?? 2.5
    this.backpedalScale = options?.backpedalScale ?? 0.5
    this.yawOffset = options?.yawOffset ?? Math.PI
    this.entries = [
      { name: clips.idle, time: 0, weight: 1 },
      { name: clips.run, time: 0, weight: 0 },
      { name: clips.sprint ?? clips.run, time: 0, weight: 0 },
    ]
    this.pose = { position: this.position, yaw: 0, rotation: this.rotation, speedLevel: 0 }
  }

  /** World-vector move input: x = +right (+X), y = +forward (+Z), magnitude clamped
   *  to 1. The character turns toward the vector, then runs along it. Call whenever
   *  input changes — the value holds between calls. */
  setMove(x: number, y: number, sprint = false): void {
    const m = Math.hypot(x, y)
    if (m > 1) {
      x /= m
      y /= m
    }
    this.tankMode = false
    this.inputX = x
    this.inputY = y
    this.inputSprint = sprint
  }

  /** Tank-style input relative to the CURRENT facing, camera-independent and fully
   *  deterministic: `steer` (−1..1, + = her right) rotates at steerRate whether
   *  standing or moving, `forward` (+1 run ahead — curving while steering — or −1
   *  backpedal at backpedalScale). Holds between calls, like setMove. */
  setDrive(forward: number, steer: number, sprint = false): void {
    this.tankMode = true
    this.inputForward = Math.max(-1, Math.min(1, forward))
    this.inputSteer = Math.max(-1, Math.min(1, steer))
    this.inputSprint = sprint
  }

  /** Place the character (initial spawn or respawn). */
  teleport(x: number, y: number, z: number, yaw = 0): void {
    this.position.setXYZ(x, y, z)
    this.yaw = yaw
  }

  getPosition(): Vec3 {
    return this.position
  }

  /** Stop driving the model's pose (the blend is cleared; the single-clip player resumes). */
  detach(): void {
    this.model.clearBlendPose()
  }

  private clipDuration(name: string): number {
    const frames = this.model.getClip(name)?.frameCount ?? 0
    return frames > 0 ? frames / FPS : 1
  }

  /** Advance one frame: integrates yaw + position, updates the clip clocks, hands the
   *  weighted pose to the model, and returns the root transform to apply. */
  update(dt: number): LocomotionPose {
    if (dt > 0.1) dt = 0.1 // tab-switch guard: never integrate a huge step

    const hasSprint = this.clips.sprint !== undefined
    let moving: boolean
    let align = 1
    let speedScale = 1

    if (this.tankMode) {
      // Steering rotates the facing directly — standing or moving — and the travel
      // direction IS the facing, so no pivot gate and no drift by construction.
      this.yaw = wrapAngle(this.yaw + this.inputSteer * this.steerRate * dt)
      const fwd = this.inputForward
      moving = Math.abs(fwd) > 0.05
      if (moving) {
        const back = fwd < 0
        this.dirX = Math.sin(this.yaw) * Math.sign(fwd)
        this.dirZ = Math.cos(this.yaw) * Math.sign(fwd)
        speedScale = Math.abs(fwd) * (back ? this.backpedalScale : 1)
      }
    } else {
      const m = Math.hypot(this.inputX, this.inputY)
      moving = m > 0.05
      // Yaw eases toward the input heading only while there is one. Turns are strictly
      // in place: zero translation outside the threshold cone, ramping smoothly to full
      // speed as the body aligns — so direction reversals (L-R-L) cannot drift.
      if (moving) {
        const desired = Math.atan2(this.inputX, this.inputY)
        const err = wrapAngle(desired - this.yaw)
        this.yaw = wrapAngle(this.yaw + err * Math.min(1, this.turnResponse * dt))
        align = Math.max(0, (Math.cos(err) - this.cosTurnThreshold) / (1 - this.cosTurnThreshold))
        this.dirX = this.inputX / m
        this.dirZ = this.inputY / m
      }
    }

    // Speed level ramps linearly toward the target; the pose blend follows it.
    // (No sprint while backpedaling.)
    const sprinting = this.inputSprint && hasSprint && !(this.tankMode && this.inputForward < 0)
    const target = moving ? (sprinting ? 2 : 1) : 0
    const maxStep = this.speedResponse * dt
    const d = target - this.speedLevel
    this.speedLevel += Math.abs(d) <= maxStep ? d : Math.sign(d) * maxStep

    // Root motion along the travel direction. In-place clips carry no horizontal root.
    const speed =
      (this.speedLevel <= 1
        ? this.runSpeed * this.speedLevel
        : this.runSpeed + (this.sprintSpeed - this.runSpeed) * (this.speedLevel - 1)) *
      align *
      speedScale
    this.position.x += this.dirX * speed * dt
    this.position.z += this.dirZ * speed * dt

    // Clocks: idle free-runs; run/sprint share one normalized gait phase so a
    // mid-blend stride stays on the same feet.
    const idleDur = this.clipDuration(this.clips.idle)
    const runDur = this.clipDuration(this.clips.run)
    const sprintDur = hasSprint ? this.clipDuration(this.clips.sprint!) : runDur
    this.idleTime = (this.idleTime + dt) % idleDur
    const gaitDur = this.speedLevel <= 1 ? runDur : runDur + (sprintDur - runDur) * (this.speedLevel - 1)
    this.gaitPhase = (this.gaitPhase + dt / gaitDur) % 1

    // Weights along the 1D speed axis.
    let wIdle: number, wRun: number, wSprint: number
    if (this.speedLevel <= 1) {
      wIdle = 1 - this.speedLevel
      wRun = this.speedLevel
      wSprint = 0
    } else {
      wIdle = 0
      wRun = 2 - this.speedLevel
      wSprint = this.speedLevel - 1
    }

    this.entries[0].time = this.idleTime
    this.entries[0].weight = wIdle
    this.entries[1].time = this.gaitPhase * runDur
    this.entries[1].weight = wRun
    this.entries[2].time = this.gaitPhase * sprintDur
    this.entries[2].weight = wSprint
    this.model.setBlendPose(this.entries)

    const ry = this.yaw + this.yawOffset
    const half = ry * 0.5
    this.rotation.setXYZW(0, Math.sin(half), 0, Math.cos(half))
    this.pose.yaw = this.yaw
    this.pose.speedLevel = this.speedLevel
    return this.pose
  }
}
