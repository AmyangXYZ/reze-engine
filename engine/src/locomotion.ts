// Game-style locomotion over the blend primitive: idle ↔ run ↔ sprint mixed by a
// smoothed speed level, yaw eased toward the input heading, root motion integrated
// in code (the clips are in-place). Zero renderer coupling — the controller only
// talks to Model.setBlendPose and returns the root transform for the host to apply
// via engine.setModelTransform.

import { Model } from "./model"
import { FPS, type BlendEntry } from "./animation"
import { Quat, Vec3 } from "./math"

export interface StrafeClipEntry {
  /** Clip name previously loaded on the model. */
  clip: string
  /** Movement direction relative to the facing, radians: 0 = forward, + = the character's right. */
  angle: number
  /** The clip's authored root speed in MMD units/s (post-conversion scale) — drives root motion. */
  speed: number
}

export interface TurnClipEntry {
  /** Clip name previously loaded on the model. */
  clip: string
  /** Signed yaw the clip turns through, radians (+ = the character's right). */
  angle: number
  /** Clip-local seconds at which the yaw is complete (the settle tail is skipped). */
  exitTime: number
}

export interface RunTurnClipEntry {
  /** Clip name previously loaded on the model. */
  clip: string
  /** Signed total yaw, radians (+ = the character's right; reversals are ±PI). */
  angle: number
  /** Clip-local seconds at which the yaw is complete (settle tail skipped). */
  exitTime: number
  /** Uniform samples over [0, exitTime] of the clip's authored forward displacement
   *  (MMD units along the heading at trigger) — the overrun-plant-return curve. */
  forward: number[]
  gear: "run" | "sprint"
  foot: "L" | "R"
}

export interface StopClipEntry {
  /** Clip name previously loaded on the model. */
  clip: string
  /** Clip-local seconds at which the deceleration settles (idle tail skipped). */
  exitTime: number
  /** Uniform samples over [0, exitTime] of the authored forward displacement
   *  (MMD units along the heading at release) — the skid-to-plant curve. */
  forward: number[]
  gear: "run" | "sprint"
  foot: "L" | "R"
}

export interface LocomotionClips {
  /** Clip names previously loaded on the model (loadVmd/loadClip). */
  idle: string
  run: string
  sprint?: string
  /** Directional ring for strafe mode (setFacing): body holds a facing while movement
   *  blends the two ring clips nearest the local move angle. */
  strafeRun?: StrafeClipEntry[]
  strafeSprint?: StrafeClipEntry[]
  /** Authored turn-in-place clips: reversal-class direction changes from near-
   *  standstill play the nearest clip (yaw baked in the bones, root held) and
   *  transfer its angle to the root at exitTime — instead of the eased pivot. */
  turnInPlace?: TurnClipEntry[]
  /** Authored RUNNING reversals (plant-and-turn): while moving fast with a
   *  reversal-class heading error, the matching clip plays with the root driven
   *  along its measured forward profile; yaw transfers at exitTime and she runs
   *  out along the new heading. */
  runTurn?: RunTurnClipEntry[]
  /** Authored stops: releasing input at speed plays the gear/foot-matched stop with
   *  the root driven along its measured skid profile, instead of a blend to idle.
   *  Re-pressing input interrupts the stop and resumes locomotion. */
  stop?: StopClipEntry[]
}

export interface LocomotionOptions {
  /** When false, update() computes the pose but does NOT call setBlendPose —
   *  read it with getBlendEntries(). For embedding in an AnimationStateMachine
   *  delegate state, which owns the final blend. Default true. */
  autoApply?: boolean
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
  /** Minimum heading error (radians) for an authored turn clip to fire (default
   *  ~100°: reversals only — smaller corrections keep the instant pivot). */
  turnClipMinAngle?: number
  /** Playback rate for turn clips (default 1.4 — the authored turns are deliberate;
   *  game pacing wants them brisker). */
  turnTimeScale?: number
  /** Playback rate for stop clips (default 1.25) — same reasoning. */
  stopTimeScale?: number
  /** Seconds a heading must be held before a release earns an authored stop
   *  (default 0.5). Below it, weaving direction changes blend to idle instead
   *  of skidding metres along the last-held heading. */
  stopCommitTime?: number
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

/** Seconds an interrupted stop's ghost pose takes to dissolve over resumed locomotion. */
const GHOST_FADE = 0.25

function wrapAngle(a: number): number {
  while (a > Math.PI) a -= TWO_PI
  while (a < -Math.PI) a += TWO_PI
  return a
}

export class LocomotionController {
  private readonly model: Model
  private readonly autoApply: boolean
  private lastEntries: BlendEntry[] | null = null
  private readonly clips: LocomotionClips
  private readonly runSpeed: number
  private readonly sprintSpeed: number
  private readonly speedResponse: number
  private readonly turnResponse: number
  private readonly cosTurnThreshold: number
  private readonly turnClipMinAngle: number
  private readonly turnTimeScale: number
  private readonly stopTimeScale: number
  private readonly stopCommitTime: number
  private turning: { entry: TurnClipEntry; time: number } | null = null
  private runTurning: {
    entry: RunTurnClipEntry
    time: number
    startX: number
    startZ: number
    dirX: number
    dirZ: number
  } | null = null
  /** An interrupted authored clip fading out OVER resumed locomotion, so breaking
   *  out of a stop is instantly responsive without a pose pop. */
  private exitGhost: { clip: string; clipTime: number; elapsed: number } | null = null
  private headingHold = 0
  private headingDirX = 0
  private headingDirY = 0
  private stopping: {
    entry: StopClipEntry
    time: number
    startX: number
    startZ: number
    dirX: number
    dirZ: number
    startLevel: number
    /** Fraction of the authored deceleration distance this stop travels. */
    scale: number
    /** Idle's share of the blend at release (level < 1 shows part idle) — the
     *  crossfade source keeps this exact mix or the first stop frame snaps. */
    fromIdleW: number
    /** The gear clip that was visibly playing at release — the crossfade source
     *  (fading over `run` after a SPRINT release would snap the pose). It keeps
     *  advancing through the fade so the motion never freezes. */
    fromClip: string
    fromTime: number
  } | null = null
  private readonly turnEntries: BlendEntry[]
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
  /** Peak-hold of speedLevel (decays ~1.5/s): reversal triggers read this, because
   *  keyboard direction flips pass through a dead moment (W+S cancel, key gap) that
   *  dips the instantaneous level right when the reversal input lands. */
  private recentSpeed = 0
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
  // Strafe mode: non-null = the world yaw the body holds (movement decoupled from facing).
  private facingYaw: number | null = null
  private readonly strafeRun: StrafeClipEntry[] | null
  private readonly strafeSprint: StrafeClipEntry[] | null
  private readonly strafeEntries: BlendEntry[]

  constructor(model: Model, clips: LocomotionClips, options?: LocomotionOptions) {
    this.model = model
    this.clips = clips
    this.runSpeed = options?.runSpeed ?? 67
    this.sprintSpeed = options?.sprintSpeed ?? 92
    this.speedResponse = options?.speedResponse ?? 5
    this.autoApply = options?.autoApply ?? true
    this.turnResponse = options?.turnResponse ?? 10
    this.cosTurnThreshold = Math.cos(options?.turnInPlaceThreshold ?? Math.PI / 4)
    this.turnClipMinAngle = options?.turnClipMinAngle ?? (100 * Math.PI) / 180
    this.turnTimeScale = options?.turnTimeScale ?? 1.4
    this.stopTimeScale = options?.stopTimeScale ?? 1.25
    this.stopCommitTime = options?.stopCommitTime ?? 0.5
    this.turnEntries = [
      { name: clips.idle, time: 0, weight: 0 },
      { name: clips.idle, time: 0, weight: 1 },
      { name: clips.idle, time: 0, weight: 0 },
    ]
    this.steerRate = options?.steerRate ?? 2.5
    this.backpedalScale = options?.backpedalScale ?? 0.5
    this.yawOffset = options?.yawOffset ?? Math.PI
    this.entries = [
      { name: clips.idle, time: 0, weight: 1 },
      { name: clips.run, time: 0, weight: 0 },
      { name: clips.sprint ?? clips.run, time: 0, weight: 0 },
      { name: clips.idle, time: 0, weight: 0 }, // fading ghost of an interrupted clip
    ]
    const byAngle = (a: StrafeClipEntry, b: StrafeClipEntry) => a.angle - b.angle
    this.strafeRun = clips.strafeRun ? [...clips.strafeRun].sort(byAngle) : null
    this.strafeSprint = clips.strafeSprint ? [...clips.strafeSprint].sort(byAngle) : null
    // idle + adjacent pair per gear; unused slots keep weight 0 and are skipped.
    this.strafeEntries = Array.from({ length: 5 }, () => ({ name: clips.idle, time: 0, weight: 0 }))
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

  /** Teleport PLUS a hard reset of every transient motion commitment: any
   *  in-flight authored stop/turn (whose stored start position would otherwise
   *  keep driving the root from where it began), the exit ghost, speed and
   *  momentum, the heading-hold gate, and the held inputs. For handing the
   *  root back after an externally-driven action (a root-motion clip state):
   *  wherever the action ended is simply where she now stands. */
  reset(x: number, y: number, z: number, yaw = 0): void {
    this.position.setXYZ(x, y, z)
    this.yaw = wrapAngle(yaw)
    this.dirX = Math.sin(this.yaw)
    this.dirZ = Math.cos(this.yaw)
    this.speedLevel = 0
    this.recentSpeed = 0
    this.headingHold = 0
    this.headingDirX = 0
    this.headingDirY = 0
    this.gaitPhase = 0
    this.stopping = null
    this.turning = null
    this.runTurning = null
    this.exitGhost = null
    this.inputX = 0
    this.inputY = 0
    this.inputSprint = false
    this.inputForward = 0
    this.inputSteer = 0
  }

  /** Strafe mode: hold the body at this world yaw (a camera forward, a lock-on target)
   *  while setMove's vector drives the directional strafe ring — requires
   *  clips.strafeRun. null returns to turn-toward-movement. */
  setFacing(yaw: number | null): void {
    this.facingYaw = this.strafeRun ? yaw : null
  }

  getPosition(): Vec3 {
    return this.position
  }

  /** Stop driving the model's pose (the blend is cleared; the single-clip player resumes). */
  detach(): void {
    this.lastEntries = null
    if (this.autoApply) this.model.clearBlendPose()
  }

  /** The most recent update()'s blend entries (null before the first update or
   *  after detach). The array is reused between frames — read, don't hold. */
  getBlendEntries(): BlendEntry[] | null {
    return this.lastEntries
  }

  private emit(entries: BlendEntry[]): void {
    this.lastEntries = entries
    if (this.autoApply) this.model.setBlendPose(entries)
  }

  private clipDuration(name: string): number {
    const frames = this.model.getClip(name)?.frameCount ?? 0
    return frames > 0 ? frames / FPS : 1
  }

  /** Advance one frame: integrates yaw + position, updates the clip clocks, hands the
   *  weighted pose to the model, and returns the root transform to apply. */
  update(dt: number): LocomotionPose {
    if (dt > 0.1) dt = 0.1 // tab-switch guard: never integrate a huge step

    if (this.facingYaw !== null && this.strafeRun !== null && !this.tankMode) {
      return this.updateStrafe(dt)
    }

    if (this.turning !== null) {
      return this.updateTurnClip(dt)
    }
    if (this.runTurning !== null) {
      return this.updateRunTurn(dt)
    }
    if (this.stopping !== null) {
      return this.updateStop(dt)
    }

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
      this.recentSpeed = Math.max(this.speedLevel, this.recentSpeed - dt * 1.5)
      // How long the CURRENT heading has been held. An authored skid is the
      // ending of a committed run; weaving through directions (D, then S+D,
      // then S) rebuilds throttle in each new heading but earns no such
      // ending — playing one there slides her metres along whichever way she
      // happened to face last, which reads as drift.
      if (moving) {
        const dx = this.inputX / m
        const dy = this.inputY / m
        this.headingHold =
          this.headingDirX * dx + this.headingDirY * dy > 0.94 ? this.headingHold + dt : 0
        this.headingDirX = dx
        this.headingDirY = dy
      }
      // Release at speed: play the authored stop instead of blending to idle.
      const stops = this.clips.stop
      // Gate on the peak-hold speed: the release decay (throttle/analog) drains the
      // instantaneous level below any threshold before `moving` goes false. The
      // small floor on the actual level keeps standstill taps from faking a skid.
      if (
        !moving &&
        stops &&
        stops.length > 0 &&
        this.recentSpeed >= 0.7 &&
        this.speedLevel >= 0.35 &&
        this.headingHold >= this.stopCommitTime
      ) {
        const gear = this.recentSpeed > 1.4 ? "sprint" : "run"
        const foot = this.gaitPhase < 0.5 ? "L" : "R"
        // Gear must match exactly: supplying stops for only one gear means the
        // other gear keeps the default quick blend-to-idle.
        let best: StopClipEntry | null = null
        let bestScore = -1
        for (const e of stops) {
          if (e.gear !== gear) continue
          const score = e.foot === foot ? 1 : 0
          if (score > bestScore) {
            best = e
            bestScore = score
          }
        }
        if (best) {
          const fromClip = gear === "sprint" && this.clips.sprint ? this.clips.sprint : this.clips.run
          this.stopping = {
            entry: best,
            time: 0,
            startX: this.position.x,
            startZ: this.position.z,
            dirX: this.dirX,
            dirZ: this.dirZ,
            startLevel: Math.min(this.recentSpeed, 2),
            // How much of the authored skid this stop has earned. Each profile
            // was captured at ITS gear's full speed (run = level 1, sprint =
            // level 2), so compare like with like. Entering slower — or moments
            // after a direction change, when the peak-hold still remembers the
            // PREVIOUS heading's sprint — must not replay the full slide.
            // Floored so a stop always still reads as a stop.
            scale: Math.max(0.3, Math.min(1, this.speedLevel / (best.gear === "sprint" ? 2 : 1))),
            fromIdleW: Math.max(0, 1 - this.speedLevel),
            fromClip,
            fromTime: this.gaitPhase * this.clipDuration(fromClip),
          }
          return this.updateStop(dt)
        }
      }
      // Yaw eases toward the input heading only while there is one. Turns are strictly
      // in place: zero translation outside the threshold cone, ramping smoothly to full
      // speed as the body aligns — so direction reversals (L-R-L) cannot drift.
      if (moving) {
        const desired = Math.atan2(this.inputX, this.inputY)
        const err = wrapAngle(desired - this.yaw)
        // Running reversal: while moving fast with a reversal-class error, play the
        // authored plant-and-turn for the current gear and gait foot.
        const runTurns = this.clips.runTurn
        if (runTurns && runTurns.length > 0 && this.recentSpeed >= 0.6 && Math.abs(err) >= (130 * Math.PI) / 180) {
          const gear = this.recentSpeed > 1.4 ? "sprint" : "run"
          const foot = this.gaitPhase < 0.5 ? "L" : "R"
          const side = err < 0 ? -1 : 1
          let best: RunTurnClipEntry | null = null
          let bestScore = -1
          for (const e of runTurns) {
            if (Math.sign(e.angle) !== side) continue
            const score = (e.gear === gear ? 2 : 0) + (e.foot === foot ? 1 : 0)
            if (score > bestScore) {
              best = e
              bestScore = score
            }
          }
          if (best) {
            // Restore the pre-dip speed so she runs OUT of the turn at pace.
            this.speedLevel = Math.max(this.speedLevel, Math.min(this.recentSpeed, 2))
            this.runTurning = {
              entry: best,
              time: 0,
              startX: this.position.x,
              startZ: this.position.z,
              dirX: Math.sin(this.yaw),
              dirZ: Math.cos(this.yaw),
            }
            return this.updateRunTurn(dt)
          }
        }
        // Reversal-class turn from near-standstill: play the authored turn clip
        // whose angle is nearest the error instead of the eased pivot.
        const turnClips = this.clips.turnInPlace
        if (turnClips && turnClips.length > 0 && this.speedLevel < 0.3 && Math.abs(err) >= this.turnClipMinAngle) {
          let best = turnClips[0]
          let bestD = Math.abs(wrapAngle(err - best.angle))
          for (const e of turnClips) {
            const d = Math.abs(wrapAngle(err - e.angle))
            if (d < bestD) {
              best = e
              bestD = d
            }
          }
          this.turning = { entry: best, time: 0 }
          return this.updateTurnClip(dt)
        }
        this.yaw = wrapAngle(this.yaw + err * Math.min(1, this.turnResponse * dt))
        align = Math.max(0, (Math.cos(err) - this.cosTurnThreshold) / (1 - this.cosTurnThreshold))
        this.dirX = this.inputX / m
        this.dirZ = this.inputY / m
      }
    }

    // Speed level ramps linearly toward the target; the pose blend follows it.
    // Input magnitude scales the target (analog sticks and arrival steering slow
    // down instead of overshooting). No sprint while backpedaling.
    const sprinting = this.inputSprint && hasSprint && !(this.tankMode && this.inputForward < 0)
    const magnitude = this.tankMode ? Math.abs(this.inputForward) : Math.min(1, Math.hypot(this.inputX, this.inputY))
    const target = moving ? (sprinting ? 2 : 1) * magnitude : 0
    const maxStep = this.speedResponse * dt
    const d = target - this.speedLevel
    this.speedLevel += Math.abs(d) <= maxStep ? d : Math.sign(d) * maxStep

    // Breakout throttle: while an interrupted skid's ghost still owns part of
    // the pose (planted feet), root motion and the gait clock scale by what's
    // LEFT of it — she accelerates exactly as the skid dissolves, so the feet
    // never slide (滑步) through the handoff.
    let ghostW = 0
    if (this.exitGhost) {
      this.exitGhost.elapsed += dt
      ghostW = Math.max(0, 1 - this.exitGhost.elapsed / GHOST_FADE)
      if (ghostW <= 0) this.exitGhost = null
    }

    // Root motion along the travel direction. In-place clips carry no horizontal root.
    const speed =
      (this.speedLevel <= 1
        ? this.runSpeed * this.speedLevel
        : this.runSpeed + (this.sprintSpeed - this.runSpeed) * (this.speedLevel - 1)) *
      align *
      speedScale *
      (1 - ghostW)
    this.position.x += this.dirX * speed * dt
    this.position.z += this.dirZ * speed * dt

    // Clocks: idle free-runs; run/sprint share one normalized gait phase so a
    // mid-blend stride stays on the same feet.
    const idleDur = this.clipDuration(this.clips.idle)
    const runDur = this.clipDuration(this.clips.run)
    const sprintDur = hasSprint ? this.clipDuration(this.clips.sprint!) : runDur
    this.idleTime = (this.idleTime + dt) % idleDur
    const gaitDur = this.speedLevel <= 1 ? runDur : runDur + (sprintDur - runDur) * (this.speedLevel - 1)
    this.gaitPhase = (this.gaitPhase + (dt * (1 - ghostW)) / gaitDur) % 1

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
    if (this.exitGhost !== null && ghostW > 0) {
      const g = this.exitGhost
      this.entries[0].weight *= 1 - ghostW
      this.entries[1].weight *= 1 - ghostW
      this.entries[2].weight *= 1 - ghostW
      this.entries[3].name = g.clip
      this.entries[3].time = g.clipTime + g.elapsed // the clip's real tail keeps playing
      this.entries[3].weight = ghostW
    } else {
      this.entries[3].weight = 0
    }
    this.emit(this.entries)

    const ry = this.yaw + this.yawOffset
    const half = ry * 0.5
    this.rotation.setXYZW(0, Math.sin(half), 0, Math.cos(half))
    this.pose.yaw = this.yaw
    this.pose.speedLevel = this.speedLevel
    return this.pose
  }

  /** The two ring clips straddling `local` (radians rel. facing), with the blend
   *  fraction between them. The ring is sorted; the last→first gap wraps. */
  private ringPair(ring: StrafeClipEntry[], local: number): { a: StrafeClipEntry; b: StrafeClipEntry; t: number } {
    const n = ring.length
    for (let i = 0; i < n; i++) {
      const a = ring[i]
      const b = ring[(i + 1) % n]
      const gap = (((b.angle - a.angle) % TWO_PI) + TWO_PI) % TWO_PI || TWO_PI
      const rel = (((local - a.angle) % TWO_PI) + TWO_PI) % TWO_PI
      if (rel <= gap + 1e-9) return { a, b, t: rel / gap }
    }
    return { a: ring[0], b: ring[0], t: 0 }
  }

  /** Strafe-mode frame: the body holds facingYaw; movement blends the two ring clips
   *  nearest the local move angle, per gear, all sharing one gait phase. Root motion
   *  follows the input direction at the pair's authored (angle-lerped) speed. */
  private updateStrafe(dt: number): LocomotionPose {
    const runRing = this.strafeRun!
    const sprintRing = this.strafeSprint

    this.yaw = wrapAngle(this.yaw + wrapAngle(this.facingYaw! - this.yaw) * Math.min(1, this.turnResponse * dt))

    const m = Math.hypot(this.inputX, this.inputY)
    const moving = m > 0.05
    const sprinting = this.inputSprint && sprintRing !== null
    const target = moving ? (sprinting ? 2 : 1) * Math.min(1, m) : 0
    const maxStep = this.speedResponse * dt
    const d = target - this.speedLevel
    this.speedLevel += Math.abs(d) <= maxStep ? d : Math.sign(d) * maxStep
    const level = this.speedLevel

    let runA = runRing[0]
    let runB = runRing[0]
    let runT = 0
    let sprintA = sprintRing ? sprintRing[0] : runRing[0]
    let sprintB = sprintA
    let sprintT = 0
    if (moving) {
      const local = wrapAngle(Math.atan2(this.inputX, this.inputY) - this.yaw)
      const rp = this.ringPair(runRing, local)
      runA = rp.a
      runB = rp.b
      runT = rp.t
      if (sprintRing) {
        const sp = this.ringPair(sprintRing, local)
        sprintA = sp.a
        sprintB = sp.b
        sprintT = sp.t
      }
      this.dirX = this.inputX / m
      this.dirZ = this.inputY / m
    }

    // Root motion at the authored (angle-lerped, gear-lerped) clip speed.
    const runSpeed = runA.speed + (runB.speed - runA.speed) * runT
    const topSpeed = sprintRing ? sprintA.speed + (sprintB.speed - sprintA.speed) * sprintT : runSpeed
    const speed = level <= 1 ? runSpeed * level : runSpeed + (topSpeed - runSpeed) * (level - 1)
    this.position.x += this.dirX * speed * dt
    this.position.z += this.dirZ * speed * dt

    // Clocks: idle free-runs; the ring shares one phase (durations are uniform per gear).
    const idleDur = this.clipDuration(this.clips.idle)
    const runDur = this.clipDuration(runA.clip)
    const sprintDur = sprintRing ? this.clipDuration(sprintA.clip) : runDur
    this.idleTime = (this.idleTime + dt) % idleDur
    const gaitDur = level <= 1 ? runDur : runDur + (sprintDur - runDur) * (level - 1)
    this.gaitPhase = (this.gaitPhase + dt / gaitDur) % 1

    let wIdle: number
    let wRun: number
    let wSprint: number
    if (level <= 1) {
      wIdle = 1 - level
      wRun = level
      wSprint = 0
    } else {
      wIdle = 0
      wRun = 2 - level
      wSprint = level - 1
    }

    const e = this.strafeEntries
    e[0].name = this.clips.idle
    e[0].time = this.idleTime
    e[0].weight = wIdle
    e[1].name = runA.clip
    e[1].time = this.gaitPhase * runDur
    e[1].weight = wRun * (1 - runT)
    e[2].name = runB.clip
    e[2].time = this.gaitPhase * runDur
    e[2].weight = wRun * runT
    e[3].name = sprintA.clip
    e[3].time = this.gaitPhase * sprintDur
    e[3].weight = wSprint * (1 - sprintT)
    e[4].name = sprintB.clip
    e[4].time = this.gaitPhase * sprintDur
    e[4].weight = wSprint * sprintT
    this.emit(e)

    const ry = this.yaw + this.yawOffset
    const half = ry * 0.5
    this.rotation.setXYZW(0, Math.sin(half), 0, Math.cos(half))
    this.pose.yaw = this.yaw
    this.pose.speedLevel = level
    return this.pose
  }

  /** Authored turn-in-place frame: the clip rotates the body through its bones while
   *  the root yaw stays frozen; at exitTime the measured angle transfers to the root
   *  in the same instant the pose hands back to idle — the same orientation expressed
   *  two ways, so the cut is seamless. No translation during the turn. */
  private updateTurnClip(dt: number): LocomotionPose {
    const t = this.turning!
    t.time += dt * this.turnTimeScale

    // Speed level settles toward 0 while turning (we were near-standstill already).
    const maxStep = this.speedResponse * dt
    this.speedLevel += Math.abs(-this.speedLevel) <= maxStep ? -this.speedLevel : -Math.sign(this.speedLevel) * maxStep

    const idleDur = this.clipDuration(this.clips.idle)
    this.idleTime = (this.idleTime + dt) % idleDur

    if (t.time >= t.entry.exitTime) {
      // Transfer the angle to the root and hand the pose to idle IN THE SAME frame —
      // leaving the previous blend up would double the rotation for one frame.
      this.yaw = wrapAngle(this.yaw + t.entry.angle)
      this.turning = null
      const e = this.turnEntries
      e[0].weight = 0
      e[1].name = this.clips.idle
      e[1].time = this.idleTime
      e[1].weight = 1
      e[2].weight = 0
      this.emit(e)
    } else {
      // Fade the turn clip over idle at the edges so entry doesn't pop.
      const w = Math.min(1, t.time / 0.12)
      const e = this.turnEntries
      e[0].name = t.entry.clip
      e[0].time = Math.min(t.time, t.entry.exitTime)
      e[0].weight = w
      e[1].name = this.clips.idle
      e[1].time = this.idleTime
      e[1].weight = 1 - w
      e[2].weight = 0
      this.emit(e)
    }

    const ry = this.yaw + this.yawOffset
    const half = ry * 0.5
    this.rotation.setXYZW(0, Math.sin(half), 0, Math.cos(half))
    this.pose.yaw = this.yaw
    this.pose.speedLevel = this.speedLevel
    return this.pose
  }

  /** Authored-stop frame: root follows the clip's measured skid profile along the
   *  release heading; input returning interrupts and resumes locomotion at a level
   *  proportional to how much of the stop remains. */
  private updateStop(dt: number): LocomotionPose {
    const t = this.stopping!
    const entry = t.entry

    // Interrupt: input came back — locomotion resumes instantly from a LOW level
    // (a stop is committed; breaking out is a fresh walk-up, never a drift), while
    // the stop pose fades out as a ghost instead of hard-cutting.
    if (Math.hypot(this.inputX, this.inputY) > 0.05) {
      this.speedLevel = Math.min(t.startLevel * (1 - Math.min(1, t.time / entry.exitTime)), 0.3)
      this.recentSpeed = this.speedLevel
      this.exitGhost = { clip: entry.clip, clipTime: Math.min(t.time, entry.exitTime), elapsed: 0 }
      this.stopping = null
      return this.update(dt)
    }

    t.time += dt * this.stopTimeScale
    const clipT = Math.min(t.time, entry.exitTime)
    const fwd = LocomotionController.profileAt(entry.forward, clipT / entry.exitTime) * t.scale
    this.position.x = t.startX + t.dirX * fwd
    this.position.z = t.startZ + t.dirZ * fwd

    this.speedLevel = t.startLevel * Math.max(0, 1 - t.time / entry.exitTime)

    const idleDur = this.clipDuration(this.clips.idle)
    this.idleTime = (this.idleTime + dt) % idleDur

    const FADE_OUT = 0.35
    if (t.time >= entry.exitTime + FADE_OUT) {
      this.stopping = null
      this.speedLevel = 0
      this.gaitPhase = 0
      const e = this.turnEntries
      e[0].weight = 0
      e[1].name = this.clips.idle
      e[1].time = this.idleTime
      e[1].weight = 1
      e[2].weight = 0
      this.emit(e)
    } else if (t.time >= entry.exitTime) {
      // Settle tail: the root is already still, so keep playing the clip's own
      // recovery (it exists past exitTime) while fading to idle — no hard cut.
      const wOut = (t.time - entry.exitTime) / FADE_OUT
      const e = this.turnEntries
      e[0].name = entry.clip
      e[0].time = t.time // real tail frames beyond exitTime
      e[0].weight = 1 - wOut
      e[1].name = this.clips.idle
      e[1].time = this.idleTime
      e[1].weight = wOut
      e[2].weight = 0
      this.emit(e)
    } else {
      const fromDur = this.clipDuration(t.fromClip)
      t.fromTime = (t.fromTime + dt) % fromDur
      const w = Math.min(1, t.time / 0.25)
      const e = this.turnEntries
      e[0].name = entry.clip
      e[0].time = clipT
      e[0].weight = w
      e[1].name = t.fromClip
      e[1].time = t.fromTime
      e[1].weight = (1 - w) * (1 - t.fromIdleW)
      e[2].name = this.clips.idle
      e[2].time = this.idleTime
      e[2].weight = (1 - w) * t.fromIdleW
      this.emit(e)
    }

    const ry = this.yaw + this.yawOffset
    const half = ry * 0.5
    this.rotation.setXYZW(0, Math.sin(half), 0, Math.cos(half))
    this.pose.yaw = this.yaw
    this.pose.speedLevel = this.speedLevel
    return this.pose
  }

  /** Linear interpolation over a uniformly sampled profile at t in [0, 1]. */
  private static profileAt(samples: number[], t: number): number {
    const n = samples.length - 1
    if (n <= 0) return 0
    const x = Math.min(Math.max(t, 0), 1) * n
    const i = Math.min(n - 1, Math.floor(x))
    return samples[i] + (samples[i + 1] - samples[i]) * (x - i)
  }

  /** Running-reversal frame: bones carry the turn while the root travels the clip's
   *  AUTHORED forward profile along the trigger heading (overrun, plant, return) —
   *  no fabricated motion, so the feet stay glued. At exitTime the yaw transfers
   *  and the normal path runs her out along the new heading. */
  private updateRunTurn(dt: number): LocomotionPose {
    const t = this.runTurning!
    t.time += dt
    const entry = t.entry
    const clipT = Math.min(t.time, entry.exitTime)

    const fwd = LocomotionController.profileAt(entry.forward, clipT / entry.exitTime)
    this.position.x = t.startX + t.dirX * fwd
    this.position.z = t.startZ + t.dirZ * fwd

    const idleDur = this.clipDuration(this.clips.idle)
    this.idleTime = (this.idleTime + dt) % idleDur

    if (t.time >= entry.exitTime) {
      this.yaw = wrapAngle(this.yaw + entry.angle)
      this.runTurning = null
      // She exits mid-stride: keep the speed level, restart the gait cleanly, and
      // point the travel direction along the new heading so the next frame runs out.
      this.gaitPhase = 0
      this.dirX = Math.sin(this.yaw)
      this.dirZ = Math.cos(this.yaw)
      const e = this.turnEntries
      e[0].name = this.clips.run
      e[0].time = 0
      e[0].weight = Math.min(1, this.speedLevel)
      e[1].name = this.clips.idle
      e[1].time = this.idleTime
      e[1].weight = 1 - Math.min(1, this.speedLevel)
      e[2].weight = 0
      this.emit(e)
    } else {
      const w = Math.min(1, t.time / 0.1)
      const e = this.turnEntries
      e[0].name = entry.clip
      e[0].time = clipT
      e[0].weight = w
      e[1].name = this.clips.run
      e[1].time = this.gaitPhase * this.clipDuration(this.clips.run)
      e[1].weight = 1 - w
      e[2].weight = 0
      this.emit(e)
    }

    const ry = this.yaw + this.yawOffset
    const half = ry * 0.5
    this.rotation.setXYZW(0, Math.sin(half), 0, Math.cos(half))
    this.pose.yaw = this.yaw
    this.pose.speedLevel = this.speedLevel
    return this.pose
  }
}
