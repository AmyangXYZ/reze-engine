// The camera on a bone: her own eyes (POV), or a phone in her hand (selfie).
//
// A third way to hold the camera, beside the orbit and the follow. The follow
// moves the point the orbit turns about and leaves the orbit's own angle and
// distance to the person at the mouse; a mount places the EYE and the LOOK from
// the rig itself, every frame, so the shot is hers.
//
// Stepped by the engine after the pose and physics, before the camera uniforms
// (as the follow is), so it never trails her by a frame and an export renders
// what the editor showed. A camera VMD or setCameraPose still wins: they are
// scene data and an explicit host statement about where the camera IS.

import { Vec3 } from "./math"
import { Quat } from "./math"
import type { Model } from "./model"

export type CameraMountMode = "pov" | "selfie"

/** What setCameraMount asks for. */
export interface CameraMountOptions {
  /** "pov": through her eyes, turning with her head. "selfie": held in her
   *  hand, aimed back at her face. */
  mode: CameraMountMode
  /** The bone the camera rides. POV 頭 (the head), selfie 右手首 (a wrist). */
  bone?: string
  /** Seconds the camera's turn trails the bone, eased (POV 0.3, selfie 0.25):
   *  a dance's head snaps read as a jolt without it. 0 is rigid. In POV the
   *  turn is eased twice over and the nod and the lens's travel take their
   *  own share of it — see POV_PITCH_TAU and POV_POS_TAU. */
  smoothing?: number
  /** POV: keep the horizon level, dropping the head's tilt. Default true. */
  level?: boolean
  /** Selfie: how far from her face the phone is held at full stretch, model
   *  units (an arm's length is about 7). Closer when her hand is. Default 7. */
  distance?: number
}

/** Where the mounted camera is this frame, world space. */
export interface MountPose {
  eye: Vec3
  forward: Vec3
  up: Vec3
}

/** The bones a face is read from, and the head. */
const HEAD = "頭"
const EYES = ["左目", "右目"] as const
/** POV: how far in front of the eyeballs the lens sits, model units — past the
 *  eye's own surface, so nothing of her that is still drawn sits on the lens. */
const POV_FORWARD = 0.25
/** Where a rig without eye bones keeps its eyes: up the head bone, model units. */
const EYES_UP_FALLBACK = 1.1
/** Selfie: the nearest the phone comes to her face, as a share of `distance`. */
const SELFIE_MIN = 0.55
const UP = new Vec3(0, 1, 0)
/**
 * POV is a STABILISED head cam, the way a gimbal or a game's first-person view
 * is: a dancer's head bobs and nods on every beat, and a lens bolted to it
 * shakes the whole frame with each one. So:
 *  - the turn is split into yaw and pitch and each eased twice over (two
 *    first-order lags in a row: a soft start and a soft stop, where one lag
 *    jerks into motion);
 *  - the nod is slower still (POV_PITCH_TAU × smoothing) and only
 *    POV_PITCH of it reaches the view — the head looks down and up far more
 *    than a person's gaze does;
 *  - the lens's travel is eased too (POV_POS_TAU × smoothing), so the bob of
 *    her head is smoothed out of the eye point. The head is hidden from this
 *    camera, so a lens a frame behind it never shows the inside of her face.
 */
const POV_PITCH = 0.6
const POV_PITCH_TAU = 1.6
const POV_POS_TAU = 0.5
/** The shortest signed turn from a to b, radians. */
const wrapAngle = (a: number): number => Math.atan2(Math.sin(a), Math.cos(a))

export function resolveMountOptions(o: CameraMountOptions): Required<CameraMountOptions> {
  const pov = o.mode !== "selfie"
  return {
    mode: pov ? "pov" : "selfie",
    bone: o.bone || (pov ? HEAD : "右手首"),
    smoothing: Math.min(2, Math.max(0, o.smoothing ?? (pov ? 0.3 : 0.25))),
    level: o.level ?? true,
    distance: Math.min(40, Math.max(2, o.distance ?? 7)),
  }
}

/** A model-space point placed as the scene has the model (setModelTransform). */
function toWorld(m: Model, p: Vec3): Vec3 {
  const s = m.scale
  const out = new Vec3(p.x * s, p.y * s, p.z * s)
  Quat.rotateVecInto(m.rotation, out, out)
  out.setXYZ(out.x + m.position.x, out.y + m.position.y, out.z + m.position.z)
  return out
}

/** A model-space direction turned as the model is; unit length. */
function dirToWorld(m: Model, x: number, y: number, z: number): Vec3 {
  const out = new Vec3(x, y, z)
  Quat.rotateVecInto(m.rotation, out, out)
  const l = Math.hypot(out.x, out.y, out.z) || 1
  out.setXYZ(out.x / l, out.y / l, out.z / l)
  return out
}

/**
 * Turn a unit vector toward another by k of the angle between them (a slerp).
 * Not a straight blend: two opposite vectors blend to nothing, so a half turn
 * — her spinning round — would never start to swing.
 */
function ease(v: Vec3, to: Vec3, k: number): void {
  if (k >= 1) {
    v.set(to)
    return
  }
  const d = Math.max(-1, Math.min(1, v.x * to.x + v.y * to.y + v.z * to.z))
  const angle = Math.acos(d)
  if (angle < 1e-5) return
  // The axis to turn about: v × to, or for a half turn any axis square to v.
  let ax = v.y * to.z - v.z * to.y
  let ay = v.z * to.x - v.x * to.z
  let az = v.x * to.y - v.y * to.x
  let al = Math.hypot(ax, ay, az)
  if (al < 1e-6) {
    // Up, unless v is itself vertical.
    if (Math.abs(v.y) < 0.9) {
      ax = -v.z
      ay = 0
      az = v.x
    } else {
      ax = 0
      ay = v.z
      az = -v.y
    }
    al = Math.hypot(ax, ay, az)
  }
  ax /= al
  ay /= al
  az /= al
  // Rodrigues: v cos θ + (a × v) sin θ + a (a·v)(1 − cos θ).
  const t = angle * k
  const c = Math.cos(t)
  const s = Math.sin(t)
  const dot = ax * v.x + ay * v.y + az * v.z
  const cx = ay * v.z - az * v.y
  const cy = az * v.x - ax * v.z
  const cz = ax * v.y - ay * v.x
  const x = v.x * c + cx * s + ax * dot * (1 - c)
  const y = v.y * c + cy * s + ay * dot * (1 - c)
  const z = v.z * c + cz * s + az * dot * (1 - c)
  const l = Math.hypot(x, y, z) || 1
  v.setXYZ(x / l, y / l, z / l)
}

/**
 * Between her eyes, world space: the eye bones' midpoint, or up the head bone
 * on a rig without them. Null without a head.
 */
export function facePoint(m: Model): Vec3 | null {
  const l = m.getBoneWorldPosition(EYES[0])
  const r = m.getBoneWorldPosition(EYES[1])
  if (l && r) return toWorld(m, new Vec3((l.x + r.x) / 2, (l.y + r.y) / 2, (l.z + r.z) / 2))
  const h = m.getBoneWorldMatrix(HEAD)
  if (!h) return null
  const up = new Vec3(h[4], h[5], h[6])
  const ul = Math.hypot(up.x, up.y, up.z) || 1
  return toWorld(m, new Vec3(h[12] + (up.x / ul) * EYES_UP_FALLBACK, h[13] + (up.y / ul) * EYES_UP_FALLBACK, h[14] + (up.z / ul) * EYES_UP_FALLBACK))
}

export class CameraMount {
  options: Required<CameraMountOptions>
  private seeded = false
  private readonly forward = new Vec3(0, 0, -1)
  private readonly up = new Vec3(0, 1, 0)
  /** Selfie: from her face toward the phone, eased. */
  private readonly arm = new Vec3(0, 0, -1)
  private reach = 0
  /** POV: yaw and pitch through two lags each ([first, second]), and the
   *  lens point through two. */
  private readonly yaw = [0, 0]
  private readonly pitch = [0, 0]
  private readonly eye1 = new Vec3(0, 0, 0)
  private readonly eye2 = new Vec3(0, 0, 0)

  constructor(
    readonly model: Model,
    options: CameraMountOptions,
  ) {
    this.options = resolveMountOptions(options)
  }

  setOptions(options: CameraMountOptions): void {
    const next = resolveMountOptions(options)
    // A new mode or bone starts from where the rig is, not from the old shot.
    if (next.mode !== this.options.mode || next.bone !== this.options.bone) this.seeded = false
    this.options = next
  }

  /** Snap to the rig on the next step: a seek or a cut is no camera move. */
  reseed(): void {
    this.seeded = false
  }

  /** This frame's pose, after her pose. Null when the rig lacks the bones. */
  step(dt: number): MountPose | null {
    const o = this.options
    const k = !this.seeded || o.smoothing <= 0 || dt <= 0 ? 1 : 1 - Math.exp(-dt / o.smoothing)
    // A big jump (a seek, a tab come back) is no move to glide through.
    const snap = !this.seeded || dt > 0.25
    return o.mode === "pov" ? this.pov(snap ? 0 : dt, snap) : this.selfie(snap ? 1 : k)
  }

  private pov(dt: number, snap: boolean): MountPose | null {
    const m = this.model
    const h = m.getBoneWorldMatrix(this.options.bone) ?? m.getBoneWorldMatrix(HEAD)
    if (!h) return null
    const face = this.options.bone === HEAD ? facePoint(m) : toWorld(m, new Vec3(h[12], h[13], h[14]))
    if (!face) return null
    // MMD rests facing −Z: forward is the head's negated third column, up its second.
    const fwd = dirToWorld(m, -h[8], -h[9], -h[10])
    const yawT = Math.atan2(fwd.x, fwd.z)
    const pitchT = Math.asin(Math.max(-1, Math.min(1, fwd.y))) * POV_PITCH
    const tau = this.options.smoothing
    const lag = (t: number) => (snap || t <= 0 ? 1 : 1 - Math.exp(-dt / t))
    const kYaw = lag(tau)
    const kPitch = lag(tau * POV_PITCH_TAU)
    const kPos = lag(tau * POV_POS_TAU)
    if (snap) {
      this.yaw[0] = this.yaw[1] = yawT
      this.pitch[0] = this.pitch[1] = pitchT
      this.eye1.set(face)
      this.eye2.set(face)
    } else {
      this.yaw[0] += wrapAngle(yawT - this.yaw[0]) * kYaw
      this.yaw[1] += wrapAngle(this.yaw[0] - this.yaw[1]) * kYaw
      this.pitch[0] += (pitchT - this.pitch[0]) * kPitch
      this.pitch[1] += (this.pitch[0] - this.pitch[1]) * kPitch
      const e1 = this.eye1
      e1.setXYZ(e1.x + (face.x - e1.x) * kPos, e1.y + (face.y - e1.y) * kPos, e1.z + (face.z - e1.z) * kPos)
      const e2 = this.eye2
      e2.setXYZ(e2.x + (e1.x - e2.x) * kPos, e2.y + (e1.y - e2.y) * kPos, e2.z + (e1.z - e2.z) * kPos)
    }
    const cp = Math.cos(this.pitch[1])
    const f = new Vec3(Math.sin(this.yaw[1]) * cp, Math.sin(this.pitch[1]), Math.cos(this.yaw[1]) * cp)
    this.forward.set(f)
    // The tilt only when asked for, and then eased like the turn.
    const up = this.options.level || Math.abs(f.y) > 0.97 ? UP : dirToWorld(m, h[4], h[5], h[6])
    ease(this.up, up, kYaw)
    this.seeded = true
    const s = POV_FORWARD * m.scale
    const e = this.eye2
    return {
      eye: new Vec3(e.x + f.x * s, e.y + f.y * s, e.z + f.z * s),
      forward: f,
      up: new Vec3(this.up.x, this.up.y, this.up.z),
    }
  }

  private selfie(k: number): MountPose | null {
    const m = this.model
    const face = facePoint(m)
    const hand = m.getBoneWorldPosition(this.options.bone)
    if (!face || !hand) return null
    const hw = toWorld(m, hand)
    let ax = hw.x - face.x
    let ay = hw.y - face.y
    let az = hw.z - face.z
    const d = Math.hypot(ax, ay, az)
    if (d < 1e-4) {
      // Hand on the face: hold the direction it last had.
      ax = this.arm.x
      ay = this.arm.y
      az = this.arm.z
    } else {
      ax /= d
      ay /= d
      az /= d
    }
    const full = this.options.distance * m.scale
    const reach = Math.min(full, Math.max(full * SELFIE_MIN, d))
    ease(this.arm, new Vec3(ax, ay, az), k)
    this.reach = this.seeded ? this.reach + (reach - this.reach) * k : reach
    this.seeded = true
    const a = this.arm
    const eye = new Vec3(face.x + a.x * this.reach, face.y + a.y * this.reach, face.z + a.z * this.reach)
    // Aimed back at her face, level: a phone is held upright, not at the wrist's twist.
    const forward = new Vec3(-a.x, -a.y, -a.z)
    const up = Math.abs(forward.y) < 0.97 ? new Vec3(0, 1, 0) : dirToWorld(m, 0, 0, -1)
    return { eye, forward, up }
  }
}

/**
 * The materials that ARE her head — hair, face, eyes, brows — for a POV shot
 * to leave out of the camera (and only the camera: her shadow and her mirror
 * image keep them). A material is hers when most of its skinning weight rides
 * 頭 or a bone under it. Computed once per mount.
 */
export function headMaterials(m: Model): string[] {
  const bones = m.getSkeleton().bones
  const head = bones.findIndex((b) => b.name === HEAD)
  if (head < 0) return []
  const under = new Uint8Array(bones.length)
  for (let i = 0; i < bones.length; i++) {
    // Walk up to the root; a rig's chain is short.
    for (let b = i, n = 0; b >= 0 && n < 256; b = bones[b].parentIndex, n++) {
      if (b === head) {
        under[i] = 1
        break
      }
    }
  }
  const { joints, weights } = m.getSkinning()
  const indices = m.getIndices()
  const out: string[] = []
  let start = 0
  for (const mat of m.getMaterials()) {
    const count = mat.vertexCount
    let onHead = 0
    let total = 0
    for (let i = start; i < start + count; i++) {
      const v = indices[i]
      for (let j = 0; j < 4; j++) {
        const w = weights[v * 4 + j]
        if (w === 0) continue
        total += w
        if (under[joints[v * 4 + j]]) onHead += w
      }
    }
    start += count
    if (total > 0 && onHead / total > 0.5) out.push(mat.name)
  }
  return out
}
