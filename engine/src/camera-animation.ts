// Samples an MMD camera track (from a VMD) at a playback time. Four bezier-interpolated
// channels — target (Vec3), rotation (Vec3 euler), distance, fov — driven off the same
// clock as the model motion so the shot stays synced to the dance.

import { Vec3 } from "./math"
import { bezierInterpolate } from "./animation"
import { DEFAULT_CAMERA_INTERPOLATION, type CameraKeyframe } from "./vmd-loader"

const FPS = 30
const DEG2RAD = Math.PI / 180

/** A sampled camera pose. `rotation` is euler radians; `fov` is radians. */
export interface CameraPose {
  target: Vec3
  rotation: Vec3
  distance: number
  fov: number
}

// MMD camera interpolation is 24 bytes, stored CONTIGUOUS per channel — each channel's 4
// bezier bytes are [x1, x2, y1, y2] in order. Channels: 0=posX 1=posY 2=posZ 3=rotation
// 4=distance 5=fov. So channel c lives at bytes [4c .. 4c+3]. (Matches babylon-mmd, which
// slices these into position/rotation/distance/fov interpolation buffers.)
function bez(ip: Uint8Array, c: number, t: number): number {
  const b = c * 4
  return bezierInterpolate(ip[b] / 127, ip[b + 1] / 127, ip[b + 2] / 127, ip[b + 3] / 127, t)
}

const lerp = (a: number, b: number, w: number): number => a + (b - a) * w

export class CameraAnimation {
  private readonly frames: CameraKeyframe[]
  /** Track length in seconds (last keyframe). */
  readonly duration: number

  constructor(frames: CameraKeyframe[]) {
    this.frames = frames
    this.duration = frames.length ? frames[frames.length - 1].frame / FPS : 0
  }

  get frameCount(): number {
    return this.frames.length
  }

  /**
   * Every keyframe's frame index, ascending.
   *
   * For a host drawing the track: a camera VMD is sparse and punchy compared to
   * a dance, so where its keys sit IS where its cuts and moves are. The poses
   * stay private — a timeline wants the rhythm, not the camera.
   */
  keyframeIndices(): number[] {
    return this.frames.map((f) => f.frame)
  }

  /**
   * Every keyframe, in full — the track as an editable document.
   *
   * The counterpart to keyframeIndices(), which deliberately withholds the
   * poses because a timeline only wants the rhythm. An EDITOR wants the poses:
   * it has to show what the shot does at a cut, let it be changed, and write
   * the result back out. Shallow copies, so a host mutating what it gets back
   * cannot reach into the track this is sampling from mid-playback.
   */
  keyframes(): CameraKeyframe[] {
    return this.frames.map((f) => ({ ...f }))
  }

  /** Sample the camera pose at time `t` (seconds). Clamps to the track ends; null if empty. */
  sample(t: number): CameraPose | null {
    const frames = this.frames
    const n = frames.length
    if (n === 0) return null

    const frame = t * FPS
    if (frame <= frames[0].frame) return this.pose(frames[0])
    if (frame >= frames[n - 1].frame) return this.pose(frames[n - 1])

    // Binary search for the segment [a, b] with a.frame <= frame < b.frame.
    let lo = 0
    let hi = n - 1
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1
      if (frames[mid].frame <= frame) lo = mid
      else hi = mid
    }
    const a = frames[lo]
    const b = frames[hi]
    // Keys one frame apart are a cut: MMD never draws between them, so the
    // earlier key holds until the later one's frame. A time meant as that frame
    // can land a hair under it (11/30 * 30), and still cuts.
    //
    // Not inside a run of keys on every frame, though: that is a baked camera
    // (a converter's or a capture's), whose keys are samples of one move. Held,
    // it steps at 30 fps, and at 60 fps each pose shows twice, a judder. A
    // baked track's own cut is the one step far bigger than its neighbours.
    if (b.frame - a.frame === 1 && this.isCut(lo)) return this.pose(frame < b.frame - 1e-4 ? a : b)
    const span = b.frame - a.frame
    const localT = span > 0 ? (frame - a.frame) / span : 0
    // The incoming interpolation curve is stored on the segment's end keyframe (b).
    const ip = b.interpolation ?? DEFAULT_CAMERA_INTERPOLATION

    return {
      target: new Vec3(
        lerp(a.target.x, b.target.x, bez(ip, 0, localT)),
        lerp(a.target.y, b.target.y, bez(ip, 1, localT)),
        lerp(a.target.z, b.target.z, bez(ip, 2, localT)),
      ),
      // MMD rotation animates as one channel (single bezier for all three euler components).
      rotation: (() => {
        const w = bez(ip, 3, localT)
        return new Vec3(
          lerp(a.rotation.x, b.rotation.x, w),
          lerp(a.rotation.y, b.rotation.y, w),
          lerp(a.rotation.z, b.rotation.z, w),
        )
      })(),
      distance: lerp(a.distance, b.distance, bez(ip, 4, localT)),
      fov: lerp(a.fov * DEG2RAD, b.fov * DEG2RAD, bez(ip, 5, localT)),
    }
  }

  /** Whether the one-frame step from key i to i + 1 is a cut, not a baked sample. */
  private isCut(i: number): boolean {
    const f = this.frames
    const before = i > 0 && f[i].frame - f[i - 1].frame === 1
    const after = i + 2 < f.length && f[i + 2].frame - f[i + 1].frame === 1
    if (!before && !after) return true
    const step = cameraStep(f[i], f[i + 1])
    const near = Math.max(before ? cameraStep(f[i - 1], f[i]) : 0, after ? cameraStep(f[i + 1], f[i + 2]) : 0)
    return step > CUT_RATIO * near + CUT_FLOOR
  }

  private pose(f: CameraKeyframe): CameraPose {
    return { target: f.target, rotation: f.rotation, distance: f.distance, fov: f.fov * DEG2RAD }
  }
}

/** A baked step this many times its neighbours' (plus the floor) is a cut. */
const CUT_RATIO = 6
const CUT_FLOOR = 0.05

/**
 * How far the camera moves in one step: its eye travel in model units plus the
 * turn in radians and the fov change in degrees scaled to read alike, so a
 * cut on any channel (a jump, a swing, a zoom) stands out from a smooth run.
 */
function cameraStep(a: CameraKeyframe, b: CameraKeyframe): number {
  const t = Math.hypot(b.target.x - a.target.x, b.target.y - a.target.y, b.target.z - a.target.z)
  const r = Math.hypot(b.rotation.x - a.rotation.x, b.rotation.y - a.rotation.y, b.rotation.z - a.rotation.z)
  return t + Math.abs(b.distance - a.distance) + r * 10 + Math.abs(b.fov - a.fov) * 0.2
}
