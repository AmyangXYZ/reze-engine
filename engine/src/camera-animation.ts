// Samples an MMD camera track (from a VMD) at a playback time. Four bezier-interpolated
// channels — target (Vec3), rotation (Vec3 euler), distance, fov — driven off the same
// clock as the model motion so the shot stays synced to the dance.

import { Vec3 } from "./math"
import { bezierInterpolate } from "./animation"
import type { CameraKeyframe } from "./vmd-loader"

const FPS = 30
const DEG2RAD = Math.PI / 180

/** A sampled camera pose. `rotation` is euler radians; `fov` is radians. */
export interface CameraPose {
  target: Vec3
  rotation: Vec3
  distance: number
  fov: number
}

// MMD camera interpolation is stored interleaved across the 6 channels:
// bytes [x1×6][x2×6][y1×6][y2×6]. Channel c's bezier is (ip[c], ip[6+c], ip[12+c], ip[18+c]).
// Channels: 0=posX 1=posY 2=posZ 3=rotation 4=distance 5=fov.
function bez(ip: Uint8Array, c: number, t: number): number {
  return bezierInterpolate(ip[c] / 127, ip[6 + c] / 127, ip[12 + c] / 127, ip[18 + c] / 127, t)
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
    const span = b.frame - a.frame
    const localT = span > 0 ? (frame - a.frame) / span : 0
    // The incoming interpolation curve is stored on the segment's end keyframe (b).
    const ip = b.interpolation

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

  private pose(f: CameraKeyframe): CameraPose {
    return { target: f.target, rotation: f.rotation, distance: f.distance, fov: f.fov * DEG2RAD }
  }
}
