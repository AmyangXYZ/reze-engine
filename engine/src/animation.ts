import { Quat, Vec3 } from "./math"

export interface ControlPoint {
  x: number
  y: number
}

export interface BoneInterpolation {
  rotation: ControlPoint[]
  translationX: ControlPoint[]
  translationY: ControlPoint[]
  translationZ: ControlPoint[]
}

export interface BoneKeyframe {
  boneName: string
  frame: number
  rotation: Quat
  translation: Vec3
  interpolation: BoneInterpolation
}

export interface MorphKeyframe {
  morphName: string
  frame: number
  weight: number
}

/** One IK chain switching on or off, keyed by its IK bone (左足ＩＫ and friends).
 *  VMD stores these as steps — a state holds until the next keyframe changes it. */
export interface IkKeyframe {
  frame: number
  enabled: boolean
}

export interface AnimationClip {
  boneTracks: Map<string, BoneKeyframe[]>
  morphTracks: Map<string, MorphKeyframe[]>
  /** Per-chain IK state over time. Absent means "leave IK as the host set it" —
   *  which is what every clip built before this existed will do. */
  ikTracks?: Map<string, IkKeyframe[]>
  frameCount: number // last keyframe frame index
}

export interface AnimationPlayOptions {
  priority?: number // Higher number = higher priority. Default: 0.
  loop?: boolean // When true, timeline wraps at end. Default: false.
}

/** Wall-clock playback progress; `current`/`duration` are seconds (clip span uses `AnimationClip.frameCount`, not `duration`). */
export interface AnimationProgress {
  animationName: string | null
  current: number
  duration: number
  percentage: number
  looping: boolean
  /** True while the timeline is advancing (not idle at end, not paused). */
  playing: boolean
  paused: boolean
}

export const FPS = 30

interface QueuedAnimationRequest {
  name: string
  priority: number
  loop: boolean
}

// Priority-aware playback: higher priority preempts, otherwise latest request is queued.
export class AnimationState {
  private animations = new Map<string, AnimationClip>()
  private currentAnimationName: string | null = null
  private currentFrame = 0
  private currentPriority = 0
  private currentLoop = false
  private isPlaying = false
  private isPaused = false
  private nextAnimation: QueuedAnimationRequest | null = null
  private onEnd: ((animationName: string) => void) | null = null

  loadAnimation(name: string, clip: AnimationClip): void {
    // Copied field by field rather than stored by reference, so the caller's
    // object cannot mutate under playback. Every field has to be listed — an
    // omission here is silent, and drops that part of the clip on the floor.
    this.animations.set(name, {
      boneTracks: clip.boneTracks,
      morphTracks: clip.morphTracks,
      ikTracks: clip.ikTracks,
      frameCount: clip.frameCount,
    })
  }

  removeAnimation(name: string): void {
    this.animations.delete(name)
    if (this.currentAnimationName === name) {
      this.currentAnimationName = null
      this.currentFrame = 0
      this.currentPriority = 0
      this.currentLoop = false
      this.isPlaying = false
      this.nextAnimation = this.nextAnimation?.name === name ? null : this.nextAnimation
    } else if (this.nextAnimation?.name === name) {
      this.nextAnimation = null
    }
  }

  play(name: string, options?: AnimationPlayOptions): boolean
  play(): void
  play(name?: string, options?: AnimationPlayOptions): boolean | void {
    if (name === undefined) {
      if (this.currentAnimationName && this.animations.has(this.currentAnimationName)) {
        this.isPaused = false
        this.isPlaying = true
      }
      return
    }
    if (!this.animations.has(name)) return false
    const priority = options?.priority ?? 0
    const loop = options?.loop ?? false

    if (this.currentAnimationName === name) {
      this.currentFrame = 0
      this.currentPriority = priority
      this.currentLoop = loop
      this.isPlaying = true
      this.isPaused = false
      return true
    }

    if (this.isPlaying && !this.isPaused) {
      if (priority > this.currentPriority) {
        this.currentAnimationName = name
        this.currentFrame = 0
        this.currentPriority = priority
        this.currentLoop = loop
        this.isPlaying = true
        this.isPaused = false
        this.nextAnimation = null
        return true
      }
      this.nextAnimation = { name, priority, loop }
      return true
    }
    this.currentAnimationName = name
    this.currentFrame = 0
    this.currentPriority = priority
    this.currentLoop = loop
    this.isPlaying = true
    this.isPaused = false
    this.nextAnimation = null
    return true
  }

  update(deltaTime: number): { ended: boolean; animationName: string | null } {
    if (!this.isPlaying || this.isPaused || this.currentAnimationName === null) {
      return { ended: false, animationName: this.currentAnimationName }
    }
    const clip = this.animations.get(this.currentAnimationName)
    if (!clip) return { ended: false, animationName: this.currentAnimationName }

    const frameCount = clip.frameCount
    if (frameCount <= 0 || !Number.isFinite(frameCount)) {
      return { ended: false, animationName: this.currentAnimationName }
    }

    this.currentFrame += deltaTime * FPS

    if (this.currentFrame >= frameCount) {
      if (this.currentLoop) {
        while (this.currentFrame >= frameCount) {
          this.currentFrame -= frameCount
        }
        return { ended: false, animationName: this.currentAnimationName }
      }
      this.currentFrame = frameCount
      const finishedName = this.currentAnimationName
      this.onEnd?.(finishedName)
      if (this.nextAnimation !== null) {
        const next = this.nextAnimation
        this.nextAnimation = null
        this.currentAnimationName = next.name
        this.currentFrame = 0
        this.currentPriority = next.priority
        this.currentLoop = next.loop
        this.isPlaying = true
        this.isPaused = false
        return { ended: true, animationName: finishedName }
      }
      this.isPlaying = false
      return { ended: true, animationName: finishedName }
    }
    return { ended: false, animationName: this.currentAnimationName }
  }

  pause(): void {
    this.isPaused = true
  }

  stop(): void {
    this.isPlaying = false
    this.isPaused = false
    this.currentFrame = 0
    this.currentPriority = 0
    this.currentLoop = false
    this.nextAnimation = null
  }

  /** stop() + deactivate the clip entirely. stop() deliberately keeps the clip
   *  current (transport re-play resumes it); clear() forgets it, so the pose
   *  stops being re-applied each frame and bone/morph resets actually show. */
  clear(): void {
    this.stop()
    this.currentAnimationName = null
  }

  // Seek by absolute timeline seconds, not frame index.
  seek(seconds: number): void {
    const clip = this.getCurrentClip()
    if (!clip || clip.frameCount <= 0 || !Number.isFinite(clip.frameCount)) return
    const targetFrame = seconds * FPS
    this.currentFrame = Math.max(0, Math.min(targetFrame, clip.frameCount))
  }

  getCurrentClip(): AnimationClip | null {
    return this.currentAnimationName !== null ? this.animations.get(this.currentAnimationName) ?? null : null
  }

  getAnimationClip(name: string): AnimationClip | null {
    return this.animations.get(name) ?? null
  }

  getCurrentAnimation(): string | null {
    return this.currentAnimationName
  }

  getCurrentTime(): number {
    const clip = this.getCurrentClip()
    if (!clip) return 0
    return this.currentFrame / FPS
  }

  getCurrentFrame(): number {
    return this.currentFrame
  }

  /** Clip length in seconds (`frameCount / FPS`). */
  getDuration(): number {
    const clip = this.getCurrentClip()
    if (!clip || clip.frameCount <= 0 || !Number.isFinite(clip.frameCount)) return 0
    return clip.frameCount / FPS
  }

  getProgress(): AnimationProgress {
    const clip = this.getCurrentClip()
    const duration = clip && clip.frameCount > 0 ? clip.frameCount / FPS : 0
    const current = clip ? this.currentFrame / FPS : 0
    const percentage = duration > 0 ? (current / duration) * 100 : 0
    return {
      animationName: this.currentAnimationName,
      current,
      duration,
      percentage,
      looping: this.currentLoop,
      playing: this.isPlaying && !this.isPaused,
      paused: this.isPaused,
    }
  }

  getAnimationNames(): string[] {
    return Array.from(this.animations.keys())
  }

  hasAnimation(name: string): boolean {
    return this.animations.has(name)
  }

  show(name: string): void {
    if (!this.animations.has(name)) return
    this.currentAnimationName = name
    this.currentFrame = 0
    this.currentPriority = 0
    this.currentLoop = false
    this.isPlaying = false
    this.isPaused = false
    this.nextAnimation = null
  }

  setOnEnd(callback: ((animationName: string) => void) | null): void {
    this.onEnd = callback
  }

  getPlaying(): boolean {
    return this.isPlaying
  }

  getPaused(): boolean {
    return this.isPaused
  }
}

export function bezierInterpolate(x1: number, x2: number, y1: number, y2: number, t: number): number {
  t = Math.max(0, Math.min(1, t))

  let start = 0
  let end = 1
  let mid = 0.5

  for (let i = 0; i < 15; i++) {
    const x = 3 * (1 - mid) * (1 - mid) * mid * x1 + 3 * (1 - mid) * mid * mid * x2 + mid * mid * mid

    if (Math.abs(x - t) < 0.0001) {
      break
    }

    if (x < t) {
      start = mid
    } else {
      end = mid
    }

    mid = (start + end) / 2
  }

  const y = 3 * (1 - mid) * (1 - mid) * mid * y1 + 3 * (1 - mid) * mid * mid * y2 + mid * mid * mid

  return y
}

const INV_127 = 1 / 127

export function rawInterpolationToBoneInterpolation(raw: Uint8Array): BoneInterpolation {
  return {
    rotation: [
      { x: raw[0], y: raw[2] },
      { x: raw[1], y: raw[3] },
    ],
    translationX: [
      { x: raw[0], y: raw[4] },
      { x: raw[8], y: raw[12] },
    ],
    translationY: [
      { x: raw[16], y: raw[20] },
      { x: raw[24], y: raw[28] },
    ],
    translationZ: [
      { x: raw[32], y: raw[36] },
      { x: raw[40], y: raw[44] },
    ],
  }
}

export function interpolateControlPoints(cp: ControlPoint[], t: number): number {
  return bezierInterpolate(
    cp[0].x * INV_127,
    cp[1].x * INV_127,
    cp[0].y * INV_127,
    cp[1].y * INV_127,
    t
  )
}
