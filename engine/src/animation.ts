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

/** One weighted contribution to a blended pose (see Model.setBlendPose). */
export interface BlendEntry {
  /** Name of a clip previously loaded with loadVmd/loadClip. */
  name: string
  /** Clip-local time in seconds. The caller owns the clock — wrap or clamp before passing. */
  time: number
  /** Relative weight >= 0. Entries are normalized over their sum; a sum below 1 is
   *  NOT renormalized — the remainder blends toward the rest pose. */
  weight: number
}

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

  /**
   * Replace a clip's morph tracks wholesale, leaving its bones alone.
   *
   * MMD authors ship expressions as their own file (表情モーション) beside the
   * body motion, and when they do it is AUTHORITATIVE: whatever morphs the body
   * motion carried are the ones the expression pass was made to replace. So
   * this overwrites rather than merges per-morph — a half-overridden face is
   * nobody's intent. With no expression file loaded, the motion's own morphs
   * play untouched, because nothing calls this.
   *
   * The clip is created if it does not exist yet, so the two files may arrive
   * in either order, and frameCount grows to cover the longer of the two — an
   * expression track running past the body motion is common and must not be
   * truncated to it.
   */
  setMorphTracks(name: string, morphTracks: Map<string, MorphKeyframe[]>, frameCount: number): void {
    const clip = this.animations.get(name)
    this.animations.set(name, {
      boneTracks: clip?.boneTracks ?? new Map(),
      morphTracks,
      ikTracks: clip?.ikTracks,
      frameCount: Math.max(clip?.frameCount ?? 0, frameCount),
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

  /** Make `name` current at frame 0 and play it, bypassing priority arbitration.
   *  Used by crossfadeTo, which owns the transition and must not be queued behind
   *  the clip it is fading away from. */
  forcePlay(name: string, loop: boolean): boolean {
    if (!this.animations.has(name)) return false
    this.currentAnimationName = name
    this.currentFrame = 0
    this.currentPriority = 0
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

/**
 * The 64 interpolation bytes of a VMD bone frame, as four bezier curves.
 *
 * One 16-byte record carries all four channels interleaved: byte `c` is channel
 * c's x1, `c + 4` its y1, `c + 8` its x2, `c + 12` its y2, for X = 0, Y = 1,
 * Z = 2, ROTATION = 3. MMD then writes that record four times over the 64 bytes,
 * each copy shifted one byte further left, so copy `r` starts at byte `r * 16`
 * and holds `record[r..15]` (the bytes past the record's end are uninitialised
 * junk — both motions in this repo have arbitrary values there).
 *
 * The shifted copies are not decoration, and reading channel c out of copy 0 is
 * the bug this function exists to not have. MMD reuses bytes 2 and 3 of the FIRST
 * copy for the per-keyframe physics toggle (`(raw[2] << 8) | raw[3]`; 0 = physics
 * on) — it overwrites Z's x1 and, more damagingly, ROTATION's x1 with zero, on
 * every keyframe of every real motion file. Rotation is what an MMD dance is
 * almost entirely made of, so an x1 pinned to 0 turns every eased keyframe
 * interval into a curve that leaps early and coasts: the dance arrives at each
 * pose ahead of the beat and sits there.
 *
 * So each channel is read out of its OWN copy, at `c * 16`, where the record's
 * byte c survives the physics field. This is what babylon-mmd does (vmdLoader.ts,
 * `boneKeyFrameInterpolation[c * 16 + {0, 8, 4, 12}]`) and the copies agree with
 * each other byte-for-byte in both motions here.
 */
export function rawInterpolationToBoneInterpolation(raw: Uint8Array): BoneInterpolation {
  const channel = (c: number): ControlPoint[] => {
    const b = c * 16
    return [
      { x: raw[b], y: raw[b + 4] },
      { x: raw[b + 8], y: raw[b + 12] },
    ]
  }
  return {
    translationX: channel(0),
    translationY: channel(1),
    translationZ: channel(2),
    rotation: channel(3),
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
