// Animation state machine: named states (clip loops or delegate pose producers),
// guarded transitions with crossfades. Sits on Model.setBlendPose like the
// LocomotionController does — a delegate state can BE a LocomotionController
// (constructed with autoApply: false) so gameplay states and locomotion mix freely.
import type { Model } from "./model"
import type { BlendEntry } from "./animation"
import { easeInOut } from "./math"

const FPS = 30

export interface AnimStateDef {
  /** Clip name previously loaded on the model. Mutually exclusive with `entries`. */
  clip?: string
  /** Loop the clip (default true). Non-loop states hold their last frame. */
  loop?: boolean
  /** Clip playback speed multiplier (default 1). */
  speed?: number
  /** Delegate pose producer — return this frame's blend entries (e.g. a
   *  LocomotionController with autoApply: false). Return null to contribute
   *  nothing (the pose relaxes toward rest per setBlendPose semantics). */
  entries?: (dt: number) => BlendEntry[] | null
  onEnter?: (from: string | null) => void
  onExit?: (to: string) => void
}

export interface AnimTransitionDef {
  /** Source state name, or "*" for any state (never fires into itself). */
  from: string | "*"
  to: string
  /** Condition, checked every update. Omitted = unconditional. */
  when?: () => boolean
  /** Fire only once the state has been active this many seconds. On a clip
   *  state with NO `when` and NO `exitTime`, the transition fires when the
   *  (non-looping) clip approaches its end — the "skill finished, back to
   *  locomotion" pattern. */
  exitTime?: number
  /** Crossfade seconds (default: machine's defaultFade). */
  fade?: number
}

export interface StateMachineOptions {
  initial: string
  /** Crossfade used when a transition does not specify one (default 0.25). */
  defaultFade?: number
}

interface ActiveState {
  name: string
  def: AnimStateDef
  time: number // seconds in state; doubles as the clip clock (pre-speed)
}

export class AnimationStateMachine {
  private readonly model: Model
  private readonly states: Record<string, AnimStateDef>
  private readonly transitions: AnimTransitionDef[]
  private readonly defaultFade: number

  private current: ActiveState
  /** Outgoing state during a crossfade — keeps playing while it fades. */
  private fading: { state: ActiveState; elapsed: number; duration: number } | null = null

  // Scratch: per-slot entry objects are reused so downstream caches (blend
  // cursors, clip-event trackers) keyed on entry identity stay warm.
  private readonly merged: BlendEntry[] = []

  constructor(model: Model, states: Record<string, AnimStateDef>, transitions: AnimTransitionDef[], options: StateMachineOptions) {
    this.model = model
    this.states = states
    this.transitions = transitions
    this.defaultFade = options.defaultFade ?? 0.25
    const def = states[options.initial]
    if (!def) throw new Error(`Unknown initial state "${options.initial}"`)
    this.current = { name: options.initial, def, time: 0 }
    def.onEnter?.(null)
  }

  get state(): string {
    return this.current.name
  }

  /** Seconds the current state has been active. */
  get stateTime(): number {
    return this.current.time
  }

  /** Force a transition now, regardless of the transition table. */
  go(to: string, fade?: number): void {
    this.begin(to, fade ?? this.defaultFade)
  }

  update(dt: number): void {
    this.current.time += dt

    // Transitions are not interruptible mid-fade (go() still is).
    if (this.fading === null) {
      for (const t of this.transitions) {
        if (t.to === this.current.name) continue
        if (t.from !== "*" && t.from !== this.current.name) continue
        if (!this.transitionReady(t)) continue
        this.begin(t.to, t.fade ?? this.defaultFade)
        break
      }
    }

    const inEntries = this.produce(this.current, dt, this.inScratch)

    if (this.fading !== null) {
      const f = this.fading
      f.elapsed += dt
      if (f.elapsed >= f.duration) {
        this.fading = null
      } else {
        f.state.time += dt
        const outEntries = this.produce(f.state, dt, this.outScratch)
        const w = easeInOut(f.elapsed / f.duration)
        this.apply(outEntries, 1 - w, inEntries, w)
        return
      }
    }
    this.apply(inEntries, 1, null, 0)
  }

  private transitionReady(t: AnimTransitionDef): boolean {
    const def = this.current.def
    if (t.exitTime !== undefined) {
      if (this.current.time < t.exitTime) return false
      return t.when ? t.when() : true
    }
    if (t.when) return t.when()
    // Unconditional, no exitTime: on a non-looping clip state this means
    // "when the clip is about to end" (start the fade so it lands at the end).
    if (def.clip && def.loop === false) {
      const dur = this.clipDuration(def.clip) / (def.speed ?? 1)
      return this.current.time >= Math.max(0, dur - (t.fade ?? this.defaultFade))
    }
    return true
  }

  private begin(to: string, fade: number): void {
    const def = this.states[to]
    if (!def) throw new Error(`Unknown state "${to}"`)
    this.current.def.onExit?.(to)
    // go() during an existing fade drops the older outgoing state (v1: no
    // three-way mixes) — the current state becomes the outgoing one.
    this.fading = fade > 0 ? { state: this.current, elapsed: 0, duration: fade } : null
    const from = this.current.name
    this.current = { name: to, def, time: 0 }
    def.onEnter?.(from)
  }

  /** This frame's entries for a state: clip states own a one-entry pose;
   *  delegate states produce their own. Returns null for "no pose". */
  private produce(state: ActiveState, dt: number, scratch: BlendEntry[]): BlendEntry[] | null {
    const def = state.def
    if (def.entries) return def.entries(dt)
    if (!def.clip) return null
    const dur = this.clipDuration(def.clip)
    let t = state.time * (def.speed ?? 1)
    if (dur > 0) t = def.loop === false ? Math.min(t, dur) : t % dur
    scratch[0].name = def.clip
    scratch[0].time = t
    scratch[0].weight = 1
    return scratch
  }

  private readonly inScratch: BlendEntry[] = [{ name: "", time: 0, weight: 1 }]
  private readonly outScratch: BlendEntry[] = [{ name: "", time: 0, weight: 1 }]

  /** Merge up to two entry lists scaled by their group weights into the stable
   *  scratch array and hand it to the model. */
  private apply(a: BlendEntry[] | null, wa: number, b: BlendEntry[] | null, wb: number): void {
    let n = 0
    const put = (src: BlendEntry[] | null, scale: number) => {
      if (src === null || scale <= 0) return
      for (const e of src) {
        if (!(e.weight > 1e-6)) continue
        let slot = this.merged[n]
        if (!slot) {
          slot = { name: "", time: 0, weight: 0 }
          this.merged[n] = slot
        }
        slot.name = e.name
        slot.time = e.time
        slot.weight = e.weight * scale
        n++
      }
    }
    // Order: incoming FIRST so slot identities stay stable for a given state
    // across the fade's start/end (cursor + event caches key on the objects).
    put(b, wb)
    put(a, wa)
    for (let i = n; i < this.merged.length; i++) this.merged[i].weight = 0
    if (n > 0) this.model.setBlendPose(this.merged)
  }

  private clipDuration(name: string): number {
    const frames = this.model.getClip(name)?.frameCount ?? 0
    return frames > 1 ? (frames - 1) / FPS : 0
  }
}
