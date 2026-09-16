"use client"

import Header from "@/components/header"
import {
  AnimationStateMachine,
  easeInOut,
  Engine,
  type EngineStats,
  LocomotionController,
  Model,
  Quat,
  Vec3,
  type AnimStateDef,
  type MaterialPresetMap,
  type RootMotionProfile,
  type StopClipEntry,
} from "reze-engine"
import { useCallback, useEffect, useRef, useState } from "react"
import { Bomb } from "lucide-react"
import Loading from "@/components/loading"
import { ASSETS, CAST } from "@/lib/assets"

// One playable Reze: WASD or the wheel to run, Space or the button to dance,
// Shift or the bomb to turn into the Bomb Devil and back, double-click her for
// a reaction — and left alone, she entertains herself.

/** Her two looks: one skeleton, one motion set. The first is the one she loads as. */
const LOOKS = [
  { id: "reze", pmx: `${CAST}/reze.pmx` },
  { id: "reze-bomb", pmx: `${CAST}/reze-bomb.pmx` },
]
/** The materials the engine's name hints leave unstyled, grouped as reze-design
 *  groups them. Every other material on both looks resolves by its name. */
const CAST_STYLE: MaterialPresetMap = {
  cloth_smooth: ["bozi", "choker"],
  cloth_rough: ["Rubber", "Leather"],
}

const ANIMATIONS = `${ASSETS}/animations`

// Locomotion: a stand loop, a run loop, and the stop played when a run is
// released. The run and stop carry their authored root travel; loading lifts it
// off センター and the controller drives the root instead.
//
// Measured on reze.pmx: the run's planted feet sweep the ground at 53 u/s while
// its authored root travels 67.1 u/s (the retarget's hip ratio overshoots her
// stride). The root runs at the foot speed, which pins her feet, and the stop's
// authored skid shrinks by the same ratio.
const STAND_VMD = `${ANIMATIONS}/1017ui@ui_stand.vmd`
const RUN_VMD = `${ANIMATIONS}/Run_Lfoot (2).vmd`
const STOP_VMD = `${ANIMATIONS}/Run_Stop_Rfoot.vmd`
const RUN_SPEED = 53
const FOOT_MATCH = RUN_SPEED / 67.1
const STOP: StopClipEntry = {
  clip: "stop",
  // The skid settles here; the clip's remainder is the recovery into the stand.
  exitTime: 1.57,
  forward: [0, 8.3, 17.3, 23.4, 30, 35.2, 37.2, 39.2, 41.2, 42.1, 42].map((v) => v * FOOT_MATCH),
  gear: "run",
  foot: "R",
}

// Main-screen reactions, each `${ANIMATIONS}/<clip>.vmd`.
/** Her first motion once loaded: a jump in from the side, landing on the spawn point. */
const ENTRANCE_CLIP = "1034@main_assistant"
/** Follows the entrance unless something interrupts it. */
const LOGIN_CLIP = "1034@main_login"
/** Double-click on her plays one of these. */
const TOUCH_CLIPS = ["1034@main_touch1", "1034@main_touch2", "1034@main_touch3", "1034@main_touch4", "1034@main_quickclick"]
/** Left still for IDLE_AUTO_AFTER seconds, she plays one of these. */
const IDLE_CLIPS = ["1034@main_emotion", LOGIN_CLIP]
const IDLE_AUTO_AFTER = 3
const FLAVOR_CLIPS = [...new Set([ENTRANCE_CLIP, ...TOUCH_CLIPS, ...IDLE_CLIPS])]

const DANCE_CLIP = "dance"
const DANCE_VMD = `${ANIMATIONS}/IRIS OUT.vmd`
const DANCE_AUDIO = `${ASSETS}/audios/IRIS OUT.m4a`
/** Plays on every look swap, either way. The skin changes on the bang, which
 *  lands this far in, after the fuse. */
const BOOM_AUDIO = `${ASSETS}/audios/boom.m4a`
const BOOM_BANG = 0.32

// Action pacing. Reactions ease in and drift back to the stand slowly; the
// dance fades in with the music; fresh movement input cuts anything short.
const FLAVOR_FADE_IN = 0.3
const FLAVOR_RETURN_FADE = 0.7
const DANCE_FADE_IN = 0.4
const DANCE_RETURN_FADE = 0.6
const CANCEL_FADE = 0.25
/** Seconds the song takes to fall silent when the dance is cut short. */
const SONG_CUT_FADE = 0.15

// Warm-up before the entrance: it starts once this many frames in a row arrive
// within the budget (the first-use stalls are over), or at the cap regardless.
const WARMUP_STEADY_FRAMES = 20
const WARMUP_FRAME_MS = 40
const WARMUP_MAX_MS = 8000

/** The orbit centre sits this far above her root. */
const CAMERA_OFFSET = new Vec3(0, 11.5, 0)
/** Seconds the camera takes to move from the spawn onto her once the entrance hands over. */
const CAMERA_GLIDE = 0.4

const ACTIVE_BUTTON = "bg-white/90 text-black border-white scale-95 shadow-[0_0_28px_rgba(255,255,255,0.5)]"
const IDLE_BUTTON =
  "bg-white/25 text-white border-white/70 backdrop-blur-md shadow-[inset_0_1px_0_rgba(255,255,255,0.3),0_2px_12px_rgba(0,0,0,0.25)]"

// Input keys the demo cares about — everything else never touches state.
const MOVE_CODES = new Set(["KeyW", "KeyA", "KeyS", "KeyD"])
// Action-game two-speed rule: any real movement intent means a full run — the
// stand⊕run blend band is a ramp to pass through, never a place to dwell
// (the mid-band "slow run" reads as interpolation, not locomotion).
const MIN_MOVE = 0.8

/** What the machine is playing besides locomotion. */
type ActionKind = "flavor" | "dance" | null
type RootXZ = { x: number; z: number; yaw: number }
/** A loaded look. `ready` once it is styled, posed and warmed, and can be swapped in. */
type Look = { id: string; model: Model; ready: boolean }

/** The page's audio output: one context, opened by the first user gesture. */
class AudioOut {
  private ctx: AudioContext | null = null

  /** Create or resume the output. Called from user gestures, which is what browsers require. */
  unlock(): AudioContext {
    if (!this.ctx) this.ctx = new AudioContext({ latencyHint: "interactive" })
    if (this.ctx.state !== "running") this.ctx.resume().catch(() => {})
    return this.ctx
  }

  dispose(): void {
    void this.ctx?.close()
    this.ctx = null
  }
}

/**
 * One sound through Web Audio. It is decoded ahead of any press, and a press
 * starts a buffer source, which sounds within the output latency. A media
 * element can take a good fraction of a second to begin, and seeking it to
 * catch up stalls it again.
 */
class Track {
  private buffer: AudioBuffer | null = null
  private ctx: AudioContext | null = null
  private source: AudioBufferSourceNode | null = null
  private gain: GainNode | null = null
  /** Bumped by every play and stop, so a start still waiting on resume() can tell it was superseded. */
  private token = 0

  constructor(private readonly out: AudioOut) {}

  get ready(): boolean {
    return this.buffer !== null
  }

  async load(url: string): Promise<void> {
    const res = await fetch(url)
    if (!res.ok) throw new Error(`Failed to fetch ${url}: ${res.status}`)
    // Decoding needs a context but not a running one. An offline context keeps
    // the real one from being created before a gesture allows it to start.
    this.buffer = await new OfflineAudioContext(2, 1, 48000).decodeAudioData(await res.arrayBuffer())
  }

  /**
   * Play from the top, restarting if already playing. With `at`, the track
   * follows that clock (the motion is the master): it starts at `at()` seconds
   * plus the output latency, read when sound can actually begin, so what is
   * heard lines up with what is seen. `onHeard` receives the performance.now()
   * time at which the start of the track reaches the speakers.
   */
  play({ at, onHeard }: { at?: () => number; onHeard?: (time: number) => void } = {}): void {
    const ctx = this.out.unlock()
    const token = ++this.token
    const start = () => {
      if (token !== this.token || !this.buffer) return
      this.silence(0)
      const gain = ctx.createGain()
      gain.connect(ctx.destination)
      const source = ctx.createBufferSource()
      source.buffer = this.buffer
      source.connect(gain)
      source.onended = () => gain.disconnect()
      const latency = (ctx.baseLatency || 0) + (ctx.outputLatency || 0)
      const offset = at ? at() + latency : 0
      source.start(0, Math.min(this.buffer.duration, Math.max(0, offset)))
      this.ctx = ctx
      this.source = source
      this.gain = gain
      if (onHeard) {
        // The output timestamp pairs a context time with when it is audible;
        // where a browser does not fill it in, the reported latency stands in.
        const ts = ctx.getOutputTimestamp?.()
        onHeard(
          ts?.performanceTime && ts.contextTime !== undefined
            ? ts.performanceTime + (ctx.currentTime - ts.contextTime) * 1000
            : performance.now() + latency * 1000
        )
      }
    }
    if (ctx.state === "running") start()
    else ctx.resume().then(start, () => {})
  }

  /** Stop, easing out over `fade` seconds. */
  stop(fade: number): void {
    this.token++
    this.silence(fade)
  }

  private silence(fade: number): void {
    const { ctx, source, gain } = this
    this.source = null
    this.gain = null
    if (!ctx || !source || !gain) return
    if (fade > 0) {
      gain.gain.setTargetAtTime(0, ctx.currentTime, fade / 3)
      source.stop(ctx.currentTime + fade)
    } else {
      source.stop()
    }
  }
}

/**
 * Every look's state machine, driven in lockstep. The hidden look always wears
 * the live pose, mid-crossfade included, so a swap is a visibility flip. The
 * first machine answers the reads.
 */
class Lockstep {
  private readonly machines: AnimationStateMachine[]

  constructor(lead: AnimationStateMachine) {
    this.machines = [lead]
  }

  get state(): string {
    return this.machines[0].state
  }

  get stateTime(): number {
    return this.machines[0].stateTime
  }

  go(to: string, fade?: number, atTime?: number): void {
    for (const m of this.machines) m.go(to, fade, atTime)
  }

  update(dt: number): void {
    for (const m of this.machines) m.update(dt)
  }

  /** Join at the lead's state and clock. A fade in progress is not carried —
   *  a joining look is hidden until well after it would have ended. */
  join(machine: AnimationStateMachine): void {
    machine.go(this.state, 0, this.stateTime)
    this.machines.push(machine)
  }
}

/** One look's animation brain: locomotion is a delegate state (the shared
 *  controller's blend); every action clip is a non-loop state that plays out in
 *  full and drifts back to the stand. */
function buildMachine(model: Model, ctl: LocomotionController): AnimationStateMachine {
  const states: Record<string, AnimStateDef> = {
    locomotion: { entries: () => ctl.getBlendEntries() },
    [DANCE_CLIP]: { clip: DANCE_CLIP, loop: false },
  }
  for (const clip of FLAVOR_CLIPS) states[clip] = { clip, loop: false }
  return new AnimationStateMachine(
    model,
    states,
    [
      { from: DANCE_CLIP, to: "locomotion", fade: DANCE_RETURN_FADE },
      ...FLAVOR_CLIPS.map((clip) => ({ from: clip, to: "locomotion", fade: FLAVOR_RETURN_FADE })),
    ],
    { initial: "locomotion", defaultFade: 0.22 }
  )
}

/** The stand clip's neutral センター offset. Every clip's horizontal センター
 *  flattens to it when its travel is lifted onto the root, so fades between
 *  clips blend without a micro-slide. */
function standRest(model: Model): { x: number; z: number } {
  const key = model.getClip("stand")?.boneTracks.get("センター")?.[0]?.translation
  return { x: key?.x ?? 0, z: key?.z ?? 0 }
}

/** Load action clips and lift their authored root travel into `profiles`, so
 *  the host moves the MODEL ROOT along it (the camera follows her, and the next
 *  state starts wherever the clip actually went). `onLoaded` hears each clip
 *  that lands; one that fails to load is skipped and the others still play. */
async function loadActions(
  model: Model,
  clips: string[],
  urlOf: (clip: string) => string,
  rest: { x: number; z: number },
  profiles: Map<string, RootMotionProfile>,
  onLoaded: (clip: string) => void
) {
  await Promise.all(
    clips.map(async (clip) => {
      try {
        await model.loadVmd(clip, urlOf(clip))
      } catch (error) {
        console.warn(`[reze] ${clip} not loaded:`, error)
        return
      }
      const profile = model.extractRootMotion(clip, rest)
      if (profile) profiles.set(clip, profile)
      onLoaded(clip)
    })
  )
}

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
const wait = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms))

/** Resolves once `count` frames in a row each arrive within `budgetMs` of the
 *  one before, or at `deadline` (performance.now() time), whichever is first. */
function steadyFrames(count: number, budgetMs: number, deadline: number): Promise<void> {
  return new Promise((resolve) => {
    let last = performance.now()
    let run = 0
    const frame = (now: number) => {
      run = now - last <= budgetMs ? run + 1 : 0
      last = now
      if (run >= count || now >= deadline) resolve()
      else requestAnimationFrame(frame)
    }
    requestAnimationFrame(frame)
  })
}

/** A clip-space offset (rest faces -Z) in world space, for a root heading `yaw`. */
function clipToWorld(dx: number, dz: number, yaw: number): { x: number; z: number } {
  const theta = yaw + Math.PI
  const cos = Math.cos(theta)
  const sin = Math.sin(theta)
  return { x: dx * cos + dz * sin, z: -dx * sin + dz * cos }
}

/** A profile's horizontal offset at fractional frame f, linearly interpolated. */
function sampleProfile(p: RootMotionProfile, f: number): { dx: number; dz: number } {
  const n = p.frames.length
  if (f <= p.frames[0]) return { dx: p.x[0], dz: p.z[0] }
  if (f >= p.frames[n - 1]) return { dx: p.x[n - 1], dz: p.z[n - 1] }
  let lo = 0
  let hi = n - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (p.frames[mid] <= f) lo = mid
    else hi = mid
  }
  const t = (f - p.frames[lo]) / (p.frames[hi] - p.frames[lo])
  return { dx: p.x[lo] + (p.x[hi] - p.x[lo]) * t, dz: p.z[lo] + (p.z[hi] - p.z[lo]) * t }
}

/** Mobile-game movement wheel. Reports {x, y (up = +forward), active} through a ref
 *  callback; the knob is moved via direct DOM transform so dragging never re-renders.
 *  `display` receives a setter the host calls to mirror OTHER input (keyboard WASD)
 *  on the knob — ignored while a drag owns it. */
function VirtualStick({
  onChange,
  display,
}: {
  onChange: (x: number, y: number, active: boolean) => void
  display: React.RefObject<((x: number, y: number) => void) | null>
}) {
  const baseRef = useRef<HTMLDivElement>(null)
  const knobRef = useRef<HTMLDivElement>(null)
  const pointerId = useRef<number | null>(null)

  /** Travel radius follows the RENDERED wheel size (responsive classes), kept
   *  near the base radius so the knob overhangs the rim at full deflection. */
  const travel = () => {
    const base = baseRef.current
    return base ? base.getBoundingClientRect().width / 2 - 6 : 66
  }

  useEffect(() => {
    display.current = (x, y) => {
      if (pointerId.current !== null || !knobRef.current) return
      const len = Math.hypot(x, y)
      if (len > 1) {
        x /= len
        y /= len
      }
      const r = travel()
      knobRef.current.style.transform = `translate(${x * r}px, ${-y * r}px)`
    }
    return () => {
      display.current = null
    }
  }, [display])

  const move = (clientX: number, clientY: number) => {
    const base = baseRef.current
    const knob = knobRef.current
    if (!base || !knob) return
    const rect = base.getBoundingClientRect()
    const r = rect.width / 2 - 6
    let dx = clientX - (rect.left + rect.width / 2)
    let dy = clientY - (rect.top + rect.height / 2)
    const len = Math.hypot(dx, dy)
    if (len > r) {
      dx *= r / len
      dy *= r / len
    }
    knob.style.transitionDuration = "0ms" // dragging tracks the finger 1:1
    knob.style.transform = `translate(${dx}px, ${dy}px)`
    onChange(dx / r, -dy / r, true)
  }

  const release = () => {
    pointerId.current = null
    const knob = knobRef.current
    if (knob) {
      knob.style.transitionDuration = "" // class easing back on → animated spring-return
      knob.style.transform = "translate(0px, 0px)"
    }
    onChange(0, 0, false)
  }

  return (
    <div
      ref={baseRef}
      className="relative w-28 h-28 sm:w-36 sm:h-36 rounded-full border-2 border-white/70 bg-white/15 backdrop-blur-[2px] touch-none select-none"
      onPointerDown={(e) => {
        e.preventDefault()
        e.currentTarget.setPointerCapture(e.pointerId)
        pointerId.current = e.pointerId
        move(e.clientX, e.clientY)
      }}
      onPointerMove={(e) => {
        if (pointerId.current === e.pointerId) move(e.clientX, e.clientY)
      }}
      onPointerUp={release}
      onPointerCancel={release}
      onContextMenu={(e) => e.preventDefault()}
    >
      {/* inner ring, like the classic wheel */}
      <div className="absolute inset-0 m-auto w-12 h-12 sm:w-16 sm:h-16 rounded-full border border-white/50 pointer-events-none" />
      {/* Eased by default so keyboard pushes glide to their direction and releases
          spring back; dragging zeroes the duration inline for 1:1 tracking. */}
      <div
        ref={knobRef}
        className="absolute inset-0 m-auto w-10 h-10 sm:w-12 sm:h-12 rounded-full bg-white/70 border-2 border-white shadow-[0_0_14px_rgba(255,255,255,0.45)] pointer-events-none transition-transform duration-150 ease-out"
      />
    </div>
  )
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const keysRef = useRef<Set<string>>(new Set())
  const stickRef = useRef({ x: 0, y: 0, active: false })
  const stickDisplayRef = useRef<((x: number, y: number) => void) | null>(null)
  /** Loaded looks, in LOOKS order; the active one is `lookRef`. */
  const looksRef = useRef<Look[]>([])
  const lookRef = useRef(0)
  /** The active look's model. Every look holds the same clips. */
  const modelRef = useRef<Model | null>(null)
  const ctlRef = useRef<LocomotionController | null>(null)
  const machineRef = useRef<Lockstep | null>(null)
  /** Whether the camera rides her root yet (it holds the spawn through the entrance). */
  const followingRef = useRef(false)
  /** performance.now() time of a pending look swap, or -1. */
  const swapAtRef = useRef(-1)
  const actionRef = useRef<ActionKind>(null)
  /** Where she IS this frame (whichever system owns the root). */
  const rootRef = useRef<RootXZ>({ x: 0, z: 0, yaw: Math.PI })
  /** Root anchor of the active action clip: the clip's profile plays out from here. */
  const anchorRef = useRef<RootXZ>({ x: 0, z: 0, yaw: Math.PI })
  /** The OUTGOING root path during an action crossfade: the previous clip's
   *  profile keeps advancing under its fading pose (or, entering from
   *  locomotion, the controller's decelerating position), blended with the same
   *  easeInOut the machine uses for the pose, so root and pose agree. */
  const fadeFromRef = useRef<{
    clip: string | null // null = came from locomotion (follow the controller)
    anchor: RootXZ
    stateTime: number
    start: number
    fade: number
  } | null>(null)
  const profilesRef = useRef(new Map<string, RootMotionProfile>())
  const lastFlavorRef = useRef<string | null>(null)
  const lastActiveRef = useRef(0)
  const lastSpeedRef = useRef(0)
  const [audio] = useState(() => {
    const out = new AudioOut()
    return { out, song: new Track(out), boom: new Track(out) }
  })
  const [dancing, setDancing] = useState(false)
  const [danceReady, setDanceReady] = useState(false)
  const [spaceHeld, setSpaceHeld] = useState(false)
  const [bombed, setBombed] = useState(false)
  const [bombReady, setBombReady] = useState(false)
  const [swapPending, setSwapPending] = useState(false)
  const [shiftHeld, setShiftHeld] = useState(false)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<EngineStats | null>(null)

  // The first gesture anywhere opens the audio output, so a later press starts
  // sound at once instead of waiting on the context to come up.
  useEffect(() => {
    const unlock = () => audio.out.unlock()
    const events = ["pointerdown", "keydown", "touchend"] as const
    for (const e of events) window.addEventListener(e, unlock, { once: true })
    return () => {
      for (const e of events) window.removeEventListener(e, unlock)
      audio.song.stop(0)
      audio.boom.stop(0)
      audio.out.dispose()
    }
  }, [audio])

  /** Enter an action state: anchor its root path where she currently is, and
   *  remember the outgoing path so the crossfade blends roots, not just poses. */
  const goAction = useCallback((clip: string, kind: Exclude<ActionKind, null>, fade: number): boolean => {
    const machine = machineRef.current
    if (!machine || !modelRef.current?.getClip(clip)) return false
    fadeFromRef.current = {
      clip: machine.state === "locomotion" ? null : machine.state,
      anchor: { ...anchorRef.current },
      stateTime: machine.stateTime,
      start: performance.now(),
      fade,
    }
    anchorRef.current = { ...rootRef.current }
    machine.go(clip, fade)
    actionRef.current = kind
    lastActiveRef.current = performance.now()
    return true
  }, [])

  /** Break out of whatever is playing — fresh movement input owns her. The
   *  music stops with the dance, not after its fade. */
  const cancelAction = useCallback(() => {
    const machine = machineRef.current
    if (!machine || machine.state === "locomotion") return
    if (actionRef.current === "dance") audio.song.stop(SONG_CUT_FADE)
    machine.go("locomotion", CANCEL_FADE) // the tick's reconcile hands the root back
  }, [audio])

  /** Space / the button: start the dance with the song from the top, or stop it. */
  const toggleDance = useCallback(() => {
    if (actionRef.current === "dance") {
      cancelAction()
      return
    }
    if (!goAction(DANCE_CLIP, "dance", DANCE_FADE_IN)) return
    audio.song.play({
      at: () => {
        const machine = machineRef.current
        return machine?.state === DANCE_CLIP ? machine.stateTime : 0
      },
    })
    setDancing(true)
  }, [goAction, cancelAction, audio])

  /** Flip to the other look. Whatever she is doing carries straight on — the
   *  hidden look already wears the pose. */
  const swapLook = useCallback(() => {
    const engine = engineRef.current
    const looks = looksRef.current
    const from = looks[lookRef.current]
    const next = (lookRef.current + 1) % LOOKS.length
    const to = looks[next]
    if (!engine || !from || !to) return
    const root = rootRef.current
    const half = (root.yaw + Math.PI) * 0.5
    engine.setModelTransform(to.id, {
      position: new Vec3(root.x, 0, root.z),
      rotation: new Quat(0, Math.sin(half), 0, Math.cos(half)),
      visible: true,
    })
    engine.setModelTransform(from.id, { visible: false })
    // Hidden looks skip physics: the incoming hair and cloth start settled on
    // the current pose rather than wherever they were left.
    engine.resetPhysics()
    if (followingRef.current) engine.setCameraFollow(to.model, undefined, CAMERA_OFFSET)
    lookRef.current = next
    modelRef.current = to.model
    setBombed(next === 1)
    setSwapPending(false)
  }, [])

  /** Shift / the bomb button: the boom starts now and the skin changes on its
   *  bang. Presses while a swap is pending are ignored. */
  const pressLook = useCallback(() => {
    const to = looksRef.current[(lookRef.current + 1) % LOOKS.length]
    if (!machineRef.current || !to?.ready || swapAtRef.current >= 0) return
    if (!audio.boom.ready) {
      swapLook()
      return
    }
    // Held to a second in case the sound never starts (a refused resume).
    swapAtRef.current = performance.now() + 1000
    audio.boom.play({ onHeard: (time) => (swapAtRef.current = time + BOOM_BANG * 1000) })
    setSwapPending(true)
  }, [audio, swapLook])

  /** A random clip from `pool` — never the one she just played, when there is a choice. */
  const startFlavor = useCallback(
    (pool: string[]) => {
      const model = modelRef.current
      if (!model) return
      const loaded = pool.filter((clip) => model.getClip(clip))
      const fresh = loaded.filter((clip) => clip !== lastFlavorRef.current)
      const choices = fresh.length > 0 ? fresh : loaded
      if (choices.length === 0) return
      const clip = choices[Math.floor(Math.random() * choices.length)]
      if (goAction(clip, "flavor", FLAVOR_FADE_IN)) lastFlavorRef.current = clip
    },
    [goAction]
  )

  const onStick = useCallback(
    (x: number, y: number, active: boolean) => {
      stickRef.current.x = x
      stickRef.current.y = y
      stickRef.current.active = active
      // Grabbing the wheel is fresh movement intent — breaks out of any action.
      if (active) cancelAction()
    },
    [cancelAction]
  )

  const initEngine = useCallback(async () => {
    if (!canvasRef.current) {
      setLoading(false)
      return
    }
    try {
      const engine = new Engine(canvasRef.current, {
        camera: { distance: 33, target: new Vec3(0, CAMERA_OFFSET.y, 0) },
        bloom: { color: new Vec3(0.75, 0.82, 1.0) },
        // reze-design's sun: azimuth 205°, elevation 21° (azElToDirection), strength 2.
        sun: { strength: 2.0, direction: new Vec3(0.3946, -0.3584, 0.8462) },
        // tailwind blue-200, display-space sRGB
        background: new Vec3(0.749, 0.859, 0.996),
        // Double-click (desktop) / tap (touch) on her → a touch reaction. Only
        // from rest: mid-run and mid-dance taps are ignored. Tapping during a
        // reaction rolls a new one.
        onRaycast: (modelName) => {
          if (modelName !== looksRef.current[lookRef.current]?.id) return
          const kind = actionRef.current
          if ((kind === null && lastSpeedRef.current < 0.2) || kind === "flavor") startFlavor(TOUCH_CLIPS)
        },
      })
      engineRef.current = engine
      await engine.init()
      engine.setOutlineEnabled(true)
      // Perf readout lives in the header's FPS pill (click it). Dev builds keep
      // a console handle for live probing; production exposes nothing.
      if (process.env.NODE_ENV === "development") (window as unknown as { engine?: Engine }).engine = engine

      // Stage first: ground up and the render loop painting before any model or
      // VMD bytes arrive — she pops in styled once ready.
      engine.addGround({
        // tailwind blue-400 in linear light
        diffuseColor: new Vec3(0.116, 0.384, 0.956),
        gridLineColor: new Vec3(0.95, 0.96, 1.0),
        gridLineOpacity: 0.5,
        noiseStrength: 0.02,
        opacity: 1,
        // Kept modest: the far grid aliases into a shimmering band at horizon
        // distances. The fade starts early and ramps long so the ground melts
        // into the backdrop instead of meeting it at a visible horizon line.
        width: 800,
        height: 800,
        fadeStart: 120,
        fadeEnd: 300,
      })

      const actionRotation = new Quat(0, 0, 0, 1)
      const actionPosition = new Vec3(0, 0, 0)
      const cameraTarget = new Vec3(0, CAMERA_OFFSET.y, 0)
      let cameraGlide = 0
      /** Entrance time at which the login takes over, or -1 once that is moot. */
      let loginAt = -1
      let last = performance.now()
      const gameTick = () => {
        // The clock ticks from the first frame, so the first armed frame's dt is
        // one frame rather than the whole load.
        const now = performance.now()
        const dt = (now - last) / 1000
        last = now
        const ctl = ctlRef.current
        const machine = machineRef.current
        const lookId = looksRef.current[lookRef.current]?.id
        if (!ctl || !machine || !lookId) return

        // Camera-relative controls: the mouse orbits the view and thereby steers
        // the run — up is always away from the camera, right is screen-right.
        const keys = keysRef.current
        const stick = stickRef.current
        let rawX: number
        let rawY: number
        if (stick.active) {
          rawX = stick.x
          rawY = stick.y
        } else {
          const kx = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0)
          const ky = (keys.has("KeyW") ? 1 : 0) - (keys.has("KeyS") ? 1 : 0)
          const len = Math.hypot(kx, ky)
          rawX = len > 0 ? kx / len : 0
          rawY = len > 0 ? ky / len : 0
          stickDisplayRef.current?.(rawX, rawY)
        }
        // Orbit eye sits at target + r·(sinα, ·, cosα), so screen-forward is
        // (-sinα, -cosα) and screen-right is (-cosα, sinα).
        const alpha = engine.getCameraAlpha()
        const sinA = Math.sin(alpha)
        const cosA = Math.cos(alpha)
        let x = rawX * -cosA + rawY * -sinA
        let y = rawX * sinA + rawY * -cosA
        const mag = Math.hypot(x, y)
        if (mag > 0.05 && mag < MIN_MOVE) {
          const k = MIN_MOVE / mag
          x *= k
          y *= k
        }

        // Reconcile: the machine came back to locomotion (clip end, or a
        // cancel) — hand the root, wherever the action carried it, back to the
        // controller in the same frame locomotion resumes.
        if (machine.state === "locomotion" && actionRef.current !== null) {
          const root = rootRef.current
          ctl.reset(root.x, 0, root.z, root.yaw)
          fadeFromRef.current = null
          if (actionRef.current === "dance") {
            // A cancel already stopped the song; at the clip's end it eases out
            // with the return to the stand.
            audio.song.stop(DANCE_RETURN_FADE)
            setDancing(false)
          }
          actionRef.current = null
          lastActiveRef.current = now
        }

        // The controller ticks every frame — with no input while an action owns
        // the root, so momentum settles under it and held keys resume after.
        const inLocomotion = machine.state === "locomotion"
        ctl.setMove(inLocomotion ? x : 0, inLocomotion ? y : 0)
        const pose = ctl.update(dt)

        if (inLocomotion) {
          rootRef.current.x = pose.position.x
          rootRef.current.z = pose.position.z
          rootRef.current.yaw = pose.yaw
          engine.setModelTransform(lookId, { position: pose.position, rotation: pose.rotation })
        } else {
          // The action owns the root: play the clip's authored path from its
          // anchor, facing frozen.
          const pathAt = (clip: string, a: RootXZ, time: number) => {
            const profile = profilesRef.current.get(clip)
            if (!profile) return { x: a.x, z: a.z }
            const { dx, dz } = sampleProfile(profile, time * 30)
            const d = clipToWorld(dx, dz, a.yaw)
            return { x: a.x + d.x, z: a.z + d.z }
          }
          const anchor = anchorRef.current
          let { x: px, z: pz } = pathAt(machine.state, anchor, machine.stateTime)
          const from = fadeFromRef.current
          if (from) {
            const elapsed = (now - from.start) / 1000
            if (elapsed >= from.fade) {
              fadeFromRef.current = null
            } else {
              const old =
                from.clip === null
                  ? { x: pose.position.x, z: pose.position.z }
                  : pathAt(from.clip, from.anchor, from.stateTime + elapsed)
              const w = easeInOut(elapsed / from.fade)
              px = old.x + (px - old.x) * w
              pz = old.z + (pz - old.z) * w
            }
          }
          rootRef.current.x = px
          rootRef.current.z = pz
          rootRef.current.yaw = anchor.yaw
          const half = (anchor.yaw + Math.PI) * 0.5
          actionRotation.setXYZW(0, Math.sin(half), 0, Math.cos(half))
          actionPosition.setXYZ(px, 0, pz)
          engine.setModelTransform(lookId, { position: actionPosition, rotation: actionRotation })
        }
        lastSpeedRef.current = pose.speedLevel

        // The camera holds the spawn (the origin) through the entrance, so she
        // jumps into frame. Once locomotion has her, it glides onto her root and
        // follows it — the gait clips don't animate 全ての親, so the camera tracks
        // the run without inheriting センター's bob and lean.
        if (!followingRef.current && inLocomotion) {
          cameraGlide = Math.min(1, cameraGlide + dt / CAMERA_GLIDE)
          if (cameraGlide >= 1) {
            engine.setCameraFollow(modelRef.current, undefined, CAMERA_OFFSET)
            followingRef.current = true
          } else {
            const w = easeInOut(cameraGlide)
            cameraTarget.setXYZ(rootRef.current.x * w, CAMERA_OFFSET.y, rootRef.current.z * w)
            engine.setCameraTarget(cameraTarget)
          }
        }

        // A pending look swap lands on the frame that is on screen when the
        // bang is heard: this tick's changes show on the next frame.
        if (swapAtRef.current >= 0 && now + dt * 1000 >= swapAtRef.current) {
          swapAtRef.current = -1
          swapLook()
        }

        // The login follows the entrance, entering where the entrance would
        // have started fading back to the stand. Anything that takes her out
        // of the entrance first (movement, a touch, the dance) calls it off.
        if (loginAt >= 0) {
          if (machine.state !== ENTRANCE_CLIP) {
            loginAt = -1
          } else if (machine.stateTime + dt >= loginAt) {
            loginAt = -1
            if (goAction(LOGIN_CLIP, "flavor", FLAVOR_RETURN_FADE)) lastFlavorRef.current = LOGIN_CLIP
          }
        }

        machine.update(dt)

        // Left truly alone for a while, she entertains herself.
        if (mag > 0.05 || stick.active || pose.speedLevel > 0.05 || actionRef.current !== null) {
          lastActiveRef.current = now
        } else if ((now - lastActiveRef.current) / 1000 > IDLE_AUTO_AFTER) {
          lastActiveRef.current = now
          startFlavor(IDLE_CLIPS)
        }

        setStats(engine.getStats())
      }
      engine.runRenderLoop(gameTick)

      const lead = LOOKS[0]
      const model = await engine.loadModel(lead.id, lead.pmx)
      engine.setModelTransform(lead.id, { visible: false })
      await engine.autoStyleGroups(lead.id, CAST_STYLE)
      looksRef.current = [{ id: lead.id, model, ready: true }]
      modelRef.current = model

      const [, , , hasEntrance] = await Promise.all([
        model.loadVmd("stand", STAND_VMD),
        model.loadVmd("run", RUN_VMD),
        model.loadVmd("stop", STOP_VMD),
        model.loadVmd(ENTRANCE_CLIP, `${ANIMATIONS}/${ENTRANCE_CLIP}.vmd`).then(
          () => true,
          (error) => {
            console.warn(`[reze] ${ENTRANCE_CLIP} not loaded:`, error)
            return false
          }
        ),
      ])
      const rest = standRest(model)
      model.extractRootMotion("run", rest)
      model.extractRootMotion("stop", rest)
      const entrance = hasEntrance ? model.extractRootMotion(ENTRANCE_CLIP, rest) : null
      if (entrance) profilesRef.current.set(ENTRANCE_CLIP, entrance)

      // Clips are parsed once, on the lead model, and shared with every other
      // look — same skeleton, and root motion already lifted.
      const loadedClips = new Set(["stand", "run", "stop", ...(entrance ? [ENTRANCE_CLIP] : [])])
      const shareClip = (clip: string) => {
        loadedClips.add(clip)
        const data = model.getClip(clip)
        if (!data) return
        for (const look of looksRef.current) if (look.model !== model) look.model.loadClip(clip, data)
      }

      const ctl = new LocomotionController(
        model,
        { idle: "stand", run: "run", stop: [STOP] },
        // The stop plays at its authored pace. autoApply off — the state
        // machines own the final blend. The controller reads only clip lengths
        // from its model, so every look shares it.
        { runSpeed: RUN_SPEED, stopTimeScale: 1, autoApply: false }
      )
      const machine = new Lockstep(buildMachine(model, ctl))
      // Spawn at the origin facing the camera (rest facing, -Z); she turns around
      // on the first input.
      ctl.teleport(0, 0, 0, Math.PI)
      const pose0 = ctl.update(1 / 60)

      // Warm-up, behind the loading bar. She is drawn at the spawn fully
      // dissolved, so every pass does its first-use work on her without a pixel
      // showing, holding her first frame (the entrance, or the stand) while the
      // physics settles on it. The dance, the reactions, the sounds and the
      // Bomb Devil load meanwhile, so their parse stalls (tens of ms each) land
      // here instead of mid-jump. The show starts once the frame pace holds
      // steady.
      if (entrance) machine.go(ENTRANCE_CLIP, 0)
      machine.update(0)
      engine.setModelDissolve(lead.id, 0)
      engine.setModelTransform(lead.id, { position: pose0.position, rotation: pose0.rotation, visible: true })
      await nextFrame()
      engine.resetPhysics()

      // The other look: its own model and machine over the shared clips and
      // controller, warmed the same way, then kept resident and hidden so a
      // swap never loads anything.
      const prepareLook = async (def: (typeof LOOKS)[number]) => {
        const other = await engine.loadModel(def.id, def.pmx)
        engine.setModelTransform(def.id, { visible: false })
        await engine.autoStyleGroups(def.id, CAST_STYLE)
        // Registered and caught up in one synchronous step, so a clip landing
        // in between cannot be missed.
        const look: Look = { id: def.id, model: other, ready: false }
        looksRef.current.push(look)
        for (const clip of loadedClips) {
          const data = model.getClip(clip)
          if (data) other.loadClip(clip, data)
        }
        machine.join(buildMachine(other, ctl))
        engine.setModelDissolve(def.id, 0)
        engine.setModelTransform(def.id, {
          position: new Vec3(rootRef.current.x, 0, rootRef.current.z),
          visible: true,
        })
        await steadyFrames(WARMUP_STEADY_FRAMES, WARMUP_FRAME_MS, performance.now() + WARMUP_MAX_MS)
        engine.setModelTransform(def.id, { visible: false })
        engine.setModelDissolve(def.id, 1)
        look.ready = true
        setBombReady(true)
      }

      const loads = Promise.all([
        (async () => {
          await Promise.all([
            loadActions(model, [DANCE_CLIP], () => DANCE_VMD, rest, profilesRef.current, shareClip),
            audio.song.load(DANCE_AUDIO).catch((error) => console.warn("[reze] song not loaded:", error)),
          ])
          setDanceReady(model.getClip(DANCE_CLIP) !== null)
          const pending = FLAVOR_CLIPS.filter((clip) => !model.getClip(clip)) // the entrance is already in
          await loadActions(model, pending, (clip) => `${ANIMATIONS}/${clip}.vmd`, rest, profilesRef.current, shareClip)
        })(),
        audio.boom.load(BOOM_AUDIO).catch((error) => console.warn("[reze] boom not loaded:", error)),
        prepareLook(LOOKS[1]).catch((error) => console.warn(`[reze] ${LOOKS[1].id} not loaded:`, error)),
      ])
      const deadline = performance.now() + WARMUP_MAX_MS
      await Promise.race([
        loads.then(() => steadyFrames(WARMUP_STEADY_FRAMES, WARMUP_FRAME_MS, deadline)),
        wait(WARMUP_MAX_MS),
      ])

      // Showtime: the tick takes over and she materializes into the entrance's
      // first frame. Physics is model-space, so moving her root to the jump's
      // start leaves the settled hair where it is.
      ctlRef.current = ctl
      machineRef.current = machine
      if (entrance) {
        // The jump starts wherever its own travel, played forward, lands her on
        // the spawn.
        const n = entrance.frames.length
        const land = clipToWorld(entrance.x[n - 1], entrance.z[n - 1], Math.PI)
        rootRef.current = { x: -land.x, z: -land.z, yaw: Math.PI }
        engine.setModelTransform(lead.id, { position: new Vec3(-land.x, 0, -land.z) })
        goAction(ENTRANCE_CLIP, "flavor", 0)
        // The machine's own return to the stand starts FLAVOR_RETURN_FADE before
        // the clip's end (its duration is (frameCount - 1) / 30).
        const frames = model.getClip(ENTRANCE_CLIP)?.frameCount ?? 0
        loginAt = Math.max(0, (frames - 1) / 30 - FLAVOR_RETURN_FADE)
      }
      engine.setModelDissolve(lead.id, 1)
      lastActiveRef.current = performance.now()
      setEngineError(null)
      setLoading(false)
    } catch (error) {
      setEngineError(error instanceof Error ? error.message : "Unknown error")
      setLoading(false)
    }
  }, [goAction, startFlavor, swapLook, audio])

  useEffect(() => {
    void initEngine()
    return () => {
      engineRef.current?.dispose()
    }
  }, [initEngine])

  // WASD movement, Space dance, Shift look swap. Blur clears everything so keys
  // can't stick when the tab loses focus mid-press.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.code === "Space") {
        e.preventDefault() // page scroll
        if (!e.repeat) {
          setSpaceHeld(true)
          toggleDance()
        }
        return
      }
      if (e.code === "ShiftLeft" || e.code === "ShiftRight") {
        if (!e.repeat) {
          setShiftHeld(true)
          pressLook()
        }
        return
      }
      if (MOVE_CODES.has(e.code)) {
        keysRef.current.add(e.code)
        // A fresh press (not a key held from before the action) breaks out of
        // whatever is playing — movement always wins.
        if (!e.repeat) cancelAction()
      }
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        setSpaceHeld(false)
        return
      }
      if (e.code === "ShiftLeft" || e.code === "ShiftRight") {
        setShiftHeld(false)
        return
      }
      keysRef.current.delete(e.code)
    }
    const blur = () => {
      keysRef.current.clear()
      setSpaceHeld(false)
      setShiftHeld(false)
    }
    window.addEventListener("keydown", down)
    window.addEventListener("keyup", up)
    window.addEventListener("blur", blur)
    return () => {
      window.removeEventListener("keydown", down)
      window.removeEventListener("keyup", up)
      window.removeEventListener("blur", blur)
    }
  }, [toggleDance, pressLook, cancelAction])

  return (
    <div className="fixed inset-0 w-full h-full overflow-hidden touch-none">
      <Header stats={stats} engineRef={engineRef} />

      {engineError && (
        <div className="absolute inset-0 w-full h-full flex items-center justify-center text-white p-6 z-50 text-lg font-medium">
          Engine Error: {engineError}
        </div>
      )}
      {loading && !engineError && <Loading loading={loading} />}

      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none pointer-events-auto z-1" />

      {!loading && !engineError && (
        // Mobile-wheel thumb zone; hold WASD or drag — any deflection is a run.
        <div className="absolute bottom-10 left-6 sm:bottom-24 sm:left-48 z-[60] pointer-events-auto">
          <VirtualStick onChange={onStick} display={stickDisplayRef} />
        </div>
      )}

      {!loading && !engineError && (danceReady || bombReady) && (
        // Action row, bottom-right — the dance button's center lines up with the
        // wheel's center on both breakpoints, and the bomb sits beside it. Each
        // key highlights its button; pointerdown for game-feel latency, and no
        // focus — a focused button would re-fire on Space.
        <div className="absolute bottom-14 right-6 sm:bottom-32 sm:right-48 z-[60] pointer-events-auto h-20 flex items-center gap-4">
          {bombReady && (
            <button
              className={`w-14 h-14 rounded-full border-2 flex items-center justify-center select-none touch-none transition-all duration-100 ${
                bombed || shiftHeld || swapPending ? ACTIVE_BUTTON : IDLE_BUTTON
              }`}
              onPointerDown={(e) => {
                e.preventDefault()
                pressLook()
              }}
              onContextMenu={(e) => e.preventDefault()}
              aria-label="Bomb Devil (Shift)"
            >
              <Bomb className="w-6 h-6" />
            </button>
          )}
          {danceReady && (
            <button
              className={`w-20 h-20 rounded-full border-2 font-mono font-semibold text-xs tracking-widest select-none touch-none transition-all duration-100 ${
                dancing || spaceHeld ? ACTIVE_BUTTON : IDLE_BUTTON
              }`}
              onPointerDown={(e) => {
                e.preventDefault()
                toggleDance()
              }}
              onContextMenu={(e) => e.preventDefault()}
              aria-label="Dance (Space)"
            >
              ♪
              <div className="text-[9px] mt-0.5 tracking-[0.2em]">DANCE</div>
            </button>
          )}
        </div>
      )}
    </div>
  )
}
