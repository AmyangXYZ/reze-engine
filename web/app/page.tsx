"use client"

import Header from "@/components/header"
import { Engine, EngineStats, LocomotionController, Model, Vec3, type StrafeClipEntry } from "reze-engine"
import { useCallback, useEffect, useRef, useState } from "react"
import Loading from "@/components/loading"

const VMD_ROOT = "/unity-fbx-locomotion"
const VMD_BASE = `${VMD_ROOT}/vmd`

// ai同屏连携: the player character plus two AI companions in follow formation.
// Every character plays VMDs converted against ITS OWN measured skeleton, with the
// pack's authored root speeds at that conversion's scale.
const PLAYER = {
  id: "thoth",
  pmx: "/models/托特/托特.pmx",
  vmdDir: VMD_BASE,
  runSpeed: 62.7,
  sprintSpeed: 86.3,
}
const COMPANIONS = [
  {
    id: "izanami",
    pmx: "/models/深空之眼—伊邪那美「初雪千华」/伊邪那美誓约2.0.pmx",
    vmdDir: `${VMD_ROOT}/vmd-izanami`,
    runSpeed: 61.3,
    sprintSpeed: 84.5,
    // formation slot in player-local space: x = player's right, z = player's forward
    slot: { x: -14, z: -11 },
  },
  {
    id: "skuld",
    pmx: "/models/深空之眼—诗蔻蒂/诗蔻蒂3.0.pmx",
    vmdDir: `${VMD_ROOT}/vmd-skuld`,
    runSpeed: 61.7,
    sprintSpeed: 85.0,
    slot: { x: 14, z: -11 },
  },
]
// Companions hold position inside the deadband; approach speed ramps in over the
// arrive radius beyond it (analog magnitude), so they settle instead of overshooting.
const FOLLOW_DEADBAND = 3
const FOLLOW_ARRIVE = 8
const FOLLOW_SPRINT_AT = 30

const deg = (d: number) => (d * Math.PI) / 180

// Authored stops: skid profiles measured from each clip's root motion at this
// conversion's scale (exit at the plateau). Release at speed plays the matched
// stop; ANY key press interrupts it instantly — anims for committed outcomes,
// immediate response for new intent.
const STOP_CLIPS = [
  { clip: "Sprint_Stop_Lfoot", exitTime: 2.08, forward: [0, 13.9, 24.1, 34.2, 42.2, 47.3, 51.1, 53.5, 53.7, 52.7], gear: "sprint" as const, foot: "L" as const },
  { clip: "Sprint_Stop_Rfoot", exitTime: 2.22, forward: [0, 13.9, 24.1, 34.2, 41, 46.7, 50.5, 53.6, 56.1, 58.5, 58.7], gear: "sprint" as const, foot: "R" as const },
]

// The 8-direction strafe rings. Angles and speeds MEASURED from each clip's authored
// root motion at this conversion's scale (FL/FR are the pure ±90° side clips;
// BL/BR are slower side-step variants, unused).
const STRAFE_RUN: StrafeClipEntry[] = [
  { clip: "StrafeRun_F", angle: deg(0), speed: 65.4 },
  { clip: "StrafeRun_R45", angle: deg(45), speed: 65.4 },
  { clip: "StrafeRun_FR", angle: deg(90), speed: 60.6 },
  { clip: "StrafeRun_R135", angle: deg(135), speed: 65.4 },
  { clip: "StrafeRun_B", angle: deg(180), speed: 65.4 },
  { clip: "StrafeRun_L135", angle: deg(-135), speed: 65.4 },
  { clip: "StrafeRun_FL", angle: deg(-90), speed: 60.6 },
  { clip: "StrafeRun_L45", angle: deg(-45), speed: 65.4 },
]
const STRAFE_SPRINT: StrafeClipEntry[] = [
  { clip: "StrafeSprint_F", angle: deg(0), speed: 83.0 },
  { clip: "StrafeSprint_R45", angle: deg(45), speed: 85.4 },
  { clip: "StrafeSprint_FR", angle: deg(90), speed: 83.1 },
  { clip: "StrafeSprint_R135", angle: deg(135), speed: 77.0 },
  { clip: "StrafeSprint_B", angle: deg(180), speed: 65.3 },
  { clip: "StrafeSprint_L135", angle: deg(-135), speed: 77.0 },
  { clip: "StrafeSprint_FL", angle: deg(-90), speed: 83.0 },
  { clip: "StrafeSprint_L45", angle: deg(-45), speed: 83.0 },
]

// Input keys the demo cares about — everything else never touches state.
const INPUT_CODES = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "Space"])

/** Stick travel radius in px; drag past SPRINT_AT of it to sprint. Travel close to
 *  the base radius (72) lets the knob overhang the rim at full deflection — the
 *  familiar mobile-wheel "pushed past the edge" look. */
const STICK_RADIUS = 66
const SPRINT_AT = 0.92
// Keyboard throttle, race-game style: holding WASD ramps the input magnitude toward
// the rim (sprint) and release decays it — one rule for both inputs: deflection =
// speed, rim = sprint. Two-phase charge: snap to run fast, then a deliberate slower
// push into the rim, so movement is responsive but sprint is intentional.
const KB_RUN_AT = 0.75 // deflection of a full run
const KB_TO_RUN = 0.35 // s from standstill to full run
const KB_TO_SPRINT = 1.6 // s of continued holding from run into the rim — sprint is earned
const KB_DECEL_TIME = 0 // instant: the stop must begin at full velocity or the seam lurches
// Action-game two-speed rule: any real movement intent means at least a full run —
// the idle⊕run blend band is a ramp to pass through, never a place to dwell
// (the mid-band "slow run" reads as interpolation, not locomotion).
const MIN_MOVE = 0.8
const KB_TURN_PAUSE = 0.35 // s: a direction change pauses the charge until the new keys are held steadily

/** Mobile-game movement wheel. Reports {x, y (up = +forward), active} through a ref
 *  callback; the knob is moved via direct DOM transform so dragging never re-renders.
 *  `display` receives a setter the host calls to mirror OTHER input (keyboard WASD)
 *  on the knob — ignored while a drag owns it. */
function VirtualStick({
  onChange,
  display,
}: {
  onChange: (x: number, y: number, active: boolean) => void
  display: React.MutableRefObject<((x: number, y: number) => void) | null>
}) {
  const baseRef = useRef<HTMLDivElement>(null)
  const knobRef = useRef<HTMLDivElement>(null)
  const pointerId = useRef<number | null>(null)

  useEffect(() => {
    display.current = (x, y) => {
      if (pointerId.current !== null || !knobRef.current) return
      const len = Math.hypot(x, y)
      if (len > 1) {
        x /= len
        y /= len
      }
      knobRef.current.style.transform = `translate(${x * STICK_RADIUS}px, ${-y * STICK_RADIUS}px)`
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
    let dx = clientX - (rect.left + rect.width / 2)
    let dy = clientY - (rect.top + rect.height / 2)
    const len = Math.hypot(dx, dy)
    if (len > STICK_RADIUS) {
      dx *= STICK_RADIUS / len
      dy *= STICK_RADIUS / len
    }
    knob.style.transitionDuration = "0ms" // dragging tracks the finger 1:1
    knob.style.transform = `translate(${dx}px, ${dy}px)`
    onChange(dx / STICK_RADIUS, -dy / STICK_RADIUS, true)
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
      className="relative w-36 h-36 rounded-full border-2 border-white/70 bg-white/15 backdrop-blur-[2px] touch-none select-none"
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
      <div className="absolute inset-0 m-auto w-16 h-16 rounded-full border border-white/50 pointer-events-none" />
      {/* Eased by default so keyboard pushes glide to their direction and releases
          spring back; dragging zeroes the duration inline for 1:1 tracking. */}
      <div
        ref={knobRef}
        className="absolute inset-0 m-auto w-12 h-12 rounded-full bg-white/70 border-2 border-white shadow-[0_0_14px_rgba(255,255,255,0.45)] pointer-events-none transition-transform duration-150 ease-out"
      />
    </div>
  )
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const keysRef = useRef<Set<string>>(new Set())
  const stickRef = useRef({ x: 0, y: 0, active: false })
  const actorsRef = useRef<Model[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const dancingRef = useRef(false)
  const danceCancelRef = useRef(false)
  const [dancing, setDancing] = useState(false)
  const [spaceHeld, setSpaceHeld] = useState(false)
  const kbThrottle = useRef({ mag: 0, dirX: 0, dirY: 1, lastKx: 0, lastKy: 0, cool: 0 })
  const [engineError, setEngineError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<EngineStats | null>(null)

  // ai同屏连携 showcase: one press (button or Space), all three play the dance together
  // with the song — a synced one-shot per model. The dance can't re-trigger itself;
  // only MOVEMENT interrupts it (WASD/wheel cancels, and you run out of the fade).
  const startDance = useCallback(() => {
    const actors = actorsRef.current
    if (actors.length === 0 || dancingRef.current) return
    if (!audioRef.current) {
      audioRef.current = new Audio("/audios/One More Last Time.wav")
      audioRef.current.preload = "auto"
    }
    const onEnd = () => {
      // fired by the player's one-shot finishing (natural end or cancel fade done)
      dancingRef.current = false
      danceCancelRef.current = false
      setDancing(false)
      audioRef.current?.pause()
    }
    let ok = false
    actors.forEach((m, i) => {
      ok = m.playOneShot("dance", { fadeIn: 0.4, fadeOut: 0.6, onEnd: i === 0 ? onEnd : undefined }) || ok
    })
    if (!ok) return
    audioRef.current.currentTime = 0
    void audioRef.current.play().catch(() => {})
    dancingRef.current = true
    danceCancelRef.current = false
    setDancing(true)
  }, [])

  const stickDisplayRef = useRef<((x: number, y: number) => void) | null>(null)

  const onStick = useCallback((x: number, y: number, active: boolean) => {
    stickRef.current.x = x
    stickRef.current.y = y
    stickRef.current.active = active
  }, [])

  const initEngine = useCallback(async () => {
    if (!canvasRef.current) {
      setLoading(false)
      return
    }
    try {
      const engine = new Engine(canvasRef.current, {
        camera: { distance: 45, target: new Vec3(0, 11.5, 0) },
        bloom: { color: new Vec3(0.75, 0.82, 1.0) },
        // reze-design's sun: azimuth 205°, elevation 21° (azElToDirection), strength 2.
        sun: { strength: 2.0, direction: new Vec3(0.3946, -0.3584, 0.8462) },
        // tailwind blue-200, display-space sRGB
        background: new Vec3(0.749, 0.859, 0.996),
      })
      engineRef.current = engine
      await engine.init()
      // Uncapped: chase native refresh. engine.setMaxFPS(n) is available for
      // hosts that prefer to spend less CPU on high-refresh displays.
      // Console access for perf work: engine.getStats() has the CPU breakdown,
      // engine.setPhysicsEnabled(false) isolates physics.
      ;(window as unknown as { engine?: Engine }).engine = engine

      const model = await engine.loadModel(PLAYER.id, PLAYER.pmx)
      await engine.autoStyleGroups(PLAYER.id, { body: ["手"], metal: ["指甲"] })
      // Room to sprint: ~10s of full sprint in any direction before the edge.
      // Pale blue-grey stage with a fine white grid, fading into the backdrop.
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

      await Promise.all([
        model.loadVmd("idle", `${PLAYER.vmdDir}/Idle.vmd`),
        model.loadVmd("run", `${PLAYER.vmdDir}/Run_Lfoot.vmd`),
        model.loadVmd("sprint", `${PLAYER.vmdDir}/Sprint_Lfoot.vmd`),
        model.loadVmd("dance", "/animations/One More Last Time.vmd"),
        ...[...STRAFE_RUN, ...STRAFE_SPRINT].map((e) => model.loadVmd(e.clip, `${PLAYER.vmdDir}/${e.clip}.vmd`)),
        ...STOP_CLIPS.map((e) => model.loadVmd(e.clip, `${PLAYER.vmdDir}/${e.clip}.vmd`)),
      ])

      const controller = new LocomotionController(
        model,
        { idle: "idle", run: "run", sprint: "sprint", strafeRun: STRAFE_RUN, strafeSprint: STRAFE_SPRINT, stop: STOP_CLIPS },
        { runSpeed: PLAYER.runSpeed, sprintSpeed: PLAYER.sprintSpeed }
      )
      // Spawn facing the camera (rest facing, -Z); she turns around on the first input.
      controller.teleport(0, 0, 0, Math.PI)
      // Follow the root (全ての親): the gait clips don't animate it, so the camera
      // tracks the run without inheriting センター's bob and lean.
      engine.setCameraFollow(model, undefined, new Vec3(0, 11.5, 0))

      // AI companions: own model, own per-skeleton VMDs, own controller — steered by
      // a follow policy through the same setMove interface the player uses.
      const companions = await Promise.all(
        COMPANIONS.map(async (def) => {
          const m = await engine.loadModel(def.id, def.pmx)
          await engine.autoStyleGroups(def.id)
          await Promise.all([
            m.loadVmd("idle", `${def.vmdDir}/Idle.vmd`),
            m.loadVmd("run", `${def.vmdDir}/Run_Lfoot.vmd`),
            m.loadVmd("sprint", `${def.vmdDir}/Sprint_Lfoot.vmd`),
            m.loadVmd("dance", "/animations/One More Last Time.vmd"),
          ])
          const c = new LocomotionController(
            m,
            { idle: "idle", run: "run", sprint: "sprint" },
            { runSpeed: def.runSpeed, sprintSpeed: def.sprintSpeed }
          )
          // Spawn already in formation around the player (player faces -Z at rest).
          c.teleport(-def.slot.x, 0, -def.slot.z, Math.PI)
          return { def, controller: c, model: m }
        })
      )
      actorsRef.current = [model, ...companions.map((c) => c.model)]

      let last = performance.now()
      engine.runRenderLoop(() => {
        const now = performance.now()
        const dt = (now - last) / 1000
        last = now

        // Camera-relative controls, FPS-style: the mouse orbits the view and thereby
        // steers the run — up is always away from the camera, right is screen-right.
        // The wheel gives analog direction/magnitude (rim = sprint); WASD + Shift
        // feed the same vector. Orbit eye sits at target + r·(sinα, ·, cosα), so
        // screen-forward is (-sinα, -cosα) and screen-right is (-cosα, sinα).
        const keys = keysRef.current
        const stick = stickRef.current
        let rawX: number
        let rawY: number
        let sprint: boolean
        if (stick.active) {
          rawX = stick.x
          rawY = stick.y
          sprint = Math.hypot(stick.x, stick.y) > SPRINT_AT
        } else {
          // Keyboard throttle: held keys steer while the magnitude ramps toward the
          // rim; release keeps the last direction and decays, so she glides down
          // through run to a stop and the knob plays the whole story.
          const kx = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0)
          const ky = (keys.has("KeyW") ? 1 : 0) - (keys.has("KeyS") ? 1 : 0)
          const t = kbThrottle.current
          const held = kx !== 0 || ky !== 0
          if (held) {
            const len = Math.hypot(kx, ky)
            t.dirX = kx / len
            t.dirY = ky / len
            // A different key combo interrupts the commitment: hold the charge
            // until the new direction has been held steadily for a moment. (The
            // first press from standstill is not a change — starts stay snappy.)
            const changed = kx !== t.lastKx || ky !== t.lastKy
            if (changed && (t.lastKx !== 0 || t.lastKy !== 0)) t.cool = KB_TURN_PAUSE
            if (t.cool > 0) {
              t.cool -= dt
            } else {
              const rate = t.mag < KB_RUN_AT ? KB_RUN_AT / KB_TO_RUN : (1 - KB_RUN_AT) / KB_TO_SPRINT
              t.mag = Math.min(1, t.mag + rate * dt)
            }
          } else {
            t.mag = Math.max(0, t.mag - dt / KB_DECEL_TIME)
            t.cool = 0
          }
          t.lastKx = kx
          t.lastKy = ky
          rawX = t.dirX * t.mag
          rawY = t.dirY * t.mag
          sprint = t.mag > SPRINT_AT
          stickDisplayRef.current?.(rawX, rawY)
        }
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
        // Souls-style unlocked camera: input is camera-relative but the character
        // turns toward the movement and runs — never glued to the camera's facing.
        // (The strafe ring stays loaded for a future lock-on mode, where side-
        // stepping is the right behavior: controller.setFacing(targetYaw).)
        // Movement interrupts the dance: one cancel fade, and she runs out of it.
        if (dancingRef.current && !danceCancelRef.current && (rawX !== 0 || rawY !== 0 || stick.active)) {
          danceCancelRef.current = true
          for (const m of actorsRef.current) m.cancelOneShot(0.25) // snappy: she's running almost immediately
        }
        controller.setMove(x, y, sprint)
        const pose = controller.update(dt)
        engine.setModelTransform(PLAYER.id, { position: pose.position, rotation: pose.rotation })

        // Companion follow AI: chase a formation slot in the player's local frame;
        // hold inside the deadband, run beyond it, sprint when left far behind.
        const fx = Math.sin(pose.yaw)
        const fz = Math.cos(pose.yaw)
        for (const { def, controller: c } of companions) {
          const tx = pose.position.x + Math.cos(pose.yaw) * def.slot.x + fx * def.slot.z
          const tz = pose.position.z - Math.sin(pose.yaw) * def.slot.x + fz * def.slot.z
          const p = c.getPosition()
          const dx = tx - p.x
          const dz = tz - p.z
          const dist = Math.hypot(dx, dz)
          if (!dancingRef.current && dist > FOLLOW_DEADBAND) {
            const mag = Math.min(1, (dist - FOLLOW_DEADBAND) / FOLLOW_ARRIVE)
            c.setMove((dx / dist) * mag, (dz / dist) * mag, dist > FOLLOW_SPRINT_AT)
          } else {
            c.setMove(0, 0)
          }
          const cp = c.update(dt)
          engine.setModelTransform(def.id, { position: cp.position, rotation: cp.rotation })
        }

        setStats(engine.getStats())
      })

      // One settled frame (bind pose → blended idle is a jump), then park the physics on it.
      await new Promise((resolve) => requestAnimationFrame(resolve))
      engine.resetPhysics()

      setEngineError(null)
    } catch (error) {
      setEngineError(error instanceof Error ? error.message : "Unknown error")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void initEngine()
    return () => {
      engineRef.current?.dispose()
    }
  }, [initEngine])

  // WASD + Shift held-key tracking. Blur clears everything so keys can't stick
  // when the tab loses focus mid-press.
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (!INPUT_CODES.has(e.code)) return
      if (e.code === "Space") {
        e.preventDefault() // page scroll
        setSpaceHeld(true)
        startDance()
        return
      }
      keysRef.current.add(e.code)
    }
    const up = (e: KeyboardEvent) => {
      if (e.code === "Space") {
        setSpaceHeld(false)
        return
      }
      keysRef.current.delete(e.code)
    }
    const blur = () => {
      keysRef.current.clear()
      setSpaceHeld(false)
    }
    window.addEventListener("keydown", down)
    window.addEventListener("keyup", up)
    window.addEventListener("blur", blur)
    return () => {
      window.removeEventListener("keydown", down)
      window.removeEventListener("keyup", up)
      window.removeEventListener("blur", blur)
    }
  }, [])

  return (
    <div className="fixed inset-0 w-full h-full overflow-hidden touch-none">
      <Header stats={stats} />

      {engineError && (
        <div className="absolute inset-0 w-full h-full flex items-center justify-center text-white p-6 z-50 text-lg font-medium">
          Engine Error: {engineError}
        </div>
      )}
      {loading && !engineError && <Loading loading={loading} />}

      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none pointer-events-auto z-1" />

      {!loading && !engineError && (
        // Mobile-wheel thumb zone; hold WASD or drag — the rim is the sprint zone.
        <div className="absolute bottom-10 left-6 sm:bottom-24 sm:left-48 z-[60] pointer-events-auto">
          <VirtualStick onChange={onStick} display={stickDisplayRef} />
        </div>
      )}

      {!loading && !engineError && (
        // Skill zone, bottom-right — the button's center lines up with the wheel
        // knob's center (wheel center = container bottom + 72px). Space triggers and
        // highlights it; while dancing it can't re-trigger — only movement interrupts.
        <div className="absolute bottom-[4.5rem] right-6 sm:bottom-32 sm:right-48 z-[60] pointer-events-auto">
          <button
            className={`w-20 h-20 rounded-full border-2 font-mono font-semibold text-xs tracking-widest select-none touch-none transition-all duration-100 ${
              dancing || spaceHeld
                ? "bg-white/90 text-black border-white scale-95 shadow-[0_0_28px_rgba(255,255,255,0.5)]"
                : "bg-white/25 text-white border-white/70 backdrop-blur-md shadow-[inset_0_1px_0_rgba(255,255,255,0.3),0_2px_12px_rgba(0,0,0,0.25)]"
            }`}
            onClick={startDance}
            onContextMenu={(e) => e.preventDefault()}
            aria-label="Dance together (Space)"
          >
            ♪
            <div className="text-[9px] mt-0.5 tracking-[0.2em]">DANCE</div>
          </button>
        </div>
      )}
    </div>
  )
}
