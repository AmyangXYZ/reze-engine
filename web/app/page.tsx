"use client"

import Header from "@/components/header"
import { Engine, EngineStats, LocomotionController, Vec3 } from "reze-engine"
import { useCallback, useEffect, useRef, useState } from "react"
import Loading from "@/components/loading"

const MODEL_ID = "thoth"
const VMD_BASE = "/unity-fbx-locomotion/vmd"

// Input keys the demo cares about — everything else never touches state.
const INPUT_CODES = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "ShiftLeft", "ShiftRight"])

/** Stick travel radius in px; drag past SPRINT_AT of it to sprint. */
const STICK_RADIUS = 52
const SPRINT_AT = 0.92

/** Mobile-game movement wheel. Reports {x, y (up = +forward), active} through a ref
 *  callback; the knob is moved via direct DOM transform so dragging never re-renders. */
function VirtualStick({ onChange }: { onChange: (x: number, y: number, active: boolean) => void }) {
  const baseRef = useRef<HTMLDivElement>(null)
  const knobRef = useRef<HTMLDivElement>(null)
  const pointerId = useRef<number | null>(null)

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
    knob.style.transform = `translate(${dx}px, ${dy}px)`
    onChange(dx / STICK_RADIUS, -dy / STICK_RADIUS, true)
  }

  const release = () => {
    pointerId.current = null
    if (knobRef.current) knobRef.current.style.transform = "translate(0px, 0px)"
    onChange(0, 0, false)
  }

  return (
    <div
      ref={baseRef}
      className="relative w-36 h-36 rounded-full border-2 border-white/35 bg-white/[0.03] backdrop-blur-[2px] touch-none select-none"
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
      <div className="absolute inset-0 m-auto w-16 h-16 rounded-full border border-white/30 pointer-events-none" />
      <div
        ref={knobRef}
        className="absolute inset-0 m-auto w-12 h-12 rounded-full bg-white/25 border-2 border-white/60 shadow-[0_0_12px_rgba(255,255,255,0.25)] pointer-events-none"
      />
    </div>
  )
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const keysRef = useRef<Set<string>>(new Set())
  const stickRef = useRef({ x: 0, y: 0, active: false })
  const [engineError, setEngineError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<EngineStats | null>(null)

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
        bloom: { color: new Vec3(0.9, 0.3, 0.6) },
        // reze-design's sun: azimuth 205°, elevation 21° (azElToDirection), strength 2.
        sun: { strength: 2.0, direction: new Vec3(0.3946, -0.3584, 0.8462) },
      })
      engineRef.current = engine
      await engine.init()

      const model = await engine.loadModel(MODEL_ID, "/models/托特/托特.pmx")
      await engine.autoStyleGroups(MODEL_ID, { body: ["手"], metal: ["指甲"] })
      // Room to sprint: ~10s of full sprint in any direction before the edge.
      engine.addGround({ diffuseColor: new Vec3(1, 0.3, 0.6), width: 2000, height: 2000, fadeStart: 600, fadeEnd: 950 })

      await Promise.all([
        model.loadVmd("idle", `${VMD_BASE}/Idle.vmd`),
        model.loadVmd("run", `${VMD_BASE}/Run_Lfoot.vmd`),
        model.loadVmd("sprint", `${VMD_BASE}/Sprint_Lfoot.vmd`),
      ])

      // Speeds = the pack's measured root motion × this conversion's scale (0.1306
      // from the measured 托特 skeleton), so strides match the ground covered.
      const controller = new LocomotionController(
        model,
        { idle: "idle", run: "run", sprint: "sprint" },
        { runSpeed: 62.7, sprintSpeed: 86.3 }
      )
      // Spawn facing the camera (rest facing, -Z); she turns around on the first input.
      controller.teleport(0, 0, 0, Math.PI)
      // Follow the root (全ての親): the gait clips don't animate it, so the camera
      // tracks the run without inheriting センター's bob and lean.
      engine.setCameraFollow(model, undefined, new Vec3(0, 11.5, 0))

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
          rawX = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0)
          rawY = (keys.has("KeyW") ? 1 : 0) - (keys.has("KeyS") ? 1 : 0)
          sprint = keys.has("ShiftLeft") || keys.has("ShiftRight")
        }
        const alpha = engine.getCameraAlpha()
        const sinA = Math.sin(alpha)
        const cosA = Math.cos(alpha)
        const x = rawX * -cosA + rawY * -sinA
        const y = rawX * sinA + rawY * -cosA
        controller.setMove(x, y, sprint)
        const pose = controller.update(dt)
        engine.setModelTransform(MODEL_ID, { position: pose.position, rotation: pose.rotation })

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
      if (INPUT_CODES.has(e.code)) keysRef.current.add(e.code)
    }
    const up = (e: KeyboardEvent) => {
      keysRef.current.delete(e.code)
    }
    const blur = () => {
      keysRef.current.clear()
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
        <div className="absolute bottom-24 left-48 z-[60] pointer-events-auto">
          <VirtualStick onChange={onStick} />
        </div>
      )}
    </div>
  )
}
