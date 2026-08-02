"use client"

import Header from "@/components/header"
import { Engine, EngineStats, LocomotionController, Vec3 } from "reze-engine"
import { useCallback, useEffect, useRef, useState } from "react"
import Loading from "@/components/loading"

const MODEL_ID = "thoth"
const VMD_BASE = "/unity-fbx-locomotion/vmd"

// Input keys the demo cares about — everything else never touches state.
const INPUT_CODES = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "ShiftLeft", "ShiftRight"])

function KeyButton({
  code,
  label,
  active,
  wide,
  onPress,
}: {
  code: string
  label: string
  active: boolean
  wide?: boolean
  onPress: (code: string, on: boolean) => void
}) {
  return (
    <button
      className={`${wide ? "w-40 h-10 text-xs tracking-[0.2em]" : "w-12 h-12 text-sm"} flex items-center justify-center rounded-xl border font-mono font-semibold select-none touch-none transition-all duration-75 ${
        active
          ? "bg-white/90 text-black border-white scale-95 shadow-[0_0_20px_rgba(255,255,255,0.45)]"
          : "bg-white/5 text-white/75 border-white/15 backdrop-blur-md shadow-[inset_0_1px_0_rgba(255,255,255,0.15),0_2px_10px_rgba(0,0,0,0.35)]"
      }`}
      onPointerDown={(e) => {
        e.preventDefault()
        e.currentTarget.setPointerCapture(e.pointerId)
        onPress(code, true)
      }}
      onPointerUp={() => onPress(code, false)}
      onPointerCancel={() => onPress(code, false)}
      onContextMenu={(e) => e.preventDefault()}
      aria-label={label}
    >
      {label}
    </button>
  )
}

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const keysRef = useRef<Set<string>>(new Set())
  const [engineError, setEngineError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<EngineStats | null>(null)
  const [pressed, setPressed] = useState<ReadonlySet<string>>(new Set())

  // One entry point for keyboard and on-screen buttons: the render loop reads
  // keysRef; `pressed` mirrors it for the button highlights.
  const press = useCallback((code: string, on: boolean) => {
    const keys = keysRef.current
    if (on === keys.has(code)) return
    if (on) keys.add(code)
    else keys.delete(code)
    setPressed(new Set(keys))
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
        // steers the run — W is always away from the camera, D is screen-right.
        // Orbit eye sits at target + r·(sinα, ·, cosα), so screen-forward is
        // (-sinα, -cosα) and screen-right is (-cosα, sinα).
        const keys = keysRef.current
        const rawX = (keys.has("KeyD") ? 1 : 0) - (keys.has("KeyA") ? 1 : 0)
        const rawY = (keys.has("KeyW") ? 1 : 0) - (keys.has("KeyS") ? 1 : 0)
        const alpha = engine.getCameraAlpha()
        const sinA = Math.sin(alpha)
        const cosA = Math.cos(alpha)
        const x = rawX * -cosA + rawY * -sinA
        const y = rawX * sinA + rawY * -cosA
        controller.setMove(x, y, keys.has("ShiftLeft") || keys.has("ShiftRight"))
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
      if (INPUT_CODES.has(e.code)) press(e.code, true)
    }
    const up = (e: KeyboardEvent) => {
      if (INPUT_CODES.has(e.code)) press(e.code, false)
    }
    const blur = () => {
      keysRef.current.clear()
      setPressed(new Set())
    }
    window.addEventListener("keydown", down)
    window.addEventListener("keyup", up)
    window.addEventListener("blur", blur)
    return () => {
      window.removeEventListener("keydown", down)
      window.removeEventListener("keyup", up)
      window.removeEventListener("blur", blur)
    }
  }, [press])

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
        // Mobile-wheel position: 20% in from the left and bottom edges.
        <div className="absolute bottom-24 left-48 z-[60] pointer-events-auto flex flex-col items-center gap-2">
          <KeyButton code="KeyW" label="W" active={pressed.has("KeyW")} onPress={press} />
          <div className="flex gap-2">
            <KeyButton code="KeyA" label="A" active={pressed.has("KeyA")} onPress={press} />
            <KeyButton code="KeyS" label="S" active={pressed.has("KeyS")} onPress={press} />
            <KeyButton code="KeyD" label="D" active={pressed.has("KeyD")} onPress={press} />
          </div>
          <KeyButton
            code="ShiftLeft"
            label="SHIFT"
            wide
            active={pressed.has("ShiftLeft") || pressed.has("ShiftRight")}
            onPress={press}
          />
        </div>
      )}
    </div>
  )
}
