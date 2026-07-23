"use client"

import Header from "@/components/header"
import { Engine, EngineStats, Model, Vec3, type AnimationProgress } from "reze-engine"
import { useCallback, useEffect, useRef, useState } from "react"
import Loading from "@/components/loading"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Pause, Play } from "lucide-react"

// Format time as M:SS or MM:SS (with leading zero)
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, "0")}`
}

// Format remaining time (negative time shows as "-0:23")
function formatRemainingTime(current: number, duration: number): string {
  const remaining = duration - current
  if (remaining <= 0) return "0:00"
  const mins = Math.floor(remaining / 60)
  const secs = Math.floor(remaining % 60)
  return `-${mins}:${secs.toString().padStart(2, "0")}`
}

/** Scene models: same order as load — transport + seek drive all entries together. */
const SCENE_MODELS = [{ id: "serqet", clip: "m1" }] as const

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const modelsRef = useRef<Model[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<EngineStats | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isPaused, setIsPaused] = useState(false)
  const [progress, setProgress] = useState<AnimationProgress>({
    current: 0,
    duration: 0,
    percentage: 0,
    animationName: null,
    looping: false,
    playing: false,
    paused: false,
  })

  // Sync progress from model (current/duration in seconds, name)
  useEffect(() => {
    let rafId: number | null = null

    const updateProgress = () => {
      const primary = modelsRef.current[0]
      if (primary && isPlaying && !isPaused) {
        const prog: AnimationProgress = primary.getAnimationProgress()
        setProgress({
          current: prog.current,
          duration: prog.duration,
          percentage: prog.percentage,
          animationName: prog.animationName ?? null,
          looping: prog.looping,
          playing: prog.playing,
          paused: prog.paused,
        })
        setIsPlaying(prog.playing)
        setIsPaused(prog.paused)
        if (prog.playing) rafId = requestAnimationFrame(updateProgress)
      }
    }

    if (isPlaying && !isPaused) {
      rafId = requestAnimationFrame(updateProgress)
    }

    return () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId)
      }
    }
  }, [isPlaying, isPaused])

  // Create and preload audio element on mount
  useEffect(() => {
    const audio = new Audio("/audios/One More Last Time.wav")
    audio.preload = "auto"
    audio.setAttribute("playsinline", "true")
    audio.setAttribute("webkit-playsinline", "true")
    audio.volume = 1.0
    audio.muted = false

    Object.assign(audio.style, {
      display: "none",
      position: "absolute",
      visibility: "hidden",
      width: "0",
      height: "0",
    })
    document.body.appendChild(audio)

    audio.load()

    audio.addEventListener("loadeddata", () => {
      audioRef.current = audio
    })

    audio.addEventListener("error", () => {
      console.warn("Audio failed to load")
    })

    return () => {
      audio.pause()
      audio.parentNode?.removeChild(audio)
    }
  }, [])

  const handlePlay = useCallback(() => {
    const models = modelsRef.current
    if (!engineRef.current || models.length === 0) return
    const prog = models[0].getAnimationProgress()
    if (prog.paused) {
      if (audioRef.current) {
        audioRef.current.muted = false
        audioRef.current.volume = 1.0
        audioRef.current.play().catch(() => {})
      }
      for (const m of models) m.play()

      setIsPlaying(true)
      setIsPaused(false)
      return
    }
    if (prog.playing) return
    if (audioRef.current) {
      audioRef.current.muted = false
      audioRef.current.volume = 1.0
      const atEnd = prog.duration > 0 && prog.current >= prog.duration - 1e-3
      audioRef.current.currentTime = atEnd ? 0 : prog.current
      audioRef.current.play().catch(() => {})
    }
    const atEnd = prog.duration > 0 && prog.current >= prog.duration - 1e-3
    if (atEnd) for (const m of models) m.seek(0)
    for (const m of models) m.play()

    setIsPlaying(true)
    setIsPaused(false)
  }, [])

  const handlePause = useCallback(() => {
    if (engineRef.current) {
      for (const m of modelsRef.current) m.pause()
      if (audioRef.current) {
        audioRef.current.pause()
      }
      setIsPaused(true)
    }
  }, [])

  const handleResume = useCallback(() => {
    if (engineRef.current) {
      if (audioRef.current) {
        audioRef.current.play().catch(() => {})
      }
      for (const m of modelsRef.current) m.play()
      modelsRef.current[0]?.setMorphWeight("抗穿模", 0.5)
      setIsPaused(false)
    }
  }, [])

  const handleSeek = useCallback(
    (value: number[]) => {
      if (engineRef.current && progress.duration > 0) {
        const seekTime = (value[0] / 100) * progress.duration
        for (const m of modelsRef.current) m.seek(seekTime)
        if (audioRef.current) {
          audioRef.current.currentTime = seekTime
        }
        setProgress((p) => ({
          ...p,
          current: seekTime,
          percentage: value[0],
        }))
        // No resetPhysics needed: the engine detects the bone-pose jump and
        // carries dynamic bodies across the seek per kinematic root.
      }
    },
    [progress.duration],
  )

  const initEngine = useCallback(async () => {
    if (!canvasRef.current) {
      setLoading(false)
      return
    }
    try {
      const engine = new Engine(canvasRef.current, {
        camera: { distance: 31.5, target: new Vec3(0, 11.5, 0) },
        bloom: { color: new Vec3(0.9, 0.3, 0.6) },
      })
      engineRef.current = engine
      await engine.init()

      const m1 = await engine.loadModel(SCENE_MODELS[0].id, "/models/塞尔凯特/塞尔凯特.pmx")

      modelsRef.current = [m1]

      // Group materials into their default style graphs (custom names → categories the
      // built-in hints miss). Ungrouped materials would render the neutral default.
      await engine.autoStyleGroups(SCENE_MODELS[0].id, {
        eye: ["眼睛", "眼白", "目白", "右瞳", "左瞳", "眉毛"],
        face: ["脸", "face01"],
        body: ["皮肤", "skin"],
        hair: ["头发", "hair_f"],
        cloth_smooth: [
          "衣服",
          "裙子",
          "裙带",
          "裙布",
          "外套",
          "外套饰",
          "裤子",
          "裤子0",
          "腿环",
          "发饰",
          "鞋子",
          "鞋子饰",
          "shirt",
          "shoes",
          "shorts",
          "trigger",
          "dress",
          "hair_accessory",
          "cloth01_shoes",
        ],
        stockings: ["袜子", "stockings"],
        metal: ["metal01", "earring"],
      })

      engine.addGround({
        diffuseColor: new Vec3(1, 0.3, 0.6),
      })

      await m1.loadVmd(SCENE_MODELS[0].clip, "/animations/One More Last Time.vmd")

      engine.runRenderLoop(() => setStats(engine.getStats()))

      await new Promise((resolve) => requestAnimationFrame(resolve))

      m1.show(SCENE_MODELS[0].clip)

      await new Promise((resolve) => requestAnimationFrame(resolve))

      engine.resetPhysics()

      const prog: AnimationProgress = m1.getAnimationProgress()
      setProgress({
        current: prog.current,
        duration: prog.duration,
        percentage: prog.percentage,
        animationName: prog.animationName ?? null,
        looping: prog.looping,
        playing: prog.playing,
        paused: prog.paused,
      })
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
      if (engineRef.current) {
        engineRef.current.dispose()
      }
      if (audioRef.current) {
        audioRef.current.pause()
        if (audioRef.current.parentNode) {
          audioRef.current.parentNode.removeChild(audioRef.current)
        }
      }
    }
  }, [initEngine])

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.code !== "Space" && e.key !== " ") return
      e.preventDefault()
      if (isPlaying && !isPaused) handlePause()
      else if (isPaused) handleResume()
      else handlePlay()
    }
    window.addEventListener("keydown", handleKeyDown)
    return () => window.removeEventListener("keydown", handleKeyDown)
  }, [isPlaying, isPaused, handlePlay, handlePause, handleResume])

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
        <div className="absolute bottom-4 left-4 right-4 z-[60] pointer-events-auto">
          <div className="max-w-4xl mx-auto px-2 pr-4 bg-black/30 backdrop-blur-xs rounded-full outline-none pointer-events-auto">
            <div className="flex items-center gap-3">
              {!isPlaying ? (
                <Button onClick={handlePlay} size="icon" variant="ghost" aria-label="Play">
                  <Play />
                </Button>
              ) : isPaused ? (
                <Button onClick={handleResume} size="icon" variant="ghost" aria-label="Resume">
                  <Play />
                </Button>
              ) : (
                <Button onClick={handlePause} size="icon" variant="ghost" aria-label="Pause">
                  <Pause />
                </Button>
              )}

              <div className="text-white text-sm font-mono tabular-nums flex items-center gap-2">
                {formatTime(progress.current)}
                {progress.looping && (
                  <span className="text-[10px] uppercase tracking-wide text-emerald-400/90">loop</span>
                )}
              </div>

              <div className="flex-1">
                <Slider
                  value={[progress.percentage]}
                  onValueChange={handleSeek}
                  min={0}
                  max={100}
                  step={0.001}
                  className="w-full"
                  disabled={progress.duration === 0}
                />
              </div>

              <div className="text-muted-foreground text-sm font-mono tabular-nums text-right">
                {formatRemainingTime(progress.current, progress.duration)}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
