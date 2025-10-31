"use client"

import Header from "@/components/header"
import { Engine, EngineStats } from "@/lib/engine"
import { useCallback, useEffect, useRef, useState } from "react"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const [webgpuNotSupported, setWebgpuNotSupported] = useState(false)
  const [stats, setStats] = useState<EngineStats>({
    fps: 0,
    frameTime: 0,
    memoryUsed: 0,
    drawCalls: 0,
    vertices: 0,
  })

  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      // Initialize engine
      try {
        const engine = new Engine(canvasRef.current)
        engineRef.current = engine
        await engine.init()
        await engine.loadPmx("/models/梵天/", "梵天-o.pmx")
        engine.runRenderLoop(() => {
          setStats(engine.getStats())
        })
      } catch (error) {
        console.error("Error initializing engine:", error)
        setWebgpuNotSupported(true)
      }
    }
  }, [])

  useEffect(() => {
    void (async () => {
      await initEngine()
    })()

    // Cleanup on unmount
    return () => {
      if (engineRef.current) {
        engineRef.current.dispose()
      }
    }
  }, [initEngine])

  return (
    <div className="fixed inset-0 w-full h-full overflow-hidden touch-none">
      <Header stats={stats} />
      {webgpuNotSupported && <div className="absolute inset-0 w-full h-full flex items-center justify-center text-white">WebGPU not supported</div>}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none z-1" />
    </div>
  )
}
