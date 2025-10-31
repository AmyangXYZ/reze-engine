"use client"

import Header from "@/components/header"
import { Engine, EngineStats } from "@/lib/engine"
import { useCallback, useEffect, useRef, useState } from "react"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
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
      const engine = new Engine(canvasRef.current)
      engineRef.current = engine
      await engine.init()

      // Load default model
      // await engine.loadRzm("/models/dummy.rzm")
      await engine.loadPmx("/models/梵天/", "梵天-o.pmx")

      // Start render loop with stats callback
      engine.runRenderLoop(() => {
        setStats(engine.getStats())
      })
    }
  }, [])

  useEffect(() => {
    initEngine()

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
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none z-1" />
    </div>
  )
}
