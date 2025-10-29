"use client"

import { Engine, EngineStats } from "@/lib/engine"
import { useCallback, useEffect, useRef, useState } from "react";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const [stats, setStats] = useState<EngineStats | null>(null)

  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      // Initialize engine
      const engine = new Engine(canvasRef.current)
      engineRef.current = engine
      await engine.init()

      // Load default model
      // await engine.loadRzm("/models/dummy.rzm")
      await engine.loadPmx("/models/梵天.pmx")

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
      {/* Full-screen canvas */}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none" />

      {/* Floating header */}
      <header className="absolute top-0 left-0 right-0 p-6 pointer-events-none">
        <h1 className="text-3xl font-bold text-white drop-shadow-lg">Reze Engine</h1>
      </header>

      {/* Stats panel */}
      {stats && (
        <div className="absolute top-20 right-6 bg-black/70 text-white p-4 rounded-lg text-sm font-mono pointer-events-none backdrop-blur-sm min-w-[180px]">
          <div className="space-y-1">
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">FPS:</span>
              <span className="font-bold">{stats.fps}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Frame:</span>
              <span>{stats.frameTime.toFixed(2)} ms</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Draw Calls:</span>
              <span>{stats.drawCalls}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted-foreground">Vertices:</span>
              <span>{stats.vertices}</span>
            </div>
            {stats.memoryUsed > 0 && (
              <div className="flex justify-between gap-4">
                <span className="text-muted-foreground">Memory:</span>
                <span>{stats.memoryUsed} MB</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
