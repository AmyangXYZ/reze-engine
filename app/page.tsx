"use client"

import Header from "@/components/header"
import { Engine, EngineStats } from "@/lib/engine"
import { Button } from "@/components/ui/button"
import { useCallback, useEffect, useRef, useState } from "react"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [showRigidbodies, setShowRigidbodies] = useState(false)
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
        setEngineError(error instanceof Error ? error.message : "Unknown error")
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

  // Sync rigidbody visibility when state changes
  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.setShowRigidbodies(showRigidbodies)
    }
  }, [showRigidbodies])

  const handleToggleRigidbodies = useCallback(() => {
    setShowRigidbodies((prev) => !prev)
  }, [])

  return (
    <div className="fixed inset-0 w-full h-full overflow-hidden touch-none">
      <Header stats={stats} />

      {/* Rigidbody toggle button - middle left */}
      <div className="absolute left-4 top-1/2 -translate-y-1/2 z-10 pointer-events-auto">
        <Button
          size="sm"
          variant="outline"
          onClick={handleToggleRigidbodies}
          className="bg-black/80 text-white hover:bg-black border-white/20 hover:border-white/40"
        >
          {showRigidbodies ? "Hide" : "Show"} Rigidbodies
        </Button>
      </div>

      {engineError && (
        <div className="absolute inset-0 w-full h-full flex items-center justify-center text-white">
          Engine Error: {engineError}
        </div>
      )}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none z-1" />
    </div>
  )
}
