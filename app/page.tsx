"use client"

import Header from "@/components/header"
import { Engine, EngineStats } from "@/lib/engine"
import { Button } from "@/components/ui/button"
import { useCallback, useEffect, useRef, useState } from "react"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [showRigidbodies, setShowRigidbodies] = useState<boolean>(false)
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
        // Sync initial state
        setShowRigidbodies(engine.getShowRigidbodies())
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

  const handleToggleRigidbodies = useCallback(() => {
    if (engineRef.current) {
      const newValue = !showRigidbodies
      engineRef.current.setShowRigidbodies(newValue)
      setShowRigidbodies(newValue)
    }
  }, [showRigidbodies])

  return (
    <div className="fixed inset-0 w-full h-full overflow-hidden touch-none">
      <Header stats={stats} />

      {/* Side panel for controls */}
      <div className="absolute left-0 top-1/2 -translate-y-1/2 p-4 pointer-events-none z-10">
        <div className="pointer-events-auto bg-black/80 backdrop-blur-sm rounded-lg p-3 shadow-lg md:block hidden">
          <Button
            size="sm"
            onClick={handleToggleRigidbodies}
            className={`w-full font-medium ${showRigidbodies
                ? "bg-yellow-500 text-black hover:bg-yellow-600"
                : "bg-gray-700 text-white hover:bg-gray-600"
              }`}
          >
            {showRigidbodies ? "Hide Rigidbodies" : "Show Rigidbodies"}
          </Button>
        </div>
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
