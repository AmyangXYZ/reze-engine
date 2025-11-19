"use client"

import { useCallback, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import { EngineV3_2 } from "../engines/v3_2"

export default function Canvas3_2() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [rendered, setRendered] = useState(false)
  const [engineError, setEngineError] = useState<string | null>(null)
  const engineRef = useRef<EngineV3_2 | null>(null)

  const render = useCallback(async () => {
    if (!canvasRef.current || rendered) return
    try {
      const engine = new EngineV3_2(canvasRef.current)
      await engine.init()
      engineRef.current = engine
      engine.runRenderLoop()
      setRendered(true)
    } catch (error) {
      setEngineError(error instanceof Error ? error.message : "Unknown error")
    }
  }, [rendered])

  const stop = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.dispose()
      setRendered(false)
    }
  }, [])

  return (
    <div className="flex flex-col items-center justify-center relative w-full max-w-md">
      {engineError && (
        <div className="absolute inset-0 w-full h-full flex items-center justify-center max-w-sm mx-auto p-4">
          Engine Error: {engineError}
        </div>
      )}
      {!rendered && !engineError && (
        <Button onClick={render} className="absolute max-w-xs mx-auto z-10 flex my-auto">
          Render
        </Button>
      )}
      {rendered && !engineError && (
        <Button onClick={stop} className="absolute top-2 right-2 max-w-xs mx-auto z-10 flex my-auto" variant="outline">
          Stop
        </Button>
      )}
      <canvas ref={canvasRef} className="w-full h-[640px] border border-muted-foreground p-4" />
    </div>
  )
}
