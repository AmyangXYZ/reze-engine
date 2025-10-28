"use client"

import { Engine } from "@/lib/engine"
import { useCallback, useEffect, useRef } from "react";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)


  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      const engine = new Engine(canvasRef.current)
      engineRef.current = engine
      await engine.init()
      engine.render()
    }
  }, [canvasRef])

  useEffect(() => {
    initEngine()
  }, [initEngine])

  return (
    <div className="flex min-h-screen items-center justify-center">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-center gap-10">
        <h1 className="text-3xl font-bold">Reze Engine</h1>
        <canvas ref={canvasRef} className="w-full h-full border-2 border-cyan-500" />
      </main>
    </div>
  );
}
