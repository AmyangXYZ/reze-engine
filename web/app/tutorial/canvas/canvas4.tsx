"use client"

import { useCallback, useRef, useState, useMemo, useEffect } from "react"
import { EngineV4 } from "../engines/v4"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Quat } from "../lib/math"

export default function Canvas4() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [rendered, setRendered] = useState(false)
  const [engineError, setEngineError] = useState<string | null>(null)
  const engineRef = useRef<EngineV4 | null>(null)

  // Bone rotation states
  const [neckRot, setNeckRot] = useState(new Quat(0.0, 0.0, 0.0, 1.0))
  const [waistRot, setWaistRot] = useState(new Quat(0.0, 0.3, 0.0, 1.0))

  const updateBoneRotation = useCallback((boneName: string, quat: Quat) => {
    if (engineRef.current) {
      engineRef.current.rotateBone(boneName, quat)
    }
  }, [])

  const render = useCallback(async () => {
    if (!canvasRef.current || rendered) return
    try {
      const engine = new EngineV4(canvasRef.current)
      await engine.init()
      engineRef.current = engine
      engine.runRenderLoop()

      // Apply initial rotations
      engine.rotateBone("首", neckRot)
      engine.rotateBone("腰", waistRot)

      setRendered(true)
    } catch (error) {
      setEngineError(error instanceof Error ? error.message : "Unknown error")
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rendered])

  const stop = useCallback(() => {
    setWaistRot(new Quat(0.0, 0.0, 0.0, 1.0))
    setNeckRot(new Quat(0.0, 0.0, 0.0, 1.0))

    if (engineRef.current) {
      engineRef.current.dispose()
      setRendered(false)
    }
  }, [])

  const BoneSliders = ({ label, boneName, rotation, setRotation }: {
    label: string
    boneName: string
    rotation: Quat
    setRotation: (rot: Quat) => void
  }) => {
    const [localRot, setLocalRot] = useState(rotation)
    const normalized = useMemo(() => localRot.normalize(), [localRot])

    // Sync localRot with parent rotation prop
    useEffect(() => {
      setLocalRot(rotation)
    }, [rotation])

    const handleChange = (axis: 'x' | 'y' | 'z' | 'w', value: number) => {
      const newRot = new Quat(
        axis === 'x' ? value : localRot.x,
        axis === 'y' ? value : localRot.y,
        axis === 'z' ? value : localRot.z,
        axis === 'w' ? value : localRot.w
      )
      setLocalRot(newRot)
      updateBoneRotation(boneName, newRot)
    }

    const handleCommit = () => {
      // Normalize and update parent state so slider reflects normalized values
      const normalizedRot = localRot.normalize()
      setRotation(normalizedRot)
    }

    return (
      <div className="flex-1 space-y-1">
        <h3 className="text-xs font-semibold mb-1">{label}</h3>
        <div className="space-y-0.5">
          <div className="flex items-center gap-1">
            <span className="text-[11px] w-5">x:</span>
            <Slider
              value={[localRot.x]}
              onValueChange={(v) => handleChange('x', v[0])}
              onValueCommit={handleCommit}
              min={-2}
              max={2}
              step={0.01}
              className="flex-1 [&>span:first-child]:h-1 [&_[role=slider]]:h-2.5 [&_[role=slider]]:w-2.5"
            />
            <span className="text-[11px] w-14 text-right text-muted-foreground">{normalized.x.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[11px] w-5">y:</span>
            <Slider
              value={[localRot.y]}
              onValueChange={(v) => handleChange('y', v[0])}
              onValueCommit={handleCommit}
              min={-2}
              max={2}
              step={0.01}
              className="flex-1 [&>span:first-child]:h-1 [&_[role=slider]]:h-2.5 [&_[role=slider]]:w-2.5"
            />
            <span className="text-[11px] w-14 text-right text-muted-foreground">{normalized.y.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[11px] w-5">z:</span>
            <Slider
              value={[localRot.z]}
              onValueChange={(v) => handleChange('z', v[0])}
              onValueCommit={handleCommit}
              min={-2}
              max={2}
              step={0.01}
              className="flex-1 [&>span:first-child]:h-1 [&_[role=slider]]:h-2.5 [&_[role=slider]]:w-2.5"
            />
            <span className="text-[11px] w-14 text-right text-muted-foreground">{normalized.z.toFixed(2)}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[11px] w-5">w:</span>
            <Slider
              value={[localRot.w]}
              onValueChange={(v) => handleChange('w', v[0])}
              onValueCommit={handleCommit}
              min={-2}
              max={2}
              step={0.01}
              className="flex-1 [&>span:first-child]:h-1 [&_[role=slider]]:h-2.5 [&_[role=slider]]:w-2.5"
            />
            <span className="text-[11px] w-14 text-right text-muted-foreground">{normalized.w.toFixed(2)}</span>
          </div>
        </div>
      </div>
    )
  }

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

      <canvas ref={canvasRef} className="w-full h-[640px] border border-muted-foreground p-4" />
      {rendered && !engineError && (
        <>
          <div className="w-full flex gap-4 mt-2">
            <BoneSliders label="腰 (Waist)" boneName="腰" rotation={waistRot} setRotation={setWaistRot} />
            <BoneSliders label="首 (Neck)" boneName="首" rotation={neckRot} setRotation={setNeckRot} />
          </div>
          <Button onClick={stop} className="absolute top-2 right-2 max-w-xs mx-auto z-10 flex my-auto" variant="outline">
            Stop
          </Button>
        </>
      )}
    </div>
  )
}
