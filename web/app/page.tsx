"use client"

import Header from "@/components/header"
import { Engine, EngineStats, Quat } from "reze-engine"
import { useCallback, useEffect, useRef, useState } from "react"
import Loading from "@/components/loading"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<EngineStats | null>(null)

  // Model rotation state
  const isDraggingModel = useRef(false)
  const lastMousePos = useRef({ x: 0, y: 0 })
  const modelRotationY = useRef(0) // Current Y-axis rotation in radians
  const rotationSensitivity = 0.002 // Similar to camera angular sensitivity

  // Touch state for mobile
  const isDraggingModelTouch = useRef(false)
  const touchIdentifier = useRef<number | null>(null)
  const lastTouchPos = useRef({ x: 0, y: 0 })

  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      // Initialize engine
      try {
        const engine = new Engine(canvasRef.current, {
          ambient: 1.0,
          bloomIntensity: 0.12,
          rimLightIntensity: 0.3,
        })
        engineRef.current = engine
        await engine.init()
        await engine.loadModel("/models/塞尔凯特2/塞尔凯特2.pmx")
        await engine.loadAnimation("/animations/animation.vmd")

        setLoading(false)

        engine.runRenderLoop(() => {
          setStats(engine.getStats())
        })

        // engine.rotateBones(
        //   ["腰", "首", "右腕", "左腕", "右ひざ"],
        //   [
        //     new Quat(-0.4, -0.3, 0, 1),
        //     new Quat(0.3, -0.3, -0.3, 1),
        //     new Quat(0.3, 0.3, 0.3, 1),
        //     new Quat(-0.3, 0.3, -0.3, 1),
        //     new Quat(-1.0, -0.3, 0.0, 1),
        //   ],
        //   1500
        // )

        // Wait a frame to ensure render loop has started and model is fully initialized
        // This prevents physics explosion when animation starts
        await new Promise((resolve) => requestAnimationFrame(resolve))
        engine.playAnimation()
      } catch (error) {
        setEngineError(error instanceof Error ? error.message : "Unknown error")
      }
    }
  }, [])

  useEffect(() => {
    void (async () => {
      initEngine()
    })()

    // Cleanup on unmount
    return () => {
      if (engineRef.current) {
        engineRef.current.dispose()
      }
    }
  }, [initEngine])

  // Mouse event handlers for model rotation
  // Use capture phase to intercept events before camera handlers
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || loading) return

    const handleMouseDown = (e: MouseEvent) => {
      // Only handle left-click (button 0) and prevent it from reaching camera
      if (e.button === 0) {
        isDraggingModel.current = true
        lastMousePos.current = { x: e.clientX, y: e.clientY }
        e.stopPropagation() // Prevent camera from receiving this event
        e.preventDefault()
      }
    }

    const handleMouseMove = (e: MouseEvent) => {
      if (!isDraggingModel.current || !engineRef.current) return

      // Stop propagation to prevent camera from handling this event
      e.stopPropagation()

      const deltaX = e.clientX - lastMousePos.current.x

      // Update rotation angle (accumulate)
      modelRotationY.current -= deltaX * rotationSensitivity

      // Create quaternion for Y-axis rotation
      const rotationQuat = Quat.fromEuler(0, modelRotationY.current, 0)

      // Rotate the center bone "センター"
      engineRef.current.rotateBones(["センター"], [rotationQuat], 0)

      lastMousePos.current = { x: e.clientX, y: e.clientY }
    }

    const handleMouseUp = (e: MouseEvent) => {
      if (e.button === 0) {
        isDraggingModel.current = false
        e.stopPropagation() // Prevent camera from receiving this event
      }
    }

    // Use capture phase (true) so our handlers run before camera's handlers
    canvas.addEventListener("mousedown", handleMouseDown, { capture: true })
    window.addEventListener("mousemove", handleMouseMove, { capture: true })
    window.addEventListener("mouseup", handleMouseUp, { capture: true })

    return () => {
      canvas.removeEventListener("mousedown", handleMouseDown, { capture: true })
      window.removeEventListener("mousemove", handleMouseMove, { capture: true })
      window.removeEventListener("mouseup", handleMouseUp, { capture: true })
    }
  }, [loading, rotationSensitivity])

  // Touch event handlers for model rotation on mobile
  // Use capture phase to intercept single-finger touch events before camera handlers
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || loading) return

    const handleTouchStart = (e: TouchEvent) => {
      // Only handle single-finger touch and prevent it from reaching camera
      if (e.touches.length === 1) {
        const touch = e.touches[0]
        isDraggingModelTouch.current = true
        touchIdentifier.current = touch.identifier
        lastTouchPos.current = { x: touch.clientX, y: touch.clientY }
        e.stopPropagation() // Prevent camera from receiving this event
        e.preventDefault()
      }
    }

    const handleTouchMove = (e: TouchEvent) => {
      if (!isDraggingModelTouch.current || !engineRef.current || touchIdentifier.current === null) return

      // Find the touch we're tracking
      let touch: Touch | null = null
      for (let i = 0; i < e.touches.length; i++) {
        if (e.touches[i].identifier === touchIdentifier.current) {
          touch = e.touches[i]
          break
        }
      }

      // If our tracked touch is gone or multiple touches, stop
      if (!touch || e.touches.length > 1) {
        isDraggingModelTouch.current = false
        touchIdentifier.current = null
        return
      }

      // Stop propagation to prevent camera from handling this event
      e.stopPropagation()
      e.preventDefault()

      const deltaX = touch.clientX - lastTouchPos.current.x

      // Update rotation angle (accumulate)
      modelRotationY.current -= deltaX * rotationSensitivity

      // Create quaternion for Y-axis rotation
      const rotationQuat = Quat.fromEuler(0, modelRotationY.current, 0)

      // Rotate the center bone "センター"
      engineRef.current.rotateBones(["センター"], [rotationQuat], 0)

      lastTouchPos.current = { x: touch.clientX, y: touch.clientY }
    }

    const handleTouchEnd = (e: TouchEvent) => {
      // Check if our tracked touch ended
      if (touchIdentifier.current !== null) {
        let touchStillActive = false
        for (let i = 0; i < e.touches.length; i++) {
          if (e.touches[i].identifier === touchIdentifier.current) {
            touchStillActive = true
            break
          }
        }

        if (!touchStillActive) {
          isDraggingModelTouch.current = false
          touchIdentifier.current = null
          e.stopPropagation() // Prevent camera from receiving this event
        }
      }

      // If all touches ended, reset state
      if (e.touches.length === 0) {
        isDraggingModelTouch.current = false
        touchIdentifier.current = null
        e.stopPropagation()
      }
    }

    // Use capture phase (true) so our handlers run before camera's handlers
    canvas.addEventListener("touchstart", handleTouchStart, { capture: true, passive: false })
    window.addEventListener("touchmove", handleTouchMove, { capture: true, passive: false })
    window.addEventListener("touchend", handleTouchEnd, { capture: true })

    return () => {
      canvas.removeEventListener("touchstart", handleTouchStart, { capture: true })
      window.removeEventListener("touchmove", handleTouchMove, { capture: true })
      window.removeEventListener("touchend", handleTouchEnd, { capture: true })
    }
  }, [loading, rotationSensitivity])

  return (
    <div className="fixed inset-0 w-full h-full overflow-hidden touch-none">
      <Header stats={stats} />

      {engineError && (
        <div className="absolute inset-0 w-full h-full flex items-center justify-center text-white p-6">
          Engine Error: {engineError}
        </div>
      )}
      {loading && !engineError && <Loading loading={loading} />}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none z-1" />
    </div>
  )
}
