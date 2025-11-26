"use client"

import Header from "@/components/header"
import { Engine, EngineStats, Quat, Vec3 } from "reze-engine"
import { useCallback, useEffect, useRef, useState } from "react"
import Loading from "@/components/loading"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Music, VolumeX } from "lucide-react"

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<EngineStats | null>(null)
  const [isMuted, setIsMuted] = useState(false)
  const audioStartedRef = useRef(false)

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
          rimLightIntensity: 0.4,
          cameraDistance: 13.5,
          cameraTarget: new Vec3(0, 17.2, 0),
        })
        engineRef.current = engine
        await engine.init()
        await engine.loadModel("/models/塞尔凯特2/塞尔凯特2.pmx")
        await engine.loadAnimation("/animations/pool.vmd")

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
        await new Promise(resolve => requestAnimationFrame(resolve))
        engine.playAnimation()

        // Attempt to autoplay audio after model is rendered and animation starts
        // This will fail silently if browser blocks autoplay, and will start on first user interaction
        setTimeout(() => {
          if (audioRef.current && !audioStartedRef.current) {
            audioStartedRef.current = true
            audioRef.current.currentTime = 22
            audioRef.current.volume = 1.0
            audioRef.current.play().catch(() => {
              // Autoplay blocked - will start on user interaction
              audioStartedRef.current = false
            })
          }
        }, 100)
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
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
    }
  }, [initEngine])

  // Start audio playback from 22 seconds (called on first user interaction)
  const startAudio = useCallback(() => {
    if (audioRef.current && !audioStartedRef.current) {
      audioStartedRef.current = true
      audioRef.current.currentTime = 22
      audioRef.current.volume = 1.0
      audioRef.current.play().catch(() => {
        // Silently handle autoplay restrictions
        audioStartedRef.current = false
      })
    }
  }, [])

  // Handle mute/unmute toggle
  const toggleMute = useCallback(() => {
    // Start audio on first interaction if not started yet
    startAudio()

    if (audioRef.current) {
      audioRef.current.muted = !isMuted
      setIsMuted(!isMuted)
    }
  }, [isMuted, startAudio])

  // Start audio on first user interaction (click or touch)
  useEffect(() => {
    if (loading) return

    const handleFirstInteraction = () => {
      startAudio()
      // Remove listeners after first interaction
      document.removeEventListener("click", handleFirstInteraction)
      document.removeEventListener("touchstart", handleFirstInteraction)
    }

    document.addEventListener("click", handleFirstInteraction, { once: true })
    document.addEventListener("touchstart", handleFirstInteraction, { once: true })

    return () => {
      document.removeEventListener("click", handleFirstInteraction)
      document.removeEventListener("touchstart", handleFirstInteraction)
    }
  }, [loading, startAudio])

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
      <div className="absolute inset-0 w-full h-full flex justify-center items-center">
        <Image src="/pool.jpeg" alt="Reze Engine" width={1000} height={1000} className="w-full h-full md:h-auto touch-none z-0 object-cover" />
      </div>
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none z-1" />

      {/* Audio element */}
      <audio
        ref={audioRef}
        src="/in the pool.mp3"
        loop
        preload="auto"
        className="hidden"
      />

      {/* Floating mute button */}
      <Button
        onClick={toggleMute}
        variant="secondary"
        size="icon"
        className="fixed bottom-6 right-6 z-50 rounded-full shadow-lg hover:shadow-xl transition-all"
        aria-label={isMuted ? "Unmute audio" : "Mute audio"}
      >
        {isMuted ? <VolumeX className="h-6 w-6" /> : <Music className="h-6 w-6" />}
      </Button>
    </div>
  )
}
