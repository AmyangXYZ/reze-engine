# Reze Engine

A lightweight engine built with WebGPU and TypeScript for real-time 3D anime character MMD model rendering.

## Features

- Physics
- Alpha blending
- Post alpha eye rendering
- Rim lighting
- Bloom
- Outlines
- Toon shading with directional lights
- MSAA 4x anti-aliasing
- GPU-accelerated skinning
- Bone rotation api

## Usage

```typescript
export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const engineRef = useRef<Engine | null>(null)
  const [engineError, setEngineError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [stats, setStats] = useState<EngineStats>({
    fps: 0,
    frameTime: 0,
    gpuMemory: 0,
  })
  const [progress, setProgress] = useState(0)

  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      // Initialize engine
      try {
        const engine = new Engine(canvasRef.current)
        engineRef.current = engine
        await engine.init()
        await engine.loadModel("/models/塞尔凯特/塞尔凯特.pmx")
        setLoading(false)

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
      initEngine()
    })()

    // Cleanup on unmount
    return () => {
      if (engineRef.current) {
        engineRef.current.dispose()
      }
    }
  }, [initEngine])

  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            return 0
          }
          return prev + 1
        })
      }, 50)

      return () => clearInterval(interval)
    }
  }, [loading])

  return (
    <div
      className="fixed inset-0 w-full h-full overflow-hidden touch-none"
      style={{
        background:
          "radial-gradient(ellipse at center, rgba(35, 35, 45, 0.8) 0%, rgba(35, 35, 45, 0.8) 8%, rgba(8, 8, 12, 0.95) 65%, rgba(0, 0, 0, 1) 100%)",
      }}
    >
      <Header stats={stats} />

      {engineError && (
        <div className="absolute inset-0 w-full h-full flex items-center justify-center text-white p-6">
          Engine Error: {engineError}
        </div>
      )}
      {loading && !engineError && (
        <div className="absolute inset-0 max-w-xs mx-auto w-full h-full flex items-center justify-center text-white p-6">
          <Progress value={progress} className="rounded-none" />
        </div>
      )}
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full touch-none z-1" />
    </div>
  )
}
```
