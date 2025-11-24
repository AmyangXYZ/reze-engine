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
- VMD animation

## Usage

```javascript
export default function MainScene() {
  const canvasRef = useRef < HTMLCanvasElement > null
  const engineRef = useRef < Engine > null
  const [engineError, setEngineError] = (useState < string) | (null > null)

  const initEngine = useCallback(async () => {
    if (canvasRef.current) {
      try {
        const engine = new Engine(canvasRef.current, {
          ambient: 1.0,
          rimLightIntensity: 0.1,
          bloomIntensity: 0.1,
        })
        engineRef.current = engine
        await engine.init()
        await engine.loadModel("/models/塞尔凯特/塞尔凯特.pmx")

        engine.runRenderLoop(() => {})
        setTimeout(() => setModelLoaded(true), 200)
      } catch (error) {
        setEngineError(error instanceof Error ? error.message : "Unknown error")
      }
    }
  }, [])

  useEffect(() => {
    void (async () => {
      initEngine()
    })()

    return () => {
      if (engineRef.current) {
        engineRef.current.dispose()
      }
    }
  }, [initEngine])

  return (
    <div className="w-full h-full flex flex-col md:flex-row">
      <div className="w-full h-full relative">
        {engineError && (
          <div className="text-red-500 z-10 absolute top-0 left-0 w-full h-full flex items-center justify-center text-lg font-medium">
            {engineError}
          </div>
        )}
        <canvas ref={canvasRef} className="w-full h-full z-1" />
      </div>
    </div>
  )
}
```

## Projects Using This Engine

- **[MiKaPo](https://mikapo.vercel.app)** - Online real-time motion capture for MMD using webcam and MediaPipe
- **[Popo](https://popo.love)** - Fine-tuned LLM that generates MMD poses from natural language descriptions
- **[MPL](https://mmd-mpl.vercel.app)** - Semantic motion programming language for scripting MMD animations with intuitive syntax

## Tutorial

Learn WebGPU from scratch by building an anime character renderer in incremental steps. The tutorial covers the complete rendering pipeline from a simple triangle to fully textured, skeletal-animated characters.

[How to Render an Anime Character with WebGPU](https://reze.one/tutorial)
