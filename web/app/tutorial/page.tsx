import Header from "@/components/header"
import { useRef } from "react"

export const metadata = {
  title: "How to render an anime character with WebGPU",
  description: "Reze Engine: WebGPU Engine Tutorial",
  keywords: ["WebGPU", "Engine", "Tutorial", "tutorial", "MMD",],
}

// interface Model {
//   vertices: Float32Array | number[] // Can be TypedArray or regular array from JSON
//   indices: Uint32Array | number[] // Can be TypedArray or regular array from JSON
//   materials: { name: string, diffuseTextureIndex: number }[]
//   textures: { name: string, path: string }[]
//   bones: { name: string, parentIndex: number, bindTranslation: Vec3 }[]
//   skinning: { joints: Uint16Array | number[], weights: Uint8Array | number[] }
// }

export default function Tutorial() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  return (
    <div className="flex flex-col items-center w-full h-full">
      <Header stats={null} />
      <div className="flex flex-col items-center justify-start w-full h-full gap-4 p-4 mt-10">
        <h1 className="scroll-m-20 text-center text-4xl font-extrabold tracking-tight text-balance">
          WebGPU Engine Tutorial: Rendering Vertices
        </h1>
        <p className="text-center text-muted-foreground max-w-2xl">
          Step 1: Minimal WebGPU setup to render vertices with a simple shader
        </p>
        <canvas ref={canvasRef} className="w-[800px] h-[600px] border border-gray-300" />
      </div>
    </div>
  )
}