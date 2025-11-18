import Header from "@/components/header"
import Canvas0 from "./canvas/canvas0"
import Canvas1 from "./canvas/canvas1"

export const metadata = {
  title: "How to render an anime character with WebGPU",
  description: "Reze Engine: WebGPU Engine Tutorial",
  keywords: ["WebGPU", "Engine", "Tutorial", "tutorial", "MMD"],
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
  return (
    <div className="flex flex-col items-center w-full h-full px-8 py-4">
      <Header stats={null} />
      <div className="flex flex-col items-center justify-start max-w-3xl w-full h-full mt-12 gap-8">
        <h1 className="scroll-m-20 text-center text-3xl font-extrabold tracking-tight text-balance">
          How to Render an Anime Character with WebGPU
        </h1>
        <p className="leading-7">
          [WIP] This tutorial covers the core pipeline for rendering anime characters with WebGPU: geometry rendering,
          skinning, material and texture handling, bone attachment and animation. We focus on the concepts and workflow
          rather than implementation details: matrix math, shader programming, and model parsing are handled by standard
          code you can generate with AI tools. By the end, you&apos;ll understand how the pieces fit together and can
          build your own rendering engine. Full source code for each example is available{" "}
          <a
            href="https://github.com/AmyangXYZ/reze-engine/tree/master/web/app/tutorial"
            className="text-blue-500"
            target="_blank"
          >
            here
          </a>
          .
        </p>

        <section className="flex flex-col items-center justify-start gap-4">
          <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
            Canvas 0: Your First Triangle
          </h2>
          <p className="leading-7">Here we walk through the hello world of WebGPU programming: rendering a triangle.</p>
          <Canvas0 />
        </section>

        <section className="flex flex-col items-center justify-start gap-4">
          <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
            Canvas 1: Add a Camera
          </h2>
          <p className="leading-7">Then we add a camera to the scene and make a render loop.</p>
          <Canvas1 />
        </section>
      </div>
    </div>
  )
}
