import Header from "@/components/header"
import Canvas0 from "./canvas/canvas0"
import Canvas1 from "./canvas/canvas1"
import Link from "next/link"

export const metadata = {
  title: "How to render an anime character with WebGPU",
  description: "Reze Engine: WebGPU Engine Tutorial",
  keywords: ["WebGPU", "Engine", "Tutorial", "tutorial", "MMD"],
}


const REPO_URL = "https://github.com/AmyangXYZ/reze-engine/tree/master/web/app/tutorial"

export default function Tutorial() {
  return (
    <div className="flex flex-col items-center w-full px-8 py-4">
      <Header stats={null} />
      <div className="flex flex-col items-center justify-start max-w-3xl w-full h-full mt-12 gap-8">
        <h1 className="scroll-m-20 text-center text-3xl font-extrabold tracking-tight text-balance">
          How to Render an Anime Character with WebGPU
        </h1>
        <p className="leading-7">
          This is a tutorial for developers who want to learn WebGPU or dive deeper after playing with high-level frameworks like three.js or babylon.js, but don&apos;t know where to start beyond the simple triangle example. This tutorial covers the core pipeline for rendering anime characters with WebGPU: geometry rendering, skinning, material and texture handling, bone attachment, and animation. We focus on concepts and workflow
          rather than implementation details—matrix math, shader programming, and model parsing are handled by standard
          code you can generate with AI tools. By the end, you&apos;ll understand how the pieces fit together and can
          build your own rendering engine like the Reze Engine. Full source code for each example is available{" "}
          <Link
            href={REPO_URL}
            className="text-blue-400"
            target="_blank"
          >
            here
          </Link>
          .
        </p>

        <section className="flex flex-col items-start justify-start gap-4 w-full">
          <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
            Canvas 0: Your First Triangle
          </h2>
          <p className="leading-7">Here we walk through the Hello World of WebGPU programming: rendering a triangle. Think of the GPU as a separate computer with its own memory and instruction set. Unlike JavaScript where you pass data directly to functions, working with the GPU involves cross-boundary communication—you need to be explicit about:</p>
          <ul className="my-2 ml-6 list-disc [&>li]:mt-2">
            <li><span className="font-semibold">The data to process</span>: vertices</li>
            <li><span className="font-semibold">Where to get the data from</span>: buffer</li>
            <li><span className="font-semibold">How to process it</span>: shaders and pipeline</li>
            <li><span className="font-semibold">The main entry point</span>: render pass</li>
          </ul>
          <p className="leading-7">Let&apos;s look at the first Engine class <Link href={`${REPO_URL}/engines/v0.ts`} target="_blank" className="text-blue-400">engines/v0.ts</Link>. The code follows the standard WebGPU initialization pattern. First, we request a GPU device and set up a rendering context on the canvas—this establishes the connection and configures where the output goes. Then we allocate a GPU buffer and write the positions of our 3 vertices into it. The buffer lives in GPU memory, not JavaScript&apos;s heap, so we explicitly marshal data across that boundary using <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">writeBuffer</code>. Next, we define shaders: the vertex shader processes each vertex, and the fragment shader determines the color of each pixel. We bundle these shaders with metadata about the buffer layout into a pipeline—this is where the GPU validates and optimizes your program. Finally, we create a render pass that executes the pipeline and produces the triangle on screen.</p>

          <div className="w-full h-full items-center justify-center flex mt-2">
            <Canvas0 />
          </div>
        </section>

        <section className="flex flex-col items-start justify-start gap-4 w-full">
          <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
            Canvas 1: Add a Camera and Make it 3D
          </h2>
          <p className="leading-7">The first example draws a single static frame. To make it 3D, we need two things: a camera and a render loop that generates continuous frames. The camera isn&apos;t a 3D object—it&apos;s a pair of transformation matrices (view and projection) that convert 3D world coordinates into 2D screen coordinates, creating the illusion of depth. Unlike in three.js or babylon.js, WebGPU doesn&apos;t have a built-in camera object, so we manage these matrices ourselves. </p>

          <p className="leading-7">Here&apos;s the camera class we use throughout the tutorial and in the Reze Engine: <Link href={`${REPO_URL}/lib/camera.ts`} target="_blank" className="text-blue-400">lib/camera.ts</Link>. The implementation details aren&apos;t important (throw to AI)—just know that it calculates view and projection matrices that update in response to mouse events (movements, zooming, and panning). </p>

          <p className="leading-7">Now look at the second Engine class <Link href={`${REPO_URL}/engines/v1.ts`} target="_blank" className="text-blue-400">engines/v1.ts</Link>. The key change is in the vertex shader, where we multiply each vertex position by the camera matrices: </p>

          <pre className="bg-zinc-800 px-4 py-2 rounded-md w-full overflow-x-auto">
            <code className="relative rounded font-mono text-sm font-semibold">
              {`@vertex
fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
  return camera.projection * camera.view * vec4f(position, 0.0, 1.0);
}            `}
            </code>
          </pre>

          <div className="w-full h-full items-center justify-center flex mt-2">
            <Canvas1 />
          </div>

          <p className="leading-7">The interesting part is how we get these matrices from the CPU (JavaScript) to the GPU (shader). This is done through a <span className="font-semibold">uniform buffer</span>—essentially a chunk of GPU memory that acts like a global variable accessible to all shaders in a pipeline. First, we write the camera data to the buffer: </p>

          <pre className="bg-zinc-800 px-4 py-2 rounded-md w-full overflow-x-auto">
            <code className="relative rounded font-mono text-sm font-semibold">
              {`this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)`}
            </code>
          </pre>

          <p className="leading-7">Next, we create a bind group that tells the GPU where to find this buffer, and attach it to the render pass:</p>

          <pre className="bg-zinc-800 px-4 py-2 rounded-md w-full overflow-x-auto">
            <code className="relative rounded font-mono text-sm font-semibold">
              {`this.bindGroup = this.device.createBindGroup({
  label: "bind group layout",
  layout: this.pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
})`}
            </code>
          </pre>

          <pre className="bg-zinc-800 px-4 py-2 rounded-md w-full overflow-x-auto">
            <code className="relative rounded font-mono text-sm font-semibold">
              {`pass.setBindGroup(0, this.bindGroup);`}
            </code>
          </pre>

          <p className="leading-7">Finally, in the shader, we define a struct matching the buffer&apos;s memory layout and bind it to group 0:</p>

          <pre className="bg-zinc-800 px-4 py-2 rounded-md w-full overflow-x-auto">
            <code className="relative rounded font-mono text-sm font-semibold">
              {`struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
  viewPos: vec3f,
  _padding: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;`}
            </code>
          </pre>

          <p className="leading-7">Now the shader can access <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">camera.view</code> and <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">camera.projection</code> directly. This uniform buffer pattern is fundamental in WebGPU—you&apos;ll use it to pass any data from CPU to GPU, including lighting parameters, material properties, and transformation matrices.</p>



        </section>
      </div>
    </div>
  )
}
