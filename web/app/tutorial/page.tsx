import Header from "@/components/header"
import Canvas0 from "./canvas/canvas0"
import Canvas1 from "./canvas/canvas1"
import Link from "next/link"
import Canvas2 from "./canvas/canvas2"
import Code from "@/components/code"
import Inline from "@/components/inline"
import Image from "next/image"
import Canvas3 from "./canvas/canvas3"
import TableOfContents from "@/components/table-of-contents"

export const metadata = {
  title: "How to render an anime character with WebGPU",
  description: "Reze Engine: WebGPU Engine Tutorial",
  keywords: ["WebGPU", "Engine", "Tutorial", "tutorial", "MMD"],
}

const REPO_URL = "https://github.com/AmyangXYZ/reze-engine/tree/master/web/app/tutorial"

export default function Tutorial() {
  return (
    <div className="flex flex-row justify-center w-full px-8 py-4">
      <Header stats={null} />
      <div className="flex flex-row items-start justify-center w-full max-w-7xl gap-8 mt-12 pb-20">
        <div className="w-64"></div>

        <div className="flex flex-col items-center justify-start max-w-3xl w-full h-full gap-10">
          <h1 className="scroll-m-20 text-center text-3xl font-extrabold tracking-tight text-balance">
            How to Render an Anime Character with WebGPU
          </h1>
          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <p className="leading-7">
              This is a tutorial for developers who want to learn WebGPU or dive deeper after playing with high-level
              frameworks like three.js or babylon.js, but don&apos;t know where to start beyond the simple triangle
              example. This tutorial covers the core pipeline for rendering anime characters with WebGPU: geometry
              rendering, skinning, material and texture handling, bone attachment, and animation. We focus on concepts
              and workflow rather than implementation details—matrix math, shader programming, and model parsing are
              handled by standard code you can generate with AI tools. By the end, you&apos;ll understand how the pieces
              fit together and can build your own rendering engine like the Reze Engine. Full source code for each
              example is available{" "}
              <Link href={REPO_URL} className="text-blue-400" target="_blank">
                here
              </Link>
              .
            </p>
            <Image src="/image-banner.png" alt="img" width={1000} height={1000} />
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Engine v0: Your First Triangle
            </h2>
            <p className="leading-7">
              Here we walk through the Hello World of WebGPU programming: rendering a triangle. Think of the GPU as a
              separate computer with its own memory and instruction set. Unlike JavaScript where you pass data directly
              to functions, working with the GPU involves cross-boundary communication—you need to be explicit about:
            </p>
            <ul className="my-2 ml-6 list-disc [&>li]:mt-2">
              <li>
                <span className="font-semibold">The data to process</span>: vertices
              </li>
              <li>
                <span className="font-semibold">Where to get the data from</span>: buffer
              </li>
              <li>
                <span className="font-semibold">How to process it</span>: shaders and pipeline
              </li>
              <li>
                <span className="font-semibold">The main entry point</span>: render pass
              </li>
            </ul>
            <p className="leading-7">
              Let&apos;s look at the first Engine class{" "}
              <Link href={`${REPO_URL}/engines/v0.ts`} target="_blank" className="text-blue-400">
                engines/v0.ts
              </Link>
              . The code follows the standard WebGPU initialization pattern. First, we request a GPU device and set up a
              rendering context on the canvas. Then we allocate a GPU buffer and write the positions of our 3 vertices
              into it using <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">writeBuffer</code>. Next, we
              define shaders: the vertex shader processes each vertex, and the fragment shader determines the color of
              each pixel. We bundle these shaders with metadata about the buffer layout into a pipeline. Finally, we
              create a render pass that executes the pipeline and produces the triangle on screen.
            </p>

            <div className="w-full h-full items-center justify-center flex mt-2">
              <Canvas0 />
            </div>
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Engine v1: Add a Camera and Make it 3D
            </h2>
            <p className="leading-7">
              The first example draws a single static frame. To make it 3D, we need two things: a camera and a render
              loop that generates continuous frames. The camera isn&apos;t a 3D object—it&apos;s a pair of
              transformation matrices (view and projection) that convert 3D world coordinates into 2D screen
              coordinates, creating the illusion of depth. Unlike in three.js or babylon.js, WebGPU doesn&apos;t have a
              built-in camera object, so we manage these matrices ourselves.{" "}
            </p>

            <p className="leading-7">
              Here&apos;s the camera class we use throughout the tutorial and in the Reze Engine:{" "}
              <Link href={`${REPO_URL}/lib/camera.ts`} target="_blank" className="text-blue-400">
                lib/camera.ts
              </Link>
              . The implementation details aren&apos;t important (throw to AI)—just know that it calculates view and
              projection matrices that update in response to mouse events (movements, zooming, and panning).{" "}
            </p>

            <p className="leading-7">
              Now look at the second Engine class{" "}
              <Link href={`${REPO_URL}/engines/v1.ts`} target="_blank" className="text-blue-400">
                engines/v1.ts
              </Link>
              . The key change is in the vertex shader, where we multiply each vertex position by the camera matrices:{" "}
            </p>

            <Code language="wgsl">
              {`@vertex
fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
  return camera.projection * camera.view * vec4f(position, 0.0, 1.0);
}            `}
            </Code>

            <div className="w-full h-full items-center justify-center flex mt-2">
              <Canvas1 />
            </div>

            <p className="leading-7">
              The interesting part is how we get these matrices from the CPU (JavaScript) to the GPU (shader). This is
              done through a <span className="font-semibold">uniform buffer</span>—essentially a chunk of GPU memory
              that acts like a global variable accessible to all shaders in a pipeline. First, we write the camera data
              to the buffer:{" "}
            </p>

            <Code language="typescript">
              {`this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)`}
            </Code>

            <p className="leading-7">
              Next, we create a bind group that tells the GPU where to find this buffer, and attach it to the render
              pass:
            </p>

            <Code language="typescript">
              {`this.bindGroup = this.device.createBindGroup({
  label: "bind group layout",
  layout: this.pipeline.getBindGroupLayout(0),
  entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
})`}
            </Code>

            <Code language="typescript">{`pass.setBindGroup(0, this.bindGroup);`}</Code>

            <p className="leading-7">
              Finally, in the shader, we define a struct matching the buffer&apos;s memory layout and bind it to group
              0:
            </p>

            <Code language="wgsl">
              {`struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
  viewPos: vec3f,
  _padding: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;`}
            </Code>

            <p className="leading-7">
              Now the shader can access <Inline>camera.view</Inline> and <Inline>camera.projection</Inline> directly.
              This uniform buffer pattern is fundamental in WebGPU—you&apos;ll use it to pass any data from CPU to GPU,
              including lighting parameters, material properties, and transformation matrices.
            </p>
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Engine v2: Render Character Geometry
            </h2>
            <p className="leading-7">
              Now we move from a hardcoded triangle to actual model geometry. We&apos;re using a pre-parsed PMX model
              file—the standard format for MMD (MikuMikuDance) anime characters. The parser itself isn&apos;t covered
              here (any model format works; use AI to generate parsers as needed). What matters is understanding the two
              key data structures: vertices and indices.
            </p>

            <p className="leading-7">
              Each vertex in the model contains three types of data, stored sequentially in memory (this is called{" "}
              <span className="font-semibold">interleaved vertex data</span>):
            </p>
            <ul className="ml-6 list-disc [&>li]:mt-2">
              <li>
                <span className="font-semibold">Position</span>: <Inline>[x, y, z]</Inline> coordinates in 3D space
              </li>
              <li>
                <span className="font-semibold">Normal</span>: <Inline>[nx, ny, nz]</Inline> direction perpendicular to
                the surface (used for lighting)
              </li>
              <li>
                <span className="font-semibold">UV coordinates</span>: <Inline>[u, v]</Inline> texture mapping
                coordinates (tells which part of a texture image to display)
              </li>
            </ul>
            <p className="leading-7">
              That&apos;s 8 floats per vertex (3 + 3 + 2 = 8 floats = 32 bytes). The index buffer specifies which
              vertices form each triangle—instead of duplicating vertex data, we reference existing vertices by their
              indices. This dramatically reduces memory usage.
            </p>

            <p className="leading-7">
              In{" "}
              <Link href={`${REPO_URL}/engines/v2.ts`} target="_blank" className="text-blue-400">
                engines/v2.ts
              </Link>
              , we create both vertex and index buffers from the model data:
            </p>

            <Code language="typescript">
              {`private initVertexBuffers() {
  const vertices = Float32Array.from(this.model.vertices)
  this.vertexBuffer = this.device.createBuffer({
    label: "model vertex buffer",
    size: vertices.byteLength,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
  })
  this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices.buffer)

  // Create index buffer
  const indices = Uint32Array.from(this.model.indices)
  this.indexBuffer = this.device.createBuffer({
    label: "model index buffer",
    size: indices.byteLength,
    usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
  })
  this.device.queue.writeBuffer(this.indexBuffer, 0, indices.buffer)
}
`}
            </Code>

            <p className="leading-7">
              The pipeline configuration changes in two ways. First, we update the vertex buffer layout to match our
              32-byte stride. The shader only needs position data for now (we&apos;ll use normals and UVs in later
              steps), but the GPU must skip the correct number of bytes to read each position:
            </p>

            <Code language="typescript">
              {`vertex: {
  module: this.shaderModule,
  buffers: [{
    arrayStride: 8 * 4, // 32 bytes: skip position + normal + UV for each vertex
    attributes: [{
      shaderLocation: 0,
      offset: 0, // position starts at byte 0
      format: "float32x3" // read 3 floats for position
    }]
  }]
}`}
            </Code>

            <p className="leading-7">
              Second, we use indexed drawing instead of direct drawing. The render pass now calls{" "}
              <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">drawIndexed</code> and specifies the index
              buffer:
            </p>

            <Code language="typescript">
              {`pass.setVertexBuffer(0, this.vertexBuffer)
pass.setIndexBuffer(this.indexBuffer, "uint32")
pass.drawIndexed(this.model.indices.length) // draw all triangles using indices`}
            </Code>

            <p className="leading-7">
              The result is a red shape of the character. Without textures or lighting (coming next), we see only the
              raw geometry. But this is a major milestone—we&apos;ve gone from 3 hardcoded vertices to rendering a
              complex model with thousands of triangles.
            </p>

            <div className="w-full h-full items-center justify-center flex mt-2">
              <Canvas2 />
            </div>
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Engine v3: Textures and Materials
            </h2>
            <p className="leading-7">
              Now we add textures to the character. Textures are images that are mapped onto the model&apos;s surface to
              give it color and detail. In WebGPU, textures are created from image data and can be sampled in the
              fragment shader to determine the color of each pixel.
            </p>

            <div className="w-full h-full items-center justify-center flex mt-2">
              <Canvas3 />
            </div>
          </section>
        </div>
        <div className="w-64 sticky top-12 self-start ">
          <TableOfContents />
        </div>
      </div>
    </div>
  )
}
