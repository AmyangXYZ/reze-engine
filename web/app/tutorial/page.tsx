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
import Canvas3_2 from "./canvas/canvas3_2"

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

        <div className="flex flex-col items-center justify-start max-w-2xl w-full h-full gap-10">
          <h1 className="scroll-m-20 text-center text-3xl font-extrabold tracking-tight text-balance">
            How to Render an Anime Character with WebGPU
          </h1>
          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <p className="leading-7">
              This is a tutorial for developers who want to learn WebGPU or dive deeper after playing with high-level
              frameworks like three.js or babylon.js, but don&apos;t know where to start beyond the simple triangle
              example. This tutorial covers the core pipeline for rendering anime characters with WebGPU: geometry
              rendering, material and texture handling, bone and skinning, and animation. We focus on concepts and
              workflow rather than implementation details—matrix math, shader programming, and model parsing are handled
              by standard code you can generate with AI tools. By the end, you&apos;ll understand how the pieces fit
              together and can build your own rendering engine like the Reze Engine. Full source code for each example
              is available{" "}
              <Link href={REPO_URL} className="text-blue-400" target="_blank">
                here
              </Link>
              .
            </p>
            <Image src="/image-banner.png" alt="img" width={1000} height={1000} loading="eager" />
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
              . The code follows the standard WebGPU initialization pattern:
            </p>
            <ol className="ml-6 list-decimal [&>li]:mt-2">
              <li>Request a GPU device and set up a rendering context on the canvas</li>
              <li>
                Allocate a GPU buffer and write the positions of our 3 vertices into it using{" "}
                <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">writeBuffer</code>
              </li>
              <li>
                Define shaders: the vertex shader processes each vertex, and the fragment shader determines the color of
                each pixel
              </li>
              <li>Bundle these shaders with metadata about the buffer layout into a pipeline</li>
              <li>Create a render pass that executes the pipeline and produces the triangle on screen</li>
            </ol>

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
              . To pass camera matrices from JavaScript to the shader, we use a{" "}
              <span className="font-semibold">uniform buffer</span>—a chunk of GPU memory that acts like a global
              variable accessible to all shaders. First, we write the camera data to the buffer:
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
              Finally, in the shader, we define a struct matching the buffer&apos;s memory layout:
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
              Now the shader can access <Inline>camera.view</Inline> and <Inline>camera.projection</Inline> directly. In
              the vertex shader, we multiply each vertex position by these matrices:
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
              This uniform buffer pattern is fundamental in WebGPU—you&apos;ll use it to pass any data from CPU to GPU,
              including lighting parameters, material properties, and transformation matrices.
            </p>
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Engine v2: Render Character Geometry
            </h2>
            <p className="leading-7">
              Now we move from a hardcoded triangle to actual model geometry. We&apos;re using a pre-parsed PMX{" "}
              <Link href={`${REPO_URL}/model.json`} target="_blank" className="text-blue-400">
                model data
              </Link>{" "}
              —the standard format for MMD (MikuMikuDance) anime characters. The parser itself isn&apos;t covered here
              (any model format works; use AI to generate parsers as needed). What matters is understanding the two key
              data structures: vertices and indices.
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
              The index buffer specifies which vertices form each triangle—instead of duplicating vertex data, we
              reference existing vertices by their indices. This dramatically reduces memory usage.
            </p>

            <p className="leading-7">
              In{" "}
              <Link href={`${REPO_URL}/engines/v2.ts`} target="_blank" className="text-blue-400">
                engines/v2.ts
              </Link>
              , we create both vertex and index buffers from the model data. Look for the{" "}
              <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">initVertexBuffers</code> method:
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
              The key change is using indexed drawing instead of direct drawing. The render pass now calls{" "}
              <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">drawIndexed</code> and specifies the index
              buffer:
            </p>

            <Code language="typescript">
              {`pass.setVertexBuffer(0, this.vertexBuffer)
pass.setIndexBuffer(this.indexBuffer, "uint32")
pass.drawIndexed(this.model.indices.length) // draw all triangles using indices`}
            </Code>

            <p className="leading-7">
              The result is a red shape of the character. Without textures (coming next), we see only the raw geometry.
              But this is a major milestone—we&apos;ve gone from 3 hardcoded vertices to rendering a complex model with
              thousands of triangles.
            </p>

            <div className="w-full h-full items-center justify-center flex mt-2">
              <Canvas2 />
            </div>
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Engine v3: Material and Texture
            </h2>
            <p className="leading-7">
              Now we add textures to bring color and detail to the character. This introduces two important concepts:{" "}
              <span className="font-semibold">materials</span> and <span className="font-semibold">textures</span>.
            </p>

            <p className="leading-7">
              A <span className="font-semibold">material</span> links a group of vertices (by their indices) and
              specifies which texture and visual parameters to use when drawing those triangles. In a character model, a
              material can be the face, hair, clothes, or other components.
            </p>

            <p className="leading-7">
              A <span className="font-semibold">texture</span> is an image file that contains color data. Each vertex
              has UV coordinates that map it to a location in the texture. The fragment shader samples the texture using
              these coordinates to determine the color for each pixel.
            </p>

            <p className="leading-7">
              In{" "}
              <Link href={`${REPO_URL}/engines/v3.ts`} target="_blank" className="text-blue-400">
                engines/v3.ts
              </Link>
              , we first load texture images and create GPU textures. Look for the{" "}
              <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">initTexture</code> method. We fetch each image
              file, create an <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">ImageBitmap</code>, then
              create a <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">GPUTexture</code> and upload the
              image data:
            </p>

            <Code language="typescript">
              {`const imageBitmap = await createImageBitmap(await response.blob())
const texture = this.device.createTexture({
  size: [imageBitmap.width, imageBitmap.height],
  format: "rgba8unorm",
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
})
this.device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture }, [
  imageBitmap.width,
  imageBitmap.height,
])`}
            </Code>

            <p className="leading-7">
              Next, we create a sampler that defines how the texture should be sampled (filtering, wrapping, etc.):
            </p>

            <Code language="typescript">
              {`this.sampler = this.device.createSampler({
  magFilter: "linear",
  minFilter: "linear",
  addressModeU: "repeat",
  addressModeV: "repeat",
})`}
            </Code>

            <p className="leading-7">
              In the shader, we need to pass UV coordinates from the vertex shader to the fragment shader. We define a{" "}
              <Inline>VertexOutput</Inline> struct to bundle the position and UV together:
            </p>

            <Code language="wgsl">
              {`struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

@vertex
fn vs(@location(2) uv: vec2<f32>) -> VertexOutput {
  var output: VertexOutput;
  output.position = camera.projection * camera.view * vec4f(position, 1.0);
  output.uv = uv;
  return output;
}`}
            </Code>

            <p className="leading-7">
              The fragment shader receives the UV coordinates and samples the texture using{" "}
              <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">textureSample</code>:
            </p>

            <Code language="wgsl">
              {`@fragment
fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
  return vec4<f32>(textureSample(texture, textureSampler, input.uv).rgb, 1.0);
}`}
            </Code>

            <p className="leading-7">
              To bind textures to the shader, we create a bind group for each material with its texture and sampler. We
              add this as a second bind group alongside the camera uniform:
            </p>

            <Code language="typescript">
              {`for (const material of this.model.materials) {
  const textureIndex = material.diffuseTextureIndex
  const materialBindGroup = this.device.createBindGroup({
    layout: this.pipeline.getBindGroupLayout(1),
    entries: [
      { binding: 0, resource: this.textures[textureIndex].createView() },
      { binding: 1, resource: this.sampler },
    ],
  })
  this.materialBindGroups.push(materialBindGroup)
}`}
            </Code>

            <p className="leading-7">
              Finally, we render each material separately. Instead of one <Inline>drawIndexed</Inline> call for the
              entire model, we iterate through materials, set each material&apos;s bind group, and draw its triangles:
            </p>

            <Code language="typescript">
              {`let firstIndex = 0
for (let i = 0; i < this.model.materials.length; i++) {
  const material = this.model.materials[i]
  if (material.vertexCount === 0) continue

  pass.setBindGroup(1, this.materialBindGroups[i])
  pass.drawIndexed(material.vertexCount, 1, firstIndex)
  firstIndex += material.vertexCount
}`}
            </Code>

            <p className="leading-7">The result transforms our red model into a fully textured character.</p>

            <div className="w-full h-full items-center justify-center flex mt-2">
              <Canvas3 />
            </div>

            <p className="leading-7">
              However, you must notice the character appears transparent or you can see through to the back faces. We
              didn&apos;t notice this issue in the previous version because the model was covered by solid red color.
              The fix is surprisingly simple—just three steps: create a depth texture, add it to the render pass, and
              configure the pipeline. No shader changes needed:
            </p>

            <Code language="typescript">
              {`// Create depth texture
this.depthTexture = this.device.createTexture({
  size: [width, height],
  format: "depth24plus",
  usage: GPUTextureUsage.RENDER_ATTACHMENT,
})

// Add to render pass
depthStencilAttachment: {
  view: this.depthTexture.createView(),
  depthClearValue: 1.0,
  depthLoadOp: "clear",
  depthStoreOp: "store",
}

// Add to pipeline
depthStencil: {
  depthWriteEnabled: true,
  depthCompare: "less",
  format: "depth24plus",
}`}
            </Code>

            <p className="leading-7">
              The complete implementation is in{" "}
              <Link href={`${REPO_URL}/engines/v3_2.ts`} target="_blank" className="text-blue-400">
                engines/v3_2.ts
              </Link>
              . With materials, textures, and depth testing in place, we now have a complete static rendering pipeline.
              The character is fully textured and looks solid from any angle.
            </p>

            <div className="w-full h-full items-center justify-center flex mt-2">
              <Canvas3_2 />
            </div>
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Engine v4: Bones and Skinning
            </h2>
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Engine v5: Animation
            </h2>
          </section>
        </div>
        <div className="w-64 sticky top-12 self-start ">
          <TableOfContents />
        </div>
      </div>
    </div>
  )
}
