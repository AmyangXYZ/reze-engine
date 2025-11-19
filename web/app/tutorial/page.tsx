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
import Canvas4 from "./canvas/canvas4"

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
      <div className="flex flex-row items-start justify-center w-full max-w-7xl gap-8 mt-12 pb-50">
        <div className="w-64"></div>

        <div className="flex flex-col items-center justify-start max-w-2xl w-full h-full gap-10">
          <h1 className="scroll-m-20 text-center text-3xl font-extrabold tracking-tight text-balance">
            How to Render an Anime Character with WebGPU
          </h1>
          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <p className="leading-7">
              You&apos;ve tried three.js or babylon.js and wanted to understand what&apos;s happening under the hood. You
              looked at WebGPU tutorials, saw the &quot;Hello Triangle&quot; example, but still don&apos;t know how to
              render a real 3D character from scratch. This tutorial bridges that gap. In five incremental steps,
              you&apos;ll go from a simple triangle to a fully textured, animated anime character—learning the complete
              rendering pipeline along the way: geometry buffers, camera transforms, materials and textures, skeletal
              animation, and the render loop that ties it all together.
            </p>
            <p className="leading-7">
              We focus on understanding the GPU pipeline itself, not implementation details. The real challenge
              isn&apos;t the math or the shaders—AI can generate those. The challenge is learning a different mental
              model: you need to know <span className="font-semibold">what components exist</span> (buffers, bind
              groups, pipelines, render passes), <span className="font-semibold">how they connect</span> (which data
              goes where, in what order), and <span className="font-semibold">why they&apos;re necessary</span> (when to
              use uniform vs storage buffers, how textures flow from CPU to GPU). By the end, you&apos;ll have built a
              working renderer and understand the architecture behind engines like the{" "}
              <Link href="/" className="text-blue-400">
                Reze Engine
              </Link>
              . Full source code is available{" "}
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
              Let&apos;s start with the classic Hello Triangle—not because it&apos;s exciting, but because it&apos;s the simplest
              example that shows every essential component of the WebGPU pipeline. Once you understand how these pieces
              connect here, scaling up to complex models is just adding more data, not learning new concepts.
            </p>
            <p className="leading-7">
              Think of the GPU as a separate computer with its own memory and instruction set. Unlike JavaScript where
              you pass data directly to functions, working with the GPU involves cross-boundary communication—you need to
              be explicit about:
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
              —the standard format for MMD (MikuMikuDance) anime characters. MMD is widely used for anime-style character
              modeling, with massive fan communities creating models from popular games like Genshin Impact (原神) and Aether Gazer (深空之眼). The parser itself isn&apos;t covered here (any model format
              works; use AI to generate parsers as needed). What matters is understanding the two key data structures:
              vertices and indices.
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
              However, you might notice the character appears transparent or you can see through to the back faces. This happens because without depth testing, the GPU draws triangles in the order they&apos;re submitted—far triangles can draw over near ones. The fix is surprisingly simple—just three steps: create a depth texture, add it to the render pass, and configure the pipeline. No shader changes needed:
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

            <h3 className="scroll-m-20 text-xl font-semibold tracking-tight">Bones and Hierarchy</h3>

            <p className="leading-7">
              A <span className="font-semibold">bone</span> is a transform in a hierarchy. Each bone has a parent (except the root), and moving a parent bone moves all its children. In MMD models, a typical arm chain looks like:
            </p>

            <p className="leading-7 pl-4 font-mono text-sm">
              センター (center) → 上半身 (upper_body) → 右肩 (shoulder_R) → 右腕 (arm_R) → 右ひじ (elbow_R) → 右手首 (wrist_R) → finger joints
            </p>

            <p className="leading-7">
              When you rotate 上半身 (upper_body), the entire upper body—shoulders, arms, elbows, wrists, and fingers—all follow. This cascading effect happens because each bone&apos;s transform is relative to its parent.
            </p>

            <h3 className="scroll-m-20 text-xl font-semibold tracking-tight">Skinning: Connecting Bones to Vertices</h3>

            <p className="leading-7">
              <span className="font-semibold">Skinning</span> is how bones deform the mesh. Each vertex stores up to 4 bone indices and 4 weights that sum to 1.0. When bones move, the vertex&apos;s final position is a weighted blend:
            </p>

            <Code language="typescript">
              {`// Vertex data
joints:  [15, 16, 0, 0]    // Bone indices
weights: [0.7, 0.3, 0, 0]  // 70% from bone 15, 30% from bone 16

// Final position = weighted sum of each bone's transform
finalPosition = (skinMatrix[15] * position) * 0.7 
              + (skinMatrix[16] * position) * 0.3`}
            </Code>

            <p className="leading-7">
              The <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">skinMatrix</code> for each bone combines the bone&apos;s current pose with its bind pose. This is what allows smooth deformation as bones rotate.
            </p>

            <h3 className="scroll-m-20 text-xl font-semibold tracking-tight">CPU-Side Bone Control</h3>

            <p className="leading-7">
              Bones live on the CPU. Animations, physics, and user input all update bone rotations here. When you rotate a bone, the engine recalculates the hierarchy (parent-to-child transforms) and uploads the results to GPU:
            </p>

            <Code language="typescript">
              {`// Your game code: rotate the neck bone
engine.rotateBone("首", rotation)

// Internally, this triggers:
// 1. evaluatePose() - recalculate all world matrices from hierarchy
// 2. Upload world matrices to GPU
// 3. Compute pass - calculate skin matrices
// 4. Next render uses updated skinning`}
            </Code>

            <h3 className="scroll-m-20 text-xl font-semibold tracking-tight">Compute Shaders: Parallel Matrix Calculations</h3>

            <p className="leading-7">
              With hundreds of bones and thousands of vertices, calculating skin matrices on the CPU is too slow. This is where <span className="font-semibold">compute shaders</span> shine—a key WebGPU advantage over WebGL. Compute shaders run massively parallel calculations on the GPU, perfect for matrix operations.
            </p>

            <p className="leading-7">
              We upload bone matrices to storage buffers, then dispatch a compute shader to calculate all skin matrices in parallel. For a model with 471 bones, this means 471 matrix multiplications happening simultaneously on the GPU:
            </p>

            <Code language="wgsl">
              {`@group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;
@group(0) @binding(2) var<storage, read> inverseBindMatrices: array<mat4x4f>;
@group(0) @binding(3) var<storage, read_write> skinMatrices: array<mat4x4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
  let boneIndex = globalId.x;
  if (boneIndex >= boneCount.count) { return; }
  
  skinMatrices[boneIndex] = worldMatrices[boneIndex] * inverseBindMatrices[boneIndex];
}`}
            </Code>

            <h3 className="scroll-m-20 text-xl font-semibold tracking-tight">Putting It Together</h3>

            <p className="leading-7">
              Here&apos;s the complete flow each frame (see the full implementation in{" "}
              <Link href={`${REPO_URL}/engines/v4.ts`} target="_blank" className="text-blue-400">
                engines/v4.ts
              </Link>
              ):
            </p>

            <ol className="ml-6 list-decimal [&>li]:mt-2">
              <li>
                <span className="font-semibold">CPU</span>: Animation or user input updates bone rotations
              </li>
              <li>
                <span className="font-semibold">CPU</span>: <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">evaluatePose()</code> walks the hierarchy to calculate world matrices
              </li>
              <li>
                <span className="font-semibold">CPU → GPU</span>: Upload world matrices to storage buffer
              </li>
              <li>
                <span className="font-semibold">GPU compute pass</span>: Calculate <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm">skinMatrix = world × inverseBind</code> for all bones in parallel
              </li>
              <li>
                <span className="font-semibold">GPU render pass</span>: Vertex shader reads skin matrices and blends each vertex by its bone weights
              </li>
            </ol>

            <p className="leading-7">
              The vertex shader performs the final skinning calculation:
            </p>

            <Code language="wgsl">
              {`@group(0) @binding(1) var<storage, read> skinMats: array<mat4x4f>;

@vertex
fn vs(
  @location(0) position: vec3<f32>,
  @location(3) joints: vec4<u32>,
  @location(4) weights: vec4<f32>
) -> VertexOutput {
  // Blend position by bone influences
  var skinnedPos = vec4f(0.0);
  for (var i = 0u; i < 4u; i++) {
    skinnedPos += (skinMats[joints[i]] * vec4f(position, 1.0)) * weights[i];
  }
  
  output.position = camera.projection * camera.view * skinnedPos;
}`}
            </Code>

            <div className="w-full h-full items-center justify-center flex mt-4">
              <Canvas4 />
            </div>

            <p className="leading-7">
              Try rotating the waist and neck bones with the sliders above to see skeletal skinning in action.
            </p>
          </section>

          <section className="flex flex-col items-start justify-start gap-6 w-full">
            <h2 className="scroll-m-20 border-b pb-2 text-2xl font-semibold tracking-tight first:mt-0">
              Conclusion
            </h2>

            <p className="leading-7">
              You&apos;ve now built a complete WebGPU rendering pipeline—from a simple triangle to a fully textured, skeletal-animated character. You understand the core components (buffers, bind groups, pipelines, render passes), how they connect (CPU to GPU data flow, shader interfaces), and why they&apos;re designed this way (uniform vs storage buffers, compute shaders for parallel work).
            </p>

            <p className="leading-7">
              This tutorial focused on WebGPU fundamentals. Advanced features like physics simulation, inverse kinematics, dynamic lighting, bloom, and post-processing build on these same concepts—they&apos;re application-level features, not new WebGPU primitives. You can explore these in the{" "}
              <Link href="/" className="text-blue-400">
                Reze Engine
              </Link>{" "}
              source code, which extends what you&apos;ve learned here into a full-featured anime character renderer.
            </p>
          </section>


        </div>
        <div className="w-64 sticky top-12 self-start ">
          <TableOfContents />
        </div>
      </div >
    </div >
  )
}
