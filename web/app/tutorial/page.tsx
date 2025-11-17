"use client"

import Header from "@/components/header"
import { useCallback, useEffect, useRef } from "react"

import modelData from './model.json'
import { Vec3 } from "reze-engine"

interface Model {
  vertices: Float32Array | number[] // Can be TypedArray or regular array from JSON
  indices: Uint32Array | number[] // Can be TypedArray or regular array from JSON
  materials: { name: string, diffuseTextureIndex: number }[]
  textures: { name: string, path: string }[]
  bones: { name: string, parentIndex: number, bindTranslation: Vec3 }[]
  skinning: { joints: Uint16Array | number[], weights: Uint8Array | number[] }
}

export default function Tutorial() {
  const deviceRef = useRef<GPUDevice | null>(null)
  const contextRef = useRef<GPUCanvasContext | null>(null)
  const presentationFormatRef = useRef<GPUTextureFormat | null>(null)
  const modelRef = useRef<Model | null>(modelData as Model)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  const vertexBufferRef = useRef<GPUBuffer | null>(null)
  const indexBufferRef = useRef<GPUBuffer | null>(null)
  const pipelineRef = useRef<GPURenderPipeline | null>(null)
  const renderPassDescriptorRef = useRef<GPURenderPassDescriptor | null>(null)
  const animationFrameRef = useRef<number | null>(null)

  // Step 1: Initialize WebGPU device and context
  const initWebGPU = useCallback(async () => {
    if (!canvasRef.current) return

    // Get WebGPU adapter and device
    const adapter = await navigator.gpu?.requestAdapter()
    if (!adapter) {
      throw new Error("WebGPU adapter not found")
    }

    const device = await adapter.requestDevice()
    if (!device) {
      throw new Error("WebGPU device not available")
    }
    deviceRef.current = device

    // Get WebGPU context from canvas
    const context = canvasRef.current.getContext("webgpu")
    if (!context) {
      throw new Error("Failed to get WebGPU context")
    }
    contextRef.current = context

    // Get preferred canvas format
    const presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    presentationFormatRef.current = presentationFormat

    // Configure canvas context
    context.configure({
      device,
      format: presentationFormat,
      alphaMode: "premultiplied",
    })
  }, [])

  // Step 2: Create shaders and render pipeline
  const createPipeline = useCallback(() => {
    if (!deviceRef.current || !presentationFormatRef.current) return

    const device = deviceRef.current
    const format = presentationFormatRef.current

    // Minimal shader: just render vertices with a simple color
    // Vertex format: position (3 floats) + normal (3 floats) + uv (2 floats) = 8 floats
    const shaderModule = device.createShaderModule({
      label: "minimal shader",
      code: /* wgsl */ `
        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) color: vec3f,
        };

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f,
        ) -> VertexOutput {
          var output: VertexOutput;
          // Simple orthographic projection for now (no camera matrix yet)
          output.position = vec4f(position, 1.0);
          // Use normal as color for visualization
          output.color = normalize(normal) * 0.5 + 0.5;
          return output;
        }

        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          return vec4f(input.color, 1.0);
        }
      `,
    })

    // Create render pipeline
    pipelineRef.current = device.createRenderPipeline({
      label: "minimal pipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: 8 * 4, // 8 floats (pos + normal + uv) * 4 bytes each = 32 bytes
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: "float32x3" as GPUVertexFormat, // position
              },
              {
                shaderLocation: 1,
                offset: 3 * 4,
                format: "float32x3" as GPUVertexFormat, // normal
              },
              {
                shaderLocation: 2,
                offset: 6 * 4,
                format: "float32x2" as GPUVertexFormat, // uv
              },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs",
        targets: [
          {
            format: format,
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
    })
  }, [])

  // Step 3: Create vertex and index buffers
  const createBuffers = useCallback(() => {
    if (!deviceRef.current || !modelRef.current) return

    const device = deviceRef.current
    const model = modelRef.current

    // Convert JSON arrays to TypedArrays (JSON doesn't preserve TypedArrays)
    let vertices: Float32Array
    if (model.vertices instanceof Float32Array) {
      vertices = model.vertices
    } else if (Array.isArray(model.vertices)) {
      vertices = new Float32Array(model.vertices)
    } else {
      console.error("Invalid vertices data:", model.vertices)
      return
    }

    // Validate vertices data
    if (!vertices || vertices.length === 0) {
      console.error("Vertices array is empty or undefined")
      return
    }

    // Create vertex buffer
    const vertexBufferSize = vertices.byteLength
    if (vertexBufferSize === 0) {
      console.error("Vertex buffer size is 0")
      return
    }

    vertexBufferRef.current = device.createBuffer({
      label: "vertex buffer",
      size: vertexBufferSize,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    // Write buffer data - WebGPU accepts TypedArrays directly
    // @ts-expect-error - TypeScript incorrectly infers ArrayBufferLike, but runtime uses ArrayBuffer
    device.queue.writeBuffer(vertexBufferRef.current, 0, vertices)

    // Convert indices array to TypedArray
    let indices: Uint32Array | null = null
    if (model.indices) {
      if (model.indices instanceof Uint32Array) {
        indices = model.indices
      } else if (Array.isArray(model.indices)) {
        indices = new Uint32Array(model.indices)
      } else {
        console.error("Invalid indices data:", model.indices)
        return
      }

      // Validate indices data
      if (indices && indices.length > 0) {
        const indexBufferSize = indices.byteLength
        indexBufferRef.current = device.createBuffer({
          label: "index buffer",
          size: indexBufferSize,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        })
        // Write buffer data - WebGPU accepts TypedArrays directly
        // @ts-expect-error - TypeScript incorrectly infers ArrayBufferLike, but runtime uses ArrayBuffer
        device.queue.writeBuffer(indexBufferRef.current, 0, indices)
      } else {
        console.error("Indices array is empty")
      }
    }
  }, [])

  // Step 4: Setup canvas and render pass descriptor
  const setupCanvas = useCallback(() => {
    if (!canvasRef.current || !contextRef.current || !deviceRef.current) return

    const canvas = canvasRef.current
    const device = deviceRef.current

    // Set canvas size
    const dpr = window.devicePixelRatio || 1
    const displayWidth = canvas.clientWidth
    const displayHeight = canvas.clientHeight
    const width = Math.floor(displayWidth * dpr)
    const height = Math.floor(displayHeight * dpr)

    canvas.width = width
    canvas.height = height

    // Create depth texture (depth-only, no stencil)
    const depthTexture = device.createTexture({
      label: "depth texture",
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })

    // Create render pass descriptor
    renderPassDescriptorRef.current = {
      label: "render pass",
      colorAttachments: [
        {
          view: contextRef.current.getCurrentTexture().createView(),
          clearValue: { r: 0.1, g: 0.1, b: 0.1, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    }
  }, [])

  // Step 5: Render one frame
  const render = useCallback(() => {
    if (
      !deviceRef.current ||
      !pipelineRef.current ||
      !vertexBufferRef.current ||
      !indexBufferRef.current ||
      !renderPassDescriptorRef.current
    ) {
      return
    }

    const device = deviceRef.current
    const pipeline = pipelineRef.current
    const vertexBuffer = vertexBufferRef.current
    const indexBuffer = indexBufferRef.current

    // Update render pass descriptor with current texture view
    const descriptor = renderPassDescriptorRef.current
    if (contextRef.current && descriptor.colorAttachments) {
      const colorAttachment = (descriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
      if (colorAttachment) {
        colorAttachment.view = contextRef.current.getCurrentTexture().createView()
      }
    }

    // Create command encoder and render pass
    const encoder = device.createCommandEncoder()
    const pass = encoder.beginRenderPass(descriptor)

    // Set pipeline and buffers
    pass.setPipeline(pipeline)
    pass.setVertexBuffer(0, vertexBuffer)
    pass.setIndexBuffer(indexBuffer, "uint32")

    // Draw all indices
    const model = modelRef.current
    if (model && model.indices) {
      // Convert to array if needed to get length
      const indicesLength = Array.isArray(model.indices)
        ? model.indices.length
        : model.indices instanceof Uint32Array
          ? model.indices.length
          : 0

      if (indicesLength > 0) {
        pass.drawIndexed(indicesLength, 1, 0, 0, 0)
      }
    }

    pass.end()
    device.queue.submit([encoder.finish()])
  }, [])

  // Render loop
  const startRenderLoop = useCallback(() => {
    const loop = () => {
      render()
      animationFrameRef.current = requestAnimationFrame(loop)
    }
    animationFrameRef.current = requestAnimationFrame(loop)
  }, [render])

  const stopRenderLoop = useCallback(() => {
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
  }, [])

  // Initialize everything
  useEffect(() => {
    void (async () => {
      try {
        await initWebGPU()
        createPipeline()
        createBuffers()
        setupCanvas()
        startRenderLoop()
      } catch (error) {
        console.error("Failed to initialize WebGPU:", error)
      }
    })()

    return () => {
      stopRenderLoop()
    }
  }, [initWebGPU, createPipeline, createBuffers, setupCanvas, startRenderLoop, stopRenderLoop])

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