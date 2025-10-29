import { Camera } from "./camera"
import { Vec3 } from "./math"
import { RzmModel } from "./rzm"
import { PmxLoader } from "./pmx-loader"

export interface EngineStats {
  fps: number
  frameTime: number // ms
  memoryUsed: number // MB
  drawCalls: number
  vertices: number
}

export class Engine {
  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  public camera!: Camera
  private cameraUniformBuffer!: GPUBuffer
  private cameraMatrixData = new Float32Array(32) // Reused every frame
  private lightUniformBuffer!: GPUBuffer
  // Multi-light system: ambient(1) + padding(3), then 4 lights: each light = direction(3) + padding(1), color(3) + intensity(1) = 8 floats per light
  // Total: 4 + (4 * 8) = 36 floats, padded to 64 floats (256 bytes) for proper alignment
  private lightData = new Float32Array(64)
  private lightCount: number = 0 // Number of active lights (0-4)
  private bindGroup!: GPUBindGroup
  private vertexBuffer!: GPUBuffer
  private vertexCount: number = 0
  private indexBuffer?: GPUBuffer
  private indexCount: number = 0
  private resizeObserver: ResizeObserver | null = null
  private renderPassDescriptor!: GPURenderPassDescriptor
  private renderPassColorAttachment!: GPURenderPassColorAttachment
  private depthTexture!: GPUTexture
  private pipeline!: GPURenderPipeline
  private multisampleTexture!: GPUTexture
  private readonly sampleCount = 4 // MSAA 4x

  // Stats tracking
  private lastFpsUpdate = performance.now()
  private framesSinceLastUpdate = 0
  private frameTimeSamples: number[] = []
  private stats: EngineStats = {
    fps: 0,
    frameTime: 0,
    memoryUsed: 0,
    drawCalls: 0,
    vertices: 0,
  }

  // Render loop
  private animationFrameId: number | null = null
  private renderLoopCallback: (() => void) | null = null

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  public async init() {
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if (!device) {
      throw new Error("need a browser that supports WebGPU")
    }
    this.device = device
    console.log("WebGPU device created")

    const context = this.canvas.getContext("webgpu")
    if (!context) {
      throw new Error("Failed to get WebGPU context.")
    }
    this.context = context

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
    })

    // Create uniform buffer for camera matrices (view + projection = 32 floats)
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 32 * 4, // 32 floats * 4 bytes each
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Create uniform buffer for lighting (aligned to 256 bytes = 64 floats)
    this.lightUniformBuffer = this.device.createBuffer({
      label: "light uniforms",
      size: 64 * 4, // 64 floats * 4 bytes = 256 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Initialize MMD-style multi-light setup
    this.setAmbient(0.4)
    this.clearLights()
    // Key light (main, bright from front-right)
    this.addLight(new Vec3(-0.5, 0.8, 0.5).normalize(), new Vec3(1.0, 0.95, 0.9), 1.2)
    // Fill light (softer from left)
    this.addLight(new Vec3(0.7, 0.5, 0.3).normalize(), new Vec3(0.8, 0.85, 1.0), 0.6)
    // Rim light (from behind for edge highlighting)
    this.addLight(new Vec3(0.3, 0.5, -1.0).normalize(), new Vec3(0.9, 0.9, 1.0), 0.4)

    // Create render pass descriptor (view will be updated each frame)
    this.renderPassColorAttachment = {
      view: this.context.getCurrentTexture().createView(), // Placeholder, updated each frame
      clearValue: [0.3, 0.3, 0.3, 1],
      loadOp: "clear",
      storeOp: "store",
    }

    this.renderPassDescriptor = {
      label: "renderPass",
      colorAttachments: [this.renderPassColorAttachment],
    }

    // Create shader and pipeline
    this.initPipeline()

    // Setup camera and resize observer
    this.initCamera()
    this.initResizeObserver()
  }

  private initPipeline() {
    const shaderModule = this.device.createShaderModule({
      label: "model shaders",
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
        };

        struct Light {
          direction: vec3f,
          _padding1: f32,
          color: vec3f,
          intensity: f32,
        };

        struct LightUniforms {
          ambient: f32,
          lightCount: f32,
          _padding1: f32,
          _padding2: f32,
          lights: array<Light, 4>,
        };

        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) normal: vec3f,
          @location(1) uv: vec2f,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(0) @binding(1) var<uniform> light: LightUniforms;

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f
        ) -> VertexOutput {
          var output: VertexOutput;
          output.position = camera.projection * camera.view * vec4f(position, 1.0);
          output.normal = normal;
          output.uv = uv;
          return output;
        }
  
        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          let normal = normalize(input.normal);
          
          // Ambient lighting
          let ambient = light.ambient;
          let numLights = u32(light.lightCount);
          
          // Accumulate diffuse from all active lights
          var totalDiffuse = vec3f(0.0);
          
          for (var i = 0u; i < numLights; i++) {
            let lightDir = normalize(light.lights[i].direction);
            let diffuseFactor = max(dot(normal, lightDir), 0.0);
            let lightColor = light.lights[i].color;
            let lightIntensity = light.lights[i].intensity;
            totalDiffuse += lightColor * (diffuseFactor * lightIntensity);
          }
          
          // Combine lighting
          let lighting = vec3f(ambient) + totalDiffuse;
          
          // Base color (albedo)
          let baseColor = vec3f(0.8, 0.6, 0.9);
          
          // Final color with lighting (linear space)
          var finalColor = baseColor * lighting;
          
          // Gamma correction (linear to sRGB for true color)
          finalColor = pow(finalColor, vec3f(1.0 / 2.2));
          
          return vec4f(finalColor, 1.0);
        }
      `,
    })

    this.pipeline = this.device.createRenderPipeline({
      label: "model pipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
        buffers: [
          {
            arrayStride: 8 * 4, // 8 floats per vertex * 4 bytes per float = 32 bytes
            attributes: [
              {
                shaderLocation: 0, // position
                offset: 0,
                format: "float32x3" as GPUVertexFormat,
              },
              {
                shaderLocation: 1, // normal
                offset: 3 * 4,
                format: "float32x3" as GPUVertexFormat,
              },
              {
                shaderLocation: 2, // uv
                offset: 6 * 4,
                format: "float32x2" as GPUVertexFormat,
              },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { cullMode: "back" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      multisample: {
        count: this.sampleCount,
      },
    })

    // Create bind group
    this.bindGroup = this.device.createBindGroup({
      label: "camera bind group",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        {
          binding: 0,
          resource: { buffer: this.cameraUniformBuffer },
        },
        {
          binding: 1,
          resource: { buffer: this.lightUniformBuffer },
        },
      ],
    })
  }

  private initResizeObserver() {
    this.resizeObserver = new ResizeObserver(() => this.handleResize())
    this.resizeObserver.observe(this.canvas)
    this.handleResize()
  }

  private handleResize() {
    // Get the display size from CSS
    const displayWidth = this.canvas.clientWidth
    const displayHeight = this.canvas.clientHeight

    // Update canvas resolution to match display size
    // Using devicePixelRatio for crisp rendering on high-DPI displays
    const dpr = window.devicePixelRatio || 1
    const width = Math.floor(displayWidth * dpr)
    const height = Math.floor(displayHeight * dpr)

    // Only resize if dimensions actually changed
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width
      this.canvas.height = height

      // Recreate multisample texture with new size
      this.multisampleTexture = this.device.createTexture({
        label: "multisample render target",
        size: [width, height],
        sampleCount: this.sampleCount,
        format: this.presentationFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })

      // Recreate depth texture
      this.depthTexture = this.device.createTexture({
        label: "depth texture",
        size: [width, height],
        sampleCount: this.sampleCount,
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })

      // Update depth attachment on the render pass descriptor
      this.renderPassDescriptor.depthStencilAttachment = {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      }

      // Update camera aspect ratio
      this.camera.aspect = width / height

      // No need to trigger render - continuous loop handles it
    }
  }

  private initCamera() {
    // Create camera with default settings for character viewing
    this.camera = new Camera(
      Math.PI, // alpha
      Math.PI / 2.5, // beta
      30, // radius
      new Vec3(0, 13, 0) // target
    )

    // Set aspect ratio
    const aspect = this.canvas.width / this.canvas.height
    this.camera.aspect = aspect

    // Attach controls
    this.camera.attachControl(this.canvas)
  }

  // Clear all lights
  public clearLights() {
    this.lightCount = 0
    this.lightData[1] = 0 // lightCount
    this.updateLightBuffer()
  }

  // Add a light (up to 4 lights)
  public addLight(direction: Vec3, color: Vec3, intensity: number = 1.0): boolean {
    if (this.lightCount >= 4) return false

    const normalized = direction.normalize()
    const baseIndex = 4 + this.lightCount * 8

    // Store direction
    this.lightData[baseIndex] = normalized.x
    this.lightData[baseIndex + 1] = normalized.y
    this.lightData[baseIndex + 2] = normalized.z
    this.lightData[baseIndex + 3] = 0 // padding

    // Store color and intensity
    this.lightData[baseIndex + 4] = color.x
    this.lightData[baseIndex + 5] = color.y
    this.lightData[baseIndex + 6] = color.z
    this.lightData[baseIndex + 7] = intensity

    this.lightCount++
    this.lightData[1] = this.lightCount
    this.updateLightBuffer()
    return true
  }

  // Set ambient light intensity (0.0 to 1.0)
  public setAmbient(intensity: number) {
    this.lightData[0] = intensity
    this.updateLightBuffer()
  }

  // Update light buffer on GPU
  private updateLightBuffer() {
    this.device.queue.writeBuffer(this.lightUniformBuffer, 0, this.lightData)
  }

  public getStats(): EngineStats {
    return { ...this.stats }
  }

  public runRenderLoop(callback?: () => void) {
    this.renderLoopCallback = callback || null

    const loop = () => {
      this.render()

      if (this.renderLoopCallback) {
        this.renderLoopCallback()
      }

      this.animationFrameId = requestAnimationFrame(loop)
    }

    this.animationFrameId = requestAnimationFrame(loop)
  }

  public stopRenderLoop() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
    this.renderLoopCallback = null
  }

  public dispose() {
    // Stop render loop
    this.stopRenderLoop()

    // Cleanup camera controls
    if (this.camera) {
      this.camera.detachControl()
    }

    // Cleanup resize observer
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }
  }

  // Load RZM model from URL
  public async loadRzm(url: string) {
    const model = await RzmModel.load(url)
    this.drawModel(model)
  }

  // Load PMX model from URL
  public async loadPmx(url: string) {
    const model = await PmxLoader.load(url)
    this.drawModel(model)
  }

  private drawModel(model: RzmModel) {
    const vertices = model.getVertices()

    // Create vertex buffer from interleaved data
    this.vertexBuffer = this.device.createBuffer({
      label: "model vertex buffer",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })

    // Write vertex data to GPU buffer
    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices)
    this.vertexCount = model.getVertexCount()

    // Create index buffer if model has indices
    const indices = model.getIndices()
    if (indices) {
      this.indexBuffer = this.device.createBuffer({
        label: "model index buffer",
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(this.indexBuffer, 0, indices)
      this.indexCount = model.getIndexCount()
    } else {
      this.indexBuffer = undefined
      this.indexCount = 0
    }
  }

  public render() {
    // Safety check - don't render if not fully initialized
    if (!this.multisampleTexture || !this.camera || !this.device) {
      return
    }

    const frameStart = performance.now()

    // Update camera matrices
    const viewMatrix = this.camera.getViewMatrix()
    const projectionMatrix = this.camera.getProjectionMatrix()

    // Combine matrices into reused buffer
    this.cameraMatrixData.set(viewMatrix.values, 0)
    this.cameraMatrixData.set(projectionMatrix.values, 16)

    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)

    // Update render targets for current frame
    this.renderPassColorAttachment.view = this.multisampleTexture.createView()
    this.renderPassColorAttachment.resolveTarget = this.context.getCurrentTexture().createView()
    if (this.renderPassDescriptor.depthStencilAttachment) {
      this.renderPassDescriptor.depthStencilAttachment.view = this.depthTexture.createView()
    }

    const encoder = this.device.createCommandEncoder({ label: "our encoder" })
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.setVertexBuffer(0, this.vertexBuffer)

    // Use indexed rendering if index buffer exists
    if (this.indexBuffer) {
      pass.setIndexBuffer(this.indexBuffer, "uint32")
      pass.drawIndexed(this.indexCount)
    } else {
      pass.draw(this.vertexCount)
    }

    pass.end()

    const commandBuffer = encoder.finish()
    this.device.queue.submit([commandBuffer])

    // Update stats
    const frameEnd = performance.now()
    this.updateStats(frameEnd - frameStart)
  }

  private updateStats(frameTime: number) {
    // Update frame time (smoothed average over last 60 frames)
    const maxSamples = 60
    this.frameTimeSamples.push(frameTime)
    if (this.frameTimeSamples.length > maxSamples) {
      this.frameTimeSamples.shift()
    }
    const avgFrameTime = this.frameTimeSamples.reduce((a, b) => a + b, 0) / this.frameTimeSamples.length
    this.stats.frameTime = Math.round(avgFrameTime * 100) / 100

    const now = performance.now()

    // Calculate FPS based on actual frame count over time
    this.framesSinceLastUpdate++
    const elapsed = now - this.lastFpsUpdate

    // Update FPS and memory every second
    if (elapsed >= 1000) {
      this.stats.fps = Math.round((this.framesSinceLastUpdate / elapsed) * 1000)
      this.framesSinceLastUpdate = 0
      this.lastFpsUpdate = now

      // Update memory stats once per second (Chrome only)
      const perf = performance as Performance & {
        memory?: { usedJSHeapSize: number; totalJSHeapSize: number }
      }
      if (perf.memory) {
        this.stats.memoryUsed = Math.round(perf.memory.usedJSHeapSize / 1048576) // Convert to MB
      }
    }

    // GPU stats (basic - draw calls and vertices this frame)
    this.stats.drawCalls = 1
    this.stats.vertices = this.vertexCount
  }
}
