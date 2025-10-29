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
  private bindGroup!: GPUBindGroup
  private vertexBuffer!: GPUBuffer
  private vertexCount: number = 0
  private indexBuffer?: GPUBuffer
  private indexCount: number = 0
  private resizeObserver: ResizeObserver | null = null
  private renderPassDescriptor!: GPURenderPassDescriptor
  private renderPassColorAttachment!: GPURenderPassColorAttachment
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

    // Create render pass descriptor (view will be updated each frame)
    this.renderPassColorAttachment = {
      view: this.context.getCurrentTexture().createView(), // Placeholder, updated each frame
      clearValue: [0.3, 0.3, 0.3, 1],
      loadOp: "clear",
      storeOp: "store",
    }

    this.renderPassDescriptor = {
      label: "our basic canvas renderPass",
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
        struct Uniforms {
          view: mat4x4f,
          projection: mat4x4f,
        };

        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) normal: vec3f,
          @location(1) uv: vec2f,
        };

        @group(0) @binding(0) var<uniform> uniforms: Uniforms;

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f
        ) -> VertexOutput {
          var output: VertexOutput;
          output.position = uniforms.projection * uniforms.view * vec4f(position, 1.0);
          output.normal = normal;
          output.uv = uv;
          return output;
        }
  
        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          // Simple diffuse lighting based on normal
          let lightDir = normalize(vec3f(0.5, 1.0, 0.3));
          let diffuse = max(dot(normalize(input.normal), lightDir), 0.3);
          let color = vec3f(0.8, 0.6, 0.9);
          return vec4f(color * diffuse, 1.0);
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
