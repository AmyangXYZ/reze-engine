import { Vec3 } from "../lib/math"
import { Camera } from "../lib/camera"
import modelData from "../model.json"

interface Model {
  vertices: Float32Array
  indices: Uint32Array
}

// Basic engine with arc rotate camera
export class EngineV2 {
  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  private pipeline!: GPURenderPipeline
  private vertexBuffer!: GPUBuffer
  private indexBuffer!: GPUBuffer
  private renderPassDescriptor!: GPURenderPassDescriptor
  private shaderModule!: GPUShaderModule

  private model!: Model

  // Camera
  private camera!: Camera
  private cameraUniformBuffer!: GPUBuffer
  private cameraMatrixData = new Float32Array(36)
  // Bind group
  private bindGroup!: GPUBindGroup
  // Render loop
  private animationFrameId: number | null = null

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  public async init() {
    this.loadModel()
    await this.initDevice()
    this.initContext()
    this.initShader()
    this.initVertexBuffers()
    this.initPipeline()
    this.setupCamera()
    this.createBindGroups()
  }

  private loadModel() {
    const model = modelData as unknown as Model
    this.model = {
      vertices: new Float32Array(model.vertices),
      indices: new Uint32Array(model.indices),
    }
  }

  private async initDevice() {
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if (!device) {
      throw new Error("WebGPU is not supported in this browser.")
    }
    this.device = device
  }

  private initContext() {
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

    // Set canvas size with device pixel ratio (like engine.ts)
    const displayWidth = this.canvas.clientWidth
    const displayHeight = this.canvas.clientHeight
    const dpr = window.devicePixelRatio || 1
    const width = Math.floor(displayWidth * dpr)
    const height = Math.floor(displayHeight * dpr)
    this.canvas.width = width
    this.canvas.height = height

    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    }
  }

  private initShader() {
    this.shaderModule = this.device.createShaderModule({
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
          viewPos: vec3f,
          _padding: f32,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;

        @vertex
        fn vs(@location(0) position: vec3<f32>) -> @builtin(position) vec4<f32> {
          return camera.projection * camera.view * vec4f(position, 1.0);
        }

        @fragment
        fn fs() -> @location(0) vec4<f32> {
          return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
      `,
    })
  }

  private initVertexBuffers() {
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

  private initPipeline() {
    this.pipeline = this.device.createRenderPipeline({
      label: "v2 pipeline",
      layout: "auto",
      vertex: {
        module: this.shaderModule,
        buffers: [
          {
            arrayStride: 8 * 4, // 8 floats * 4 bytes each = 32 bytes per vertex (position + normal + UV)
            attributes: [
              {
                shaderLocation: 0, // matches @location(0) in the shader
                offset: 0, // position is at offset 0
                format: "float32x3", // vec3<f32> = 3 float32s for position
              },
            ],
          },
        ],
      },
      fragment: {
        module: this.shaderModule,
        targets: [{ format: this.presentationFormat }],
      },
    })
  }

  private setupCamera() {
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 40 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.camera = new Camera(Math.PI, Math.PI / 2.5, 27, new Vec3(0, 12, 0))

    this.camera.aspect = this.canvas.width / this.canvas.height
    this.camera.attachControl(this.canvas)
  }

  private createBindGroups() {
    this.bindGroup = this.device.createBindGroup({
      label: "bind group layout",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
    })
  }

  private updateCameraUniforms() {
    const viewMatrix = this.camera.getViewMatrix()
    const projectionMatrix = this.camera.getProjectionMatrix()
    const cameraPos = this.camera.getPosition()
    this.cameraMatrixData.set(viewMatrix.values, 0)
    this.cameraMatrixData.set(projectionMatrix.values, 16)
    this.cameraMatrixData[32] = cameraPos.x
    this.cameraMatrixData[33] = cameraPos.y
    this.cameraMatrixData[34] = cameraPos.z
    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)
  }

  public render() {
    // Update render target views
    ;(this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.context
      .getCurrentTexture()
      .createView()

    // Update camera uniforms
    this.updateCameraUniforms()

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.setIndexBuffer(this.indexBuffer, "uint32")
    pass.drawIndexed(this.model.indices.length)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  public runRenderLoop() {
    const loop = () => {
      this.render()
      this.animationFrameId = requestAnimationFrame(loop)
    }

    this.animationFrameId = requestAnimationFrame(loop)
  }

  public dispose() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
    if (this.camera) {
      this.camera.detachControl()
    }
    this.context.unconfigure()
  }
}
