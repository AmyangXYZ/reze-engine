import { Vec3 } from "../lib/math"
import { Camera } from "../lib/camera"

// Basic engine with arc rotate camera
export class EngineV1 {
  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  private pipeline!: GPURenderPipeline
  private vertexBuffer!: GPUBuffer
  private renderPassDescriptor!: GPURenderPassDescriptor
  private shaderModule!: GPUShaderModule

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
    await this.initDevice()
    this.initContext()
    this.initShader()
    this.initVertexBuffers()
    this.initPipeline()
    this.setupCamera()
    this.createBindGroups()
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
        fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
          return camera.projection * camera.view * vec4f(position, 0.0, 1.0);
        }

        @fragment
        fn fs() -> @location(0) vec4<f32> {
          return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
      `,
    })
  }

  private initVertexBuffers() {
    const triangleVertices = new Float32Array([-0.5, -0.5, 0.5, -0.5, 0.0, 0.5])
    this.vertexBuffer = this.device.createBuffer({
      label: "v0 vertex buffer",
      size: triangleVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.vertexBuffer, 0, triangleVertices)
  }

  private initPipeline() {
    this.pipeline = this.device.createRenderPipeline({
      label: "v0 pipeline",
      layout: "auto",
      vertex: {
        module: this.shaderModule,
        buffers: [
          {
            arrayStride: 2 * 4, // 2 floats * 4 bytes each = 8 bytes per vertex
            attributes: [
              {
                shaderLocation: 0, // matches @location(0) in the shader
                offset: 0,
                format: "float32x2", // vec2<f32> = 2 float32s
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

    this.camera = new Camera(0, Math.PI / 2.5, 2, new Vec3(0, 0, 0))

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
    ;(this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.context
      .getCurrentTexture()
      .createView()

    // V1: Update camera uniforms
    this.updateCameraUniforms()

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.draw(3)
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
