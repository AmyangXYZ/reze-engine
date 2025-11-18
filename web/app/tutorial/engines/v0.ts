// Basic engine rendering a triangle
export class EngineV0 {
  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  private pipeline!: GPURenderPipeline
  private vertexBuffer!: GPUBuffer
  private renderPassDescriptor!: GPURenderPassDescriptor
  private shaderModule!: GPUShaderModule

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  public async init() {
    await this.initDevice()
    this.initContext()
    this.initShader()
    this.initVertexBuffers()
    this.initPipeline()
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

    const displayWidth = this.canvas.clientWidth
    const displayHeight = this.canvas.clientHeight
    const dpr = window.devicePixelRatio || 1
    console.log(displayWidth, displayHeight, dpr)
    const width = Math.floor(displayWidth * dpr)
    const height = Math.floor(displayHeight * dpr)
    this.canvas.width = width
    this.canvas.height = height
    console.log(width, height)

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
        @vertex
        fn vs(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
          return vec4<f32>(position, 0.0, 1.0);
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

  public render() {
    ;(this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.context
      .getCurrentTexture()
      .createView()

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    pass.setPipeline(this.pipeline)
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.draw(3)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }
}
