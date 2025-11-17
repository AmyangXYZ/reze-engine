export class EngineV0 {
  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  private pipeline!: GPURenderPipeline
  private vertexBuffer!: GPUBuffer
  private indexBuffer!: GPUBuffer
  private renderPassDescriptor!: GPURenderPassDescriptor

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  public async init() {
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if (!device) {
      throw new Error("WebGPU is not supported in this browser.")
    }
    this.device = device
  }

  public render() {
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    pass.setPipeline(this.pipeline)
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.setIndexBuffer(this.indexBuffer, "uint32")
    pass.drawIndexed(this.indexBuffer.size, 1, 0, 0, 0)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }
}
