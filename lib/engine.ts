export class Engine {
  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat

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
  }

  public render() {
    const shaderModule = this.device.createShaderModule({
      label: "our hardcoded red triangle shaders",
      code: /* wgsl */ `
        @vertex fn vs(
          @builtin(vertex_index) vertexIndex : u32
        ) -> @builtin(position) vec4f {
          let pos = array(
            vec2f( 0.0,  0.5),  // top center
            vec2f(-0.5, -0.5),  // bottom left
            vec2f( 0.5, -0.5)   // bottom right
          );
  
          return vec4f(pos[vertexIndex], 0.0, 1.0);
        }
  
        @fragment fn fs() -> @location(0) vec4f {
          return vec4f(1, 0, 0, 1);
        }
      `,
    })

    const pipeline = this.device.createRenderPipeline({
      label: "our hardcoded red triangle pipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
      },
      fragment: {
        module: shaderModule,
        targets: [{ format: this.presentationFormat }],
      },
    })

    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "our basic canvas renderPass",
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: [0.3, 0.3, 0.3, 1],
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    }
    const encoder = this.device.createCommandEncoder({ label: "our encoder" })

    const pass = encoder.beginRenderPass(renderPassDescriptor)
    pass.setPipeline(pipeline)
    pass.draw(3)
    pass.end()

    const commandBuffer = encoder.finish()
    this.device.queue.submit([commandBuffer])
  }
}
