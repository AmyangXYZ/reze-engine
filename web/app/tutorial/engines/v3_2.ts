import { Vec3 } from "../lib/math"
import { Camera } from "../lib/camera"
import modelData from "../model.json"

interface Model {
  vertices: Float32Array
  indices: Uint32Array
  textures: Texture[]
  materials: Material[]
}

interface Texture {
  path: string
  name: string
}

interface Material {
  name: string
  diffuseTextureIndex: number
  vertexCount: number
}

// Basic engine with arc rotate camera
export class EngineV3_2 {
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
  // Bind groups
  private bindGroup!: GPUBindGroup
  private materialBindGroups: GPUBindGroup[] = []
  // Textures
  private textures: GPUTexture[] = []
  private sampler!: GPUSampler
  // Depth
  private depthTexture!: GPUTexture
  // Render loop
  private animationFrameId: number | null = null

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  public async init() {
    this.loadModel()
    await this.initDevice()
    this.initContext()
    await this.initTexture()
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
      textures: model.textures,
      materials: model.materials,
    }
    console.log(this.model)
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

    // Create depth texture
    this.depthTexture = this.device.createTexture({
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
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
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
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

        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) uv: vec2<f32>,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(1) @binding(0) var texture: texture_2d<f32>;
        @group(1) @binding(1) var textureSampler: sampler;

        @vertex
        fn vs(
          @location(0) position: vec3<f32>,
          @location(1) normal: vec3<f32>,
          @location(2) uv: vec2<f32>
        ) -> VertexOutput {
          var output: VertexOutput;
          output.position = camera.projection * camera.view * vec4f(position, 1.0);
          output.uv = uv;
          return output;
        }

        @fragment
        fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
          return vec4<f32>(textureSample(texture, textureSampler, input.uv).rgb, 1.0);
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
      label: "v3 pipeline",
      layout: "auto",
      vertex: {
        module: this.shaderModule,
        buffers: [
          {
            arrayStride: 8 * 4, // 8 floats * 4 bytes each = 32 bytes per vertex (position + normal + UV)
            attributes: [
              {
                shaderLocation: 0, // position
                offset: 0,
                format: "float32x3",
              },
              {
                shaderLocation: 1, // normal (not used in shader, but we need to skip it)
                offset: 3 * 4, // 12 bytes (3 floats)
                format: "float32x3",
              },
              {
                shaderLocation: 2, // UV
                offset: 6 * 4, // 24 bytes (3 floats position + 3 floats normal)
                format: "float32x2",
              },
            ],
          },
        ],
      },
      fragment: {
        module: this.shaderModule,
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { cullMode: "none" },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
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

  private async initTexture() {
    // Collect all unique texture indices used by materials
    const textureIndices = new Set<number>()
    for (const material of this.model.materials) {
      if (material.diffuseTextureIndex >= 0 && material.diffuseTextureIndex < this.model.textures.length) {
        textureIndices.add(material.diffuseTextureIndex)
      }
    }
    const textureDir = "/models/塞尔凯特"
    // Load all textures referenced by materials
    const textureLoadPromises = Array.from(textureIndices).map(async (textureIndex) => {
      const texturePath = `${textureDir}/${this.model.textures[textureIndex].path}`
      const response = await fetch(texturePath)
      if (!response.ok) {
        throw new Error(`Failed to load texture: ${texturePath}`)
      }

      const imageBitmap = await createImageBitmap(await response.blob(), {
        premultiplyAlpha: "none",
        colorSpaceConversion: "none",
      })

      // Create texture
      const texture = this.device.createTexture({
        label: `texture: ${this.model.textures[textureIndex].name}`,
        size: [imageBitmap.width, imageBitmap.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })

      // Upload image data to texture
      this.device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture }, [
        imageBitmap.width,
        imageBitmap.height,
      ])

      return { textureIndex, texture }
    })

    const loadedTextures = await Promise.all(textureLoadPromises)

    // Store textures in array indexed by texture index
    this.textures = new Array(this.model.textures.length)
    for (const { textureIndex, texture } of loadedTextures) {
      this.textures[textureIndex] = texture
    }

    // Create sampler
    this.sampler = this.device.createSampler({
      label: "texture sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    })
  }

  private createBindGroups() {
    this.bindGroup = this.device.createBindGroup({
      label: "camera bind group",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
    })

    // Create bind group for each material with its own texture
    for (const material of this.model.materials) {
      const textureIndex = material.diffuseTextureIndex
      const materialBindGroup = this.device.createBindGroup({
        label: `texture bind group: ${material.name}`,
        layout: this.pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: this.textures[textureIndex].createView() },
          { binding: 1, resource: this.sampler },
        ],
      })

      this.materialBindGroups.push(materialBindGroup)
    }
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

    // Render each material separately with its own texture
    let firstIndex = 0
    for (let i = 0; i < this.model.materials.length; i++) {
      const material = this.model.materials[i]
      if (material.vertexCount === 0) continue

      const bindGroup = this.materialBindGroups[i]
      pass.setBindGroup(1, bindGroup)
      pass.drawIndexed(material.vertexCount, 1, firstIndex)
      firstIndex += material.vertexCount
    }

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
