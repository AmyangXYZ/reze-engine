import { Camera } from "./camera"
import { Quat, Vec3 } from "./math"
import { Model } from "./model"
import { PmxLoader } from "./pmx-loader"
import { Physics } from "./physics"

export interface EngineStats {
  fps: number
  frameTime: number // ms
  memoryUsed: number // MB
  vertices: number
  drawCalls: number
}

export class Engine {
  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  public camera!: Camera
  private cameraUniformBuffer!: GPUBuffer
  private cameraMatrixData = new Float32Array(36) // view(16) + projection(16) + viewPos(3) + padding(1) = 36 floats
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
  private diffuseTexture?: GPUTexture
  private diffuseSampler?: GPUSampler
  private resizeObserver: ResizeObserver | null = null
  private renderPassDescriptor!: GPURenderPassDescriptor
  private renderPassColorAttachment!: GPURenderPassColorAttachment
  private depthTexture!: GPUTexture
  private pipeline!: GPURenderPipeline
  private gridPipeline?: GPURenderPipeline
  private jointsBuffer?: GPUBuffer
  private weightsBuffer?: GPUBuffer
  private skinMatrixBuffer?: GPUBuffer
  private fallbackSkinMatrixBuffer?: GPUBuffer
  private multisampleTexture!: GPUTexture
  private readonly sampleCount = 4 // MSAA 4x
  private currentModel: Model | null = null
  private modelDir: string = "" // Directory for loading model textures
  // Grid
  private gridVertexBuffer?: GPUBuffer
  private gridVertexCount: number = 0
  private gridBindGroup?: GPUBindGroup
  private physics: Physics | null = null

  // Stats tracking
  private lastFpsUpdate = performance.now()
  private framesSinceLastUpdate = 0
  private frameTimeSamples: number[] = []
  private drawCallCount: number = 0 // Per-frame draw call counter
  private lastFrameTime = performance.now() // For spring bone deltaTime calculation
  private stats: EngineStats = {
    fps: 0,
    frameTime: 0,
    memoryUsed: 0,
    vertices: 0,
    drawCalls: 0,
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
      throw new Error("WebGPU is not supported in this browser.")
    }
    this.device = device

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

    // Create uniform buffer for camera matrices (view + projection + viewPos = 36 floats, padded to 40 for alignment)
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 40 * 4, // 40 floats * 4 bytes = 160 bytes (aligned)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Create uniform buffer for lighting (aligned to 256 bytes = 64 floats)
    this.lightUniformBuffer = this.device.createBuffer({
      label: "light uniforms",
      size: 64 * 4, // 64 floats * 4 bytes = 256 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Initialize MMD-style multi-light setup
    this.setAmbient(0.65) // Reduced ambient to make lights more visible
    this.clearLights()
    // Key light (main, bright from front-right)
    this.addLight(new Vec3(-0.5, -0.8, 0.5).normalize(), new Vec3(1.0, 0.95, 0.9), 1.2)
    // Fill light (softer from left)`
    this.addLight(new Vec3(0.7, -0.5, 0.3).normalize(), new Vec3(0.8, 0.85, 1.0), 1.1)
    // Rim light (from behind for edge highlighting)
    this.addLight(new Vec3(0.3, -0.5, -1.0).normalize(), new Vec3(0.9, 0.9, 1.0), 1.0)

    // Create render pass descriptor (view will be updated each frame)
    this.renderPassColorAttachment = {
      view: this.context.getCurrentTexture().createView(), // Placeholder, updated each frame
      clearValue: { r: 0.05, g: 0.066, b: 0.086, a: 1.0 },
      loadOp: "clear",
      storeOp: "store",
    }

    this.renderPassDescriptor = {
      label: "renderPass",
      colorAttachments: [this.renderPassColorAttachment],
    }

    // Create shader and pipeline
    this.initPipeline()
    this.initGridPipeline()

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
          viewPos: vec3f,
          _padding: f32,
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
          @location(2) worldPos: vec3f,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(0) @binding(1) var<uniform> light: LightUniforms;
        @group(0) @binding(2) var diffuseTexture: texture_2d<f32>;
        @group(0) @binding(3) var diffuseSampler: sampler;
        @group(0) @binding(4) var<storage, read> skinMats: array<mat4x4f>;

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f,
          @location(3) joints0: vec4<u32>,
          @location(4) weights0: vec4<f32>
        ) -> VertexOutput {
          var output: VertexOutput;
          // GPU skinning (LBS4)
          var skinnedPos = vec4f(0.0, 0.0, 0.0, 0.0);
          var skinnedNrm = vec3f(0.0, 0.0, 0.0);
          for (var i = 0u; i < 4u; i++) {
            let j = joints0[i];
            let w = weights0[i];
            let m = skinMats[j];
            skinnedPos += (m * vec4f(position, 1.0)) * w;
            // normal (upper-left 3x3)
            let r3 = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
            skinnedNrm += (r3 * normal) * w;
          }
          let worldPos = skinnedPos.xyz;
          output.position = camera.projection * camera.view * vec4f(worldPos, 1.0);
          output.normal = normalize(skinnedNrm);
          output.uv = uv;
          output.worldPos = worldPos;
          return output;
        }

        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          let n = normalize(input.normal);
          let albedo = textureSample(diffuseTexture, diffuseSampler, input.uv).rgb;

          // Ambient term
          var color = albedo * vec3f(light.ambient);

          // View direction for subtle specular
          let v = normalize(camera.viewPos - input.worldPos);

          // Lambert diffuse with 1/pi plus a gentle Blinn-Phong highlight
          let diffuse = albedo / 3.14159265;
          let numLights = u32(light.lightCount);
          for (var i = 0u; i < numLights; i++) {
            let l = normalize(-light.lights[i].direction);
            let nDotL = max(dot(n, l), 0.0);
            if (nDotL > 0.0) {
              let radiance = light.lights[i].color * light.lights[i].intensity;
              // subtle specular
              let h = normalize(l + v);
              let spec = pow(max(dot(n, h), 0.0), 24.0) * 0.06;
              color += diffuse * radiance * nDotL + vec3f(spec) * radiance;
            }
          }

          // Soft rolloff to keep brightness natural without looking flat
          color = color / (vec3f(1.0) + color * 0.15);
          color = clamp(color, vec3f(0.0), vec3f(1.0));
          return vec4f(color, 1.0);
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
          {
            arrayStride: 4 * 2, // 4 * uint16
            attributes: [{ shaderLocation: 3, offset: 0, format: "uint16x4" as GPUVertexFormat }],
          },
          {
            arrayStride: 4, // 4 * unorm8 packed
            attributes: [{ shaderLocation: 4, offset: 0, format: "unorm8x4" as GPUVertexFormat }],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      multisample: {
        count: this.sampleCount,
      },
    })

    // Create bind group (will be updated when textures are loaded)
    this.updateBindGroup()
  }

  private initGridPipeline() {
    // Simple grid shader using only camera uniforms
    const shader = this.device.createShaderModule({
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
          viewPos: vec3f,
          _padding: f32,
        };

        struct VSOut { @builtin(position) pos: vec4f, @location(0) color: vec3f };
        @group(0) @binding(0) var<uniform> camera: CameraUniforms;

        @vertex fn vs(@location(0) position: vec3f, @location(1) color: vec3f) -> VSOut {
          var o: VSOut;
          o.pos = camera.projection * camera.view * vec4f(position, 1.0);
          o.color = color;
          return o;
        }

        @fragment fn fs(i: VSOut) -> @location(0) vec4f { return vec4f(i.color, 1.0); }
      `,
    })

    this.gridPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shader,
        buffers: [
          {
            arrayStride: 6 * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
              { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat },
            ],
          },
        ],
      },
      fragment: { module: shader, targets: [{ format: this.presentationFormat }] },
      primitive: { topology: "line-list", cullMode: "none" },
      depthStencil: { format: "depth24plus", depthWriteEnabled: true, depthCompare: "less" },
      multisample: { count: this.sampleCount },
    })

    // Bind group with only camera uniforms
    this.gridBindGroup = this.device.createBindGroup({
      layout: this.gridPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
    })

    // Build grid vertex buffer (moderate extent and spacing)
    this.buildGrid(20, 2)
  }

  private buildGrid(halfLines: number, step: number) {
    const count = halfLines
    const size = count * step
    const lines = [] as number[]
    const faint = [0.3, 0.4, 0.45] // subtle cyan tint
    const strong = [0.35, 0.72, 0.82]
    for (let i = -count; i <= count; i++) {
      const c = i === 0 ? strong : faint
      // lines parallel to X (varying Z)
      lines.push(-size, 0, i * step, c[0], c[1], c[2])
      lines.push(size, 0, i * step, c[0], c[1], c[2])
      // lines parallel to Z (varying X)
      lines.push(i * step, 0, -size, c[0], c[1], c[2])
      lines.push(i * step, 0, size, c[0], c[1], c[2])
    }
    const data = new Float32Array(lines)
    this.gridVertexCount = data.length / 6
    this.gridVertexBuffer = this.device.createBuffer({
      label: "grid",
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.gridVertexBuffer, 0, data)
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
      27, // radius
      new Vec3(0, 12.5, 0) // target
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

  // Load PMX model from directory and filename
  public async loadPmx(dir: string, fileName: string) {
    this.modelDir = dir.endsWith("/") ? dir : dir + "/"
    const url = this.modelDir + fileName
    const model = await PmxLoader.load(url)

    this.physics = new Physics(model.getRigidbodies(), model.getJoints())
    await this.drawModel(model)

    setTimeout(() => {
      model.rotateBones(["腰"], [new Quat(-0.7, -0.0, 0, 1)], 500)
    }, 2000)
    setTimeout(() => {
      model.rotateBones(["腰"], [new Quat(-0.0, 0.0, 0, 1)], 500)
    }, 3000)
  }

  private async drawModel(model: Model) {
    this.currentModel = model
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

    // Skinning buffers
    const skinning = model.getSkinning()
    // joints buffer (u16x4 per vertex)
    this.jointsBuffer = this.device.createBuffer({
      label: "joints buffer",
      size: skinning.joints.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(
      this.jointsBuffer,
      0,
      skinning.joints.buffer,
      skinning.joints.byteOffset,
      skinning.joints.byteLength
    )

    // weights buffer (unorm8x4 per vertex)
    this.weightsBuffer = this.device.createBuffer({
      label: "weights buffer",
      size: skinning.weights.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(
      this.weightsBuffer,
      0,
      skinning.weights.buffer,
      skinning.weights.byteOffset,
      skinning.weights.byteLength
    )

    // skin matrices storage buffer
    const skeleton = model.getSkeleton()
    const boneCount = skeleton.bones.length
    const byteSize = boneCount * 16 * 4
    this.skinMatrixBuffer = this.device.createBuffer({
      label: "skin matrices",
      size: Math.max(256, byteSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

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

    // Load textures and prepare per-material draws
    await this.prepareMaterialDraws(model)
  }

  private materialDraws: { count: number; firstIndex: number; bindGroup: GPUBindGroup }[] = []

  private async prepareMaterialDraws(model: Model) {
    const materials = model.getMaterials()
    if (materials.length === 0) {
      // Fallback single bind group with default texture
      this.updateBindGroup()
      return
    }

    // Load all unique diffuse textures
    const textures = model.getTextures()

    const textureCache = new Map<number, { texture: GPUTexture; sampler: GPUSampler }>()

    const loadTextureByIndex = async (texIndex: number) => {
      if (texIndex < 0 || texIndex >= textures.length) return null
      if (textureCache.has(texIndex)) return textureCache.get(texIndex)!

      const path = this.modelDir + textures[texIndex].path
      const resource = await this.createTextureFromPath(path)
      if (resource) textureCache.set(texIndex, resource)
      return resource
    }

    // Preload all referenced textures (diffuse only for now)
    const uniqueDiffuse = Array.from(new Set(materials.map((m) => m.diffuseTextureIndex).filter((i) => i >= 0)))
    await Promise.all(uniqueDiffuse.map(loadTextureByIndex))

    // Default white texture if missing
    const defaultTexture = this.createDefaultTexture()
    const defaultSampler = this.device.createSampler({ magFilter: "linear", minFilter: "linear" })

    // Build batched draw calls - group consecutive materials with same texture together
    this.materialDraws = []
    let runningFirstIndex = 0
    let currentTexResource: { texture: GPUTexture; sampler: GPUSampler } | null = null
    let currentBindGroup: GPUBindGroup | null = null
    let batchedCount = 0
    let batchedFirstIndex = runningFirstIndex

    for (let i = 0; i < materials.length; i++) {
      const mat = materials[i]
      const texResource = await loadTextureByIndex(mat.diffuseTextureIndex)
      const matCount = Math.max(0, mat.vertexCount | 0) // PMX stores indexCount here

      // Check if we can batch with previous draw (same texture resource)
      const canBatch = currentTexResource === texResource && currentBindGroup !== null && matCount > 0

      if (!canBatch) {
        // Flush previous batched draw if any
        if (currentBindGroup !== null && batchedCount > 0) {
          this.materialDraws.push({
            count: batchedCount,
            firstIndex: batchedFirstIndex,
            bindGroup: currentBindGroup,
          })
        }

        // Start new batch
        const entries: GPUBindGroupEntry[] = [
          { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
          { binding: 1, resource: { buffer: this.lightUniformBuffer } },
          { binding: 2, resource: (texResource ? texResource.texture : defaultTexture).createView() },
          { binding: 3, resource: texResource ? texResource.sampler : defaultSampler },
        ]

        // Bind skin buffer (fallback if actual not present)
        entries.push({ binding: 4, resource: { buffer: this.skinMatrixBuffer || this.fallbackSkinMatrixBuffer! } })

        currentBindGroup = this.device.createBindGroup({
          label: `batched material bind group`,
          layout: this.pipeline.getBindGroupLayout(0),
          entries,
        })

        currentTexResource = texResource
        batchedFirstIndex = runningFirstIndex
        batchedCount = matCount
      } else {
        // Extend current batch
        batchedCount += matCount
      }

      runningFirstIndex += matCount
    }

    // Flush final batch if any
    if (currentBindGroup !== null && batchedCount > 0) {
      this.materialDraws.push({
        count: batchedCount,
        firstIndex: batchedFirstIndex,
        bindGroup: currentBindGroup,
      })
    }
    const total = this.materialDraws.reduce((a, d) => a + d.count, 0)
    if (this.indexCount && total !== this.indexCount) {
      console.warn(`[PMX] material index sum ${total} != indexCount ${this.indexCount}`)
    }
  }

  private async loadTexture(path: string): Promise<void> {
    try {
      // Load image
      const response = await fetch(path)
      const imageBitmap = await createImageBitmap(await response.blob())

      // Create texture
      this.diffuseTexture = this.device.createTexture({
        label: `texture: ${path}`,
        size: [imageBitmap.width, imageBitmap.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })

      // Create sampler
      this.diffuseSampler = this.device.createSampler({
        label: `sampler: ${path}`,
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "repeat",
        addressModeV: "repeat",
      })

      // Copy image data to texture
      this.device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture: this.diffuseTexture }, [
        imageBitmap.width,
        imageBitmap.height,
      ])

      // Update bind group with new texture (legacy single-texture path)
      this.updateBindGroup()
    } catch (error) {
      console.warn(`Failed to load texture: ${path}`, error)
    }
  }

  private async createTextureFromPath(path: string): Promise<{ texture: GPUTexture; sampler: GPUSampler } | null> {
    try {
      const response = await fetch(path)
      const imageBitmap = await createImageBitmap(await response.blob())
      const texture = this.device.createTexture({
        label: `texture: ${path}`,
        size: [imageBitmap.width, imageBitmap.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture }, [
        imageBitmap.width,
        imageBitmap.height,
      ])
      const sampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        addressModeU: "repeat",
        addressModeV: "repeat",
      })
      return { texture, sampler }
    } catch (e) {
      console.warn(`Failed to load texture: ${path}`, e)
      return null
    }
  }

  private updateBindGroup() {
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
      { binding: 1, resource: { buffer: this.lightUniformBuffer } },
    ]

    // Add texture and sampler if available
    if (this.diffuseTexture && this.diffuseSampler) {
      entries.push(
        { binding: 2, resource: this.diffuseTexture.createView() },
        { binding: 3, resource: this.diffuseSampler }
      )
    } else {
      // Use a default white texture if no texture is loaded
      const defaultTexture = this.createDefaultTexture()
      const defaultSampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
      })

      entries.push({ binding: 2, resource: defaultTexture.createView() }, { binding: 3, resource: defaultSampler })
    }

    // Ensure binding 4 is always present
    if (!this.skinMatrixBuffer && !this.fallbackSkinMatrixBuffer) {
      const size = 16 * 4
      this.fallbackSkinMatrixBuffer = this.device.createBuffer({
        label: "fallback skin matrices",
        size,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      })
      const identity = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
      this.device.queue.writeBuffer(this.fallbackSkinMatrixBuffer, 0, identity)
    }
    entries.push({ binding: 4, resource: { buffer: this.skinMatrixBuffer || this.fallbackSkinMatrixBuffer! } })

    this.bindGroup = this.device.createBindGroup({
      label: "model bind group",
      layout: this.pipeline.getBindGroupLayout(0),
      entries,
    })
  }

  private createDefaultTexture(): GPUTexture {
    const size = 1
    const texture = this.device.createTexture({
      size: [size, size],
      format: "rgba8unorm",
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })

    // Fill with white
    const data = new Uint8Array([255, 255, 255, 255])
    this.device.queue.writeTexture({ texture }, data, { bytesPerRow: size * 4 }, [size, size])

    return texture
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
    const cameraPos = this.camera.getPosition()

    // Combine matrices and camera position into reused buffer
    this.cameraMatrixData.set(viewMatrix.values, 0)
    this.cameraMatrixData.set(projectionMatrix.values, 16)
    this.cameraMatrixData[32] = cameraPos.x
    this.cameraMatrixData[33] = cameraPos.y
    this.cameraMatrixData[34] = cameraPos.z
    this.cameraMatrixData[35] = 0 // padding

    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)

    // Update render targets for current frame
    this.renderPassColorAttachment.view = this.multisampleTexture.createView()
    this.renderPassColorAttachment.resolveTarget = this.context.getCurrentTexture().createView()
    if (this.renderPassDescriptor.depthStencilAttachment) {
      this.renderPassDescriptor.depthStencilAttachment.view = this.depthTexture.createView()
    }

    // Reset draw call counter for this frame
    this.drawCallCount = 0

    // Calculate delta time once at the start of the frame
    const currentTime = performance.now()
    const deltaTime = this.lastFrameTime > 0 ? (currentTime - this.lastFrameTime) / 1000.0 : 0.016 // Default to ~60fps if first frame
    this.lastFrameTime = currentTime

    // Update pose and physics before rendering
    if (this.currentModel) {
      // Evaluate pose (handles rotation tweens, computes world/skin matrices)
      this.currentModel.evaluatePose(deltaTime)

      // Update physics (rigidbody states) if enabled
      if (this.physics) {
        // Step physics: syncs bone-driven rigidbodies and simulates dynamics
        // Physics modifies bone world matrices in-place, so we need to recompute skin matrices
        const skeleton = this.currentModel.getSkeleton()
        const boneCount = skeleton.bones.length
        const boneWorldMatrices = this.currentModel.getBoneWorldMatrices()
        const boneInverseBindMatrices = skeleton.inverseBindMatrices

        // Physics step modifies boneWorldMatrices in-place for dynamic rigidbodies
        this.physics.step(deltaTime, boneWorldMatrices, boneInverseBindMatrices, boneCount)

        // Recompute skin matrices from the (potentially modified) world matrices
        // This is more efficient than re-evaluating the entire pose
        this.currentModel.updateSkinMatrices()
      }

      // Update skin matrices buffer after all pose evaluation is complete
      if (this.skinMatrixBuffer) {
        const mats = this.currentModel.getSkinMatrices()
        if (mats) {
          this.device.queue.writeBuffer(this.skinMatrixBuffer, 0, mats.buffer, mats.byteOffset, mats.byteLength)
        }
      }
    }

    // Begin render pass
    const encoder = this.device.createCommandEncoder({ label: "our encoder" })
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    // Draw grid (if created)
    if (this.gridPipeline && this.gridVertexBuffer && this.gridBindGroup && this.gridVertexCount > 0) {
      pass.setPipeline(this.gridPipeline)
      pass.setVertexBuffer(0, this.gridVertexBuffer)
      pass.setBindGroup(0, this.gridBindGroup)
      pass.draw(this.gridVertexCount)
      this.drawCallCount++
    }
    // Draw model first
    pass.setPipeline(this.pipeline)
    pass.setVertexBuffer(0, this.vertexBuffer)
    if (this.jointsBuffer) pass.setVertexBuffer(1, this.jointsBuffer)
    if (this.weightsBuffer) pass.setVertexBuffer(2, this.weightsBuffer)

    // Use indexed rendering if index buffer exists
    if (this.indexBuffer) {
      pass.setIndexBuffer(this.indexBuffer, "uint32")
      if (this.materialDraws.length > 0) {
        for (const draw of this.materialDraws) {
          pass.setBindGroup(0, draw.bindGroup)
          if (draw.count > 0) {
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
            this.drawCallCount++
          }
        }
      } else {
        // Legacy single-bind-group path
        pass.setBindGroup(0, this.bindGroup)
        pass.drawIndexed(this.indexCount)
        this.drawCallCount++
      }
    } else {
      pass.setBindGroup(0, this.bindGroup)
      pass.draw(this.vertexCount)
      this.drawCallCount++
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

    this.stats.vertices = this.vertexCount
    this.stats.drawCalls = this.drawCallCount
  }
}
