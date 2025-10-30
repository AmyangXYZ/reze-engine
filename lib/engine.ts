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
  private cameraMatrixData = new Float32Array(36) // view(16) + projection(16) + viewPos(3) + padding(1) = 36 floats
  private lightUniformBuffer!: GPUBuffer
  // Multi-light system: ambient(1) + padding(3), then 4 lights: each light = direction(3) + padding(1), color(3) + intensity(1) = 8 floats per light
  // Total: 4 + (4 * 8) = 36 floats, padded to 64 floats (256 bytes) for proper alignment
  private lightData = new Float32Array(64)
  private lightCount: number = 0 // Number of active lights (0-4)
  private materialUniformBuffer!: GPUBuffer
  // Material properties: baseColor(3) + metallic(1), roughness(1) + padding(3) = 8 floats (32 bytes, needs 64 byte alignment)
  private materialData = new Float32Array(16) // padded to 64 bytes
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

    // Create uniform buffer for material properties (64 bytes aligned)
    this.materialUniformBuffer = this.device.createBuffer({
      label: "material uniforms",
      size: 16 * 4, // 16 floats * 4 bytes = 64 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Initialize default PBR material (non-metallic, semi-glossy - good for anime characters)
    this.setMaterial(new Vec3(0.94, 0.88, 0.82), 0.0, 0.45)

    // Initialize MMD-style multi-light setup
    this.setAmbient(0.6) // Reduced ambient to make lights more visible
    this.clearLights()
    // Key light (main, bright from front-right)
    this.addLight(new Vec3(-0.5, -0.8, 0.5).normalize(), new Vec3(1.0, 0.95, 0.9), 1.2)
    // Fill light (softer from left)`
    this.addLight(new Vec3(0.7, -0.5, 0.3).normalize(), new Vec3(0.8, 0.85, 1.0), 1.1)
    // Rim light (from behind for edge highlighting)
    this.addLight(new Vec3(0.3, -0.5, -1.0).normalize(), new Vec3(0.9, 0.9, 1.0), 1)

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

        struct MaterialUniforms {
          baseColor: vec3f,
          metallic: f32,
          roughness: f32,
          _padding: vec3f,
        };

        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) normal: vec3f,
          @location(1) uv: vec2f,
          @location(2) worldPos: vec3f,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(0) @binding(1) var<uniform> light: LightUniforms;
        @group(0) @binding(2) var<uniform> material: MaterialUniforms;
        @group(0) @binding(3) var diffuseTexture: texture_2d<f32>;
        @group(0) @binding(4) var diffuseSampler: sampler;

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f
        ) -> VertexOutput {
          var output: VertexOutput;
          let worldPos = position; // Model space = world space for now (no model matrix)
          output.position = camera.projection * camera.view * vec4f(position, 1.0);
          output.normal = normal;
          output.uv = uv;
          output.worldPos = worldPos;
          return output;
        }
  
        // PBR helper functions (Cook-Torrance BRDF)
        fn fresnelSchlick(cosTheta: f32, f0: vec3f) -> vec3f {
          return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
        }

        fn distributionGGX(n: vec3f, h: vec3f, roughness: f32) -> f32 {
          let a = roughness * roughness;
          let a2 = a * a;
          let nDotH = max(dot(n, h), 0.0);
          let nDotH2 = nDotH * nDotH;
          let denom = (nDotH2 * (a2 - 1.0) + 1.0);
          return a2 / (3.14159265 * denom * denom);
        }

        fn geometrySchlickGGX(nDotV: f32, roughness: f32) -> f32 {
          let r = (roughness + 1.0);
          let k = (r * r) / 8.0;
          return nDotV / (nDotV * (1.0 - k) + k);
        }

        fn geometrySmith(n: vec3f, v: vec3f, l: vec3f, roughness: f32) -> f32 {
          let nDotV = max(dot(n, v), 0.0);
          let nDotL = max(dot(n, l), 0.0);
          let ggx2 = geometrySchlickGGX(nDotV, roughness);
          let ggx1 = geometrySchlickGGX(nDotL, roughness);
          return ggx1 * ggx2;
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

  // Set PBR material properties
  public setMaterial(baseColor: Vec3, metallic: number, roughness: number) {
    // Clamp values to valid ranges
    this.materialData[0] = Math.max(0.0, Math.min(1.0, baseColor.x))
    this.materialData[1] = Math.max(0.0, Math.min(1.0, baseColor.y))
    this.materialData[2] = Math.max(0.0, Math.min(1.0, baseColor.z))
    this.materialData[3] = Math.max(0.0, Math.min(1.0, metallic))
    this.materialData[4] = Math.max(0.01, Math.min(1.0, roughness)) // Roughness must be > 0
    // Padding is already zeros
    this.device.queue.writeBuffer(this.materialUniformBuffer, 0, this.materialData)
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
    await this.drawModel(model)
  }

  private async drawModel(model: RzmModel) {
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

    // Load textures and prepare per-material draws
    await this.prepareMaterialDraws(model)
  }

  private materialDraws: { count: number; firstIndex: number; bindGroup: GPUBindGroup }[] = []

  private async prepareMaterialDraws(model: RzmModel) {
    const materials = model.getMaterials()
    if (materials.length === 0) {
      // Fallback single bind group with default texture
      this.updateBindGroup()
      return
    }

    // Load all unique diffuse textures
    const modelDir = "/models/梵天/"
    const textures = model.getTextures()

    const textureCache = new Map<number, { texture: GPUTexture; sampler: GPUSampler }>()

    const loadTextureByIndex = async (texIndex: number) => {
      if (texIndex < 0 || texIndex >= textures.length) return null
      if (textureCache.has(texIndex)) return textureCache.get(texIndex)!

      const path = modelDir + textures[texIndex].path
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

    // Build per-material bind groups and draw ranges
    this.materialDraws = []
    let runningFirstIndex = 0
    for (let i = 0; i < materials.length; i++) {
      const mat = materials[i]
      const texResource = await loadTextureByIndex(mat.diffuseTextureIndex)

      const entries: GPUBindGroupEntry[] = [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.lightUniformBuffer } },
        { binding: 3, resource: (texResource ? texResource.texture : defaultTexture).createView() },
        { binding: 4, resource: texResource ? texResource.sampler : defaultSampler },
      ]

      const bindGroup = this.device.createBindGroup({
        label: `material ${i} bind group`,
        layout: this.pipeline.getBindGroupLayout(0),
        entries,
      })

      const count = Math.max(0, mat.vertexCount | 0) // PMX stores indexCount here
      this.materialDraws.push({ count, firstIndex: runningFirstIndex, bindGroup })
      runningFirstIndex += count
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
        { binding: 3, resource: this.diffuseTexture.createView() },
        { binding: 4, resource: this.diffuseSampler }
      )
    } else {
      // Use a default white texture if no texture is loaded
      const defaultTexture = this.createDefaultTexture()
      const defaultSampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear",
      })

      entries.push({ binding: 3, resource: defaultTexture.createView() }, { binding: 4, resource: defaultSampler })
    }

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

    const encoder = this.device.createCommandEncoder({ label: "our encoder" })
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    pass.setPipeline(this.pipeline)
    pass.setVertexBuffer(0, this.vertexBuffer)

    // Use indexed rendering if index buffer exists
    if (this.indexBuffer) {
      pass.setIndexBuffer(this.indexBuffer, "uint32")
      if (this.materialDraws.length > 0) {
        for (const draw of this.materialDraws) {
          pass.setBindGroup(0, draw.bindGroup)
          if (draw.count > 0) {
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          }
        }
      } else {
        // Legacy single-bind-group path
        pass.setBindGroup(0, this.bindGroup)
        pass.drawIndexed(this.indexCount)
      }
    } else {
      pass.setBindGroup(0, this.bindGroup)
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
