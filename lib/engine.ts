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
  private cameraMatrixData = new Float32Array(36)
  private lightUniformBuffer!: GPUBuffer
  private lightData = new Float32Array(64)
  private lightCount = 0
  private vertexBuffer!: GPUBuffer
  private vertexCount: number = 0
  private indexBuffer?: GPUBuffer
  private indexCount: number = 0
  private resizeObserver: ResizeObserver | null = null
  private depthTexture!: GPUTexture
  private pipeline!: GPURenderPipeline
  private jointsBuffer!: GPUBuffer
  private weightsBuffer!: GPUBuffer
  private skinMatrixBuffer?: GPUBuffer
  private multisampleTexture!: GPUTexture
  private readonly sampleCount = 4 // MSAA 4x
  private currentModel: Model | null = null
  private modelDir: string = ""
  private physics: Physics | null = null
  private textureSampler!: GPUSampler
  private textureCache = new Map<string, GPUTexture>() // Cache for loaded textures by path

  // Stats tracking
  private lastFpsUpdate = performance.now()
  private framesSinceLastUpdate = 0
  private frameTimeSamples: number[] = []
  private frameTimeSum: number = 0
  private drawCallCount: number = 0
  private lastFrameTime = performance.now()
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

    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 40 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.lightUniformBuffer = this.device.createBuffer({
      label: "light uniforms",
      size: 64 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.textureSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    })

    this.setAmbient(0.7)
    this.addLight(new Vec3(-0.5, -0.8, 0.5).normalize(), new Vec3(1.0, 0.95, 0.9), 0.28)
    this.addLight(new Vec3(0.7, -0.5, 0.3).normalize(), new Vec3(0.8, 0.85, 1.0), 0.24)
    this.addLight(new Vec3(0.3, -0.5, -1.0).normalize(), new Vec3(0.9, 0.9, 1.0), 0.2)

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
        @group(0) @binding(5) var toonTexture: texture_2d<f32>;
        @group(0) @binding(6) var toonSampler: sampler;

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f,
          @location(3) joints0: vec4<u32>,
          @location(4) weights0: vec4<f32>
        ) -> VertexOutput {
          var output: VertexOutput;
          let pos4 = vec4f(position, 1.0);
          var skinnedPos = vec4f(0.0, 0.0, 0.0, 0.0);
          var skinnedNrm = vec3f(0.0, 0.0, 0.0);
          for (var i = 0u; i < 4u; i++) {
            let j = joints0[i];
            let w = weights0[i];
            let m = skinMats[j];
            skinnedPos += (m * pos4) * w;
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

          // Accumulate light contribution
          var lightAccum = vec3f(light.ambient);
          let numLights = u32(light.lightCount);
          for (var i = 0u; i < numLights; i++) {
            let l = -light.lights[i].direction;
            let nDotL = max(dot(n, l), 0.0);
            
            // Sample toon texture: use N·L as U coordinate, 0.5 as V (middle of texture)
            // Must sample outside conditional for uniform control flow
            let toonUV = vec2f(nDotL, 0.5);
            let toonFactor = textureSample(toonTexture, toonSampler, toonUV).rgb;
            
            // Accumulate light contribution through toon factor
            let radiance = light.lights[i].color * light.lights[i].intensity;
            lightAccum += toonFactor * radiance * nDotL;
          }
          
          // Multiply albedo with accumulated lighting
          var color = albedo * lightAccum;
          
          return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), 1.0);
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

      // Update camera aspect ratio
      this.camera.aspect = width / height
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

  public clearLights() {
    this.lightCount = 0
    this.lightData[1] = 0
    this.updateLightBuffer()
  }

  public addLight(direction: Vec3, color: Vec3, intensity: number = 1.0): boolean {
    if (this.lightCount >= 4) return false

    const normalized = direction.normalize()
    const baseIndex = 4 + this.lightCount * 8
    this.lightData[baseIndex] = normalized.x
    this.lightData[baseIndex + 1] = normalized.y
    this.lightData[baseIndex + 2] = normalized.z
    this.lightData[baseIndex + 3] = 0
    this.lightData[baseIndex + 4] = color.x
    this.lightData[baseIndex + 5] = color.y
    this.lightData[baseIndex + 6] = color.z
    this.lightData[baseIndex + 7] = intensity

    this.lightCount++
    this.lightData[1] = this.lightCount
    this.updateLightBuffer()
    return true
  }

  public setAmbient(intensity: number) {
    this.lightData[0] = intensity
    this.updateLightBuffer()
  }

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
    this.stopRenderLoop()
    if (this.camera) this.camera.detachControl()
    if (this.resizeObserver) {
      this.resizeObserver.disconnect()
      this.resizeObserver = null
    }
  }

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
    const skinning = model.getSkinning()
    const skeleton = model.getSkeleton()

    this.vertexBuffer = this.device.createBuffer({
      label: "model vertex buffer",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices)
    this.vertexCount = model.getVertexCount()

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

    const boneCount = skeleton.bones.length
    this.skinMatrixBuffer = this.device.createBuffer({
      label: "skin matrices",
      size: Math.max(256, boneCount * 16 * 4),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

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
      throw new Error("Model must have index buffer")
    }

    await this.prepareMaterialDraws(model)
  }

  private materialDraws: { count: number; firstIndex: number; bindGroup: GPUBindGroup }[] = []

  private async prepareMaterialDraws(model: Model) {
    const materials = model.getMaterials()
    if (materials.length === 0) {
      throw new Error("Model has no materials")
    }

    const textures = model.getTextures()

    const loadTextureByIndex = async (texIndex: number): Promise<GPUTexture | null> => {
      if (texIndex < 0 || texIndex >= textures.length) {
        return null
      }

      const path = this.modelDir + textures[texIndex].path
      const texture = await this.createTextureFromPath(path)
      return texture
    }

    // Load default toon texture (shared toon textures are typically in "toon" folder)
    // If material has toonTextureIndex >= 0, use that; otherwise use a default
    const loadToonTexture = async (toonTextureIndex: number): Promise<GPUTexture> => {
      if (toonTextureIndex >= 0 && toonTextureIndex < textures.length) {
        const texture = await loadTextureByIndex(toonTextureIndex)
        if (texture) return texture
      }
      // Fallback: create a simple default toon gradient (white to gray)
      // In practice, PMX models usually have toon textures, but this is a safe fallback
      const defaultToonData = new Uint8Array(256 * 2 * 4) // 256x2 RGBA
      for (let i = 0; i < 256; i++) {
        const factor = i / 255.0
        const gray = Math.floor(128 + factor * 127) // 128 to 255
        defaultToonData[i * 4] = gray
        defaultToonData[i * 4 + 1] = gray
        defaultToonData[i * 4 + 2] = gray
        defaultToonData[i * 4 + 3] = 255
        // Second row same as first
        defaultToonData[(256 + i) * 4] = gray
        defaultToonData[(256 + i) * 4 + 1] = gray
        defaultToonData[(256 + i) * 4 + 2] = gray
        defaultToonData[(256 + i) * 4 + 3] = 255
      }
      const defaultToonTexture = this.device.createTexture({
        label: "default toon texture",
        size: [256, 2],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      this.device.queue.writeTexture(
        { texture: defaultToonTexture },
        defaultToonData,
        { bytesPerRow: 256 * 4 },
        [256, 2]
      )
      return defaultToonTexture
    }

    // Create one draw call per material
    this.materialDraws = []
    const bindGroupLayout = this.pipeline.getBindGroupLayout(0)
    let runningFirstIndex = 0

    for (const mat of materials) {
      const matCount = mat.vertexCount | 0
      if (matCount === 0) continue

      const diffuseTexture = await loadTextureByIndex(mat.diffuseTextureIndex)
      if (!diffuseTexture) throw new Error(`Material "${mat.name}" has no diffuse texture`)

      const toonTexture = await loadToonTexture(mat.toonTextureIndex)

      const bindGroup = this.device.createBindGroup({
        label: `material bind group: ${mat.name}`,
        layout: bindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
          { binding: 1, resource: { buffer: this.lightUniformBuffer } },
          { binding: 2, resource: diffuseTexture.createView() },
          { binding: 3, resource: this.textureSampler },
          { binding: 4, resource: { buffer: this.skinMatrixBuffer! } },
          { binding: 5, resource: toonTexture.createView() },
          { binding: 6, resource: this.textureSampler },
        ],
      })

      this.materialDraws.push({
        count: matCount,
        firstIndex: runningFirstIndex,
        bindGroup,
      })

      runningFirstIndex += matCount
    }
  }

  private async createTextureFromPath(path: string): Promise<GPUTexture | null> {
    // Check cache first
    const cached = this.textureCache.get(path)
    if (cached) {
      return cached
    }

    try {
      const response = await fetch(path)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
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

      // Cache the texture
      this.textureCache.set(path, texture)
      return texture
    } catch (e) {
      console.warn(`Failed to load texture: ${path}`, e)
      return null
    }
  }

  public render() {
    if (!this.multisampleTexture || !this.camera || !this.device || !this.currentModel) return

    const currentTime = performance.now()
    const deltaTime = this.lastFrameTime > 0 ? (currentTime - this.lastFrameTime) / 1000 : 0.016
    this.lastFrameTime = currentTime

    // Update camera uniforms
    const viewMatrix = this.camera.getViewMatrix()
    const projectionMatrix = this.camera.getProjectionMatrix()
    const cameraPos = this.camera.getPosition()
    this.cameraMatrixData.set(viewMatrix.values, 0)
    this.cameraMatrixData.set(projectionMatrix.values, 16)
    this.cameraMatrixData[32] = cameraPos.x
    this.cameraMatrixData[33] = cameraPos.y
    this.cameraMatrixData[34] = cameraPos.z
    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)

    // Create render pass descriptor
    const multisampleTextureView = this.multisampleTexture.createView()
    const depthTextureView = this.depthTexture.createView()
    const renderPassDescriptor: GPURenderPassDescriptor = {
      label: "renderPass",
      colorAttachments: [
        {
          view: multisampleTextureView,
          resolveTarget: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.05, g: 0.066, b: 0.086, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: depthTextureView,
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    }

    // Update pose and physics
    this.currentModel.evaluatePose()
    if (this.physics) {
      const skeleton = this.currentModel.getSkeleton()
      this.physics.step(
        deltaTime,
        this.currentModel.getBoneWorldMatrices(),
        skeleton.inverseBindMatrices,
        skeleton.bones.length
      )
      this.currentModel.updateSkinMatrices()
    }

    // Update skin matrices buffer
    const mats = this.currentModel.getSkinMatrices()
    this.device.queue.writeBuffer(this.skinMatrixBuffer!, 0, mats.buffer, mats.byteOffset, mats.byteLength)

    // Render
    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(renderPassDescriptor)
    pass.setPipeline(this.pipeline)
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.setVertexBuffer(1, this.jointsBuffer)
    pass.setVertexBuffer(2, this.weightsBuffer)
    pass.setIndexBuffer(this.indexBuffer!, "uint32")

    this.drawCallCount = 0
    for (const draw of this.materialDraws) {
      if (draw.count > 0) {
        pass.setBindGroup(0, draw.bindGroup)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
        this.drawCallCount++
      }
    }

    pass.end()
    this.device.queue.submit([encoder.finish()])
    this.updateStats(performance.now() - currentTime)
  }

  private updateStats(frameTime: number) {
    const maxSamples = 60
    this.frameTimeSamples.push(frameTime)
    this.frameTimeSum += frameTime
    if (this.frameTimeSamples.length > maxSamples) {
      const removed = this.frameTimeSamples.shift()!
      this.frameTimeSum -= removed
    }
    const avgFrameTime = this.frameTimeSum / this.frameTimeSamples.length
    this.stats.frameTime = Math.round(avgFrameTime * 100) / 100

    const now = performance.now()
    this.framesSinceLastUpdate++
    const elapsed = now - this.lastFpsUpdate

    if (elapsed >= 1000) {
      this.stats.fps = Math.round((this.framesSinceLastUpdate / elapsed) * 1000)
      this.framesSinceLastUpdate = 0
      this.lastFpsUpdate = now

      const perf = performance as Performance & {
        memory?: { usedJSHeapSize: number; totalJSHeapSize: number }
      }
      if (perf.memory) {
        this.stats.memoryUsed = Math.round(perf.memory.usedJSHeapSize / 1024 / 1024)
      }
    }

    this.stats.vertices = this.vertexCount
    this.stats.drawCalls = this.drawCallCount
  }
}
