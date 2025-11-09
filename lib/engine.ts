import { Camera } from "./camera"
import { Quat, Vec3 } from "./math"
import { Model, Material } from "./model"
import { PmxLoader } from "./pmx-loader"
import { Physics } from "./physics"

export interface EngineStats {
  fps: number
  frameTime: number // ms
  memoryUsed: number // MB
  vertices: number
  drawCalls: number
}

interface MaterialDraw {
  count: number
  firstIndex: number
  bindGroup: GPUBindGroup
  isTransparent: boolean
  texturePath: string
  materialIndices: number[] // Material indices for this batched draw
}

interface OutlineDraw {
  count: number
  firstIndex: number
  bindGroup: GPUBindGroup
  isTransparent: boolean
}

interface MaterialDrawInfo {
  mat: Material
  matIndex: number
  firstIndex: number
  count: number
  diffuseTexture: GPUTexture
  toonTexture: GPUTexture
  texturePath: string
  isTransparent: boolean
  materialUniformBuffer: GPUBuffer
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
  private resizeObserver: ResizeObserver | null = null
  private depthTexture!: GPUTexture
  private pipeline!: GPURenderPipeline
  private outlinePipeline!: GPURenderPipeline
  private jointsBuffer!: GPUBuffer
  private weightsBuffer!: GPUBuffer
  private skinMatrixBuffer?: GPUBuffer
  private multisampleTexture!: GPUTexture
  private readonly sampleCount = 4 // MSAA 4x
  private renderPassDescriptor!: GPURenderPassDescriptor
  private currentModel: Model | null = null
  private modelDir: string = ""
  private physics: Physics | null = null
  private textureSampler!: GPUSampler
  private textureCache = new Map<string, GPUTexture>()

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
      alphaMode: "premultiplied",
    })

    this.initCamera()
    this.initLighting()
    this.initPipeline()
    this.initOutline()
    this.initResizeObserver()
  }

  private initPipeline() {
    this.textureSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    })

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
          diffuse: vec4f,
          ambient: vec3f,
          _padding: f32,
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
        @group(0) @binding(7) var<uniform> material: MaterialUniforms;

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f,
          @location(3) joints0: vec4<u32>,
          @location(4) weights0: vec4<f32>
        ) -> VertexOutput {
          var output: VertexOutput;
          let pos4 = vec4f(position, 1.0);
          
          // Normalize weights to ensure they sum to 1.0 (handles floating-point precision issues)
          let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
          var normalizedWeights: vec4f;
          if (weightSum > 0.0001) {
            normalizedWeights = weights0 / weightSum;
          } else {
            normalizedWeights = vec4f(1.0, 0.0, 0.0, 0.0);
          }
          
          var skinnedPos = vec4f(0.0, 0.0, 0.0, 0.0);
          var skinnedNrm = vec3f(0.0, 0.0, 0.0);
          for (var i = 0u; i < 4u; i++) {
            let j = joints0[i];
            let w = normalizedWeights[i];
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
          // MMD formula: textureColor * materialDiffuseColor
          let texColor = textureSample(diffuseTexture, diffuseSampler, input.uv).rgb;
          let albedo = texColor * material.diffuse.rgb;

          // MMD uses per-material ambient color (adds warmth/redness to skin)
          // Start with material ambient instead of global ambient
          var lightAccum = material.ambient;
          
          let numLights = u32(light.lightCount);
          for (var i = 0u; i < numLights; i++) {
            let l = -light.lights[i].direction;
            let nDotL = max(dot(n, l), 0.0);
            let toonUV = vec2f(nDotL, 0.5);
            let toonFactor = textureSample(toonTexture, toonSampler, toonUV).rgb;
            let radiance = light.lights[i].color * light.lights[i].intensity;
            lightAccum += toonFactor * radiance * nDotL;
          }
          
          let color = albedo * lightAccum;
          let finalAlpha = material.diffuse.a;
          if (finalAlpha < 0.001) {
            discard;
          }
          
          return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), finalAlpha);
        }
      `,
    })

    // Single pipeline for all materials with alpha blending
    this.pipeline = this.device.createRenderPipeline({
      label: "model pipeline",
      layout: "auto",
      vertex: {
        module: shaderModule,
        buffers: [
          {
            arrayStride: 8 * 4,
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
              { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat },
              { shaderLocation: 2, offset: 6 * 4, format: "float32x2" as GPUVertexFormat },
            ],
          },
          {
            arrayStride: 4 * 2,
            attributes: [{ shaderLocation: 3, offset: 0, format: "uint16x4" as GPUVertexFormat }],
          },
          {
            arrayStride: 4,
            attributes: [{ shaderLocation: 4, offset: 0, format: "unorm8x4" as GPUVertexFormat }],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        targets: [
          {
            format: this.presentationFormat,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
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

  private initOutline() {
    const outlineShaderModule = this.device.createShaderModule({
      label: "outline shaders",
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
          viewPos: vec3f,
          _padding: f32,
        };

        struct MaterialUniforms {
          edgeColor: vec4f,
          edgeSize: f32,
          _padding1: f32,
          _padding2: f32,
          _padding3: f32,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(0) @binding(1) var<uniform> material: MaterialUniforms;
        @group(0) @binding(2) var<storage, read> skinMats: array<mat4x4f>;

        struct VertexOutput {
          @builtin(position) position: vec4f,
        };

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f,
          @location(3) joints0: vec4<u32>,
          @location(4) weights0: vec4<f32>
        ) -> VertexOutput {
          var output: VertexOutput;
          let pos4 = vec4f(position, 1.0);
          
          // Normalize weights to ensure they sum to 1.0 (handles floating-point precision issues)
          let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
          var normalizedWeights: vec4f;
          if (weightSum > 0.0001) {
            normalizedWeights = weights0 / weightSum;
          } else {
            normalizedWeights = vec4f(1.0, 0.0, 0.0, 0.0);
          }
          
          var skinnedPos = vec4f(0.0, 0.0, 0.0, 0.0);
          var skinnedNrm = vec3f(0.0, 0.0, 0.0);
          for (var i = 0u; i < 4u; i++) {
            let j = joints0[i];
            let w = normalizedWeights[i];
            let m = skinMats[j];
            skinnedPos += (m * pos4) * w;
            let r3 = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
            skinnedNrm += (r3 * normal) * w;
          }
          let worldPos = skinnedPos.xyz;
          let worldNormal = normalize(skinnedNrm);
          
          // MMD invert hull: expand vertices outward along normals
          let scaleFactor = 0.008;
          let expandedPos = worldPos + worldNormal * material.edgeSize * scaleFactor;
          output.position = camera.projection * camera.view * vec4f(expandedPos, 1.0);
          return output;
        }

        @fragment fn fs() -> @location(0) vec4f {
          return material.edgeColor;
        }
      `,
    })

    this.outlinePipeline = this.device.createRenderPipeline({
      label: "outline pipeline",
      layout: "auto",
      vertex: {
        module: outlineShaderModule,
        buffers: [
          {
            arrayStride: 8 * 4,
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: "float32x3" as GPUVertexFormat,
              },
              {
                shaderLocation: 1,
                offset: 3 * 4,
                format: "float32x3" as GPUVertexFormat,
              },
              {
                shaderLocation: 2,
                offset: 6 * 4,
                format: "float32x2" as GPUVertexFormat,
              },
            ],
          },
          {
            arrayStride: 4 * 2,
            attributes: [{ shaderLocation: 3, offset: 0, format: "uint16x4" as GPUVertexFormat }],
          },
          {
            arrayStride: 4,
            attributes: [{ shaderLocation: 4, offset: 0, format: "unorm8x4" as GPUVertexFormat }],
          },
        ],
      },
      fragment: {
        module: outlineShaderModule,
        targets: [
          {
            format: this.presentationFormat,
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
                operation: "add",
              },
            },
          },
        ],
      },
      primitive: {
        cullMode: "back",
      },
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
    const displayWidth = this.canvas.clientWidth
    const displayHeight = this.canvas.clientHeight

    const dpr = window.devicePixelRatio || 1
    const width = Math.floor(displayWidth * dpr)
    const height = Math.floor(displayHeight * dpr)

    if (!this.multisampleTexture || this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width
      this.canvas.height = height

      this.multisampleTexture = this.device.createTexture({
        label: "multisample render target",
        size: [width, height],
        sampleCount: this.sampleCount,
        format: this.presentationFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })

      this.depthTexture = this.device.createTexture({
        label: "depth texture",
        size: [width, height],
        sampleCount: this.sampleCount,
        format: "depth24plus",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })

      const depthTextureView = this.depthTexture.createView()

      const colorAttachment: GPURenderPassColorAttachment =
        this.sampleCount > 1
          ? {
              view: this.multisampleTexture.createView(),
              resolveTarget: this.context.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: "clear",
              storeOp: "store",
            }
          : {
              view: this.context.getCurrentTexture().createView(),
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: "clear",
              storeOp: "store",
            }

      this.renderPassDescriptor = {
        label: "renderPass",
        colorAttachments: [colorAttachment],
        depthStencilAttachment: {
          view: depthTextureView,
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
        },
      }

      this.camera.aspect = width / height
    }
  }

  private initCamera() {
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 40 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.camera = new Camera(Math.PI, Math.PI / 2.5, 26.6, new Vec3(0, 12.5, 0))

    this.camera.aspect = this.canvas.width / this.canvas.height
    this.camera.attachControl(this.canvas)
  }

  private initLighting() {
    this.lightUniformBuffer = this.device.createBuffer({
      label: "light uniforms",
      size: 64 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.lightCount = 0

    this.addLight(new Vec3(-0.5, -0.8, 0.5).normalize(), new Vec3(1.0, 0.95, 0.9), 0.24)
    this.addLight(new Vec3(0.7, -0.5, 0.3).normalize(), new Vec3(0.8, 0.85, 1.0), 0.18)
    this.addLight(new Vec3(0.3, -0.5, -1.0).normalize(), new Vec3(0.9, 0.9, 1.0), 0.12)
    this.device.queue.writeBuffer(this.lightUniformBuffer, 0, this.lightData)
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
    return true
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
    } else {
      throw new Error("Model must have index buffer")
    }

    await this.prepareMaterialDraws(model)
  }

  private materialDraws: MaterialDraw[] = []
  private outlineDraws: OutlineDraw[] = []

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

    const loadToonTexture = async (toonTextureIndex: number): Promise<GPUTexture> => {
      const texture = await loadTextureByIndex(toonTextureIndex)
      if (texture) return texture

      // Default toon texture fallback
      const defaultToonData = new Uint8Array(256 * 2 * 4)
      for (let i = 0; i < 256; i++) {
        const factor = i / 255.0
        const gray = Math.floor(128 + factor * 127)
        defaultToonData[i * 4] = gray
        defaultToonData[i * 4 + 1] = gray
        defaultToonData[i * 4 + 2] = gray
        defaultToonData[i * 4 + 3] = 255
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

    // First pass: collect all material draw info
    const materialInfos: MaterialDrawInfo[] = []
    let runningFirstIndex = 0

    for (let matIndex = 0; matIndex < materials.length; matIndex++) {
      const mat = materials[matIndex]
      const matCount = mat.vertexCount | 0
      if (matCount === 0) continue

      const diffuseTexture = await loadTextureByIndex(mat.diffuseTextureIndex)
      if (!diffuseTexture) throw new Error(`Material "${mat.name}" has no diffuse texture`)

      const toonTexture = await loadToonTexture(mat.toonTextureIndex)
      const texturePath =
        mat.diffuseTextureIndex >= 0 && mat.diffuseTextureIndex < textures.length
          ? textures[mat.diffuseTextureIndex].path
          : ""

      const materialAlpha = mat.diffuse[3]
      const EPSILON = 0.001
      const isTransparent = materialAlpha < 1.0 - EPSILON

      // MMD material uniforms: diffuse (RGBA) + ambient (RGB)
      // Material ambient adds warmth/redness to skin tones
      const materialUniformData = new Float32Array(8)
      // Diffuse color (RGBA)
      materialUniformData[0] = mat.diffuse[0] // R
      materialUniformData[1] = mat.diffuse[1] // G
      materialUniformData[2] = mat.diffuse[2] // B
      materialUniformData[3] = mat.diffuse[3] // A
      // Ambient color (RGB) - important for skin tone warmth
      materialUniformData[4] = mat.ambient[0] // R
      materialUniformData[5] = mat.ambient[1] // G
      materialUniformData[6] = mat.ambient[2] // B
      materialUniformData[7] = 0.0 // padding

      const materialUniformBuffer = this.device.createBuffer({
        label: `material uniform: ${mat.name}`,
        size: materialUniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(materialUniformBuffer, 0, materialUniformData)

      materialInfos.push({
        mat,
        matIndex,
        firstIndex: runningFirstIndex,
        count: matCount,
        diffuseTexture,
        toonTexture,
        texturePath,
        isTransparent,
        materialUniformBuffer,
      })

      runningFirstIndex += matCount
    }

    // Sort materials by texture path to group same textures together
    // This reduces texture switching overhead, especially for large textures
    materialInfos.sort((a, b) => {
      // First sort by transparency (opaque first)
      if (a.isTransparent !== b.isTransparent) {
        return a.isTransparent ? 1 : -1
      }
      // Then sort by texture path
      return a.texturePath.localeCompare(b.texturePath)
    })

    // Build bind groups and batch consecutive materials with same texture
    this.materialDraws = []
    this.outlineDraws = []
    const outlineBindGroupLayout = this.outlinePipeline.getBindGroupLayout(0)

    // Batch consecutive materials that share the same texture
    for (let i = 0; i < materialInfos.length; i++) {
      const firstInfo = materialInfos[i]
      let batchCount = firstInfo.count
      const batchFirstIndex = firstInfo.firstIndex
      const batchMaterialIndices = [firstInfo.matIndex]

      // Combine consecutive materials with the same texture AND same material properties
      let j = i + 1
      while (
        j < materialInfos.length &&
        materialInfos[j].texturePath === firstInfo.texturePath &&
        materialInfos[j].isTransparent === firstInfo.isTransparent &&
        materialInfos[j].firstIndex === batchFirstIndex + batchCount
      ) {
        // Check if material properties are identical (can share the same uniform)
        const mat1 = firstInfo.mat
        const mat2 = materialInfos[j].mat
        const sameDiffuse =
          mat1.diffuse[0] === mat2.diffuse[0] &&
          mat1.diffuse[1] === mat2.diffuse[1] &&
          mat1.diffuse[2] === mat2.diffuse[2] &&
          mat1.diffuse[3] === mat2.diffuse[3]
        const sameAmbient =
          mat1.ambient[0] === mat2.ambient[0] &&
          mat1.ambient[1] === mat2.ambient[1] &&
          mat1.ambient[2] === mat2.ambient[2]

        if (sameDiffuse && sameAmbient) {
          // Same texture, same transparency, same material properties, and consecutive index range - can batch!
          batchCount += materialInfos[j].count
          batchMaterialIndices.push(materialInfos[j].matIndex)
          j++
        } else {
          // Different material properties - can't batch
          break
        }
      }

      // Use the first material's bind group (they all share the same texture)
      const bindGroup = this.device.createBindGroup({
        label: `material bind group: ${firstInfo.mat.name} (batched)`,
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
          { binding: 1, resource: { buffer: this.lightUniformBuffer } },
          { binding: 2, resource: firstInfo.diffuseTexture.createView() },
          { binding: 3, resource: this.textureSampler },
          { binding: 4, resource: { buffer: this.skinMatrixBuffer! } },
          { binding: 5, resource: firstInfo.toonTexture.createView() },
          { binding: 6, resource: this.textureSampler },
          { binding: 7, resource: { buffer: firstInfo.materialUniformBuffer } },
        ],
      })

      this.materialDraws.push({
        count: batchCount,
        firstIndex: batchFirstIndex,
        bindGroup,
        isTransparent: firstInfo.isTransparent,
        texturePath: firstInfo.texturePath,
        materialIndices: batchMaterialIndices,
      })

      // Skip the batched materials
      i = j - 1
    }

    // Process outlines for all materials (in original order to match indices)
    runningFirstIndex = 0
    for (const mat of materials) {
      const matCount = mat.vertexCount | 0
      if (matCount === 0) continue

      const materialAlpha = mat.diffuse[3]
      const EPSILON = 0.001
      const isTransparent = materialAlpha < 1.0 - EPSILON

      // Outline for all materials (including transparent)
      if ((mat.edgeFlag & 0x01) !== 0 || mat.edgeSize > 0) {
        const materialUniformData = new Float32Array(8)
        materialUniformData[0] = mat.edgeColor[0]
        materialUniformData[1] = mat.edgeColor[1]
        materialUniformData[2] = mat.edgeColor[2]
        materialUniformData[3] = mat.edgeColor[3]
        materialUniformData[4] = mat.edgeSize

        const materialUniformBuffer = this.device.createBuffer({
          label: `outline material uniform: ${mat.name}`,
          size: materialUniformData.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(materialUniformBuffer, 0, materialUniformData)

        const outlineBindGroup = this.device.createBindGroup({
          label: `outline bind group: ${mat.name}`,
          layout: outlineBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
            { binding: 1, resource: { buffer: materialUniformBuffer } },
            { binding: 2, resource: { buffer: this.skinMatrixBuffer! } },
          ],
        })

        // All outlines use the same pipeline
        this.outlineDraws.push({
          count: matCount,
          firstIndex: runningFirstIndex,
          bindGroup: outlineBindGroup,
          isTransparent,
        })
      }

      runningFirstIndex += matCount
    }
  }

  private async createTextureFromPath(path: string): Promise<GPUTexture | null> {
    const cached = this.textureCache.get(path)
    if (cached) {
      return cached
    }

    try {
      const response = await fetch(path)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const imageBitmap = await createImageBitmap(await response.blob(), {
        premultiplyAlpha: "none",
        colorSpaceConversion: "none",
      })
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

      this.textureCache.set(path, texture)
      return texture
    } catch {
      return null
    }
  }

  public render() {
    if (this.multisampleTexture && this.camera && this.device && this.currentModel) {
      const currentTime = performance.now()
      const deltaTime = this.lastFrameTime > 0 ? (currentTime - this.lastFrameTime) / 1000 : 0.016
      this.lastFrameTime = currentTime

      const viewMatrix = this.camera.getViewMatrix()
      const projectionMatrix = this.camera.getProjectionMatrix()
      const cameraPos = this.camera.getPosition()
      this.cameraMatrixData.set(viewMatrix.values, 0)
      this.cameraMatrixData.set(projectionMatrix.values, 16)
      this.cameraMatrixData[32] = cameraPos.x
      this.cameraMatrixData[33] = cameraPos.y
      this.cameraMatrixData[34] = cameraPos.z
      this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)

      const colorAttachment = (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
      if (this.sampleCount > 1) {
        colorAttachment.resolveTarget = this.context.getCurrentTexture().createView()
      } else {
        colorAttachment.view = this.context.getCurrentTexture().createView()
      }

      this.currentModel.evaluatePose()
      if (this.physics) {
        this.physics.step(
          deltaTime,
          this.currentModel.getBoneWorldMatrices(),
          this.currentModel.getBoneInverseBindMatrices()
        )
        this.currentModel.updateSkinMatrices()
      }

      const mats = this.currentModel.getSkinMatrices()
      this.device.queue.writeBuffer(this.skinMatrixBuffer!, 0, mats.buffer, mats.byteOffset, mats.byteLength)

      const encoder = this.device.createCommandEncoder()
      const pass = encoder.beginRenderPass(this.renderPassDescriptor)

      pass.setVertexBuffer(0, this.vertexBuffer)
      pass.setVertexBuffer(1, this.jointsBuffer)
      pass.setVertexBuffer(2, this.weightsBuffer)
      pass.setIndexBuffer(this.indexBuffer!, "uint32")

      this.drawCallCount = 0

      // Render opaque outlines first (MMD invert hull method)
      if (this.outlineDraws.length > 0) {
        pass.setPipeline(this.outlinePipeline)
        for (const draw of this.outlineDraws) {
          if (draw.count > 0 && !draw.isTransparent) {
            pass.setBindGroup(0, draw.bindGroup)
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          }
        }
      }

      // Render opaque materials
      pass.setPipeline(this.pipeline)
      for (const draw of this.materialDraws) {
        if (draw.count > 0 && !draw.isTransparent) {
          pass.setBindGroup(0, draw.bindGroup)
          pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          this.drawCallCount++
        }
      }

      // Render transparent materials
      for (const draw of this.materialDraws) {
        if (draw.count > 0 && draw.isTransparent) {
          pass.setBindGroup(0, draw.bindGroup)
          pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          this.drawCallCount++
        }
      }

      // Render transparent outlines after their materials
      if (this.outlineDraws.length > 0) {
        pass.setPipeline(this.outlinePipeline)
        for (const draw of this.outlineDraws) {
          if (draw.count > 0 && draw.isTransparent) {
            pass.setBindGroup(0, draw.bindGroup)
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          }
        }
      }

      pass.end()
      this.device.queue.submit([encoder.finish()])
      this.updateStats(performance.now() - currentTime)
    }
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
