import { Camera } from "./camera"
import { Quat, Vec3 } from "./math"
import { Model } from "./model"
import { PmxLoader } from "./pmx-loader"
import { Physics } from "./physics"

export interface EngineStats {
  fps: number
  frameTime: number // ms
  memoryUsed: number // MB (JS heap)
  vertices: number
  drawCalls: number
  triangles: number
  materials: number
  textures: number
  textureMemory: number // MB
  bufferMemory: number // MB
  gpuMemory: number // MB (estimated total GPU memory)
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
  private worldMatrixBuffer?: GPUBuffer
  private inverseBindMatrixBuffer?: GPUBuffer
  private skinMatrixComputePipeline?: GPUComputePipeline
  private boneCountBuffer?: GPUBuffer
  private multisampleTexture!: GPUTexture
  private readonly sampleCount = 4 // MSAA 4x
  private renderPassDescriptor!: GPURenderPassDescriptor
  private currentModel: Model | null = null
  private modelDir: string = ""
  private physics: Physics | null = null
  private textureSampler!: GPUSampler
  private textureCache = new Map<string, GPUTexture>()
  private textureSizes = new Map<string, { width: number; height: number }>()

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
    triangles: 0,
    materials: 0,
    textures: 0,
    textureMemory: 0,
    bufferMemory: 0,
    gpuMemory: 0,
  }
  private animationFrameId: number | null = null
  private renderLoopCallback: (() => void) | null = null

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  // Step 1: Get WebGPU device and context
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

    this.setupCamera()
    this.setupLighting()
    this.createPipelines()
    this.setupResize()
  }

  // Step 2: Create shaders and render pipelines
  private createPipelines() {
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
          alpha: f32,
          _padding1: f32,
          _padding2: f32,
          _padding3: f32,
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
          let albedo = textureSample(diffuseTexture, diffuseSampler, input.uv).rgb;

          var lightAccum = vec3f(light.ambient);
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
          let finalAlpha = material.alpha;
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
          let scaleFactor = 0.01;
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

  // Create compute shader for skin matrix computation
  private createSkinMatrixComputePipeline() {
    const computeShader = this.device.createShaderModule({
      label: "skin matrix compute",
      code: /* wgsl */ `
        struct BoneCountUniform {
          count: u32,
          _padding1: u32,
          _padding2: u32,
          _padding3: u32,
          _padding4: vec4<u32>,
        };
        
        @group(0) @binding(0) var<uniform> boneCount: BoneCountUniform;
        @group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;
        @group(0) @binding(2) var<storage, read> inverseBindMatrices: array<mat4x4f>;
        @group(0) @binding(3) var<storage, read_write> skinMatrices: array<mat4x4f>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
          let boneIndex = globalId.x;
          // Bounds check: we dispatch workgroups (64 threads each), so some threads may be out of range
          if (boneIndex >= boneCount.count) {
            return;
          }
          let worldMat = worldMatrices[boneIndex];
          let invBindMat = inverseBindMatrices[boneIndex];
          skinMatrices[boneIndex] = worldMat * invBindMat;
        }
      `,
    })

    this.skinMatrixComputePipeline = this.device.createComputePipeline({
      label: "skin matrix compute pipeline",
      layout: "auto",
      compute: {
        module: computeShader,
      },
    })
  }

  // Step 3: Setup canvas resize handling
  private setupResize() {
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

  // Step 4: Create camera and uniform buffer
  private setupCamera() {
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 40 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.camera = new Camera(Math.PI, Math.PI / 2.5, 26.6, new Vec3(0, 12.5, 0))

    this.camera.aspect = this.canvas.width / this.canvas.height
    this.camera.attachControl(this.canvas)
  }

  // Step 5: Create lighting buffers
  private setupLighting() {
    this.lightUniformBuffer = this.device.createBuffer({
      label: "light uniforms",
      size: 64 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.lightCount = 0

    this.setAmbient(0.96)
    this.addLight(new Vec3(-0.5, -0.8, 0.5).normalize(), new Vec3(1.0, 0.95, 0.9), 0.12)
    this.addLight(new Vec3(0.7, -0.5, 0.3).normalize(), new Vec3(0.8, 0.85, 1.0), 0.1)
    this.addLight(new Vec3(0.3, -0.5, -1.0).normalize(), new Vec3(0.9, 0.9, 1.0), 0.08)
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

  public setAmbient(intensity: number) {
    this.lightData[0] = intensity
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

  // Step 6: Load PMX model file
  public async loadModel(path: string) {
    const pathParts = path.split("/")
    pathParts.pop()
    const dir = pathParts.join("/") + "/"
    this.modelDir = dir

    const model = await PmxLoader.load(path)
    this.physics = new Physics(model.getRigidbodies(), model.getJoints())
    await this.setupModelBuffers(model)
  }

  public rotateBones(bones: string[], rotations: Quat[], durationMs?: number) {
    this.currentModel?.rotateBones(bones, rotations, durationMs)
  }

  // Step 7: Create vertex, index, and joint buffers
  private async setupModelBuffers(model: Model) {
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
    const matrixSize = boneCount * 16 * 4

    this.skinMatrixBuffer = this.device.createBuffer({
      label: "skin matrices",
      size: Math.max(256, matrixSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    })

    this.worldMatrixBuffer = this.device.createBuffer({
      label: "world matrices",
      size: Math.max(256, matrixSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    this.inverseBindMatrixBuffer = this.device.createBuffer({
      label: "inverse bind matrices",
      size: Math.max(256, matrixSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    const invBindMatrices = skeleton.inverseBindMatrices
    this.device.queue.writeBuffer(
      this.inverseBindMatrixBuffer,
      0,
      invBindMatrices.buffer,
      invBindMatrices.byteOffset,
      invBindMatrices.byteLength
    )

    this.boneCountBuffer = this.device.createBuffer({
      label: "bone count uniform",
      size: 32, // Minimum uniform buffer size is 32 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const boneCountData = new Uint32Array(8) // 32 bytes total
    boneCountData[0] = boneCount
    this.device.queue.writeBuffer(this.boneCountBuffer, 0, boneCountData)

    this.createSkinMatrixComputePipeline()

    const indices = model.getIndices()
    if (indices) {
      this.indexBuffer = this.device.createBuffer({
        label: "model index buffer",
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(this.indexBuffer, 0, indices)
    } else {
      throw new Error("Model has no index buffer")
    }

    await this.setupMaterials(model)
  }

  private materialDraws: { count: number; firstIndex: number; bindGroup: GPUBindGroup; isTransparent: boolean }[] = []
  private outlineDraws: { count: number; firstIndex: number; bindGroup: GPUBindGroup; isTransparent: boolean }[] = []

  // Step 8: Load textures and create material bind groups
  private async setupMaterials(model: Model) {
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
      this.textureSizes.set("__default_toon__", { width: 256, height: 2 })
      return defaultToonTexture
    }

    this.materialDraws = []
    this.outlineDraws = []
    const outlineBindGroupLayout = this.outlinePipeline.getBindGroupLayout(0)
    let runningFirstIndex = 0

    for (const mat of materials) {
      const matCount = mat.vertexCount | 0
      if (matCount === 0) continue

      const diffuseTexture = await loadTextureByIndex(mat.diffuseTextureIndex)
      if (!diffuseTexture) throw new Error(`Material "${mat.name}" has no diffuse texture`)

      const toonTexture = await loadToonTexture(mat.toonTextureIndex)

      const materialAlpha = mat.diffuse[3]
      const EPSILON = 0.001
      const isTransparent = materialAlpha < 1.0 - EPSILON

      const materialUniformData = new Float32Array(4)
      materialUniformData[0] = materialAlpha
      materialUniformData[1] = 0.0
      materialUniformData[2] = 0.0
      materialUniformData[3] = 0.0

      const materialUniformBuffer = this.device.createBuffer({
        label: `material uniform: ${mat.name}`,
        size: materialUniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(materialUniformBuffer, 0, materialUniformData)

      const bindGroup = this.device.createBindGroup({
        label: `material bind group: ${mat.name}`,
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
          { binding: 1, resource: { buffer: this.lightUniformBuffer } },
          { binding: 2, resource: diffuseTexture.createView() },
          { binding: 3, resource: this.textureSampler },
          { binding: 4, resource: { buffer: this.skinMatrixBuffer! } },
          { binding: 5, resource: toonTexture.createView() },
          { binding: 6, resource: this.textureSampler },
          { binding: 7, resource: { buffer: materialUniformBuffer } },
        ],
      })

      // All materials use the same pipeline
      this.materialDraws.push({
        count: matCount,
        firstIndex: runningFirstIndex,
        bindGroup,
        isTransparent,
      })

      // Outline for all materials (including transparent)
      // Edge flag is at bit 4 (0x10) in PMX format, not bit 0 (0x01)
      if ((mat.edgeFlag & 0x10) !== 0 && mat.edgeSize > 0) {
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

  // Helper: Load texture from file path
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
      this.textureSizes.set(path, { width: imageBitmap.width, height: imageBitmap.height })
      return texture
    } catch {
      return null
    }
  }

  // Step 9: Render one frame
  public render() {
    if (this.multisampleTexture && this.camera && this.device && this.currentModel) {
      const currentTime = performance.now()
      const deltaTime = this.lastFrameTime > 0 ? (currentTime - this.lastFrameTime) / 1000 : 0.016
      this.lastFrameTime = currentTime

      this.updateCameraUniforms()
      this.updateRenderTarget()

      this.updateModelPose(deltaTime)

      const encoder = this.device.createCommandEncoder()
      const pass = encoder.beginRenderPass(this.renderPassDescriptor)

      pass.setVertexBuffer(0, this.vertexBuffer)
      pass.setVertexBuffer(1, this.jointsBuffer)
      pass.setVertexBuffer(2, this.weightsBuffer)
      pass.setIndexBuffer(this.indexBuffer!, "uint32")

      this.drawCallCount = 0
      this.drawOutlines(pass, false)
      this.drawModel(pass, false)
      this.drawModel(pass, true)
      this.drawOutlines(pass, true)

      pass.end()
      this.device.queue.submit([encoder.finish()])
      this.updateStats(performance.now() - currentTime)
    }
  }

  // Update camera uniform buffer each frame
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

  // Update render target texture view
  private updateRenderTarget() {
    const colorAttachment = (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
    if (this.sampleCount > 1) {
      colorAttachment.resolveTarget = this.context.getCurrentTexture().createView()
    } else {
      colorAttachment.view = this.context.getCurrentTexture().createView()
    }
  }

  // Update model pose and physics
  private updateModelPose(deltaTime: number) {
    this.currentModel!.evaluatePose()

    // Upload world matrices to GPU
    const worldMats = this.currentModel!.getBoneWorldMatrices()
    this.device.queue.writeBuffer(
      this.worldMatrixBuffer!,
      0,
      worldMats.buffer,
      worldMats.byteOffset,
      worldMats.byteLength
    )

    if (this.physics) {
      this.physics.step(deltaTime, worldMats, this.currentModel!.getBoneInverseBindMatrices())
      // Re-upload world matrices after physics (physics may have updated bones)
      this.device.queue.writeBuffer(
        this.worldMatrixBuffer!,
        0,
        worldMats.buffer,
        worldMats.byteOffset,
        worldMats.byteLength
      )
    }

    // Compute skin matrices on GPU
    this.computeSkinMatrices()
  }

  // Compute skin matrices on GPU
  private computeSkinMatrices() {
    const boneCount = this.currentModel!.getSkeleton().bones.length
    const workgroupSize = 64
    // Dispatch exactly enough threads for all bones (no bounds check needed)
    const workgroupCount = Math.ceil(boneCount / workgroupSize)

    // Update bone count uniform
    const boneCountData = new Uint32Array(8) // 32 bytes total
    boneCountData[0] = boneCount
    this.device.queue.writeBuffer(this.boneCountBuffer!, 0, boneCountData)

    const bindGroup = this.device.createBindGroup({
      label: "skin matrix compute bind group",
      layout: this.skinMatrixComputePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.boneCountBuffer! } },
        { binding: 1, resource: { buffer: this.worldMatrixBuffer! } },
        { binding: 2, resource: { buffer: this.inverseBindMatrixBuffer! } },
        { binding: 3, resource: { buffer: this.skinMatrixBuffer! } },
      ],
    })

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.skinMatrixComputePipeline!)
    pass.setBindGroup(0, bindGroup)
    pass.dispatchWorkgroups(workgroupCount)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  // Draw outlines (opaque or transparent)
  private drawOutlines(pass: GPURenderPassEncoder, transparent: boolean) {
    if (this.outlineDraws.length === 0) return
    pass.setPipeline(this.outlinePipeline)
    for (const draw of this.outlineDraws) {
      if (draw.count > 0 && draw.isTransparent === transparent) {
        pass.setBindGroup(0, draw.bindGroup)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    }
  }

  // Draw model materials (opaque or transparent)
  private drawModel(pass: GPURenderPassEncoder, transparent: boolean) {
    pass.setPipeline(this.pipeline)
    for (const draw of this.materialDraws) {
      if (draw.count > 0 && draw.isTransparent === transparent) {
        pass.setBindGroup(0, draw.bindGroup)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
        this.drawCallCount++
      }
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

    // Calculate triangles from index buffer
    if (this.indexBuffer) {
      const indexCount = this.currentModel?.getIndices()?.length || 0
      this.stats.triangles = Math.floor(indexCount / 3)
    } else {
      this.stats.triangles = Math.floor(this.vertexCount / 3)
    }

    // Material count
    this.stats.materials = this.materialDraws.length

    // Texture stats
    this.stats.textures = this.textureCache.size
    let textureMemoryBytes = 0
    for (const [path, size] of this.textureSizes.entries()) {
      if (this.textureCache.has(path)) {
        // RGBA8 = 4 bytes per pixel
        textureMemoryBytes += size.width * size.height * 4
      }
    }
    // Add render target textures (multisample + depth)
    if (this.multisampleTexture) {
      const width = this.canvas.width
      const height = this.canvas.height
      textureMemoryBytes += width * height * 4 * this.sampleCount // multisample color
      textureMemoryBytes += width * height * 4 // depth (depth24plus = 4 bytes)
    }
    this.stats.textureMemory = Math.round((textureMemoryBytes / 1024 / 1024) * 100) / 100

    // Buffer memory estimate
    let bufferMemoryBytes = 0
    if (this.vertexBuffer) {
      const vertices = this.currentModel?.getVertices()
      if (vertices) bufferMemoryBytes += vertices.byteLength
    }
    if (this.indexBuffer) {
      const indices = this.currentModel?.getIndices()
      if (indices) bufferMemoryBytes += indices.byteLength
    }
    if (this.jointsBuffer) {
      const skinning = this.currentModel?.getSkinning()
      if (skinning) bufferMemoryBytes += skinning.joints.byteLength
    }
    if (this.weightsBuffer) {
      const skinning = this.currentModel?.getSkinning()
      if (skinning) bufferMemoryBytes += skinning.weights.byteLength
    }
    if (this.skinMatrixBuffer) {
      const skeleton = this.currentModel?.getSkeleton()
      if (skeleton) bufferMemoryBytes += Math.max(256, skeleton.bones.length * 16 * 4)
    }
    bufferMemoryBytes += 40 * 4 // cameraUniformBuffer
    bufferMemoryBytes += 64 * 4 // lightUniformBuffer
    // Material uniform buffers (estimate: 4 bytes per material)
    bufferMemoryBytes += this.materialDraws.length * 4
    this.stats.bufferMemory = Math.round((bufferMemoryBytes / 1024 / 1024) * 100) / 100

    // Total GPU memory estimate
    this.stats.gpuMemory = Math.round((this.stats.textureMemory + this.stats.bufferMemory) * 100) / 100
  }
}
