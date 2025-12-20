import { Camera } from "./camera"
import { Mat4, Quat, Vec3 } from "./math"
import { Model } from "./model"
import { PmxLoader } from "./pmx-loader"
import { Physics } from "./physics"
import { AnimationPose, Player } from "./player"

export type EngineOptions = {
  ambientColor?: Vec3
  bloomIntensity?: number
  rimLightIntensity?: number
  cameraDistance?: number
  cameraTarget?: Vec3
}

export interface EngineStats {
  fps: number
  frameTime: number // ms
}

type DrawCallType =
  | "opaque"
  | "eye"
  | "hair-over-eyes"
  | "hair-over-non-eyes"
  | "transparent"
  | "opaque-outline"
  | "eye-outline"
  | "hair-outline"
  | "transparent-outline"

interface DrawCall {
  type: DrawCallType
  count: number
  firstIndex: number
  bindGroup: GPUBindGroup
}

export class Engine {
  private readonly CAMERA_UNIFORM_STRIDE = 256

  private sbsEnabled = false
  private sbsIpdMeters = 0.064

  private xrSession: XRSession | null = null
  private xrRefSpace: XRReferenceSpace | null = null
  private xrBinding: XRGPUBinding | null = null
  private xrLayer: XRProjectionLayer | null = null
  private xrMsaaTexture: GPUTexture | null = null
  private xrDepthTexture: GPUTexture | null = null

  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  private camera!: Camera
  private cameraUniformBuffer!: GPUBuffer
  private cameraMatrixData = new Float32Array(36)
  private cameraDistance: number = 26.6
  private cameraTarget: Vec3 = new Vec3(0, 12.5, 0)
  private lightUniformBuffer!: GPUBuffer
  private lightData = new Float32Array(4)
  private vertexBuffer!: GPUBuffer
  private indexBuffer?: GPUBuffer
  private resizeObserver: ResizeObserver | null = null
  private depthTexture!: GPUTexture
  // Material rendering pipelines
  private modelPipeline!: GPURenderPipeline
  private eyePipeline!: GPURenderPipeline
  private hairPipelineOverEyes!: GPURenderPipeline
  private hairPipelineOverNonEyes!: GPURenderPipeline
  private hairDepthPipeline!: GPURenderPipeline
  // Outline pipelines
  private outlinePipeline!: GPURenderPipeline
  private hairOutlinePipeline!: GPURenderPipeline
  private mainBindGroupLayout!: GPUBindGroupLayout
  private outlineBindGroupLayout!: GPUBindGroupLayout
  private jointsBuffer!: GPUBuffer
  private weightsBuffer!: GPUBuffer
  private skinMatrixBuffer?: GPUBuffer
  private inverseBindMatrixBuffer?: GPUBuffer
  private multisampleTexture!: GPUTexture
  private readonly sampleCount = 4
  private renderPassDescriptor!: GPURenderPassDescriptor
  // Constants
  private readonly STENCIL_EYE_VALUE = 1
  private readonly BLOOM_DOWNSCALE_FACTOR = 2

  // Default values
  private static readonly DEFAULT_BLOOM_THRESHOLD = 0.01
  private static readonly DEFAULT_BLOOM_INTENSITY = 0.12
  private static readonly DEFAULT_RIM_LIGHT_INTENSITY = 0.45
  private static readonly DEFAULT_CAMERA_DISTANCE = 26.6
  private static readonly DEFAULT_CAMERA_TARGET = new Vec3(0, 12.5, 0)
  private static readonly TRANSPARENCY_EPSILON = 0.001
  private static readonly STATS_FPS_UPDATE_INTERVAL_MS = 1000
  private static readonly STATS_FRAME_TIME_ROUNDING = 100

  // Ambient light settings
  private ambientColor: Vec3 = new Vec3(1.0, 1.0, 1.0)
  // Bloom post-processing textures
  private sceneRenderTexture!: GPUTexture
  private sceneRenderTextureView!: GPUTextureView // Cached view (recreated on resize)
  private bloomExtractTexture!: GPUTexture
  private bloomBlurTexture1!: GPUTexture
  private bloomBlurTexture2!: GPUTexture
  // Post-processing pipelines
  private bloomExtractPipeline!: GPURenderPipeline
  private bloomBlurPipeline!: GPURenderPipeline
  private bloomComposePipeline!: GPURenderPipeline
  private blurDirectionBuffer!: GPUBuffer
  private bloomIntensityBuffer!: GPUBuffer
  private bloomThresholdBuffer!: GPUBuffer
  private linearSampler!: GPUSampler
  // Bloom bind groups (created once, reused every frame)
  private bloomExtractBindGroup?: GPUBindGroup
  private bloomBlurHBindGroup?: GPUBindGroup
  private bloomBlurVBindGroup?: GPUBindGroup
  private bloomComposeBindGroup?: GPUBindGroup
  // Bloom settings
  private bloomThreshold: number = Engine.DEFAULT_BLOOM_THRESHOLD
  private bloomIntensity: number = Engine.DEFAULT_BLOOM_INTENSITY
  // Rim light settings
  private rimLightIntensity: number = Engine.DEFAULT_RIM_LIGHT_INTENSITY

  private currentModel: Model | null = null
  private modelDir: string = ""
  private physics: Physics | null = null
  private materialSampler!: GPUSampler
  private textureCache = new Map<string, GPUTexture>()
  private vertexBufferNeedsUpdate = false
  // Unified draw call list
  private drawCalls: DrawCall[] = []

  private lastFpsUpdate = performance.now()
  private framesSinceLastUpdate = 0
  private lastFrameTime = performance.now()
  private frameTimeSum = 0
  private frameTimeCount = 0
  private stats: EngineStats = {
    fps: 0,
    frameTime: 0,
  }
  private animationFrameId: number | null = null
  private renderLoopCallback: (() => void) | null = null

  private player: Player = new Player()
  private hasAnimation = false // Set to true when loadAnimation is called

  constructor(canvas: HTMLCanvasElement, options?: EngineOptions) {
    this.canvas = canvas
    if (options) {
      this.ambientColor = options.ambientColor ?? new Vec3(1.0, 1.0, 1.0)
      this.bloomIntensity = options.bloomIntensity ?? Engine.DEFAULT_BLOOM_INTENSITY
      this.rimLightIntensity = options.rimLightIntensity ?? Engine.DEFAULT_RIM_LIGHT_INTENSITY
      this.cameraDistance = options.cameraDistance ?? Engine.DEFAULT_CAMERA_DISTANCE
      this.cameraTarget = options.cameraTarget ?? Engine.DEFAULT_CAMERA_TARGET
    }
  }

  // Step 1: Get WebGPU device and context
  public async init(options?: { xrCompatible?: boolean }) {
    const adapter = await navigator.gpu?.requestAdapter(
      options?.xrCompatible ? ({ xrCompatible: true } as unknown as GPURequestAdapterOptions) : undefined
    )
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
    this.createBloomPipelines()
    this.setupResize()
  }

  private createRenderPipeline(config: {
    label: string
    layout: GPUPipelineLayout
    shaderModule: GPUShaderModule
    vertexBuffers: GPUVertexBufferLayout[]
    fragmentTarget?: GPUColorTargetState
    fragmentEntryPoint?: string
    cullMode?: GPUCullMode
    depthStencil?: GPUDepthStencilState
    multisample?: GPUMultisampleState
  }): GPURenderPipeline {
    return this.device.createRenderPipeline({
      label: config.label,
      layout: config.layout,
      vertex: {
        module: config.shaderModule,
        buffers: config.vertexBuffers,
      },
      fragment: config.fragmentTarget
        ? {
            module: config.shaderModule,
            entryPoint: config.fragmentEntryPoint,
            targets: [config.fragmentTarget],
          }
        : undefined,
      primitive: { cullMode: config.cullMode ?? "none" },
      depthStencil: config.depthStencil,
      multisample: config.multisample ?? { count: this.sampleCount },
    })
  }

  private createPipelines() {
    this.materialSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    })

    // Shared vertex buffer layouts
    const fullVertexBuffers: GPUVertexBufferLayout[] = [
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
    ]

    const outlineVertexBuffers: GPUVertexBufferLayout[] = [
      {
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
          { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat },
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
    ]

    const depthOnlyVertexBuffers: GPUVertexBufferLayout[] = [
      {
        arrayStride: 8 * 4,
        attributes: [
          { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
          { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat },
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
    ]

    const standardBlend: GPUColorTargetState = {
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
    }

    const shaderModule = this.device.createShaderModule({
      label: "model shaders",
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
          viewPos: vec3f,
          _padding: f32,
        };

        struct LightUniforms {
          ambientColor: vec3f,
        };

        struct MaterialUniforms {
          alpha: f32,
          alphaMultiplier: f32,
          rimIntensity: f32,
          _padding1: f32,
          rimColor: vec3f,
          isOverEyes: f32, // 1.0 if rendering over eyes, 0.0 otherwise
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
        @group(0) @binding(5) var<uniform> material: MaterialUniforms;

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(2) uv: vec2f,
          @location(3) joints0: vec4<u32>,
          @location(4) weights0: vec4<f32>
        ) -> VertexOutput {
          var output: VertexOutput;
          let pos4 = vec4f(position, 1.0);
          
          // Branchless weight normalization (avoids GPU branch divergence)
          let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
          let invWeightSum = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
          let normalizedWeights = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);
          
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
          // Early alpha test - discard before expensive calculations
          var finalAlpha = material.alpha * material.alphaMultiplier;
          if (material.isOverEyes > 0.5) {
            finalAlpha *= 0.5; // Hair over eyes gets 50% alpha
          }
          if (finalAlpha < 0.001) {
            discard;
          }
          
          let n = normalize(input.normal);
          let albedo = textureSample(diffuseTexture, diffuseSampler, input.uv).rgb;

          let lightAccum = light.ambientColor;
          
          // Rim light calculation
          let viewDir = normalize(camera.viewPos - input.worldPos);
          var rimFactor = 1.0 - max(dot(n, viewDir), 0.0);
          rimFactor = rimFactor * rimFactor; // Optimized: direct multiply instead of pow(x, 2.0)
          let rimLight = material.rimColor * material.rimIntensity * rimFactor;
          
          let color = albedo * lightAccum + rimLight;
          
          return vec4f(color, finalAlpha);
        }
      `,
    })

    // Create explicit bind group layout for all pipelines using the main shader
    this.mainBindGroupLayout = this.device.createBindGroupLayout({
      label: "main material bind group layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 144 },
        }, // camera
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // light
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // diffuseTexture
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} }, // diffuseSampler
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // skinMats
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // material
      ],
    })

    const mainPipelineLayout = this.device.createPipelineLayout({
      label: "main pipeline layout",
      bindGroupLayouts: [this.mainBindGroupLayout],
    })

    this.modelPipeline = this.createRenderPipeline({
      label: "model pipeline",
      layout: mainPipelineLayout,
      shaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTarget: standardBlend,
      cullMode: "none",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    // Create bind group layout for outline pipelines
    this.outlineBindGroupLayout = this.device.createBindGroupLayout({
      label: "outline bind group layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: { type: "uniform", hasDynamicOffset: true, minBindingSize: 144 },
        }, // camera
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // material
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // skinMats
      ],
    })

    const outlinePipelineLayout = this.device.createPipelineLayout({
      label: "outline pipeline layout",
      bindGroupLayouts: [this.outlineBindGroupLayout],
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
          isOverEyes: f32, // 1.0 if rendering over eyes, 0.0 otherwise (for hair outlines)
          _padding1: f32,
          _padding2: f32,
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
          @location(3) joints0: vec4<u32>,
          @location(4) weights0: vec4<f32>
        ) -> VertexOutput {
          var output: VertexOutput;
          let pos4 = vec4f(position, 1.0);
          
          // Branchless weight normalization (avoids GPU branch divergence)
          let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
          let invWeightSum = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
          let normalizedWeights = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);
          
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
          var color = material.edgeColor;
          
          if (material.isOverEyes > 0.5) {
            color.a *= 0.5; // Hair outlines over eyes get 50% alpha
          }
          
          return color;
        }
      `,
    })

    this.outlinePipeline = this.createRenderPipeline({
      label: "outline pipeline",
      layout: outlinePipelineLayout,
      shaderModule: outlineShaderModule,
      vertexBuffers: outlineVertexBuffers,
      fragmentTarget: standardBlend,
      cullMode: "back",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
      },
    })

    // Hair outline pipeline
    this.hairOutlinePipeline = this.createRenderPipeline({
      label: "hair outline pipeline",
      layout: outlinePipelineLayout,
      shaderModule: outlineShaderModule,
      vertexBuffers: outlineVertexBuffers,
      fragmentTarget: standardBlend,
      cullMode: "back",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: false,
        depthCompare: "less-equal",
        depthBias: -0.0001,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
      },
    })

    // Eye overlay pipeline (renders after opaque, writes stencil)
    this.eyePipeline = this.createRenderPipeline({
      label: "eye overlay pipeline",
      layout: mainPipelineLayout,
      shaderModule,
      vertexBuffers: fullVertexBuffers,
      fragmentTarget: standardBlend,
      cullMode: "front",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        depthBias: -0.00005,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
        stencilFront: {
          compare: "always",
          failOp: "keep",
          depthFailOp: "keep",
          passOp: "replace",
        },
        stencilBack: {
          compare: "always",
          failOp: "keep",
          depthFailOp: "keep",
          passOp: "replace",
        },
      },
    })

    // Depth-only shader for hair pre-pass (reduces overdraw by early depth rejection)
    const depthOnlyShaderModule = this.device.createShaderModule({
      label: "depth only shader",
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
          viewPos: vec3f,
          _padding: f32,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(0) @binding(4) var<storage, read> skinMats: array<mat4x4f>;

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) normal: vec3f,
          @location(3) joints0: vec4<u32>,
          @location(4) weights0: vec4<f32>
        ) -> @builtin(position) vec4f {
          let pos4 = vec4f(position, 1.0);
          
          // Branchless weight normalization (avoids GPU branch divergence)
          let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
          let invWeightSum = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
          let normalizedWeights = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);
          
          var skinnedPos = vec4f(0.0, 0.0, 0.0, 0.0);
          for (var i = 0u; i < 4u; i++) {
            let j = joints0[i];
            let w = normalizedWeights[i];
            let m = skinMats[j];
            skinnedPos += (m * pos4) * w;
          }
          let worldPos = skinnedPos.xyz;
          let clipPos = camera.projection * camera.view * vec4f(worldPos, 1.0);
          return clipPos;
        }

        @fragment fn fs() -> @location(0) vec4f {
          return vec4f(0.0, 0.0, 0.0, 0.0); // Transparent - color writes disabled via writeMask
        }
      `,
    })

    // Hair depth pre-pass pipeline: depth-only with color writes disabled to eliminate overdraw
    this.hairDepthPipeline = this.createRenderPipeline({
      label: "hair depth pre-pass",
      layout: mainPipelineLayout,
      shaderModule: depthOnlyShaderModule,
      vertexBuffers: depthOnlyVertexBuffers,
      fragmentTarget: {
        format: this.presentationFormat,
        writeMask: 0,
      },
      fragmentEntryPoint: "fs",
      cullMode: "front",
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        depthBias: 0.0,
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
      },
    })

    // Hair pipelines for rendering over eyes vs non-eyes (only differ in stencil compare mode)
    const createHairPipeline = (isOverEyes: boolean): GPURenderPipeline => {
      return this.createRenderPipeline({
        label: `hair pipeline (${isOverEyes ? "over eyes" : "over non-eyes"})`,
        layout: mainPipelineLayout,
        shaderModule,
        vertexBuffers: fullVertexBuffers,
        fragmentTarget: standardBlend,
        cullMode: "front",
        depthStencil: {
          format: "depth24plus-stencil8",
          depthWriteEnabled: false,
          depthCompare: "less-equal",
          stencilFront: {
            compare: isOverEyes ? "equal" : "not-equal",
            failOp: "keep",
            depthFailOp: "keep",
            passOp: "keep",
          },
          stencilBack: {
            compare: isOverEyes ? "equal" : "not-equal",
            failOp: "keep",
            depthFailOp: "keep",
            passOp: "keep",
          },
        },
      })
    }

    this.hairPipelineOverEyes = createHairPipeline(true)
    this.hairPipelineOverNonEyes = createHairPipeline(false)
  }

  // Create bloom post-processing pipelines
  private createBloomPipelines() {
    // Bloom extraction shader (extracts bright areas)
    const bloomExtractShader = this.device.createShaderModule({
      label: "bloom extract",
      code: /* wgsl */ `
        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) uv: vec2f,
        };

        @vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
          var output: VertexOutput;
          // Generate fullscreen quad from vertex index
          let x = f32((vertexIndex << 1u) & 2u) * 2.0 - 1.0;
          let y = f32(vertexIndex & 2u) * 2.0 - 1.0;
          output.position = vec4f(x, y, 0.0, 1.0);
          output.uv = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
          return output;
        }

        struct BloomExtractUniforms {
          threshold: f32,
          _padding1: f32,
          _padding2: f32,
          _padding3: f32,
          _padding4: f32,
          _padding5: f32,
          _padding6: f32,
          _padding7: f32,
        };

        @group(0) @binding(0) var inputTexture: texture_2d<f32>;
        @group(0) @binding(1) var inputSampler: sampler;
        @group(0) @binding(2) var<uniform> extractUniforms: BloomExtractUniforms;

        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          let color = textureSample(inputTexture, inputSampler, input.uv);
          // Extract bright areas above threshold
          let threshold = extractUniforms.threshold;
          let bloom = max(vec3f(0.0), color.rgb - vec3f(threshold)) / max(0.001, 1.0 - threshold);
          return vec4f(bloom, color.a);
        }
      `,
    })

    // Bloom blur shader (gaussian blur - can be used for both horizontal and vertical)
    const bloomBlurShader = this.device.createShaderModule({
      label: "bloom blur",
      code: /* wgsl */ `
        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) uv: vec2f,
        };

        @vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
          var output: VertexOutput;
          let x = f32((vertexIndex << 1u) & 2u) * 2.0 - 1.0;
          let y = f32(vertexIndex & 2u) * 2.0 - 1.0;
          output.position = vec4f(x, y, 0.0, 1.0);
          output.uv = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
          return output;
        }

        struct BlurUniforms {
          direction: vec2f,
          _padding1: f32,
          _padding2: f32,
          _padding3: f32,
          _padding4: f32,
          _padding5: f32,
          _padding6: f32,
        };

        @group(0) @binding(0) var inputTexture: texture_2d<f32>;
        @group(0) @binding(1) var inputSampler: sampler;
        @group(0) @binding(2) var<uniform> blurUniforms: BlurUniforms;

        // 3-tap gaussian blur using bilinear filtering trick (40% fewer texture fetches!)
        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          let texelSize = 1.0 / vec2f(textureDimensions(inputTexture));
          
          // Bilinear optimization: leverage hardware filtering to sample between pixels
          // Original 5-tap: weights [0.06136, 0.24477, 0.38774, 0.24477, 0.06136] at offsets [-2, -1, 0, 1, 2]
          // Optimized 3-tap: combine adjacent samples using weighted offsets
          let weight0 = 0.38774; // Center sample
          let weight1 = 0.24477 + 0.06136; // Combined outer samples = 0.30613
          let offset1 = (0.24477 * 1.0 + 0.06136 * 2.0) / weight1; // Weighted position = 1.2
          
          var result = textureSample(inputTexture, inputSampler, input.uv) * weight0;
          let offsetVec = offset1 * texelSize * blurUniforms.direction;
          result += textureSample(inputTexture, inputSampler, input.uv + offsetVec) * weight1;
          result += textureSample(inputTexture, inputSampler, input.uv - offsetVec) * weight1;
          
          return result;
        }
      `,
    })

    // Bloom composition shader (combines original scene with bloom)
    const bloomComposeShader = this.device.createShaderModule({
      label: "bloom compose",
      code: /* wgsl */ `
        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) uv: vec2f,
        };

        @vertex fn vs(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
          var output: VertexOutput;
          let x = f32((vertexIndex << 1u) & 2u) * 2.0 - 1.0;
          let y = f32(vertexIndex & 2u) * 2.0 - 1.0;
          output.position = vec4f(x, y, 0.0, 1.0);
          output.uv = vec2f(x * 0.5 + 0.5, 1.0 - (y * 0.5 + 0.5));
          return output;
        }

        struct BloomComposeUniforms {
          intensity: f32,
          _padding1: f32,
          _padding2: f32,
          _padding3: f32,
          _padding4: f32,
          _padding5: f32,
          _padding6: f32,
          _padding7: f32,
        };

        @group(0) @binding(0) var sceneTexture: texture_2d<f32>;
        @group(0) @binding(1) var sceneSampler: sampler;
        @group(0) @binding(2) var bloomTexture: texture_2d<f32>;
        @group(0) @binding(3) var bloomSampler: sampler;
        @group(0) @binding(4) var<uniform> composeUniforms: BloomComposeUniforms;

        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          let scene = textureSample(sceneTexture, sceneSampler, input.uv);
          let bloom = textureSample(bloomTexture, bloomSampler, input.uv);
          // Additive blending with intensity control
          let result = scene.rgb + bloom.rgb * composeUniforms.intensity;
          return vec4f(result, scene.a);
        }
      `,
    })

    // Create uniform buffer for blur direction (minimum 32 bytes for WebGPU)
    const blurDirectionBuffer = this.device.createBuffer({
      label: "blur direction",
      size: 32, // Minimum 32 bytes required for uniform buffers in WebGPU
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Create uniform buffer for bloom intensity (minimum 32 bytes for WebGPU)
    const bloomIntensityBuffer = this.device.createBuffer({
      label: "bloom intensity",
      size: 32, // Minimum 32 bytes required for uniform buffers in WebGPU
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Create uniform buffer for bloom threshold (minimum 32 bytes for WebGPU)
    const bloomThresholdBuffer = this.device.createBuffer({
      label: "bloom threshold",
      size: 32, // Minimum 32 bytes required for uniform buffers in WebGPU
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Set default bloom values
    const intensityData = new Float32Array(8) // f32 + 7 padding floats = 8 floats = 32 bytes
    intensityData[0] = this.bloomIntensity
    this.device.queue.writeBuffer(bloomIntensityBuffer, 0, intensityData)

    const thresholdData = new Float32Array(8) // f32 + 7 padding floats = 8 floats = 32 bytes
    thresholdData[0] = this.bloomThreshold
    this.device.queue.writeBuffer(bloomThresholdBuffer, 0, thresholdData)

    // Create linear sampler for post-processing
    const linearSampler = this.device.createSampler({
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "clamp-to-edge",
      addressModeV: "clamp-to-edge",
    })

    // Bloom extraction pipeline
    this.bloomExtractPipeline = this.device.createRenderPipeline({
      label: "bloom extract",
      layout: "auto",
      vertex: {
        module: bloomExtractShader,
        entryPoint: "vs",
      },
      fragment: {
        module: bloomExtractShader,
        entryPoint: "fs",
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    })

    // Bloom blur pipeline
    this.bloomBlurPipeline = this.device.createRenderPipeline({
      label: "bloom blur",
      layout: "auto",
      vertex: {
        module: bloomBlurShader,
        entryPoint: "vs",
      },
      fragment: {
        module: bloomBlurShader,
        entryPoint: "fs",
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    })

    // Bloom composition pipeline
    this.bloomComposePipeline = this.device.createRenderPipeline({
      label: "bloom compose",
      layout: "auto",
      vertex: {
        module: bloomComposeShader,
        entryPoint: "vs",
      },
      fragment: {
        module: bloomComposeShader,
        entryPoint: "fs",
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { topology: "triangle-list" },
    })

    // Store buffers and sampler for later use
    this.blurDirectionBuffer = blurDirectionBuffer
    this.bloomIntensityBuffer = bloomIntensityBuffer
    this.bloomThresholdBuffer = bloomThresholdBuffer
    this.linearSampler = linearSampler
  }

  private setupBloom(width: number, height: number) {
    const bloomWidth = Math.floor(width / this.BLOOM_DOWNSCALE_FACTOR)
    const bloomHeight = Math.floor(height / this.BLOOM_DOWNSCALE_FACTOR)
    this.bloomExtractTexture = this.device.createTexture({
      label: "bloom extract",
      size: [bloomWidth, bloomHeight],
      format: this.presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.bloomBlurTexture1 = this.device.createTexture({
      label: "bloom blur 1",
      size: [bloomWidth, bloomHeight],
      format: this.presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })
    this.bloomBlurTexture2 = this.device.createTexture({
      label: "bloom blur 2",
      size: [bloomWidth, bloomHeight],
      format: this.presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    })

    // Create bloom bind groups
    this.bloomExtractBindGroup = this.device.createBindGroup({
      layout: this.bloomExtractPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sceneRenderTexture.createView() },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: { buffer: this.bloomThresholdBuffer } },
      ],
    })

    this.bloomBlurHBindGroup = this.device.createBindGroup({
      layout: this.bloomBlurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.bloomExtractTexture.createView() },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: { buffer: this.blurDirectionBuffer } },
      ],
    })

    this.bloomBlurVBindGroup = this.device.createBindGroup({
      layout: this.bloomBlurPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.bloomBlurTexture1.createView() },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: { buffer: this.blurDirectionBuffer } },
      ],
    })

    this.bloomComposeBindGroup = this.device.createBindGroup({
      layout: this.bloomComposePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sceneRenderTexture.createView() },
        { binding: 1, resource: this.linearSampler },
        { binding: 2, resource: this.bloomBlurTexture2.createView() },
        { binding: 3, resource: this.linearSampler },
        { binding: 4, resource: { buffer: this.bloomIntensityBuffer } },
      ],
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
        format: "depth24plus-stencil8",
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
      })

      // Create scene render texture (non-multisampled for post-processing)
      this.sceneRenderTexture = this.device.createTexture({
        label: "scene render texture",
        size: [width, height],
        format: this.presentationFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      })

      // Setup bloom textures and bind groups
      this.setupBloom(width, height)

      const depthTextureView = this.depthTexture.createView()
      // Cache the scene render texture view (only recreate on resize)
      this.sceneRenderTextureView = this.sceneRenderTexture.createView()

      // Render scene to texture instead of directly to canvas
      const colorAttachment: GPURenderPassColorAttachment =
        this.sampleCount > 1
          ? {
              view: this.multisampleTexture.createView(),
              resolveTarget: this.sceneRenderTextureView,
              clearValue: { r: 0, g: 0, b: 0, a: 0 },
              loadOp: "clear",
              storeOp: "store",
            }
          : {
              view: this.sceneRenderTextureView,
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
          stencilClearValue: 0,
          stencilLoadOp: "clear",
          stencilStoreOp: "discard", // Discard stencil after frame to save bandwidth (we only use it during rendering)
        },
      }

      this.camera.aspect = width / height
    }
  }

  // Step 4: Create camera and uniform buffer
  private setupCamera() {
    // Dynamic uniform buffer: 2 views (mono/left + right) with 256-byte alignment.
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: this.CAMERA_UNIFORM_STRIDE * 2,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.camera = new Camera(Math.PI, Math.PI / 2.5, this.cameraDistance, this.cameraTarget)

    this.camera.aspect = this.canvas.width / this.canvas.height
    this.camera.attachControl(this.canvas)
  }

  // Step 5: Create lighting buffers
  private setupLighting() {
    this.lightUniformBuffer = this.device.createBuffer({
      label: "light uniforms",
      size: 4 * 4, // 4 floats: ambientColor vec3f (3) + padding (1)
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.setAmbientColor(this.ambientColor)

    this.device.queue.writeBuffer(this.lightUniformBuffer, 0, this.lightData)
  }

  private setAmbientColor(color: Vec3) {
    // Layout: ambientColor (0-2), padding (3)
    this.lightData[0] = color.x
    this.lightData[1] = color.y
    this.lightData[2] = color.z
    this.lightData[3] = 0.0 // Padding for vec3f alignment
  }

  public async loadAnimation(url: string) {
    await this.player.loadVmd(url)
    this.hasAnimation = true

    // Show first frame (time 0) immediately
    if (this.currentModel) {
      const initialPose = this.player.getPoseAtTime(0)
      this.resetBonesAndPhysics(initialPose, true)
    }
  }

  public playAnimation() {
    if (!this.hasAnimation || !this.currentModel) return

    const wasPaused = this.player.isPausedState
    const wasPlaying = this.player.isPlayingState

    // Only reset pose and physics if starting from beginning (not resuming)
    if (!wasPlaying && !wasPaused) {
      const initialPose = this.player.getPoseAtTime(0)
      this.resetBonesAndPhysics(initialPose, true)
    }

    // Start playback (or resume if paused)
    this.player.play()
  }

  public stopAnimation() {
    this.player.stop()
  }

  public pauseAnimation() {
    this.player.pause()
  }

  public seekAnimation(time: number) {
    if (!this.currentModel || !this.hasAnimation) return

    this.player.seek(time)

    // Immediately apply pose at seeked time (don't reset bones without keyframes)
    const pose = this.player.getPoseAtTime(time)
    this.resetBonesAndPhysics(pose, false)
  }

  public getAnimationProgress() {
    return this.player.getProgress()
  }

  /**
   * Apply animation pose to model
   */
  private applyPose(pose: AnimationPose): void {
    if (!this.currentModel) return

    // Apply bone rotations
    if (pose.boneRotations.size > 0) {
      const boneNames = Array.from(pose.boneRotations.keys())
      const rotations = Array.from(pose.boneRotations.values())
      this.rotateBones(boneNames, rotations, 0)
    }

    // Apply bone translations
    if (pose.boneTranslations.size > 0) {
      const boneNames = Array.from(pose.boneTranslations.keys())
      const translations = Array.from(pose.boneTranslations.values())
      this.moveBones(boneNames, translations, 0)
    }

    // Apply morph weights
    for (const [morphName, weight] of pose.morphWeights.entries()) {
      this.setMorphWeight(morphName, weight, 0)
    }
  }

  /**
   * Reset bones and physics to match a given pose
   * @param pose The pose to apply
   * @param resetBonesWithoutKeyframes If true, reset bones that don't have keyframes in the pose to identity
   */
  private resetBonesAndPhysics(pose: AnimationPose, resetBonesWithoutKeyframes: boolean = false): void {
    if (!this.currentModel) return

    this.applyPose(pose)

    // Reset bones without keyframes if requested (for initial animation setup)
    if (resetBonesWithoutKeyframes) {
      const skeleton = this.currentModel.getSkeleton()
      const bonesWithPose = new Set(pose.boneRotations.keys())
      const bonesToReset: string[] = []
      for (const bone of skeleton.bones) {
        if (!bonesWithPose.has(bone.name)) {
          bonesToReset.push(bone.name)
        }
      }

      if (bonesToReset.length > 0) {
        const identityQuat = new Quat(0, 0, 0, 1)
        const identityQuats = new Array(bonesToReset.length).fill(identityQuat)
        this.rotateBones(bonesToReset, identityQuats, 0)
      }
    }

    // Update model pose and physics
    this.currentModel.evaluatePose()

    if (this.physics) {
      const worldMats = this.currentModel.getBoneWorldMatrices()
      this.physics.reset(worldMats, this.currentModel.getBoneInverseBindMatrices())

      // Compute and upload skin matrices immediately
      this.computeSkinMatrices()
    }
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
    this.stopAnimation()
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

  // moveBones now takes relative translations (VMD-style) by default
  public moveBones(bones: string[], relativeTranslations: Vec3[], durationMs?: number) {
    this.currentModel?.moveBones(bones, relativeTranslations, durationMs)
  }

  public setMorphWeight(name: string, weight: number, durationMs?: number): void {
    if (!this.currentModel) return
    this.currentModel.setMorphWeight(name, weight, durationMs)
    if (!durationMs || durationMs === 0) {
      this.vertexBufferNeedsUpdate = true
    }
  }

  private updateVertexBuffer(): void {
    if (!this.currentModel || !this.vertexBuffer) return
    const vertices = this.currentModel.getVertices()
    if (!vertices || vertices.length === 0) return
    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices)
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
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
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

    this.drawCalls = []
    let currentIndexOffset = 0

    for (const mat of materials) {
      const indexCount = mat.vertexCount
      if (indexCount === 0) continue

      const diffuseTexture = await loadTextureByIndex(mat.diffuseTextureIndex)
      if (!diffuseTexture) throw new Error(`Material "${mat.name}" has no diffuse texture`)

      const materialAlpha = mat.diffuse[3]
      const isTransparent = materialAlpha < 1.0 - Engine.TRANSPARENCY_EPSILON

      const materialUniformBuffer = this.createMaterialUniformBuffer(mat.name, materialAlpha, 0.0)

      // Create bind groups using the shared bind group layout - All pipelines (main, eye, hair multiply, hair opaque) use the same shader and layout
      const bindGroup = this.device.createBindGroup({
        label: `material bind group: ${mat.name}`,
        layout: this.mainBindGroupLayout,
        entries: [
          { binding: 0, resource: { buffer: this.cameraUniformBuffer, size: 144 } },
          { binding: 1, resource: { buffer: this.lightUniformBuffer } },
          { binding: 2, resource: diffuseTexture.createView() },
          { binding: 3, resource: this.materialSampler },
          { binding: 4, resource: { buffer: this.skinMatrixBuffer! } },
          { binding: 5, resource: { buffer: materialUniformBuffer } },
        ],
      })

      if (indexCount > 0) {
        if (mat.isEye) {
          this.drawCalls.push({ type: "eye", count: indexCount, firstIndex: currentIndexOffset, bindGroup })
        } else if (mat.isHair) {
          // Hair materials: create separate bind groups for over-eyes vs over-non-eyes
          const createHairBindGroup = (isOverEyes: boolean) => {
            const buffer = this.createMaterialUniformBuffer(
              `${mat.name} (${isOverEyes ? "over eyes" : "over non-eyes"})`,
              materialAlpha,
              isOverEyes ? 1.0 : 0.0
            )

            return this.device.createBindGroup({
              label: `material bind group (${isOverEyes ? "over eyes" : "over non-eyes"}): ${mat.name}`,
              layout: this.mainBindGroupLayout,
              entries: [
                { binding: 0, resource: { buffer: this.cameraUniformBuffer, size: 144 } },
                { binding: 1, resource: { buffer: this.lightUniformBuffer } },
                { binding: 2, resource: diffuseTexture.createView() },
                { binding: 3, resource: this.materialSampler },
                { binding: 4, resource: { buffer: this.skinMatrixBuffer! } },
                { binding: 5, resource: { buffer: buffer } },
              ],
            })
          }

          const bindGroupOverEyes = createHairBindGroup(true)
          const bindGroupOverNonEyes = createHairBindGroup(false)

          this.drawCalls.push({
            type: "hair-over-eyes",
            count: indexCount,
            firstIndex: currentIndexOffset,
            bindGroup: bindGroupOverEyes,
          })
          this.drawCalls.push({
            type: "hair-over-non-eyes",
            count: indexCount,
            firstIndex: currentIndexOffset,
            bindGroup: bindGroupOverNonEyes,
          })
        } else if (isTransparent) {
          this.drawCalls.push({ type: "transparent", count: indexCount, firstIndex: currentIndexOffset, bindGroup })
        } else {
          this.drawCalls.push({ type: "opaque", count: indexCount, firstIndex: currentIndexOffset, bindGroup })
        }
      }

      // Edge flag is at bit 4 (0x10) in PMX format
      if ((mat.edgeFlag & 0x10) !== 0 && mat.edgeSize > 0) {
        const materialUniformData = new Float32Array([
          mat.edgeColor[0],
          mat.edgeColor[1],
          mat.edgeColor[2],
          mat.edgeColor[3],
          mat.edgeSize,
          0,
          0,
          0,
        ])
        const materialUniformBuffer = this.createUniformBuffer(
          `outline material uniform: ${mat.name}`,
          materialUniformData
        )

        const outlineBindGroup = this.device.createBindGroup({
          label: `outline bind group: ${mat.name}`,
          layout: this.outlineBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.cameraUniformBuffer, size: 144 } },
            { binding: 1, resource: { buffer: materialUniformBuffer } },
            { binding: 2, resource: { buffer: this.skinMatrixBuffer! } },
          ],
        })

        if (indexCount > 0) {
          const outlineType: DrawCallType = mat.isEye
            ? "eye-outline"
            : mat.isHair
            ? "hair-outline"
            : isTransparent
            ? "transparent-outline"
            : "opaque-outline"
          this.drawCalls.push({
            type: outlineType,
            count: indexCount,
            firstIndex: currentIndexOffset,
            bindGroup: outlineBindGroup,
          })
        }
      }

      currentIndexOffset += indexCount
    }
  }

  private createMaterialUniformBuffer(label: string, alpha: number, isOverEyes: number): GPUBuffer {
    const data = new Float32Array(8)
    data.set([alpha, 1.0, this.rimLightIntensity, 0.0, 1.0, 1.0, 1.0, isOverEyes])
    return this.createUniformBuffer(`material uniform: ${label}`, data)
  }

  private createUniformBuffer(label: string, data: Float32Array | Uint32Array): GPUBuffer {
    const buffer = this.device.createBuffer({
      label,
      size: data.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(buffer, 0, data as ArrayBufferView<ArrayBuffer>)
    return buffer
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

  // Helper: Render eyes with stencil writing (for post-alpha-eye effect)
  private renderEyes(pass: GPURenderPassEncoder, dynamicOffsets: number[]) {
    pass.setPipeline(this.eyePipeline)
    pass.setStencilReference(this.STENCIL_EYE_VALUE)
    for (const draw of this.drawCalls) {
      if (draw.type === "eye") {
        pass.setBindGroup(0, draw.bindGroup, dynamicOffsets)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    }
  }

  // Helper: Render hair with post-alpha-eye effect (depth pre-pass + stencil-based shading + outlines)
  private renderHair(pass: GPURenderPassEncoder, dynamicOffsets: number[]) {
    // Hair depth pre-pass (reduces overdraw via early depth rejection)
    const hasHair = this.drawCalls.some((d) => d.type === "hair-over-eyes" || d.type === "hair-over-non-eyes")
    if (hasHair) {
      pass.setPipeline(this.hairDepthPipeline)
      for (const draw of this.drawCalls) {
        if (draw.type === "hair-over-eyes" || draw.type === "hair-over-non-eyes") {
          pass.setBindGroup(0, draw.bindGroup, dynamicOffsets)
          pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
        }
      }
    }

    // Hair shading (split by stencil for transparency over eyes)
    const hairOverEyes = this.drawCalls.filter((d) => d.type === "hair-over-eyes")
    if (hairOverEyes.length > 0) {
      pass.setPipeline(this.hairPipelineOverEyes)
      pass.setStencilReference(this.STENCIL_EYE_VALUE)
      for (const draw of hairOverEyes) {
        pass.setBindGroup(0, draw.bindGroup, dynamicOffsets)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    }

    const hairOverNonEyes = this.drawCalls.filter((d) => d.type === "hair-over-non-eyes")
    if (hairOverNonEyes.length > 0) {
      pass.setPipeline(this.hairPipelineOverNonEyes)
      pass.setStencilReference(this.STENCIL_EYE_VALUE)
      for (const draw of hairOverNonEyes) {
        pass.setBindGroup(0, draw.bindGroup, dynamicOffsets)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    }

    // Hair outlines
    const hairOutlines = this.drawCalls.filter((d) => d.type === "hair-outline")
    if (hairOutlines.length > 0) {
      pass.setPipeline(this.hairOutlinePipeline)
      for (const draw of hairOutlines) {
        pass.setBindGroup(0, draw.bindGroup, dynamicOffsets)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    }
  }

  // Render strategy: 1) Opaque non-eye/hair 2) Eyes (stencil=1) 3) Hair (depth pre-pass + split by stencil) 4) Transparent 5) Bloom
  public render() {
    if (!this.multisampleTexture || !this.camera || !this.device) {
      return
    }

    if (this.sbsEnabled) {
      this.renderSbs()
    } else {
      this.renderStandard()
    }
  }

  public setSbsEnabled(enabled: boolean, options?: { ipdMeters?: number }) {
    this.sbsEnabled = enabled
    if (options?.ipdMeters !== undefined) {
      this.sbsIpdMeters = options.ipdMeters
    }
  }

  private prepareFrame(currentTime: number): number {
    const deltaTime = this.lastFrameTime > 0 ? (currentTime - this.lastFrameTime) / 1000 : 0.016
    this.lastFrameTime = currentTime

    // Animate VMD animation if playing
    if (this.hasAnimation && this.currentModel) {
      const pose = this.player.update(currentTime)
      if (pose) {
        this.applyPose(pose)
      }
    }

    // Update model pose first (this may update morph weights via tweens)
    // We need to do this before creating the encoder to ensure vertex buffer is ready
    if (this.currentModel) {
      const hasActiveMorphTweens = this.currentModel.evaluatePose()
      if (hasActiveMorphTweens) {
        this.vertexBufferNeedsUpdate = true
      }
    }

    // Update vertex buffer if morphs changed
    if (this.vertexBufferNeedsUpdate) {
      this.updateVertexBuffer()
      this.vertexBufferNeedsUpdate = false
    }

    // Update model pose (computes skin matrices on CPU)
    if (this.currentModel) {
      this.updateModelPose(deltaTime)
    }

    return deltaTime
  }

  private renderScene(pass: GPURenderPassEncoder, cameraOffset: number) {
    if (!this.currentModel) {
      return
    }

    const dynamicOffsets = [cameraOffset]

    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.setVertexBuffer(1, this.jointsBuffer)
    pass.setVertexBuffer(2, this.weightsBuffer)
    pass.setIndexBuffer(this.indexBuffer!, "uint32")

    // Pass 1: Opaque
    pass.setPipeline(this.modelPipeline)
    for (const draw of this.drawCalls) {
      if (draw.type === "opaque") {
        pass.setBindGroup(0, draw.bindGroup, dynamicOffsets)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    }

    // Pass 2: Eyes (writes stencil value for hair to test against)
    this.renderEyes(pass, dynamicOffsets)

    this.drawOutlines(pass, dynamicOffsets, false)

    // Pass 3: Hair rendering (depth pre-pass + shading + outlines)
    this.renderHair(pass, dynamicOffsets)

    // Pass 4: Transparent
    pass.setPipeline(this.modelPipeline)
    for (const draw of this.drawCalls) {
      if (draw.type === "transparent") {
        pass.setBindGroup(0, draw.bindGroup, dynamicOffsets)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    }

    this.drawOutlines(pass, dynamicOffsets, true)
  }

  private renderStandard() {
    const frameStart = performance.now()

    this.prepareFrame(frameStart)
    this.updateCameraUniforms()
    this.updateRenderTarget()

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    this.renderScene(pass, 0)
    pass.end()

    this.device.queue.submit([encoder.finish()])

    this.applyBloom()

    this.updateStats(performance.now() - frameStart)
  }

  private updateSbsCameraUniforms() {
    const up = new Vec3(0, 1, 0)

    const baseEye = this.camera.getPosition()
    const baseTarget = this.camera.target

    const forward = baseTarget.subtract(baseEye).normalize()
    const right = up.cross(forward).normalize()

    const halfIpd = this.sbsIpdMeters * 0.5
    const eyeOffset = right.scale(halfIpd)

    const leftEye = baseEye.subtract(eyeOffset)
    const rightEye = baseEye.add(eyeOffset)

    // Parallel cameras: shift both eye and target by the same offset.
    const leftTarget = baseTarget.subtract(eyeOffset)
    const rightTarget = baseTarget.add(eyeOffset)

    const projection = Mat4.perspective(this.camera.fov, this.camera.aspect / 2, this.camera.near, this.camera.far)

    this.writeCameraUniform(0, Mat4.lookAt(leftEye, leftTarget, up), projection, leftEye)
    this.writeCameraUniform(
      this.CAMERA_UNIFORM_STRIDE,
      Mat4.lookAt(rightEye, rightTarget, up),
      projection,
      rightEye
    )
  }

  private renderSbs() {
    const frameStart = performance.now()

    this.prepareFrame(frameStart)
    this.updateSbsCameraUniforms()

    const currentTextureView = this.context.getCurrentTexture().createView()

    const colorAttachment: GPURenderPassColorAttachment =
      this.sampleCount > 1
        ? {
            view: this.multisampleTexture.createView(),
            resolveTarget: currentTextureView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          }
        : {
            view: currentTextureView,
            clearValue: { r: 0, g: 0, b: 0, a: 0 },
            loadOp: "clear",
            storeOp: "store",
          }

    const passDescriptor: GPURenderPassDescriptor = {
      label: "renderPass (sbs)",
      colorAttachments: [colorAttachment],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
        stencilClearValue: 0,
        stencilLoadOp: "clear",
        stencilStoreOp: "discard",
      },
    }

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(passDescriptor)

    const halfWidth = Math.floor(this.canvas.width / 2)
    const height = this.canvas.height

    // Left eye
    pass.setViewport(0, 0, halfWidth, height, 0, 1)
    pass.setScissorRect(0, 0, halfWidth, height)
    this.renderScene(pass, 0)

    // Right eye
    pass.setViewport(halfWidth, 0, halfWidth, height, 0, 1)
    pass.setScissorRect(halfWidth, 0, halfWidth, height)
    this.renderScene(pass, this.CAMERA_UNIFORM_STRIDE)

    pass.end()
    this.device.queue.submit([encoder.finish()])

    this.updateStats(performance.now() - frameStart)
  }

  public async startWebXR(session: XRSession, options?: { referenceSpaceType?: XRReferenceSpaceType }) {
    if (!globalThis.XRGPUBinding) {
      throw new Error("WebXR WebGPU binding (XRGPUBinding) is not available in this browser.")
    }

    this.xrSession = session
    this.xrBinding = new globalThis.XRGPUBinding(session, this.device)

    const layer = this.xrBinding.createProjectionLayer({
      colorFormat: this.presentationFormat,
      depthFormat: "depth24plus-stencil8",
      textureType: "texture-array",
    } as unknown)
    this.xrLayer = layer

    ;(session as unknown as { updateRenderState: (state: unknown) => void }).updateRenderState({ layers: [layer] })

    const referenceSpaceType = options?.referenceSpaceType ?? "local-floor"
    this.xrRefSpace = await session
      .requestReferenceSpace(referenceSpaceType)
      .catch(async () => session.requestReferenceSpace("local"))

    this.xrMsaaTexture = this.device.createTexture({
      label: "xr multisample render target",
      size: [layer.textureWidth, layer.textureHeight],
      sampleCount: this.sampleCount,
      format: this.presentationFormat,
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })

    this.xrDepthTexture = this.device.createTexture({
      label: "xr depth texture",
      size: [layer.textureWidth, layer.textureHeight],
      sampleCount: this.sampleCount,
      format: "depth24plus-stencil8",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })

    this.camera.detachControl()
  }

  public stopWebXR() {
    this.xrSession = null
    this.xrRefSpace = null
    this.xrBinding = null
    this.xrLayer = null
    this.xrMsaaTexture = null
    this.xrDepthTexture = null

    this.camera.attachControl(this.canvas)
  }

  private convertXrMat4RhsToLhs(matrix: Float32Array): Mat4 {
    const out = new Float32Array(16)
    const s = [1, 1, -1, 1]
    for (let c = 0; c < 4; c++) {
      for (let r = 0; r < 4; r++) {
        out[c * 4 + r] = matrix[c * 4 + r] * s[r] * s[c]
      }
    }
    return new Mat4(out)
  }

  public renderWebXRFrame(frame: XRFrame) {
    if (!this.xrSession || !this.xrRefSpace || !this.xrBinding || !this.xrLayer || !this.xrMsaaTexture || !this.xrDepthTexture) {
      return
    }

    const frameStart = performance.now()
    this.prepareFrame(frameStart)

    const pose = frame.getViewerPose(this.xrRefSpace)
    if (!pose) {
      return
    }

    const encoder = this.device.createCommandEncoder()

    for (let viewIndex = 0; viewIndex < pose.views.length; viewIndex++) {
      const view = pose.views[viewIndex]
      if (viewIndex > 1) {
        break
      }

      const subImage = this.xrBinding.getViewSubImage(this.xrLayer, view)

      const cameraOffset = viewIndex * this.CAMERA_UNIFORM_STRIDE

      const viewMatrix = this.convertXrMat4RhsToLhs(view.transform.inverse.matrix as Float32Array)
      const projectionMatrix = this.convertXrMat4RhsToLhs(view.projectionMatrix as Float32Array)
      const p = view.transform.position
      const cameraPos = new Vec3(p.x, p.y, -p.z)

      this.writeCameraUniform(cameraOffset, viewMatrix, projectionMatrix, cameraPos)

      const baseArrayLayer = subImage.imageIndex ?? 0
      const colorView = subImage.colorTexture.createView({ baseArrayLayer, arrayLayerCount: 1 })

      const colorAttachment: GPURenderPassColorAttachment = {
        view: this.xrMsaaTexture.createView(),
        resolveTarget: colorView,
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: "clear",
        storeOp: "store",
      }

      const pass = encoder.beginRenderPass({
        label: `renderPass (xr) view ${viewIndex}`,
        colorAttachments: [colorAttachment],
        depthStencilAttachment: {
          view: this.xrDepthTexture.createView(),
          depthClearValue: 1.0,
          depthLoadOp: "clear",
          depthStoreOp: "store",
          stencilClearValue: 0,
          stencilLoadOp: "clear",
          stencilStoreOp: "discard",
        },
      })

      const { x, y, width, height } = subImage.viewport
      pass.setViewport(x, y, width, height, 0, 1)
      pass.setScissorRect(Math.floor(x), Math.floor(y), Math.floor(width), Math.floor(height))

      this.renderScene(pass, cameraOffset)
      pass.end()
    }

    this.device.queue.submit([encoder.finish()])

    this.updateStats(performance.now() - frameStart)
  }

  private applyBloom() {
    if (!this.sceneRenderTexture || !this.bloomExtractTexture) {
      return
    }

    // Update bloom parameters
    const thresholdData = new Float32Array(8)
    thresholdData[0] = this.bloomThreshold
    this.device.queue.writeBuffer(this.bloomThresholdBuffer, 0, thresholdData)

    const intensityData = new Float32Array(8)
    intensityData[0] = this.bloomIntensity
    this.device.queue.writeBuffer(this.bloomIntensityBuffer, 0, intensityData)

    const encoder = this.device.createCommandEncoder()

    // Extract bright areas
    const extractPass = encoder.beginRenderPass({
      label: "bloom extract",
      colorAttachments: [
        {
          view: this.bloomExtractTexture.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    })

    extractPass.setPipeline(this.bloomExtractPipeline)
    extractPass.setBindGroup(0, this.bloomExtractBindGroup!)
    extractPass.draw(6, 1, 0, 0)
    extractPass.end()

    // Horizontal blur
    const hBlurData = new Float32Array(4)
    hBlurData[0] = 1.0
    hBlurData[1] = 0.0
    this.device.queue.writeBuffer(this.blurDirectionBuffer, 0, hBlurData)
    const blurHPass = encoder.beginRenderPass({
      label: "bloom blur horizontal",
      colorAttachments: [
        {
          view: this.bloomBlurTexture1.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    })

    blurHPass.setPipeline(this.bloomBlurPipeline)
    blurHPass.setBindGroup(0, this.bloomBlurHBindGroup!)
    blurHPass.draw(6, 1, 0, 0)
    blurHPass.end()

    // Vertical blur
    const vBlurData = new Float32Array(4)
    vBlurData[0] = 0.0
    vBlurData[1] = 1.0
    this.device.queue.writeBuffer(this.blurDirectionBuffer, 0, vBlurData)
    const blurVPass = encoder.beginRenderPass({
      label: "bloom blur vertical",
      colorAttachments: [
        {
          view: this.bloomBlurTexture2.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    })

    blurVPass.setPipeline(this.bloomBlurPipeline)
    blurVPass.setBindGroup(0, this.bloomBlurVBindGroup!)
    blurVPass.draw(6, 1, 0, 0)
    blurVPass.end()

    // Compose to canvas
    const composePass = encoder.beginRenderPass({
      label: "bloom compose",
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    })

    composePass.setPipeline(this.bloomComposePipeline)
    composePass.setBindGroup(0, this.bloomComposeBindGroup!)
    composePass.draw(6, 1, 0, 0)
    composePass.end()

    this.device.queue.submit([encoder.finish()])
  }

  private writeCameraUniform(offset: number, viewMatrix: Mat4, projectionMatrix: Mat4, cameraPos: Vec3) {
    this.cameraMatrixData.set(viewMatrix.values, 0)
    this.cameraMatrixData.set(projectionMatrix.values, 16)
    this.cameraMatrixData[32] = cameraPos.x
    this.cameraMatrixData[33] = cameraPos.y
    this.cameraMatrixData[34] = cameraPos.z
    this.device.queue.writeBuffer(this.cameraUniformBuffer, offset, this.cameraMatrixData)
  }

  private updateCameraUniforms() {
    this.writeCameraUniform(0, this.camera.getViewMatrix(), this.camera.getProjectionMatrix(), this.camera.getPosition())
  }

  private updateRenderTarget() {
    // Use cached view (only recreated on resize in handleResize)
    const colorAttachment = (this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0]
    if (this.sampleCount > 1) {
      colorAttachment.resolveTarget = this.sceneRenderTextureView
    } else {
      colorAttachment.view = this.sceneRenderTextureView
    }
  }

  private updateModelPose(deltaTime: number) {
    // Note: evaluatePose is called earlier in render() to update vertex buffer before encoder creation
    // Here we just get the matrices and update physics/compute
    const worldMats = this.currentModel!.getBoneWorldMatrices()

    if (this.physics) {
      this.physics.step(deltaTime, worldMats, this.currentModel!.getBoneInverseBindMatrices())
    }

    this.computeSkinMatrices()
  }

  private computeSkinMatrices() {
    const skinMatrices = this.currentModel!.getSkinMatrices()
    this.device.queue.writeBuffer(
      this.skinMatrixBuffer!,
      0,
      skinMatrices.buffer,
      skinMatrices.byteOffset,
      skinMatrices.byteLength
    )
  }

  private drawOutlines(pass: GPURenderPassEncoder, dynamicOffsets: number[], transparent: boolean) {
    pass.setPipeline(this.outlinePipeline)
    const outlineType: DrawCallType = transparent ? "transparent-outline" : "opaque-outline"
    for (const draw of this.drawCalls) {
      if (draw.type === outlineType) {
        pass.setBindGroup(0, draw.bindGroup, dynamicOffsets)
        pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
      }
    }
  }

  private updateStats(frameTime: number) {
    // Simplified frame time tracking - rolling average with fixed window
    const maxSamples = 60
    this.frameTimeSum += frameTime
    this.frameTimeCount++
    if (this.frameTimeCount > maxSamples) {
      // Maintain rolling window by subtracting oldest sample estimate
      const avg = this.frameTimeSum / maxSamples
      this.frameTimeSum -= avg
      this.frameTimeCount = maxSamples
    }
    this.stats.frameTime =
      Math.round((this.frameTimeSum / this.frameTimeCount) * Engine.STATS_FRAME_TIME_ROUNDING) /
      Engine.STATS_FRAME_TIME_ROUNDING

    // FPS tracking
    const now = performance.now()
    this.framesSinceLastUpdate++
    const elapsed = now - this.lastFpsUpdate

    if (elapsed >= Engine.STATS_FPS_UPDATE_INTERVAL_MS) {
      this.stats.fps = Math.round((this.framesSinceLastUpdate / elapsed) * Engine.STATS_FPS_UPDATE_INTERVAL_MS)
      this.framesSinceLastUpdate = 0
      this.lastFpsUpdate = now
    }
  }
}
