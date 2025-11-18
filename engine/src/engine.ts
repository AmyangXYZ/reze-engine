import { Camera } from "./camera"
import { Quat, Vec3 } from "./math"
import { Model } from "./model"
import { PmxLoader } from "./pmx-loader"
import { Physics } from "./physics"
import { VMDKeyFrame, VMDLoader } from "./vmd-loader"

export type EngineOptions = {
  ambient?: number
  bloomIntensity?: number
  rimLightIntensity?: number
}

export interface EngineStats {
  fps: number
  frameTime: number // ms
  gpuMemory: number // MB (estimated total GPU memory)
}

// Internal type for organizing bone keyframes during animation playback
type BoneKeyFrame = {
  boneName: string
  time: number
  rotation: Quat
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
  private indexBuffer?: GPUBuffer
  private resizeObserver: ResizeObserver | null = null
  private depthTexture!: GPUTexture
  private pipeline!: GPURenderPipeline
  private outlinePipeline!: GPURenderPipeline
  private hairUnifiedOutlinePipeline!: GPURenderPipeline
  private hairUnifiedPipelineOverEyes!: GPURenderPipeline
  private hairUnifiedPipelineOverNonEyes!: GPURenderPipeline
  private hairDepthPipeline!: GPURenderPipeline
  private eyePipeline!: GPURenderPipeline
  private hairBindGroupLayout!: GPUBindGroupLayout
  private outlineBindGroupLayout!: GPUBindGroupLayout
  private jointsBuffer!: GPUBuffer
  private weightsBuffer!: GPUBuffer
  private skinMatrixBuffer?: GPUBuffer
  private worldMatrixBuffer?: GPUBuffer
  private inverseBindMatrixBuffer?: GPUBuffer
  private skinMatrixComputePipeline?: GPUComputePipeline
  private skinMatrixComputeBindGroup?: GPUBindGroup
  private boneCountBuffer?: GPUBuffer
  private multisampleTexture!: GPUTexture
  private readonly sampleCount = 4 // MSAA 4x
  private renderPassDescriptor!: GPURenderPassDescriptor
  // Ambient light settings
  private ambient: number = 1.0
  // Bloom post-processing textures
  private sceneRenderTexture!: GPUTexture
  private sceneRenderTextureView!: GPUTextureView
  private bloomExtractTexture!: GPUTexture
  private bloomBlurTexture1!: GPUTexture
  private bloomBlurTexture2!: GPUTexture
  // Bloom post-processing pipelines
  private bloomExtractPipeline!: GPURenderPipeline
  private bloomBlurPipeline!: GPURenderPipeline
  private bloomComposePipeline!: GPURenderPipeline
  // Fullscreen quad for post-processing
  private fullscreenQuadBuffer!: GPUBuffer
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
  private bloomThreshold: number = 0.3
  private bloomIntensity: number = 0.12
  // Rim light settings
  private rimLightIntensity: number = 0.45
  private rimLightPower: number = 2.0

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
    gpuMemory: 0,
  }
  private animationFrameId: number | null = null
  private renderLoopCallback: (() => void) | null = null

  private animationFrames: VMDKeyFrame[] = []
  private animationTimeouts: number[] = []

  constructor(canvas: HTMLCanvasElement, options?: EngineOptions) {
    this.canvas = canvas
    if (options) {
      this.ambient = options.ambient ?? 1.0
      this.bloomIntensity = options.bloomIntensity ?? 0.12
      this.rimLightIntensity = options.rimLightIntensity ?? 0.45
    }
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
    this.createFullscreenQuad()
    this.createBloomPipelines()
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
          alphaMultiplier: f32,
          rimIntensity: f32,
          rimPower: f32,
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
          
          // Rim light calculation
          let viewDir = normalize(camera.viewPos - input.worldPos);
          var rimFactor = 1.0 - max(dot(n, viewDir), 0.0);
          rimFactor = pow(rimFactor, material.rimPower);
          let rimLight = material.rimColor * material.rimIntensity * rimFactor;
          
          let color = albedo * lightAccum + rimLight;
          
          var finalAlpha = material.alpha * material.alphaMultiplier;
          if (material.isOverEyes > 0.5) {
            finalAlpha *= 0.5; // Hair over eyes gets 50% alpha
          }
          
          if (finalAlpha < 0.001) {
            discard;
          }
          
          return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), finalAlpha);
        }
      `,
    })

    // Create explicit bind group layout for all pipelines using the main shader
    this.hairBindGroupLayout = this.device.createBindGroupLayout({
      label: "shared material bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // camera
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // light
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // diffuseTexture
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: {} }, // diffuseSampler
        { binding: 4, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }, // skinMats
        { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: {} }, // toonTexture
        { binding: 6, visibility: GPUShaderStage.FRAGMENT, sampler: {} }, // toonSampler
        { binding: 7, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // material
      ],
    })

    const sharedPipelineLayout = this.device.createPipelineLayout({
      label: "shared pipeline layout",
      bindGroupLayouts: [this.hairBindGroupLayout],
    })

    // Single pipeline for all materials with alpha blending
    this.pipeline = this.device.createRenderPipeline({
      label: "model pipeline",
      layout: sharedPipelineLayout,
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
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      multisample: {
        count: this.sampleCount,
      },
    })

    // Create bind group layout for outline pipelines
    this.outlineBindGroupLayout = this.device.createBindGroupLayout({
      label: "outline bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }, // camera
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
          var color = material.edgeColor;
          
          if (material.isOverEyes > 0.5) {
            color.a *= 0.5; // Hair outlines over eyes get 50% alpha
          }
          
          return color;
        }
      `,
    })

    this.outlinePipeline = this.device.createRenderPipeline({
      label: "outline pipeline",
      layout: outlinePipelineLayout,
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
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      multisample: {
        count: this.sampleCount,
      },
    })

    // Unified hair outline pipeline: single pass without stencil testing, uses depth test "less-equal" to draw everywhere hair exists
    this.hairUnifiedOutlinePipeline = this.device.createRenderPipeline({
      label: "unified hair outline pipeline",
      layout: outlinePipelineLayout,
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
        format: "depth24plus-stencil8",
        depthWriteEnabled: false, // Don't write depth - let hair geometry control depth
        depthCompare: "less-equal", // Only draw where hair depth exists (no stencil test needed)
        depthBias: -0.0001, // Small negative bias to bring outline slightly closer for depth test
        depthBiasSlopeScale: 0.0,
        depthBiasClamp: 0.0,
      },
      multisample: {
        count: this.sampleCount,
      },
    })

    // Eye overlay pipeline (renders after opaque, writes stencil)
    this.eyePipeline = this.device.createRenderPipeline({
      label: "eye overlay pipeline",
      layout: sharedPipelineLayout,
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
        format: "depth24plus-stencil8",
        depthWriteEnabled: false, // Don't write depth
        depthCompare: "less", // Respect existing depth
        stencilFront: {
          compare: "always",
          failOp: "keep",
          depthFailOp: "keep",
          passOp: "replace", // Write stencil value 1
        },
        stencilBack: {
          compare: "always",
          failOp: "keep",
          depthFailOp: "keep",
          passOp: "replace",
        },
      },
      multisample: { count: this.sampleCount },
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
          
          // Normalize weights
          let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
          var normalizedWeights: vec4f;
          if (weightSum > 0.0001) {
            normalizedWeights = weights0 / weightSum;
          } else {
            normalizedWeights = vec4f(1.0, 0.0, 0.0, 0.0);
          }
          
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
    this.hairDepthPipeline = this.device.createRenderPipeline({
      label: "hair depth pre-pass",
      layout: sharedPipelineLayout,
      vertex: {
        module: depthOnlyShaderModule,
        buffers: [
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
        ],
      },
      fragment: {
        module: depthOnlyShaderModule,
        entryPoint: "fs",
        targets: [
          {
            format: this.presentationFormat,
            writeMask: 0, // Disable all color writes - we only care about depth
          },
        ],
      },
      primitive: { cullMode: "none" },
      depthStencil: {
        format: "depth24plus-stencil8",
        depthWriteEnabled: true,
        depthCompare: "less",
      },
      multisample: { count: this.sampleCount },
    })

    // Unified hair pipeline for over-eyes (stencil == 1): single pass with dynamic branching
    this.hairUnifiedPipelineOverEyes = this.device.createRenderPipeline({
      label: "unified hair pipeline (over eyes)",
      layout: sharedPipelineLayout,
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
        format: "depth24plus-stencil8",
        depthWriteEnabled: false, // Don't write depth (already written in pre-pass)
        depthCompare: "equal", // Only render where depth matches pre-pass
        stencilFront: {
          compare: "equal", // Only render where stencil == 1 (over eyes)
          failOp: "keep",
          depthFailOp: "keep",
          passOp: "keep",
        },
        stencilBack: {
          compare: "equal",
          failOp: "keep",
          depthFailOp: "keep",
          passOp: "keep",
        },
      },
      multisample: { count: this.sampleCount },
    })

    // Unified pipeline for hair over non-eyes (stencil != 1)
    this.hairUnifiedPipelineOverNonEyes = this.device.createRenderPipeline({
      label: "unified hair pipeline (over non-eyes)",
      layout: sharedPipelineLayout,
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
        format: "depth24plus-stencil8",
        depthWriteEnabled: false, // Don't write depth (already written in pre-pass)
        depthCompare: "equal", // Only render where depth matches pre-pass
        stencilFront: {
          compare: "not-equal", // Only render where stencil != 1 (over non-eyes)
          failOp: "keep",
          depthFailOp: "keep",
          passOp: "keep",
        },
        stencilBack: {
          compare: "not-equal",
          failOp: "keep",
          depthFailOp: "keep",
          passOp: "keep",
        },
      },
      multisample: { count: this.sampleCount },
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

  // Create fullscreen quad for post-processing
  private createFullscreenQuad() {
    // Fullscreen quad vertices: two triangles covering the entire screen - Format: position (x, y), uv (u, v)
    const quadVertices = new Float32Array([
      // Triangle 1
      -1.0,
      -1.0,
      0.0,
      0.0, // bottom-left
      1.0,
      -1.0,
      1.0,
      0.0, // bottom-right
      -1.0,
      1.0,
      0.0,
      1.0, // top-left
      // Triangle 2
      -1.0,
      1.0,
      0.0,
      1.0, // top-left
      1.0,
      -1.0,
      1.0,
      0.0, // bottom-right
      1.0,
      1.0,
      1.0,
      1.0, // top-right
    ])

    this.fullscreenQuadBuffer = this.device.createBuffer({
      label: "fullscreen quad",
      size: quadVertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.fullscreenQuadBuffer, 0, quadVertices)
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

        // 9-tap gaussian blur
        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          let texelSize = 1.0 / vec2f(textureDimensions(inputTexture));
          var result = vec4f(0.0);
          
          // Gaussian weights for 9-tap filter
          let weights = array<f32, 9>(
            0.01621622, 0.05405405, 0.12162162,
            0.19459459, 0.22702703,
            0.19459459, 0.12162162, 0.05405405, 0.01621622
          );
          
          let offsets = array<f32, 9>(-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0);
          
          for (var i = 0u; i < 9u; i++) {
            let offset = offsets[i] * texelSize * blurUniforms.direction;
            result += textureSample(inputTexture, inputSampler, input.uv + offset) * weights[i];
          }
          
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

  // Setup bloom textures and bind groups (called when canvas is resized)
  private setupBloom(width: number, height: number) {
    // Create bloom textures (half resolution for performance)
    const bloomWidth = Math.floor(width / 2)
    const bloomHeight = Math.floor(height / 2)
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
      this.sceneRenderTextureView = this.sceneRenderTexture.createView()

      // Setup bloom textures and bind groups
      this.setupBloom(width, height)

      const depthTextureView = this.depthTexture.createView()

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

    this.setAmbient(this.ambient)
    this.addLight(new Vec3(-0.5, -0.8, 0.5).normalize(), new Vec3(1.0, 0.95, 0.9), 0.02)
    this.addLight(new Vec3(0.7, -0.5, 0.3).normalize(), new Vec3(0.8, 0.85, 1.0), 0.015)
    this.addLight(new Vec3(0.3, -0.5, -1.0).normalize(), new Vec3(0.9, 0.9, 1.0), 0.01)
    this.device.queue.writeBuffer(this.lightUniformBuffer, 0, this.lightData)
  }

  private addLight(direction: Vec3, color: Vec3, intensity: number = 1.0): boolean {
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

  private setAmbient(intensity: number) {
    this.lightData[0] = intensity
  }

  public async loadAnimation(url: string) {
    const frames = await VMDLoader.load(url)
    this.animationFrames = frames
  }

  public playAnimation() {
    if (this.animationFrames.length === 0) return

    this.stopAnimation()

    const allBoneKeyFrames: BoneKeyFrame[] = []
    for (const keyFrame of this.animationFrames) {
      for (const boneFrame of keyFrame.boneFrames) {
        allBoneKeyFrames.push({
          boneName: boneFrame.boneName,
          time: keyFrame.time,
          rotation: boneFrame.rotation,
        })
      }
    }

    const boneKeyFramesByBone = new Map<string, BoneKeyFrame[]>()
    for (const boneKeyFrame of allBoneKeyFrames) {
      if (!boneKeyFramesByBone.has(boneKeyFrame.boneName)) {
        boneKeyFramesByBone.set(boneKeyFrame.boneName, [])
      }
      boneKeyFramesByBone.get(boneKeyFrame.boneName)!.push(boneKeyFrame)
    }

    for (const keyFrames of boneKeyFramesByBone.values()) {
      keyFrames.sort((a, b) => a.time - b.time)
    }

    const time0Rotations: Array<{ boneName: string; rotation: Quat }> = []
    const bonesWithTime0 = new Set<string>()
    for (const [boneName, keyFrames] of boneKeyFramesByBone.entries()) {
      if (keyFrames.length > 0 && keyFrames[0].time === 0) {
        time0Rotations.push({
          boneName: boneName,
          rotation: keyFrames[0].rotation,
        })
        bonesWithTime0.add(boneName)
      }
    }

    if (this.currentModel) {
      if (time0Rotations.length > 0) {
        const boneNames = time0Rotations.map((r) => r.boneName)
        const rotations = time0Rotations.map((r) => r.rotation)
        this.rotateBones(boneNames, rotations, 0)
      }

      const skeleton = this.currentModel.getSkeleton()
      const bonesToReset: string[] = []
      for (const bone of skeleton.bones) {
        if (!bonesWithTime0.has(bone.name)) {
          bonesToReset.push(bone.name)
        }
      }

      if (bonesToReset.length > 0) {
        const identityQuat = new Quat(0, 0, 0, 1)
        const identityQuats = new Array(bonesToReset.length).fill(identityQuat)
        this.rotateBones(bonesToReset, identityQuats, 0)
      }

      this.currentModel.evaluatePose()

      // Reset physics immediately and upload matrices to prevent A-pose flash
      if (this.physics) {
        const worldMats = this.currentModel.getBoneWorldMatrices()
        this.physics.reset(worldMats, this.currentModel.getBoneInverseBindMatrices())

        // Upload matrices immediately so next frame shows correct pose
        this.device.queue.writeBuffer(
          this.worldMatrixBuffer!,
          0,
          worldMats.buffer,
          worldMats.byteOffset,
          worldMats.byteLength
        )
        this.computeSkinMatrices()
      }
    }
    for (const [_, keyFrames] of boneKeyFramesByBone.entries()) {
      for (let i = 0; i < keyFrames.length; i++) {
        const boneKeyFrame = keyFrames[i]
        const previousBoneKeyFrame = i > 0 ? keyFrames[i - 1] : null

        if (boneKeyFrame.time === 0) continue

        let durationMs = 0
        if (i === 0) {
          durationMs = boneKeyFrame.time * 1000
        } else if (previousBoneKeyFrame) {
          durationMs = (boneKeyFrame.time - previousBoneKeyFrame.time) * 1000
        }

        const scheduleTime = i > 0 && previousBoneKeyFrame ? previousBoneKeyFrame.time : 0
        const delayMs = scheduleTime * 1000

        if (delayMs <= 0) {
          this.rotateBones([boneKeyFrame.boneName], [boneKeyFrame.rotation], durationMs)
        } else {
          const timeoutId = window.setTimeout(() => {
            this.rotateBones([boneKeyFrame.boneName], [boneKeyFrame.rotation], durationMs)
          }, delayMs)
          this.animationTimeouts.push(timeoutId)
        }
      }
    }
  }

  public stopAnimation() {
    for (const timeoutId of this.animationTimeouts) {
      clearTimeout(timeoutId)
    }
    this.animationTimeouts = []
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
    // console.log({
    //   vertices: Array.from(model.getVertices()),
    //   indices: Array.from(model.getIndices()),
    //   materials: model.getMaterials(),
    //   textures: model.getTextures(),
    //   bones: model.getSkeleton().bones,
    //   skinning: { joints: Array.from(model.getSkinning().joints), weights: Array.from(model.getSkinning().weights) },
    // })
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

    // Create compute bind group once (reused every frame)
    this.skinMatrixComputeBindGroup = this.device.createBindGroup({
      layout: this.skinMatrixComputePipeline!.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.boneCountBuffer } },
        { binding: 1, resource: { buffer: this.worldMatrixBuffer } },
        { binding: 2, resource: { buffer: this.inverseBindMatrixBuffer } },
        { binding: 3, resource: { buffer: this.skinMatrixBuffer } },
      ],
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
      throw new Error("Model has no index buffer")
    }

    await this.setupMaterials(model)
  }

  private opaqueNonEyeNonHairDraws: {
    count: number
    firstIndex: number
    bindGroup: GPUBindGroup
    isTransparent: boolean
  }[] = []
  private eyeDraws: { count: number; firstIndex: number; bindGroup: GPUBindGroup; isTransparent: boolean }[] = []
  private hairDrawsOverEyes: { count: number; firstIndex: number; bindGroup: GPUBindGroup; isTransparent: boolean }[] =
    []
  private hairDrawsOverNonEyes: {
    count: number
    firstIndex: number
    bindGroup: GPUBindGroup
    isTransparent: boolean
  }[] = []
  private transparentNonEyeNonHairDraws: {
    count: number
    firstIndex: number
    bindGroup: GPUBindGroup
    isTransparent: boolean
  }[] = []
  private opaqueNonEyeNonHairOutlineDraws: {
    count: number
    firstIndex: number
    bindGroup: GPUBindGroup
    isTransparent: boolean
  }[] = []
  private eyeOutlineDraws: { count: number; firstIndex: number; bindGroup: GPUBindGroup; isTransparent: boolean }[] = []
  private hairOutlineDraws: { count: number; firstIndex: number; bindGroup: GPUBindGroup; isTransparent: boolean }[] =
    []
  private transparentNonEyeNonHairOutlineDraws: {
    count: number
    firstIndex: number
    bindGroup: GPUBindGroup
    isTransparent: boolean
  }[] = []

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

      // Default toon texture fallback - cache it
      const defaultToonPath = "__default_toon__"
      const cached = this.textureCache.get(defaultToonPath)
      if (cached) return cached

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
      this.textureCache.set(defaultToonPath, defaultToonTexture)
      this.textureSizes.set(defaultToonPath, { width: 256, height: 2 })
      return defaultToonTexture
    }

    this.opaqueNonEyeNonHairDraws = []
    this.eyeDraws = []
    this.hairDrawsOverEyes = []
    this.hairDrawsOverNonEyes = []
    this.transparentNonEyeNonHairDraws = []
    this.opaqueNonEyeNonHairOutlineDraws = []
    this.eyeOutlineDraws = []
    this.hairOutlineDraws = []
    this.transparentNonEyeNonHairOutlineDraws = []
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

      // Create material uniform data
      const materialUniformData = new Float32Array(8)
      materialUniformData[0] = materialAlpha
      materialUniformData[1] = 1.0 // alphaMultiplier: 1.0 for non-hair materials
      materialUniformData[2] = this.rimLightIntensity
      materialUniformData[3] = this.rimLightPower
      materialUniformData[4] = 1.0 // rimColor.r
      materialUniformData[5] = 1.0 // rimColor.g
      materialUniformData[6] = 1.0 // rimColor.b
      materialUniformData[7] = 0.0

      const materialUniformBuffer = this.device.createBuffer({
        label: `material uniform: ${mat.name}`,
        size: materialUniformData.byteLength,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      })
      this.device.queue.writeBuffer(materialUniformBuffer, 0, materialUniformData)

      // Create bind groups using the shared bind group layout - All pipelines (main, eye, hair multiply, hair opaque) use the same shader and layout
      const bindGroup = this.device.createBindGroup({
        label: `material bind group: ${mat.name}`,
        layout: this.hairBindGroupLayout,
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

      // Classify materials into appropriate draw lists
      if (mat.isEye) {
        this.eyeDraws.push({
          count: matCount,
          firstIndex: runningFirstIndex,
          bindGroup,
          isTransparent,
        })
      } else if (mat.isHair) {
        // Hair materials: create bind groups for unified pipeline with dynamic branching
        const materialUniformDataHair = new Float32Array(8)
        materialUniformDataHair[0] = materialAlpha
        materialUniformDataHair[1] = 1.0 // alphaMultiplier: base value, shader will adjust
        materialUniformDataHair[2] = this.rimLightIntensity
        materialUniformDataHair[3] = this.rimLightPower
        materialUniformDataHair[4] = 1.0 // rimColor.r
        materialUniformDataHair[5] = 1.0 // rimColor.g
        materialUniformDataHair[6] = 1.0 // rimColor.b
        materialUniformDataHair[7] = 0.0

        // Create uniform buffers for both modes
        const materialUniformBufferOverEyes = this.device.createBuffer({
          label: `material uniform (over eyes): ${mat.name}`,
          size: materialUniformDataHair.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        const materialUniformDataOverEyes = new Float32Array(materialUniformDataHair)
        materialUniformDataOverEyes[7] = 1.0
        this.device.queue.writeBuffer(materialUniformBufferOverEyes, 0, materialUniformDataOverEyes)

        const materialUniformBufferOverNonEyes = this.device.createBuffer({
          label: `material uniform (over non-eyes): ${mat.name}`,
          size: materialUniformDataHair.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        const materialUniformDataOverNonEyes = new Float32Array(materialUniformDataHair)
        materialUniformDataOverNonEyes[7] = 0.0
        this.device.queue.writeBuffer(materialUniformBufferOverNonEyes, 0, materialUniformDataOverNonEyes)

        // Create bind groups for both modes
        const bindGroupOverEyes = this.device.createBindGroup({
          label: `material bind group (over eyes): ${mat.name}`,
          layout: this.hairBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
            { binding: 1, resource: { buffer: this.lightUniformBuffer } },
            { binding: 2, resource: diffuseTexture.createView() },
            { binding: 3, resource: this.textureSampler },
            { binding: 4, resource: { buffer: this.skinMatrixBuffer! } },
            { binding: 5, resource: toonTexture.createView() },
            { binding: 6, resource: this.textureSampler },
            { binding: 7, resource: { buffer: materialUniformBufferOverEyes } },
          ],
        })

        const bindGroupOverNonEyes = this.device.createBindGroup({
          label: `material bind group (over non-eyes): ${mat.name}`,
          layout: this.hairBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
            { binding: 1, resource: { buffer: this.lightUniformBuffer } },
            { binding: 2, resource: diffuseTexture.createView() },
            { binding: 3, resource: this.textureSampler },
            { binding: 4, resource: { buffer: this.skinMatrixBuffer! } },
            { binding: 5, resource: toonTexture.createView() },
            { binding: 6, resource: this.textureSampler },
            { binding: 7, resource: { buffer: materialUniformBufferOverNonEyes } },
          ],
        })

        // Store both bind groups for unified pipeline
        this.hairDrawsOverEyes.push({
          count: matCount,
          firstIndex: runningFirstIndex,
          bindGroup: bindGroupOverEyes,
          isTransparent,
        })

        this.hairDrawsOverNonEyes.push({
          count: matCount,
          firstIndex: runningFirstIndex,
          bindGroup: bindGroupOverNonEyes,
          isTransparent,
        })
      } else if (isTransparent) {
        this.transparentNonEyeNonHairDraws.push({
          count: matCount,
          firstIndex: runningFirstIndex,
          bindGroup,
          isTransparent,
        })
      } else {
        this.opaqueNonEyeNonHairDraws.push({
          count: matCount,
          firstIndex: runningFirstIndex,
          bindGroup,
          isTransparent,
        })
      }

      // Outline for all materials (including transparent) - Edge flag is at bit 4 (0x10) in PMX format, not bit 0 (0x01)
      if ((mat.edgeFlag & 0x10) !== 0 && mat.edgeSize > 0) {
        const materialUniformData = new Float32Array(8)
        materialUniformData[0] = mat.edgeColor[0] // edgeColor.r
        materialUniformData[1] = mat.edgeColor[1] // edgeColor.g
        materialUniformData[2] = mat.edgeColor[2] // edgeColor.b
        materialUniformData[3] = mat.edgeColor[3] // edgeColor.a
        materialUniformData[4] = mat.edgeSize
        materialUniformData[5] = 0.0 // isOverEyes: 0.0 for all (unified pipeline doesn't use stencil)
        materialUniformData[6] = 0.0 // _padding1
        materialUniformData[7] = 0.0 // _padding2

        const materialUniformBuffer = this.device.createBuffer({
          label: `outline material uniform: ${mat.name}`,
          size: materialUniformData.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(materialUniformBuffer, 0, materialUniformData)

        const outlineBindGroup = this.device.createBindGroup({
          label: `outline bind group: ${mat.name}`,
          layout: this.outlineBindGroupLayout,
          entries: [
            { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
            { binding: 1, resource: { buffer: materialUniformBuffer } },
            { binding: 2, resource: { buffer: this.skinMatrixBuffer! } },
          ],
        })

        // Classify outlines into appropriate draw lists
        if (mat.isEye) {
          this.eyeOutlineDraws.push({
            count: matCount,
            firstIndex: runningFirstIndex,
            bindGroup: outlineBindGroup,
            isTransparent,
          })
        } else if (mat.isHair) {
          this.hairOutlineDraws.push({
            count: matCount,
            firstIndex: runningFirstIndex,
            bindGroup: outlineBindGroup,
            isTransparent,
          })
        } else if (isTransparent) {
          this.transparentNonEyeNonHairOutlineDraws.push({
            count: matCount,
            firstIndex: runningFirstIndex,
            bindGroup: outlineBindGroup,
            isTransparent,
          })
        } else {
          this.opaqueNonEyeNonHairOutlineDraws.push({
            count: matCount,
            firstIndex: runningFirstIndex,
            bindGroup: outlineBindGroup,
            isTransparent,
          })
        }
      }

      runningFirstIndex += matCount
    }
  }

  // Helper: Load texture from file path with optional max size limit
  private async createTextureFromPath(path: string, maxSize: number = 2048): Promise<GPUTexture | null> {
    const cached = this.textureCache.get(path)
    if (cached) {
      return cached
    }

    try {
      const response = await fetch(path)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      let imageBitmap = await createImageBitmap(await response.blob(), {
        premultiplyAlpha: "none",
        colorSpaceConversion: "none",
      })

      // Downscale if texture is too large
      let finalWidth = imageBitmap.width
      let finalHeight = imageBitmap.height
      if (finalWidth > maxSize || finalHeight > maxSize) {
        const scale = Math.min(maxSize / finalWidth, maxSize / finalHeight)
        finalWidth = Math.floor(finalWidth * scale)
        finalHeight = Math.floor(finalHeight * scale)

        // Create canvas to downscale
        const canvas = new OffscreenCanvas(finalWidth, finalHeight)
        const ctx = canvas.getContext("2d")
        if (ctx) {
          ctx.drawImage(imageBitmap, 0, 0, finalWidth, finalHeight)
          imageBitmap = await createImageBitmap(canvas)
        }
      }

      const texture = this.device.createTexture({
        label: `texture: ${path}`,
        size: [finalWidth, finalHeight],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })
      this.device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture }, [finalWidth, finalHeight])

      this.textureCache.set(path, texture)
      this.textureSizes.set(path, { width: finalWidth, height: finalHeight })
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

      // PASS 1: Opaque non-eye, non-hair
      pass.setPipeline(this.pipeline)
      for (const draw of this.opaqueNonEyeNonHairDraws) {
        if (draw.count > 0) {
          pass.setBindGroup(0, draw.bindGroup)
          pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          this.drawCallCount++
        }
      }

      // PASS 2: Eyes (writes stencil = 1)
      pass.setPipeline(this.eyePipeline)
      pass.setStencilReference(1) // Set stencil reference value to 1
      for (const draw of this.eyeDraws) {
        if (draw.count > 0) {
          pass.setBindGroup(0, draw.bindGroup)
          pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          this.drawCallCount++
        }
      }

      // PASS 3: Hair rendering with depth pre-pass and unified pipeline
      this.drawOutlines(pass, false)

      // 3a: Hair depth pre-pass (eliminates overdraw by rejecting fragments early)
      if (this.hairDrawsOverEyes.length > 0 || this.hairDrawsOverNonEyes.length > 0) {
        pass.setPipeline(this.hairDepthPipeline)
        for (const draw of this.hairDrawsOverEyes) {
          if (draw.count > 0) {
            pass.setBindGroup(0, draw.bindGroup)
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          }
        }
        for (const draw of this.hairDrawsOverNonEyes) {
          if (draw.count > 0) {
            pass.setBindGroup(0, draw.bindGroup)
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          }
        }
      }

      // 3b: Hair shading pass with unified pipeline and dynamic branching
      if (this.hairDrawsOverEyes.length > 0) {
        pass.setPipeline(this.hairUnifiedPipelineOverEyes)
        pass.setStencilReference(1)
        for (const draw of this.hairDrawsOverEyes) {
          if (draw.count > 0) {
            pass.setBindGroup(0, draw.bindGroup)
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
            this.drawCallCount++
          }
        }
      }

      if (this.hairDrawsOverNonEyes.length > 0) {
        pass.setPipeline(this.hairUnifiedPipelineOverNonEyes)
        pass.setStencilReference(1)
        for (const draw of this.hairDrawsOverNonEyes) {
          if (draw.count > 0) {
            pass.setBindGroup(0, draw.bindGroup)
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
            this.drawCallCount++
          }
        }
      }

      // 3c: Hair outlines - unified single pass without stencil testing
      if (this.hairOutlineDraws.length > 0) {
        pass.setPipeline(this.hairUnifiedOutlinePipeline)
        for (const draw of this.hairOutlineDraws) {
          if (draw.count > 0) {
            pass.setBindGroup(0, draw.bindGroup)
            pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          }
        }
      }

      // PASS 4: Transparent non-eye, non-hair
      pass.setPipeline(this.pipeline)
      for (const draw of this.transparentNonEyeNonHairDraws) {
        if (draw.count > 0) {
          pass.setBindGroup(0, draw.bindGroup)
          pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
          this.drawCallCount++
        }
      }

      this.drawOutlines(pass, true)

      pass.end()
      this.device.queue.submit([encoder.finish()])

      // Apply bloom post-processing
      this.applyBloom()

      this.updateStats(performance.now() - currentTime)
    }
  }

  // Apply bloom post-processing
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
    const width = this.canvas.width
    const height = this.canvas.height
    const bloomWidth = Math.floor(width / 2)
    const bloomHeight = Math.floor(height / 2)

    // Pass 1: Extract bright areas (downsample to half resolution)
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

    // Pass 2: Horizontal blur
    const hBlurData = new Float32Array(4) // vec2f + padding = 4 floats
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

    // Pass 3: Vertical blur
    const vBlurData = new Float32Array(4) // vec2f + padding = 4 floats
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

    // Pass 4: Compose scene + bloom to canvas
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
      // Resolve to scene render texture for post-processing
      colorAttachment.resolveTarget = this.sceneRenderTextureView
    } else {
      // Render directly to scene render texture
      colorAttachment.view = this.sceneRenderTextureView
    }
  }

  private updateModelPose(deltaTime: number) {
    this.currentModel!.evaluatePose()
    const worldMats = this.currentModel!.getBoneWorldMatrices()

    if (this.physics) {
      this.physics.step(deltaTime, worldMats, this.currentModel!.getBoneInverseBindMatrices())
    }

    this.device.queue.writeBuffer(
      this.worldMatrixBuffer!,
      0,
      worldMats.buffer,
      worldMats.byteOffset,
      worldMats.byteLength
    )
    this.computeSkinMatrices()
  }

  // Compute skin matrices on GPU
  private computeSkinMatrices() {
    const boneCount = this.currentModel!.getSkeleton().bones.length
    const workgroupSize = 64
    // Dispatch exactly enough threads for all bones (no bounds check needed)
    const workgroupCount = Math.ceil(boneCount / workgroupSize)

    // Bone count is written once in setupModelBuffers() and never changes

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.skinMatrixComputePipeline!)
    pass.setBindGroup(0, this.skinMatrixComputeBindGroup!)
    pass.dispatchWorkgroups(workgroupCount)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  // Draw outlines (opaque or transparent)
  private drawOutlines(pass: GPURenderPassEncoder, transparent: boolean) {
    pass.setPipeline(this.outlinePipeline)
    if (transparent) {
      // Draw transparent outlines (if any)
      for (const draw of this.transparentNonEyeNonHairOutlineDraws) {
        if (draw.count > 0) {
          pass.setBindGroup(0, draw.bindGroup)
          pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
        }
      }
    } else {
      // Draw opaque outlines before main geometry
      for (const draw of this.opaqueNonEyeNonHairOutlineDraws) {
        if (draw.count > 0) {
          pass.setBindGroup(0, draw.bindGroup)
          pass.drawIndexed(draw.count, 1, draw.firstIndex, 0, 0)
        }
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
    }

    // Calculate GPU memory: textures + buffers + render targets
    let textureMemoryBytes = 0
    for (const [path, size] of this.textureSizes.entries()) {
      if (this.textureCache.has(path)) {
        textureMemoryBytes += size.width * size.height * 4 // RGBA8 = 4 bytes per pixel
      }
    }

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
    if (this.worldMatrixBuffer) {
      const skeleton = this.currentModel?.getSkeleton()
      if (skeleton) bufferMemoryBytes += Math.max(256, skeleton.bones.length * 16 * 4)
    }
    if (this.inverseBindMatrixBuffer) {
      const skeleton = this.currentModel?.getSkeleton()
      if (skeleton) bufferMemoryBytes += Math.max(256, skeleton.bones.length * 16 * 4)
    }
    bufferMemoryBytes += 40 * 4 // cameraUniformBuffer
    bufferMemoryBytes += 64 * 4 // lightUniformBuffer
    bufferMemoryBytes += 32 // boneCountBuffer
    bufferMemoryBytes += 32 // blurDirectionBuffer
    bufferMemoryBytes += 32 // bloomIntensityBuffer
    bufferMemoryBytes += 32 // bloomThresholdBuffer
    if (this.fullscreenQuadBuffer) {
      bufferMemoryBytes += 24 * 4 // fullscreenQuadBuffer (6 vertices * 4 floats)
    }

    // Material uniform buffers: Float32Array(8) = 32 bytes each
    const totalMaterialDraws =
      this.opaqueNonEyeNonHairDraws.length +
      this.eyeDraws.length +
      this.hairDrawsOverEyes.length +
      this.hairDrawsOverNonEyes.length +
      this.transparentNonEyeNonHairDraws.length
    bufferMemoryBytes += totalMaterialDraws * 32 // Material uniform buffers (8 floats = 32 bytes)

    // Outline material uniform buffers: Float32Array(8) = 32 bytes each
    const totalOutlineDraws =
      this.opaqueNonEyeNonHairOutlineDraws.length +
      this.eyeOutlineDraws.length +
      this.hairOutlineDraws.length +
      this.transparentNonEyeNonHairOutlineDraws.length
    bufferMemoryBytes += totalOutlineDraws * 32 // Outline material uniform buffers

    let renderTargetMemoryBytes = 0
    if (this.multisampleTexture) {
      const width = this.canvas.width
      const height = this.canvas.height
      renderTargetMemoryBytes += width * height * 4 * this.sampleCount // multisample color
      renderTargetMemoryBytes += width * height * 4 // depth (depth24plus-stencil8 = 4 bytes)
    }
    if (this.sceneRenderTexture) {
      const width = this.canvas.width
      const height = this.canvas.height
      renderTargetMemoryBytes += width * height * 4 // sceneRenderTexture (non-multisampled)
    }
    if (this.bloomExtractTexture) {
      const width = Math.floor(this.canvas.width / 2)
      const height = Math.floor(this.canvas.height / 2)
      renderTargetMemoryBytes += width * height * 4 // bloomExtractTexture
      renderTargetMemoryBytes += width * height * 4 // bloomBlurTexture1
      renderTargetMemoryBytes += width * height * 4 // bloomBlurTexture2
    }

    const totalGPUMemoryBytes = textureMemoryBytes + bufferMemoryBytes + renderTargetMemoryBytes
    this.stats.gpuMemory = Math.round((totalGPUMemoryBytes / 1024 / 1024) * 100) / 100
  }
}
