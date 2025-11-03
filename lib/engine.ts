import { Camera } from "./camera"
import { Vec3, Mat4 } from "./math"
import { Model } from "./model"
import { PmxLoader } from "./pmx-loader"
import { Physics, Rigidbody, RigidbodyType, RigidbodyShape } from "./physics"

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
  // Rigidbody visualization
  private rigidbodyPipeline?: GPURenderPipeline
  private rigidbodyVertexBuffer?: GPUBuffer
  private rigidbodyIndexBuffer?: GPUBuffer
  private rigidbodyVertexCount: number = 0
  private rigidbodyIndexCount: number = 0
  private rigidbodyBindGroup?: GPUBindGroup
  private physics: Physics | null = null
  private showRigidbodies: boolean = false
  private rigidbodyMeshes: Array<{
    baseVertices: Float32Array // Local space vertices (pos3 + color3 + normal3 = 9 floats per vertex)
    indices: Uint32Array
    vertexOffset: number // Starting vertex index in combined buffer
    indexOffset: number // Starting index in combined buffer
    color: [number, number, number] // RGB color for this rigidbody
  }> = []
  private rigidbodyMeshIndexMap: number[] = [] // Map from mesh index to definition index

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
    this.initRigidbodyPipeline()

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

  private initRigidbodyPipeline() {
    // Pipeline for solid/surfaced rigidbody visualization
    const shader = this.device.createShaderModule({
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
          viewPos: vec3f,
          _padding: f32,
        };

        struct VSOut { 
          @builtin(position) pos: vec4f, 
          @location(0) color: vec3f,
        };
        @group(0) @binding(0) var<uniform> camera: CameraUniforms;

        @vertex fn vs(@location(0) position: vec3f, @location(1) color: vec3f, @location(2) normal: vec3f) -> VSOut {
          var o: VSOut;
          var clipPos = camera.projection * camera.view * vec4f(position, 1.0);
          // Push depth forward to ensure rigidbodies render above model parts
          clipPos.z = clipPos.z - clipPos.w * 0.05;
          o.pos = clipPos;
          o.color = color;
          return o;
        }

        @fragment fn fs(i: VSOut) -> @location(0) vec4f { 
          return vec4f(i.color, 0.7); // Semi-transparent with per-rigidbody color
        }
      `,
    })

    this.rigidbodyPipeline = this.device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: shader,
        buffers: [
          {
            arrayStride: 9 * 4, // position(3) + color(3) + normal(3)
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" as GPUVertexFormat },
              { shaderLocation: 1, offset: 3 * 4, format: "float32x3" as GPUVertexFormat },
              { shaderLocation: 2, offset: 6 * 4, format: "float32x3" as GPUVertexFormat },
            ],
          },
        ],
      },
      fragment: {
        module: shader,
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
      primitive: { topology: "triangle-list", cullMode: "none" },
      depthStencil: {
        format: "depth24plus",
        depthWriteEnabled: true,
        depthCompare: "less-equal", // Allow depth ordering among rigidbodies while still rendering above model
      },
      multisample: { count: this.sampleCount },
    })

    // Bind group with only camera uniforms
    this.rigidbodyBindGroup = this.device.createBindGroup({
      layout: this.rigidbodyPipeline.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.cameraUniformBuffer } }],
    })
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

  // Get color for rigidbody based on type and group (common physics engine practice)
  private getRigidbodyColor(rb: Rigidbody): [number, number, number] {
    if (rb.type === RigidbodyType.Static) {
      return [0.9, 0.1, 0.1] // Red
    } else if (rb.type === RigidbodyType.Kinematic) {
      return [0.2, 0.6, 1.0] // Light blue (more visible than white)
    } else {
      // Dynamic: use green as default, vary by collision group for differentiation
      const groupColors: [number, number, number][] = [
        [0.2, 0.9, 0.2], // Group 0: Green (active dynamic)
        [1.0, 0.9, 0.2], // Group 1: Yellow
        [0.2, 0.8, 0.9], // Group 2: Cyan
        [0.9, 0.2, 0.9], // Group 3: Magenta
        [1.0, 0.6, 0.2], // Group 4: Orange
        [0.6, 0.2, 0.9], // Group 5: Purple
        [0.9, 0.9, 0.2], // Group 6: Yellow-green
        [0.2, 0.7, 1.0], // Group 7: Light blue
      ]
      const groupIdx = rb.group % groupColors.length
      return groupColors[groupIdx]
    }
  }

  // Build base meshes in local space once (called on load)
  private buildRigidbodyBaseMeshes() {
    if (!this.physics) return

    this.rigidbodyMeshes = []
    const rigidbodies = this.physics.getRigidbodies()
    const segments = 4

    const addLocalVertex = (
      vertices: number[],
      pos: [number, number, number],
      normal: [number, number, number],
      color: [number, number, number]
    ) => {
      vertices.push(pos[0], pos[1], pos[2], color[0], color[1], color[2], normal[0], normal[1], normal[2])
    }

    const addLocalTriangle = (
      vertices: number[],
      indices: number[],
      v0: [number, number, number],
      v1: [number, number, number],
      v2: [number, number, number],
      normal: [number, number, number],
      vertexOffset: number,
      color: [number, number, number]
    ) => {
      const i0 = vertexOffset
      addLocalVertex(vertices, v0, normal, color)
      const i1 = vertexOffset + 1
      addLocalVertex(vertices, v1, normal, color)
      const i2 = vertexOffset + 2
      addLocalVertex(vertices, v2, normal, color)
      indices.push(i0, i1, i2)
      return vertexOffset + 3
    }

    let globalVertexOffset = 0
    let globalIndexOffset = 0
    const meshIndexMap: number[] = [] // Map from definition index to mesh index

    for (let rbIdx = 0; rbIdx < rigidbodies.length; rbIdx++) {
      const rb = rigidbodies[rbIdx]

      // Skip kinematic rigidbodies without bone attachment (boneIndex=-1)
      if (rb.type === RigidbodyType.Kinematic && rb.boneIndex < 0) {
        continue
      }

      const vertices: number[] = []
      const indices: number[] = []
      let localVertexOffset = 0
      const color = this.getRigidbodyColor(rb)

      if (rb.shape === RigidbodyShape.Sphere) {
        const radius = rb.size.x
        const rings = segments
        const sectors = segments

        for (let i = 0; i < rings; i++) {
          const theta1 = (i / rings) * Math.PI
          const theta2 = ((i + 1) / rings) * Math.PI
          const sinT1 = Math.sin(theta1)
          const cosT1 = Math.cos(theta1)
          const sinT2 = Math.sin(theta2)
          const cosT2 = Math.cos(theta2)

          for (let j = 0; j < sectors; j++) {
            const phi1 = (j / sectors) * Math.PI * 2
            const phi2 = ((j + 1) / sectors) * Math.PI * 2
            const cosP1 = Math.cos(phi1)
            const sinP1 = Math.sin(phi1)
            const cosP2 = Math.cos(phi2)
            const sinP2 = Math.sin(phi2)

            const p1x = radius * sinT1 * cosP1
            const p1y = radius * cosT1
            const p1z = radius * sinT1 * sinP1
            const p2x = radius * sinT1 * cosP2
            const p2y = radius * cosT1
            const p2z = radius * sinT1 * sinP2
            const p3x = radius * sinT2 * cosP2
            const p3y = radius * cosT2
            const p3z = radius * sinT2 * sinP2
            const p4x = radius * sinT2 * cosP1
            const p4y = radius * cosT2
            const p4z = radius * sinT2 * sinP1

            const len3 = Math.sqrt(p3x * p3x + p3y * p3y + p3z * p3z)
            const len4 = Math.sqrt(p4x * p4x + p4y * p4y + p4z * p4z)
            const invLen3 = len3 > 0 ? 1 / len3 : 1
            const invLen4 = len4 > 0 ? 1 / len4 : 1

            localVertexOffset = addLocalTriangle(
              vertices,
              indices,
              [p1x, p1y, p1z],
              [p2x, p2y, p2z],
              [p3x, p3y, p3z],
              [p3x * invLen3, p3y * invLen3, p3z * invLen3],
              localVertexOffset,
              color
            )
            localVertexOffset = addLocalTriangle(
              vertices,
              indices,
              [p1x, p1y, p1z],
              [p3x, p3y, p3z],
              [p4x, p4y, p4z],
              [p4x * invLen4, p4y * invLen4, p4z * invLen4],
              localVertexOffset,
              color
            )
          }
        }
      } else if (rb.shape === RigidbodyShape.Box) {
        const hw = rb.size.x
        const hh = rb.size.y
        const hd = rb.size.z

        // Box faces: +Z, -Z, +Y, -Y, +X, -X
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [-hw, -hh, hd],
          [hw, -hh, hd],
          [hw, hh, hd],
          [0, 0, 1],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [-hw, -hh, hd],
          [hw, hh, hd],
          [-hw, hh, hd],
          [0, 0, 1],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [hw, -hh, -hd],
          [-hw, -hh, -hd],
          [-hw, hh, -hd],
          [0, 0, -1],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [hw, -hh, -hd],
          [-hw, hh, -hd],
          [hw, hh, -hd],
          [0, 0, -1],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [-hw, hh, -hd],
          [-hw, hh, hd],
          [hw, hh, hd],
          [0, 1, 0],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [-hw, hh, -hd],
          [hw, hh, hd],
          [hw, hh, -hd],
          [0, 1, 0],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [-hw, -hh, -hd],
          [hw, -hh, -hd],
          [hw, -hh, hd],
          [0, -1, 0],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [-hw, -hh, -hd],
          [hw, -hh, hd],
          [-hw, -hh, hd],
          [0, -1, 0],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [hw, -hh, -hd],
          [hw, hh, -hd],
          [hw, hh, hd],
          [1, 0, 0],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [hw, -hh, -hd],
          [hw, hh, hd],
          [hw, -hh, hd],
          [1, 0, 0],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [-hw, -hh, -hd],
          [-hw, -hh, hd],
          [-hw, hh, hd],
          [-1, 0, 0],
          localVertexOffset,
          color
        )
        localVertexOffset = addLocalTriangle(
          vertices,
          indices,
          [-hw, -hh, -hd],
          [-hw, hh, hd],
          [-hw, hh, -hd],
          [-1, 0, 0],
          localVertexOffset,
          color
        )
      } else if (rb.shape === RigidbodyShape.Capsule) {
        const radius = rb.size.x
        const height = rb.size.y
        const halfHeight = height * 0.5
        const rings = segments / 2
        const sectors = segments

        // Top hemisphere
        for (let i = 0; i < rings / 2; i++) {
          const theta1 = (i / rings) * Math.PI
          const theta2 = ((i + 1) / rings) * Math.PI
          const sinT1 = Math.sin(theta1)
          const cosT1 = Math.cos(theta1)
          const sinT2 = Math.sin(theta2)
          const cosT2 = Math.cos(theta2)

          for (let j = 0; j < sectors; j++) {
            const phi1 = (j / sectors) * Math.PI * 2
            const phi2 = ((j + 1) / sectors) * Math.PI * 2
            const cosP1 = Math.cos(phi1)
            const sinP1 = Math.sin(phi1)
            const cosP2 = Math.cos(phi2)
            const sinP2 = Math.sin(phi2)

            const p1x = radius * sinT1 * cosP1
            const p1y = radius * cosT1 + halfHeight
            const p1z = radius * sinT1 * sinP1
            const p2x = radius * sinT1 * cosP2
            const p2y = radius * cosT1 + halfHeight
            const p2z = radius * sinT1 * sinP2
            const p3x = radius * sinT2 * cosP2
            const p3y = radius * cosT2 + halfHeight
            const p3z = radius * sinT2 * sinP2
            const p4x = radius * sinT2 * cosP1
            const p4y = radius * cosT2 + halfHeight
            const p4z = radius * sinT2 * sinP1

            const n3x = Math.sin(theta2) * cosP2
            const n3y = Math.cos(theta2)
            const n3z = Math.sin(theta2) * sinP2
            const len3 = Math.sqrt(n3x * n3x + n3y * n3y + n3z * n3z)
            const invLen3 = len3 > 0 ? 1 / len3 : 1

            const n4x = Math.sin(theta2) * cosP1
            const n4y = Math.cos(theta2)
            const n4z = Math.sin(theta2) * sinP1
            const len4 = Math.sqrt(n4x * n4x + n4y * n4y + n4z * n4z)
            const invLen4 = len4 > 0 ? 1 / len4 : 1

            localVertexOffset = addLocalTriangle(
              vertices,
              indices,
              [p1x, p1y, p1z],
              [p2x, p2y, p2z],
              [p3x, p3y, p3z],
              [n3x * invLen3, n3y * invLen3, n3z * invLen3],
              localVertexOffset,
              color
            )
            localVertexOffset = addLocalTriangle(
              vertices,
              indices,
              [p1x, p1y, p1z],
              [p3x, p3y, p3z],
              [p4x, p4y, p4z],
              [n4x * invLen4, n4y * invLen4, n4z * invLen4],
              localVertexOffset,
              color
            )
          }
        }

        // Cylindrical middle
        for (let j = 0; j < sectors; j++) {
          const phi1 = (j / sectors) * Math.PI * 2
          const phi2 = ((j + 1) / sectors) * Math.PI * 2
          const cosP1 = Math.cos(phi1)
          const sinP1 = Math.sin(phi1)
          const cosP2 = Math.cos(phi2)
          const sinP2 = Math.sin(phi2)

          const p1x = radius * cosP1
          const p1y = halfHeight
          const p1z = radius * sinP1
          const p2x = radius * cosP2
          const p2y = halfHeight
          const p2z = radius * sinP2
          const p3x = radius * cosP2
          const p3y = -halfHeight
          const p3z = radius * sinP2
          const p4x = radius * cosP1
          const p4y = -halfHeight
          const p4z = radius * sinP1

          const nx = cosP1
          const ny = 0
          const nz = sinP1

          localVertexOffset = addLocalTriangle(
            vertices,
            indices,
            [p1x, p1y, p1z],
            [p2x, p2y, p2z],
            [p3x, p3y, p3z],
            [nx, ny, nz],
            localVertexOffset,
            color
          )
          localVertexOffset = addLocalTriangle(
            vertices,
            indices,
            [p1x, p1y, p1z],
            [p3x, p3y, p3z],
            [p4x, p4y, p4z],
            [nx, ny, nz],
            localVertexOffset,
            color
          )
        }

        // Bottom hemisphere
        for (let i = rings / 2; i < rings; i++) {
          const theta1 = (i / rings) * Math.PI
          const theta2 = ((i + 1) / rings) * Math.PI
          const sinT1 = Math.sin(theta1)
          const cosT1 = Math.cos(theta1)
          const sinT2 = Math.sin(theta2)
          const cosT2 = Math.cos(theta2)

          for (let j = 0; j < sectors; j++) {
            const phi1 = (j / sectors) * Math.PI * 2
            const phi2 = ((j + 1) / sectors) * Math.PI * 2
            const cosP1 = Math.cos(phi1)
            const sinP1 = Math.sin(phi1)
            const cosP2 = Math.cos(phi2)
            const sinP2 = Math.sin(phi2)

            const p1x = radius * sinT1 * cosP1
            const p1y = radius * cosT1 - halfHeight
            const p1z = radius * sinT1 * sinP1
            const p2x = radius * sinT1 * cosP2
            const p2y = radius * cosT1 - halfHeight
            const p2z = radius * sinT1 * sinP2
            const p3x = radius * sinT2 * cosP2
            const p3y = radius * cosT2 - halfHeight
            const p3z = radius * sinT2 * sinP2
            const p4x = radius * sinT2 * cosP1
            const p4y = radius * cosT2 - halfHeight
            const p4z = radius * sinT2 * sinP1

            const n3x = Math.sin(theta2) * cosP2
            const n3y = Math.cos(theta2)
            const n3z = Math.sin(theta2) * sinP2
            const len3 = Math.sqrt(n3x * n3x + n3y * n3y + n3z * n3z)
            const invLen3 = len3 > 0 ? 1 / len3 : 1

            const n4x = Math.sin(theta2) * cosP1
            const n4y = Math.cos(theta2)
            const n4z = Math.sin(theta2) * sinP1
            const len4 = Math.sqrt(n4x * n4x + n4y * n4y + n4z * n4z)
            const invLen4 = len4 > 0 ? 1 / len4 : 1

            localVertexOffset = addLocalTriangle(
              vertices,
              indices,
              [p1x, p1y, p1z],
              [p2x, p2y, p2z],
              [p3x, p3y, p3z],
              [n3x * invLen3, n3y * invLen3, n3z * invLen3],
              localVertexOffset,
              color
            )
            localVertexOffset = addLocalTriangle(
              vertices,
              indices,
              [p1x, p1y, p1z],
              [p3x, p3y, p3z],
              [p4x, p4y, p4z],
              [n4x * invLen4, n4y * invLen4, n4z * invLen4],
              localVertexOffset,
              color
            )
          }
        }
      }

      if (vertices.length > 0 && indices.length > 0) {
        // Adjust indices for global offset
        const adjustedIndices = indices.map((idx) => idx + globalVertexOffset)
        this.rigidbodyMeshes.push({
          baseVertices: new Float32Array(vertices),
          indices: new Uint32Array(adjustedIndices),
          vertexOffset: globalVertexOffset,
          indexOffset: globalIndexOffset,
          color: color,
        })
        meshIndexMap.push(rbIdx) // Store mapping from mesh index to definition index
        globalVertexOffset += vertices.length / 9
        globalIndexOffset += indices.length
      }
    }

    // Store mesh index map for transform updates
    this.rigidbodyMeshIndexMap = meshIndexMap

    // Create combined index buffer once
    if (this.rigidbodyMeshes.length > 0) {
      const allIndices: number[] = []
      for (const mesh of this.rigidbodyMeshes) {
        allIndices.push(...mesh.indices)
      }
      const indexData = new Uint32Array(allIndices)
      this.rigidbodyIndexCount = indexData.length

      if (this.rigidbodyIndexBuffer && this.rigidbodyIndexBuffer.size >= indexData.byteLength) {
        this.device.queue.writeBuffer(this.rigidbodyIndexBuffer, 0, indexData)
      } else {
        if (this.rigidbodyIndexBuffer) {
          this.rigidbodyIndexBuffer.destroy()
        }
        this.rigidbodyIndexBuffer = this.device.createBuffer({
          label: "rigidbody visualization indices",
          size: indexData.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(this.rigidbodyIndexBuffer, 0, indexData)
      }
    }
  }

  // Update transforms each frame (transforms cached base vertices)
  private updateRigidbodyTransforms() {
    if (!this.physics || this.rigidbodyMeshes.length === 0 || !this.rigidbodyVertexBuffer) return

    const definitions = this.physics.getRigidbodies()
    const allVertices: number[] = []

    for (let meshIdx = 0; meshIdx < this.rigidbodyMeshes.length; meshIdx++) {
      const defIdx = this.rigidbodyMeshIndexMap[meshIdx]
      const rb = definitions[defIdx]
      const mesh = this.rigidbodyMeshes[meshIdx]
      const baseVertices = mesh.baseVertices
      const color = mesh.color
      const rotMat = Mat4.fromQuat(rb.rotation.x, rb.rotation.y, rb.rotation.z, rb.rotation.w)
      const m = rotMat.values
      const cx = rb.position.x
      const cy = rb.position.y
      const cz = rb.position.z

      // Transform each vertex
      for (let j = 0; j < baseVertices.length; j += 9) {
        const px = baseVertices[j]
        const py = baseVertices[j + 1]
        const pz = baseVertices[j + 2]
        const nx = baseVertices[j + 6]
        const ny = baseVertices[j + 7]
        const nz = baseVertices[j + 8]

        // Transform position
        const wx = m[0] * px + m[4] * py + m[8] * pz + cx
        const wy = m[1] * px + m[5] * py + m[9] * pz + cy
        const wz = m[2] * px + m[6] * py + m[10] * pz + cz

        // Transform normal
        const tnx = m[0] * nx + m[4] * ny + m[8] * nz
        const tny = m[1] * nx + m[5] * ny + m[9] * nz
        const tnz = m[2] * nx + m[6] * ny + m[10] * nz
        const len = Math.sqrt(tnx * tnx + tny * tny + tnz * tnz)
        const invLen = len > 0 ? 1 / len : 1

        allVertices.push(wx, wy, wz, color[0], color[1], color[2], tnx * invLen, tny * invLen, tnz * invLen)
      }
    }

    if (allVertices.length > 0) {
      const vertexData = new Float32Array(allVertices)
      this.rigidbodyVertexCount = vertexData.length / 9
      this.device.queue.writeBuffer(this.rigidbodyVertexBuffer, 0, vertexData)
    }
  }

  private buildRigidbodyVisualization() {
    if (!this.currentModel || !this.physics) return

    // If meshes are already cached, just create the vertex buffer
    if (this.rigidbodyMeshes.length > 0) {
      const totalVertices = this.rigidbodyMeshes.reduce((sum, mesh) => sum + mesh.baseVertices.length / 9, 0)
      const vertexDataSize = totalVertices * 9 * 4 // 9 floats per vertex, 4 bytes per float

      if (!this.rigidbodyVertexBuffer || this.rigidbodyVertexBuffer.size < vertexDataSize) {
        if (this.rigidbodyVertexBuffer) {
          this.rigidbodyVertexBuffer.destroy()
        }
        this.rigidbodyVertexBuffer = this.device.createBuffer({
          label: "rigidbody visualization vertices",
          size: vertexDataSize,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        })
      }

      // Initial transform update
      this.updateRigidbodyTransforms()
      return
    }

    // Fallback: old code path if meshes not built
    const vertices: number[] = []
    const indices: number[] = []
    let vertexOffset = 0
    const color = [1.0, 1.0, 0.0]

    // Optimized transform functions using direct array operations to avoid object creation
    const transformPointAndNormal = (
      px: number,
      py: number,
      pz: number,
      nx: number,
      ny: number,
      nz: number,
      m: Float32Array,
      cx: number,
      cy: number,
      cz: number
    ) => {
      // Transform position
      const wx = m[0] * px + m[4] * py + m[8] * pz + cx
      const wy = m[1] * px + m[5] * py + m[9] * pz + cy
      const wz = m[2] * px + m[6] * py + m[10] * pz + cz

      // Transform normal
      const tnx = m[0] * nx + m[4] * ny + m[8] * nz
      const tny = m[1] * nx + m[5] * ny + m[9] * nz
      const tnz = m[2] * nx + m[6] * ny + m[10] * nz
      const len = Math.sqrt(tnx * tnx + tny * tny + tnz * tnz)
      const invLen = len > 0 ? 1 / len : 1

      vertices.push(wx, wy, wz, color[0], color[1], color[2], tnx * invLen, tny * invLen, tnz * invLen)
    }

    const addVertexSimple = (
      px: number,
      py: number,
      pz: number,
      nx: number,
      ny: number,
      nz: number,
      cx: number,
      cy: number,
      cz: number
    ) => {
      vertices.push(px + cx, py + cy, pz + cz, color[0], color[1], color[2], nx, ny, nz)
    }

    const addTriangle = (
      v0x: number,
      v0y: number,
      v0z: number,
      v1x: number,
      v1y: number,
      v1z: number,
      v2x: number,
      v2y: number,
      v2z: number,
      nx: number,
      ny: number,
      nz: number,
      cx: number,
      cy: number,
      cz: number,
      m?: Float32Array
    ) => {
      const i0 = vertexOffset
      if (m) {
        transformPointAndNormal(v0x, v0y, v0z, nx, ny, nz, m, cx, cy, cz)
      } else {
        addVertexSimple(v0x, v0y, v0z, nx, ny, nz, cx, cy, cz)
      }
      vertexOffset++
      const i1 = vertexOffset
      if (m) {
        transformPointAndNormal(v1x, v1y, v1z, nx, ny, nz, m, cx, cy, cz)
      } else {
        addVertexSimple(v1x, v1y, v1z, nx, ny, nz, cx, cy, cz)
      }
      vertexOffset++
      const i2 = vertexOffset
      if (m) {
        transformPointAndNormal(v2x, v2y, v2z, nx, ny, nz, m, cx, cy, cz)
      } else {
        addVertexSimple(v2x, v2y, v2z, nx, ny, nz, cx, cy, cz)
      }
      vertexOffset++
      indices.push(i0, i1, i2)
    }

    const addSphere = (cx: number, cy: number, cz: number, radius: number, segments: number, m?: Float32Array) => {
      const rings = segments
      const sectors = segments

      for (let i = 0; i < rings; i++) {
        const theta1 = (i / rings) * Math.PI
        const theta2 = ((i + 1) / rings) * Math.PI
        const sinT1 = Math.sin(theta1)
        const cosT1 = Math.cos(theta1)
        const sinT2 = Math.sin(theta2)
        const cosT2 = Math.cos(theta2)

        for (let j = 0; j < sectors; j++) {
          const phi1 = (j / sectors) * Math.PI * 2
          const phi2 = ((j + 1) / sectors) * Math.PI * 2
          const cosP1 = Math.cos(phi1)
          const sinP1 = Math.sin(phi1)
          const cosP2 = Math.cos(phi2)
          const sinP2 = Math.sin(phi2)

          // Generate 4 points for a quad (avoid Vec3 creation)
          const p1x = radius * sinT1 * cosP1
          const p1y = radius * cosT1
          const p1z = radius * sinT1 * sinP1
          const p2x = radius * sinT1 * cosP2
          const p2y = radius * cosT1
          const p2z = radius * sinT1 * sinP2
          const p3x = radius * sinT2 * cosP2
          const p3y = radius * cosT2
          const p3z = radius * sinT2 * sinP2
          const p4x = radius * sinT2 * cosP1
          const p4y = radius * cosT2
          const p4z = radius * sinT2 * sinP1

          // Normals (same as positions for sphere, normalized)
          const len3 = Math.sqrt(p3x * p3x + p3y * p3y + p3z * p3z)
          const len4 = Math.sqrt(p4x * p4x + p4y * p4y + p4z * p4z)
          const invLen3 = len3 > 0 ? 1 / len3 : 1
          const invLen4 = len4 > 0 ? 1 / len4 : 1

          addTriangle(
            p1x,
            p1y,
            p1z,
            p2x,
            p2y,
            p2z,
            p3x,
            p3y,
            p3z,
            p3x * invLen3,
            p3y * invLen3,
            p3z * invLen3,
            cx,
            cy,
            cz,
            m
          )
          addTriangle(
            p1x,
            p1y,
            p1z,
            p3x,
            p3y,
            p3z,
            p4x,
            p4y,
            p4z,
            p4x * invLen4,
            p4y * invLen4,
            p4z * invLen4,
            cx,
            cy,
            cz,
            m
          )
        }
      }
    }

    const addBox = (cx: number, cy: number, cz: number, sx: number, sy: number, sz: number, m?: Float32Array) => {
      const hw = sx,
        hh = sy,
        hd = sz

      // Face 1: +Z
      addTriangle(-hw, -hh, hd, hw, -hh, hd, hw, hh, hd, 0, 0, 1, cx, cy, cz, m)
      addTriangle(-hw, -hh, hd, hw, hh, hd, -hw, hh, hd, 0, 0, 1, cx, cy, cz, m)
      // Face 2: -Z
      addTriangle(hw, -hh, -hd, -hw, -hh, -hd, -hw, hh, -hd, 0, 0, -1, cx, cy, cz, m)
      addTriangle(hw, -hh, -hd, -hw, hh, -hd, hw, hh, -hd, 0, 0, -1, cx, cy, cz, m)
      // Face 3: +Y
      addTriangle(-hw, hh, -hd, -hw, hh, hd, hw, hh, hd, 0, 1, 0, cx, cy, cz, m)
      addTriangle(-hw, hh, -hd, hw, hh, hd, hw, hh, -hd, 0, 1, 0, cx, cy, cz, m)
      // Face 4: -Y
      addTriangle(-hw, -hh, -hd, hw, -hh, -hd, hw, -hh, hd, 0, -1, 0, cx, cy, cz, m)
      addTriangle(-hw, -hh, -hd, hw, -hh, hd, -hw, -hh, hd, 0, -1, 0, cx, cy, cz, m)
      // Face 5: +X
      addTriangle(hw, -hh, -hd, hw, hh, -hd, hw, hh, hd, 1, 0, 0, cx, cy, cz, m)
      addTriangle(hw, -hh, -hd, hw, hh, hd, hw, -hh, hd, 1, 0, 0, cx, cy, cz, m)
      // Face 6: -X
      addTriangle(-hw, -hh, -hd, -hw, -hh, hd, -hw, hh, hd, -1, 0, 0, cx, cy, cz, m)
      addTriangle(-hw, -hh, -hd, -hw, hh, hd, -hw, hh, -hd, -1, 0, 0, cx, cy, cz, m)
    }

    // Helper function to add a solid capsule (optimized)
    const addCapsule = (
      cx: number,
      cy: number,
      cz: number,
      radius: number,
      height: number,
      segments: number,
      m?: Float32Array
    ) => {
      const halfHeight = height * 0.5
      const rings = segments / 2
      const sectors = segments

      // Top hemisphere
      for (let i = 0; i < rings / 2; i++) {
        const theta1 = (i / rings) * Math.PI
        const theta2 = ((i + 1) / rings) * Math.PI
        const sinT1 = Math.sin(theta1)
        const cosT1 = Math.cos(theta1)
        const sinT2 = Math.sin(theta2)
        const cosT2 = Math.cos(theta2)

        for (let j = 0; j < sectors; j++) {
          const phi1 = (j / sectors) * Math.PI * 2
          const phi2 = ((j + 1) / sectors) * Math.PI * 2
          const cosP1 = Math.cos(phi1)
          const sinP1 = Math.sin(phi1)
          const cosP2 = Math.cos(phi2)
          const sinP2 = Math.sin(phi2)

          const p1x = radius * sinT1 * cosP1
          const p1y = radius * cosT1 + halfHeight
          const p1z = radius * sinT1 * sinP1
          const p2x = radius * sinT1 * cosP2
          const p2y = radius * cosT1 + halfHeight
          const p2z = radius * sinT1 * sinP2
          const p3x = radius * sinT2 * cosP2
          const p3y = radius * cosT2 + halfHeight
          const p3z = radius * sinT2 * sinP2
          const p4x = radius * sinT2 * cosP1
          const p4y = radius * cosT2 + halfHeight
          const p4z = radius * sinT2 * sinP1

          const n3x = Math.sin(theta2) * cosP2
          const n3y = Math.cos(theta2)
          const n3z = Math.sin(theta2) * sinP2
          const len3 = Math.sqrt(n3x * n3x + n3y * n3y + n3z * n3z)
          const invLen3 = len3 > 0 ? 1 / len3 : 1

          const n4x = Math.sin(theta2) * cosP1
          const n4y = Math.cos(theta2)
          const n4z = Math.sin(theta2) * sinP1
          const len4 = Math.sqrt(n4x * n4x + n4y * n4y + n4z * n4z)
          const invLen4 = len4 > 0 ? 1 / len4 : 1

          addTriangle(
            p1x,
            p1y,
            p1z,
            p2x,
            p2y,
            p2z,
            p3x,
            p3y,
            p3z,
            n3x * invLen3,
            n3y * invLen3,
            n3z * invLen3,
            cx,
            cy,
            cz,
            m
          )
          addTriangle(
            p1x,
            p1y,
            p1z,
            p3x,
            p3y,
            p3z,
            p4x,
            p4y,
            p4z,
            n4x * invLen4,
            n4y * invLen4,
            n4z * invLen4,
            cx,
            cy,
            cz,
            m
          )
        }
      }

      // Cylindrical middle part
      for (let j = 0; j < sectors; j++) {
        const phi1 = (j / sectors) * Math.PI * 2
        const phi2 = ((j + 1) / sectors) * Math.PI * 2
        const cosP1 = Math.cos(phi1)
        const sinP1 = Math.sin(phi1)
        const cosP2 = Math.cos(phi2)
        const sinP2 = Math.sin(phi2)

        const p1x = radius * cosP1
        const p1y = halfHeight
        const p1z = radius * sinP1
        const p2x = radius * cosP2
        const p2y = halfHeight
        const p2z = radius * sinP2
        const p3x = radius * cosP2
        const p3y = -halfHeight
        const p3z = radius * sinP2
        const p4x = radius * cosP1
        const p4y = -halfHeight
        const p4z = radius * sinP1

        const nx = cosP1
        const ny = 0
        const nz = sinP1

        addTriangle(p1x, p1y, p1z, p2x, p2y, p2z, p3x, p3y, p3z, nx, ny, nz, cx, cy, cz, m)
        addTriangle(p1x, p1y, p1z, p3x, p3y, p3z, p4x, p4y, p4z, nx, ny, nz, cx, cy, cz, m)
      }

      // Bottom hemisphere
      for (let i = rings / 2; i < rings; i++) {
        const theta1 = (i / rings) * Math.PI
        const theta2 = ((i + 1) / rings) * Math.PI
        const sinT1 = Math.sin(theta1)
        const cosT1 = Math.cos(theta1)
        const sinT2 = Math.sin(theta2)
        const cosT2 = Math.cos(theta2)

        for (let j = 0; j < sectors; j++) {
          const phi1 = (j / sectors) * Math.PI * 2
          const phi2 = ((j + 1) / sectors) * Math.PI * 2
          const cosP1 = Math.cos(phi1)
          const sinP1 = Math.sin(phi1)
          const cosP2 = Math.cos(phi2)
          const sinP2 = Math.sin(phi2)

          const p1x = radius * sinT1 * cosP1
          const p1y = radius * cosT1 - halfHeight
          const p1z = radius * sinT1 * sinP1
          const p2x = radius * sinT1 * cosP2
          const p2y = radius * cosT1 - halfHeight
          const p2z = radius * sinT1 * sinP2
          const p3x = radius * sinT2 * cosP2
          const p3y = radius * cosT2 - halfHeight
          const p3z = radius * sinT2 * sinP2
          const p4x = radius * sinT2 * cosP1
          const p4y = radius * cosT2 - halfHeight
          const p4z = radius * sinT2 * sinP1

          const n3x = Math.sin(theta2) * cosP2
          const n3y = Math.cos(theta2)
          const n3z = Math.sin(theta2) * sinP2
          const len3 = Math.sqrt(n3x * n3x + n3y * n3y + n3z * n3z)
          const invLen3 = len3 > 0 ? 1 / len3 : 1

          const n4x = Math.sin(theta2) * cosP1
          const n4y = Math.cos(theta2)
          const n4z = Math.sin(theta2) * sinP1
          const len4 = Math.sqrt(n4x * n4x + n4y * n4y + n4z * n4z)
          const invLen4 = len4 > 0 ? 1 / len4 : 1

          addTriangle(
            p1x,
            p1y,
            p1z,
            p2x,
            p2y,
            p2z,
            p3x,
            p3y,
            p3z,
            n3x * invLen3,
            n3y * invLen3,
            n3z * invLen3,
            cx,
            cy,
            cz,
            m
          )
          addTriangle(
            p1x,
            p1y,
            p1z,
            p3x,
            p3y,
            p3z,
            p4x,
            p4y,
            p4z,
            n4x * invLen4,
            n4y * invLen4,
            n4z * invLen4,
            cx,
            cy,
            cz,
            m
          )
        }
      }
    }

    const definitions = this.physics.getRigidbodies()
    const segments = 4 // Minimal segments for debug visualization

    // Pre-compute rotation matrices once
    const rotationMatrices: (Float32Array | undefined)[] = []
    for (let i = 0; i < definitions.length; i++) {
      const rb = definitions[i]
      const rotMat = Mat4.fromQuat(rb.rotation.x, rb.rotation.y, rb.rotation.z, rb.rotation.w)
      rotationMatrices.push(rotMat.values)
    }

    for (let i = 0; i < definitions.length; i++) {
      const rb = definitions[i]
      const cx = rb.position.x
      const cy = rb.position.y
      const cz = rb.position.z
      const m = rotationMatrices[i]

      if (rb.shape === RigidbodyShape.Sphere) {
        addSphere(cx, cy, cz, rb.size.x, segments, m)
      } else if (rb.shape === RigidbodyShape.Box) {
        addBox(cx, cy, cz, rb.size.x, rb.size.y, rb.size.z, m)
      } else if (rb.shape === RigidbodyShape.Capsule) {
        addCapsule(cx, cy, cz, rb.size.x, rb.size.y, segments, m)
      }
    }

    if (vertices.length > 0 && indices.length > 0) {
      const vertexData = new Float32Array(vertices)
      const indexData = new Uint32Array(indices)
      this.rigidbodyVertexCount = vertexData.length / 9
      this.rigidbodyIndexCount = indexData.length

      // Reuse buffers if they exist and are the right size, otherwise recreate
      if (this.rigidbodyVertexBuffer && this.rigidbodyVertexBuffer.size >= vertexData.byteLength) {
        this.device.queue.writeBuffer(this.rigidbodyVertexBuffer, 0, vertexData)
      } else {
        if (this.rigidbodyVertexBuffer) {
          this.rigidbodyVertexBuffer.destroy()
        }
        this.rigidbodyVertexBuffer = this.device.createBuffer({
          label: "rigidbody visualization vertices",
          size: vertexData.byteLength,
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(this.rigidbodyVertexBuffer, 0, vertexData)
      }

      if (this.rigidbodyIndexBuffer && this.rigidbodyIndexBuffer.size >= indexData.byteLength) {
        this.device.queue.writeBuffer(this.rigidbodyIndexBuffer, 0, indexData)
      } else {
        if (this.rigidbodyIndexBuffer) {
          this.rigidbodyIndexBuffer.destroy()
        }
        this.rigidbodyIndexBuffer = this.device.createBuffer({
          label: "rigidbody visualization indices",
          size: indexData.byteLength,
          usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        })
        this.device.queue.writeBuffer(this.rigidbodyIndexBuffer, 0, indexData)
      }

      console.log(
        `[Engine] Built rigidbody visualization: ${this.physics?.getRigidbodies().length || 0} rigidbodies, ${
          this.rigidbodyVertexCount
        } vertices, ${this.rigidbodyIndexCount} indices`
      )
    }
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

  public setShowRigidbodies(show: boolean): void {
    this.showRigidbodies = show
  }

  public getShowRigidbodies(): boolean {
    return this.showRigidbodies
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
    // model.rotateBones(
    //   ["腰", "左腕", "左足"],
    //   [new Quat(-0.5, -0.3, 0, 1), new Quat(-0.3, 0.3, -0.3, 1), new Quat(-0.3, -0.3, 0.3, 1)],
    //   2000
    // )
    this.physics = new Physics(model.getRigidbodies(), model.getJoints())
    await this.drawModel(model)

    if (this.physics.getRigidbodies().length > 0) {
      this.buildRigidbodyBaseMeshes()
      this.buildRigidbodyVisualization()
    }

    const boneNames = model.getBoneNames()
    console.log("Available bones:", boneNames)
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

        // Update rigidbody visualization if enabled
        if (this.showRigidbodies) {
          this.updateRigidbodyTransforms()
        }
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
    // Draw rigidbodies last (always on top)
    if (
      this.showRigidbodies &&
      this.rigidbodyPipeline &&
      this.rigidbodyVertexBuffer &&
      this.rigidbodyIndexBuffer &&
      this.rigidbodyBindGroup &&
      this.rigidbodyIndexCount > 0
    ) {
      pass.setPipeline(this.rigidbodyPipeline)
      pass.setVertexBuffer(0, this.rigidbodyVertexBuffer)
      pass.setIndexBuffer(this.rigidbodyIndexBuffer, "uint32")
      pass.setBindGroup(0, this.rigidbodyBindGroup)
      pass.drawIndexed(this.rigidbodyIndexCount)
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
