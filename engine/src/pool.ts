import { Vec3 } from "./math"

export interface PoolOptions {
  y?: number // Y position (default: 12)
  size?: number // Plane size (default: 100)
  segments?: number // Subdivision count (default: 50)
}

export class Pool {
  private device!: GPUDevice
  private vertexBuffer!: GPUBuffer
  private indexBuffer!: GPUBuffer
  private pipeline!: GPURenderPipeline
  private bindGroup!: GPUBindGroup
  private bindGroupLayout!: GPUBindGroupLayout
  private uniformBuffer!: GPUBuffer
  private cameraBindGroupLayout!: GPUBindGroupLayout
  private cameraBindGroup!: GPUBindGroup
  private cameraUniformBuffer!: GPUBuffer
  private indexCount: number = 0
  private y: number
  private size: number
  private segments: number
  private seaColor: Vec3
  private seaLight: Vec3
  private startTime: number = performance.now()

  constructor(
    device: GPUDevice,
    cameraBindGroupLayout: GPUBindGroupLayout,
    cameraUniformBuffer: GPUBuffer,
    options?: PoolOptions
  ) {
    this.device = device
    this.cameraBindGroupLayout = cameraBindGroupLayout
    this.cameraUniformBuffer = cameraUniformBuffer
    this.y = options?.y ?? 15
    this.size = options?.size ?? 100
    this.segments = options?.segments ?? 50
    // Hardcoded dark night pool colors (not used in shader, but kept for uniform buffer)
    this.seaColor = new Vec3(0.02, 0.05, 0.12) // Dark night pool base
    this.seaLight = new Vec3(0.1, 0.3, 0.5) // Light cyan for lit areas
  }

  public async init() {
    this.createGeometry()
    this.createShader()
    this.createUniforms()
  }

  private createGeometry() {
    const segments = this.segments
    const size = this.size
    const halfSize = size / 2
    const step = size / segments

    // Generate vertices
    const vertices: number[] = []
    for (let i = 0; i <= segments; i++) {
      for (let j = 0; j <= segments; j++) {
        const x = -halfSize + j * step
        const z = -halfSize + i * step
        const y = this.y
        const u = j / segments
        const v = i / segments

        // Position (x, y, z) + UV (u, v)
        vertices.push(x, y, z, u, v)
      }
    }

    // Generate indices
    const indices: number[] = []
    for (let i = 0; i < segments; i++) {
      for (let j = 0; j < segments; j++) {
        const topLeft = i * (segments + 1) + j
        const topRight = topLeft + 1
        const bottomLeft = (i + 1) * (segments + 1) + j
        const bottomRight = bottomLeft + 1

        // Two triangles per quad
        indices.push(topLeft, bottomLeft, topRight)
        indices.push(topRight, bottomLeft, bottomRight)
      }
    }

    this.indexCount = indices.length

    // Create buffers
    this.vertexBuffer = this.device.createBuffer({
      label: "pool vertices",
      size: vertices.length * 4,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.vertexBuffer, 0, new Float32Array(vertices))

    const indexBufferSize = indices.length * 4
    this.indexBuffer = this.device.createBuffer({
      label: "pool indices",
      size: indexBufferSize,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.indexBuffer, 0, new Uint32Array(indices))

    // Verify: segments=50 should give 50*50*6 = 15000 indices = 60000 bytes
    if (this.indexCount !== indices.length) {
      console.warn(`Pool index count mismatch: expected ${indices.length}, got ${this.indexCount}`)
    }
  }

  private createShader() {
    const shaderModule = this.device.createShaderModule({
      label: "pool shader",
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
          viewPos: vec3f,
          _padding: f32,
        };

        struct PoolUniforms {
          time: f32,
          poolY: f32,
          seaColor: vec3f,
          seaLight: vec3f,
        };

        struct VertexOutput {
          @builtin(position) position: vec4f,
          @location(0) worldPos: vec3f,
          @location(1) uv: vec2f,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(1) @binding(0) var<uniform> pool: PoolUniforms;

        // Procedural noise function (simplified)
        fn hash(p: vec2f) -> f32 {
          var p3 = fract(vec3f(p.xyx) * vec3f(443.8975, 397.2973, 491.1871));
          p3 += dot(p3, p3.yzx + 19.19);
          return fract((p3.x + p3.y) * p3.z);
        }

        fn noise(p: vec2f) -> f32 {
          let i = floor(p);
          var f = fract(p);
          f = f * f * (3.0 - 2.0 * f);
          
          let a = hash(i);
          let b = hash(i + vec2f(1.0, 0.0));
          let c = hash(i + vec2f(0.0, 1.0));
          let d = hash(i + vec2f(1.0, 1.0));
          
          return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        // Layered noise for water height (adapted from Shadertoy - matches reference exactly)
        fn waterHeight(uv: vec2f, time: f32) -> f32 {
          var e = 0.0;
          // Match Shadertoy: time*mod(j,.789)*.1 - time*.05
          for (var j = 1.0; j < 6.0; j += 1.0) {
            let timeOffset = time * (j % 0.789) * 0.1 - time * 0.05;
            let scaledUV = uv * (j * 1.789) + j * 159.45 + timeOffset;
            e += noise(scaledUV) / j;
          }
          return e / 6.0;
        }

        // Calculate water normals from height gradients (matches Shadertoy reference)
        fn waterNormals(uv: vec2f, time: f32) -> vec3f {
          // Match Shadertoy: uv.x *= .25 (scale X differently for more wave detail)
          let scaledUV = vec2f(uv.x * 0.25, uv.y);
          let eps = 0.008; // Match Shadertoy epsilon
          let h = waterHeight(scaledUV, time);
          let hx = waterHeight(scaledUV + vec2f(eps, 0.0), time);
          let hz = waterHeight(scaledUV + vec2f(0.0, eps), time);
          
          // Match Shadertoy normal calculation exactly
          let n = vec3f(h - hx, 1.0, h - hz);
          return normalize(n);
        }

        @vertex fn vs(
          @location(0) position: vec3f,
          @location(1) uv: vec2f
        ) -> VertexOutput {
          var output: VertexOutput;
          
          // Displace Y based on water height - much higher waves
          let time = pool.time;
          // More wave detail - smaller scale for more waves
          // Wave direction: back-left to front-right (both U and V increase)
          let waveUV = uv * 12.0 + vec2f(time * 0.3, time * 0.2); // Front-right direction (both positive)
          let height = waterHeight(waveUV, time) * 2; // Much higher wave amplitude
          let displacedY = position.y + height;
          
          let worldPos = vec3f(position.x, displacedY, position.z);
          output.worldPos = worldPos;
          output.uv = uv;
          output.position = camera.projection * camera.view * vec4f(worldPos, 1.0);
          
          return output;
        }

        @fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
          let time = pool.time;
          // More wave detail - smaller scale for more waves (matches Shadertoy approach)
          // Wave direction: back-left to front-right (both U and V increase)
          let uv = input.uv * 12.0 + vec2f(time * 0.3, time * 0.2); // Front-right direction (both positive)
          
          // Calculate water normals from height gradients (this creates the wave effect)
          let n = waterNormals(uv, time);
          
          // View direction
          let viewDir = normalize(camera.viewPos - input.worldPos);
          
          // Fresnel effect for reflection (stronger at glancing angles)
          var fresnel = 1.0 - max(dot(n, viewDir), 0.0);
          fresnel = fresnel * fresnel;
          
          // Dark night pool - very dark base
          let darkPoolColor = vec3f(0.01, 0.02, 0.05); // Very dark blue-black
          
          // Center spotlight effect - reflection-like bright center
          let centerUV = input.uv - vec2f(0.5, 0.5); // Center at (0.5, 0.5)
          let distFromCenter = length(centerUV);
          // Smaller spotlight area with very smooth, subtle gradient
          let spotlightFalloff = 1.0 - smoothstep(0.0, 0.12, distFromCenter); // Smaller radius (0.12)
          
          // Reflection-like bright center - brighter, balanced blue
          let spotlightColor = vec3f(0.2, 0.4, 0.6); // Balanced blue
          let spotlightCenter = vec3f(0.5, 0.65, 0.85); // More white center
          
          // Very smooth, subtle gradient mix - multiple smoothstep layers for smoother transition
          let falloff1 = smoothstep(0.0, 0.12, distFromCenter); // Outer edge
          let falloff2 = smoothstep(0.0, 0.08, distFromCenter); // Inner edge
          var color = mix(darkPoolColor, spotlightColor, (1.0 - falloff1) * 0.9); // Brighter outer gradient
          color = mix(color, spotlightCenter, (1.0 - falloff2) * (1.0 - falloff2) * 1.0); // Very bright inner reflection
          
          // Add reflection-like effect based on view angle and normals
          let reflectionFactor = max(dot(n, vec3f(0.0, 1.0, 0.0)), 0.0); // More reflection when looking down
          let reflectionBrightness = spotlightFalloff * reflectionFactor * 0.5;
          color += spotlightCenter * reflectionBrightness; // Add reflection-like brightness
          
          // Wave-based color variation (matches Shadertoy transparency approach)
          // Match Shadertoy: transparency = dot(n, vec3(0.,.2,1.5)) * 12. + 1.5
          var transparency = dot(n, vec3f(0.0, 0.2, 1.5));
          transparency = (transparency * 12.0 + 1.5);
          
          // Match Shadertoy color mixing: mix with seaColor and seaLight (brighter, balanced blue)
          let seaColor = vec3f(0.08, 0.18, 0.3); // Balanced blue
          let seaLight = vec3f(0.12, 0.25, 0.45); // Balanced blue
          // Only apply this mixing subtly to avoid green tint
          color = mix(color, seaColor, clamp(transparency, 0.0, 1.0) * 0.3);
          color = mix(color, seaLight, max(0.0, transparency - 1.5) * 0.2);
          
          // Enhanced wave-based color variation for more visible waves
          let waveHeight = waterHeight(uv, time);
          let waveContrast = (waveHeight - 0.5) * 0.3; // Amplify wave contrast
          color += vec3f(0.05, 0.08, 0.15) * waveContrast * spotlightFalloff; // Balanced blue wave highlights
          
          // Enhanced underwater glow - white/neutral glow with subtle blue tint, much brighter
          let glowIntensity = spotlightFalloff * 0.6 + fresnel * 0.3 + waveHeight * 0.2; // Stronger glow
          let glowColor = vec3f(0.35, 0.35, 0.45); // Brighter glow with subtle blue tint
          color += glowColor * glowIntensity;
          
          // Additional subtle white glow around the center, brighter and more white
          let centerGlow = spotlightFalloff * spotlightFalloff * 0.4; // Soft glow falloff
          let whiteGlow = vec3f(0.55, 0.55, 0.6); // Brighter white glow
          color += whiteGlow * centerGlow * 0.7; // Brighter white center glow
          
          // Reflection of dark night sky
          let nightSkyColor = vec3f(0.02, 0.04, 0.08); // Very dark night sky
          let reflection = mix(darkPoolColor, nightSkyColor, fresnel * 0.2);
          color = mix(color, reflection, fresnel * 0.3 * (1.0 - spotlightFalloff)); // Less reflection in spotlight
          
          // Specular highlights from underwater lights (bokeh-like bright spots)
          let lightDir1 = normalize(vec3f(0.3, -0.4, 0.6));
          let lightDir2 = normalize(vec3f(-0.3, -0.3, 0.7));
          let reflDir1 = reflect(-lightDir1, n);
          let reflDir2 = reflect(-lightDir2, n);
          var specular1 = max(dot(viewDir, reflDir1), 0.0);
          var specular2 = max(dot(viewDir, reflDir2), 0.0);
          specular1 = pow(specular1, 150.0); // Tight, bright highlights
          specular2 = pow(specular2, 180.0);
          // Subtle white/blue highlights (bokeh effect) - darker, less blue
          color += vec3f(0.8, 0.8, 0.9) * specular1 * 1.2 * spotlightFalloff; // Darker white
          color += vec3f(0.4, 0.5, 0.7) * specular2 * 0.9 * spotlightFalloff; // Darker, less blue
          
          return vec4f(color, 0.8); // Half transparent water
        }
      `,
    })

    // Create bind group layout for pool uniforms
    this.bindGroupLayout = this.device.createBindGroupLayout({
      label: "pool bind group layout",
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
          buffer: {
            type: "uniform",
          },
        },
      ],
    })

    // Create render pipeline
    this.pipeline = this.device.createRenderPipeline({
      label: "pool pipeline",
      layout: this.device.createPipelineLayout({
        bindGroupLayouts: [this.cameraBindGroupLayout, this.bindGroupLayout],
      }),
      vertex: {
        module: shaderModule,
        entryPoint: "vs",
        buffers: [
          {
            arrayStride: 5 * 4, // 3 floats (position) + 2 floats (uv)
            attributes: [
              {
                shaderLocation: 0,
                offset: 0,
                format: "float32x3",
              },
              {
                shaderLocation: 1,
                offset: 3 * 4,
                format: "float32x2",
              },
            ],
          },
        ],
      },
      fragment: {
        module: shaderModule,
        entryPoint: "fs",
        targets: [
          {
            format: "bgra8unorm",
            blend: {
              color: {
                srcFactor: "src-alpha",
                dstFactor: "one-minus-src-alpha",
              },
              alpha: {
                srcFactor: "one",
                dstFactor: "one-minus-src-alpha",
              },
            },
          },
        ],
      },
      primitive: {
        topology: "triangle-list",
        cullMode: "none",
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less-equal",
        format: "depth24plus-stencil8",
      },
      multisample: {
        count: 4,
      },
    })
  }

  private createUniforms() {
    // Create uniform buffer
    // WGSL uniform buffers require 16-byte alignment:
    // time: f32 (4) + poolY (4) + padding (8) = 16 bytes
    // seaColor: vec3f (12) + padding (4) = 16 bytes
    // seaLight: vec3f (12) + padding (4) = 16 bytes
    // Total: 48 bytes
    this.uniformBuffer = this.device.createBuffer({
      label: "pool uniforms",
      size: 48,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    // Create bind group
    this.bindGroup = this.device.createBindGroup({
      label: "pool bind group",
      layout: this.bindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.uniformBuffer,
          },
        },
      ],
    })

    // Create camera bind group
    this.cameraBindGroup = this.device.createBindGroup({
      label: "pool camera bind group",
      layout: this.cameraBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.cameraUniformBuffer,
          },
        },
      ],
    })
  }

  public updateUniforms() {
    const time = (performance.now() - this.startTime) / 1000.0
    // WGSL uniform buffer layout (16-byte aligned):
    // offset 0: time (f32), poolY (f32), padding (8 bytes)
    // offset 16: seaColor (vec3f), padding (4 bytes)
    // offset 32: seaLight (vec3f), padding (4 bytes)
    const data = new Float32Array(12)
    data[0] = time
    data[1] = this.y
    // data[2-3] = padding (unused)
    data[4] = this.seaColor.x
    data[5] = this.seaColor.y
    data[6] = this.seaColor.z
    // data[7] = padding (unused)
    data[8] = this.seaLight.x
    data[9] = this.seaLight.y
    data[10] = this.seaLight.z
    // data[11] = padding (unused)

    this.device.queue.writeBuffer(this.uniformBuffer, 0, data)
  }

  public render(
    pass: GPURenderPassEncoder,
    restoreBuffers?: {
      vertexBuffer: GPUBuffer
      jointsBuffer: GPUBuffer
      weightsBuffer: GPUBuffer
      indexBuffer: GPUBuffer
    }
  ) {
    this.updateUniforms()

    // Set pool's pipeline and bind groups FIRST
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.cameraBindGroup)
    pass.setBindGroup(1, this.bindGroup)

    // IMPORTANT: Set pool's own buffers AFTER setting pipeline
    // Pool only needs vertex buffer 0 (position + UV), but we must keep buffers 1 and 2 set
    // for subsequent model rendering (eyes, hair, etc.)
    pass.setVertexBuffer(0, this.vertexBuffer)
    // Explicitly keep model's buffers 1 and 2 set - pool pipeline doesn't use them but they must stay
    if (restoreBuffers) {
      pass.setVertexBuffer(1, restoreBuffers.jointsBuffer)
      pass.setVertexBuffer(2, restoreBuffers.weightsBuffer)
    }

    // Set pool's index buffer - this MUST be set to override model's index buffer
    pass.setIndexBuffer(this.indexBuffer, "uint32")

    // Draw all pool indices starting from 0
    // Parameters: indexCount, instanceCount, firstIndex, baseVertex, firstInstance
    // We always draw from index 0 with all indices
    pass.drawIndexed(this.indexCount, 1, 0, 0, 0)

    // Restore model's buffers for subsequent rendering (eyes, hair, etc.)
    // This ensures vertex buffer 0 and index buffer are restored to model's buffers
    if (restoreBuffers) {
      pass.setVertexBuffer(0, restoreBuffers.vertexBuffer)
      pass.setVertexBuffer(1, restoreBuffers.jointsBuffer)
      pass.setVertexBuffer(2, restoreBuffers.weightsBuffer)
      pass.setIndexBuffer(restoreBuffers.indexBuffer, "uint32")
    }
  }

  public dispose() {
    // Buffers will be cleaned up automatically when device is lost
    // But we could explicitly destroy them if needed
  }
}
