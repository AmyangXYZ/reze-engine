// One-shot GPU particle emissions (model dissolve/materialize, hit sparks…).
// Stateless by construction: emit() uploads static per-particle attributes
// once, the vertex shader evaluates each particle's whole flight from the
// elapsed time, and the emission frees itself when it expires. Zero per-frame
// CPU work beyond a single time-uniform write shared by every emission.

import { Vec3 } from "./math"
import { PARTICLES_SHADER_WGSL } from "./shaders/passes/particles"

export interface ParticleEmitOptions {
  /** World-space anchor points, xyz-packed (see Model.sampleSurfacePoints).
   *  One particle spawns per point. */
  points: Float32Array
  /** "burst" flies point → scattered and fades (dissolve); "converge" flies
   *  scattered → point and vanishes on arrival (materialize). Default burst. */
  mode?: "burst" | "converge"
  /** Seconds the full effect lasts, staggers included (default 1.6). */
  duration?: number
  /** HDR color — components above 1 glow through bloom (default warm gold). */
  color?: Vec3
  /** Particle radius in world units (default 0.5). */
  size?: number
  /** Scatter distance in world units at the far end of the flight (default 14). */
  scatter?: number
  /** Upward drift across the flight, world units (default 6). */
  lift?: number
  /** Radians the cloud winds around its centroid across the flight (default 2). */
  swirl?: number
}

interface Emission {
  particleBuffer: GPUBuffer
  uniformBuffer: GPUBuffer
  bindGroup: GPUBindGroup
  count: number
  endTime: number
}

/** Floats per particle: point.xyz, scatter.xyz, rand(delay, lifeScale, seed, sizeScale). */
const PARTICLE_STRIDE = 10

export class ParticleSystem {
  private readonly device: GPUDevice
  private readonly pipeline: GPURenderPipeline
  private readonly bindGroupLayout: GPUBindGroupLayout
  private readonly cameraBuffer: GPUBuffer
  private readonly timeBuffer: GPUBuffer
  private readonly timeData = new Float32Array(1)
  private emissions: Emission[] = []

  constructor(
    device: GPUDevice,
    cameraBuffer: GPUBuffer,
    formats: { hdr: GPUTextureFormat; mask: GPUTextureFormat; depth: GPUTextureFormat; sampleCount: number }
  ) {
    this.device = device
    this.cameraBuffer = cameraBuffer
    this.timeBuffer = device.createBuffer({
      label: "particle time",
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.bindGroupLayout = device.createBindGroupLayout({
      label: "particle bind group layout",
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
        { binding: 1, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } },
        { binding: 2, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
      ],
    })
    const module = device.createShaderModule({ label: "particles", code: PARTICLES_SHADER_WGSL })
    const additive: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one" },
      alpha: { srcFactor: "one", dstFactor: "one" },
    }
    this.pipeline = device.createRenderPipeline({
      label: "particle pipeline",
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.bindGroupLayout] }),
      vertex: {
        module,
        entryPoint: "vs_main",
        buffers: [
          {
            arrayStride: PARTICLE_STRIDE * 4,
            stepMode: "instance",
            attributes: [
              { shaderLocation: 0, offset: 0, format: "float32x3" },
              { shaderLocation: 1, offset: 12, format: "float32x3" },
              { shaderLocation: 2, offset: 24, format: "float32x4" },
            ],
          },
        ],
      },
      fragment: {
        module,
        entryPoint: "fs_main",
        targets: [
          { format: formats.hdr, blend: additive },
          { format: formats.mask, blend: additive },
        ],
      },
      primitive: { topology: "triangle-list" },
      // Occluded by the scene, never occluding: glitter reads as light, and
      // light writing depth would punch holes into the transparent phase.
      depthStencil: { format: formats.depth, depthWriteEnabled: false, depthCompare: "less-equal" },
      multisample: { count: formats.sampleCount },
    })
  }

  /** Upload one emission. `now` is the shared clock (seconds) draw() receives. */
  emit(opts: ParticleEmitOptions, now: number): void {
    const count = Math.floor(opts.points.length / 3)
    if (count === 0) return
    const mode = opts.mode === "converge" ? 1 : 0
    const duration = opts.duration ?? 1.6
    const color = opts.color ?? new Vec3(4.0, 3.0, 1.6)
    const size = opts.size ?? 0.5
    const scatter = opts.scatter ?? 14
    const lift = opts.lift ?? 6
    const swirl = opts.swirl ?? 2

    // Delays stagger the front so the cloud dissolves as a sweep, and each
    // particle's life fits delay + flight inside the emission's duration.
    const data = new Float32Array(count * PARTICLE_STRIDE)
    let cx = 0
    let cy = 0
    let cz = 0
    for (let i = 0; i < count; i++) {
      const o = i * PARTICLE_STRIDE
      const px = opts.points[i * 3]
      const py = opts.points[i * 3 + 1]
      const pz = opts.points[i * 3 + 2]
      cx += px
      cy += py
      cz += pz
      data[o] = px
      data[o + 1] = py
      data[o + 2] = pz
      // Random direction on the sphere, mildly flattened so the cloud spreads
      // outward more than it dives into the floor.
      const theta = Math.random() * Math.PI * 2
      const zr = Math.random() * 2 - 1
      const r = Math.sqrt(Math.max(0, 1 - zr * zr))
      const dist = scatter * (0.35 + 0.65 * Math.random())
      data[o + 3] = Math.cos(theta) * r * dist
      data[o + 4] = zr * 0.5 * dist
      data[o + 5] = Math.sin(theta) * r * dist
      const delay = Math.random() * duration * 0.3
      data[o + 6] = delay
      data[o + 7] = (duration - delay) / duration // lifeScale: everyone lands inside duration
      data[o + 8] = Math.random()
      data[o + 9] = 0.5 + Math.random()
    }

    const particleBuffer = this.device.createBuffer({
      label: "particle emission",
      size: data.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(particleBuffer, 0, data)

    const uni = new Float32Array(12)
    uni[0] = color.x
    uni[1] = color.y
    uni[2] = color.z
    uni[3] = size
    uni[4] = cx / count
    uni[5] = cy / count
    uni[6] = cz / count
    uni[7] = mode
    uni[8] = now
    uni[9] = duration
    uni[10] = lift
    uni[11] = swirl
    const uniformBuffer = this.device.createBuffer({
      label: "particle emission uniforms",
      size: uni.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(uniformBuffer, 0, uni)

    const bindGroup = this.device.createBindGroup({
      label: "particle emission bind group",
      layout: this.bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: this.cameraBuffer } },
        { binding: 1, resource: { buffer: uniformBuffer } },
        { binding: 2, resource: { buffer: this.timeBuffer } },
      ],
    })

    this.emissions.push({ particleBuffer, uniformBuffer, bindGroup, count, endTime: now + duration + 0.1 })
  }

  get active(): boolean {
    return this.emissions.length > 0
  }

  /** Record draws into the main pass (after the transparent phase). */
  draw(pass: GPURenderPassEncoder, now: number): void {
    if (this.emissions.length === 0) return
    const live: Emission[] = []
    let bound = false
    for (const e of this.emissions) {
      if (now >= e.endTime) {
        e.particleBuffer.destroy()
        e.uniformBuffer.destroy()
        continue
      }
      if (!bound) {
        this.timeData[0] = now
        this.device.queue.writeBuffer(this.timeBuffer, 0, this.timeData)
        pass.setPipeline(this.pipeline)
        bound = true
      }
      pass.setBindGroup(0, e.bindGroup)
      pass.setVertexBuffer(0, e.particleBuffer)
      pass.draw(6, e.count)
      live.push(e)
    }
    this.emissions = live
  }

  dispose(): void {
    for (const e of this.emissions) {
      e.particleBuffer.destroy()
      e.uniformBuffer.destroy()
    }
    this.emissions = []
    this.timeBuffer.destroy()
  }
}
