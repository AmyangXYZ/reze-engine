// Native looks: a model dressed in the game's own materials.
//
// A look names translated shader variants (host.ts), the shared images they
// sample (ramps, matcaps), and per material the passes it draws and the values
// it carries — what a game character's material held, with the model's own
// texture standing in for its _MainTex. Optionally a CharacterRig, which makes
// the model a character in the game's sense (character.ts).
//
// The frame, as the game's CharacterFeature lays it out after the opaques:
//   1. every native opaque material's passes (Outline, ForwardBase), in queue
//      order — after the engine's own opaque bundle, so both share one depth;
//   2. CharHairShadow, CharFaceShadow, CharHairTransE, each over every
//      material that has one (the hair's shadow on the face, by stencil);
//   3. the eyes' see-through-hair passes, Override1 for all, then 2, then 3;
// and the transparent queue in the engine's transparent phase. Shadows are the
// engine's: a native material still casts through its pass.

import { NativeHost, sameValue, type NativeShader, type NativeStream, type NativeValue, type ValueSource } from "./host"
import type { PassState } from "./state"
import { UNITY_SKIN_WGSL, tangentsFor } from "./skin"
import { characterBlock, type CharacterRig } from "./character"
import { engineToGame, gameToEngine } from "./globals"

export type NativeTexture = {
  width: number
  height: number
  /** RGBA8, rows top first. */
  data: Uint8Array
  /** Colour (sRGB) or data (linear) — how the shader expects to read it. */
  srgb: boolean
  /** Unity's wrap mode as the game imported it (Repeat, Clamp, Mirror,
   *  MirrorOnce); repeating when absent. A clamped mask sampled past its edge
   *  reads its border, not the far side. */
  wrap?: string
  /** Unity's filter mode (Point, Bilinear, Trilinear); bilinear when absent. */
  filter?: string
}

const WRAP: Record<string, GPUAddressMode> = {
  Repeat: "repeat",
  Clamp: "clamp-to-edge",
  Mirror: "mirror-repeat",
  MirrorOnce: "mirror-repeat",
}

export type NativeMaterialSpec = {
  /** The model's materials this dresses, by name. */
  materials: string[]
  /** Unity render queue: above 2500 draws in the transparent phase. */
  queue: number
  passes: { lightMode: string; shader: string; state: PassState }[]
  /** Material properties, by name (colours linear, as Unity uploads them). */
  values: Record<string, NativeValue>
  /** Texture slots: a key into the look's images, or "@diffuse" for the
   *  material's own texture. */
  textures: Record<string, string>
}

export type NativeLook = {
  shaders: NativeShader[]
  textures: Record<string, NativeTexture>
  materials: NativeMaterialSpec[]
  rig?: CharacterRig
  /** The game vertex streams the model's PMX carries, by semantic: the
   *  additional UV channel (1–4) that holds each, four floats a vertex, keyed
   *  over time by that channel's UV morphs (PMX types 4–7) — a particle's
   *  colour over its life on a mesh's own vertex colours. A stream named here
   *  reads that channel; any other reads as Unity binds a missing one. */
  streams?: Record<string, number>
}

/** What the engine lends for one model. */
export type NativeModel = {
  name: string
  vertices: Float32Array
  indices: Uint32Array
  vertexBuffer: GPUBuffer
  jointsBuffer: GPUBuffer
  weightsBuffer: GPUBuffer
  skinMatrixBuffer: GPUBuffer
  indexBuffer: GPUBuffer
  draws: {
    materialName: string
    firstIndex: number
    count: number
    diffuse: GPUTextureView
  }[]
  hidden: (material: string) => boolean
  /** The head bone's current skinning matrix, or null. */
  headSkin: () => ArrayLike<number> | null
  headRest: [number, number, number]
  /** Its rendering-layer bits, for a look without a rig. */
  layers: number
  /** A PMX additional UV channel (1–4), four floats a vertex, or null. */
  additionalUv: (channel: number) => Float32Array | null
  /** Its morphs, in the order of morphWeights. */
  morphs: readonly { type: number; uvOffsets?: { vertexIndex: number; offset: [number, number, number, number] }[] }[]
  /** The current effective (group-resolved, clamped) morph weights. */
  morphWeights: () => Float32Array
}

/** A vertex stream out of an additional UV channel: its rest values, the
 *  channel's morphs, and the weights it was last built at. */
type CarriedStream = {
  buffer: GPUBuffer
  base: Float32Array
  data: Float32Array
  morphs: { index: number; offsets: { vertexIndex: number; offset: [number, number, number, number] }[] }[]
  weights: Float32Array
}

type Install = {
  model: NativeModel
  look: NativeLook
  views: Map<string, GPUTextureView>
  /** Per look texture: its sampler, as the game imported the texture. */
  samplers: Map<string, GPUSampler>
  textures: GPUTexture[]
  pos: GPUBuffer
  nrm: GPUBuffer
  tan: GPUBuffer
  restTan: GPUBuffer
  params: GPUBuffer
  skinBind: GPUBindGroup
  /** Per draw of the model: the spec that dresses it, and the values set on
   *  that material since (setUniforms), over the spec's. */
  dressed: { draw: NativeModel["draws"][number]; spec: NativeMaterialSpec; values: Record<string, NativeValue> }[]
  block: Record<string, NativeValue>
  /** The look's streams (NativeLook.streams), by semantic. */
  carried: Map<string, CarriedStream>
}

const OPAQUE_MODES = ["ALWAYS", "FORWARDBASE"]
const CHARACTER_MODES = ["CHARHAIRSHADOW", "CHARFACESHADOW", "CHARHAIRTRANSE"]
const OVERRIDE_MODES = ["OVERRIDE1", "OVERRIDE2", "OVERRIDE3"]

export class NativeLooks {
  readonly host: NativeHost
  private installs = new Map<string, Install>()
  /** Per model: its passes' pipelines compiling (see install). */
  private warming = new Map<string, Promise<void>>()
  private skinPipeline: GPUComputePipeline
  private zero: GPUBuffer
  private white: GPUBuffer
  private globals: Record<string, NativeValue> = {}
  private frameTextures: Record<string, GPUTextureView | undefined> = {}
  private scale = 8

  constructor(
    private device: GPUDevice,
    host: NativeHost,
  ) {
    this.host = host
    this.skinPipeline = device.createComputePipeline({
      label: "native skinning",
      layout: "auto",
      compute: {
        module: device.createShaderModule({
          label: "native skinning",
          code: UNITY_SKIN_WGSL,
        }),
        entryPoint: "main",
      },
    })
    this.zero = device.createBuffer({
      label: "native zero stream",
      size: 64,
      usage: GPUBufferUsage.VERTEX,
    })
    this.white = device.createBuffer({
      label: "native white stream",
      size: 64,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    })
    new Float32Array(this.white.getMappedRange()).fill(1)
    this.white.unmap()
  }

  get size(): number {
    return this.installs.size
  }

  private samplerCache = new Map<string, GPUSampler>()

  /** A texture's sampler: its wrap and filter, shared by every texture alike. */
  private sampler(t: NativeTexture): GPUSampler {
    const wrap = WRAP[t.wrap ?? "Repeat"] ?? "repeat"
    const filter: GPUFilterMode = t.filter === "Point" ? "nearest" : "linear"
    const key = `${wrap}|${filter}`
    let s = this.samplerCache.get(key)
    if (!s) {
      s = this.device.createSampler({
        label: `native look sampler ${key}`,
        addressModeU: wrap,
        addressModeV: wrap,
        magFilter: filter,
        minFilter: filter,
        mipmapFilter: filter,
      })
      this.samplerCache.set(key, s)
    }
    return s
  }

  has(model: string): boolean {
    return this.installs.has(model)
  }

  /** Resolves once every pass of the model's look has its pipeline — until
   *  then those draws sit out (NativeHost never compiles at a draw). */
  ready(model: string): Promise<void> {
    return this.warming.get(model) ?? Promise.resolve()
  }

  /** Is this model's material drawn by its native look? */
  dresses(model: string, material: string): boolean {
    const i = this.installs.get(model)
    return !!i && i.dressed.some((d) => d.draw.materialName === material)
  }

  install(model: NativeModel, look: NativeLook): void {
    this.remove(model.name)
    const d = this.device
    for (const s of look.shaders) this.host.register(s)
    const views = new Map<string, GPUTextureView>()
    const samplers = new Map<string, GPUSampler>()
    const textures: GPUTexture[] = []
    for (const [key, t] of Object.entries(look.textures)) {
      if (t.wrap || t.filter) samplers.set(key, this.sampler(t))
      const tex = d.createTexture({
        label: `${model.name}: ${key}`,
        size: [t.width, t.height],
        format: t.srgb ? "rgba8unorm-srgb" : "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      d.queue.writeTexture({ texture: tex }, t.data as Uint8Array<ArrayBuffer>, { bytesPerRow: t.width * 4 }, [t.width, t.height])
      textures.push(tex)
      views.set(key, tex.createView())
    }
    const n = model.vertices.length / 8
    const buf = (label: string, floats: number, usage: number) =>
      d.createBuffer({
        label: `${model.name}: ${label}`,
        size: Math.max(16, floats * 4),
        usage,
      })
    const out = GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX
    const pos = buf("native position", n * 3, out)
    const nrm = buf("native normal", n * 3, out)
    const tan = buf("native tangent", n * 4, out)
    const restTan = buf("rest tangent", n * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST)
    d.queue.writeBuffer(restTan, 0, tangentsFor(model.vertices, model.indices) as Float32Array<ArrayBuffer>)
    const params = d.createBuffer({
      label: `${model.name}: skin params`,
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    d.queue.writeBuffer(params, 0, new Uint32Array([n, 0, 0, 0]))
    const skinBind = d.createBindGroup({
      label: `${model.name}: native skinning`,
      layout: this.skinPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: params } },
        { binding: 1, resource: { buffer: model.vertexBuffer } },
        { binding: 2, resource: { buffer: model.jointsBuffer } },
        { binding: 3, resource: { buffer: model.weightsBuffer } },
        { binding: 4, resource: { buffer: model.skinMatrixBuffer } },
        { binding: 5, resource: { buffer: restTan } },
        { binding: 6, resource: { buffer: pos } },
        { binding: 7, resource: { buffer: nrm } },
        { binding: 8, resource: { buffer: tan } },
      ],
    })
    const dressed: Install["dressed"] = []
    for (const draw of model.draws) {
      const spec = look.materials.find((m) => m.materials.includes(draw.materialName))
      if (spec) dressed.push({ draw, spec, values: {} })
    }
    const carried = new Map<string, CarriedStream>()
    for (const [semantic, channel] of Object.entries(look.streams ?? {})) {
      const base = model.additionalUv(channel)
      if (!base || base.length < n * 4) continue
      const morphs = model.morphs
        .map((m, index) => ({ index, type: m.type, offsets: m.uvOffsets ?? [] }))
        .filter((m) => m.type === 3 + channel && m.offsets.length)
        .map(({ index, offsets }) => ({ index, offsets }))
      const buffer = buf(`stream ${semantic}`, n * 4, GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST)
      const data = new Float32Array(base.subarray(0, n * 4))
      d.queue.writeBuffer(buffer, 0, data)
      // NaN: the first prepare builds the stream whatever the weights are
      carried.set(semantic, { buffer, base: data.slice(), data, morphs, weights: new Float32Array(morphs.length).fill(NaN) })
    }
    const install: Install = {
      model,
      look,
      views,
      samplers,
      textures,
      pos,
      nrm,
      tan,
      restTan,
      params,
      skinBind,
      dressed,
      block: {},
      carried,
    }
    this.installs.set(model.name, install)
    // Every pass this look will draw, compiled now and side by side — the
    // shader, its state and the streams are all known here.
    const drawn = [...OPAQUE_MODES, ...CHARACTER_MODES, ...OVERRIDE_MODES]
    const jobs: Promise<void>[] = []
    for (const d of dressed)
      for (const p of d.spec.passes) if (drawn.includes(p.lightMode)) jobs.push(this.host.warm(p.shader, p.state, this.streams(install, p.shader)))
    this.warming.set(
      model.name,
      Promise.all(jobs).then(() => {}),
    )
  }

  remove(name: string): void {
    const i = this.installs.get(name)
    if (!i) return
    this.installs.delete(name)
    this.host.release(`${name}|`)
    const bufs = [i.pos, i.nrm, i.tan, i.restTan, i.params, ...[...i.carried.values()].map((c) => c.buffer)]
    const texs = i.textures
    // Freed once the GPU is done with the frames that may still name them.
    void this.device.queue.onSubmittedWorkDone().then(() => {
      for (const b of bufs) b.destroy()
      for (const t of texs) t.destroy()
    })
  }

  /**
   * Before the scene pass: skin every dressed model and compute each
   * character's block; the frame's Unity globals come computed (engine.ts).
   */
  prepare(
    encoder: GPUCommandEncoder,
    globals: Record<string, NativeValue>,
    scale: number,
    textures: Record<string, GPUTextureView | undefined>,
  ): void {
    if (!this.installs.size) return
    this.scale = scale
    this.globals = globals
    this.frameTextures = textures
    const pass = encoder.beginComputePass({ label: "native skinning" })
    pass.setPipeline(this.skinPipeline)
    for (const i of this.installs.values()) {
      pass.setBindGroup(0, i.skinBind)
      pass.dispatchWorkgroups(Math.ceil(i.model.vertices.length / 8 / 64))
      i.block = i.look.rig
        ? characterBlock(i.look.rig, i.model.headSkin(), i.model.headRest, scale)
        : {
            unity_RenderingLayer: new Uint32Array([i.model.layers >>> 0, 0, 0, 0]),
          }
      if (i.carried.size) this.keyStreams(i)
    }
    pass.end()
  }

  /** Each carried stream at the model's current morph weights: its rest values
   *  plus every one of its channel's morphs at its weight, rebuilt and uploaded
   *  only when one of those weights moved. */
  private keyStreams(i: Install): void {
    const weights = i.model.morphWeights()
    for (const c of i.carried.values()) {
      if (!c.morphs.length) continue
      let moved = false
      for (let k = 0; k < c.morphs.length; k++) {
        const w = weights[c.morphs[k].index] ?? 0
        if (w !== c.weights[k]) {
          c.weights[k] = w
          moved = true
        }
      }
      if (!moved) continue
      c.data.set(c.base)
      for (let k = 0; k < c.morphs.length; k++) {
        const w = c.weights[k]
        if (!w) continue
        for (const o of c.morphs[k].offsets) {
          const at = o.vertexIndex * 4
          if (at + 3 >= c.data.length) continue
          c.data[at] += o.offset[0] * w
          c.data[at + 1] += o.offset[1] * w
          c.data[at + 2] += o.offset[2] * w
          c.data[at + 3] += o.offset[3] * w
        }
      }
      this.device.queue.writeBuffer(c.buffer, 0, c.data as Float32Array<ArrayBuffer>)
    }
  }

  /**
   * Set some of one dressed material's values (its Unity material properties,
   * as the look's `values` hold them) — what a game animates on a material
   * over a take. Each value stands until set again; the look's own stand
   * under it. Only names the material's spec carries are taken: they are the
   * ones its draws' uniform blocks read per draw. A value equal to the one
   * already set keeps that one's object, so the host, which refills a
   * uniform member only when its value object changes, writes nothing for
   * it. Returns whether the model wears a look that dresses this material.
   */
  setUniforms(model: string, material: string, values: Record<string, NativeValue>): boolean {
    const i = this.installs.get(model)
    if (!i) return false
    let found = false
    for (const d of i.dressed) {
      if (d.draw.materialName !== material) continue
      found = true
      for (const name in values) {
        const v = values[name]
        if (v === undefined || v === null || !(name in d.spec.values)) continue
        const was = d.values[name]
        if (was === undefined || !sameValue(was, v)) d.values[name] = v
      }
    }
    return found
  }

  private streams(i: Install, shader: string): NativeStream[] {
    return this.host.inputs(shader).map(({ semantic }) => {
      const carried = i.carried.get(semantic)
      if (carried)
        return {
          buffer: carried.buffer,
          offset: 0,
          stride: 16,
          format: "float32x4" as const,
        }
      if (semantic === "POSITION")
        return {
          buffer: i.pos,
          offset: 0,
          stride: 12,
          format: "float32x3" as const,
        }
      if (semantic === "NORMAL")
        return {
          buffer: i.nrm,
          offset: 0,
          stride: 12,
          format: "float32x3" as const,
        }
      if (semantic === "TANGENT")
        return {
          buffer: i.tan,
          offset: 0,
          stride: 16,
          format: "float32x4" as const,
        }
      if (semantic === "TEXCOORD0")
        return {
          buffer: i.model.vertexBuffer,
          offset: 24,
          stride: 32,
          format: "float32x2" as const,
        }
      // A missing vertex colour reads white, as Unity binds it; any other
      // missing stream reads zero.
      if (semantic === "COLOR")
        return {
          buffer: this.white,
          offset: 0,
          stride: 0,
          format: "float32x4" as const,
        }
      return {
        buffer: this.zero,
        offset: 0,
        stride: 0,
        format: "float32x4" as const,
      }
    })
  }

  private drawPass(pass: GPURenderPassEncoder, i: Install, d: Install["dressed"][number], p: NativeMaterialSpec["passes"][number]): void {
    if (i.model.hidden(d.draw.materialName)) return
    const s = this.scale
    const perDraw: ValueSource = {
      // The skinned streams are in engine units; the game's world is 1/scale of them.
      unity_ObjectToWorld: engineToGame(s),
      unity_WorldToObject: gameToEngine(s),
      unity_WorldTransformParams: [0, 0, 0, 1],
      unity_LightData: [0, 0, 0, 0],
      unity_LightIndices: [
        [0, 0, 0, 0],
        [0, 0, 0, 0],
      ],
      unity_LODFade: [1, 1, 0, 0],
    }
    const textures: Record<string, GPUTextureView | undefined> = {
      ...this.frameTextures,
    }
    const samplers: Record<string, GPUSampler | undefined> = {}
    for (const [slot, key] of Object.entries(d.spec.textures)) {
      textures[slot] = key === "@diffuse" ? d.draw.diffuse : i.views.get(key)
      samplers[slot] = i.samplers.get(key)
    }
    this.host.draw(pass, {
      shader: p.shader,
      state: p.state,
      streams: this.streams(i, p.shader),
      index: {
        buffer: i.model.indexBuffer,
        format: "uint32",
        offset: d.draw.firstIndex * 4,
        count: d.draw.count,
      },
      sources: [perDraw, i.block, d.values, d.spec.values, this.globals],
      textures,
      samplers,
      key: `${i.model.name}|${d.draw.materialName}|${p.lightMode}|${p.shader}`,
      pass: "main",
    })
  }

  /** Every dressed material whose queue is in range, in queue order. */
  private each(opaque: boolean): { i: Install; d: Install["dressed"][number] }[] {
    const out: { i: Install; d: Install["dressed"][number] }[] = []
    for (const i of this.installs.values()) for (const d of i.dressed) if (d.spec.queue <= 2500 === opaque) out.push({ i, d })
    return out.sort((a, b) => a.d.spec.queue - b.d.spec.queue)
  }

  /** After the engine's opaque phase: the opaques, then the character passes. */
  drawOpaque(pass: GPURenderPassEncoder): void {
    if (!this.installs.size) return
    const list = this.each(true)
    for (const { i, d } of list) for (const p of d.spec.passes) if (OPAQUE_MODES.includes(p.lightMode)) this.drawPass(pass, i, d, p)
    for (const mode of [...CHARACTER_MODES, ...OVERRIDE_MODES])
      for (const { i, d } of list) for (const p of d.spec.passes) if (p.lightMode === mode) this.drawPass(pass, i, d, p)
  }

  /** In the engine's transparent phase. */
  drawTransparent(pass: GPURenderPassEncoder): void {
    if (!this.installs.size) return
    for (const { i, d } of this.each(false))
      for (const p of d.spec.passes) if (OPAQUE_MODES.includes(p.lightMode)) this.drawPass(pass, i, d, p)
  }
}
