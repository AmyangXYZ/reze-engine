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

import { NativeHost, type NativeShader, type NativeStream, type NativeValue, type ValueSource } from "./host"
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
}

type Install = {
  model: NativeModel
  look: NativeLook
  views: Map<string, GPUTextureView>
  textures: GPUTexture[]
  pos: GPUBuffer
  nrm: GPUBuffer
  tan: GPUBuffer
  restTan: GPUBuffer
  params: GPUBuffer
  skinBind: GPUBindGroup
  /** Per draw of the model: the spec that dresses it, if any. */
  dressed: { draw: NativeModel["draws"][number]; spec: NativeMaterialSpec }[]
  block: Record<string, NativeValue>
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
    const textures: GPUTexture[] = []
    for (const [key, t] of Object.entries(look.textures)) {
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
      if (spec) dressed.push({ draw, spec })
    }
    const install: Install = {
      model,
      look,
      views,
      textures,
      pos,
      nrm,
      tan,
      restTan,
      params,
      skinBind,
      dressed,
      block: {},
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
    const bufs = [i.pos, i.nrm, i.tan, i.restTan, i.params]
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
    }
    pass.end()
  }

  private streams(i: Install, shader: string): NativeStream[] {
    return this.host.inputs(shader).map(({ semantic }) => {
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
    for (const [slot, key] of Object.entries(d.spec.textures)) textures[slot] = key === "@diffuse" ? d.draw.diffuse : i.views.get(key)
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
      sources: [perDraw, i.block, d.spec.values, this.globals],
      textures,
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
