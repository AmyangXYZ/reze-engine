// A game stage drawn by its own shaders: the package ag-rip's stage_native.py
// writes from a recording of the game's scene — its meshes, textures and
// materials as the game holds them, every renderer at its place, and the
// translated shader variants each material's passes use.
//
// This is the compare page's renderer (agtools/web/agplay) inside the engine,
// with one difference that matters: nothing here is recorded per frame. The
// camera, the clock, the sun and lamps and the cascades are the engine's, turned
// into the game's globals by globals.ts; the stage brings only what is its own —
// geometry, materials, and its scene settings (fog, tint, ambient, environment
// cube), which it hands the engine to set for every native draw.
//
// SPACE. Meshes and renderer matrices stay in the game's world, as recorded:
// the globals carry the engine's camera into it (globals.ts, gameToEngine), so
// a native draw lands where the game drew it, as the MMD export places it.
//
// SHADOWS. Every caster draws its material's own SHADOWCASTER pass into the
// engine's cascade tiles, as the game drew its own atlas — alpha-tested leaves
// cast their leaves, not their quads.

import { NativeHost, type NativeShader, type NativeStream, type NativeValue, type ValueSource } from "./host"
import type { PassState } from "./state"
import { gameToEngine, invert4, mul4, scale4, flipY } from "./globals"
import { SHADOW_CASCADES } from "../shadow-cascades"

export type NativeStageMesh = {
  id: string
  vertexCount: number
  /** Semantic -> [byte offset, components] (float32). */
  streams: Record<string, [number, number]>
  submeshes: { offset: number; count: number; topology?: string }[]
  bounds: { center: [number, number, number]; size: [number, number, number] }
}

export type NativeStageTexture = {
  id: string
  name: string
  width: number
  height: number
  hdr: number
  srgb: number
  cube: number
  wrap?: string
  filter?: string
  mips?: number
  mipsExported?: number
  aniso?: number
  files: string[]
}

export type NativeStageMaterial = {
  id: string
  name: string
  shader: string
  queue: number
  passes: { lightMode: string; shader: string; state: PassState }[]
  values: Record<string, NativeValue>
  textures: Record<string, string>
  /** An empty slot's stand-in, per the shader: white, black, grey, bump. */
  defaults?: Record<string, string>
}

export type NativeStageRenderer = {
  id: string
  path: string
  mesh: string
  materials: (string | null)[]
  /** Object to world, game units, column-major. */
  matrix: number[]
  layers: number
  castShadows: number
  receiveShadows: number
  values: Record<string, NativeValue>
  textures: Record<string, string>
}

export type NativeStageLights = {
  main: {
    direction: [number, number, number]
    color: [number, number, number]
    shadow: number
    /** As the game's pipeline packed it, recorded: _MainLightColor and
     *  SimMainLightColor (linear, intensity in w), SimMainLightColorNoInt. */
    unity?: { color: number[]; simColor: number[]; simColorNoInt: number[] }
    /** The Light's shadowBias and shadowNormalBias, in shadow-map texels. */
    shadowBias?: number
    shadowNormalBias?: number
  }
  additional: {
    position: [number, number, number]
    range: number
    color: [number, number, number]
    type: "point" | "spot"
    aim?: [number, number, number]
    cosOuter?: number
    cosInner?: number
    unity?: {
      spotDir: number[]
      colorW: number
      extra: number[]
      worldToLight: number[] | null
      lightType: number
      simSpotDir?: number[] | null
      simColorW?: number
    }
  }[]
}

/** stage.json, as stage_native.py writes it. */
export type NativeStagePackage = {
  version: number
  kind: "unity-native-stage"
  name: string
  source?: string
  /** Engine units per game unit. */
  scale: number
  keywords: string[]
  shaders: string[]
  meshes: NativeStageMesh[]
  textures: NativeStageTexture[]
  materials: NativeStageMaterial[]
  renderers: NativeStageRenderer[]
  lights: NativeStageLights
  settings: Record<string, NativeValue>
  globalTextures: Record<string, string>
}

/** Reads one file of the package by its path inside it (stage.json's folder). */
export type NativeStageReader = (path: string) => Promise<ArrayBuffer>

type Tex = { texture: GPUTexture; view: GPUTextureView; sampler: GPUSampler }

type Item = {
  renderer: { id: string; castShadows: number }
  material: NativeStageMaterial
  sub: number
  mesh: { info: NativeStageMesh; buffer: GPUBuffer }
  pass: NativeStageMaterial["passes"][number]
  perDraw: ValueSource
  textures: Record<string, GPUTextureView | undefined>
  samplers: Record<string, GPUSampler | undefined>
  /** World-space centre, engine units — for sorting. */
  center: [number, number, number]
  key: string
}

const MAIN_MODES = ["ALWAYS", "FORWARDBASE"]

export class NativeStage {
  readonly name: string
  readonly pkg: NativeStagePackage
  /** Engine-unit box around everything that casts, for the cascades. */
  readonly bounds: {
    min: [number, number, number]
    max: [number, number, number]
  }
  /**
   * How far the stage reaches from the origin, in engine units: every
   * renderer's box, casters or not. A game's sky is a dome in the stage
   * (X317's reaches 1900 units out), and the camera's far plane has to reach it.
   */
  get extent(): number {
    return this.reach
  }
  private reach = 0
  private buffers: GPUBuffer[] = []
  private textures: GPUTexture[] = []
  private opaque: Item[] = []
  private transparent: Item[] = []
  private casters: Item[] = []
  private zero: GPUBuffer
  private white: GPUBuffer

  private constructor(
    private device: GPUDevice,
    pkg: NativeStagePackage,
  ) {
    this.name = pkg.name
    this.pkg = pkg
    this.bounds = {
      min: [Infinity, Infinity, Infinity],
      max: [-Infinity, -Infinity, -Infinity],
    }
    this.zero = device.createBuffer({
      label: "native stage: zero stream",
      size: 64,
      usage: GPUBufferUsage.VERTEX,
    })
    this.white = device.createBuffer({
      label: "native stage: white stream",
      size: 64,
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    })
    new Float32Array(this.white.getMappedRange()).fill(1)
    this.white.unmap()
    this.buffers.push(this.zero, this.white)
  }

  /**
   * Load a package: every mesh and texture it names through `read`, every
   * shader registered with the hosts (the scene pass's and the shadow atlas's).
   */
  static async load(
    device: GPUDevice,
    pkg: NativeStagePackage,
    read: NativeStageReader,
    hosts: { scene: NativeHost; shadow: NativeHost },
    tools: {
      mipmaps: (t: GPUTexture, levels: number) => void
      fallback: Record<string, GPUTextureView>
    },
    onProgress?: (done: number, total: number) => void,
  ): Promise<NativeStage> {
    if (pkg.kind !== "unity-native-stage") throw new Error("not a native stage package")
    const st = new NativeStage(device, pkg)
    // Progress by weight, not by file: a 2048² picture is most of a second to
    // decode, a shader or a small mesh nothing — so pictures weigh their
    // megapixels, everything else one unit of a quarter of one.
    const texWeight = (t: NativeStageTexture) => Math.max(0.25, ((t.cube ? 6 : 1) * t.width * t.height) / (1 << 20))
    const total = 0.25 * (pkg.shaders.length + pkg.meshes.length) + pkg.textures.reduce((s, t) => s + texWeight(t), 0)
    let done = 0
    const tick = (w = 0.25) => onProgress?.((done += w), total)
    const text = async (p: string) => new TextDecoder().decode(await read(p))

    const shaders = new Map<string, NativeShader>()
    await Promise.all(
      pkg.shaders.map(async (n) => {
        const [vert, frag, info] = await Promise.all([
          text(`shaders/${n}.vert.wgsl`),
          text(`shaders/${n}.frag.wgsl`),
          text(`shaders/${n}.json`),
        ])
        shaders.set(n, { name: n, vert, frag, info: JSON.parse(info) })
        tick()
      }),
    )
    const casterShaders = new Set<string>()
    for (const m of pkg.materials) for (const p of m.passes) if (p.lightMode === "SHADOWCASTER") casterShaders.add(p.shader)
    for (const [n, s] of shaders) {
      if (casterShaders.has(n)) hosts.shadow.register(s)
      else hosts.scene.register(s)
    }

    const meshes = new Map<string, { info: NativeStageMesh; buffer: GPUBuffer }>()
    await Promise.all(
      pkg.meshes.map(async (m) => {
        const data = await read(`meshes/${m.id}.bin`)
        const buffer = device.createBuffer({
          label: `native stage mesh ${m.id}`,
          size: Math.max(16, Math.ceil(data.byteLength / 4) * 4),
          usage: GPUBufferUsage.VERTEX | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        })
        device.queue.writeBuffer(buffer, 0, data, 0, data.byteLength & ~3)
        st.buffers.push(buffer)
        meshes.set(m.id, { info: m, buffer })
        tick()
      }),
    )

    // Loaded means drawable: every pass's pipeline is compiled before this
    // resolves, rather than one at a time in the frames after it.
    const warming = st.warm(hosts, meshes)

    const textures = new Map<string, Tex>()
    await Promise.all(
      pkg.textures.map(async (t) => {
        try {
          textures.set(t.id, await st.loadTexture(t, read, tools.mipmaps))
        } catch (e) {
          console.warn(`native stage: texture ${t.id} ${t.name}:`, e)
        }
        tick(texWeight(t))
      }),
    )

    st.build(meshes, textures, tools.fallback)
    await warming
    return st
  }

  private async loadTexture(
    t: NativeStageTexture,
    read: NativeStageReader,
    mipmaps: (t: GPUTexture, levels: number) => void,
  ): Promise<Tex> {
    const d = this.device
    const full = Math.max(1, Math.floor(Math.log2(Math.max(t.width, t.height))) + 1)
    const format: GPUTextureFormat = t.hdr ? "rgba16float" : t.srgb ? "rgba8unorm-srgb" : "rgba8unorm"
    if (t.cube) {
      // Every face and mip as recorded: face-major <id>_f<face>_m<mip>.bin, RGBA16F
      // rows along WebGPU's face directions (AGDlcRecord's AGCubeFace).
      const mips = t.mipsExported || 1
      const texture = d.createTexture({
        label: `native stage ${t.name}`,
        size: [t.width, t.height, 6],
        format,
        mipLevelCount: mips,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
      })
      await Promise.all(
        t.files.map(async (file, k) => {
          const face = Math.floor(k / mips)
          const mip = k % mips
          const s = Math.max(1, t.width >> mip)
          const data = await read(`textures/${file}`)
          d.queue.writeTexture({ texture, origin: [0, 0, face], mipLevel: mip }, data, { bytesPerRow: s * 8, rowsPerImage: s }, [s, s, 1])
        }),
      )
      this.textures.push(texture)
      const sampler = d.createSampler({
        magFilter: "linear",
        minFilter: "linear",
        mipmapFilter: "linear",
      })
      return {
        texture,
        view: texture.createView({ dimension: "cube" }),
        sampler,
      }
    }
    const mips = t.hdr ? 1 : full
    const texture = d.createTexture({
      label: `native stage ${t.name}`,
      size: [t.width, t.height],
      format,
      mipLevelCount: mips,
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    })
    const data = await read(`textures/${t.files[0]}`)
    if (t.hdr) {
      // Raw RGBA16F rows as Unity holds them (row 0 is v 0): as they are.
      d.queue.writeTexture({ texture }, data, { bytesPerRow: t.width * 8 }, [t.width, t.height])
    } else {
      // A PNG runs top down; Unity's row 0 is the bottom (v 0): flipped on the way in.
      const bmp = await createImageBitmap(new Blob([data]), {
        imageOrientation: "flipY",
        colorSpaceConversion: "none",
        premultiplyAlpha: "none",
        // A picture swapped into the folder need not be the size it replaced.
        resizeWidth: t.width,
        resizeHeight: t.height,
        resizeQuality: "high",
      })
      d.queue.copyExternalImageToTexture({ source: bmp }, { texture }, [t.width, t.height])
      bmp.close()
      if (mips > 1) mipmaps(texture, mips)
    }
    this.textures.push(texture)
    const wrap: GPUAddressMode =
      (
        {
          Repeat: "repeat",
          Clamp: "clamp-to-edge",
          Mirror: "mirror-repeat",
          MirrorOnce: "mirror-repeat",
        } as Record<string, GPUAddressMode>
      )[t.wrap ?? "Repeat"] ?? "repeat"
    const filter: GPUFilterMode = t.filter === "Point" ? "nearest" : "linear"
    const trilinear = t.filter === "Trilinear"
    const sampler = d.createSampler({
      addressModeU: wrap,
      addressModeV: wrap,
      magFilter: filter,
      minFilter: filter,
      mipmapFilter: trilinear ? "linear" : "nearest",
      maxAnisotropy: filter === "linear" && trilinear ? Math.min(16, Math.max(1, t.aniso || 1)) : 1,
    })
    return { texture, view: texture.createView(), sampler }
  }

  /** Every renderer's draws, resolved once: the stage does not move. */
  private build(
    meshes: Map<string, { info: NativeStageMesh; buffer: GPUBuffer }>,
    textures: Map<string, Tex>,
    fallback: Record<string, GPUTextureView>,
  ): void {
    const pkg = this.pkg
    const s = pkg.scale
    const mats = new Map(pkg.materials.map((m) => [m.id, m]))
    const globalViews: Record<string, GPUTextureView | undefined> = {}
    const globalSamplers: Record<string, GPUSampler | undefined> = {}
    for (const [slot, id] of Object.entries(pkg.globalTextures)) {
      const t = textures.get(id)
      if (t) {
        globalViews[slot] = t.view
        globalSamplers[slot] = t.sampler
      }
    }
    for (const r of pkg.renderers) {
      const mesh = meshes.get(r.mesh)
      if (!mesh) continue
      const m = r.matrix
      // centre and corners of the mesh box, to engine units
      const b = mesh.info.bounds
      const cx = b.center[0],
        cy = b.center[1],
        cz = b.center[2]
      const center: [number, number, number] = [
        -(m[0] * cx + m[4] * cy + m[8] * cz + m[12]) * s,
        (m[1] * cx + m[5] * cy + m[9] * cz + m[13]) * s,
        -(m[2] * cx + m[6] * cy + m[10] * cz + m[14]) * s,
      ]
      const ex = b.size[0] / 2,
        ey = b.size[1] / 2,
        ez = b.size[2] / 2
      const half = [0, 1, 2].map((i) => (Math.abs(m[i]) * ex + Math.abs(m[4 + i]) * ey + Math.abs(m[8 + i]) * ez) * s)
      this.reach = Math.max(this.reach, Math.hypot(center[0], center[1], center[2]) + Math.hypot(half[0], half[1], half[2]))
      const world = Float32Array.from(m)
      const det = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2])
      const perDraw: ValueSource = {
        unity_ObjectToWorld: world,
        unity_WorldToObject: invert4(world),
        unity_WorldTransformParams: [0, 0, 0, det < 0 ? -1 : 1],
        unity_LightData: [0, 0, 0, 0],
        unity_LightIndices: [
          [0, 0, 0, 0],
          [0, 0, 0, 0],
        ],
        unity_RenderingLayer: new Uint32Array([r.layers >>> 0, 0, 0, 0]),
        unity_LODFade: [1, 1, 0, 0],
        ...r.values,
      }
      this.addItems(r, mesh, perDraw, center, mats, globalViews, globalSamplers, textures, fallback)
      if (r.castShadows) {
        for (let i = 0; i < 3; i++) {
          this.bounds.min[i] = Math.min(this.bounds.min[i], center[i] - half[i])
          this.bounds.max[i] = Math.max(this.bounds.max[i], center[i] + half[i])
        }
      }
    }
  }

  private addItems(
    r: { id: string; materials: (string | null)[]; castShadows: number; textures: Record<string, string> },
    mesh: { info: NativeStageMesh; buffer: GPUBuffer },
    perDraw: ValueSource,
    center: [number, number, number],
    mats: Map<string, NativeStageMaterial>,
    globalViews: Record<string, GPUTextureView | undefined>,
    globalSamplers: Record<string, GPUSampler | undefined>,
    textures: Map<string, Tex>,
    fallback: Record<string, GPUTextureView>,
  ): void {
    r.materials.forEach((mid, sub) => {
        const mat = mid ? mats.get(mid) : undefined
        if (!mat || sub >= mesh.info.submeshes.length) return
        const views: Record<string, GPUTextureView | undefined> = {
          ...globalViews,
        }
        const samplers: Record<string, GPUSampler | undefined> = {
          ...globalSamplers,
        }
        // an empty slot reads what the shader declared it defaults to. A default
        // with no name ("") is no picture at all — a cube's, usually — and is
        // left to the host, which stands in by the binding's own kind.
        for (const [slot, def] of Object.entries(mat.defaults ?? {})) if (def && fallback[def]) views[slot] = fallback[def]
        for (const [slot, id] of [...Object.entries(mat.textures), ...Object.entries(r.textures)]) {
          const t = textures.get(id)
          if (!t) continue
          views[slot] = t.view
          samplers[slot] = t.sampler
        }
        for (const pass of mat.passes) {
          const item: Item = {
            renderer: r,
            material: mat,
            sub,
            mesh,
            pass,
            perDraw,
            textures: views,
            samplers,
            center,
            key: `stage|${r.id}|${sub}|${pass.lightMode}|${pass.shader}`,
          }
          if (pass.lightMode === "SHADOWCASTER") {
            if (r.castShadows) this.casters.push(item)
          } else if (MAIN_MODES.includes(pass.lightMode)) {
            ;(mat.queue > 2500 ? this.transparent : this.opaque).push(item)
          }
        }
      })
  }

  /**
   * Every pass's pipeline, compiled before the stage reports loaded (see
   * NativeHost.warm) — side by side, and never at the first draw. Needs only
   * the meshes' stream layouts, so it runs while the pictures decode. The
   * passes are the ones build() keeps: casters that cast, and the main modes.
   */
  private warm(
    hosts: { scene: NativeHost; shadow: NativeHost },
    meshes: Map<string, { info: NativeStageMesh; buffer: GPUBuffer }>,
  ): Promise<void> {
    const jobs: Promise<void>[] = []
    const mats = new Map(this.pkg.materials.map((m) => [m.id, m]))
    for (const r of this.pkg.renderers) {
      const mesh = meshes.get(r.mesh)
      if (!mesh) continue
      r.materials.forEach((mid, sub) => {
        const mat = mid ? mats.get(mid) : undefined
        if (!mat || sub >= mesh.info.submeshes.length) return
        for (const pass of mat.passes) {
          const caster = pass.lightMode === "SHADOWCASTER"
          if (caster ? !r.castShadows : !MAIN_MODES.includes(pass.lightMode)) continue
          const host = caster ? hosts.shadow : hosts.scene
          jobs.push(host.warm(pass.shader, pass.state, this.streams(host, { mesh, pass } as Item)))
        }
      })
    }
    return Promise.all(jobs).then(() => {})
  }

  private streams(host: NativeHost, it: Item): NativeStream[] {
    const info = it.mesh.info
    return host.inputs(it.pass.shader).map(({ semantic }) => {
      const st = info.streams[semantic]
      if (!st) {
        // A missing vertex colour reads white, as Unity binds it; anything else zero.
        return {
          buffer: semantic === "COLOR" ? this.white : this.zero,
          offset: 0,
          stride: 0,
          format: "float32x4" as const,
        }
      }
      const n = st[1]
      return {
        buffer: it.mesh.buffer,
        offset: st[0],
        stride: n * 4,
        format: (n === 1 ? "float32" : `float32x${n}`) as GPUVertexFormat,
      }
    })
  }

  private draw(
    pass: GPURenderPassEncoder,
    host: NativeHost,
    it: Item,
    globals: Record<string, NativeValue>,
    frame: Record<string, GPUTextureView | undefined>,
    keySuffix = "",
  ): void {
    const sm = it.mesh.info.submeshes[it.sub]
    host.draw(pass, {
      shader: it.pass.shader,
      state: it.pass.state,
      streams: this.streams(host, it),
      index: { buffer: it.mesh.buffer, format: "uint32", offset: sm.offset, count: sm.count },
      sources: [it.perDraw, it.material.values, globals],
      textures: { ...it.textures, ...frame },
      samplers: it.samplers,
      key: it.key + keySuffix,
      pass: keySuffix || "main",
    })
  }

  /** One cascade tile of the engine's atlas: every caster's SHADOWCASTER pass. */
  drawShadow(
    pass: GPURenderPassEncoder,
    host: NativeHost,
    globals: Record<string, NativeValue>,
    cascade: number,
    viewProj: ArrayLike<number>,
    engineScale: number,
  ): void {
    const vp = flipY(mul4(viewProj, gameToEngine(engineScale)))
    // ShadowUtils.GetShadowBias as the game sets it: (texel x the light's
    // shadowBias, -texel x its shadowNormalBias), a texel being this tile's
    // width in game units over its resolution. The stage's casters read it as
    // _ShadowBias (Replica's CascadeShadowPass); the character's DrawShadowPass
    // names it sim_ShadowBias. A package that predates the recorded biases
    // takes half a texel of each.
    const rowLen = Math.hypot(vp[0], vp[4], vp[8])
    const texel = rowLen > 0 ? 2 / rowLen / SHADOW_CASCADES[cascade].mapSize : 0
    const main = this.pkg.lights.main
    const bias = [texel * (main.shadowBias ?? 0.5), -texel * (main.shadowNormalBias ?? 0.5), 0, 0]
    const g: Record<string, NativeValue> = {
      ...globals,
      unity_MatrixVP: vp,
      unity_MatrixV: scale4(1),
      unity_MatrixP: vp,
      glstate_matrix_projection: vp,
      _ShadowBias: bias,
      sim_ShadowBias: bias,
    }
    for (const it of this.casters) this.draw(pass, host, it, g, {}, `|c${cascade}`)
  }

  /** The opaque queue, front to back within each queue. */
  drawOpaque(
    pass: GPURenderPassEncoder,
    host: NativeHost,
    globals: Record<string, NativeValue>,
    frame: Record<string, GPUTextureView | undefined>,
    eye: [number, number, number],
  ): void {
    const d = (it: Item) => (it.center[0] - eye[0]) ** 2 + (it.center[1] - eye[1]) ** 2 + (it.center[2] - eye[2]) ** 2
    const list = this.opaque.slice().sort((a, b) => a.material.queue - b.material.queue || d(a) - d(b))
    for (const it of list) this.draw(pass, host, it, globals, frame)
  }

  /** The transparent queue, back to front within each queue. */
  drawTransparent(
    pass: GPURenderPassEncoder,
    host: NativeHost,
    globals: Record<string, NativeValue>,
    frame: Record<string, GPUTextureView | undefined>,
    eye: [number, number, number],
  ): void {
    const d = (it: Item) => (it.center[0] - eye[0]) ** 2 + (it.center[1] - eye[1]) ** 2 + (it.center[2] - eye[2]) ** 2
    const list = this.transparent.slice().sort((a, b) => a.material.queue - b.material.queue || d(b) - d(a))
    for (const it of list) this.draw(pass, host, it, globals, frame)
  }

  destroy(hosts: NativeHost[]): void {
    for (const h of hosts) h.release("stage|")
    const bufs = this.buffers
    const texs = this.textures
    void this.device.queue.onSubmittedWorkDone().then(() => {
      for (const b of bufs) b.destroy()
      for (const t of texs) t.destroy()
    })
  }
}
