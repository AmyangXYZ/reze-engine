// The native-material host: draws the game's own shaders, translated to WGSL
// (ag-rip's shader_wgsl.py: HLSL -> dxc -> SPIR-V -> naga), inside the engine's
// scene pass.
//
// A translated shader is a vertex module, a fragment module and the binding map
// the translator wrote beside them: every uniform block with its members'
// offsets, every texture and sampler, the vertex inputs by Unity semantic. The
// host builds the pipeline from that map (an explicit layout: the modules
// declare everything the decompiled block did, used or not), fills each block
// BY NAME from value sources — the frame's Unity globals, the draw's
// UnityPerDraw, the character's property block, the material's properties —
// and draws. This is the compare page's renderer (agtools/web/agplay), whose
// output matched the game's frames to a few levels of 255, moved into the
// engine's frame.
//
// What changes on the way in is only what the engine's pass needs: the
// fragment returns the engine's attachments (colour, the aux mask, the id when
// the device has one), depth tests flip with the engine's reversed depth, and
// the pass's MSAA count and formats are the engine's.

import { blendState, colorMask, cull, depthCompare, stencilState, zwrite, type PassState } from "./state"

export type NativeBinding = {
  group: number
  binding: number
  space: string
  name: string
  type: string
  stages: string[]
}
export type NativeMember = {
  name: string
  offset: number
  size: number
  type: string
}
export type NativeStruct = { size: number; members: NativeMember[] }
export type NativeShaderInfo = {
  bindings: Record<string, NativeBinding>
  structs: Record<string, NativeStruct>
  vertexInputs: { location: number; name: string; type: string }[]
}
/** One translated shader variant: what shader_wgsl.py writes for a pass. */
export type NativeShader = {
  name: string
  vert: string
  frag: string
  info: NativeShaderInfo
}

/** A value a uniform member takes: a scalar, a flat vector or matrix, an array of
 *  those, or raw integer bits (Uint32Array — never passed through a float). */
export type NativeValue = number | ArrayLike<number> | ArrayLike<ArrayLike<number>>
/** Name -> value; the first source that has a name wins. */
export type ValueSource = Record<string, NativeValue | undefined>

/** One vertex stream a draw binds, by the semantic a shader input names. */
export type NativeStream = {
  buffer: GPUBuffer
  offset: number
  stride: number
  format: GPUVertexFormat
}

/** Members read as integers rather than floats (the shader asint()s them). */
const INT_MEMBERS = new Set(["unity_LightIndices"])

/** "in_u002e_var_u002e_TEXCOORD2_" -> "TEXCOORD2"; TEXCOORD alone is 0. */
export function semantic(inputName: string): string {
  let s = inputName.replace(/^in_u002e_var_u002e_/, "").replace(/_+$/, "")
  if (s === "TEXCOORD") s = "TEXCOORD0"
  if (s === "COLOR0") s = "COLOR"
  return s
}

/**
 * The fragment module, returning the engine's attachments.
 *
 * naga writes the entry point last: it copies its inputs to private globals,
 * calls the translated body, and returns the one SV_Target. That return is
 * rewritten to the engine's output struct — the colour as the game computed
 * it, the aux mask (bloom on, canvas alpha from the colour's alpha) and, when
 * the device has an id attachment, a zero id. Anything else about the module
 * is left exactly as translated.
 */
export function wrapFragment(frag: string, ids: boolean): string {
  const at = frag.lastIndexOf("@fragment")
  if (at < 0) throw new Error("native shader: no fragment entry point")
  const head = frag.slice(at)
  const sig = /->\s*@location\(0\)\s*vec4<f32>\s*\{/
  if (!sig.test(head)) throw new Error("native shader: the fragment entry does not return one vec4 target")
  let body = head.replace(sig, "-> RzNativeOut {")
  const ret = /return\s+(_e\d+)\s*;\s*\}\s*$/
  const m = ret.exec(body)
  if (!m) throw new Error("native shader: the fragment entry's return is not the translator's")
  const v = m[1]
  body = body.replace(ret, `return RzNativeOut(${v}, vec4f(1.0, 1.0, 0.0, ${v}.a)${ids ? ", vec2u(0u)" : ""});\n}\n`)
  const struct = `struct RzNativeOut {\n  @location(0) color: vec4f,\n  @location(1) mask: vec4f,\n${ids ? "  @location(2) id: vec2u,\n" : ""}};\n`
  // After the module's directives (diagnostic, enable): WGSL wants them first.
  const pre = frag.slice(0, at)
  const directives = /^(?:\s*(?:diagnostic|enable|requires)\b[^;]*;)*/.exec(pre)![0]
  return directives + "\n" + struct + pre.slice(directives.length) + body
}

type Prepared = {
  shader: NativeShader
  vert: GPUShaderModule
  frag: GPUShaderModule
  bindLayout: GPUBindGroupLayout
  layout: GPUPipelineLayout
}

/**
 * One uniform block's buffer and what it was last filled from: per member, the
 * value object and the source it came from. A frame refills only the members
 * whose value object changed — the engine keeps an unchanged global's object
 * from frame to frame (engine.ts, stableGlobals) — and uploads only the bytes
 * between the first and the last of those.
 */
type Block = {
  gpu: GPUBuffer
  data: ArrayBuffer
  f32: Float32Array
  i32: Int32Array
  u32: Uint32Array
  refs: (NativeValue | undefined)[]
  /** Every member came from the last source (the frame's globals). */
  framewide: boolean
  /** The frame it was last brought up to date in (shared blocks). */
  frame: number
}

type DrawCache = {
  buffers: Map<string, Block>
  bind: GPUBindGroup | null
  views: Map<string, GPUTextureView>
  pipeline: GPURenderPipeline | null
}

export type NativeHostTargets = {
  /** The scene pass's colour formats, in attachment order: hdr, aux[, id]. */
  colorFormats: GPUTextureFormat[]
  depthFormat: GPUTextureFormat
  sampleCount: number
  reversedZ: boolean
}

export type NativeDraw = {
  shader: string
  state: PassState
  streams: NativeStream[]
  index: {
    buffer: GPUBuffer
    format: GPUIndexFormat
    offset: number
    count: number
  }
  /** Name -> value, first match wins. */
  sources: ValueSource[]
  /** Texture name -> view (material slots, globals, render targets). */
  textures: Record<string, GPUTextureView | undefined>
  /** Texture name -> its sampler (wrap and filter as the game imported it);
   *  a texture without one samples repeating, trilinear. */
  samplers?: Record<string, GPUSampler | undefined>
  /** Separates draws' uniform buffers: one renderer's pass, one slot. */
  key: string
  /** The pass the frame's globals belong to (main, a shadow cascade): a block
   *  filled from the globals alone is one buffer per shader and pass, shared by
   *  every draw in it. */
  pass?: string
}

export class NativeHost {
  private prepared = new Map<string, Prepared>()
  private pipelines = new Map<string, { pipeline: GPURenderPipeline; stencilRef: number } | null>()
  private draws = new Map<string, DrawCache>()
  private shared = new Map<string, Block>()
  private frame = 0
  private sampler: GPUSampler
  /** Names no source supplied, with how often — the audit the compare page showed. */
  readonly missing = new Map<string, number>()
  readonly errors: string[] = []

  constructor(
    private device: GPUDevice,
    private targets: NativeHostTargets,
    private fallback: {
      white: GPUTextureView
      black: GPUTextureView
      blackCube: GPUTextureView
      depth: GPUTextureView
      comparison: GPUSampler
    },
  ) {
    this.sampler = device.createSampler({
      label: "native material sampler",
      magFilter: "linear",
      minFilter: "linear",
      mipmapFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
      maxAnisotropy: 8,
    })
  }

  has(name: string): boolean {
    return this.prepared.has(name)
  }

  /** Register a translated shader. Idempotent by name. */
  register(shader: NativeShader): void {
    if (this.prepared.has(shader.name)) return
    const d = this.device
    const ids = this.targets.colorFormats.length > 2
    const entries: GPUBindGroupLayoutEntry[] = Object.values(shader.info.bindings).map((b) => {
      const visibility =
        (b.stages.includes("vertex") ? GPUShaderStage.VERTEX : 0) | (b.stages.includes("fragment") ? GPUShaderStage.FRAGMENT : 0)
      if (b.space === "uniform")
        return {
          binding: b.binding,
          visibility,
          buffer: { type: "uniform" as const },
        }
      if (/^sampler_comparison/.test(b.type))
        return {
          binding: b.binding,
          visibility,
          sampler: { type: "comparison" as const },
        }
      if (/^sampler/.test(b.type))
        return {
          binding: b.binding,
          visibility,
          sampler: { type: "filtering" as const },
        }
      if (/texture_depth/.test(b.type))
        return {
          binding: b.binding,
          visibility,
          texture: { sampleType: "depth" as const },
        }
      const viewDimension: GPUTextureViewDimension = /texture_cube/.test(b.type) ? "cube" : /texture_3d/.test(b.type) ? "3d" : "2d"
      return {
        binding: b.binding,
        visibility,
        texture: { sampleType: "float" as const, viewDimension },
      }
    })
    const bindLayout = d.createBindGroupLayout({
      label: `${shader.name} bindings`,
      entries,
    })
    const module = (code: string, label: string) => {
      const m = d.createShaderModule({ label, code })
      void m.getCompilationInfo().then((ci) => {
        for (const x of ci.messages) if (x.type === "error") this.errors.push(`${label}: ${x.message}`)
      })
      return m
    }
    this.prepared.set(shader.name, {
      shader,
      vert: module(shader.vert, `${shader.name} vertex`),
      // A depth-only host keeps the translated fragment: it has no targets to wrap for.
      frag: module(this.targets.colorFormats.length ? wrapFragment(shader.frag, ids) : shader.frag, `${shader.name} fragment`),
      bindLayout,
      layout: d.createPipelineLayout({
        label: shader.name,
        bindGroupLayouts: [bindLayout],
      }),
    })
  }

  /** The vertex inputs a shader reads, by Unity semantic, in location order. */
  inputs(name: string): { location: number; semantic: string }[] {
    const p = this.prepared.get(name)
    return p
      ? p.shader.info.vertexInputs.map((v) => ({
          location: v.location,
          semantic: semantic(v.name),
        }))
      : []
  }

  private pipeline(p: Prepared, state: PassState, streams: NativeStream[]) {
    const key = `${p.shader.name}|${JSON.stringify(state)}|${streams.map((s) => s.format).join(",")}`
    if (this.pipelines.has(key)) return this.pipelines.get(key)!
    const t = this.targets
    const stencil = stencilState(state)
    const blend = blendState(state)
    // A depth-only host (the shadow atlas) has no colour targets at all.
    const targets: GPUColorTargetState[] = !t.colorFormats.length
      ? []
      : [
          {
            format: t.colorFormats[0],
            blend,
            writeMask: colorMask(state.ColorMask?.[0]),
          },
          // The aux mask: bloom on, and the canvas alpha laid over as the engine's
          // own transparent classes lay it.
          {
            format: t.colorFormats[1],
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
            writeMask: colorMask(state.ColorMask?.[0]) === 0 ? 0 : 0xf,
          },
        ]
    if (t.colorFormats[2]) targets.push({ format: t.colorFormats[2], writeMask: 0 })
    const offset = state.Offset
    let result: { pipeline: GPURenderPipeline; stencilRef: number } | null
    try {
      result = {
        pipeline: this.device.createRenderPipeline({
          label: `${p.shader.name} pipeline`,
          layout: p.layout,
          vertex: {
            module: p.vert,
            entryPoint: "vert",
            buffers: p.shader.info.vertexInputs.map((vi, k) => ({
              arrayStride: streams[k].stride,
              stepMode: "vertex" as const,
              attributes: [
                {
                  shaderLocation: vi.location,
                  offset: 0,
                  format: streams[k].format,
                },
              ],
            })),
          },
          fragment: { module: p.frag, entryPoint: "frag", targets },
          // Unity's D3D winding, which is PMX's too: clockwise is the front.
          primitive: {
            topology: "triangle-list",
            cullMode: cull(state.Cull?.[0]),
            frontFace: "cw",
          },
          depthStencil: {
            format: t.depthFormat,
            depthWriteEnabled: zwrite(state.ZWrite?.[0]),
            depthCompare: depthCompare(state.ZTest?.[0], t.reversedZ),
            ...(stencil
              ? {
                  stencilFront: stencil.face,
                  stencilBack: stencil.face,
                  stencilReadMask: stencil.read,
                  stencilWriteMask: stencil.write,
                }
              : {}),
            // Offset (factor, units): pulls toward the viewer, which under
            // reversed depth is the other sign.
            ...(offset
              ? {
                  depthBias: Math.round((+offset[1] || 0) * (t.reversedZ ? 1 : -1)),
                  depthBiasSlopeScale: (+offset[0] || 0) * (t.reversedZ ? 1 : -1),
                }
              : {}),
          },
          multisample: { count: t.sampleCount },
        }),
        stencilRef: stencil?.ref ?? 0,
      }
    } catch (e) {
      this.errors.push(`pipeline ${p.shader.name}: ${(e as Error).message}`)
      result = null
    }
    this.pipelines.set(key, result)
    return result
  }

  private lookup(name: string, sources: ValueSource[]): NativeValue | undefined {
    for (const s of sources) {
      const v = s[name]
      if (v !== undefined && v !== null) return v
    }
    // naga renames a member that ends in a digit with a trailing "_"
    if (name.endsWith("_")) return this.lookup(name.replace(/_+$/, ""), sources)
    return undefined
  }

  /** A new frame: shared blocks are brought up to date once in it. */
  beginFrame(): void {
    this.frame++
  }

  private lookupFrom(name: string, sources: ValueSource[]): [NativeValue | undefined, number] {
    for (let i = 0; i < sources.length; i++) {
      const v = sources[i][name]
      if (v !== undefined && v !== null) return [v, i]
    }
    // naga renames a member that ends in a digit with a trailing "_"
    if (name.endsWith("_")) return this.lookupFrom(name.replace(/_+$/, ""), sources)
    return [undefined, -1]
  }

  private newBlock(struct: NativeStruct, label: string): Block {
    const size = Math.max(16, struct.size)
    const data = new ArrayBuffer(size)
    return {
      gpu: this.device.createBuffer({ label, size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      data,
      f32: new Float32Array(data),
      i32: new Int32Array(data),
      u32: new Uint32Array(data),
      refs: new Array(struct.members.length).fill(null),
      framewide: true,
      frame: -1,
    }
  }

  /** Bring a block up to date with its sources; upload what changed. */
  private fill(struct: NativeStruct, sources: ValueSource[], blk: Block, first: boolean): void {
    let lo = Infinity
    let hi = -1
    const last = sources.length - 1
    for (let mi = 0; mi < struct.members.length; mi++) {
      const m = struct.members[mi]
      const [v, src] = this.lookupFrom(m.name, sources)
      if (first && src >= 0 && src !== last) blk.framewide = false
      if (!first && v === blk.refs[mi]) continue
      blk.refs[mi] = v
      const at = m.offset / 4
      const n = m.size / 4
      lo = Math.min(lo, m.offset)
      hi = Math.max(hi, m.offset + m.size)
      blk.f32.fill(0, at, at + n)
      if (v === undefined) {
        if (first) this.missing.set(m.name, (this.missing.get(m.name) ?? 0) + 1)
        continue
      }
      // Integer bit patterns (light masks, bins) stay integers: through a float
      // a mask could spell a NaN, which a Float32Array does not keep bit for bit.
      const target = v instanceof Uint32Array ? blk.u32 : INT_MEMBERS.has(m.name.replace(/_+$/, "")) ? blk.i32 : blk.f32
      if (typeof v === "number") {
        target[at] = v
        continue
      }
      const arr = /^array<vec4<\w+>,\s*(\d+)>/.exec(m.type)
      const list = v as ArrayLike<number | ArrayLike<number>>
      if (arr && list.length === +arr[1] && typeof list[0] === "number") {
        // A scalar array (float[N] in HLSL, widened to float4[N]): one element
        // per 16-byte slot, read through .x.
        for (let k = 0; k < list.length; k++) target[at + k * 4] = list[k] as number
        continue
      }
      let k = 0
      for (let i = 0; i < list.length && k < n; i++) {
        const e = list[i]
        if (typeof e === "number") target[at + k++] = e
        else for (let j = 0; j < e.length && k < n; j++) target[at + k++] = e[j]
      }
    }
    if (hi > lo) {
      lo &= ~3
      this.device.queue.writeBuffer(blk.gpu, lo, blk.data, lo, Math.min(blk.data.byteLength, (hi + 3) & ~3) - lo)
    }
  }

  /** Encode one draw into the scene pass. */
  draw(pass: GPURenderPassEncoder, d: NativeDraw): void {
    const p = this.prepared.get(d.shader)
    if (!p) return
    const pl = this.pipeline(p, d.state, d.streams)
    if (!pl) return
    let cache = this.draws.get(d.key)
    if (!cache) {
      cache = {
        buffers: new Map(),
        bind: null,
        views: new Map(),
        pipeline: null,
      }
      this.draws.set(d.key, cache)
    }
    const info = p.shader.info
    const entries: GPUBindGroupEntry[] = []
    let rebind = !cache.bind || cache.pipeline !== pl.pipeline
    for (const [name, b] of Object.entries(info.bindings)) {
      if (b.space === "uniform") {
        const struct = info.structs[name]
        if (!struct) continue
        let gb = cache.buffers.get(name)
        if (!gb) {
          gb = this.newBlock(struct, `${d.key} ${name}`)
          this.fill(struct, d.sources, gb, true)
          // Filled from the frame's globals alone: one buffer for the pass,
          // shared by every draw of this shader in it.
          if (gb.framewide && d.pass) {
            const sk = `${d.pass}|${d.shader}|${name}`
            const sh = this.shared.get(sk)
            if (sh) {
              gb.gpu.destroy()
              gb = sh
            } else {
              gb.frame = this.frame
              this.shared.set(sk, gb)
            }
          }
          cache.buffers.set(name, gb)
          rebind = true
        } else if (!gb.framewide || !d.pass) {
          this.fill(struct, d.sources, gb, false)
        } else if (gb.frame !== this.frame) {
          gb.frame = this.frame
          this.fill(struct, d.sources, gb, false)
        }
        entries.push({ binding: b.binding, resource: { buffer: gb.gpu } })
      } else if (/^texture/.test(b.type)) {
        const view = this.textureView(name, b.type, d.textures)
        if (cache.views.get(name) !== view) {
          cache.views.set(name, view)
          rebind = true
        }
        entries.push({ binding: b.binding, resource: view })
      } else if (/^sampler/.test(b.type)) {
        entries.push({
          binding: b.binding,
          resource: /comparison/.test(b.type) ? this.fallback.comparison : this.samplerFor(name, info, d.samplers),
        })
      }
    }
    if (rebind) {
      try {
        cache.bind = this.device.createBindGroup({
          label: `${d.key} bindings`,
          layout: p.bindLayout,
          entries,
        })
        cache.pipeline = pl.pipeline
      } catch (e) {
        this.errors.push(`bind ${d.shader}: ${(e as Error).message}`)
        return
      }
    }
    pass.setPipeline(pl.pipeline)
    pass.setBindGroup(0, cache.bind!)
    pass.setStencilReference(pl.stencilRef)
    d.streams.forEach((s, k) => pass.setVertexBuffer(k, s.buffer, s.offset))
    pass.setIndexBuffer(d.index.buffer, d.index.format, d.index.offset)
    pass.drawIndexed(d.index.count)
  }

  /** sampler_AlbedoTex / samplerAlbedoTex -> the sampler of the texture it pairs with. */
  private samplerFor(name: string, info: NativeShaderInfo, samplers: NativeDraw["samplers"]): GPUSampler {
    if (!samplers) return this.sampler
    const bare = name.replace(/^sampler_?/, "")
    const tex = info.bindings[bare] ? bare : info.bindings[`_${bare}`] ? `_${bare}` : bare
    return samplers[tex] ?? this.sampler
  }

  private textureView(name: string, type: string, textures: Record<string, GPUTextureView | undefined>): GPUTextureView {
    const v = textures[name]
    if (v) return v
    if (/texture_depth/.test(type)) return this.fallback.depth
    if (/texture_cube/.test(type)) return this.fallback.blackCube
    return this.fallback.white
  }

  /** Forget a draw's buffers (its model left). */
  release(prefix: string): void {
    for (const [k, c] of this.draws) {
      if (!k.startsWith(prefix)) continue
      // A pass-wide block outlives any one draw that uses it.
      const kept = new Set(this.shared.values())
      for (const b of c.buffers.values()) if (!kept.has(b)) b.gpu.destroy()
      this.draws.delete(k)
    }
  }
}

export function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false
  const x = new Uint32Array(a.buffer, a.byteOffset, a.length >> 2)
  const y = new Uint32Array(b.buffer, b.byteOffset, b.length >> 2)
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false
  return true
}
