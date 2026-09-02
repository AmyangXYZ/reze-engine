// The PMX file, as data, losslessly.
//
// SEPARATE from Model on purpose. Model is what the renderer needs and drops
// everything else — English names, memos, display frames, additional UVs, SDEF
// and QDEF weights, per-vertex edge scale. That is correct for drawing: making
// Model lossless would make every viewer pay per-vertex costs for a scene it
// never edits. It is fatal for editing. A writer fed from Model would turn
// "rename a bone and save" into "rename a bone and lose every morph's panel
// assignment", and the user would rightly call that a broken editor.
//
// So this reads the file into a structure that mirrors it one for one, and
// writes that structure back. Read and write live in ONE module and each section
// keeps its reader and writer adjacent, because the only real failure mode here
// is the two drifting apart in field order — which the format cannot detect and
// which produces a file that loads as garbage.
//
// Round-tripping is checked byte-for-byte against real models in
// tests/pmx-roundtrip.test.mjs. That test is the product promise in executable
// form, so it is worth more than any of the code below.

/** 0 = UTF-16LE, 1 = UTF-8. Preserved: writing back in the other one is a
 *  different file, even where it decodes the same. */
export type PmxEncoding = 0 | 1

export interface PmxGlobals {
  encoding: PmxEncoding
  additionalUvCount: number
  /** Vertex indices are UNSIGNED; every other index is signed, with -1 for none. */
  vertexIndexSize: 1 | 2 | 4
  textureIndexSize: 1 | 2 | 4
  materialIndexSize: 1 | 2 | 4
  boneIndexSize: 1 | 2 | 4
  morphIndexSize: 1 | 2 | 4
  rigidbodyIndexSize: 1 | 2 | 4
  /** Anything past the eight PMX 2.0 knows about, kept so the header's own
   *  length survives a round trip. */
  extra: number[]
}

export type PmxWeightType = 0 | 1 | 2 | 3 | 4 // BDEF1, BDEF2, BDEF4, SDEF, QDEF

export interface PmxVertex {
  position: [number, number, number]
  normal: [number, number, number]
  uv: [number, number]
  /** additionalUvCount entries, each a vec4. */
  additionalUv: [number, number, number, number][]
  weightType: PmxWeightType
  /** 1, 2 or 4 entries depending on weightType. */
  bones: number[]
  /** BDEF2 keeps one, BDEF4 and QDEF keep four, BDEF1 keeps none. */
  weights: number[]
  /** SDEF only: C, R0, R1. */
  sdef?: { c: [number, number, number]; r0: [number, number, number]; r1: [number, number, number] }
  edgeScale: number
}

export interface PmxMaterial {
  name: string
  nameEn: string
  diffuse: [number, number, number, number]
  specular: [number, number, number]
  specularPower: number
  ambient: [number, number, number]
  /** Bit 0 double-sided, 1 ground shadow, 2 to self-shadow map, 3 from
   *  self-shadow, 4 edge, 5 vertex colour, 6 point draw, 7 line draw. */
  drawFlags: number
  edgeColor: [number, number, number, number]
  edgeSize: number
  textureIndex: number
  sphereIndex: number
  /** 0 none, 1 multiply, 2 add, 3 subtexture. */
  sphereMode: number
  /** 0 = toonIndex is a texture index, 1 = it is a shared toon 0-9. */
  toonShared: number
  toonIndex: number
  memo: string
  /** How many INDICES this material owns, not faces. Materials own a contiguous
   *  run of the index buffer, which is why face order is document identity. */
  indexCount: number
}

export interface PmxIkLink {
  boneIndex: number
  hasLimit: boolean
  limitMin?: [number, number, number]
  limitMax?: [number, number, number]
}

export interface PmxBone {
  name: string
  nameEn: string
  position: [number, number, number]
  parentIndex: number
  layer: number
  /** 0x0001 tail is a bone · 0x0002 rotatable · 0x0004 movable · 0x0008 visible
   *  0x0010 operable · 0x0020 IK · 0x0080 local append · 0x0100 append rotate
   *  0x0200 append move · 0x0400 fixed axis · 0x0800 local axes
   *  0x1000 after physics · 0x2000 external parent */
  flags: number
  tailBoneIndex?: number
  tailPosition?: [number, number, number]
  appendParentIndex?: number
  appendRatio?: number
  fixedAxis?: [number, number, number]
  localAxisX?: [number, number, number]
  localAxisZ?: [number, number, number]
  externalKey?: number
  ik?: {
    targetIndex: number
    loopCount: number
    limitAngle: number
    links: PmxIkLink[]
  }
}

export type PmxMorphOffset =
  | { kind: "group"; index: number; influence: number }
  | { kind: "vertex"; index: number; offset: [number, number, number] }
  | { kind: "bone"; index: number; translation: [number, number, number]; rotation: [number, number, number, number] }
  | { kind: "uv"; index: number; offset: [number, number, number, number] }
  | {
      kind: "material"
      index: number
      /** 0 multiply, 1 add. */
      operation: number
      diffuse: [number, number, number, number]
      specular: [number, number, number]
      specularPower: number
      ambient: [number, number, number]
      edgeColor: [number, number, number, number]
      edgeSize: number
      textureTint: [number, number, number, number]
      sphereTint: [number, number, number, number]
      toonTint: [number, number, number, number]
    }
  | { kind: "flip"; index: number; influence: number }
  | {
      kind: "impulse"
      index: number
      local: number
      velocity: [number, number, number]
      torque: [number, number, number]
    }

export interface PmxMorph {
  name: string
  nameEn: string
  /** 0 reserved, 1 eyebrow, 2 eye, 3 mouth, 4 other. What MMD groups sliders by. */
  panel: number
  /** 0 group, 1 vertex, 2 bone, 3-7 UV, 8 material, 9 flip, 10 impulse. */
  type: number
  offsets: PmxMorphOffset[]
}

export interface PmxDisplayFrame {
  name: string
  nameEn: string
  /** 1 for the two frames MMD reserves (Root and 表情). */
  special: number
  /** type 0 = bone index, 1 = morph index. */
  elements: { type: number; index: number }[]
}

export interface PmxRigidbody {
  name: string
  nameEn: string
  boneIndex: number
  group: number
  nonCollideMask: number
  /** 0 sphere, 1 box, 2 capsule. */
  shape: number
  size: [number, number, number]
  position: [number, number, number]
  rotation: [number, number, number]
  mass: number
  linearDamping: number
  angularDamping: number
  restitution: number
  friction: number
  /** 0 follows its bone, 1 physics, 2 physics then aligned to its bone. */
  physicsMode: number
}

export interface PmxJoint {
  name: string
  nameEn: string
  /** 0 is 6DOF-with-spring, the only one PMX 2.0 has. 2.1 adds 1-5. */
  type: number
  rigidbodyA: number
  rigidbodyB: number
  position: [number, number, number]
  rotation: [number, number, number]
  positionMin: [number, number, number]
  positionMax: [number, number, number]
  rotationMin: [number, number, number]
  rotationMax: [number, number, number]
  springPosition: [number, number, number]
  springRotation: [number, number, number]
}

export interface PmxDocument {
  version: number
  globals: PmxGlobals
  name: string
  nameEn: string
  comment: string
  commentEn: string
  vertices: PmxVertex[]
  /** Flat triples. Materials own contiguous runs of this, so its ORDER is
   *  identity: reordering it silently re-assigns geometry between materials. */
  indices: Uint32Array
  textures: string[]
  materials: PmxMaterial[]
  bones: PmxBone[]
  morphs: PmxMorph[]
  displayFrames: PmxDisplayFrame[]
  rigidbodies: PmxRigidbody[]
  joints: PmxJoint[]
  /** PMX 2.1 soft bodies, kept as the bytes they arrived as. They are vanishingly
   *  rare and nothing here edits them — but a model that has them must still save
   *  as the same file, and preserving the tail costs nothing. */
  trailing: Uint8Array | null
}

// ──────────────────────────────────────────────────────────────────
// Bytes

class Cursor {
  readonly view: DataView
  readonly bytes: Uint8Array
  at = 0
  constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer)
    this.bytes = new Uint8Array(buffer)
  }
  u8() {
    return this.view.getUint8(this.at++)
  }
  u16() {
    const v = this.view.getUint16(this.at, true)
    this.at += 2
    return v
  }
  i32() {
    const v = this.view.getInt32(this.at, true)
    this.at += 4
    return v
  }
  f32() {
    const v = this.view.getFloat32(this.at, true)
    this.at += 4
    return v
  }
  vec2(): [number, number] {
    return [this.f32(), this.f32()]
  }
  vec3(): [number, number, number] {
    return [this.f32(), this.f32(), this.f32()]
  }
  vec4(): [number, number, number, number] {
    return [this.f32(), this.f32(), this.f32(), this.f32()]
  }
  /** Unsigned — vertex indices only. */
  uindex(size: number): number {
    if (size === 1) return this.u8()
    if (size === 2) return this.u16()
    return this.view.getUint32((this.at += 4) - 4, true)
  }
  /** Signed, with the all-ones pattern meaning "none". Every index but vertex. */
  index(size: number): number {
    if (size === 1) return this.view.getInt8(this.at++)
    if (size === 2) {
      const v = this.view.getInt16(this.at, true)
      this.at += 2
      return v
    }
    return this.i32()
  }
  text(encoding: PmxEncoding): string {
    const length = this.i32()
    const slice = this.bytes.subarray(this.at, this.at + length)
    this.at += length
    if (encoding === 1) return new TextDecoder("utf-8").decode(slice)
    return new TextDecoder("utf-16le").decode(slice)
  }
}

class Sink {
  private chunks: Uint8Array[] = []
  private buf = new Uint8Array(65536)
  private view = new DataView(this.buf.buffer)
  private at = 0
  private need(n: number) {
    if (this.at + n <= this.buf.length) return
    this.chunks.push(this.buf.subarray(0, this.at))
    this.buf = new Uint8Array(Math.max(65536, n))
    this.view = new DataView(this.buf.buffer)
    this.at = 0
  }
  u8(v: number) {
    this.need(1)
    this.view.setUint8(this.at++, v)
  }
  u16(v: number) {
    this.need(2)
    this.view.setUint16(this.at, v, true)
    this.at += 2
  }
  i32(v: number) {
    this.need(4)
    this.view.setInt32(this.at, v, true)
    this.at += 4
  }
  f32(v: number) {
    this.need(4)
    this.view.setFloat32(this.at, v, true)
    this.at += 4
  }
  vec(v: readonly number[]) {
    for (const n of v) this.f32(n)
  }
  uindex(size: number, v: number) {
    if (size === 1) this.u8(v)
    else if (size === 2) this.u16(v)
    else this.i32(v | 0)
  }
  index(size: number, v: number) {
    if (size === 1) this.u8(v & 0xff)
    else if (size === 2) this.u16(v & 0xffff)
    else this.i32(v)
  }
  raw(v: Uint8Array) {
    this.need(v.length)
    this.buf.set(v, this.at)
    this.at += v.length
  }
  text(encoding: PmxEncoding, v: string) {
    const bytes = encoding === 1 ? new TextEncoder().encode(v) : utf16le(v)
    this.i32(bytes.length)
    this.raw(bytes)
  }
  finish(): ArrayBuffer {
    this.chunks.push(this.buf.subarray(0, this.at))
    let total = 0
    for (const c of this.chunks) total += c.length
    const out = new Uint8Array(total)
    let at = 0
    for (const c of this.chunks) {
      out.set(c, at)
      at += c.length
    }
    return out.buffer
  }
}

/** TextEncoder has no utf-16le, so the surrogate pairs are written by hand. */
function utf16le(s: string): Uint8Array {
  const out = new Uint8Array(s.length * 2)
  const view = new DataView(out.buffer)
  for (let i = 0; i < s.length; i++) view.setUint16(i * 2, s.charCodeAt(i), true)
  return out
}

// ──────────────────────────────────────────────────────────────────
// Read and write, one section at a time, each pair adjacent

/** Parses a PMX file into a document that writes back byte for byte. */
export function readPmxDocument(buffer: ArrayBuffer): PmxDocument {
  const c = new Cursor(buffer)

  // Signature. "Pmx " appears in the wild alongside "PMX ", and PMD is the older
  // format entirely — say which it was rather than "not a PMX file".
  const sig = String.fromCharCode(c.u8(), c.u8(), c.u8(), c.u8())
  if (sig.slice(0, 3).toUpperCase() !== "PMX") {
    if (sig.slice(0, 3).toUpperCase() === "PMD") {
      throw new Error("This is a PMD file (MMD's older format). Convert it to PMX in PMXEditor first.")
    }
    throw new Error(`Not a PMX file (signature "${sig.replace(/[^\x20-\x7e]/g, "?")}")`)
  }
  const version = c.f32()
  const globalsCount = c.u8()
  if (globalsCount < 8) throw new Error(`PMX header declares ${globalsCount} globals, expected at least 8`)
  const globals: PmxGlobals = {
    encoding: c.u8() as PmxEncoding,
    additionalUvCount: c.u8(),
    vertexIndexSize: c.u8() as 1 | 2 | 4,
    textureIndexSize: c.u8() as 1 | 2 | 4,
    materialIndexSize: c.u8() as 1 | 2 | 4,
    boneIndexSize: c.u8() as 1 | 2 | 4,
    morphIndexSize: c.u8() as 1 | 2 | 4,
    rigidbodyIndexSize: c.u8() as 1 | 2 | 4,
    extra: [],
  }
  for (let i = 8; i < globalsCount; i++) globals.extra.push(c.u8())
  const e = globals.encoding

  const doc: PmxDocument = {
    version,
    globals,
    name: c.text(e),
    nameEn: c.text(e),
    comment: c.text(e),
    commentEn: c.text(e),
    vertices: [],
    indices: new Uint32Array(0),
    textures: [],
    materials: [],
    bones: [],
    morphs: [],
    displayFrames: [],
    rigidbodies: [],
    joints: [],
    trailing: null,
  }

  // ── Vertices ──
  const vertexCount = c.i32()
  doc.vertices.length = vertexCount
  for (let i = 0; i < vertexCount; i++) {
    const v: PmxVertex = {
      position: c.vec3(),
      normal: c.vec3(),
      uv: c.vec2(),
      additionalUv: [],
      weightType: 0,
      bones: [],
      weights: [],
      edgeScale: 0,
    }
    for (let k = 0; k < globals.additionalUvCount; k++) v.additionalUv.push(c.vec4())
    v.weightType = c.u8() as PmxWeightType
    const bi = globals.boneIndexSize
    switch (v.weightType) {
      case 0:
        v.bones = [c.index(bi)]
        break
      case 1:
        v.bones = [c.index(bi), c.index(bi)]
        v.weights = [c.f32()]
        break
      case 2:
      case 4:
        v.bones = [c.index(bi), c.index(bi), c.index(bi), c.index(bi)]
        v.weights = [c.f32(), c.f32(), c.f32(), c.f32()]
        break
      case 3:
        v.bones = [c.index(bi), c.index(bi)]
        v.weights = [c.f32()]
        v.sdef = { c: c.vec3(), r0: c.vec3(), r1: c.vec3() }
        break
    }
    v.edgeScale = c.f32()
    doc.vertices[i] = v
  }

  // ── Faces ──
  const indexCount = c.i32()
  const indices = new Uint32Array(indexCount)
  for (let i = 0; i < indexCount; i++) indices[i] = c.uindex(globals.vertexIndexSize)
  doc.indices = indices

  // ── Textures ──
  const textureCount = c.i32()
  for (let i = 0; i < textureCount; i++) doc.textures.push(c.text(e))

  // ── Materials ──
  const materialCount = c.i32()
  for (let i = 0; i < materialCount; i++) {
    const m: PmxMaterial = {
      name: c.text(e),
      nameEn: c.text(e),
      diffuse: c.vec4(),
      specular: c.vec3(),
      specularPower: c.f32(),
      ambient: c.vec3(),
      drawFlags: c.u8(),
      edgeColor: c.vec4(),
      edgeSize: c.f32(),
      textureIndex: c.index(globals.textureIndexSize),
      sphereIndex: c.index(globals.textureIndexSize),
      sphereMode: c.u8(),
      toonShared: c.u8(),
      toonIndex: 0,
      memo: "",
      indexCount: 0,
    }
    m.toonIndex = m.toonShared === 1 ? c.u8() : c.index(globals.textureIndexSize)
    m.memo = c.text(e)
    m.indexCount = c.i32()
    doc.materials.push(m)
  }

  // ── Bones ──
  const boneCount = c.i32()
  const bi = globals.boneIndexSize
  for (let i = 0; i < boneCount; i++) {
    const b: PmxBone = {
      name: c.text(e),
      nameEn: c.text(e),
      position: c.vec3(),
      parentIndex: c.index(bi),
      layer: c.i32(),
      flags: c.u16(),
    }
    if (b.flags & 0x0001) b.tailBoneIndex = c.index(bi)
    else b.tailPosition = c.vec3()
    if (b.flags & 0x0300) {
      b.appendParentIndex = c.index(bi)
      b.appendRatio = c.f32()
    }
    if (b.flags & 0x0400) b.fixedAxis = c.vec3()
    if (b.flags & 0x0800) {
      b.localAxisX = c.vec3()
      b.localAxisZ = c.vec3()
    }
    if (b.flags & 0x2000) b.externalKey = c.i32()
    if (b.flags & 0x0020) {
      const targetIndex = c.index(bi)
      const loopCount = c.i32()
      const limitAngle = c.f32()
      const linkCount = c.i32()
      const links: PmxIkLink[] = []
      for (let k = 0; k < linkCount; k++) {
        const boneIndex = c.index(bi)
        const hasLimit = c.u8() === 1
        links.push(hasLimit ? { boneIndex, hasLimit, limitMin: c.vec3(), limitMax: c.vec3() } : { boneIndex, hasLimit })
      }
      b.ik = { targetIndex, loopCount, limitAngle, links }
    }
    doc.bones.push(b)
  }

  // ── Morphs ──
  const morphCount = c.i32()
  for (let i = 0; i < morphCount; i++) {
    const name = c.text(e)
    const nameEn = c.text(e)
    const panel = c.u8()
    const type = c.u8()
    const offsetCount = c.i32()
    const offsets: PmxMorphOffset[] = []
    for (let k = 0; k < offsetCount; k++) {
      switch (type) {
        case 0:
          offsets.push({ kind: "group", index: c.index(globals.morphIndexSize), influence: c.f32() })
          break
        case 1:
          offsets.push({ kind: "vertex", index: c.uindex(globals.vertexIndexSize), offset: c.vec3() })
          break
        case 2:
          offsets.push({ kind: "bone", index: c.index(bi), translation: c.vec3(), rotation: c.vec4() })
          break
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
          offsets.push({ kind: "uv", index: c.uindex(globals.vertexIndexSize), offset: c.vec4() })
          break
        case 8:
          offsets.push({
            kind: "material",
            index: c.index(globals.materialIndexSize),
            operation: c.u8(),
            diffuse: c.vec4(),
            specular: c.vec3(),
            specularPower: c.f32(),
            ambient: c.vec3(),
            edgeColor: c.vec4(),
            edgeSize: c.f32(),
            textureTint: c.vec4(),
            sphereTint: c.vec4(),
            toonTint: c.vec4(),
          })
          break
        case 9:
          offsets.push({ kind: "flip", index: c.index(globals.morphIndexSize), influence: c.f32() })
          break
        case 10:
          offsets.push({
            kind: "impulse",
            index: c.index(globals.rigidbodyIndexSize),
            local: c.u8(),
            velocity: c.vec3(),
            torque: c.vec3(),
          })
          break
        default:
          throw new Error(`Morph "${name}" has unknown type ${type}`)
      }
    }
    doc.morphs.push({ name, nameEn, panel, type, offsets })
  }

  // ── Display frames ──
  const frameCount = c.i32()
  for (let i = 0; i < frameCount; i++) {
    const name = c.text(e)
    const nameEn = c.text(e)
    const special = c.u8()
    const elementCount = c.i32()
    const elements: { type: number; index: number }[] = []
    for (let k = 0; k < elementCount; k++) {
      const type = c.u8()
      elements.push({ type, index: c.index(type === 1 ? globals.morphIndexSize : bi) })
    }
    doc.displayFrames.push({ name, nameEn, special, elements })
  }

  // ── Rigidbodies ──
  const bodyCount = c.i32()
  for (let i = 0; i < bodyCount; i++) {
    doc.rigidbodies.push({
      name: c.text(e),
      nameEn: c.text(e),
      boneIndex: c.index(bi),
      group: c.u8(),
      nonCollideMask: c.u16(),
      shape: c.u8(),
      size: c.vec3(),
      position: c.vec3(),
      rotation: c.vec3(),
      mass: c.f32(),
      linearDamping: c.f32(),
      angularDamping: c.f32(),
      restitution: c.f32(),
      friction: c.f32(),
      physicsMode: c.u8(),
    })
  }

  // ── Joints ──
  const jointCount = c.i32()
  const ri = globals.rigidbodyIndexSize
  for (let i = 0; i < jointCount; i++) {
    doc.joints.push({
      name: c.text(e),
      nameEn: c.text(e),
      type: c.u8(),
      rigidbodyA: c.index(ri),
      rigidbodyB: c.index(ri),
      position: c.vec3(),
      rotation: c.vec3(),
      positionMin: c.vec3(),
      positionMax: c.vec3(),
      rotationMin: c.vec3(),
      rotationMax: c.vec3(),
      springPosition: c.vec3(),
      springRotation: c.vec3(),
    })
  }

  if (c.at < c.bytes.length) doc.trailing = c.bytes.slice(c.at)
  return doc
}

/** Writes a document back to PMX bytes. */
export function writePmxDocument(doc: PmxDocument): ArrayBuffer {
  const w = new Sink()
  const g = doc.globals
  const e = g.encoding
  const bi = g.boneIndexSize

  for (const ch of "PMX ") w.u8(ch.charCodeAt(0))
  w.f32(doc.version)
  w.u8(8 + g.extra.length)
  w.u8(g.encoding)
  w.u8(g.additionalUvCount)
  w.u8(g.vertexIndexSize)
  w.u8(g.textureIndexSize)
  w.u8(g.materialIndexSize)
  w.u8(g.boneIndexSize)
  w.u8(g.morphIndexSize)
  w.u8(g.rigidbodyIndexSize)
  for (const x of g.extra) w.u8(x)
  w.text(e, doc.name)
  w.text(e, doc.nameEn)
  w.text(e, doc.comment)
  w.text(e, doc.commentEn)

  w.i32(doc.vertices.length)
  for (const v of doc.vertices) {
    w.vec(v.position)
    w.vec(v.normal)
    w.vec(v.uv)
    for (let k = 0; k < g.additionalUvCount; k++) w.vec(v.additionalUv[k] ?? [0, 0, 0, 0])
    w.u8(v.weightType)
    switch (v.weightType) {
      case 0:
        w.index(bi, v.bones[0])
        break
      case 1:
        w.index(bi, v.bones[0])
        w.index(bi, v.bones[1])
        w.f32(v.weights[0])
        break
      case 2:
      case 4:
        for (let k = 0; k < 4; k++) w.index(bi, v.bones[k])
        for (let k = 0; k < 4; k++) w.f32(v.weights[k])
        break
      case 3:
        w.index(bi, v.bones[0])
        w.index(bi, v.bones[1])
        w.f32(v.weights[0])
        w.vec(v.sdef!.c)
        w.vec(v.sdef!.r0)
        w.vec(v.sdef!.r1)
        break
    }
    w.f32(v.edgeScale)
  }

  w.i32(doc.indices.length)
  for (let i = 0; i < doc.indices.length; i++) w.uindex(g.vertexIndexSize, doc.indices[i])

  w.i32(doc.textures.length)
  for (const t of doc.textures) w.text(e, t)

  w.i32(doc.materials.length)
  for (const m of doc.materials) {
    w.text(e, m.name)
    w.text(e, m.nameEn)
    w.vec(m.diffuse)
    w.vec(m.specular)
    w.f32(m.specularPower)
    w.vec(m.ambient)
    w.u8(m.drawFlags)
    w.vec(m.edgeColor)
    w.f32(m.edgeSize)
    w.index(g.textureIndexSize, m.textureIndex)
    w.index(g.textureIndexSize, m.sphereIndex)
    w.u8(m.sphereMode)
    w.u8(m.toonShared)
    if (m.toonShared === 1) w.u8(m.toonIndex)
    else w.index(g.textureIndexSize, m.toonIndex)
    w.text(e, m.memo)
    w.i32(m.indexCount)
  }

  w.i32(doc.bones.length)
  for (const b of doc.bones) {
    w.text(e, b.name)
    w.text(e, b.nameEn)
    w.vec(b.position)
    w.index(bi, b.parentIndex)
    w.i32(b.layer)
    w.u16(b.flags)
    if (b.flags & 0x0001) w.index(bi, b.tailBoneIndex!)
    else w.vec(b.tailPosition!)
    if (b.flags & 0x0300) {
      w.index(bi, b.appendParentIndex!)
      w.f32(b.appendRatio!)
    }
    if (b.flags & 0x0400) w.vec(b.fixedAxis!)
    if (b.flags & 0x0800) {
      w.vec(b.localAxisX!)
      w.vec(b.localAxisZ!)
    }
    if (b.flags & 0x2000) w.i32(b.externalKey!)
    if (b.flags & 0x0020) {
      const ik = b.ik!
      w.index(bi, ik.targetIndex)
      w.i32(ik.loopCount)
      w.f32(ik.limitAngle)
      w.i32(ik.links.length)
      for (const l of ik.links) {
        w.index(bi, l.boneIndex)
        w.u8(l.hasLimit ? 1 : 0)
        if (l.hasLimit) {
          w.vec(l.limitMin!)
          w.vec(l.limitMax!)
        }
      }
    }
  }

  w.i32(doc.morphs.length)
  for (const m of doc.morphs) {
    w.text(e, m.name)
    w.text(e, m.nameEn)
    w.u8(m.panel)
    w.u8(m.type)
    w.i32(m.offsets.length)
    for (const o of m.offsets) {
      switch (o.kind) {
        case "group":
          w.index(g.morphIndexSize, o.index)
          w.f32(o.influence)
          break
        case "vertex":
          w.uindex(g.vertexIndexSize, o.index)
          w.vec(o.offset)
          break
        case "bone":
          w.index(bi, o.index)
          w.vec(o.translation)
          w.vec(o.rotation)
          break
        case "uv":
          w.uindex(g.vertexIndexSize, o.index)
          w.vec(o.offset)
          break
        case "material":
          w.index(g.materialIndexSize, o.index)
          w.u8(o.operation)
          w.vec(o.diffuse)
          w.vec(o.specular)
          w.f32(o.specularPower)
          w.vec(o.ambient)
          w.vec(o.edgeColor)
          w.f32(o.edgeSize)
          w.vec(o.textureTint)
          w.vec(o.sphereTint)
          w.vec(o.toonTint)
          break
        case "flip":
          w.index(g.morphIndexSize, o.index)
          w.f32(o.influence)
          break
        case "impulse":
          w.index(g.rigidbodyIndexSize, o.index)
          w.u8(o.local)
          w.vec(o.velocity)
          w.vec(o.torque)
          break
      }
    }
  }

  w.i32(doc.displayFrames.length)
  for (const f of doc.displayFrames) {
    w.text(e, f.name)
    w.text(e, f.nameEn)
    w.u8(f.special)
    w.i32(f.elements.length)
    for (const el of f.elements) {
      w.u8(el.type)
      w.index(el.type === 1 ? g.morphIndexSize : bi, el.index)
    }
  }

  w.i32(doc.rigidbodies.length)
  for (const b of doc.rigidbodies) {
    w.text(e, b.name)
    w.text(e, b.nameEn)
    w.index(bi, b.boneIndex)
    w.u8(b.group)
    w.u16(b.nonCollideMask)
    w.u8(b.shape)
    w.vec(b.size)
    w.vec(b.position)
    w.vec(b.rotation)
    w.f32(b.mass)
    w.f32(b.linearDamping)
    w.f32(b.angularDamping)
    w.f32(b.restitution)
    w.f32(b.friction)
    w.u8(b.physicsMode)
  }

  w.i32(doc.joints.length)
  for (const j of doc.joints) {
    w.text(e, j.name)
    w.text(e, j.nameEn)
    w.u8(j.type)
    w.index(g.rigidbodyIndexSize, j.rigidbodyA)
    w.index(g.rigidbodyIndexSize, j.rigidbodyB)
    w.vec(j.position)
    w.vec(j.rotation)
    w.vec(j.positionMin)
    w.vec(j.positionMax)
    w.vec(j.rotationMin)
    w.vec(j.rotationMax)
    w.vec(j.springPosition)
    w.vec(j.springRotation)
  }

  if (doc.trailing) w.raw(doc.trailing)
  return w.finish()
}
