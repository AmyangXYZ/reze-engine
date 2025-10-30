import { RzmModel, RzmTexture, RzmMaterial } from "./rzm"

export class PmxLoader {
  private view: DataView
  private offset = 0
  private decoder!: TextDecoder
  private encoding = 0
  private additionalVec4Count = 0
  private vertexIndexSize = 0
  private textureIndexSize = 0
  private materialIndexSize = 0
  private boneIndexSize = 0
  private textures: RzmTexture[] = []
  private materials: RzmMaterial[] = []

  private constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer)
  }

  static async load(url: string): Promise<RzmModel> {
    const buffer = await fetch(url).then((r) => r.arrayBuffer())
    return new PmxLoader(buffer).parse()
  }

  private parse(): RzmModel {
    this.parseHeader()
    const { positions, normals, uvs } = this.parseVertices()
    const indices = this.parseIndices()
    this.parseTextures()
    this.parseMaterials()
    return this.toRzmModel(positions, normals, uvs, indices)
  }

  private parseHeader() {
    if (this.getString(3) !== "PMX") throw new Error("Not a PMX file")

    // PMX: 1-byte alignment after signature
    this.getUint8()

    // PMX: version (float32)
    const version = this.getFloat32()
    if (version < 2.0 || version > 2.2) {
      // Continue, but warn for unexpected version
      console.warn(`PMX version ${version} may not be fully supported`)
    }

    // PMX: globals count (uint8) followed by that many bytes describing encoding and index sizes
    const globalsCount = this.getUint8()

    // Read globals (8 bytes expected for PMX 2.x)
    this.encoding = this.getUint8() // 0:utf16le, 1:utf8
    this.additionalVec4Count = this.getUint8()
    this.vertexIndexSize = this.getUint8()
    this.textureIndexSize = this.getUint8()
    this.materialIndexSize = this.getUint8()
    this.boneIndexSize = this.getUint8()
    this.getUint8() // currently unused
    this.getUint8() // currently unused

    // Skip any extra globals beyond the first 8
    if (globalsCount > 8) this.offset += globalsCount - 8

    this.decoder = new TextDecoder(this.encoding === 0 ? "utf-16le" : "utf-8")

    // Skip model info (4 text fields)
    this.getText()
    this.getText()
    this.getText()
    this.getText()
  }

  private parseVertices() {
    const count = this.getInt32()
    const positions: number[] = []
    const normals: number[] = []
    const uvs: number[] = []

    for (let i = 0; i < count; i++) {
      positions.push(this.getFloat32(), this.getFloat32(), this.getFloat32())
      normals.push(this.getFloat32(), this.getFloat32(), this.getFloat32())

      const u = this.getFloat32()
      const v = this.getFloat32()
      // PMX UVs are in the same orientation as WebGPU sampling; no flip
      uvs.push(u, v)

      this.offset += this.additionalVec4Count * 16
      this.skipBoneWeight(this.getUint8())
      this.offset += 4 // edge scale
    }

    return { positions, normals, uvs }
  }

  private skipBoneWeight(type: number) {
    const sizes = [1, 2, 4, 2, 4]
    const boneCount = sizes[type]
    if (boneCount === undefined) throw new Error(`Invalid bone weight type: ${type}`)

    this.offset += boneCount * this.boneIndexSize

    if (type === 0) return
    if (type === 1) this.offset += 4
    if (type === 2 || type === 4) this.offset += 16
    if (type === 3) this.offset += 40 // SDEF: weight + c + r0 + r1
  }

  private parseIndices() {
    const count = this.getInt32()
    const indices: number[] = []

    for (let i = 0; i < count; i++) {
      indices.push(this.getIndex(this.vertexIndexSize))
    }

    // Reverse winding order (DirectX to OpenGL)
    for (let i = 0; i < indices.length; i += 3) {
      ;[indices[i + 1], indices[i + 2]] = [indices[i + 2], indices[i + 1]]
    }

    return indices
  }

  private parseTextures() {
    try {
      const count = this.getInt32()
      this.textures = []

      for (let i = 0; i < count; i++) {
        const textureName = this.getText()
        this.textures.push({
          path: textureName,
          name: textureName.split("/").pop() || textureName, // Extract filename
        })
      }
    } catch (error) {
      console.error("Error parsing textures:", error)
      this.textures = []
    }
  }

  private parseMaterials() {
    try {
      const count = this.getInt32()
      this.materials = []

      for (let i = 0; i < count; i++) {
        const name = this.getText()
        this.getText() // englishName (skip)

        const diffuse = [this.getFloat32(), this.getFloat32(), this.getFloat32(), this.getFloat32()] as [
          number,
          number,
          number,
          number
        ]
        const specular = [this.getFloat32(), this.getFloat32(), this.getFloat32()] as [number, number, number]
        const shininess = this.getFloat32()
        const ambient = [this.getFloat32(), this.getFloat32(), this.getFloat32()] as [number, number, number]

        const flag = this.getUint8()
        // edgeColor vec4 (skip 4 floats)
        this.getFloat32()
        this.getFloat32()
        this.getFloat32()
        this.getFloat32()
        // edgeSize float (skip 1 float)
        this.getFloat32()

        const textureIndex = this.getNonVertexIndex(this.textureIndexSize)
        const sphereTextureIndex = this.getNonVertexIndex(this.textureIndexSize)
        const sphereTextureMode = this.getUint8()

        const isSharedToonTexture = this.getUint8() === 1
        const toonTextureIndex = isSharedToonTexture ? this.getUint8() : this.getNonVertexIndex(this.textureIndexSize)

        this.getText() // comment (skip)
        const vertexCount = this.getInt32()

        this.materials.push({
          name,
          diffuse,
          specular,
          ambient,
          shininess,
          diffuseTextureIndex: textureIndex,
          normalTextureIndex: -1, // Not used in basic PMX
          sphereTextureIndex,
          sphereMode: sphereTextureMode,
          toonTextureIndex,
          edgeFlag: flag,
          vertexCount,
        })
      }
    } catch (error) {
      console.error("Error parsing materials:", error)
      this.materials = []
    }
  }

  private toRzmModel(positions: number[], normals: number[], uvs: number[], indices: number[]): RzmModel {
    // Create indexed vertex buffer
    const vertexCount = positions.length / 3
    const vertexData = new Float32Array(vertexCount * 8)

    for (let i = 0; i < vertexCount; i++) {
      const pi = i * 3
      const ui = i * 2
      const vi = i * 8

      vertexData[vi] = positions[pi]
      vertexData[vi + 1] = positions[pi + 1]
      vertexData[vi + 2] = positions[pi + 2]
      vertexData[vi + 3] = normals[pi]
      vertexData[vi + 4] = normals[pi + 1]
      vertexData[vi + 5] = normals[pi + 2]
      vertexData[vi + 6] = uvs[ui]
      vertexData[vi + 7] = uvs[ui + 1]
    }

    // Create index buffer
    const indexData = new Uint32Array(indices)

    return new RzmModel(vertexData, indexData, this.textures, this.materials)
  }

  private getUint8() {
    return this.view.getUint8(this.offset++)
  }

  // Vertex index: 1->uint8, 2->uint16, 4->int32
  private getVertexIndex(size: number) {
    if (size === 1) return this.getUint8()
    if (size === 2) {
      const v = this.view.getUint16(this.offset, true)
      this.offset += 2
      return v
    }
    return this.getInt32()
  }

  // Non-vertex indices (texture/material/bone/morph/rigid): 1->int8, 2->int16, 4->int32
  private getNonVertexIndex(size: number) {
    if (size === 1) {
      const v = this.view.getInt8(this.offset)
      this.offset += 1
      return v
    }
    if (size === 2) {
      const v = this.view.getInt16(this.offset, true)
      this.offset += 2
      return v
    }
    return this.getInt32()
  }

  private getInt32() {
    const v = this.view.getInt32(this.offset, true)
    this.offset += 4
    return v
  }

  private getFloat32() {
    const v = this.view.getFloat32(this.offset, true)
    this.offset += 4
    return v
  }

  private getString(len: number) {
    const bytes = new Uint8Array(this.view.buffer, this.offset, len)
    this.offset += len
    return String.fromCharCode(...bytes)
  }

  private getText() {
    const len = this.getInt32()
    if (len <= 0) return ""

    // Debug: log problematic string lengths
    if (len > 1000 || len < -1000) {
      console.warn(`Suspicious string length: ${len} at offset ${this.offset}`)
    }

    // Ensure we don't read beyond buffer bounds
    if (this.offset + len > this.view.buffer.byteLength) {
      console.warn(`String length ${len} exceeds buffer bounds, skipping`)
      this.offset += len
      return ""
    }

    const bytes = new Uint8Array(this.view.buffer, this.offset, len)
    this.offset += len
    return this.decoder.decode(bytes)
  }

  private getIndex(size: number) {
    // Backward-compat helper: defaults to vertex index behavior
    return this.getVertexIndex(size)
  }
}
