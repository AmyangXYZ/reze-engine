import { RzmModel } from "./rzm"

export class PmxLoader {
  private view: DataView
  private offset = 0
  private decoder!: TextDecoder
  private encoding = 0
  private additionalVec4Count = 0
  private vertexIndexSize = 0
  private boneIndexSize = 0

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
    return this.toRzmModel(positions, normals, uvs, indices)
  }

  private parseHeader() {
    if (this.getString(4) !== "PMX ") throw new Error("Not a PMX file")

    this.offset += 4 // version (float)
    const globalsCount = this.getUint8()

    // Read globals (8 bytes expected)
    this.encoding = this.getUint8()
    this.additionalVec4Count = this.getUint8()
    this.vertexIndexSize = this.getUint8()
    this.getUint8() // textureIndexSize (unused for now)
    this.getUint8() // materialIndexSize (unused for now)
    this.boneIndexSize = this.getUint8()
    this.getUint8() // morphIndexSize (unused for now)
    this.getUint8() // rigidBodyIndexSize (unused for now)

    // Skip any extra globals
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
      uvs.push(u, 1.0 - v) // DirectX to OpenGL

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

    return new RzmModel(vertexData, indexData)
  }

  private getUint8() {
    return this.view.getUint8(this.offset++)
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
    if (len === 0) return ""
    const bytes = new Uint8Array(this.view.buffer, this.offset, len)
    this.offset += len
    return this.decoder.decode(bytes)
  }

  private getIndex(size: number) {
    if (size === 1) return this.getUint8()
    if (size === 2) {
      const v = this.view.getUint16(this.offset, true)
      this.offset += 2
      return v
    }
    return this.getInt32()
  }
}
