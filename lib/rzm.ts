// RZM (Reze Model) Format - WebGPU-native model format
// Phase 2: Geometry + Textures + Materials

const RZM_MAGIC = 0x455a4552 // "REZE" in little-endian
const RZM_VERSION = 2 // Updated version for texture support
const VERTEX_STRIDE = 8 // floats per vertex: position(3) + normal(3) + uv(2) = 8

export interface RzmTexture {
  path: string
  name: string
}

export interface RzmMaterial {
  name: string
  diffuse: [number, number, number, number]
  specular: [number, number, number]
  ambient: [number, number, number]
  shininess: number
  diffuseTextureIndex: number
  normalTextureIndex: number
  sphereTextureIndex: number
  sphereMode: number
  toonTextureIndex: number
  edgeFlag: number
  vertexCount: number
}

export class RzmModel {
  private vertexData: Float32Array<ArrayBuffer>
  private vertexCount: number
  private indexData?: Uint32Array<ArrayBuffer>
  private indexCount: number
  private textures: RzmTexture[] = []
  private materials: RzmMaterial[] = []

  constructor(
    vertexData: Float32Array<ArrayBuffer>,
    indexData?: Uint32Array<ArrayBuffer>,
    textures: RzmTexture[] = [],
    materials: RzmMaterial[] = []
  ) {
    this.vertexData = vertexData
    this.vertexCount = vertexData.length / VERTEX_STRIDE
    this.indexData = indexData
    this.indexCount = indexData ? indexData.length : 0
    this.textures = textures
    this.materials = materials
  }

  // Load RZM model from URL
  static async load(url: string): Promise<RzmModel> {
    const response = await fetch(url)
    const buffer = await response.arrayBuffer()
    return RzmModel.parse(buffer)
  }

  // Parse RZM model from ArrayBuffer
  static parse(buffer: ArrayBuffer): RzmModel {
    const view = new DataView(buffer)

    // Read header
    const magic = view.getUint32(0, true)
    const version = view.getUint32(4, true)
    const vertexCount = view.getUint32(12, true)
    const indexCount = view.getUint32(16, true)

    // Validate
    if (magic !== RZM_MAGIC) {
      throw new Error(`Invalid RZM file: magic mismatch`)
    }
    if (version !== RZM_VERSION) {
      throw new Error(`Unsupported RZM version: ${version}`)
    }

    // Read vertex data (starts after 64-byte header)
    const vertexDataLength = vertexCount * VERTEX_STRIDE
    const sourceVertexData = new Float32Array(buffer, 64, vertexDataLength)
    const vertexData = new Float32Array(sourceVertexData) // Copy to new array

    // Read index data if present (starts after vertex data)
    let indexData: Uint32Array<ArrayBuffer> | undefined
    if (indexCount > 0) {
      const indexOffset = 64 + vertexDataLength * 4 // 4 bytes per float
      const sourceIndexData = new Uint32Array(buffer, indexOffset, indexCount)
      indexData = new Uint32Array(sourceIndexData) // Copy to new array
    }

    return new RzmModel(vertexData, indexData)
  }

  // Get interleaved vertex data for GPU upload
  // Format: [x,y,z, nx,ny,nz, u,v, x,y,z, nx,ny,nz, u,v, ...]
  getVertices(): Float32Array<ArrayBuffer> {
    return this.vertexData
  }

  // Get texture information
  getTextures(): RzmTexture[] {
    return this.textures
  }

  // Get material information
  getMaterials(): RzmMaterial[] {
    return this.materials
  }

  // Get diffuse texture path for a material
  getDiffuseTexturePath(materialIndex: number): string | null {
    if (materialIndex < 0 || materialIndex >= this.materials.length) return null
    const material = this.materials[materialIndex]
    if (material.diffuseTextureIndex < 0 || material.diffuseTextureIndex >= this.textures.length) return null
    return this.textures[material.diffuseTextureIndex].path
  }

  // Get vertex count
  getVertexCount(): number {
    return this.vertexCount
  }

  // Get index data for GPU upload
  getIndices(): Uint32Array<ArrayBuffer> | undefined {
    return this.indexData
  }

  // Get index count
  getIndexCount(): number {
    return this.indexCount
  }

  // Create RZM model from position-only data
  // Generates dummy normals and UVs
  static fromPositions(positions: Float32Array): RzmModel {
    const vertexCount = positions.length / 3
    const vertexData = new Float32Array(vertexCount * VERTEX_STRIDE)

    for (let i = 0; i < vertexCount; i++) {
      const posIdx = i * 3
      const vertIdx = i * VERTEX_STRIDE

      // Position
      vertexData[vertIdx + 0] = positions[posIdx + 0]
      vertexData[vertIdx + 1] = positions[posIdx + 1]
      vertexData[vertIdx + 2] = positions[posIdx + 2]

      // Normal (dummy, pointing up)
      vertexData[vertIdx + 3] = 0
      vertexData[vertIdx + 4] = 1
      vertexData[vertIdx + 5] = 0

      // UV (dummy)
      vertexData[vertIdx + 6] = 0
      vertexData[vertIdx + 7] = 0
    }

    return new RzmModel(vertexData)
  }

  // Export to RZM binary format
  toArrayBuffer(): ArrayBuffer {
    const headerSize = 64
    const vertexDataSize = this.vertexData.byteLength
    const indexDataSize = this.indexData ? this.indexData.byteLength : 0
    const totalSize = headerSize + vertexDataSize + indexDataSize

    const buffer = new ArrayBuffer(totalSize)
    const view = new DataView(buffer)

    // Write header
    view.setUint32(0, RZM_MAGIC, true) // magic
    view.setUint32(4, RZM_VERSION, true) // version
    view.setUint32(8, 0, true) // flags
    view.setUint32(12, this.vertexCount, true) // vertexCount
    view.setUint32(16, this.indexCount, true) // indexCount
    view.setUint32(20, 0, true) // materialCount (TODO)
    view.setUint32(24, 0, true) // boneCount (TODO)
    // Rest of header is padding

    // Write vertex data
    new Float32Array(buffer, 64, this.vertexData.length).set(this.vertexData)

    // Write index data if present
    if (this.indexData) {
      const indexOffset = 64 + vertexDataSize
      new Uint32Array(buffer, indexOffset, this.indexData.length).set(this.indexData)
    }

    return buffer
  }
}
