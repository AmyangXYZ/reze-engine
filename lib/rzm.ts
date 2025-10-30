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

export interface RzmBone {
  name: string
  parentIndex: number // -1 if no parent
  bindTranslation: [number, number, number]
}

export interface RzmSkeleton {
  bones: RzmBone[]
  // One inverse-bind matrix per bone (column-major mat4, 16 floats per bone)
  inverseBindMatrices: Float32Array
}

import { Mat4, Vec3, Quat } from "./math"

export class RzmModel {
  private vertexData: Float32Array<ArrayBuffer>
  private vertexCount: number
  private indexData?: Uint32Array<ArrayBuffer>
  private indexCount: number
  private textures: RzmTexture[] = []
  private materials: RzmMaterial[] = []
  // Static skeleton/skinning (not necessarily serialized yet)
  private skeleton?: RzmSkeleton
  private joints0?: Uint16Array // length = vertexCount * 4
  private weights0?: Uint8Array // UNORM8, length = vertexCount * 4, sums ~ 255 per-vertex

  // Runtime (non-serialized) pose data
  private boneLocalRotations?: Float32Array // quat per bone (x,y,z,w) length = boneCount*4
  private boneLocalTranslations?: Float32Array // vec3 per bone length = boneCount*3
  private boneWorldMatrices?: Float32Array // mat4 per bone length = boneCount*16
  private boneSkinMatrices?: Float32Array // mat4 per bone length = boneCount*16

  constructor(
    vertexData: Float32Array<ArrayBuffer>,
    indexData?: Uint32Array<ArrayBuffer>,
    textures: RzmTexture[] = [],
    materials: RzmMaterial[] = [],
    skeleton?: RzmSkeleton,
    skinning?: { joints0: Uint16Array; weights0: Uint8Array }
  ) {
    this.vertexData = vertexData
    this.vertexCount = vertexData.length / VERTEX_STRIDE
    this.indexData = indexData
    this.indexCount = indexData ? indexData.length : 0
    this.textures = textures
    this.materials = materials
    this.skeleton = skeleton
    if (skinning) {
      this.joints0 = skinning.joints0
      this.weights0 = skinning.weights0
    }
  }

  // --- Simple per-bone rotation tween state (runtime only) ---
  private _rotTweenActive?: Uint8Array // 0/1 per bone
  private _rotTweenStartQuat?: Float32Array // quat per bone (x,y,z,w)
  private _rotTweenTargetQuat?: Float32Array // quat per bone (x,y,z,w)
  private _rotTweenStartTimeMs?: Float32Array // one float per bone (ms)
  private _rotTweenDurationMs?: Float32Array // one float per bone (ms)

  private ensureRotTweenBuffers(): void {
    if (!this.skeleton) return
    const n = this.skeleton.bones.length
    if (!this._rotTweenActive) this._rotTweenActive = new Uint8Array(n)
    if (!this._rotTweenStartQuat) this._rotTweenStartQuat = new Float32Array(n * 4)
    if (!this._rotTweenTargetQuat) this._rotTweenTargetQuat = new Float32Array(n * 4)
    if (!this._rotTweenStartTimeMs) this._rotTweenStartTimeMs = new Float32Array(n)
    if (!this._rotTweenDurationMs) this._rotTweenDurationMs = new Float32Array(n)
  }

  private static easeInOut(t: number): number {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
  }

  private static slerp(
    aX: number,
    aY: number,
    aZ: number,
    aW: number,
    bX: number,
    bY: number,
    bZ: number,
    bW: number,
    t: number
  ): [number, number, number, number] {
    // Robust basic slerp with shortest path
    let cos = aX * bX + aY * bY + aZ * bZ + aW * bW
    let bx = bX,
      by = bY,
      bz = bZ,
      bw = bW
    if (cos < 0) {
      cos = -cos
      bx = -bx
      by = -by
      bz = -bz
      bw = -bw
    }
    if (cos > 0.9995) {
      // Linear fallback
      const x = aX + t * (bx - aX)
      const y = aY + t * (by - aY)
      const z = aZ + t * (bz - aZ)
      const w = aW + t * (bw - aW)
      const invLen = 1 / Math.hypot(x, y, z, w)
      return [x * invLen, y * invLen, z * invLen, w * invLen]
    }
    const theta0 = Math.acos(cos)
    const sinTheta0 = Math.sin(theta0)
    const theta = theta0 * t
    const s0 = Math.sin(theta0 - theta) / sinTheta0
    const s1 = Math.sin(theta) / sinTheta0
    const x = s0 * aX + s1 * bx
    const y = s0 * aY + s1 * by
    const z = s0 * aZ + s1 * bz
    const w = s0 * aW + s1 * bw
    return [x, y, z, w]
  }

  private updateRotationTweens(): void {
    if (!this.skeleton) return
    if (!this._rotTweenActive) return
    this.ensureRuntimePose()
    this.ensureRotTweenBuffers()
    if (!this.boneLocalRotations) return
    const now = performance.now()
    const n = this.skeleton.bones.length
    for (let i = 0; i < n; i++) {
      if (this._rotTweenActive[i] !== 1) continue
      const startMs = this._rotTweenStartTimeMs![i]
      const durMs = Math.max(1, this._rotTweenDurationMs![i])
      const t = Math.max(0, Math.min(1, (now - startMs) / durMs))
      const e = RzmModel.easeInOut(t)
      const qi = i * 4
      const a0 = this._rotTweenStartQuat![qi]
      const a1 = this._rotTweenStartQuat![qi + 1]
      const a2 = this._rotTweenStartQuat![qi + 2]
      const a3 = this._rotTweenStartQuat![qi + 3]
      const b0 = this._rotTweenTargetQuat![qi]
      const b1 = this._rotTweenTargetQuat![qi + 1]
      const b2 = this._rotTweenTargetQuat![qi + 2]
      const b3 = this._rotTweenTargetQuat![qi + 3]
      const [x, y, z, w] = RzmModel.slerp(a0, a1, a2, a3, b0, b1, b2, b3, e)
      this.boneLocalRotations[qi] = x
      this.boneLocalRotations[qi + 1] = y
      this.boneLocalRotations[qi + 2] = z
      this.boneLocalRotations[qi + 3] = w
      if (t >= 1) {
        this._rotTweenActive[i] = 0
      }
    }
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

  // Accessors for skeleton/skinning
  getSkeleton(): RzmSkeleton | undefined {
    return this.skeleton
  }

  getSkinning(): { joints0: Uint16Array; weights0: Uint8Array } | undefined {
    if (this.joints0 && this.weights0) return { joints0: this.joints0, weights0: this.weights0 }
    return undefined
  }

  getSkinMatrices(): Float32Array | undefined {
    if (!this.boneSkinMatrices || !this.skeleton) return undefined
    return this.boneSkinMatrices
  }

  // Runtime pose (non-serialized) setters/getters (minimal for now)
  ensureRuntimePose(): void {
    if (!this.skeleton) return
    const boneCount = this.skeleton.bones.length
    if (!this.boneLocalRotations) this.boneLocalRotations = new Float32Array(boneCount * 4)
    if (!this.boneLocalTranslations) this.boneLocalTranslations = new Float32Array(boneCount * 3)
    if (!this.boneWorldMatrices) this.boneWorldMatrices = new Float32Array(boneCount * 16)
    if (!this.boneSkinMatrices) this.boneSkinMatrices = new Float32Array(boneCount * 16)
    // Initialize rotations to identity (0,0,0,1). Translations default to zero
    for (let i = 0; i < boneCount; i++) {
      const qi = i * 4
      if (this.boneLocalRotations[qi + 3] === 0) {
        this.boneLocalRotations[qi] = 0
        this.boneLocalRotations[qi + 1] = 0
        this.boneLocalRotations[qi + 2] = 0
        this.boneLocalRotations[qi + 3] = 1
      }
    }
  }

  // ------- Bone helpers (public API) -------
  getBoneCount(): number {
    return this.skeleton ? this.skeleton.bones.length : 0
  }

  getBoneNames(): string[] {
    if (!this.skeleton) return []
    return this.skeleton.bones.map((b) => b.name)
  }

  getBoneIndexByName(name: string): number {
    if (!this.skeleton) return -1
    for (let i = 0; i < this.skeleton.bones.length; i++) {
      if (this.skeleton.bones[i].name === name) return i
    }
    return -1
  }

  getBoneName(index: number): string | undefined {
    if (!this.skeleton) return undefined
    if (index < 0 || index >= this.skeleton.bones.length) return undefined
    return this.skeleton.bones[index].name
  }

  getBoneLocal(index: number): { rotation: Quat; translation: Vec3 } | undefined {
    if (!this.skeleton) return undefined
    this.ensureRuntimePose()
    if (!this.boneLocalRotations || !this.boneLocalTranslations) return undefined
    if (index < 0 || index >= this.skeleton.bones.length) return undefined
    const qi = index * 4
    const ti = index * 3
    return {
      rotation: new Quat(
        this.boneLocalRotations[qi],
        this.boneLocalRotations[qi + 1],
        this.boneLocalRotations[qi + 2],
        this.boneLocalRotations[qi + 3]
      ),
      translation: new Vec3(
        this.boneLocalTranslations[ti],
        this.boneLocalTranslations[ti + 1],
        this.boneLocalTranslations[ti + 2]
      ),
    }
  }

  // Batched rotation with optional interpolation and retargeting.
  rotateBones(indicesOrNames: Array<number | string>, quats: Quat[], durationMs?: number): void {
    if (!this.skeleton) return
    this.ensureRuntimePose()
    this.ensureRotTweenBuffers()
    if (!this.boneLocalRotations || !this._rotTweenActive) return

    quats = quats.map((q) => q.normalize())
    // Resolve indices
    const indices: number[] = new Array(indicesOrNames.length)
    for (let i = 0; i < indicesOrNames.length; i++) {
      const v = indicesOrNames[i]
      const idx = typeof v === "number" ? v : this.getBoneIndexByName(v)
      if (idx < 0 || idx >= this.skeleton.bones.length) continue
      indices[i] = idx
    }

    // Quats are normalized above; use toArray() for xyzw

    const now = performance.now()
    const dur = durationMs && durationMs > 0 ? durationMs : 0

    for (let i = 0; i < indices.length; i++) {
      const bi = indices[i]
      if (bi === undefined) continue
      const qi = bi * 4
      const [tx, ty, tz, tw] = quats[i].toArray()

      if (dur === 0) {
        // Immediate set, cancel any tween
        this.boneLocalRotations[qi] = tx
        this.boneLocalRotations[qi + 1] = ty
        this.boneLocalRotations[qi + 2] = tz
        this.boneLocalRotations[qi + 3] = tw
        this._rotTweenActive[bi] = 0
        continue
      }

      // Retarget: if active, compute current pose as new start; else start from current local
      let sx = this.boneLocalRotations[qi]
      let sy = this.boneLocalRotations[qi + 1]
      let sz = this.boneLocalRotations[qi + 2]
      let sw = this.boneLocalRotations[qi + 3]
      if (this._rotTweenActive[bi] === 1) {
        const startMs = this._rotTweenStartTimeMs![bi]
        const prevDur = Math.max(1, this._rotTweenDurationMs![bi])
        const t = Math.max(0, Math.min(1, (now - startMs) / prevDur))
        const e = RzmModel.easeInOut(t)
        const a0 = this._rotTweenStartQuat![qi]
        const a1 = this._rotTweenStartQuat![qi + 1]
        const a2 = this._rotTweenStartQuat![qi + 2]
        const a3 = this._rotTweenStartQuat![qi + 3]
        const b0 = this._rotTweenTargetQuat![qi]
        const b1 = this._rotTweenTargetQuat![qi + 1]
        const b2 = this._rotTweenTargetQuat![qi + 2]
        const b3 = this._rotTweenTargetQuat![qi + 3]
        const cur = RzmModel.slerp(a0, a1, a2, a3, b0, b1, b2, b3, e)
        sx = cur[0]
        sy = cur[1]
        sz = cur[2]
        sw = cur[3]
      }

      // Write start and target, mark active
      this._rotTweenStartQuat![qi] = sx
      this._rotTweenStartQuat![qi + 1] = sy
      this._rotTweenStartQuat![qi + 2] = sz
      this._rotTweenStartQuat![qi + 3] = sw
      this._rotTweenTargetQuat![qi] = tx
      this._rotTweenTargetQuat![qi + 1] = ty
      this._rotTweenTargetQuat![qi + 2] = tz
      this._rotTweenTargetQuat![qi + 3] = tw
      this._rotTweenStartTimeMs![bi] = now
      this._rotTweenDurationMs![bi] = dur
      this._rotTweenActive[bi] = 1
    }
  }

  setBoneRotation(indexOrName: number | string, quat: Quat): void {
    if (!this.skeleton) return
    const index = typeof indexOrName === "number" ? indexOrName : this.getBoneIndexByName(indexOrName)
    if (index < 0 || index >= this.skeleton.bones.length) return
    this.ensureRuntimePose()
    if (!this.boneLocalRotations) return
    const qi = index * 4
    this.boneLocalRotations[qi] = quat.x
    this.boneLocalRotations[qi + 1] = quat.y
    this.boneLocalRotations[qi + 2] = quat.z
    this.boneLocalRotations[qi + 3] = quat.w
  }

  setBoneTranslation(indexOrName: number | string, t: Vec3): void {
    if (!this.skeleton) return
    const index = typeof indexOrName === "number" ? indexOrName : this.getBoneIndexByName(indexOrName)
    if (index < 0 || index >= this.skeleton.bones.length) return
    this.ensureRuntimePose()
    if (!this.boneLocalTranslations) return
    const ti = index * 3
    this.boneLocalTranslations[ti] = t.x
    this.boneLocalTranslations[ti + 1] = t.y
    this.boneLocalTranslations[ti + 2] = t.z
  }

  resetBone(index: number): void {
    if (!this.skeleton) return
    this.ensureRuntimePose()
    if (!this.boneLocalRotations || !this.boneLocalTranslations) return
    const qi = index * 4
    const ti = index * 3
    this.boneLocalRotations[qi] = 0
    this.boneLocalRotations[qi + 1] = 0
    this.boneLocalRotations[qi + 2] = 0
    this.boneLocalRotations[qi + 3] = 1
    this.boneLocalTranslations[ti] = 0
    this.boneLocalTranslations[ti + 1] = 0
    this.boneLocalTranslations[ti + 2] = 0
  }

  resetAllBones(): void {
    const count = this.getBoneCount()
    for (let i = 0; i < count; i++) this.resetBone(i)
  }

  getBoneWorldMatrix(index: number): Float32Array | undefined {
    if (!this.skeleton) return undefined
    this.ensureRuntimePose()
    this.evaluatePose()
    if (!this.boneWorldMatrices) return undefined
    const start = index * 16
    return this.boneWorldMatrices.slice(start, start + 16)
  }

  // Evaluate world and skin matrices from local TR and bind
  evaluatePose(): void {
    if (!this.skeleton) return
    this.ensureRuntimePose()
    // Advance rotation tweens before composing matrices
    this.updateRotationTweens()
    if (!this.boneLocalRotations || !this.boneLocalTranslations || !this.boneWorldMatrices || !this.boneSkinMatrices)
      return
    const bones = this.skeleton.bones
    const invBind = this.skeleton.inverseBindMatrices
    // Local references (non-null) for inner closure
    const localRot = this.boneLocalRotations!
    const worldBuf = this.boneWorldMatrices!
    const skinBuf = this.boneSkinMatrices!

    // compute recursively to respect parent-before-child regardless of file order
    const computed: boolean[] = new Array(bones.length).fill(false)

    const computeWorld = (i: number): void => {
      if (computed[i]) return
      if (bones[i].parentIndex >= bones.length) {
        console.warn(`[RZM] bone ${i} parent out of range: ${bones[i].parentIndex}`)
      }
      const qi = i * 4
      // runtime local translation ignored in pure-rotation mode
      const rotateM = Mat4.fromQuat(localRot[qi], localRot[qi + 1], localRot[qi + 2], localRot[qi + 3])
      const translateBind = Mat4.identity().translateInPlace(
        bones[i].bindTranslation[0],
        bones[i].bindTranslation[1],
        bones[i].bindTranslation[2]
      )
      // Local: move from parent to joint, then rotate around joint
      const localM = translateBind.multiply(rotateM)

      const worldSeg = worldBuf.subarray(i * 16, i * 16 + 16)
      let worldM: Mat4
      if (bones[i].parentIndex >= 0) {
        const p = bones[i].parentIndex
        if (!computed[p]) computeWorld(p)
        const parentSeg = worldBuf.subarray(p * 16, p * 16 + 16)
        worldM = new Mat4(parentSeg).multiply(localM)
      } else {
        worldM = localM
      }
      worldSeg.set(worldM.values)
      computed[i] = true

      const skinSeg = skinBuf.subarray(i * 16, i * 16 + 16)
      const invSeg = invBind.subarray(i * 16, i * 16 + 16)
      const skinM = new Mat4(worldSeg).multiply(new Mat4(invSeg))
      skinSeg.set(skinM.values)
    }

    for (let i = 0; i < bones.length; i++) computeWorld(i)
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
    view.setUint32(20, this.materials.length, true) // materialCount
    view.setUint32(24, this.skeleton ? this.skeleton.bones.length : 0, true) // boneCount
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
