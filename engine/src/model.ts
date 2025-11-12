import { Mat4, Quat, easeInOut } from "./math"
import { Rigidbody, Joint } from "./physics"

const VERTEX_STRIDE = 8

export interface Texture {
  path: string
  name: string
}

export interface Material {
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
  edgeColor: [number, number, number, number]
  edgeSize: number
  vertexCount: number
  isEye?: boolean // New: marks eye materials
  isFace?: boolean // New: marks face/skin materials
  isHair?: boolean // New: marks hair materials
}

export interface Bone {
  name: string
  parentIndex: number // -1 if no parent
  bindTranslation: [number, number, number]
  children: number[] // child bone indices (built on skeleton creation)
  appendParentIndex?: number // index of the bone to inherit from
  appendRatio?: number // 0..1
  appendRotate?: boolean
  appendMove?: boolean
}

export interface Skeleton {
  bones: Bone[]
  inverseBindMatrices: Float32Array // One inverse-bind matrix per bone (column-major mat4, 16 floats per bone)
}

export interface Skinning {
  joints: Uint16Array // length = vertexCount * 4, bone indices per vertex
  weights: Uint8Array // UNORM8, length = vertexCount * 4, sums ~ 255 per-vertex
}

// Runtime skeleton pose state (updated each frame)
export interface SkeletonRuntime {
  nameIndex: Record<string, number> // Cached lookup: bone name -> bone index (built on initialization)
  localRotations: Float32Array // quat per bone (x,y,z,w) length = boneCount*4
  localTranslations: Float32Array // vec3 per bone length = boneCount*3
  worldMatrices: Float32Array // mat4 per bone length = boneCount*16
  computedBones: boolean[] // length = boneCount
}

// Rotation tween state per bone
interface RotationTweenState {
  active: Uint8Array // 0/1 per bone
  startQuat: Float32Array // quat per bone (x,y,z,w)
  targetQuat: Float32Array // quat per bone (x,y,z,w)
  startTimeMs: Float32Array // one float per bone (ms)
  durationMs: Float32Array // one float per bone (ms)
}

export class Model {
  private vertexData: Float32Array<ArrayBuffer>
  private vertexCount: number
  private indexData: Uint32Array<ArrayBuffer>
  private textures: Texture[] = []
  private materials: Material[] = []
  // Static skeleton/skinning (not necessarily serialized yet)
  private skeleton: Skeleton
  private skinning: Skinning

  // Physics data from PMX
  private rigidbodies: Rigidbody[] = []
  private joints: Joint[] = []

  // Runtime skeleton pose state (updated each frame)
  private runtimeSkeleton!: SkeletonRuntime

  // Cached identity matrices to avoid allocations in computeWorldMatrices
  private cachedIdentityMat1 = Mat4.identity()
  private cachedIdentityMat2 = Mat4.identity()

  private rotTweenState!: RotationTweenState

  constructor(
    vertexData: Float32Array<ArrayBuffer>,
    indexData: Uint32Array<ArrayBuffer>,
    textures: Texture[],
    materials: Material[],
    skeleton: Skeleton,
    skinning: Skinning,
    rigidbodies: Rigidbody[] = [],
    joints: Joint[] = []
  ) {
    this.vertexData = vertexData
    this.vertexCount = vertexData.length / VERTEX_STRIDE
    this.indexData = indexData
    this.textures = textures
    this.materials = materials
    this.skeleton = skeleton
    this.skinning = skinning
    this.rigidbodies = rigidbodies
    this.joints = joints

    if (this.skeleton.bones.length == 0) {
      throw new Error("Model has no bones")
    }

    this.initializeRuntimeSkeleton()
    this.initializeRotTweenBuffers()
  }

  private initializeRuntimeSkeleton(): void {
    const boneCount = this.skeleton.bones.length

    this.runtimeSkeleton = {
      localRotations: new Float32Array(boneCount * 4),
      localTranslations: new Float32Array(boneCount * 3),
      worldMatrices: new Float32Array(boneCount * 16),
      nameIndex: this.skeleton.bones.reduce((acc, bone, index) => {
        acc[bone.name] = index
        return acc
      }, {} as Record<string, number>),
      computedBones: new Array(boneCount).fill(false),
    }

    const rotations = this.runtimeSkeleton.localRotations
    for (let i = 0; i < this.skeleton.bones.length; i++) {
      const qi = i * 4
      if (rotations[qi + 3] === 0) {
        rotations[qi] = 0
        rotations[qi + 1] = 0
        rotations[qi + 2] = 0
        rotations[qi + 3] = 1
      }
    }
  }

  private initializeRotTweenBuffers(): void {
    const n = this.skeleton.bones.length
    this.rotTweenState = {
      active: new Uint8Array(n),
      startQuat: new Float32Array(n * 4),
      targetQuat: new Float32Array(n * 4),
      startTimeMs: new Float32Array(n),
      durationMs: new Float32Array(n),
    }
  }

  private updateRotationTweens(): void {
    const state = this.rotTweenState
    const now = performance.now()
    const rotations = this.runtimeSkeleton.localRotations
    const boneCount = this.skeleton.bones.length

    for (let i = 0; i < boneCount; i++) {
      if (state.active[i] !== 1) continue

      const startMs = state.startTimeMs[i]
      const durMs = Math.max(1, state.durationMs[i])
      const t = Math.max(0, Math.min(1, (now - startMs) / durMs))
      const e = easeInOut(t)

      const qi = i * 4
      const startQuat = new Quat(
        state.startQuat[qi],
        state.startQuat[qi + 1],
        state.startQuat[qi + 2],
        state.startQuat[qi + 3]
      )
      const targetQuat = new Quat(
        state.targetQuat[qi],
        state.targetQuat[qi + 1],
        state.targetQuat[qi + 2],
        state.targetQuat[qi + 3]
      )
      const result = Quat.slerp(startQuat, targetQuat, e)

      rotations[qi] = result.x
      rotations[qi + 1] = result.y
      rotations[qi + 2] = result.z
      rotations[qi + 3] = result.w

      if (t >= 1) state.active[i] = 0
    }
  }

  // Get interleaved vertex data for GPU upload
  // Format: [x,y,z, nx,ny,nz, u,v, x,y,z, nx,ny,nz, u,v, ...]
  getVertices(): Float32Array<ArrayBuffer> {
    return this.vertexData
  }

  // Get texture information
  getTextures(): Texture[] {
    return this.textures
  }

  // Get material information
  getMaterials(): Material[] {
    return this.materials
  }

  // Get vertex count
  getVertexCount(): number {
    return this.vertexCount
  }

  // Get index data for GPU upload
  getIndices(): Uint32Array<ArrayBuffer> {
    return this.indexData
  }

  // Accessors for skeleton/skinning
  getSkeleton(): Skeleton {
    return this.skeleton
  }

  getSkinning(): Skinning {
    return this.skinning
  }

  // Accessors for physics data
  getRigidbodies(): Rigidbody[] {
    return this.rigidbodies
  }

  getJoints(): Joint[] {
    return this.joints
  }

  // ------- Bone helpers (public API) -------

  getBoneNames(): string[] {
    return this.skeleton.bones.map((b) => b.name)
  }

  rotateBones(names: string[], quats: Quat[], durationMs?: number): void {
    const state = this.rotTweenState
    const normalized = quats.map((q) => q.normalize())
    const now = performance.now()
    const dur = durationMs && durationMs > 0 ? durationMs : 0

    for (let i = 0; i < names.length; i++) {
      const name = names[i]
      const idx = this.runtimeSkeleton.nameIndex[name] ?? -1
      if (idx < 0 || idx >= this.skeleton.bones.length) continue

      const qi = idx * 4
      const rotations = this.runtimeSkeleton.localRotations
      const [tx, ty, tz, tw] = normalized[i].toArray()

      if (dur === 0) {
        rotations[qi] = tx
        rotations[qi + 1] = ty
        rotations[qi + 2] = tz
        rotations[qi + 3] = tw
        state.active[idx] = 0
        continue
      }

      let sx = rotations[qi]
      let sy = rotations[qi + 1]
      let sz = rotations[qi + 2]
      let sw = rotations[qi + 3]

      if (state.active[idx] === 1) {
        const startMs = state.startTimeMs[idx]
        const prevDur = Math.max(1, state.durationMs[idx])
        const t = Math.max(0, Math.min(1, (now - startMs) / prevDur))
        const e = easeInOut(t)
        const startQuat = new Quat(
          state.startQuat[qi],
          state.startQuat[qi + 1],
          state.startQuat[qi + 2],
          state.startQuat[qi + 3]
        )
        const targetQuat = new Quat(
          state.targetQuat[qi],
          state.targetQuat[qi + 1],
          state.targetQuat[qi + 2],
          state.targetQuat[qi + 3]
        )
        const result = Quat.slerp(startQuat, targetQuat, e)
        const cx = result.x
        const cy = result.y
        const cz = result.z
        const cw = result.w
        sx = cx
        sy = cy
        sz = cz
        sw = cw
      }

      state.startQuat[qi] = sx
      state.startQuat[qi + 1] = sy
      state.startQuat[qi + 2] = sz
      state.startQuat[qi + 3] = sw
      state.targetQuat[qi] = tx
      state.targetQuat[qi + 1] = ty
      state.targetQuat[qi + 2] = tz
      state.targetQuat[qi + 3] = tw
      state.startTimeMs[idx] = now
      state.durationMs[idx] = dur
      state.active[idx] = 1
    }
  }

  getBoneWorldMatrices(): Float32Array {
    return this.runtimeSkeleton.worldMatrices
  }

  getBoneInverseBindMatrices(): Float32Array {
    return this.skeleton.inverseBindMatrices
  }

  evaluatePose(): void {
    this.updateRotationTweens()
    this.computeWorldMatrices()
  }

  private computeWorldMatrices(): void {
    const bones = this.skeleton.bones
    const localRot = this.runtimeSkeleton.localRotations
    const localTrans = this.runtimeSkeleton.localTranslations
    const worldBuf = this.runtimeSkeleton.worldMatrices
    const computed = this.runtimeSkeleton.computedBones.fill(false)
    const boneCount = bones.length

    if (boneCount === 0) return

    const computeWorld = (i: number): void => {
      if (computed[i]) return

      const b = bones[i]
      if (b.parentIndex >= boneCount) {
        console.warn(`[RZM] bone ${i} parent out of range: ${b.parentIndex}`)
      }

      const qi = i * 4
      let rotateM = Mat4.fromQuat(localRot[qi], localRot[qi + 1], localRot[qi + 2], localRot[qi + 3])
      let addLocalTx = 0,
        addLocalTy = 0,
        addLocalTz = 0

      // Optimized append rotation check - only check necessary conditions
      const appendParentIdx = b.appendParentIndex
      const hasAppend =
        b.appendRotate && appendParentIdx !== undefined && appendParentIdx >= 0 && appendParentIdx < boneCount

      if (hasAppend) {
        const ratio = b.appendRatio === undefined ? 1 : Math.max(-1, Math.min(1, b.appendRatio))
        const hasRatio = Math.abs(ratio) > 1e-6

        if (hasRatio) {
          const apQi = appendParentIdx * 4
          const apTi = appendParentIdx * 3

          if (b.appendRotate) {
            let ax = localRot[apQi]
            let ay = localRot[apQi + 1]
            let az = localRot[apQi + 2]
            const aw = localRot[apQi + 3]
            const absRatio = ratio < 0 ? -ratio : ratio
            if (ratio < 0) {
              ax = -ax
              ay = -ay
              az = -az
            }
            const identityQuat = new Quat(0, 0, 0, 1)
            const appendQuat = new Quat(ax, ay, az, aw)
            const result = Quat.slerp(identityQuat, appendQuat, absRatio)
            const rx = result.x
            const ry = result.y
            const rz = result.z
            const rw = result.w
            rotateM = Mat4.fromQuat(rx, ry, rz, rw).multiply(rotateM)
          }

          if (b.appendMove) {
            const appendRatio = b.appendRatio ?? 1
            addLocalTx = localTrans[apTi] * appendRatio
            addLocalTy = localTrans[apTi + 1] * appendRatio
            addLocalTz = localTrans[apTi + 2] * appendRatio
          }
        }
      }

      // Build local matrix: identity + bind translation, then rotation, then append translation
      this.cachedIdentityMat1
        .setIdentity()
        .translateInPlace(b.bindTranslation[0], b.bindTranslation[1], b.bindTranslation[2])
      this.cachedIdentityMat2.setIdentity().translateInPlace(addLocalTx, addLocalTy, addLocalTz)
      const localM = this.cachedIdentityMat1.multiply(rotateM).multiply(this.cachedIdentityMat2)

      const worldOffset = i * 16
      if (b.parentIndex >= 0) {
        const p = b.parentIndex
        if (!computed[p]) computeWorld(p)
        const parentOffset = p * 16
        // Use cachedIdentityMat2 as temporary buffer for parent * local multiplication
        Mat4.multiplyArrays(worldBuf, parentOffset, localM.values, 0, this.cachedIdentityMat2.values, 0)
        worldBuf.subarray(worldOffset, worldOffset + 16).set(this.cachedIdentityMat2.values)
      } else {
        worldBuf.subarray(worldOffset, worldOffset + 16).set(localM.values)
      }
      computed[i] = true
    }

    // Process all bones (recursion handles dependencies automatically)
    for (let i = 0; i < boneCount; i++) computeWorld(i)
  }
}
