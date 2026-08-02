// IK solver (MMD-style; see Saba MMDIkSolver.cpp)

import { Mat4, Quat, Vec3, type EulerOrder } from "./math"
import { Bone, IKLink, IKSolver, IKChainInfo } from "./model"

// Callback type for updating world matrix (provided by model to handle append transformations)
export type UpdateWorldMatrixFn = (boneIndex: number, applyIK: boolean) => void

const enum InternalEulerRotationOrder {
  YXZ = 0,
  ZYX = 1,
  XZY = 2,
}

const enum InternalSolveAxis {
  None = 0,
  Fixed = 1,
  X = 2,
  Y = 3,
  Z = 4,
}

class IKChain {
  readonly boneIndex: number
  readonly minimumAngle: Vec3 | null
  readonly maximumAngle: Vec3 | null
  readonly rotationOrder: InternalEulerRotationOrder
  readonly solveAxis: InternalSolveAxis

  constructor(boneIndex: number, link: IKLink) {
    this.boneIndex = boneIndex

    if (link.hasLimit && link.minAngle && link.maxAngle) {
      // Normalize min/max angles
      const minX = Math.min(link.minAngle.x, link.maxAngle.x)
      const minY = Math.min(link.minAngle.y, link.maxAngle.y)
      const minZ = Math.min(link.minAngle.z, link.maxAngle.z)
      const maxX = Math.max(link.minAngle.x, link.maxAngle.x)
      const maxY = Math.max(link.minAngle.y, link.maxAngle.y)
      const maxZ = Math.max(link.minAngle.z, link.maxAngle.z)
      this.minimumAngle = new Vec3(minX, minY, minZ)
      this.maximumAngle = new Vec3(maxX, maxY, maxZ)

      // Determine rotation order based on constraint ranges
      const halfPi = Math.PI * 0.5
      if (-halfPi < minX && maxX < halfPi) {
        this.rotationOrder = InternalEulerRotationOrder.YXZ
      } else if (-halfPi < minY && maxY < halfPi) {
        this.rotationOrder = InternalEulerRotationOrder.ZYX
      } else {
        this.rotationOrder = InternalEulerRotationOrder.XZY
      }

      // Determine solve axis optimization
      if (minX === 0 && maxX === 0 && minY === 0 && maxY === 0 && minZ === 0 && maxZ === 0) {
        this.solveAxis = InternalSolveAxis.Fixed
      } else if (minY === 0 && maxY === 0 && minZ === 0 && maxZ === 0) {
        this.solveAxis = InternalSolveAxis.X
      } else if (minX === 0 && maxX === 0 && minZ === 0 && maxZ === 0) {
        this.solveAxis = InternalSolveAxis.Y
      } else if (minX === 0 && maxX === 0 && minY === 0 && maxY === 0) {
        this.solveAxis = InternalSolveAxis.Z
      } else {
        this.solveAxis = InternalSolveAxis.None
      }
    } else {
      this.minimumAngle = null
      this.maximumAngle = null
      this.rotationOrder = InternalEulerRotationOrder.XZY // not used
      this.solveAxis = InternalSolveAxis.None
    }
  }
}

// IK-local scratch pool. Safe because solve() runs synchronously and all scratch
// use completes before the updateWorldMatrix callback is invoked.
const _ikVec: Vec3[] = [
  new Vec3(0, 0, 0), new Vec3(0, 0, 0), new Vec3(0, 0, 0), new Vec3(0, 0, 0), new Vec3(0, 0, 0),
  new Vec3(0, 0, 0), new Vec3(0, 0, 0), new Vec3(0, 0, 0), new Vec3(0, 0, 0),
]
const _ikQuat: Quat[] = [
  new Quat(0, 0, 0, 1), new Quat(0, 0, 0, 1), new Quat(0, 0, 0, 1),
]
const _ikMat: Float32Array[] = [
  new Float32Array(16), new Float32Array(16), new Float32Array(16), new Float32Array(16),
]

// An IKChain is derived purely from static link topology (bone index + angle limits +
// solve axis), which never changes at runtime. Build the chain array once per solver and
// reuse it — avoids reallocating the array and re-running the per-link limit/axis setup
// (IKChain ctor) on every solve() every frame.
const _chainCache = new WeakMap<IKSolver, IKChain[]>()

export class IKSolverSystem {
  private static readonly EPSILON = 1.0e-8
  private static readonly THRESHOLD = (88 * Math.PI) / 180

  static solve(
    ikSolvers: IKSolver[],
    bones: Bone[],
    localRotations: Quat[],
    localTranslations: Vec3[],
    worldMatrices: Mat4[],
    ikChainInfo: IKChainInfo[],
    updateWorldMatrix?: UpdateWorldMatrixFn
  ): void {
    for (const solver of ikSolvers) {
      this.solveIK(solver, bones, localRotations, localTranslations, worldMatrices, ikChainInfo, updateWorldMatrix)
    }
  }

  private static solveIK(
    solver: IKSolver,
    bones: Bone[],
    localRotations: Quat[],
    localTranslations: Vec3[],
    worldMatrices: Mat4[],
    ikChainInfo: IKChainInfo[],
    updateWorldMatrix?: UpdateWorldMatrixFn
  ): void {
    if (solver.links.length === 0) return

    const ikBoneIndex = solver.ikBoneIndex
    const targetBoneIndex = solver.targetBoneIndex

    // Reset IK rotations (in place — same object is accumulated into during solve)
    for (const link of solver.links) {
      const chainInfo = ikChainInfo[link.boneIndex]
      if (chainInfo) {
        chainInfo.ikRotation.setXYZW(0, 0, 0, 1)
      }
    }

    if (this.getDistance(ikBoneIndex, targetBoneIndex, worldMatrices) < this.EPSILON) return

    // Build IK chains (cached per solver — static topology, see _chainCache)
    let chains = _chainCache.get(solver)
    if (chains === undefined) {
      chains = solver.links.map((link) => new IKChain(link.boneIndex, link))
      _chainCache.set(solver, chains)
    }

    // Update chain bones and target bone world matrices (initial state, no IK yet)
    if (updateWorldMatrix) {
      for (let i = chains.length - 1; i >= 0; i--) {
        updateWorldMatrix(chains[i].boneIndex, false)
      }
      updateWorldMatrix(targetBoneIndex, false)
    } else {
      for (let i = chains.length - 1; i >= 0; i--) {
        this.updateWorldMatrix(chains[i].boneIndex, bones, localRotations, localTranslations, worldMatrices, undefined)
      }
      this.updateWorldMatrix(targetBoneIndex, bones, localRotations, localTranslations, worldMatrices, undefined)
    }

    if (this.getDistance(ikBoneIndex, targetBoneIndex, worldMatrices) < this.EPSILON) return

    // Solve iteratively
    const iteration = Math.min(solver.iterationCount, 256)
    const halfIteration = iteration >> 1

    for (let i = 0; i < iteration; i++) {
      for (let chainIndex = 0; chainIndex < chains.length; chainIndex++) {
        const chain = chains[chainIndex]
        if (chain.solveAxis !== InternalSolveAxis.Fixed) {
          this.solveChain(
            chain,
            chainIndex,
            solver,
            ikBoneIndex,
            targetBoneIndex,
            bones,
            localRotations,
            localTranslations,
            worldMatrices,
            ikChainInfo,
            i < halfIteration,
            updateWorldMatrix
          )
        }
      }

      if (this.getDistance(ikBoneIndex, targetBoneIndex, worldMatrices) < this.EPSILON) break
    }

    // Apply IK rotations to local rotations (mutate localRot in place)
    for (const link of solver.links) {
      const chainInfo = ikChainInfo[link.boneIndex]
      if (chainInfo?.ikRotation) {
        const localRot = localRotations[link.boneIndex]
        Quat.multiplyInto(chainInfo.ikRotation, localRot, localRot)
        localRot.normalize()
      }
    }
  }

  private static solveChain(
    chain: IKChain,
    chainIndex: number,
    solver: IKSolver,
    ikBoneIndex: number,
    targetBoneIndex: number,
    bones: Bone[],
    localRotations: Quat[],
    localTranslations: Vec3[],
    worldMatrices: Mat4[],
    ikChainInfo: IKChainInfo[],
    useAxis: boolean,
    updateWorldMatrix?: UpdateWorldMatrixFn
  ): void {
    const chainBoneIndex = chain.boneIndex

    // scratch layout:
    //   _ikVec[0]=chainPos, [1]=ikPos, [2]=targetPos, [3]=chainTargetVec, [4]=chainIkVec,
    //           [5]=rotAxis, [6]=finalAxis, [7]=eulerTmp, [8]=limitedEuler
    //   _ikMat[0]=parentRot, [1]=invParentRot, [2]=quatToMatTmp
    //   _ikQuat[0]=ikRotation, [1]=combinedRot, [2]=localRotConj

    const chainPos = Vec3.setFromMat4Translation(worldMatrices[chainBoneIndex].values, _ikVec[0])
    const ikPos = Vec3.setFromMat4Translation(worldMatrices[ikBoneIndex].values, _ikVec[1])
    const targetPos = Vec3.setFromMat4Translation(worldMatrices[targetBoneIndex].values, _ikVec[2])

    const chainTargetVec = Vec3.subtractInto(chainPos, targetPos, _ikVec[3])
    chainTargetVec.normalize()
    const chainIkVec = Vec3.subtractInto(chainPos, ikPos, _ikVec[4])
    chainIkVec.normalize()

    const rotAxis = Vec3.crossInto(chainTargetVec, chainIkVec, _ikVec[5])
    if (rotAxis.length() < this.EPSILON) return

    // Parent world rotation matrix (translation removed) into _ikMat[0]
    this.getParentWorldRotationMatrixInto(chainBoneIndex, bones, worldMatrices, _ikMat[0])

    let finalAxis: Vec3
    if (chain.minimumAngle !== null && useAxis) {
      switch (chain.solveAxis) {
        case InternalSolveAxis.None: {
          if (!Mat4.inverseInto(_ikMat[0], _ikMat[1])) {
            finalAxis = rotAxis
          } else {
            finalAxis = Vec3.transformMat4RotationInto(rotAxis, _ikMat[1], _ikVec[6])
            finalAxis.normalize()
          }
          break
        }
        case InternalSolveAxis.X:
        case InternalSolveAxis.Y:
        case InternalSolveAxis.Z: {
          const m = _ikMat[0]
          const axisOffset = (chain.solveAxis - InternalSolveAxis.X) * 4
          const ax = m[axisOffset], ay = m[axisOffset + 1], az = m[axisOffset + 2]
          const dotA = rotAxis.x * ax + rotAxis.y * ay + rotAxis.z * az
          const sign = dotA >= 0 ? 1 : -1
          finalAxis =
            chain.solveAxis === InternalSolveAxis.X
              ? _ikVec[6].setXYZ(sign, 0, 0)
              : chain.solveAxis === InternalSolveAxis.Y
              ? _ikVec[6].setXYZ(0, sign, 0)
              : _ikVec[6].setXYZ(0, 0, sign)
          break
        }
        default:
          finalAxis = rotAxis
      }
    } else {
      if (!Mat4.inverseInto(_ikMat[0], _ikMat[1])) {
        finalAxis = rotAxis
      } else {
        finalAxis = Vec3.transformMat4RotationInto(rotAxis, _ikMat[1], _ikVec[6])
        finalAxis.normalize()
      }
    }

    let dotTI = chainTargetVec.dot(chainIkVec)
    dotTI = Math.max(-1.0, Math.min(1.0, dotTI))

    const angle = Math.min(solver.limitAngle * (chainIndex + 1), Math.acos(dotTI))
    const ikRotation = Quat.fromAxisAngleInto(finalAxis.x, finalAxis.y, finalAxis.z, angle, _ikQuat[0])

    const chainInfo = ikChainInfo[chainBoneIndex]
    if (chainInfo) {
      // chainInfo.ikRotation = ikRotation * chainInfo.ikRotation (in place)
      Quat.multiplyInto(ikRotation, chainInfo.ikRotation, chainInfo.ikRotation)

      if (chain.minimumAngle && chain.maximumAngle) {
        const localRot = localRotations[chainBoneIndex]
        chainInfo.localRotation.set(localRot)

        // combinedRot = chainInfo.ikRotation * localRot
        Quat.multiplyInto(chainInfo.ikRotation, localRot, _ikQuat[1])
        // extract euler into _ikVec[7]
        this.extractEulerAnglesInto(_ikQuat[1], chain.rotationOrder, _ikVec[7])
        // limit into _ikVec[8]
        this.limitEulerAnglesInto(_ikVec[7], chain.minimumAngle, chain.maximumAngle, useAxis, _ikVec[8])
        // reconstruct quat into chainInfo.ikRotation (uses _ikQuat[3..5] as tmp)
        this.reconstructQuatFromEulerInto(_ikVec[8], chain.rotationOrder, chainInfo.ikRotation)
        // localRot conjugate into _ikQuat[2] (localRot is unit, so conjugate == inverse)
        _ikQuat[2].setXYZW(-localRot.x, -localRot.y, -localRot.z, localRot.w)
        // chainInfo.ikRotation *= localRotConj
        Quat.multiplyInto(chainInfo.ikRotation, _ikQuat[2], chainInfo.ikRotation)
      }
    }

    // Update world matrices for affected bones (using callback - handles append correctly)
    if (updateWorldMatrix) {
      for (let i = chainIndex; i >= 0; i--) {
        const link = solver.links[i]
        updateWorldMatrix(link.boneIndex, true)  // applyIK = true
      }
      updateWorldMatrix(targetBoneIndex, false)
    } else {
      for (let i = chainIndex; i >= 0; i--) {
        const link = solver.links[i]
        this.updateWorldMatrix(link.boneIndex, bones, localRotations, localTranslations, worldMatrices, ikChainInfo)
      }
      this.updateWorldMatrix(ikBoneIndex, bones, localRotations, localTranslations, worldMatrices, undefined)
      this.updateWorldMatrix(targetBoneIndex, bones, localRotations, localTranslations, worldMatrices, undefined)
    }
  }

  private static limitAngle(angle: number, min: number, max: number, useAxis: boolean): number {
    if (angle < min) {
      const diff = 2 * min - angle
      return diff <= max && useAxis ? diff : min
    } else if (angle > max) {
      const diff = 2 * max - angle
      return diff >= min && useAxis ? diff : max
    } else {
      return angle
    }
  }

  private static getDistance(boneIndex1: number, boneIndex2: number, worldMatrices: Mat4[]): number {
    const m1 = worldMatrices[boneIndex1].values
    const m2 = worldMatrices[boneIndex2].values
    const dx = m1[12] - m2[12]
    const dy = m1[13] - m2[13]
    const dz = m1[14] - m2[14]
    return Math.sqrt(dx * dx + dy * dy + dz * dz)
  }

  // Maps InternalEulerRotationOrder to the shared math-layer order names.
  private static readonly EULER_ORDER_NAMES: readonly EulerOrder[] = ["YXZ", "ZYX", "XZY"]

  private static extractEulerAnglesInto(quat: Quat, order: InternalEulerRotationOrder, out: Vec3): void {
    Mat4.fromQuatInto(quat.x, quat.y, quat.z, quat.w, _ikMat[2], 0)
    const m = _ikMat[2]

    switch (order) {
      case InternalEulerRotationOrder.YXZ: {
        let rX = Math.asin(-m[9])
        if (Math.abs(rX) > this.THRESHOLD) rX = rX < 0 ? -this.THRESHOLD : this.THRESHOLD
        let cosX = Math.cos(rX)
        if (cosX !== 0) cosX = 1 / cosX
        out.x = rX
        out.y = Math.atan2(m[8] * cosX, m[10] * cosX)
        out.z = Math.atan2(m[1] * cosX, m[5] * cosX)
        return
      }
      case InternalEulerRotationOrder.ZYX: {
        let rY = Math.asin(-m[2])
        if (Math.abs(rY) > this.THRESHOLD) rY = rY < 0 ? -this.THRESHOLD : this.THRESHOLD
        let cosY = Math.cos(rY)
        if (cosY !== 0) cosY = 1 / cosY
        out.x = Math.atan2(m[6] * cosY, m[10] * cosY)
        out.y = rY
        out.z = Math.atan2(m[1] * cosY, m[0] * cosY)
        return
      }
      case InternalEulerRotationOrder.XZY: {
        let rZ = Math.asin(-m[4])
        if (Math.abs(rZ) > this.THRESHOLD) rZ = rZ < 0 ? -this.THRESHOLD : this.THRESHOLD
        let cosZ = Math.cos(rZ)
        if (cosZ !== 0) cosZ = 1 / cosZ
        out.x = Math.atan2(m[6] * cosZ, m[5] * cosZ)
        out.y = Math.atan2(m[8] * cosZ, m[0] * cosZ)
        out.z = rZ
        return
      }
    }
  }

  private static limitEulerAnglesInto(euler: Vec3, min: Vec3, max: Vec3, useAxis: boolean, out: Vec3): void {
    out.x = this.limitAngle(euler.x, min.x, max.x, useAxis)
    out.y = this.limitAngle(euler.y, min.y, max.y, useAxis)
    out.z = this.limitAngle(euler.z, min.z, max.z, useAxis)
  }

  private static reconstructQuatFromEulerInto(euler: Vec3, order: InternalEulerRotationOrder, out: Quat): void {
    Quat.fromEulerOrderInto(euler.x, euler.y, euler.z, this.EULER_ORDER_NAMES[order], out)
  }

  // Write parent's world rotation (translation stripped) into out Float32Array.
  private static getParentWorldRotationMatrixInto(
    boneIndex: number,
    bones: Bone[],
    worldMatrices: Mat4[],
    out: Float32Array
  ): void {
    const bone = bones[boneIndex]
    if (bone.parentIndex >= 0) {
      Mat4.copyRotationInto(worldMatrices[bone.parentIndex].values, out)
    } else {
      // Identity
      out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0
      out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0
      out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0
      out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1
    }
  }

  private static updateWorldMatrix(
    boneIndex: number,
    bones: Bone[],
    localRotations: Quat[],
    localTranslations: Vec3[],
    worldMatrices: Mat4[],
    ikChainInfo?: IKChainInfo[]
  ): void {
    const bone = bones[boneIndex]
    const localRot = localRotations[boneIndex]
    const localTrans = localTranslations[boneIndex]

    let fx = localRot.x, fy = localRot.y, fz = localRot.z, fw = localRot.w
    if (ikChainInfo) {
      const chainInfo = ikChainInfo[boneIndex]
      if (chainInfo && chainInfo.ikRotation) {
        const ik = chainInfo.ikRotation
        const nx = ik.w * fx + ik.x * fw + ik.y * fz - ik.z * fy
        const ny = ik.w * fy - ik.x * fz + ik.y * fw + ik.z * fx
        const nz = ik.w * fz + ik.x * fy - ik.y * fx + ik.z * fw
        const nw = ik.w * fw - ik.x * fx - ik.y * fy - ik.z * fz
        const len = Math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
        const inv = len > 0 ? 1 / len : 0
        fx = nx * inv; fy = ny * inv; fz = nz * inv; fw = nw * inv
      }
    }

    const localMVals = _ikMat[3]
    Mat4.localTransformInto(
      bone.bindTranslation[0], bone.bindTranslation[1], bone.bindTranslation[2],
      fx, fy, fz, fw,
      localTrans.x, localTrans.y, localTrans.z,
      localMVals
    )

    const worldMat = worldMatrices[boneIndex]
    if (bone.parentIndex >= 0) {
      const parentMat = worldMatrices[bone.parentIndex]
      Mat4.multiplyArrays(parentMat.values, 0, localMVals, 0, worldMat.values, 0)
    } else {
      worldMat.values.set(localMVals)
    }
  }
}

