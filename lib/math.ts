export class Vec3 {
  x: number
  y: number
  z: number

  constructor(x: number, y: number, z: number) {
    this.x = x
    this.y = y
    this.z = z
  }

  add(other: Vec3): Vec3 {
    return new Vec3(this.x + other.x, this.y + other.y, this.z + other.z)
  }

  subtract(other: Vec3): Vec3 {
    return new Vec3(this.x - other.x, this.y - other.y, this.z - other.z)
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z)
  }

  normalize(): Vec3 {
    const len = this.length()
    if (len === 0) return new Vec3(0, 0, 0)
    return new Vec3(this.x / len, this.y / len, this.z / len)
  }

  cross(other: Vec3): Vec3 {
    return new Vec3(
      this.y * other.z - this.z * other.y,
      this.z * other.x - this.x * other.z,
      this.x * other.y - this.y * other.x
    )
  }

  dot(other: Vec3): number {
    return this.x * other.x + this.y * other.y + this.z * other.z
  }

  scale(scalar: number): Vec3 {
    return new Vec3(this.x * scalar, this.y * scalar, this.z * scalar)
  }

  clone(): Vec3 {
    return new Vec3(this.x, this.y, this.z)
  }
}

export class Quat {
  x: number
  y: number
  z: number
  w: number

  constructor(x: number, y: number, z: number, w: number) {
    this.x = x
    this.y = y
    this.z = z
    this.w = w
  }

  add(other: Quat): Quat {
    return new Quat(this.x + other.x, this.y + other.y, this.z + other.z, this.w + other.w)
  }

  clone(): Quat {
    return new Quat(this.x, this.y, this.z, this.w)
  }

  multiply(other: Quat): Quat {
    // Proper quaternion multiplication (not component-wise)
    return new Quat(
      this.w * other.x + this.x * other.w + this.y * other.z - this.z * other.y,
      this.w * other.y - this.x * other.z + this.y * other.w + this.z * other.x,
      this.w * other.z + this.x * other.y - this.y * other.x + this.z * other.w,
      this.w * other.w - this.x * other.x - this.y * other.y - this.z * other.z
    )
  }

  conjugate(): Quat {
    // Conjugate (inverse for unit quaternions): (x, y, z, w) -> (-x, -y, -z, w)
    return new Quat(-this.x, -this.y, -this.z, this.w)
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w)
  }

  normalize(): Quat {
    const len = this.length()
    if (len === 0) return new Quat(0, 0, 0, 1)
    return new Quat(this.x / len, this.y / len, this.z / len, this.w / len)
  }

  // Rotate a vector by this quaternion: result = q * v * q^-1
  rotateVec(v: Vec3): Vec3 {
    // Treat v as pure quaternion (x, y, z, 0)
    const qx = this.x,
      qy = this.y,
      qz = this.z,
      qw = this.w
    const vx = v.x,
      vy = v.y,
      vz = v.z

    // t = 2 * cross(q.xyz, v)
    const tx = 2 * (qy * vz - qz * vy)
    const ty = 2 * (qz * vx - qx * vz)
    const tz = 2 * (qx * vy - qy * vx)

    // result = v + q.w * t + cross(q.xyz, t)
    return new Vec3(
      vx + qw * tx + (qy * tz - qz * ty),
      vy + qw * ty + (qz * tx - qx * tz),
      vz + qw * tz + (qx * ty - qy * tx)
    )
  }

  // Rotate a vector by this quaternion (Babylon.js style naming)
  rotate(v: Vec3): Vec3 {
    const qv = new Vec3(this.x, this.y, this.z)
    const uv = qv.cross(v)
    const uuv = qv.cross(uv)
    return v.add(uv.scale(2 * this.w)).add(uuv.scale(2))
  }

  // Static method: create quaternion that rotates from one direction to another
  static fromTo(from: Vec3, to: Vec3): Quat {
    const dot = from.dot(to)
    if (dot > 0.999999) return new Quat(0, 0, 0, 1) // Already aligned
    if (dot < -0.999999) {
      // 180 degrees
      let axis = from.cross(new Vec3(1, 0, 0))
      if (axis.length() < 0.001) axis = from.cross(new Vec3(0, 1, 0))
      return new Quat(axis.x, axis.y, axis.z, 0).normalize()
    }

    const axis = from.cross(to)
    const w = Math.sqrt((1 + dot) * 2)
    const invW = 1 / w
    return new Quat(axis.x * invW, axis.y * invW, axis.z * invW, w * 0.5).normalize()
  }

  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w]
  }

  // Convert Euler angles to quaternion (ZXY order, left-handed, PMX format)
  static fromEuler(rotX: number, rotY: number, rotZ: number): Quat {
    const cx = Math.cos(rotX * 0.5)
    const sx = Math.sin(rotX * 0.5)
    const cy = Math.cos(rotY * 0.5)
    const sy = Math.sin(rotY * 0.5)
    const cz = Math.cos(rotZ * 0.5)
    const sz = Math.sin(rotZ * 0.5)

    const w = cy * cx * cz + sy * sx * sz
    const x = cy * sx * cz + sy * cx * sz
    const y = sy * cx * cz - cy * sx * sz
    const z = cy * cx * sz - sy * sx * cz

    return new Quat(x, y, z, w).normalize()
  }

  // Convert quaternion to Euler angles (ZXY order, inverse of fromEuler)
  toEuler(): Vec3 {
    const qx = this.x
    const qy = this.y
    const qz = this.z
    const qw = this.w

    // ZXY order (left-handed)
    // Roll (X): rotation around X axis
    const sinr_cosp = 2 * (qw * qx + qy * qz)
    const cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    const rotX = Math.atan2(sinr_cosp, cosr_cosp)

    // Pitch (Y): rotation around Y axis
    const sinp = 2 * (qw * qy - qz * qx)
    const rotY = Math.abs(sinp) >= 1 ? (sinp >= 0 ? Math.PI / 2 : -Math.PI / 2) : Math.asin(sinp)

    // Yaw (Z): rotation around Z axis
    const siny_cosp = 2 * (qw * qz + qx * qy)
    const cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    const rotZ = Math.atan2(siny_cosp, cosy_cosp)

    return new Vec3(rotX, rotY, rotZ)
  }
}

export class Mat4 {
  values: Float32Array

  constructor(values: Float32Array) {
    this.values = values
  }

  static identity(): Mat4 {
    return new Mat4(new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))
  }

  // Perspective matrix for LEFT-HANDED coordinate system (Z+ forward)
  // For left-handed: Z goes from 0 (near) to 1 (far), +Z is forward
  static perspective(fov: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1.0 / Math.tan(fov / 2)
    const rangeInv = 1.0 / (far - near) // Positive for left-handed

    return new Mat4(
      new Float32Array([
        f / aspect,
        0,
        0,
        0,
        0,
        f,
        0,
        0,
        0,
        0,
        (far + near) * rangeInv,
        1, // Positive for left-handed (Z+ forward)
        0,
        0,
        -near * far * rangeInv * 2, // Negated for left-handed
        0,
      ])
    )
  }

  // LookAt matrix for LEFT-HANDED coordinate system (Z+ forward)
  // For left-handed: camera looks along +Z direction
  static lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
    // In left-handed: forward = target - eye (Z+ direction)
    const forward = target.subtract(eye).normalize()
    const right = up.cross(forward).normalize() // X+ is right
    const upVec = forward.cross(right).normalize() // Y+ is up

    return new Mat4(
      new Float32Array([
        right.x,
        upVec.x,
        forward.x,
        0,
        right.y,
        upVec.y,
        forward.y,
        0,
        right.z,
        upVec.z,
        forward.z,
        0,
        -right.dot(eye),
        -upVec.dot(eye),
        -forward.dot(eye),
        1,
      ])
    )
  }

  multiply(other: Mat4): Mat4 {
    // Column-major multiplication (matches WGSL/GLSL convention):
    // result = a * b
    const out = new Float32Array(16)
    const a = this.values
    const b = other.values
    for (let c = 0; c < 4; c++) {
      const b0 = b[c * 4 + 0]
      const b1 = b[c * 4 + 1]
      const b2 = b[c * 4 + 2]
      const b3 = b[c * 4 + 3]
      out[c * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3
      out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3
      out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3
      out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3
    }
    return new Mat4(out)
  }

  clone(): Mat4 {
    return new Mat4(this.values.slice())
  }

  static fromQuat(x: number, y: number, z: number, w: number): Mat4 {
    // Column-major rotation matrix from quaternion (matches glMatrix/WGSL)
    const out = new Float32Array(16)
    const x2 = x + x,
      y2 = y + y,
      z2 = z + z
    const xx = x * x2,
      xy = x * y2,
      xz = x * z2
    const yy = y * y2,
      yz = y * z2,
      zz = z * z2
    const wx = w * x2,
      wy = w * y2,
      wz = w * z2
    out[0] = 1 - (yy + zz)
    out[1] = xy + wz
    out[2] = xz - wy
    out[3] = 0
    out[4] = xy - wz
    out[5] = 1 - (xx + zz)
    out[6] = yz + wx
    out[7] = 0
    out[8] = xz + wy
    out[9] = yz - wx
    out[10] = 1 - (xx + yy)
    out[11] = 0
    out[12] = 0
    out[13] = 0
    out[14] = 0
    out[15] = 1
    return new Mat4(out)
  }

  // Create transform matrix from position and rotation
  static fromPositionRotation(position: Vec3, rotation: Quat): Mat4 {
    const rotMat = Mat4.fromQuat(rotation.x, rotation.y, rotation.z, rotation.w)
    rotMat.values[12] = position.x
    rotMat.values[13] = position.y
    rotMat.values[14] = position.z
    return rotMat
  }

  // Extract position from transform matrix
  getPosition(): Vec3 {
    return new Vec3(this.values[12], this.values[13], this.values[14])
  }

  // Extract quaternion rotation from this matrix (upper-left 3x3 rotation part)
  toQuat(): Quat {
    const m = this.values
    const m00 = m[0],
      m01 = m[4],
      m02 = m[8]
    const m10 = m[1],
      m11 = m[5],
      m12 = m[9]
    const m20 = m[2],
      m21 = m[6],
      m22 = m[10]
    const trace = m00 + m11 + m22
    let x = 0,
      y = 0,
      z = 0,
      w = 1
    if (trace > 0) {
      const s = Math.sqrt(trace + 1.0) * 2
      w = 0.25 * s
      x = (m21 - m12) / s
      y = (m02 - m20) / s
      z = (m10 - m01) / s
    } else if (m00 > m11 && m00 > m22) {
      const s = Math.sqrt(1.0 + m00 - m11 - m22) * 2
      w = (m21 - m12) / s
      x = 0.25 * s
      y = (m01 + m10) / s
      z = (m02 + m20) / s
    } else if (m11 > m22) {
      const s = Math.sqrt(1.0 + m11 - m00 - m22) * 2
      w = (m02 - m20) / s
      x = (m01 + m10) / s
      y = 0.25 * s
      z = (m12 + m21) / s
    } else {
      const s = Math.sqrt(1.0 + m22 - m00 - m11) * 2
      w = (m10 - m01) / s
      x = (m02 + m20) / s
      y = (m12 + m21) / s
      z = 0.25 * s
    }
    const invLen = 1 / Math.hypot(x, y, z, w)
    return new Quat(x * invLen, y * invLen, z * invLen, w * invLen)
  }

  translateInPlace(tx: number, ty: number, tz: number): this {
    this.values[12] += tx
    this.values[13] += ty
    this.values[14] += tz
    return this
  }

  // Full 4x4 matrix inverse using adjugate method
  // This works for any invertible matrix, not just orthonormal transforms
  // The previous implementation assumed orthonormal rotation matrices, which fails
  // when matrices have scaling or are not perfectly orthonormal (e.g., after
  // bone hierarchy transformations)
  inverse(): Mat4 {
    const m = this.values
    const out = new Float32Array(16)

    const a00 = m[0],
      a01 = m[1],
      a02 = m[2],
      a03 = m[3]
    const a10 = m[4],
      a11 = m[5],
      a12 = m[6],
      a13 = m[7]
    const a20 = m[8],
      a21 = m[9],
      a22 = m[10],
      a23 = m[11]
    const a30 = m[12],
      a31 = m[13],
      a32 = m[14],
      a33 = m[15]

    const b00 = a00 * a11 - a01 * a10
    const b01 = a00 * a12 - a02 * a10
    const b02 = a00 * a13 - a03 * a10
    const b03 = a01 * a12 - a02 * a11
    const b04 = a01 * a13 - a03 * a11
    const b05 = a02 * a13 - a03 * a12
    const b06 = a20 * a31 - a21 * a30
    const b07 = a20 * a32 - a22 * a30
    const b08 = a20 * a33 - a23 * a30
    const b09 = a21 * a32 - a22 * a31
    const b10 = a21 * a33 - a23 * a31
    const b11 = a22 * a33 - a23 * a32

    let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06

    if (Math.abs(det) < 1e-10) {
      console.warn("Matrix is not invertible (determinant near zero)")
      return Mat4.identity()
    }

    det = 1.0 / det

    out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det
    out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det
    out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det
    out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det
    out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det
    out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det
    out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det
    out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det
    out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det
    out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det
    out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det
    out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det
    out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det
    out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det
    out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det
    out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det

    return new Mat4(out)
  }
}
