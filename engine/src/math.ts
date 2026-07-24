// Easing function: ease-in-out quadratic
export function easeInOut(t: number): number {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2
}

export class Vec3 {
  x: number
  y: number
  z: number

  constructor(x: number, y: number, z: number) {
    this.x = x
    this.y = y
    this.z = z
  }

  static zeros(): Vec3 {
    return new Vec3(0, 0, 0)
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

  // Normalize this vector in-place (mutates this object)
  normalize(): Vec3 {
    const len = this.length()
    if (len === 0) {
      this.x = 0
      this.y = 0
      this.z = 0
    } else {
      const invLen = 1 / len
      this.x *= invLen
      this.y *= invLen
      this.z *= invLen
    }
    return this
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

  // Set this vector's components from another vector (in-place mutation)
  set(other: Vec3): Vec3 {
    this.x = other.x
    this.y = other.y
    this.z = other.z
    return this
  }

  setXYZ(x: number, y: number, z: number): Vec3 {
    this.x = x
    this.y = y
    this.z = z
    return this
  }

  // out = a - b (no allocation)
  static subtractInto(a: Vec3, b: Vec3, out: Vec3): Vec3 {
    out.x = a.x - b.x
    out.y = a.y - b.y
    out.z = a.z - b.z
    return out
  }

  // out = a × b (no allocation). Safe when out === a or out === b.
  static crossInto(a: Vec3, b: Vec3, out: Vec3): Vec3 {
    const ax = a.x, ay = a.y, az = a.z
    const bx = b.x, by = b.y, bz = b.z
    out.x = ay * bz - az * by
    out.y = az * bx - ax * bz
    out.z = ax * by - ay * bx
    return out
  }

  // Read translation from Mat4 values array (column-major) into out.
  static setFromMat4Translation(m: Float32Array, out: Vec3): Vec3 {
    out.x = m[12]
    out.y = m[13]
    out.z = m[14]
    return out
  }

  // Transform normal by the upper-left 3x3 of a Mat4 (column-major) into out.
  // Safe when out === normal.
  static transformMat4RotationInto(normal: Vec3, m: Float32Array, out: Vec3): Vec3 {
    const nx = normal.x, ny = normal.y, nz = normal.z
    out.x = m[0] * nx + m[4] * ny + m[8] * nz
    out.y = m[1] * nx + m[5] * ny + m[9] * nz
    out.z = m[2] * nx + m[6] * ny + m[10] * nz
    return out
  }

  // In-place normalize returning length squared info via Vec3. Alias for normalize() but explicit.
  normalizeInPlace(): Vec3 {
    return this.normalize()
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

  static identity(): Quat {
    return new Quat(0, 0, 0, 1)
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

  // Conjugate this quaternion in-place (mutates this object)
  // Conjugate (inverse for unit quaternions): (x, y, z, w) -> (-x, -y, -z, w)
  conjugate(): Quat {
    this.x = -this.x
    this.y = -this.y
    this.z = -this.z
    return this
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w)
  }

  // Normalize this quaternion in-place (mutates this object)
  normalize(): Quat {
    const len = this.length()
    if (len === 0) {
      this.x = 0
      this.y = 0
      this.z = 0
      this.w = 1
    } else {
      const invLen = 1 / len
      this.x *= invLen
      this.y *= invLen
      this.z *= invLen
      this.w *= invLen
    }
    return this
  }

  // Static method: create quaternion from rotation axis and angle
  static fromAxisAngle(axis: Vec3, angle: number): Quat {
    // Clone to avoid mutating input, then normalize
    const nx = axis.x
    const ny = axis.y
    const nz = axis.z
    const len = Math.sqrt(nx * nx + ny * ny + nz * nz)
    const invLen = len > 0 ? 1 / len : 0
    const normalizedX = nx * invLen
    const normalizedY = ny * invLen
    const normalizedZ = nz * invLen

    const halfAngle = angle * 0.5
    const sinHalf = Math.sin(halfAngle)
    const cosHalf = Math.cos(halfAngle)
    return new Quat(normalizedX * sinHalf, normalizedY * sinHalf, normalizedZ * sinHalf, cosHalf)
  }

  toArray(): [number, number, number, number] {
    return [this.x, this.y, this.z, this.w]
  }

  // Set this quaternion's components from another quaternion (in-place mutation)
  set(other: Quat): Quat {
    this.x = other.x
    this.y = other.y
    this.z = other.z
    this.w = other.w
    return this
  }

  // Spherical linear interpolation between two quaternions
  static slerp(a: Quat, b: Quat, t: number): Quat {
    let cos = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    let bx = b.x,
      by = b.y,
      bz = b.z,
      bw = b.w

    // If dot product is negative, negate one quaternion to take shorter path
    if (cos < 0) {
      cos = -cos
      bx = -bx
      by = -by
      bz = -bz
      bw = -bw
    }

    // If quaternions are very close, use linear interpolation
    if (cos > 0.9995) {
      const x = a.x + t * (bx - a.x)
      const y = a.y + t * (by - a.y)
      const z = a.z + t * (bz - a.z)
      const w = a.w + t * (bw - a.w)
      const invLen = 1 / Math.hypot(x, y, z, w)
      return new Quat(x * invLen, y * invLen, z * invLen, w * invLen)
    }

    // Standard SLERP
    const theta0 = Math.acos(cos)
    const sinTheta0 = Math.sin(theta0)
    const theta = theta0 * t
    const s0 = Math.sin(theta0 - theta) / sinTheta0
    const s1 = Math.sin(theta) / sinTheta0
    return new Quat(s0 * a.x + s1 * bx, s0 * a.y + s1 * by, s0 * a.z + s1 * bz, s0 * a.w + s1 * bw)
  }

  // out = a * b (quaternion multiplication, rotation composition).
  // Safe when out === a or out === b.
  static multiplyInto(a: Quat, b: Quat, out: Quat): Quat {
    const ax = a.x, ay = a.y, az = a.z, aw = a.w
    const bx = b.x, by = b.y, bz = b.z, bw = b.w
    out.x = aw * bx + ax * bw + ay * bz - az * by
    out.y = aw * by - ax * bz + ay * bw + az * bx
    out.z = aw * bz + ax * by - ay * bx + az * bw
    out.w = aw * bw - ax * bx - ay * by - az * bz
    return out
  }

  // out = quat from axis (unnormalized) and angle.
  static fromAxisAngleInto(ax: number, ay: number, az: number, angle: number, out: Quat): Quat {
    const len = Math.sqrt(ax * ax + ay * ay + az * az)
    const invLen = len > 0 ? 1 / len : 0
    const nx = ax * invLen, ny = ay * invLen, nz = az * invLen
    const half = angle * 0.5
    const s = Math.sin(half), c = Math.cos(half)
    out.x = nx * s
    out.y = ny * s
    out.z = nz * s
    out.w = c
    return out
  }

  // out = slerp(a, b, t). Safe when out === a or out === b.
  static slerpInto(a: Quat, b: Quat, t: number, out: Quat): Quat {
    let cos = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w
    let bx = b.x, by = b.y, bz = b.z, bw = b.w
    if (cos < 0) {
      cos = -cos
      bx = -bx; by = -by; bz = -bz; bw = -bw
    }
    if (cos > 0.9995) {
      const x = a.x + t * (bx - a.x)
      const y = a.y + t * (by - a.y)
      const z = a.z + t * (bz - a.z)
      const w = a.w + t * (bw - a.w)
      const invLen = 1 / Math.hypot(x, y, z, w)
      out.x = x * invLen; out.y = y * invLen; out.z = z * invLen; out.w = w * invLen
      return out
    }
    const theta0 = Math.acos(cos)
    const sinTheta0 = Math.sin(theta0)
    const theta = theta0 * t
    const s0 = Math.sin(theta0 - theta) / sinTheta0
    const s1 = Math.sin(theta) / sinTheta0
    out.x = s0 * a.x + s1 * bx
    out.y = s0 * a.y + s1 * by
    out.z = s0 * a.z + s1 * bz
    out.w = s0 * a.w + s1 * bw
    return out
  }

  setXYZW(x: number, y: number, z: number, w: number): Quat {
    this.x = x; this.y = y; this.z = z; this.w = w
    return this
  }

  setIdentity(): Quat {
    this.x = 0; this.y = 0; this.z = 0; this.w = 1
    return this
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
  // Zero-alloc perspective into an existing column-major array (same values as perspective).
  static perspectiveInto(out: Float32Array, fov: number, aspect: number, near: number, far: number): void {
    const f = 1.0 / Math.tan(fov / 2)
    const rangeInv = 1.0 / (far - near) // Positive for left-handed
    out[0] = f / aspect; out[1] = 0; out[2] = 0; out[3] = 0
    out[4] = 0; out[5] = f; out[6] = 0; out[7] = 0
    out[8] = 0; out[9] = 0; out[10] = (far + near) * rangeInv; out[11] = 1 // Z+ forward (LH)
    out[12] = 0; out[13] = 0; out[14] = -near * far * rangeInv * 2; out[15] = 0
  }

  static perspective(fov: number, aspect: number, near: number, far: number): Mat4 {
    const m = new Mat4(new Float32Array(16))
    Mat4.perspectiveInto(m.values, fov, aspect, near, far)
    return m
  }

  // Zero-alloc LH lookAt into an existing column-major array (scalar math, no Vec3 temps;
  // identical result to lookAt: forward=norm(target-eye), right=norm(up×forward), up=norm(forward×right)).
  static lookAtInto(
    out: Float32Array,
    ex: number, ey: number, ez: number,
    tx: number, ty: number, tz: number,
    ux: number, uy: number, uz: number
  ): void {
    // sqrt(dot) (not hypot) to match Vec3.length()/normalize() bit-for-bit.
    let fx = tx - ex, fy = ty - ey, fz = tz - ez
    const fl = Math.sqrt(fx * fx + fy * fy + fz * fz); const fi = fl > 0 ? 1 / fl : 0
    fx *= fi; fy *= fi; fz *= fi
    let rx = uy * fz - uz * fy, ry = uz * fx - ux * fz, rz = ux * fy - uy * fx
    const rl = Math.sqrt(rx * rx + ry * ry + rz * rz); const ri = rl > 0 ? 1 / rl : 0
    rx *= ri; ry *= ri; rz *= ri
    let vx = fy * rz - fz * ry, vy = fz * rx - fx * rz, vz = fx * ry - fy * rx
    const vl = Math.sqrt(vx * vx + vy * vy + vz * vz); const vi = vl > 0 ? 1 / vl : 0
    vx *= vi; vy *= vi; vz *= vi
    out[0] = rx; out[1] = vx; out[2] = fx; out[3] = 0
    out[4] = ry; out[5] = vy; out[6] = fy; out[7] = 0
    out[8] = rz; out[9] = vz; out[10] = fz; out[11] = 0
    out[12] = -(rx * ex + ry * ey + rz * ez)
    out[13] = -(vx * ex + vy * ey + vz * ez)
    out[14] = -(fx * ex + fy * ey + fz * ez)
    out[15] = 1
  }

  // LookAt matrix for LEFT-HANDED coordinate system (Z+ forward)
  static lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
    const m = new Mat4(new Float32Array(16))
    Mat4.lookAtInto(m.values, eye.x, eye.y, eye.z, target.x, target.y, target.z, up.x, up.y, up.z)
    return m
  }

  // LH ortho, NDC depth 0=near 1=far
  static orthographicLh(left: number, right: number, bottom: number, top: number, near: number, far: number): Mat4 {
    const rl = 1 / (right - left)
    const tb = 1 / (top - bottom)
    const fn = 1 / (far - near)
    return new Mat4(
      new Float32Array([
        2 * rl,
        0,
        0,
        0,
        0,
        2 * tb,
        0,
        0,
        0,
        0,
        fn,
        0,
        -(right + left) * rl,
        -(top + bottom) * tb,
        -near * fn,
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

  // Static method to multiply two matrix array segments directly into output array (no object creation)
  // Column-major multiplication: result = a * b
  static multiplyArrays(
    a: Float32Array,
    aOffset: number,
    b: Float32Array,
    bOffset: number,
    out: Float32Array,
    outOffset: number
  ): void {
    for (let c = 0; c < 4; c++) {
      const b0 = b[bOffset + c * 4 + 0]
      const b1 = b[bOffset + c * 4 + 1]
      const b2 = b[bOffset + c * 4 + 2]
      const b3 = b[bOffset + c * 4 + 3]
      out[outOffset + c * 4 + 0] =
        a[aOffset + 0] * b0 + a[aOffset + 4] * b1 + a[aOffset + 8] * b2 + a[aOffset + 12] * b3
      out[outOffset + c * 4 + 1] =
        a[aOffset + 1] * b0 + a[aOffset + 5] * b1 + a[aOffset + 9] * b2 + a[aOffset + 13] * b3
      out[outOffset + c * 4 + 2] =
        a[aOffset + 2] * b0 + a[aOffset + 6] * b1 + a[aOffset + 10] * b2 + a[aOffset + 14] * b3
      out[outOffset + c * 4 + 3] =
        a[aOffset + 3] * b0 + a[aOffset + 7] * b1 + a[aOffset + 11] * b2 + a[aOffset + 15] * b3
    }
  }

  clone(): Mat4 {
    return new Mat4(this.values.slice())
  }

  // Write rotation matrix from quaternion into existing Float32Array (column-major).
  static fromQuatInto(x: number, y: number, z: number, w: number, out: Float32Array, offset: number = 0): void {
    const x2 = x + x, y2 = y + y, z2 = z + z
    const xx = x * x2, xy = x * y2, xz = x * z2
    const yy = y * y2, yz = y * z2, zz = z * z2
    const wx = w * x2, wy = w * y2, wz = w * z2
    out[offset + 0] = 1 - (yy + zz)
    out[offset + 1] = xy + wz
    out[offset + 2] = xz - wy
    out[offset + 3] = 0
    out[offset + 4] = xy - wz
    out[offset + 5] = 1 - (xx + zz)
    out[offset + 6] = yz + wx
    out[offset + 7] = 0
    out[offset + 8] = xz + wy
    out[offset + 9] = yz - wx
    out[offset + 10] = 1 - (xx + yy)
    out[offset + 11] = 0
    out[offset + 12] = 0
    out[offset + 13] = 0
    out[offset + 14] = 0
    out[offset + 15] = 1
  }

  // Fused local transform: out = T(bindT) · R(quat) · T(localT).
  // Result translation = bindT + R * localT; rotation column block = R.
  // Column-major. Zero allocations.
  static localTransformInto(
    bx: number, by: number, bz: number,
    qx: number, qy: number, qz: number, qw: number,
    lx: number, ly: number, lz: number,
    out: Float32Array
  ): void {
    const x2 = qx + qx, y2 = qy + qy, z2 = qz + qz
    const xx = qx * x2, xy = qx * y2, xz = qx * z2
    const yy = qy * y2, yz = qy * z2, zz = qz * z2
    const wx = qw * x2, wy = qw * y2, wz = qw * z2
    const m00 = 1 - (yy + zz), m01 = xy + wz,       m02 = xz - wy
    const m10 = xy - wz,       m11 = 1 - (xx + zz), m12 = yz + wx
    const m20 = xz + wy,       m21 = yz - wx,       m22 = 1 - (xx + yy)
    out[0] = m00; out[1] = m01; out[2] = m02; out[3] = 0
    out[4] = m10; out[5] = m11; out[6] = m12; out[7] = 0
    out[8] = m20; out[9] = m21; out[10] = m22; out[11] = 0
    out[12] = bx + m00 * lx + m10 * ly + m20 * lz
    out[13] = by + m01 * lx + m11 * ly + m21 * lz
    out[14] = bz + m02 * lx + m12 * ly + m22 * lz
    out[15] = 1
  }

  // Write position+rotation transform into existing Float32Array.
  static fromPositionRotationInto(
    px: number, py: number, pz: number,
    qx: number, qy: number, qz: number, qw: number,
    out: Float32Array
  ): void {
    Mat4.fromQuatInto(qx, qy, qz, qw, out, 0)
    out[12] = px
    out[13] = py
    out[14] = pz
  }

  // Position + rotation + UNIFORM scale. Scales the rotation basis (columns 0–2) by s;
  // uniform scale keeps normals correct after the shader's normalize() (no inverse-transpose).
  static fromPositionRotationScaleInto(
    px: number, py: number, pz: number,
    qx: number, qy: number, qz: number, qw: number,
    s: number,
    out: Float32Array
  ): void {
    Mat4.fromQuatInto(qx, qy, qz, qw, out, 0)
    out[0] *= s; out[1] *= s; out[2] *= s
    out[4] *= s; out[5] *= s; out[6] *= s
    out[8] *= s; out[9] *= s; out[10] *= s
    out[12] = px
    out[13] = py
    out[14] = pz
  }

  // In-place 4x4 inverse into out array. Returns true on success, false if singular (out untouched).
  static inverseInto(m: Float32Array, out: Float32Array): boolean {
    const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3]
    const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7]
    const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11]
    const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15]
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
    if (Math.abs(det) < 1e-10) return false
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
    return true
  }

  // Copy only the rotation (upper-left 3x3) of src into dst, zero out translation, identity w.
  // Column-major in both.
  static copyRotationInto(src: Float32Array, dst: Float32Array): void {
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = 0
    dst[4] = src[4]; dst[5] = src[5]; dst[6] = src[6]; dst[7] = 0
    dst[8] = src[8]; dst[9] = src[9]; dst[10] = src[10]; dst[11] = 0
    dst[12] = 0; dst[13] = 0; dst[14] = 0; dst[15] = 1
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
    return Mat4.toQuatFromArray(this.values, 0)
  }

  // Extract quaternion from matrix array into an existing Quat (no allocation).
  static toQuatFromArrayInto(m: Float32Array, offset: number, out: Quat): Quat {
    const m00 = m[offset + 0], m01 = m[offset + 4], m02 = m[offset + 8]
    const m10 = m[offset + 1], m11 = m[offset + 5], m12 = m[offset + 9]
    const m20 = m[offset + 2], m21 = m[offset + 6], m22 = m[offset + 10]
    const trace = m00 + m11 + m22
    let x = 0, y = 0, z = 0, w = 1
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
    out.x = x * invLen
    out.y = y * invLen
    out.z = z * invLen
    out.w = w * invLen
    return out
  }

  // Static method to extract quaternion from matrix array (avoids creating Mat4 object)
  static toQuatFromArray(m: Float32Array, offset: number): Quat {
    const m00 = m[offset + 0],
      m01 = m[offset + 4],
      m02 = m[offset + 8]
    const m10 = m[offset + 1],
      m11 = m[offset + 5],
      m12 = m[offset + 9]
    const m20 = m[offset + 2],
      m21 = m[offset + 6],
      m22 = m[offset + 10]
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

  // Reset matrix to identity in place
  setIdentity(): this {
    const v = this.values
    v[0] = 1
    v[1] = 0
    v[2] = 0
    v[3] = 0
    v[4] = 0
    v[5] = 1
    v[6] = 0
    v[7] = 0
    v[8] = 0
    v[9] = 0
    v[10] = 1
    v[11] = 0
    v[12] = 0
    v[13] = 0
    v[14] = 0
    v[15] = 1
    return this
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

// Preallocated scratch instances for hot paths. Each subsystem should use its own
// slot to avoid cross-call stomping. Bump the count if more call sites need scratch.
export const scratchMat4Values: Float32Array[] = [
  new Float32Array(16),
  new Float32Array(16),
  new Float32Array(16),
  new Float32Array(16),
  new Float32Array(16),
  new Float32Array(16),
]

export const scratchVec3: Vec3[] = [
  new Vec3(0, 0, 0),
  new Vec3(0, 0, 0),
  new Vec3(0, 0, 0),
  new Vec3(0, 0, 0),
  new Vec3(0, 0, 0),
  new Vec3(0, 0, 0),
  new Vec3(0, 0, 0),
  new Vec3(0, 0, 0),
]

export const scratchQuat: Quat[] = [
  new Quat(0, 0, 0, 1),
  new Quat(0, 0, 0, 1),
  new Quat(0, 0, 0, 1),
  new Quat(0, 0, 0, 1),
]
