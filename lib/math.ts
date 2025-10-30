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
    return new Quat(this.x * other.x, this.y * other.y, this.z * other.z, this.w * other.w)
  }

  length(): number {
    return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w)
  }

  normalize(): Quat {
    const len = this.length()
    if (len === 0) return new Quat(0, 0, 0, 1)
    return new Quat(this.x / len, this.y / len, this.z / len, this.w / len)
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

  static perspective(fov: number, aspect: number, near: number, far: number): Mat4 {
    const f = 1.0 / Math.tan(fov / 2)
    const rangeInv = 1.0 / (near - far)

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
        (near + far) * rangeInv,
        -1,
        0,
        0,
        near * far * rangeInv * 2,
        0,
      ])
    )
  }

  static lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
    const zAxis = eye.subtract(target).normalize()
    const xAxis = up.cross(zAxis).normalize()
    const yAxis = zAxis.cross(xAxis)

    return new Mat4(
      new Float32Array([
        xAxis.x,
        yAxis.x,
        zAxis.x,
        0,
        xAxis.y,
        yAxis.y,
        zAxis.y,
        0,
        xAxis.z,
        yAxis.z,
        zAxis.z,
        0,
        -xAxis.dot(eye),
        -yAxis.dot(eye),
        -zAxis.dot(eye),
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

  translateInPlace(tx: number, ty: number, tz: number): this {
    this.values[12] += tx
    this.values[13] += ty
    this.values[14] += tz
    return this
  }
}
