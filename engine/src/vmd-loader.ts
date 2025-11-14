import { Quat } from "./math"

export interface BoneFrame {
  boneName: string
  frame: number
  rotation: Quat
}

export interface VMDKeyFrame {
  time: number // in seconds
  boneFrames: BoneFrame[]
}

export class VMDLoader {
  private view: DataView
  private offset = 0
  private decoder: TextDecoder

  private constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer)
    // Try to use Shift-JIS decoder, fallback to UTF-8 if not available
    try {
      this.decoder = new TextDecoder("shift-jis")
    } catch {
      // Fallback to UTF-8 if Shift-JIS is not supported
      this.decoder = new TextDecoder("utf-8")
    }
  }

  static async load(url: string): Promise<VMDKeyFrame[]> {
    const loader = new VMDLoader(await fetch(url).then((r) => r.arrayBuffer()))
    return loader.parse()
  }

  static loadFromBuffer(buffer: ArrayBuffer): VMDKeyFrame[] {
    const loader = new VMDLoader(buffer)
    return loader.parse()
  }

  private parse(): VMDKeyFrame[] {
    // Read header (30 bytes)
    const header = this.getString(30)
    if (!header.startsWith("Vocaloid Motion Data")) {
      throw new Error("Invalid VMD file header")
    }

    // Skip model name (20 bytes)
    this.skip(20)

    // Read bone frame count (4 bytes, u32 little endian)
    const boneFrameCount = this.getUint32()

    // Read all bone frames
    const allBoneFrames: Array<{ time: number; boneFrame: BoneFrame }> = []

    for (let i = 0; i < boneFrameCount; i++) {
      const boneFrame = this.readBoneFrame()

      // Convert frame number to time (assuming 30 FPS like the Rust code)
      const FRAME_RATE = 30.0
      const time = boneFrame.frame / FRAME_RATE

      allBoneFrames.push({ time, boneFrame })
    }

    // Group by time and convert to VMDKeyFrame format
    // Sort by time first
    allBoneFrames.sort((a, b) => a.time - b.time)

    const keyFrames: VMDKeyFrame[] = []
    let currentTime = -1.0
    let currentBoneFrames: BoneFrame[] = []

    for (const { time, boneFrame } of allBoneFrames) {
      if (Math.abs(time - currentTime) > 0.001) {
        // New time frame
        if (currentBoneFrames.length > 0) {
          keyFrames.push({
            time: currentTime,
            boneFrames: currentBoneFrames,
          })
        }
        currentTime = time
        currentBoneFrames = [boneFrame]
      } else {
        // Same time frame
        currentBoneFrames.push(boneFrame)
      }
    }

    // Add the last frame
    if (currentBoneFrames.length > 0) {
      keyFrames.push({
        time: currentTime,
        boneFrames: currentBoneFrames,
      })
    }

    return keyFrames
  }

  private readBoneFrame(): BoneFrame {
    // Read bone name (15 bytes)
    const nameBuffer = new Uint8Array(this.view.buffer, this.offset, 15)
    this.offset += 15

    // Find the actual length of the bone name (stop at first null byte)
    let nameLength = 15
    for (let i = 0; i < 15; i++) {
      if (nameBuffer[i] === 0) {
        nameLength = i
        break
      }
    }

    // Decode Shift-JIS bone name
    let boneName: string
    try {
      const nameSlice = nameBuffer.slice(0, nameLength)
      boneName = this.decoder.decode(nameSlice)
    } catch {
      // Fallback to lossy decoding if there were encoding errors
      boneName = String.fromCharCode(...nameBuffer.slice(0, nameLength))
    }

    // Read frame number (4 bytes, little endian)
    const frame = this.getUint32()

    // Skip position (12 bytes: 3 x f32, little endian)
    this.skip(12)

    // Read rotation quaternion (16 bytes: 4 x f32, little endian)
    const rotX = this.getFloat32()
    const rotY = this.getFloat32()
    const rotZ = this.getFloat32()
    const rotW = this.getFloat32()
    const rotation = new Quat(rotX, rotY, rotZ, rotW)

    // Skip interpolation parameters (64 bytes)
    this.skip(64)

    return {
      boneName,
      frame,
      rotation,
    }
  }

  private getUint32(): number {
    if (this.offset + 4 > this.view.buffer.byteLength) {
      throw new RangeError(`Offset ${this.offset} + 4 exceeds buffer bounds ${this.view.buffer.byteLength}`)
    }
    const v = this.view.getUint32(this.offset, true) // true = little endian
    this.offset += 4
    return v
  }

  private getFloat32(): number {
    if (this.offset + 4 > this.view.buffer.byteLength) {
      throw new RangeError(`Offset ${this.offset} + 4 exceeds buffer bounds ${this.view.buffer.byteLength}`)
    }
    const v = this.view.getFloat32(this.offset, true) // true = little endian
    this.offset += 4
    return v
  }

  private getString(len: number): string {
    const bytes = new Uint8Array(this.view.buffer, this.offset, len)
    this.offset += len
    return String.fromCharCode(...bytes)
  }

  private skip(bytes: number): void {
    if (this.offset + bytes > this.view.buffer.byteLength) {
      throw new RangeError(`Offset ${this.offset} + ${bytes} exceeds buffer bounds ${this.view.buffer.byteLength}`)
    }
    this.offset += bytes
  }
}
