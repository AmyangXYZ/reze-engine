import { Quat, Vec3 } from "./math"

interface BoneFrame {
  boneName: string
  frame: number
  rotation: Quat
  translation: Vec3
  interpolation: Uint8Array // 64 bytes of interpolation parameters
}

interface MorphFrame {
  morphName: string
  frame: number
  weight: number // 0.0 to 1.0
}

export interface VMDKeyFrame {
  time: number // in seconds
  boneFrames: BoneFrame[]
  morphFrames: MorphFrame[]
}

/** One MMD camera keyframe. The camera looks at `target` from `distance` away, oriented by
 *  `rotation` (euler radians), with `fov` degrees. `interpolation` is 24 bytes: 6 channels
 *  (posX, posY, posZ, rotation, distance, fov) × 4 bezier bytes (x1, x2, y1, y2). */
export interface CameraKeyframe {
  frame: number
  distance: number
  target: Vec3
  rotation: Vec3 // euler radians
  fov: number // degrees
  interpolation: Uint8Array // 24 bytes
}

/** A VMD "IK/display" record: one moment at which chains are switched. */
export interface IkFrame {
  frame: number
  /** Model visibility rides in the same record. Parsed, unused by playback. */
  visible: boolean
  states: { boneName: string; enabled: boolean }[]
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

  /** Parse only the camera track (a dedicated camera VMD, or the camera block of any VMD).
   *  Returns [] if the file has no camera block (motion-only VMDs end after the morph block). */
  static async loadCamera(url: string): Promise<CameraKeyframe[]> {
    const loader = new VMDLoader(await fetch(url).then((r) => r.arrayBuffer()))
    return loader.parseCamera()
  }

  static loadCameraFromBuffer(buffer: ArrayBuffer): CameraKeyframe[] {
    return new VMDLoader(buffer).parseCamera()
  }

  // Seek past the bone + morph blocks (fixed record sizes) to the camera block, then read it.
  // bone frame = 111 B (15 name + 4 frame + 12 pos + 16 rot + 64 interp); morph = 23 B
  // (15 name + 4 frame + 4 weight); camera = 61 B.
  private parseCamera(): CameraKeyframe[] {
    this.offset = 0
    const header = this.getString(30)
    if (!header.startsWith("Vocaloid Motion Data")) throw new Error("Invalid VMD file header")
    this.skip(20)
    const boneCount = this.getUint32()
    this.skip(boneCount * 111)
    const morphCount = this.getUint32()
    this.skip(morphCount * 23)
    if (this.offset + 4 > this.view.buffer.byteLength) return [] // no camera block
    const cameraCount = this.getUint32()

    const frames: CameraKeyframe[] = []
    for (let i = 0; i < cameraCount; i++) {
      const frame = this.getUint32()
      const distance = this.getFloat32()
      const target = new Vec3(this.getFloat32(), this.getFloat32(), this.getFloat32())
      const rotation = new Vec3(this.getFloat32(), this.getFloat32(), this.getFloat32())
      const interpolation = new Uint8Array(24)
      for (let j = 0; j < 24; j++) interpolation[j] = this.getUint8()
      const fov = this.getUint32()
      this.skip(1) // perspective flag (0 = perspective) — unused
      frames.push({ frame, distance, target, rotation, fov, interpolation })
    }
    frames.sort((a, b) => a.frame - b.frame)
    return frames
  }

  static loadIkFromBuffer(buffer: ArrayBuffer): IkFrame[] {
    return new VMDLoader(buffer).parseIk()
  }

  /**
   * The IK/display block, at the very end of the file.
   *
   * MMD writes it after camera, light and self-shadow, and a motion-only VMD
   * often stops before any of them — so every block is bounds-checked and a
   * short file simply reports no IK data rather than throwing. Record sizes:
   * bone 111 B, morph 23 B, camera 61 B, light 28 B, self-shadow 9 B.
   */
  private parseIk(): IkFrame[] {
    this.offset = 0
    const header = this.getString(30)
    if (!header.startsWith("Vocaloid Motion Data")) throw new Error("Invalid VMD file header")
    this.skip(20)
    const end = this.view.buffer.byteLength
    const seek = (size: number): boolean => {
      if (this.offset + 4 > end) return false
      this.skip(this.getUint32() * size)
      return this.offset <= end
    }
    if (!seek(111) || !seek(23) || !seek(61) || !seek(28) || !seek(9)) return []
    if (this.offset + 4 > end) return []

    const count = this.getUint32()
    const frames: IkFrame[] = []
    for (let i = 0; i < count && this.offset + 9 <= end; i++) {
      const frame = this.getUint32()
      const visible = this.getUint8() !== 0
      const ikCount = this.getUint32()
      const states: { boneName: string; enabled: boolean }[] = []
      for (let j = 0; j < ikCount && this.offset + 21 <= end; j++) {
        states.push({ boneName: this.getShiftJisName(20), enabled: this.getUint8() !== 0 })
      }
      frames.push({ frame, visible, states })
    }
    frames.sort((a, b) => a.frame - b.frame)
    return frames
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

      // Convert frame number to time (30 FPS)
      const FRAME_RATE = 30.0
      const time = boneFrame.frame / FRAME_RATE

      allBoneFrames.push({ time, boneFrame })
    }

    // Read morph frame count (4 bytes, u32 little endian)
    const morphFrameCount = this.getUint32()

    // Read all morph frames
    const allMorphFrames: Array<{ time: number; morphFrame: MorphFrame }> = []

    for (let i = 0; i < morphFrameCount; i++) {
      const morphFrame = this.readMorphFrame()

      // Convert frame number to time (30 FPS)
      const FRAME_RATE = 30.0
      const time = morphFrame.frame / FRAME_RATE

      allMorphFrames.push({ time, morphFrame })
    }

    // Combine all frames and group by time
    const allFrames: Array<{ time: number; boneFrame?: BoneFrame; morphFrame?: MorphFrame }> = []
    for (const { time, boneFrame } of allBoneFrames) {
      allFrames.push({ time, boneFrame })
    }
    for (const { time, morphFrame } of allMorphFrames) {
      allFrames.push({ time, morphFrame })
    }

    // Sort by time
    allFrames.sort((a, b) => a.time - b.time)

    // Group by time and convert to VMDKeyFrame format
    const keyFrames: VMDKeyFrame[] = []
    let currentTime = -1.0
    let currentBoneFrames: BoneFrame[] = []
    let currentMorphFrames: MorphFrame[] = []

    for (const frame of allFrames) {
      if (Math.abs(frame.time - currentTime) > 0.001) {
        // New time frame
        if (currentBoneFrames.length > 0 || currentMorphFrames.length > 0) {
          keyFrames.push({
            time: currentTime,
            boneFrames: currentBoneFrames,
            morphFrames: currentMorphFrames,
          })
        }
        currentTime = frame.time
        currentBoneFrames = frame.boneFrame ? [frame.boneFrame] : []
        currentMorphFrames = frame.morphFrame ? [frame.morphFrame] : []
      } else {
        // Same time frame
        if (frame.boneFrame) {
          currentBoneFrames.push(frame.boneFrame)
        }
        if (frame.morphFrame) {
          currentMorphFrames.push(frame.morphFrame)
        }
      }
    }

    // Add the last frame
    if (currentBoneFrames.length > 0 || currentMorphFrames.length > 0) {
      keyFrames.push({
        time: currentTime,
        boneFrames: currentBoneFrames,
        morphFrames: currentMorphFrames,
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

    // Read position/translation (12 bytes: 3 x f32, little endian)
    const posX = this.getFloat32()
    const posY = this.getFloat32()
    const posZ = this.getFloat32()
    const translation = new Vec3(posX, posY, posZ)

    // Read rotation quaternion (16 bytes: 4 x f32, little endian)
    const rotX = this.getFloat32()
    const rotY = this.getFloat32()
    const rotZ = this.getFloat32()
    const rotW = this.getFloat32()
    const rotation = new Quat(rotX, rotY, rotZ, rotW)

    // Read interpolation parameters (64 bytes)
    const interpolation = new Uint8Array(64)
    for (let i = 0; i < 64; i++) {
      interpolation[i] = this.getUint8()
    }

    return {
      boneName,
      frame,
      rotation,
      translation,
      interpolation,
    }
  }

  private readMorphFrame(): MorphFrame {
    // Read morph name (15 bytes)
    const nameBuffer = new Uint8Array(this.view.buffer, this.offset, 15)
    this.offset += 15

    // Find the actual length of the morph name (stop at first null byte)
    let nameLength = 15
    for (let i = 0; i < 15; i++) {
      if (nameBuffer[i] === 0) {
        nameLength = i
        break
      }
    }

    // Decode Shift-JIS morph name
    let morphName: string
    try {
      const nameSlice = nameBuffer.slice(0, nameLength)
      morphName = this.decoder.decode(nameSlice)
    } catch {
      // Fallback to lossy decoding if there were encoding errors
      morphName = String.fromCharCode(...nameBuffer.slice(0, nameLength))
    }

    // Read frame number (4 bytes, little endian)
    const frame = this.getUint32()

    // Read weight (4 bytes, f32, little endian)
    const weight = this.getFloat32()

    return {
      morphName,
      frame,
      weight,
    }
  }

  private getUint8(): number {
    if (this.offset + 1 > this.view.buffer.byteLength) {
      throw new RangeError(`Offset ${this.offset} + 1 exceeds buffer bounds ${this.view.buffer.byteLength}`)
    }
    const v = this.view.getUint8(this.offset)
    this.offset += 1
    return v
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

  /** A fixed-width, NUL-padded Shift-JIS name — the form every VMD name takes. */
  private getShiftJisName(len: number): string {
    const bytes = new Uint8Array(this.view.buffer, this.offset, len)
    this.offset += len
    let end = bytes.indexOf(0)
    if (end < 0) end = len
    const slice = bytes.slice(0, end)
    try {
      return this.decoder.decode(slice)
    } catch {
      return String.fromCharCode(...slice)
    }
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
