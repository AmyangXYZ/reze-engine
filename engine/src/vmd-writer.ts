import { AnimationClip, BoneInterpolation, ControlPoint } from "./animation"
import { DEFAULT_CAMERA_INTERPOLATION, type CameraKeyframe } from "./vmd-loader"

const VMD_HEADER = "Vocaloid Motion Data 0002"
const HEADER_SIZE = 30
const MODEL_NAME_SIZE = 20
const BONE_NAME_SIZE = 15
const MORPH_NAME_SIZE = 15
const BONE_FRAME_SIZE = BONE_NAME_SIZE + 4 + 12 + 16 + 64 // 111 bytes
const MORPH_FRAME_SIZE = MORPH_NAME_SIZE + 4 + 4 // 23 bytes
/** IK bone names get 20 bytes in the IK block, not the 15 bones get elsewhere. */
const IK_NAME_SIZE = 20
/** frame + distance + target(3) + rotation(3) + interpolation(24) + fov + perspective. */
const CAMERA_FRAME_SIZE = 4 + 4 + 12 + 12 + 24 + 4 + 1 // 61 bytes
/** What MMD stamps in the model-name field of a camera VMD. Tools sniff it to
 *  tell a camera file from a motion at a glance, so write what they expect. */
const CAMERA_MODEL_NAME = "\u30ab\u30e1\u30e9\u30fb\u7167\u660e"

// Build a Unicode-to-Shift-JIS lookup by inverting the TextDecoder mapping.
let shiftJISTable: Map<string, number[]> | null = null

function getShiftJISTable(): Map<string, number[]> {
  if (shiftJISTable) return shiftJISTable
  const decoder = new TextDecoder("shift-jis")
  const map = new Map<string, number[]>()
  // Single-byte range
  for (let i = 0; i < 256; i++) {
    const char = decoder.decode(new Uint8Array([i]))
    if (char !== "\ufffd") map.set(char, [i])
  }
  // Two-byte range (JIS X 0208)
  for (let hi = 0x81; hi <= 0xfc; hi++) {
    if (hi >= 0xa0 && hi <= 0xdf) continue
    for (let lo = 0x40; lo <= 0xfc; lo++) {
      if (lo === 0x7f) continue
      const char = decoder.decode(new Uint8Array([hi, lo]))
      if (char !== "\ufffd" && !map.has(char)) {
        map.set(char, [hi, lo])
      }
    }
  }
  shiftJISTable = map
  return map
}

function encodeShiftJIS(str: string): Uint8Array {
  const table = getShiftJISTable()
  const bytes: number[] = []
  for (const char of str) {
    const b = table.get(char)
    if (b) bytes.push(...b)
  }
  return new Uint8Array(bytes)
}

/** Which half of a clip to write. Mirrors `Model.loadVmd`'s `tracks` option, so
 *  a file this writer splits out is one the loader can read straight back:
 *
 *    "all"    bone + morph (+ IK) — one file, what MMD itself exports
 *    "motion" bone (+ IK) only — the dance, no expressions
 *    "morphs" morph only — an expression file (\u8868\u60c5\u30e2\u30fc\u30b7\u30e7\u30f3) to lay over a motion
 *
 *  IK rides with "motion" rather than "morphs" because it is bone state: which
 *  chains solve says nothing about a face. */
export type VmdTrackSelection = "all" | "motion" | "morphs"

export class VMDWriter {
  write(clip: AnimationClip, options?: { tracks?: VmdTrackSelection }): ArrayBuffer {
    const tracks = options?.tracks ?? "all"
    const wantBones = tracks !== "morphs"
    const wantMorphs = tracks !== "motion"

    let totalBoneFrames = 0
    if (wantBones) {
      for (const frames of clip.boneTracks.values()) {
        totalBoneFrames += frames.length
      }
    }
    let totalMorphFrames = 0
    if (wantMorphs) {
      for (const frames of clip.morphTracks.values()) {
        totalMorphFrames += frames.length
      }
    }

    // IK state is stored per MOMENT, not per bone: one record lists every chain
    // and its state at that frame. So the tracks are transposed back into the
    // frames they were flattened from.
    const ikByFrame = new Map<number, { boneName: string; enabled: boolean }[]>()
    for (const [boneName, keys] of (wantBones ? clip.ikTracks : undefined) ?? []) {
      for (const key of keys) {
        const at = ikByFrame.get(key.frame)
        if (at) at.push({ boneName, enabled: key.enabled })
        else ikByFrame.set(key.frame, [{ boneName, enabled: key.enabled }])
      }
    }
    const ikFrames = [...ikByFrame.entries()].sort((a, b) => a[0] - b[0])
    let ikSize = 0
    for (const [, states] of ikFrames) ikSize += 4 + 1 + 4 + states.length * (IK_NAME_SIZE + 1)

    const size =
      HEADER_SIZE +
      MODEL_NAME_SIZE +
      4 + totalBoneFrames * BONE_FRAME_SIZE +
      4 + totalMorphFrames * MORPH_FRAME_SIZE +
      // Camera, light, self-shadow and IK counts, ALWAYS — 16 bytes even when
      // every one of them is zero.
      //
      // A VMD's sections are positional: a reader reaches each block by walking
      // every count before it. Writing these only when there was IK state to
      // carry left an ordinary motion ending right after the morph block, so
      // anything looking for the sections that follow read past the end of the
      // file and took whatever it found. MMD's own motion files carry all five
      // counts unconditionally; so do ours now.
      4 * 4 +
      ikSize

    const buffer = new ArrayBuffer(size)
    const view = new DataView(buffer)
    let offset = 0

    // Header (30 bytes, ASCII)
    offset = writeFixedString(buffer, offset, VMD_HEADER, HEADER_SIZE)

    // Model name (20 bytes, zeroed)
    offset += MODEL_NAME_SIZE

    // Bone frame count
    view.setUint32(offset, totalBoneFrames, true)
    offset += 4

    // Bone frames
    for (const frames of wantBones ? clip.boneTracks.values() : []) {
      for (const kf of frames) {
        // Bone name (15 bytes, Shift-JIS)
        offset = writeFixedShiftJIS(buffer, offset, kf.boneName, BONE_NAME_SIZE)

        // Frame number (u32 LE)
        view.setUint32(offset, kf.frame, true)
        offset += 4

        // Translation (3 x f32 LE)
        view.setFloat32(offset, kf.translation.x, true); offset += 4
        view.setFloat32(offset, kf.translation.y, true); offset += 4
        view.setFloat32(offset, kf.translation.z, true); offset += 4

        // Rotation quaternion (4 x f32 LE)
        view.setFloat32(offset, kf.rotation.x, true); offset += 4
        view.setFloat32(offset, kf.rotation.y, true); offset += 4
        view.setFloat32(offset, kf.rotation.z, true); offset += 4
        view.setFloat32(offset, kf.rotation.w, true); offset += 4

        // Interpolation (64 bytes)
        const raw = boneInterpolationToRaw(kf.interpolation)
        new Uint8Array(buffer, offset, 64).set(raw)
        offset += 64
      }
    }

    // Morph frame count
    view.setUint32(offset, totalMorphFrames, true)
    offset += 4

    // Morph frames
    for (const frames of wantMorphs ? clip.morphTracks.values() : []) {
      for (const kf of frames) {
        // Morph name (15 bytes, Shift-JIS)
        offset = writeFixedShiftJIS(buffer, offset, kf.morphName, MORPH_NAME_SIZE)

        // Frame number (u32 LE)
        view.setUint32(offset, kf.frame, true)
        offset += 4

        // Weight (f32 LE)
        view.setFloat32(offset, kf.weight, true)
        offset += 4
      }
    }

    // Empty camera, light and self-shadow blocks, then the IK count — written
    // whether or not there is IK state, so the file always ends on a complete
    // section table.
    for (let i = 0; i < 3; i++, offset += 4) view.setUint32(offset, 0, true)
    view.setUint32(offset, ikFrames.length, true)
    offset += 4

    if (ikFrames.length > 0) {
      for (const [frame, states] of ikFrames) {
        view.setUint32(offset, frame, true)
        offset += 4
        view.setUint8(offset, 1) // model visible
        offset += 1
        view.setUint32(offset, states.length, true)
        offset += 4
        for (const state of states) {
          offset = writeFixedShiftJIS(buffer, offset, state.boneName, IK_NAME_SIZE)
          view.setUint8(offset, state.enabled ? 1 : 0)
          offset += 1
        }
      }
    }

    return buffer
  }

  /**
   * A camera VMD: the shot's own file, with no model motion in it.
   *
   * Every section count is written, including the empty ones. The format is
   * positional, so a reader reaches each block by walking the counts before it —
   * bone and morph to arrive at the camera block, and light, self-shadow and IK
   * to leave it cleanly.
   *
   * The trailing three used to be omitted, on the theory that every reader
   * bounds-checks past the camera block. They do not. This file type is
   * カメラ・照明 — camera AND lighting — so MMD looks for the light block exactly
   * where the file used to stop, read past the end, and lit the scene from
   * whatever it found there. It reached users as a stray coloured light on a
   * camera-only export.
   *
   * `frames` is sorted by frame on the way out: CameraAnimation binary-searches
   * the track it loads, and an out-of-order file would sample wrong rather than
   * fail loudly.
   */
  writeCamera(frames: CameraKeyframe[]): ArrayBuffer {
    const sorted = [...frames].sort((a, b) => a.frame - b.frame)
    const size =
      HEADER_SIZE + MODEL_NAME_SIZE + 4 + 4 + 4 + sorted.length * CAMERA_FRAME_SIZE + 4 * 3
    const buffer = new ArrayBuffer(size)
    const view = new DataView(buffer)
    let offset = 0

    offset = writeFixedString(buffer, offset, VMD_HEADER, HEADER_SIZE)
    offset = writeFixedShiftJIS(buffer, offset, CAMERA_MODEL_NAME, MODEL_NAME_SIZE)

    view.setUint32(offset, 0, true) // bone frame count
    offset += 4
    view.setUint32(offset, 0, true) // morph frame count
    offset += 4
    view.setUint32(offset, sorted.length, true)
    offset += 4

    for (const kf of sorted) {
      view.setUint32(offset, kf.frame, true); offset += 4
      view.setFloat32(offset, kf.distance, true); offset += 4
      view.setFloat32(offset, kf.target.x, true); offset += 4
      view.setFloat32(offset, kf.target.y, true); offset += 4
      view.setFloat32(offset, kf.target.z, true); offset += 4
      // Euler radians, as the loader reads them.
      view.setFloat32(offset, kf.rotation.x, true); offset += 4
      view.setFloat32(offset, kf.rotation.y, true); offset += 4
      view.setFloat32(offset, kf.rotation.z, true); offset += 4
      // 24 bytes, contiguous per channel — see camera-animation.ts's `bez`.
      // Short or missing tables are padded with a linear default rather than
      // writing junk: a hand-built keyframe should not have to know the layout.
      const ip = new Uint8Array(24)
      ip.set(DEFAULT_CAMERA_INTERPOLATION)
      if (kf.interpolation) ip.set(kf.interpolation.subarray(0, 24))
      new Uint8Array(buffer, offset, 24).set(ip)
      offset += 24
      // fov is degrees, and an integer in the file — MMD's own field is u32.
      view.setUint32(offset, Math.max(0, Math.round(kf.fov)), true); offset += 4
      view.setUint8(offset, 0) // 0 = perspective
      offset += 1
    }

    // Light, self-shadow and IK/display counts. See the note above: the light
    // block begins here, and leaving it off is what lit MMD from past the end
    // of the file.
    for (let i = 0; i < 3; i++, offset += 4) view.setUint32(offset, 0, true)

    return buffer
  }
}


function writeFixedString(buffer: ArrayBuffer, offset: number, str: string, maxBytes: number): number {
  const bytes = new Uint8Array(buffer, offset, maxBytes)
  bytes.fill(0)
  for (let i = 0; i < str.length && i < maxBytes; i++) {
    bytes[i] = str.charCodeAt(i) & 0xff
  }
  return offset + maxBytes
}

function writeFixedShiftJIS(buffer: ArrayBuffer, offset: number, str: string, maxBytes: number): number {
  const target = new Uint8Array(buffer, offset, maxBytes)
  target.fill(0)
  const encoded = encodeShiftJIS(str)
  target.set(encoded.subarray(0, maxBytes))
  return offset + maxBytes
}

/**
 * Convert BoneInterpolation back to the 64-byte raw VMD interpolation table.
 * Exact inverse of rawInterpolationToBoneInterpolation in animation.ts — see the
 * layout note there for why the record is written four times.
 */
function boneInterpolationToRaw(interp: BoneInterpolation): Uint8Array {
  // The one 16-byte record: each channel's x1 / y1 / x2 / y2, interleaved.
  const record = new Uint8Array(16)
  const put = (c: number, cp: ControlPoint[]): void => {
    record[c] = cp[0].x
    record[c + 4] = cp[0].y
    record[c + 8] = cp[1].x
    record[c + 12] = cp[1].y
  }
  put(0, interp.translationX)
  put(1, interp.translationY)
  put(2, interp.translationZ)
  put(3, interp.rotation)

  // Copy `r` starts at byte r * 16 and holds record[r..15], so channel r's own
  // bytes land where the reader (and MMD, and babylon-mmd) look for them.
  const raw = new Uint8Array(64)
  for (let r = 0; r < 4; r++) {
    for (let i = r; i < 16; i++) raw[r * 16 + (i - r)] = record[i]
  }
  // Bytes 2 and 3 of the first copy are not interpolation — they are the physics
  // toggle. 0x0000 is "physics on", which is what both reference motions carry and
  // the only state a clip can round-trip: BoneInterpolation does not model the flag.
  raw[2] = 0
  raw[3] = 0

  return raw
}
