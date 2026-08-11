// DDS → RGBA8, decoded on the CPU.
//
// Stage models converted out of games ship their textures as DDS far more often
// than MMD character models do, and the browser cannot read one: createImageBitmap
// refuses it, and the TGA fallback reads the header as garbage and reports a
// height of zero. Every such material fell back to white.
//
// Decoded rather than uploaded as-is on purpose. WebGPU can sample BC formats
// natively, but only where the texture-compression-bc feature was requested at
// device creation — it is absent on much mobile hardware, and requesting a
// feature that may not exist to read a file that may not appear is a worse trade
// than spending a few milliseconds per texture at load.

import type { DecodedImage } from "./tga-loader"

const MAGIC = 0x20534444 // "DDS "
const fourCC = (s: string) => s.charCodeAt(0) | (s.charCodeAt(1) << 8) | (s.charCodeAt(2) << 16) | (s.charCodeAt(3) << 24)
const FOURCC_DXT1 = fourCC("DXT1")
const FOURCC_DXT3 = fourCC("DXT3")
const FOURCC_DXT5 = fourCC("DXT5")
const FOURCC_DX10 = fourCC("DX10")

// The DXGI formats worth answering. BC1/2/3 are what a converted stage carries;
// the two RGBA8 spellings turn up in tools that write a DX10 header for an
// uncompressed surface.
const DXGI_BC1 = new Set([70, 71, 72])
const DXGI_BC2 = new Set([73, 74, 75])
const DXGI_BC3 = new Set([76, 77, 78])
const DXGI_RGBA8 = new Set([27, 28, 29])
const DXGI_BGRA8 = new Set([87, 88, 91])

/** True when these bytes are a DDS, by magic rather than by file extension. */
export function isDds(buffer: ArrayBuffer): boolean {
  return buffer.byteLength >= 128 && new DataView(buffer).getUint32(0, true) === MAGIC
}

function rgb565(c: number, out: Uint8Array, o: number): void {
  const r = (c >> 11) & 0x1f
  const g = (c >> 5) & 0x3f
  const b = c & 0x1f
  // Bit-replication, not a shift: (r << 3) leaves white at 248 and tints every
  // bright surface, which reads as a dull texture rather than a decode bug.
  out[o] = (r << 3) | (r >> 2)
  out[o + 1] = (g << 2) | (g >> 4)
  out[o + 2] = (b << 3) | (b >> 2)
  out[o + 3] = 255
}

/**
 * One BC1 colour block into `dst`.
 *
 * `opaque` forces the four-colour mode. BC1 on its own picks its mode per block
 * from the endpoint order — c0 <= c1 means three colours and a transparent
 * fourth — but when a BC1 block is the colour half of a BC2/BC3 pair the alpha
 * lives in the other half and the four-colour mode is unconditional.
 */
function bc1Block(v: DataView, off: number, dst: Uint8Array, x0: number, y0: number, w: number, h: number, opaque: boolean): void {
  const c0 = v.getUint16(off, true)
  const c1 = v.getUint16(off + 2, true)
  const bits = v.getUint32(off + 4, true)
  const pal = new Uint8Array(16)
  rgb565(c0, pal, 0)
  rgb565(c1, pal, 4)
  if (c0 > c1 || opaque) {
    for (let i = 0; i < 3; i++) {
      pal[8 + i] = (2 * pal[i] + pal[4 + i] + 1) / 3
      pal[12 + i] = (pal[i] + 2 * pal[4 + i] + 1) / 3
    }
    pal[11] = 255
    pal[15] = 255
  } else {
    for (let i = 0; i < 3; i++) pal[8 + i] = (pal[i] + pal[4 + i]) >> 1
    pal[11] = 255
    // The fourth entry is transparent black in this mode — the 1-bit alpha.
    pal[12] = 0
    pal[13] = 0
    pal[14] = 0
    pal[15] = 0
  }
  for (let py = 0; py < 4; py++) {
    for (let px = 0; px < 4; px++) {
      const x = x0 + px
      const y = y0 + py
      if (x >= w || y >= h) continue
      const idx = ((bits >> (2 * (4 * py + px))) & 3) * 4
      const o = (y * w + x) * 4
      dst[o] = pal[idx]
      dst[o + 1] = pal[idx + 1]
      dst[o + 2] = pal[idx + 2]
      dst[o + 3] = pal[idx + 3]
    }
  }
}

/** BC3's 3-bit interpolated alpha block. */
function bc3Alpha(v: DataView, off: number, dst: Uint8Array, x0: number, y0: number, w: number, h: number): void {
  const a0 = v.getUint8(off)
  const a1 = v.getUint8(off + 1)
  const a = new Uint8Array(8)
  a[0] = a0
  a[1] = a1
  if (a0 > a1) {
    for (let i = 1; i < 7; i++) a[i + 1] = ((7 - i) * a0 + i * a1) / 7
  } else {
    for (let i = 1; i < 5; i++) a[i + 1] = ((5 - i) * a0 + i * a1) / 5
    a[6] = 0
    a[7] = 255
  }
  // 16 three-bit indices over six bytes; read as two 24-bit halves so the
  // shifts stay inside the 32-bit range JS bit ops actually work in.
  const lo = v.getUint8(off + 2) | (v.getUint8(off + 3) << 8) | (v.getUint8(off + 4) << 16)
  const hi = v.getUint8(off + 5) | (v.getUint8(off + 6) << 8) | (v.getUint8(off + 7) << 16)
  for (let i = 0; i < 16; i++) {
    const x = x0 + (i & 3)
    const y = y0 + (i >> 2)
    if (x >= w || y >= h) continue
    const bitsFor = i < 8 ? (lo >> (3 * i)) & 7 : (hi >> (3 * (i - 8))) & 7
    dst[(y * w + x) * 4 + 3] = a[bitsFor]
  }
}

/** BC2's 4-bit explicit alpha block. */
function bc2Alpha(v: DataView, off: number, dst: Uint8Array, x0: number, y0: number, w: number, h: number): void {
  for (let i = 0; i < 16; i++) {
    const x = x0 + (i & 3)
    const y = y0 + (i >> 2)
    if (x >= w || y >= h) continue
    const nib = v.getUint8(off + (i >> 1))
    const a = i & 1 ? nib >> 4 : nib & 0x0f
    dst[(y * w + x) * 4 + 3] = (a << 4) | a
  }
}

/**
 * Decode the top mip of a DDS to RGBA8.
 *
 * Only the first surface: the engine generates its own mip chain, and a cube map
 * or an array reaching here is a texture slot that was never going to mean what
 * the file meant anyway.
 */
export function decodeDds(buffer: ArrayBuffer): DecodedImage {
  const v = new DataView(buffer)
  if (buffer.byteLength < 128 || v.getUint32(0, true) !== MAGIC) throw new Error("not a DDS")

  const headerFlags = v.getUint32(8, true)
  const height = v.getUint32(12, true)
  const width = v.getUint32(16, true)
  const pitch = v.getUint32(20, true)
  if (width <= 0 || height <= 0) throw new Error(`DDS bad dimensions ${width}x${height}`)

  const pfFlags = v.getUint32(80, true)
  const pfFourCC = v.getUint32(84, true)
  const rgbBits = v.getUint32(88, true)
  const rMask = v.getUint32(92, true)
  const aMask = v.getUint32(104, true)

  let data = 128
  let kind: "bc1" | "bc2" | "bc3" | "rgba" | "bgra" | null = null

  if (pfFlags & 0x4) {
    if (pfFourCC === FOURCC_DXT1) kind = "bc1"
    else if (pfFourCC === FOURCC_DXT3) kind = "bc2"
    else if (pfFourCC === FOURCC_DXT5) kind = "bc3"
    else if (pfFourCC === FOURCC_DX10) {
      const dxgi = v.getUint32(128, true)
      data = 148 // 128 header + 20-byte DX10 extension
      if (DXGI_BC1.has(dxgi)) kind = "bc1"
      else if (DXGI_BC2.has(dxgi)) kind = "bc2"
      else if (DXGI_BC3.has(dxgi)) kind = "bc3"
      else if (DXGI_RGBA8.has(dxgi)) kind = "rgba"
      else if (DXGI_BGRA8.has(dxgi)) kind = "bgra"
      else throw new Error(`DDS unsupported DXGI format ${dxgi}`)
    } else {
      throw new Error(`DDS unsupported fourCC 0x${pfFourCC.toString(16)}`)
    }
  } else if (pfFlags & 0x40 && (rgbBits === 32 || rgbBits === 24)) {
    // Uncompressed. The masks are stated over the little-endian DWORD, so the
    // channel owning the LOW byte is the one stored first: red low is RGBA,
    // and the classic D3D A8R8G8B8 (red at 0x00ff0000) is BGRA in memory.
    kind = rMask === 0x000000ff ? "rgba" : "bgra"
  }
  if (!kind) throw new Error("DDS unsupported pixel format")

  const rgba = new Uint8Array(width * height * 4)

  if (kind === "rgba" || kind === "bgra") {
    const bytes = rgbBits === 24 ? 3 : 4
    const hasAlpha = bytes === 4 && aMask !== 0
    // Rows can be padded, and the header says by how much — DDSD_PITCH means the
    // pitch field is a byte stride rather than a total size. Reading past it
    // shears the image diagonally, which looks like a corrupt texture.
    const stride = headerFlags & 0x8 && pitch >= width * bytes ? pitch : width * bytes
    if (data + stride * height > buffer.byteLength) throw new Error("DDS truncated")
    const src = new Uint8Array(buffer, data)
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const s = y * stride + x * bytes
        const o = (y * width + x) * 4
        if (kind === "bgra") {
          rgba[o] = src[s + 2]
          rgba[o + 1] = src[s + 1]
          rgba[o + 2] = src[s]
        } else {
          rgba[o] = src[s]
          rgba[o + 1] = src[s + 1]
          rgba[o + 2] = src[s + 2]
        }
        rgba[o + 3] = hasAlpha ? src[s + 3] : 255
      }
    }
    return { rgba, width, height }
  }

  const blockBytes = kind === "bc1" ? 8 : 16
  const bw = Math.max(1, (width + 3) >> 2)
  const bh = Math.max(1, (height + 3) >> 2)
  if (data + bw * bh * blockBytes > buffer.byteLength) throw new Error("DDS truncated")

  for (let by = 0; by < bh; by++) {
    for (let bx = 0; bx < bw; bx++) {
      const off = data + (by * bw + bx) * blockBytes
      const x0 = bx * 4
      const y0 = by * 4
      if (kind === "bc1") {
        bc1Block(v, off, rgba, x0, y0, width, height, false)
      } else {
        // Colour first so the alpha block can overwrite what BC1 wrote — in
        // BC2/BC3 the colour half is always the opaque four-colour mode.
        bc1Block(v, off + 8, rgba, x0, y0, width, height, true)
        if (kind === "bc2") bc2Alpha(v, off, rgba, x0, y0, width, height)
        else bc3Alpha(v, off, rgba, x0, y0, width, height)
      }
    }
  }
  return { rgba, width, height }
}
