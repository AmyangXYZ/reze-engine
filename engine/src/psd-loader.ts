// PSD → RGBA8, from the merged composite.
//
// MMD texture packs are often shipped as the artist's working files, so a
// material's texture path points at a .psd that no browser will decode. The
// layers are none of our business — Photoshop writes a flattened composite of
// the whole document at the end of the file, which is exactly the image the
// artist saw, so that is what this reads.
//
// (A file saved with "Maximize Compatibility" off has no useful composite. There
// is nothing to be done about that here beyond failing clearly: reconstructing it
// would mean compositing every layer, blend mode and clipping group — a Photoshop,
// not a texture loader.)

import type { DecodedImage } from "./tga-loader"

const MAGIC = 0x38425053 // "8BPS", big-endian — PSD is big-endian throughout

const enum ColorMode {
  Bitmap = 0,
  Grayscale = 1,
  Indexed = 2,
  RGB = 3,
  CMYK = 4,
  Multichannel = 7,
  Duotone = 8,
  Lab = 9,
}

const MODE_NAMES: Record<number, string> = {
  [ColorMode.Bitmap]: "bitmap",
  [ColorMode.CMYK]: "CMYK",
  [ColorMode.Multichannel]: "multichannel",
  [ColorMode.Lab]: "Lab",
}

/** True when these bytes are a PSD/PSB, by magic rather than by file extension. */
export function isPsd(buffer: ArrayBuffer): boolean {
  return buffer.byteLength >= 26 && new DataView(buffer).getUint32(0, false) === MAGIC
}

/**
 * PackBits, one row at a time.
 *
 * `end` bounds the row rather than the buffer: the row-length table says how many
 * compressed bytes this row occupies, and trusting the control bytes past that
 * would let one malformed row eat the next one's data.
 */
function unpackBits(src: Uint8Array, start: number, end: number, dst: Uint8Array, at: number, limit: number): number {
  let i = start
  let o = at
  while (i < end && o < limit) {
    const n = (src[i++] << 24) >> 24 // to signed
    if (n >= 0) {
      const count = Math.min(n + 1, limit - o, end - i)
      for (let k = 0; k < count; k++) dst[o++] = src[i++]
    } else if (n !== -128) {
      // -128 is a no-op by the spec, not a run of 129.
      const count = Math.min(1 - n, limit - o)
      const b = src[i++]
      for (let k = 0; k < count; k++) dst[o++] = b
    }
  }
  return o
}

/**
 * Decode a PSD's composite to RGBA8.
 *
 * RGB, grayscale, indexed and duotone at 8 or 16 bits per channel, raw or RLE —
 * which is every texture that has actually turned up. CMYK and Lab throw: they
 * need a colour conversion that would be guesswork without a profile, and a
 * silently wrong-coloured texture is worse than a missing one.
 */
export function decodePsd(buffer: ArrayBuffer): DecodedImage {
  const v = new DataView(buffer)
  if (buffer.byteLength < 26 || v.getUint32(0, false) !== MAGIC) throw new Error("not a PSD")

  const version = v.getUint16(4, false) // 1 = PSD, 2 = PSB
  if (version !== 1 && version !== 2) throw new Error(`PSD unsupported version ${version}`)
  const channels = v.getUint16(12, false)
  const height = v.getUint32(14, false)
  const width = v.getUint32(18, false)
  const depth = v.getUint16(22, false)
  const mode = v.getUint16(24, false)

  if (width <= 0 || height <= 0) throw new Error(`PSD bad dimensions ${width}x${height}`)
  if (depth !== 8 && depth !== 16) throw new Error(`PSD unsupported bit depth ${depth}`)
  if (mode in MODE_NAMES) throw new Error(`PSD unsupported colour mode: ${MODE_NAMES[mode]}`)

  // Three variable-length sections stand between the header and the pixels. Only
  // the indexed palette is worth reading; the rest is skipped by its length.
  let p = 26
  const colorDataLen = v.getUint32(p, false)
  const colorData = p + 4
  p = colorData + colorDataLen
  p += 4 + v.getUint32(p, false) // image resources
  // PSB states this length in 8 bytes. Only the low word can matter — the high
  // one would mean a layer section larger than 4GB.
  if (version === 2) {
    p += 8 + v.getUint32(p + 4, false)
  } else {
    p += 4 + v.getUint32(p, false)
  }
  if (p + 2 > buffer.byteLength) throw new Error("PSD truncated before the composite")

  const compression = v.getUint16(p, false)
  p += 2

  // Only the channels that carry colour, plus one alpha. A PSD can hold spot and
  // mask channels past those; they are stored after, so ignoring them is a matter
  // of not reading that far.
  const colourChannels = mode === ColorMode.RGB ? 3 : 1
  const hasAlpha = channels > colourChannels
  const used = colourChannels + (hasAlpha ? 1 : 0)
  if (channels < colourChannels) throw new Error(`PSD has ${channels} channel(s), expected ${colourChannels}`)

  const bytesPerSample = depth === 16 ? 2 : 1
  const planeSamples = width * height
  const planeBytes = planeSamples * bytesPerSample
  const planes = new Uint8Array(used * planeBytes)

  if (compression === 0) {
    const need = used * planeBytes
    if (p + need > buffer.byteLength) throw new Error("PSD truncated composite")
    planes.set(new Uint8Array(buffer, p, need))
  } else if (compression === 1) {
    // A table of per-row compressed lengths for EVERY channel comes first,
    // including the channels being skipped — so the table is read in full even
    // though only the first `used` channels' rows are decoded.
    const countBytes = version === 2 ? 4 : 2
    const tableBytes = channels * height * countBytes
    if (p + tableBytes > buffer.byteLength) throw new Error("PSD truncated row table")
    const rowLengths = new Uint32Array(channels * height)
    for (let i = 0; i < rowLengths.length; i++) {
      rowLengths[i] = countBytes === 2 ? v.getUint16(p + i * 2, false) : v.getUint32(p + i * 4, false)
    }
    const src = new Uint8Array(buffer)
    let at = p + tableBytes
    for (let c = 0; c < channels; c++) {
      for (let y = 0; y < height; y++) {
        const len = rowLengths[c * height + y]
        if (at + len > buffer.byteLength) throw new Error("PSD truncated composite")
        if (c < used) {
          const rowStart = c * planeBytes + y * width * bytesPerSample
          unpackBits(src, at, at + len, planes, rowStart, rowStart + width * bytesPerSample)
        }
        at += len
      }
    }
  } else {
    // 2 and 3 are the Zip codes. They appear on layer data, effectively never on
    // the composite, and inflating would mean shipping a decompressor.
    throw new Error(`PSD unsupported compression ${compression}`)
  }

  // 16-bit samples are big-endian; the high byte is the 8-bit value.
  const sample = (plane: number, i: number): number =>
    depth === 16 ? planes[plane * planeBytes + i * 2] : planes[plane * planeBytes + i]

  const rgba = new Uint8Array(planeSamples * 4)

  if (mode === ColorMode.Indexed) {
    // The palette is the colour mode data: 256 reds, then 256 greens, then blues.
    if (colorDataLen < 768) throw new Error("PSD indexed image has no palette")
    const pal = new Uint8Array(buffer, colorData, 768)
    for (let i = 0; i < planeSamples; i++) {
      const idx = sample(0, i)
      rgba[i * 4] = pal[idx]
      rgba[i * 4 + 1] = pal[256 + idx]
      rgba[i * 4 + 2] = pal[512 + idx]
      rgba[i * 4 + 3] = hasAlpha ? sample(1, i) : 255
    }
    return { rgba, width, height }
  }

  for (let i = 0; i < planeSamples; i++) {
    if (mode === ColorMode.RGB) {
      rgba[i * 4] = sample(0, i)
      rgba[i * 4 + 1] = sample(1, i)
      rgba[i * 4 + 2] = sample(2, i)
    } else {
      // Grayscale and duotone: one plane across all three. Duotone's ink colours
      // live in the colour mode data, and the spec's own advice is to treat the
      // data as grayscale — which is what every other reader does.
      const g = sample(0, i)
      rgba[i * 4] = g
      rgba[i * 4 + 1] = g
      rgba[i * 4 + 2] = g
    }
    rgba[i * 4 + 3] = hasAlpha ? sample(colourChannels, i) : 255
  }
  return { rgba, width, height }
}
