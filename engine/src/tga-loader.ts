// Minimal TGA decoder. TGA is not a web-native format (createImageBitmap can't decode
// it), yet it's common in PMX assets — especially sphere maps (.spa/.sph) and eye/detail
// textures — so without this those materials render untextured. Produces top-left-origin
// RGBA8, ready for queue.writeTexture into an rgba8unorm(-srgb) texture.
//
// Handles the variants that actually appear in the wild: true-color (16/24/32), grayscale
// (8), and color-mapped (8-bit index), each raw or RLE-compressed, with the image
// descriptor's origin bits honored. Throws on malformed/unsupported input — the caller
// catches, logs, and falls back to the white texture (never panics the render loop).

export type DecodedImage = { width: number; height: number; rgba: Uint8Array }

// Image type byte (header offset 2).
const enum TgaType {
  ColorMapped = 1,
  TrueColor = 2,
  Grayscale = 3,
  RleColorMapped = 9,
  RleTrueColor = 10,
  RleGrayscale = 11,
}

export function decodeTga(buffer: ArrayBuffer): DecodedImage {
  const bytes = new Uint8Array(buffer)
  const view = new DataView(buffer)
  if (bytes.length < 18) throw new Error("TGA too small for header")

  const idLength = view.getUint8(0)
  const colorMapType = view.getUint8(1)
  const imageType = view.getUint8(2) as TgaType
  const cmapLength = view.getUint16(5, true)
  const cmapEntryBits = view.getUint8(7)
  const width = view.getUint16(12, true)
  const height = view.getUint16(14, true)
  const pixelDepth = view.getUint8(16)
  const descriptor = view.getUint8(17)

  if (width <= 0 || height <= 0) throw new Error(`TGA bad dimensions ${width}x${height}`)

  const rle =
    imageType === TgaType.RleColorMapped ||
    imageType === TgaType.RleTrueColor ||
    imageType === TgaType.RleGrayscale
  const colorMapped = imageType === TgaType.ColorMapped || imageType === TgaType.RleColorMapped
  const grayscale = imageType === TgaType.Grayscale || imageType === TgaType.RleGrayscale
  const trueColor = imageType === TgaType.TrueColor || imageType === TgaType.RleTrueColor
  if (!colorMapped && !grayscale && !trueColor) throw new Error(`TGA unsupported image type ${imageType}`)

  let offset = 18 + idLength

  // Color map (BGR/BGRA entries) → precomputed RGBA lookup.
  let colorMap: Uint8Array | null = null
  if (colorMapType === 1) {
    const entryBytes = Math.ceil(cmapEntryBits / 8)
    colorMap = new Uint8Array(cmapLength * 4)
    for (let i = 0; i < cmapLength; i++) {
      const [r, g, b, a] = unpackColor(bytes, offset + i * entryBytes, cmapEntryBits)
      colorMap.set([r, g, b, a], i * 4)
    }
    offset += cmapLength * entryBytes
  } else if (colorMapped) {
    throw new Error("TGA color-mapped image without a color map")
  }

  // Bytes per stored element (index byte for color-mapped, else pixel bytes).
  const elemBits = colorMapped ? pixelDepth : pixelDepth
  const elemBytes = Math.ceil(elemBits / 8)
  const pixelCount = width * height

  // Decode the (possibly RLE) element stream into one element per pixel, file order.
  const elements = new Uint8Array(pixelCount * elemBytes)
  if (rle) {
    let out = 0
    let src = offset
    while (out < elements.length) {
      if (src >= bytes.length) throw new Error("TGA RLE stream truncated")
      const packet = bytes[src++]
      const count = (packet & 0x7f) + 1
      if (packet & 0x80) {
        // RLE packet: one element repeated `count` times.
        if (src + elemBytes > bytes.length) throw new Error("TGA RLE packet truncated")
        for (let i = 0; i < count && out < elements.length; i++) {
          elements.set(bytes.subarray(src, src + elemBytes), out)
          out += elemBytes
        }
        src += elemBytes
      } else {
        // Raw packet: `count` elements follow.
        const span = count * elemBytes
        if (src + span > bytes.length) throw new Error("TGA raw packet truncated")
        elements.set(bytes.subarray(src, src + span), out)
        out += span
        src += span
      }
    }
  } else {
    const span = pixelCount * elemBytes
    if (offset + span > bytes.length) throw new Error("TGA pixel data truncated")
    elements.set(bytes.subarray(offset, offset + span), 0)
  }

  // Convert elements → RGBA in file order.
  const linear = new Uint8Array(pixelCount * 4)
  for (let p = 0; p < pixelCount; p++) {
    let rgba: [number, number, number, number]
    if (colorMapped) {
      const index = elements[p] // 8-bit index (pixelDepth 8)
      const c = index * 4
      rgba = colorMap ? [colorMap[c], colorMap[c + 1], colorMap[c + 2], colorMap[c + 3]] : [0, 0, 0, 255]
    } else if (grayscale) {
      const g = elements[p * elemBytes]
      rgba = [g, g, g, 255]
    } else {
      rgba = unpackColor(elements, p * elemBytes, pixelDepth)
    }
    linear.set(rgba, p * 4)
  }

  // Honor origin bits: bit5 set → top-to-bottom (no flip); clear → bottom-to-top.
  // bit4 set → right-to-left. Output is always top-left origin for GPU upload.
  const flipY = (descriptor & 0x20) === 0
  const flipX = (descriptor & 0x10) !== 0
  if (!flipX && !flipY) return { width, height, rgba: linear }

  const rgba = new Uint8Array(pixelCount * 4)
  for (let y = 0; y < height; y++) {
    const sy = flipY ? height - 1 - y : y
    for (let x = 0; x < width; x++) {
      const sx = flipX ? width - 1 - x : x
      const src = (sy * width + sx) * 4
      const dst = (y * width + x) * 4
      rgba[dst] = linear[src]
      rgba[dst + 1] = linear[src + 1]
      rgba[dst + 2] = linear[src + 2]
      rgba[dst + 3] = linear[src + 3]
    }
  }
  return { width, height, rgba }
}

// Unpack one true-color element (16/24/32-bit, stored BGR[A]) to RGBA8.
function unpackColor(src: Uint8Array, o: number, bits: number): [number, number, number, number] {
  if (bits === 32) return [src[o + 2], src[o + 1], src[o], src[o + 3]]
  if (bits === 24) return [src[o + 2], src[o + 1], src[o], 255]
  if (bits === 16 || bits === 15) {
    // 1-5-5-5: bit15 = attribute, then 5 red, 5 green, 5 blue (little-endian).
    const val = src[o] | (src[o + 1] << 8)
    const r = ((val >> 10) & 0x1f) * 255
    const g = ((val >> 5) & 0x1f) * 255
    const b = (val & 0x1f) * 255
    return [Math.round(r / 31), Math.round(g / 31), Math.round(b / 31), 255]
  }
  throw new Error(`TGA unsupported pixel depth ${bits}`)
}
