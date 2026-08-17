// Radiance .hdr (RGBE) — the HDRI format Blender worlds ship in.
//
// Pure and dependency-free like everything else here: the whole format is a
// text header, a resolution line, and scanlines of shared-exponent RGBE
// pixels, so a parser is a hundred lines and a WASM decoder would be absurd.
// The output is scene-linear RGB — the space the view transform expects —
// which is the entire point: an 8-bit sky is display-space wallpaper, this is
// radiance the film can actually roll off.

export type HdrImage = {
  width: number
  height: number
  /** RGBA, scene-linear, alpha 1 — padded for direct rgba16float upload. */
  data: Float32Array
}

/**
 * Parse a Radiance .hdr file. Throws with a reason on anything malformed —
 * the caller shows it as an upload error, not a blank sky.
 */
export function parseHDR(buffer: ArrayBuffer): HdrImage {
  const bytes = new Uint8Array(buffer)
  let pos = 0
  const readLine = (): string => {
    let end = pos
    while (end < bytes.length && bytes[end] !== 0x0a) end++
    const line = String.fromCharCode(...bytes.subarray(pos, end))
    pos = end + 1
    return line
  }

  const magic = readLine()
  if (!magic.startsWith("#?RADIANCE") && !magic.startsWith("#?RGBE")) {
    throw new Error("not a Radiance .hdr file (missing #?RADIANCE header)")
  }
  let formatOk = false
  for (;;) {
    const line = readLine()
    if (line === "") break
    if (line.startsWith("FORMAT=")) {
      if (line !== "FORMAT=32-bit_rle_rgbe") throw new Error(`unsupported .hdr format: ${line.slice(7)}`)
      formatOk = true
    }
    if (pos >= bytes.length) throw new Error("truncated .hdr header")
  }
  if (!formatOk) throw new Error(".hdr header has no FORMAT line")

  // Only the universal orientation. Rotated/flipped variants exist in theory;
  // no tool in this pipeline writes them, and silently mis-orienting a sky is
  // worse than refusing one.
  const res = readLine()
  const m = /^-Y (\d+) \+X (\d+)$/.exec(res)
  if (!m) throw new Error(`unsupported .hdr orientation: "${res}"`)
  const height = parseInt(m[1], 10)
  const width = parseInt(m[2], 10)

  const data = new Float32Array(width * height * 4)
  const rgbe = new Uint8Array(width * 4)

  for (let y = 0; y < height; y++) {
    if (pos + 4 > bytes.length) throw new Error(`truncated at scanline ${y}`)
    const isNewRle = bytes[pos] === 2 && bytes[pos + 1] === 2 && ((bytes[pos + 2] << 8) | bytes[pos + 3]) === width
    if (isNewRle && width >= 8 && width < 32768) {
      pos += 4
      // Four planes, each RLE'd independently: count>128 is a run, else literals.
      for (let c = 0; c < 4; c++) {
        let x = 0
        while (x < width) {
          if (pos >= bytes.length) throw new Error(`truncated RLE at scanline ${y}`)
          let count = bytes[pos++]
          if (count > 128) {
            count -= 128
            const v = bytes[pos++]
            if (x + count > width) throw new Error(`RLE run overflows scanline ${y}`)
            for (let i = 0; i < count; i++) rgbe[(x + i) * 4 + c] = v
          } else {
            if (x + count > width) throw new Error(`RLE literals overflow scanline ${y}`)
            for (let i = 0; i < count; i++) rgbe[(x + i) * 4 + c] = bytes[pos++]
          }
          x += count
        }
      }
    } else {
      // Flat scanline (small widths, or files that never bothered with RLE).
      // The old-style (1,1,1,n) repeat marker is part of this path.
      let x = 0
      while (x < width) {
        if (pos + 4 > bytes.length) throw new Error(`truncated at scanline ${y}`)
        const r = bytes[pos]
        const g = bytes[pos + 1]
        const b = bytes[pos + 2]
        const e = bytes[pos + 3]
        pos += 4
        if (r === 1 && g === 1 && b === 1 && x > 0) {
          // Old-style RLE: repeat the previous pixel e times (shift ignored —
          // scanlines this parser accepts are under 2^15 wide).
          const px = (x - 1) * 4
          for (let i = 0; i < e && x < width; i++, x++) {
            rgbe[x * 4] = rgbe[px]
            rgbe[x * 4 + 1] = rgbe[px + 1]
            rgbe[x * 4 + 2] = rgbe[px + 2]
            rgbe[x * 4 + 3] = rgbe[px + 3]
          }
        } else {
          rgbe[x * 4] = r
          rgbe[x * 4 + 1] = g
          rgbe[x * 4 + 2] = b
          rgbe[x * 4 + 3] = e
          x++
        }
      }
    }
    // RGBE → linear: a shared exponent over three 8-bit mantissas.
    const row = y * width * 4
    for (let x = 0; x < width; x++) {
      const e = rgbe[x * 4 + 3]
      const scale = e === 0 ? 0 : Math.pow(2, e - 136) // 2^(e-128) / 256
      data[row + x * 4] = rgbe[x * 4] * scale
      data[row + x * 4 + 1] = rgbe[x * 4 + 1] * scale
      data[row + x * 4 + 2] = rgbe[x * 4 + 2] * scale
      data[row + x * 4 + 3] = 1
    }
  }
  return { width, height, data }
}

/**
 * Pack float32 to IEEE half, the format rgba16float uploads want. Clamps to
 * the half range rather than producing Inf — a 70000-nit sun texel should
 * read as "the brightest representable", not poison the filter.
 */
export function packHalf(src: Float32Array): Uint16Array<ArrayBuffer> {
  const out = new Uint16Array(new ArrayBuffer(src.length * 2))
  const f32 = new Float32Array(1)
  const u32 = new Uint32Array(f32.buffer)
  for (let i = 0; i < src.length; i++) {
    f32[0] = src[i]
    const x = u32[0]
    const sign = (x >>> 16) & 0x8000
    let exp = (x >>> 23) & 0xff
    const frac = x & 0x7fffff
    let h: number
    if (exp === 0xff) {
      h = sign | 0x7bff // NaN/Inf → max finite, by the clamp policy above
    } else if (exp > 142) {
      h = sign | 0x7bff // > 65504 → max finite
    } else if (exp < 113) {
      // Subnormal or zero in half precision.
      h = exp < 103 ? sign : sign | ((frac | 0x800000) >> (126 - exp))
    } else {
      h = sign | ((exp - 112) << 10) | (frac >> 13)
    }
    out[i] = h
  }
  return out
}
