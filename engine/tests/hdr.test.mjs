// The Radiance parser and the half-float pack. Run: npm test.
//
// The test WRITES a .hdr in memory — header, new-RLE scanlines, flat
// scanlines — and reads it back, because a fixture binary would pin bytes
// nobody can review. Golden values are computed from the RGBE definition
// directly: value = mantissa * 2^(exponent - 136).

import { test } from "node:test"
import assert from "node:assert/strict"
import { parseHDR, packHalf } from "../dist/hdr.js"

const HEADER = (w, h) => `#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y ${h} +X ${w}\n`

function bytesOf(str) {
  return [...str].map((c) => c.charCodeAt(0))
}

/** New-style RLE scanline: 4 planes, literal-encoded (count ≤ 128). */
function newRleScanline(width, rgbePixels) {
  const out = [2, 2, (width >> 8) & 0xff, width & 0xff]
  for (let c = 0; c < 4; c++) {
    let x = 0
    while (x < width) {
      const n = Math.min(128, width - x)
      out.push(n)
      for (let i = 0; i < n; i++) out.push(rgbePixels[(x + i) * 4 + c])
      x += n
    }
  }
  return out
}

test("a new-RLE file round-trips to the RGBE definition's own values", () => {
  const width = 8
  // One bright texel (sun: mantissa 200, exp 140 → 200 * 2^4 = 3200) among
  // mid-grays (128 * 2^-8 = 0.5), and one black (exp 0 → exactly 0).
  const px = []
  for (let x = 0; x < width; x++) px.push(128, 128, 128, 128)
  px[0 * 4 + 0] = 200; px[0 * 4 + 3] = 140
  px[3 * 4 + 0] = 0; px[3 * 4 + 1] = 0; px[3 * 4 + 2] = 0; px[3 * 4 + 3] = 0
  const file = new Uint8Array([...bytesOf(HEADER(width, 1)), ...newRleScanline(width, px)])
  const img = parseHDR(file.buffer)
  assert.equal(img.width, 8)
  assert.equal(img.height, 1)
  assert.ok(Math.abs(img.data[0] - 200 * Math.pow(2, 140 - 136)) < 1e-6, `sun texel ${img.data[0]}`)
  assert.ok(img.data[0] > 1.0, "an HDR sun must exceed display white — that is the whole point")
  assert.ok(Math.abs(img.data[4 * 1 + 1] - 0.5) < 1e-6, "mid-gray green")
  assert.equal(img.data[4 * 3], 0, "exponent 0 is exactly black")
  assert.equal(img.data[3], 1, "alpha padded to 1")
})

test("a flat (unencoded) file parses, old-style repeats included", () => {
  const width = 4
  // Pixel, then an old-style (1,1,1,2) repeat covering two more, then one more.
  const scan = [100, 50, 25, 129, 1, 1, 1, 2, 10, 20, 30, 128]
  const file = new Uint8Array([...bytesOf(HEADER(width, 1)), ...scan])
  const img = parseHDR(file.buffer)
  const s = Math.pow(2, 129 - 136)
  assert.ok(Math.abs(img.data[0] - 100 * s) < 1e-9)
  // The repeat copied pixel 0 twice.
  assert.equal(img.data[4], img.data[0])
  assert.equal(img.data[8], img.data[0])
  assert.ok(Math.abs(img.data[12] - 10 * Math.pow(2, 128 - 136)) < 1e-9)
})

test("malformed files throw with a reason, not a blank sky", () => {
  assert.throws(() => parseHDR(new Uint8Array(bytesOf("PNG nonsense\n")).buffer), /Radiance/)
  assert.throws(() => parseHDR(new Uint8Array(bytesOf("#?RADIANCE\nFORMAT=32-bit_rle_xyze\n\n-Y 1 +X 1\n")).buffer), /unsupported .hdr format/)
  assert.throws(() => parseHDR(new Uint8Array(bytesOf("#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n+Y 1 +X 1\n")).buffer), /orientation/)
  const truncated = new Uint8Array([...bytesOf(HEADER(8, 2)), 2, 2, 0, 8, 12])
  assert.throws(() => parseHDR(truncated.buffer), /truncated|overflow/)
})

test("packHalf hits the known bit patterns and clamps instead of overflowing", () => {
  const h = packHalf(new Float32Array([0, 1, 0.5, -2, 65504, 1e9, Infinity, NaN, 6.1035156e-5, 3.0517578e-5]))
  assert.equal(h[0], 0x0000)
  assert.equal(h[1], 0x3c00)
  assert.equal(h[2], 0x3800)
  assert.equal(h[3], 0xc000)
  assert.equal(h[4], 0x7bff, "half max")
  assert.equal(h[5], 0x7bff, "overflow clamps to max finite")
  assert.equal(h[6], 0x7bff, "Inf clamps — a sun texel must not poison the filter")
  assert.equal((h[7] & 0x7fff), 0x7bff, "NaN clamps too")
  assert.equal(h[8], 0x0400, "smallest normal half (2^-14)")
  assert.equal(h[9], 0x0200, "2^-15 lands in the subnormal range, half precision kept")
})
