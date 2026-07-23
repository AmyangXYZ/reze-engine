// TGA decoder tests. Run: npm test.
import { test } from "node:test"
import assert from "node:assert/strict"
import { decodeTga } from "../dist/tga-loader.js"

// Build a TGA buffer: header (18B) + pixel bytes. descriptor bit5 (0x20) = top-left origin.
function tga({ type, width, height, depth, descriptor = 0, pixels }) {
  const h = new Uint8Array(18)
  h[2] = type
  h[12] = width & 0xff
  h[13] = width >> 8
  h[14] = height & 0xff
  h[15] = height >> 8
  h[16] = depth
  h[17] = descriptor
  return new Uint8Array([...h, ...pixels]).buffer
}

test("uncompressed 24-bit truecolor, bottom-left origin → vertical flip", () => {
  // BGR pixels, bottom row first: [red, green] then [blue, white].
  const img = decodeTga(
    tga({
      type: 2,
      width: 2,
      height: 2,
      depth: 24,
      pixels: [0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255],
    }),
  )
  assert.equal(img.width, 2)
  assert.equal(img.height, 2)
  // Flipped to top-left: row0 = blue, white; row1 = red, green.
  assert.deepEqual([...img.rgba.slice(0, 8)], [0, 0, 255, 255, 255, 255, 255, 255])
  assert.deepEqual([...img.rgba.slice(8, 16)], [255, 0, 0, 255, 0, 255, 0, 255])
})

test("top-left origin (descriptor bit5) is not flipped", () => {
  const img = decodeTga(
    tga({ type: 2, width: 1, height: 2, depth: 24, descriptor: 0x20, pixels: [0, 0, 255, 0, 255, 0] }),
  )
  assert.deepEqual([...img.rgba.slice(0, 4)], [255, 0, 0, 255]) // row0 red
  assert.deepEqual([...img.rgba.slice(4, 8)], [0, 255, 0, 255]) // row1 green
})

test("32-bit truecolor keeps the alpha channel (BGRA → RGBA)", () => {
  const img = decodeTga(tga({ type: 2, width: 1, height: 1, depth: 32, descriptor: 0x20, pixels: [10, 20, 30, 128] }))
  // BGRA (10,20,30,128) → RGBA (30,20,10,128)
  assert.deepEqual([...img.rgba], [30, 20, 10, 128])
})

test("RLE truecolor (type 10) decodes to the same pixels as raw", () => {
  // One RLE packet: repeat count=3 of BGR (0,0,255)=red, then a raw packet of 1 white.
  // width*height = 4. RLE packet header 0x82 = repeat, (0x82&0x7f)+1 = 3 reds.
  // Raw packet header 0x00 = raw, (0)+1 = 1 pixel.
  const pixels = [0x82, 0, 0, 255, 0x00, 255, 255, 255]
  const img = decodeTga(tga({ type: 10, width: 4, height: 1, depth: 24, descriptor: 0x20, pixels }))
  assert.equal(img.width, 4)
  assert.deepEqual([...img.rgba], [255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255, 255, 255, 255])
})

test("grayscale (type 3) expands to RGB", () => {
  const img = decodeTga(tga({ type: 3, width: 2, height: 1, depth: 8, descriptor: 0x20, pixels: [64, 200] }))
  assert.deepEqual([...img.rgba], [64, 64, 64, 255, 200, 200, 200, 255])
})

test("malformed input throws (caller catches → white fallback)", () => {
  assert.throws(() => decodeTga(new Uint8Array([1, 2, 3]).buffer))
  // Unsupported image type.
  assert.throws(() => decodeTga(tga({ type: 99, width: 1, height: 1, depth: 24, pixels: [0, 0, 0] })))
})
