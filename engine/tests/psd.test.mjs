// PSD decoder tests. Run: npm test.
//
// MMD texture packs are often shipped as the artist's working files, so a PMX
// can reference a .psd the browser will not decode. These pin the composite
// reader: the section skipping that finds it, and the two compressions it uses.
import { test } from "node:test"
import assert from "node:assert/strict"
import { decodePsd, isPsd } from "../dist/psd-loader.js"

const RGB = 3
const GRAYSCALE = 1
const INDEXED = 2

// Build a PSD: 26-byte header, three skippable sections, then the composite.
// `colorData` is the indexed palette when there is one.
function psd({
  width,
  height,
  channels,
  depth = 8,
  mode = RGB,
  version = 1,
  compression,
  colorData = [],
  imageResources = [],
  layerInfo = [],
  payload,
}) {
  const head = []
  const u16 = (n) => head.push((n >> 8) & 0xff, n & 0xff)
  const u32 = (n) => head.push((n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff)
  head.push(0x38, 0x42, 0x50, 0x53) // "8BPS"
  u16(version)
  head.push(0, 0, 0, 0, 0, 0) // reserved
  u16(channels)
  u32(height)
  u32(width)
  u16(depth)
  u16(mode)
  u32(colorData.length)
  head.push(...colorData)
  u32(imageResources.length)
  head.push(...imageResources)
  // PSB states the layer/mask length in eight bytes rather than four.
  if (version === 2) u32(0)
  u32(layerInfo.length)
  head.push(...layerInfo)
  u16(compression)
  return new Uint8Array([...head, ...payload]).buffer
}

const texel = (img, x, y) => [...img.rgba.slice((y * img.width + x) * 4, (y * img.width + x) * 4 + 4)]

test("raw RGB is planar — one whole channel, then the next", () => {
  // 2x1: red then green. Stored R,R then G,G then B,B — NOT interleaved.
  const img = decodePsd(
    psd({ width: 2, height: 1, channels: 3, compression: 0, payload: [255, 0, 0, 255, 0, 0] }),
  )
  assert.equal(img.width, 2)
  assert.equal(img.height, 1)
  assert.deepEqual(texel(img, 0, 0), [255, 0, 0, 255])
  assert.deepEqual(texel(img, 1, 0), [0, 255, 0, 255])
})

test("a fourth channel is alpha", () => {
  const img = decodePsd(
    psd({ width: 1, height: 1, channels: 4, compression: 0, payload: [10, 20, 30, 128] }),
  )
  assert.deepEqual([...img.rgba], [10, 20, 30, 128])
})

test("three channels come back fully opaque", () => {
  const img = decodePsd(psd({ width: 1, height: 1, channels: 3, compression: 0, payload: [10, 20, 30] }))
  assert.deepEqual([...img.rgba], [10, 20, 30, 255])
})

test("spot and mask channels past the alpha are ignored", () => {
  // Five channels: RGBA plus a spot channel that must not be read as colour.
  const img = decodePsd(
    psd({ width: 1, height: 1, channels: 5, compression: 0, payload: [10, 20, 30, 128, 99] }),
  )
  assert.deepEqual([...img.rgba], [10, 20, 30, 128])
})

test("RLE rows are PackBits, bounded by the row-length table", () => {
  // Each row of each channel: a repeat packet of 4, then one literal byte.
  // -3 → repeat the next byte 4 times; 0 → copy 1 literal byte.
  const row = (run, last) => [0xfd, run, 0x00, last]
  const rows = [row(255, 0), row(0, 255), row(0, 0)] // R, G, B — one row each
  const table = [0, 4, 0, 4, 0, 4] // three uint16 lengths, one per channel-row
  const img = decodePsd(
    psd({ width: 5, height: 1, channels: 3, compression: 1, payload: [...table, ...rows.flat()] }),
  )
  assert.equal(img.width, 5)
  assert.deepEqual(texel(img, 0, 0), [255, 0, 0, 255]) // from the runs
  assert.deepEqual(texel(img, 4, 0), [0, 255, 0, 255]) // from the literals
})

test("the row table covers every channel, including skipped ones", () => {
  // Four channels but the third row-run belongs to B and the fourth to alpha.
  // Getting the table stride wrong reads alpha as blue.
  const row = (b) => [0xff, b] // -1 → repeat next byte twice
  const table = [0, 2, 0, 2, 0, 2, 0, 2]
  const img = decodePsd(
    psd({
      width: 2,
      height: 1,
      channels: 4,
      compression: 1,
      payload: [...table, ...row(10), ...row(20), ...row(30), ...row(128)],
    }),
  )
  assert.deepEqual(texel(img, 0, 0), [10, 20, 30, 128])
  assert.deepEqual(texel(img, 1, 0), [10, 20, 30, 128])
})

test("16-bit samples are big-endian and take the high byte", () => {
  // 0xAB12 → 0xAB. Getting the endianness backwards yields 0x12.
  const img = decodePsd(
    psd({
      width: 1,
      height: 1,
      channels: 3,
      depth: 16,
      compression: 0,
      payload: [0xab, 0x12, 0xcd, 0x34, 0xef, 0x56],
    }),
  )
  assert.deepEqual([...img.rgba], [0xab, 0xcd, 0xef, 255])
})

test("grayscale fills all three channels", () => {
  const img = decodePsd(
    psd({ width: 2, height: 1, channels: 1, mode: GRAYSCALE, compression: 0, payload: [0, 200] }),
  )
  assert.deepEqual(texel(img, 0, 0), [0, 0, 0, 255])
  assert.deepEqual(texel(img, 1, 0), [200, 200, 200, 255])
})

test("indexed reads the palette out of the colour mode data", () => {
  // 768 bytes: 256 reds, 256 greens, 256 blues. Entry 1 = (7, 8, 9).
  const pal = new Array(768).fill(0)
  pal[1] = 7
  pal[256 + 1] = 8
  pal[512 + 1] = 9
  const img = decodePsd(
    psd({ width: 1, height: 1, channels: 1, mode: INDEXED, compression: 0, colorData: pal, payload: [1] }),
  )
  assert.deepEqual([...img.rgba], [7, 8, 9, 255])
})

test("the three variable sections are skipped by their stated length", () => {
  // Non-empty resources and layer info: reading past either lands on garbage.
  const img = decodePsd(
    psd({
      width: 1,
      height: 1,
      channels: 3,
      compression: 0,
      imageResources: [1, 2, 3, 4, 5],
      layerInfo: [9, 9, 9],
      payload: [10, 20, 30],
    }),
  )
  assert.deepEqual([...img.rgba], [10, 20, 30, 255])
})

test("PSB states its layer length in eight bytes", () => {
  const img = decodePsd(
    psd({ width: 1, height: 1, channels: 3, version: 2, compression: 0, payload: [10, 20, 30] }),
  )
  assert.deepEqual([...img.rgba], [10, 20, 30, 255])
})

test("isPsd keys on the magic, not the extension", () => {
  assert.equal(isPsd(psd({ width: 1, height: 1, channels: 3, compression: 0, payload: [0, 0, 0] })), true)
  assert.equal(isPsd(new Uint8Array(64).buffer), false)
  assert.equal(isPsd(new Uint8Array(4).buffer), false)
})

test("colour modes needing a profile throw rather than guessing", () => {
  assert.throws(
    () => decodePsd(psd({ width: 1, height: 1, channels: 4, mode: 4, compression: 0, payload: [0, 0, 0, 0] })),
    /CMYK/,
  )
  assert.throws(
    () => decodePsd(psd({ width: 1, height: 1, channels: 3, mode: 9, compression: 0, payload: [0, 0, 0] })),
    /Lab/,
  )
})

test("zip-compressed composites and odd depths throw clearly", () => {
  assert.throws(
    () => decodePsd(psd({ width: 1, height: 1, channels: 3, compression: 2, payload: [] })),
    /unsupported compression/,
  )
  assert.throws(
    () => decodePsd(psd({ width: 1, height: 1, channels: 3, depth: 32, compression: 0, payload: [] })),
    /bit depth/,
  )
})

test("a truncated composite throws instead of decoding past the buffer", () => {
  assert.throws(
    () => decodePsd(psd({ width: 64, height: 64, channels: 3, compression: 0, payload: [1, 2, 3] })),
    /truncated/,
  )
})
