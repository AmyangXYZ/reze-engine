// DDS decoder tests. Run: npm test.
//
// Stage models converted out of games ship DDS, which the browser cannot decode
// and the TGA fallback misreads (it reported a height of zero and every such
// material went white). These pin the block formats those stages actually use.
import { test } from "node:test"
import assert from "node:assert/strict"
import { decodeDds, isDds } from "../dist/dds-loader.js"

const DDPF_ALPHAPIXELS = 0x1
const DDPF_FOURCC = 0x4
const DDPF_RGB = 0x40

// Build a DDS: 128-byte header + payload. Field offsets per the DDS_HEADER spec.
function dds({ width, height, pfFlags, fourCC = 0, rgbBits = 0, masks = [0, 0, 0, 0], flags = 0, pitch = 0, payload }) {
  const buf = new ArrayBuffer(128 + payload.length)
  const v = new DataView(buf)
  v.setUint32(0, 0x20534444, true) // "DDS "
  v.setUint32(4, 124, true)
  v.setUint32(8, flags, true)
  v.setUint32(12, height, true)
  v.setUint32(16, width, true)
  v.setUint32(20, pitch, true)
  v.setUint32(76, 32, true)
  v.setUint32(80, pfFlags, true)
  v.setUint32(84, fourCC, true)
  v.setUint32(88, rgbBits, true)
  masks.forEach((m, i) => v.setUint32(92 + i * 4, m, true))
  new Uint8Array(buf, 128).set(payload)
  return buf
}

const fourCC = (s) => s.charCodeAt(0) | (s.charCodeAt(1) << 8) | (s.charCodeAt(2) << 16) | (s.charCodeAt(3) << 24)
const texel = (img, x, y) => [...img.rgba.slice((y * img.width + x) * 4, (y * img.width + x) * 4 + 4)]

// 565 endpoints used throughout: pure red and pure blue, which bit-replicate to
// exactly 255 so the interpolated entries are easy to state.
const RED565 = [0x00, 0xf8] // 0xF800
const BLUE565 = [0x1f, 0x00] // 0x001F
// Texel indices 0,1,2,3 across the first row: 0b11_10_01_00 = 0xE4.
const ROW0_0123 = [0xe4, 0x00, 0x00, 0x00]

test("BC1 four-colour mode interpolates both midpoints", () => {
  // c0 > c1 selects the opaque four-colour mode.
  const img = decodeDds(
    dds({
      width: 4,
      height: 4,
      pfFlags: DDPF_FOURCC,
      fourCC: fourCC("DXT1"),
      payload: [...RED565, ...BLUE565, ...ROW0_0123],
    }),
  )
  assert.equal(img.width, 4)
  assert.equal(img.height, 4)
  assert.deepEqual(texel(img, 0, 0), [255, 0, 0, 255]) // c0
  assert.deepEqual(texel(img, 1, 0), [0, 0, 255, 255]) // c1
  assert.deepEqual(texel(img, 2, 0), [170, 0, 85, 255]) // 2/3 c0
  assert.deepEqual(texel(img, 3, 0), [85, 0, 170, 255]) // 1/3 c0
})

test("BC1 three-colour mode makes index 3 transparent", () => {
  // c0 <= c1 is the punch-through mode: one midpoint, and index 3 is a hole.
  const img = decodeDds(
    dds({
      width: 4,
      height: 4,
      pfFlags: DDPF_FOURCC,
      fourCC: fourCC("DXT1"),
      payload: [...BLUE565, ...RED565, ...ROW0_0123],
    }),
  )
  assert.deepEqual(texel(img, 0, 0), [0, 0, 255, 255])
  assert.deepEqual(texel(img, 1, 0), [255, 0, 0, 255])
  assert.deepEqual(texel(img, 2, 0), [127, 0, 127, 255]) // midpoint
  assert.deepEqual(texel(img, 3, 0), [0, 0, 0, 0]) // the hole
})

test("BC3 reads its own alpha block and forces the colour half opaque", () => {
  // Alpha endpoints 255/0 with a0 > a1 → eight interpolated values. Only texel 0
  // takes index 1 (= a1 = 0); every other texel is index 0 (= a0 = 255).
  const alpha = [255, 0, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00]
  // Colour endpoints in the order that would mean "three-colour" for a bare BC1.
  // Inside BC3 it must not: the alpha block owns transparency here.
  const colour = [...BLUE565, ...RED565, ...ROW0_0123]
  const img = decodeDds(
    dds({ width: 4, height: 4, pfFlags: DDPF_FOURCC, fourCC: fourCC("DXT5"), payload: [...alpha, ...colour] }),
  )
  assert.deepEqual(texel(img, 0, 0), [0, 0, 255, 0]) // alpha index 1 → 0
  assert.deepEqual(texel(img, 1, 0), [255, 0, 0, 255])
  // Both midpoints interpolated, and index 3 is a colour rather than a hole —
  // a bare BC1 with these endpoints would have given [127,0,127] and [0,0,0,0].
  assert.deepEqual(texel(img, 2, 0), [85, 0, 170, 255])
  assert.deepEqual(texel(img, 3, 0), [170, 0, 85, 255])
})

test("BC2 reads its 4-bit explicit alpha, low nibble first", () => {
  // Texel 0 = 0x0, texel 1 = 0xF; the rest 0.
  const alpha = [0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
  const colour = [...RED565, ...BLUE565, ...ROW0_0123]
  const img = decodeDds(
    dds({ width: 4, height: 4, pfFlags: DDPF_FOURCC, fourCC: fourCC("DXT3"), payload: [...alpha, ...colour] }),
  )
  assert.equal(texel(img, 0, 0)[3], 0)
  assert.equal(texel(img, 1, 0)[3], 255) // 0xF replicated to 0xFF, not 0x0F
})

test("blocks past the edge are clipped, not written out of bounds", () => {
  // 2x2 image is one padded 4x4 block. Only the four real texels come back.
  const img = decodeDds(
    dds({
      width: 2,
      height: 2,
      pfFlags: DDPF_FOURCC,
      fourCC: fourCC("DXT1"),
      payload: [...RED565, ...BLUE565, ...ROW0_0123],
    }),
  )
  assert.equal(img.rgba.length, 2 * 2 * 4)
  assert.deepEqual(texel(img, 0, 0), [255, 0, 0, 255])
  assert.deepEqual(texel(img, 1, 0), [0, 0, 255, 255])
})

test("uncompressed A8R8G8B8 is BGRA in memory", () => {
  // The classic D3D spelling: red owns 0x00ff0000, so red is the third byte.
  const img = decodeDds(
    dds({
      width: 1,
      height: 1,
      pfFlags: DDPF_RGB | DDPF_ALPHAPIXELS,
      rgbBits: 32,
      masks: [0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000],
      payload: [10, 20, 30, 128], // B,G,R,A
    }),
  )
  assert.deepEqual([...img.rgba], [30, 20, 10, 128])
})

test("uncompressed A8B8G8R8 is RGBA in memory", () => {
  const img = decodeDds(
    dds({
      width: 1,
      height: 1,
      pfFlags: DDPF_RGB | DDPF_ALPHAPIXELS,
      rgbBits: 32,
      masks: [0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000],
      payload: [10, 20, 30, 128],
    }),
  )
  assert.deepEqual([...img.rgba], [10, 20, 30, 128])
})

test("24-bit has no alpha channel and comes back opaque", () => {
  const img = decodeDds(
    dds({
      width: 1,
      height: 1,
      pfFlags: DDPF_RGB,
      rgbBits: 24,
      masks: [0x00ff0000, 0x0000ff00, 0x000000ff, 0],
      payload: [10, 20, 30],
    }),
  )
  assert.deepEqual([...img.rgba], [30, 20, 10, 255])
})

test("a padded row pitch does not shear the image", () => {
  // DDSD_PITCH (0x8) with a stride of 12 bytes for a 2px-wide 32-bit surface:
  // four bytes of padding per row. Reading rows at width*4 would walk into it.
  const row = (a, b) => [...a, ...b, 0, 0, 0, 0]
  const img = decodeDds(
    dds({
      width: 2,
      height: 2,
      flags: 0x8,
      pitch: 12,
      pfFlags: DDPF_RGB | DDPF_ALPHAPIXELS,
      rgbBits: 32,
      masks: [0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000],
      payload: [...row([0, 0, 255, 255], [0, 255, 0, 255]), ...row([255, 0, 0, 255], [255, 255, 255, 255])],
    }),
  )
  assert.deepEqual(texel(img, 0, 0), [255, 0, 0, 255])
  assert.deepEqual(texel(img, 1, 0), [0, 255, 0, 255])
  assert.deepEqual(texel(img, 0, 1), [0, 0, 255, 255])
})

test("DX10 headers skip their extension before the payload", () => {
  // DXGI_FORMAT_BC1_UNORM = 71. The 20-byte extension sits between the header
  // and the blocks; missing it decodes the extension bytes as pixels.
  const ext = [71, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
  const img = decodeDds(
    dds({
      width: 4,
      height: 4,
      pfFlags: DDPF_FOURCC,
      fourCC: fourCC("DX10"),
      payload: [...ext, ...RED565, ...BLUE565, ...ROW0_0123],
    }),
  )
  assert.deepEqual(texel(img, 0, 0), [255, 0, 0, 255])
  assert.deepEqual(texel(img, 1, 0), [0, 0, 255, 255])
})

test("isDds keys on the magic, not the extension", () => {
  const real = dds({
    width: 4,
    height: 4,
    pfFlags: DDPF_FOURCC,
    fourCC: fourCC("DXT1"),
    payload: [...RED565, ...BLUE565, ...ROW0_0123],
  })
  assert.equal(isDds(real), true)
  assert.equal(isDds(new Uint8Array(200).buffer), false)
  assert.equal(isDds(new Uint8Array(8).buffer), false) // too short to hold a header
})

test("an unsupported pixel format throws rather than returning garbage", () => {
  assert.throws(
    () => decodeDds(dds({ width: 4, height: 4, pfFlags: DDPF_FOURCC, fourCC: fourCC("DXT2"), payload: [] })),
    /unsupported fourCC/,
  )
  assert.throws(
    () =>
      decodeDds(
        dds({ width: 4, height: 4, pfFlags: DDPF_FOURCC, fourCC: fourCC("DX10"), payload: new Array(20).fill(95) }),
      ),
    /unsupported DXGI format/,
  )
})

test("a truncated payload throws instead of decoding past the buffer", () => {
  assert.throws(
    () => decodeDds(dds({ width: 64, height: 64, pfFlags: DDPF_FOURCC, fourCC: fourCC("DXT1"), payload: [1, 2, 3] })),
    /truncated/,
  )
})
