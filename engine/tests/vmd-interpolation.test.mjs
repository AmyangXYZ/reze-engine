import { test } from "node:test"
import assert from "node:assert/strict"
import { rawInterpolationToBoneInterpolation, interpolateControlPoints } from "../dist/animation.js"
import { VMDWriter } from "../dist/vmd-writer.js"
import { VMDLoader } from "../dist/vmd-loader.js"

// The 16-byte record MMD actually interpolates from: byte c is channel c's x1,
// c+4 its y1, c+8 its x2, c+12 its y2, for X=0 Y=1 Z=2 rotation=3. Distinct
// values everywhere so a misread lands on the wrong number rather than a
// coincidence.
const RECORD = [
  /* x1 */ 11, 12, 13, 14,
  /* y1 */ 21, 22, 23, 24,
  /* x2 */ 31, 32, 33, 34,
  /* y2 */ 41, 42, 43, 44,
]

/** A bone frame's 64 bytes the way MMD writes them: the record four times, each
 *  copy shifted one byte left, bytes 2-3 of the first copy taken by the physics
 *  toggle, and uninitialised junk past the end of each copy. */
function mmdBlock(physics = 0x0000, junk = 0xab) {
  const raw = new Uint8Array(64).fill(junk)
  for (let r = 0; r < 4; r++) {
    for (let i = r; i < 16; i++) raw[r * 16 + (i - r)] = RECORD[i]
  }
  raw[2] = (physics >> 8) & 0xff
  raw[3] = physics & 0xff
  return raw
}

test("each channel is read out of its own shifted copy", () => {
  const interp = rawInterpolationToBoneInterpolation(mmdBlock())
  const expect = (c) => [
    { x: RECORD[c], y: RECORD[c + 4] },
    { x: RECORD[c + 8], y: RECORD[c + 12] },
  ]
  assert.deepEqual(interp.translationX, expect(0))
  assert.deepEqual(interp.translationY, expect(1))
  assert.deepEqual(interp.translationZ, expect(2))
  assert.deepEqual(interp.rotation, expect(3))
})

test("the physics toggle does not eat rotation's and Z's x1", () => {
  // Bytes 2 and 3 of the first copy are the physics field, not interpolation.
  // Reading channel c at raw[c] instead of raw[c * 16] pins Z's and rotation's
  // x1 to whatever the toggle happens to be — zero, on every real motion file.
  for (const physics of [0x0000, 25359 /* physics off */]) {
    const interp = rawInterpolationToBoneInterpolation(mmdBlock(physics))
    assert.equal(interp.rotation[0].x, RECORD[3])
    assert.equal(interp.translationZ[0].x, RECORD[2])
  }
})

test("junk past the end of each copy is never read", () => {
  const a = rawInterpolationToBoneInterpolation(mmdBlock(0, 0x00))
  const b = rawInterpolationToBoneInterpolation(mmdBlock(0, 0xff))
  assert.deepEqual(a, b)
})

test("a rotation curve with a real x1 eases instead of leaping", () => {
  // 92,54 → 61,117 is the curve both reference motions carry on their eased
  // keyframes. With x1 lost to zero the curve is well ahead of itself at the
  // midpoint; with x1 intact it is still behind linear there.
  const eased = [{ x: 92, y: 54 }, { x: 61, y: 117 }]
  const broken = [{ x: 0, y: 54 }, { x: 61, y: 117 }]
  const mid = interpolateControlPoints(eased, 0.5)
  assert.ok(mid < 0.5, `expected ease-in at the midpoint, got ${mid}`)
  assert.ok(interpolateControlPoints(broken, 0.5) > mid + 0.2)
})

test("interpolation survives a writer → loader round trip", () => {
  const source = rawInterpolationToBoneInterpolation(mmdBlock())
  const clip = {
    boneTracks: new Map([
      [
        "センター",
        [
          { boneName: "センター", frame: 0, rotation: { x: 0, y: 0, z: 0, w: 1 }, translation: { x: 0, y: 0, z: 0 }, interpolation: source },
          { boneName: "センター", frame: 30, rotation: { x: 0, y: 0, z: 0, w: 1 }, translation: { x: 1, y: 2, z: 3 }, interpolation: source },
        ],
      ],
    ]),
    morphTracks: new Map(),
    frameCount: 30,
  }
  const keyFrames = VMDLoader.loadFromBuffer(new VMDWriter().write(clip))
  const boneFrames = keyFrames.flatMap((k) => k.boneFrames)
  assert.equal(boneFrames.length, 2)
  for (const bf of boneFrames) {
    assert.deepEqual(rawInterpolationToBoneInterpolation(bf.interpolation), source)
  }
})
