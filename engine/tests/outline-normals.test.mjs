// Outline normals. Run: npm test.
//
// The hull tears wherever two vertices share a position and push two ways — a
// hard edge, a UV seam. The smoothed stream exists so they push ONE way.

import { test } from "node:test"
import assert from "node:assert/strict"
import { buildOutlineVertices } from "../dist/outline-normals.js"

// A cube corner split three ways, as a hard-edged mesh stores it: three copies
// of the origin, each carrying its own face's normal, plus one far vertex per
// face so each copy has a triangle.
const v = (p, n) => [...p, ...n, 0, 0]
const verts = [
  v([0, 0, 0], [1, 0, 0]), v([0, 1, 0], [1, 0, 0]), v([0, 0, 1], [1, 0, 0]),
  v([0, 0, 0], [0, 1, 0]), v([0, 0, 1], [0, 1, 0]), v([1, 0, 0], [0, 1, 0]),
  v([0, 0, 0], [0, 0, 1]), v([1, 0, 0], [0, 0, 1]), v([0, 1, 0], [0, 0, 1]),
].flat()
const indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]

test("vertices at one position get one outline normal", () => {
  const out = buildOutlineVertices(new Float32Array(verts), 8, indices, new Float32Array([0.5, 1, 1, 1, 1, 1, 1, 1, 1]))
  const n = (i) => [out[i * 4], out[i * 4 + 1], out[i * 4 + 2]]
  for (const i of [3, 6]) n(i).forEach((c, k) => assert.ok(Math.abs(c - n(0)[k]) < 1e-6, `copy ${i} differs`))
  // the corner's bisector, unit length
  const s = 1 / Math.sqrt(3)
  n(0).forEach((c) => assert.ok(Math.abs(c - s) < 1e-5, `not the bisector: ${n(0)}`))
  // the edge scale rides along in w
  assert.equal(out[3], 0.5)
  assert.equal(out[7], 1)
})

test("back-to-back sheets keep their own normals", () => {
  const sheet = [v([0, 0, 0], [0, 0, 1]), v([1, 0, 0], [0, 0, 1]), v([0, 1, 0], [0, 0, 1]),
    v([0, 0, 0], [0, 0, -1]), v([1, 0, 0], [0, 0, -1]), v([0, 1, 0], [0, 0, -1])].flat()
  const out = buildOutlineVertices(new Float32Array(sheet), 8, [0, 1, 2, 3, 5, 4])
  assert.equal(out[2], 1)
  assert.equal(out[3 * 4 + 2], -1)
})
