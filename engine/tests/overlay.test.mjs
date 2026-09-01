// The overlay pass draws whatever buildOverlayShapes laid out, and the capsule's
// cap-sign trick is the one piece that is not obvious by eye. Run: npm test.

import { test } from "node:test"
import assert from "node:assert/strict"
import {
  buildOverlayShapes,
  writeOverlayInstance,
  lineBetween,
  OVERLAY_SHAPES,
  OVERLAY_VERTEX_FLOATS,
  OVERLAY_INSTANCE_FLOATS,
} from "../dist/overlay.js"
import { Quat, Vec3 } from "../dist/math.js"

const geometry = buildOverlayShapes()

function vertex(i) {
  const o = i * OVERLAY_VERTEX_FLOATS
  const v = geometry.vertices
  return { pos: [v[o], v[o + 1], v[o + 2]], capSign: v[o + 6] }
}

test("every shape has geometry, and the ranges tile the buffer as whole lines", () => {
  let next = 0
  for (const shape of OVERLAY_SHAPES) {
    const range = geometry.ranges[shape]
    assert.equal(range.first, next, `${shape} does not start where the last shape ended`)
    assert.ok(range.count > 0, `${shape} is empty`)
    assert.equal(range.count % 3, 0, `${shape} is not whole triangles`)
    next += range.count
  }
  assert.equal(next, geometry.vertices.length / OVERLAY_VERTEX_FLOATS)
})

test("a capsule is a radius and a length, whatever the two are", () => {
  const { first, count } = geometry.ranges.capsule
  const radius = 0.4
  const halfHeight = 2.5
  // The vertex shader's placement: pos * scale + (0, capSign * extent, 0).
  const place = (v) => [v.pos[0] * radius, v.pos[1] * radius + v.capSign * halfHeight, v.pos[2] * radius]

  let sideLines = 0
  // Each stroked segment is 6 vertices: both ends of the ribbon quad, twice over.
  for (let i = first; i < first + count; i += 6) {
    const a = vertex(i)
    const b = vertex(i + 2)
    if (a.capSign !== b.capSign) {
      // A cylinder side line: vertical, the body's full length, on its surface.
      sideLines++
      const p0 = place(a)
      const p1 = place(b)
      assert.ok(Math.abs(p0[0] - p1[0]) < 1e-6 && Math.abs(p0[2] - p1[2]) < 1e-6, "a side line is not vertical")
      assert.ok(Math.abs(Math.abs(p0[1] - p1[1]) - 2 * halfHeight) < 1e-6, "a side line is not the body's length")
      assert.ok(Math.abs(Math.hypot(p0[0], p0[2]) - radius) < 1e-6, "a side line left the cylinder")
    } else {
      // Everything else rides a sphere of the radius — the caps stay round
      // however long the body gets, which is the whole point of the cap sign.
      for (const v of [a, b]) {
        const r = Math.hypot(v.pos[0] * radius, v.pos[1] * radius, v.pos[2] * radius)
        assert.ok(Math.abs(r - radius) < 1e-6, `capsule vertex is off the unit sphere`)
      }
    }
  }
  assert.equal(sideLines, 4, "a capsule should have exactly four side lines")
})

test("the instance packing round-trips a primitive", () => {
  const out = new Float32Array(OVERLAY_INSTANCE_FLOATS)
  writeOverlayInstance({ shape: "line", position: [1, 2, 3], color: [0.1, 0.2, 0.3, 0.4] }, out, 0)
  assert.deepEqual([...out.slice(0, 4)], [0, 0, 0, 1], "default rotation is identity")
  assert.deepEqual([...out.slice(4, 8)], [1, 2, 3, 0])
  assert.deepEqual([...out.slice(8, 11)], [1, 1, 1], "default scale is unit")
  assert.deepEqual(
    [...out.slice(12, 16)].map((n) => Math.round(n * 10) / 10),
    [0.1, 0.2, 0.3, 0.4],
  )
})

test("lineBetween points the unit segment at its target", () => {
  const from = [1, -2, 3]
  const to = [4, 5, -1]
  const line = lineBetween(from, to, [1, 1, 1, 1])
  const length = Math.hypot(to[0] - from[0], to[1] - from[1], to[2] - from[2])
  assert.ok(Math.abs(line.scale[1] - length) < 1e-6)
  const tail = Quat.rotateVec(new Quat(...line.rotation), new Vec3(0, length, 0))
  assert.ok(Math.abs(from[0] + tail.x - to[0]) < 1e-5)
  assert.ok(Math.abs(from[1] + tail.y - to[1]) < 1e-5)
  assert.ok(Math.abs(from[2] + tail.z - to[2]) < 1e-5)
  assert.equal(lineBetween(from, from, [1, 1, 1, 1]), null)
  assert.equal(lineBetween(from, to, [1, 1, 1, 1], true).shape, "dashedLine")
})
