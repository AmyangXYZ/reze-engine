// The inverted hull's own push direction, one per vertex, built once at load.
//
// The hull is the model pushed out along a normal. Pushed along the MESH's
// normals it tears wherever the mesh carries two vertices at one position —
// every hard edge and every UV seam — because the two copies go two ways and
// the gap between them shows the body through the line: the jaw, the finger
// joints, the parting in the hair. Aether Gazer bakes a smoothed normal into
// vertex colour for exactly this (Character/Debug.shader, pass "Outline"); a
// PMX has nowhere to carry one, so it is computed here instead: the normals of
// every vertex at one position, averaged, so the copies agree and the hull
// stays closed.
//
// Each vertex's vote is weighted by the corner angles it owns, which is what
// makes a hard corner come out as the corner's true bisector rather than
// leaning toward whichever side happened to be cut into more triangles.
//
// Rest pose only, as the game's is: a morph moves the positions and leaves this
// alone, and skinning turns it exactly as it turns the normal.

/** Positions closer than this (model units) are one position. */
export const OUTLINE_WELD_EPSILON = 1e-4

/**
 * Smoothed outline normals, packed with the PMX per-vertex edge scale:
 * xyz = unit direction, w = edge scale — four floats per vertex.
 *
 * @param vertexData interleaved [pos3, normal3, uv2] per vertex
 * @param indices    triangle list
 * @param edgeScales per-vertex PMX edge scale, or omitted for 1
 */
export function buildOutlineVertices(
  vertexData: ArrayLike<number>,
  stride: number,
  indices: ArrayLike<number>,
  edgeScales?: ArrayLike<number>,
): Float32Array<ArrayBuffer> {
  const n = Math.floor(vertexData.length / stride)
  const out = new Float32Array(n * 4)

  // Corner angles each vertex owns. A vertex no triangle uses keeps a token
  // weight, so it still has a direction of its own.
  const weight = new Float32Array(n).fill(1e-6)
  for (let t = 0; t + 2 < indices.length; t += 3) {
    const a = indices[t], b = indices[t + 1], c = indices[t + 2]
    if (a >= n || b >= n || c >= n) continue
    weight[a] += cornerAngle(vertexData, stride, a, b, c)
    weight[b] += cornerAngle(vertexData, stride, b, c, a)
    weight[c] += cornerAngle(vertexData, stride, c, a, b)
  }

  // Weld by quantized position.
  const inv = 1 / OUTLINE_WELD_EPSILON
  const group = new Int32Array(n)
  const ids = new Map<string, number>()
  let groups = 0
  for (let i = 0; i < n; i++) {
    const o = i * stride
    const key = `${Math.round(vertexData[o] * inv)},${Math.round(vertexData[o + 1] * inv)},${Math.round(vertexData[o + 2] * inv)}`
    let g = ids.get(key)
    if (g === undefined) {
      g = groups++
      ids.set(key, g)
    }
    group[i] = g
  }

  const sum = new Float32Array(groups * 3)
  for (let i = 0; i < n; i++) {
    const o = i * stride
    const w = weight[i]
    const g = group[i] * 3
    sum[g] += vertexData[o + 3] * w
    sum[g + 1] += vertexData[o + 4] * w
    sum[g + 2] += vertexData[o + 5] * w
  }

  for (let i = 0; i < n; i++) {
    const o = i * stride
    const nx = vertexData[o + 3], ny = vertexData[o + 4], nz = vertexData[o + 5]
    const g = group[i] * 3
    let x = sum[g], y = sum[g + 1], z = sum[g + 2]
    const len = Math.hypot(x, y, z)
    // Two sheets back to back at one position (a double-sided skirt, a shell
    // and its lining) cancel to nothing, or to a direction pointing INTO one of
    // them. Either way the welded answer is wrong for this vertex, and its own
    // normal is the honest one.
    if (len < 1e-6 || x * nx + y * ny + z * nz <= 0) {
      x = nx
      y = ny
      z = nz
    }
    const l = Math.hypot(x, y, z) || 1
    out[i * 4] = x / l
    out[i * 4 + 1] = y / l
    out[i * 4 + 2] = z / l
    out[i * 4 + 3] = edgeScales ? edgeScales[i] : 1
  }
  return out
}

/** The angle at vertex `a` of triangle (a, b, c), in radians. */
function cornerAngle(v: ArrayLike<number>, stride: number, a: number, b: number, c: number): number {
  const oa = a * stride, ob = b * stride, oc = c * stride
  const ux = v[ob] - v[oa], uy = v[ob + 1] - v[oa + 1], uz = v[ob + 2] - v[oa + 2]
  const wx = v[oc] - v[oa], wy = v[oc + 1] - v[oa + 1], wz = v[oc + 2] - v[oa + 2]
  const lu = Math.hypot(ux, uy, uz), lw = Math.hypot(wx, wy, wz)
  if (lu < 1e-12 || lw < 1e-12) return 0
  const cos = (ux * wx + uy * wy + uz * wz) / (lu * lw)
  return Math.acos(Math.max(-1, Math.min(1, cos)))
}
