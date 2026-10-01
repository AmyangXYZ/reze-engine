// Skinned vertex streams for native (game-shader) materials.
//
// The engine skins in the vertex stage of its own material shaders, from the
// bone palette (skinMats) and the post-morph vertex buffer. A game shader skins
// nothing: Unity hands it world-space-ready POSITION / NORMAL / TANGENT streams
// from its SkinnedMeshRenderer. So a model that wears native materials is
// skinned once per frame here, by compute, into three plain vertex buffers the
// game shaders read as Unity's — the same streams the compare page feeds them.
//
// Inputs are the engine's own buffers, untouched: the interleaved vertex
// buffer (position, normal, uv — 8 floats, morphs already applied), the joints
// (uint16x4, two words a vertex), the weights (unorm8x4, one word) and the
// skin matrices. Tangents are not in the engine's vertex format; they are
// derived once at install (tangentsFor) and skinned here with the normal.

export const UNITY_SKIN_WGSL = /* wgsl */ `
struct Params { count: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> verts: array<f32>;      // pos(3) nrm(3) uv(2)
@group(0) @binding(2) var<storage, read> joints: array<u32>;     // uint16x4: two words
@group(0) @binding(3) var<storage, read> weights: array<u32>;    // unorm8x4: one word
@group(0) @binding(4) var<storage, read> skinMats: array<mat4x4f>;
@group(0) @binding(5) var<storage, read> tangents: array<f32>;   // rest tangent(4)
@group(0) @binding(6) var<storage, read_write> outPos: array<f32>;
@group(0) @binding(7) var<storage, read_write> outNrm: array<f32>;
@group(0) @binding(8) var<storage, read_write> outTan: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
  let v = id.x;
  if (v >= P.count) { return; }
  let p = vec4f(verts[v * 8u], verts[v * 8u + 1u], verts[v * 8u + 2u], 1.0);
  let n = vec3f(verts[v * 8u + 3u], verts[v * 8u + 4u], verts[v * 8u + 5u]);
  let t = vec4f(tangents[v * 4u], tangents[v * 4u + 1u], tangents[v * 4u + 2u], tangents[v * 4u + 3u]);
  let j01 = joints[v * 2u];
  let j23 = joints[v * 2u + 1u];
  let j = vec4u(j01 & 0xffffu, j01 >> 16u, j23 & 0xffffu, j23 >> 16u);
  var w = unpack4x8unorm(weights[v]);
  // Renormalised as the material vertex stage does: PMX ships unnormalised
  // weights on extras, and a sum of zero is the bind pose.
  let s = w.x + w.y + w.z + w.w;
  w = select(vec4f(1.0, 0.0, 0.0, 0.0), w / s, s > 0.0001);
  var sp = vec3f(0.0);
  var sn = vec3f(0.0);
  var st = vec3f(0.0);
  for (var k = 0u; k < 4u; k++) {
    let m = skinMats[j[k]];
    let r = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
    sp += (m * p).xyz * w[k];
    sn += (r * n) * w[k];
    st += (r * t.xyz) * w[k];
  }
  // Degenerate blends (opposing bones at 50/50) cancel to zero; up, not NaN.
  let ln = dot(sn, sn);
  let lt = dot(st, st);
  sn = select(vec3f(0.0, 1.0, 0.0), sn * inverseSqrt(ln), ln > 1e-12);
  st = select(vec3f(1.0, 0.0, 0.0), st * inverseSqrt(lt), lt > 1e-12);
  outPos[v * 3u] = sp.x; outPos[v * 3u + 1u] = sp.y; outPos[v * 3u + 2u] = sp.z;
  outNrm[v * 3u] = sn.x; outNrm[v * 3u + 1u] = sn.y; outNrm[v * 3u + 2u] = sn.z;
  outTan[v * 4u] = st.x; outTan[v * 4u + 1u] = st.y; outTan[v * 4u + 2u] = st.z; outTan[v * 4u + 3u] = t.w;
}
`

/**
 * Per-vertex tangents (xyz, w = handedness) from positions, normals and uvs,
 * accumulated per triangle and orthogonalised against the normal — the
 * standard construction, which is what a mesh imported without tangents gets
 * in Unity too. `verts` is the engine's interleaved layout (8 floats).
 */
export function tangentsFor(verts: Float32Array, indices: Uint32Array | Uint16Array): Float32Array {
  const n = verts.length / 8
  const tan = new Float64Array(n * 3)
  const bit = new Float64Array(n * 3)
  for (let i = 0; i + 2 < indices.length; i += 3) {
    const a = indices[i],
      b = indices[i + 1],
      c = indices[i + 2]
    const p = (v: number, k: number) => verts[v * 8 + k]
    const e1 = [p(b, 0) - p(a, 0), p(b, 1) - p(a, 1), p(b, 2) - p(a, 2)]
    const e2 = [p(c, 0) - p(a, 0), p(c, 1) - p(a, 1), p(c, 2) - p(a, 2)]
    const du1 = p(b, 6) - p(a, 6),
      dv1 = p(b, 7) - p(a, 7)
    const du2 = p(c, 6) - p(a, 6),
      dv2 = p(c, 7) - p(a, 7)
    const det = du1 * dv2 - du2 * dv1
    if (Math.abs(det) < 1e-12) continue
    const r = 1 / det
    for (const v of [a, b, c]) {
      for (let k = 0; k < 3; k++) {
        tan[v * 3 + k] += (e1[k] * dv2 - e2[k] * dv1) * r
        bit[v * 3 + k] += (e2[k] * du1 - e1[k] * du2) * r
      }
    }
  }
  const out = new Float32Array(n * 4)
  for (let v = 0; v < n; v++) {
    const nx = verts[v * 8 + 3],
      ny = verts[v * 8 + 4],
      nz = verts[v * 8 + 5]
    let tx = tan[v * 3],
      ty = tan[v * 3 + 1],
      tz = tan[v * 3 + 2]
    const d = tx * nx + ty * ny + tz * nz
    tx -= nx * d
    ty -= ny * d
    tz -= nz * d
    let len = Math.hypot(tx, ty, tz)
    if (len < 1e-12) {
      // Any direction perpendicular to the normal: a surface with no uv
      // gradient has no tangent to speak of, and a zero one is NaN downstream.
      const ax = Math.abs(nx) < 0.9 ? 1 : 0
      tx = ax ? 0 : -nz
      ty = ax ? nz : 0
      tz = ax ? -ny : nx
      len = Math.hypot(tx, ty, tz) || 1
    }
    tx /= len
    ty /= len
    tz /= len
    // Handedness: whether the bitangent agrees with n × t.
    const cx = ny * tz - nz * ty,
      cy = nz * tx - nx * tz,
      cz = nx * ty - ny * tx
    const w = cx * bit[v * 3] + cy * bit[v * 3 + 1] + cz * bit[v * 3 + 2] < 0 ? -1 : 1
    out.set([tx, ty, tz, w], v * 4)
  }
  return out
}
