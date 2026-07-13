// GPU vertex-morph compute pass. Produces per-vertex morphed positions and writes them
// straight into the interleaved vertex buffer (stride 8: pos3 + normal3 + uv2), so every
// downstream pass (main / shadow / outline / pick) sees the morphed geometry with no VS
// change. Runs once per frame only when a model's morph weights changed.
//
// Morph data is a CSR (compressed sparse row) inversion of the per-morph vertex-offset
// lists: for vertex v, its entries live in [rowStart[v], rowStart[v+1]); each entry is a
// (morphIndex, xyz-offset) pair, ordered by ascending morph index — the same accumulation
// order the CPU path used, so results match. Group morphs are resolved to effective
// weights on the CPU and uploaded in `weights`.

export const MORPH_COMPUTE_WGSL = /* wgsl */ `
struct Params { vertexCount: u32 };

@group(0) @binding(0) var<storage, read> basePos: array<f32>;      // vertexCount * 3
@group(0) @binding(1) var<storage, read> rowStart: array<u32>;     // vertexCount + 1
@group(0) @binding(2) var<storage, read> colMorph: array<u32>;     // entryCount
@group(0) @binding(3) var<storage, read> colOffset: array<f32>;    // entryCount * 3
@group(0) @binding(4) var<storage, read> weights: array<f32>;      // morphCount (effective)
@group(0) @binding(5) var<storage, read_write> vtx: array<f32>;    // vertexCount * 8
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= params.vertexCount) { return; }

  let b3 = v * 3u;
  var px = basePos[b3];
  var py = basePos[b3 + 1u];
  var pz = basePos[b3 + 2u];

  let s = rowStart[v];
  let e = rowStart[v + 1u];
  for (var i = s; i < e; i = i + 1u) {
    let w = weights[colMorph[i]];
    if (w != 0.0) {
      let o3 = i * 3u;
      px = px + w * colOffset[o3];
      py = py + w * colOffset[o3 + 1u];
      pz = pz + w * colOffset[o3 + 2u];
    }
  }

  let vo = v * 8u; // VERTEX_STRIDE — write only the position slots
  vtx[vo] = px;
  vtx[vo + 1u] = py;
  vtx[vo + 2u] = pz;
}
`
