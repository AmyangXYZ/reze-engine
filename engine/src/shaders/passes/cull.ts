// GPU frustum cull writing indirect draw arguments.
//
// One dispatch covers every material draw in the scene and writes TWO argument
// buffers — the camera pass and the shadow pass — because they differ only in
// which frustum they test against, and splitting them into two dispatches would
// read the same metadata twice. See docs/frame-graph.md.
//
// The compute writes ONLY `instanceCount` (word 1 of each 5-word record). The
// other four words — indexCount, firstIndex, baseVertex, firstInstance — are
// seeded by the CPU when the draw list changes, which is the same moment the
// metadata buffer is uploaded. Nothing about them varies per frame, so shipping
// them through the compute would be a per-frame write in exchange for nothing.
//
// Two bound kinds, because a skinned mesh has no usable model-space bounds:
//   RIGID   — every bone of the model shares one skin matrix (a stage, or a
//             character still in its bind pose), so each material's load-time
//             model-space AABB is valid and gets transformed by that matrix.
//             This is what makes a 200-material stage worth culling at all.
//   SKINNED — animation invalidates per-material bounds, so the whole model is
//             one world-space sphere derived from its posed bone positions.
//
// Planes are NORMALIZED on the CPU. The AABB test would tolerate unnormalized
// planes (distance and extent scale together), the sphere test would not.

export const CULL_COMPUTE_WGSL = /* wgsl */ `
struct DrawMeta {
  lo: vec3f,      // model-space AABB min — read only for RIGID models
  model: u32,     // index into models[]
  hi: vec3f,      // model-space AABB max
  flags: u32,     // bit0 = casts shadow
};

struct ModelRec {
  xform: mat4x4f, // model space -> world; meaningful only when RIGID
  sphere: vec4f,  // world-space centre.xyz + radius.w; meaningful when SKINNED
  flags: u32,     // bit0 = visible, bit1 = rigid
  pad0: u32,
  pad1: u32,
  pad2: u32,
};

// Camera planes at 0..5, light planes at 6..11.
// counts.x = draw count, counts.y = 0 to pass everything (setCullEnabled).
struct Frusta {
  planes: array<vec4f, 12>,
  counts: vec4u,
};

@group(0) @binding(0) var<storage, read> metas: array<DrawMeta>;
@group(0) @binding(1) var<storage, read> models: array<ModelRec>;
@group(0) @binding(2) var<uniform> frusta: Frusta;
@group(0) @binding(3) var<storage, read_write> cameraArgs: array<u32>;
@group(0) @binding(4) var<storage, read_write> shadowArgs: array<u32>;

const DRAW_CASTS_SHADOW: u32 = 1u;
const MODEL_VISIBLE: u32 = 1u;
const MODEL_RIGID: u32 = 2u;

// Signed distance of the AABB's most-positive corner: the centre distance plus
// the extent projected onto |n|. Negative means every corner is behind the
// plane, which is the only case that can reject.
fn aabbVisible(base: u32, c: vec3f, e: vec3f) -> bool {
  for (var i = 0u; i < 6u; i = i + 1u) {
    let p = frusta.planes[base + i];
    if (dot(p.xyz, c) + p.w + dot(e, abs(p.xyz)) < 0.0) { return false; }
  }
  return true;
}

fn sphereVisible(base: u32, s: vec4f) -> bool {
  for (var i = 0u; i < 6u; i = i + 1u) {
    let p = frusta.planes[base + i];
    if (dot(p.xyz, s.xyz) + p.w + s.w < 0.0) { return false; }
  }
  return true;
}

@compute @workgroup_size(64)
fn cs(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= frusta.counts.x) { return; }

  // Named dm, not meta: "meta" is a WGSL reserved word (reserved for future use,
  // so it parses nowhere). No backticks anywhere in this string either — it is a
  // template literal, and one would end the shader mid-comment.
  let dm = metas[i];
  let m = models[dm.model];
  let visible = (m.flags & MODEL_VISIBLE) != 0u;

  var inCamera = false;
  var inLight = false;
  if (visible) {
    if ((m.flags & MODEL_RIGID) != 0u) {
      let c = (dm.lo + dm.hi) * 0.5;
      let e = (dm.hi - dm.lo) * 0.5;
      let wc = (m.xform * vec4f(c, 1.0)).xyz;
      // Extent of the transformed box: the abs of the linear part applied to the
      // half-extent. Exact for the axis-aligned case, conservative for a rotation.
      let a = mat3x3f(abs(m.xform[0].xyz), abs(m.xform[1].xyz), abs(m.xform[2].xyz));
      let we = a * e;
      inCamera = aabbVisible(0u, wc, we);
      inLight = aabbVisible(6u, wc, we);
    } else {
      inCamera = sphereVisible(0u, m.sphere);
      inLight = sphereVisible(6u, m.sphere);
    }
  }

  // Culling off (setCullEnabled) passes everything, so a scene can be A/B'd
  // against its own unculled self without rebuilding anything. The pass still
  // runs, so the diagnostics keep reporting.
  let off = frusta.counts.y == 0u;
  let castsShadow = (dm.flags & DRAW_CASTS_SHADOW) != 0u;
  cameraArgs[i * 5u + 1u] = select(0u, 1u, inCamera || off);
  shadowArgs[i * 5u + 1u] = select(0u, 1u, (inLight && castsShadow) || off);
}
`
