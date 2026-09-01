// The mesh itself, as lines, in the overlay layer.
//
// It skins here rather than reading the loader's CPU positions, and that is most
// of why the pass exists. What is on screen is skinned and morphed on the GPU;
// the CPU-side array is BIND POSE. Draw from that and the wireframe sits
// perfectly on a T-posed model and slides off every animated one — which is
// exactly the state a user is in while looking at weights.
//
// Each edge is expanded into a screen-space QUAD rather than drawn with line
// topology. WebGPU pins a line to one device pixel, and one device pixel on a 2x
// display is half a CSS pixel, which 4x MSAA then spreads across two of them:
// the result is a grey suggestion of a wireframe rather than a wireframe. So the
// draw is six vertices per edge, instanced, and the mesh is read through STORAGE
// buffers — there is no per-vertex stream, because a quad's two corners come
// from two different vertices of the model.
//
// The depth prepass shares this module's `vs` through its own entry point: it
// draws the solid triangles with vertex buffers, writing depth only, so the
// wireframe can be occluded by the body it belongs to.

export const WIREFRAME_SHADER_WGSL = /* wgsl */ `
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _pad: f32 };
struct WireUniforms { color: vec4f, viewport: vec2f, thicknessPx: f32, _pad: f32 };

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> wire: WireUniforms;

@group(1) @binding(0) var<storage, read> skinMats: array<mat4x4f>;
// The model's own vertex buffer: pos(3), normal(3), uv(2) — 8 floats a vertex.
@group(1) @binding(1) var<storage, read> vertices: array<f32>;
// uint16x4 per vertex, so two u32 each.
@group(1) @binding(2) var<storage, read> joints: array<u32>;
// unorm8x4 per vertex, one u32 each.
@group(1) @binding(3) var<storage, read> weights: array<u32>;
// Pairs of vertex indices, one pair an edge.
@group(1) @binding(4) var<storage, read> edges: array<u32>;

fn skinnedPos(i: u32) -> vec3f {
  let o = i * 8u;
  let rest = vec4f(vertices[o], vertices[o + 1u], vertices[o + 2u], 1.0);

  let j0 = joints[i * 2u];
  let j1 = joints[i * 2u + 1u];
  let packed = weights[i];
  var w = vec4f(
    f32(packed & 0xffu),
    f32((packed >> 8u) & 0xffu),
    f32((packed >> 16u) & 0xffu),
    f32((packed >> 24u) & 0xffu),
  ) / 255.0;
  let sum = w.x + w.y + w.z + w.w;
  w = select(vec4f(1.0, 0.0, 0.0, 0.0), w / max(sum, 1e-4), sum > 0.0001);

  var acc = vec4f(0.0);
  acc += (skinMats[j0 & 0xffffu] * rest) * w.x;
  acc += (skinMats[j0 >> 16u] * rest) * w.y;
  acc += (skinMats[j1 & 0xffffu] * rest) * w.z;
  acc += (skinMats[j1 >> 16u] * rest) * w.w;
  return acc.xyz;
}

// Six vertices an edge: (a,-1) (a,+1) (b,-1) (a,+1) (b,+1) (b,-1).
@vertex fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> @builtin(position) vec4f {
  let a = edges[ii * 2u];
  let b = edges[ii * 2u + 1u];
  let vp = camera.projection * camera.view;
  let ca = vp * vec4f(skinnedPos(a), 1.0);
  let cb = vp * vec4f(skinnedPos(b), 1.0);

  let atB = vi == 2u || vi == 4u || vi == 5u;
  let side = select(-1.0, 1.0, vi == 1u || vi == 3u || vi == 4u);
  let own = select(ca, cb, atB);
  let other = select(cb, ca, atB);

  let half = 0.5 * wire.viewport;
  let s0 = (own.xy / max(abs(own.w), 1e-6)) * half;
  let s1 = (other.xy / max(abs(other.w), 1e-6)) * half;
  let d = s1 - s0;
  // A degenerate edge has no direction on screen; any perpendicular will do,
  // and it is a dot either way.
  let tangent = select(vec2f(1.0, 0.0), normalize(d), length(d) > 1e-6);
  let normalPx = vec2f(-tangent.y, tangent.x);
  let offsetPx = normalPx * side * wire.thicknessPx * 0.5;
  return vec4f(own.xy + (offsetPx / half) * own.w, own.z, own.w);
}

// The depth prepass: the solid mesh through the ordinary vertex stream, writing
// depth and nothing else, so the far side of a body is hidden behind the near.
@vertex fn vsDepth(
  @location(0) position: vec3f,
  @location(1) joints0: vec4<u32>,
  @location(2) weights0: vec4<f32>,
) -> @builtin(position) vec4f {
  let rest = vec4f(position, 1.0);
  let sum = weights0.x + weights0.y + weights0.z + weights0.w;
  let w = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 / max(sum, 1e-4), sum > 0.0001);
  var acc = vec4f(0.0);
  for (var i = 0u; i < 4u; i++) {
    acc += (skinMats[joints0[i]] * rest) * w[i];
  }
  return camera.projection * camera.view * vec4f(acc.xyz, 1.0);
}

@fragment fn fs() -> @location(0) vec4f {
  // Premultiplied, like every other fragment in this layer.
  return vec4f(wire.color.rgb * wire.color.a, wire.color.a);
}
`
