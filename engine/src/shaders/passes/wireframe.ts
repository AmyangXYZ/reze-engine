// The mesh itself, as lines, in the overlay layer.
//
// It skins here rather than reading the loader's CPU positions, and that is the
// whole point of the pass existing. What is on screen is skinned and morphed on
// the GPU; the CPU-side array is BIND POSE. Draw from that and the wireframe
// sits perfectly on a T-posed model and slides off every animated one — which
// is exactly the state a user is in while looking at weights. So it reads the
// same vertex buffer, the same joints and weights, and the same skin matrices
// the colour pass does, through the same arithmetic.
//
// Line width in WebGPU is always one pixel, which suits a dense mesh: a thicker
// stroke on thirty thousand triangles is a filled shape, not a wireframe. The
// overlay layer is multisampled, so the hairlines antialias.

export const WIREFRAME_SHADER_WGSL = /* wgsl */ `
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _pad: f32 };
struct WireframeUniforms { color: vec4f };

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> wire: WireframeUniforms;
@group(1) @binding(0) var<storage, read> skinMats: array<mat4x4f>;

@vertex fn vs(
  @location(0) position: vec3f,
  @location(1) joints0: vec4<u32>,
  @location(2) weights0: vec4<f32>,
) -> @builtin(position) vec4f {
  let pos4 = vec4f(position, 1.0);
  let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
  let invWeightSum = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
  let w = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);
  var skinned = vec4f(0.0);
  for (var i = 0u; i < 4u; i++) {
    skinned += (skinMats[joints0[i]] * pos4) * w[i];
  }
  return camera.projection * camera.view * vec4f(skinned.xyz, 1.0);
}

@fragment fn fs() -> @location(0) vec4f {
  // Premultiplied, like every other fragment in this layer.
  return vec4f(wire.color.rgb * wire.color.a, wire.color.a);
}
`
