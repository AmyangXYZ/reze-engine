// Depth-only prepass for the TRANSPARENT bucket. Transparent color draws keep
// depth write OFF so self-overlapping sheer cloth blends both layers instead of
// a triangle-order patchwork — but that leaves no depth record, so anything
// drawn later (the outline hulls) shows straight through the fabric as black
// shapes. This pass re-draws the transparent geometry depth-only AFTER its
// color pass: solid-enough texels (alpha ≥ 0.5) write depth, so outlines get
// occluded behind fabric exactly like they are behind opaque cloth, while
// truly sheer texels (a veil) stay non-occluding.
//
// Reuses mainPipelineLayout: camera g0b0, diffuseSampler g0b2, skinMats g1b0,
// diffuse texture g2b0, material uniforms g2b1 — the same bind groups the
// color draws already set, so drawing it costs no extra binding work.
//
// A FUNCTION rather than the constant it was, for the reason commonFsOutWgsl is
// one: the fragment outputs below depend on whether the device carries the id
// attachment, and that answer does not exist at import time. The constant could
// not have taken the outputs at all, which is most of why it did not have them.

import { sceneFsOutWgsl, sceneIdPadWgsl } from "./scene-contract"

export function transparentDepthPrepassWgsl(): string {
  return /* wgsl */ `
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _p: f32, };
struct MaterialUniforms {
  diffuseColor: vec3f,
  alpha: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(2) var diffuseSampler: sampler;
@group(1) @binding(0) var<storage, read> skinMats: array<mat4x4f>;
@group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
@group(2) @binding(1) var<uniform> material: MaterialUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex fn vs(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) joints0: vec4<u32>,
  @location(4) weights0: vec4<f32>,
) -> VSOut {
  let pos4 = vec4f(position, 1.0);
  let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
  let invWeightSum = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
  let w = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);
  var skinned = vec4f(0.0);
  for (var i = 0u; i < 4u; i++) {
    skinned += (skinMats[joints0[i]] * pos4) * w[i];
  }
  var o: VSOut;
  o.position = camera.projection * camera.view * vec4f(skinned.xyz, 1.0);
  o.uv = uv;
  return o;
}

// Every attachment the pass carries, declared and then not written: the
// pipeline takes all of them at writeMask 0 (see sceneTargets), so what this
// returns is discarded by the hardware and only the depth write survives — which
// is the entire purpose of the pass. Declaring them anyway is what keeps the
// pipeline valid on a browser that requires an output per target rather than
// per WRITTEN target. Costs one dead struct store on a fragment that already
// runs, because the alpha test below needs it to.
${sceneFsOutWgsl({ name: "PrepassOut", aux: "mask" })}
@fragment fn fs(in: VSOut) -> PrepassOut {
  let a = material.alpha * textureSample(diffuseTexture, diffuseSampler, in.uv).a;
  if (a < 0.5) { discard; }
  var out: PrepassOut;
  out.color = vec4f(0.0);
  out.mask = vec4f(0.0);
${sceneIdPadWgsl("out")}  return out;
}
`
}
