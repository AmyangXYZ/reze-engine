// The depth-only prime, one module behind four pipelines.
//
// The oldest fps complaint the engine had — zoom close and the frame drops, in
// every material generation — was per-fragment shading TIMES layering: an MMD
// model at close-up is cloth over body over face under hair, author-order drawn,
// every buried layer fully shaded and then covered. This module lets depth go
// down FIRST so the colour passes shade each pixel once:
//
//   · opaque prepass  (CUTOFF 0.5) — plain auto-class opaque draws, before the
//     opaque colour walk. See drawOpaqueDepthPrepass for who is in and why.
//   · hair prime      (CUTOFF 1.0, stencil not-equal) — between the non-hair and
//     hair colour walks, fenced off the eye silhouette the see-through-hair
//     stencil pass needs. See drawHairDepthPrime.
//   · transparent solid prime (CUTOFF 1.0) — a translucent material's alpha-1
//     texels, where over-blending is plain replacement and the buried work
//     provably never shows. See drawTransparentSolidPrepass.
//   · transparent depth prepass (CUTOFF 0.5, AFTER colour) — the original
//     occupant, dormant: re-records sheer fabric's depth so outline hulls are
//     occluded behind it. Kept for a future OIT path.
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
  // @invariant, to the same end as the material VertexOutput: the colour pass
  // must land on exactly the depths this wrote.
  @builtin(position) @invariant position: vec4f,
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
// The cutout threshold, per pipeline. 0.5 is the OPAQUE prime's "solid enough"
// — safe there because opaque colour replaces rather than blends. The
// TRANSPARENT prime overrides it to 1.0: a translucent fragment's blend reads
// what is behind it, so only a texel at EXACTLY alpha 1 — where over-blending
// collapses to plain replacement and the destination stops mattering — may
// claim depth ahead of its buried layers without changing the pixel.
override CUTOFF: f32 = 0.5;
@fragment fn fs(in: VSOut) -> PrepassOut {
  let a = material.alpha * textureSample(diffuseTexture, diffuseSampler, in.uv).a;
  if (a < CUTOFF) { discard; }
  var out: PrepassOut;
  out.color = vec4f(0.0);
  out.mask = vec4f(0.0);
${sceneIdPadWgsl("out")}  return out;
}
`
}
