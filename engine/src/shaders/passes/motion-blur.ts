/**
 * Motion blur: a per-pixel velocity buffer and a gather along it.
 *
 * TWO PASSES, both run only while it is on (Engine.setMotionBlur):
 *
 *  · VELOCITY. Every model is drawn again against the scene's own depth (read
 *    only, compare "ahead or equal"), so exactly the visible surface writes:
 *    where its pixel is now minus where the same skinned vertex was last frame
 *    — last frame's skin matrices and last frame's camera. Its own pass and its
 *    own pipelines, so no scene pipeline changes and nothing costs anything
 *    while blur is off. The same skinned vertex path and @invariant position as
 *    the depth prepass, so it lands on the depths the scene wrote; the same
 *    alpha cutout, dissolve and x-ray tests, so a hair card's holes do not
 *    claim the card's motion. The target holds (velocity × coverage, coverage)
 *    so the MSAA resolve averages an edge into a fraction instead of garbage.
 *
 *  · GATHER, on the resolved HDR before bloom and the view transform (bloom
 *    then blooms the streak, as a camera's would). Where no model drew — the
 *    ground, the sky, a game stage — the motion is the CAMERA's, rebuilt from
 *    depth: the pixel's world point this frame, projected through last frame's
 *    camera. The blur is the pixel's own velocity × the shutter, a straight
 *    average of samples along it, clamped to a few percent of the frame.
 *
 * What it does not do (yet): a fast arm smears within itself, not out over a
 * still background, which would take a neighbourhood max velocity (tile max).
 * Morphs carry no motion: a blink is not blurred.
 */

import { DISSOLVE_WGSL, JELLY_VERTEX_WGSL, OBJECT_LIGHT_STRUCT_WGSL, XRAY_WGSL } from "../materials/common"

/** The velocity target: (velocity NDC xy × coverage, coverage, 1). */
export const MOTION_VELOCITY_FORMAT: GPUTextureFormat = "rgba16float"

export function motionVelocityWgsl(): string {
  return /* wgsl */ `
${DISSOLVE_WGSL}
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _p: f32, };
// The head of the material block, as the depth prepass declares it.
struct MaterialUniforms {
  diffuseColor: vec3f,
  alpha: f32,
  _skip0: vec4f,
  _skip1: vec4f,
  _skip2: f32,
  materialId: f32,
  _skip3: f32,
  dissolve: f32,
};
struct PrevCamera { viewProj: mat4x4f, };

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> prevCamera: PrevCamera;
@group(0) @binding(2) var diffuseSampler: sampler;
@group(1) @binding(0) var<storage, read> skinMats: array<mat4x4f>;
${OBJECT_LIGHT_STRUCT_WGSL}
@group(1) @binding(1) var<uniform> objectLight: ObjectLight;
@group(1) @binding(2) var<storage, read> prevSkinMats: array<mat4x4f>;
${XRAY_WGSL}
${JELLY_VERTEX_WGSL}
@group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
@group(2) @binding(1) var<uniform> material: MaterialUniforms;

struct VSOut {
  @builtin(position) @invariant position: vec4f,
  @location(0) uv: vec2f,
  @location(1) restPos: vec3f,
  @location(2) clipNow: vec4f,
  @location(3) clipPrev: vec4f,
};

@vertex fn vs(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) joints0: vec4<u32>,
  @location(4) weights0: vec4<f32>,
) -> VSOut {
  let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
  let invWeightSum = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
  let w = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);
  var pos4 = vec4f(position, 1.0);
  if (objectLight.jelly[0].w > 0.0) { pos4 = vec4f(rzJellyVertex(position, joints0, w), 1.0); }
  var skinned = vec4f(0.0);
  var prev = vec4f(0.0);
  for (var i = 0u; i < 4u; i++) {
    skinned += (skinMats[joints0[i]] * pos4) * w[i];
    prev += (prevSkinMats[joints0[i]] * pos4) * w[i];
  }
  var o: VSOut;
  o.position = camera.projection * camera.view * vec4f(skinned.xyz, 1.0);
  o.uv = uv;
  o.restPos = position;
  o.clipNow = o.position;
  o.clipPrev = prevCamera.viewProj * vec4f(prev.xyz, 1.0);
  return o;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let a = material.alpha * textureSample(diffuseTexture, diffuseSampler, in.uv).a;
  if (a < 0.5) { discard; }
  if (material.dissolve < 0.9995 && rz_dissolve_threshold(in.restPos) > material.dissolve) { discard; }
  if (rz_xray_cut(in.position, material.materialId)) { discard; }
  let now = in.clipNow.xy / in.clipNow.w;
  let prev = in.clipPrev.xy / max(in.clipPrev.w, 1e-5);
  return vec4f(now - prev, 1.0, 1.0);
}
`
}

/** Uniforms of the gather, in floats: invViewProj (16), prevViewProj (16),
 *  (scale, maxPx, width, height), (depthClear, 0, 0, 0). */
export const MOTION_BLUR_UNIFORM_FLOATS = 40

export const MOTION_BLUR_WGSL = /* wgsl */ `
struct Uniforms {
  invViewProj: mat4x4f,
  prevViewProj: mat4x4f,
  // x: shutter scale (this frame's motion → streak), y: longest streak in px,
  // zw: target size in px.
  params: vec4f,
  // x: the depth a cleared pixel holds (the far plane, either end).
  depth: vec4f,
};
@group(0) @binding(0) var hdr: texture_2d<f32>;
@group(0) @binding(1) var velocity: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_depth_multisampled_2d;
@group(0) @binding(3) var<uniform> u: Uniforms;
@group(0) @binding(4) var linearSampler: sampler;

struct VOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  o.uv = vec2f(p.x, 1.0 - p.y);
  return o;
}

const SAMPLES: i32 = 16;

@fragment fn fs(in: VOut) -> @location(0) vec4f {
  let size = u.params.zw;
  let px = vec2i(in.pos.xy);
  let vel = textureLoad(velocity, px, 0);
  // The camera's share: where this pixel's world point was last frame.
  let ndc = vec2f(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  var d = textureLoad(depthTex, px, 0);
  // A cleared pixel is the far plane, which an infinite projection cannot
  // invert: nudged a hair inside it, it reprojects as a direction.
  d = select(min(d, 0.999999), max(d, 0.000001), u.depth.x < 0.5);
  let world = u.invViewProj * vec4f(ndc, d, 1.0);
  let prevClip = u.prevViewProj * vec4f(world.xyz / world.w, 1.0);
  let camVel = ndc - prevClip.xy / max(prevClip.w, 1e-5);
  // Where models drew, theirs, by how much of the pixel they cover.
  let v = vel.xy + (1.0 - clamp(vel.z, 0.0, 1.0)) * camVel;
  // NDC to pixels (y down), times the shutter, clamped.
  var streak = vec2f(v.x, -v.y) * 0.5 * size * u.params.x;
  let len = length(streak);
  if (len > u.params.y) { streak *= u.params.y / len; }
  let centre = textureLoad(hdr, px, 0);
  if (length(streak) < 0.5) { return centre; }
  var sum = vec3f(0.0);
  for (var i = 0; i < SAMPLES; i++) {
    let t = (f32(i) + 0.5) / f32(SAMPLES) - 0.5;
    let uv = in.uv + streak * t / size;
    sum += textureSampleLevel(hdr, linearSampler, uv, 0.0).rgb;
  }
  return vec4f(sum / f32(SAMPLES), centre.a);
}
`
