// Shadow map depth pass. Skinned VS + alpha-test FS (depth-only attachment, no
// color targets): texels where diffuse-texture alpha × material alpha fall below
// the cutoff are discarded, so lace casts lace-shaped shadows and a true veil
// casts nothing — per texel, with no per-material sheerness classification.
// Group 1 is the main pass's per-material bind group reused as-is; only
// bindings 0/1 are declared here (a layout may carry bindings a shader ignores).

export const SHADOW_DEPTH_SHADER_WGSL = /* wgsl */ `
struct LightVP { viewProj: mat4x4f, };
@group(0) @binding(0) var<uniform> lp: LightVP;
@group(0) @binding(1) var<storage, read> skinMats: array<mat4x4f>;
@group(0) @binding(2) var texSampler: sampler;
@group(1) @binding(0) var diffuseTexture: texture_2d<f32>;
struct MaterialDiffuse { diffuse: vec4f, };
@group(1) @binding(1) var<uniform> material: MaterialDiffuse;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

@vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f,
  @location(3) joints0: vec4<u32>, @location(4) weights0: vec4<f32>) -> VSOut {
  let pos4 = vec4f(position, 1.0);
  let ws = weights0.x + weights0.y + weights0.z + weights0.w;
  let inv = select(1.0, 1.0 / ws, ws > 0.0001);
  let nw = select(vec4f(1.0,0.0,0.0,0.0), weights0 * inv, ws > 0.0001);
  var sp = vec4f(0.0);
  for (var i = 0u; i < 4u; i++) { sp += (skinMats[joints0[i]] * pos4) * nw[i]; }
  var out: VSOut;
  out.position = lp.viewProj * vec4f(sp.xyz, 1.0);
  out.uv = uv;
  return out;
}

@fragment fn fs(in: VSOut) {
  let alpha = textureSample(diffuseTexture, texSampler, in.uv).a * material.diffuse.a;
  if (alpha < 0.5) { discard; }
}
`
