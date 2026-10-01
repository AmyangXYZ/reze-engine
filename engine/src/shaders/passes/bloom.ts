// Bloom — Aether Gazer's own chain (AGSimPostFX.Bloom, URP-style): a soft-knee
// prefilter to half size, a separable Gaussian down to a few pixels, then back
// up level by level, each blend leaning toward the coarser level by `scatter`.
// The composite adds the result × tint × intensity before the view transform.

const FULLSCREEN_VS = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
`

// Full-res HDR → half-res. The game takes ONE bilinear sample at the half-res
// pixel's centre, which is the 2×2 box of the block under it; here the four
// texels are loaded so each can be un-premultiplied (the scene target is
// premultiplied; its alpha lives in the aux target's .g) and gated by the bloom
// mask (.r) first.
export const BLOOM_PREFILTER_SHADER_WGSL = `${FULLSCREEN_VS}
@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: vec4<f32>; // threshold, knee, clamp, scatter
@group(0) @binding(2) var maskTex: texture_2d<f32>;

fn fetch(c: vec2<i32>) -> vec3f {
  let d = vec2<i32>(textureDimensions(hdrTex));
  let cc = clamp(c, vec2<i32>(0), d - vec2<i32>(1));
  let aux = textureLoad(maskTex, cc, 0);
  return max(textureLoad(hdrTex, cc, 0).rgb / max(aux.g, 1e-6), vec3f(0.0)) * aux.r;
}

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let base = vec2<i32>(p.xy - vec2f(0.5)) * 2;
  let avg = (fetch(base) + fetch(base + vec2<i32>(1, 0)) + fetch(base + vec2<i32>(0, 1)) + fetch(base + vec2<i32>(1, 1))) * 0.25;
  let c = min(avg, vec3f(params.z));
  // URP's quadratic soft knee on the brightest channel.
  let bright = max(c.r, max(c.g, c.b));
  let knee = params.y;
  let soft = clamp(bright - params.x + knee, 0.0, 2.0 * knee);
  let curve = soft * soft / (4.0 * knee + 1e-4);
  return vec4f(c * (max(curve, bright - params.x) / max(bright, 1e-4)), 1.0);
}
`

// Horizontal 9-tap Gaussian (σ ≈ 2 at the destination's scale), its taps two
// SOURCE texels apart: the destination is half the source, so this both halves
// and blurs. Each tap lands on a texel boundary, so bilinear averages pairs.
export const BLOOM_BLUR_H_SHADER_WGSL = `${FULLSCREEN_VS}
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let srcDims = vec2f(textureDimensions(srcTex));
  let dstDims = max(floor(srcDims * 0.5), vec2f(1.0));
  let uv = p.xy / dstDims;
  let dx = vec2f(2.0 / srcDims.x, 0.0);
  var w = array<f32, 5>(0.22702703, 0.19459459, 0.12162162, 0.05405405, 0.01621622);
  var o = textureSampleLevel(srcTex, srcSamp, uv, 0.0).rgb * w[0];
  for (var i = 1; i < 5; i++) {
    let off = dx * f32(i);
    o += (textureSampleLevel(srcTex, srcSamp, uv - off, 0.0).rgb + textureSampleLevel(srcTex, srcSamp, uv + off, 0.0).rgb) * w[i];
  }
  return vec4f(o, 1.0);
}
`

// Vertical pass at the same size: the same 9-tap Gaussian as five bilinear taps.
export const BLOOM_BLUR_V_SHADER_WGSL = `${FULLSCREEN_VS}
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var srcSamp: sampler;

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let dims = vec2f(textureDimensions(srcTex));
  let uv = p.xy / dims;
  let near = vec2f(0.0, 1.3846154 / dims.y);
  let far = vec2f(0.0, 3.2307692 / dims.y);
  var o = textureSampleLevel(srcTex, srcSamp, uv, 0.0).rgb * 0.22702703;
  o += (textureSampleLevel(srcTex, srcSamp, uv - near, 0.0).rgb + textureSampleLevel(srcTex, srcSamp, uv + near, 0.0).rgb) * 0.31621623;
  o += (textureSampleLevel(srcTex, srcSamp, uv - far, 0.0).rgb + textureSampleLevel(srcTex, srcSamp, uv + far, 0.0).rgb) * 0.07027027;
  return vec4f(o, 1.0);
}
`

// One level up: this level's blur leaned toward the coarser one by scatter.
export const BLOOM_UPSAMPLE_SHADER_WGSL = `${FULLSCREEN_VS}
@group(0) @binding(0) var highTex: texture_2d<f32>;  // down[i]
@group(0) @binding(1) var lowTex: texture_2d<f32>;   // coarser: down[i+1] at the top, up[i+1] after
@group(0) @binding(2) var srcSamp: sampler;
@group(0) @binding(3) var<uniform> params: vec4<f32>; // threshold, knee, clamp, scatter

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  let uv = p.xy / vec2f(textureDimensions(highTex));
  let high = textureSampleLevel(highTex, srcSamp, uv, 0.0).rgb;
  let low = textureSampleLevel(lowTex, srcSamp, uv, 0.0).rgb;
  return vec4f(mix(high, low, params.w), 1.0);
}
`
