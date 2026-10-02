// The reflection probe's GPU passes, and the sky's prefilter.
//
// A game does not reflect the sky alone: it bakes a cube of the stage itself
// from somewhere in the middle of it, prefilters it down its mips with the
// specular lobe, and every surface reads it box-projected onto the stage's
// bounds. The engine does the same once a stage has loaded (Engine:
// captureReflectionProbe): six renders of the stage into a 128² cube, the
// sky laid in wherever nothing was drawn, a mip chain, then a GGX convolution
// into the 64² cube the materials sample (rzProbeSpecular in nodes.ts).
//
// The world's sky gets the same convolution down its equirect mips, in place
// of the box average they were built from.

import { NODES_WGSL } from "../materials/nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL, DISSOLVE_WGSL } from "../materials/common"

/** Face size of the capture, and of the cube the materials read. The game
 *  bakes 64²; the capture is twice that so the base level is antialiased. */
export const PROBE_CAPTURE_SIZE = 128
export const PROBE_SIZE = 64
/** 64² down to 1²: seven levels, the game's own count. */
export const PROBE_LEVELS = 7
export const PROBE_CAPTURE_LEVELS = 8

/**
 * The perceptual roughness level m of the probe was convolved with, on
 * Unity's probe curve: m = r·(1.7 − 0.7r)·6 (UNITY_SPECCUBE_LOD_STEPS), the
 * convention the game's baked probes follow and its lookup (×8, clamped at 6)
 * reads against. Level 0 is the unblurred capture.
 */
export function probeLevelRoughness(level: number): number {
  if (level <= 0) return 0
  const m = Math.min(level, 6)
  return Math.min((10.2 - Math.sqrt(Math.max(104.04 - 16.8 * m, 0))) / 8.4, 1)
}

/**
 * The perceptual roughness level m of the sky's equirect chain was convolved
 * with: the engine's own lookup is lod = sqrt(r)·(levels − 1), so level m is
 * r = (m / (levels − 1))².
 */
export function skyLevelRoughness(level: number, levels: number): number {
  if (level <= 0 || levels <= 1) return 0
  const t = level / (levels - 1)
  return Math.min(t * t, 1)
}

/** A cube face's direction for a uv on it (uv.y down), in the convention the
 *  GPU samples a cube with — the same one the capture cameras render. */
const FACE_DIR_WGSL = /* wgsl */ `
fn rz_face_dir(face: u32, uv: vec2f) -> vec3f {
  let s = uv.x * 2.0 - 1.0;
  let t = uv.y * 2.0 - 1.0;
  switch face {
    case 0u: { return normalize(vec3f(1.0, -t, -s)); }
    case 1u: { return normalize(vec3f(-1.0, -t, s)); }
    case 2u: { return normalize(vec3f(s, 1.0, t)); }
    case 3u: { return normalize(vec3f(s, -1.0, -t)); }
    case 4u: { return normalize(vec3f(s, -t, 1.0)); }
    default: { return normalize(vec3f(-s, -t, -1.0)); }
  }
}

struct RzFullOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs_full(@builtin(vertex_index) i: u32) -> RzFullOut {
  var o: RzFullOut;
  let x = f32(i32(i / 2u) * 4 - 1);
  let y = f32(i32(i % 2u) * 4 - 1);
  o.pos = vec4f(x, y, 0.0, 1.0);
  o.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
  return o;
}
`

/** GGX importance sampling, the filtered kind (Křivánek & Colbert): each
 *  sample reads the source at the mip whose texel matches its solid angle, so
 *  a few dozen taps integrate a wide lobe without the fireflies a point sample
 *  of a bright lamp would leave. */
const GGX_SAMPLING_WGSL = /* wgsl */ `
fn rz_hammersley(i: u32, n: u32) -> vec2f {
  var b = (i << 16u) | (i >> 16u);
  b = ((b & 0x55555555u) << 1u) | ((b & 0xAAAAAAAAu) >> 1u);
  b = ((b & 0x33333333u) << 2u) | ((b & 0xCCCCCCCCu) >> 2u);
  b = ((b & 0x0F0F0F0Fu) << 4u) | ((b & 0xF0F0F0F0u) >> 4u);
  b = ((b & 0x00FF00FFu) << 8u) | ((b & 0xFF00FF00u) >> 8u);
  return vec2f(f32(i) / f32(n), f32(b) * 2.3283064365386963e-10);
}

/** A half vector about N for the lobe at alpha = a (= perceptual r²). */
fn rz_ggx_half(xi: vec2f, N: vec3f, a: f32) -> vec3f {
  let phi = 6.283185307 * xi.x;
  let cosT = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
  let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
  let h = vec3f(sinT * cos(phi), sinT * sin(phi), cosT);
  let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(N.z) < 0.999);
  let tx = normalize(cross(up, N));
  let ty = cross(N, tx);
  return normalize(tx * h.x + ty * h.y + N * h.z);
}

fn rz_ggx_d(nh: f32, a: f32) -> f32 {
  let a2 = a * a;
  let d = nh * nh * (a2 - 1.0) + 1.0;
  return a2 / (3.141592653589793 * d * d);
}
`

/**
 * The compose step: the stage as captured, with the world's sky laid in
 * behind it wherever the capture's coverage says nothing was drawn — an open
 * stage reflects its sky, an interior its walls. Built on the material
 * prelude for rzWorldSpecularLod, which knows every kind of sky the scene may
 * have (an HDRI, a gradient, one colour). Only the bindings it reads end up in
 * its layout ("auto"): the light uniform, the sampler and the sky at group 0,
 * and its own three past the material set's.
 */
export const PROBE_COMPOSE_WGSL =
  NODES_WGSL +
  COMMON_MATERIAL_PRELUDE_WGSL +
  DISSOLVE_WGSL +
  FACE_DIR_WGSL +
  /* wgsl */ `
struct RzProbeStep { face: u32, lod: f32, roughness: f32, srcSize: f32 };
@group(0) @binding(20) var rzCaptureColor: texture_2d_array<f32>;
@group(0) @binding(21) var rzCaptureAux: texture_2d_array<f32>;
@group(0) @binding(22) var<uniform> rzStep: RzProbeStep;

/** Radiance the probe can hold: finite, not negative, inside half float.
 *  One pixel that is not — a NaN from any surface in the capture, or an
 *  emissive sheet that overflowed the half-float target to infinity — spreads
 *  through every GGX level built from it (each texel averages up to 128
 *  samples), and every Lit surface reading the probe went dark with it. A
 *  bad pixel here costs one texel, not the stage. */
fn rz_probe_radiance(c: vec3f) -> vec3f {
  let finite = (c == c) & (abs(c) < vec3f(65000.0));
  return clamp(select(vec3f(0.0), c, finite), vec3f(0.0), vec3f(65000.0));
}

@fragment fn fs_compose(in: RzFullOut) -> @location(0) vec4f {
  let size = textureDimensions(rzCaptureColor);
  let px = vec2u(min(in.pos.xy, vec2f(size) - vec2f(1.0)));
  let c = rz_probe_radiance(textureLoad(rzCaptureColor, px, rzStep.face, 0).rgb);
  let cover = saturate(textureLoad(rzCaptureAux, px, rzStep.face, 0).g);
  let sky = rz_probe_radiance(rzWorldSpecularLod(rz_face_dir(rzStep.face, in.uv), 0.0, 0.0));
  return vec4f(c + sky * (1.0 - cover), 1.0);
}
`

/**
 * The cube's own passes: a mip from the level above (mode 0 — one bilinear
 * tap at the child texel's direction is the 2×2 average), and the GGX
 * convolution into the probe the materials read (mode 1).
 */
export const PROBE_FILTER_WGSL =
  FACE_DIR_WGSL +
  GGX_SAMPLING_WGSL +
  /* wgsl */ `
struct RzProbeStep { face: u32, lod: f32, roughness: f32, srcSize: f32, mode: u32 };
@group(0) @binding(0) var rzSrc: texture_cube<f32>;
@group(0) @binding(1) var rzSamp: sampler;
@group(0) @binding(2) var<uniform> rzStep: RzProbeStep;

const RZ_PROBE_SAMPLES: u32 = 128u;

@fragment fn fs_filter(in: RzFullOut) -> @location(0) vec4f {
  let N = rz_face_dir(rzStep.face, in.uv);
  let r = rzStep.roughness;
  if (rzStep.mode == 0u || r < 0.002) {
    return vec4f(textureSampleLevel(rzSrc, rzSamp, N, rzStep.lod).rgb, 1.0);
  }
  let a = r * r;
  let texelSolid = 4.0 * 3.141592653589793 / (6.0 * rzStep.srcSize * rzStep.srcSize);
  let maxLod = f32(textureNumLevels(rzSrc) - 1u);
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var i = 0u; i < RZ_PROBE_SAMPLES; i++) {
    let H = rz_ggx_half(rz_hammersley(i, RZ_PROBE_SAMPLES), N, a);
    let L = 2.0 * dot(N, H) * H - N;
    let nl = dot(N, L);
    if (nl <= 0.0) { continue; }
    // N = V: the pdf over L is D·(N·H)/(4·V·H) = D/4.
    let pdf = rz_ggx_d(saturate(dot(N, H)), a) * 0.25;
    let sampleSolid = 1.0 / (f32(RZ_PROBE_SAMPLES) * pdf + 1e-6);
    let lod = clamp(0.5 * log2(sampleSolid / texelSolid) + 1.0, 0.0, maxLod);
    sum += textureSampleLevel(rzSrc, rzSamp, L, lod).rgb * nl;
    weight += nl;
  }
  return vec4f(sum / max(weight, 1e-4), 1.0);
}
`

/**
 * The sky's mips, convolved: level m of the equirect the materials read
 * (rzWorldSpecularLod) becomes the GGX-filtered sky at the roughness that
 * level stands for, sampled from the box-averaged chain the upload built.
 * Directions follow the composite's dome: u = 0.5 + atan2(x, z)/2π,
 * v = 0.5 − asin(y)/π.
 */
export const SKY_PREFILTER_WGSL =
  GGX_SAMPLING_WGSL +
  /* wgsl */ `
struct RzFullOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex fn vs_full(@builtin(vertex_index) i: u32) -> RzFullOut {
  var o: RzFullOut;
  let x = f32(i32(i / 2u) * 4 - 1);
  let y = f32(i32(i % 2u) * 4 - 1);
  o.pos = vec4f(x, y, 0.0, 1.0);
  o.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
  return o;
}

struct RzSkyStep { roughness: f32, srcWidth: f32, srcHeight: f32, spare: f32 };
@group(0) @binding(0) var rzSky: texture_2d<f32>;
@group(0) @binding(1) var rzSamp: sampler;
@group(0) @binding(2) var<uniform> rzStep: RzSkyStep;

const RZ_SKY_SAMPLES: u32 = 64u;

fn rz_sky_dir(uv: vec2f) -> vec3f {
  let phi = (uv.x - 0.5) * 6.283185307;
  let lat = (0.5 - uv.y) * 3.141592653589793;
  return vec3f(cos(lat) * sin(phi), sin(lat), cos(lat) * cos(phi));
}

fn rz_sky_uv(d: vec3f) -> vec2f {
  return vec2f(0.5 + atan2(d.x, d.z) * 0.15915494309, 0.5 - asin(clamp(d.y, -1.0, 1.0)) * 0.31830988618);
}

@fragment fn fs_sky(in: RzFullOut) -> @location(0) vec4f {
  let N = rz_sky_dir(in.uv);
  let a = rzStep.roughness * rzStep.roughness;
  let texelSolid = 4.0 * 3.141592653589793 / (rzStep.srcWidth * rzStep.srcHeight);
  let maxLod = f32(textureNumLevels(rzSky) - 1u);
  var sum = vec3f(0.0);
  var weight = 0.0;
  for (var i = 0u; i < RZ_SKY_SAMPLES; i++) {
    let H = rz_ggx_half(rz_hammersley(i, RZ_SKY_SAMPLES), N, a);
    let L = 2.0 * dot(N, H) * H - N;
    let nl = dot(N, L);
    if (nl <= 0.0) { continue; }
    let pdf = rz_ggx_d(saturate(dot(N, H)), a) * 0.25;
    let sampleSolid = 1.0 / (f32(RZ_SKY_SAMPLES) * pdf + 1e-6);
    let lod = clamp(0.5 * log2(sampleSolid / texelSolid) + 1.0, 0.0, maxLod);
    sum += textureSampleLevel(rzSky, rzSamp, rz_sky_uv(L), lod).rgb * nl;
    weight += nl;
  }
  return vec4f(sum / max(weight, 1e-4), 1.0);
}
`
