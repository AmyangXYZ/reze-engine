import { sceneLightApi } from "../scene-light-api"
import { lightsApi, MAX_LIGHTS } from "../lights"

/**
 * The frame, read back as numbers — what an agent (or a test) measures instead
 * of squinting at a picture.
 *
 * A screenshot answers "what does it look like". It cannot answer the questions
 * a lighting artist actually asks of a render: is her face in the sun's shadow,
 * which lamp is carrying the key, how many stops over is that highlight, which
 * pixels are skin. Every one of those is a number the frame already computed
 * and threw away; these passes keep them.
 *
 * TWO ENTRY POINTS over one prelude:
 *
 *   pixels — one invocation per texel of the frame just rendered. Reads the id
 *            attachment, the depth and the linear HDR resolve, rebuilds the
 *            world point from the depth and a normal from its neighbours, and
 *            asks the same shadow atlas and lamp records the materials asked.
 *   probe  — the same light questions at points the caller names (a bone, a
 *            face), with each lamp's share kept apart.
 *
 * The normal is RECONSTRUCTED from depth, not the one the material shaded
 * with: the frame never stores normals, and storing them for every frame to
 * serve an occasional readback would be the cost the id attachment's
 * store-only-when-read rule exists to avoid. On a silhouette edge it is the
 * flatter of the two neighbours, which is right for every texel that is not
 * the edge itself.
 */

/** Floats per texel in the pixels output — see PIXEL_FIELDS. */
export const AOV_PIXEL_STRIDE = 12

/** What each of a texel's floats is, in order. */
export const AOV_PIXEL_FIELDS = [
  "r", "g", "b", // linear HDR, before exposure and the view transform
  "depth", // distance from the eye, world units; 0 where nothing drew
  "sunShadow", // 1 lit, 0 in a caster's shadow; -1 where nothing drew
  "sunFacing", // N·L toward the sun, -1..1 (the toon ramp's input)
  "lampR", "lampG", "lampB", // every positional lamp's light arriving here
  "ambR", "ambG", "ambB", // the world's light arriving here
] as const

/** Floats per probe point before its per-lamp block. */
export const AOV_PROBE_HEAD = 12
/** Floats per probe point: the head, then rgb+pad for every lamp slot. */
export const AOV_PROBE_STRIDE = AOV_PROBE_HEAD + MAX_LIGHTS * 4

// Bindings 0..3 are the scene light API, 4..6 the lamps and their cookies.
const PRELUDE = /* wgsl */ `
${sceneLightApi(true, 0, 0)}
${lightsApi(0, 4, "0xffffffffu", { binding: 5, sampler: "_aovSampler" })}
@group(0) @binding(6) var _aovSampler: sampler;

/** The sun's cast shadow at p, nudged along n off the surface — the
 *  materials' rzSunOcclusion, without the facing test (sunFacing carries it). */
fn aovSunShadow(p: vec3f, n: vec3f) -> f32 {
  let castAmt = _rzLight.lights[0].direction.w;
  if (castAmt <= 0.0) { return 1.0; }
  let a = _rzSunLocate(p + n * 0.08);
  if (a.w < 0.0) { return 1.0; }
  return mix(1.0, _rzSunTaps(a), castAmt);
}

fn aovToSun() -> vec3f { return normalize(-_rzLight.lights[0].direction.xyz); }
`

export const AOV_PIXELS_WGSL = /* wgsl */ `
${PRELUDE}
@group(0) @binding(7) var idTex: texture_multisampled_2d<u32>;
@group(0) @binding(8) var depthTex: texture_depth_multisampled_2d;
@group(0) @binding(9) var hdrTex: texture_2d<f32>;
struct AovU { invViewProj: mat4x4f, eye: vec4f, size: vec4u, }
@group(0) @binding(10) var<uniform> u: AovU;
@group(0) @binding(11) var<storage, read_write> outF: array<f32>;
@group(0) @binding(12) var<storage, read_write> outId: array<u32>;

fn worldAt(c: vec2i, z: f32) -> vec3f {
  let uv = (vec2f(c) + 0.5) / vec2f(u.size.xy);
  let h = u.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, z, 1.0);
  return h.xyz / h.w;
}

fn drawn(c: vec2i) -> bool {
  if (any(c < vec2i(0)) || any(c >= vec2i(u.size.xy))) { return false; }
  let id = textureLoad(idTex, c, 0).xy;
  return id.x != 0u || id.y != 0u;
}

/** The step from p to the neighbour at c + d (or c - d, negated), whichever is
 *  nearer in depth: across a silhouette the far neighbour belongs to something
 *  else, and a normal built from it tilts toward the background. */
fn tangent(c: vec2i, d: vec2i, p: vec3f) -> vec3f {
  let a = c + d;
  let b = c - d;
  let okA = drawn(a);
  let okB = drawn(b);
  var ta = vec3f(0.0);
  var tb = vec3f(0.0);
  if (okA) { ta = worldAt(a, textureLoad(depthTex, a, 0)) - p; }
  if (okB) { tb = p - worldAt(b, textureLoad(depthTex, b, 0)); }
  if (okA && okB) { return select(tb, ta, dot(ta, ta) <= dot(tb, tb)); }
  if (okA) { return ta; }
  return tb;
}

@compute @workgroup_size(8, 8)
fn pixels(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= u.size.x || gid.y >= u.size.y) { return; }
  let c = vec2i(gid.xy);
  let i = gid.y * u.size.x + gid.x;
  let o = i * ${AOV_PIXEL_STRIDE}u;
  let id = textureLoad(idTex, c, 0).xy;
  let rgb = textureLoad(hdrTex, c, 0).rgb;
  // Material in the low half, object in the high — the attachment's own order
  // (scene-contract: vec2u(material, object)).
  outId[i] = (id.y << 16u) | (id.x & 0xffffu);
  outF[o] = rgb.r;
  outF[o + 1u] = rgb.g;
  outF[o + 2u] = rgb.b;
  for (var k = 3u; k < ${AOV_PIXEL_STRIDE}u; k++) { outF[o + k] = 0.0; }
  if (id.x == 0u && id.y == 0u) {
    outF[o + 4u] = -1.0;
    return;
  }
  let p = worldAt(c, textureLoad(depthTex, c, 0));
  var n = cross(tangent(c, vec2i(0, 1), p), tangent(c, vec2i(1, 0), p));
  let toEye = u.eye.xyz - p;
  if (dot(n, n) < 1e-12) { n = toEye; }
  n = normalize(n);
  // Toward the eye: the side of the surface this texel shows.
  if (dot(n, toEye) < 0.0) { n = -n; }
  let lamps = rzLightsDiffuse(p, n);
  let amb = rzWorldAmbient(n);
  outF[o + 3u] = length(toEye);
  outF[o + 4u] = aovSunShadow(p, n);
  outF[o + 5u] = dot(n, aovToSun());
  outF[o + 6u] = lamps.r;
  outF[o + 7u] = lamps.g;
  outF[o + 8u] = lamps.b;
  outF[o + 9u] = amb.r;
  outF[o + 10u] = amb.g;
  outF[o + 11u] = amb.b;
}
`

export const AOV_PROBE_WGSL = /* wgsl */ `
${PRELUDE}
struct ProbePoint { p: vec4f, n: vec4f, }
@group(0) @binding(7) var<storage, read> points: array<ProbePoint>;
@group(0) @binding(8) var<storage, read_write> outF: array<f32>;

@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&points)) { return; }
  let p = points[i].p.xyz;
  let n = normalize(points[i].n.xyz);
  let o = i * ${AOV_PROBE_STRIDE}u;
  // Colour and strength travel apart in the light uniform (writeSun); the
  // light arriving is their product.
  let sun = _rzLight.lights[0].color.rgb * _rzLight.lights[0].color.w;
  let amb = rzWorldAmbient(n);
  outF[o] = aovSunShadow(p, n);
  outF[o + 1u] = dot(n, aovToSun());
  outF[o + 2u] = sun.r;
  outF[o + 3u] = sun.g;
  outF[o + 4u] = sun.b;
  outF[o + 5u] = amb.r;
  outF[o + 6u] = amb.g;
  outF[o + 7u] = amb.b;
  let count = rzLightCount();
  outF[o + 8u] = f32(count);
  // The document's lamps come first (setLights order); the rest are effects'.
  outF[o + 9u] = f32(_rzLightDocCount());
  for (var k = 10u; k < ${AOV_PROBE_HEAD}u; k++) { outF[o + k] = 0.0; }
  for (var l = 0u; l < ${MAX_LIGHTS}u; l++) {
    var c = vec3f(0.0);
    if (l < count) { c = _rzLightOne(l, p, n); }
    let b = o + ${AOV_PROBE_HEAD}u + l * 4u;
    outF[b] = c.r;
    outF[b + 1u] = c.g;
    outF[b + 2u] = c.b;
    outF[b + 3u] = 0.0;
  }
}
`
