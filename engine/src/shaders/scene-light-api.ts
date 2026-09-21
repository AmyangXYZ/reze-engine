import { SHADOW_CASCADES } from "../shadow-cascades"
// The scene's own light, as an effect author sees it: the sun's shadow and the
// world's ambient.
//
// A lawn standing in a character's shadow has to darken where the floor under
// it does, or the figure floats: the ground catcher draws her shadow, and the
// blades drawn over it hide it. And the shade has to be the SCENE'S shade — her
// shaded side takes the world's colour, and a lawn shaded by its own guess
// reads as a patch of different flowers beside her. Both are already computed
// for the materials every frame; this hands the same lookups to effects.
//
// Bound for real in the particle SHADING stage only. That is where geometry an
// effect draws inside the scene pass is shaded, and the shadow pass has already
// run by then. Every other module that compiles the author's file gets stubs:
// rzShadow answers 1, lit, which is what a point with no occlusion information
// is — the same answer sampleShadow gives outside every cascade — and
// rzWorldAmbient answers black, no light.

/**
 * The world's light at a surface facing n — the flat colour, or the installed
 * HDRI's irradiance (sh[0].w = 1), evaluated from folded SH coefficients (see
 * ibl.ts for the folding; the shader is a plain polynomial in the normal).
 *
 * Written against whichever name the module gives the light uniform, so the
 * materials, the ground and an effect all evaluate one polynomial.
 */
export const worldAmbientWgsl = (u: string) => /* wgsl */ `
fn rzWorldAmbient(n: vec3f) -> vec3f {
  if (${u}.sh[0].w < 0.5) { return ${u}.ambientColor.xyz; }
  let x = n.x;
  let y = n.y;
  let z = n.z;
  let c = ${u}.sh[0].xyz
    + ${u}.sh[1].xyz * y + ${u}.sh[2].xyz * z + ${u}.sh[3].xyz * x
    + ${u}.sh[4].xyz * (x * y) + ${u}.sh[5].xyz * (y * z)
    + ${u}.sh[6].xyz * (3.0 * z * z - 1.0) + ${u}.sh[7].xyz * (x * z)
    + ${u}.sh[8].xyz * (x * x - y * y);
  return max(c, vec3f(0.0));
}
`

/**
 * 3×3 PCF over one cascade's map, for a point already in that cascade's NDC.
 *
 * Shared with the materials, so an effect's shadow and her own shadow soften
 * alike. The sampler is linear-filtered, which makes each tap a 2×2 bilinear
 * compare. Unrolled — Safari's Metal backend doesn't unroll nested shadow
 * loops reliably. The body expects `ndc` in scope and returns.
 */
export const pcf9 = (map: string, ts: string, sampler = "shadowSampler") => /* wgsl */ `
  let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let cmpZ = ndc.z - 0.001;
  let ts = ${ts};
  let s00 = textureSampleCompareLevel(${map}, ${sampler}, suv + vec2f(-ts, -ts), cmpZ);
  let s10 = textureSampleCompareLevel(${map}, ${sampler}, suv + vec2f(0.0, -ts), cmpZ);
  let s20 = textureSampleCompareLevel(${map}, ${sampler}, suv + vec2f( ts, -ts), cmpZ);
  let s01 = textureSampleCompareLevel(${map}, ${sampler}, suv + vec2f(-ts, 0.0), cmpZ);
  let s11 = textureSampleCompareLevel(${map}, ${sampler}, suv, cmpZ);
  let s21 = textureSampleCompareLevel(${map}, ${sampler}, suv + vec2f( ts, 0.0), cmpZ);
  let s02 = textureSampleCompareLevel(${map}, ${sampler}, suv + vec2f(-ts,  ts), cmpZ);
  let s12 = textureSampleCompareLevel(${map}, ${sampler}, suv + vec2f(0.0,  ts), cmpZ);
  let s22 = textureSampleCompareLevel(${map}, ${sampler}, suv + vec2f( ts,  ts), cmpZ);
  return (s00 + s10 + s20 + s01 + s11 + s21 + s02 + s12 + s22) * (1.0 / 9.0);
`

/** Bindings the real accessors take, from `binding` upward. */
export const SCENE_LIGHT_API_BINDINGS = 5

/**
 * `rzShadow(p)`: how much of the sun reaches world point `p` — 1 lit, 0 in a
 * caster's shadow, soft in between.
 *
 * `rzWorldAmbient(n)`: the world's light arriving at a surface facing `n` —
 * the materials' own ambient, so an effect's shade can match hers.
 *
 * With `on`, declares five bindings from `binding`:
 *   +0 the light uniform (ambient, the sun — direction, and in .w the shadow
 *      switch — and the HDRI's irradiance)
 *   +1 the cascade view-projections, then the caster sphere
 *   +2 the near cascade's map   +3 the far cascade's map
 *   +4 the comparison sampler
 *
 * It takes a POINT and no normal. The materials' sampleShadow also rejects a
 * surface facing away from the sun, which is right for skin and wrong for a
 * blade of grass or a petal: those are thin, lit through, and shaded by their
 * own N·L. The point is nudged toward the sun instead, which lifts a blade's
 * root off a stage floor that casts into the same map.
 */
export function sceneLightApi(on: boolean, group: number, binding: number): string {
  if (!on) {
    return /* wgsl */ `
fn rzShadow(p: vec3f) -> f32 { return 1.0; }
fn rzWorldAmbient(n: vec3f) -> vec3f { return vec3f(0.0); }
`
  }
  const n = SHADOW_CASCADES.length
  return /* wgsl */ `
// The light uniform, as the materials lay it out; lights[0] is the sun.
struct _RzSceneLight { direction: vec4f, color: vec4f, }
struct _RzSceneLightU { ambientColor: vec4f, lights: array<_RzSceneLight, 4>, sh: array<vec4f, 9>, }
// The cascades' matrices, inner to outer, then every caster in one sphere
// (x, y, z, radius): radius 0 = nothing casts, negative = do not test.
struct _RzShadowBlock { viewProj: array<mat4x4f, ${n}>, casters: vec4f, }
@group(${group}) @binding(${binding}) var<uniform> _rzLight: _RzSceneLightU;
@group(${group}) @binding(${binding + 1}) var<uniform> _rzShadowVP: _RzShadowBlock;
@group(${group}) @binding(${binding + 2}) var _rzShadowNear: texture_depth_2d;
@group(${group}) @binding(${binding + 3}) var _rzShadowFar: texture_depth_2d;
@group(${group}) @binding(${binding + 4}) var _rzShadowCmp: sampler_comparison;

fn _rzShadowNearTaps(ndc: vec3f) -> f32 {
${pcf9("_rzShadowNear", `1.0 / ${SHADOW_CASCADES[0].mapSize}.0`, "_rzShadowCmp")}
}

fn _rzShadowFarTaps(ndc: vec3f) -> f32 {
${pcf9("_rzShadowFar", `1.0 / ${SHADOW_CASCADES[n - 1].mapSize}.0`, "_rzShadowCmp")}
}

fn rzShadow(p: vec3f) -> f32 {
  // The scene's one shadow switch, on the sun — see sampleShadow.
  let castAmt = _rzLight.lights[0].direction.w;
  if (castAmt <= 0.0) { return 1.0; }
  let toSun = normalize(-_rzLight.lights[0].direction.xyz);
  // Can anything cast onto this point at all? The ground's test, for the same
  // two reasons: most of a lawn is nowhere near her, and a scene with no
  // models never renders the map, whose untouched zeroes compare as shadowed.
  let cs = _rzShadowVP.casters;
  if (cs.w == 0.0) { return 1.0; }
  if (cs.w > 0.0) {
    let toCaster = cs.xyz - p;
    let along = dot(toCaster, toSun);
    let perp = length(toCaster - toSun * along);
    if (along <= -cs.w || perp > cs.w) { return 1.0; }
  }
  let q = p + toSun * 0.08;
  let c0 = _rzShadowVP.viewProj[0] * vec4f(q, 1.0);
  let n0 = c0.xyz / max(c0.w, 1e-6);
  if (all(abs(n0.xy) < vec2f(0.98)) && n0.z > 0.0 && n0.z < 1.0) {
    return mix(1.0, _rzShadowNearTaps(n0), castAmt);
  }
  let c1 = _rzShadowVP.viewProj[${n - 1}] * vec4f(q, 1.0);
  let n1 = c1.xyz / max(c1.w, 1e-6);
  if (all(abs(n1.xy) < vec2f(0.98)) && n1.z > 0.0 && n1.z < 1.0) {
    return mix(1.0, _rzShadowFarTaps(n1), castAmt);
  }
  return 1.0;
}
${worldAmbientWgsl("_rzLight")}`
}
