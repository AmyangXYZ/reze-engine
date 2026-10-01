import { SHADOW_ATLAS_SIZE, SHADOW_CASCADES } from "../shadow-cascades"
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

/**
 * The world's light averaged over every direction: its colour and brightness,
 * with no shape. Every term but sh[0] averages to zero over the sphere (x, xy,
 * 3z²-1 ... all do), so the average is sh[0] itself; a flat world is its own.
 * What an anime cast takes from a sky - see ModelLight.opts.
 */
fn rzWorldAmbientAvg() -> vec3f {
  if (${u}.sh[0].w < 0.5) { return ${u}.ambientColor.xyz; }
  return max(${u}.sh[0].xyz, vec3f(0.0));
}
`

/**
 * The sun's shadow atlas, as every reader looks it up: `<p>Locate(q)` finds
 * world point q in the first cascade whose box holds it and answers
 * (atlas uv, depth, cascade) — cascade -1 outside them all — and `<p>Taps(a)`
 * is the 3×3 PCF at that place.
 *
 * ONE implementation for the materials, the ground and effects, so a blade of
 * grass and her own skin go dark at the same edge. `vp` names the cascades'
 * view-projection array in the module that includes it.
 *
 * The box test keeps a margin (SHADOW_MARGIN of the tile's half-width) so the
 * whole filter stays inside the tile: a tap that crossed into the neighbouring
 * tile would compare against another cascade's depths. The sampler is linear,
 * so each tap is itself a 2×2 compare. Unrolled — Safari's Metal backend
 * doesn't unroll nested shadow loops reliably.
 */
export const SHADOW_MARGIN = 0.98

export const sunShadowWgsl = (p: string, map: string, sampler: string, vp: string) => {
  const n = SHADOW_CASCADES.length
  const tiles = SHADOW_CASCADES.map((c) => `vec2f(${c.origin[0] / SHADOW_ATLAS_SIZE}, ${c.origin[1] / SHADOW_ATLAS_SIZE})`).join(", ")
  const scale = SHADOW_CASCADES[0].mapSize / SHADOW_ATLAS_SIZE
  return /* wgsl */ `
const ${p}Tiles = array<vec2f, ${n}>(${tiles});

fn ${p}Locate(q: vec3f) -> vec4f {
  for (var i = 0u; i < ${n}u; i++) {
    let c = ${vp}[i] * vec4f(q, 1.0);
    let ndc = c.xyz / max(c.w, 1e-6);
    if (all(abs(ndc.xy) < vec2f(${SHADOW_MARGIN})) && ndc.z > 0.0 && ndc.z < 1.0) {
      let uv = ${p}Tiles[i] + vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * ${scale};
      return vec4f(uv, ndc.z, f32(i));
    }
  }
  return vec4f(0.0, 0.0, 0.0, -1.0);
}

fn ${p}Taps(a: vec4f) -> f32 {
  let cmpZ = a.z - 0.001;
  let ts = ${1 / SHADOW_ATLAS_SIZE};
  let s00 = textureSampleCompareLevel(${map}, ${sampler}, a.xy + vec2f(-ts, -ts), cmpZ);
  let s10 = textureSampleCompareLevel(${map}, ${sampler}, a.xy + vec2f(0.0, -ts), cmpZ);
  let s20 = textureSampleCompareLevel(${map}, ${sampler}, a.xy + vec2f( ts, -ts), cmpZ);
  let s01 = textureSampleCompareLevel(${map}, ${sampler}, a.xy + vec2f(-ts, 0.0), cmpZ);
  let s11 = textureSampleCompareLevel(${map}, ${sampler}, a.xy, cmpZ);
  let s21 = textureSampleCompareLevel(${map}, ${sampler}, a.xy + vec2f( ts, 0.0), cmpZ);
  let s02 = textureSampleCompareLevel(${map}, ${sampler}, a.xy + vec2f(-ts,  ts), cmpZ);
  let s12 = textureSampleCompareLevel(${map}, ${sampler}, a.xy + vec2f(0.0,  ts), cmpZ);
  let s22 = textureSampleCompareLevel(${map}, ${sampler}, a.xy + vec2f( ts,  ts), cmpZ);
  return (s00 + s10 + s20 + s01 + s11 + s21 + s02 + s12 + s22) * (1.0 / 9.0);
}
`
}

/** Bindings the real accessors take, from `binding` upward. */
export const SCENE_LIGHT_API_BINDINGS = 4

/**
 * `rzShadow(p)`: how much of the sun reaches world point `p` — 1 lit, 0 in a
 * caster's shadow, soft in between.
 *
 * `rzWorldAmbient(n)`: the world's light arriving at a surface facing `n` —
 * the materials' own ambient, so an effect's shade can match hers.
 *
 * With `on`, declares four bindings from `binding`:
 *   +0 the light uniform (ambient, the sun — direction, and in .w the shadow
 *      switch — and the HDRI's irradiance)
 *   +1 the cascade view-projections, then the caster sphere
 *   +2 the sun's shadow atlas   +3 the comparison sampler
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
@group(${group}) @binding(${binding + 2}) var _rzShadowAtlas: texture_depth_2d;
@group(${group}) @binding(${binding + 3}) var _rzShadowCmp: sampler_comparison;
${sunShadowWgsl("_rzSun", "_rzShadowAtlas", "_rzShadowCmp", "_rzShadowVP.viewProj")}

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
  let a = _rzSunLocate(p + toSun * 0.08);
  // Outside every cascade there is no occlusion information: lit.
  if (a.w < 0.0) { return 1.0; }
  return mix(1.0, _rzSunTaps(a), castAmt);
}
${worldAmbientWgsl("_rzLight")}`
}
