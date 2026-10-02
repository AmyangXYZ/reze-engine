import { SCENE_TAP_STUB } from "./scene-tap"
import { subjectMaskApi } from "./cast-api"
// Positional lights, as data — the sibling of the cast, audio and score
// interfaces, and shaped like them: one shared buffer, read through accessors,
// never touched directly.
//
// WHAT THIS IS NOT. The sun is still the ONE key light and it still owns the
// toon ramp. These are an ADDITIVE layer on top of whatever the material's
// graph decided, and they deliberately do not re-ramp: two ramped terminators
// crossing a cheek read as plastic, which is the failure every stylised
// renderer that bolted a second key light onto a toon shader has shipped. A
// light here brightens; it does not restate the shading.
//
// So there are no per-light shadows and no area lights. A fragment walks the
// DOCUMENT's lamps through a world-space grid (see light-grid.ts): its cell
// names the lamps that can reach it, so a stage rig of fifty costs each pixel
// the handful standing near it. The lamps an EFFECT emits are placed by a
// compute pass every frame, which the grid never sees, and are walked in full.
//
// LAYOUT, in 32-bit words, read by the shader as vec4u. A 20-word header:
//
//   [0] light count (f32)       [1] how many are the document's (f32)
//   [4..6] grid origin (f32)    [7] 1 / cell edge (f32)
//   [8..10] grid dims (u32)     [12..19] the OUTSIDE mask (u32 bits)
//
// then MAX_LIGHTS records of 16 floats — four vec4s:
//
//   [0..2] position, world space               [3] radius
//   [4..6] colour PREMULTIPLIED by intensity   [7] type
//   [8..10] aim, unit, pointing away from the light   [11] cos of the outer angle
//   [12] cos of the inner angle
//   [13] the rendering layers it does NOT reach (u32 bits)
//   [14] its bulb: the distance inside which the inverse square is held flat
//        (0 = RZ_LAMP_NEAR)                    [15] its cookie: layer + 1, 0 none
//
// then the grid: LIGHT_MASK_WORDS words of lamp bits per cell.
//
// Colour carries intensity because nothing reads them apart: every use is the
// product, and storing two numbers that are only ever multiplied is two numbers
// that can disagree.
//
// A POINT LIGHT IS A SPOT WITH NO CONE, and that is how the loop stays
// branchless: it stores aim (0,0,0) and both cosines at -1, which makes the
// cone term below saturate to exactly 1. `type` says which a light is for
// anyone reading the buffer; the shading never asks.
//
// LAYERS, as the game's pipeline (URP's rendering layers) gates its lights: a
// draw carries layer bits, and a light reaches it only where its own mask
// shares one. Stored INVERTED — the layers a light does not reach — so the
// zero every writer that predates layers leaves in [13] reaches everything,
// which is what an effect's lamp and a plain document light mean.

import { castDistanceStub } from "./passes/cast-distance"
import { audioApi } from "./audio-api"
import { midiApi } from "./midi-api"
import { lyricsApi } from "./lyrics-api"
import { clockApi, trailSlotsApi, viewportApi } from "./passes/hosted-api"
import { idApi } from "./id-api"
import { sceneLightApi, worldAmbientWgsl } from "./scene-light-api"
import { pointsApi } from "./points-api"
import { textureApi } from "./texture-api"

/** Words before the first record: the counts, the grid's placement and the
 *  outside mask, five vec4s — see the layout above. */
export const LIGHT_HEADER = 20
/** Floats per light — see the layout above. */
export const LIGHT_STRIDE = 16
/**
 * The cap. The loop below runs per fragment over the lights a scene HAS, so
 * this bounds the worst case rather than the ordinary one, and the buffer is
 * storage rather than uniform — 128 records is 8 KiB, which costs nothing to
 * declare and is paid for only by the scenes that fill it.
 *
 * Sized against the work, not against a round number: a Unity stage arrives
 * with a lamp per fixture. One rip of a resort brought thirty-three, a lit
 * interior brings a hundred or more, and the ceilings of sixteen and then
 * forty-eight each cut a real scene in half. Past this the extras are dropped.
 *
 * A fragment pays for the document lamps its grid cell names, so the cost
 * follows how many reach a place rather than how many the scene holds. 256,
 * two vec4u of bits per cell, is URP's own Forward+ budget for visible lights
 * — and holds a lit interior's whole rig (X317's 153).
 */
export const MAX_LIGHTS = 256
/** Words of lamp bits per grid cell — one bit per lamp the cap allows. */
export const LIGHT_MASK_WORDS = MAX_LIGHTS / 32
/** The grid's cell budget. 32k cells over a stage a couple of hundred units
 *  across is a cell of about four — finer than the lamps it separates. */
export const LIGHT_GRID_CELLS = 32768
/** Where the grid starts, in words. */
export const LIGHT_GRID_BASE = LIGHT_HEADER + MAX_LIGHTS * LIGHT_STRIDE
/** Words in the whole buffer: header, records, grid. */
export const LIGHTS_FLOATS = LIGHT_GRID_BASE + LIGHT_GRID_CELLS * LIGHT_MASK_WORDS

/**
 * The RzLight struct, declared in EVERY module a user's source is spliced into.
 *
 * One effect file goes into every module it has a mount in, so a foreground
 * effect that also emits lights compiles its lightEmit inside the FIELD shader
 * too — where nothing calls it, but it still has to resolve. Leaving the struct
 * out of those modules is a compile error on a function the author was right to
 * write, which is the same trap the grid's step-only half documents.
 */
export const RZ_LIGHT_STRUCT_WGSL = /* wgsl */ `
/** What an effect returns for one of its lights. */
struct RzLight {
  pos: vec3f,
  color: vec3f,
  intensity: f32,
  radius: f32,
  /** Its bulb: the distance inside which it stops getting brighter. 0 is the
   *  engine's default; a glow worn on a hand wants one the size of the glow. */
  near: f32,
}
`

/** rzWorldAmbient against the materials' and the ground's `light` uniform. */
export const WORLD_AMBIENT_WGSL = worldAmbientWgsl("light")

/** Does this source define the emit mount? */
export function hasLightEmit(wgsl: string): boolean {
  return /\bfn\s+lightEmit\s*\(/.test(wgsl)
}

/**
 * The compute module that runs an effect's lightEmit once per light per frame.
 *
 *     fn lightEmit(i: u32, time: f32) -> RzLight
 *
 * A COMPUTE stage rather than a CPU callback, and that is the whole point:
 * Fireworks knows where its bursts are as a closed form in WGSL, and mirroring
 * that on the CPU to place a light would be two derivations of one trajectory
 * that drift apart. Emitting in the shader means the light is wherever the
 * effect says it is, on the scene clock — which is also what makes it survive
 * an offline export frame-stepped at a different rate.
 *
 * The author writes local index 0..n-1 and never learns the global one, the
 * same aliasing the anchor table uses — so installing another effect ahead of
 * this one moves its lights without touching its source.
 *
 * The base arrives in the UNIFORM rather than baked into the text. Baking it
 * would mean recompiling every emitting effect the moment a scene gained or
 * lost a document light, because that is what shifts the slots underneath
 * them — a shader rebuild triggered by moving a lamp.
 *
 * The scene API arrives as a STRING rather than being imported, so this module
 * depends on nothing that depends on it. That API is what lets a lamp aim at
 * someone: Stage Lights points its beams at rzSubject().root, and a light that
 * did not know where she was could only sit where the fixture hangs. It also
 * brings RzLight, which is why this builder does not declare it again.
 */
export function buildLightEmitShader(
  wgsl: string,
  sceneApi: string,
  cast: { trailCount: number },
  paramsDecl = "",
): string {
  return /* wgsl */ `
// read_write HERE and read-only in the material shaders. Different passes, so
// the two never coexist: this compute runs before the scene pass that reads it.
@group(0) @binding(0) var<storage, read_write> _rzLightsOut: array<f32>;
// [0] (time, base slot, count, weight) — see buildLightEmitShader on why the
// base is here and not in the text — and [1].x the subject mask, which is the
// only thing the second vec4 is for: an emitter reads the cast through the same
// accessors its drawing half does, so a rig aimed at one dancer must see one.
@group(0) @binding(1) var<uniform> _rzLightU: array<vec4f, 2>;
// The camera block and the cast — the two buffers the scene API reads. Same
// contents the field and grid modules bind, so an effect's lightEmit sees the
// scene exactly as its drawing half does.
@group(0) @binding(2) var<uniform> viewU: array<vec4<f32>, 15>;
@group(0) @binding(3) var<storage, read> _rzCast: array<vec4f>;
${sceneApi}
${subjectMaskApi("u32(_rzLightU[1].x)")}
${SCENE_TAP_STUB}
// Audio and score at 4 and 5, the same bindings the particle and trail modules
// put them on. A lamp that pulses on the beat or lights on a note is the whole
// point of a light an effect owns rather than one the document places, so this
// is not compile-safety padding — it is the mount's reason to exist.
${audioApi(0, 4)}
${midiApi(0, 5)}
${lyricsApi(0, 6)}
// The rest of the hosted API. Every one of these is here because the AUTHOR'S
// WHOLE FILE lands below, not because lightEmit needs it: a trail effect that
// grows a lamp at its tip compiles its ribbon code in this module too. The math
// helpers and the Particle struct are NOT repeated — they arrive with the scene
// API above, and a second copy is a redefinition error in engine code.
${clockApi("_rzLightU[0].x", "0.0")}
// Canvas height — the same number the drawing modules read out of their camera
// struct, both written from canvas.height, so this name means ONE value in
// every module. Verified against the writers, not assumed: cameraMatrixData[35]
// and viewU[6].w have the same source.
${viewportApi("viewU[6].w")}
${trailSlotsApi(cast.trailCount)}
// The id accessors, stubbed: this module cannot read an attachment the
// scene pass writes. See id-api.ts — the author's whole file compiles here.
${idApi(false, 0, 0) + castDistanceStub() + sceneLightApi(false, 0, 0) + pointsApi(false) + textureApi(0)}
// The dials the author declared, if any. A lamp is exactly the thing someone
// retunes — its colour and its reach — so an emitter reads params like every
// other mount rather than being the one place a #param resolves to nothing.
${paramsDecl}
${wgsl}

@compute @workgroup_size(64)
fn lightEmitMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  // The dispatch is sized to the count, but a workgroup is 64 wide and the
  // count rarely is — the tail must not write into the next effect's slots.
  if (i >= u32(_rzLightU[0].z)) { return; }
  // Time is a PARAMETER, not an rzTime() call: this same source compiles
  // inside the field, particle, trail and grid modules, and those already
  // define rzTime differently or not at all. A parameter needs nothing from
  // the module it lands in, which is why the field mounts take theirs too.
  let l = lightEmit(i, _rzLightU[0].x);
  // SANITIZED at the one write site, because lightEmit is HOSTED USER CODE and
  // this buffer feeds every fragment of every material: one NaN position would
  // poison the whole frame, and WGSL leaves max(NaN, 0) indeterminate, so it
  // would not even fail the same way on every GPU. A light that fails the
  // check writes zeros — radius 0 is off. Colour is clamped at zero on top:
  // this layer is ADDITIVE, and a negative channel would darken what it lands
  // on and can push HDR negative into bloom.
  let finite = l.pos.x == l.pos.x && l.pos.y == l.pos.y && l.pos.z == l.pos.z &&
    l.radius == l.radius && l.intensity == l.intensity &&
    l.color.x == l.color.x && l.color.y == l.color.y && l.color.z == l.color.z;
  // Weight rides along with the sanitisation, which is already the one place
  // this buffer is written: a lamp at half weight is half as bright, and one
  // at zero never reaches here because the dispatch is skipped.
  let c = select(vec3f(0.0), max(l.color * l.intensity, vec3f(0.0)), finite) * _rzLightU[0].w;
  let b = ${LIGHT_HEADER}u + (u32(_rzLightU[0].y) + i) * ${LIGHT_STRIDE}u;
  _rzLightsOut[b] = select(0.0, l.pos.x, finite);
  _rzLightsOut[b + 1u] = select(0.0, l.pos.y, finite);
  _rzLightsOut[b + 2u] = select(0.0, l.pos.z, finite);
  // Radius goes to zero when the effect is OFF, and is left alone at every
  // weight above it. Scaling it with the fade would shrink a dimming lamp's
  // reach, which is a different thing than dimming it; zeroing it at nothing
  // is what lets a material's distance cull drop the slot entirely.
  _rzLightsOut[b + 3u] = select(0.0, max(l.radius, 0.0), finite && _rzLightU[0].w > 0.0);
  // Colour carries intensity, exactly as the CPU writer stores it — one product,
  // one place, so the two producers cannot disagree about what a slot means.
  _rzLightsOut[b + 4u] = c.x;
  _rzLightsOut[b + 5u] = c.y;
  _rzLightsOut[b + 6u] = c.z;
  // An effect emits POINT lights: RzLight carries no aim, and widening it would
  // fail to compile every emitter already written against it. The rest of the
  // record is written anyway — these slots are reused frame to frame, and a spot
  // the document placed here last frame would otherwise keep its cone.
  _rzLightsOut[b + 7u] = 0.0;
  _rzLightsOut[b + 8u] = 0.0;
  _rzLightsOut[b + 9u] = 0.0;
  _rzLightsOut[b + 10u] = 0.0;
  _rzLightsOut[b + 11u] = -1.0;
  _rzLightsOut[b + 12u] = -1.0;
  _rzLightsOut[b + 13u] = 0.0;
  _rzLightsOut[b + 14u] = select(0.0, max(l.near, 0.0), finite && l.near == l.near);
  _rzLightsOut[b + 15u] = 0.0;
}
`
}

/** The rz*Light accessors, with the buffer declared at the given binding.
 *  `layers` is the WGSL expression for the drawing's rendering-layer bits. */
export function lightsApi(
  group: number,
  binding: number,
  layers: string,
  cookies: { binding: number; sampler: string },
): string {
  const R = LIGHT_HEADER / 4
  const S = LIGHT_STRIDE / 4
  return /* wgsl */ `
// vec4u, so a record is four loads and a cell's lamp bits are one. Floats come
// out through bitcast; the bits must NOT pass through f32 on the way, where a
// mask that happens to spell a NaN is not guaranteed to survive a load.
@group(${group}) @binding(${binding}) var<storage, read> _rzLights: array<vec4u>;
// The spot lamps' cookies, one picture a layer (Engine.loadLightCookie).
@group(${group}) @binding(${cookies.binding}) var _rzCookies: texture_2d_array<f32>;

const RZ_MAX_LIGHTS: u32 = ${MAX_LIGHTS}u;
// The DEFAULT bulb, in world units, for a record that names none (word 14):
// the inverse square is held flat inside it, so the spike beside the lamp is
// finite. URP's own floor — distance² at least HALF_MIN, 6.1e-5 m² — which at
// MMD scale (8 units a metre) is a bulb of 0.0625. A lamp with a real size (a
// game rig's shapeRadius) says so in word 14.
const RZ_LAMP_NEAR: f32 = 0.0625;

/** How many positional lights the scene has. Zero is the ordinary case. */
fn rzLightCount() -> u32 { return min(u32(bitcast<f32>(_rzLights[0].x)), RZ_MAX_LIGHTS); }

/** How many of them the document placed — the ones the grid indexes. */
fn _rzLightDocCount() -> u32 { return min(u32(bitcast<f32>(_rzLights[0].y)), rzLightCount()); }

/** One vec4 of light i's record. */
fn _rzLightVec(i: u32, k: u32) -> vec4f { return bitcast<vec4f>(_rzLights[${R}u + i * ${S}u + k]); }

/** Light i's world position. */
fn rzLightPos(i: u32) -> vec3f { return _rzLightVec(i, 0u).xyz; }

/** How far light i reaches. Its falloff is zero AT this distance, not merely
 *  small, which is the bound the grid is built from. */
fn rzLightRadius(i: u32) -> f32 { return _rzLightVec(i, 0u).w; }

/** Light i's colour, already multiplied by its intensity. */
fn rzLightColor(i: u32) -> vec3f { return _rzLightVec(i, 1u).xyz; }

/** Where light i points, away from itself. The zero vector for a point light. */
fn rzLightAim(i: u32) -> vec3f { return _rzLightVec(i, 2u).xyz; }

/** A spot's cookie, as URP's: its picture projected along the cone, so the
 *  light a point takes is the picture's colour there. The picture's square
 *  spans the cone's outer angle, up on the picture toward world up (or +z for
 *  a lamp aimed straight up or down). 1 for a lamp without one, and for a
 *  point lamp, whose cookie would need a cube. */
fn rzLightCookie(i: u32, toLight: vec3f) -> vec3f {
  let k = _rzLightVec(i, 3u).w;
  let a = rzLightAim(i);
  if (k < 0.5 || dot(a, a) < 0.5) { return vec3f(1.0); }
  let up0 = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(a.y) > 0.99);
  let right = normalize(cross(up0, a));
  let up = cross(a, right);
  let d = -toLight;
  let c = rzLightCone(i).x;
  let spread = sqrt(max(1.0 - c * c, 0.0)) / max(c, 1e-4);
  let z = max(dot(d, a), 1e-4);
  let uv = vec2f(dot(d, right), -dot(d, up)) / (z * spread) * 0.5 + 0.5;
  return textureSampleLevel(_rzCookies, ${cookies.sampler}, uv, i32(k) - 1, 0.0).rgb;
}

/** Light i's bulb: inside this distance the inverse square is held flat. A
 *  record that names none (0, every writer that predates the word) takes
 *  RZ_LAMP_NEAR. A game lamp's is its shapeRadius: Unity caps 1/d² at
 *  1/shapeRadius, so the flat core is sqrt(shapeRadius) - and a character rig
 *  tuned with a bulb wider than its reach is a lamp of constant strength inside
 *  a soft sphere, which a single global bulb cannot spell. */
fn rzLightNear(i: u32) -> f32 { let r = _rzLightVec(i, 3u).z; return select(RZ_LAMP_NEAR, r, r > 0.0); }

/** The cosines a spot fades between: x its outer edge, y its inner one. A point
 *  light stores (-1, -1), which saturates the cone term to 1. */
fn rzLightCone(i: u32) -> vec2f { return vec2f(_rzLightVec(i, 2u).w, _rzLightVec(i, 3u).x); }

/** Does light i reach this drawing's rendering layers? Read as bits, never as
 *  a float — see the layout. */
fn _rzLightReaches(i: u32) -> bool { return ((${layers}) & ~_rzLights[${R}u + i * ${S}u + 3u].y) != 0u; }

/** The document lamps that can reach p — 256 bits, lamps 0..127 in .lo and
 *  128..255 in .hi: its grid cell's, or the outside mask beyond the grid.
 *  Written so a NaN position fails the inside test. */
struct _RzLampMask { lo: vec4u, hi: vec4u };
fn _rzLightCellMask(p: vec3f) -> _RzLampMask {
  let g = bitcast<vec4f>(_rzLights[1]);
  let dims = _rzLights[2].xyz;
  let c = floor((p - g.xyz) * g.w);
  if (!(all(c >= vec3f(0.0)) && all(c < vec3f(dims)))) { return _RzLampMask(_rzLights[3], _rzLights[4]); }
  let ci = vec3u(c);
  let at = ${LIGHT_GRID_BASE / 4}u + ((ci.z * dims.y + ci.y) * dims.x + ci.x) * 2u;
  return _RzLampMask(_rzLights[at], _rzLights[at + 1u]);
}

/**
 * One light's contribution at a surface point.
 *
 * A LIGHT FALLS OFF AS THE INVERSE SQUARE, the curve Unity, Unreal, Blender and
 * glTF all light with: intensity / max(d², bulb²) (rzLightNear), so its intensity is
 * the brightness one unit away, windowed by (1 − (d/R)⁴)² so it is exactly zero
 * at its radius and the bound the grid is built from is real.
 *
 * LIGHT UNITS ARE UNITY'S, for every light in this engine — the sun, the
 * directional slots and these lamps: a surface returns albedo × light × N·L
 * with no 1/π (URP folds it into its intensities), and a highlight is the
 * lobe as URP writes it. So a white lamp of intensity 1 one unit away lights
 * albedo 1:1 head on, as a Unity point light does. A Blender light carries
 * divided by π (the app's GLB loader does it).
 */
fn _rzLightOne(i: u32, p: vec3f, n: vec3f) -> vec3f {
  if (!_rzLightReaches(i)) { return vec3f(0.0); }
  let pr = _rzLightVec(i, 0u);
  let d = pr.xyz - p;
  let dist = length(d);
  // Out of reach before anything else is computed. The grid is conservative —
  // a cell's bit means the lamp CAN reach part of it — so this still runs.
  if (dist >= pr.w) { return vec3f(0.0); }
  let toLight = d / max(dist, 1e-4);
  // Facing the light, and nothing behind it. No wrap or half-lambert: this
  // layer adds light, and a wrapped term would lift the shadow side, which is
  // the ramp's business and not this one's.
  let ndl = max(dot(n, toLight), 0.0);
  if (ndl <= 0.0) { return vec3f(0.0); }
  let t = clamp(dist / max(pr.w, 1e-4), 0.0, 1.0);
  let t2 = t * t;
  let window = 1.0 - t2 * t2;
  let bulb = rzLightNear(i);
  let falloff = window * window / max(dist * dist, bulb * bulb);
  // How far inside the cone this point sits: 1 within the inner angle, 0 past
  // the outer one, squared for the same soft edge the falloff has. A point
  // light's (-1, -1) divides by the floor and clamps to 1, so it pays one
  // dot product and no branch.
  let cone = rzLightCone(i);
  let aim = clamp((dot(-toLight, rzLightAim(i)) - cone.x) / max(cone.y - cone.x, 1e-4), 0.0, 1.0);
  return rzLightColor(i) * rzLightCookie(i, toLight) * (ndl * falloff * aim * aim);
}

/** The lamps named by one word of a cell's bits, lowest first. */
fn _rzLightWord(bits0: u32, base: u32, p: vec3f, n: vec3f) -> vec3f {
  var acc = vec3f(0.0);
  var bits = bits0;
  loop {
    if (bits == 0u) { break; }
    let i = base + firstTrailingBit(bits);
    bits = bits & (bits - 1u);
    acc = acc + _rzLightOne(i, p, n);
  }
  return acc;
}

/**
 * Every positional light's contribution at a surface point, as light — not as a
 * finished colour. Multiply by whatever the surface's albedo is.
 *
 * WITH NO LIGHTS THIS RETURNS EXACTLY ZERO and neither walk runs, so a scene
 * that declares none is arithmetically identical to one compiled before lights
 * existed. That is the property the whole feature is gated on: adding this to
 * every material must cost nothing until someone asks for a light.
 */
fn rzLightsDiffuse(p: vec3f, n: vec3f) -> vec3f {
  return _rzLightsIrradiance(p, n);
}

// ONE WALK PER FRAGMENT. A principled closure walks the lamps for its specular
// anyway, and every lamp's reach, falloff and cone are the same numbers the
// diffuse layer needs — so that walk computes both, against the normal the
// material prelude recorded here, and leaves the diffuse for the epilogue.
// Walking the grid twice was ~60% of a lit stage's frame.
var<private> _rzLampN: vec3f;
var<private> _rzLampDiffuse: vec3f;
var<private> _rzLampDiffuseSet: bool = false;
// Set by a closure that shaded the lamps' diffuse itself (a Unity-mode
// principled, which tints it by its own base) — the epilogue then adds none.
var<private> _rzLampDiffuseTaken: bool = false;

/** rzLightsDiffuse(p, _rzLampN), taken from a principled walk when one ran. */
fn rzLightsDiffuseOnce(p: vec3f, n: vec3f) -> vec3f {
  if (_rzLampDiffuseTaken) { return vec3f(0.0); }
  if (_rzLampDiffuseSet) { return _rzLampDiffuse; }
  return rzLightsDiffuse(p, n);
}

fn _rzLightsIrradiance(p: vec3f, n: vec3f) -> vec3f {
  var acc = vec3f(0.0);
  let count = rzLightCount();
  let docs = _rzLightDocCount();
  // The document's lamps: only those the grid says can reach this cell.
  if (docs > 0u) {
    let m = _rzLightCellMask(p);
    acc = acc + _rzLightWord(m.lo.x, 0u, p, n) + _rzLightWord(m.lo.y, 32u, p, n) +
      _rzLightWord(m.lo.z, 64u, p, n) + _rzLightWord(m.lo.w, 96u, p, n) +
      _rzLightWord(m.hi.x, 128u, p, n) + _rzLightWord(m.hi.y, 160u, p, n) +
      _rzLightWord(m.hi.z, 192u, p, n) + _rzLightWord(m.hi.w, 224u, p, n);
  }
  // The effects' lamps, which move every frame on the GPU: all of them.
  for (var i = docs; i < count; i = i + 1u) {
    acc = acc + _rzLightOne(i, p, n);
  }
  return acc;
}
`
}
