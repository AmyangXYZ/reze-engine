import { SCENE_TAP_STUB } from "./scene-tap"
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
// So there are no per-light shadows, no area lights and no clustering. The loop
// runs over the lights a scene HAS, not over the cap, so a scene with four pays
// for four; the cap only bounds the buffer and the worst case.
//
// LAYOUT. A 4-float header (count, then padding that keeps the records
// vec4-aligned), then MAX_LIGHTS records of 16 floats — four vec4s:
//
//   [0..2] position, world space               [3] radius
//   [4..6] colour PREMULTIPLIED by intensity   [7] type
//   [8..10] aim, unit, pointing away from the light   [11] cos of the outer angle
//   [12] cos of the inner angle                [13..15] spare
//
// Colour carries intensity because nothing reads them apart: every use is the
// product, and storing two numbers that are only ever multiplied is two numbers
// that can disagree.
//
// A POINT LIGHT IS A SPOT WITH NO CONE, and that is how the loop stays
// branchless: it stores aim (0,0,0) and both cosines at -1, which makes the
// cone term below saturate to exactly 1. `type` says which a light is for
// anyone reading the buffer; the shading never asks.

import { castDistanceStub } from "./passes/cast-distance"
import { audioApi } from "./audio-api"
import { midiApi } from "./midi-api"
import { lyricsApi } from "./lyrics-api"
import { clockApi, trailSlotsApi, viewportApi } from "./passes/hosted-api"
import { idApi } from "./id-api"

/** Floats before the first record. One is the count; the rest keep the records
 *  vec4-aligned, which is what lets a future pass read them as vec4s. */
export const LIGHT_HEADER = 4
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
 * What it does NOT buy: a scene that genuinely lights a fragment from a hundred
 * lamps pays for a hundred iterations of the loop. The distance test below
 * rejects most of them in a few instructions, which is what makes a rig of this
 * size affordable at all; clustering is what replaces the linear walk when a
 * scene wants every one of them close enough to matter.
 */
export const MAX_LIGHTS = 128
/** Floats in the whole buffer. */
export const LIGHTS_FLOATS = LIGHT_HEADER + MAX_LIGHTS * LIGHT_STRIDE

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
}
`

/**
 * The world's light at a surface facing n — the flat colour, or the installed
 * HDRI's irradiance (sh[0].w = 1), evaluated from folded SH coefficients (see
 * ibl.ts for the folding; the shader is a plain polynomial in the normal).
 *
 * One string included by every module that declares LightUniforms with the sh
 * block — the hosted-api lesson: a helper defined in some modules and not
 * others is a compile error waiting for the first file that crosses them.
 */
export const WORLD_AMBIENT_WGSL = /* wgsl */ `
fn rzWorldAmbient(n: vec3f) -> vec3f {
  if (light.sh[0].w < 0.5) { return light.ambientColor.xyz; }
  let x = n.x;
  let y = n.y;
  let z = n.z;
  let c = light.sh[0].xyz
    + light.sh[1].xyz * y + light.sh[2].xyz * z + light.sh[3].xyz * x
    + light.sh[4].xyz * (x * y) + light.sh[5].xyz * (y * z)
    + light.sh[6].xyz * (3.0 * z * z - 1.0) + light.sh[7].xyz * (x * z)
    + light.sh[8].xyz * (x * x - y * y);
  return max(c, vec3f(0.0));
}
`

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
// (time, base slot, count, weight) — see buildLightEmitShader on why the base
// is here and not in the text.
@group(0) @binding(1) var<uniform> _rzLightU: vec4f;
// The camera block and the cast — the two buffers the scene API reads. Same
// contents the field and grid modules bind, so an effect's lightEmit sees the
// scene exactly as its drawing half does.
@group(0) @binding(2) var<uniform> viewU: array<vec4<f32>, 15>;
@group(0) @binding(3) var<storage, read> _rzCast: array<vec4f>;
${sceneApi}
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
${clockApi("_rzLightU.x", "0.0")}
// Canvas height — the same number the drawing modules read out of their camera
// struct, both written from canvas.height, so this name means ONE value in
// every module. Verified against the writers, not assumed: cameraMatrixData[35]
// and viewU[6].w have the same source.
${viewportApi("viewU[6].w")}
${trailSlotsApi(cast.trailCount)}
// The id accessors, stubbed: this module cannot read an attachment the
// scene pass writes. See id-api.ts — the author's whole file compiles here.
${idApi(false, 0, 0) + castDistanceStub()}
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
  if (i >= u32(_rzLightU.z)) { return; }
  // Time is a PARAMETER, not an rzTime() call: this same source compiles
  // inside the field, particle, trail and grid modules, and those already
  // define rzTime differently or not at all. A parameter needs nothing from
  // the module it lands in, which is why the field mounts take theirs too.
  let l = lightEmit(i, _rzLightU.x);
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
  let c = select(vec3f(0.0), max(l.color * l.intensity, vec3f(0.0)), finite) * _rzLightU.w;
  let b = ${LIGHT_HEADER}u + (u32(_rzLightU.y) + i) * ${LIGHT_STRIDE}u;
  _rzLightsOut[b] = select(0.0, l.pos.x, finite);
  _rzLightsOut[b + 1u] = select(0.0, l.pos.y, finite);
  _rzLightsOut[b + 2u] = select(0.0, l.pos.z, finite);
  // Radius goes to zero when the effect is OFF, and is left alone at every
  // weight above it. Scaling it with the fade would shrink a dimming lamp's
  // reach, which is a different thing than dimming it; zeroing it at nothing
  // is what lets a material's distance cull drop the slot entirely.
  _rzLightsOut[b + 3u] = select(0.0, max(l.radius, 0.0), finite && _rzLightU.w > 0.0);
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
  _rzLightsOut[b + 14u] = 0.0;
  _rzLightsOut[b + 15u] = 0.0;
}
`
}

/** The rz*Light accessors, with the buffer declared at the given binding. */
export function lightsApi(group: number, binding: number): string {
  return /* wgsl */ `
@group(${group}) @binding(${binding}) var<storage, read> _rzLights: array<f32>;

const RZ_MAX_LIGHTS: u32 = ${MAX_LIGHTS}u;

/** How many positional lights the scene has. Zero is the ordinary case. */
fn rzLightCount() -> u32 { return min(u32(_rzLights[0]), RZ_MAX_LIGHTS); }

/** Light i's world position. */
fn rzLightPos(i: u32) -> vec3f {
  let b = ${LIGHT_HEADER}u + i * ${LIGHT_STRIDE}u;
  return vec3f(_rzLights[b], _rzLights[b + 1u], _rzLights[b + 2u]);
}

/** How far light i reaches. Its falloff is zero AT this distance, not merely
 *  small, so the light has a bound a cull can be derived from later. */
fn rzLightRadius(i: u32) -> f32 { return _rzLights[${LIGHT_HEADER}u + i * ${LIGHT_STRIDE}u + 3u]; }

/** Light i's colour, already multiplied by its intensity. */
fn rzLightColor(i: u32) -> vec3f {
  let b = ${LIGHT_HEADER}u + i * ${LIGHT_STRIDE}u + 4u;
  return vec3f(_rzLights[b], _rzLights[b + 1u], _rzLights[b + 2u]);
}

/** Where light i points, away from itself. The zero vector for a point light. */
fn rzLightAim(i: u32) -> vec3f {
  let b = ${LIGHT_HEADER}u + i * ${LIGHT_STRIDE}u + 8u;
  return vec3f(_rzLights[b], _rzLights[b + 1u], _rzLights[b + 2u]);
}

/** The cosines a spot fades between: x its outer edge, y its inner one. A point
 *  light stores (-1, -1), which saturates the cone term to 1. */
fn rzLightCone(i: u32) -> vec2f {
  let b = ${LIGHT_HEADER}u + i * ${LIGHT_STRIDE}u + 11u;
  return vec2f(_rzLights[b], _rzLights[b + 1u]);
}

/**
 * Every positional light's contribution at a surface point, as light — not as a
 * finished colour. Multiply by whatever the surface's albedo is.
 *
 * WITH NO LIGHTS THIS RETURNS EXACTLY ZERO and the loop never runs, so a scene
 * that declares none is arithmetically identical to one compiled before lights
 * existed. That is the property the whole feature is gated on: adding this to
 * every material must cost nothing until someone asks for a light.
 *
 * FALLOFF IS RELATIVE TO THE RADIUS, and deliberately not physical.
 *
 * The first version windowed a real inverse-square, and it was unusable: 1/d²
 * is measured in world units, an MMD character is about 18 of them tall, so a
 * lamp two metres off her shoulder divided by 37 and an intensity of 4 landed
 * as 0.06 — invisible. Radius and intensity were fighting, and intensity had no
 * scale a person could learn.
 *
 * So: intensity is the brightness AT the light, radius is where it reaches
 * zero, and the curve between them is the same shape whatever the scene's
 * scale. Both dials now mean what they say, which for a composer beats being
 * right about photons. (1 - t²)² — smooth at both ends, exactly 0 at the
 * radius, so the bound a cull could be derived from is still real.
 */
fn rzLightsDiffuse(p: vec3f, n: vec3f) -> vec3f {
  var acc = vec3f(0.0);
  let count = rzLightCount();
  for (var i = 0u; i < count; i = i + 1u) {
    let d = rzLightPos(i) - p;
    let dist = length(d);
    // Out of reach before anything else is computed. The falloff is already
    // exactly zero at the radius, so this changes no pixel — it is what keeps a
    // stage rig's far lamps off the bill at every fragment they do not light.
    if (dist >= rzLightRadius(i)) { continue; }
    let toLight = d / max(dist, 1e-4);
    // Facing the light, and nothing behind it. No wrap or half-lambert: this
    // layer adds light, and a wrapped term would lift the shadow side, which is
    // the ramp's business and not this one's.
    let ndl = max(dot(n, toLight), 0.0);
    if (ndl <= 0.0) { continue; }
    let t = clamp(dist / max(rzLightRadius(i), 1e-4), 0.0, 1.0);
    let falloff = 1.0 - t * t;
    // How far inside the cone this point sits: 1 within the inner angle, 0 past
    // the outer one, squared for the same soft edge the falloff has. A point
    // light's (-1, -1) divides by the floor and clamps to 1, so it pays one
    // dot product and no branch.
    let cone = rzLightCone(i);
    let aim = clamp((dot(-toLight, rzLightAim(i)) - cone.x) / max(cone.y - cone.x, 1e-4), 0.0, 1.0);
    acc = acc + rzLightColor(i) * (ndl * falloff * falloff * aim * aim);
  }
  return acc;
}
`
}
