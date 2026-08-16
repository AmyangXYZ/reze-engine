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
// So there are no per-light shadows, no area lights and no clustering. At this
// count a flat loop in the fragment shader is cheaper than anything that would
// avoid it, and the cap is what keeps that true.
//
// LAYOUT. A 4-float header (count, then padding that keeps the records
// vec4-aligned), then MAX_LIGHTS records of 8 floats:
//
//   [0..2] position, world space      [3] radius
//   [4..6] colour PREMULTIPLIED by intensity   [7] type
//
// Colour carries intensity because nothing reads them apart: every use is the
// product, and storing two numbers that are only ever multiplied is two numbers
// that can disagree. `type` is reserved — every light is a point light today
// and the loop does not branch on it, so it is honest padding rather than a
// switch with one case.

/** Floats before the first record. One is the count; the rest keep the records
 *  vec4-aligned, which is what lets a future pass read them as vec4s. */
export const LIGHT_HEADER = 4
/** Floats per light — see the layout above. */
export const LIGHT_STRIDE = 8
/**
 * The cap, and it is a real one: the loop below runs per fragment, so this is
 * the number that decides whether lights are free or a cost. Sixteen is the
 * bounded middle tier the design settled on — enough for a stage rig, far below
 * the point where clustering would start to pay for itself.
 */
export const MAX_LIGHTS = 16
/** Floats in the whole buffer. */
export const LIGHTS_FLOATS = LIGHT_HEADER + MAX_LIGHTS * LIGHT_STRIDE

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

/**
 * Every positional light's contribution at a surface point, as light — not as a
 * finished colour. Multiply by whatever the surface's albedo is.
 *
 * WITH NO LIGHTS THIS RETURNS EXACTLY ZERO and the loop never runs, so a scene
 * that declares none is arithmetically identical to one compiled before lights
 * existed. That is the property the whole feature is gated on: adding this to
 * every material must cost nothing until someone asks for a light.
 *
 * Falloff is inverse-square WINDOWED to the radius. Pure inverse-square never
 * reaches zero, so every light would touch every fragment in the scene and the
 * cap would be the only thing bounding the cost; the window makes the radius
 * mean what it says.
 */
fn rzLightsDiffuse(p: vec3f, n: vec3f) -> vec3f {
  var acc = vec3f(0.0);
  let count = rzLightCount();
  for (var i = 0u; i < count; i = i + 1u) {
    let d = rzLightPos(i) - p;
    let dist2 = dot(d, d);
    let dist = sqrt(dist2);
    // Facing the light, and nothing behind it. No wrap or half-lambert: this
    // layer adds light, and a wrapped term would lift the shadow side, which is
    // the ramp's business and not this one's.
    let ndl = max(dot(n, d / max(dist, 1e-4)), 0.0);
    if (ndl <= 0.0) { continue; }
    let window = clamp(1.0 - dist / max(rzLightRadius(i), 1e-4), 0.0, 1.0);
    // 1 + dist2 rather than dist2: an unbounded 1/r² is infinite at the source,
    // and a light sitting inside geometry would blow the frame out.
    acc = acc + rzLightColor(i) * (ndl * window * window / (1.0 + dist2));
  }
  return acc;
}
`
}
