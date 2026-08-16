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

/**
 * `// @lights 3` — how many lights this effect emits.
 *
 * Declared, like every other mount: what the file says is what gets allocated,
 * so an effect that emits none costs no slots and nobody pays for a cap they
 * did not ask for. Clamped rather than rejected, the same choice `@particles`
 * makes — an author asking for a hundred gets the most the engine will give and
 * a scene that still runs.
 */
export function parseLightCount(wgsl: string, max: number): number {
  const m = /^\s*\/\/\s*@lights\s+(\d+)\s*$/m.exec(wgsl)
  if (!m) return 0
  return Math.max(1, Math.min(max, parseInt(m[1], 10)))
}

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
 * Deliberately NARROW for now: the emit stage gets its own clock and nothing
 * else. Fireworks — the case this mount exists for — is closed-form in time, so
 * that is enough to be useful, and every extra interface spliced in here is
 * another binding to get wrong. The cast, audio and score buffers are the
 * obvious next ones (a light on a wrist, a light on the beat) and each is a
 * binding plus its accessor module.
 */
export function buildLightEmitShader(wgsl: string): string {
  return /* wgsl */ `
// read_write HERE and read-only in the material shaders. Different passes, so
// the two never coexist: this compute runs before the scene pass that reads it.
@group(0) @binding(0) var<storage, read_write> _rzLightsOut: array<f32>;
// (time, base slot, count, _) — see buildLightEmitShader on why the base is
// here and not in the text.
@group(0) @binding(1) var<uniform> _rzLightU: vec4f;

${RZ_LIGHT_STRUCT_WGSL}
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
  let b = ${LIGHT_HEADER}u + (u32(_rzLightU.y) + i) * ${LIGHT_STRIDE}u;
  _rzLightsOut[b] = l.pos.x;
  _rzLightsOut[b + 1u] = l.pos.y;
  _rzLightsOut[b + 2u] = l.pos.z;
  _rzLightsOut[b + 3u] = l.radius;
  // Colour carries intensity, exactly as the CPU writer stores it — one product,
  // one place, so the two producers cannot disagree about what a slot means.
  _rzLightsOut[b + 4u] = l.color.x * l.intensity;
  _rzLightsOut[b + 5u] = l.color.y * l.intensity;
  _rzLightsOut[b + 6u] = l.color.z * l.intensity;
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
    // Facing the light, and nothing behind it. No wrap or half-lambert: this
    // layer adds light, and a wrapped term would lift the shadow side, which is
    // the ramp's business and not this one's.
    let ndl = max(dot(n, d / max(dist, 1e-4)), 0.0);
    if (ndl <= 0.0) { continue; }
    let t = clamp(dist / max(rzLightRadius(i), 1e-4), 0.0, 1.0);
    let falloff = 1.0 - t * t;
    acc = acc + rzLightColor(i) * (ndl * falloff * falloff);
  }
  return acc;
}
`
}
