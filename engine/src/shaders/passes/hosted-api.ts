// The half of the effect API whose VALUES are module-local.
//
// An effect file is spliced WHOLE into every module it has a mount in. A file
// with a trail and a lightEmit compiles its trail code inside the light module,
// where it never runs but still has to resolve — so `rzTime()` has to exist
// there even though a compute pass that writes lamp positions has no obvious
// clock, and `rzFalloff()` has to exist there even though nothing calls it.
//
// EFFECT_SCENE_API (composite.ts) is the other half: names that mean the same
// thing everywhere, written once. These are the ones that cannot be, because
// each module keeps its clock and its viewport in a different uniform. So they
// are written once HERE and the expression is the parameter — which is the
// same trick the grid pass already used for one of them, generalised to all.
//
// The rule this file exists to make keepable: a hosting module includes every
// block below. Not the ones its own mount happens to need — every one, because
// what lands in it is the author's whole file.
//
// EXCEPT that EFFECT_SCENE_API already carries EFFECT_MATH_API and the Particle
// struct, so the three modules built on it take those two by inheritance and
// must not add them again. tests/hosted-api.test.mjs fails on a name declared
// twice for exactly this reason — it is the mistake sharing blocks invites.

/**
 * The particle record, laid out by hand — declared in EVERY hosting module.
 *
 * Not only in the particle ones: `fn particleStep(p: Particle)` is part of the
 * author's file, so the trail, field and lightEmit modules compile that
 * signature as dead code and a missing struct is a compile error on a function
 * the author was right to write. The trail module used to carry a hand-copied
 * duplicate for exactly this reason.
 *
 * `age` and `life` sit in the padding that vec3f alignment would waste anyway
 * (a vec3f occupies 12 bytes but aligns the next field to 16), so the struct is
 * 48 bytes rather than the 64 a naive ordering costs. At 4096 particles that is
 * 192KB instead of 256KB, and it is read every frame by both stages.
 *
 * `life <= 0` means "not alive" and is what the pool checks to recycle a slot,
 * so a freshly zeroed buffer is entirely dead and every particle is born on the
 * first step rather than needing a separate seeding pass.
 *
 * RzLight is the same kind of thing and lives in lights.ts, because it is that
 * module's own type; every hosting module splices it for this same reason.
 */
export const PARTICLE_STRUCT_WGSL = /* wgsl */ `
struct Particle {
  pos: vec3f,
  age: f32,
  vel: vec3f,
  life: f32,
  size: f32,
  rot: f32,
  seed: f32,
  // Aspect along the direction of travel. 1 or less is a square billboard; a
  // raindrop is 10 or 20. Zero-initialised, so an effect that never sets it gets
  // the square it expects.
  stretch: f32,
}
`

/**
 * `rzTime`/`rzDt`, from whatever the module calls its clock.
 *
 * A module with no clock of its own passes a literal. Zero dt is honest there:
 * nothing spliced into it is being stepped, so a kernel that integrates would
 * integrate by nothing rather than by a plausible-looking wrong number.
 */
export function clockApi(timeExpr: string, dtExpr: string): string {
  return /* wgsl */ `
fn rzTime() -> f32 { return ${timeExpr}; }
fn rzDt() -> f32 { return ${dtExpr}; }
`
}

/** `rzViewportHeight` — the render target's height in pixels, which is what a
 *  point size in world units has to be divided by to become a pixel radius. */
export function viewportApi(heightExpr: string): string {
  return /* wgsl */ `
fn rzViewportHeight() -> f32 { return ${heightExpr}; }
`
}

/**
 * The one cast number that is PER EFFECT rather than per engine.
 *
 * Author-visible, and published effects loop over it, so its meaning is pinned:
 * how many of THIS effect's anchors asked for a trail. Deliberately not the
 * anchor address space — that is RZ_MAX_ANCHORS in CAST_API, and the two being
 * one number was the old trail bug. Everything else about the cast's shape is a
 * constant and lives there.
 */
export function trailSlotsApi(trailCount: number): string {
  return /* wgsl */ `
const RZ_TRAIL_SLOTS: i32 = ${trailCount};
`
}

/**
 * Pure math — no bindings, no uniforms, identical in every module.
 *
 * These were duplicated in the particle and trail modules, byte for byte in the
 * case of rzFalloff and to a `+ vec3f(0.0)` in the case of rzValueNoise. Two
 * copies of a noise function is one bad merge away from an effect that looks
 * different depending on which mount drew it, which is not a bug anyone would
 * think to look for.
 *
 * EVERY hosting module includes this and none defines any of these itself, so
 * "which module am I in" can never change what a helper returns.
 */
export const EFFECT_MATH_API = /* wgsl */ `
fn rzHash11(x: f32) -> f32 {
  var p = fract(x * 0.1031);
  p = p * (p + 33.33);
  return fract(p * (p + p));
}
fn rzHash21(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
/** Three independent randoms from one seed — the usual need when spawning. */
fn rzHash13(x: f32) -> vec3f {
  return vec3f(rzHash11(x), rzHash11(x + 17.13), rzHash11(x + 41.71));
}
/**
 * Compact-support falloff: 1 at the centre, exactly 0 at r, smooth between.
 *
 * It reaches exactly zero rather than merely getting small, because a glow that
 * never quite ends has to be culled somewhere, and culling it wherever it looks
 * close enough is what put a visible hard edge on the first halo effect.
 */
fn rzFalloff(d: f32, r: f32) -> f32 {
  let x = clamp(d / max(r, 1e-6), 0.0, 1.0);
  let f = 1.0 - x;
  return f * f * f;
}
fn rzHash31(p: vec3f) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 = p3 + dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
fn rzValueNoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let n000 = rzHash31(i);
  let n100 = rzHash31(i + vec3f(1.0, 0.0, 0.0));
  let n010 = rzHash31(i + vec3f(0.0, 1.0, 0.0));
  let n110 = rzHash31(i + vec3f(1.0, 1.0, 0.0));
  let n001 = rzHash31(i + vec3f(0.0, 0.0, 1.0));
  let n101 = rzHash31(i + vec3f(1.0, 0.0, 1.0));
  let n011 = rzHash31(i + vec3f(0.0, 1.0, 1.0));
  let n111 = rzHash31(i + vec3f(1.0, 1.0, 1.0));
  let x00 = mix(n000, n100, u.x);
  let x10 = mix(n010, n110, u.x);
  let x01 = mix(n001, n101, u.x);
  let x11 = mix(n011, n111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
/** Divergence-free flow — the field a wisp of smoke follows without a solver. */
fn rzCurlNoise(p: vec3f) -> vec3f {
  let e = 0.1;
  let dx = vec3f(e, 0.0, 0.0);
  let dy = vec3f(0.0, e, 0.0);
  let dz = vec3f(0.0, 0.0, e);
  let x0 = rzValueNoise(p - dx); let x1 = rzValueNoise(p + dx);
  let y0 = rzValueNoise(p - dy); let y1 = rzValueNoise(p + dy);
  let z0 = rzValueNoise(p - dz); let z1 = rzValueNoise(p + dz);
  return normalize(vec3f((y1 - y0) - (z1 - z0), (z1 - z0) - (x1 - x0), (x1 - x0) - (y1 - y0)) + vec3f(1e-6));
}
`
