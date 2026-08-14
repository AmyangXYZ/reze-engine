import { audioApi } from "../audio-api"
import { scoreApi } from "../score-api"
// GPU particles for user effects: a compute step and an instanced quad draw.
//
// Its own shader MODULE rather than more source spliced into composite.ts, for
// three reasons that all point the same way. Cost: a fullscreen effect pays for
// every pixel on screen whether it drew anything there or not, so a ribbon walked
// as a field cost O(pixels × samples) where the same picture as geometry costs
// O(segments) — the difference between one effect and a scene full of them.
// Occlusion: geometry drawn inside the scene pass is depth-tested, so a particle
// behind the character is simply hidden, which a composited field can only fake.
// And names: separate modules cannot collide, so two effects may both define
// `hash21` and never meet — which is what makes running several at once safe.
//
// The author writes three functions and the engine owns the loop. That split is
// deliberate: spawning, ageing, recycling, billboarding and sorting are the same
// in every particle effect ever written, and an author who has to re-derive them
// gets them subtly wrong in a way that only shows up on someone else's machine.

/** Where the cast/trail history sits in the shared storage buffer. */
export type CastLayout = { subjects: number; samples: number; base: number; trailBase: number; slots: number }

/**
 * The trail accessors, in the PARTICLE module.
 *
 * Sparks are the reason: the original hand ribbon shed sparks along its path,
 * and as real particles they need to SPAWN on that path — which means
 * particleInit reading the same recorded history the trail draws from. One
 * effect file, two mounts, one buffer.
 */
/** Shared with the sim pass, which reads the same buffer for the same reason:
 *  a kernel that displaces fog has to know where the dancer's feet are. */
export function castApi(cast: CastLayout): string {
  return `
const RZ_SUBJECTS: i32 = ${cast.subjects};
const RZ_SAMPLES: i32 = ${cast.samples};
const RZ_TRAIL_SLOTS: i32 = ${cast.slots};
fn rzSubjectCount() -> i32 {
  var n = 0;
  for (var i = 0; i < RZ_SUBJECTS; i++) {
    if (_rzCast[i * 3 + 2].w > 0.0) { n = i + 1; }
  }
  return n;
}
fn rzTrailCount(subject: i32, slot: i32) -> i32 {
  if (subject < 0 || subject >= RZ_SUBJECTS || slot < 0 || slot >= RZ_TRAIL_SLOTS) { return 0; }
  return i32(_rzCast[${cast.base} + (slot * RZ_SUBJECTS + subject) * 3 + 2].w);
}
/** Sample i of a path: xyz where it was, w how many seconds ago. i = 0 is now. */
fn rzTrail(subject: i32, slot: i32, i: i32) -> vec4f {
  let n = rzTrailCount(subject, slot);
  if (i < 0 || i >= n) { return vec4f(0.0); }
  return _rzCast[${cast.trailBase} + (slot * RZ_SUBJECTS + subject) * RZ_SAMPLES + i];
}
`
}

/** How the author's quads combine with the scene. */
export type ParticleBlend = "alpha" | "additive"

export type ParticleSource = {
  /** The author's WGSL verbatim. */
  wgsl: string
  /** Simultaneous particles. Fixed at install; the pool recycles rather than grows. */
  count: number
  blend: ParticleBlend
  /** Feed the bloom pyramid. Off by default — rain should not glow. */
  bloom: boolean
}

/** Bytes per particle. Explicitly padded — see the struct below. */
export const PARTICLE_STRIDE = 48

/**
 * The particle record, laid out by hand.
 *
 * `age` and `life` sit in the padding that vec3f alignment would waste anyway
 * (a vec3f occupies 12 bytes but aligns the next field to 16), so the struct is
 * 48 bytes rather than the 64 a naive ordering costs. At 4096 particles that is
 * 192KB instead of 256KB, and it is read every frame by both stages.
 *
 * `life <= 0` means "not alive" and is what the pool checks to recycle a slot,
 * so a freshly zeroed buffer is entirely dead and every particle is born on the
 * first step rather than needing a separate seeding pass.
 */
const PARTICLE_STRUCT = /* wgsl */ `
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

const CAMERA_STRUCT = /* wgsl */ `
struct CameraU {
  view: mat4x4f,
  proj: mat4x4f,
  camPos: vec3f,
  targetHeight: f32,
}
`

const PARTICLE_UNIFORMS = /* wgsl */ `
struct ParticleU {
  time: f32,
  dt: f32,
  count: u32,
  frame: u32,
}
`

/**
 * The shared prelude, everything `rz`-prefixed.
 *
 * Not convenience — correctness. Every effect written against the old contract
 * re-derived its own hash and its own falloff, which is duplicated code and
 * duplicated bugs; `rzFalloff` in particular has COMPACT SUPPORT (it reaches
 * exactly zero at r), because an exponential glow that never quite reaches zero
 * has to be culled somewhere, and culling it wherever it "looks close enough"
 * is what put a visible hard edge on the first halo effect.
 */
const PRELUDE = /* wgsl */ `
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
fn rzHash31(p: vec3f) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 = p3 + dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
/** Three independent randoms from one seed — the usual need when spawning. */
fn rzHash13(x: f32) -> vec3f {
  return vec3f(rzHash11(x), rzHash11(x + 17.13), rzHash11(x + 41.71));
}
fn rzValueNoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let n000 = rzHash31(i + vec3f(0.0, 0.0, 0.0));
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
/** Divergence-free flow — the standard drifting-air force. Snow and mist want this. */
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
/** Compact-support falloff: 1 at the centre, exactly 0 at r, smooth between. */
fn rzFalloff(d: f32, r: f32) -> f32 {
  let x = clamp(d / max(r, 1e-6), 0.0, 1.0);
  let f = 1.0 - x;
  return f * f * f;
}
fn rzTime() -> f32 { return pu.time; }
fn rzViewportHeight() -> f32 { return cam.targetHeight; }
fn rzCameraPos() -> vec3f { return cam.camPos; }
fn rzCameraRight() -> vec3f { return vec3f(cam.view[0][0], cam.view[1][0], cam.view[2][0]); }
fn rzCameraUp() -> vec3f { return vec3f(cam.view[0][1], cam.view[1][1], cam.view[2][1]); }
fn rzCameraForward() -> vec3f { return vec3f(cam.view[0][2], cam.view[1][2], cam.view[2][2]); }
/** World point → (uv, view distance), same contract as the field mounts' rzProject. */
fn rzProject(p: vec3f) -> vec3f {
  let clip = cam.proj * cam.view * vec4f(p, 1.0);
  let w = max(clip.w, 1e-4);
  return vec3f(clip.xy / w * 0.5 + 0.5, clip.w);
}
fn rzDt() -> f32 { return pu.dt; }
fn rzCamPos() -> vec3f { return cam.camPos; }
`

/** `// @particles 4096` — how many live at once. */
export function parseParticleCount(wgsl: string, max: number): number {
  const m = /^\s*\/\/\s*@particles\s+(\d+)\s*$/m.exec(wgsl)
  if (!m) return 0
  // Clamped rather than rejected: an author asking for a million gets the most
  // the engine will give and a scene that still runs, which is a better failure
  // than a compile error naming a number they had no way to know.
  return Math.max(1, Math.min(max, parseInt(m[1], 10)))
}

/** `// @bloom` — opt in to the bloom pyramid. Sparks want it; rain does not. */
export function parseParticleBloom(wgsl: string): boolean {
  return /^\s*\/\/\s*@bloom\s*$/m.test(wgsl)
}

/** `// @blend additive` — default is straight alpha. */
export function parseParticleBlend(wgsl: string): ParticleBlend {
  return /^\s*\/\/\s*@blend\s+additive\s*$/m.test(wgsl) ? "additive" : "alpha"
}

/** Does the source define the particle contract? All three are required together. */
export function particleEntryPoints(wgsl: string): { init: boolean; step: boolean; shade: boolean } {
  return {
    init: /\bfn\s+particleInit\s*\(/.test(wgsl),
    step: /\bfn\s+particleStep\s*\(/.test(wgsl),
    shade: /\bfn\s+particleShade\s*\(/.test(wgsl),
  }
}

/**
 * Spawn, age, recycle.
 *
 * One kernel for both birth and update, because a dead slot and a new particle
 * are the same write — a pool that recycles has no allocation and therefore no
 * spawn-rate bookkeeping to get wrong. The cost is that lifetimes are staggered
 * only by whatever the author randomises in `particleInit`, which for rain and
 * snow is exactly right, and for a burst is what the age offset is for.
 */
export function buildParticleComputeShader(src: ParticleSource, cast: CastLayout): string {
  return (
    PARTICLE_STRUCT +
    CAMERA_STRUCT +
    PARTICLE_UNIFORMS +
    `
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> pu: ParticleU;
@group(0) @binding(2) var<uniform> cam: CameraU;
@group(0) @binding(3) var<storage, read> _rzCast: array<vec4f>;
` +
    castApi(cast) +
    audioApi(0, 4) +
    scoreApi(0, 5) +
    PRELUDE +
    "\n// ── user effect ──\n" +
    src.wgsl +
    /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= pu.count) { return; }
  var p = particles[i];
  if (p.life <= 0.0 || p.age >= p.life) {
    // The seed is stable per SLOT and per generation, so a particle looks the
    // same every time the scene is replayed at the same moment — which is what
    // keeps an exported video identical to the preview.
    let generation = floor(pu.time * 0.37) + f32(i) * 0.618;
    p = particleInit(i, rzHash11(generation));
    // age is NOT reset here. WGSL zero-initialises a var, so an author who
    // ignores it starts at zero anyway — while one who sets it to a fraction of
    // its life staggers the pool, which is the difference between snow and a
    // pulse of snow arriving all at once every few seconds.
  } else {
    p = particleStep(p, pu.dt);
    p.age = p.age + pu.dt;
  }
  particles[i] = p;
}
`
  )
}

/**
 * One camera-facing quad per live particle.
 *
 * Six vertices, no index or vertex buffer: the corners are derived from
 * `vertex_index` and the particle is read from storage by `instance_index`, so a
 * draw is `draw(6, count)` and there is nothing to upload per frame. The billboard
 * basis comes from the VIEW matrix's rows rather than from a look-at, which keeps
 * the quad square on screen no matter where the camera rolls.
 *
 * A dead particle collapses to a degenerate quad instead of being culled on the
 * CPU — the alternative is a compacted draw list, which costs a prefix sum and a
 * readback every frame to save vertices the rasteriser was going to reject anyway.
 */
export function buildParticleRenderShader(src: ParticleSource, cast: CastLayout): string {
  return (
    `override BLOOM: bool = ${src.bloom ? "true" : "false"};
override ADDITIVE: bool = ${src.blend === "additive" ? "true" : "false"};\n` +
    PARTICLE_STRUCT +
    CAMERA_STRUCT +
    PARTICLE_UNIFORMS +
    `
@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> pu: ParticleU;
@group(0) @binding(2) var<uniform> cam: CameraU;
@group(0) @binding(3) var<storage, read> _rzCast: array<vec4f>;
` +
    castApi(cast) +
    audioApi(0, 4) +
    scoreApi(0, 5) +
    PRELUDE +
    "\n// ── user effect ──\n" +
    src.wgsl +
    /* wgsl */ `
struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) @interpolate(flat) id: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var out: VSOut;
  out.id = ii;
  let p = particles[ii];
  // Two triangles, corners in the order 0,1,2, 2,1,3.
  let quad = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(-1.0, 1.0),
    vec2f(-1.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let c = quad[vi];
  out.uv = c * 0.5 + 0.5;
  if (p.life <= 0.0) {
    // Degenerate: off the near plane, rasterises nothing.
    out.clip = vec4f(0.0, 0.0, -2.0, 1.0);
    return out;
  }
  let right = vec3f(cam.view[0][0], cam.view[1][0], cam.view[2][0]);
  let up = vec3f(cam.view[0][1], cam.view[1][1], cam.view[2][1]);
  let s = sin(p.rot);
  let k = cos(p.rot);
  var r = vec2f(c.x * k - c.y * s, c.x * s + c.y * k);
  // Stretched along the direction of travel ON SCREEN — which is not the world
  // direction once the camera is off-axis. Rain falling straight down is nearly
  // a point when viewed from above and a long streak from the side, and taking
  // the velocity's components in the camera's own basis is what gets both right.
  // Rotation is ignored while stretched: the velocity IS the orientation.
  if (p.stretch > 1.0) {
    let vr = dot(p.vel, right);
    let vu = dot(p.vel, up);
    let vlen = length(vec2f(vr, vu));
    if (vlen > 1e-5) {
      let d = vec2f(vr, vu) / vlen;
      r = vec2f(d.y, -d.x) * c.x + d * (c.y * p.stretch);
    }
  }
  let world = p.pos + (right * r.x + up * r.y) * p.size;
  out.clip = cam.proj * cam.view * vec4f(world, 1.0);
  return out;
}

struct FSOut {
  @location(0) color: vec4f,
  // The scene's aux target: (bloom mask, coverage). Materials write it, so a
  // particle that skipped it would punch a hole in the mask of whatever it drew
  // over.
  //
  // vec4f even though the target is rg8unorm and only .rg land: that target's
  // blend factors reference SrcAlpha, and a fragment with no alpha channel is
  // rejected outright — "reading alpha but it is missing from fragment output".
  // The material shaders declare vec4f here for the same reason.
  @location(1) mask: vec4f,
}

@fragment
fn fs(in: VSOut) -> FSOut {
  let p = particles[in.id];
  let c = particleShade(p, in.uv);
  if (c.a <= 0.0) { discard; }
  var out: FSOut;
  // PREMULTIPLIED: the scene's colour target blends with srcFactor \"one\", so a
  // straight-alpha fragment would come out over-bright wherever it is
  // translucent — which is most of a soft particle.
  out.color = vec4f(c.rgb * c.a, c.a);
  // The mask's OPERATOR must match the colour's, or the composite invents bands.
  //
  // The composite divides the HDR colour by this coverage to un-premultiply
  // before tone mapping. An ALPHA effect writes (gate, 1.0) and lets the
  // src-alpha blend make alpha-over coverage, exactly as the materials do. An
  // ADDITIVE effect's pipeline blends this target with factor ONE instead, and
  // writes its values directly — coverage SUMS like the colour does, so
  // Σ(rgb·a)/Σa returns the true colour even where the effect overlaps itself.
  // With summed colour over alpha-over coverage, every self-overlap divided into
  // a bright white bar — visible only over the background, because the model's
  // own coverage is already 1 there and the divide is a no-op. That mismatch,
  // not geometry, was the banding that survived every geometry fix.
  let mg = select(vec2f(select(0.0, 1.0, BLOOM), 1.0), vec2f(select(0.0, c.a, BLOOM), c.a), ADDITIVE);
  out.mask = vec4f(mg.x, mg.y, 0.0, c.a);
  return out;
}
`
  )
}
