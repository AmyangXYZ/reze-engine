#anchor 左手首 trail
#anchor 右手首 trail
#particles 320
#blend additive
#bloom

// A neon ribbon along the path each hand actually took — with the sparks it
// sheds living as real GPU particles.
//
// The fullscreen original walked up to 128 trail samples PER PIXEL and drew its
// sparks by looping over every path sample at every pixel; it looked right and
// cost the frame rate. This version draws the same recorded history as
// geometry: 127 spline-smoothed segments per hand, and the sparks are the first
// use of both mounts in one file — particleInit spawns them ON the trail, from
// the same buffer the ribbon reads.
//
// The ribbon renders into its OWN layer, max-blended — the original's
// core-takes-the-max rule as a blend mode — then composites over the frame
// AFTER tone mapping, exactly where the fullscreen version ran. Max is why the
// parallel strands of a circling hand cannot double into bright dashes, and
// post-tonemap is why these colours reach the screen verbatim, never bleached
// by AgX. (The @blend/@bloom pragmas below apply to the SPARKS, which are
// particles in the scene pass; the ribbon's blending is fixed by the layer.)
//
// Tunables — edit and ⌘⏎.
const CORE_COLOR = vec3f(0.86, 0.97, 1.0);   // white-hot, faintly blue
const BODY_COLOR = vec3f(0.12, 0.42, 1.0);   // electric blue
const HAZE_COLOR = vec3f(0.34, 0.10, 0.98);  // violet, in the outer light

const LIFETIME = 0.80;   // seconds of path drawn behind the hand
// PIXELS, exactly as the original measured — the ribbon is extruded in screen
// space now, so the constant means what it meant there, and the trail holds its
// width on screen at any camera distance.
const WIDTH = 32.0;      // half-width of the GLOW in pixels at 1080p
const CORE_FRAC = 0.21;  // the hot core inside it — the original's 7px in 34
const TAIL_WIDTH = 0.40; // core width at the far end, as a fraction
const TAIL_EASE = 0.22;  // last stretch of life spent fading to nothing
// Rebalanced toward the saber grammar: the white-hot part is a THIN line and
// the saturated colour is most of the visible width. CORE high makes white
// spread; HALO carries the neon.
const CORE = 2.6;        // weight of the core — what makes it read as light
// 0.55 in the original — times the overlap its `+=` accumulated. There every
// pixel SUMMED every segment within the glow radius (about 1.5 segments' worth
// on a straight path); here the quads tile, so each pixel samples the profile
// once and the weight carries the difference.
// The wide soft skirt. It is what BLOOM has to catch: the pyramid works on
// what is already on screen, so a thin hot filament with nothing around it
// blooms into a thin hot filament. Widening the skirt is what makes the
// light spread rather than just brighten.
const HALO = 0.85;
const LIGHT = 1.35;      // final gain on the ribbon's alpha
// Ribbons draw INSIDE the scene pass now — HDR, before tone mapping — so this
// colour is no longer what reaches the screen. vec3f(1.0) goes through AgX and
// lands as grey, which is why the ribbon dimmed the moment it stopped being
// pasted over the finished frame. Same constant Snow carries, for the same
// reason: anything meant to read as LIGHT has to out-shine the paper.
const INTENSITY = 3.2;

// Sparks. World units; a model stands ~20 units tall, judge against that.
// Sparks are PARTICLES, so unlike the ribbon they render in the scene pass and
// pass through AgX — which bleaches any bright tint toward white. The colour
// has to be DEEP, with the lightness confined to the pinpoint centre.
const SPARK_COLOR = vec3f(0.36, 0.24, 1.0);  // deep neon violet-blue, the halo
const SPARK_HOT = vec3f(0.72, 0.60, 1.0);    // the pinpoint, kept lavender
const SPARK_LIFE = 0.80;  // seconds — at or below LIFETIME, or dots outlive the trail
const SPARK_EMERGE = 0.30;// spent fading in, while it swings clear of the ribbon
const SPARK_SIZE = 0.07;
const ORBIT_SPEED = 2.5;  // how hard the path curls sideways as it flies
const DRIFT = 2.6;        // outward speed at birth — clear the ribbon fast
const RISE = 2.4;         // upward drift
const TWINKLE = 9.0;
const SPARK_INTENSITY = 1.7;   // low, or AgX's shoulder eats the hue

// ── The ribbon ──────────────────────────────────────────────────────────────

fn trailWidth(u: f32, age: f32) -> f32 {
  // The quad IS the glow, and the glow keeps a constant radius — only the core
  // narrows down the tail (in trailShade). The width itself just closes the very
  // end so the oldest sample is not a hard edge, and it deliberately ignores how
  // fast the hand moves: pinching where the hand slows breaks the ribbon at
  // every turn, because a turn is where a hand decelerates.
  let a = clamp(1.0 - age / LIFETIME, 0.0, 1.0);
  // rzViewportHeight()/1080 is the original's pxK: the same effect at a 4K
  // export and in the editor, instead of a different-looking one per resolution.
  return WIDTH * (rzViewportHeight() / 1080.0) * smoothstep(0.0, TAIL_EASE, a);
}

fn trailShade(u: f32, v: f32, age: f32, weight: f32, slot: i32) -> vec4f {
  // `a` runs 1 at the hand to 0 at the end of LIFETIME, exactly as the original.
  let a = clamp(1.0 - age / LIFETIME, 0.0, 1.0);
  let ease = smoothstep(0.0, TAIL_EASE, a);
  let bright = a * a * ease;
  let d = abs(v);

  // A GAUSSIAN core inside a compact-support halo — the original's two shapes.
  // `weight` is the engine's segment length over a full contribution: the line
  // integral, so a pausing hand goes dim and blue instead of white.
  let frac = CORE_FRAC * mix(TAIL_WIDTH, 1.0, pow(a, 0.7));
  let core = exp(-(d * d) / max(frac * frac, 1e-6)) * bright * weight;
  let halo = rzFalloff(d, 1.0) * bright * weight;
  let heat = core * CORE + halo * HALO;

  // Colour FROM intensity: white only where it is genuinely hot, azure through
  // the body, deep blue in the outer light.
  var rgb = mix(HAZE_COLOR, BODY_COLOR, smoothstep(0.03, 0.40, heat));
  rgb = mix(rgb, CORE_COLOR, smoothstep(0.45, 1.30, heat));
  // No gain, no tone-map compensation: the layer composites after AgX, so what
  // is written here is what appears — the original's own contract.
  return vec4f(rgb * INTENSITY, clamp(heat * LIGHT, 0.0, 1.0));
}

// ── The sparks ──────────────────────────────────────────────────────────────

fn particleInit(id: u32, seed: f32) -> Particle {
  var p: Particle;
  let h = rzHash13(seed + f32(id) * 0.0193);
  let h2 = rzHash13(seed * 3.1 + f32(id) * 0.0711);
  // A hand, a character, a recent point on that hand's path.
  let slot = i32(h.x * f32(RZ_TRAIL_SLOTS));
  var subject = i32(h.y * 4.0) % 4;
  if (rzTrailCount(subject, slot) < 6) { subject = 0; }
  let n = rzTrailCount(subject, slot);
  // Half a second of real history before the FIRST spark — a catch-all for
  // every first-frame path shape (bind pose, load pop-in, first-pose snap):
  // whatever those look like, a freshly started trail sits out the grace and a
  // burst at scene start is impossible by construction, not by enumeration.
  if (n < 6 || rzTrail(subject, slot, n - 1).w < 0.5) {
    p.life = 0.0;
    return p;
  }
  // Spread along the recent path, not piled at the live end — 512 sparks all
  // born at the newest five samples was a permanent starburst welded to the
  // hand, and its bloom spikes read as burrs on the ribbon.
  let k = 1 + i32(h.z * 24.0);
  let sampleA = rzTrail(subject, slot, k);
  let sampleB = rzTrail(subject, slot, min(k + 2, n - 1));
  // Sparks are shed by MOTION, so a still path sheds none. A resting hand keeps
  // a trail — a cluster of near-identical samples — and unguarded spawns
  // fountained from it while the ribbon itself rightly drew nothing. The same
  // guard kills the burst at the world origin while a model loads: its bones
  // sit at (0,0,0) then, which is just another perfectly still path.
  if (distance(sampleA.xyz, sampleB.xyz) < 0.06) {
    p.life = 0.0;
    return p;
  }
  p.pos = sampleA.xyz;
  // Thrown outward and upward; the swirl in the step curls the path.
  let phase = h2.x * 6.2831853;
  p.vel = vec3f(cos(phase) * DRIFT, RISE * 0.5, sin(phase) * DRIFT);
  p.size = SPARK_SIZE * (0.6 + 0.9 * h2.y);
  p.life = SPARK_LIFE * (0.7 + 0.5 * h2.z);
  p.seed = h2.x * 39.0 + h.z * 17.0;
  return p;
}

fn particleStep(p: Particle, dt: f32) -> Particle {
  var q = p;
  // Curl the horizontal velocity, so a spark orbits away from the path rather
  // than flying a straight line off it.
  let c = cos(ORBIT_SPEED * dt);
  let s = sin(ORBIT_SPEED * dt);
  q.vel = vec3f(q.vel.x * c - q.vel.z * s, q.vel.y + RISE * 0.6 * dt, q.vel.x * s + q.vel.z * c);
  q.pos = q.pos + q.vel * dt;
  return q;
}

fn particleShade(p: Particle, uv: vec2f) -> vec4f {
  let q = (uv - vec2f(0.5)) * 2.0;
  let dd = length(q);
  let t = clamp(p.age / max(p.life, 1e-3), 0.0, 1.0);
  // Fade IN first: a spark born at full brightness is a bright dot sitting on
  // the ribbon. By the time it is visible it has swung clear.
  let emerge = smoothstep(0.0, SPARK_EMERGE / SPARK_LIFE, t);
  let fall = (1.0 - t) * (1.0 - t);
  let twinkle = 0.4 + 0.6 * sin(rzTime() * TWINKLE + p.seed);
  // A hard pinpoint inside a soft bloom — a spark, not a smudge.
  let soft = exp(-(dd * dd) / 0.5);
  let hot = exp(-(dd * dd) / 0.08);
  let alpha = clamp((soft * 0.55 + hot) * fall * twinkle * emerge, 0.0, 1.0);
  // Two-tone: violet halo, lavender heart — colour from intensity, the ribbon's
  // own rule at spark scale.
  let rgb = mix(SPARK_COLOR, SPARK_HOT, clamp(hot, 0.0, 1.0));
  return vec4f(rgb * SPARK_INTENSITY, alpha);
}

// ── The light the ribbons throw ──────────────────────────────────────────────
//
// The ribbons already GLOW: they draw inside the scene pass, so the bloom
// pyramid catches them. What they never did is LIGHT anything — a neon band
// whipping past a dress left it exactly as the sun had it, which is the same
// tell the fire and the follow-spots had before they declared lights.
//
// One light per wrist, at the anchors the ribbons are already drawn from, so
// the light is where the band is by construction rather than by a second guess
// at where her hands are.
#lights 2
const RIBBON_LIGHT_I = 1.4;   // brightness at the wrist
const RIBBON_LIGHT_R = 7.0;   // world units — a band lights what it passes,
                              // not the room; this is about an arm's length

fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  // BODY_COLOR, not CORE_COLOR: the core is white-hot and would wash whatever
  // it touched, and the colour a viewer reads off the ribbon is the body.
  l.color = BODY_COLOR;
  l.intensity = 0.0;
  l.radius = 1.0;
  l.pos = vec3f(0.0, 0.0, 0.0);

  // Slot 0 is the left wrist, 1 the right — the same order the anchors are
  // declared in at the top of this file, which is what rzAnchor's slots mean.
  //
  // SUBJECT 0, deliberately. The ribbons themselves are drawn for every
  // character in the scene, but a light is a declared cost: covering four
  // dancers would spend eight of the sixteen slots on one effect, and leave a
  // composer no room for the stage. One dancer is what this is for.
  let a = rzAnchor(0, i32(i));
  if (!a.valid) { return l; }
  l.pos = a.pos;
  l.radius = RIBBON_LIGHT_R;

  // Brighter the faster the hand moves, because that is when the ribbon is
  // longest and brightest — a still hand trails almost nothing, and a light
  // blazing off a motionless wrist reads as a lamp she is holding.
  let speed = length(a.vel);
  l.intensity = RIBBON_LIGHT_I * clamp(speed / 12.0, 0.0, 1.0);
  return l;
}
