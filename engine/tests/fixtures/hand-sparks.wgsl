#anchor 左手首 trail
#anchor 右手首 trail
#particles 3600
#blend additive
#bloom
#lights 2

// Hand Sparks — both hands throwing sparks.
//
// Bound to BOTH WRISTS, so the source travels with her: sparks leave each hand
// wherever she puts it, and a fast hand throws them further than a still one.
// That is the difference between an effect that is IN a performance and one that
// is behind it — change the anchors at the top of this file and they throw from
// wherever you name instead.
//
// NO RIBBON. The trail mount would draw a solid band between the samples of a
// hand's path, and a band is a different object: it reads as cloth or as light
// wrapped around her. These are only ever separate sparks, thrown and falling,
// so what draws her movement is the shape of the shower rather than a line.
//
// A TEMPERATURE RUN, NOT A PALETTE. Flame goes blue, then white, then yellow,
// then red as it cools, so the hottest thing in the picture is the blue one —
// the instant a spark leaves her hand. Reading the colours in that order is what
// makes the shower look like something burning rather than something tinted, and
// it is where the colour comes from without any of it being invented.
//
// The blue segment is deliberately SHORT. The blue core of a real flame is
// always its smallest part; stretch it and the whole thing reads as a gas jet.
//
// It pours continuously rather than in bursts. The engine staggers the pool's
// lifetimes anyway, so the fan is dense at every instant instead of pulsing.
//
// THE STREAK IS THE SPEED, which is the one idea worth taking from the spark
// shaders that draw these as cylinders. A spark's length on screen is its
// velocity: it leaves at fifteen metres a second as a long thin line, the air
// takes that away in a fraction of a second, and by the time it falls it is a
// dot. Drawing every spark the same length is what makes a shower look like
// confetti. The engine stretches a quad along its own velocity in the CAMERA's
// basis, so the length is recomputed from the live speed each frame and reads
// correctly from any angle.
//
// The sparks are BALLISTIC. Thrown outward, slowed hard by drag, pulled down by
// gravity, and dying young — which is what separates a spark from a firefly.

// Tunables.
const SPEED = 4.6;         // metres a second, leaving the bone
const SPEED_VAR = 0.85;    // how much that varies spark to spark — a few go far
const CONE = 0.75;         // half-angle of the spray, radians. 0.75 is about 43 degrees
const INHERIT = 0.5;       // how much of the hand's own motion they carry away
const GRAVITY = 4.2;
const DRAG = 2.4;          // per second. Air stops a spark fast, and that is the look
const LIFE = 1.0;          // they carry past the hand that threw them
const LIFE_VAR = 0.7;
// HAIRLINE. The width is the whole difference between a needle and a dash: at
// 0.012 a spark is about a pixel across at stage distance, and the stretch below
// turns that pixel into a scratch. Widen it and the fan becomes a spray of rice.
const SIZE = 0.024;
const LENGTH = 1.9;        // streak length per metre/second of speed
const LENGTH_MAX = 16.0;   // and its ceiling, so a fast one is a line and not a smear
const BLUE = vec3f(0.42, 0.68, 1.00); // the instant it leaves her — hottest
const HOT = vec3f(1.00, 0.97, 0.86);  // white
const WARM = vec3f(1.00, 0.46, 0.07); // burning down
const DEAD = vec3f(0.44, 0.04, 0.01); // and out
const BLUE_END = 0.12;    // how much of a spark's life is still blue — keep it short
const WHITE_END = 0.34;
const GAIN = 4.6;          // HDR: white here is grey after AgX
const LIGHT_COLOR = vec3f(1.00, 0.62, 0.22);  // the shower's body, not its blue tip
const LIGHT_I = 1.15;
const LIGHT_R = 7.0;      // world units — an arm's length, not the room
const LIGHT_REST = 0.4;   // what a still hand keeps
const LIGHT_FULL = 9.0;   // hand speed at which the light is fully up

/**
 * The temperature run: 0 the instant a spark is thrown, 1 as it goes out.
 * Blue, white, orange, red — in that order, because that is the order metal
 * cools in.
 */
fn burnColor(t: f32) -> vec3f {
  if (t < BLUE_END) {
    return mix(BLUE, HOT, clamp(t / BLUE_END, 0.0, 1.0));
  }
  if (t < WHITE_END) {
    return mix(HOT, WARM, clamp((t - BLUE_END) / (WHITE_END - BLUE_END), 0.0, 1.0));
  }
  return mix(WARM, DEAD, clamp((t - WHITE_END) / (1.0 - WHITE_END), 0.0, 1.0));
}

fn particleInit(id: u32, seed: f32) -> Particle {
  var p: Particle;
  let r = rzHash13(seed + f32(id) * 0.0211);
  let r2 = rzHash13(seed * 2.17 + f32(id) * 0.0473);

  // EITHER HAND. The slot is chosen per spark, so both wrists throw from the
  // same pool and neither has to be given half of it up front — a hand held
  // still simply stops drawing from it.
  let slot = select(0, 1, r.z > 0.5);
  let a = rzAnchor(0, slot);
  // No such bone on this rig: park it dead rather than showering the world
  // origin, which is what an unchecked anchor does on every model that spells
  // the name differently.
  if (!a.valid) {
    p.life = 0.0;
    p.pos = vec3f(0.0, -1000.0, 0.0);
    return p;
  }
  p.pos = a.pos;

  // A NARROW CONE ALONG THE BONE, not a sphere. A torch throws its spray one
  // way — the way the tool is pointing — and a full-sphere emitter is a
  // dandelion. rzAnchor gives the bone's own forward axis, so the spray turns
  // with her wrist for free.
  //
  // Uniform in cos(theta) rather than in theta: sampling the angle directly
  // crowds the axis and leaves the rim of the cone thin.
  var axis = a.fwd;
  if (dot(axis, axis) < 1e-6) { axis = vec3f(0.0, 1.0, 0.0); }
  axis = normalize(axis);
  // A frame around it. Crossing with +Y collapses when the bone points up, so
  // the second axis is chosen away from it. (Not named "ref" — WGSL reserves it,
  // and the error lands on the line that USES the variable rather than the one
  // that declares it.)
  let away = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(axis.y) > 0.9);
  let t = normalize(cross(away, axis));
  let b = cross(axis, t);
  let phi = r.x * 6.2831853;
  let cz = mix(1.0, cos(CONE), r.y);
  let sz = sqrt(max(0.0, 1.0 - cz * cz));
  let dir = normalize(axis * cz + (t * cos(phi) + b * sin(phi)) * sz);

  // Fast, and varying hard: a shower where every spark leaves at the same rate
  // has a visible front edge to it.
  let speed = SPEED * mix(1.0 - SPEED_VAR, 1.0 + SPEED_VAR, r2.x * r2.x);
  // And it carries some of the hand away with it, so a thrown arm throws sparks.
  p.vel = dir * speed + a.vel * INHERIT;

  p.size = SIZE * mix(0.6, 1.4, r2.y);
  // Set here and recomputed every step: the length IS the speed.
  p.stretch = clamp(speed * LENGTH, 1.0, LENGTH_MAX);
  p.life = LIFE * (1.0 - LIFE_VAR * 0.5 + LIFE_VAR * r2.z);
  p.seed = seed + f32(id) * 0.0211;
  return p;
}

fn particleStep(p: Particle, dt: f32) -> Particle {
  var q = p;
  // Drag first, then gravity: a spark is light, so the air takes its speed away
  // long before the ground takes its height. Exponential, so the result does not
  // depend on the frame rate.
  q.vel = q.vel * exp(-DRAG * dt);
  q.vel.y = q.vel.y - GRAVITY * dt;
  q.pos = q.pos + q.vel * dt;
  // The streak follows the speed down. A spark that keeps its launch length
  // after the air has stopped it is a dash hanging in the air.
  q.stretch = clamp(length(q.vel) * LENGTH, 1.0, LENGTH_MAX);
  return q;
}

fn particleShade(p: Particle, uv: vec2f) -> vec4f {
  let q = (uv - vec2f(0.5)) * 2.0;
  let r = length(q);
  // Compact support: exactly zero at the quad's edge, so bloom has no square to
  // find. A hard core inside a soft one — the shape of something incandescent.
  let core = pow(1.0 - clamp(r, 0.0, 1.0), 3.0);
  let hot = core * core;

  let age = clamp(p.age / max(p.life, 1e-4), 0.0, 1.0);
  let col = burnColor(age);
  // It dims as it cools, and it dies out rather than being cut off.
  let fade = pow(1.0 - age, 1.6);
  let e = clamp(hot + core * 0.35, 0.0, 1.0) * fade;
  if (e <= 0.0) { return vec4f(0.0); }
  return vec4f(col * GAIN, e);
}

/**
 * One lamp at the fuse, so the spit actually lights her hand.
 *
 * Bloom spreads bright pixels in screen space and never enters shading — a
 * bloomed spark does not illuminate the sleeve next to it. This does.
 */
fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  // The shower's BODY, never the blue tip. A blue-white lamp on skin reads as
  // moonlight, and the colour a viewer takes off the sparks is the orange they
  // spend most of their life at.
  l.color = LIGHT_COLOR;
  l.intensity = 0.0;
  l.radius = 1.0;
  l.pos = vec3f(0.0, 0.0, 0.0);

  // One lamp per wrist. Slot 0 is the left and 1 the right, the order the
  // anchors are declared at the top — the light index IS the slot here, unlike
  // the sparks, which pick a hand each.
  //
  // SUBJECT 0 only. A light is a declared cost, and covering four dancers would
  // spend eight of the sixteen slots on one effect.
  let a = rzAnchor(0, i32(i));
  if (!a.valid) { return l; }
  l.pos = a.pos;
  l.radius = LIGHT_R;

  // HOW FAST SHE IS MOVING, AVERAGED OVER THE WHOLE RECORDED PATH.
  //
  // Not length(a.vel): that is one frame's velocity, and a pose track is sampled
  // rather than smooth, so it jumps between frames. A light driven by it
  // flickers on the model and no curve fixes that, because the input is the
  // noise. Walking the path and dividing the distance covered by how long it
  // took is an average over the trail's whole span, so the light can change only
  // as fast as a hand can change its mind.
  //
  // Path LENGTH, not the distance between the ends — a circling hand returns to
  // where it started and would otherwise read as standing still. Divine Ribbon
  // arrived at the same thing for the same reason.
  let n = rzTrailCount(0, i32(i));
  var travelled = 0.0;
  if (n >= 2) {
    var prev = rzTrail(0, i32(i), 0).xyz;
    for (var s = 1; s < n; s = s + 1) {
      let cur = rzTrail(0, i32(i), s).xyz;
      travelled = travelled + distance(prev, cur);
      prev = cur;
    }
  }
  let span = max(rzTrail(0, i32(i), max(n - 1, 0)).w, 1e-3);
  l.intensity = LIGHT_I * mix(LIGHT_REST, 1.0, clamp((travelled / span) / LIGHT_FULL, 0.0, 1.0));
  return l;
}
