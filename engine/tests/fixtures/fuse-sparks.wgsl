#anchor 右手首
#particles 2200
#blend additive
#bloom
#lights 1

// Fuse Sparks — a struck sparkler at a bone.
//
// Bound to a BONE, so the source travels with her: the sparks leave the right
// wrist wherever she puts it, and a fast hand throws them further than a still
// one. That is the difference between an effect that is IN a performance and one
// that is behind it — change the anchor at the top of this file and it burns
// from wherever you name instead.
//
// SMALL AND FINE, BUT HOT. Not a torch throwing molten metal across the stage —
// a fuse catching against her: a compact star of hairline needles, most an inch
// long, a few carrying much further, and gone almost at once. The whole thing
// should read as a bright point with a fan of scratches around it, not as
// weather.
//
// The GEOMETRY is a sparkler's and the COLOUR is a bomb's, and that pairing is
// deliberate — cool white at this size read as clean and electrical, which is
// the wrong feeling entirely for something about to go off.
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
const SPEED = 5.2;         // metres a second, leaving the bone
const SPEED_VAR = 0.85;    // how much that varies spark to spark — a few go far
const CONE = 0.55;         // half-angle of the spray, radians. 0.55 is about 31 degrees
const INHERIT = 0.5;       // how much of the hand's own motion they carry away
const GRAVITY = 5.0;
const DRAG = 4.6;          // per second. Air stops a spark fast, and that is the look
const LIFE = 0.42;
const LIFE_VAR = 0.6;
// HAIRLINE. The width is the whole difference between a needle and a dash: at
// 0.012 a spark is about a pixel across at stage distance, and the stretch below
// turns that pixel into a scratch. Widen it and the fan becomes a spray of rice.
const SIZE = 0.012;
const LENGTH = 3.4;        // streak length per metre/second of speed
const LENGTH_MAX = 34.0;   // and its ceiling, so a fast one is a line and not a smear
const HOT = vec3f(1.00, 0.96, 0.78);  // newborn — white, running warm
const WARM = vec3f(1.00, 0.46, 0.07); // most of its life — burning orange
const DEAD = vec3f(0.44, 0.04, 0.01); // the last of it — going out red
const GAIN = 3.4;          // HDR: white here is grey after AgX
const LIGHT_COLOR = vec3f(1.0, 0.58, 0.18);
const LIGHT_POWER = 0.9;
const LIGHT_R = 2.2;

fn particleInit(id: u32, seed: f32) -> Particle {
  var p: Particle;
  let r = rzHash13(seed + f32(id) * 0.0211);
  let r2 = rzHash13(seed * 2.17 + f32(id) * 0.0473);

  let a = rzAnchor(0, 0);
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
  // COOLING IS THE PICTURE, and it is what makes these read as burning metal
  // rather than as light. White for the first quarter, then orange for most of
  // the flight, then out red — the same run a real spark makes, and short enough
  // that the eye reads it as one flash.
  let col = select(mix(WARM, DEAD, clamp((age - 0.22) / 0.78, 0.0, 1.0)),
                   mix(HOT, WARM, clamp(age / 0.22, 0.0, 1.0)),
                   age < 0.22);
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
  l.color = LIGHT_COLOR;
  l.intensity = 0.0;
  l.radius = 1.0;
  l.pos = vec3f(0.0, 0.0, 0.0);
  let a = rzAnchor(0, 0);
  if (!a.valid) { return l; }
  l.pos = a.pos;
  l.radius = LIGHT_R;
  // Steady, because the shower is. It only breathes enough to look alive.
  l.intensity = LIGHT_POWER * (0.9 + 0.1 * sin(time * 11.0));
  return l;
}
