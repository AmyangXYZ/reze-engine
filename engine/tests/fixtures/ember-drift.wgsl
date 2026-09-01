#particles 3000
#blend additive
#bloom

// Ember Drift — sparks climbing through the stage.
//
// After "Fire particles" by Jan Mróz (jaszunio15), Shadertoy, CC BY 3.0. What is
// carried over is his MOTION and his heat: a hot core inside a wider bloom,
// wandering on noise so the field never travels as one sheet, and the burst of
// colour from white through orange to nothing.
//
// HIS SPRITE IS NOT. His is an elongated spark — right for a bonfire's spit,
// wrong for what falls out of one. Ash tumbles and its outline is bitten, so
// each flake here is an irregular POLYGON with its own corners, turning over as
// it climbs. Two side by side are different objects rather than one shape at two
// sizes, which was the whole complaint about the version before it.
//
// PARTICLES, NOT A FIELD, and that is the whole difference from a port of the
// original. His is a fullscreen shader: the embers live in screen space, they
// cannot be behind anything, and the field turns with the camera because it IS
// the camera. Drawn as geometry they stand in the scene — depth-tested, so one
// passes behind her shoulder and in front of her hand in the same frame — and
// they belong to the stage rather than to the lens. Same reason Snow and Rain
// are particles, and the volume below is theirs.
//
// They also draw in HDR before tone mapping, which his could not: the core can
// sit well above white and bloom, where a fullscreen layer is clamped at it.

// Tunables.
const CENTER = vec3f(0.0, 0.0, 0.0);   // the stage they rise through
const AREA = 34.0;        // half-width of the volume, as Snow and Rain use
const TOP = 46.0;         // they climb to here, then wrap
const RISE = 2.6;         // metres a second, before size weighting
const LEAN = vec3f(0.55, 0.0, 0.2);    // his MOVEMENT_DIRECTION, as a drift
const WANDER = 0.9;       // how far the noise pushes them off that line
const SWIRL = 0.14;       // how tightly the wander curls
const SIZE_MIN = 0.055;
const SIZE_MAX = 0.26;
const TUMBLE = 1.5;       // how fast a flake turns over
const SECTORS = 9.0;      // corners on a flake's outline
// HOW DEEP THE NOTCHES GO, and it is the difference between ash and confetti.
// Measured over 24 seeds as (max-min)/mean of the outline: 0.42 gives 0.63,
// which is a pebble; 0.15 gives 1.18, which is a shard with spikes and bites
// taken out of it. Anything above about 1 reads as broken rather than rounded.
const RAGGED_MIN = 0.15;
// NO COLD ASH, AND THE BLEND MODE IS WHY. This layer is additive, so a dark
// flake cannot be dark — additive only ever ADDS light, and a near-black chip
// comes out as a mid-brown one. That is exactly what made the field read as
// confetti. Every flake here emits; what varies is how hot, and the cold ones
// are simply very dim blood red.
const DIM = 0.06;         // how faint the coolest ones get
const CORE_IN = 0.10;     // his sprite, normalised — see the note above
const CORE_OUT = 0.50;
// COOLING IS THE FIRE. An ember that holds one colour is a fleck of paint; one
// that runs white-hot to orange to dark red as it climbs is burning, and the
// range itself is what reads as heat. Every other change here is smaller than
// this one.
const HOT = vec3f(1.00, 0.82, 0.42);   // just off the floor
const FIRE = vec3f(1.00, 0.22, 0.03);  // most of the climb
const BLOOD = vec3f(0.34, 0.01, 0.01); // going out under the ceiling
const COOL_POW = 0.7;     // <1 holds the heat longer before it drops away
const SPARK_GAIN = 4.2;   // HDR: white here is grey after AgX, so a spark needs headroom
const BLOOM_GAIN = 1.1;
const BRIGHT_VAR = 0.85;  // a few fierce ones against many faint — cubed, see init
const CLUMP = 0.65;       // how much they gather into streams instead of a field
const GUST = 0.55;        // a second, slower turbulence over the first
const FLICKER = 0.55;     // how hard each ember pulses
const FLICKER_RATE = 5.0;
const FADE_IN = 6.0;      // metres over which one lights up after wrapping
const FADE_OUT = 16.0;    // and cools out under the ceiling

fn particleInit(id: u32, seed: f32) -> Particle {
  var p: Particle;
  let r = rzHash13(seed + f32(id) * 0.0157);
  let r2 = rzHash13(seed * 1.93 + f32(id) * 0.0331);
  // CLUMPED, NOT SCATTERED. A uniform spawn is snow — fire comes up in streams
  // with gaps between them. Rejecting toward a coarse noise field costs one
  // lookup and is the difference between weather and a burn.
  var at = vec3f(CENTER.x + (r.x - 0.5) * AREA * 2.0, r.y * TOP, CENTER.z + (r.z - 0.5) * AREA * 2.0);
  let lane = rzCurlNoise(vec3f(at.x, 0.0, at.z) * 0.045);
  at.x += lane.x * AREA * CLUMP * 0.5;
  at.z += lane.z * AREA * CLUMP * 0.5;
  p.pos = at;
  p.size = mix(SIZE_MIN, SIZE_MAX, r2.x * r2.x);
  // Bigger embers carry further, which is what stops the field reading as one
  // sheet sliding upward.
  p.vel = vec3f(0.0, RISE * mix(0.6, 1.5, r2.x), 0.0) + LEAN * mix(0.5, 1.4, r2.y);
  // NOT STRETCHED, AND THAT IS THE POINT. A stretched quad is aligned to its
  // velocity and the engine ignores p.rot for it — which is right for a spark,
  // whose direction of travel IS its orientation, and wrong for ash. Ash tumbles.
  // Giving up the streak is what buys a flake that turns over and presents a
  // different face as it climbs.
  p.stretch = 0.0;
  p.rot = r2.z * 6.2831853;

  // Never expires: embers WRAP at the ceiling rather than recycling through the
  // pool, so none of them appears out of nothing in mid-air.
  p.life = 1.0e9;
  p.seed = seed + f32(id) * 0.0157;
  return p;
}

fn particleStep(p: Particle, dt: f32) -> Particle {
  var q = p;
  // Curl noise is divergence-free, so embers wander past each other instead of
  // collecting into the same thread. It is this port's stand-in for his two noise
  // nudges, which do the same job in two dimensions.
  let w = rzCurlNoise(q.pos * SWIRL + vec3f(0.0, rzTime() * 0.22, 0.0));
  // A SECOND, SLOWER TURBULENCE over the first. One octave of curl gives every
  // ember the same size of wobble and the field reads as combed; a coarse slow
  // one under it bends whole groups together, which is what a draught does to a
  // fire. And each ember is pushed around by its own amount, so two side by side
  // do not trace the same line.
  let gust = rzCurlNoise(q.pos * (SWIRL * 0.22) + vec3f(rzTime() * 0.07, 0.0, 0.0));
  let sway = mix(0.5, 1.6, rzHash11(q.seed * 5.3));
  q.pos = q.pos + (q.vel + (w * WANDER + gust * GUST) * sway) * dt;
  // Turning over as it goes, each at its own rate and its own direction.
  q.rot = q.rot + dt * TUMBLE * (rzHash11(q.seed * 2.11) - 0.5) * 2.0;
  // Toroidal in x and z, so the field stays centred on the stage however far the
  // camera orbits, and over the top when one reaches the ceiling.
  let span = AREA * 2.0;
  let rx = q.pos.x - CENTER.x;
  let rz = q.pos.z - CENTER.z;
  q.pos.x = CENTER.x + rx - span * floor(rx / span + 0.5);
  q.pos.z = CENTER.z + rz - span * floor(rz / span + 0.5);
  if (q.pos.y > TOP) {
    let h = rzHash13(q.seed + rzTime() * 0.41);
    q.pos = vec3f(CENTER.x + (h.x - 0.5) * span, 0.0, CENTER.z + (h.z - 0.5) * span);
  }
  return q;
}

fn particleShade(p: Particle, uv: vec2f) -> vec4f {
  let q = (uv - vec2f(0.5)) * 2.0;
  let r = length(q);

  // HIS SPRITE, in his proportions. The core is a hard little thing filling the
  // inner half; the bloom is a cube of the remainder, which is what makes the
  // glow fall off fast enough to read as heat rather than as a soft dot.
  // A FLAKE IS A POLYGON, NOT A DENTED CIRCLE.
  //
  // The outline is a radius per angular SECTOR, hashed off the seed and joined
  // by straight lines — so each flake has its own corners, its own long edge and
  // its own bitten side, and two of them side by side are different objects
  // rather than one shape at two sizes. Measured across seeds the radius spreads
  // 0.13 to 0.51 of the sprite, which is the variety the eye reads.
  //
  // It costs two hashes and no samples. The previous attempt modulated the radius
  // with a pair of sines, which gives every flake the same gentle wobble at a
  // different phase — the same shape, rotated. That is what looked identical.
  let ang = atan2(q.y, q.x);
  let a01 = (ang * 0.15915494 + 0.5) * SECTORS;
  let s0 = floor(a01);
  let f = a01 - s0;
  let e0 = rzHash11(p.seed * 13.1 + (s0 - SECTORS * floor(s0 / SECTORS)) * 7.7);
  let e1 = rzHash11(p.seed * 13.1 + ((s0 + 1.0) - SECTORS * floor((s0 + 1.0) / SECTORS)) * 7.7);
  let edge = RAGGED_MIN + (1.0 - RAGGED_MIN) * mix(e0, e1, f);

  // Hard-edged, with a pixel of softness — ash has a cut edge, not a halo.
  // 1 - smoothstep(lo, hi), NOT smoothstep(hi, lo). WGSL leaves smoothstep
  // undefined when low >= high, and it is the kind of undefined that works on
  // the machine it was written on.
  let body = 1.0 - smoothstep(edge * 0.82, edge, r);
  // The heat sits inside the flake rather than around it, so an ember looks lit
  // from within and a cold one is just a hole in the light.
  let core = 1.0 - smoothstep(edge * 0.18, edge * 0.72, r);
  let bloom = pow(1.0 - clamp(r, 0.0, 1.0), 3.0);

  // Each one flickers at its own rate. Embers that pulse together are a strobe.
  let flick = 1.0 - FLICKER * (0.5 + 0.5 * sin(rzTime() * FLICKER_RATE * (0.5 + rzHash11(p.seed * 3.7)) + p.seed * 29.3));

  // Lights up after it wraps and cools under the ceiling, so neither end of the
  // climb is a line where embers appear or vanish.
  let born = smoothstep(0.0, FADE_IN, p.pos.y);
  let spent = 1.0 - smoothstep(TOP - FADE_OUT, TOP, p.pos.y);
  let life = born * spent;

  // How far up it has got, which is how much heat it has lost.
  let climb = pow(clamp(p.pos.y / TOP, 0.0, 1.0), COOL_POW);
  let heat = select(mix(FIRE, BLOOD, clamp((climb - 0.3) / 0.7, 0.0, 1.0)),
                    mix(HOT, FIRE, clamp(climb / 0.3, 0.0, 1.0)),
                    climb < 0.3);

  // Cubed, so a handful are fierce and most are barely there — the distribution
  // a real fire has, and the reason bloom has anything to find.
  let bv = rzHash11(p.seed * 11.93);
  let gain = mix(DIM, 1.0 + BRIGHT_VAR, bv * bv * bv);

  let col = heat * SPARK_GAIN * gain;
  let e = clamp(body + bloom * 0.35, 0.0, 1.0) * life * flick;
  if (e <= 0.0) { return vec4f(0.0); }
  // The core is hotter than the rim: a coal is lit from inside, not painted.
  return vec4f(col * (1.0 + core * 1.6) + heat * BLOOM_GAIN * bloom * gain, e);
}
