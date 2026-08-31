#particles 24000
#blend additive
#bloom

// Floating Stars — motes of light drifting around the cast.
//
// The companion to Shining Stars rather than a replacement: that one is the SKY,
// a field with no position, and this one is IN the scene. Particles draw in HDR
// before tone mapping, so these can sit several times brighter than paper and
// bloom, and they are depth-tested — a mote passes behind her shoulder and in
// front of her hand in the same frame.
//
// The volume is THE STAGE, the same box Snow and Rain fall through, and the
// count is in their range: a scattering of motes reads as dirt on the lens, and
// only density reads as weather. It is a cylinder rather than a sphere because a
// skybox-sized sphere puts a few of its stars right against the lens, and a mote
// a metre from the camera is a soft grey smear over everything. Sizes are fixed
// and small for the same reason.
//
// Gathered gently inward, so the figure stands in the thick of it rather than in
// a clearing.
//
// They rise slowly and wander. Nothing about them is fast: the drift is there so
// the field is alive between twinkles, not to read as motion.

// Tunables.
const CENTER = vec3f(0.0, 0.0, 0.0);  // the stage they gather around
const INNER = 1.5;        // clear of the body
const OUTER = 50.0;       // and how far out they reach
const GATHER = 1.3;       // >1 crowds them toward the cast; 1 is an even spread
const FLOOR = 0.0;
const TOP = 58.0;         // they rise to here, then wrap — well above the frame
const RISE = 0.055;       // metres per second — barely
const WANDER = 0.11;      // sideways drift
const SWIRL = 0.07;       // how tightly that drift curls
const SIZE_MIN = 0.045;
const SIZE_MAX = 0.18;
const WARM = vec3f(1.0, 0.95, 0.86);
const COOL = vec3f(0.87, 0.93, 1.0);
const INTENSITY = 6.5;    // HDR brightness — see the note below
const BRIGHT_VAR = 0.8;   // the faint many against the bright few
const CORE = 0.17;        // core radius, in quad halves
const HALO = 0.16;      // only ever seen at the peak of a flash
const SPIKE = 0.46;     // likewise — the flare IS the shine
const SPIKE_THIN = 0.08;
const TWINKLE = 0.42;   // how often a mote comes round to its flash
const FLASH = 0.7;      // and how much of one it is when it does

// A NOTE ON BRIGHTNESS. Particles draw in HDR before tone mapping, so white here
// is not the white that comes out: vec3f(1.0) goes through AgX and lands as grey.
// INTENSITY is what puts a mote above the paper it is drawn on, and above the
// bloom threshold. It is also why these read on a LIGHT background, where an
// additive glow at 1.0 would simply vanish.

fn particleInit(id: u32, seed: f32) -> Particle {
  var p: Particle;
  let a = rzHash13(seed + f32(id) * 0.0193);
  let b = rzHash13(seed * 1.61 + f32(id) * 0.0377);

  // Around the cast, gathered inward. sqrt alone spreads evenly over the disc,
  // which puts most of them at the rim where nobody is looking; the extra power
  // pulls them back toward the figure.
  let ang = a.x * 6.2831853;
  let radial = pow(sqrt(a.y), GATHER);
  let radius = mix(INNER, OUTER, radial);
  p.pos = CENTER + vec3f(cos(ang) * radius, mix(FLOOR, TOP, a.z), sin(ang) * radius);

  p.size = mix(SIZE_MIN, SIZE_MAX, b.x * b.x);
  p.vel = vec3f(0.0, RISE * mix(0.5, 1.5, b.y), 0.0);
  // Never expires: motes WRAP at the ceiling rather than recycling through the
  // pool, so none of them appears out of nothing in mid-air.
  p.life = 1.0e9;
  p.rot = b.z * 6.2831853;
  p.seed = seed + f32(id) * 0.0193;
  return p;
}

fn particleStep(p: Particle, dt: f32) -> Particle {
  var q = p;
  // Curl noise is divergence-free, so motes wander past each other instead of
  // collecting into the same thread the way a plain sine drift does.
  let w = rzCurlNoise(q.pos * SWIRL + vec3f(0.0, rzTime() * 0.07, 0.0));
  q.pos = q.pos + (q.vel + w * WANDER) * dt;
  // Over the top and back to the floor, on a freshly drawn ring so the same mote
  // never rises up the same line twice.
  if (q.pos.y > TOP) {
    let h = rzHash13(q.seed + rzTime() * 0.31);
    let ang = h.x * 6.2831853;
    let radius = mix(INNER, OUTER, pow(sqrt(h.y), GATHER));
    q.pos = vec3f(CENTER.x + cos(ang) * radius, FLOOR, CENTER.z + sin(ang) * radius);
  }
  return q;
}

fn particleShade(p: Particle, uv: vec2f) -> vec4f {
  let q = (uv - vec2f(0.5)) * 2.0;
  let r = length(q);
  let ax = abs(q);

  // Every piece reaches EXACTLY zero at the quad's edge — rzFalloff has compact
  // support. A glow that merely gets small has to be cut off somewhere, and the
  // cut is a square edge that bloom then draws a box around.
  //
  // Squared rather than powed: pow is undefined for a negative base, and x*x
  // says the same thing about a value that reaches zero.
  let cf = rzFalloff(r, CORE);
  let core = cf * cf;
  let halo = rzFalloff(r, 1.0) * HALO;
  let sx = rzFalloff(ax.x, 1.0) * rzFalloff(ax.y, SPIKE_THIN);
  let sy = rzFalloff(ax.y, 1.0) * rzFalloff(ax.x, SPIKE_THIN);
  let flare = (sx + sy) * SPIKE;

  // A sharp pulse, not a swell. Raised to a sixth power the cycle sits near zero
  // for most of its length and flares briefly, which is what makes the shine an
  // EVENT rather than a state — and raising the power makes it RARER without
  // making it faster, which is the difference between a sky and a switchboard.
  let rate = 0.35 + 1.55 * rzHash11(p.seed * 3.41);
  let sw = 0.5 + 0.5 * sin(rzTime() * rate * TWINKLE * 6.2831853 + p.seed * 37.7);
  let sw2 = sw * sw;
  let flash = sw2 * sw2 * sw2 * FLASH;

  // THE POINT IS ALWAYS THERE; THE GLOW ONLY ARRIVES WITH THE FLASH.
  //
  // Halo and flare are gated on the pulse, the core is not. A halo that burns
  // constantly is what turns a field of stars into a field of blooming blobs:
  // every mote wears a permanent soft disc, and the twinkle is lost inside it.
  // Gated, a mote is a fine point until its moment and a burst of light during
  // it — and since bloom keys off brightness, it only blooms then too.
  //
  // SHAPE in alpha, BRIGHTNESS in colour: the fragment is premultiplied
  // (rgb * a), so a profile carried in both would be squared.
  let steady = mix(0.55, 1.0, rzHash11(p.seed * 2.29));
  let e = clamp(core * steady + (halo + flare) * flash, 0.0, 1.0);

  let warmth = rzHash11(p.seed * 5.09);
  let tint = mix(COOL, WARM, warmth);
  let bv = rzHash11(p.seed * 11.71);
  let bright = mix(0.4, 1.0 + BRIGHT_VAR, bv * bv * bv);
  return vec4f(tint * INTENSITY * bright, e);
}
