// @layer additive
// @anchor 左足ＩＫ trail
// @anchor 右足ＩＫ trail
// @anchor 左足首 trail
// @anchor 右足首 trail

// Tunables — edit and ⌘⏎.
// Colour comes from INTENSITY, not from a single tint. A disc filled with one
// saturated colour and then alpha-blended over a dark floor lands somewhere in
// the middle and reads muddy — which is how a red print came out dark orange.
// Light goes WHITE where it is strong and keeps its colour only as it falls off.
const HOT_COLOR = vec3f(0.88, 0.99, 1.0);    // white-cyan, centre of a fresh print
const GLOW_COLOR = vec3f(0.14, 0.76, 1.0);   // cyan, the body of it
const RIM_COLOR = vec3f(0.05, 0.30, 0.95);   // deep blue, at the edge
const RADIUS = 1.25;     // world units — a mark a foot sits INSIDE, not one it
                         // covers, or the print reads as a dot under the shoe.
                         // Sized against THIS cast: an 18-unit model has a foot
                         // a bit over a unit long, and the 2.4 this started at
                         // drew a pool wider than her stride, so both prints
                         // merged into one puddle she stood in the middle of.
const EDGE = 0.5;        // soft rim as a fraction of the radius
// How much the foot must have dropped for a sample to read as a landing. A
// RELATIVE test, never an absolute height: 足ＩＫ sits at the ANKLE, already a
// hand's width up on a bare foot and higher in heels, so any constant measured
// from the floor is wrong on some rig.
const TOUCH_EPS = 0.03;
const FADE = 2.0;        // seconds a print takes to burn out
const STANCE_HOLD = 0.85;// how bright a print stays while the foot is STILL on
                         // it, regardless of that timer — see the use site
// How much the score brightens the floor. The prints and the falling notes are
// the same music seen twice, and having the ground answer the same beat is what
// makes them one scene instead of two effects that happen to share a frame.
// ADDED, never multiplied: with no score loaded this is zero and the prints are
// exactly what they always were, rather than going dark.
const PULSE = 0.55;
// What rises off a print as it burns out. Motes, not a column: a column is one
// solid shape and it reads as a shape — two rims, a lit tube between them. A
// print going out like a candle is a disc that dims while sparks leave it, and
// sparks are what carry that. They are placed in SCREEN space around the
// projected print, so seven of them cost two projections rather than seven.
// The plume scales with the mark, or a half-size print throws a full-size
// spray and the sparks read as belonging to something else.
const EMBERS = 80;        // motes rising off each print
const EMBER_LIFE = 1.60;  // seconds one lives
const EMBER_RISE = 2.4;   // world units per second it climbs
const EMBER_VARY = 0.70;  // how much that speed differs mote to mote
const EMBER_SPREAD = 0.10;// how much further OUT it drifts as it climbs. Small
                          // on purpose: anything generous here and the sparks
                          // leave from outside the mark they belong to
const EMBER_R = 0.075;    // its radius, the same
const EMBER_CORE = 0.72;  // fraction of that which is SOLID before it falls off
const EMBER_SIZES = 0.6;  // how much THAT differs mote to mote
const EMBER_I = 1.7;      // how bright, against the mark on the floor
const FLOOR_BIAS = 0.35; // world units of slack before the scene is treated as
                         // standing in FRONT of a mark. A drawn ground sits on
                         // the same plane as the marks, so without this it
                         // occludes them by a rounding error and they flicker.
const ELONGATE = 1.0;    // 1 = round; above 1 stretches along the step
const STRENGTH = 2.3;
const FAR_CLAMP = 400.0;

// Marks where the feet met the ground, left behind as the character walks away.
//
// The worked example for rzSubject().root and for reading a trail in WORLD space
// rather than on screen. It was called Footfalls and only lit the foot's current
// position, which followed the foot around like a spotlight — a foot's location,
// not a footprint.
//
// A print is a place and a TIME: born where the foot met the floor, then burning
// down while the foot walks on. rzTrail carries exactly that — every sample is a
// position plus how many seconds ago it was there — so the samples whose height
// is near the floor ARE the contacts, and their age is how long ago each one
// happened. That is a real print that stays behind, out of data the engine keeps
// for a hand ribbon.
//
// It is still an approximation of contact rather than the thing itself: engine
// contact events would give the exact frame a foot landed, its facing and how
// hard, where this infers it from height. Good enough to look right, and the
// comment is here so the next person knows which part is the stand-in.
//
// The height test is also why the band has to be tight: a shoe's toe is a few
// centimetres off the ground, so a generous band marks the toe as readily as the
// floor under it. Position alone cannot tell them apart — they are in the same
// place. Knowing which pixels are the CHARACTER would, and that needs the subject
// mask the engine does not expose yet.
/** Returns the mark on the FLOOR in x, and the light standing on it in y. */
/** Two uncorrelated 0..1 values from a 2D key, WITHOUT a transcendental.
 *
 *  This was `fract(sin(q) * 43758.5453)`, the Shadertoy standard — two sines
 *  per call, two calls per mote, eighty motes per print. Measured at 1.7ms of
 *  the effect's 3.6, which is a lot of frame to spend on randomness nobody can
 *  see the shape of. Dave Hoskins' hash22 is the same job in multiplies and
 *  fracts, and a spark that lands a pixel elsewhere is not a spark anyone can
 *  tell was moved. */
fn fpHash(p: vec2f) -> vec2f {
  var q = fract(vec3f(p.x, p.y, p.x) * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yzx + 33.33);
  return fract(vec2f((q.x + q.y) * q.z, (q.x + q.z) * q.y));
}

/** Returns the mark on the FLOOR in x, and what rises off it in y. */
fn markFor(subject: i32, slot: i32, alt: i32, p: vec3f, sp: vec2f, aspect: f32,
           ro: vec3f, ray: vec3f, depth: f32) -> vec2f {
  var n = rzTrailCount(subject, slot);
  var pick = slot;
  if (n < 2) {
    n = rzTrailCount(subject, alt);
    pick = alt;
  }
  if (n < 4) { return vec2f(0.0); }

  // Cull in world space: the whole walk fits in a circle, and most of the floor
  // is nowhere near it.
  var mid = vec2f(0.0);
  for (var k = 0; k < 4; k++) {
    mid += rzTrail(subject, pick, min(k * n / 3, n - 1)).xz;
  }
  mid *= 0.25;
  var reach = 0.0;
  for (var k = 0; k < 4; k++) {
    reach = max(reach, distance(rzTrail(subject, pick, min(k * n / 3, n - 1)).xz, mid));
  }
  // Cull on the RAY, not on the surface point.
  //
  // The floor test alone was right while this only drew on the floor: the pixel
  // was the mark. A pillar stands in the air, so a ray that passes straight
  // through one on its way to a wall fifty metres behind has a surface point
  // nowhere near the walk, and culling on it would erase the very pixels the
  // pillar lives in. Closest approach to a vertical axis through the middle of
  // the walk covers both — it is exactly the surface test when the ray ends on
  // the floor there, and looser everywhere the pillar needs it to be.
  let denom = max(1.0 - ray.y * ray.y, 1e-4);
  let w0 = ro - vec3f(mid.x, 0.0, mid.y);
  let near = ro + ray * max((ray.y * w0.y - dot(ray, w0)) / denom, 0.0);
  if (distance(near.xz, mid) > reach * 1.5 + RADIUS) { return vec2f(0.0); }

  var acc = 0.0;
  var col = 0.0;
  // EVERY sample. The ring shifts by one each frame, so a given moment of the
  // path sits on an even index one frame and an odd one the next — striding
  // over it means a landing is detected every OTHER frame, and the print
  // strobes at half the framerate.
  //
  // The three reads per step look redundant — consecutive iterations overlap by
  // two — and carrying them across the loop in registers instead was MEASURED
  // SLOWER, 1.9ms to 3.6ms. They land in cache and cost almost nothing; two
  // vec4f held live across the body cost occupancy, which is the scarcer thing.
  // Left alone deliberately.
  for (var i = 1; i + 1 < n; i = i + 1) {
    let s = rzTrail(subject, pick, i);
    if (s.w > FADE) { break; }               // older than a print lasts

    // TOUCHDOWN, not "low". Index 0 is now and rising indices run backwards in
    // time, so `older` is where the foot was before this sample and `newer` is
    // where it went after. A landing is the moment it STOPS descending: it was
    // higher a moment ago, and it did not keep dropping.
    //
    // Marking every low sample instead — which is what this did — puts a print
    // every 20ms through the whole stance, forty overlapping discs that merge
    // into one blob and flicker as each new one is born a few centimetres from
    // the last. A step should leave ONE print, and the path already says
    // exactly when it happened.
    let older = rzTrail(subject, pick, i + 1);
    let newer = rzTrail(subject, pick, i - 1).y;
    if (older.y - s.y < TOUCH_EPS) { continue; }   // was not descending
    if (s.y - newer > TOUCH_EPS) { continue; }     // still descending

    // Elongate along the direction of travel: a footprint is longer than it is
    // wide, and the step direction is the only orientation trail data carries.
    // (The foot's actual facing would need per-contact data — see the note above.)
    let travel = s.xz - older.xz;
    var axis = vec2f(0.0, 1.0);
    if (length(travel) > 1e-4) { axis = normalize(travel); }
    let perp = vec2f(-axis.y, axis.x);
    // STILL STANDING ON IT? A print fades on a timer, which is right for one
    // she has walked away from and wrong for the one under her weight: a
    // planted foot ended up sitting on a mark two seconds into dying, so the
    // moment she was most obviously in contact with the floor was the moment
    // the floor said least about it. On a dark stage — where a shadow has
    // nothing to fall on — that print is the only thing tying her to the
    // ground, so it holds while the foot is on it and fades once she lifts off.
    //
    // The height test is against the TOUCHDOWN's own height, never a constant:
    // 足ＩＫ sits at the ankle, a hand's width up on a bare foot and higher in
    // heels, so anything measured from the floor is wrong on some rig. Compared
    // against where this same foot actually landed, it is right on all of them.
    let cur = rzTrail(subject, pick, 0);
    let onIt = select(0.0, 1.0, distance(cur.xz, s.xz) < RADIUS * 0.5 && cur.y - s.y < TOUCH_EPS * 6.0);
    let age = 1.0 - clamp(s.w / FADE, 0.0, 1.0);
    let fade = max(age * age, onIt * STANCE_HOLD);

    // The mark on the floor, BURNING DOWN: it dims and it also draws in, the
    // way a candle's pool of light does. Fading alone leaves a disc of the same
    // size going grey, which reads as a light being turned down rather than as
    // something being used up.
    let burn = RADIUS * (0.60 + 0.40 * age);
    let rel = p.xz - s.xz;
    let local = vec2f(dot(rel, axis) / ELONGATE, dot(rel, perp));
    let dist = length(local);
    if (dist <= burn) {
      // Compact support: zero at the rim, so the cull leaves no hard edge.
      acc = max(acc, (1.0 - smoothstep(burn * (1.0 - EDGE), burn, dist)) * fade);
    }

    // The light standing on it, INTEGRATED through the slab of air it occupies.
    //
    // Closest approach to the axis is the right primitive for a filled
    // cylinder, where the nearest point is also the brightest and carries the
    // whole shape. It is the wrong one for a WALL: a ray that goes straight
    // through a pillar crosses the wall twice and should come out bright, yet
    // its closest approach lands in the hollow middle and reports nothing at
    // all. Only rays that happened to graze a wall tangentially survived, which
    // is why the light arrived as a few thin streaks sitting nowhere near the
    // prints they belonged to.
    //
    // Six samples between where the ray enters the pillar's height band and
    // where it leaves — or where the scene stops it, so a leg in front cuts the
    // light rather than glowing through it.
    // The embers. Placed in SCREEN space off the projected print: one
    // projection puts the print on the frame, a second puts a world-up metre on
    // it, and from those two every mote's rise, drift and size follow without a
    // projection of its own. Rising along a projected world up rather than the
    // frame's own +y is what keeps them going up when the camera is tilted.
    // From the FLOOR, not from the sample. A trail sample is a bone, and 足ＩＫ
    // sits at the ankle — already a hand's width up on a bare foot and higher in
    // heels — so sparks launched from it start in mid-air above the mark they
    // are supposed to be leaving. The subject's root is where the ground is.
    let gy = rzSubject(subject).root.y;
    let g0 = vec3f(s.x, gy, s.z);
    let pc = rzProject(g0);
    if (pc.z > 0.0) {
      // The GROUND PLANE, as two screen vectors. Screen up alone can only place
      // a mote along a line, and the disc is a disc: to leave from anywhere on
      // it, a mote needs the print's own two ground axes projected as well.
      // Four projections per print, and every one of the hundred-odd motes
      // below is then plain arithmetic.
      let pu = rzProject(g0 + vec3f(0.0, 1.0, 0.0));
      let pa = rzProject(g0 + vec3f(axis.x, 0.0, axis.y));
      let pp = rzProject(g0 + vec3f(-axis.y, 0.0, axis.x));
      let base = vec2f(pc.x * aspect, pc.y);
      let upv = vec2f(pu.x * aspect, pu.y) - base;
      let ax2 = vec2f(pa.x * aspect, pa.y) - base;
      let pz2 = vec2f(pp.x * aspect, pp.y) - base;

      // One circle around the whole plume, before any of it is worked out.
      // At this count the motes are the entire cost of the effect, and all of
      // them are inside a patch of frame a hand's breadth across — so the
      // pixels that are not in that patch must never touch the loop at all.
      let climbMax = EMBER_RISE * (1.0 + EMBER_VARY * 0.5) * EMBER_LIFE;
      let wide = burn * (1.0 + EMBER_SPREAD) * max(length(ax2), length(pz2))
               + EMBER_R * (1.0 + EMBER_SIZES) * length(upv);
      let plumeC = base + upv * (climbMax * 0.5);
      let plumeR = length(upv) * climbMax * 0.5 + wide;
      // Hidden where the scene is NEARER than the print — a leg in front cuts
      // the sparks rather than letting them glow through it.
      let occ = 1.0 - smoothstep(pc.z - 0.4, pc.z + 0.4, depth);
      if (occ < 0.999 && dot(sp - plumeC, sp - plumeC) <= plumeR * plumeR) {
        for (var k = 0; k < EMBERS; k++) {
          let key = floor(s.x * 13.0 + s.z * 7.0);
          let h = fpHash(vec2f(f32(k) + 1.0, key));
          let g = fpHash(vec2f(key + 3.7, f32(k) * 2.3 + 5.1));
          // Staggered births across the print's life, so they leave in a
          // stream rather than all at once.
          let el = s.w - (f32(k) + h.x) / f32(EMBERS) * FADE;
          if (el < 0.0 || el > EMBER_LIFE) { continue; }
          let t = el / EMBER_LIFE;
          // WHERE ON THE DISC it left from — uniform over the area, which is
          // what the square root does. Without it they crowd the centre, and
          // the print looks like it is venting through a hole.
          let ang = h.y * 6.2831853;
          // Against the disc's LIT radius, not its geometric one. The mark
          // fades out across EDGE of itself, so the circle anyone can actually
          // see ends well inside `burn` — spawning against `burn` put the whole
          // spray out in the invisible skirt, reading as sparks leaving from a
          // ring around the print rather than from the print. Derived from EDGE
          // so it follows if the rim is ever made softer or harder.
          let lit = burn * (1.0 - EDGE * 0.65);
          let rad = sqrt(g.y) * lit * (1.0 + EMBER_SPREAD * t);
          let out = ax2 * cos(ang) + pz2 * sin(ang);
          // Its own climb rate and its own size. Without these a crowd this
          // large rises as a rank of identical dots — the variation is what
          // turns a hundred motes into a spray rather than a row.
          let climb = EMBER_RISE * (1.0 - EMBER_VARY * 0.5 + EMBER_VARY * g.x);
          let q = base + out * rad + upv * (climb * el);
          let r = EMBER_R * (1.0 - EMBER_SIZES * 0.5 + EMBER_SIZES * fract(g.x * 7.3))
                * (1.0 - t * 0.55) * max(length(upv), 1e-5);
          // Solid to EMBER_CORE, then falling off. Ramping from the very
          // centre puts half the brightness in the outer skirt, so a mote this
          // small only ever reaches full at a single point and the field pass
          // runs at half resolution — between them, the dot upsamples into
          // nothing. A flat core survives that; a peak does not.
          let mote = 1.0 - smoothstep(r * EMBER_CORE, r, length(sp - q));
          col = max(col, mote * (1.0 - t) * (1.0 - t) * fade * EMBER_I * (1.0 - occ));
        }
      }
    }
  }
  return vec2f(acc, col);
}

/** Loudest key sounding right now, 0..1 — the floor's share of the music.
 *  Twelve samples across the score's own range rather than all 128: this is a
 *  brightness nudge, not a readout, and it runs per pixel. Zero when no score
 *  is installed, which is what keeps this effect standing on its own. */
fn scorePulse() -> f32 {
  if (rzNoteCount() == 0) { return 0.0; }
  let lo = rzPitchLow();
  let hi = rzPitchHigh();
  var e = 0.0;
  for (var k = 0; k < 12; k++) {
    e = max(e, rzKeyEnergy(round(mix(lo, hi, f32(k) / 11.0))));
  }
  return e;
}

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let ro = rzCameraPos();
  // Where the scene put a surface. Meaningless past the far clamp, and the
  // floor test below rejects it there — but the pass no longer returns early on
  // it, because a pillar over open sky is drawn on a pixel that has no surface
  // at all.
  let p = rzWorldPos(ray, min(depth, FAR_CLAMP));
  let res = rzResolution();
  let aspect = res.x / max(res.y, 1.0);
  // The pixel itself, in a square space — where the embers are measured.
  let sp = vec2f(uv.x * aspect, uv.y);

  // How far along this ray the scene actually put something, for occlusion.
  let surfT = select(1.0e9, length(p - ro), depth < FAR_CLAMP);

  var acc = 0.0;
  for (var i = 0; i < rzSubjectCount(); i++) {
    let s = rzSubject(i);
    if (!s.valid) { continue; }

    // THE FLOOR IS A PLANE, not whatever the depth buffer happens to hold.
    //
    // This read the drawn surface point and asked whether it was at floor
    // height, which works only where the floor is actually drawn. On a dark
    // stage it often is not — a scene with ground opacity 0 renders nothing
    // there and writes no depth — so every mark failed the test and vanished,
    // leaving only the embers, which are placed in SCREEN space and so hung in
    // the air near the feet instead of under them. It read as the prints being
    // offset; they were absent.
    //
    // Intersecting the ray with the subject's own ground height puts the mark
    // where the floor IS, drawn or not. Depth is still used, but only for what
    // it can answer: whether something stands in front of the mark.
    var pf = p;
    var flat = 0.0;
    if (abs(ray.y) > 1.0e-5) {
      let tf = (s.root.y - ro.y) / ray.y;
      if (tf > 0.0) {
        pf = ro + ray * tf;
        // Hidden behind a leg, but never behind the floor it lies on: the
        // tolerance is what stops a drawn ground from occluding its own marks.
        flat = select(1.0, 0.0, surfT < length(pf - ro) - FLOOR_BIAS);
      }
    }
    let a = markFor(i, 0, 2, pf, sp, aspect, ro, ray, depth);
    let b = markFor(i, 1, 3, pf, sp, aspect, ro, ray, depth);
    acc = max(acc, max(a.x, b.x) * flat);
    acc = max(acc, max(a.y, b.y));
  }
  let heat = acc * STRENGTH * (1.0 + PULSE * scorePulse());
  if (heat <= 0.004) { return vec4f(0.0); }
  // Deep blue at the rim, cyan through the body, white-hot in the middle of a
  // fresh print — and the whole thing cools as it ages, because `acc` carries
  // the age fade into the intensity this reads.
  var rgb = mix(RIM_COLOR, GLOW_COLOR, smoothstep(0.04, 0.45, heat));
  rgb = mix(rgb, HOT_COLOR, smoothstep(0.55, 1.25, heat));
  return vec4f(rgb, clamp(heat, 0.0, 1.0));
}
