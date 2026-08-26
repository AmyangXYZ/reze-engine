#layer additive
#anchor センター trail

// Tunables — edit and ⌘⏎.
const COLOR_L = vec3f(0.231, 0.510, 0.965);  // stage left — blue-500 #3b82f6
const COLOR_T = vec3f(1.000, 1.000, 1.000);  // overhead — white, the key
const COLOR_R = vec3f(0.937, 0.267, 0.267);  // stage right — red-500 #ef4444
const HEIGHT = 42.0;     // world units above the floor the lamps hang
const SPREAD = 25.0;     // how far left and right of centre they sit — this
                         // against HEIGHT is the rake angle of the outer beams
const FRONT = 7.0;       // ...and how far toward the audience
const TOP_R = 2.4;       // beam radius at the lamp
const BOT_R = 4.8;       // ...and where it lands — close to TOP_R keeps the
                         // shaft near parallel instead of a wide fan
const EDGE = 0.09;       // softness of the shaft's rim, as a fraction of BOT_R
const DENSITY = 1.15;    // how much haze the beam lights up — carries the
                         // shaft's brightness, since coverage IS its intensity
const POOL_R = 3.6;      // bright pool on the floor — inside the beam's landing
const POOL = 0.26;
const FLATNESS = 0.35;   // world units of ground the pool may lie on
const LAG = 0.30;        // seconds the aim trails the dancer
const AIM_STRIDE = 4;    // trail samples skipped when averaging the aim
const STEPS = 20;        // march steps. Twenty against the beam's own cylinder
                         // samples DENSER than thirty-four did against the old
                         // bounding sphere, which was mostly empty air
const FAR_CLAMP = 400.0;
const LIGHT_VIVID = 1.5;  // saturation of the CAST light vs the shaft's own
                          // colour — see vivid(). 1.0 casts exactly what the
                          // beam looks like, which reads white on a lit surface

// Follow-spots that track the cast.
//
// The worked example for rzSubject().root: the beams aim at the FLOOR under each
// character, so they follow a dancer across the stage the way an operator would,
// and land where the feet are rather than around the waist.
//
// The lamps are placed relative to the CAMERA — left, overhead, right, all in
// front of the character — so they read as stage lighting from wherever you
// orbit. Pinned to world axes they end up behind the subject half the time,
// which is backlight, not a follow-spot.
//
// The beam is volumetric, marched through the air, which is why it is occluded
// correctly: the march ends at the scene's own depth, so a shaft passes in front
// of a character and stops behind her.
fn hash21(p: vec2f) -> f32 {
  return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

/**
 * Where lamp i hangs: side is -1, 0, +1 for left, overhead, right.
 *
 * WORLD-FIXED, fully: rigged to the stage's own axes, no camera term and no
 * sway. Earlier versions placed the rig around the camera's side of the stage
 * so it always read left/overhead/right — but that swings the lamp heads on
 * every orbit and every camera cut, and a fixture that moves is no fixture.
 * Only the AIM follows anything now; orbit behind the stage and you are simply
 * seeing the rig from behind, which is what a rig does.
 */
fn lampPos(side: f32, floorY: f32) -> vec3f {
  return vec3f(side * SPREAD, floorY + HEIGHT, FRONT);
}

/**
 * Where the beams point: the dancer, but lagged.
 *
 * An operator does not snap a follow-spot onto a moving target and neither
 * should this — an aim locked exactly to the hips twitches on every step and
 * makes the beam feel weightless. This is the same damping the follow camera
 * uses, done differently because a shader keeps nothing between frames: an
 * exponentially weighted average over the センター trail IS a low-pass filter,
 * and the trail is a position history we already have.
 *
 * LAG is the time constant. Larger drifts more lazily behind her; zero would be
 * the twitchy lock this exists to avoid.
 */
fn aimPoint(subject: i32, root: vec3f) -> vec3f {
  let n = rzTrailCount(subject, 0);
  if (n < 2) { return root; }
  var sum = vec3f(0.0);
  var wsum = 0.0;
  for (var i = 0; i < n; i = i + AIM_STRIDE) {
    let s = rzTrail(subject, 0, i);
    // Past two time constants the weight is already under a seventh — the rest
    // of the trail cannot move the average enough to be worth walking.
    if (s.w > LAG * 2.0) { break; }
    let we = exp(-s.w / LAG);
    sum += s.xyz * we;
    wsum += we;
  }
  if (wsum < 1e-4) { return root; }
  let p = sum / wsum;
  // Horizontal only: the aim follows where she IS on the stage, and the beam
  // still lands on the floor rather than chasing her hips up and down.
  return vec3f(p.x, root.y, p.z);
}

fn beamFor(side: f32, aim: vec3f, origin: vec3f, ray: vec3f, tMax: f32, time: f32, jitter: f32) -> f32 {
  let apex = lampPos(side, aim.y);
  let toFloor = aim - apex;
  let len = length(toFloor);
  if (len < 1e-3) { return 0.0; }
  let dir = toFloor / len;

  // March the BEAM's true envelope: its cylinder, cut by the lamp and floor
  // planes. Stepping uniformly from the lens to the scene gives every pixel a
  // different interval; and the earlier fix — a bounding SPHERE around the whole
  // beam — was still five times fatter than the beam it contained (radius
  // len/2 + BOT_R against a shaft at most BOT_R wide), so most of a fixed step
  // budget landed in empty air and failed the d < r test below. Against the
  // tight envelope, twenty steps hold the sampling density thirty-four needed
  // inside the sphere, and the jitter hides the rest.
  let w0 = origin - apex;
  let rd = dot(ray, dir);
  let wd = dot(w0, dir);
  var t0 = 0.0;
  var t1 = tMax;
  if (abs(rd) > 1e-4) {
    let ta = (0.0 - wd) / rd;
    let tb = (len - wd) / rd;
    t0 = max(t0, min(ta, tb));
    t1 = min(t1, max(ta, tb));
  } else if (wd < 0.0 || wd > len) {
    return 0.0;
  }
  let aq = 1.0 - rd * rd;
  let bq = dot(w0, ray) - wd * rd;
  let cq = dot(w0, w0) - wd * wd - BOT_R * BOT_R;
  if (aq > 1e-5) {
    let disc = bq * bq - aq * cq;
    if (disc < 0.0) { return 0.0; }
    let sq = sqrt(disc);
    t0 = max(t0, (-bq - sq) / aq);
    t1 = min(t1, (-bq + sq) / aq);
  } else if (cq > 0.0) {
    return 0.0;
  }
  if (t1 <= t0) { return 0.0; }

  var acc = 0.0;
  let dt = (t1 - t0) / f32(STEPS);
  for (var s = 0; s < STEPS; s++) {
    // JITTERED start. Every pixel sampling at the same offsets makes each step
    // deposit a visible disc, and a marched cone comes out as a stack of
    // spheres — worse the harder the beam's edge is, which is exactly when you
    // want it. Offsetting each pixel by a hash turns that banding into fine
    // noise, which is what haze looks like anyway.
    let p = origin + ray * (t0 + dt * (f32(s) + jitter));
    let along = clamp(dot(p - apex, dir) / len, 0.0, 1.0);
    let r = mix(TOP_R, BOT_R, along);
    let d = distance(p, apex + dir * (along * len));
    if (d < r) {
      // A searchlight has a defined SHAFT: flat across its width, falling off
      // only at the rim. The rim is a WORLD width, not a fraction of the local
      // radius — as a fraction it widens with the cone, leaving the beam sharp
      // at the lamp and smeared where it lands, which is the end you look at.
      let soft = max(EDGE * BOT_R, 1e-3);
      acc += smoothstep(r, r - soft, d) * mix(0.55, 1.0, along) * dt;
    }
  }
  return acc;
}

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let origin = rzCameraPos();
  let dir = normalize(ray);
  // A beam is light in the air BETWEEN the lens and whatever was drawn, so
  // stopping at depth is what occludes it.
  let tMax = min(depth, FAR_CLAMP);
  if (tMax <= 0.01) { return vec4f(0.0); }
  let jitter = hash21(uv * rzResolution());

  // Composited OVER one another, not summed.
  //
  // Three beams converge on the dancer by design, and summing their coverage
  // put the total past 1 exactly where they meet — so the most interesting part
  // of the shot clipped to flat white and lost both the shafts and her legs.
  // Over-compositing is how overlapping lights actually behave: each adds what
  // is left rather than its whole self, so the total approaches 1 and never
  // slams into it.
  var rgbPm = vec3f(0.0);   // premultiplied
  var alpha = 0.0;
  for (var c = 0; c < rzSubjectCount(); c++) {
    let subj = rzSubject(c);
    if (!subj.valid) { continue; }
    let aim = aimPoint(c, subj.root);

    for (var i = 0; i < 3; i++) {
      let side = f32(i) - 1.0;   // -1 left, 0 overhead, +1 right
      var col = COLOR_T;
      if (i == 0) { col = COLOR_L; }
      if (i == 2) { col = COLOR_R; }
      let beam = clamp(beamFor(side, aim, origin, dir, tMax, time, jitter) * DENSITY * 0.06, 0.0, 1.0);
      if (beam > 0.001) {
        rgbPm += col * beam * (1.0 - alpha);
        alpha += beam * (1.0 - alpha);
      }
      // Saturated is saturated: with alpha this close to 1, a third beam's whole
      // march could not move the pixel a code value.
      if (alpha > 0.985) { break; }
    }

    // The pool where the light lands. Read off the pixel's own world position,
    // so it lies on whatever the floor actually is and bends over it.
    if (depth < FAR_CLAMP) {
      let p = rzWorldPos(dir, depth);
      if (abs(p.y - subj.root.y) < FLATNESS) {
        let d = distance(p.xz, aim.xz);
        if (d < POOL_R) {
          let x = 1.0 - d / POOL_R;
          let pool = clamp(x * x * POOL, 0.0, 1.0);
          rgbPm += COLOR_T * pool * (1.0 - alpha);
          alpha += pool * (1.0 - alpha);
        }
      }
    }
  }

  let a = clamp(alpha, 0.0, 1.0);
  if (a <= 0.002) { return vec4f(0.0); }
  // Un-premultiply for the straight alpha the mount wants.
  //
  // The tint is HELD now. This used to lerp toward white from alpha 0.8, which
  // whitened the core of every shaft — the exact place a coloured beam should
  // read most strongly, and most of why three lamps looked like one. Neon is
  // saturated where it is brightest; only the last sliver before full coverage
  // blooms out, and at alpha 1 that is a couple of per cent rather than a fifth.
  let lit = rgbPm / max(alpha, 1e-4);
  return vec4f(mix(lit, vec3f(1.0), smoothstep(0.96, 1.7, alpha)), a);
}

/** Saturation about the colour's own luminance. k = 1 is unchanged; above it
 *  pushes away from grey while holding hue and rough brightness. Clamped at
 *  zero because k > 1 drives the weakest channel negative, and a negative
 *  channel in an ADDITIVE light darkens what it lands on. */
fn vivid(c: vec3f, k: f32) -> vec3f {
  let luma = dot(c, vec3f(0.2126, 0.7152, 0.0722));
  return max(mix(vec3f(luma), c, k), vec3f(0.0));
}

// ── The light the beams actually cast ────────────────────────────────────────
//
// Until this mount existed a beam could only PAINT light: the shaft was drawn
// over the finished frame, so the pool on the floor was a bright patch that lit
// nothing, and the dancer standing in it was shaded by the sun alone. Three
// obvious follow-spots converging on someone who does not react to them is the
// tell that the whole effect is a decal.
//
// One light per beam, placed where that beam LANDS rather than where the
// fixture hangs. The lamp is 42 units up and its pool is on the floor; putting
// the light at the head would wash the whole stage evenly from above, which is
// the one thing a follow-spot does not do. At the landing point the falloff
// does the work the beam's cone is already doing visually.
//
// The aim is the same lagged aim the shafts use — literally the same function,
// so the light cannot drift away from the shaft supposed to be delivering it.
#lights 3
fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  // The SAME palette as the shafts, pushed. Haze reads pale because it is
  // scattered light seen edge-on; the same lamp landing on a dress has to read
  // by its hue or three spots converging look like one white one — which is
  // what they did. Deriving it rather than keeping a second set of constants
  // means editing a beam's colour moves the light it casts with it.
  //
  // AgX is the other half of why this is needed: it desaturates as it
  // compresses, so a colour that survives the tone map has to start further out
  // than it would in a linear render.
  l.color = vivid(select(select(COLOR_L, COLOR_T, i == 1u), COLOR_R, i == 2u), LIGHT_VIVID);
  // Sides match beamFor's: -1 stage left, 0 overhead, +1 stage right.
  let side = f32(i32(i) - 1);

  // The first subject is who the rig follows, exactly as the beams do.
  let subj = rzSubject(0);
  if (!subj.valid) {
    // Nobody on stage. The fixtures still hang, but there is no pool to light,
    // so this slot goes dark rather than parking a lamp at the world origin.
    l.pos = lampPos(side, 0.0);
    l.intensity = 0.0;
    l.radius = 1.0;
    return l;
  }
  let aim = aimPoint(0, subj.root);
  // Lifted off the floor by the pool's own radius. A light exactly ON the plane
  // it lights grazes it — the floor's normal is perpendicular to the direction
  // to the light, N·L is zero, and the pool would be invisible from directly
  // above. Standing it off makes the brightest point the middle of the pool,
  // which is where the beam is brightest too.
  l.pos = vec3f(aim.x + side * POOL_R * 0.5, aim.y + POOL_R, aim.z);
  // The overhead lamp is the key of the three and the rakes are fill — the same
  // relationship the drawn beams have, where the white one reads as strongest.
  l.intensity = select(1.6, 2.4, i == 1u);
  // Reaches a little past the pool it paints, so someone standing in it is lit
  // to about the waist rather than only at the ankles.
  l.radius = POOL_R * 4.5;
  return l;
}
