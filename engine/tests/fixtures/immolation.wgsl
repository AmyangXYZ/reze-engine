// @layer additive
// @anchor 頭
// @anchor 上半身2
// @anchor 左ひじ
// @anchor 右ひじ
// @anchor 左手首
// @anchor 右手首
// @fullres

// Immolation — the upper body alight, after Elden Ring's flame incantations.
//
// The obvious way to build this is one plume per emitter: pick a dozen points
// on the body, give each a rising puff of noise, add them up. It costs a noise
// field per emitter per pixel and it still looks like a dozen separate fires.
//
// This does it the other way round. Six UPPER-BODY segments — neck, torso, both
// upper arms, both forearms — are joined from the anchors, and the distance to
// the nearest of them is a cheap silhouette field of her top half: one number
// that already knows where she is and which way her arm points. The fire is a
// single noise field, evaluated once per pixel, thresholded against it. One
// evaluation lights the whole figure, and because the field is continuous the
// flames wrap her instead of sitting in tufts.
//
// Three things make it read as fire rather than as orange fog, and the first
// two are the ones a particle system could not do at all:
//
//   DOMAIN WARP. The noise is sampled at coordinates displaced by another
//   noise. Straight fbm gives billows; warped fbm gives the curling licks and
//   hooks that flame actually makes, which is the whole texture of the
//   reference.
//
//   ANISOTROPY, twice over. The field is measured with up cheap and down
//   expensive, so it dilates upward and nothing hangs beneath her. And the
//   noise itself is sampled on a lattice stretched vertically, so its cells are
//   tall — flame structures are tall and thin, never round.
//
//   A HIGH FLOOR. Most of the field is cut away entirely. Fire is mostly gaps;
//   a threshold low enough to keep every wisp is a threshold that fills the
//   frame, which is what a soft falloff wants to do if you let it.
//
// Weighted to the upper body by height, so it erupts off the shoulders and head
// and dies out around the waist. And deliberately faint where it crosses her —
// at the silhouette you look through the whole depth of the fire at once, and
// that is where it belongs; drawn at full strength over her body it just hides
// the character it is supposed to be burning.
//
// Full resolution. Billowing noise upsamples for free and this ran at half for
// exactly that reason, but filaments do not: at half res the fine field that
// splits the tongues is reconstructed by a bilinear blend, and what should be
// a thread of flame arrives as a smear. It shows worst on the faint fire
// crossing her body, where there is no bright structure to hide it.
//
// Tunables — edit and ⌘⏎.
const SMOKE_COLOR = vec3f(0.42, 0.02, 0.01);  // the cool tips — crimson, not brown
const RED_COLOR = vec3f(0.92, 0.13, 0.02);
const ORANGE_COLOR = vec3f(1.0, 0.45, 0.05);
const HOT_COLOR = vec3f(1.0, 0.90, 0.52);     // the hearts of the tongues
const LIMB = 0.08;      // limb half-thickness, in hip heights
const REACH = 0.26;     // how far flames reach SIDEWAYS, the same
const UP_K = 0.24;      // upward distance counts this much: lower climbs higher
const DOWN_K = 3.0;     // and downward this much: higher keeps fire off the floor
const FX = 13.0;        // noise cells across a hip height
const FY = 5.5;         // and up it — fewer, so the cells are tall
const WARP = 0.60;      // how far the field displaces its own lookup
const RISE = 2.8;       // hip heights per second the tongues climb
const ROIL = 0.80;      // how much the noise breaks the falloff into tongues
const FLOOR = 0.46;     // threshold — most of the field is cut away
const WISP = 0.55;      // depth of the fine filament breakup
const WISP_F = 2.8;     // and how fine it is, relative to the main field
const INSIDE = 0.22;    // OPACITY of fire over her body, relative to off her
                        // edge — opacity only, never brightness
const LOW = -0.10;      // height where fire dies, in hip heights off the hips
const HIGH = 0.45;      // and where it is fully lit
const FLICKER = 0.14;
const FLICKER_HZ = 6.0;

const BONE_COUNT = 6;

/** Which pairs of the points are joined by limb. Upper body only — the legs
 *  carry no fire, so their anchors are not even declared. */
fn fireBone(i: i32) -> vec2<i32> {
  switch i {
    case 0: { return vec2<i32>(1, 0); }   // neck
    case 1: { return vec2<i32>(6, 1); }   // torso, hips to chest
    case 2: { return vec2<i32>(1, 2); }   // upper arm, left
    case 3: { return vec2<i32>(1, 3); }   //            right
    case 4: { return vec2<i32>(2, 4); }   // forearm, left
    default: { return vec2<i32>(3, 5); }  //          right
  }
}

/** Six declared anchors, then the hips from rzSubject. */
fn fireWorld(subject: i32, idx: i32) -> vec4f {
  if (idx >= 6) {
    let s = rzSubject(subject);
    if (!s.valid) { return vec4f(0.0); }
    return vec4f(s.center, 1.0);
  }
  let a = rzAnchor(subject, idx);
  return vec4f(a.pos, select(0.0, 1.0, a.valid));
}

fn fireHash(p: vec2f) -> f32 {
  var q = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  q = q + dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

fn fireNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(fireHash(i), fireHash(i + vec2f(1.0, 0.0)), u.x),
    mix(fireHash(i + vec2f(0.0, 1.0)), fireHash(i + vec2f(1.0, 1.0)), u.x),
    u.y
  );
}

/** Four octaves. Fire has detail at every scale it is worth paying for; three
 *  reads as cloud, five is past what a half-res target can show. */
fn fireFbm(p: vec2f) -> f32 {
  var sum = 0.0;
  var amp = 0.5;
  var q = p;
  for (var i = 0; i < 4; i++) {
    sum = sum + amp * fireNoise(q);
    q = q * 2.07 + vec2f(1.7, 9.2);
    amp = amp * 0.5;
  }
  return sum / 0.9375;
}

/**
 * Turbulence — the same octaves folded through abs(2n - 1).
 *
 * Plain fbm is smooth everywhere, so thresholding it gives rounded blobs with
 * soft shoulders. Folding each octave puts a CREASE wherever the noise crosses
 * its midpoint, and the creases are thin, branching and continuous across
 * scales. Inverted, they are exactly the filigree that runs through real flame:
 * thread-thin bright lines curling through the body of the fire, which is what
 * the reference has everywhere and smooth noise cannot produce at any
 * frequency. Two octaves — the third is below what survives the threshold.
 */
fn fireRidge(p: vec2f) -> f32 {
  var sum = 0.0;
  var amp = 0.6;
  var q = p;
  for (var i = 0; i < 2; i++) {
    sum = sum + amp * abs(2.0 * fireNoise(q) - 1.0);
    q = q * 2.11 + vec2f(4.3, 2.9);
    amp = amp * 0.5;
  }
  return 1.0 - sum / 0.9;
}

/** The vector from the nearest point of a segment to p — the direction matters
 *  here, because up and down are not the same distance to a flame. */
fn fireDelta(p: vec2f, a: vec2f, b: vec2f) -> vec2f {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return pa - ba * h;
}

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let res = rzResolution();
  let aspect = res.x / max(res.y, 1.0);
  // A square space, or the flames lean with the frame's aspect.
  let p = vec2f(uv.x * aspect, uv.y);

  var heat = 0.0;
  var cover = 1.0;

  for (var c = 0; c < rzSubjectCount(); c++) {
    let s = rzSubject(c);
    if (!s.valid) { continue; }

    // How big this character is ON SCREEN, so every length below can be written
    // in hip heights and hold as the camera pushes in or pulls out.
    //
    // Measured from the HIPS, not the bounding sphere: bounds.w is a
    // deliberately generous cull radius rather than a fit, and reaching a
    // fraction of it in every direction is what filled the frame with fire.
    let H = max(s.center.y - s.root.y, 0.05);
    let mid = rzProject(s.center);
    if (mid.z <= 0.0) { continue; }
    let mid2 = vec2f(mid.x * aspect, mid.y);
    // WHICH WAY IS UP, on screen, for this character.
    //
    // Everything below — the anisotropy, the height weighting, the direction
    // the noise scrolls — is measured against this rather than against the
    // frame's own +y. Screen up is only world up when the camera is level, so
    // measuring against it makes flames lean with the camera: tilt and they
    // tilt with you, orbit under her and they climb out of frame sideways.
    // A world-up vector projected from her hips gives both the direction AND
    // its foreshortened length, so fire correctly looks shorter from above.
    let upW = rzProject(s.center + vec3f(0.0, H, 0.0));
    let upV = vec2f(upW.x * aspect, upW.y) - mid2;
    // Looking straight down the up axis this collapses; fall back to the
    // camera's own up for scale so the fire keeps a sane size rather than
    // exploding as the divisor goes to nothing.
    let camE = rzProject(s.center + rzCameraUp() * H);
    let srCam = max(abs(camE.y - mid.y), 1e-4);
    let sr = max(length(upV), srCam * 0.35);
    let upDir = select(normalize(upV), vec2f(0.0, 1.0), length(upV) < 1e-5);
    let sideDir = vec2f(-upDir.y, upDir.x);
    // Bounding reject — and it has to be DERIVED, not guessed.
    //
    // A guessed radius clips the fire at the top of its own circle, and near
    // the apex a circle is flat to within a pixel across a wide span, so the
    // failure does not look like a circle at all: it looks like someone drew a
    // straight line across the tallest flames. The reach is: however far the
    // furthest limb can be from the hips, plus however far fire climbs above a
    // limb. The first is the bounding SPHERE, which is generous precisely
    // because it already covers a raised arm; the second falls out of the
    // anisotropy, since a pixel LIMB + REACH away along up is at UP_K times
    // that distance in the field.
    let off = length(s.bounds.xyz - s.center) / H;
    let cull = sr * (off + s.bounds.w / H + (REACH + LIMB) / UP_K);
    if (dot(p - mid2, p - mid2) > cull * cull) { continue; }

    // Height off the hips ALONG WORLD UP, in hip heights — the upper-body
    // weight, which must not slide around her as the camera swings.
    let rel = p - mid2;
    let ly = dot(rel, upDir) / sr;
    let lx = dot(rel, sideDir) / sr;
    let tall = smoothstep(LOW, HIGH, ly);
    if (tall <= 0.001) { continue; }

    // Her top half, as six segments on screen.
    var pts: array<vec2f, 7>;
    var good: array<bool, 7>;
    for (var i = 0; i < 7; i++) {
      let w = fireWorld(c, i);
      let pr = rzProject(w.xyz);
      pts[i] = vec2f(pr.x * aspect, pr.y);
      good[i] = w.w > 0.5 && pr.z > 0.0;
    }

    // Nearest limb, measured with up cheap and down expensive.
    var d = 1e9;
    for (var i = 0; i < BONE_COUNT; i++) {
      let bone = fireBone(i);
      if (!good[bone.x] || !good[bone.y]) { continue; }
      let v = fireDelta(p, pts[bone.x], pts[bone.y]);
      // Split along world up and across it, then charge each differently.
      let vy = dot(v, upDir);
      let vx = dot(v, sideDir);
      let ky = select(vy * DOWN_K, vy * UP_K, vy > 0.0);
      d = min(d, length(vec2f(vx, ky)));
    }
    if (d > 1e8) { continue; }

    // Hip heights from the nearest limb surface, so the numbers above are
    // readable and the fire is the same size on a chibi and on a tall model.
    let t = 1.0 - clamp((d / sr - LIMB) / REACH, 0.0, 1.0);
    if (t <= 0.0) { continue; }

    // ONE noise field for the whole body, in body-local coordinates so it does
    // not swim across her when the camera moves, on a lattice stretched
    // vertically so its cells are tall, climbing with time.
    // The lattice stands upright in the WORLD, so the tall cells stay tall and
    // the scroll climbs the way she does, whatever the camera is doing.
    let q = vec2f(lx * FX, ly * FY) + vec2f(0.0, -time * RISE * FY);
    // Warped by itself, TWICE. One stage of displacement bends the field and
    // gives licks; feeding that stage into a second, finer one folds the bends
    // back over themselves, and folded bends are the curls and hooks — the
    // closed loops of flame in the reference that a single warp never makes,
    // because a single warp can only ever push the field sideways.
    let w1 = vec2f(fireFbm(q * 0.34), fireFbm(q * 0.34 + vec2f(5.2, 1.3))) - 0.5;
    let w2 = vec2f(fireFbm(q * 0.95 + w1), fireFbm(q * 0.95 + w1 + vec2f(3.1, 7.7))) - 0.5;
    let n = fireFbm(q + (w1 * 1.5 + w2 * 0.8) * WARP * FY);
    let flick = 1.0 - FLICKER * fireNoise(vec2f(time * FLICKER_HZ + f32(c) * 3.7, 0.5));

    // The falloff says how far fire COULD reach here; the noise carves that
    // into tongues, and the floor cuts the gaps between them clean away.
    var h = clamp((t * tall * (1.0 - ROIL + ROIL * 2.0 * n) - FLOOR) / (1.0 - FLOOR), 0.0, 1.0);
    // The filigree, laid over the tongues and carried by the same warp so it
    // curls with them rather than sliding across them.
    let fine = fireRidge(q * WISP_F + w1 * WARP * FY + vec2f(0.0, -time * RISE * FY * 1.5));
    h = h * (1.0 - WISP + WISP * 2.0 * fine) * flick;

    // Fainter where it crosses her — but by OPACITY, not by strength. Scaling
    // the heat slid every pixel down the colour ramp, so fire over her body
    // came out as the dark smoky end of it: a brown wash with no structure,
    // which is what made it read as a blur rather than as flame. The shape and
    // the colour are now decided at full strength and only the coverage is
    // pulled back, so the tongues stay tongues and you see her through them.
    let covered = 1.0 - smoothstep(H * 1.2, H * 1.8, abs(depth - mid.z));
    if (h > heat) {
      heat = h;
      cover = mix(1.0, INSIDE, covered);
    }
  }

  if (heat <= 0.004) { return vec4f(0.0); }
  // A WIDE red plateau before the orange starts. In the reference the yellow
  // hearts are small and everything around them is deep red for a long way out;
  // ramping straight from red to orange makes the whole fire read as orange,
  // which is the difference between a bonfire and this.
  var rgb = mix(SMOKE_COLOR, RED_COLOR, smoothstep(0.02, 0.20, heat));
  rgb = mix(rgb, ORANGE_COLOR, smoothstep(0.36, 0.68, heat));
  rgb = mix(rgb, HOT_COLOR, smoothstep(0.78, 0.97, heat));
  // Never fully opaque: flame is something you see through, and the character
  // behind it has to stay readable.
  return vec4f(rgb, smoothstep(0.03, 0.45, heat) * 0.88 * cover);
}

// ── The light the fire throws ────────────────────────────────────────────────
//
// Fire is the least excusable thing to draw without lighting: a body wrapped in
// flame, shaded only by the sun, reads as a sticker of fire over a person. The
// sources are already known — fireWorld(subject, idx) is what the tongues are
// drawn from — so the lights re-read the same bones the picture does.
//
// Two, not six. The lights are a soft orange wash, and six point lights inside
// one body would be six copies of nearly the same wash for six slots of budget.
// The torso carries the body, the head carries the face — and a face lit from
// its own flames is the shot.
// @lights 2
const FIRE_I = 1.9;        // brightness at the source
const FIRE_REACH = 1.5;    // in hip heights — fire lights its own body and a
                           // little of the floor, not the room
const FIRE_UP = 0.10;      // lifted toward the tongues, which climb from the
                           // bone rather than sitting on it

fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  l.color = ORANGE_COLOR;
  l.intensity = 0.0;
  l.radius = 1.0;
  l.pos = vec3f(0.0, 0.0, 0.0);

  let s = rzSubject(0);
  if (!s.valid) { return l; }
  let H = max(s.center.y - s.root.y, 0.05);

  // 0 the torso (fireWorld index 6, the hips), 1 the head (anchor 0, 頭).
  let src = fireWorld(0, select(0, 6, i == 0u));
  if (src.w < 0.5) { return l; }

  l.pos = vec3f(src.x, src.y + H * FIRE_UP, src.z);
  l.radius = H * FIRE_REACH;
  // The SAME flicker the flames carry, offset per source so the two do not
  // pulse in lockstep — fire that breathes in unison reads as one lamp on a
  // dimmer, which is exactly what this is trying not to be.
  // The literal rather than TAU: this file has no such constant, and adding
  // one to a shared namespace for one line invites a collision later.
  let phase = time * FLICKER_HZ * 6.2831853 + f32(i) * 2.4;
  l.intensity = FIRE_I * (1.0 - FLICKER + FLICKER * sin(phase));
  // The head sits in the smaller, cooler part of the fire.
  l.intensity = l.intensity * select(1.0, 0.7, i == 1u);
  return l;
}
