#layer additive
#anchor 頭
#anchor 上半身2
#anchor 左ひじ
#anchor 右ひじ
#anchor 左手首
#anchor 右手首
#anchor 左ひざ
#anchor 右ひざ


// Vyke's Dragonbolt — the red fire-lightning of Elden Ring, clinging to a body.
//
// These are ARCS, not lightning. Lightning is a path from somewhere to
// somewhere else and it looks it: straight, spanning, purposeful. An electric
// arc has nowhere to go, so it curves — it crawls around what it is clinging
// to. That is the whole difference between this and a bolt effect, and it is
// built into the geometry rather than tuned in afterwards:
//
//   CRACKLES orbit a joint, on a tilted ring, through less than a third of a
//   turn. CONNECTORS are short bowed pieces of the way between two joints,
//   never the whole span — electricity crawling over a body, not a cable strung
//   between its corners.
//
// Both ORBIT IN DEPTH as well as across the screen, which is what makes them
// look wrapped rather than painted on. Every point of every path carries its
// own distance from the camera, and the half of a ring that swings behind a
// shoulder is hidden by that shoulder. rzProject hands back z in metres on the
// same axis as the depth the mount is given, so the test costs one comparison
// per path — no marching, no 3D distance field.
//
// The depth is derived rather than projected point by point: the ring's screen
// radius is converted to metres once per arc (one extra projection, after the
// bound rejects), and the swing is that radius through the tilt. A tilted ring
// projects to an ellipse and this draws a circle, which is a lie no one can see
// in a jagged filament, and it keeps the whole path in 2D where it is cheap.
//
// Both FORK, off the tangent wherever the branch tears. Eight anchors (head,
// chest, elbows, wrists, knees) plus the hips and the floor point, free from
// rzSubject, cover the whole silhouette. Channels RESTRIKE on a timer rather
// than animating — a path lives for a blink, then a new seed tears a new one
// somewhere else. That flicker is what reads as electricity.
//
// The halo is drawn in two tiers rather than one. A foreground mount composites
// after tone mapping and is clamped to display range, so it never reaches the
// bloom pyramid: every bit of the glow around these arcs has to be in the
// falloff here, and a single radius reads as a flat disc.
//
// Tunables — edit and ⌘⏎.
const CORE_COLOR = vec3f(1.0, 0.78, 0.68);   // the filament — hot, still red
const BODY_COLOR = vec3f(1.0, 0.11, 0.04);   // the body of the arc
const HAZE_COLOR = vec3f(0.52, 0.03, 0.01);  // deep ember, the outer light
const STRIKE_HZ = 5.0;    // new channels per second
const ATTACK = 0.05;      // of a life spent snapping on
const HOLD = 0.55;        // and how much of it is spent at full before dying
const SIZZLE = 9.0;       // flickers per life
const BOLTS = 20;         // channels alive at once — most of them crackles
const CONNECTORS = 6;     // how many of those crawl between two joints
const SEGS = 7;           // jags per arc — enough that the curve stays a curve
const FORK_SEGS = 3;      // and per branch
const CRACKLE_R = 0.024;  // how wide a crackle orbits its joint
const SWEEP_MIN = 0.7;    // and through how much of a turn, in radians
const SWEEP_MAX = 1.9;
const TILT_MIN = 0.45;    // how far the ring leans into the screen: 0 is flat
const TILT_MAX = 1.0;     // on to the frame, 1 is swinging fully through it
const SPAN_MIN = 0.20;    // a connector's length, as a fraction of the joint gap
const SPAN_MAX = 0.38;
const HUG = 0.85;         // how far along its bone a connector may slide
const BOW = 0.5;          // how far a connector bows, relative to JAG
const JAG = 0.016;        // sideways wander — small, or it eats the curve
const CORE_W = 0.0018;    // filament half-width — a hairline, and it stays one
const GLOW_W = 0.0090;    // the tight halo hugging it
const HAZE_W = 0.0500;    // and the wide, faint aura it sits in
const CORE_I = 1.9;
const GLOW_I = 0.90;
const HAZE_I = 0.30;
const FLICKER = 0.40;     // depth of the per-channel brightness flicker
const WOBBLE = 0.35;      // how much the filament writhes while it is alive
const WOBBLE_HZ = 7.0;    // and how fast
const Z_BIAS = 0.02;      // metres of slack, so an arc lying on the skin shows
const Z_FADE = 0.03;      // and how sharply it goes as it passes behind

const TAU = 6.2831853;

fn arcHash(p: vec2f) -> f32 {
  var q = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  q = q + dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

/**
 * The aura around the filament, falling off cubically.
 *
 * The shape is the whole difference between neon and a blob. A linear ramp
 * across the radius fills its disc almost evenly, so a wide one reads as a lit
 * tube; cubed, the same radius stays concentrated on the line and trails away
 * to nothing at its edge — which is what a glass tube full of gas actually does
 * to the air around it. That is what lets the aura be WIDE and faint at once,
 * and a wide faint aura next to a hairline core is the neon look.
 *
 * Bounded, not a 1/r tail: the reject below is derived from HAZE_W, and a glow
 * that never quite reaches zero has no radius to derive one from.
 */
fn arcHalo(dist: f32) -> f32 {
  let t = 1.0 - smoothstep(0.0, HAZE_W, dist);
  return t * t * t;
}

fn arcSeg(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h);
}

/** Ten points on the body: eight declared anchors, then the hips and the floor.
 *  World space — the caller projects, because it also wants the world position
 *  to work out how big a metre is on screen there. w is 0 on a rig without it. */
fn arcWorld(subject: i32, idx: i32) -> vec4f {
  if (idx >= 8) {
    let s = rzSubject(subject);
    if (!s.valid) { return vec4f(0.0); }
    return vec4f(select(s.center, s.root, idx == 9), 1.0);
  }
  let a = rzAnchor(subject, idx);
  return vec4f(a.pos, select(0.0, 1.0, a.valid));
}

const BONE_COUNT = 10;

/**
 * The body as SEGMENTS, not points — which pairs of the ten are joined by limb.
 *
 * Anchors are capped at eight and all eight are spent, but the coverage problem
 * was never the count: ten discrete points leave the forearms, the thighs and
 * the torso bare no matter how many channels orbit the joints, because there is
 * nothing between them to orbit. Naming the segments instead lets a channel sit
 * ANYWHERE along a limb, so eight anchors cover a whole body continuously and
 * more anchors would buy nothing.
 *
 * It also fixes what the connectors were doing. Picking two joints at random
 * drew head-to-knee lines through open air; along a bone, every one of them
 * lies on the body by construction, and the end-hugging that used to paper over
 * that is now just where along the limb it sits.
 *
 * The two lower legs run knee to the FLOOR POINT rather than to an ankle: at
 * eight slots the wrists earn their place more than the ankles do, and knee to
 * floor tracks a shin closely enough while she is standing on it. A lifted leg
 * is the case it approximates worst.
 */
fn arcBone(i: i32) -> vec2<i32> {
  switch i {
    case 0: { return vec2<i32>(1, 0); }   // neck
    case 1: { return vec2<i32>(8, 1); }   // torso
    case 2: { return vec2<i32>(1, 2); }   // upper arm, left
    case 3: { return vec2<i32>(1, 3); }   //            right
    case 4: { return vec2<i32>(2, 4); }   // forearm, left
    case 5: { return vec2<i32>(3, 5); }   //          right
    case 6: { return vec2<i32>(8, 6); }   // thigh, left
    case 7: { return vec2<i32>(8, 7); }   //        right
    case 8: { return vec2<i32>(6, 9); }   // lower leg, left
    default: { return vec2<i32>(7, 9); }  //            right
  }
}

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let res = rzResolution();
  let aspect = res.x / max(res.y, 1.0);
  // Measure in a square space, or every arc is stretched into an ellipse.
  let p = vec2f(uv.x * aspect, uv.y);

  var core = 0.0;
  var glow = 0.0;
  var haze = 0.0;

  for (var c = 0; c < rzSubjectCount(); c++) {
    let s = rzSubject(c);
    if (!s.valid) { continue; }

    // ── Reject the whole character before touching a single channel ──
    //
    // The two-tier cull below is tight, but every channel had to be SET UP
    // before it could be reached: a hash for its phase, four more for its
    // shape, ten trig calls to sketch it and a projection or two to place it.
    // Twenty-odd of those ran on every pixel of the frame, including the empty
    // corners — which is most of any frame, and was most of the cost.
    //
    // One circle around the character answers all of them at once. Two
    // projections, and a pixel that is nowhere near her is done.
    let bc = rzProject(s.bounds.xyz);
    if (bc.z <= 0.0) { continue; }
    let be = rzProject(s.bounds.xyz + rzCameraUp() * s.bounds.w);
    let br = max(abs(be.y - bc.y), 1e-4);
    let bc2 = vec2f(bc.x * aspect, bc.y);
    // Her silhouette, plus how far past it a channel can stray: a branch may
    // leave a bone by half its length, and an orbit and its aura add their own.
    let bound = br * 1.5 + CRACKLE_R + JAG + HAZE_W;
    if (dot(p - bc2, p - bc2) > bound * bound) { continue; }

    // ── The body, projected once for every channel that will ask ──
    //
    // Each channel used to project the bone it sits on for itself, which is one
    // or two projections apiece and the same joints over and over. Ten points
    // cover every bone, so ten projections cover every channel.
    var pts: array<vec2f, 10>;
    var zs: array<f32, 10>;
    var ok: array<bool, 10>;
    for (var i = 0; i < 10; i++) {
      let wp = arcWorld(c, i);
      let pr = rzProject(wp.xyz);
      pts[i] = vec2f(pr.x * aspect, pr.y);
      zs[i] = pr.z;
      ok[i] = wp.w > 0.5 && pr.z > 0.0;
    }

    // ── How near is this pixel to each LIMB ──
    //
    // The reject above answers "is this pixel anywhere near her" and no more,
    // and it cannot do better: bounds.w is the generous cull sphere, so the
    // circle it gives covers most of the frame whenever she is framed at all.
    // Inside it every channel was still being set up — a hash for its phase,
    // four for its shape, ten trig calls to sketch it — and twenty of those on
    // most of the frame is the whole cost of this effect.
    //
    // Ten segment distances answer it properly. A channel can only reach so far
    // from the bone it lives on, so a pixel beside one arm can throw out every
    // channel on the other arm and both legs for the price of ONE array lookup
    // in the loop below, before any of that setup runs. Ten cheap tests replace
    // most of twenty expensive ones.
    var bd: array<f32, 10>;
    var near = false;
    for (var i = 0; i < BONE_COUNT; i++) {
      let bn = arcBone(i);
      if (!ok[bn.x] || !ok[bn.y]) {
        bd[i] = 1e9;
        continue;
      }
      // Its own longest branch scales with its own length, so a torso claims
      // more room than a forearm and neither is given the other's slack.
      let len = length(pts[bn.y] - pts[bn.x]);
      let reach = CRACKLE_R * 1.25 + JAG + HAZE_W + 0.55 * SPAN_MAX * len;
      bd[i] = arcSeg(p, pts[bn.x], pts[bn.y]) - reach;
      near = near || bd[i] <= 0.0;
    }
    if (!near) { continue; }

    for (var b = 0; b < BOLTS; b++) {
      // Each channel restrikes on its own phase, so they never blink together.
      let phase = time * STRIKE_HZ + f32(b) * 0.317;
      let strike = floor(phase);
      let life = fract(phase);
      let seed = vec2f(strike * 1.7 + f32(b) * 31.7, f32(c) * 11.3 + f32(b) * 5.1);

      // WHICH LIMB this channel lives on — asked FIRST, because the answer is
      // also the cheapest possible rejection. Both shapes below sit on it: a
      // crackle rings some point along it, a connector runs a slice of it.
      let bi = i32(arcHash(seed) * f32(BONE_COUNT) * 0.9999);
      if (bd[bi] > 0.0) { continue; }   // one lookup, and the channel is gone
      let bone = arcBone(bi);
      let e2 = pts[bone.x];
      let f2 = pts[bone.y];
      let ez = zs[bone.x];
      let fz = zs[bone.y];

      // Snap on, HOLD, then die. A plain decay across a life this long reads
      // as a slow fade rather than as an arc that struck and stayed lit; the
      // hold is what lets a channel linger without going sluggish.
      let env = smoothstep(0.0, ATTACK, life) * (1.0 - smoothstep(HOLD, 1.0, life));
      let flick = 1.0 - FLICKER * arcHash(seed + vec2f(life * SIZZLE, 0.0));
      let amp = env * flick;
      if (amp < 0.02) { continue; }

      // The two shapes share one walk below; these are the parameters it needs.
      let curved = b >= CONNECTORS;
      var rad = 0.0;              // curved: orbit radius, on screen
      var a0 = 0.0;               //         from what angle
      var sweep = 0.0;            //         through how much of a turn
      var p0 = vec2f(0.0);        // straight: from here
      var p1 = vec2f(0.0);        //           to here
      var bow = 0.0;              //           bowing this far off the chord
      var z0 = 0.0;               // depth at the head of the run
      var z1 = 0.0;               //           and at its tail
      var anchor = vec2f(0.0);    // what a ring is centred on, on screen
      var anchorZ = 0.0;
      var boneLen = 1.0;          // its bone's length on screen, for the scale

      if (curved) {
        // A crackle: a ring around a point ANYWHERE ALONG the limb. Small
        // radius, partial turn, either handedness — it clings, it does not
        // radiate. Projected from the interpolated world point rather than
        // interpolated between two projections, which is the same thing only
        // for a bone pointed across the frame.
        // Interpolated between the bone's two PROJECTED ends rather than
        // projected from the interpolated world point. Those differ only by
        // what perspective does across the length of one limb, which at the
        // scale of a hairline arc is nothing, and it saves a projection on
        // every channel.
        let tb = arcHash(seed + vec2f(0.9, 3.4));
        anchor = mix(e2, f2, tb);
        anchorZ = mix(ez, fz, tb);
        boneLen = length(f2 - e2);
        z0 = anchorZ;
        z1 = anchorZ;
        rad = CRACKLE_R * (0.55 + 0.7 * arcHash(seed + vec2f(1.1, 8.8)));
        a0 = arcHash(seed + vec2f(5.1, 2.9)) * TAU;
        let turn = SWEEP_MIN + (SWEEP_MAX - SWEEP_MIN) * arcHash(seed + vec2f(2.4, 6.3));
        sweep = select(-turn, turn, arcHash(seed + vec2f(9.2, 0.4)) > 0.5);
      } else {
        // A connector: a SHORT bowed slice OF the limb, not the whole of it.
        let frac = SPAN_MIN + (SPAN_MAX - SPAN_MIN) * arcHash(seed + vec2f(1.1, 8.8));
        // Where along the bone it sits, measured from one end or the other.
        let slide = arcHash(seed + vec2f(4.9, 3.1)) * (1.0 - frac) * HUG;
        let t0 = select(slide, 1.0 - frac - slide, arcHash(seed + vec2f(2.2, 9.6)) > 0.5);
        p0 = mix(e2, f2, t0);
        p1 = mix(e2, f2, t0 + frac);
        // The slice runs along the limb in depth as well as across screen.
        z0 = mix(ez, fz, t0);
        z1 = mix(ez, fz, t0 + frac);
        anchorZ = z0;
        // Signed, so it curves one way instead of wobbling around the chord.
        bow = select(-BOW, BOW, arcHash(seed + vec2f(9.2, 0.4)) > 0.5) * JAG;
      }

      // ── Culling, in two tiers ──
      //
      // Walking the real path costs ten segment tests and ten sines, and a
      // circular bound is a bad fit for something long and thin — worse, it
      // admits every pixel out to the AURA radius, where the wander is far
      // smaller than the distance and nothing about the exact path can be seen.
      // Widening the aura for the neon look therefore cost roughly the square
      // of the widening in walked pixels, which is where the frame rate went.
      //
      // So measure once against a three-piece SKETCH of the shape — both ends,
      // the bulge between them, and the branch as one straight run — and decide
      // from that. Beyond the aura, skip. Inside the aura but outside the
      // filament, shade the aura straight off the sketch: at that distance a
      // wander of JAG is a rounding error on a cubic falloff, and the walk buys
      // nothing. Only pixels near enough to resolve the filament pay for it.
      //
      // The sketch is a capsule rather than a disc, which is the second saving:
      // for a connector it admits about half the pixels the old bound did.
      let run = select(length(p1 - p0), abs(sweep) * rad, curved);
      let fi = 1 + i32(arcHash(seed + vec2f(6.6, 1.3)) * f32(SEGS - 2));
      let tf = f32(fi + 1) / f32(SEGS);
      var mid = (p0 + p1) * 0.5;
      var head = p0;
      var tail = p1;
      var fo = mix(p0, p1, tf);   // where the branch tears, near enough for a cull
      var tang = vec2f(1.0, 0.0);
      if (curved) {
        let am = a0 + sweep * 0.5;
        let af = a0 + sweep * tf;
        head = anchor + vec2f(cos(a0), sin(a0)) * rad;
        tail = anchor + vec2f(cos(a0 + sweep), sin(a0 + sweep)) * rad;
        mid = anchor + vec2f(cos(am), sin(am)) * rad;
        fo = anchor + vec2f(cos(af), sin(af)) * rad;
        tang = vec2f(-sin(af), cos(af)) * sign(sweep);
      } else {
        let dirn = normalize(p1 - p0 + vec2f(1e-5));
        let perp = vec2f(-dirn.y, dirn.x);
        mid = mid + perp * bow;
        fo = fo + perp * (bow * sin(tf * 3.14159265));
        tang = dirn;
      }
      // Same handedness hash the branch itself uses, and its longest reach.
      let fangC = select(-1.1, 1.1, arcHash(seed + vec2f(0.3, 4.4)) > 0.5);
      let fdirC = vec2f(
        tang.x * cos(fangC) - tang.y * sin(fangC),
        tang.x * sin(fangC) + tang.y * cos(fangC)
      );
      let dc = min(
        min(arcSeg(p, head, mid), arcSeg(p, mid, tail)),
        arcSeg(p, fo, fo + fdirC * run * 0.55)
      );
      if (dc > HAZE_W + JAG) { continue; }
      if (dc > GLOW_W + JAG) {
        // Aura only. One depth for the whole shape is enough for something this
        // soft, and it saves working out the metre scale at all.
        let zc = select(mix(z0, z1, 0.5), anchorZ, curved);
        let visC = 1.0 - smoothstep(depth + Z_BIAS - Z_FADE, depth + Z_BIAS + Z_FADE, zc);
        haze = max(haze, arcHalo(dc) * amp * visC);
        continue;
      }

      // How far this ring swings toward and away from the camera. The radius is
      // on screen; the bone below says how much of the frame a metre is; the tilt is
      // how much of the ring is pointed through the frame rather than across
      // it. A flat ring never hides, a steep one spends half its sweep behind.
      var zAmp = 0.0;
      var zAxis = 0.0;
      if (curved) {
        let tilt = TILT_MIN + (TILT_MAX - TILT_MIN) * arcHash(seed + vec2f(3.7, 5.5));
        // How many metres a frame-height is HERE, read off the bone the arc
        // is sitting on: its length on screen against its length in the world.
        // That was a second projection per channel when it went through
        // a helper of its own, and the bone already answers it.
        let wl = length(arcWorld(c, bone.y).xyz - arcWorld(c, bone.x).xyz);
        let perFrame = max(boneLen, 1e-5) / max(wl, 1e-4);
        // Capped at the bounding sphere: a camera looking down the up axis
        // makes the metre-to-frame conversion degenerate, and an uncapped
        // swing would then bury or float every arc at once.
        zAmp = min(rad / perFrame * tilt, s.bounds.w);
        zAxis = arcHash(seed + vec2f(8.1, 1.9)) * TAU;
      }

      // fi — where the branch tears off — was chosen by the cull above. Never
      // the first or last jag: a fork at the root reads as two arcs, and one at
      // the tip reads as a fray.
      var forkAt = mid;
      var forkDir = vec2f(1.0, 0.0);
      var forkZ = anchorZ;
      var prev = head;
      var zPrev = select(z0, anchorZ + zAmp * cos(a0 - zAxis), curved);
      // Nearest point on the path, and — the reason this is tracked at all —
      // how far away that point is. One shade, one depth test, per path.
      var d = 1e9;
      var dz = zPrev;

      for (var i = 0; i < SEGS; i++) {
        let t = f32(i + 1) / f32(SEGS);
        // Wander biggest mid-run, so both ends stay where they belong.
        let bulge = sin(t * 3.14159265);
        // The wander is fixed for the life of the channel, so a longer life
        // would sit there as a still picture. This is the one term that moves
        // while it is lit: each jag breathes on its own phase, which is what an
        // arc does when it is hunting for a path.
        let wob = sin(time * WOBBLE_HZ * TAU + arcHash(seed + vec2f(f32(i) * 3.1, 5.5)) * TAU);
        let jit = ((arcHash(seed + vec2f(f32(i) * 9.1, 2.2)) - 0.5) * 2.0 + wob * WOBBLE) * JAG * bulge;
        var next: vec2f;
        var zNext: f32;
        if (curved) {
          // Radially, so the jag roughens the orbit instead of flattening it.
          let ang = a0 + sweep * t;
          next = anchor + vec2f(cos(ang), sin(ang)) * (rad + jit);
          zNext = anchorZ + zAmp * cos(ang - zAxis);
        } else {
          let dirn = normalize(p1 - p0 + vec2f(1e-5));
          next = mix(p0, p1, t) + vec2f(-dirn.y, dirn.x) * (jit + bow * bulge);
          zNext = mix(z0, z1, t);
        }
        let ds = arcSeg(p, prev, next);
        if (ds < d) {
          d = ds;
          dz = (zPrev + zNext) * 0.5;
        }
        if (i == fi) {
          forkAt = next;
          forkDir = normalize(next - prev + vec2f(1e-5));
          forkZ = zNext;
        }
        prev = next;
        zPrev = zNext;
      }

      // The branch: off the tangent, short, thinner, bowed one way so it curls
      // rather than zigzags. Short enough that it takes its depth from where it
      // tore off and does not need one of its own.
      let fang = select(-1.1, 1.1, arcHash(seed + vec2f(0.3, 4.4)) > 0.5);
      let fdir = vec2f(
        forkDir.x * cos(fang) - forkDir.y * sin(fang),
        forkDir.x * sin(fang) + forkDir.y * cos(fang)
      );
      let fperp = vec2f(-fdir.y, fdir.x);
      let fbow = select(-1.0, 1.0, arcHash(seed + vec2f(7.7, 2.1)) > 0.5) * JAG * 0.7;
      let fend = forkAt + fdir * run * (0.25 + 0.30 * arcHash(seed + vec2f(0.7, 6.1)));
      var fa = forkAt;
      var df = 1e9;
      for (var i = 0; i < FORK_SEGS; i++) {
        let t = f32(i + 1) / f32(FORK_SEGS);
        let e = sin(t * 3.14159265);
        let wob = sin(time * WOBBLE_HZ * TAU + arcHash(seed + vec2f(f32(i) * 6.3, 1.4)) * TAU);
        let jit = ((arcHash(seed + vec2f(f32(i) * 4.7, 8.3)) - 0.5) + wob * WOBBLE) * JAG * 0.5 * e;
        let next = mix(forkAt, fend, t) + fperp * (fbow * e + jit);
        df = min(df, arcSeg(p, fa, next));
        fa = next;
      }

      // Hidden where the body is nearer than the arc. Soft, over a few
      // centimetres: a hard test crawls along the silhouette as she turns.
      let visM = 1.0 - smoothstep(depth + Z_BIAS - Z_FADE, depth + Z_BIAS + Z_FADE, dz);
      let visF = 1.0 - smoothstep(depth + Z_BIAS - Z_FADE, depth + Z_BIAS + Z_FADE, forkZ);
      if (visM + visF <= 0.001) { continue; }

      // BRIGHTEST WINS, rather than summing. Twenty-two channels over ten
      // joints means two or three of them share a limb at any moment, and added
      // light compounds there: the halos merge into one blob and the filament
      // saturates past white in the middle of it. Under a max, an arc crossing
      // another stays as bright as the brighter of the two and no brighter, so
      // each one keeps its own edge and the discharge reads as many thin arcs
      // instead of one lit tube.
      //
      // The branch carries a thinner filament but the same light, so it reads
      // as part of the same discharge rather than as a second, dimmer one.
      core = max(core, (1.0 - smoothstep(0.0, CORE_W, d)) * amp * visM);
      core = max(core, (1.0 - smoothstep(0.0, CORE_W * 0.6, df)) * amp * 0.8 * visF);
      glow = max(glow, (1.0 - smoothstep(0.0, GLOW_W, d)) * amp * visM);
      glow = max(glow, (1.0 - smoothstep(0.0, GLOW_W, df)) * amp * visF);
      haze = max(haze, arcHalo(d) * amp * visM);
      haze = max(haze, arcHalo(df) * amp * visF);
    }
  }

  let heat = core * CORE_I + glow * GLOW_I + haze * HAZE_I;
  if (heat <= 0.004) { return vec4f(0.0); }
  // Colour from intensity: ember in the outer light, red through the body of
  // the arc, and the pale core only where it is genuinely hot — a high
  // threshold, because a red bolt that whites out along its length is an
  // orange one.
  var rgb = mix(HAZE_COLOR, BODY_COLOR, smoothstep(0.03, 0.50, heat));
  rgb = mix(rgb, CORE_COLOR, smoothstep(1.3, 2.4, heat));
  return vec4f(rgb, clamp(heat, 0.0, 1.0));
}
