// Fireworks — ported to WGSL from "Fireworks" by Gatomoi
// https://www.shadertoy.com/view/73sXWH
// The click-launched shells and the feedback trail are dropped: both needed
// extra buffers, and an effect here is a single pass.

// Tunables — edit and ⌘⏎.
//
// Sized UP for the half-res field pass: the sky holds nearly twice the shells,
// each bigger and hotter, and the arithmetic lands almost exactly on what the
// smaller show used to cost at full resolution. The culls are what let it
// scale — every pixel still rejects whole shells before touching a spark.
const SHELLS = 10;             // shells in the rotation
const PARTICLES = 72;          // sparks per shell — THE cost dial
const SECONDARY = 4;           // crackles each qualifying spark throws
const PERIOD = 5.0;            // seconds between a slot's launches
// 0.382 of the period — the golden-ratio conjugate, kept in step with PERIOD. A
// rounder fraction resonates: were this an eighth of the period, slots 0 and 8
// would launch in lockstep and the sky would clump instead of filling.
const STAGGER = 1.25;          // seconds between one slot and the next — at
                               // 3.35s of life that is ~2.7 shells in the air
                               // at once, against ~1.75 before
const CLIMB_SCALE = 1.6;       // stretches the climb only — a shell that shoots
                               // up instantly has no flight to watch
const SEQUENCE = 3.35;         // total life: climb + burst, and no longer — a slot
                               // past it is dead, so ending the window early is
                               // free frames
const BURST_TIME = 1.55;       // the reference's own burst length
// The reference's own value. Raising it was a mistake I made trying to force a
// visible fall — the fall is visible there because the trails DRAW it, not
// because the sparks drop harder, and a heavier one only made the burst hang
// below the point it went off at.
const GRAVITY = 0.32;
const CHILD_GRAVITY = 0.06;    // the same, for the embers
const CRACKLE_AT = 0.57;       // fraction of the burst before the secondaries go
const EXPOSURE = 1.8;
const SPEED = 0.85;            // clock multiplier — lower is slower, all of it
const BOOM = 1.15;             // how far the sparks throw — smaller sits further back
const WORLD_Z = 46.0;          // how far BEHIND the stage the shells rise
const WORLD_SPAN = 30.0;       // half-width of the firing line, world units
const WORLD_LOW = 17.0;        // burst heights, world units
const WORLD_HIGH = 30.0;
const REF_DIST = 55.0;         // the distance at which a shell wears its tuned size
const CLIMB_TRAIL = 0.045;     // how much of the climb the rocket's streak covers.
                               // Small, because the climb is EASED: at 0.2 the
                               // rocket has already flown half its path, so that
                               // much of "its past" is half the frame.
const HEAD_HALO = 0.022;       // how far the rocket's own glow carries
const CLIMB_GLOW = 2.6;        // brightness of that streak. Brightness IS opacity
                               // here: the layer's alpha is its own energy, so a
                               // dim streak is a see-through one.
const CLIMB_HALO = 0.006;       // and how WIDE it is allowed to be. Brightness and
                               // width are separate dials on purpose — a 1/r glow
                               // spreads as you brighten it, so without a bound
                               // the only legible streak is a fat one.

// The reference kept its streaks in a buffer that decayed 0.955/0.925/0.875 per
// frame, so a streak lived about this long — and PER CHANNEL, blue fastest. Both
// numbers below are that buffer, solved rather than eyeballed: at 60fps the decay
// rates are ln(1/0.955)·60 = 2.76 for red against 8.0 for blue, so a 0.37s-old
// piece of streak keeps 36% of its red and 5% of its blue. That is why the tails
// go orange, and it is a colour we can compute instead of guess at.
const TRAIL_SECONDS = 0.37;
const TRAIL_DECAY = vec3f(2.76, 4.68, 8.0);
const TRAIL_MIX = 0.72;        // what the reference composited that buffer at
const PLAIN_SHELLS = 0.1;      // fraction that burst with no trails at all
const SMALLEST = 0.7;          // the runt of the rotation, as a fraction of BOOM

const TAU = 6.28318530718;
// Every 1/r glow inside the cull is faded to exactly zero at this radius, and the
// cull is this radius past the farthest spark. That makes the boundary a
// definition rather than an estimate: nothing can be truncated, because nothing
// is still alive out there. Estimating it is what drew the ring — a glow reads as
// visible far further than its raw value suggests, since the tone map lifts it
// (alpha = 1 − exp(−1.52·c)) before the frame is quantised.
// Derived, not chosen: a spark's glow is size·0.78/d, so at the size above it
// falls under one code value at 0.32. Every glow inside the cull is faded to
// zero exactly here, which is what makes the boundary safe to cull on.
const GLOW_REACH = 0.52;
// One 8-bit code value, referred back through the tone map (alpha = 1 − exp(−1.52·c)).
// Anything dimmer than this cannot change a pixel, so it need not be drawn.
const VISIBLE = 0.0026;

fn background(ray: vec3f, uv0: vec2f, clock: f32) -> vec4f {
  // One dial for the whole display: shells, climbs, bursts and the sparkle rate
  // all read this clock, so slowing it slows them together instead of drifting
  // out of step with each other.
  let time = clock * SPEED;
  let res = bgResolution();
  // Square units, y in −1..1, x carrying the aspect — the convention this came
  // from, and the one the ballistics constants are tuned in.
  let uv = vec2f((uv0.x * 2.0 - 1.0) * (res.x / max(res.y, 1.0)), uv0.y * 2.0 - 1.0);

  var col = vec3f(0.0);
  for (var s = 0; s < SHELLS; s++) {
    let slot = f32(s);
    let offset = slot * STAGGER;
    if (time < offset) {
      continue;
    }
    // Each slot fires on its own cycle; floor() names the launch, and every
    // random the shell wears is drawn from that name, so it keeps its colour and
    // its scatter for as long as it burns.
    let cycle = floor((time - offset) / PERIOD);
    let elapsed = time - (cycle * PERIOD + offset);
    if (elapsed <= 0.0 || elapsed >= SEQUENCE) {
      continue;
    }
    let seed = vec2f(slot * 91.7 + cycle * 17.3, cycle * 41.9 + slot * 8.2);
    let wt = fwWorldTarget(slot, cycle);
    let pr = rzProject(wt);
    // Behind the lens, or close enough that the scale would explode: skip.
    if (pr.z < 4.0) { continue; }
    let pad = vec3f(wt.x + (fwHash(seed + vec2f(31.0, 4.0)) - 0.5) * 12.0, 0.0, wt.z);
    let pp = rzProject(pad);
    if (pp.z < 4.0) { continue; }
    let shellScale = clamp(REF_DIST / pr.z, 0.4, 1.6);
    col += fwSequence(uv, pr.xy * res, pp.xy * res, elapsed, seed, res, shellScale) * 0.82;
  }

  let vignette = smoothstep(0.84, 0.2, length(uv0 - 0.5));
  col *= 0.9 + 0.1 * vignette;

  // A soft shoulder rather than a clamp: highlights roll off instead of clipping
  // flat to white, which is what keeps a bright core its own colour.
  let mapped = 1.0 - exp(-max(col, vec3f(0.0)) * EXPOSURE);
  let energy = max(max(mapped.r, mapped.g), mapped.b);
  if (energy <= 0.0) {
    return vec4f(0.0);
  }
  // Hue in rgb, energy in alpha: the composite does rgb·a + background·(1−a),
  // so this reads as additive light while embers still let the sky through.
  return vec4f(mapped / energy, energy);
}

// Climb, then burst. One shell is only ever doing one of the two.
fn fwSequence(uv: vec2f, targetPixel: vec2f, startPixel: vec2f, elapsed: f32, seed: vec2f, res: vec2f, k: f32) -> vec3f {
  let climb = (0.9 + fwHash(seed + vec2f(45.0, 12.0)) * 0.18) * CLIMB_SCALE;
  if (elapsed < climb) {
    return fwRocket(uv, startPixel, targetPixel, elapsed / climb, seed, res, k);
  }
  let t = (elapsed - climb) / BURST_TIME;
  if (t < 1.0) {
    return fwBurst(uv, fwToWorld(targetPixel, res), t, seed, k);
  }
  return vec3f(0.0);
}

// ── The climb ──

fn fwRocket(uv: vec2f, startPixel: vec2f, targetPixel: vec2f, t: f32, seed: vec2f, res: vec2f, k: f32) -> vec3f {
  let pos = fwRocketPos(startPixel, targetPixel, t, seed, res);
  // Where it was an instant ago: the streak is the segment between the two, which
  // is the only trail we get in a single pass — the original accumulated one in a
  // feedback buffer.
  let prev = fwRocketPos(startPixel, targetPixel, max(t - CLIMB_TRAIL, 0.0), seed, res);

  let tint = normalize(fwPalette(seed) + vec3f(1.0, 0.72, 0.35));
  // The head needs its own bound for the same reason the streak does: at this
  // amplitude a bare 1/r stays above one code value two whole screen-heights
  // out, so the rocket dragged a frame-sized smudge around with it. Bounded, it
  // is a point with a small halo.
  // Both endpoints are PROJECTED, so the streak's length and slope already
  // carry the perspective; only the widths need the shell's scale.
  let hd = length(uv - pos) / k;
  let head = (0.0022 / (hd + 0.0011) * smoothstep(HEAD_HALO, 0.0, hd) +
              exp(-hd * hd * 55000.0) * 1.7);

  // The streak gets a hard core as well as a glow, for the same reason the
  // sparks do: a glow alone is a smudge, and it is the core that reads as a
  // line. `age` is 0 at the far end, so the streak thins and dims backwards
  // instead of ending square.
  var arc = fwArc(uv, prev, pos);
  arc.x = arc.x / k;
  // Holds full strength for the first third of the climb rather than starting to
  // go immediately, so the streak is still legible when the shell is high.
  let fadeIn = smoothstep(1.3, 0.35, t);
  let taper = mix(0.55, 1.0, arc.y);
  // The halo is faded to nothing at CLIMB_HALO, so the streak has an actual
  // edge; the core carries the light. Roughly three pixels wide at 1080p, which
  // is a wire rather than a band.
  let glow = 0.0014 / (arc.x + 0.0016) * smoothstep(CLIMB_HALO, 0.0, arc.x) *
             fadeIn * taper * CLIMB_GLOW;
  // 260000 against the head's 55000: the streak's falloff is a bit over half the
  // head's width, so the rocket reads as a point DRAGGING a wire rather than a
  // wire with a bulge in it. They were within 10% of each other before.
  let core = exp(-arc.x * arc.x * 260000.0) * 2.2 * fadeIn * taper * CLIMB_GLOW;

  return tint * head + mix(tint, vec3f(1.0, 0.38, 0.12), 0.35) * (glow + core);
}

fn fwRocketPos(startPixel: vec2f, targetPixel: vec2f, t: f32, seed: vec2f, res: vec2f) -> vec2f {
  // Eased so it leaves fast and arrives slow, the way a shell running out of
  // thrust does. A FOURTH power, not a cube: it puts the rocket within a fraction
  // of a percent of its apex by four-fifths of the climb, so the last stretch is
  // a hang rather than an arrival. Without that pause the burst reads as going
  // off mid-flight — there is no moment of being at the top to go off AT.
  let x = clamp(t, 0.0, 1.0);
  let inv = 1.0 - x;
  let inv2 = inv * inv;
  var p = mix(fwToWorld(startPixel, res), fwToWorld(targetPixel, res), 1.0 - inv2 * inv2);
  // A little lateral bow, biggest mid-flight.
  p.x += sin(t * 3.14159265359) * (fwHash(seed + vec2f(17.0, 91.0)) - 0.5) * 0.055;
  return p;
}

// ── The burst ──

fn fwBurst(uvIn: vec2f, center: vec2f, t: f32, seed: vec2f, k: f32) -> vec3f {
  // The shell's whole private frame, scaled about its own centre: every tuned
  // distance below — velocities, reaches, glow radii — lands k times larger on
  // screen, which is all "nearer" means to a billboard burst.
  let uv = center + (uvIn - center) / k;
  // Nothing in this shell reaches farther than its fastest spark, plus the
  // crackles it throws, plus the distance a glow stays visible. Outside that
  // circle the loop below cannot contribute, so it is skipped whole — and a young
  // shell is nearly a point, which is when the saving is largest.
  let rel = uv - center;
  // Not every shell is the same shell: the small ones read as further away and
  // give the big ones something to be big against.
  let boom = BOOM * mix(SMALLEST, 1.0, fwHash(seed + vec2f(5.5, 2.2)));

  // The opening flash is a 1/r glow, so it reaches much further than the sparks
  // early on and nothing at all once it has decayed — its radius is derived
  // rather than guessed: the distance at which it falls under one code value.
  let flashAmp = exp(-t * 18.0) * 0.010;
  let flashReach = max(flashAmp / VISIBLE, 1e-4);

  // ONE test for the whole shell, against whichever of the two is currently
  // larger. Returning here costs a pixel nothing at all — not even the palette,
  // which is four hashes and two normalises that most pixels have no use for.
  let reach = max(flashReach,
                  (1.02 * t + 0.16 + 0.5 * GRAVITY * t * t + CHILD_GRAVITY) * boom + GLOW_REACH);
  if (dot(rel, rel) > reach * reach) {
    return vec3f(0.0);
  }

  let tint = fwPalette(seed);
  let flashDistance = length(rel);
  var col = normalize(tint + vec3f(1.0)) * flashAmp / (flashDistance + 0.0025) *
            smoothstep(flashReach, 0.0, flashDistance);

  let emberTint = normalize(tint + vec3f(0.9, 0.55, 0.25));
  let hotTint = normalize(tint + vec3f(1.0, 0.62, 0.22));
  // A tenth of shells burst with no streaks at all, which is what keeps the sky
  // from reading as one firework repeated.
  let trails = fwHash(seed + vec2f(101.7, 53.3)) >= PLAIN_SHELLS;
  // The streak's span in burst-time: one unit of t is BURST_TIME/SPEED seconds.
  let trailLen = select(0.0, TRAIL_SECONDS * SPEED / BURST_TIME, trails);
  let hSeed = fwHash(seed);

  let fade = smoothstep(0.82, 0.08, t);
  // The streaked layer outlives the sharp one and ramps in over the first frames
  // rather than arriving whole — both straight from the reference's second pass.
  let trailFade = smoothstep(0.95, 0.08, t) * smoothstep(0.0, 0.06, t);
  let crackleT = clamp((t - CRACKLE_AT) / (1.0 - CRACKLE_AT), 0.0, 1.0);

  // Radii for the rejects in the loop, hoisted — the same for every spark in the
  // shell. A spark's arc is no longer than its top speed plus gravity over the
  // streak's span; its embers stay within their own scatter.
  let sparkReach = (1.02 + GRAVITY) * boom * trailLen + GLOW_REACH;
  let emberReach = (0.16 + CHILD_GRAVITY) * boom + GLOW_REACH;

  for (var particle = 0; particle < PARTICLES; particle++) {
    let i = f32(particle);
    // An even fan, jittered, so the shell is round without being a stencil.
    let angle = i * (TAU / f32(PARTICLES)) + (fwHash(seed + vec2f(i, 13.1)) - 0.5) * 0.22;
    let velocity = vec2f(cos(angle), sin(angle)) * ((0.3 + 0.72 * fwHash(vec2f(i, hSeed))) * boom);

    let pos = fwBallistic(center, velocity, t, boom);

    // A pixel further from the spark than its whole arc plus a glow's reach
    // cannot be lit by any part of it. Skipping here skips the second ballistic,
    // the segment solve and all the shading — and a shell's sparks fan out over
    // ten times a glow's radius, so nearly all of them miss any given pixel.
    let toSpark = uv - pos;
    if (dot(toSpark, toSpark) <= sparkReach * sparkReach) {
      let sparkle = sin(t * 58.0 + i + hSeed * 12.0) * 0.5 + 0.5;

      // ── The sharp spark. Every one of them, drawn at where it IS. Gated on its
      // OWN radius, which is tighter than the reject above: that one has to cover
      // the streak's whole length, so between the two radii this maths would run
      // only to be multiplied by a window of zero.
      let d = length(toSpark);
      if (d < GLOW_REACH) {
        let size = 0.00125 * fade * (0.55 + 0.45 * sparkle);
        let glow = size / (d + 0.00062) * smoothstep(GLOW_REACH, 0.0, d);
        let core = exp(-d * d * 120000.0) * 1.35 * fade * (0.75 + 0.25 * sparkle);
        col += tint * (glow * 0.78 + core);
      }

      // ── The streak. EVERY OTHER spark, exactly as the reference's second pass
      // stepped its particle index by two: half the shell streaks and half stays
      // sharp, and it is the mixture that reads as fireworks rather than as a
      // net. Drawing it on all of them was what made the sky look woven.
      if (trails && (particle & 1) == 0) {
        let prev = fwBallistic(center, velocity, max(t - trailLen, 0.0), boom);
        // .x = distance to the arc, .y = where along it — 0 at the tail, 1 at
        // the spark. The second is the streak's own age, in seconds.
        let arc = fwArc(uv, prev, pos);
        let td = arc.x;
        if (td < GLOW_REACH) {
          // The decay the buffer applied over frames, applied over LENGTH
          // instead. Per channel, so the tail goes orange on its own rather than
          // being tinted orange by hand.
          let held = exp(-TRAIL_DECAY * ((1.0 - arc.y) * TRAIL_SECONDS));
          let tSize = 0.00105 * trailFade * (0.6 + 0.4 * sparkle);
          let tGlow = tSize / (td + 0.00072) * smoothstep(GLOW_REACH, 0.0, td);
          let tCore = exp(-td * td * 95000.0) * trailFade * (0.65 + 0.35 * sparkle);
          col += mix(tint, hotTint, 0.28 + sparkle * 0.2) * held *
                 (tGlow * 0.52 + tCore * 0.46) * TRAIL_MIX;
        }
      }
    }

    // Roughly two in five sparks crackle into a handful of embers. Tested
    // separately from the spark above, and NOT skipped with it: a crackle goes
    // off where the spark was at CRACKLE_AT, which by now is somewhere else
    // entirely.
    if (crackleT > 0.0 && fwHash(seed + vec2f(i, 71.7)) >= 0.58) {
      let origin = fwBallistic(center, velocity, CRACKLE_AT, boom);
      let toOrigin = uv - origin;
      if (dot(toOrigin, toOrigin) > emberReach * emberReach) {
        continue;
      }
      let flashDist = length(toOrigin);
      col += emberTint * exp(-crackleT * 20.0) * 0.0032 / (flashDist + 0.0018) *
             smoothstep(GLOW_REACH, 0.0, flashDist);

      for (var child = 0; child < SECONDARY; child++) {
        let j = f32(child);
        let cSeed = fwHash(seed + vec2f(i * 17.0, j * 23.0));
        let cAngle = TAU * (j / f32(SECONDARY) + cSeed * 0.35);
        let cVel = vec2f(cos(cAngle), sin(cAngle)) *
                   ((0.06 + 0.10 * fwHash(seed + vec2f(j * 9.0, i * 5.0))) * boom);

        var cPos = origin + cVel * crackleT;
        cPos.y -= CHILD_GRAVITY * boom * crackleT * crackleT;

        let cd = length(uv - cPos);
        let cFade = smoothstep(1.0, 0.05, crackleT);
        col += mix(emberTint, vec3f(1.0, 0.82, 0.42), cSeed * 0.45) *
               ((0.00088 * cFade) / (cd + 0.00062) * 0.75 * smoothstep(GLOW_REACH, 0.0, cd) +
                exp(-cd * cd * 150000.0) * 0.85 * cFade);
      }
    }
  }

  return col;
}

fn fwBallistic(center: vec2f, velocity: vec2f, t: f32, boom: f32) -> vec2f {
  // Gravity scales with the shell alongside the velocity it fights. Throwing the
  // sparks half again as far while they still drop the same 0.15 flattens every
  // arc into a straight fan — the trajectory has to grow with the shell, or the
  // shell stops falling.
  return center + velocity * t - vec2f(0.0, 0.5 * GRAVITY * boom * t * t);
}

// ── Toolbox ──

/**
 * Where shell (slot, cycle) bursts — a WORLD point behind the stage.
 *
 * The display used to be composed in screen space, which is why it moved with
 * the camera: it was wallpaper. Anchored in the world and projected each frame
 * (rzProject), the shells stay put while you orbit, sit behind the character
 * because the background mount composites behind the scene, and gain parallax
 * for free — a shell farther down the line is smaller because it IS farther.
 */
fn fwWorldTarget(slot: f32, cycle: f32) -> vec3f {
  let seed = vec2f(slot * 37.17 + cycle * 11.31, cycle * 23.73 + slot * 5.91);
  return vec3f(
    mix(-WORLD_SPAN, WORLD_SPAN, fwHash(seed + vec2f(1.0, 0.0))),
    mix(WORLD_LOW, WORLD_HIGH, fwHash(seed + vec2f(0.0, 1.0))),
    WORLD_Z + (fwHash(seed + vec2f(2.0, 7.0)) - 0.5) * 14.0,
  );
}

/** Pixels → the square, y-normalised space everything above is tuned in. */
fn fwToWorld(pixel: vec2f, res: vec2f) -> vec2f {
  return (pixel * 2.0 - res) / max(res.y, 1.0);
}

fn fwPalette(seed: vec2f) -> vec3f {
  // A pyrotechnician's palette, not noise. Random RGB pulled toward warm or
  // cool and normalised always averaged out to pastel — every shell the same
  // washed grey-gold. Real shows read as NAMED colours: strontium red, sodium
  // gold, barium green, copper blue... eight of them, picked per shell, with a
  // small wobble so two reds are sisters rather than twins.
  let pick = i32(fwHash(seed + vec2f(11.3, 0.7)) * 7.99);
  var c = vec3f(1.0, 0.92, 0.80);                  // 7: white-gold
  if (pick == 0) { c = vec3f(1.0, 0.22, 0.20); }   // strontium red
  if (pick == 1) { c = vec3f(1.0, 0.62, 0.16); }   // sodium gold
  if (pick == 2) { c = vec3f(1.0, 0.92, 0.35); }   // champagne
  if (pick == 3) { c = vec3f(0.30, 1.0, 0.42); }   // barium green
  if (pick == 4) { c = vec3f(0.25, 0.85, 1.0); }   // cyan
  if (pick == 5) { c = vec3f(0.35, 0.45, 1.0); }   // copper blue
  if (pick == 6) { c = vec3f(1.0, 0.35, 0.75); }   // magenta
  let wob = (fwHash(seed + vec2f(2.1, 19.8)) - 0.5) * 0.18;
  return normalize(c + vec3f(wob, -wob * 0.5, wob * 0.3));
}

/** Distance to the segment ab — the rocket's streak. */
fn fwSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  return fwArc(p, a, b).x;
}

/** Distance to segment ab, and how far along it the nearest point sits (0 at a,
 *  1 at b). A spark's streak needs both: the distance draws it, the position
 *  along it ages it. */
fn fwArc(p: vec2f, a: vec2f, b: vec2f) -> vec2f {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.00001), 0.0, 1.0);
  return vec2f(length(pa - ba * h), h);
}

fn fwHash(p0: vec2f) -> f32 {
  var p = fract(p0 * vec2f(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

// ── The light a burst throws ─────────────────────────────────────────────────
//
// This is the case the lightEmit mount was built for. A shell's burst position
// is already a closed form in this file — fwWorldTarget(slot, cycle) — so the
// light re-evaluates the SAME expression the picture is drawn from. Nothing is
// mirrored on the CPU, so there is no second derivation to drift, and an
// offline export stepped at another rate gets the same answer as playback.
//
// Four lights against ten shells, because a slot only qualifies while it is
// BURSTING: 1.55s of a 5s period, so four slots average a bit over one lit at
// a time and peak at four. Ten lights would be ten slots of budget for the
// same picture.
//
// The climb throws nothing. A rocket is a spark against the sky; it is the
// burst that lights a field, and spending a slot on the climb would keep a
// light alive for the two-thirds of the sequence when it is doing nothing.
// @lights 4
const FLASH_PEAK = 2.2;    // brightness at the burst itself
const FLASH_REACH = 90.0;  // world units. The shells sit WORLD_Z behind the
                           // stage and WORLD_LOW..HIGH up, so the stage is ~50
                           // units away — a reach under that lights nothing
const FLASH_DECAY = 4.0;   // how fast the flash falls off across the burst

fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  // Dark by default, and every early return leaves it that way: a slot between
  // bursts is not a light at zero brightness sitting somewhere, it is no light.
  l.pos = vec3f(0.0, WORLD_LOW, WORLD_Z);
  l.color = vec3f(1.0);
  l.intensity = 0.0;
  l.radius = 1.0;

  // The same clock the picture runs on, including SPEED — slowing the display
  // has to slow the light with it or they come apart.
  let t = time * SPEED;
  let slot = f32(i);
  let offset = slot * STAGGER;
  if (t < offset) { return l; }

  // floor() names the launch, exactly as background() does, so this light is
  // reading the same shell the same way — same cycle, same seed, same colour.
  let cycle = floor((t - offset) / PERIOD);
  let elapsed = t - (cycle * PERIOD + offset);
  let climb = SEQUENCE - BURST_TIME;
  if (elapsed < climb || elapsed >= SEQUENCE) { return l; }

  let seed = vec2f(slot * 91.7 + cycle * 17.3, cycle * 41.9 + slot * 8.2);
  l.pos = fwWorldTarget(slot, cycle);
  l.color = fwPalette(seed);
  l.radius = FLASH_REACH;

  // A burst is a FLASH: almost all of the light in the first fraction, then a
  // fall. exp() alone never reaches zero, so the tail is faded out over the
  // last of the burst — a light still lit when its sparks have gone is a lamp
  // hanging in the sky.
  let bt = (elapsed - climb) / BURST_TIME;
  l.intensity = FLASH_PEAK * exp(-bt * FLASH_DECAY) * (1.0 - smoothstep(0.8, 1.0, bt));
  return l;
}
