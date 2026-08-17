// @grid 768
// @fullres
// @anchor 左足首
// @anchor 右足首

// Dry Ice — David Gallardo (xjorma)'s heightfield fog, ported whole.
// CC BY-NC-SA 3.0. Common + Buffer A + B/C/D + Image, all four.
//
// The earlier port took Buffer A and the Image pass and rewrote the rest from
// memory of what a fluid needs. This one is the arithmetic, converted rather
// than reinterpreted — and the conversion is the interesting part, because the
// original is written entirely in PIXELS AND FRAMES:
//
//   velocity is pixels per frame; it backtraces by `dissipation * velocity`
//   pixels; density decays by a factor per frame; the noise is sampled in a uv
//   normalised by the viewport HEIGHT.
//
// None of those units exist here. A grid is a fixed number of texels whatever
// the window is, and dt is whatever the machine managed. So every constant
// below is the original's number times the conversion that makes it mean the
// same thing:
//
//   per frame   → per second, × 60
//   per pixel   → per FIELD (0..1 across the grid), ÷ 288
//
// 288 because that is the short side of the picture this was matched against.
// The original's look genuinely depends on its resolution — at 1080p the same
// source runs a slower, finer fluid, because a velocity of "one pixel" is a
// smaller fraction of the screen. Pinning the conversion to one resolution is
// what makes the port resolution-INDEPENDENT: raising @grid below now buys
// sharpness and changes nothing else, which is the property you want and the
// original does not have.

// Tunables — every const below is one, grouped by what it shapes. Edit and ⌘⏎.
// ── Where it lies, and how big ────────────────────────────────────────────────
//
// The original ties every length to one: fog 0.24 deep across a field 2 wide,
// emitters reaching 0.12, the light at 1.0. Held to exactly, on a stage, that
// gives a cloud sea — because the thing it is scaled around is a ball 3% of the
// field across, and a dancer is 12% of it and has to stay visible standing in
// the stuff. So the pool's WIDTH and the fog's DEPTH are set apart here, and
// what stays tied to the depth is everything that decides how the fog LOOKS:
// the light's height above it, and the size of the noise that shapes it.
//
// Read the three together. Widening the pool without moving the other two is
// what turns billows into hills.
const SIM_AT = vec2f(0.0, 0.0);   // the pool's centre, in world x/z
const SIM_SPAN = 160.0;           // and its width — the default ground, covered
const FLOOR_Y = 0.0;
// Thigh-deep on a model that stands ~20, which is where dry ice sits. Not a
// share of the span: at the original's 12% this would be 19 and she would be in
// it to the chest.
const HEIGHT = 7.0;
// The lamp, at the original's 4.17 fog-heights up (its 1.0 over a fog 0.24
// deep). Tied to HEIGHT and NOT to the span, and that is the whole difference
// between a lit pool and a white sheet: hung at half the span instead — 80 units
// for this pool — it lights everything from so far above that the 1/d² is flat
// across the whole floor and every billow shadow is too short to see.
const LIGHT_Y = FLOOR_Y + HEIGHT * 4.17;
// A foot. Absolute, because a foot does not get bigger when the stage does —
// this is the one length the original scales that has a real size here.
const KICK_R = 2.6;

// ── The fluid (Buffer A) ─────────────────────────────────────────────────────
//
// Velocity is in FIELDS PER SECOND: 1.0 crosses the pool in a second. Density is
// 0..1, the fraction of HEIGHT the fog reaches here.
const ADVECT = 0.95;      // `dissipation` — the backtrace falls 5% short on purpose
// The two noise fields, in cycles across the pool. The original samples them at
// uv*40 and uv*1.5 over a uv spanning 2, so: 80 and 3.
//
// The coarse one is the workhorse, which reads as the wrong way round until you
// count texels. At 3 cycles its cells are a third of the pool, and seven octaves
// carry it down to about four of them — so it alone spans everything from a bank
// drifting through to the fray on a single billow. The fine one is grain: its
// base cell is a hundredth of the pool and its finest is well under a texel, and
// its job is to give advection something to stretch, not to be seen.
// The two stirring rates are the one place the conversion bites twice, and
// getting it wrong is invisible in the arithmetic and obvious on screen. The
// original ADDS 0.2·noise to a velocity already measured per frame, so it is an
// acceleration: fields per second, per second. Both 60s apply — one for the
// velocity's own unit and one for the rate at which it is handed out. Convert it
// as a velocity, the way it reads, and the fluid comes out sixty times too
// slack; it still churns, because the noise fields keep moving underneath it,
// which is exactly what makes the mistake survive a look.
// Sized against the FOG'S DEPTH, not the pool's width. The original's coarse
// cells are a third of its field, which is 2.8 times its fog depth — hold that
// ratio and the billows stay billow-shaped at any pool size. Held to a share of
// the span instead, this pool's cells would be 53 units across against a fog 7
// deep, and a lump eight times wider than it is tall does not read as a billow,
// it reads as a hill.
const PUFF = SIM_SPAN / (HEIGHT * 2.8);   // ≈ 8 cycles across the pool
const PUFF_HZ = 0.05;
const PUFF_V = 1.25;      // 0.1 px added per frame
const PUFF_D = 2.40;      // 0.04 /frame
const PUFF_OCT = 7;
// The fine field is GRAIN, and its octaves stop at the texel on purpose.
//
// The original runs seven here too, and at its resolution the bottom four are
// sub-pixel and read as film grain. On this grid they are sub-TEXEL, which is a
// different thing entirely: a sub-texel octave cannot be advected, because every
// frame re-randomises it rather than carrying it, so it is not grain that flows
// — it is fresh white noise stirred into density sixty times a second, and it
// lands on screen as wet sand poured over the fog. Three octaves puts the
// finest at about a texel and stops.
const WISP = PUFF * 26.7;   // the original's 40-to-1.5 ratio
const WISP_HZ = 0.25;
const WISP_V = 2.50;      // 0.2 px added per frame → fields/s per second
const WISP_D = 0.60;      // 0.01 /frame — a density, so only one 60
const WISP_OCT = 3;
// How hard the whole thing is stirred, against the original's own figures above.
//
// Theirs is a shadertoy: a field two units wide that the eye takes in at once,
// where fog crossing it in six seconds reads as lively. The same number over a
// stage is the pool sliding past at 26 units a second, which is faster than the
// dancer can run. Dry ice creeps.
const FLOW = 0.20;
const DECAY = 0.60;       // ×0.99 a frame is exp(-0.6·t)
// The rim, and it is the original's screen vignette rather than an invention:
// density and velocity are multiplied by 0.98 at the border of the buffer and by
// 1.0 at its centre, every frame. Two percent is nothing per frame and
// everything in the steady state — it is what makes the fog a POOL with a soft
// edge instead of a square that stops. Measured radially here, because a square
// grid faded on a square profile announces its own corners.
const EDGE = 1.2;         // ×0.98 a frame
// And then an actual END, which the original has no need of: its buffer edge IS
// its floor slab's edge, so the straight cut where its fog stops is hidden by
// the scenery. This pool lies on a floor of the user's own size, so it has to
// run out on its own — inside the grid, and before any corner of it can show.
// A radius in field widths, frayed so the boundary is not a circle somebody
// drew. The fray stays SMALL on purpose: the coarse field has three cycles
// across the pool, so a large one would not fray the rim, it would give the pool
// three lobes and turn it into a shape.
const EDGE_FRAY = 0.05;
const EDGE_FADE = 0.16;
// ── Her feet ──
//
// The original's spheres do three things, and only the first is obvious: they
// SUBTRACT density (a hole is a hole, not fog pushed aside), they hand the fluid
// a little of their own motion, and they do it whether they are moving or not.
const KICK_CLEAR = 9.0;   // 0.15 /frame at the centre of the reach
// How fast fog under a sole is brought up to the sole's own speed — a rate, so
// half a second of contact gets it most of the way there. The original writes
// this as `velocity -= ballVelocity * 5` per frame, which reads like a hard
// shove and converts to 2.1 a second, because the 5 multiplies a PER-FRAME
// displacement before landing in a per-frame velocity.
const KICK_DRAG = 2.1;
// How far below 足首 the sole is. The bone sits at the ankle, and the height that
// decides whether a foot is in the fog is the sole's.
const FOOT_DROP = 1.3;
const FOOT_SOFT = 1.5;    // and over what height the reach fades out

// ── The picture (Image) ──────────────────────────────────────────────────────
//
// A volume march, and at every step of it a second march toward the light. What
// this accumulates is scattered light and transmittance; there is no surface and
// no normal anywhere in it. That is the whole reason the original's fog has
// depth — you are seeing INTO it, and the dark is fog standing in the lee of
// fog, which is a fact about the neighbourhood and not about the slope underfoot.
const SLICES = 24;            // the original's nbSlice, exactly
const SHADOW_SLICES = 12;     // half of it, at twice the stride — same path
// Optical depth through the FULL height: fogDensity 20 and shadowDensity 25 over
// a fog 0.24 deep. Stated as the product so they survive a change to HEIGHT.
const FOG_DENSITY = 4.8;
const SHADOW_DENSITY = 6.0;
// ONE POINT LIGHT, hanging over the cast, and this is the composition of the
// original: a lamp above the emitter, everything falling off as 1/d² around it.
// The pool is bright where she is and sinks toward black at its edges, billows
// throw shadows that lengthen outward, and none of that is available to a
// directional light — which is what the previous version used, and why its fog
// read as one flat grey sheet however much relief was in the field.
//
// Bounded by construction: the lamp hangs half a pool-width above the floor, so
// d² never approaches zero and the 1/d² never blows up.
const AMBIENT = 0.0;          // the original gives shadowed fog nothing at all
const MAX_FOG = 1.0;          // and lets it close over the emitters completely
const MIN_SLOPE = 0.12;       // flattest ray the march will step along; the
                              // original refuses anything flatter outright, and
                              // a dance camera is flatter than that most of the
                              // time, so it is clamped rather than rejected

/**
 * SINE-FREE, and it has to be.
 *
 * The usual sine hash works beautifully near the origin and falls apart away
 * from it: a float32 sine of a six-figure argument has no meaningful fractional
 * part left, and the hash stops being random and becomes an interference
 * pattern — concentric rings and oil-slick swirls sliding under everything. The
 * original has exactly that (its `fract(sin(h)*43758.5)` at seven octaves), and
 * at its resolution the artefact is sub-pixel and reads as grain.
 */
fn dfHash(p: vec3f) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}

fn dfNoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = mix(mix(dfHash(i), dfHash(i + vec3f(1.0, 0.0, 0.0)), u.x),
              mix(dfHash(i + vec3f(0.0, 1.0, 0.0)), dfHash(i + vec3f(1.0, 1.0, 0.0)), u.x), u.y);
  let b = mix(mix(dfHash(i + vec3f(0.0, 0.0, 1.0)), dfHash(i + vec3f(1.0, 0.0, 1.0)), u.x),
              mix(dfHash(i + vec3f(0.0, 1.0, 1.0)), dfHash(i + vec3f(1.0, 1.0, 1.0)), u.x), u.y);
  return mix(a, b, u.z);
}

/**
 * TWO channels out of one walk, which is the original's `fbm` and is not a
 * saving. A vec2 of noise is a DIRECTION to push in; a scalar could only ever
 * have been a strength, and a field stirred by a scalar has no curl in it.
 *
 * Time doubles with space, octave by octave, exactly as the original's
 * `p = p*2.0 + shift` does to all three components — so every octave drifts at
 * the same rate measured in its own cells, and the field boils self-similarly
 * instead of having one scale crawl while another flickers. Capped at 8× so the
 * argument stays inside float32's useful range over a long take; past that the
 * octaves are well under a texel and their rate cannot be seen anyway.
 */
fn dfFbm2(p: vec2f, t: f32, octaves: i32) -> vec2f {
  var acc = vec2f(0.0);
  var amp = 0.5;
  var total = 0.0;
  var q = p;
  var tz = t;
  for (var i = 0; i < octaves; i++) {
    acc += vec2f(dfNoise(vec3f(q, tz)), dfNoise(vec3f(q, tz + 10.0))) * amp;
    total += amp;
    q = q * 2.0 + vec2f(100.0);
    tz = select(tz, tz * 2.0, i < 3) + 100.0;
    amp *= 0.5;
  }
  // Centred on its OWN mean, where the original subtracts a flat 0.5.
  //
  // An fbm of k octaves averages 0.5·(1 - 2⁻ᵏ), not 0.5, so subtracting 0.5
  // leaves every sample biased low by the same amount everywhere — and a
  // constant added to a VELOCITY field every frame is a wind. Nothing removes
  // it either: the pressure solve only takes out what has divergence, and a
  // uniform flow has none. At seven octaves the bias is 0.4% and the original
  // rides it (it is a good half of where its steady drift comes from); at the
  // four this file wants for the grain it would be 3%, and the pool would simply
  // sail off downwind. Centring it properly makes the octave count free, and
  // leaves the drift to FLOW, where it can be read.
  return acc - vec2f(0.5 * total);
}

/** Grid uv to world x/z, and back. The engine imposes no mapping — this is it. */
fn dfWorld(uv: vec2f) -> vec2f { return SIM_AT + (uv - vec2f(0.5)) * SIM_SPAN; }
fn dfUV(xz: vec2f) -> vec2f { return (xz - SIM_AT) / SIM_SPAN + vec2f(0.5); }

/**
 * The field with the pressure gradient already taken out of it — the original's
 * `sampleMinusGradient`, and the one line the previous port dropped.
 *
 * It matters where it is CALLED, not just that it exists. Advection reads
 * through this at the point the parcel came FROM, so what gets carried forward
 * and stored is the corrected velocity. Reading the raw field there instead —
 * and correcting only the direction of the backtrace, which is the natural
 * mistake — leaves the divergent half of the flow in the grid permanently. The
 * solve then spends every frame fighting a component that never leaves, the
 * pressure climbs without bound, and the fluid never circulates.
 */
fn dfMinusGrad(uv: vec2f, te: f32) -> vec3f {
  let c = rzGridPrev(uv);
  let l = rzGridPrev(uv - vec2f(te, 0.0)).w;
  let r = rzGridPrev(uv + vec2f(te, 0.0)).w;
  let d = rzGridPrev(uv - vec2f(0.0, te)).w;
  let u = rzGridPrev(uv + vec2f(0.0, te)).w;
  return vec3f(c.xy - vec2f(r - l, u - d) * 0.5, c.z);
}

fn gridStep(uv: vec2f, prev: vec4f, dt: f32) -> vec4f {
  // Starts EMPTY. There is nothing to seed: injection fills the pool within a
  // couple of seconds and then never stops, and what you look at is the balance
  // between that and the decay, not a bank that was placed there.
  if (rzGridFrame() == 0) { return vec4f(0.0); }

  let te = rzGridTexel();
  let t = rzTime();
  let sL = rzGridPrev(uv - vec2f(te, 0.0));
  let sR = rzGridPrev(uv + vec2f(te, 0.0));
  let sD = rzGridPrev(uv - vec2f(0.0, te));
  let sU = rzGridPrev(uv + vec2f(0.0, te));

  // ── Project, then advect backwards, both through dfMinusGrad ──
  let here = prev.xy - vec2f(sR.w - sL.w, sU.w - sD.w) * 0.5;
  let carried = dfMinusGrad(uv - here * (dt * ADVECT), te);
  var vel = carried.xy;
  var density = carried.z;

  // ── Injection, two scales, into velocity AND density ──
  //
  // Raw fbm, not its curl. Curl noise is divergence-free by construction and
  // looks like the obvious improvement — it is how the previous port avoided
  // needing a real solve — but it also means every eddy is one the noise
  // function decided on. Half of what goes in here is divergent, and the
  // pressure solve is what turns that half into circulation that answers to the
  // fog already present: fluid piling into a bank has to go somewhere, and where
  // it goes is around.
  let wispN = dfFbm2(uv * WISP, t * WISP_HZ, WISP_OCT);
  let puffN = dfFbm2(uv * PUFF, t * PUFF_HZ, PUFF_OCT);
  vel += (wispN * WISP_V + puffN * PUFF_V) * (FLOW * dt);

  // WHERE fog is made — the end of the pool, applied to the INJECTION and not as
  // one more decay. A mask multiplied into density every frame compounds: it
  // lands in the steady state as inject/(1 - keep·mask), so halving it does not
  // halve the fog, it divides it by fifty. Injecting nothing past the brink just
  // means no fog is made there, which is the thing that was meant.
  // Written as 1 - smoothstep and not as smoothstep with its edges swapped:
  // WGSL leaves smoothstep undefined when low >= high, and it is the kind of
  // undefined that works on the machine you wrote it on.
  let brink = 0.5 + EDGE_FRAY * puffN.x;
  let made = 1.0 - smoothstep(brink - EDGE_FADE, brink, length(uv - vec2f(0.5)));
  density += (length(wispN) * WISP_D + length(puffN) * PUFF_D) * (dt * made);

  // ── Her feet ──
  let world = dfWorld(uv);
  for (var c = 0; c < rzSubjectCount(); c++) {
    for (var slot = 0; slot < 2; slot++) {
      let foot = rzAnchor(c, slot);
      if (!foot.valid) { continue; }
      let dist = distance(world, foot.pos.xz);
      if (dist > KICK_R) { continue; }
      // IS THE SOLE IN THE FOG. The original never asks, because its emitters
      // roll on the floor and cannot be anywhere else; a dancer's foot spends
      // half its time in the air, and without this a kick thrown at head height
      // carves the floor as readily as a step does. Measured against the LOCAL
      // fog top, so a sole clears deep bank and passes over a thin patch
      // untouched.
      let immersion = clamp(
        ((FLOOR_Y + density * HEIGHT) - (foot.pos.y - FOOT_DROP)) / FOOT_SOFT, 0.0, 1.0);
      if (immersion <= 0.0) { continue; }
      let f = (KICK_R - dist) / KICK_R * immersion;
      density -= f * (KICK_CLEAR * dt);
      vel = mix(vel, foot.vel.xz / SIM_SPAN, clamp(f * KICK_DRAG * dt, 0.0, 1.0));
    }
  }

  density = clamp(density, 0.0, 1.0);

  // ── Decay, and the rim ──
  //
  // The decay is DENSITY'S ALONE. Velocity is damped by nothing here and nothing
  // anywhere else — the original never damps it either, and what bounds it is
  // the projection above plus the 5% the backtrace falls short. Handing velocity
  // the density's decay as well is the quiet way to kill this effect: the fluid
  // then loses in half a second whatever circulation it built, everything reads
  // as blobs drifting, and the field looks noise-driven because effectively it is.
  let off = uv - vec2f(0.5);
  let bowl = clamp(1.0 - dot(off, off) * 4.0, 0.0, 1.0);  // 1 at the centre, 0 at the rim
  let rim = exp(-EDGE * (1.0 - bowl) * dt);
  density *= exp(-DECAY * dt) * rim;
  vel *= rim;

  // ── One Jacobi sweep toward incompressible ──
  //
  // The original spends about twenty of these a frame, spread over three buffers
  // and a pair of hand-generated 121-tap kernels that are ten sweeps each. One
  // sweep here, over a grid that keeps its pressure between frames, is sixty a
  // second chasing a field that turns over in one — the same destination,
  // several frames behind.
  //
  // Full step, not the half step the previous version took. Under-relaxation was
  // put in to damp a texel-scale checkerboard, which it does by damping
  // everything; the checkerboard was the pressure field growing without bound
  // against divergence that advection was quietly putting back every frame, and
  // that is fixed above, at its cause.
  let divergence = ((sR.x - sL.x) + (sU.y - sD.y)) * 0.5;
  let pressure = (sL.w + sR.w + sD.w + sU.w - divergence) * 0.25;

  return vec4f(vel, density, pressure);
}

/** The top of the fog over a point of floor, in world units. Flat off the pool. */
fn dfFogTop(xz: vec2f) -> f32 {
  let g = dfUV(xz);
  if (g.x < 0.0 || g.x > 1.0 || g.y < 0.0 || g.y > 1.0) { return FLOOR_Y; }
  return FLOOR_Y + rzGrid(g).z * HEIGHT;
}

/**
 * How much of the lamp reaches a point inside the fog — the original's inner
 * loop, and the entire reason the picture has shape in it.
 *
 * Steps rise by one shadow slice each, so the march clears the full depth in
 * SHADOW_SLICES of them however shallow the light lies. Sizing the step by the
 * VIEW slice instead and then taking fewer of them, which is what the previous
 * version did, marches a third of the way up and stops: everything comes back
 * barely shadowed, and the fog renders flat.
 */
fn dfShadow(pos: vec3f, lamp: vec3f, jitter: f32) -> f32 {
  let toLamp = lamp - pos;
  let dist2 = dot(toLamp, toLamp);
  let L = toLamp / sqrt(dist2);
  let rise = HEIGHT / f32(SHADOW_SLICES);
  let step = L * (rise / max(L.y, 0.15));
  let per = length(step) / rise;
  var p = pos + step * jitter;
  var through = 0.0;
  for (var i = 0; i < SHADOW_SLICES; i++) {
    p += step;
    if (p.y > FLOOR_Y + HEIGHT) { break; }
    through += clamp(dfFogTop(p.xz) - p.y, 0.0, rise) * per;
  }
  // 1/d², normalised so the floor directly under the lamp reads exactly 1 — the
  // original's `/lightDist2` with its light one unit up over a field of two.
  return exp(-through * (SHADOW_DENSITY / HEIGHT)) * (LIGHT_Y * LIGHT_Y / dist2);
}

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let ro = rzCameraPos();
  let tScene = depth / max(dot(ray, rzCameraForward()), 1e-3);
  let top = FLOOR_Y + HEIGHT;

  // Enter at the top of the slab, or at the lens when it is already inside.
  var t0 = 0.0;
  if (ro.y > top) {
    if (ray.y > -1e-4) { return vec4f(0.0); }
    t0 = (top - ro.y) / ray.y;
  }
  if (t0 >= tScene) { return vec4f(0.0); }

  // The lamp rides over the cast, as the original's rides over its white ball.
  var lamp = vec3f(SIM_AT.x, LIGHT_Y, SIM_AT.y);
  if (rzSubjectCount() > 0) {
    let hip = rzSubjectHip(0);
    lamp = vec3f(hip.x, LIGHT_Y, hip.z);
  }

  let slice = HEIGHT / f32(SLICES);
  // One slice of RISE per step, so the march covers the depth in SLICES samples
  // however steep the ray is.
  let stepV = slice / max(abs(ray.y), MIN_SLOPE);
  // Jittered, as the original's `fudge` is — a march this coarse lays down
  // visible contour rings without it. Seeded on the PIXEL, which is what
  // `hash12(fragCoord + iTime)` is: seeding on uv scaled by a round number
  // instead leaves neighbouring pixels a fraction apart in the hash's input,
  // which is where a dither stops being a dither. fract keeps it bounded over a
  // long take.
  //
  // HALF a step of it, not the original's whole one. It has more slices and
  // fewer pixels to spread the noise across; at full resolution a whole step of
  // white noise, feeding the shadow march as well as this one, is a speckle over
  // the entire fog rather than a dither along its contours. Half still breaks
  // the rings.
  let jitter = 0.25 + 0.5 * dfHash(vec3f(uv * rzResolution(), fract(time * 7.0)));

  var trans = 1.0;
  var lit = 0.0;
  for (var i = 0; i < SLICES; i++) {
    let t = t0 + (f32(i) + jitter) * stepV;
    if (t > tScene) { break; }
    let p = ro + ray * t;
    if (p.y < FLOOR_Y - slice) { break; }
    let below = dfFogTop(p.xz) - p.y;
    if (below <= 0.0) { continue; }
    // Clamped at one slice, as the original clamps: a sample stands for the fog
    // in ITS slice, not for everything underneath it.
    let dens = min(below, slice) * (stepV / slice) * (FOG_DENSITY / HEIGHT);
    if (dens <= 0.001) { continue; }
    // Beer–Lambert, where the original writes `transmittance *= 1 - density`.
    // That linear form is this one's small-step approximation and is fine at its
    // step sizes; a grazing ray here carries a density past 1 in a single step,
    // and the linear form then returns a NEGATIVE transmittance and the
    // accumulation comes apart.
    let a = 1.0 - exp(-dens);
    lit += (AMBIENT + dfShadow(p, lamp, jitter)) * a * trans;
    trans *= 1.0 - a;
    if (trans < 0.02) { break; }
  }

  let alpha = clamp(1.0 - trans, 0.0, MAX_FOG);
  if (alpha <= 0.004) { return vec4f(0.0); }
  // The colour IS the scattered light, greyscale, with no tint anywhere: dry ice
  // is water vapour, it has no colour of its own, and the only hue it can show
  // is the lamp's. Divided by coverage because this returns a straight-alpha
  // layer — the original folds the two together and its thin fog comes out
  // darker than it should for it.
  //
  // sqrt is the original's own `sqrt(tot)`: the march accumulates a linear
  // radiance and this mount hands back display-space sRGB. Without it the fog
  // arrives about half as bright as the picture it came from.
  let energy = clamp(lit / alpha, 0.0, 1.0);
  return vec4f(vec3f(sqrt(energy)), alpha);
}
