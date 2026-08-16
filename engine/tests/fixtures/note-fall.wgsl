// @layer additive
// @fullres

// Note Fall — the score falling onto a keyboard, behind the cast.
//
// Reads the installed MIDI through rzNote*. With no score loaded it draws the
// line alone and nothing falls, which is a scene waiting for a file rather than
// an error.
//
// A BACKGROUND, not particles. An early version put quads in the scene pass in
// front of the character and they cut across her legs; a background sits behind
// everything by construction, so the cast reads clean and the notes are light
// behind her rather than debris in front. Additive for the same reason — this is
// light, so it adds to whatever backdrop the scene already has.
//
// AND FULL RESOLUTION, which is not the default. The field layer runs at half
// res and upsamples, because field effects are fog and shafts and gradients —
// low-frequency things a bilinear stretch cannot hurt. This is the exception
// the flag exists for: crisp bar edges, key seams and single-pixel glints are
// exactly the detail an upsample destroys.
//
// THE ROLL IS A TIME AXIS DRAWN VERTICALLY. That is the one idea the geometry
// hangs from: height above the line IS time ahead of now, at (uv.y - LINE_Y) /
// speed. So a bar's length is its DURATION times the fall speed — not a
// separate size dial — and the bar covers a pixel exactly when its note is
// sounding at the moment that pixel stands for. An earlier version scaled
// length by an unrelated constant, which made every bar 3.5x too short and then
// clipped 13% of the piece flat on top of that.
//
// The same idea decides what happens at the line: the bar is EATEN by it. The
// bottom edge arrives at note-on and the top edge at note-off, so what is left
// above the line is exactly what is left to sound. No separate collapse
// animation, and a held note stays visible for as long as you hear it.
//
// HOW A FULLSCREEN PASS AFFORDS THIS. Notes are sorted by onset, so a pixel
// binary-searches for its own band instead of scanning. An earlier version
// walked a fixed 28 notes from t - TAIL, which is fine for the average of six
// notes a second and badly wrong for a piano, where a chord puts six notes on
// ONE instant: the 28 were used up by the oldest and the newest were never
// reached, so falling notes vanished and popped back as the window slid. It
// read as bad sync; it was dropped geometry. The band below is derived, and
// WINDOW is sized from the real file — see it.
//
// The glow is mafik's trick from the Waveform effect: brightness as 1/distance
// rather than a smoothstep. A soft edge gives a flat sticker; a reciprocal
// gives a thin hot core inside a wide falloff, which is what reads as emissive
// once bloom picks it up.

// ── Layout ──────────────────────────────────────────────────────────────────
const LINE_Y: f32 = 0.20;      // the strike line, 0 = bottom of frame
const FALL: f32 = 2.2;         // seconds of music visible above the line
const KEYS_LO: f32 = 0.06;     // keyboard margins across the frame
const KEYS_HI: f32 = 0.94;

/**
 * Longest and shortest a note may DRAW as, in seconds.
 *
 * DUR_MAX is a cost bound, not a look: the per-pixel search band is exactly
 * this wide (a pixel must reach back far enough to find any note still
 * sounding), so it sets WINDOW. Measured against the Reze arc transcription:
 * durations run p50 0.26s, p90 0.41s, p99 3.97s, max 8.61s — the shape of a
 * piece that is fast runs over long pedalled bass. Four seconds clips 16 of
 * 1562 notes, one percent, and those keep their key lit for the true length
 * anyway. DUR_MIN keeps a staccato note from drawing as nothing.
 */
const DUR_MAX: f32 = 4.0;
const DUR_MIN: f32 = 0.06;

/** Notes considered per pixel. The band is DUR_MAX + SPARK_LIFE + 2*PAD/speed,
 *  about 4.9s, and the densest 5.0s of the real file holds 72 onsets. 80 is
 *  that with headroom; past it a note is simply not drawn, which is a ceiling
 *  on cost rather than an error. */
const WINDOW: i32 = 80;

// ── The keyboard ────────────────────────────────────────────────────────────
// Laid out as a piano actually is — whites of equal width, blacks narrower and
// sitting ON the seam between two of them — rather than a slot per semitone.
// Both mappings put a note somewhere sensible; only this one lets you read WHICH
// note, because the eye finds a pitch by counting the black-key groups of two
// and three, and a uniform semitone grid destroys that grouping. keyX below is
// the single mapping for notes AND keys, so the two cannot disagree.
const KEY_TOP: f32 = LINE_Y - 0.004;   // the keys start where the line is, so a
                                       // note lands ON the keyboard rather than
                                       // stopping short of it
const KEY_H: f32 = 0.115;              // white key length
const BLACK_H: f32 = 0.072;
const KEY_GAP: f32 = 0.0013;           // the seam, in screen widths. Additive light
                                       // cannot draw a dark line, so the seam is
                                       // simply where no key is drawn.
// The keyboard is ALWAYS there and reads as a keyboard: bright whites, dark
// blacks. A press then shifts the key to blue rather than lighting it up.
//
// That direction is the point. The first version drew every key dim and lit the
// struck one, which is backwards from every piano and every roll app: on a real
// keyboard the white keys are the bright thing, so a bright key reads as the
// DEFAULT and a dark one as pressed. This layer is additive and cannot subtract
// light, so "darker" has to be spent on hue — blue is both unmistakably
// different and lower in luminance than ivory, which is exactly the read wanted.
// A press DIMS the key, as a real one falls into its own shadow. Additive light
// cannot darken what is behind it, but a key's colour IS what this layer emits,
// so emitting less is a real dim rather than a trick — and it costs no hue,
// which is what the coloured version spent and did not need.
//
// Black keys go the other way by necessity: dimming something already at 0.05
// is invisible. They lift to the same dark grey the whites fall to, so a
// pressed key of either kind reads as the same state.
const KEY_IVORY = vec3f(0.72, 0.78, 0.88);    // an unpressed white key
const KEY_EBONY = vec3f(0.045, 0.055, 0.080); // and an unpressed black one
const KEY_PRESS = vec3f(0.20, 0.23, 0.30);    // a pressed white, in shadow
const KEY_PRESS_B = vec3f(0.14, 0.16, 0.21);  // a pressed black, just readable
const KEY_SHADE: f32 = 0.82;           // how much the back of a key falls off,
                                       // so the keys have a face and not just
                                       // a fill

/**
 * A key shows a STRIKE, not a hold.
 *
 * rzKeyEnergy holds at a flat 1.0 for as long as a note sounds, which is right
 * for the fog and glow effects it was built for and wrong here: this piece
 * holds notes for 4.6 and 8.6 seconds, so a key lit from it simply stays
 * glaring after the music has moved on. So the bright part is a short flash
 * from the note list, and rzKeyEnergy only contributes a low sustain — which is
 * also what a struck piano string actually does.
 */
const KEY_FLASH: f32 = 0.45;      // seconds the strike takes to fade
const KEY_FLASH_GAIN: f32 = 1.6;
const KEY_SUSTAIN: f32 = 0.34;    // how much a merely-held key keeps

// ── Bars ────────────────────────────────────────────────────────────────────
// Lengths are in units of screen HEIGHT on BOTH axes — x is divided by aspect
// before use. Mixing raw uv.x with uv.y was why bars read as slabs: on a 16:9
// frame the same number is 1.8x wider than it is tall, and the round glow came
// out an ellipse.
//
// WIDTH is the exception: a fraction of a white key rather than a fixed number,
// so a two-octave piece and a seven-octave one both get bars that sit on their
// keys instead of swamping them or vanishing between them.
const BAR_FILL: f32 = 0.17;    // half-width, as a fraction of a white slot
const BAR_FILL_B: f32 = 0.12;  // black-key notes are narrower, as their keys are
// The GLOW is what made neighbouring notes touch, not the box. At forty white
// keys a slot is about 42px, and the old halo reached 11px past a 26px box —
// 48px of ink in a 42px slot, so adjacent notes overlapped however narrow the
// box got. Both falloffs are now tight enough that the ink stays inside its key.
const BAR_SHARP: f32 = 220000.0; // core falloff — half brightness ~2.4px out
const BAR_HALO: f32 = 340.0;   // and the wider one, ~3px
const BAR_HALO_W: f32 = 0.12;  // how much of the halo to mix in
const PAD: f32 = 0.035;        // how far a bar's glow reaches past its edge.
                               // The search band is derived from this, so
                               // widening the glow without widening PAD clips it.

const CORE = vec3f(0.86, 0.95, 1.00);   // white, faintly cold
const GLOW = vec3f(0.14, 0.42, 1.00);   // the blue that carries the falloff

// ── The line ────────────────────────────────────────────────────────────────
// Matched to Waveform's weight deliberately. Waveform composes twenty copies of
// 1/(200*p.y) over [-1,1] at a 1.9 tint, which works out near 1/(400*dy) at
// several times this gain; an earlier version used 1/(620*dy) at 0.5, about
// thirty times dimmer, and read as a hairline.
const LINE_TINT = vec3f(0.50, 0.80, 1.00);
const LINE_SHARP: f32 = 800.0; // the CORE: higher is thinner
const LINE_GAIN: f32 = 1.5;
const LINE_CAP: f32 = 4.0;     // the reciprocal is unbounded at dy = 0
// …and a second, much wider falloff around it. One reciprocal cannot be both
// thin and glowing — its width and its reach are the same number — so thinning
// the core kept costing the glow. Two terms separate the questions: SHARP is
// how thin the filament is, HALO is how far the light carries.
const LINE_HALO: f32 = 55.0;
const LINE_HALO_W: f32 = 0.42;
/** Past this far from the line, in screen heights, it contributes nothing worth
 *  a twenty-four-key sample — so pixels beyond it skip lineY entirely. At the
 *  edge the wide falloff is under 0.03, below what bloom can lift. */
const LINE_REACH: f32 = 0.30;

// The line is the surface notes land ON, so it has to read as level. It bends
// where keys sound, but only just: the bend is per-key and SUMS across the
// sampled pitches, so a chord with a wide voicing stacked several of them and
// swung the whole line into a wave taller than the bars falling onto it.
// RIPPLE_MAX caps the total, which keeps a dense passage a disturbed surface
// rather than a different shape.
// Barely anything. The line is the surface notes land ON, so it has to read as
// level — a visible swell on every hit made it the loudest thing in the frame
// and turned the strike into an event the line was having rather than one the
// note was. What is left is a flicker of life at the point of contact.
const RIPPLE: f32 = 0.0015;    // how far one struck key bends the line
const RIPPLE_MAX: f32 = 0.003; // ceiling on the total bend, however many sound
const RIPPLE_W: f32 = 0.085;   // how wide that bend reaches, in screen widths
const FLOW: f32 = 0.55;        // idle drift, so the line breathes when silent

// ── Impact ──────────────────────────────────────────────────────────────────
// A burst, not a puff: twelve glints fanned across the half-circle rather than
// five thrown at random angles, each its own size and speed. Uniform dots at
// random angles read as a handful of copies of one sprite; a fan with varied
// size reads as something breaking apart.
const SPARK_LIFE: f32 = 0.7;
const SPARK_N: i32 = 24;
const SPARK_REACH: f32 = 0.22;  // how far a glint travels. This is the size dial
                                // for the whole burst; SPARK_TOP below is
                                // derived from it and has to follow it up.
const SPARK_RISE: f32 = 1.3;    // how much of that goes upward
const SPARK_GRAV: f32 = 0.12;   // and what pulls it back
const SPARK_SHARP: f32 = 600000.0; // per-glint tightness — ~1.4px, which only
                                   // resolves at all because of @fullres
const SPARK_GAIN: f32 = 0.95;
/** The highest a glint gets, and how far below the line one may dip. Bounds the
 *  block so a pixel outside it does not pay twenty-four iterations to learn
 *  that, so it must stay at or above the arc's real apex: with the constants
 *  above the peak works out near 0.26. */
const SPARK_TOP: f32 = 0.30;
const SPARK_DIP: f32 = 0.03;

// ── Piano geometry ──────────────────────────────────────────────────────────
// Closed forms rather than lookup tables: WGSL cannot index a const array with
// a runtime value, and a var<private> table would be state this needs none of.

/** White keys below this semitone class, within its octave. The 7-in-12 pattern
 *  C D E F G A B, as arithmetic — verified for all twelve classes. */
fn whiteInOctave(c: i32) -> i32 { return (c * 7 + 6) / 12; }

/** Is this semitone class one of the five black keys? Below E the black ones
 *  are odd, from F up they are even — which is exactly the seam in the pattern
 *  that makes the groups of two and three. */
fn isBlackClass(c: i32) -> bool { return select(c % 2 == 1, c % 2 == 0, c >= 5); }

/** How many white keys sit below this pitch on a full keyboard. A black key
 *  shares the ordinal of the white ABOVE it, which is what places it on the
 *  seam between two whites instead of in a slot of its own. */
fn whiteOrd(p: i32) -> i32 { return 7 * (p / 12) + whiteInOctave(p % 12); }

/** The pitch of the nth white key — whiteOrd inverted over the whites. */
fn pitchOfWhiteOrd(n: i32) -> i32 { return 12 * (n / 7) + (12 * (n % 7) + 5) / 7; }

/** White ordinal of the score's lowest pitch: the keyboard's left edge.
 *
 *  One white LOWER when that lowest note is itself black. A black key sits on
 *  the seam below the white it shares an ordinal with, so without this it lands
 *  at seam index 0 — the frame's left edge, where there is no white beneath it
 *  and the drawing code skips it. Its bars would then fall on a key that was
 *  never drawn. */
fn kbLowOrd() -> i32 {
  let lo = i32(round(rzPitchLow()));
  return whiteOrd(lo) - select(0, 1, isBlackClass(lo % 12));
}

/** White keys the score spans. The keyboard is the PIECE's range, not 0..127 —
 *  a piece that never leaves two octaves fills the width instead of huddling in
 *  the middle of an 88-key drawing that is mostly empty. */
fn kbWhites() -> i32 {
  return max(whiteOrd(i32(round(rzPitchHigh()))) - kbLowOrd() + 1, 1);
}

/** Width of one white key, in screen widths. */
fn kbKeyW() -> f32 { return (KEYS_HI - KEYS_LO) / f32(kbWhites()); }

/**
 * x of a pitch: the centre of its white key, or the seam its black key sits on.
 * The one mapping a falling note and its key both go through.
 *
 * ord0 and kw are passed IN, not looked up. They used to be read here through
 * kbLowOrd()/kbKeyW(), which is correct and was costing the whole effect: each
 * of those does integer division and reads the score header, and this function
 * is called about a hundred times per pixel — twenty-four by the line and up to
 * eighty by the note loop. Integer division is among the slowest things a GPU
 * does, so that worked out at several hundred divides on every pixel of a
 * full-resolution frame, and the field pass measured 23.8ms. They are loop
 * invariants; hoisting them to the caller is the whole fix.
 */
fn keyX(pitch: f32, ord0: i32, kw: f32) -> f32 {
  let p = i32(round(pitch));
  let n = f32(whiteOrd(p) - ord0);
  let off = select(0.5, 0.0, isBlackClass(p % 12));
  return KEYS_LO + (n + off) * kw;
}

/** Signed distance to a rounded box — the bar's shape before it becomes light. */
fn boxSdf(p: vec2f, half: vec2f, r: f32) -> f32 {
  let q = abs(p) - half + vec2f(r);
  return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}

/** First note whose onset is at or after t, by binary search over the sorted
 *  score. Returns the count when every note is earlier. */
fn firstAfter(t: f32) -> i32 {
  var lo = 0;
  var hi = rzNoteCount();
  // 16 halvings covers 65536 notes. WGSL cannot see that hi - lo shrinks, so
  // the bound has to be written out; too FEW iterations does not fail loudly,
  // it returns a wrong index and puts notes at wrong times.
  for (var k = 0; k < 16; k = k + 1) {
    if (lo >= hi) { break; }
    let mid = (lo + hi) / 2;
    if (rzNoteStart(mid) < t) { lo = mid + 1; } else { hi = mid; }
  }
  return lo;
}

/** How hard this key was struck in the last KEY_FLASH seconds — see KEY_FLASH.
 *  A narrow scan: only onsets in that window can matter, and the keyboard is a
 *  thin strip of the frame, so this is far cheaper than it looks. */
fn strikeAt(pitch: i32, t: f32) -> f32 {
  let count = rzNoteCount();
  // Not named 'from': that is a WGSL reserved keyword, and the compiler reports
  // it as a syntax error at the USE site rather than as a bad declaration.
  let head = firstAfter(t - KEY_FLASH);
  var v = 0.0;
  // 32, not 24: this walks EVERY onset in the flash window, not just the ones
  // on this key, so a chord-heavy passage exhausts it quickly — and running out
  // does not fail, it silently leaves a key dark while its note falls past.
  for (var k = 0; k < 32; k = k + 1) {
    let i = head + k;
    if (i >= count) { break; }
    let age = t - rzNoteStart(i);
    if (age < 0.0) { break; }     // sorted, so nothing after this has sounded
    if (i32(round(rzNotePitch(i))) == pitch) { v = max(v, 1.0 - age / KEY_FLASH); }
  }
  return v;
}

/** How pressed a key looks, 0..1 — a strike that fades, plus a low sustain
 *  while the note is still held. Saturated, so a struck key goes fully over to
 *  the pressed colour and then returns. */
fn pressAt(pitch: i32, t: f32) -> f32 {
  return clamp(KEY_SUSTAIN * rzKeyEnergy(f32(pitch)) + KEY_FLASH_GAIN * strikeAt(pitch, t), 0.0, 1.0);
}

/** The keyboard under the line: which note is which, and which were just hit. */
fn keyboard(uv: vec2f, t: f32) -> vec3f {
  if (uv.y > KEY_TOP || uv.y < KEY_TOP - KEY_H) { return vec3f(0.0); }
  let nW = kbWhites();
  let ord0 = kbLowOrd();
  let kw = kbKeyW();
  let wf = (uv.x - KEYS_LO) / kw;
  if (wf < 0.0 || wf >= f32(nW)) { return vec3f(0.0); }

  // A black key first, since it sits on top of the two whites it divides. It
  // exists only where those two are a whole tone apart — which is what leaves
  // the gaps at E-F and B-C that make the groups readable.
  let bIdx = i32(round(wf));
  if (bIdx > 0 && bIdx < nW) {
    let below = pitchOfWhiteOrd(ord0 + bIdx - 1);
    let above = pitchOfWhiteOrd(ord0 + bIdx);
    if (above - below == 2 && abs(uv.x - (KEYS_LO + f32(bIdx) * kw)) < kw * 0.30 && uv.y > KEY_TOP - BLACK_H) {
      let p = below + 1;
      let press = pressAt(p, t);
      let v = (uv.y - (KEY_TOP - BLACK_H)) / BLACK_H;
      return mix(KEY_EBONY, KEY_PRESS_B, press) * mix(1.0, KEY_SHADE * 0.85, v);
    }
  }

  // Otherwise the white key whose slot this is, inset so the seams read.
  let wi = i32(floor(wf));
  let frac = wf - f32(wi);
  let inset = KEY_GAP / kw;
  if (frac < inset || frac > 1.0 - inset) { return vec3f(0.0); }
  let p = pitchOfWhiteOrd(ord0 + wi);
  let press = pressAt(p, t);
  let v = (uv.y - (KEY_TOP - KEY_H)) / KEY_H;
  return mix(KEY_IVORY, KEY_PRESS, press) * mix(1.0, KEY_SHADE, v);
}

/** The line's height at x: a flowing baseline, nudged where a key sounds.
 *  ord0/kw/lo/hi are hoisted for the reason keyX explains. */
fn lineY(x: f32, t: f32, ord0: i32, kw: f32, lo: f32, hi: f32) -> f32 {
  // Idle motion, so the line is fluid rather than a rule drawn across the frame.
  let base = LINE_Y
           + sin(x * 9.0 + t * FLOW) * 0.0022
           + sin(x * 19.0 - t * FLOW * 1.6) * 0.0011
           + sin(x * 37.0 + t * FLOW * 2.3) * 0.0005;
  // Every sounding key pushes it up near its own x. Sampling the key map rather
  // than the note list is the whole reason that map exists: this is a per-pixel
  // question about NOW, which a list of onsets cannot answer cheaply.
  var bend = 0.0;
  for (var k = 0; k < 24; k = k + 1) {
    let pitch = round(mix(lo, hi, f32(k) / 23.0));
    let d = (x - keyX(pitch, ord0, kw)) / RIPPLE_W;
    // Distance first: past three widths the gaussian is nothing, and this is
    // the innermost loop of a full-resolution fullscreen pass.
    if (abs(d) > 3.0) { continue; }
    bend += RIPPLE * rzKeyEnergy(pitch) * exp(-d * d);
  }
  return base + min(bend, RIPPLE_MAX);
}

fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f {
  let res = rzResolution();
  let aspect = res.x / max(res.y, 1.0);
  let t = rzScoreTime();
  var col = vec3f(0.0);

  // The keyboard's shape, read ONCE. Every one of these costs an integer divide
  // or a score-header read, and everything below wants them — see keyX.
  let ord0 = kbLowOrd();
  let nW = kbWhites();
  let kw = (KEYS_HI - KEYS_LO) / f32(nW);
  let pLo = rzPitchLow();
  let pHi = rzPitchHigh();

  // ── The line ──────────────────────────────────────────────────────────────
  // 1/|distance| — a hot filament inside a wide glow, which is what a soft edge
  // cannot give and what bloom needs something to catch.
  //
  // Skipped outright where it cannot be seen. lineY samples twenty-four keys,
  // and it was doing that for every pixel of the frame including the empty sky
  // — most of a full-resolution pass spent computing a line that contributes
  // nothing that far from it. LINE_REACH is past where the wide falloff has
  // anything left to add.
  if (abs(uv.y - LINE_Y) < LINE_REACH) {
    let dy = uv.y - lineY(uv.x, time, ord0, kw, pLo, pHi);
    col += LINE_TINT * (min(abs(1.0 / (LINE_SHARP * dy)), LINE_CAP) * LINE_GAIN
                      + LINE_HALO_W / (1.0 + LINE_HALO * abs(dy)));
  }

  let count = rzNoteCount();
  if (count == 0) {
    // No score: the line alone, and no keyboard — the pitch range it would be
    // drawn against is zero, and one key stretched across the frame is worse
    // than nothing while a file is on its way.
    return vec4f(col, clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0));
  }
  col += keyboard(uv, t);

  let speed = (1.0 - LINE_Y) / FALL;
  let above = uv.y - LINE_Y;
  if (above < -SPARK_DIP) {
    // Below the line: the keyboard and the line's own glow live here.
    return vec4f(col, clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0));
  }

  // ── Which notes can reach THIS pixel ──────────────────────────────────────
  // The height IS a time — see the header — so a bar covers this pixel exactly
  // when its note is sounding at tau. That inverts into a band of onsets: a
  // note must have started by tau, and cannot have started more than DUR_MAX
  // before it or it would already be over.
  let tau = t + above / speed;
  let sTo = tau + PAD / speed;
  let sFrom = min(tau, t) - PAD / speed - DUR_MAX - SPARK_LIFE;

  let start = firstAfter(sFrom);
  for (var k = 0; k < WINDOW; k = k + 1) {
    let i = start + k;
    if (i >= count) { break; }
    let s = rzNoteStart(i);
    if (s > sTo) { break; }        // sorted, so everything after is later still

    let dur = clamp(rzNoteLength(i), DUR_MIN, DUR_MAX);
    // The bar is eaten by the line: bottom edge at note-on, top edge at
    // note-off, so its length is what is left to sound.
    let bot = max((s - t) * speed, 0.0);
    let top = max((s + dur - t) * speed, 0.0);
    let age = t - s;
    // Emission runs the length of the note, and the last glint still has a
    // life to live after that — which is why the band below reaches back
    // DUR_MAX + SPARK_LIFE rather than DUR_MAX.
    let sparking = age > 0.0 && age < dur + SPARK_LIFE;

    // Vertical reject BEFORE touching pitch: keyX costs integer divides, and
    // most of a 4-second band is notes this pixel is nowhere near in time.
    if (top <= bot && !sparking) { continue; }
    if ((above < bot - PAD || above > top + PAD) && !sparking) { continue; }

    let pitch = rzNotePitch(i);
    let black = isBlackClass(i32(round(pitch)) % 12);
    // Height units on both axes, so the glow is round in pixels.
    let dx = (uv.x - keyX(pitch, ord0, kw)) * aspect;
    let halfW = kw * aspect * select(BAR_FILL, BAR_FILL_B, black);
    // 2.3x SPARK_REACH, not 1.0: a glint's reach is scaled by its own hash to
    // 1.25, by its note's energy to 1.35, and then travelled at up to 1.25
    // speed — every one of those multiplies the spread. Raise this whenever any
    // of them is raised: an unraised bound does not fail, it silently clips the
    // outermost glints of exactly the biggest hits.
    if (abs(dx) > halfW + PAD + SPARK_REACH * 2.3) { continue; }

    if (top > bot) {
      let len = top - bot;
      let cy = LINE_Y + (bot + top) * 0.5;
      let d = max(boxSdf(vec2f(dx, uv.y - cy), vec2f(halfW, len * 0.5), min(halfW * 0.5, len * 0.4)), 0.0);
      // Velocity is normalised against the score's own range inside rzNote*,
      // but a quiet transcription still tops out low — so lift it rather than
      // trust it.
      let hit = 0.55 + 0.45 * clamp(rzNoteVelocity(i) * 2.2, 0.0, 1.0);
      // A brief overshoot as it arrives — what sells the note LANDING rather
      // than merely crossing.
      let flash = 1.0 + 1.6 * exp(-max(age, 0.0) * 12.0) * step(0.0, age);
      let core = 1.0 / (1.0 + BAR_SHARP * d * d);
      let halo = 1.0 / (1.0 + BAR_HALO * d);
      col += mix(GLOW, CORE, clamp(core, 0.0, 1.0)) * (core * 1.5 + halo * BAR_HALO_W) * hit * flash;
    }

    // ── The burst. It KEEPS EMITTING for as long as the note sounds instead of
    //    firing once: contact with the line lasts the whole note, and a single
    //    puff at the start read as the bar bouncing off rather than grinding
    //    against it. Each glint is born on its own staggered cycle, so the
    //    stream is continuous, and the cycle index seeds the hash so no two
    //    waves repeat. Still closed-form in age, so a scrub replays it exactly.
    if (sparking && uv.y < LINE_Y + SPARK_TOP) {
      // Short notes pack their glints into the time they have; long ones spread
      // over a full life and then recycle. Without the floor, a staccato note
      // emitted only the first two or three of them.
      // THIS NOTE'S OWN CHARACTER, drawn once and shared by every glint it
      // throws. Without it every burst was the same fan at a different place —
      // same width, same count, same energy — so a run of notes read as one
      // animation repeating rather than as a piece being played. The pitch is
      // in the seed as well as the index, so the same key struck twice differs
      // and two different keys differ more.
      let nh = rzHash11(f32(i) * 1.37 + pitch * 0.29 + 0.11);
      let nh2 = rzHash11(f32(i) * 2.71 + pitch * 0.73 + 7.3);
      let nh3 = rzHash11(f32(i) * 5.11 + pitch * 1.13 + 3.7);
      // How hard this one went: velocity, plus a draw of its own so two notes
      // at the same velocity are still not the same burst.
      let energy = mix(0.65, 1.35, nh2) * (0.6 + 0.4 * clamp(rzNoteVelocity(i) * 2.2, 0.0, 1.0));
      let spread = mix(0.5, 1.0, nh);        // how wide the fan opens
      let tilt = (nh3 - 0.5) * 0.85;         // and which way it leans
      // Not every hit throws the same number. A light one is a few glints.
      let live = max(i32(mix(f32(SPARK_N) * 0.4, f32(SPARK_N), nh2 * energy)), 3);

      let step = min(SPARK_LIFE, max(dur, 0.12)) / f32(live);
      for (var j = 0; j < live; j = j + 1) {
        let tj = age - f32(j) * step;
        if (tj < 0.0) { continue; }
        let cyc = floor(tj / SPARK_LIFE);
        if (f32(j) * step + cyc * SPARK_LIFE > dur) { continue; }  // it stopped sounding
        let sp = (tj - cyc * SPARK_LIFE) / SPARK_LIFE;             // 0..1 through its life
        let h = rzHash11(f32(i) * 3.17 + f32(j) * 0.71 + cyc * 5.13);
        let h2 = rzHash11(f32(i) * 7.31 + f32(j) * 2.13 + cyc * 1.77 + 11.0);
        // Fanned by j and only JITTERED by the hash. Pure hash angles clump —
        // random draws leave gaps and pairs, which reads as a handful rather
        // than as a burst. The fan is centred on straight up and then leaned,
        // so the SHAPE varies per note while the spacing stays even.
        let ang = 1.5707963 + tilt
                + ((f32(j) + 0.5) / f32(live) - 0.5) * 3.14159265 * spread
                + (h - 0.5) * 0.42;
        let reach = SPARK_REACH * energy * (0.35 + 0.9 * h2);
        let sp2 = sp * (0.75 + 0.5 * h);   // and not all at one speed
        let px = cos(ang) * reach * sp2;
        let py = LINE_Y + sin(ang) * reach * SPARK_RISE * sp2 - SPARK_GRAV * sp2 * sp2;
        let sd = length(vec2f(dx - px, uv.y - py));
        // Each glint its own size. One size for all is the other half of what
        // reads as copies of a sprite.
        let sharp = SPARK_SHARP * (0.5 + h2 * 1.6);
        let fade = (1.0 - sp) * (1.0 - sp);
        col += CORE * (1.0 / (1.0 + sharp * sd * sd)) * fade * SPARK_GAIN;
      }
    }
  }

  // Alpha is the light's own strength: additive means it is never asked to
  // cover anything, but the composite still wants to know how present it is.
  return vec4f(col, clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0));
}
