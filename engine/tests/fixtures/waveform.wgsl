#layer additive — this effect is LIGHT, not matter. The layer ADDS to what
// the scene already drew, so a backdrop image, a flat colour, a skybox or
// another effect behind it all stay visible and nothing here covers them.
//
// Waveform — mafik's Shadertoy visualiser, ported onto the rzAudio* interface.
// Original: https://www.shadertoy.com/view/XsjGz3 — the structure below follows
// it line for line; the adaptations are named where they happen.
//
// The idea that makes it beautiful: twenty spectrum bands each bend a
// TRAVELLING COSINE into the same coordinate, faster per band, amplitude from
// the band's PEAKINESS (its value against its neighbours' — a second
// difference, so flat noise moves nothing and a beat kicks). The glow is the
// classic 1/|y| falloff around whatever curve all that bending composed.
// Shadertoy feeds a live analyser; here rzAudioBand reads the precomputed
// analysis, which is why an export shows the same picture.
//
// WHAT THE PORT LEAVES BEHIND. A Shadertoy owns its whole canvas, so mafik
// opens by painting one: a blue-to-red gradient, the bass eighth of the
// spectrum as vertical columns, grain standing in for his noise texture, the
// lot pulled down by 0.7. A scene here already HAS a backdrop, and painting
// over it is the one thing a visualiser must not do — the gradient survives
// as the wave's own tint across the frame, which is where its colour was
// doing the work anyway, and the columns and grain go, because a full-frame
// wash of added light is opacity by another name.

// Tunables — edit and ⌘⏎.
const COLOR1 = vec3f(0.30, 0.55, 1.00);  // the wave's tint at screen left
const COLOR2 = vec3f(1.00, 0.35, 0.50);  // ...and at screen right
const BANDS_N = 20;      // cosines composed — the original's n
const BOUNCE = 0.5;      // how far the music bends the wave. THE size dial:
                         // 1.0 is mafik's full swing, which reaches the frame's
                         // edges on big drops; 0.5 keeps the dance in the middle
                         // of the frame. Scales GEOMETRY only — brightness and
                         // the crisp 1/|y| lines are untouched, which is what
                         // the earlier limiter experiments got wrong.
const GLOW = 200.0;      // the 1/|y| glow's tightness — original's constant
const WAVE_TINT = vec3f(1.9, 1.0, 1.5); // pink-white, r > b > g as mafik had it
const BRIGHT = 1.0;      // how much light the curve adds, overall
const CALM = 0.10;       // amplitude left when paused, or with no track at all

/** The original's audio texture row, as our bands: x 0..1 across the spectrum.
 *  Read raw — the AnalyserNode smoothing already lives in the analysis. */
fn wvSpectrum(x: f32) -> f32 {
  return rzAudioBand(i32(clamp(x, 0.0, 1.0) * f32(rzAudioBandCount() - 1)));
}

fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f {
  // Paused or trackless goes CALM: the maths below would happily oscillate
  // over a frozen spectrum forever — and with no track at all, the 0.6 seed in
  // the walk fabricates waves out of silence.
  let drive = mix(CALM, 1.0, rzAudioPlaying());

  // The wave. uv into [-1,1], the original's offsets verbatim.
  var p = vec2f(-1.0 + 2.0 * uv.x, -1.0 + 2.0 * uv.y);
  p.y += 0.1;
  p.x *= 2.0;

  var prev = 0.0;
  var curr = 0.6;
  var next = wvSpectrum(0.0);
  var wave = 0.0;
  for (var i = 0; i < BANDS_N; i++) {
    prev = curr;
    curr = next;
    next = wvSpectrum(f32(i + 1) / f32(BANDS_N));
    // Mafik's amplitude, undamped — including the i = 0 line, whose cos(0) is
    // constant: a straight bar displaced by the first band's peakiness squared,
    // the thick centre line that BUMPS with the music. The tempering happens in
    // the analysis instead: bands arrive smoothed with the AnalyserNode's own
    // 0.8, exactly the sluggishness Shadertoy fed him.
    let amp = max(0.0, curr * 2.0 - prev - next) * drive;
    p.y += cos((p.x * 2.0 * f32(i) / f32(BANDS_N) * 10.0 + time * f32(i)) % 6.2831853) * amp * amp * BOUNCE;
    p.x += 0.1;
    // Scalar now: the tint is applied once at the end, so the gradient can
    // colour the curve without being mixed into the falloff twenty times.
    wave += abs(1.0 / (GLOW * p.y)) * (5.0 / f32(BANDS_N));
  }

  // Straight alpha over a layer that adds: the COLOUR is what the light is,
  // the ALPHA is how much of it lands, and where the curve is far away that
  // alpha is zero — which is the whole point, since zero is what lets the
  // backdrop through.
  //
  // The field layer composites after tone mapping, so anything past 1 would
  // simply clip. The overshoot is spent on the tint going white instead: an
  // overexposed core reads as heat rather than as a flat clamped stripe.
  let lit = wave * BRIGHT;
  let tint = clamp(WAVE_TINT * mix(COLOR1, COLOR2, uv.x), vec3f(0.0), vec3f(1.0));
  let color = mix(tint, vec3f(1.0), clamp(lit - 1.0, 0.0, 1.0));
  return vec4f(color, clamp(lit, 0.0, 1.0));
}
