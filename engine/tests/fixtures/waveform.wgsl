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
// Tunables — edit and ⌘⏎.
const COLOR1 = vec3f(0.0, 0.0, 0.3);   // deep blue, screen left
const COLOR2 = vec3f(0.5, 0.0, 0.0);   // deep red, screen right
const BANDS_N = 20;      // cosines composed — the original's n
const BOUNCE = 0.5;      // how far the music bends the wave. THE size dial:
                         // 1.0 is mafik's full swing, which reaches the frame's
                         // edges on big drops; 0.5 keeps the dance in the middle
                         // of the frame. Scales GEOMETRY only — brightness and
                         // the crisp 1/|y| lines are untouched, which is what
                         // the earlier limiter experiments got wrong.
const GLOW = 200.0;      // the 1/|y| glow's tightness — original's constant
const WAVE_TINT = vec3f(1.9, 1.0, 1.5); // pink-white, r > b > g as mafik had it
const GRAIN = 0.14;      // background texture strength — stands in for iChannel0
                         // against a live analyser; our per-band p95 normalise
                         // runs hotter, so the same maths over-bends
const CALM = 0.10;       // amplitude left when paused, or with no track at all

fn wvHash21(p: vec2f) -> f32 {
  var q = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  q = q + dot(q, q.yzx + 33.33);
  return fract((q.x + q.y) * q.z);
}
/** Bilinear value noise — the stand-in for the original's noise texture. */
fn wvNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = wvHash21(i);
  let b = wvHash21(i + vec2f(1.0, 0.0));
  let c = wvHash21(i + vec2f(0.0, 1.0));
  let d = wvHash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
/** The original's audio texture row, as our bands: x 0..1 across the spectrum.
 *  Read raw — the AnalyserNode smoothing already lives in the analysis. */
fn wvSpectrum(x: f32) -> f32 {
  return rzAudioBand(i32(clamp(x, 0.0, 1.0) * f32(rzAudioBandCount() - 1)));
}

fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f {
  // ── The background, as mafik built it: a blue→red gradient, plus the bass
  // eighth of the spectrum as vertical light columns, darkened, plus grain. ──
  // Paused or trackless goes CALM: the maths below would happily oscillate
  // over a frozen spectrum forever — and with no track at all, the 0.6 seed in
  // the walk fabricates waves out of silence.
  let drive = mix(CALM, 1.0, rzAudioPlaying());
  var bg = mix(COLOR1, COLOR2, uv.x) + vec3f(wvSpectrum(uv.x / 8.0) * drive) - 0.7;
  // Far smoother than the first attempt: at cell size 64 the stand-in noise
  // read as red static, which mafik's texture never was.
  bg += (wvNoise(uv * 9.0) * 0.7 + wvNoise(uv * 23.0) * 0.3) * GRAIN;

  // ── The wave. uv into [-1,1], the original's offsets verbatim. ────────────
  // The wave. uv into [-1,1], the original's offsets verbatim.
  var p = vec2f(-1.0 + 2.0 * uv.x, -1.0 + 2.0 * uv.y);
  p.y += 0.1;
  p.x *= 2.0;

  var prev = 0.0;
  var curr = 0.6;
  var next = wvSpectrum(0.0);
  var wave = vec3f(0.0);
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
    let w = abs(1.0 / (GLOW * p.y));
    wave += WAVE_TINT * w * (5.0 / f32(BANDS_N));
  }

  // Full-frame: this IS the background, so alpha is 1 and the scene's own
  // backdrop is replaced, exactly as a Shadertoy canvas would be.
  return vec4f(max(bg + wave, vec3f(0.0)), 1.0);
}
