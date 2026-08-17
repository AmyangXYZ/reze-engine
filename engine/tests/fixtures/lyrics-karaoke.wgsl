// @fullres — glyph edges are sub-pixel detail the half-res field pass cannot carry.
//
// The words of the track, drawn the way anison broadcasts drew them: one line
// low on the screen, white glyphs inside a heavy dark rim, a soft shadow under
// it so the words hold over a bright stage, and the sung part wiping to colour
// as the clock crosses the line.
//
// The TEXT arrives from the host — pair an .lrc with the track (X.mp3 beside
// X.lrc) and each line is rasterised once into the lyric atlas. What this file
// owns is everything about how the words LOOK. Edit a value and hit ⌘⏎.
//
//   rzLyricIndex(t)      which line is live at time t (-1 between lines)
//   rzLyricText(i, uv)   glyph coverage of line i; uv is 0..1 across its box
//   rzLyricAspect(i)     the line's width over its height, as rasterised
//   rzLyricProgress(i,t) 0..1 through the line — the karaoke sweep

// Tunables — edit and ⌘⏎.
const FILL = vec3f(1.0, 1.0, 1.0);      // unsung glyphs
const SUNG = vec3f(0.42, 0.78, 1.0);    // what the sweep leaves behind
const RIM = vec3f(0.04, 0.04, 0.12);    // the outline around both
const LINE_H = 0.050;                   // line box height, fraction of screen height
const BOTTOM = 0.070;                   // gap under the line, fraction of screen height
const RIM_R = 0.085;                    // outline reach, fraction of box height
const SHADOW = 0.5;                     // drop shadow strength, 0 for none
const RISE = 0.10;                      // how far the line rises as it appears, in box heights
const FADE_IN = 0.12;                   // seconds; broadcast subs cut fast
const FADE_OUT = 0.18;

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let t = rzAudioTime();
  let i = rzLyricIndex(t);
  if (i < 0 || !rzLyricHasText(i)) { return vec4f(0.0); }

  let inT = clamp((t - rzLyricStart(i)) / FADE_IN, 0.0, 1.0);
  let fade = inT * clamp((rzLyricEnd(i) - t) / FADE_OUT, 0.0, 1.0);
  if (fade <= 0.0) { return vec4f(0.0); }

  // The line's box: height is fixed, width follows the rasterised proportions,
  // so a two-word interjection and a full verse both sit centred and unstretched.
  let res = rzResolution();
  let w = LINE_H * rzLyricAspect(i) * (res.y / res.x);
  let ease = 1.0 - (1.0 - inT) * (1.0 - inT);
  let baseY = BOTTOM - RISE * LINE_H * (1.0 - ease);
  let lu = vec2f((uv.x - 0.5) / w + 0.5, (uv.y - baseY) / LINE_H);
  // Reject early and generously — the rim and shadow reach past the box.
  if (lu.x < -0.2 || lu.x > 1.2 || lu.y < -0.4 || lu.y > 1.4) { return vec4f(0.0); }

  // Equal reach on screen in both axes: y is a fraction of the box, x converts
  // that same screen distance into the box's own width units.
  let r = vec2f(RIM_R * LINE_H * (res.y / res.x) / w, RIM_R);
  let glyph = rzLyricText(i, lu);

  // The rim is the glyph dilated over two rings — one ring scallops on thin
  // strokes, which reads as a wobbly outline rather than a drawn one.
  var rim = glyph;
  for (var k = 0; k < 8; k = k + 1) {
    let a = 6.2831853 * f32(k) / 8.0;
    let d = vec2f(cos(a), sin(a));
    rim = max(rim, rzLyricText(i, lu + d * r));
    let b = a + 6.2831853 / 16.0;
    rim = max(rim, rzLyricText(i, lu + vec2f(cos(b), sin(b)) * r * 0.55));
  }

  // A soft shadow below, from the same dilation at a wider reach — it is what
  // keeps white-on-white readable when the stage behind blows out.
  var shade = 0.0;
  if (SHADOW > 0.0) {
    let off = vec2f(0.0, -r.y * 1.6);
    for (var k = 0; k < 4; k = k + 1) {
      let a = 6.2831853 * f32(k) / 4.0 + 0.7854;
      shade = max(shade, rzLyricText(i, lu + off + vec2f(cos(a), sin(a)) * r * 1.9));
    }
  }

  // The sweep: everything left of the line's progress has been sung. The edge
  // is soft by a hair — a hard one crawls along the glyph as it crosses.
  let sung = smoothstep(-0.008, 0.008, rzLyricProgress(i, t) - lu.x);
  let ink = mix(FILL, SUNG, sung);

  // Three layers, one straight-alpha return: shadow under rim under fill. Each
  // coverage is >= the one above it, so the mixes stack in that order and the
  // rim never darkens the glyph it surrounds.
  let color = mix(mix(vec3f(0.0), RIM, rim), ink, glyph);
  let a = max(max(shade * SHADOW * 0.85, rim), glyph) * fade;
  return vec4f(color, clamp(a, 0.0, 1.0));
}
