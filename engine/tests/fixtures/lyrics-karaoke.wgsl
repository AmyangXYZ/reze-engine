// @fullres — glyph edges are sub-pixel detail the half-res field pass cannot carry.
//
// The words of the track, drawn the way anison broadcasts drew them: one bold
// line low on the screen, white glyphs in a dark rim, the sung fraction
// wiping to colour as the clock crosses the line.
//
// The TEXT arrives from the host — pair an .lrc with the track (X.mp3 beside
// X.lrc) and each line is rasterised once into the lyric atlas. What this
// file owns is everything about how the words LOOK: colours, size, rim,
// wipe, fades. Edit a value and hit ⌘⏎.
//
//   rzLyricIndex(t)      which line is live at time t (-1 between lines)
//   rzLyricText(i, uv)   glyph coverage of line i; uv is 0..1 across its box
//   rzLyricAspect(i)     the line's width over its height, as rasterised
//   rzLyricProgress(i,t) 0..1 through the line — the karaoke sweep

// Tunables — edit and ⌘⏎.
const FILL = vec3f(1.0, 1.0, 1.0);      // unsung glyphs
const SUNG = vec3f(0.56, 0.82, 1.0);    // what the sweep leaves behind
const RIM = vec3f(0.05, 0.05, 0.16);    // the outline around both
const LINE_H = 0.072;                   // line height, fraction of screen height
const BOTTOM = 0.085;                   // gap under the line, fraction of screen height
const RIM_R = 0.07;                     // outline reach, fraction of line height
const FADE_IN = 0.10;                   // seconds; broadcast subs cut fast
const FADE_OUT = 0.15;

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let t = rzAudioTime();
  let i = rzLyricIndex(t);
  if (i < 0 || !rzLyricHasText(i)) { return vec4f(0.0); }

  // The line's box: height is fixed, width follows the rasterised proportions
  // so a short interjection and a full verse line both sit centred, unstretched.
  let res = rzResolution();
  let w = LINE_H * rzLyricAspect(i) * (res.y / res.x);
  let lu = vec2f((uv.x - 0.5) / w + 0.5, (uv.y - BOTTOM) / LINE_H);
  if (lu.x < -0.1 || lu.x > 1.1 || lu.y < -0.25 || lu.y > 1.25) { return vec4f(0.0); }

  let glyph = rzLyricText(i, lu);
  // The rim is the glyph dilated a step in eight directions — the classic
  // broadcast outline, cheap because the atlas is a single bilinear read.
  // Equal reach on screen in both axes: y is RIM_R of the box; x converts
  // that screen distance into the box's own width units.
  let r = vec2f(RIM_R * LINE_H * (res.y / res.x) / w, RIM_R);
  var rim = glyph;
  rim = max(rim, rzLyricText(i, lu + vec2f(r.x, 0.0)));
  rim = max(rim, rzLyricText(i, lu - vec2f(r.x, 0.0)));
  rim = max(rim, rzLyricText(i, lu + vec2f(0.0, r.y)));
  rim = max(rim, rzLyricText(i, lu - vec2f(0.0, r.y)));
  rim = max(rim, rzLyricText(i, lu + r * 0.707));
  rim = max(rim, rzLyricText(i, lu - r * 0.707));
  rim = max(rim, rzLyricText(i, lu + vec2f(r.x, -r.y) * 0.707));
  rim = max(rim, rzLyricText(i, lu + vec2f(-r.x, r.y) * 0.707));

  // The sweep: everything left of the line's progress has been sung.
  let sung = smoothstep(-0.012, 0.012, rzLyricProgress(i, t) - lu.x);
  let ink = mix(FILL, SUNG, sung);

  let fade = clamp((t - rzLyricStart(i)) / FADE_IN, 0.0, 1.0) * clamp((rzLyricEnd(i) - t) / FADE_OUT, 0.0, 1.0);
  let color = mix(RIM, ink, glyph);
  let a = max(glyph, rim * 0.88) * fade;
  return vec4f(color, a);
}
