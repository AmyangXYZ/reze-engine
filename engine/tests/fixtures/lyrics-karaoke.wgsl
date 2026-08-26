// glyph edges are sub-pixel detail the half-res field pass cannot carry.
//
// The words of the track, drawn the way anison broadcasts drew them: a line
// low on the screen, white glyphs inside a heavy dark rim, a soft shadow under
// it so the words hold over a bright stage, and the sung part wiping to colour
// as the clock crosses the line.
//
// The TEXT arrives from the host — pair an .lrc with the track (X.mp3 beside
// X.lrc) and each line is rasterised once into the lyric atlas. What this file
// owns is where the words go, when they land, and how they look.
//
//   rzLyricIndex(t)      which line is live at time t (-1 between lines)
//   rzLyricText(i, uv)   glyph coverage of line i; uv is 0..1 across its box
//   rzLyricAspect(i)     the line's width over its height, as rasterised
//   rzLyricPixels(i)     the line's box in atlas texels
//   rzLyricProgress(i,t) 0..1 through the line — the karaoke sweep

// Tunables — edit and ⌘⏎.
const LEAD = 0.30;                      // seconds EARLY; a subtitle read just
                                        // before it is sung is what feels in time
const FILL = vec3f(1.0, 1.0, 1.0);      // unsung glyphs
const SUNG = vec3f(0.42, 0.78, 1.0);    // what the sweep leaves behind
const RIM = vec3f(0.04, 0.04, 0.12);    // the outline around both
const LINE_H = 0.050;                   // line box height, fraction of screen height
const RIM_R = 0.085;                    // outline reach, fraction of box height
const SHADOW = 0.5;                     // drop shadow strength, 0 for none
const SHADOW_DROP = 0.5;                // how far BELOW the words it sits, in rim radii
const SHARP = 0.35;                     // edge contrast; 0.5 is raw coverage,
                                        // lower is harder — under ~0.2 it steps
const RISE = 0.10;                      // how far the line rises as it appears, in box heights
const FADE_IN = 0.12;                   // seconds; broadcast subs cut fast
const FADE_OUT = 0.18;

/**
 * WHERE each line goes: xy is the box's anchor in screen fractions (x is its
 * centre, y its bottom edge, both measured from the bottom-left), z scales its
 * height. This is the function to edit when the words should move — it gets
 * the line's index and the clock, so placement can follow the song.
 *
 * Some ideas, each one line:
 *   alternate high and low   vec3f(0.5, select(0.08, 0.60, (i % 2) == 1), 1.0)
 *   the chorus, bigger       vec3f(0.5, 0.08, select(1.0, 1.6, i >= 18))
 *   drift with the music     vec3f(0.5 + 0.04 * sin(t * 0.7), 0.08, 1.0)
 *   step up the screen       vec3f(0.5, 0.08 + 0.03 * f32(i % 4), 1.0)
 */
fn linePlace(i: i32, t: f32) -> vec3f {
  return vec3f(0.5, 0.070, 1.0);
}

/**
 * Glyph coverage over ONE SCREEN PIXEL, not at one point.
 *
 * The atlas holds a line at whatever size it was rasterised, and the box is
 * drawn at whatever size the canvas and linePlace ask for — so the two rarely
 * agree, and where the atlas has more texels than the box has pixels a single
 * bilinear tap simply misses some of them. That is what stair-steps a glyph
 * edge. Four taps on a rotated grid estimate the area instead, which is what
 * an antialiased edge actually is.
 */
fn lyricCoverage(i: i32, lu: vec2f, px: vec2f) -> f32 {
  var s = rzLyricText(i, lu + vec2f(0.125, 0.375) * px);
  s = s + rzLyricText(i, lu + vec2f(-0.375, 0.125) * px);
  s = s + rzLyricText(i, lu + vec2f(-0.125, -0.375) * px);
  s = s + rzLyricText(i, lu + vec2f(0.375, -0.125) * px);
  return s * 0.25;
}

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let t = rzAudioTime() + LEAD;
  let i = rzLyricIndex(t);
  if (i < 0 || !rzLyricHasText(i)) { return vec4f(0.0); }

  let inT = clamp((t - rzLyricStart(i)) / FADE_IN, 0.0, 1.0);
  let fade = inT * clamp((rzLyricEnd(i) - t) / FADE_OUT, 0.0, 1.0);
  if (fade <= 0.0) { return vec4f(0.0); }

  // The box: height from linePlace, width from the rasterised proportions, so
  // a two-word interjection and a full verse both sit unstretched.
  let res = rzResolution();
  let place = linePlace(i, t);
  let lineH = LINE_H * place.z;
  let w = lineH * rzLyricAspect(i) * (res.y / res.x);
  let ease = 1.0 - (1.0 - inT) * (1.0 - inT);
  let baseY = place.y - RISE * lineH * (1.0 - ease);
  let lu = vec2f((uv.x - place.x) / w + 0.5, (uv.y - baseY) / lineH);
  // Reject early and generously — the rim and shadow reach past the box.
  if (lu.x < -0.2 || lu.x > 1.2 || lu.y < -0.4 || lu.y > 1.4) { return vec4f(0.0); }

  // One screen pixel, in the box's own units. Everything below measures in it.
  let px = vec2f(1.0 / max(w * res.x, 1.0), 1.0 / max(lineH * res.y, 1.0));
  let glyph = smoothstep(0.5 - SHARP, 0.5 + SHARP, lyricCoverage(i, lu, px));

  // The rim is the glyph dilated over two rings, each tap nudged a quarter
  // pixel off its spoke so the dilated edge is sampled at more than one phase.
  let r = vec2f(RIM_R * lineH * (res.y / res.x) / w, RIM_R);
  var raw = 0.0;
  for (var k = 0; k < 8; k = k + 1) {
    let a = 6.2831853 * f32(k) / 8.0;
    let d = vec2f(cos(a), sin(a));
    raw = max(raw, rzLyricText(i, lu + d * r + d.yx * px * 0.25));
    let b = a + 6.2831853 / 16.0;
    let e = vec2f(cos(b), sin(b));
    raw = max(raw, rzLyricText(i, lu + e * r * 0.55 - e.yx * px * 0.25));
  }
  let rim = max(glyph, smoothstep(0.5 - SHARP, 0.5 + SHARP, raw));

  // A soft shadow UNDER the words — what keeps them readable when the stage
  // behind blows out. Both the direction and the softness are load-bearing.
  //
  // DIRECTION. A tap reads the line at lu + off and draws what it finds at lu,
  // so the ink lands opposite the offset: to hang the shadow BELOW the glyphs
  // the tap has to reach ABOVE them, and the other sign puts a second set of
  // words over the line, which reads as the line drawn twice.
  //
  // SOFTNESS. The taps are AVERAGED across a disc rather than maxed at one
  // radius. A max is a hard-edged copy of the glyph shifted bodily off its
  // own position — the same duplicate by another route, and the reason this
  // is a weighted sum with the outer ring turned half a step off the inner
  // one, so no spoke is sampled twice.
  var shade = 0.0;
  if (SHADOW > 0.0) {
    let drop = vec2f(0.0, r.y * SHADOW_DROP);
    var wsum = 1.0;
    shade = rzLyricText(i, lu + drop);
    for (var k = 0; k < 6; k = k + 1) {
      let a = 6.2831853 * f32(k) / 6.0;
      let d = vec2f(cos(a), sin(a));
      let e = vec2f(cos(a + 0.5235988), sin(a + 0.5235988));
      shade = shade + 0.70 * rzLyricText(i, lu + drop + d * r * 0.60);
      shade = shade + 0.40 * rzLyricText(i, lu + drop + e * r * 1.10);
      wsum = wsum + 1.10;
    }
    shade = shade / wsum;
  }

  // The sweep: everything left of the line's progress has been sung. Its edge
  // is one pixel wide, like every other edge here.
  let sung = smoothstep(-px.x, px.x, rzLyricProgress(i, t) - lu.x);
  let ink = mix(FILL, SUNG, sung);

  // Three layers, one straight-alpha return: shadow under rim under fill. Each
  // coverage is >= the one above it, so the mixes stack in that order and the
  // rim never darkens the glyph it surrounds.
  let color = mix(mix(vec3f(0.0), RIM, rim), ink, glyph);
  let a = max(max(shade * SHADOW, rim), glyph) * fade;
  return vec4f(color, clamp(a, 0.0, 1.0));
}
