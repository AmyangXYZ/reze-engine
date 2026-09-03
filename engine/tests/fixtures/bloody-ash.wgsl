#lights 3

// Bloody Ash — Holy Light, bled.
//
// It is the SAME SHADER as Holy Light with two colours changed, and that is the
// point of shipping the pair. Everything expensive about a silhouette glow — the
// distance to her outline — is answered once by the engine, so a whole second
// look costs two constants. An author who wants a third does not need to
// understand any of the machinery below; they need to know which two lines to
// edit.
//
// It draws ONLY OUTSIDE HER. A foreground mount lands over the finished frame,
// so a stray pixel on the body is paint on the material: her blacks would go
// grey and her shading would flatten.
//
// There are no particles in it. Ash flakes were tried and taken out — against a
// rim this bright they read as dirt on the lens rather than as something coming
// off her, and they fought the one thing the effect is for.

// Tunables — edit a value and hit Cmd-Enter to see it live.
// TWO TIERS, AND SATURATED AT HER EDGE.
//
// Divine Ribbon and Divine Teleportation put a warm-WHITE pinpoint at the source,
// and it is right for them: they are built for a dark stage, where near-white is
// the brightest thing available and reads as the hottest part of the light. Over
// a pale background it is invisible, and what it draws is a white ring between
// the figure and her own glow — a gap, exactly where the effect should be
// strongest. So the colour is saturated where it leaves her and decays outward,
// and the divinity comes from the lamps below rather than from a white core.
const GOLD = vec3f(1.00, 0.16, 0.08);  // the body of the glow, and its hottest point
const HALO = vec3f(0.34, 0.01, 0.02);  // deep blood, the outer light
/** What she is LIT by. The body's gold, never the pinpoint's near-white: a
 *  warm-white lamp washes whatever it lands on, and the colour a viewer reads
 *  off the glow is its body. Divine Ribbon says the same in its own file. */
const LIGHT_COLOR = vec3f(0.95, 0.16, 0.10);
const LIGHT_POWER = 1.5;   // how hard she is lit
const LIGHT_R = 9.0;       // reach of each lamp, in world units
const REACH = 110.0;     // how far the light carries, in pixels of a 1440-line frame
const RIM = 5.0;         // width of the bright band against her outline
const RIM_GAIN = 0.9;
const AURA = 0.7;        // the broad glow's strength
const FADE = 1.8;        // how sharply it falls away, >1 pulls it in tight
const RAYS = 13.0;       // shafts around her; 0 for a plain halo
const RAY_DEPTH = 0.28;  // how deeply they cut into it
const TURN = 0.06;       // how fast they turn
const BREATH = 0.14;     // a slow swell, 0 holds it still
const OPACITY = 1.0;

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  // REACH AND RIM ARE FRACTIONS OF THE PICTURE, not counts of device pixels.
  // rzCastDistance answers in screen pixels, so bare numbers draw a glow half as
  // wide in a 4K export as in the preview it was tuned in. Measured against 1440
  // lines — about what the editor's preview is — it is the same glow at every
  // resolution.
  let px = rzResolution().y / 1440.0;
  let reach = REACH * px;
  let d = rzCastDistance(uv);
  if (d >= reach) { return vec4f(0.0); }

  // OUTSIDE ONLY, faded across the one pixel her edge lives in. The field reads
  // 0 on her and runs to -0.5 at a half-covered pixel, so this is 0 everywhere
  // she is, 1 a pixel out, and a clean ramp between — the light meets the same
  // anti-aliased edge she is drawn with, and never covers her.
  let outside = clamp(d + 0.5, 0.0, 1.0);
  if (outside <= 0.0) { return vec4f(0.0); }

  // The broad aura, falling to exactly zero at REACH. A glow that merely gets
  // small has to be cut somewhere, and the cut is an edge.
  let t = clamp(d / reach, 0.0, 1.0);
  let aura = pow(1.0 - t, FADE) * AURA;

  // The bright band against her.
  //
  // A smoothstep holds near 1 for its first few pixels, so this and the aura
  // together pin the alpha at 1.0 for a short distance out from her edge. That
  // saturated band is a deliberate choice and not an oversight: it was replaced
  // once with a cube that peaks and falls immediately, which is smoother and
  // measurably more like light — and it looked weaker and further from what this
  // effect is for. A hot edge is the whole read.
  let rim = (1.0 - smoothstep(0.0, RIM * px, d)) * RIM_GAIN;

  // Shafts, radiating from her middle rather than the frame's. Projecting her hip
  // is what makes them hers: they swing with her instead of sitting on the screen
  // like a filter.
  var shafts = 1.0;
  if (RAYS > 0.0 && rzSubjectCount() > 0) {
    let centre = rzProject(rzSubjectHip(0));
    // uv is normalised, so a raw angle is squashed by the aspect and the shafts
    // come out elliptical. One multiply puts them back on a circle.
    let res = rzResolution();
    let v = (uv - centre.xy) * vec2f(res.x / max(res.y, 1.0), 1.0);
    let ang = atan2(v.y, v.x);
    shafts = 1.0 - RAY_DEPTH * (0.5 + 0.5 * sin(ang * RAYS + time * TURN * 6.2831853));
  }

  let breath = 1.0 + BREATH * sin(time * 0.7);
  let e = clamp((rim + aura * shafts) * breath, 0.0, 1.0) * outside;
  return vec4f(mix(GOLD, HALO, t), e * OPACITY);
}

/**
 * SHE IS ACTUALLY LIT — three lamps standing in her, not a glow drawn behind her.
 *
 * A foreground field paints over the finished frame and can never touch the
 * shading, so the border it draws sits ON the picture: the figure inside stays
 * lit exactly as she was, which is what makes the effect read as a decal. These
 * are declared lights, they enter shading like any other, and the gold lands on
 * her arms and her cheek by the ordinary rules. That is what the divine effects
 * do and it is why they look lit rather than outlined.
 *
 * SUBJECT 0 ONLY, and three slots. A light is a declared cost — covering four
 * dancers would spend twelve of the sixteen and leave a composer no room for the
 * stage. The same call Divine Ribbon makes, for the same reason.
 */
fn lightEmit(i: u32, time: f32) -> RzLight {
  var l: RzLight;
  l.color = LIGHT_COLOR;
  l.intensity = 0.0;
  l.radius = 1.0;
  l.pos = vec3f(0.0, 0.0, 0.0);

  let s = rzSubject(0);
  if (!s.valid) { return l; }

  // Chest, hips and above the head — up the body rather than around it, so she
  // is lit from within her own silhouette and the falloff does the wrapping.
  let hip = max(s.center.y - s.root.y, 0.05);
  let up = vec3f(0.0, 1.0, 0.0);
  var at = s.center;
  if (i == 1u) { at = s.root + up * (hip * 0.35); }
  if (i == 2u) { at = s.root + up * (hip * 2.15); }
  l.pos = at;
  l.radius = LIGHT_R;
  // The same slow swell the glow breathes with, so the lamp and the halo cannot
  // disagree about how bright the moment is.
  l.intensity = LIGHT_POWER * (1.0 + BREATH * sin(time * 0.7));
  return l;
}
