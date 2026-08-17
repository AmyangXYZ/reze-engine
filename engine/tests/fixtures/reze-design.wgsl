// @fullres — this effect draws scanlines and glyph edges: sub-pixel detail the half-res field
// pass cannot carry, so it opts out and pays its own full price.
// Tunables — edit and ⌘⏎.
const NEON_COLOR = vec3f(0.96, 0.45, 0.71);  // tube color (brand pink)
const NEON_COLOR_B = vec3f(0.45, 0.65, 0.98); // second hue the shimmer travels to
const GLOW = 0.8;                            // halo strength
const FLICKER = 0.1;                         // 0 = steady sign
const TEXT_SCALE = 0.075;                    // sign size on screen
const POS_Y = 0.24;                          // height above center

fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
  return length(pa - ba * h);
}

// "REZE DESIGN" as 40 pre-baked segments (glyph boxes 0..0.9 × 0..1, advance 1.2, word gap
fn sdText(p: vec2f) -> f32 {
  // Coarse reject: pixels far from the sign's bounding box skip the letters entirely (the halo
  let toBox = vec2f(max(abs(p.x - 6.20) - 6.20, 0.0), max(abs(p.y - 0.5) - 0.5, 0.0));
  let boxD = length(toBox);
  if (boxD > 1.0) { return boxD; }
  var d = 1e5;
  d = min(d, sdSegment(p, vec2f(0.0, 0), vec2f(0.0, 1)));
  d = min(d, sdSegment(p, vec2f(0.0, 1), vec2f(0.8, 1)));
  d = min(d, sdSegment(p, vec2f(0.8, 1), vec2f(0.8, 0.5)));
  d = min(d, sdSegment(p, vec2f(0.8, 0.5), vec2f(0.0, 0.5)));
  d = min(d, sdSegment(p, vec2f(0.4, 0.5), vec2f(0.9, 0)));
  d = min(d, sdSegment(p, vec2f(1.2, 0), vec2f(1.2, 1)));
  d = min(d, sdSegment(p, vec2f(1.2, 1), vec2f(2.1, 1)));
  d = min(d, sdSegment(p, vec2f(1.2, 0.5), vec2f(1.9, 0.5)));
  d = min(d, sdSegment(p, vec2f(1.2, 0), vec2f(2.1, 0)));
  d = min(d, sdSegment(p, vec2f(2.4, 1), vec2f(3.3, 1)));
  d = min(d, sdSegment(p, vec2f(3.3, 1), vec2f(2.4, 0)));
  d = min(d, sdSegment(p, vec2f(2.4, 0), vec2f(3.3, 0)));
  d = min(d, sdSegment(p, vec2f(3.6, 0), vec2f(3.6, 1)));
  d = min(d, sdSegment(p, vec2f(3.6, 1), vec2f(4.5, 1)));
  d = min(d, sdSegment(p, vec2f(3.6, 0.5), vec2f(4.3, 0.5)));
  d = min(d, sdSegment(p, vec2f(3.6, 0), vec2f(4.5, 0)));
  d = min(d, sdSegment(p, vec2f(5.5, 0), vec2f(5.5, 1)));
  d = min(d, sdSegment(p, vec2f(5.5, 1), vec2f(6.15, 1)));
  d = min(d, sdSegment(p, vec2f(6.15, 1), vec2f(6.4, 0.75)));
  d = min(d, sdSegment(p, vec2f(6.4, 0.75), vec2f(6.4, 0.25)));
  d = min(d, sdSegment(p, vec2f(6.4, 0.25), vec2f(6.15, 0)));
  d = min(d, sdSegment(p, vec2f(6.15, 0), vec2f(5.5, 0)));
  d = min(d, sdSegment(p, vec2f(6.7, 0), vec2f(6.7, 1)));
  d = min(d, sdSegment(p, vec2f(6.7, 1), vec2f(7.6, 1)));
  d = min(d, sdSegment(p, vec2f(6.7, 0.5), vec2f(7.4, 0.5)));
  d = min(d, sdSegment(p, vec2f(6.7, 0), vec2f(7.6, 0)));
  d = min(d, sdSegment(p, vec2f(8.8, 1), vec2f(7.9, 1)));
  d = min(d, sdSegment(p, vec2f(7.9, 1), vec2f(7.9, 0.5)));
  d = min(d, sdSegment(p, vec2f(7.9, 0.5), vec2f(8.8, 0.5)));
  d = min(d, sdSegment(p, vec2f(8.8, 0.5), vec2f(8.8, 0)));
  d = min(d, sdSegment(p, vec2f(8.8, 0), vec2f(7.9, 0)));
  d = min(d, sdSegment(p, vec2f(9.55, 0), vec2f(9.55, 1)));
  d = min(d, sdSegment(p, vec2f(11.2, 1), vec2f(10.3, 1)));
  d = min(d, sdSegment(p, vec2f(10.3, 1), vec2f(10.3, 0)));
  d = min(d, sdSegment(p, vec2f(10.3, 0), vec2f(11.2, 0)));
  d = min(d, sdSegment(p, vec2f(11.2, 0), vec2f(11.2, 0.45)));
  d = min(d, sdSegment(p, vec2f(11.2, 0.45), vec2f(10.8, 0.45)));
  d = min(d, sdSegment(p, vec2f(11.5, 0), vec2f(11.5, 1)));
  d = min(d, sdSegment(p, vec2f(11.5, 1), vec2f(12.4, 0)));
  d = min(d, sdSegment(p, vec2f(12.4, 0), vec2f(12.4, 1)));
  return d;
}

fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f {
  let res = rzResolution();
  var p = (uv - 0.5) * vec2f(res.x / res.y, 1.0);
  p.y -= POS_Y;
  // The sign breathes and sways — quiet motion that says "live shader", not GIF.
  let wobble = 0.05 * sin(time * 0.9);
  let breathe = 1.0 + 0.05 * sin(time * 1.7);
  let c = cos(wobble);
  let si = sin(wobble);
  p = mat2x2f(c, -si, si, c) * p;
  // Screen space → text space (centered).
  let tp = p / (TEXT_SCALE * breathe) + vec2f(12.4 * 0.5, 0.5);
  let d = sdText(vec2f(tp.x, tp.y));

  // Neon = crisp tube + inner hot line + tight halo.
  let aa = fwidth(d) * 1.5;
  let tube = 1.0 - smoothstep(0.06 - aa, 0.06 + aa, d);
  let hot = 1.0 - smoothstep(0.02 - aa, 0.02 + aa, d);
  let core = tube * 0.75 + hot * 0.45;
  let halo = exp(-d * 5.0) * GLOW;
  // Gentle electrical shimmer (two incommensurate frequencies, shallow depth).
  let flicker = 1.0 - FLICKER * (0.5 + 0.5 * sin(time * 7.3) * sin(time * 3.1));
  let s = (core + halo) * flicker;

  // A hue gradient travels along the sign — the "this is a shader" tell.
  let hue = mix(NEON_COLOR, NEON_COLOR_B, 0.5 + 0.5 * sin(time * 1.1 + tp.x * 0.45));
  let color = hue + vec3f(0.35) * core; // core burns toward white
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), clamp(s, 0.0, 1.0));
}
