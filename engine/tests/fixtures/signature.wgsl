// glyph edges are sub-pixel detail the half-res field pass cannot carry.
//
// The maker's mark: REZE DESIGN with 黯灭小羊 beneath it, one neon sign. The
// title row is the shipped REZE DESIGN lettering; the handle row is the four
// characters' stroke skeletons as segments, drawn by the same tube-and-halo
// math so both rows read as one piece of glass.
// Tunables — edit and ⌘⏎.
const NEON_COLOR = vec3f(0.96, 0.45, 0.71);  // tube color (brand pink)
const NEON_COLOR_B = vec3f(0.45, 0.65, 0.98); // second hue the shimmer travels to
const GLOW = 0.8;                            // halo strength
const FLICKER = 0.1;                         // 0 = steady sign
const TEXT_SCALE = 0.075;                    // sign size on screen
const POS_Y = 0.30;                          // height above center
const HANZI = 1.35;                          // handle row height, in cap heights
const ROW_GAP = 0.5;                         // space between the rows, in cap heights

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

fn sdHz0(p: vec2f) -> f32 {
  let toBox = vec2f(max(abs(p.x - 0.523) - 0.416, 0.0), max(abs(p.y - 0.520) - 0.393, 0.0));
  let boxD = length(toBox);
  if (boxD > 0.7) { return boxD; }
  var d = 1e5;
  d = min(d, sdSegment(p, vec2f(0.132, 0.798), vec2f(0.199, 0.571)));
  d = min(d, sdSegment(p, vec2f(0.156, 0.798), vec2f(0.438, 0.825)));
  d = min(d, sdSegment(p, vec2f(0.438, 0.825), vec2f(0.459, 0.801)));
  d = min(d, sdSegment(p, vec2f(0.459, 0.801), vec2f(0.413, 0.604)));
  d = min(d, sdSegment(p, vec2f(0.222, 0.714), vec2f(0.253, 0.670)));
  d = min(d, sdSegment(p, vec2f(0.369, 0.764), vec2f(0.341, 0.686)));
  d = min(d, sdSegment(p, vec2f(0.216, 0.603), vec2f(0.375, 0.617)));
  d = min(d, sdSegment(p, vec2f(0.276, 0.792), vec2f(0.308, 0.742)));
  d = min(d, sdSegment(p, vec2f(0.308, 0.742), vec2f(0.289, 0.447)));
  d = min(d, sdSegment(p, vec2f(0.195, 0.510), vec2f(0.401, 0.529)));
  d = min(d, sdSegment(p, vec2f(0.137, 0.408), vec2f(0.432, 0.464)));
  d = min(d, sdSegment(p, vec2f(0.135, 0.317), vec2f(0.107, 0.182)));
  d = min(d, sdSegment(p, vec2f(0.206, 0.319), vec2f(0.245, 0.241)));
  d = min(d, sdSegment(p, vec2f(0.302, 0.349), vec2f(0.350, 0.286)));
  d = min(d, sdSegment(p, vec2f(0.396, 0.380), vec2f(0.447, 0.312)));
  d = min(d, sdSegment(p, vec2f(0.628, 0.913), vec2f(0.720, 0.845)));
  d = min(d, sdSegment(p, vec2f(0.548, 0.731), vec2f(0.828, 0.762)));
  d = min(d, sdSegment(p, vec2f(0.564, 0.653), vec2f(0.608, 0.585)));
  d = min(d, sdSegment(p, vec2f(0.736, 0.706), vec2f(0.757, 0.672)));
  d = min(d, sdSegment(p, vec2f(0.757, 0.672), vec2f(0.692, 0.546)));
  d = min(d, sdSegment(p, vec2f(0.457, 0.503), vec2f(0.939, 0.534)));
  d = min(d, sdSegment(p, vec2f(0.524, 0.420), vec2f(0.554, 0.376)));
  d = min(d, sdSegment(p, vec2f(0.554, 0.376), vec2f(0.551, 0.127)));
  d = min(d, sdSegment(p, vec2f(0.574, 0.410), vec2f(0.757, 0.439)));
  d = min(d, sdSegment(p, vec2f(0.757, 0.439), vec2f(0.795, 0.407)));
  d = min(d, sdSegment(p, vec2f(0.795, 0.407), vec2f(0.796, 0.185)));
  d = min(d, sdSegment(p, vec2f(0.796, 0.185), vec2f(0.784, 0.155)));
  d = min(d, sdSegment(p, vec2f(0.784, 0.155), vec2f(0.730, 0.171)));
  d = min(d, sdSegment(p, vec2f(0.586, 0.283), vec2f(0.715, 0.302)));
  d = min(d, sdSegment(p, vec2f(0.580, 0.166), vec2f(0.739, 0.185)));
  return d;
}

fn sdHz1(p: vec2f) -> f32 {
  let toBox = vec2f(max(abs(p.x - 1.630) - 0.365, 0.0), max(abs(p.y - 0.467) - 0.333, 0.0));
  let boxD = length(toBox);
  if (boxD > 0.7) { return boxD; }
  var d = 1e5;
  d = min(d, sdSegment(p, vec2f(1.338, 0.771), vec2f(1.870, 0.801)));
  d = min(d, sdSegment(p, vec2f(1.302, 0.574), vec2f(1.417, 0.459)));
  d = min(d, sdSegment(p, vec2f(1.825, 0.673), vec2f(1.845, 0.634)));
  d = min(d, sdSegment(p, vec2f(1.845, 0.634), vec2f(1.698, 0.497)));
  d = min(d, sdSegment(p, vec2f(1.534, 0.750), vec2f(1.597, 0.696)));
  d = min(d, sdSegment(p, vec2f(1.597, 0.696), vec2f(1.538, 0.338)));
  d = min(d, sdSegment(p, vec2f(1.538, 0.338), vec2f(1.431, 0.221)));
  d = min(d, sdSegment(p, vec2f(1.431, 0.221), vec2f(1.265, 0.147)));
  d = min(d, sdSegment(p, vec2f(1.606, 0.447), vec2f(1.820, 0.178)));
  d = min(d, sdSegment(p, vec2f(1.820, 0.178), vec2f(1.995, 0.134)));
  return d;
}

fn sdHz2(p: vec2f) -> f32 {
  let toBox = vec2f(max(abs(p.x - 2.712) - 0.346, 0.0), max(abs(p.y - 0.525) - 0.351, 0.0));
  let boxD = length(toBox);
  if (boxD > 0.7) { return boxD; }
  var d = 1e5;
  d = min(d, sdSegment(p, vec2f(2.673, 0.876), vec2f(2.721, 0.834)));
  d = min(d, sdSegment(p, vec2f(2.721, 0.834), vec2f(2.716, 0.209)));
  d = min(d, sdSegment(p, vec2f(2.716, 0.209), vec2f(2.698, 0.174)));
  d = min(d, sdSegment(p, vec2f(2.698, 0.174), vec2f(2.550, 0.230)));
  d = min(d, sdSegment(p, vec2f(2.473, 0.586), vec2f(2.366, 0.389)));
  d = min(d, sdSegment(p, vec2f(2.892, 0.604), vec2f(3.028, 0.492)));
  d = min(d, sdSegment(p, vec2f(3.028, 0.492), vec2f(3.058, 0.425)));
  return d;
}

fn sdHz3(p: vec2f) -> f32 {
  let toBox = vec2f(max(abs(p.x - 3.809) - 0.404, 0.0), max(abs(p.y - 0.516) - 0.448, 0.0));
  let boxD = length(toBox);
  if (boxD > 0.7) { return boxD; }
  var d = 1e5;
  d = min(d, sdSegment(p, vec2f(3.637, 0.915), vec2f(3.736, 0.827)));
  d = min(d, sdSegment(p, vec2f(3.944, 0.964), vec2f(3.969, 0.924)));
  d = min(d, sdSegment(p, vec2f(3.969, 0.924), vec2f(3.850, 0.812)));
  d = min(d, sdSegment(p, vec2f(3.600, 0.720), vec2f(4.013, 0.763)));
  d = min(d, sdSegment(p, vec2f(3.623, 0.562), vec2f(3.964, 0.600)));
  d = min(d, sdSegment(p, vec2f(3.404, 0.399), vec2f(3.467, 0.381)));
  d = min(d, sdSegment(p, vec2f(3.467, 0.381), vec2f(4.105, 0.452)));
  d = min(d, sdSegment(p, vec2f(4.105, 0.452), vec2f(4.213, 0.421)));
  d = min(d, sdSegment(p, vec2f(3.821, 0.703), vec2f(3.795, 0.068)));
  return d;
}

// The handle row: per-character box rejects keep the segment loops off almost
// every pixel; the min of the boxes still approximates distance for the halo.
fn sdHanzi(p: vec2f) -> f32 {
  return min(min(sdHz0(p), sdHz1(p)), min(sdHz2(p), sdHz3(p)));
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
  // Screen space → title space (centered); the handle row lives below it in the
  // same units, so one hue gradient and one flicker cross both.
  let q = p / (TEXT_SCALE * breathe) + vec2f(12.4 * 0.5, 0.5);
  let d1 = sdText(q);
  // Handle row: 4.26 hanzi units wide, scaled by HANZI and centered under the
  // title. Distances scale back by HANZI so both rows share one unit.
  let h = (q - vec2f((12.4 - 4.26 * HANZI) * 0.5, -(ROW_GAP + HANZI))) / HANZI;
  let d2 = sdHanzi(h) * HANZI;

  // Neon = crisp tube + inner hot line + tight halo. The handle's tubes are
  // thinner — its strokes sit closer than any pair of latin stems, and at the
  // title's radius they would fuse into a blob.
  let aa1 = fwidth(d1) * 1.5;
  let core1 = (1.0 - smoothstep(0.06 - aa1, 0.06 + aa1, d1)) * 0.75 + (1.0 - smoothstep(0.02 - aa1, 0.02 + aa1, d1)) * 0.45;
  let aa2 = fwidth(d2) * 1.5;
  let core2 = (1.0 - smoothstep(0.038 - aa2, 0.038 + aa2, d2)) * 0.75 + (1.0 - smoothstep(0.014 - aa2, 0.014 + aa2, d2)) * 0.45;
  let halo = exp(-d1 * 5.0) * GLOW + exp(-d2 * 6.5) * GLOW * 0.9;
  // Gentle electrical shimmer (two incommensurate frequencies, shallow depth).
  let flicker = 1.0 - FLICKER * (0.5 + 0.5 * sin(time * 7.3) * sin(time * 3.1));
  let core = core1 + core2;
  let s = (core + halo) * flicker;

  // A hue gradient travels along the sign — the "this is a shader" tell.
  let hue = mix(NEON_COLOR, NEON_COLOR_B, 0.5 + 0.5 * sin(time * 1.1 + q.x * 0.45));
  let color = hue + vec3f(0.35) * core; // core burns toward white
  return vec4f(clamp(color, vec3f(0.0), vec3f(1.0)), clamp(s, 0.0, 1.0));
}
