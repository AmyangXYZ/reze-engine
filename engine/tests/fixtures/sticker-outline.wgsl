// Sticker Outline — a die-cut border around the cast.
//
// A TEMPLATE more than an effect: it is the shortest thing that shows what
// rzCastDistance is for. Every silhouette look is these same three lines with a
// different curve on the end — swap the smoothstep for exp(-d/k) and it is an
// aura, for sin(d*k - t) and it is a shield, for a second read at uv + offset
// and it is a drop shadow. Start here and change the last line.
//
// The engine answers the hard part. Distance to the nearest cast pixel is not a
// property of the pixel — it depends on every pixel around it — so a shader with
// one pass can only go looking, and searching a disc costs O(radius^2). This
// effect used to do exactly that: 64 id samples on every background pixel for a
// 9 pixel border, and over 96 for a 16 pixel one. rzCastDistance is built once
// per frame by a jump flood, costs the same whatever width anyone asks for, and
// is shared by every effect that reads it.
//
// It borders the CAST and nothing else. The ground, a stage and a media plane
// all draw into the id buffer exactly as she does, but only the subjects seed
// the field, so the floor gets no white rectangle around it.

// Tunables — edit a value and hit Cmd-Enter to see it live.
const COLOR = vec3f(1.0, 1.0, 1.0);
const WIDTH = 12.0;       // border thickness, in pixels of a 1440-line frame
const FEATHER = 1.4;      // softness at the outer edge, in real pixels
const OPACITY = 1.0;      // 0..1

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  // THE WIDTH IS A FRACTION OF THE PICTURE, not a count of device pixels.
  //
  // rzCastDistance answers in screen pixels, so a bare 12 is twelve pixels of
  // whatever frame is being drawn — the border tuned in the preview comes out
  // half as thick in a 4K export, which is the copy anyone keeps. Measured
  // against 1440 lines — about what the editor's preview is — it is the same
  // border at every resolution, and the export is the preview enlarged.
  //
  // FEATHER stays in real pixels: it is the anti-aliasing of the cut, and a cut
  // wants the same pixel and a half however many the frame has.
  let width = WIDTH * rzResolution().y / 1440.0;
  let d = rzCastDistance(uv);
  if (d >= width) { return vec4f(0.0); }
  // WHERE IT MEETS HER, faded across the one pixel her edge actually lives in.
  //
  // The distance is signed and its zero crossing is sub-pixel — it comes from the
  // MSAA coverage she is drawn with, not from a single sample — so fading over
  // that crossing lands the border against the same edge the eye sees. Stopping
  // dead at d > 0 instead puts a hard binary step next to an anti-aliased figure,
  // which is what makes a border look stuck on rather than attached.
  let inner = clamp(d + 0.5, 0.0, 1.0);
  // And the outer edge is a cut, not a glow: solid to the feather, then off.
  // 1 - smoothstep(lo, hi), never smoothstep(hi, lo): WGSL leaves it undefined
  // when low >= high, and it is the kind of undefined that works locally.
  let outer = 1.0 - smoothstep(width - FEATHER, width, d);
  return vec4f(COLOR, inner * outer * OPACITY);
}
