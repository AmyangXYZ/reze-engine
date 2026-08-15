// @layer additive
// @fullres

// Summoning Circle — a sigil on the ground under the cast, and the light
// standing on it.
//
// The sigil is a port of nayk's summoningCircle():
//   https://www.shadertoy.com/view/M3KGD3          (nayk, 2024-06-14)
// which itself credits
//   https://glslsandbox.com/e#109235.1
//   https://www.shadertoy.com/view/43V3zG
// The construction below is transcribed rather than reinterpreted — the same
// helpers (lPoly, mPoly, strokeStar, wtrz), the same figures in the same order,
// the same constants. Its shape is three chords struck across a triangle, a
// rimmed ring whose outer stroke breathes on a trapezoidal wave, nested
// triangles and a hexagon, and a hexagram set into a disc at each vertex.
//
// What is NOT the original is where it lives. This draws on the GROUND PLANE
// through rzSubject().root, found by intersecting the view ray with it, so the
// figure lies under the character in the world instead of across the frame.
//
// It is a FOREGROUND mount even though it draws on the ground, and that is
// deliberate. A background mount only survives where the scene did not cover
// the pixel, so a scene with a ground plane would hide the sigil completely.
// Drawn forward and depth-tested by hand, it lies on the floor whether or not
// there is a floor, and her legs cut into it correctly either way.
//
// The one place the port had to change: the original's line widths and feathers
// are constants in a space where the figure fills the frame, so a pixel is
// always far smaller than the thinnest of them. On a ground plane raking away
// toward the horizon a pixel covers a great deal of sigil, and those strokes
// fall below it and break into dashes. So every width and every feather has a
// floor of one measured pixel — metres-per-pixel converted at the point being
// drawn, which holds at any distance, any field of view and any grazing angle.
// Up close nothing is touched and the figure is the original's exactly.
//
// Tunables — edit and ⌘⏎.
const RING_COLOR = vec3f(0.64, 0.34, 1.0);   // the lines — violet
const HOT_COLOR = vec3f(0.92, 0.84, 1.0);    // where they are brightest
const COL_COLOR = vec3f(0.40, 0.07, 0.90);   // deep purple, in the faint light
const SIZE = 1.60;        // sigil radius, in hip heights
const EXTENT = 1.30;      // how far past that radius the FIGURE actually goes.
                          // The rim is at 1.0, but mPoly centres the three
                          // vertex discs ON it and they are 0.2 across, so the
                          // drawing reaches 1.2 — gate any tighter and the
                          // discs come back sliced off, which is exactly what
                          // a rim-radius gate did to them.
const SPIN = 0.05;        // turns per second
const LINE_W = 1.3;       // the one-pixel floor, in pixels
const RING_I = 1.35;      // how hot a line is
const COL_R = 1.05;       // column radius, in hip heights — wide enough to
                          // stand the whole body inside it
const COL_H = 3.0;        // and height, the same
const COL_I = 0.50;
const PULSE = 0.18;       // depth of the breathing brightness
const PULSE_HZ = 0.5;
const Z_BIAS = 0.03;      // metres of slack before the scene occludes it
const Z_FADE = 0.04;

const PI = 3.14159265;
const TAU = 6.2831853;

fn sat(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

/** p turned by a. GLSL's `p * rot(a)` is a row vector times a column-major
 *  matrix, which comes out as the ordinary counter-clockwise turn. */
fn rot2(p: vec2f, a: f32) -> vec2f {
  let c = cos(a);
  let s = sin(a);
  return vec2f(p.x * c - p.y * s, p.x * s + p.y * c);
}

// The original's signed-distance shorthand: inside is positive.
fn sdR(d: f32, r: f32) -> f32 { return r - d; }
fn sd1(d: f32) -> f32 { return 1.0 - d; }

fn fillA(d: f32, aa: f32) -> f32 { return smoothstep(0.0, aa, d); }

/** Both the width and the feather take a floor of one pixel. Without it the
 *  0.005-wide strokes below vanish wherever the plane rakes away. */
fn strokeA(d: f32, w: f32, aa: f32) -> f32 {
  let ww = max(w, aa * 0.5);
  return 1.0 - smoothstep(ww, ww + aa, abs(d));
}
fn strokeInnerA(d: f32, w: f32, aa: f32) -> f32 { return strokeA(d - w, w, aa); }
fn strokeOuterA(d: f32, w: f32, aa: f32) -> f32 { return strokeA(d + w, w, aa); }

fn lSquare(p: vec2f) -> f32 { let q = abs(p); return max(q.x, q.y); }

/** Distance to a regular n-gon, by folding the angle onto its nearest edge. */
fn lPoly(p: vec2f, n: f32) -> f32 {
  let a = atan2(p.x, p.y) + PI;
  let r = TAU / n;
  return cos(floor(0.5 + a / r) * r - a) * length(p) / cos(r * 0.5);
}

/** Fold the plane into n wedges, each measured from a point s out along its
 *  own axis — how the vertex discs get drawn once and appear three times. */
fn mPoly(p: vec2f, n: f32, s: f32) -> vec2f {
  let r = TAU / n;
  let a = floor(atan2(p.y, p.x) / r) * r + r * 0.5;
  return rot2(vec2f(cos(a), sin(a)) * s - p, -a - PI * 0.5);
}

fn wtri(x: f32) -> f32 { return abs(2.0 * fract(x * 0.5 - 0.25) - 1.0) * 2.0 - 1.0; }
/** Trapezoidal wave — the outer ring's width steps between two values on it. */
fn wtrz(x: f32, w: f32) -> f32 { return clamp(wtri(x * 2.0) * w, -1.0, 1.0); }

/** Two n/2-gons, the second turned or mirrored: for n = 6, a hexagram. */
fn strokeStar(p: vec2f, n: f32, w: f32, aa: f32) -> f32 {
  var l = strokeInnerA(sd1(lPoly(p, n * 0.5)), w, aa);
  let odd = (n - 2.0 * floor(n * 0.5)) != 0.0;
  let p2 = select(rot2(p, TAU / n), vec2f(-p.x, p.y), odd);
  l = l + strokeInnerA(sd1(lPoly(p2, n * 0.5)), w, aa);
  return l;
}

/** nayk's figure, transcribed. p is in sigil radii; 1.0 is the rim. */
fn summoningCircle(p: vec2f, aa: f32) -> f32 {
  var l = 0.0;
  // Three chords struck across the figure — thin bands through the centre,
  // squashed a hundred to one so lSquare reads them as lines.
  l = l + fillA(sdR(lSquare(rot2(p, PI / 3.0 * 1.5) * vec2f(100.0, 1.0)), 1.0), aa);
  l = l + fillA(sdR(lSquare(rot2(p, PI / 3.0 * 2.5) * vec2f(100.0, 1.0)), 1.0), aa);
  l = l + fillA(sdR(lSquare(rot2(p, PI / 3.0 * 3.5) * vec2f(100.0, 1.0)), 1.0), aa);
  l = sat(l);
  // and cut back out of the triangle, so they read as spokes from its edges.
  l = l - fillA(sd1(lPoly(p, 3.0)), aa);
  l = sat(l);

  let r = atan2(p.y, p.x);
  // The rim: outer stroke breathing between two widths three times around.
  l = l + strokeOuterA(sdR(length(p), 0.98), 0.008 + wtrz(r / TAU * 3.0, 12.0) * 0.005, aa);
  l = l + strokeInnerA(sdR(length(p), 0.95), 0.005, aa);
  // Nested triangles, and the hexagon between them.
  l = l + strokeInnerA(sd1(lPoly(p, 3.0)), 0.01, aa);
  l = l + strokeInnerA(sdR(lPoly(p, 3.0), 0.88), 0.02, aa);
  l = l + strokeInnerA(sdR(lPoly(p, 6.0), 0.53), 0.01, aa);
  let q = mPoly(rot2(p, PI * 0.5), 3.0, 0.5);
  l = l + fillA(sdR(lPoly(q, 3.0), 0.3), aa);
  let q2 = mPoly(rot2(p, PI / 3.0 + PI * 0.5), 3.0, 0.7);
  l = l + fillA(sdR(lPoly(q2, 3.0), 0.1), aa);
  l = l + strokeInnerA(sdR(lPoly(rot2(p, PI), 3.0), 0.5), 0.02, aa);
  l = l + fillA(sdR(length(p), 0.05), aa);

  // A disc at each triangle vertex: punched out of everything above, ringed
  // twice, with a hexagram set into it.
  let q3 = mPoly(rot2(p, PI * 0.5), 3.0, 1.0);
  l = sat(l);
  l = l - fillA(sdR(length(q3), 0.2), aa);
  l = sat(l);
  l = l + strokeInnerA(sdR(length(q3), 0.18), 0.005, aa);
  l = l + strokeInnerA(sdR(length(q3), 0.15), 0.005, aa);
  l = l + strokeStar(rot2(q3, PI) * 7.0, 6.0, 0.1, aa);
  return l;
}

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let res = rzResolution();
  let ro = rzCameraPos();
  var lines = 0.0;
  var beam = 0.0;

  for (var c = 0; c < rzSubjectCount(); c++) {
    let s = rzSubject(c);
    if (!s.valid) { continue; }
    // Sized off the HIP HEIGHT. bounds.w is a deliberately generous cull
    // sphere, not a fit — sizing off it put the sigil several body heights
    // across, so only ever a fragment of it was on screen. Root to hips is a
    // dependable fraction of any rig's height, and it costs nothing to read.
    let H = max(s.center.y - s.root.y, 0.05);
    let R = H * SIZE;
    // Breathing, per character and out of phase, so a duo does not pulse as one.
    let pulse = 1.0 + PULSE * sin((time * PULSE_HZ + f32(c) * 0.37) * TAU);

    // ── The sigil, on the ground plane through root ──
    if (abs(ray.y) > 1e-4) {
      let t = (s.root.y - ro.y) / ray.y;
      if (t > 0.0) {
        let hit = ro + ray * t;
        let rel = hit.xz - s.root.xz;
        if (dot(rel, rel) < R * R * EXTENT * EXTENT) {
          let at = rzProject(hit);
          // Hidden by anything nearer — her legs, a prop, the lip of a stage.
          // Soft, or the boundary crawls as she moves over it.
          let vis = 1.0 - smoothstep(depth + Z_BIAS - Z_FADE, depth + Z_BIAS + Z_FADE, at.z);
          if (vis > 0.001) {
            // One pixel, in sigil radii, HERE — measured rather than assumed,
            // which is what keeps the strokes whole as the plane rakes away and
            // one pixel starts covering a great deal of ground.
            let up = rzProject(hit + rzCameraUp() * 0.25);
            let uvPerM = max(abs(up.y - at.y) * 4.0, 1e-5);
            let aa = max(0.01, LINE_W / (uvPerM * max(res.y, 1.0) * R));
            let sig = summoningCircle(rot2(rel / R, time * SPIN * TAU), aa);
            lines = max(lines, sat(sig) * vis * pulse);
          }
        }
      }
    }

    // ── The column standing on it ──
    //
    // A vertical cylinder, shaded from the ray's CLOSEST APPROACH to its axis
    // rather than marched: for a shape this soft the nearest point carries the
    // whole look, and a march would cost thirty samples to say the same thing.
    let denom = 1.0 - ray.y * ray.y;
    if (denom > 1e-4) {
      let w0 = ro - s.root;
      let tc = (ray.y * w0.y - dot(ray, w0)) / denom;
      if (tc > 0.0) {
        let q = ro + ray * tc;
        let h = (q.y - s.root.y) / (H * COL_H);
        let rad = length(q.xz - s.root.xz) / (H * COL_R);
        if (h > 0.0 && h < 1.0 && rad < 1.0) {
          let qat = rzProject(q);
          let vis = 1.0 - smoothstep(depth + Z_BIAS - Z_FADE, depth + Z_BIAS + Z_FADE, qat.z);
          // Squared across the radius rather than cubed: a shaft of light
          // still, but one that carries out to its edge instead of collapsing
          // onto its axis — a cubic falloff is down to an eighth at the halfway
          // mark, which leaves the body standing outside its own light. Fading
          // out at the top, and in off the ground so it does not end in a hard
          // disc where it meets the sigil.
          let radial = 1.0 - smoothstep(0.0, 1.0, rad);
          let vert = (1.0 - h) * smoothstep(0.0, 0.12, h);
          beam = max(beam, radial * radial * vert * vis * pulse);
        }
      }
    }
  }

  let heat = lines * RING_I + beam * COL_I;
  if (heat <= 0.004) { return vec4f(0.0); }
  // Deep purple in the faint light, violet through the strokes, and a pale
  // lilac only where they are hottest.
  var rgb = mix(COL_COLOR, RING_COLOR, smoothstep(0.05, 0.6, heat));
  rgb = mix(rgb, HOT_COLOR, smoothstep(1.0, 1.9, heat));
  return vec4f(rgb, clamp(heat, 0.0, 1.0));
}
