import { RZ_LIGHT_STRUCT_WGSL } from "../lights"
import { audioApi } from "../audio-api"
import { anchorAliasWgsl, ribbonSlotWgsl } from "../anchor-table"
import { scoreApi } from "../score-api"
// Ribbons along a bone's recorded path, drawn as geometry.
//
// This is the effect the whole geometry path was built for. As a fullscreen
// field the hand ribbon cost O(pixels × trail samples) — every pixel on screen
// walking 128 samples to decide it was not on the ribbon — and it was the single
// most expensive built-in by a wide margin. As a strip of quads it costs the
// segments: 127 of them, whatever the resolution.
//
// It draws from the SAME history the field version read through rzTrail. The
// engine already records every declared anchor's path at a fixed rate on the
// scene clock, so nothing new is sampled and an exported ribbon is identical to
// the one in the editor.
//
// The author writes two functions: how wide the ribbon is at a point, and what
// colour. Everything else — which bones, which characters, where the samples
// are, how to face the camera — is the engine's.

export type TrailSource = {
  /** The author's WGSL verbatim. */
  wgsl: string
  /** How many ribbons to draw — one per trailed anchor. */
  slots: number
  /** For each ribbon, the LOCAL anchor slot it belongs to. Identity when every
   *  anchor is trailed; otherwise it skips the untrailed ones. */
  ribbonSlots: number[]
  /** Additive, like most glowing ribbons, or straight alpha. */
  blend: "alpha" | "additive"
  bloom: boolean
}

/**
 * Sub-segments drawn between each pair of recorded samples.
 *
 * The path is sampled at a fixed rate on the scene clock, so a fast hand leaves
 * its samples far apart and a strip drawn straight between them is visibly
 * faceted. Four sub-segments on a Catmull-Rom curve through the neighbours costs
 * four times the vertices — which is nothing, they are vertices — and removes
 * both the faceting and most of the jitter, since a spline tangent varies
 * smoothly where a per-segment direction snaps about whenever the hand slows.
 */
export const TRAIL_SUBDIVISIONS = 6

/** Does the source define the trail contract? Both are required together. */
export function trailEntryPoints(wgsl: string): { width: boolean; shade: boolean } {
  return {
    width: /\bfn\s+trailWidth\s*\(/.test(wgsl),
    shade: /\bfn\s+trailShade\s*\(/.test(wgsl),
  }
}

/**
 * One quad per segment, laid out flat across every anchor and character.
 *
 * The instance index encodes all three — segment, subject, slot — so a scene
 * with three dancers and eight declared bones is still ONE draw call and needs
 * nothing computed on the CPU per frame. Instances past the end of a real trail
 * collapse to a degenerate quad, which costs a vertex shader invocation and no
 * fragments; the alternative is a compacted instance list, which costs a
 * readback every frame to save exactly that.
 *
 * The ribbon faces the camera per SEGMENT rather than as a whole: the side
 * vector is the segment direction crossed with the direction to the eye, so a
 * ribbon that loops back on itself stays visible along its entire length instead
 * of vanishing edge-on where it turns.
 */
export function buildTrailShader(
  src: TrailSource,
  cast: {
    subjects: number
    samples: number
    base: number
    trailBase: number
    slots: number
    alias: number[]
    /** Depth convention of the scene buffer this layer tests against. The
     *  occlusion compare below is MANUAL (the layer has no depth attachment),
     *  so it does not flip with the pipelines' depthCompare — it has to be
     *  emitted the right way round at build time. On a reversed-Z device,
     *  larger z is CLOSER; the unflipped test drew ribbons only when occluded. */
    reversedZ: boolean
  },
): string {
  return (
    `const RZ_SUBJECTS: i32 = ${cast.subjects};
const RZ_SAMPLES: i32 = ${cast.samples};
const RZ_SLOTS: i32 = ${src.slots};
const RZ_TRAIL_SLOTS: i32 = ${src.slots};
// The anchor ADDRESS SPACE — distinct from RZ_SLOTS, which is how many ribbons
// this effect draws. The accessors bound by this; the instance loop by that.
const RZ_MAX_ANCHORS: i32 = ${cast.slots};
${ribbonSlotWgsl(src.ribbonSlots)}
${anchorAliasWgsl(cast.alias)}
const SUB: i32 = ${TRAIL_SUBDIVISIONS};

// The same struct and helpers the particle modules define. The whole effect
// file is spliced into EVERY module its mounts compile into, so an effect that
// declares both trails and particles carries its particle functions through
// this module as dead code — dead code that still has to type-check. Keeping
// the two preludes' public surface identical is what makes one file, two
// mounts work.
struct Particle {
  pos: vec3f,
  age: f32,
  vel: vec3f,
  life: f32,
  size: f32,
  rot: f32,
  seed: f32,
  stretch: f32,
}
fn rzDt() -> f32 { return 0.0; }
fn rzHash21(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  p3 = p3 + dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}
fn rzHash31(p: vec3f) -> f32 {
  var p3 = fract(p * 0.1031);
  p3 = p3 + dot(p3, p3.zyx + 31.32);
  return fract((p3.x + p3.y) * p3.z);
}
fn rzHash13(x: f32) -> vec3f {
  return vec3f(rzHash11(x), rzHash11(x + 17.13), rzHash11(x + 41.71));
}
fn rzValueNoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let n000 = rzHash31(i);
  let n100 = rzHash31(i + vec3f(1.0, 0.0, 0.0));
  let n010 = rzHash31(i + vec3f(0.0, 1.0, 0.0));
  let n110 = rzHash31(i + vec3f(1.0, 1.0, 0.0));
  let n001 = rzHash31(i + vec3f(0.0, 0.0, 1.0));
  let n101 = rzHash31(i + vec3f(1.0, 0.0, 1.0));
  let n011 = rzHash31(i + vec3f(0.0, 1.0, 1.0));
  let n111 = rzHash31(i + vec3f(1.0, 1.0, 1.0));
  let x00 = mix(n000, n100, u.x);
  let x10 = mix(n010, n110, u.x);
  let x01 = mix(n001, n101, u.x);
  let x11 = mix(n011, n111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
// World units a sample pair must span to count as a FULL contribution — the
// original's REF_SEG_PX, converted out of pixels.
//
// This is a line integral along a path, so a segment should contribute in
// proportion to the path it covers. Without it a hand that pauses stacks fifty
// samples on one point and every one of them contributes fully, which is what
// turned the ribbon white wherever it slowed. It scales INTENSITY, not width:
// scaling width pinches the strip shut at exactly the moments a hand decelerates,
// which is every turn — and a ribbon that closes to nothing at each turn reads as
// dark breaks in the middle of it.
const RZ_REF_SPAN: f32 = 0.37;

struct CameraU {
  view: mat4x4f,
  proj: mat4x4f,
  camPos: vec3f,
  targetHeight: f32,
}
struct TrailU {
  time: f32,
  _pad0: f32,
  _pad1: f32,
  _pad2: f32,
}
@group(0) @binding(0) var<storage, read> _rzCast: array<vec4f>;
@group(0) @binding(1) var<uniform> tu: TrailU;
@group(0) @binding(2) var<uniform> cam: CameraU;
// Binding 3 is GONE, and it had to go. Ribbons draw inside the scene pass now,
// and the depth buffer is that pass's render attachment — binding it for
// sampling in the same pass is a usage conflict WebGPU rejects outright:
// "includes writable usage and another usage in the same synchronization
// scope". The hardware depth test replaced what it was read for.

fn rzTime() -> f32 { return tu.time; }
${RZ_LIGHT_STRUCT_WGSL}
fn rzCameraPos() -> vec3f { return cam.camPos; }
fn rzCameraRight() -> vec3f { return vec3f(cam.view[0][0], cam.view[1][0], cam.view[2][0]); }
fn rzCameraUp() -> vec3f { return vec3f(cam.view[0][1], cam.view[1][1], cam.view[2][1]); }
fn rzCameraForward() -> vec3f { return vec3f(cam.view[0][2], cam.view[1][2], cam.view[2][2]); }
/** World point → (uv, view distance), same contract as the field mounts' rzProject. */
fn rzProject(p: vec3f) -> vec3f {
  let clip = cam.proj * cam.view * vec4f(p, 1.0);
  let w = max(clip.w, 1e-4);
  return vec3f(clip.xy / w * 0.5 + 0.5, clip.w);
}
fn rzViewportHeight() -> f32 { return cam.targetHeight; }
fn rzCamPos() -> vec3f { return cam.camPos; }
fn rzSubjectCount() -> i32 {
  var n = 0;
  for (var i = 0; i < RZ_SUBJECTS; i++) {
    if (_rzCast[i * 3 + 2].w > 0.0) { n = i + 1; }
  }
  return n;
}
fn rzTrailCount(subject: i32, slot: i32) -> i32 {
  // Bounded by the scene's anchor cap, NOT by how many anchors asked for a
  // trail. Those are different index spaces: storage is addressed by anchor
  // slot, so an untrailed @anchor followed by a trailed one put the trail at
  // index 1 with a bound of 1, and rzTrail returned zero — a ribbon that
  // silently did not draw. The bound was redundant: an untrailed slot already
  // reports a recorded count of zero.
  let g = _rzSlot(slot);
  if (subject < 0 || subject >= RZ_SUBJECTS || g < 0 || g >= RZ_MAX_ANCHORS) { return 0; }
  return i32(_rzCast[${cast.base} + (g * RZ_SUBJECTS + subject) * 3 + 2].w);
}
/** Sample i of a path: xyz where it was, w how many seconds ago. i = 0 is now. */
fn rzTrail(subject: i32, slot: i32, i: i32) -> vec4f {
  let n = rzTrailCount(subject, slot);
  if (i < 0 || i >= n) { return vec4f(0.0); }
  return _rzCast[${cast.trailBase} + (_rzSlot(slot) * RZ_SUBJECTS + subject) * RZ_SAMPLES + i];
}
/** Catmull-Rom through four samples — passes through p1 and p2. */
fn rzSpline(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let t2 = t * t;
  let t3 = t2 * t;
  return 0.5 * ((2.0 * p1) + (-p0 + p2) * t + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2 +
                (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
}
/** Its derivative — the ribbon's direction, smooth by construction. */
fn rzSplineTangent(p0: vec3f, p1: vec3f, p2: vec3f, p3: vec3f, t: f32) -> vec3f {
  let t2 = t * t;
  let d = 0.5 * ((-p0 + p2) + 2.0 * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t +
                 3.0 * (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t2);
  if (length(d) < 1e-6) { return vec3f(0.0, 1.0, 0.0); }
  return normalize(d);
}
fn rzTrailAt(subject: i32, slot: i32, i: i32, n: i32) -> vec4f {
  return rzTrail(subject, slot, clamp(i, 0, n - 1));
}

/**
 * The direction the ribbon runs AT a sample, not along one segment.
 *
 * Averaging the incoming and outgoing directions gives a mitre joint: both quads
 * meeting at a sample compute the same tangent, so they share an edge exactly.
 * Taking each segment's own direction instead leaves a wedge at every bend —
 * which under additive blending is a bright band, and is what put visible
 * stripes across the ribbon wherever the hand turned.
 */
fn rzTangentAt(subject: i32, slot: i32, i: i32, n: i32) -> vec3f {
  let p = rzTrail(subject, slot, i).xyz;
  var d = vec3f(0.0);
  if (i > 0) {
    let q = rzTrail(subject, slot, i - 1).xyz;
    let e = p - q;
    if (length(e) > 1e-7) { d = d + normalize(e); }
  }
  if (i + 1 < n) {
    let q = rzTrail(subject, slot, i + 1).xyz;
    let e = q - p;
    if (length(e) > 1e-7) { d = d + normalize(e); }
  }
  if (length(d) < 1e-6) { return vec3f(0.0, 1.0, 0.0); }
  return normalize(d);
}
/**
 * Circumradius of three consecutive samples — the path's local turn radius.
 *
 * The ribbon's half-width must not exceed this. Where it does, the inner edges
 * of adjacent quads cross the centre of the turn and overlap, and under
 * additive blending every overlap doubles into a bright transverse wedge — the
 * bands that survived every brightness fix, because they were geometry, not
 * shading. Collinear or stationary samples return a huge radius: a straight
 * path clamps nothing.
 */
fn rzTurnRadius(a: vec3f, b: vec3f, c: vec3f) -> f32 {
  let ab = b - a;
  let bc = c - b;
  let ca = a - c;
  let la = length(ab);
  let lb = length(bc);
  let cr = length(cross(ab, ca));
  if (la < 1e-5 || lb < 1e-5 || cr < 1e-7) { return 1e6; }
  return (la * lb * length(ca)) / (2.0 * cr);
}
fn rzFalloff(d: f32, r: f32) -> f32 {
  let x = clamp(d / max(r, 1e-6), 0.0, 1.0);
  let f = 1.0 - x;
  return f * f * f;
}
fn rzHash11(x: f32) -> f32 {
  var p = fract(x * 0.1031);
  p = p * (p + 33.33);
  return fract(p * (p + p));
}
` +
    audioApi(0, 4) +
    scoreApi(0, 5) +
    "\n// ── user effect ──\n" +
    src.wgsl +
    /* wgsl */ `
struct VSOut {
  @builtin(position) clip: vec4f,
  @location(0) uv: vec2f,
  @location(1) age: f32,
  @location(2) weight: f32,
  // Which declared anchor this quad belongs to — so one effect can dress its
  // hand and foot ribbons differently instead of sharing a single look.
  @location(3) @interpolate(flat) slot: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  var out: VSOut;
  let subs = u32((RZ_SAMPLES - 1) * SUB);
  let k = i32(ii % subs);
  let rest = ii / subs;
  let seg = k / SUB;
  let sub = k % SUB;
  let subject = i32(rest % u32(RZ_SUBJECTS));
  // Instance index counts RIBBONS; the cast buffer is addressed by anchor slot.
  // One ribbon per trailed anchor, so these are the same number only when every
  // anchor is trailed — _rzRibbonSlot is what bridges them.
  let ribbon = i32(rest / u32(RZ_SUBJECTS));
  let slot = _rzRibbonSlot(ribbon);

  let n = rzTrailCount(subject, slot);
  if (ribbon >= RZ_SLOTS || seg + 1 >= n) {
    out.clip = vec4f(0.0, 0.0, -2.0, 1.0);
    out.uv = vec2f(0.0);
    out.age = 0.0;
    return out;
  }

  let quad = array<vec2f, 6>(
    vec2f(0.0, -1.0), vec2f(1.0, -1.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
  );
  let c = quad[vi];
  let atEnd = c.x > 0.5;

  // Position ON THE CURVE, not on the chord. Both quads meeting at a sub-sample
  // evaluate the same t, so they share an edge exactly — no wedge to double-blend
  // into a bright band, which is what the per-segment version left at every bend.
  let t0 = f32(sub) / f32(SUB);
  let t1 = (f32(sub) + 1.0) / f32(SUB);
  let t = select(t0, t1, atEnd);
  let p0 = rzTrailAt(subject, slot, seg - 1, n).xyz;
  let p1 = rzTrail(subject, slot, seg).xyz;
  let p2 = rzTrail(subject, slot, seg + 1).xyz;
  let p3 = rzTrailAt(subject, slot, seg + 2, n).xyz;
  let H = cam.targetHeight;
  let toPx = vec2f((cam.proj[1][1] / cam.proj[0][0]) * H, H) * 0.5;
  // The original's MIN_SEG_PX gate, restored as a HARD gate — not a fade.
  //
  // "Below this the hand is not moving", its comment said, and it meant it: a
  // segment under 0.7px contributed NOTHING. Softening that into a weight that
  // approaches zero let a pause stack dozens of near-stationary quads at full
  // glow width, each individually faint, summing into a wide bright bar exactly
  // where the hand slows — aggregation the width taper never clamps, because
  // width follows AGE, not speed. Where the hand is not moving there is no path,
  // and a path effect should draw nothing there at all.
  let c1 = cam.proj * cam.view * vec4f(p1, 1.0);
  let c2 = cam.proj * cam.view * vec4f(p2, 1.0);
  if (c1.w <= 0.01 || c2.w <= 0.01 ||
      distance(c1.xy / c1.w * toPx, c2.xy / c2.w * toPx) < 0.7 * (H / 1080.0)) {
    out.clip = vec4f(0.0, 0.0, -2.0, 1.0);
    out.uv = vec2f(0.0);
    out.age = 0.0;
    out.weight = 0.0;
    return out;
  }
  let pA = rzSpline(p0, p1, p2, p3, t0);
  let pB = rzSpline(p0, p1, p2, p3, t1);
  let p = select(pA, pB, atEnd);

  // Age and position along the ribbon interpolate across the sub-segment too.
  let ageA = rzTrail(subject, slot, seg).w;
  let ageB = rzTrail(subject, slot, seg + 1).w;
  let age = mix(ageA, ageB, t);
  let u = (f32(seg) + t) / f32(max(1, n - 1));

  // SCREEN-SPACE extrusion — the fresh take, and the faithful one.
  //
  // Every artefact this ribbon has had came from building it in 3D with a
  // camera-facing side vector: fold-over wedges at tight turns, bowties when the
  // tangent swept past the view axis, and twisted quads whose core line rotated
  // ACROSS the strip and drew as a bright rung. The fullscreen original had none
  // of them, because it never left 2D: it shaded distance to the PROJECTED path,
  // in pixels. So project the spline to pixel space and extrude there — the same
  // geometry the original shaded, at a quad's price. A 2D perpendicular cannot
  // twist, and pixel width is what the original's constants meant all along.
  let clipA = cam.proj * cam.view * vec4f(pA, 1.0);
  let clipB = cam.proj * cam.view * vec4f(pB, 1.0);
  if (clipA.w <= 0.01 || clipB.w <= 0.01) {
    out.clip = vec4f(0.0, 0.0, -2.0, 1.0);
    out.uv = vec2f(0.0);
    out.age = 0.0;
    out.weight = 0.0;
    return out;
  }
  // The projected tangent by the quotient rule — exact at the knot, so the two
  // quads meeting there derive the identical perpendicular and share their edge.
  let dA4 = cam.proj * cam.view * vec4f(rzSplineTangent(p0, p1, p2, p3, t0), 0.0);
  let dB4 = cam.proj * cam.view * vec4f(rzSplineTangent(p0, p1, p2, p3, t1), 0.0);
  var tanA = (dA4.xy * clipA.w - clipA.xy * dA4.w) * toPx;
  var tanB = (dB4.xy * clipB.w - clipB.xy * dB4.w) * toPx;
  if (length(tanA) < 1e-5) { tanA = vec2f(1.0, 0.0); }
  if (length(tanB) < 1e-5) { tanB = vec2f(1.0, 0.0); }
  var perpA = normalize(vec2f(-tanA.y, tanA.x));
  var perpB = normalize(vec2f(-tanB.y, tanB.x));
  // A projected cusp flips the perpendicular. The shade is symmetric in |v|, so
  // forcing the quad's ends to agree is invisible — and without it the quad is a
  // 2D bowtie.
  if (dot(perpA, perpB) < 0.0) { perpB = -perpB; }
  let clipP = select(clipA, clipB, atEnd);
  let perp = select(perpA, perpB, atEnd);
  // trailWidth speaks PIXELS, like the original's constants; rzViewportHeight()
  // lets the effect scale them to the frame.
  let wPx = max(0.0, trailWidth(u, age));
  let ndc = clipP.xy / clipP.w + perp * (c.y * wPx) / toPx;
  out.clip = vec4f(ndc * clipP.w, clipP.z, clipP.w);
  out.slot = u32(slot);
  // The line-integral weight: how far the hand travelled, evaluated AT EACH KNOT
  // by central difference and mixed by t so it is continuous across boundaries —
  // per-pair spans made dashes, per-segment averages made terraces.
  let spanA = distance(p2, p0) * 0.5;
  let spanB = distance(p3, p1) * 0.5;
  out.weight = clamp(mix(spanA, spanB, t) / RZ_REF_SPAN, 0.0, 1.0);
  out.uv = vec2f(u, c.y);
  out.age = age;
  return out;
}

struct TrailFSOut {
  @location(0) color: vec4f,
  // The scene's aux target: (bloom mask, coverage). Writing mask 1 is what
  // makes a ribbon BLOOM — it is light, and the gate is what says so.
  @location(1) aux: vec4f,
}

@fragment
fn fs(in: VSOut) -> TrailFSOut {
  // No manual depth test any more. This drew into its own attachment-less
  // layer and compared position.z against the scene's depth texture by hand,
  // with the direction baked per depth convention. Inside the scene pass the
  // hardware does it, correctly and for free, and the reversed-Z trap that
  // needed its own regression test goes with it.
  let c = trailShade(in.uv.x, in.uv.y, in.age, in.weight, i32(in.slot));
  if (c.a <= 0.0) { discard; }
  var o: TrailFSOut;
  // STRAIGHT colour into an ADDITIVE target, which reverses the old MAX rule
  // deliberately: max existed so parallel strands could not double into bright
  // dashes on a layer composited after tone mapping. In HDR before bloom,
  // overlapping light SHOULD sum — that is what neon does — and the tone
  // mapper is what keeps the sum from clipping.
  o.color = vec4f(c.rgb, c.a);
  o.aux = vec4f(${src.bloom ? "1.0" : "0.0"}, 1.0, 0.0, c.a);
  return o;
}
`
  )
}
