import { EFFECT_SUBJECTS } from "../cast-layout"

// How far is this pixel from the cast? — the primitive behind every silhouette
// look, computed once and read by anyone.
//
// AN EFFECT CANNOT ANSWER THIS ITSELF, and that is the whole reason this pass
// exists. Distance to the nearest drawn pixel is not a function of the pixel;
// it is a function of every pixel around it. A shader with one pass can only go
// looking, and searching a disc costs O(radius^2) per pixel — measured on the
// first sticker-outline effect, a 9 pixel border cost 64 id samples on every
// background pixel and a 16 pixel one cost 96 to 128. The look everybody wants
// next is a wide one, and that is the direction the cost runs away in.
//
// A JUMP FLOOD costs log2(resolution) passes and answers for ANY distance. It
// is the same 37 million reads at 1080p whether the effect wants a 4 pixel
// border or a 400 pixel aura, and it is shared: two effects reading it pay for
// it once.
//
// NO CAP ON THE DISTANCE. A partial flood — stopping early to save a pass —
// would silently limit how far an effect could reach, and the engine has no
// business deciding that. The author writes the falloff and pays for the width
// they asked for in their own arithmetic, not in ours.
//
// FULL RESOLUTION, after trying not to be.
//
// Half res was tried twice and rejected on sight both times, and the reason is
// worth recording because the arithmetic looked fine. A half-res distance field
// really does place the edge to within a pixel — measured against an exact
// transform, 0.22 px mean and 0.68 px at the 99th percentile. What that average
// hides is that the error is not noise: it VARIES along the silhouette, so the
// border breathes in and out by a fraction of a pixel as the edge turns, and an
// edge that wobbles reads as jagged no matter how smoothly it is feathered.
// Seeding at the sub-texel centroid rather than the texel centre halved it
// (0.83 to 0.28 px mean on a curve) and it was still visible.
//
// So the field is the size of the frame and the seeds are exact. What that costs
// at 1080p is about 224 million texture reads a frame, against 47 million at
// half — real, and the honest price of an edge that does not crawl. It is still
// the same cost whatever width an effect asks for, which is the whole point: the
// per-pixel search this replaced cost 64 reads a pixel for a 9 px border and
// four times that for an 18 px one.

/** Whether any of this effect's source reads the field, and so whether the
 *  engine should spend the passes building it. Nothing else turns it on. */
export function castDistanceUsed(wgsl: string): boolean {
  return /\brzCastDistance\s*\(/.test(wgsl)
}

/** Seed and ping-pong target: the nearest seed's texel coordinate, or (-1,-1).
 *  32-bit because these are coordinates — a half would quantise them at 2048
 *  and put the seam of that error straight through a 4K frame. */
export const CAST_SEED_FORMAT: GPUTextureFormat = "rg32float"
/** The resolved distance, in FIELD texels. Sampled bilinearly, so it is filtered
 *  rather than loaded, and r32float is not filterable everywhere — r16float is,
 *  and half a texel of precision on a distance is nothing. */
export const CAST_DIST_FORMAT: GPUTextureFormat = "r16float"

/** Per-pixel MSAA coverage of the cast, written beside the seeds and read by the
 *  resolve to place the edge inside the seed pixel. */
export const CAST_COVERAGE_FORMAT: GPUTextureFormat = "r8unorm"

/** The field is the size of the frame. See the note above for what half cost. */
export const CAST_FIELD_DIV = 1

const FULLSCREEN_VS = /* wgsl */ `
struct VSOut { @builtin(position) pos: vec4f };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  // One oversized triangle rather than two: no seam down the diagonal, and the
  // rasteriser does not have to think about a shared edge.
  var p = array<vec2f, 3>(vec2f(-1.0, -3.0), vec2f(-1.0, 1.0), vec2f(3.0, 1.0));
  var o: VSOut;
  o.pos = vec4f(p[i], 0.0, 1.0);
  return o;
}
`

/**
 * Pass 1 — plant a seed on every pixel the CAST drew.
 *
 * The ground, a stage and a media plane all write ids exactly as she does, so
 * seeding "anything drawn" would grow a border around the floor: a rectangle
 * round the frame rather than a sticker. Only the subject ids seed.
 */
export function buildCastSeedShader(samples: number): string {
  return (
    /* wgsl */ `
const RZ_SUBJECTS: i32 = ${EFFECT_SUBJECTS};

const RZ_ID_SAMPLES: i32 = ${samples};

@group(0) @binding(0) var _rzIdTex: texture_multisampled_2d<u32>;
@group(0) @binding(1) var<storage, read> _rzCast: array<vec4f>;

// The two cast accessors this pass needs, spelled out rather than pulled in.
// CAST_API brings subjects, trails, anchors and their aliases with it, and a
// seed pass wants none of that — it asks one question about one id.
fn rzSubjectCount() -> i32 {
  var n = 0;
  for (var i = 0; i < RZ_SUBJECTS; i++) {
    if (_rzCast[i * 3 + 2].w > 0.0) { n = i + 1; }
  }
  return n;
}
fn rzSubjectId(i: i32) -> u32 {
  if (i < 0 || i >= rzSubjectCount()) { return 0u; }
  return u32(_rzCast[i * 3 + 1].w);
}

${FULLSCREEN_VS}

struct SeedOut {
  /** Where the nearest cast pixel is, for the flood to carry. */
  @location(0) seed: vec2f,
  /** How much of this pixel she covers, 0..1, for the resolve to place the edge
   *  inside it. */
  @location(1) coverage: f32,
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> SeedOut {
  // COVERAGE, NOT A YES OR NO — this is the whole reason the border used to sit
  // badly against her.
  //
  // The scene draws at 4x MSAA and the frame you see is resolved, so her edge is
  // smooth. The id attachment is deliberately NOT resolved (an averaged id
  // belongs to nothing), so reading sample 0 gives a hard, binary, aliased
  // silhouette — a different edge from the one she is drawn with. A border grown
  // off that can never sit flush against her: it traces the staircase while she
  // has none.
  //
  // Counting the samples recovers what MSAA already knew. Four samples give five
  // levels of coverage, the resolve turns that into a sub-pixel zero crossing,
  // and the border meets her where she actually ends.
  let dim = vec2<i32>(textureDimensions(_rzIdTex));
  let p = clamp(vec2<i32>(pos.xy), vec2<i32>(0), dim - vec2<i32>(1));
  let n = rzSubjectCount();
  var covered = 0.0;
  for (var sample = 0; sample < RZ_ID_SAMPLES; sample++) {
    let o = textureLoad(_rzIdTex, p, sample).y;
    if (o == 0u) { continue; }
    for (var i = 0; i < n; i++) {
      if (o == rzSubjectId(i)) { covered += 1.0; break; }
    }
  }
  var out: SeedOut;
  out.coverage = covered / f32(RZ_ID_SAMPLES);
  // Anything she touches at all seeds, including a pixel she barely clips: a
  // sliver is where the sub-pixel edge lives, and dropping it would put the
  // staircase straight back.
  // -1 is the empty marker, and every step below tests for it before believing
  // a candidate.
  out.seed = select(vec2f(-1.0, -1.0), pos.xy, covered > 0.0);
  return out;
}
`
  )
}

/**
 * Pass 2 — one jump-flood step, run once per halving of the stride.
 *
 * Each texel asks its eight neighbours at the current stride what seed THEY
 * know about, and keeps whichever is nearest to itself. Starting at half the
 * field's width and halving to one, a seed reaches every texel that is nearer
 * to it than to any other — which is the definition of the field.
 */
export function buildCastStepShader(): string {
  return /* wgsl */ `
@group(0) @binding(0) var _rzPrev: texture_2d<f32>;
@group(0) @binding(1) var<uniform> _rzStep: vec4f;   // (stride, _, _, _)

${FULLSCREEN_VS}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec2f {
  let me = pos.xy;
  let dim = vec2<i32>(textureDimensions(_rzPrev));
  let stride = i32(_rzStep.x);
  var best = textureLoad(_rzPrev, vec2<i32>(me), 0).xy;
  // Squared throughout: the comparison is the only thing that matters and a
  // square root per candidate is nine of them per texel per pass.
  var bestD = select(1.0e30, dot(best - me, best - me), best.x >= 0.0);
  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let p = clamp(vec2<i32>(me) + vec2<i32>(dx, dy) * stride, vec2<i32>(0), dim - vec2<i32>(1));
      let c = textureLoad(_rzPrev, p, 0).xy;
      if (c.x < 0.0) { continue; }
      let d = dot(c - me, c - me);
      if (d < bestD) { bestD = d; best = c; }
    }
  }
  return best;
}
`
}

/**
 * Pass 3 — turn the seed coordinates into a distance.
 *
 * Its own pass because the READ is bilinear. Interpolating coordinates across
 * the boundary between two seeds averages two unrelated points and lands the
 * result on neither; interpolating the DISTANCE is the smooth, honest thing,
 * and it is what makes a half-res field cut a full-res edge.
 */
export function buildCastResolveShader(): string {
  return /* wgsl */ `
@group(0) @binding(0) var _rzSeed: texture_2d<f32>;
@group(0) @binding(1) var _rzCoverage: texture_2d<f32>;

${FULLSCREEN_VS}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) f32 {
  let s = textureLoad(_rzSeed, vec2<i32>(pos.xy), 0).xy;
  // No cast on screen at all: everything is unreachably far from her, which is
  // the answer that makes an effect draw nothing rather than everything.
  if (s.x < 0.0) { return 1.0e30; }
  let c = textureLoad(_rzCoverage, vec2<i32>(s), 0).x;
  // SIGNED, WITH THE ZERO CROSSING INSIDE THE SEED PIXEL.
  //
  // A fully covered seed means her true edge runs about half a pixel outside its
  // centre, so the distance to the EDGE is half a pixel less than the distance to
  // the centre. A half-covered seed has the edge through its centre and wants no
  // correction. A barely covered one has the edge nearly a half-pixel further in.
  // All three are (c - 0.5).
  //
  // It also makes the field NEGATIVE inside her, and that is what lets an effect
  // fade where it meets her rather than stopping dead on a texel boundary — the
  // difference between a border that is attached and one that is merely nearby.
  return length(s - pos.xy) - (c - 0.5);
}
`
}

/**
 * `rzCastDistance(uv)`, for the effects that read it.
 *
 * ALWAYS COMPILED IN, even when the pass is not running — it then samples a 1x1
 * holding a very large number, so an effect keyed on it draws nothing instead of
 * failing to compile. Same rule the grid's accessor follows, and the id
 * accessors before it: an author should never have to guard a name.
 */
export function castDistanceApi(group: number, tex: number, samp: number, scale: number): string {
  return /* wgsl */ `
@group(${group}) @binding(${tex}) var _rzCastDistTex: texture_2d<f32>;
@group(${group}) @binding(${samp}) var _rzCastDistSamp: sampler;

/**
 * Distance from this point to the nearest pixel the CAST drew, in SCREEN pixels.
 *
 * ZERO ON HER, positive outside, and the crossing is sub-pixel — it comes from
 * MSAA coverage, so it lands on the same edge she is drawn with rather than on
 * the staircase a single sample sees. Fade an effect across zero and it attaches
 * to her cleanly.
 *
 * IT IS NOT AN INTERIOR DEPTH. Every pixel she covers is a seed, so the nearest
 * seed to a pixel deep inside her is itself: the answer there is 0, not "far
 * in". What the coverage buys is half a pixel of sub-pixel placement at the
 * boundary, which is exactly enough to anti-alias against her and no more — so
 * the value inside her runs to -0.5 and no further. An effect that wants to
 * reach INTO her wants a second flood seeded on the background, which this pass
 * does not build. Guard on d >= 0 and fade across it.
 *
 * There is no ceiling. A pixel on the far side of the frame gets the real
 * number. The ground, a stage and a media plane are not the cast and do not
 * seed it.
 *
 * Screen pixels whatever the field's own resolution is, so an author writes the
 * width they mean and never has to know how this is built. uv is the effect's
 * own convention, origin bottom-left; the flip to texture rows happens here.
 */
fn rzCastDistance(uv: vec2f) -> f32 {
  let flipped = vec2f(clamp(uv.x, 0.0, 1.0), 1.0 - clamp(uv.y, 0.0, 1.0));
  let d = textureSampleLevel(_rzCastDistTex, _rzCastDistSamp, flipped, 0.0).x;
  return d * ${scale > 0 ? scale : 1}.0;
}
`
}

/**
 * The accessor as a STUB, for every module that is not a field.
 *
 * An effect's whole source is spliced into each module its entry points reach —
 * declare lightEmit and the light module compiles your foreground() along with
 * it. So a name that exists only in the field module is a compile error the
 * moment one effect uses both, which is exactly what a silhouette glow that
 * also lights the cast wants to do.
 *
 * Answers "unreachably far", the same shape of answer idApi gives when the id
 * buffer is off: the arithmetic still runs and produces nothing, rather than the
 * shader failing to build. Nothing outside a field mount has a screen position
 * to ask about anyway.
 */
export function castDistanceStub(): string {
  return /* wgsl */ `
fn rzCastDistance(uv: vec2f) -> f32 { return 1.0e30; }
`
}
