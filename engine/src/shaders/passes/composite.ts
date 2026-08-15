import { audioApi } from "../audio-api"
import { scoreApi } from "../score-api"
import { simClockApi, simReadApi } from "./sim"
// Composite: HDR scene + bloom pyramid → Filmic tone map → gamma → swapchain.
// Bloom tint/intensity applied at combine (EEVEE treats them as combine-stage params, not prefilter).
//
// The shader is a TEMPLATE: buildCompositeShader() emits either the base pass or
// a variant with user WGSL injected at one or both effect MOUNTS (setEffect).
// The two mounts are the same idea on either side of the scene:
//
//   background(...)  under the scene — a sibling of the 360 equirect (mode 2),
//                    reusing the same per-pixel view-ray reconstruction.
//   foreground(...)  over the finished frame, handed the scene's depth in metres
//                    so it can be occluded by whatever it passes behind.
//
// Both composite in display space, so neither affects lighting, bloom, or
// tonemapping, and both are captured by offline export like any background.

/** What user effect WGSL may define, documented once. A file declares its own
 *  mounts by which of these it defines — defining both is how one file is one
 *  weather system (dark sky behind, rain in front):
 *
 *    fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f
 *    fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f
 *
 *  - `ray`   — normalized world-space view direction of this pixel (left-handed,
 *              +Z forward; identical to what the 360 skybox samples by).
 *  - `uv`    — 0..1 across the canvas, origin bottom-left (shadertoy-style).
 *  - `time`  — seconds since the effect was applied.
 *  - `depth` — FOREGROUND ONLY. Camera-space distance in metres of whatever the
 *              scene drew at this pixel, the far plane where it drew nothing.
 *              Compare a particle's own distance against it and the model
 *              occludes it; fog needs no comparison at all, its alpha simply IS
 *              a function of distance.
 *  - `rzResolution()` — canvas size in pixels, for aspect correction.
 *  - `rzCameraPos()`, `rzWorldPos(ray, depth)` — the lens, and the place a pixel
 *              was drawn.
 *  - `rzSubjectCount()`, `rzSubjectHip(i)` — the cast, at HIP height (see the
 *              function; it is not the floor, and reading it as the floor is a
 *              mistake this API's own comment used to invite).
 *  - `rzProject(p)` — a world point as uv + view-axis distance; the cheap way to
 *              anchor anything, and `z` compares directly against `depth`.
 *  - the bg* spellings of all of the above still resolve, permanently.
 *  - declared params arrive as `params.<name>` (f32 or vec3f), shared by both.
 *
 *  Return display-space sRGB + alpha, 0..1. Both mounts are alpha-composited
 *  LAYERS, so alpha is what decides how much they replace: a background effect
 *  at alpha 1 covers the base (solid color / 360 equirect / transparent) and at
 *  0 lets it through, which is how a starfield is stars over the user's color;
 *  a foreground at alpha 1 covers the frame. No mode flag anywhere — the alpha
 *  channel already says it. */
/**
 * The bones an effect asked for, in declaration order — the slots rzAnchor reads.
 *
 *     // @anchor 左手首 trail
 *     // @anchor 頭
 *
 * A declaration in the source, like the mounts: what a file names is what gets
 * resolved and uploaded, so naming none costs nothing and nobody pays for a
 * rig's other five hundred bones. Anchored to the start of a line so that
 * writing the word @anchor in ordinary prose does not silently add a slot —
 * which would shift every slot after it.
 *
 * `trail` additionally keeps that bone's recent PATH, for rzTrail. Opt-in
 * because a path is two orders of magnitude more data than a point, and most
 * anchors want a point.
 *
 * Names are passed through verbatim: any bone the rig has works, and one it does
 * not have simply reports invalid.
 */
export function parseEffectAnchors(wgsl: string, max: number): { bone: string; trail: boolean }[] {
  return [...wgsl.matchAll(/^[ \t]*\/\/[ \t]*@anchor[ \t]+(\S+)([ \t]+trail)?[ \t]*$/gm)]
    .map((m) => ({ bone: m[1], trail: m[2] !== undefined }))
    .slice(0, max)
}

/**
 * The caps the cast buffer is built to, shared by the shader below and by the
 * engine that fills it. Interpolated into the WGSL rather than written twice:
 * the layout arithmetic on both sides has to agree exactly, and two literals
 * that must match are two literals that eventually will not.
 *
 * All three are MINIMUMS. Raising one breaks nothing, because effects read
 * through accessors and loop to the count functions; lowering one does.
 */
export const EFFECT_SUBJECTS = 4
export const EFFECT_ANCHORS = 8
export const EFFECT_TRAIL_SAMPLES = 128
/** vec4 slot where the trails begin — after the subjects and the anchors. */
export const EFFECT_TRAIL_BASE = EFFECT_SUBJECTS * 3 + EFFECT_ANCHORS * EFFECT_SUBJECTS * 3

export type CompositeEffectSource = {
  /** The user's WGSL verbatim: helpers plus whichever entry points it defines. */
  wgsl: string
  /** Codegen'd `struct EffectParams {...}` + binding decl; empty when no params. */
  paramsDecl: string
  /** Defines `fn background(...)` — mount under the scene. */
  hasBackground: boolean
  /** Defines `fn foreground(...)` — mount over the finished frame. */
  hasForeground: boolean
  /** Grid resolution when the effect declared `// @sim`, else 0. */
  simSize: number
}

const COMPOSITE_HEAD = /* wgsl */ `
// Pipeline-override constant: the engine creates two composite pipelines, one
// with APPLY_GAMMA=false (gamma=1 fast path) and one with APPLY_GAMMA=true.
// The 'if (APPLY_GAMMA)' below is resolved at pipeline-compile time — the
// dead branch is dropped by the shader compiler (no runtime branch, no pow
// invocation on Safari's Metal backend in the common case).
override APPLY_GAMMA: bool = true;

@group(0) @binding(0) var hdrTex: texture_2d<f32>;
@group(0) @binding(1) var bloomTex: texture_2d<f32>;   // bloomUpTexture mip 0 (full pyramid top)
@group(0) @binding(2) var bloomSamp: sampler;
@group(0) @binding(3) var<uniform> viewU: array<vec4<f32>, 15>;
// Aux mask/alpha texture. .r = bloom mask (unused here; bloom blit uses it).
// .g = accumulated canvas alpha (what hdr.a carried before the HDR format
// became rg11b10ufloat). We unpremultiply HDR by this alpha for tonemap, then
// re-premultiply the tonemapped color for output so the premultiplied canvas
// alphaMode composites the WebGPU surface over the page background correctly.
@group(0) @binding(4) var maskTex: texture_2d<f32>;
// Filmic tone curve baked to a WIDTH×1 r16float LUT (bakeFilmicLut on the CPU side).
// Domain: log2(linear) mapped to [0,13]. Replaces the old array<f32,14> that was indexed
// by a runtime u32 (a Metal-backend smell — dynamic local-array indexing lowers to a
// per-invocation copy/switch) and interpolated piecewise-linearly (C0, so segment slope
// discontinuities showed as Mach bands in smooth skin/shadow gradients). The LUT is a
// monotone-cubic (Fritsch–Carlson) fit through the same 14 anchors — same values, C1
// continuity kills the banding — sampled with hardware linear filtering.
@group(0) @binding(5) var filmicLut: texture_2d<f32>;
// viewU[0] = (exposure, invGamma, _, _);  viewU[1] = (tint.rgb, intensity)
// viewU[2] = (background.rgb, mode) — display-space sRGB, composited UNDER the
//            scene post-tonemap. BASE-layer mode: 0 transparent (DOM shows),
//            1 solid color, 2 = 360 equirect skybox sampled by view ray. A user
//            WGSL effect is a separate LAYER over the base — no mode of its own,
//            and no on/off uniform either: the pipeline is REBUILT per effect, so
//            the compiled variant IS the flag.
// viewU[3] = (camera right, tanHalfFov·aspect); viewU[4] = (camera up, tanHalfFov);
// viewU[5] = (camera forward, _) — refreshed per frame while skybox/effect active.
// viewU[6] = (time seconds, view transform id, canvas width, canvas height).
// viewU[7] = (grade offset.rgb, contrast);  viewU[8] = (grade power.rgb, saturation);
// viewU[9] = (grade slope.rgb, grade on/off) — see grade() below.
// viewU[10] = (camera world position, _) — refreshed with the basis above.
// invGamma = 1/gamma precomputed on CPU — avoids a per-pixel divide.
@group(0) @binding(6) var bgEquirect: texture_2d<f32>;
// The scene pass's own MSAA depth buffer, bound depth-only. NOT an extra
// render target: with neither depth of field nor a foreground effect active the
// scene pass discards depth (TBDR tile memory never spills) and this binding is
// never read. Either feature makes the pass store it instead, and both read
// sample 0 — the DoF gather, and linearDepth() for the depth handed to
// foreground().
@group(0) @binding(8) var depthTex: texture_depth_multisampled_2d;
// dofU[0] = (enabled, focusDistance, focusRange, aperture)
// dofU[1] = (maxBlurRadiusPx, bladeCount, sampleCount, anamorphicRatio)
// dofU[2] = (projA, projB, _, _) — z-buffer → camera-space depth inversion,
//           viewZ = projB / (z - projA), rebuilt per frame because near/far
//           track the camera radius. Cleared depth (1.0) inverts to the far
//           plane, so empty sky reads as maximally defocused background.
@group(0) @binding(9) var<uniform> dofU: array<vec4<f32>, 3>;
// Blender's AgX, as the 57³ lookup it ships as rather than a reconstruction of
// it. Sampled in the log-encoded E-Gamut space the cube expects — see agxTransform.
@group(0) @binding(10) var agxLut: texture_3d<f32>;
// The cast, as data. Read through rzSubject/rzAnchor below — the LAYOUT IS NOT
// STABLE and never will be, because it depends on what each effect declared.
// Reading it directly is the one thing that would freeze it forever.
//
// vec4 slots: [0 .. 11] four subjects, three each (root+valid, hip, bounds);
// then MAX_ANCHORS × four subjects, three each (pos+valid, vel, fwd).
@group(0) @binding(11) var<storage, read> _rzCast: array<vec4f>;
// The FIELD LAYER: the user's background/foreground mounts, rendered at half
// resolution in their own pass (see buildFieldShader) and sampled here. Field
// effects are glows, fog, shafts, gradients — low-frequency by nature — and
// running them per quarter-pixel was the single largest per-frame cost an
// effect could add. Bilinear upsampling is invisible at that frequency.
@group(0) @binding(15) var fieldBgTex: texture_2d<f32>;
@group(0) @binding(16) var fieldFgTex: texture_2d<f32>;
// Geometry ribbons, max-blended in their own layer (trails.ts). Composited in
// display space below, where the fullscreen ribbon effects always ran.
@group(0) @binding(12) var trailTex: texture_2d<f32>;

// Must match FILMIC_LUT_WIDTH in engine.ts (bakeFilmicLut).
const FILMIC_LUT_W: f32 = 256.0;

fn linearDepth(coord: vec2<i32>) -> f32 {
  let z = textureLoad(depthTex, coord, 0);
  // projA and projB are m[10] and m[14] of whatever projection the camera built,
  // so this one line inverts both conventions. Non-reversed puts projA just above
  // 1; reversed puts it slightly below 0. Either way (z - projA) keeps its sign
  // across the whole of z in [0,1], so the divisor never crosses zero.
  return clamp(dofU[2].y / (z - dofU[2].x), 0.05, 100000.0);
}

/** Signed circle of confusion in device pixels — negative in front of the
 *  focus band, positive behind it, zero inside it. */
fn circleOfConfusion(depth: f32) -> f32 {
  let focus = max(dofU[0].y, 0.05);
  let halfRange = max(dofU[0].z * 0.5, 0.01);
  let delta = depth - focus;
  let outside = max(abs(delta) - halfRange, 0.0);
  let radius = min(outside / max(depth, 0.05) * dofU[0].w * dofU[1].x, dofU[1].x);
  return select(-radius, radius, delta >= 0.0);
}

/** Premultiplied scene color (HDR + bloom) at an arbitrary pixel, for the
 *  bokeh gather. Explicit-LOD sampling only — legal in non-uniform flow. */
fn sceneSample(coord: vec2<i32>, fullSzI: vec2<i32>, fullSz: vec2f) -> vec4f {
  let p = clamp(coord, vec2<i32>(0), fullSzI - vec2<i32>(1));
  let sAlpha = textureLoad(maskTex, p, 0).g;
  let sHdr = textureLoad(hdrTex, p, 0).rgb / max(sAlpha, 1e-6);
  let sUv = (vec2f(p) + vec2f(0.5)) / fullSz;
  let sBloom = textureSampleLevel(bloomTex, bloomSamp, sUv, 0.0).rgb * viewU[1].xyz * viewU[1].w;
  return vec4f((sHdr + sBloom) * sAlpha, sAlpha);
}

fn filmic(x: f32) -> f32 {
  // Reference checkpoints (Blender 3.6 Filmic MHC, sobotka/filmic-blender
  // look_medium-high-contrast.spi1d): linear 0.18 → ~0.395, linear 1.0 → ~0.83.
  // NOTE: version-pinned to Blender 3.6 — 4.x defaults to AgX, not Filmic.
  let t = clamp(log2(max(x, 1e-10)) + 10.0, 0.0, 13.0);
  // Map t∈[0,13] to the texel-center of baked sample j = t·(W-1)/13.
  let u = (t * (FILMIC_LUT_W - 1.0) / 13.0 + 0.5) / FILMIC_LUT_W;
  // textureSampleLevel (explicit LOD, no derivatives) is legal in non-uniform flow.
  return textureSampleLevel(filmicLut, bloomSamp, vec2f(u, 0.5), 0.0).r;
}

/** The sRGB display encoding — Blender's "Standard" view transform, which is what
 *  NPR work uses: no film curve at all, so the colours a graph computes are the
 *  colours that land. Not a 2.2 power law; sRGB has a linear toe. */
fn srgb_encode(x: f32) -> f32 {
  let c = max(x, 0.0);
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

/**
 * AgX, the chain config.ocio states for "AgX Base Rec.1886" on an sRGB display:
 *
 *   scene-linear Rec.709 → 3x3 into FilmLight E-Gamut
 *   → log2 across 25 stops, [-12.47393, +12.5261] → 0..1
 *   → the 57³ cube
 *   → Rec.1886 (pure γ2.4) decoded, then re-encoded as sRGB for the swapchain
 *
 * The last step is not pedantry: Rec.1886 is a pure power law and sRGB has a
 * linear toe, so skipping it lifts the darkest few code values.
 *
 * Blender interpolates the cube tetrahedrally and a GPU sampler is trilinear.
 * The difference is small and confined to saturated gradients; if it ever shows,
 * tetrahedral is a dozen lines here.
 */
fn agxTransform(c: vec3f) -> vec3f {
  let m = mat3x3<f32>(
    vec3f(0.55937113, 0.07622071, 0.06552670),
    vec3f(0.30478326, 0.78797183, 0.16454675),
    vec3f(0.13584556, 0.13580748, 0.76992653),
  );
  let e = m * max(c, vec3f(0.0));
  // The cube's domain: an unlit pixel would take log2 to -inf, so floor it at
  // the bottom stop rather than sampling outside the LUT.
  let t = clamp((log2(max(e, vec3f(1e-10))) + vec3f(12.47393)) / 25.0, vec3f(0.0), vec3f(1.0));
  // Half-texel inset, so the end stops are the cube's own values and not a blend
  // with the clamp-to-edge border.
  let n = 57.0;
  let uvw = t * ((n - 1.0) / n) + (0.5 / n);
  let formed = textureSampleLevel(agxLut, bloomSamp, uvw, 0.0).rgb;
  let linear = pow(max(formed, vec3f(0.0)), vec3f(2.4));
  return vec3f(srgb_encode(linear.r), srgb_encode(linear.g), srgb_encode(linear.b));
}

/** Which display transform, chosen per frame at viewU[6].y (0 filmic, 1 standard, 2 agx).
 *  A uniform branch rather than a pipeline variant: switching is rare, and both
 *  arms are cheap enough that specialising the pipeline would buy nothing. */
fn viewTransform(c: vec3f) -> vec3f {
  let mode = viewU[6].y;
  if (mode > 1.5) { return agxTransform(c); }
  if (mode > 0.5) { return vec3f(srgb_encode(c.r), srgb_encode(c.g), srgb_encode(c.b)); }
  return vec3f(filmic(c.r), filmic(c.g), filmic(c.b));
}
`

/**
 * The scene half of the effect API — camera, projection, cast, trails.
 *
 * Split out of COMPOSITE_HEAD so the SIM pass can have it too. One effect file
 * is spliced into every module it has a mount in, so a file with both a grid and
 * a foreground compiles its foreground inside the sim shader, where every
 * rzCameraPos() in it has to resolve. Sharing the block is better than stubbing
 * it twice, and it means a kernel can legitimately ask where a bone is — which
 * is exactly what a wake wants.
 *
 * Depends on `viewU` and `_rzCast` being declared by the including module, and
 * on nothing else: no textures, no samplers.
 */
export const EFFECT_SCENE_API = /* wgsl */ `
// ── The effect API ────────────────────────────────────────────────────────────
//
// Named rz*, for the engine. The prefix earns its place twice: user code is
// concatenated into THIS module, so an unprefixed rzAnchor() would collide with
// exactly the helper an author would write, and the old bg* prefix stopped being
// true in 0.41.0 when effects gained a mount over the finished frame.
//
// The bg* names below are permanent aliases, not a deprecation with an end date.
// A published link is immutable, so a scene pinned to a bg* effect has to keep
// compiling forever. They are one-line and inlined; no new function gets one.

/** Canvas size in pixels — for aspect correction. */
fn rzResolution() -> vec2f { return viewU[6].zw; }

/** The camera's world position. */
fn rzCameraPos() -> vec3f { return viewU[10].xyz; }

/** How many characters are in the scene, up to four. */
fn rzSubjectCount() -> i32 { return i32(viewU[10].w); }

/**
 * A world point as the camera sees it: xy the uv it lands on, z its distance
 * along the VIEW AXIS in metres.
 *
 * The exact inverse of the ray this pass builds per pixel, so it is the cheap way
 * to work with anything anchored in the world. Marching a curve or a trail in 3D
 * costs a distance evaluation per sample per pixel; projecting its points once
 * and measuring in 2D costs a subtraction, which is the difference between a
 * ribbon that runs at 4K and one that does not.
 *
 * z is directly comparable to the depth handed to foreground(), so occlusion is
 * a single test: draw where your z is nearer than the scene's. It is returned
 * SIGNED and unclamped — behind the camera is negative, and worth rejecting
 * before you use the uv, which is meaningless there.
 */
fn rzProject(p: vec3f) -> vec3f {
  let d = p - viewU[10].xyz;
  let z = dot(d, viewU[5].xyz);
  // Guard only the divide. z itself is returned as it is, so the caller can see
  // the sign; clamping it here would put points behind the lens on the horizon.
  let inv = 1.0 / select(z, 1e-4, z < 1e-4);
  let ndc = vec2f(dot(d, viewU[3].xyz) * inv / viewU[3].w, dot(d, viewU[4].xyz) * inv / viewU[4].w);
  return vec3f(ndc * 0.5 + 0.5, z);
}

fn rzCamPos() -> vec3f { return rzCameraPos(); }
fn rzCameraRight() -> vec3f { return viewU[3].xyz; }
fn rzCameraUp() -> vec3f { return viewU[4].xyz; }
fn rzCameraForward() -> vec3f { return viewU[5].xyz; }

/** A character, as much of one as a shader needs. */
struct RzSubject {
  /** On the FLOOR, under the body — where a ring or a magic circle belongs. */
  root: vec3f,
  /** At the hips, the middle of the body — where an aura belongs. */
  center: vec3f,
  /** Bounding sphere: xyz centre, w radius. Deliberately generous — cull with it. */
  bounds: vec4f,
  /** False past the end of the cast, and every field is then zero. */
  valid: bool,
}

/** One bone an effect asked for, by name, at the top of its own source. */
struct RzAnchor {
  pos: vec3f,
  /** World units per second, from the previous frame. Direction for a trail,
   *  magnitude for anything that should react to how hard someone is moving. */
  vel: vec3f,
  /** The bone's forward axis — which way a foot points, where a head looks. */
  fwd: vec3f,
  /** False when this rig has no such bone. Check it: the alternative is drawing
   *  a hand effect at the world origin on every model that spells it differently. */
  valid: bool,
}

const RZ_MAX_ANCHORS: i32 = ${EFFECT_ANCHORS};

/**
 * Character i. Loop to rzSubjectCount(), never to a constant — the caps here are
 * MINIMUMS and are free to grow, which is only true while nobody hardcodes them.
 */
fn rzSubject(i: i32) -> RzSubject {
  var s: RzSubject;
  s.valid = i >= 0 && i < rzSubjectCount();
  if (!s.valid) { return s; }
  let b = i * 3;
  s.root = _rzCast[b].xyz;
  s.center = _rzCast[b + 1].xyz;
  s.bounds = _rzCast[b + 2];
  return s;
}

/** Local slot → scene slot. IDENTITY here, hardcoded rather than spliced: with
 *  one effect installed the scene table is exactly this effect's declaration
 *  order, so the two spaces coincide. setEffects replaces this line with the
 *  effect's real alias (anchorAliasWgsl) when the table is shared — the
 *  accessors below already route through it, so only this definition changes. */
fn _rzSlot(local: i32) -> i32 { return local; }

/**
 * The slot-th bone this effect declared, on character subject.
 *
 * Slots are the order of the declarations at the top of your source:
 *
 *     // @anchor 左手首
 *     // @anchor 頭
 *
 * gives you slot 0 and slot 1. Any bone name the model has works; valid is
 * false when it does not have it, which is the normal case across rigs that
 * spell things differently.
 */
fn rzAnchor(subject: i32, slot: i32) -> RzAnchor {
  var a: RzAnchor;
  a.valid = false;
  // The author's slot is a NAME; _rzSlot turns it into the scene's address. With
  // one effect installed the two coincide and this folds away.
  let g = _rzSlot(slot);
  if (subject < 0 || subject >= rzSubjectCount() || g < 0 || g >= RZ_MAX_ANCHORS) { return a; }
  let b = ${EFFECT_SUBJECTS * 3} + (g * ${EFFECT_SUBJECTS} + subject) * 3;
  a.valid = _rzCast[b].w > 0.5;
  a.pos = _rzCast[b].xyz;
  a.vel = _rzCast[b + 1].xyz;
  a.fwd = _rzCast[b + 2].xyz;
  return a;
}

const RZ_TRAIL_SAMPLES: i32 = ${EFFECT_TRAIL_SAMPLES};

/**
 * How many path samples this anchor has. Zero unless it was declared with
 * trail, and it climbs from zero as the trail fills after the effect loads.
 *
 * Loop to THIS, never to RZ_TRAIL_SAMPLES: the cap is a minimum and is free to
 * grow, which stays true only while nobody hardcodes it.
 */
fn rzTrailCount(subject: i32, slot: i32) -> i32 {
  let g = _rzSlot(slot);
  if (subject < 0 || subject >= rzSubjectCount() || g < 0 || g >= RZ_MAX_ANCHORS) { return 0; }
  return i32(_rzCast[${EFFECT_SUBJECTS * 3} + (g * ${EFFECT_SUBJECTS} + subject) * 3 + 2].w);
}

/**
 * Sample i of an anchor's path: xyz where it was, w how many seconds ago.
 *
 * i = 0 is NOW and they run backwards in time, so a ribbon is drawn by walking i
 * upward and fading on .w. Sampled at a fixed rate on the SCENE clock, not the
 * display's — so the path is identical in the editor, in an export, and in a
 * re-export, and its spacing does not change with framerate.
 *
 * This is what a hand trail wants instead of position and velocity. One position
 * and one velocity is a straight segment that jitters, because a velocity is a
 * difference between two frames; a path is what actually happened.
 */
fn rzTrail(subject: i32, slot: i32, i: i32) -> vec4f {
  let n = rzTrailCount(subject, slot);
  if (i < 0 || i >= n) { return vec4f(0.0); }
  let base = ${EFFECT_TRAIL_BASE} + (_rzSlot(slot) * ${EFFECT_SUBJECTS} + subject) * RZ_TRAIL_SAMPLES;
  return _rzCast[base + i];
}

fn bgResolution() -> vec2f { return rzResolution(); }
fn bgCameraPos() -> vec3f { return rzCameraPos(); }
fn bgSubjectCount() -> i32 { return rzSubjectCount(); }

/**
 * Where a character IS, in world space — at the hips, not on the floor.
 *
 * An effect that wants to RESPOND to the cast — a glow that follows someone,
 * dust kicked up where they are — needs to know where they are, and the ray and
 * the depth cannot tell it: they describe the pixel, not the scene.
 *
 * The value is model.position + センター + 全ての親. センター sits at hip
 * height on every standard MMD rig, so this is a point in the middle of the
 * body. It is NOT the contact point: a ripple drawn here appears at the waist.
 * Ground effects want the .xz of this and their own floor height, which is what
 * the effects that shipped against it already do.
 *
 * The comment here used to claim it was "between the feet on the floor", which
 * is where that habit came from. Left as it is regardless of the name: a
 * published link is immutable, so every shared scene pinning an effect that
 * reads this depends on it meaning exactly what it has always meant.
 *
 * Clamped rather than bounds-checked: an effect looping past the count reads the
 * last subject instead of sampling whatever follows the array, which is a wrong
 * ripple rather than an undefined one.
 */
fn rzSubjectHip(i: i32) -> vec3f { return viewU[11 + clamp(i, 0, 3)].xyz; }

fn bgSubjectPos(i: i32) -> vec3f { return rzSubjectHip(i); }

/** Where in the WORLD the scene drew this pixel — the depth handed to
 *  foreground() turned into a place. Without it an effect can only think in
 *  distances from the lens, which is no use to anything that belongs somewhere:
 *  fog lying on the ground has to know where the ground is.
 *
 *  depth measures along the VIEW AXIS, not along the ray, so it is divided by
 *  the ray's projection onto camera-forward before being walked out. At the far
 *  plane (nothing drawn) this lands a very long way off, which is what a sky
 *  should do to anything reading it. */
fn rzWorldPos(ray: vec3f, depth: f32) -> vec3f {
  let axis = max(dot(normalize(ray), viewU[5].xyz), 1e-4);
  return rzCameraPos() + normalize(ray) * (depth / axis);
}

fn bgWorldPos(ray: vec3f, depth: f32) -> vec3f { return rzWorldPos(ray, depth); }

/** Color grading, applied to the tonemapped SCENE (not the background — see the
 *  call site). The core is ASC CDL, the film-industry interchange standard:
 *
 *      out = (in · slope + offset) ^ power        then saturation  (SOP → SAT)
 *
 *  Using the real standard rather than invented controls means a look authored
 *  here maps onto any grading tool. slope/offset/power are derived on the CPU
 *  from the UI's shadow/midtone/highlight colors (see setColorGrading), so the
 *  per-pixel cost is one mul-add, one pow, one lerp. */
fn grade(c: vec3f) -> vec3f {
  var x = pow(max(c * viewU[9].xyz + viewU[7].xyz, vec3f(0.0)), viewU[8].xyz);
  // Contrast pivots on 0.5 — display-referred midpoint, since we grade post-Filmic.
  x = (x - vec3f(0.5)) * viewU[7].w + vec3f(0.5);
  // Rec.709 luma, matching the ASC SAT node.
  let luma = dot(x, vec3f(0.2126, 0.7152, 0.0722));
  return max(mix(vec3f(luma), x, viewU[8].w), vec3f(0.0));
}
`

const COMPOSITE_BODY = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let coord = vec2<i32>(fragCoord.xy);
  let hdr = textureLoad(hdrTex, coord, 0);
  let alpha = textureLoad(maskTex, coord, 0).g;
  let a = max(alpha, 1e-6);
  let straight = hdr.rgb / a;
  let fullSz = vec2f(textureDimensions(hdrTex));
  // Bloom is at half-res (pyramid mip 0). Sampler interpolates back to full-res UVs.
  // fragCoord.xy is already at pixel center (e.g. 0.5, 0.5 for first pixel).
  let bloomUv = fragCoord.xy / max(fullSz, vec2f(1.0));
  let tint = viewU[1].xyz;
  let intensity = viewU[1].w;
  let bloom = textureSampleLevel(bloomTex, bloomSamp, bloomUv, 0.0).rgb * tint * intensity;
  let combined = straight + bloom;

  // ── Depth of field ──
  // Single-pass golden-angle gather over a polygonal (bladed) disk, in
  // premultiplied HDR before tonemap. Near-field taps that see focused
  // background are heavily down-weighted so a sharp subject doesn't bleed into
  // a blurred foreground; the reverse leak (background bokeh over the subject
  // edge) is damped less — that soft halo is what real lenses do. Scene layer
  // only: the composited background (solid / 360 / effect) stays sharp, which
  // is invisible while it sits at infinity behind a far-blurred stage.
  var scenePm = vec4f(combined * alpha, alpha);
  if (dofU[0].x > 0.5) {
    let centerDepth = linearDepth(coord);
    let centerCoc = circleOfConfusion(centerDepth);
    let radius = abs(centerCoc);
    if (radius > 0.35) {
      let fullSzI = vec2<i32>(fullSz);
      let sampleCount = clamp(dofU[1].z, 6.0, 24.0);
      let blades = clamp(dofU[1].y, 3.0, 12.0);
      let sector = 6.28318530718 / blades;
      var accum = scenePm;
      var weightSum = 1.0;
      for (var i = 0u; i < 24u; i++) {
        if (f32(i) >= sampleCount) { break; }
        let fi = f32(i) + 0.5;
        let ring = sqrt(fi / sampleCount);
        let angle = fi * 2.39996323;
        let localAngle = (fract((angle + 3.14159265359) / sector) - 0.5) * sector;
        let polygonRadius = cos(3.14159265359 / blades) / max(cos(localAngle), 0.01);
        var disk = vec2f(cos(angle), sin(angle)) * ring * polygonRadius;
        disk.x *= max(dofU[1].w, 0.25);
        let sp = coord + vec2<i32>(round(disk * radius));
        let cp = clamp(sp, vec2<i32>(0), fullSzI - vec2<i32>(1));
        let sampleDepth = linearDepth(cp);
        let sampleCoc = circleOfConfusion(sampleDepth);
        var w = 1.0;
        if (centerCoc < 0.0 && sampleDepth > centerDepth + dofU[0].z) {
          w *= 0.08;
        } else if (centerCoc > 0.0 && sampleDepth > centerDepth + dofU[0].z * 2.0) {
          w *= 0.35;
        }
        // A tap only contributes where its own blur circle reaches this pixel.
        let sampleRadius = abs(sampleCoc);
        w *= mix(0.2, 1.0, smoothstep(ring * radius - 1.0, ring * radius + 1.0, sampleRadius));
        accum += sceneSample(cp, fullSzI, fullSz) * w;
        weightSum += w;
      }
      scenePm = mix(scenePm, accum / max(weightSum, 1e-5), smoothstep(0.35, 1.75, radius));
    }
  }
  let sceneAlpha = scenePm.a;
  let sceneStraight = scenePm.rgb / max(sceneAlpha, 1e-6);

  let exposed = sceneStraight * exp2(viewU[0].x);
  var disp = max(viewTransform(exposed), vec3f(0.0));
  // Grade the SCENE only, before the display gamma. Deliberately not applied to
  // the background: it keeps a picked background color exactly as picked, and —
  // load-bearing — leaves green-screen mode's key color unshifted so chroma
  // keying still works. Skipped entirely when the grade is neutral.
  if (viewU[9].w > 0.5) {
    disp = grade(disp);
  }
  if (APPLY_GAMMA) {
    disp = pow(disp, vec3f(viewU[0].y));
  }
  // Composite over the background in display space (premultiplied out). The
  // background is TWO layers: a base (transparent / solid color / 360 equirect)
  // and an optional user WGSL effect over-composited onto it.
  let bg = viewU[2];
  var bgA = select(0.0, 1.0, bg.w > 0.5);
  var bgPm = bg.rgb * bgA;  // premultiplied accumulator
  // This pixel's world-space view ray, rebuilt from the camera basis — what the
  // equirect samples by, and what both effect mounts navigate by. The dome sits
  // at infinity (no parallax): PhotoDome-style, display-only. Hoisted out of the
  // branch below because the foreground mount is past the end of it; it is pure
  // arithmetic on uniforms, which every backend sinks into whatever reads it.
  let ndc = vec2f(fragCoord.x / fullSz.x * 2.0 - 1.0, 1.0 - fragCoord.y / fullSz.y * 2.0);
  let dir = normalize(viewU[5].xyz + ndc.x * viewU[3].w * viewU[3].xyz + ndc.y * viewU[4].w * viewU[4].xyz);
  if (BACKGROUND_COND) {
    if (bg.w > 1.5) {
      // LH world (+Z forward): longitude = atan2(x, z), Babylon-PhotoDome convention.
      let su = 0.5 + atan2(dir.x, dir.z) * 0.15915494309;  // 1/(2π)
      let sv = 0.5 - asin(clamp(dir.y, -1.0, 1.0)) * 0.31830988618;  // 1/π
      bgPm = textureSampleLevel(bgEquirect, bloomSamp, vec2f(su, sv), 0.0).rgb;
    }
    BACKGROUND_CALL
  }
  // The frame, premultiplied: scene over background. A var, not the return
  // expression, because the foreground mount composites onto it.
  var outRgb = disp * sceneAlpha + bgPm * (1.0 - sceneAlpha);
  var outA = sceneAlpha + bgA * (1.0 - sceneAlpha);
  // The trail layer, straight-alpha over the tone-mapped frame — the author's
  // colours verbatim, never through AgX.
  let trailFx = textureLoad(trailTex, vec2i(fragCoord.xy), 0);
  outRgb = trailFx.rgb * trailFx.a + outRgb * (1.0 - trailFx.a);
  outA = trailFx.a + outA * (1.0 - trailFx.a);
  FOREGROUND_CALL
  return vec4f(outRgb, outA);
}
`

// uv flipped to bottom-left origin (shadertoy convention); clamped so a stray
// effect can't push negatives/NaN into the premultiplied composite. Standard
// OVER onto the base layer. No `if` around it: the pipeline is rebuilt per
// effect, so this text only exists in variants whose WGSL defines background().
const BACKGROUND_CALL = /* wgsl */ `
    // The field layer is PREMULTIPLIED: N effects blend into it in document
    // order, and premultiplied is the only form in which repeated OVER composes
    // associatively — straight alpha would need the divide back out on every
    // draw. So rgb is already scaled by its own alpha and must not be again.
    // With one effect drawing over a cleared target this is identical to the
    // straight form it replaced.
    let bgFx = clamp(textureSampleLevel(fieldBgTex, bloomSamp, fragCoord.xy / fullSz, 0.0), vec4f(0.0), vec4f(1.0));
    bgPm = bgFx.rgb + bgPm * (1.0 - bgFx.a);
    bgA = bgFx.a + bgA * (1.0 - bgFx.a);
`

// Same OVER, one layer later — onto the finished frame rather than onto the
// base. Ungated by design: a foreground runs at every pixel, including the ones
// the model covers, because covering them is the point.
const FOREGROUND_CALL = /* wgsl */ `
  // Premultiplied, as the background layer above — same reason.
  let fgFx = clamp(textureSampleLevel(fieldFgTex, bloomSamp, fragCoord.xy / fullSz, 0.0), vec4f(0.0), vec4f(1.0));
  outRgb = fgFx.rgb + outRgb * (1.0 - fgFx.a);
  outA = fgFx.a + outA * (1.0 - fgFx.a);
`

// Derivative builtins are illegal in non-uniform control flow (WGSL uniformity
// analysis rejects the pipeline), so the coverage gate below can only wrap
// effect code that doesn't use them. Checked textually at build time.
const USES_DERIVATIVES = /\b(?:fwidth|dpdx|dpdy)(?:Fine|Coarse)?\s*\(/

/** The condition on the background block (equirect sample + background effect).
 *
 *  Two jobs. It skips the block behind pixels the model fully covers — the
 *  composite multiplies the result by (1 - alpha) = 0 there anyway, and on a
 *  full-screen effect that's a third or more of the frame (the cost Safari feels
 *  most). And with no background effect compiled in, it also skips the block
 *  entirely unless the equirect needs it.
 *
 *  The equirect uses explicit-LOD sampling, which is always legal in non-uniform
 *  flow; only derivative-using effects must keep uniform control flow and forgo
 *  the coverage half. The test is textual over the whole file, so a foreground
 *  that uses fwidth costs the background its gate — conservative, and only ever
 *  in the direction of correctness. (The foreground mount itself sits in uniform
 *  flow, so derivatives are always legal there.) */
function backgroundCondition(effect?: CompositeEffectSource | null): string {
  // sceneAlpha, not alpha: the bokeh gather spreads coverage, so a pixel the
  // sharp scene fully covered can end up needing background behind its blur.
  // (The old derivative carve-out is gone with the inline user code: the field
  // pass runs the whole quad, which is uniform control flow by construction.)
  const coverage = "sceneAlpha < 0.999"
  if (!effect?.hasBackground) return `bg.w > 1.5 && ${coverage}`
  return coverage
}

export function buildCompositeShader(effect?: CompositeEffectSource | null): string {
  const body = COMPOSITE_BODY.replace("BACKGROUND_COND", backgroundCondition(effect))
    .replace("BACKGROUND_CALL", effect?.hasBackground ? BACKGROUND_CALL.trim() : "")
    .replace("FOREGROUND_CALL", effect?.hasForeground ? FOREGROUND_CALL.trim() : "")
  // The composite is STATIC either way now: the user's code compiles in the
  // field module alone, and the composite only decides whether to sample it.
  return COMPOSITE_HEAD +
    EFFECT_SCENE_API + audioApi(0, 13) + scoreApi(0, 19) + body
}

/**
 * The field pass: the user's background/foreground mounts at half resolution,
 * into two rgba16f targets the composite bilinearly upsamples.
 *
 * The fragment reconstructs the FULL-resolution pixel it stands in for and runs
 * the original derivation verbatim — same ndc, same ray, same uv, same depth
 * read — so an effect cannot tell it moved; it is simply asked half as often
 * in each direction.
 */
export function buildFieldShader(effect: CompositeEffectSource): string {
  const bgLine = effect.hasBackground
    ? "out.bg = clamp(background(dir, uv, viewU[6].x), vec4f(0.0), vec4f(1.0));"
    : ""
  const fgLine = effect.hasForeground
    ? "out.fg = clamp(foreground(dir, uv, viewU[6].x, linearDepth(vec2<i32>(min(fx, fullSz - 1.0)))), vec4f(0.0), vec4f(1.0));"
    : ""
  return (
    COMPOSITE_HEAD +
    EFFECT_SCENE_API +
    audioApi(0, 13) +
    scoreApi(0, 19) +
    // The persistent grid, always bound — a 1×1 of zeroes when the effect has
    // none, so rzSim() is a function that always exists rather than one an
    // author has to know whether they are allowed to call.
    simReadApi(0, 17, 18, effect.simSize) +
    simClockApi("viewU[6].x") +
    "\n// ── user effect (setEffect) ──\n" +
    effect.paramsDecl +
    "\n" +
    effect.wgsl +
    "\n" +
    /* wgsl */ `
@group(0) @binding(14) var<uniform> fieldU: vec4f;

@vertex fn fieldVs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

struct FieldOut {
  @location(0) bg: vec4f,
  @location(1) fg: vec4f,
}

@fragment fn fieldFs(@builtin(position) fragCoord: vec4f) -> FieldOut {
  let fullSz = fieldU.zw;
  let fx = fragCoord.xy * (fullSz / max(fieldU.xy, vec2f(1.0)));
  let ndc = vec2f(fx.x / fullSz.x * 2.0 - 1.0, 1.0 - fx.y / fullSz.y * 2.0);
  let dir = normalize(viewU[5].xyz + ndc.x * viewU[3].w * viewU[3].xyz + ndc.y * viewU[4].w * viewU[4].xyz);
  let uv = vec2f(fx.x / fullSz.x, 1.0 - fx.y / fullSz.y);
  var out: FieldOut;
  out.bg = vec4f(0.0);
  out.fg = vec4f(0.0);
  ${bgLine}
  ${fgLine}
  return out;
}
`
  )
}

/** Kept for compatibility with existing imports (the base, no-effect shader). */
export const COMPOSITE_SHADER_WGSL = buildCompositeShader(null)
