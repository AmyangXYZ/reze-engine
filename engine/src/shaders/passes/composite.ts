import { RZ_LIGHT_STRUCT_WGSL } from "../lights"
import { sceneTapApi } from "../scene-tap"
import { anchorAliasWgsl } from "../anchor-table"
import { CAST_API, CAST_MASK_ALL, subjectMaskApi } from "../cast-api"
import { clockApi, EFFECT_MATH_API, PARTICLE_STRUCT_WGSL, trailSlotsApi, viewportApi } from "./hosted-api"
import { EFFECT_ANCHORS, EFFECT_SUBJECTS, EFFECT_TRAIL_BASE, EFFECT_TRAIL_SAMPLES } from "../cast-layout"
import { audioApi } from "../audio-api"
import { idApi } from "../id-api"
import { sceneLightApi } from "../scene-light-api"
import { pointsApi } from "../points-api"
import { textureApi } from "../texture-api"
import { castDistanceApi, CAST_FIELD_DIV } from "./cast-distance"
import { lyricsApi, lyricsTextApi } from "../lyrics-api"
import { midiApi } from "../midi-api"
import { gridReadApi } from "./grid"
// Composite: HDR scene + bloom → view transform (soft / neutral / aces / none) → gamma → swapchain.
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
 *     #anchor 左手首 trail
 *     #anchor 頭
 *
 * A declaration in the source, like the mounts: what a file names is what gets
 * resolved and uploaded, so naming none costs nothing and nobody pays for a
 * rig's other five hundred bones. Anchored to the start of a line so that
 * writing the word #anchor in ordinary prose does not silently add a slot —
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
  return [...wgsl.matchAll(/^[ \t]*\/\/[ \t]*#anchor[ \t]+(\S+)([ \t]+trail)?[ \t]*$/gm)]
    .map((m) => ({ bone: m[1], trail: m[2] !== undefined }))
    .slice(0, max)
}

// The cast's caps live in cast-layout.ts — re-exported here because half the
// engine imports them from this file and the constants did not move in meaning.
export { EFFECT_ANCHORS, EFFECT_SUBJECTS, EFFECT_TRAIL_BASE, EFFECT_TRAIL_SAMPLES } from "../cast-layout"

type CompositeEffectSource = {
  /** The user's WGSL verbatim: helpers plus whichever entry points it defines. */
  wgsl: string
  /** Codegen'd `struct EffectParams {...}` + binding decl; empty when no params. */
  paramsDecl: string
  /** Defines `fn background(...)` — mount under the scene. */
  hasBackground: boolean
  /** Defines `fn foreground(...)` — mount over the finished frame. */
  hasForeground: boolean
  /** Calls rzSceneFrame — a FILTER, which carries the layers it read forward
   *  (buildFieldShader) and so draws with blending off. */
  filter?: boolean
  /** `#layer additive`: how a filter's own output joins the layers it carries. */
  additiveLayer?: boolean
  /** Grid resolution when the effect declared `#grid`, else 0. */
  gridSize: number
  /** Whether the scene pass carries the id attachment, so the field module can
   *  bind it. False emits accessors that answer 0 rather than nothing at all. */
  ids?: boolean
  /** How many of this effect's anchors asked for a trail — RZ_TRAIL_SLOTS,
   *  which a hosted particle or trail loop reads even in this module. REQUIRED,
   *  and deliberately not defaulted: a silent 0 is a ribbon loop that iterates
   *  nothing, which looks exactly like an effect that drew nothing. */
  trailCount: number
  /** This effect's local anchor slot → scene slot, from the shared table.
   *  Omitted or identity when it owns the table. The particle and trail modules
   *  have always taken this; the field module not taking it was the bug. */
  alias?: number[]
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
@group(0) @binding(3) var<uniform> viewU: array<vec4<f32>, 16>;
// Aux mask/alpha texture. .r = bloom mask (unused here; bloom blit uses it).
// .g = accumulated canvas alpha (what hdr.a carried before the HDR format
// became rg11b10ufloat). We unpremultiply HDR by this alpha for tonemap, then
// re-premultiply the tonemapped color for output so the premultiplied canvas
// alphaMode composites the WebGPU surface over the page background correctly.
@group(0) @binding(4) var maskTex: texture_2d<f32>;
// viewU[0] = (exposure, invGamma, grain amount, grain seed);  viewU[1] = (tint.rgb, intensity)
// viewU[2] = (background.rgb, mode) — display-space sRGB, composited UNDER the
//            scene post-tonemap. BASE-layer mode: 0 transparent (DOM shows),
//            1 solid color, 2 = 360 equirect skybox sampled by view ray. A user
//            WGSL effect is a separate LAYER over the base — no mode of its own,
//            and no on/off uniform either: the pipeline is REBUILT per effect, so
//            the compiled variant IS the flag.
// viewU[3] = (camera right, tanHalfFov·aspect); viewU[4] = (camera up, tanHalfFov);
// viewU[5] = (camera forward, half pair consumed by a filter this frame) — refreshed
//            per frame while skybox/effect active.
// viewU[6] = (time seconds, view transform id, canvas width, canvas height).
// viewU[7] = (grade offset.rgb, contrast);  viewU[8] = (grade power.rgb, saturation);
// viewU[9] = (grade slope.rgb, grade flags) — bit 0 the CDL grade(), bit 1 the
//            scene's own LUT (setStageGrade) — see _rzGradeScene() below.
// viewU[10] = (camera world position, _) — refreshed with the basis above.
// viewU[11..14] = the cast's positions (effects; count in viewU[10].w).
// viewU[15] = (soft curve contrast, _, _, _).
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
// The scene's own colour grade, as a cube — see _rzStageGrade. sRGB-encoded
// storage, so what a sample returns is already linear.
@group(0) @binding(12) var stageGradeLut: texture_3d<f32>;
// The cast, as data. Read through rzSubject/rzAnchor below — the LAYOUT IS NOT
// STABLE and never will be, because it depends on what each effect declared.
// Reading it directly is the one thing that would freeze it forever.
//
// vec4 slots: [0 .. 15] four subjects, EFFECT_SUBJECT_VEC4S each (root+dissolve,
// hip+objectId, bounds, gaze+looking); then MAX_ANCHORS × four subjects, three
// each (pos+valid, vel, fwd).
@group(0) @binding(11) var<storage, read> _rzCast: array<vec4f>;
// The FIELD LAYER: the user's background/foreground mounts, rendered at half
// resolution in their own pass (see buildFieldShader) and sampled here. Field
// effects are glows, fog, shafts, gradients — low-frequency by nature — and
// running them per quarter-pixel was the single largest per-frame cost an
// effect could add. Bilinear upsampling is invisible at that frequency.
// The field layer, ONE PAIR PER RESOLUTION. 15/16 are full res, 20/21 half.
// An effect draws into the pair it declared, so a starfield that upsamples
// perfectly no longer pays full-resolution pixels because a neighbour needed
// crisp edges. Both pairs are read every frame and combined full-over-half
// below; a pair NO effect draws into is bound to a 1x1 transparent texture
// instead of a target, so nothing has to clear a full-res pair to hand this
// shader the transparent black it would then read.
@group(0) @binding(15) var fieldBgTex: texture_2d<f32>;
@group(0) @binding(16) var fieldFgTex: texture_2d<f32>;
@group(0) @binding(20) var fieldBgHalfTex: texture_2d<f32>;
@group(0) @binding(21) var fieldFgHalfTex: texture_2d<f32>;

/** Two field layers into one, premultiplied OVER, top in front. A resolution
 *  boundary is a layer boundary: within a pair, effects blend in document
 *  order; across pairs, full res wins. (No backticks in here — this string is
 *  a TS template literal and one ends it mid-shader.) */
fn rzFieldMerge(top: vec4f, bot: vec4f) -> vec4f {
  return vec4f(top.rgb + bot.rgb * (1.0 - top.a), top.a + bot.a * (1.0 - top.a));
}

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

/** The sRGB display encoding — "none": no tone curve at all, so the colours a
 *  graph computes are the colours that land. Not a 2.2 power law; sRGB has a
 *  linear toe. Every transform below ends in it. */
fn srgb_encode(x: f32) -> f32 {
  let c = max(x, 0.0);
  return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

/**
 * "soft" — Aether Gazer's own curve, from its final pass
 * (Hidden/SimPipeline/Final, decompiled): (1 - e^(-exposure x))^contrast, at the
 * 2.5 and 1.4 its scenes set, into a linear target the swapchain then encodes.
 * The game's ACES branch is off in every scene measured.
 */
fn softTransform(c: vec3f) -> vec3f {
  let t = pow(max(vec3f(1.0) - exp(-2.5 * max(c, vec3f(0.0))), vec3f(0.0)), vec3f(viewU[15].x));
  return vec3f(srgb_encode(t.r), srgb_encode(t.g), srgb_encode(t.b));
}

fn _rzSrgbEncode3(c: vec3f) -> vec3f {
  return vec3f(srgb_encode(c.r), srgb_encode(c.g), srgb_encode(c.b));
}

/**
 * "neutral" — Unity URP's Neutral tonemapper as the Khronos PBR Neutral
 * operator, to the published formula: below the start of compression colour
 * passes through (less a small toe offset); above it the peak rolls off toward
 * 1 and desaturates in proportion to how much it lost.
 */
fn neutralTransform(color: vec3f) -> vec3f {
  let startCompression = 0.8 - 0.04;
  let desaturation = 0.15;
  var c = max(color, vec3f(0.0));
  let x = min(c.r, min(c.g, c.b));
  let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
  c = c - vec3f(offset);
  let peak = max(c.r, max(c.g, c.b));
  if (peak >= startCompression) {
    let d = 1.0 - startCompression;
    let newPeak = 1.0 - d * d / (peak + d - startCompression);
    c = c * (newPeak / peak);
    let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    c = mix(c, vec3f(newPeak), g);
  }
  return _rzSrgbEncode3(clamp(c, vec3f(0.0), vec3f(1.0)));
}

// "aces" — Unity URP's ACES: unity_to_ACES then AcesTonemap, both from the Core
// RP's Color.hlsl (the branch URP compiles, not full ACES): the RRT's glow and
// red modifier and global desaturation, the luminance fit of RRT +
// ODT.Academy.RGBmonitor_100nits_dim, the dim-surround gamma, the ODT
// desaturation, then AP1 → XYZ → D60→D65 → Rec.709, saturated as URP's
// ApplyTonemap does. HLSL's row-major matrices are written as rows here.
fn _rzRows(r0: vec3f, r1: vec3f, r2: vec3f, v: vec3f) -> vec3f {
  return vec3f(dot(r0, v), dot(r1, v), dot(r2, v));
}
fn _rzAp1ToXyz(v: vec3f) -> vec3f {
  return _rzRows(vec3f(0.6624541811, 0.1340042065, 0.1561876870),
                 vec3f(0.2722287168, 0.6740817658, 0.0536895174),
                 vec3f(-0.0055746495, 0.0040607335, 1.0103391003), v);
}
fn _rzAp1Luma(v: vec3f) -> f32 {
  return dot(v, vec3f(0.272229, 0.674082, 0.0536895));
}
fn acesTransform(color: vec3f) -> vec3f {
  // unity_to_ACES: linear sRGB → AP0.
  var aces = _rzRows(vec3f(0.4397010, 0.3829780, 0.1773350),
                     vec3f(0.0897923, 0.8134230, 0.0967616),
                     vec3f(0.0175440, 0.1115440, 0.8707040), max(color, vec3f(0.0)));
  // Glow module.
  let mi = min(aces.r, min(aces.g, aces.b));
  let ma = max(aces.r, max(aces.g, aces.b));
  let saturation = (max(ma, 1e-4) - max(mi, 1e-4)) / max(ma, 1e-2);
  let k = max(aces.b * (aces.b - aces.g) + aces.g * (aces.g - aces.r) + aces.r * (aces.r - aces.b), 0.0);
  let ycIn = (aces.b + aces.g + aces.r + 1.75 * sqrt(k)) / 3.0;
  let sx = (saturation - 0.4) / 0.2;
  let st = max(1.0 - abs(sx / 2.0), 0.0);
  let s = (1.0 + select(-1.0, 1.0, sx >= 0.0) * (1.0 - st * st)) / 2.0;
  let glowGain = 0.05 * s;
  let glowMid = 0.08;
  var glow = glowGain * (glowMid / max(ycIn, 1e-6) - 0.5);
  if (ycIn <= 2.0 / 3.0 * glowMid) { glow = glowGain; }
  if (ycIn >= 2.0 * glowMid) { glow = 0.0; }
  aces = aces * (1.0 + glow);
  // Red modifier.
  var hue = 0.0;
  if (!(aces.r == aces.g && aces.g == aces.b)) {
    hue = degrees(atan2(sqrt(3.0) * (aces.g - aces.b), 2.0 * aces.r - aces.g - aces.b));
  }
  if (hue < 0.0) { hue = hue + 360.0; }
  var centered = hue;
  if (centered < -180.0) { centered = centered + 360.0; } else if (centered > 180.0) { centered = centered - 360.0; }
  var hueWeight = smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centered / 135.0));
  hueWeight = hueWeight * hueWeight;
  aces.r = aces.r + hueWeight * saturation * (0.03 - aces.r) * (1.0 - 0.82);
  // AP0 → AP1 (ACEScg), then the RRT's global desaturation.
  var cg = max(_rzRows(vec3f(1.4514393161, -0.2365107469, -0.2149285693),
                       vec3f(-0.0765537734, 1.1762296998, -0.0996759264),
                       vec3f(0.0083161484, -0.0060324498, 0.9977163014), aces), vec3f(0.0));
  cg = mix(vec3f(_rzAp1Luma(cg)), cg, vec3f(0.96));
  // The luminance fit of RRT + ODT.Academy.RGBmonitor_100nits_dim.
  let post = (cg * (2.785085 * cg + 0.107772)) / (cg * (2.936045 * cg + 0.887122) + 0.806889);
  // Dark → dim surround: Y^0.9811, in xyY.
  var xyz = _rzAp1ToXyz(post);
  let div = max(xyz.x + xyz.y + xyz.z, 1e-4);
  let xyY = vec3f(xyz.x / div, xyz.y / div, pow(clamp(xyz.y, 0.0, 65504.0), 0.9811));
  let m = xyY.z / max(xyY.y, 1e-4);
  xyz = vec3f(xyY.x * m, xyY.z, (1.0 - xyY.x - xyY.y) * m);
  var lin = _rzRows(vec3f(1.6410233797, -0.3248032942, -0.2364246952),
                    vec3f(-0.6636628587, 1.6153315917, 0.0167563477),
                    vec3f(0.0117218943, -0.0082844420, 0.9883948585), xyz);
  // The ODT's desaturation, then to the display's primaries.
  lin = mix(vec3f(_rzAp1Luma(lin)), lin, vec3f(0.93));
  xyz = _rzRows(vec3f(0.98722400, -0.00611327, 0.0159533),
                vec3f(-0.00759836, 1.00186000, 0.0053302),
                vec3f(0.00307257, -0.00509595, 1.0816800), _rzAp1ToXyz(lin));
  let rec709 = _rzRows(vec3f(3.2409699419, -1.5373831776, -0.4986107603),
                       vec3f(-0.9692436363, 1.8759675015, 0.0415550574),
                       vec3f(0.0556300797, -0.2039769589, 1.0569715142), xyz);
  return _rzSrgbEncode3(clamp(rec709, vec3f(0.0), vec3f(1.0)));
}

/** Which display transform, chosen per frame at viewU[6].y (0 soft, 1 none,
 *  2 neutral, 3 aces). A uniform branch rather than a pipeline variant: switching
 *  is rare, and every arm is cheap enough that specialising the pipeline would
 *  buy nothing. */
fn viewTransform(c: vec3f) -> vec3f {
  let mode = viewU[6].y;
  if (mode > 2.5) { return acesTransform(c); }
  if (mode > 1.5) { return neutralTransform(c); }
  if (mode > 0.5) { return _rzSrgbEncode3(c); }
  return softTransform(c);
}

fn _rzSrgbDecode(x: f32) -> f32 {
  let c = max(x, 0.0);
  return select(pow((c + 0.055) / 1.055, 2.4), c / 12.92, c <= 0.04045);
}

/** The grade a scene arrived with — a game's, baked to a cube by its converter
 *  (setStageGrade). Looked up the way that game's final pass looks it up: the
 *  formed colour, linear, clamped to the cube, with texel centres at the ends,
 *  and trilinear between. The lookup is the whole grade; it has no dials. */
fn _rzStageGrade(c: vec3f) -> vec3f {
  let lin = clamp(vec3f(_rzSrgbDecode(c.r), _rzSrgbDecode(c.g), _rzSrgbDecode(c.b)), vec3f(0.0), vec3f(1.0));
  let n = f32(textureDimensions(stageGradeLut).x);
  let graded = textureSampleLevel(stageGradeLut, bloomSamp, lin * ((n - 1.0) / n) + 0.5 / n, 0.0).rgb;
  return vec3f(srgb_encode(graded.r), srgb_encode(graded.g), srgb_encode(graded.b));
}

/** Both grades, in order: the scene's own first, as the room was graded, then
 *  the author's CDL on top of it. Each is behind its own flag bit in viewU[9].w
 *  so a scene with neither pays one uniform branch. */
fn _rzGradeScene(c: vec3f) -> vec3f {
  let flags = u32(viewU[9].w);
  var x = c;
  if ((flags & 2u) != 0u) { x = _rzStageGrade(x); }
  if ((flags & 1u) != 0u) { x = grade(x); }
  return x;
}
`

/**
 * The scene half of the effect API — camera, projection, cast, trails.
 *
 * Split out of COMPOSITE_HEAD so the SIM pass can have it too. One effect file
 * is spliced into every module it has a mount in, so a file with both a grid and
 * a foreground compiles its foreground inside the grid shader, where every
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

/** How many characters the SCENE holds, up to four — the number the cast api
 *  narrows to this effect's own subjects. See subjectMaskApi. */
fn _rzCastLive() -> i32 { return i32(viewU[10].w); }

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

// The cast — subjects, anchors, trails — is CAST_API, shared verbatim with the
// particle and trail modules. It used to be written out here, a second time, and
// the two copies had drifted: this one had rzAnchor and that one did not.
${CAST_API}

// The hashes, the noise and rzFalloff — EFFECT_MATH_API, the same text the
// particle and trail modules get. The field module used to be missing most of
// it, so a helper an author wrote for one mount failed to compile in another
// for no reason visible in the file.
${EFFECT_MATH_API}${PARTICLE_STRUCT_WGSL}


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
 *
 * THE INDEX IS THE EFFECT'S OWN, like every other cast accessor — Dry Ice, Water,
 * Holy Light and Bloody Ash all read this one and all of them should place their
 * fog, their ripple and their shaft under the model the scene aimed them at. The
 * mapping is the identity for an effect aimed at nobody in particular, which is
 * every scene published before one could be aimed.
 */
fn rzSubjectHip(i: i32) -> vec3f { return viewU[11 + clamp(_rzSubjectSlot(i), 0, 3)].xyz; }

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
${RZ_LIGHT_STRUCT_WGSL}

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
    disp = _rzGradeScene(disp);
  }
  if (APPLY_GAMMA) {
    disp = pow(disp, vec3f(viewU[0].y));
  }
  // ── Film grain, on the SCENE ONLY ─────────────────────────────────────────
  //
  // Applied here, before the background is composited under, so it rides on what
  // the engine drew and nothing else: a backdrop photo or video already carries
  // its own grain, and a second helping over the top would grade the picture
  // rather than match it.
  //
  // Multiplicative and weighted toward the mid-tones, which is how film behaves:
  // little grain in the blacks, and the highlights clip it off.
  if (viewU[0].z > 0.0) {
    let gp = fragCoord.xy + vec2f(viewU[0].w, viewU[0].w * 1.7);
    let gn = fract(sin(dot(gp, vec2f(12.9898, 78.233))) * 43758.5453) - 0.5;
    let glum = dot(disp, vec3f(0.2126, 0.7152, 0.0722));
    disp = max(disp * (1.0 + gn * viewU[0].z * 4.0 * glum * (1.0 - glum)), vec3f(0.0));
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
      if (bg.w > 2.5) {
        // Mode 3: the texels are scene-linear RADIANCE, not display wallpaper.
        // Same exposure, same view transform, same user gamma as the scene —
        // one film for everything in frame, which is what makes a sun roll off
        // like a sun instead of clipping at texture white. bg.x carries the
        // world STRENGTH (the colour slot is dead in equirect modes). The
        // scene's grade stays scene-only, the documented rule above.
        var sky = max(viewTransform(bgPm * bg.x * viewU[0].x), vec3f(0.0));
        if (APPLY_GAMMA) {
          sky = pow(sky, vec3f(viewU[0].y));
        }
        bgPm = sky;
      }
    }
    BACKGROUND_CALL
  }
  // The frame, premultiplied: scene over background. A var, not the return
  // expression, because the foreground mount composites onto it.
  var outRgb = disp * sceneAlpha + bgPm * (1.0 - sceneAlpha);
  var outA = sceneAlpha + bgA * (1.0 - sceneAlpha);
  // Ribbons are NOT read here any more: they draw inside the scene pass, so
  // they are already in disp — tone mapped, and bloomed, which they never
  // were while this line existed. Sampling them here as well would draw them
  // twice, and the layer nothing clears would go stale the moment an effect
  // was removed.
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
    let bgFx = rzFieldMerge(
      clamp(textureSampleLevel(fieldBgTex, bloomSamp, fragCoord.xy / fullSz, 0.0), vec4f(0.0), vec4f(1.0)),
      // The half pair, unless a filter consumed it this frame (viewU[5].w).
      clamp(textureSampleLevel(fieldBgHalfTex, bloomSamp, fragCoord.xy / fullSz, 0.0), vec4f(0.0), vec4f(1.0)) * (1.0 - viewU[5].w));
    bgPm = bgFx.rgb + bgPm * (1.0 - bgFx.a);
    bgA = bgFx.a + bgA * (1.0 - bgFx.a);
`

// Same OVER, one layer later — onto the finished frame rather than onto the
// base. Ungated by design: a foreground runs at every pixel, including the ones
// the model covers, because covering them is the point.
const FOREGROUND_CALL = /* wgsl */ `
  // Premultiplied, as the background layer above — same reason.
  let fgFx = rzFieldMerge(
    clamp(textureSampleLevel(fieldFgTex, bloomSamp, fragCoord.xy / fullSz, 0.0), vec4f(0.0), vec4f(1.0)),
    clamp(textureSampleLevel(fieldFgHalfTex, bloomSamp, fragCoord.xy / fullSz, 0.0), vec4f(0.0), vec4f(1.0)) * (1.0 - viewU[5].w));
  outRgb = fgFx.rgb + outRgb * (1.0 - fgFx.a);
  outA = fgFx.a + outA * (1.0 - fgFx.a);
`

/** The condition on the background block (equirect sample + background effect).
 *
 *  Two jobs. It skips the block behind pixels the model fully covers — the
 *  composite multiplies the result by (1 - alpha) = 0 there anyway, and on a
 *  full-screen effect that's a third or more of the frame (the cost Safari feels
 *  most). And with no background effect compiled in, it also skips the block
 *  entirely unless the equirect needs it.
 *
 *  Everything the gate wraps is an explicit-LOD sample or a texture read, both
 *  always legal in non-uniform flow. It used to also wrap the user's code, which
 *  meant an effect using a derivative builtin had to forfeit the gate; that
 *  carve-out went with the field pass, and the last of it is below. */
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
    EFFECT_SCENE_API +
    subjectMaskApi(CAST_MASK_ALL) +
    anchorAliasWgsl(effect?.alias ?? []) +
    audioApi(0, 13) +
    midiApi(0, 19) +
    lyricsApi(0, 24) +
    body
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
    ? "out.bg = clamp(background(dir, uv, _rzFieldClock.x), vec4f(0.0), vec4f(1.0));"
    : ""
  const fgLine = effect.hasForeground
    ? "out.fg = clamp(foreground(dir, uv, _rzFieldClock.x, linearDepth(vec2<i32>(min(fx, fullSz - 1.0)))), vec4f(0.0), vec4f(1.0));"
    : ""
  // A FILTER read the other effects' layers through rzSceneFrame and now
  // carries them: its own output goes over (or adds to) what they drew, and
  // the pipeline writes the result with blending off. Weight has already
  // scaled its alpha, so a filter faded to nothing passes the layers through
  // untouched.
  const carry = effect.filter
    ? effect.additiveLayer
      ? "out.bg = _rzCarryAdd(out.bg, _rzLayerBg(uv));\n  out.fg = _rzCarryAdd(out.fg, _rzLayerFg(uv));"
      : "out.bg = _rzCarryOver(out.bg, _rzLayerBg(uv));\n  out.fg = _rzCarryOver(out.fg, _rzLayerFg(uv));"
    : ""
  return (
    COMPOSITE_HEAD +
    EFFECT_SCENE_API +
    // Which characters this effect is on, out of its own clock block — see the
    // declaration of _rzFieldClock below for why that buffer is per effect.
    subjectMaskApi("u32(_rzFieldClock.z)") +
    anchorAliasWgsl(effect.alias ?? []) +
    audioApi(0, 13) +
    midiApi(0, 19) +
    lyricsApi(0, 24) +
    // The words themselves — the atlas rides the grid's sampler, which is
    // declared just below and resolves module-wide.
    lyricsTextApi(0, 25, "_rzGridSamp") +
    // The persistent grid, always bound — a 1×1 of zeroes when the effect has
    // none, so rzGrid() is a function that always exists rather than one an
    // author has to know whether they are allowed to call.
    gridReadApi(0, 17, 18, effect.gridSize) +
    clockApi("_rzFieldClock.x", "0.0") +
    viewportApi("viewU[6].w") +
    trailSlotsApi(effect.trailCount) +
    idApi(effect.ids === true, 0, 23) +
    sceneLightApi(false, 0, 0) +
    pointsApi(false) +
    textureApi(0) +
    // Distance to the cast, in SCREEN pixels: the field is half-res, so a field
    // texel is CAST_FIELD_DIV of them and the accessor scales on the way out.
    // An author writes the width they mean and never learns how it is built.
    castDistanceApi(0, 26, 18, CAST_FIELD_DIV) +
    sceneTapApi(0, 27, 28, [29, 30, 31, 32]) +
    "\n// ── user effect (setEffect) ──\n" +
    effect.paramsDecl +
    "\n" +
    effect.wgsl +
    "\n" +
    /* wgsl */ `
@group(0) @binding(14) var<uniform> fieldU: vec4f;
/**
 * THIS EFFECT'S OWN clock, seconds since it was installed.
 *
 * Per effect, and that is the whole point of it existing. The time argument
 * used to come from viewU[6].x, which is measured from the FIRST installed
 * effect's epoch — so every later effect started mid-stream, and an effect
 * whose lightEmit read its own epoch disagreed with its own background()
 * about what time it was. One buffer per effect, one answer.
 */
@group(0) @binding(22) var<uniform> _rzFieldClock: vec4f;   // (time, weight, subject mask, _)

/** A filter's own straight-alpha output over the premultiplied layers it read
 *  (the layer blend state, restated) and the additive form of the same. */
fn _rzCarryOver(own: vec4f, under: vec4f) -> vec4f {
  return vec4f(own.rgb * own.a + under.rgb * (1.0 - own.a), own.a + under.a * (1.0 - own.a));
}
fn _rzCarryAdd(own: vec4f, under: vec4f) -> vec4f {
  return vec4f(own.rgb * own.a + under.rgb, own.a + under.a);
}

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
  // WEIGHT, applied where the author cannot decline it.
  //
  // Alpha only: both field blends multiply the fragment's colour by src-alpha,
  // so this is the fade for the alpha-over layer and the additive one alike.
  // Scaling colour as well would fade as the square.
  out.bg.a *= _rzFieldClock.y;
  out.fg.a *= _rzFieldClock.y;
  ${carry}
  return out;
}
`
  )
}

/** Kept for compatibility with existing imports (the base, no-effect shader). */
export const COMPOSITE_SHADER_WGSL = buildCompositeShader(null)
