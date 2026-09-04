import { sceneFsOutWgsl, sceneIdWriteWgsl } from "./scene-contract"
import { lightsApi } from "../lights"
import { SHADOW_CASCADES } from "../../shadow-cascades"
import { WORLD_AMBIENT_WGSL } from "../lights"

// Ground shadow-catcher: receives directional shadow, grid lines, frosted noise,
// radial distance fade. Writes bloom mask = 0 (ground never bloom-bleeds).
//
// A FUNCTION, not a constant, since the id attachment is a device capability:
// the struct this returns depends on what the probe at init found, and a string
// baked at import time cannot know that. Called once, when the module is built.

// How far the softest setting spreads the taps, in units of the sharp kernel's
// own step, and how many taps carry that spread. Sixteen is what keeps a wide
// disk reading as a penumbra instead of sixteen shadows.
const SOFT_TAPS = 16
const SOFT_MAX_SPREAD = 14

/**
 * PCF taps for one cascade, emitted per shadow map.
 *
 * At softness 0 this is the 3x3 box the ground has always used — same offsets,
 * same weights, same result to the bit. Above it the taps spread over a Vogel
 * disk whose radius IS the penumbra: an overcast sky throws no edge, and a
 * razor-edged shadow under one is the loudest thing wrong in a composite.
 *
 * The branch is on a uniform, so a scene at softness 0 genuinely takes the
 * nine-tap side rather than masking the wide one — the same reasoning the
 * shadowStrength branch above it already documents.
 *
 * A string rather than a WGSL function because a texture is the one handle type
 * this shader has never passed as a parameter, and the file is already a
 * template that interpolates its constants.
 *
 * `acc` must already be declared and zeroed; the caller reads it back
 * normalised, so there is no `/ 9.0` left at the call site.
 */
function pcfWgsl(map: string, uv: string, texel: string, z: string, acc: string, pad: string): string {
  const golden = 2.39996323
  return [
    `if (material.shadowSoftness <= 0.0) {`,
    `  let st = ${texel} * 2.0;`,
    `  for (var y = -1; y <= 1; y++) {`,
    `    for (var x = -1; x <= 1; x++) {`,
    // ...Level, not the implicit-derivative form: identical on a single-mip
    // shadow map, and legal inside a branch.
    `      ${acc} += textureSampleCompareLevel(${map}, shadowSampler, ${uv} + vec2f(f32(x), f32(y)) * st, ${z});`,
    `    }`,
    `  }`,
    `  ${acc} *= ${1 / 9};`,
    `} else {`,
    `  let radius = ${texel} * 2.0 * (1.0 + material.shadowSoftness * ${SOFT_MAX_SPREAD}.0);`,
    `  for (var s = 0; s < ${SOFT_TAPS}; s++) {`,
    `    let fs = f32(s);`,
    // sqrt of the index spaces the ring radii evenly by AREA; the golden angle
    // keeps successive taps from lining up into spokes.
    `    let r = sqrt((fs + 0.5) * ${1 / SOFT_TAPS});`,
    `    let a = fs * ${golden} + rot;`,
    `    ${acc} += textureSampleCompareLevel(${map}, shadowSampler, ${uv} + vec2f(cos(a), sin(a)) * (r * radius), ${z});`,
    `  }`,
    `  ${acc} *= ${1 / SOFT_TAPS};`,
    `}`,
  ]
    .map((l) => pad + l)
    .join("\n")
}

export function groundShaderWgsl(): string {
  return /* wgsl */ `
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _p: f32, };
struct Light { direction: vec4f, color: vec4f, };
struct LightUniforms { ambientColor: vec4f, lights: array<Light, 4>, sh: array<vec4f, 9>, };
struct GroundShadowMat {
  diffuseColor: vec3f, fadeStart: f32,
  fadeEnd: f32, shadowStrength: f32, pcfTexel: f32, gridSpacing: f32,
  gridLineWidth: f32, gridLineOpacity: f32, noiseStrength: f32, opacity: f32,
  gridLineColor: vec3f, mirror: f32,
  // farCascade: 1 while a stage is loaded, 0 otherwise. See the branch below —
  // with no stage the far map is never drawn into, so its taps are known.
  mirrorBlur: f32, farCascade: f32, shadowSoftness: f32, _mb2: f32,
  // Every shadow caster in one sphere, refreshed per frame. w = radius; 0 means
  // nothing casts, negative means "do not use this" (a rigid caster has no
  // sphere, so a scene with a stage keeps the taps). See rzShadowPossible.
  casterSphere: vec4f,
};
// One view-projection per shadow cascade, inner to outer — same buffer and
// same order the materials read.
struct LightVP { viewProj: array<mat4x4f, ${SHADOW_CASCADES.length}>, };
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> light: LightUniforms;
@group(0) @binding(2) var shadowMap: texture_depth_2d;
@group(0) @binding(3) var shadowSampler: sampler_comparison;
@group(0) @binding(4) var<uniform> material: GroundShadowMat;
@group(0) @binding(5) var<uniform> lightVP: LightVP;
// The far cascade's map, binding 7 as in the materials' layout.
@group(0) @binding(7) var shadowMapFar: texture_depth_2d;
// The floor mirror (step 7D): the reflection target, the camera that rendered
// it, and an ordinary sampler — the two shadow bindings above are comparison.
// params = (projA, projB, _, _) — the depth-linearisation pair of the SHARED
// projection, refreshed per frame beside the matrix.
struct MirrorVP { viewProj: mat4x4f, params: vec4f, };
@group(0) @binding(8) var<uniform> mirrorVP: MirrorVP;
@group(0) @binding(9) var mirrorTex: texture_2d<f32>;
@group(0) @binding(10) var linearSampler: sampler;
@group(0) @binding(11) var mirrorDepth: texture_depth_multisampled_2d;
// The frost noise, PRE-BAKED — see GROUND_NOISE_BAKE_WGSL below. Sampled with
// the repeat sampler already at binding 10.
@group(0) @binding(12) var noiseTex: texture_2d<f32>;
${WORLD_AMBIENT_WGSL}
${lightsApi(0, 6)}

struct VO { @builtin(position) position: vec4f, @location(0) worldPos: vec3f, @location(1) normal: vec3f, };
@vertex fn vs(@location(0) position: vec3f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> VO {
  var o: VO; o.worldPos = position; o.normal = normal;
  o.position = camera.projection * camera.view * vec4f(position, 1.0); return o;
}
${sceneFsOutWgsl()}@fragment fn fs(i: VO) -> FSOut {
  // Derivatives first, unconditionally: WGSL requires fwidth in uniform control
  // flow and everything below branches on world position.
  let gp = i.worldPos.xz / material.gridSpacing;
  let gridDeriv = fwidth(gp);

  var out: FSOut;
  let n = normalize(i.normal);
  let centerDist = length(i.worldPos.xz);
  let edgeFade = 1.0 - smoothstep(0.0, 1.0, clamp((centerDist - material.fadeStart) / max(material.fadeEnd - material.fadeStart, 0.001), 0.0, 1.0));

  // Past the radial fade every term below is multiplied by edgeFade and the
  // pixel is fully transparent. Shading it anyway produces nothing.
  if (edgeFade <= 0.0) {
    out.color = vec4f(0.0);
    out.mask = vec4f(0.0, 1.0, 0.0, 0.0);
    return out;
  }

  // Outside a cascade's frustum there IS no shadow information — the clamped
  // border samples compare against unrelated depths and read "shadowed",
  // darkening the whole far plane with a visible band at the frustum edge
  // (masked by the opaque surface normally, glaring in green-screen mode where
  // only the shadow-catcher term renders). So each cascade fades out near its
  // border — the near one INTO the far one's answer, the far one into lit.
  let l0 = lightVP.viewProj[0] * vec4f(i.worldPos, 1.0);
  let ndc = l0.xyz / max(l0.w, 1e-6);
  let inZ = select(0.0, 1.0, ndc.z > 0.0 && ndc.z < 1.0);
  let frustum = (1.0 - smoothstep(0.88, 0.96, abs(ndc.x))) * (1.0 - smoothstep(0.88, 0.96, abs(ndc.y))) * inZ;
  let l1 = lightVP.viewProj[1] * vec4f(i.worldPos, 1.0);
  let ndc1 = l1.xyz / max(l1.w, 1e-6);
  let inZ1 = select(0.0, 1.0, ndc1.z > 0.0 && ndc1.z < 1.0);
  let frustum1 = (1.0 - smoothstep(0.88, 0.96, abs(ndc1.x))) * (1.0 - smoothstep(0.88, 0.96, abs(ndc1.y))) * inZ1;

  // THE cost of this shader, measured: a full-screen ground ran 25 shadow
  // comparisons per pixel, and each one is hardware-bilinear, so ~100 depth
  // samples per pixel. Gutting the rest of the shader changed nothing; cutting
  // this is what makes looking down cheap.
  //
  // 3x3 at two-texel spacing covers the same +/-2 texel penumbra the 5x5 at
  // one-texel spacing did — that kernel was oversampled, its taps overlapping
  // almost entirely once the comparison sampler's own 2x2 filter is counted.
  // Nine taps instead of twenty-five for the same blur radius.
  //
  // The far cascade's taps run ONLY where the near one is fading or absent
  // (frustum < 1), so a pixel in the near core costs exactly what it did with
  // one cascade — and that core is where the camera usually looks.
  // Can anything cast a shadow ONTO this point at all?
  //
  // The ground is vastly larger than what stands on it, and the answer for most
  // of it is no. A point is shadowed only if the ray from it toward the light
  // meets a caster, so testing that ray against one bounding sphere decides in a
  // few ALU ops what nine hardware-bilinear depth comparisons would otherwise be
  // spent discovering. Same pixels: outside the sphere vis stays 1.0, which is
  // exactly what the taps return.
  //
  // Conservative in all three directions that matter. The sphere is a union of
  // per-model bounds built from POSED bones plus the skin margin, so a jump or a
  // flying skirt is inside it. A negative radius disables the test outright. And
  // the halfspace check uses -R rather than 0, so a caster the point is standing
  // inside still counts.
  var shadowPossible = material.casterSphere.w > 0.0;
  if (shadowPossible) {
    let toLight = normalize(-light.lights[0].direction.xyz);
    let toCaster = material.casterSphere.xyz - i.worldPos;
    let along = dot(toCaster, toLight);
    let perp = length(toCaster - toLight * along);
    shadowPossible = along > -material.casterSphere.w && perp <= material.casterSphere.w;
  } else {
    // Negative radius is the opt-out, not "nothing casts".
    shadowPossible = material.casterSphere.w < 0.0;
  }

  var vis = 1.0;
  // The whole cascade lookup, behind the strength dial.
  //
  // This is a FULL-COVERAGE draw and each branch below is nine comparison-sampler
  // taps per pixel — eighteen where the cascades overlap — so on a retina canvas
  // it is tens of millions of shadow-map fetches a frame. All of it was computed
  // and then multiplied by material.shadowStrength, which is ZERO whenever the
  // host turns the ground's shadow catcher off. Turning shadows off in the UI
  // therefore cost exactly what leaving them on cost, which is what a report of
  // "no fps gain from disabling shadows" looks like from the outside.
  //
  // shadowStrength is a uniform, so this branch is uniform across the draw:
  // every invocation takes the same side and the taps are genuinely skipped
  // rather than merely masked. The dark term below is (1.0 - vis) * strength, so
  // with strength at zero the result is identical either way — a pure
  // deletion of work, not a change of look.
  //
  // The same reasoning the noise tint below already got, applied to the term
  // that costs a hundred times more.
  if (material.shadowStrength > 0.0 && shadowPossible) {
  // Per-pixel rotation for the soft disk, so its rings break up into fine noise
  // rather than banding. Interleaved gradient noise: a function of the pixel
  // alone, so a still camera gives a still shadow — the sharp path ignores it.
  let ign = fract(52.9829189 * fract(dot(i.position.xy, vec2f(0.06711056, 0.00583715))));
  let rot = ign * 6.28318530718;
  // The far cascade's taps, skipped entirely when nothing ever drew into it.
  //
  // This branch is the expensive one on a wide floor: it runs wherever the NEAR
  // cascade does not reach, and the near cascade is a 64-unit box around the
  // camera target, so a floor receding to the horizon takes it almost
  // everywhere. Nine comparison taps, per pixel, on the largest draw in the
  // frame — against a map that is cleared unless a stage is in the scene.
  //
  // Cleared depth compares as "no occluder", so the skipped path leaves vis at
  // 1.0, which is exactly what the taps returned. Free, not cheaper.
  if (material.farCascade > 0.0 && frustum < 1.0 && frustum1 > 0.0) {
    let suv1 = vec2f(ndc1.x * 0.5 + 0.5, 0.5 - ndc1.y * 0.5);
    let suv1_c = clamp(suv1, vec2f(0.02), vec2f(0.98));
    let compareZ1 = ndc1.z - 0.0035;
    var acc1 = 0.0;
${pcfWgsl("shadowMapFar", "suv1_c", `${1 / SHADOW_CASCADES[SHADOW_CASCADES.length - 1].mapSize}`, "compareZ1", "acc1", "    ")}
    vis = mix(1.0, acc1, frustum1);
  }
  if (frustum > 0.0) {
    let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let suv_c = clamp(suv, vec2f(0.02), vec2f(0.98));
    let compareZ = ndc.z - 0.0035;
    var acc = 0.0;
${pcfWgsl("shadowMap", "suv_c", "material.pcfTexel", "compareZ", "acc", "    ")}
    // The base is whatever the far cascade decided, so the near border blends
    // cascade to cascade rather than snapping to lit mid-floor.
    vis = mix(vis, acc, frustum);
  }
  }

  // Frosted/matte micro-texture. Scenes leaving it at zero were still paying
  // sixteen hash rounds a pixel for a tint of exactly 1.
  var noiseTint = 1.0;
  if (material.noiseStrength != 0.0) {
    // One texture sample where four octaves of value noise used to be
    // EVALUATED — sixteen hash rounds per pixel, on the largest draw in the
    // frame. Bisection on the device that was slow pinned the ground's whole
    // cost here: flat colour was smooth, the shader minus its shadow taps was
    // still slow, and the shader minus only this was smooth. (0.33.2 built
    // this same bake and measured no win — the cost then was the PCF. The
    // instrument was right both times; the term changed.)
    //
    // The tile is the SAME fbm, baked once at init over a 64-p-unit period
    // (~21 world units before it repeats — invisible at frost strengths), and
    // the interior of the tile is bit-identical to the old evaluation because
    // only lattice cells on the seam are wrapped. Sampled at level 0 with the
    // repeat sampler: the old evaluation had no filtering either, so distant
    // sparkle is unchanged rather than quietly "improved".
    let noiseVal = textureSampleLevel(noiseTex, linearSampler, i.worldPos.xz * ${3 / 64}, 0.0).r;
    noiseTint = 1.0 + (noiseVal - 0.5) * material.noiseStrength;
  }

  // Grid lines — anti-aliased via screen-space derivatives
  let gridFrac = abs(fract(gp - 0.5) - 0.5);
  let halfLine = material.gridLineWidth * 0.5;
  let gridLine = 1.0 - min(
    smoothstep(halfLine - gridDeriv.x, halfLine + gridDeriv.x, gridFrac.x),
    smoothstep(halfLine - gridDeriv.y, halfLine + gridDeriv.y, gridFrac.y)
  );
  let sun = rzWorldAmbient(n) + light.lights[0].color.xyz * light.lights[0].color.w * max(dot(n, -light.lights[0].direction.xyz), 0.0);
  // The positional layer reaches the floor too. A stage light that lit the
  // cast and not the ground under her would read as a sticker, which is the
  // same failure the shadow catcher exists to prevent.
  let lamps = rzLightsDiffuse(i.worldPos, n);
  let dark = (1.0 - vis) * material.shadowStrength;
  // The shadow darkens the SUN only. The dark term comes from the sun's shadow
  // map, so applying it to the lamps too would have a lamp on the far side of
  // the stage dimmed by a shadow cast away from it — the one light in the scene
  // that provably is not blocking it.
  // (No backticks in here: this string is a TS template literal and one ends it
  // mid-shader, which is exactly how this line broke the build a moment ago.)
  var baseColor = material.diffuseColor * (sun * (1.0 - dark * 0.65) + lamps);
  baseColor *= noiseTint;
  // The floor mirror — its OWN LAYER, not a blend into the surface. The
  // reflection was rendered by the MIRROR camera, so projecting this
  // fragment's world position through that camera yields exactly the texel
  // where its reflection landed — projective mapping, no screen-space guess.
  // There is deliberately NO strength dial: mirror is on or off, and how much
  // of it shows is the SURFACE's own opacity covering it — the same
  // independence the grid won earlier. The branch is on a uniform, so the
  // whole cost vanishes for the scenes that leave it off.
  var reflShadowed = vec3f(0.0);
  if (material.mirror > 0.0) {
    let mc = mirrorVP.viewProj * vec4f(i.worldPos, 1.0);
    let mndc = mc.xyz / max(mc.w, 1e-6);
    let muv = clamp(vec2f(mndc.x * 0.5 + 0.5, 0.5 - mndc.y * 0.5), vec2f(0.0), vec2f(1.0));
    var lod = 0.0;
    if (material.mirrorBlur > 0.0) {
      // DEPTH-PROPORTIONAL softness: the real driver is how far BEHIND the
      // mirror surface the reflected geometry sits — a foot on the floor
      // reflects sharp, a head reflects soft, the empty backdrop softest.
      //
      // Read the mirror pass's own depth at this texel, linearise it with the
      // shared projection's pair (viewZ = projB / (z - projA); the formula
      // inverts both depth conventions, see the composite's linearDepth), and
      // reconstruct the reflected image point along the ray from the MIRRORED
      // eye through this fragment. The plane is y = 0, so mirroring the eye
      // and the camera forward is a sign flip on y, and the image point's
      // depth below the plane IS the reflected object's height above it.
      let dims = vec2f(textureDimensions(mirrorDepth));
      let texel = clamp(vec2i(muv * dims), vec2i(0), vec2i(dims) - vec2i(1));
      let z = textureLoad(mirrorDepth, texel, 0);
      let viewZ = clamp(mirrorVP.params.y / (z - mirrorVP.params.x), 0.05, 100000.0);
      let eyeM = vec3f(camera.viewPos.x, -camera.viewPos.y, camera.viewPos.z);
      let fwd = vec3f(camera.view[0][2], camera.view[1][2], camera.view[2][2]);
      let fwdM = vec3f(fwd.x, -fwd.y, fwd.z);
      let toFrag = i.worldPos - eyeM;
      let dir = toFrag * inverseSqrt(max(dot(toFrag, toFrag), 1e-8));
      let t = viewZ / max(dot(dir, fwdM), 1e-4);
      let height = max(-(eyeM.y + dir.y * t), 0.0);
      // Full softness by BLUR_SPAN units above the floor — about a character's
      // height. The hardware clamps the level, so the nominal 5.0 needs no
      // knowledge of the real chain length, and blur 0 reads exactly level 0.
      let BLUR_SPAN = 14.0;
      lod = material.mirrorBlur * 5.0 * clamp(height / BLUR_SPAN, 0.0, 1.0);
    }
    let refl = textureSampleLevel(mirrorTex, linearSampler, muv, lod).rgb;
    // Still under the received shadow: a polished floor in shade shows a dim
    // reflection, and a mirror that ignored the shadow would glow in it.
    reflShadowed = refl * (1.0 - dark * 0.65);
  }
  // THREE LAYERS, composited premultiplied, bottom to top: the shadow the
  // catcher receives, the ground surface, and the grid.
  //
  // The grid used to be MIXED INTO the surface and then weighted by the
  // surface's own alpha, so turning the ground down to nothing took the grid
  // with it — you could not have a bare grid over a photo, which is most of
  // what a grid is for. It is its own layer now: the ground plane keeps an
  // alpha, and the grid is simply on or off.
  let surfA = edgeFade * material.opacity;
  let gridA = gridLine * material.gridLineOpacity * edgeFade;
  // The mirror is opaque across the ground's extent when on — it always shows
  // SOMETHING, because the backdrop rides in the reflection target itself.
  let mirrA = material.mirror * edgeFade;
  // As opacity drops the received shadow becomes a translucent dark layer
  // (Blender's Shadow Catcher), so models still feel grounded on a photo or a
  // 360 backdrop. Colourless by construction: it darkens by covering. The
  // MIRROR subsumes it: the reflection already carries the received shadow,
  // and a catcher stacked on top would darken the same shadow twice.
  let catchA = dark * 0.65 * edgeFade * (1.0 - material.opacity) * (1.0 - material.mirror);

  // FOUR LAYERS, premultiplied, bottom to top: reflection, received shadow,
  // ground surface, grid. Each with its own coverage, none scaling another —
  // ground opacity REVEALS the mirror rather than the mirror carrying a
  // strength of its own.
  var pm = reflShadowed * mirrA;
  var cov = mirrA;
  // Shadow catcher over the reflection-or-nothing: black, it covers.
  pm = pm * (1.0 - catchA);
  cov = catchA + cov * (1.0 - catchA);
  // Surface over shadow.
  pm = baseColor * surfA + pm * (1.0 - surfA);
  cov = surfA + cov * (1.0 - surfA);
  // Grid over surface. Unshadowed, as it has always been — a painted line reads
  // as paint, and darkening it was never what the mix did either.
  pm = material.gridLineColor * gridA + pm * (1.0 - gridA);
  cov = gridA + cov * (1.0 - gridA);

  let outA = cov;
  out.color = vec4f(pm, outA);
  // mask.r = 0: ground never contributes to bloom. mask.g = 1.0 with src.a =
  // outA turns the aux blend into alpha-over, so the drawable alpha goes
  // from outA at the center to 0 at the radial edge — letting the page
  // background show through under the premultiplied canvas alphaMode.
  out.mask = vec4f(0.0, 1.0, 0.0, outA);
${sceneIdWriteWgsl("out", `${GROUND_MATERIAL_ID}u`, `${GROUND_OBJECT_ID}u`)}  return out;
}
`
}

/**
 * The ground belongs to no model instance, so it takes ids of its own — at the
 * TOP of the u16 range, not at the bottom.
 *
 * Both of the engine's counters are 1-based (model id is instance count + 1,
 * material id counts from one), so 1 is the first model's first material and
 * the floor would have shared it. 0 stays unclaimed in the other direction: it
 * is what the attachment clears to, and so what every pixel nothing drew
 * reports.
 */
export const GROUND_MATERIAL_ID = 0xffff
export const GROUND_OBJECT_ID = 0xffff

/** Texel edge of the baked frost tile. 1024 over a 64-p-unit period is two
 *  texels per finest-octave noise cell — the bake resolves everything the
 *  evaluation produced. r8unorm: the value lives in [0,1] and feeds a 0.05
 *  tint; eight bits is already 5x below what that tint can show. */
export const GROUND_NOISE_SIZE = 1024

/**
 * The frost fbm as a bake: identical math, periodic lattice.
 *
 * hash2/valueNoise/fbm are the ground's old helpers verbatim, with one change —
 * each octave's lattice wraps at its own period (64 p-units at octave 0,
 * doubling with frequency), so the tile is seamless under the repeat sampler.
 * Wrapping only renames lattice cells on the seam; every interior cell hashes
 * exactly as before, which is what makes the bake a relocation of the old
 * pixels rather than a new look.
 */
export const GROUND_NOISE_BAKE_WGSL = /* wgsl */ `
@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}
fn hash2(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  p3 += dot(p3, vec3f(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  return fract((p3.x + p3.y) * p3.z);
}
fn wrapCell(i: vec2f, period: f32) -> vec2f { return i - period * floor(i / period); }
fn valueNoiseP(p: vec2f, period: f32) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(wrapCell(i, period)), hash2(wrapCell(i + vec2f(1.0, 0.0), period)), u.x),
             mix(hash2(wrapCell(i + vec2f(0.0, 1.0), period)), hash2(wrapCell(i + vec2f(1.0, 1.0), period)), u.x), u.y);
}
@fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let p = pos.xy * (64.0 / ${GROUND_NOISE_SIZE}.0);
  var v = 0.0;
  var a = 0.5;
  var pp = p;
  var period = 64.0;
  for (var o = 0; o < 4; o++) {
    v += a * valueNoiseP(pp, period);
    pp *= 2.0;
    period *= 2.0;
    a *= 0.5;
  }
  return vec4f(v, 0.0, 0.0, 1.0);
}
`
