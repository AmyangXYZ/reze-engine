import { sceneFsOutWgsl, sceneIdWriteWgsl } from "./scene-contract"

// Ground shadow-catcher: receives directional shadow, grid lines, frosted noise,
// radial distance fade. Writes bloom mask = 0 (ground never bloom-bleeds).
//
// A FUNCTION, not a constant, since the id attachment is a device capability:
// the struct this returns depends on what the probe at init found, and a string
// baked at import time cannot know that. Called once, when the module is built.

export function groundShaderWgsl(): string {
  return /* wgsl */ `
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _p: f32, };
struct Light { direction: vec4f, color: vec4f, };
struct LightUniforms { ambientColor: vec4f, lights: array<Light, 4>, };
struct GroundShadowMat {
  diffuseColor: vec3f, fadeStart: f32,
  fadeEnd: f32, shadowStrength: f32, pcfTexel: f32, gridSpacing: f32,
  gridLineWidth: f32, gridLineOpacity: f32, noiseStrength: f32, opacity: f32,
  gridLineColor: vec3f, _pad2: f32,
};
struct LightVP { viewProj: mat4x4f, };
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> light: LightUniforms;
@group(0) @binding(2) var shadowMap: texture_depth_2d;
@group(0) @binding(3) var shadowSampler: sampler_comparison;
@group(0) @binding(4) var<uniform> material: GroundShadowMat;
@group(0) @binding(5) var<uniform> lightVP: LightVP;

fn hash2(p: vec2f) -> f32 {
  var p3 = fract(vec3f(p.x, p.y, p.x) * 0.1031);
  p3 += dot(p3, vec3f(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
  return fract((p3.x + p3.y) * p3.z);
}
fn valueNoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), u.x),
             mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), u.x), u.y);
}
fn fbmNoise(p: vec2f) -> f32 {
  var v = 0.0;
  var a = 0.5;
  var pp = p;
  for (var i = 0; i < 4; i++) {
    v += a * valueNoise(pp);
    pp *= 2.0;
    a *= 0.5;
  }
  return v;
}

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

  let lclip = lightVP.viewProj * vec4f(i.worldPos, 1.0);
  let ndc = lclip.xyz / max(lclip.w, 1e-6);
  // Outside the light's frustum there IS no shadow information — the clamped
  // border samples compare against unrelated depths and read "shadowed",
  // darkening the whole far plane with a visible band at the frustum edge
  // (masked by the opaque surface normally, glaring in green-screen mode where
  // only the shadow-catcher term renders). Fade to fully lit near the border.
  let inZ = select(0.0, 1.0, ndc.z > 0.0 && ndc.z < 1.0);
  let frustum = (1.0 - smoothstep(0.88, 0.96, abs(ndc.x))) * (1.0 - smoothstep(0.88, 0.96, abs(ndc.y))) * inZ;

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
  // Skipped altogether outside the light frustum, where the mix below discards
  // the result anyway: the shadow map covers a small area around the character
  // and the ground is vastly larger than it.
  var vis = 1.0;
  if (frustum > 0.0) {
    let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let suv_c = clamp(suv, vec2f(0.02), vec2f(0.98));
    let st = material.pcfTexel * 2.0;
    let compareZ = ndc.z - 0.0035;
    var acc = 0.0;
    for (var y = -1; y <= 1; y++) {
      for (var x = -1; x <= 1; x++) {
        // ...Level, not the implicit-derivative form: identical on a single-mip
        // shadow map, and legal inside this branch.
        acc += textureSampleCompareLevel(shadowMap, shadowSampler, suv_c + vec2f(f32(x), f32(y)) * st, compareZ);
      }
    }
    vis = mix(1.0, acc * (1.0 / 9.0), frustum);
  }

  // Frosted/matte micro-texture. Scenes leaving it at zero were still paying
  // sixteen hash rounds a pixel for a tint of exactly 1.
  var noiseTint = 1.0;
  if (material.noiseStrength != 0.0) {
    let noiseVal = fbmNoise(i.worldPos.xz * 3.0);
    noiseTint = 1.0 + (noiseVal - 0.5) * material.noiseStrength;
  }

  // Grid lines — anti-aliased via screen-space derivatives
  let gridFrac = abs(fract(gp - 0.5) - 0.5);
  let halfLine = material.gridLineWidth * 0.5;
  let gridLine = 1.0 - min(
    smoothstep(halfLine - gridDeriv.x, halfLine + gridDeriv.x, gridFrac.x),
    smoothstep(halfLine - gridDeriv.y, halfLine + gridDeriv.y, gridFrac.y)
  );
  let sun = light.ambientColor.xyz + light.lights[0].color.xyz * light.lights[0].color.w * max(dot(n, -light.lights[0].direction.xyz), 0.0);
  let dark = (1.0 - vis) * material.shadowStrength;
  var baseColor = material.diffuseColor * sun * (1.0 - dark * 0.65);
  baseColor *= noiseTint;
  let finalColor = mix(baseColor, material.gridLineColor, gridLine * material.gridLineOpacity * edgeFade);
  // Whole-ground opacity fades the SURFACE (color, grid) but the shadow stays —
  // as opacity drops, the received shadow becomes a translucent dark layer
  // (Blender's Shadow Catcher), so models still feel grounded on a photo or
  // 360 backdrop. At opacity 1 this reduces exactly to the plain surface.
  let surfA = edgeFade * material.opacity;
  let catchA = dark * 0.65 * edgeFade * (1.0 - material.opacity);
  let outA = surfA + catchA;
  out.color = vec4f(finalColor * surfA, outA);
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
