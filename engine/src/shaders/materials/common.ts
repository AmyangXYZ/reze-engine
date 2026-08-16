import { sceneFsOutWgsl } from "../passes/scene-contract"
import { lightsApi } from "../lights"

// Shared WGSL blocks concatenated by every material shader.
// Splits the boilerplate (uniform structs, bind group layout, skinning VS, PCF shadow)
// away from the per-material fragment code so each material file only contains what
// makes it visually distinct.
//
// Concat order in every material:
//   NODES_WGSL              (nodes.ts — math/noise/BSDF helpers)
//   COMMON_BINDINGS_WGSL    (uniform structs + @group/@binding declarations)
//   SAMPLE_SHADOW_WGSL      (3×3 PCF shadow sampler; reads bindings above)
//   COMMON_VS_WGSL          (skinning vertex shader; reads bindings above)
//   <material's own constants + @fragment fn fs>
//
// WGSL is a whole-module compile — declaration order at module scope doesn't matter,
// but the readable order is: types → bindings → helpers → entry points.

// ─── Uniform structs + bind group layout ────────────────────────────
// Every material pipeline uses the same bind group layout, so the same bindings are
// declared here once. Groups:
//   group(0): per-frame scene (camera, lights, shadow map, BRDF LUT via nodes.ts)
//   group(1): per-model skinning
//   group(2): per-material (diffuse texture + material uniforms)

export const COMMON_BINDINGS_WGSL = /* wgsl */ `

struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
  viewPos: vec3f,
  _padding: f32,
};

struct Light {
  direction: vec4f,
  color: vec4f,
};

struct LightUniforms {
  ambientColor: vec4f,
  lights: array<Light, 4>,
};

// Per-material uniforms. Every material binds this layout even if it ignores fields;
// the engine keeps one bind group layout across all material pipelines. The PMX
// classic-material fields (ambient/specular/shininess) are carried for graph nodes that
// want them; most graphs read only diffuseColor + alpha.
struct MaterialUniforms {
  diffuseColor: vec3f,   // PMX diffuse rgb — the material_diffuse node reads this
  alpha: f32,            // 0 → discard; <1 → transparent draw call
  ambient: vec3f,        // PMX ambient rgb
  shininess: f32,        // PMX specular power
  specular: vec3f,       // PMX specular rgb
  sphereMode: f32,       // 0 none · 1 multiply (sph) · 2 add (spa)
  // Skeleton index of the 頭 (head) bone, or -1. Lets the eye shader gate
  // the post-alpha-eye stencil by camera-vs-face hemisphere.
  headBoneIndex: f32,
  // The draw's identity, for the id attachment: which material, which object.
  // They ride HERE, in padding this struct already carried, so the buffer's
  // size and layout are untouched and the indirect-draw path keeps working —
  // the ids arrive with the same per-draw uniform the material already binds,
  // rather than needing a channel of their own. Zero while ids are off, and
  // zero is the reserved "nothing" value, so reading them then is not wrong,
  // just empty. f32 because the buffer is written as floats; the shader casts.
  materialId: f32,
  objectId: f32,
  _pad2: f32,
};

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) normal: vec3f,
  @location(1) uv: vec2f,
  @location(2) worldPos: vec3f,
  // Bind-pose object-space position (the raw pre-skin vertex attribute). Procedural
  // textures (noise bump, sparkle, Generated-coord gradients) key off this instead of
  // worldPos so the pattern rides with the surface — otherwise the mesh swims through a
  // world-static noise field under any skinning deformation or root (センター) motion.
  // At rest skinMats are identity so restPos == worldPos, which is why existing noise-
  // scale constants stay valid without retuning.
  @location(3) restPos: vec3f,
};

struct LightVP { viewProj: mat4x4f, };

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> light: LightUniforms;
@group(0) @binding(2) var diffuseSampler: sampler;
@group(0) @binding(3) var shadowMap: texture_depth_2d;
@group(0) @binding(4) var shadowSampler: sampler_comparison;
@group(0) @binding(5) var<uniform> lightVP: LightVP;
// binding(9) brdfLut is declared inside NODES_WGSL (nodes.ts).
@group(1) @binding(0) var<storage, read> skinMats: array<mat4x4f>;
@group(2) @binding(0) var diffuseTexture: texture_2d<f32>;
@group(2) @binding(1) var<uniform> material: MaterialUniforms;
// Reserved for future sphere/toon graph nodes; graphs that don't read them get the
// 1×1 white fallback bound here.
@group(2) @binding(2) var toonTexture: texture_2d<f32>;
@group(2) @binding(3) var sphereTexture: texture_2d<f32>;
// Extra maps supplied by the STYLE GROUP rather than by the PMX — the lightmap /
// ILM / ramp textures a Blender-authored look is built on. A PMX material carries
// exactly one image, and this whole family of shading encodes shadow thresholds,
// specular masks and material IDs in the channels of a second and third. Unset
// slots get the 1×1 white fallback, so a graph reading one that was never
// supplied sees white rather than garbage.
@group(2) @binding(5) var groupTexture0: texture_2d<f32>;
@group(2) @binding(6) var groupTexture1: texture_2d<f32>;
@group(2) @binding(7) var groupTexture2: texture_2d<f32>;
@group(2) @binding(8) var groupTexture3: texture_2d<f32>;

// Four-bone blended normals can cancel to ~zero on physics-driven parts
// (opposing bone rotations at 50/50 weights) — normalize(0) is 0/0 = NaN,
// which poisons the whole shading stack and flashes through bloom. Fall
// back to up for degenerate normals instead.
fn safe_normal(nIn: vec3f) -> vec3f {
  let l2 = dot(nIn, nIn);
  if (l2 < 1e-12) { return vec3f(0.0, 1.0, 0.0); }
  return nIn * inverseSqrt(l2);
}

`;

// ─── Shadow sampler (3×3 PCF) ───────────────────────────────────────
// 4096-map (MUST match Engine.SHADOW_MAP_SIZE), normal-bias 0.08, depth-bias
// 0.001. Unrolled — Safari's Metal backend doesn't unroll nested shadow loops
// reliably. The texel size was stale at 1/2048 after the map grew to 4096:
// PCF taps landed TWO texels apart, quantizing self-shadow edges into
// texel-sized gray/black squares that crawled with the animation.

export const SAMPLE_SHADOW_WGSL = /* wgsl */ `

fn sampleShadow(worldPos: vec3f, n: vec3f) -> f32 {
  if (dot(n, -light.lights[0].direction.xyz) <= 0.0) { return 0.0; }
  let biasedPos = worldPos + n * 0.08;
  let lclip = lightVP.viewProj * vec4f(biasedPos, 1.0);
  let ndc = lclip.xyz / max(lclip.w, 1e-6);
  let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let cmpZ = ndc.z - 0.001;
  let ts = 1.0 / 4096.0;
  let s00 = textureSampleCompareLevel(shadowMap, shadowSampler, suv + vec2f(-ts, -ts), cmpZ);
  let s10 = textureSampleCompareLevel(shadowMap, shadowSampler, suv + vec2f(0.0, -ts), cmpZ);
  let s20 = textureSampleCompareLevel(shadowMap, shadowSampler, suv + vec2f( ts, -ts), cmpZ);
  let s01 = textureSampleCompareLevel(shadowMap, shadowSampler, suv + vec2f(-ts, 0.0), cmpZ);
  let s11 = textureSampleCompareLevel(shadowMap, shadowSampler, suv, cmpZ);
  let s21 = textureSampleCompareLevel(shadowMap, shadowSampler, suv + vec2f( ts, 0.0), cmpZ);
  let s02 = textureSampleCompareLevel(shadowMap, shadowSampler, suv + vec2f(-ts,  ts), cmpZ);
  let s12 = textureSampleCompareLevel(shadowMap, shadowSampler, suv + vec2f(0.0,  ts), cmpZ);
  let s22 = textureSampleCompareLevel(shadowMap, shadowSampler, suv + vec2f( ts,  ts), cmpZ);
  return (s00 + s10 + s20 + s01 + s11 + s21 + s02 + s12 + s22) * (1.0 / 9.0);
}

`;

// ─── Skinning vertex shader ─────────────────────────────────────────
// Four-bone linear blend skinning. Renormalizes weights when they don't sum to 1
// (PMX models occasionally ship with unnormalized weights on extras like hair tips).
// VS normalize on the outgoing normal is skipped — interpolation denormalizes it
// anyway and every fragment shader does `normalize(input.normal)` as its first line.

export const COMMON_VS_WGSL = /* wgsl */ `

@vertex fn vs(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) joints0: vec4<u32>,
  @location(4) weights0: vec4<f32>
) -> VertexOutput {
  var output: VertexOutput;
  let pos4 = vec4f(position, 1.0);
  let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
  let invWeightSum = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
  let nw = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);
  var skinnedPos = vec4f(0.0);
  var skinnedNrm = vec3f(0.0);
  for (var i = 0u; i < 4u; i++) {
    let m = skinMats[joints0[i]];
    let w = nw[i];
    skinnedPos += (m * pos4) * w;
    skinnedNrm += (mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz) * normal) * w;
  }
  output.position = camera.projection * camera.view * vec4f(skinnedPos.xyz, 1.0);
  output.normal = skinnedNrm;
  output.uv = uv;
  output.worldPos = skinnedPos.xyz;
  output.restPos = position;
  return output;
}

`;

// ─── FS output struct ───────────────────────────────────────────────
// Location 0: final radiance+alpha (blended into rg11b10ufloat; the HDR target
// has no alpha channel, but the blend equation still uses the .a you write here
// as the src-alpha factor that premultiplies rgb into the HDR target).
// Location 1: auxiliary rg8unorm carrying
//   .r = bloom mask (1 = contributes to bloom, 0 = skip — e.g. ground).
//   .g = accumulated canvas alpha — the channel that used to live in hdr.a
//        before the switch to rg11b10ufloat. Sampled by composite to
//        un-premultiply color for tonemap and to set the final drawable alpha
//        (needed for the `premultiplied` canvas alphaMode that blends the
//        WebGPU surface over the page background).
// FS output at location 1 must be vec4f — the blend state references src.a, and
// WebGPU requires the fragment output to provide an alpha component even though
// the rg8unorm target only stores .r and .g (extra components are discarded).
// Materials write mask = vec4f(1.0, 1.0, 0.0, color.a); ground writes
// vec4f(0.0, 1.0, 0.0, edgeFade). With src.a coming from the 4th component and
// src-alpha blending enabled:
//   out.r = mask_r · src.a + dst.r · (1-src.a)   (bloom mask, weighted by alpha)
//   out.g = 1.0    · src.a + dst.g · (1-src.a)   (canonical premultiplied alpha-over)

// The struct itself comes from scene-contract, which owns what the scene pass's
// attachments ARE — this is one of the two shaders that gains an output when
// one is added, and the graph generator emits against this same declaration.
//
// A FUNCTION, and no longer part of the prelude constant below, because the id
// attachment is a device capability probed at init: a struct baked at import
// time would be built before the answer exists. The graph appends it where the
// constant used to end, so the assembled module is unchanged.
export function commonFsOutWgsl(): string {
  return `\n\n${sceneFsOutWgsl()}\n`;
}

// ─── Convenience: full shared prelude ───────────────────────────────
// Material files compose this as `${NODES_WGSL}${COMMON_MATERIAL_PRELUDE_WGSL}` to
// pull in everything structural. Each material then adds its own constants + fs().

// The FSOut struct is NOT in here any more — see commonFsOutWgsl above. Every
// consumer of this constant appends it immediately, which is where it was.
export const COMMON_MATERIAL_PRELUDE_WGSL =
  COMMON_BINDINGS_WGSL + lightsApi(0, 6) + SAMPLE_SHADOW_WGSL + COMMON_VS_WGSL
