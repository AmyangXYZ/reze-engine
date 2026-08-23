import { sceneFsOutWgsl } from "../passes/scene-contract"
import { lightsApi } from "../lights"
import { SHADOW_CASCADES } from "../../shadow-cascades"
import { WORLD_AMBIENT_WGSL } from "../lights"

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
  /** Irradiance SH of the HDRI world; [0].w is the on-flag. */
  sh: array<vec4f, 9>,
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
  // How much of this material is still THERE: 1 whole, 0 gone. Rides the last
  // of the padding this struct already carried, for the same reason the ids do
  // — the buffer's size and layout are untouched, so the indirect-draw path and
  // every existing pipeline keep working, and a material that never dissolves
  // pays one float it was already paying.
  //
  // A material morph rebuilds this block from its base copy, which is why the
  // engine writes the value into that copy as well as into the live buffer: a
  // face morphing while she dissolves must not come back solid for those frames.
  dissolve: f32,
};

struct VertexOutput {
  // @invariant: the opaque depth prepass rasterises this same skinned position
  // through a DIFFERENT shader module, and the colour pass then depth-tests
  // less-equal against what it wrote. Without invariance a backend is free to
  // optimise the two position computations differently, and a one-ulp
  // disagreement is a pixel of missing character. Invariance pins both to the
  // same result; it costs only that freedom.
  @builtin(position) @invariant position: vec4f,
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

// One view-projection per shadow cascade, inner to outer — the volumes built
// by shadow-cascades.ts, in the same order.
struct LightVP { viewProj: array<mat4x4f, ${SHADOW_CASCADES.length}>, };

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> light: LightUniforms;
@group(0) @binding(2) var diffuseSampler: sampler;
@group(0) @binding(3) var shadowMap: texture_depth_2d;
@group(0) @binding(4) var shadowSampler: sampler_comparison;
@group(0) @binding(5) var<uniform> lightVP: LightVP;
// The far cascade's map — coarser texels over a much wider box, so the stage
// keeps its shadows when the crisp near volume ends. Binding 7: 6 is the
// positional lights, 9 the BRDF LUT.
@group(0) @binding(7) var shadowMapFar: texture_depth_2d;
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

// ─── Shadow sampler (3×3 PCF, cascade-selected) ─────────────────────
// Normal-bias 0.08, depth-bias 0.001 NDC. Unrolled — Safari's Metal backend
// doesn't unroll nested shadow loops reliably. Texel sizes are interpolated
// from SHADOW_CASCADES so they cannot go stale the way the hardcoded 1/2048
// once did (PCF taps landed TWO texels apart after the map grew to 4096,
// quantizing self-shadow edges into crawling gray squares).
//
// Selection: the crisp near cascade wherever its box covers the point — with a
// margin that keeps the whole PCF kernel inside the map, so selection never
// mixes maps mid-kernel — else the far cascade, else LIT. Lit is a fix, not
// merely a default: the old single-volume path clamped, and past the light's
// far plane the comparison failed against every stored depth, silently
// shadowing any stage deeper than the box.

const pcf9 = (map: string, ts: string) => /* wgsl */ `
  let suv = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
  let cmpZ = ndc.z - 0.001;
  let ts = ${ts};
  let s00 = textureSampleCompareLevel(${map}, shadowSampler, suv + vec2f(-ts, -ts), cmpZ);
  let s10 = textureSampleCompareLevel(${map}, shadowSampler, suv + vec2f(0.0, -ts), cmpZ);
  let s20 = textureSampleCompareLevel(${map}, shadowSampler, suv + vec2f( ts, -ts), cmpZ);
  let s01 = textureSampleCompareLevel(${map}, shadowSampler, suv + vec2f(-ts, 0.0), cmpZ);
  let s11 = textureSampleCompareLevel(${map}, shadowSampler, suv, cmpZ);
  let s21 = textureSampleCompareLevel(${map}, shadowSampler, suv + vec2f( ts, 0.0), cmpZ);
  let s02 = textureSampleCompareLevel(${map}, shadowSampler, suv + vec2f(-ts,  ts), cmpZ);
  let s12 = textureSampleCompareLevel(${map}, shadowSampler, suv + vec2f(0.0,  ts), cmpZ);
  let s22 = textureSampleCompareLevel(${map}, shadowSampler, suv + vec2f( ts,  ts), cmpZ);
  return (s00 + s10 + s20 + s01 + s11 + s21 + s02 + s12 + s22) * (1.0 / 9.0);
`

const SAMPLE_SHADOW_WGSL = /* wgsl */ `

fn sampleShadowNear(ndc: vec3f) -> f32 {
${pcf9("shadowMap", `1.0 / ${SHADOW_CASCADES[0].mapSize}.0`)}
}

fn sampleShadowFar(ndc: vec3f) -> f32 {
${pcf9("shadowMapFar", `1.0 / ${SHADOW_CASCADES[SHADOW_CASCADES.length - 1].mapSize}.0`)}
}

fn sampleShadow(worldPos: vec3f, n: vec3f) -> f32 {
  if (dot(n, -light.lights[0].direction.xyz) <= 0.0) { return 0.0; }
  let biasedPos = worldPos + n * 0.08;
  let c0 = lightVP.viewProj[0] * vec4f(biasedPos, 1.0);
  let n0 = c0.xyz / max(c0.w, 1e-6);
  if (all(abs(n0.xy) < vec2f(0.98)) && n0.z > 0.0 && n0.z < 1.0) {
    return sampleShadowNear(n0);
  }
  let c1 = lightVP.viewProj[1] * vec4f(biasedPos, 1.0);
  let n1 = c1.xyz / max(c1.w, 1e-6);
  if (all(abs(n1.xy) < vec2f(0.98)) && n1.z > 0.0 && n1.z < 1.0) {
    return sampleShadowFar(n1);
  }
  // Outside every cascade there is no occlusion information; lit is the only
  // honest answer.
  return 1.0;
}

`;

// ─── Skinning vertex shader ─────────────────────────────────────────
// Four-bone linear blend skinning. Renormalizes weights when they don't sum to 1
// (PMX models occasionally ship with unnormalized weights on extras like hair tips).
// VS normalize on the outgoing normal is skipped — interpolation denormalizes it
// anyway and every fragment shader does `normalize(input.normal)` as its first line.

const COMMON_VS_WGSL = /* wgsl */ `

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

// ─── Dissolve ───────────────────────────────────────────────────────
/**
 * Whether a fragment survives the dissolve, and how close it is to the front.
 *
 * ONE implementation, shared by the colour pass and the depth prepass, and that
 * is the whole reason it lives here rather than in either of them. The prepass
 * claims depth for fragments the colour pass will shade; if the two disagreed
 * about which flakes are gone, the model would keep writing depth where it no
 * longer draws — holes that occlude the floor behind her and read as sky.
 *
 * OBJECT SPACE, off the bind-pose position, for the same reason the procedural
 * texture nodes use restPos: a world-space field would let the flakes swim
 * through her as she moves, and a screen-space one would leave them hanging in
 * the air while she turns. In object space the pattern is painted ON the
 * surface — the flakes stay where they were on her arm as the arm moves.
 *
 * The hash is value noise rather than a bare hash so the flakes come apart in
 * clumps of a few millimetres rather than as single-pixel snow, which at any
 * distance is just a fade.
 */
export const DISSOLVE_WGSL = /* wgsl */ `
/** Clump size, in object-space units — MMD's are roughly centimetres. */
const RZ_DISSOLVE_GRAIN: f32 = 0.42;
/** How much the clumps pull a fragment off its own grit value — the whole of
 *  the LOOK, and none of the rate. See rz_dissolve_threshold. */
const RZ_DISSOLVE_CLUMP: f32 = 0.6;
/** Grit cell, as a fraction of a clump. See rz_dissolve_threshold: this is what
 *  keeps a material SMALLER than a clump from having one threshold for all of
 *  it, and it has to be smaller than the SMALLEST material a model carries —
 *  an eye highlight, a button, a buckle — not merely small. Two and a half
 *  millimetres on a normal model, which is under all of them. */
const RZ_DISSOLVE_GRIT: f32 = 0.06;
/** How wide the glowing front is, in threshold units. Thin: the front is a
 *  rim, and a wide one lights whole limbs at once. */
const RZ_DISSOLVE_EDGE: f32 = 0.11;
/** What the front burns with. Emission, added after lighting and after the
 *  graph — a colour that ramps or takes a shadow reads as paint, not as heat.
 *  Above 1 on purpose: it is meant to reach the bloom pyramid. Blue-violet,
 *  the same light the motes leaving her are drawn and lit with. */
const RZ_BURN_COLOR: vec3f = vec3f(0.45, 0.62, 2.40);

fn rz_dissolve_hash(p: vec3f) -> f32 {
  // INTEGER, for exactly the reason _hash33 in nodes.ts is, and its comment
  // says it: above 24 bits an f32 loses the low ones. The float hash this
  // replaces multiplied three terms into the hundreds of millions and took
  // fract() of the product — which past 2^24 is not noise, it is a quantised
  // staircase, and wherever that staircase landed on zero the threshold was
  // zero. A surface whose threshold is zero never crosses it: it stays behind
  // while the rest of her leaves, and it sits inside the burning front the
  // whole time she is gone. Those were the white pieces that would not go.
  //
  // Floored first, unlike _hash33: vec3i() truncates toward zero, so -0.5 and
  // +0.5 are the same cell, and half this model is at negative x.
  var h = vec3u(vec3i(floor(p)) + vec3i(32768));
  h = h * vec3u(1664525u, 1013904223u, 2654435761u);
  h = (h.yzx ^ h) * vec3u(2246822519u, 3266489917u, 668265263u);
  h = h ^ (h >> vec3u(16u));
  return f32((h.x ^ h.y ^ h.z) & 16777215u) * (1.0 / 16777216.0);
}

/** Value noise over that hash — smooth within a flake, uncorrelated between. */
fn rz_dissolve_field(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let c000 = rz_dissolve_hash(i);
  let c100 = rz_dissolve_hash(i + vec3f(1.0, 0.0, 0.0));
  let c010 = rz_dissolve_hash(i + vec3f(0.0, 1.0, 0.0));
  let c110 = rz_dissolve_hash(i + vec3f(1.0, 1.0, 0.0));
  let c001 = rz_dissolve_hash(i + vec3f(0.0, 0.0, 1.0));
  let c101 = rz_dissolve_hash(i + vec3f(1.0, 0.0, 1.0));
  let c011 = rz_dissolve_hash(i + vec3f(0.0, 1.0, 1.0));
  let c111 = rz_dissolve_hash(i + vec3f(1.0, 1.0, 1.0));
  let x00 = mix(c000, c100, u.x);
  let x10 = mix(c010, c110, u.x);
  let x01 = mix(c001, c101, u.x);
  let x11 = mix(c011, c111, u.x);
  return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}

/**
 * The threshold this fragment is measured against.
 *
 * NOISE ALONE, so the whole body comes apart at once — a hand, a shoulder and
 * an ankle all thinning together rather than a line travelling up her. It was
 * tilted by height at first, on the argument that a sweep reads as something
 * happening TO her; what it actually reads as is a wipe, and a wipe has a
 * direction, an edge and a speed the rest of the effect knows nothing about.
 * Everywhere at once is also the honest match for what leaves: the motes are
 * shed from every limb in the same breath.
 *
 * TWO SCALES. Clumps alone meant every fragment of a material SMALLER than one
 * cell shared a single threshold: an eye highlight, a brow, a button, a hair
 * clip. All of it sat inside the glowing front at once, so the piece flashed
 * white as a whole and then popped — the two things a disintegration must never
 * do. The grit is a per-cell hash at a fraction of the clump, so a part a few
 * millimetres across still comes apart in pieces.
 *
 * THE GRIT IS THE BASE, though, and the clumps only push a fragment off it.
 * That ordering is what makes the dissolve smooth, and it is not obvious: the
 * two used to be WEIGHTED AND ADDED, and a sum is not uniformly distributed
 * however uniform each half is. It piles up in the middle, so most of the
 * surface had a threshold near a half — all of it went as the dissolve swept
 * through the middle, and what remained was the thin tail near zero. That tail
 * was the second stage everyone could see: a body that mostly vanished, then a
 * scatter of glowing scraps, then nothing.
 *
 * WRAPPED, not added. Offsetting a uniform value and taking fract() leaves it
 * EXACTLY uniform, where scaling a sum back into range only flattens the middle
 * — measured over four hundred thousand samples, the sum removed 12% of her per
 * step at the halfway mark and 0.1% at each end (a 110x swing), the scaled sum
 * still swung 89x, and the wrap is flat: 5.0% per step from the first flake to
 * the last, a ratio of 1.0.
 *
 * That flatness IS the smoothness. An uneven rate is what produced the two
 * stages you could see — a body that mostly vanished at once, then a scatter of
 * scraps that took as long again to follow.
 *
 * The clumping survives the wrap: neighbouring fragments share a clump value,
 * so the offset moves them together. What wraps past 1 comes back at 0 and goes
 * at the other end instead, which costs a little speckle in a region and buys
 * an even rate everywhere.
 */
fn rz_dissolve_threshold(restPos: vec3f) -> f32 {
  let grit = rz_dissolve_hash(floor(restPos / (RZ_DISSOLVE_GRAIN * RZ_DISSOLVE_GRIT)));
  let clump = rz_dissolve_field(restPos / RZ_DISSOLVE_GRAIN) - 0.5;
  return fract(grit + clump * RZ_DISSOLVE_CLUMP);
}
`

// ─── Convenience: full shared prelude ───────────────────────────────
// Material files compose this as `${NODES_WGSL}${COMMON_MATERIAL_PRELUDE_WGSL}` to
// pull in everything structural. Each material then adds its own constants + fs().

// The FSOut struct is NOT in here any more — see commonFsOutWgsl above. Every
// consumer of this constant appends it immediately, which is where it was.
export const COMMON_MATERIAL_PRELUDE_WGSL =
  COMMON_BINDINGS_WGSL + WORLD_AMBIENT_WGSL + lightsApi(0, 6) + SAMPLE_SHADOW_WGSL + COMMON_VS_WGSL
