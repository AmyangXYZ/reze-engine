// MMD-style inverted-hull outline, ported from babylon-mmd's mmdOutline shader
// (the reference implementation whose output matches MMD):
//   1. Extrude along the VIEW-SPACE normal's XY, normalized — a pure screen
//      direction, so rims never smear toward the camera at grazing angles.
//   2. Offset in clip space by edgeSize · 4/viewport · w. The ×w cancels the
//      perspective divide → CONSTANT screen thickness of exactly
//      2·edgeSize device pixels (babylon: `screenNormal / (viewport*0.25) *
//      offset * projectedPosition.w`). PMX edgeSize ~0.3–1.0 ⇒ fine 0.6–2px
//      rims, matching MMD instead of our former chunky constants.
//   3. The FRAGMENT stage samples the material's own diffuse texture and
//      MODULATES the rim's alpha by it (discarding only near-zero cut-out
//      margins) — sheer fabric gets a proportional rim, never a solid black
//      hull, without dropping the author's edge flag.

// A FUNCTION rather than a constant, for the reason commonFsOutWgsl is one: the
// fragment outputs depend on whether the device carries the id attachment, and
// a string baked at import cannot know what the device said.

import { sceneFsOutWgsl, sceneIdPadWgsl } from "./scene-contract"
import { DISSOLVE_WGSL } from "../materials/common"

/**
 * Where `dissolve` sits in the outline's own uniform, in bytes.
 *
 * Exported because the engine writes it and this file declares it, and the two
 * agreeing is not something either can check alone. edgeColor takes 0..16 and
 * edgeSize 16..20, so this is 20.
 */
export const RZ_OUTLINE_DISSOLVE_OFFSET = 20
/** Where the scene-wide width multiplier sits, in bytes — the float after
 *  dissolve (Engine.setOutlineWidth). */
export const RZ_OUTLINE_WIDTH_OFFSET = 24
/** Where the scene-wide colour override sits, in bytes: rgb and a weight, 0
 *  for the material's own edge colour (Engine.setOutlineColor). */
export const RZ_OUTLINE_COLOR_OFFSET = 32
/** Where a style graph's outline (ShaderGraph.outline) sits, in bytes: the
 *  textured flag at 28, then its shadow colour at 48. Unset, a hull draws its
 *  PMX edge colour flat, as MMD does. */
export const RZ_OUTLINE_TEXTURED_OFFSET = 28
export const RZ_OUTLINE_SHADOW_OFFSET = 48
/** The hull's own block, in floats. */
export const RZ_OUTLINE_UNIFORM_FLOATS = 16

export function outlineShaderWgsl(): string {
  return /* wgsl */ `
${DISSOLVE_WGSL}
struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
  viewPos: vec3f,
  // Render-target height in device pixels (engine writes it each frame);
  // width is recovered via the projection matrix's aspect.
  viewportHeight: f32,
};

// THE HULL'S OWN BLOCK, not the material's. 48 bytes of edge data, which is all
// this pass shades with — and dissolve, which it has to obey.
//
// It reached for the material block's own layout first, declaring the skipped
// middle to land dissolve on byte 60. That compiles and then fails validation
// at draw time: the buffer actually bound here is 32 bytes, and a pipeline
// asking for 64 is a pipeline nothing can satisfy. The lesson is the ordinary
// one — the struct is a claim about the BUFFER, and the buffer is built
// somewhere else.
struct MaterialUniforms {
  edgeColor: vec4f,
  edgeSize: f32,
  dissolve: f32,
  /** The scene's width multiplier (setOutlineWidth), the same on every hull:
   *  scales the author's edgeSize rather than replacing it, so a model's thin
   *  and thick lines keep their proportion. */
  widthScale: f32,
  /** 1: a style graph's outline, the game's (Uber pass "Outline"): edgeSize is
   *  a world width, held between 1.2 and 3 px of a 1920-wide frame, and the
   *  colour is the picture × lerp(shadowColor, edgeColor, the key's
   *  half-Lambert). 0: the PMX edge, flat. */
  textured: f32,
  /** The scene's colour in place of edgeColor.rgb, by .a (0 or 1). The
   *  material's own alpha still holds — a line its author made sheer stays so. */
  colorOverride: vec4f,
  shadowColor: vec4f,
};

struct OutlineLight { direction: vec4f, color: vec4f, };
/** The head of the scene's light block: the key is lights[0]. */
struct OutlineLights { ambientColor: vec4f, lights: array<OutlineLight, 4>, };

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var edgeSampler: sampler;
@group(0) @binding(2) var<uniform> light: OutlineLights;
@group(1) @binding(0) var<storage, read> skinMats: array<mat4x4f>;
@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var diffuseTexture: texture_2d<f32>;
/** A look's outline tint (ShaderGraph.outline.tint): one of its images, white
 *  when it names none. */
@group(2) @binding(2) var tintTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  /** BIND-POSE position, for the dissolve — the same value the colour pass and
   *  the depth prepass measure against, so all three agree about which flakes
   *  are gone. */
  @location(1) restPos: vec3f,
  /** The triangle's own threshold, flat — see the material VertexOutput. The
   *  hull traces the body's faces, so it has to lose the same ones. */
  @location(2) @interpolate(flat) faceT: f32,
  /** How much of a pixel-wide line this one really is (1 = all of it): a line
   *  held at the minimum width fades by what it was short. */
  @location(3) coverage: f32,
  @location(4) normal: vec3f,
};

/**
 * Half the frame height (engine units, at the subject) below which a line keeps
 * its constant screen width; wider than that it thins as a world-size line.
 * 12.5 = a frame 25 units tall — a full figure (an MMD model is ~20) with
 * room around her.
 *
 * Not the game's own break (w = 1 game unit = 8 engine units). The game's
 * near width is ~3× MMD's 2·edgeSize px; its world-size line comes down to
 * MMD's width only at a full-figure framing, so that is where the break goes:
 * close-ups and full figures keep today's width, and past that the game's rule
 * holds — NDC offset ∝ 1/w, scaled by the zoom as its projection._m11 scales
 * it. At the game's 8 the demo's own camera (33 units) drew a quarter of
 * today's width, which put every line on the faded minimum.
 */
const RZ_OUTLINE_FULL_FIGURE = 12.5;
/** The thinnest a line is drawn, in pixels at 1080p (and never under this many
 *  device pixels). A line due to be thinner is drawn this wide and faded by
 *  the shortfall, so far lines thin out without aliasing into dashes. */
const RZ_OUTLINE_MIN_PX = 1.5;

fn safe_normal2(v: vec2f) -> vec2f {
  let l = length(v);
  return select(vec2f(0.0), v / l, l > 1e-6);
}

@vertex fn vs(
  @location(0) position: vec3f,
  @location(1) normal: vec3f,
  @location(2) uv: vec2f,
  @location(3) joints0: vec4<u32>,
  @location(4) weights0: vec4<f32>,
  /** xyz: the smoothed rest normal (outline-normals.ts), w: PMX edge scale. */
  @location(5) outlineNormal: vec4f
) -> VertexOutput {
  var output: VertexOutput;
  let pos4 = vec4f(position, 1.0);

  let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
  let invWeightSum = select(1.0, 1.0 / weightSum, weightSum > 0.0001);
  let normalizedWeights = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);

  var skinnedPos = vec4f(0.0, 0.0, 0.0, 0.0);
  var skinnedNrm = vec3f(0.0, 0.0, 0.0);
  for (var i = 0u; i < 4u; i++) {
    let j = joints0[i];
    let w = normalizedWeights[i];
    let m = skinMats[j];
    skinnedPos += (m * pos4) * w;
    let r3 = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
    skinnedNrm += (r3 * outlineNormal.xyz) * w;
  }
  let worldPos = skinnedPos.xyz;
  let worldNormal = normalize(skinnedNrm);

  let clipPos = camera.projection * camera.view * vec4f(worldPos, 1.0);

  // The push direction is the view-space normal's XY, NOT renormalized — the
  // game's: full length at the silhouette, where the normal lies across the
  // view, and shrinking on surfaces turned toward the camera, so creases
  // facing her draw finer than the rim. Skinned exactly as the colour pass
  // skins the normal, from the smoothed rest normal, so every vertex sharing a
  // position pushes the same way and the hull stays closed over hard edges
  // and UV seams.
  let viewNormal = (camera.view * vec4f(worldNormal, 0.0)).xyz;

  // Reference-height normalization (babylon-mmd ships this variant commented
  // out as \`renderHeight = 1080\`): the near width is a constant FRACTION of
  // the frame — 2·edgeSize px at 1080p — so retina DPR and 4K export don't
  // thin the rims to sub-pixel. Width follows the projection aspect.
  // projection[1][1]/projection[0][0] = width/height for a symmetric frustum.
  let aspect = camera.projection[1][1] / camera.projection[0][0];
  let refViewport = vec2f(1080.0 * aspect, 1080.0);
  let deviceViewport = vec2f(camera.viewportHeight * aspect, camera.viewportHeight);

  // The game's width (Character/Debug.shader, pass "Outline"): the clip offset
  // is the near offset × min(w, break depth). Up close the ×w cancels the
  // perspective divide — constant screen width; past the break it stops
  // growing, so the NDC offset falls as 1/w like a line of fixed world width.
  // The break is the depth at which the frame is a full figure tall, so a
  // zoomed lens moves it out (see RZ_OUTLINE_FULL_FIGURE). Under an
  // orthographic camera w is 1 and the same min() makes the line world-size
  // once the frame is taller than a figure. The PMX per-vertex edge scale
  // multiplies it all, as vertex colour alpha does in the game.
  let w = max(clipPos.w, 1e-4);
  var ndc: vec2f;
  var coverage = 1.0;
  if (material.textured > 0.5) {
    // The game's: a view-space offset of edgeSize world units along the view
    // normal, projected, then each axis held between 1.2 px and 1.2 px ×
    // (1 + _OutlineMaxOffsetMultiplier, 1.5) of a 1920-wide frame; a line the
    // minimum had to push fades by how far it was pushed.
    let off = viewNormal.xy * material.edgeSize * material.widthScale * outlineNormal.w;
    let d1 = vec2f(off.x * camera.projection[0][0], off.y * camera.projection[1][1]) / w;
    // the minimum along the push direction (the game's dir · 2/screen · 1.2)
    let pxNdc = 2.0 / refViewport * 1.2 * safe_normal2(viewNormal.xy);
    let mx = abs(pxNdc) * 2.5;
    let o = min(mx, max(abs(d1), abs(pxNdc)));
    // fadeOut = min(length(saturate(off − |d1|))·75, 1), on clip offsets (× w)
    // in game units: w here is MMD units, 8 to the game's metre
    coverage = 1.0 - min(length(saturate((o - abs(d1)) * w / 8.0)) * 75.0, 1.0);
    ndc = o * select(vec2f(1.0), sign(d1), abs(d1) > vec2f(0.0));
  } else {
    let breakDepth = RZ_OUTLINE_FULL_FIGURE * camera.projection[1][1];
    let nearNdc = viewNormal.xy * (material.edgeSize * material.widthScale * outlineNormal.w * 4.0 / refViewport);
    ndc = nearNdc * (min(w, breakDepth) / w);

    // Antialiased minimum (the game's _OutlineAntialias): held at the minimum
    // width and faded by the shortfall, never thinner. The game fades by
    // sqrt(coverage.x · coverage.y), per axis; for one width that is the ratio
    // itself.
    let px = length(ndc * deviceViewport * 0.5);
    let minPx = RZ_OUTLINE_MIN_PX * max(1.0, camera.viewportHeight / 1080.0);
    if (px < minPx) {
      coverage = px / minPx;
      ndc = select(vec2f(0.0), ndc * (minPx / px), px > 1e-6);
    }
  }
  output.normal = worldNormal;
  output.coverage = coverage;
  output.position = vec4f(clipPos.xy + ndc * w, clipPos.z, clipPos.w);
  output.uv = uv;
  output.restPos = position;
  output.faceT = rz_dissolve_threshold(position);
  return output;
}

// The id output is declared and padded, never written for real: a hull would
// overwrite the id of the body it traces, so the pipeline takes the id target at
// writeMask 0. Declared all the same — see sceneFsOutWgsl.
${sceneFsOutWgsl({ name: "FSOut", aux: "mask" })}
@fragment fn fs(input: VertexOutput) -> FSOut {
  // Rim alpha FOLLOWS the fabric's texture alpha instead of a hard alpha test:
  // MMD draws blend-material edges solid (only cutout materials alpha-test), so
  // a 0.4 discard erased the whole hull on semi-transparent cloth — stockinged
  // legs crossing lost their outline entirely. Modulating instead keeps a
  // proportional rim on sheer weave (never a solid black hull) and still
  // discards true cut-out margins like hair-card borders.
  // THE HULL GOES WITH THE SURFACE IT TRACES.
  //
  // Without this the outline pass was the one pass that ignored the dissolve:
  // the body discarded flake by flake and its hulls stayed, so a character who
  // had gone left a solid silhouette in her own edge colour standing where she
  // had been. Only models whose materials carry MMD's edge flag showed it,
  // which is what made it look like some models "would not dissolve".
  //
  // The same test the colour pass and the depth prepass run, against the same
  // bind-pose position — three passes, one rule, or they disagree about which
  // pieces are still there.
  if (material.dissolve < 0.9995 && input.faceT > material.dissolve) { discard; }
  let tex = textureSample(diffuseTexture, edgeSampler, input.uv);
  let texA = tex.a;
  if (texA < 0.05) {
    discard;
  }
  var out: FSOut;
  // Blended, not alpha-to-coverage: this pass already blends (the texture
  // alpha above), it is drawn right after the surface it traces so what lies
  // under a faded line is already there, and 4× MSAA would quantize a fade to
  // four steps.
  var rgb = mix(material.edgeColor.rgb, material.colorOverride.rgb, material.colorOverride.a);
  if (material.textured > 0.5) {
    let lit = dot(-light.lights[0].direction.xyz, normalize(input.normal)) * 0.5 + 0.5;
    let tint = textureSample(tintTexture, edgeSampler, input.uv).rgb;
    rgb = tex.rgb * tint * mix(material.shadowColor.rgb, rgb, lit);
  }
  out.color = vec4f(rgb, material.edgeColor.a * texA * input.coverage);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
${sceneIdPadWgsl("out")}  return out;
}
`
}
