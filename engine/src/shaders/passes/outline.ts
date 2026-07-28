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

export const OUTLINE_SHADER_WGSL = /* wgsl */ `
struct CameraUniforms {
  view: mat4x4f,
  projection: mat4x4f,
  viewPos: vec3f,
  // Render-target height in device pixels (engine writes it each frame);
  // width is recovered via the projection matrix's aspect.
  viewportHeight: f32,
};

struct MaterialUniforms {
  edgeColor: vec4f,
  edgeSize: f32,
  _padding1: f32,
  _padding2: f32,
  _padding3: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var edgeSampler: sampler;
@group(1) @binding(0) var<storage, read> skinMats: array<mat4x4f>;
@group(2) @binding(0) var<uniform> material: MaterialUniforms;
@group(2) @binding(1) var diffuseTexture: texture_2d<f32>;

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
};

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
  let normalizedWeights = select(vec4f(1.0, 0.0, 0.0, 0.0), weights0 * invWeightSum, weightSum > 0.0001);

  var skinnedPos = vec4f(0.0, 0.0, 0.0, 0.0);
  var skinnedNrm = vec3f(0.0, 0.0, 0.0);
  for (var i = 0u; i < 4u; i++) {
    let j = joints0[i];
    let w = normalizedWeights[i];
    let m = skinMats[j];
    skinnedPos += (m * pos4) * w;
    let r3 = mat3x3f(m[0].xyz, m[1].xyz, m[2].xyz);
    skinnedNrm += (r3 * normal) * w;
  }
  let worldPos = skinnedPos.xyz;
  let worldNormal = normalize(skinnedNrm);

  let clipPos = camera.projection * camera.view * vec4f(worldPos, 1.0);

  // babylon-mmd: screenNormal = normalize((view * worldNormal).xy)
  let viewNormal = (camera.view * vec4f(worldNormal, 0.0)).xyz;
  let snLen = length(viewNormal.xy);
  let screenNormal = select(vec2f(0.0, 0.0), viewNormal.xy / snLen, snLen > 1e-5);

  // Reference-height normalization (babylon-mmd ships this variant commented
  // out as \`renderHeight = 1080\`): thickness is a constant FRACTION of the
  // frame — 2·edgeSize px at 1080p — so retina DPR and 4K export don't thin
  // the rims to sub-pixel. Width follows the projection aspect.
  // projection[1][1]/projection[0][0] = width/height for a symmetric frustum.
  let aspect = camera.projection[1][1] / camera.projection[0][0];
  let viewport = vec2f(1080.0 * aspect, 1080.0);

  // NDC offset = edgeSize · 4/viewport, ×w so the perspective divide cancels:
  // constant screen thickness at any distance (babylon-mmd parity).
  let offset = screenNormal * (material.edgeSize * 4.0 / viewport) * clipPos.w;
  output.position = vec4f(clipPos.xy + offset, clipPos.z, clipPos.w);
  output.uv = uv;
  return output;
}

struct FSOut { @location(0) color: vec4f, @location(1) mask: vec4f };
@fragment fn fs(input: VertexOutput) -> FSOut {
  // Rim alpha FOLLOWS the fabric's texture alpha instead of a hard alpha test:
  // MMD draws blend-material edges solid (only cutout materials alpha-test), so
  // a 0.4 discard erased the whole hull on semi-transparent cloth — stockinged
  // legs crossing lost their outline entirely. Modulating instead keeps a
  // proportional rim on sheer weave (never a solid black hull) and still
  // discards true cut-out margins like hair-card borders.
  let texA = textureSample(diffuseTexture, edgeSampler, input.uv).a;
  if (texA < 0.05) {
    discard;
  }
  var out: FSOut;
  out.color = vec4f(material.edgeColor.rgb, material.edgeColor.a * texA);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}
`
