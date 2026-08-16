import { GROUND_MATERIAL_ID } from "./ground"

/**
 * The id attachment, drawn so a person can look at it.
 *
 * The id buffer is correct or incorrect in ways nothing else in the frame can
 * show: with no consumer, a perfect id buffer and a completely wrong one
 * produce exactly the same picture. This pass exists so that "it works" is
 * something seen rather than inferred.
 *
 * WHAT CORRECT LOOKS LIKE, and what each failure looks like instead:
 *
 *   - Every material is ONE FLAT COLOUR, with hard edges. Ids are names, not
 *     quantities: a gradient anywhere, or fringing along an edge, means
 *     something is interpolating or resolving them — the two failures the
 *     no-resolve/no-blend rules exist to prevent.
 *   - Parts differ from each other. A whole model in one colour means the
 *     material index never made it into the uniform.
 *   - The FLOOR is a fixed light grey, the reserved id, and never shares a
 *     colour with a body part.
 *   - Where nothing was drawn is BLACK — id 0, the reserved nothing, which is
 *     also what the attachment clears to. Black *over geometry* means that draw
 *     wrote no id; colour where the sky should be means the clear is wrong.
 *
 * Colours come from a hash of the pair, so neighbouring materials land on
 * unrelated hues rather than adjacent shades of one — the point is telling them
 * apart, not ordering them.
 */
export const ID_DEBUG_SHADER_WGSL = /* wgsl */ `
// Multisampled and NEVER resolved, so it is read the way it is written: one
// sample, by texel. Sample 0 is what every consumer of this attachment reads.
@group(0) @binding(0) var idTex: texture_multisampled_2d<u32>;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

/** Integer hash → 0..1. Distinct inputs land on unrelated outputs, which is the
 *  whole requirement: adjacent ids must not read as adjacent colours. */
fn hashId(x: u32) -> f32 {
  var h = x * 747796405u + 2891336453u;
  h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
  return f32((h >> 22u) ^ h) / 4294967295.0;
}

@fragment fn fs(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
  let ids = textureLoad(idTex, vec2<i32>(fragCoord.xy), 0);
  let materialId = ids.x;
  let objectId = ids.y;

  // Nothing drawn here. Black, and it must cover exactly the empty space.
  if (materialId == 0u && objectId == 0u) {
    return vec4f(0.0, 0.0, 0.0, 1.0);
  }
  // The floor, at the top of the range. Flat light grey so it is unmistakable
  // and cannot be confused with a hashed colour.
  if (materialId == ${GROUND_MATERIAL_ID}u) {
    return vec4f(0.72, 0.72, 0.75, 1.0);
  }
  // Hue from the material, brightness from the object, so two models wearing
  // the same material index still read apart.
  let h = hashId(materialId * 1973u + 9277u);
  let shade = 0.55 + 0.45 * hashId(objectId * 6151u + 1u);
  // Cheap hue ramp: three offset cosines. Saturated on purpose — this is a
  // diagnostic, not a look.
  let rgb = vec3f(
    0.5 + 0.5 * cos(6.28318 * (h + 0.00)),
    0.5 + 0.5 * cos(6.28318 * (h + 0.33)),
    0.5 + 0.5 * cos(6.28318 * (h + 0.67)),
  );
  return vec4f(rgb * shade, 1.0);
}
`
