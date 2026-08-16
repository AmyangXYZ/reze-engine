// The mirror camera, as arithmetic — pure and headlessly testable, the
// shadow-cascades precedent.
//
// A planar reflection is not a second camera aimed by hand; it is the SAME
// camera with the world reflected about the floor plane. Fold the reflection
// into the view matrix and everything downstream is untouched: world positions
// stay TRUE world positions, so sun, shadows and positional lights evaluate at
// the unmirrored point — which is exactly what a mirror shows, an object lit
// as it is, seen from a mirrored eye. The only other value that must mirror is
// the eye itself, because specular reads the view direction from it.
//
// Winding: a reflection has determinant -1, so triangle orientation flips.
// Every scene-pass pipeline that draws into the mirror culls "none", which is
// what makes this legal without a flipped-frontFace pipeline set. The OUTLINE
// culls "back" and is therefore skipped in the mirror — its hull would face
// the wrong way and ink over the model.

/**
 * The debug view: the reflection target drawn over the finished frame — the
 * only way to SEE whether the mirror pass is right before anything consumes
 * it, the same instrument discipline as setIdDebug. The target is HDR linear;
 * a Reinhard fold plus a square-root keeps highlights readable without
 * involving the real view transform, which a diagnostic does not need.
 */
export const REFLECTION_DEBUG_WGSL = /* wgsl */ `
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f, };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VSOut {
  var out: VSOut;
  let x = f32(i32(i / 2u) * 4 - 1);
  let y = f32(i32(i % 2u) * 4 - 1);
  out.pos = vec4f(x, y, 0.0, 1.0);
  out.uv = vec2f(x * 0.5 + 0.5, 0.5 - y * 0.5);
  return out;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let c = textureSample(t, s, in.uv).rgb;
  return vec4f(sqrt(c / (vec3f(1.0) + c)), 1.0);
}
`

/**
 * Reflection about the horizontal plane y = h, column-major.
 *
 *   p' = (x, 2h - y, z)
 */
export function reflectionAboutY(h: number): Float32Array {
  // prettier-ignore
  return new Float32Array([
    1, 0, 0, 0,
    0, -1, 0, 0,
    0, 0, 1, 0,
    0, 2 * h, 0, 1,
  ])
}

/**
 * Fill a camera-uniform block for the mirror pass from the live one.
 *
 * Layout is the material CameraUniforms: view at 0, projection at 16, eye at
 * 32, render-target height at 35 — the same 36 floats the main camera writes,
 * copied rather than re-derived so the two cannot disagree about anything but
 * the reflection.
 *
 * view' = view × R (column-vector convention, matching `projection * view *
 * pos` in the vertex shaders); projection unchanged; eye reflected.
 */
export function buildMirrorCamera(camera: Float32Array, planeY: number, out: Float32Array): Float32Array {
  const v = camera
  const h = planeY
  // view × R where R = reflectionAboutY(h). R only touches column 1 (scaled by
  // -1) and adds 2h·col1 to the translation — write the product directly
  // rather than through a generic multiply, so the arithmetic is exact and the
  // cost is a handful of ops.
  for (let i = 0; i < 16; i++) out[i] = v[i]
  out[4] = -v[4]
  out[5] = -v[5]
  out[6] = -v[6]
  out[7] = -v[7]
  out[12] = v[12] + 2 * h * v[4]
  out[13] = v[13] + 2 * h * v[5]
  out[14] = v[14] + 2 * h * v[6]
  out[15] = v[15] + 2 * h * v[7]
  for (let i = 16; i < 32; i++) out[i] = v[i]
  out[32] = v[32]
  out[33] = 2 * h - v[33]
  out[34] = v[34]
  out[35] = v[35]
  return out
}
