// The mirror camera, as arithmetic — pure and headlessly testable, the
// shadow-cascades precedent.
//
// A planar reflection is not a second camera aimed by hand; it is the SAME
// camera with the world reflected about the mirror's plane. Fold the reflection
// into the view matrix and everything downstream is untouched: world positions
// stay TRUE world positions, so sun, shadows and positional lights evaluate at
// the unmirrored point — which is exactly what a mirror shows, an object lit
// as it is, seen from a mirrored eye. The only other value that must mirror is
// the eye itself, because specular reads the view direction from it.
//
// The plane is carried as (n, d) with n a UNIT normal and the surface at
// n·p + d = 0 — general from the start, because a mirror standing on the floor
// is the ordinary case and a floor is only the one plane that happens to be
// horizontal. The ground is (0, 1, 0, 0).
//
// WINDING: a reflection has determinant -1, so triangle orientation flips, and
// any pipeline that culls a face culls the WRONG one in the mirror pass.
//
// This note used to claim every scene-pass pipeline culls "none". That was
// never true, and each exception cost a round of "the mirror looks broken":
//
//   ground   culls "back"  -> the floor vanished whole, and with it the grid
//                             and the received shadow, since the shadow is a
//                             layer of the ground shader
//   outline  culls "back"  -> no ink line on the reflection at all
//   eye      culls "front" -> the back of the eyeball kept, its front discarded
//
// All three now build a mirror twin with the cull answered (the ground goes to
// "none", the outline to "front", the eye to "back"). Anything ADDED to the
// scene pass with a cullMode must do the same — grep cullMode in engine.ts,
// which is the whole list.

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
 * A plane as the mirror wants it: 4 numbers, (nx, ny, nz, d), surface at
 * n·p + d = 0.
 *
 * Written into a caller-owned array rather than returned, because the engine
 * holds one per mirror and rebuilds it whenever the mirror moves — which is
 * every frame a gizmo is being dragged.
 *
 * `n` is normalized here rather than trusted: it arrives from a rotated quad's
 * basis, and a scale baked into that basis would otherwise scale `d` against
 * it and put the plane somewhere else entirely.
 */
export function planeFromPointNormal(
  px: number, py: number, pz: number,
  nx: number, ny: number, nz: number,
  out: Float32Array,
): Float32Array {
  const len = Math.hypot(nx, ny, nz) || 1
  const x = nx / len
  const y = ny / len
  const z = nz / len
  out[0] = x
  out[1] = y
  out[2] = z
  // d places the plane through the point: n·p + d = 0.
  out[3] = -(x * px + y * py + z * pz)
  return out
}

/**
 * The Householder reflection about (n, d), column-major.
 *
 *   p' = p - 2 (n·p + d) n
 */
export function reflectionAboutPlane(plane: ArrayLike<number>, out?: Float32Array): Float32Array {
  const nx = plane[0], ny = plane[1], nz = plane[2], d = plane[3]
  const m = out ?? new Float32Array(16)
  m[0] = 1 - 2 * nx * nx
  m[1] = -2 * ny * nx
  m[2] = -2 * nz * nx
  m[3] = 0
  m[4] = -2 * nx * ny
  m[5] = 1 - 2 * ny * ny
  m[6] = -2 * nz * ny
  m[7] = 0
  m[8] = -2 * nx * nz
  m[9] = -2 * ny * nz
  m[10] = 1 - 2 * nz * nz
  m[11] = 0
  m[12] = -2 * d * nx
  m[13] = -2 * d * ny
  m[14] = -2 * d * nz
  m[15] = 1
  return m
}

/** Scratch for the product below — this module is called once per mirror per
 *  frame and allocating a matrix each time is a garbage-collector visit for
 *  nothing. */
const R = new Float32Array(16)

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
export function buildMirrorCamera(
  camera: Float32Array,
  plane: ArrayLike<number>,
  out: Float32Array,
): Float32Array {
  reflectionAboutPlane(plane, R)
  // view × R, column-major: out.col_j = Σ_k view.col_k · R[j*4 + k].
  for (let j = 0; j < 4; j++) {
    const b0 = R[j * 4 + 0], b1 = R[j * 4 + 1], b2 = R[j * 4 + 2], b3 = R[j * 4 + 3]
    for (let r = 0; r < 4; r++) {
      out[j * 4 + r] =
        camera[0 + r] * b0 + camera[4 + r] * b1 + camera[8 + r] * b2 + camera[12 + r] * b3
    }
  }
  for (let i = 16; i < 32; i++) out[i] = camera[i]
  // The eye through the same reflection, so specular reads the mirrored view
  // direction. Written from the plane directly rather than through R: it is one
  // dot product, and going via the matrix would round-trip it through nine
  // multiplies to reach the same number.
  const nx = plane[0], ny = plane[1], nz = plane[2], d = plane[3]
  const t = 2 * (nx * camera[32] + ny * camera[33] + nz * camera[34] + d)
  out[32] = camera[32] - t * nx
  out[33] = camera[33] - t * ny
  out[34] = camera[34] - t * nz
  out[35] = camera[35]
  return out
}
