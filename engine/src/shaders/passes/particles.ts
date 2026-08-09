// Stateless one-shot particle pass: every particle's position at time t is a
// pure function of its static per-instance attributes — no compute, no
// ping-pong buffers, nothing to update per frame but one time uniform.
// Draws additively into the HDR target inside the main pass (after the
// transparent phase), writing bloom mask + alpha so the sparks glow through
// the existing pyramid and survive the composite.

export const PARTICLES_SHADER_WGSL = /* wgsl */ `
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _p: f32, };
struct EmissionUniforms {
  color: vec3f, size: f32,
  centroidPos: vec3f, mode: f32,      // 0 = burst (point -> scatter), 1 = converge (scatter -> point)
  startTime: f32, duration: f32, lift: f32, swirl: f32,
};
struct TimeUniforms { now: f32, };
@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> emission: EmissionUniforms;
@group(0) @binding(2) var<uniform> timeU: TimeUniforms;

struct VSOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) fade: f32,
};

fn easeOutCubic(t: f32) -> f32 { let u = 1.0 - t; return 1.0 - u * u * u; }
fn easeInOut(t: f32) -> f32 { return t * t * (3.0 - 2.0 * t); }

@vertex
fn vs_main(
  @builtin(vertex_index) vi: u32,
  @location(0) point: vec3f,       // mesh anchor (spawn for burst, target for converge)
  @location(1) scatter: vec3f,     // pre-scaled scatter offset at the far end
  @location(2) rand: vec4f,        // delay, lifeScale, seed, sizeScale
) -> VSOut {
  var o: VSOut;
  let elapsed = timeU.now - emission.startTime;
  let life = max(emission.duration * rand.y, 1e-3);
  let t = clamp((elapsed - rand.x) / life, 0.0, 1.0);

  // Flight progress k runs point->scatter for burst, the reverse for converge;
  // the whole cloud winds around its centroid as it goes.
  var k: f32;
  if (emission.mode < 0.5) { k = easeOutCubic(t); } else { k = 1.0 - easeInOut(t); }
  let base = point + scatter * k + vec3f(0.0, emission.lift * k, 0.0);
  let rel = base - emission.centroidPos;
  let a = emission.swirl * k;
  let c = cos(a);
  let s = sin(a);
  let world0 = emission.centroidPos + vec3f(rel.x * c + rel.z * s, rel.y, -rel.x * s + rel.z * c);

  // Burst dies as it scatters; converge lights up fast and vanishes on arrival
  // (the reveal underneath takes over).
  var alpha: f32;
  if (emission.mode < 0.5) {
    alpha = (1.0 - t) * (1.0 - t);
  } else {
    alpha = min(t * 6.0, 1.0) * (1.0 - smoothstep(0.9, 1.0, t));
  }
  alpha *= 0.7 + 0.3 * sin(rand.z * 251.0 + elapsed * 24.0); // sparkle flicker
  let alive = select(0.0, 1.0, elapsed >= rand.x && t < 1.0);

  let size = emission.size * rand.w * (0.7 + 0.6 * (1.0 - t)) * alive;
  // Camera-facing quad from the view matrix's right/up rows.
  let right = vec3f(camera.view[0][0], camera.view[1][0], camera.view[2][0]);
  let up = vec3f(camera.view[0][1], camera.view[1][1], camera.view[2][1]);
  var corner = vec2f(-1.0, -1.0);
  switch (vi) {
    case 1u, 4u: { corner = vec2f(1.0, -1.0); }
    case 2u, 3u: { corner = vec2f(-1.0, 1.0); }
    case 5u: { corner = vec2f(1.0, 1.0); }
    default: {}
  }
  let world = world0 + (right * corner.x + up * corner.y) * size;
  o.position = camera.projection * camera.view * vec4f(world, 1.0);
  o.uv = corner;
  o.fade = alpha;
  return o;
}

struct FSOut {
  @location(0) color: vec4f,
  @location(1) mask: vec2f, // r = bloom mask, g = scene alpha (HDR has none)
};

@fragment
fn fs_main(i: VSOut) -> FSOut {
  let d2 = dot(i.uv, i.uv);
  if (d2 > 1.0) { discard; }
  var o: FSOut;
  // Quadratic falloff to EXACT zero at the rim (a gaussian alone leaves a
  // visible cutoff ring at the quad edge) with a hot bloom-feeding center.
  let t = 1.0 - d2;
  let core = t * t * (0.3 + 0.7 * exp(-d2 * 6.0));
  let w = core * i.fade;
  o.color = vec4f(emission.color * w, 0.0);
  o.mask = vec2f(w, w);
  return o;
}
`
