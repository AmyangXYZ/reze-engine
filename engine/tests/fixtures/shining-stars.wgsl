#layer additive
// Tunables — edit a value and hit ⌘⏎ to see it live.
const TINT = vec3f(1.0, 0.96, 0.88);  // star color
const DENSITY = 0.5;                  // 0..1 — how crowded the sky is
const TWINKLE = 0.5;                  // 0..1 — flicker speed
const INTENSITY = 1.2;                // overall brightness

fn bgHash2(p: vec2f) -> vec2f {
  let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(q) * 43758.5453);
}

fn starLayer(sph: vec2f, scale: f32, thresh: f32, time: f32, speed: f32) -> f32 {
  let cell = floor(sph * scale);
  let h = bgHash2(cell);
  // Star sits at a hashed point inside its cell — kills the grid look.
  let local = fract(sph * scale) - (0.15 + 0.7 * h);
  let d = length(local);
  let bright = bgHash2(cell + 7.31).x;
  if (bright < thresh) { return 0.0; }
  // Per-star period from its own hash: some stars breathe over seconds, some blink quickly
  let period = 0.25 + 2.75 * bgHash2(cell + 31.7).y;
  let tw = 0.55 + 0.45 * sin(time * speed * period + h.x * 40.0);
  let size = (bright - thresh) / (1.0 - thresh);
  // Tight core + faint halo — a hard point of light, not a blurry blob.
  let r = 0.03 + 0.045 * size;
  let core = smoothstep(r, r * 0.25, d);
  let halo = smoothstep(r * 4.0, r, d) * 0.18;
  return (core + halo) * tw * (0.45 + 0.75 * size);
}

fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f {
  let d = normalize(ray);
  // Latitude/longitude projection
  let sph = vec2f(atan2(d.x, d.z) * 1.2, asin(clamp(d.y, -1.0, 1.0)) * 1.5);
  // Denser = MORE stars, not smaller ones (sizes untouched)
  let dens = mix(0.95, 0.82, clamp(DENSITY, 0.0, 1.0));
  var s = starLayer(sph, 30.0, dens, time, TWINKLE * 1.6);
  // Faint dust layer: denser, smaller, slower.
  s += 0.4 * starLayer(sph + 3.7, 70.0, mix(0.985, 0.9, clamp(DENSITY, 0.0, 1.0)), time, TWINKLE * 0.9);
  let a = clamp(s * INTENSITY, 0.0, 1.0);
  // Straight alpha — the engine's over-composite premultiplies.
  return vec4f(TINT, a);
}
