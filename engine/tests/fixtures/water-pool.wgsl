// Water Pool — the cast stands chest-deep in still water.
//
// A FOREGROUND plane, evaluated in closed form: the water surface is where the
// pixel's view ray crosses y = WATER_Y, and the scene's own depth decides
// whether the body is in front of that crossing (above the waterline, drawn
// clear) or behind it (underwater, drawn through the water's tint). The foam
// is DEPTH-BASED: any scene surface within a hand's width of the plane gets
// the contact line, which is what draws the ring around her chest and around
// anything else that pierces the surface — no geometry, no particles.
//
// Full res, because the waterline is the one crisp edge this effect owns and
// half-res sampling would shimmer it.
// @fullres

// ── Tuning ───────────────────────────────────────────────────────────────────
const WATER_Y: f32 = 12.0;        // chest height on an ordinary MMD rig
const POOL_RADIUS: f32 = 26.0;    // world units from the origin; soft edge
const EDGE_SOFT: f32 = 3.0;       // how wide the pool's rim fade is
const FOAM_WIDTH: f32 = 0.55;     // world units either side of the waterline
const CLARITY_DEPTH: f32 = 6.0;   // how far down the water stays see-through
const DEEP_COLOR = vec3f(0.06, 0.27, 0.38);   // the pool's own body
const SKY_COLOR = vec3f(0.62, 0.86, 0.95);    // what grazing angles borrow
const FOAM_COLOR = vec3f(0.94, 0.98, 1.0);
const SPARKLE_DIR = vec3f(-0.37, 0.82, -0.44); // a fixed fake sun; the field
                                               // mount has no light uniform
const RIPPLE_SCALE: f32 = 0.55;   // waves per world unit
const RIPPLE_HEIGHT: f32 = 0.22;  // how hard ripples tilt the normal

// ── Ripples ──────────────────────────────────────────────────────────────────

fn vnoise(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(rzHash21(i), rzHash21(i + vec2f(1.0, 0.0)), u.x),
    mix(rzHash21(i + vec2f(0.0, 1.0)), rzHash21(i + vec2f(1.0, 1.0)), u.x),
    u.y,
  );
}

/** Two octaves, scrolled two ways so the pattern never reads as a conveyor. */
fn rippleH(p: vec2f, t: f32) -> f32 {
  let a = vnoise(p * RIPPLE_SCALE + vec2f(t * 0.21, t * 0.13));
  let b = vnoise(p * RIPPLE_SCALE * 2.3 - vec2f(t * 0.17, t * 0.29));
  return a * 0.7 + b * 0.3;
}

/** Surface normal by finite differences — closed form, no marching. */
fn rippleN(p: vec2f, t: f32) -> vec3f {
  let e = 0.35;
  let hx = rippleH(p + vec2f(e, 0.0), t) - rippleH(p - vec2f(e, 0.0), t);
  let hz = rippleH(p + vec2f(0.0, e), t) - rippleH(p - vec2f(0.0, e), t);
  return normalize(vec3f(-hx * RIPPLE_HEIGHT / e, 1.0, -hz * RIPPLE_HEIGHT / e));
}

// ── The surface ──────────────────────────────────────────────────────────────

fn foreground(ray: vec3f, uv: vec2f, time: f32, depth: f32) -> vec4f {
  let ro = rzCameraPos();
  let FAR_CLAMP: f32 = 4000.0;
  // Where the scene put a surface. Past the clamp there is none, and the
  // select keeps that case from pretending to be a wall four kilometres out.
  let p = rzWorldPos(ray, min(depth, FAR_CLAMP));
  let surfT = select(1.0e9, length(p - ro), depth < FAR_CLAMP);

  // Underwater camera: one deliberate wash rather than a broken special case.
  // Distance fogs toward the deep colour; the waterline from below is the
  // ordinary path's business, not this one's.
  if (ro.y < WATER_Y) {
    let fog = clamp(surfT / 40.0, 0.0, 1.0);
    return vec4f(DEEP_COLOR, mix(0.42, 0.85, fog));
  }

  // The waterline wobbles with the same field the surface ripples with, so the
  // foam band on her chest moves with the waves it belongs to.
  let wob = (rippleH(p.xz, time) - 0.5) * 0.5;
  let inPool = 1.0 - smoothstep(POOL_RADIUS - EDGE_SOFT, POOL_RADIUS, length(p.xz));
  // Depth-based contact foam: any surface the scene drew near the plane. This
  // is what rings the body — computed BEFORE the occlusion test, because the
  // band straddles the line and its upper half sits on skin that occludes the
  // water itself.
  var foam = (1.0 - smoothstep(0.0, FOAM_WIDTH, abs(p.y - (WATER_Y + wob)))) * inPool;
  foam = foam * step(surfT, 1.0e8);

  // The crossing. Looking level or up, or the scene in front of it: no water
  // at this pixel — only whatever foam the surface near the line earned.
  if (ray.y >= -1.0e-4) {
    return vec4f(FOAM_COLOR, foam * 0.85);
  }
  let t = (WATER_Y - ro.y) / ray.y;
  if (t >= surfT) {
    return vec4f(FOAM_COLOR, foam * 0.85);
  }

  let hit = ro + ray * t;
  let edge = 1.0 - smoothstep(POOL_RADIUS - EDGE_SOFT, POOL_RADIUS, length(hit.xz));
  if (edge <= 0.0) {
    return vec4f(FOAM_COLOR, foam * 0.85);
  }

  let n = rippleN(hit.xz, time);
  // Grazing angles borrow the sky, steep ones look into the deep — the fresnel
  // shape without the physics bill, which is the house style.
  let fresnel = pow(1.0 - clamp(-ray.y, 0.0, 1.0), 3.0);
  var col = mix(DEEP_COLOR, SKY_COLOR, 0.22 + 0.58 * fresnel);
  // Glints where a ripple faces the fake sun. Bounded by construction: the
  // pow caps at 1 and the weight is fixed, so no shoulder-blowing highlight.
  let glint = pow(max(dot(n, normalize(SPARKLE_DIR)), 0.0), 28.0);
  col = col + vec3f(0.5, 0.55, 0.6) * glint;

  // How far below the surface the thing behind this pixel sits. Shallow shows
  // her legs through the tint; deep goes opaque — CLARITY_DEPTH is the whole
  // dial. A pixel with no scene behind it reads as bottomless.
  let below = clamp((WATER_Y - p.y) / CLARITY_DEPTH, 0.0, 1.0);
  var a = mix(0.34, 0.88, max(fresnel, below));

  // The contact line wins over the surface it sits on.
  col = mix(col, FOAM_COLOR, foam);
  a = max(a, foam * 0.9);

  return vec4f(col, a * edge);
}
