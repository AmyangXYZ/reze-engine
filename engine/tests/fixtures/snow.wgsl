// @particles 8000

// Snow — the first effect drawn as geometry rather than marched per pixel.
//
// Every flake is one instanced quad, so the cost is the flakes and not the
// screen: eight thousand of these cost less than one fullscreen effect that
// touches every pixel to decide most of them are empty. The old path could not
// have afforded a tenth as many, and the count is the whole difference between
// "it is snowing" and a few dots drifting past.
//
// The volume is FIXED TO THE STAGE, not to the camera.
//
// Following the eye seemed right — weather is everywhere, so keep it where you
// are looking — but it makes the weather a property of the camera rather than of
// the scene, and a camera VMD is the case that proves it wrong: a VMD dollies and
// cuts, and the whole field would lurch after it on every cut. A stage that snows
// is a stage that snows regardless of where you stand to watch it.
//
// The cost is honest: fly the camera outside CENTER ± AREA and the weather is
// behind you. MMD scenes are staged around the origin, so that is a corner rather
// than the common case, and AREA is the dial if a scene needs a bigger stage.

// Tunables.
const FALL = 2.2;        // metres per second, before size weighting
const DRIFT = 0.55;      // sideways wander
const SWIRL = 0.18;      // how tightly the wander curls
const CENTER = vec3f(0.0, 0.0, 0.0);   // the stage the weather falls on
const AREA = 34.0;       // half-width of the volume
const TOP = 46.0;        // ceiling they fall from — well above the frame
const SIZE_MIN = 0.035;
const SIZE_MAX = 0.17;    // a few big ones near the lens, most of them small
const TWINKLE = 0.35;     // how much a flake catches the light as it turns
const SOFT = 0.5;        // softness of the flake's halo, 0..1
const INTENSITY = 3.0;   // HDR brightness — see the note below
const ARMS = 0.55;       // how far the arms reach, 0..1 of the quad
const CORE = 0.16;       // the hub at the centre

// A NOTE ON BRIGHTNESS, because it is the one thing that surprises everyone.
//
// Particles draw INSIDE the scene pass, in HDR, before tone mapping — unlike a
// background or foreground, which composites over the finished frame. So white
// here is not the white you get out: vec3f(1.0) goes through AgX and lands as
// grey. Snow has to be brighter than the paper it is drawn on, which is what
// INTENSITY is for. The upside of the same fact is that the scene's grade and
// exposure apply to the snow, so it belongs to the shot instead of sitting on
// top of it.

fn particleInit(id: u32, seed: f32) -> Particle {
  var p: Particle;
  let r = rzHash13(seed + f32(id) * 0.0173);
  let r2 = rzHash13(seed * 1.77 + f32(id) * 0.0411);
  p.pos = vec3f(CENTER.x + (r.x - 0.5) * AREA * 2.0, r.y * TOP, CENTER.z + (r.z - 0.5) * AREA * 2.0);
  p.size = mix(SIZE_MIN, SIZE_MAX, r2.x);
  // Bigger flakes fall faster, which is what stops the field looking like one
  // sheet sliding down the screen.
  p.vel = vec3f(0.0, -FALL * mix(0.6, 1.4, r2.x), 0.0);
  // Never expires: flakes WRAP at the floor (see step) rather than recycling
  // through the pool, so none of them appears out of nothing in mid-air.
  p.life = 1.0e9;
  p.rot = r2.z * 6.2831853;
  p.seed = seed + f32(id) * 0.0173;
  return p;
}

fn particleStep(p: Particle, dt: f32) -> Particle {
  var q = p;
  // Curl noise is divergence-free, so flakes wander past each other instead of
  // piling into the same line the way a plain sine drift does.
  let w = rzCurlNoise(q.pos * SWIRL + vec3f(0.0, rzTime() * 0.15, 0.0));
  q.pos = q.pos + (q.vel + w * DRIFT) * dt;
  q.rot = q.rot + dt * 0.6;
  // TOROIDAL in x/z as well as y.
  //
  // Wrapping only at the floor meant the volume moved with the camera but its
  // contents did not: orbiting left the old field draining where it was while a
  // new one filled in slowly, a fall-time behind. Folding x and z about the
  // centre too means a flake that leaves one side re-enters on the other in the
  // SAME frame, so the field is always centred and never has to refill. The
  // seam sits AREA away, off to the side and far from what you are looking at.
  let span = AREA * 2.0;
  let rx = q.pos.x - CENTER.x;
  let rz = q.pos.z - CENTER.z;
  q.pos.x = CENTER.x + rx - span * floor(rx / span + 0.5);
  q.pos.z = CENTER.z + rz - span * floor(rz / span + 0.5);
  // And over the top when it reaches the floor, in a freshly chosen column so the
  // same flake never falls down the same line twice.
  if (q.pos.y < 0.0) {
    let h = rzHash13(q.seed + rzTime() * 0.29);
    q.pos = vec3f(CENTER.x + (h.x - 0.5) * span, TOP, CENTER.z + (h.z - 0.5) * span);
  }
  return q;
}

fn sdSegment(p: vec2f, a: vec2f, b: vec2f) -> f32 {
  let pa = p - a;
  let ba = b - a;
  let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
  return length(pa - ba * h);
}

/**
 * A six-armed flake, folded rather than repeated.
 *
 * The obvious way to build one is to loop six times and rotate — which is what
 * the hand-written fullscreen version had to do. Folding the angle into a single
 * 60-degree wedge gets the same six-fold symmetry from ONE arm, so this is three
 * segment tests instead of thirty. That trade is worth spelling out: a particle
 * shader runs on the handful of pixels its own quad covers, not on the screen,
 * so detail here is cheap in a way it never was in a fullscreen march.
 */
fn flake(q: vec2f) -> f32 {
  let k = 1.0471975;                     // 60 degrees
  let a = atan2(q.y, q.x);
  let folded = abs(a - k * floor(a / k + 0.5));
  let r = length(q);
  let p = vec2f(cos(folded), sin(folded)) * r;
  var d = sdSegment(p, vec2f(0.0), vec2f(ARMS, 0.0)) - 0.035;
  d = min(d, sdSegment(p, vec2f(ARMS * 0.45, 0.0), vec2f(ARMS * 0.72, ARMS * 0.30)) - 0.022);
  d = min(d, sdSegment(p, vec2f(ARMS * 0.75, 0.0), vec2f(ARMS * 0.95, ARMS * 0.22)) - 0.018);
  return min(d, r - CORE);
}

fn particleShade(p: Particle, uv: vec2f) -> vec4f {
  let q = (uv - vec2f(0.5)) * 2.0;
  let d = flake(q);
  // The crystal itself, plus a soft bloom of light around it — near flakes read
  // as a shape, distant ones collapse to the halo, which is what snow does.
  let solid = smoothstep(0.04, -0.02, d);
  let halo = rzFalloff(length(q), 1.0) * SOFT * 0.5;
  // No lifetime fade — a wrapped flake never dies, and fading a share of them at
  // any moment is what makes a field look like it is flickering.
  let a = clamp(solid + halo, 0.0, 1.0);
  // Each flake catches the light at its own rate as it tumbles. Snow that all
  // brightens together reads as a flicker; snow that does not sparkle at all
  // reads as paper.
  let twinkle = 1.0 + TWINKLE * sin(rzTime() * (1.3 + p.seed * 2.7) + p.seed * 39.1);
  return vec4f(vec3f(INTENSITY * twinkle), a);
}
