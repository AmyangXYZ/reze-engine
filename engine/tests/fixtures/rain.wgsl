#particles 5000

// Rain — the same rain, drawn as geometry.
//
// The fullscreen version drew two flat CURTAINS, one behind the cast and one in
// front, and feathered the far one against the silhouette by hand. That was the
// only way to fake depth from a screen-space effect. These drops are real points
// in the scene, depth-tested like anything else, so a drop passes behind her
// shoulder and in front of her hand in the same frame with nothing to tune.
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
const COLOR = vec3f(0.82, 0.90, 1.0);
const INTENSITY = 2.4;   // HDR: particles draw before tone mapping, so white needs headroom
const FALL = 24.0;       // metres per second — rain is much faster than it looks
const SLANT = 0.11;      // wind, as a fraction of fall speed
const CENTER = vec3f(0.0, 0.0, 0.0);   // the stage the weather falls on
const AREA = 34.0;       // half-width of the volume
const TOP = 44.0;        // drops enter well above the frame, not just off the top
const WIDTH = 0.009;     // a drop's thickness
const LENGTH = 18.0;     // and how many times longer than it is wide
const HEAD = 0.55;       // how much brighter the leading end is

fn particleInit(id: u32, seed: f32) -> Particle {
  var p: Particle;
  let r = rzHash13(seed + f32(id) * 0.0131);
  let r2 = rzHash13(seed * 2.13 + f32(id) * 0.0357);
  p.pos = vec3f(CENTER.x + (r.x - 0.5) * AREA * 2.0, r.y * TOP, CENTER.z + (r.z - 0.5) * AREA * 2.0);
  let speed = FALL * mix(0.85, 1.2, r2.x);
  p.vel = vec3f(SLANT * speed, -speed, SLANT * 0.4 * speed);
  p.size = WIDTH * mix(0.75, 1.35, r2.y);
  // The engine orients a stretched quad along the drop's SCREEN velocity, so a
  // streak stays a streak from any angle instead of becoming a smear from above.
  p.stretch = LENGTH * mix(0.8, 1.25, r2.z);
  // Long enough never to expire: these drops WRAP rather than die (see step), so
  // the pool is a continuous curtain instead of one that thins and refills.
  p.life = 1.0e9;
  p.seed = seed + f32(id) * 0.0131;
  return p;
}

fn particleStep(p: Particle, dt: f32) -> Particle {
  var q = p;
  q.pos = q.pos + q.vel * dt;
  // WRAPPED, not recycled.
  //
  // Recycling through the pool was wrong twice. A respawn lands at a random
  // height, so drops popped into existence in mid-air; and staggering their age
  // to avoid that put them at random points in a fade curve, so a share of the
  // curtain was always fading out — which is what made the rain intermittent.
  // A drop that reappears at the top is seamless by construction.
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
  // And over the top at the floor, in a freshly chosen column so the same line
  // does not fall forever.
  if (q.pos.y < 0.0) {
    let h = rzHash13(q.seed + rzTime() * 0.37);
    q.pos = vec3f(CENTER.x + (h.x - 0.5) * span, TOP, CENTER.z + (h.z - 0.5) * span);
  }
  return q;
}

fn particleShade(p: Particle, uv: vec2f) -> vec4f {
  let q = (uv - vec2f(0.5)) * 2.0;
  // Across the drop: soft, so it is a filament rather than a hard bar.
  let across = rzFalloff(abs(q.x), 1.0);
  // Along it: taper both ends, and weight the leading one — a falling drop is
  // brightest where it is going, which is what reads as speed.
  let along = smoothstep(1.0, 0.55, abs(q.y));
  let head = mix(1.0 - HEAD, 1.0, clamp(q.y * 0.5 + 0.5, 0.0, 1.0));
  // No lifetime fade: a wrapped drop never dies, so there is nothing to fade —
  // and fading a fraction of them at any moment is exactly what made the
  // previous version flicker.
  return vec4f(COLOR * INTENSITY * head, across * along);
}
