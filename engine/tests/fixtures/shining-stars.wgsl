#layer additive

// Shining Stars — a night sky, drawn as points rather than blooms.
//
// The cells, the projection and the hash are the ones this effect has always
// used. What changed is the STAR: it used to reach four times its own radius as
// a soft halo, which at 1080p is a disc 26 pixels across, and it painted that
// whole disc in one flat colour. That is a bloom, not a star.
//
// A field composites in DISPLAY space, after tone mapping, and the engine clamps
// what this returns to [0,1]. Brightness is therefore not available as a dial —
// turning it up only widens the saturated middle. What reads as a bright star is
// a SHARP core that goes white while its halo keeps its colour, and a pair of
// diffraction spikes. That gradient is the whole trick.

// Tunables — edit a value and hit Cmd-Enter to see it live.
const WARM = vec3f(1.0, 0.93, 0.82);   // the colour of the warm half
const COOL = vec3f(0.82, 0.89, 1.0);   // and the cool half
const DENSITY = 0.5;        // 0..1 — how crowded the sky is
const TWINKLE = 0.5;        // 0..1 — flicker speed
const INTENSITY = 1.15;     // overall brightness
const REACH = 0.24;         // a star's whole extent, in cell widths
const HALO = 0.15;          // the glow around it — kept low, it is what discs
const SPIKE = 0.8;          // diffraction flare — what gives a big star structure
const SPIKE_THIN = 0.075;   // how fine its arms are

fn bgHash2(p: vec2f) -> vec2f {
  let q = vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)));
  return fract(sin(q) * 43758.5453);
}

/** Compact support: 1 at the centre, exactly 0 at r, smooth between. A glow that
 *  merely gets small has to be cut off somewhere, and the cut is an edge. */
fn bgFall(d: f32, r: f32) -> f32 {
  let x = clamp(d / max(r, 1e-6), 0.0, 1.0);
  let f = 1.0 - x;
  return f * f * f;
}

/** One layer of stars, as (colour * energy, energy). */
fn starLayer(sph: vec2f, scale: f32, thresh: f32, time: f32, speed: f32,
             spikes: f32, latCos: f32) -> vec4f {
  let cell = floor(sph * scale);
  let h = bgHash2(cell);
  let bright = bgHash2(cell + 7.31).x;
  if (bright < thresh) { return vec4f(0.0); }

  // Star sits at a hashed point inside its cell — kills the grid look.
  //
  // Corrected for the projection's stretch: one step of longitude covers cos(lat)
  // as much sky as one of latitude, so an uncorrected star is a circle at the
  // equator and an ellipse overhead.
  let local = (fract(sph * scale) - (0.15 + 0.7 * h)) * vec2f(latCos, 1.0);
  let d = length(local);

  let size = (bright - thresh) / (1.0 - thresh);
  let reach = REACH * (0.62 + 0.38 * size);
  if (d >= reach) { return vec4f(0.0); }

  // Cubed, then cubed again: a ninth-power core is a true point of light with no
  // plateau at all. The old smoothstep held a flat top and then ran on for four
  // times its own radius, which is what drew a disc.
  let f = bgFall(d, reach);
  let core = f * f * f;
  let halo = f * HALO;

  // Diffraction spikes, turned to the star's own angle. What a lens does to a
  // point source, and the thing that says "star" rather than "dot".
  var flare = 0.0;
  if (spikes > 0.0) {
    let rot = h.y * 6.2831853;
    let cr = cos(rot);
    let sr = sin(rot);
    let q = vec2f(local.x * cr - local.y * sr, local.x * sr + local.y * cr) / reach;
    let ax = abs(q);
    let sx = bgFall(ax.x, 1.0) * bgFall(ax.y, SPIKE_THIN);
    let sy = bgFall(ax.y, 1.0) * bgFall(ax.x, SPIKE_THIN);
    // Only the brightest carry a flare, which is what keeps the sky fine.
    flare = (sx + sy) * spikes * size * size;
  }

  // Per-star period from its own hash: some breathe over seconds, some blink.
  let period = 0.25 + 2.75 * bgHash2(cell + 31.7).y;
  let tw = 0.55 + 0.45 * sin(time * speed * period + h.x * 40.0);

  let e = (core + flare + halo) * tw * (0.45 + 0.75 * size);
  // White at the core, its own colour in the halo — the gradient that reads as
  // heat. One flat colour across the whole star is what made it look painted on.
  let tint = mix(COOL, WARM, h.x);
  let col = mix(tint, vec3f(1.0), clamp(core * 2.2, 0.0, 1.0));
  return vec4f(col * e, e);
}

fn background(ray: vec3f, uv: vec2f, time: f32) -> vec4f {
  let d = normalize(ray);
  // Latitude/longitude projection
  let sph = vec2f(atan2(d.x, d.z) * 1.2, asin(clamp(d.y, -1.0, 1.0)) * 1.5);
  // How much this latitude squashes longitude, for the correction in starLayer.
  let latCos = max(0.2, sqrt(max(0.0, 1.0 - d.y * d.y)));

  // Denser = MORE stars, not smaller ones (sizes untouched)
  let dens = mix(0.95, 0.82, clamp(DENSITY, 0.0, 1.0));
  var acc = starLayer(sph, 30.0, dens, time, TWINKLE * 1.6, SPIKE, latCos);
  // Faint dust layer: denser, smaller, slower, and no flare of its own.
  acc += 0.4 * starLayer(sph + 3.7, 70.0, mix(0.985, 0.9, clamp(DENSITY, 0.0, 1.0)),
                         time, TWINKLE * 0.9, 0.0, latCos);

  let a = clamp(acc.w * INTENSITY, 0.0, 1.0);
  if (a <= 0.0) { return vec4f(0.0); }
  // Straight alpha — the engine's over-composite premultiplies. The colour is the
  // energy-weighted average, so two stars overlapping mix instead of the second
  // one winning.
  return vec4f(acc.rgb / max(acc.w, 1e-4), a);
}
