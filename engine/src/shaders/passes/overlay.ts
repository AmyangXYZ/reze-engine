// Editor overlays — bones, rigidbodies and joints — as one instanced pass.
//
// Two kinds of geometry, because thick lines and thick rings want different
// answers.
//
// STROKED shapes (mode 0, 2) expand each segment into a screen-space quad, so
// their width is a pixel count and holds at any zoom. That works while a segment
// is longer on screen than the stroke is wide; past that, neighbouring quads
// meet at a corner nothing mitres and a circle drawn this way turns into a
// sunburst. So the one shape that must be both small and thick — the bone
// marker — is not stroked at all.
//
// FILLED shapes (mode 3) are triangles already: the marker's ring is an annulus
// and its centre is a disc, both facing the camera. No joins exist to get wrong,
// and the ring can be as heavy as it likes.
//
// The layer is 4x multisampled and composited over the finished frame rather
// than drawn into it: the swapchain is single-sample and an MSAA attachment
// cannot load from one. So fragments here are PREMULTIPLIED — rgb already
// carries alpha — and the two blends compose correctly, once against a
// transparent layer and once onto the frame.
//
// The capsule is why `caps` exists. A PMX capsule is a radius and a cylinder
// length, so scaling a unit capsule non-uniformly would stretch its hemispheres
// into ellipsoids. Instead the unit shape carries both caps at the origin and
// each vertex names which cap it belongs to (-1, 0, +1); the VS pushes it along
// local Y by that sign times `extent`. A segment can straddle the two — the
// cylinder's side lines run from +1 to -1 — so `caps` holds the sign at BOTH
// ends, not one.

export const OVERLAY_SHADER_WGSL = /* wgsl */ `
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _pad: f32 };
// 16 bytes exactly. Scalars rather than a vec3f tail: a vec3f aligns to 16 and
// would push the struct past the buffer the pipeline is handed.
struct OverlayUniforms { viewport: vec2f, dashPeriodPx: f32, _pad0: f32 };

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> overlay: OverlayUniforms;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) @interpolate(flat) color: vec4f,
}

fn qrot(q: vec4f, v: vec3f) -> vec3f {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}

// Placement mode, carried per VERTEX because it belongs to the shape rather than
// to the copy.
//
//   0  world    — rotate, scale, translate. The shapes that ARE geometry:
//                 rigidbody volumes, joint crosses, plain lines.
//   3  facing   — a flat filled shape in the camera's plane, radius scale.x.
//                 The bone marker's solid centre.
//   5  solid    — placed in world like mode 0, but filled: the rigidbody
//                 volumes, which read as volumes rather than as a cage of lines.
//   4  annulus  — the marker's ring: world radius scale.x, and a stroke that is
//                 a PIXEL width like every other line, so the two links leave it
//                 as a continuation of the same stroke rather than a step. The
//                 inner and outer rims are offset radially in screen space,
//                 which is what lets a small circle carry a thick stroke — the
//                 stroked polyline that preceded it tore into a sunburst as soon
//                 as the width passed the segment length.
//   2  link     — the taper from a bone to its child: along the bone, splayed
//                 across it in whichever direction the camera can actually see,
//                 so the two lines never collapse into one.
fn placeWorld(p: vec3f, capSign: f32, iRot: vec4f, iPos: vec4f, iScale: vec4f, mode: f32) -> vec3f {
  let right = vec3f(camera.view[0].x, camera.view[1].x, camera.view[2].x);
  let up = vec3f(camera.view[0].y, camera.view[1].y, camera.view[2].y);

  if (mode > 2.5) {
    return iPos.xyz + (right * p.x + up * p.y) * iScale.x;
  }
  if (mode > 1.5) {
    let boneDir = normalize(qrot(iRot, vec3f(0.0, 1.0, 0.0)));
    let camFwd = vec3f(camera.view[0].z, camera.view[1].z, camera.view[2].z);
    var splay = cross(boneDir, camFwd);
    let sl = length(splay);
    // Looking straight down the bone there is no visible across, and any
    // perpendicular does — the taper is a dot on screen either way.
    splay = select(right, splay / max(sl, 1e-6), sl > 1e-4);
    return iPos.xyz + splay * (p.x * iScale.x) + boneDir * (p.y * iScale.y);
  }
  return qrot(iRot, p * iScale.xyz + vec3f(0.0, capSign * iPos.w, 0.0)) + iPos.xyz;
}

@vertex fn vs(
  @location(0) pos: vec3f,
  @location(1) dir: vec3f,
  @location(2) caps: vec2f,
  @location(3) side: f32,
  @location(4) t: f32,
  @location(5) mode: f32,
  @location(6) iRot: vec4f,
  @location(7) iPos: vec4f,
  @location(8) iScale: vec4f,
  @location(9) iColor: vec4f,
) -> VSOut {
  let vp = camera.projection * camera.view;
  let half = 0.5 * overlay.viewport;

  var out: VSOut;
  out.color = iColor;

  // A solid volume is placed in world, like a stroked shape, but filled.
  if (mode > 4.5) {
    out.pos = vp * vec4f(placeWorld(pos, caps.x, iRot, iPos, iScale, 0.0), 1.0);
    return out;
  }

  // The marker's ring: a circle of world radius, stroked to a pixel width.
  if (mode > 3.5) {
    let centre = vp * vec4f(iPos.xyz, 1.0);
    let w = max(abs(centre.w), 1e-6);
    let pxPerWorld = camera.projection[1].y * half.y / w;
    // side is -1 on the inner rim and +1 on the outer.
    let radiusPx = iScale.x * pxPerWorld + side * abs(iScale.w) * 0.5;
    let radial = normalize(pos.xy);
    out.pos = vec4f(centre.xy + ((radial * radiusPx) / half) * centre.w, centre.z, centre.w);
    return out;
  }

  let c0 = vp * vec4f(placeWorld(pos, caps.x, iRot, iPos, iScale, mode), 1.0);

  // A filled vertex is already where it belongs.
  if (mode > 2.5) {
    out.pos = c0;
    return out;
  }

  let c1 = vp * vec4f(placeWorld(pos + dir, caps.y, iRot, iPos, iScale, mode), 1.0);
  let w0 = max(abs(c0.w), 1e-6);
  let w1 = max(abs(c1.w), 1e-6);
  let s0 = (c0.xy / w0) * half;
  let s1 = (c1.xy / w1) * half;
  let delta = s1 - s0;
  let segLenPx = max(length(delta), 1e-6);
  let tangent = delta / segLenPx;
  let normalPx = vec2f(-tangent.y, tangent.x);
  let offsetPx = normalPx * side * abs(iScale.w) * 0.5;
  out.pos = vec4f(c0.xy + (offsetPx / half) * c0.w, c0.z, c0.w);
  return out;
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  return vec4f(in.color.rgb * in.color.a, in.color.a);
}
`

// Lays the resolved overlay layer over the finished frame. Its input is
// premultiplied, so the blend is a straight `src + dst * (1 - src.a)`.
export const OVERLAY_COMPOSITE_SHADER_WGSL = /* wgsl */ `
@group(0) @binding(0) var overlayTex: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {
  return textureLoad(overlayTex, vec2<i32>(p.xy), 0);
}
`
