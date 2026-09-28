/**
 * Screen-space subsurface scattering — ray-mmd's skin pass (PostProcessScattering.fxsub,
 * SSSGaussBlurPS), ported constant for constant.
 *
 * A skin surface is not shaded by the light at its own pixel alone: light that
 * enters a little way off comes back out here, red furthest. ray-mmd models that
 * as a separable blur of the lit skin in screen space, one axis per pass, with a
 * per-channel Gaussian (red widest) and a depth reject so light never crosses from
 * a forearm onto the body behind it. The width is a length ON THE SURFACE, so it
 * is divided by depth, and it narrows where the surface turns away from the view.
 * Then a share of the unblurred picture is laid back over it — the "spike" that
 * keeps pores, painted lines and the terminator from dissolving.
 *
 * What marks a pixel as skin is the scene pass's aux .b, written by the graph's
 * `subsurface` node (0 everywhere else), as a quarter of its strength. The strength
 * scales the radius, as ray-mmd's sssStrength does; 1 is ray-mmd's skin, 4 the most.
 *
 * One departure, and it is a routing one: ray-mmd blurs the lit colour BEFORE the
 * albedo multiply and adds specular after. This engine resolves one HDR colour per
 * pixel, so the blur runs over the finished colour. At this radius — a few pixels,
 * with the spike keeping 14–20% of the original — the texture is not visibly
 * softened; the lighting is.
 *
 * Two entry points share the module: X reads the HDR resolve and writes the
 * scratch target, Y reads the scratch and writes back INTO the HDR resolve. The
 * spike mix (lerp toward the original) is done by the Y pipeline's blend state,
 * with the blend constant holding ray-mmd's per-channel spike: the destination IS
 * the original, so no third copy of the frame is needed.
 */

/** ray-mmd, skin profile (index 1): profileSpikeRadArr[1].xyz × (1 − sssAmount),
 *  with sssAmount = customA = 0.6 in Materials/Skin/material_skin.fx. */
export const SSS_SPIKE: [number, number, number] = [0.35 * 0.4, 0.4 * 0.4, 0.5 * 0.4]

export const SUBSURFACE_WGSL = /* wgsl */ `
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var maskTex: texture_2d<f32>;
@group(0) @binding(2) var depthTex: texture_depth_multisampled_2d;
// dofU[2] = (projA, projB): the same z-buffer inversion the composite uses.
@group(0) @binding(3) var<uniform> dofU: array<vec4f, 3>;
// sssU[0] = (direction.xy, radius, _); sssU[1] = (tanHalfFov·aspect, tanHalfFov, width, height).
@group(0) @binding(4) var<uniform> sssU: array<vec4f, 2>;

struct VsOut { @builtin(position) pos: vec4f };

@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
  let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
  var o: VsOut;
  o.pos = vec4f(p * 2.0 - 1.0, 0.0, 1.0);
  return o;
}

fn size() -> vec2i { return vec2i(sssU[1].zw); }

fn texel(uv: vec2f) -> vec2i {
  return clamp(vec2i(floor(uv * sssU[1].zw)), vec2i(0), size() - 1);
}

fn linearDepth(c: vec2i) -> f32 {
  let z = textureLoad(depthTex, c, 0);
  return clamp(dofU[2].y / (z - dofU[2].x), 0.05, 100000.0);
}

// View-space position of a texel centre — ray-mmd's GetViewPosition, with y up.
fn viewPos(c: vec2i, d: f32) -> vec3f {
  let uv = (vec2f(c) + 0.5) / sssU[1].zw;
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
  return vec3f(ndc * sssU[1].xy * d, d);
}

fn skin(c: vec2i) -> f32 { return textureLoad(maskTex, c, 0).b; }

fn blur(fragPos: vec4f) -> vec4f {
  let c = vec2i(fragPos.xy);
  let amount = skin(c) * 4.0;
  let dir = sssU[0].xy;
  let uv = (vec2f(c) + 0.5) / sssU[1].zw;
  let centre = textureLoad(srcTex, c, 0);

  // ray-mmd, verbatim: six offsets, and the skin variances per channel.
  // A var, not a const: a const array cannot be indexed by the loop counter.
  var offsets = array<f32, 6>(0.352, 0.719, 1.117, 1.579, 2.177, 3.213);
  const profileVar = vec3f(9.6, 2.8, 2.2);
  // mSSSSScale at the controller's default; it sets both the width and how hard
  // a depth step rejects.
  const SSSS_SCALE: f32 = 0.04;

  let d = linearDepth(c);

  // The surface's normal, rebuilt from depth (ray-mmd reads it from its G-buffer;
  // this engine keeps none). Only its angle to the view matters: the blur narrows
  // along an axis the surface is turning away on.
  let s = size();
  let cx = min(c + vec2i(1, 0), s - 1);
  let cy = min(c + vec2i(0, 1), s - 1);
  let P = viewPos(c, d);
  let N = cross(viewPos(cx, linearDepth(cx)) - P, viewPos(cy, linearDepth(cy)) - P);
  let nxz = select(vec2f(0.0, 1.0), normalize(N.xz), dot(N.xz, N.xz) > 1e-12);
  let nyz = select(vec2f(0.0, 1.0), normalize(N.yz), dot(N.yz, N.yz) > 1e-12);
  let scaleX = abs(dot(nxz, normalize(P.xz)));
  let scaleY = abs(dot(nyz, normalize(P.yz)));
  let perspectiveScale = max(dot(dir, vec2f(scaleX, scaleY)), 0.3);

  let radius = 0.0055 * amount * perspectiveScale / SSSS_SCALE;
  let step = dir * radius / d;

  var weightSum = vec3f(1.0);
  var colour = centre.rgb;
  for (var i = 0; i < 6; i++) {
    let o = offsets[i] / 5.5 * step;
    for (var side = 0; side < 2; side++) {
      let tc = texel(select(uv - o, uv + o, side == 0));
      // Skin only: ray-mmd zeroes a non-skin pixel's depth so the depth term
      // rejects it; the mask says the same thing directly.
      let isSkin = select(0.0, 1.0, skin(tc) > 0.004);
      let dd = (linearDepth(tc) - d) * 1000.0 * SSSS_SCALE;
      let w = exp(-(offsets[i] * offsets[i] + dd * dd) / profileVar) * isSkin;
      weightSum += w;
      colour += w * textureLoad(srcTex, tc, 0).rgb;
    }
  }
  return vec4f(colour / weightSum, centre.a);
}

@fragment fn fsX(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  // Nothing reads the scratch target off skin: the Y pass rejects those taps.
  if (skin(vec2i(pos.xy)) <= 0.004) { return vec4f(0.0); }
  return blur(pos);
}

@fragment fn fsY(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  // Off skin the HDR resolve must stay exactly as the scene left it.
  if (skin(vec2i(pos.xy)) <= 0.004) { discard; }
  return blur(pos);
}
`
