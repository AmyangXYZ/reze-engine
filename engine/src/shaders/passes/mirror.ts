// The mirror surface — the one thing that consumes the reflection pass as a
// SURFACE rather than as a layer under the floor.
//
// The ground has sampled the mirror target since step 7D, and it does it the
// only way a planar reflection can be sampled correctly: project the fragment's
// world position through the camera that rendered the reflection, and the texel
// you land on is exactly where that fragment's reflection was drawn. No
// screen-space ray, no guess. This file is that same projection on a quad you
// can put anywhere, which is all a standing mirror is.
//
// What it deliberately does NOT do:
//   - light. A mirror shows what it reflects; a sun term on top of that is a
//     second surface nobody asked for.
//   - receive shadow. The reflection already carries the shadow the reflected
//     geometry is standing in.
//   - carry a strength. Coverage is 1 — the tint is how a mirror falls short of
//     perfect, the same way real glass does, and it is a colour rather than a
//     dial that fades the surface out into nothing.

import { sceneFsOutWgsl, sceneIdPadWgsl } from "./scene-contract"

/** Bytes in MirrorMat below: mat4x4f + three 16-byte rows. */
export const MIRROR_MAT_BYTES = 64 + 16 * 3

export function mirrorShaderWgsl(): string {
  return /* wgsl */ `
struct CameraUniforms { view: mat4x4f, projection: mat4x4f, viewPos: vec3f, _p: f32, };
// params = (projA, projB, _, _) — the depth-linearisation pair, carried beside
// the matrix exactly as the ground reads it. Unused here (the surface takes a
// flat blur level, not a depth-proportional one) and kept so the two consumers
// bind one buffer rather than two shapes of the same thing.
struct MirrorVP { viewProj: mat4x4f, params: vec4f, };
struct MirrorMat {
  model: mat4x4f,
  tint: vec3f, blur: f32,
  // frame = the band's width in WORLD units, 0 for a bare pane. World rather
  // than a fraction of the quad so that a wide mirror and a narrow one in the
  // same scene are framed in the same moulding.
  frameColor: vec3f, frame: f32,
  size: vec2f, _pad: vec2f,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> mirrorVP: MirrorVP;
@group(0) @binding(2) var<uniform> material: MirrorMat;
@group(0) @binding(3) var mirrorTex: texture_2d<f32>;
@group(0) @binding(4) var linearSampler: sampler;
// The mirror pass's aux, resolved. .g is accumulated alpha — the reflection's
// COVERAGE, and the only place it exists: the HDR attachment is rg11b10ufloat
// wherever the device allows it, and that format has no alpha channel, so a
// mirror reading the colour target's own alpha reads 1 everywhere and paints
// the empty black of the reflection across the whole pane.
@group(0) @binding(5) var mirrorMask: texture_2d<f32>;

struct VO {
  @builtin(position) position: vec4f,
  @location(0) worldPos: vec3f,
  @location(1) normal: vec3f,
  /** Across the quad, 0..1 — what the frame is measured in. */
  @location(2) uv: vec2f,
};

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  // The quad is generated, not buffered: it is six vertices of a unit square
  // and the model matrix carries every difference between one mirror and
  // another, so a vertex buffer would be an allocation per mirror holding the
  // same twelve numbers.
  //
  // Local XY, facing local +Z. A mirror at identity therefore faces the way the
  // default camera looks FROM, which is the orientation someone dropping one
  // into a scene means by "in front of her".
  // Local Y runs 0..1 from the BASE, not -0.5..0.5 from the centre. A mirror
  // stands on the floor: shortening it has to take the height off the top and
  // leave the foot where it is, and turning it has to pivot on that foot. With
  // a centred quad both of those move the whole mirror.
  var q = array<vec2f, 6>(
    vec2f(-0.5, 0.0), vec2f(0.5, 0.0), vec2f(0.5, 1.0),
    vec2f(-0.5, 0.0), vec2f(0.5, 1.0), vec2f(-0.5, 1.0),
  );
  let local = vec4f(q[vi], 0.0, 1.0);
  let uv = q[vi] + vec2f(0.5, 0.0);
  let world = material.model * local;
  var o: VO;
  o.worldPos = world.xyz;
  o.uv = uv;
  // The local +Z axis through the model's rotation. Column 2 of a matrix whose
  // scale is uniform per axis is that axis scaled — normalized here because the
  // side test below is a sign, and a scaled normal would still give the right
  // sign but a wrong-length one is a trap for whatever reads this next.
  o.normal = normalize((material.model * vec4f(0.0, 0.0, 1.0, 0.0)).xyz);
  o.position = camera.projection * camera.view * world;
  return o;
}

${sceneFsOutWgsl()}@fragment fn fs(i: VO) -> FSOut {
  var out: FSOut;
${sceneIdPadWgsl("out")}
  let toEye = camera.viewPos - i.worldPos;
  // Facing the viewer whichever side is up: the quad has one normal and two
  // sides, and the moulding is on both.
  let nrm = select(-i.normal, i.normal, dot(toEye, i.normal) > 0.0);

  // ── The moulding ──
  //
  // Measured in world units off each edge, so the width means the same thing on
  // a tall mirror as on a wide one.
  let dx = min(i.uv.x, 1.0 - i.uv.x) * material.size.x;
  let dy = min(i.uv.y, 1.0 - i.uv.y) * material.size.y;
  let edge = min(dx, dy);
  if (edge < material.frame) {
    // A MOULDING with a cross-section, not a painted border. The band's normal
    // rolls from facing outward at the rim to facing the viewer at the inner
    // lip, and that roll is the whole trick: a flat rectangle of colour reads as
    // a decal on a plane, while a normal that turns catches the view and reads
    // as metal bent around glass.
    let t = clamp(edge / max(material.frame, 1e-4), 0.0, 1.0);
    // Which way this point of the band faces, in the plane. The axes come off
    // the model matrix, so a rotated mirror's moulding rotates with it.
    let ax = normalize(material.model[0].xyz);
    let ay = normalize(material.model[1].xyz);
    var outward = ax * select(-1.0, 1.0, i.uv.x > 0.5);
    if (dy < dx) { outward = ay * select(-1.0, 1.0, i.uv.y > 0.5); }
    let roll = cos(t * 1.5707963);
    let n = normalize(nrm * (1.0 - roll) + outward * roll);
    let v = normalize(toEye);
    // Two FIXED studio lights, not the scene's. A frame lit by the scene goes
    // black whenever the sun is behind it, and a black frame reads as a hole cut
    // in the picture rather than as metal in shadow.
    let l1 = normalize(vec3f(0.40, 0.80, 0.45));
    let l2 = normalize(vec3f(-0.50, 0.20, -0.60));
    let spec = pow(max(dot(n, normalize(l1 + v)), 0.0), 64.0) * 2.6
             + pow(max(dot(n, normalize(l2 + v)), 0.0), 20.0) * 0.7;
    let diff = max(dot(n, l1), 0.0) * 0.55 + max(dot(n, l2), 0.0) * 0.25;
    // The grazing rim is what makes a metal edge read as an edge.
    let fres = pow(1.0 - abs(dot(n, v)), 4.0);
    // A METAL TINTS ITS OWN HIGHLIGHT. White specular on a gold frame reads as
    // cream plastic; carrying the frame's colour into the highlight and the rim
    // is what separates gold from a yellow-painted thing.
    let sheen = mix(vec3f(1.0), material.frameColor, 0.65);
    out.color = vec4f(material.frameColor * (0.45 + diff) + sheen * (spec + fres * 0.40), 1.0);
    out.mask = vec4f(0.0, 1.0, 0.0, 1.0);
    return out;
  }

  // Which side is being looked at. Behind a mirror there is no reflection to
  // show — the fold puts the mirror camera on the viewer's own side and the
  // projection returns a smear — so the back reads as what the back of a mirror
  // is: a flat panel. Without this a mirror rotated the wrong way is not
  // wrong-looking, it is broken-looking, and the fix is invisible from the
  // transform panel.
  if (dot(toEye, i.normal) <= 0.0) {
    // The BACKING BOARD. A mirror seen from behind is a board in a frame, and
    // it is opaque: the scene must not show through it, or the pane reads as a
    // pane of dark glass with the cast standing inside it.
    out.color = vec4f(material.frameColor * 0.55, 1.0);
    out.mask = vec4f(0.0, 1.0, 0.0, 1.0);
    return out;
  }

  // The projective sample: this fragment's world position through the mirror
  // camera IS the texel its reflection landed on.
  let mc = mirrorVP.viewProj * vec4f(i.worldPos, 1.0);
  let mndc = mc.xyz / max(mc.w, 1e-6);
  let muv = clamp(vec2f(mndc.x * 0.5 + 0.5, 0.5 - mndc.y * 0.5), vec2f(0.0), vec2f(1.0));
  // A flat level, not the ground's depth-proportional one. That trick
  // reconstructs how far behind the surface the reflected geometry sits by
  // mirroring the eye through y — arithmetic that is only true for a floor.
  // The general form belongs here eventually; a uniform frosting is the honest
  // thing to ship before it, and blur 0 (the default) reads level 0 exactly.
  let refl = textureSampleLevel(mirrorTex, linearSampler, muv, material.blur * 5.0);
  // THE SAME LEVEL as the colour, and that is the whole of it. refl.rgb is
  // premultiplied by this coverage, so the two are one quantity in two
  // textures: blur the colour into the empty part of the reflection while
  // reading coverage sharp, and the pane divides a spread-out colour by a
  // coverage that never spread — which is a reflection that goes dark as the
  // frosting comes up, exactly backwards.
  let cov = textureSampleLevel(mirrorMask, linearSampler, muv, material.blur * 5.0).g;

  // ALPHA IS COVERAGE, and it is what keeps the glass clear.
  //
  // The reflection target clears TRANSPARENT, so coverage is 1 exactly where
  // the mirror pass drew something and 0 where it drew nothing. Filling the empty
  // part with the scene's background colour is the obvious thing and it is
  // wrong: that colour would then ride the view transform a second time (the
  // real backdrop composites after it), and the panel comes out paler and
  // flatter than the backdrop it sits against — a mirror that fogs.
  //
  // Letting it through instead costs nothing and is also what a mirror does: an
  // empty mirror shows the room, and here the room IS the backdrop.
  // PREMULTIPLIED, and it must not be premultiplied twice. refl.rgb came out of
  // the HDR target, which stores colour already weighted by its own alpha, and
  // the mirror class blends premultiplied for exactly that reason. Multiplying
  // by cov here as well is what made the reflection a ghost.
  //
  // ALPHA 1, ALWAYS — the pane REPLACES what is behind it. A mirror is opaque;
  // blending here let the real floor show through the glass beside the
  // reflected one, which is two grids in one pane. Premultiplied over at alpha
  // 1 is a straight overwrite, and refl.rgb is already premultiplied, so the
  // empty sky of the reflection writes 0 rather than black-over-something.
  out.color = vec4f(refl.rgb * material.tint, 1.0);
  // .g is the ACCUMULATED CANVAS ALPHA the composite un-premultiplies by, and a
  // surface that writes nothing here is a surface the composite treats as
  // absent — the reflection then survived only where something else had already
  // written alpha under it, which is to say only against the reflected floor.
  // .r is the bloom gate, left at 0: the glow of a reflected ribbon is already
  // in refl.rgb, and gating it again would bloom the one thing in the scene
  // most likely to be over the threshold twice.
  // COVERAGE goes here instead, and it REPLACES too (.a = 1, so the alpha-over
  // blend takes the source whole). The composite reads .g as accumulated alpha:
  // it un-premultiplies the colour by it and paints the scene's backdrop where
  // it falls short. That is what puts the true sky behind the reflection at the
  // true colour — the pane cannot paint the backdrop itself, because anything
  // it writes here goes through the view transform a second time.
  out.mask = vec4f(0.0, cov, 0.0, 1.0);
  return out;
}
`
}

/**
 * The mirror in the SHADOW pass — the frame throwing shade on the floor.
 *
 * Its own shader because the mirror has no vertex buffer and no skeleton: the
 * quad is six generated vertices and the model matrix is the whole of it, while
 * the ordinary shadow shader is built for skinned geometry with an alpha test.
 *
 * Depth only, no fragment stage. The pane is opaque across its whole rectangle
 * — glass included, because this glass is not see-through — so there is nothing
 * to alpha-test and nothing to colour.
 *
 * The cascade index arrives in its own tiny uniform rather than as a dynamic
 * offset: the cascade matrices sit 64 bytes apart in one array, and a dynamic
 * uniform offset has to be 256-aligned.
 */
/**
 * Downsample for the reflection's COVERAGE chain.
 *
 * Colour and coverage are a premultiplied PAIR, and a blur has to move them
 * together. Blurring colour alone drags it toward zero wherever it spreads into
 * the empty part of the reflection, while a coverage still sampled sharp says
 * "fully covered" — so the reflection simply got darker as the dial came up.
 *
 * One bilinear tap at the destination texel centre averages the source's 2x2
 * exactly, which is the box filter this wants and the cheapest way to write it.
 */
export const MIRROR_MASK_DOWNSAMPLE_WGSL = /* wgsl */ `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

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
  return textureSampleLevel(src, samp, in.uv, 0.0);
}
`

export function mirrorShadowWgsl(cascades: number): string {
  return /* wgsl */ `
struct LightVP { viewProj: array<mat4x4f, ${cascades}>, };
struct Cascade { index: u32, _p0: u32, _p1: u32, _p2: u32, };
struct MirrorMat {
  model: mat4x4f,
  tint: vec3f, blur: f32,
  frameColor: vec3f, frame: f32,
  size: vec2f, _pad: vec2f,
};

@group(0) @binding(0) var<uniform> lightVP: LightVP;
@group(0) @binding(1) var<uniform> cascade: Cascade;
@group(0) @binding(2) var<uniform> material: MirrorMat;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var q = array<vec2f, 6>(
    vec2f(-0.5, 0.0), vec2f(0.5, 0.0), vec2f(0.5, 1.0),
    vec2f(-0.5, 0.0), vec2f(0.5, 1.0), vec2f(-0.5, 1.0),
  );
  return lightVP.viewProj[cascade.index] * material.model * vec4f(q[vi], 0.0, 1.0);
}
`
}
