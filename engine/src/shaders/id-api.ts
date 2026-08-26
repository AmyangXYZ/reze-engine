// The scene's id attachment, as an author sees it.
//
// ITS OWN MODULE because every mount compiles the author's WHOLE FILE, not just
// the entry point it needs. An effect that masks a glow by character has one
// function reading rzObjectAt and, very often, a lightEmit beside it — and the
// light module has no id texture, so the file failed to compile there with
// "unresolved call target". The gap, not the effect, was the bug.
//
// Only the field module can bind the real thing, and that is not an oversight
// to close later: the id attachment is WRITTEN BY THE SCENE PASS. The light
// emitter runs before it, and the particle, trail and grid steps run inside it,
// so there is no ordering in which any of them could read a finished buffer.
// They get the stub, permanently, and it answers 0 — the reserved nothing.
//
// lights.ts imports this and must not import the composite, which imports
// lights.ts for RzLight. One small module both can depend on is what keeps that
// from being a cycle.

/**
 * Reading the scene's id attachment from a field effect — the consumer the MRT
 * work exists for.
 *
 * A field effect covers the whole screen and has no idea what it is drawing
 * over. These make it addressable: mask a glow to one character, dissolve one
 * material, outline the thing someone selected. Without them the id buffer is
 * written every frame and read by nobody.
 *
 * MULTISAMPLED AND UNRESOLVED, so it is read the way it is written —
 * textureLoad of sample 0, the same rule linearDepth already follows. An
 * averaged id belongs to nothing.
 *
 * When ids are OFF the buffer does not exist, so the accessors are still
 * DECLARED and answer 0 — the reserved nothing. An effect that masks by id then
 * masks nothing at all, which is a scene that renders rather than a shader that
 * will not compile.
 */
export function idApi(on: boolean, group: number, binding: number): string {
  if (!on) {
    return /* wgsl */ `
fn rzObjectAt(uv: vec2f) -> u32 { return 0u; }
fn rzMaterialAt(uv: vec2f) -> u32 { return 0u; }
`
  }
  return /* wgsl */ `
@group(${group}) @binding(${binding}) var _rzIdTex: texture_multisampled_2d<u32>;

/**
 * uv to a texel of the id attachment, AND THE Y FLIP IS THE POINT.
 *
 * An effect's uv is origin BOTTOM-LEFT — shadertoy's convention, and what every
 * mount hands the author. A texture's rows run the other way, and the field
 * shader's own derivation says so: uv.y is 1 - fx.y/h. (No backticks in this
 * comment — the whole block is a TS template literal and one would end it.)
 * Multiplying that uv straight into texel coordinates reads the frame upside
 * down.
 *
 * It did, from the day these accessors were written until the first effect
 * actually called one. Nothing caught it because nothing used it — the comment
 * above this pair said as much: the id buffer was written every frame and read
 * by nobody. The first effect to mask by silhouette got her mirrored top to
 * bottom, which draws as blocky shadow in the wrong half of the frame.
 */
fn _rzIdTexel(uv: vec2f) -> vec2<i32> {
  let sz = vec2f(textureDimensions(_rzIdTex));
  let flipped = vec2f(clamp(uv.x, 0.0, 1.0), 1.0 - clamp(uv.y, 0.0, 1.0));
  return clamp(vec2<i32>(flipped * sz), vec2<i32>(0), vec2<i32>(sz) - vec2<i32>(1));
}

/** Which OBJECT drew this pixel — compare against rzSubjectId(i). 0 = nothing. */
fn rzObjectAt(uv: vec2f) -> u32 {
  return textureLoad(_rzIdTex, _rzIdTexel(uv), 0).y;
}

/** Which MATERIAL drew it, within that object. 0 = nothing. */
fn rzMaterialAt(uv: vec2f) -> u32 {
  return textureLoad(_rzIdTex, _rzIdTexel(uv), 0).x;
}
`
}
