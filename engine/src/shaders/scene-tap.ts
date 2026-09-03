/**
 * The scene tap: the frame as the scene left it, for effects that bend light.
 *
 * ITS OWN FILE, AND IT IMPORTS NOTHING — the same rule and the same reason as
 * cast-layout.ts. composite.ts already imports from lights.ts, so putting the
 * stub in composite and importing it back into lights closes a cycle, and the
 * first thing to touch that cycle throws "Cannot access RZ_LIGHT_STRUCT_WGSL
 * before initialization" at module load: the whole engine failing to start on
 * an import order nobody chose. Three modules need these strings; none of them
 * may depend on another to get them.
 */

/**
 * rzScene(uv) — the frame as the scene left it.
 *
 * The one thing a foreground could not do: an effect knew WHERE the scene was,
 * through bgWorldPos and the depth tap, but never what COLOUR it was. Anything
 * that RESAMPLES the picture needs this — refraction, heat haze, glass, a
 * pixelated cast, and every CRT trick worth having (barrel curve, the RGB
 * ghost, a torn line dragged sideways) is a read of the frame at a uv other
 * than this pixel's own.
 *
 * WHAT IT IS NOT. The field pass runs BEFORE the composite, so this is the
 * resolved scene — models, stage, ground — and nothing the composite adds
 * afterwards: not the background colour, not a background effect, not the tone
 * map, not the grade. Warping it warps the cast and the floor while the
 * backdrop behind them stays put. That is a real limit, not a bug to route
 * around, and an effect that wants the whole finished picture wants a pass
 * after the composite instead.
 *
 * HDR and LINEAR for the same reason — it is read before the tone map, so
 * values run past 1 and an effect doing display-space arithmetic on them
 * should tone them itself.
 *
 * Legal because of pass order, and only that: the field pass runs after the
 * scene pass has ended and RESOLVED, so this samples a finished texture rather
 * than one being written into. Same reason the depth tap works, and it breaks
 * the same way if either ever moves.
 *
 * ITS OWN SAMPLER, deliberately. castDistanceApi and gridReadApi already both
 * declare one at binding 18, which is legal only while no single entry point
 * references both; borrowing either made every effect that also touches the
 * grid reference two variables at one binding, and WGSL rejects the module.
 * Clamp-to-edge is what stops a warped offset wrapping the far side of the
 * screen into the picture.
 */
export function sceneTapApi(group: number, tex: number, samp: number): string {
  return /* wgsl */ `
@group(${group}) @binding(${tex}) var _rzSceneTex: texture_2d<f32>;
@group(${group}) @binding(${samp}) var _rzSceneSamp: sampler;

/** uv as the mount gives it (y=0 at the BOTTOM, the shadertoy convention the
 *  composite documents) turned into a texture coordinate (y=0 at the top).
 *  Sampling the mount's uv straight returns the scene upside down, which is a
 *  bug that looks like an art direction — so it is done here once rather than in
 *  every effect that reads the frame. */
fn _rzTapUv(uv: vec2f) -> vec2f {
  return clamp(vec2f(uv.x, 1.0 - uv.y), vec2f(0.0), vec2f(1.0));
}

/**
 * How much of this pixel the scene COVERS, 0..1.
 *
 * The half of the frame that colour alone cannot tell you, and the reason a
 * ground that fades to nothing at its rim came back as a hard-edged slab. The
 * scene renders into a PREMULTIPLIED target, so a plane drawn at 48% with a
 * radial falloff stores a dimmer colour, not a lighter one — read the colour and
 * ignore this and you get an opaque dark rectangle where there was a fading
 * sheet. It is also finer than any depth test: depth is binary and a plane
 * writes it across its whole square whatever its alpha does.
 */
fn rzSceneAlpha(uv: vec2f) -> f32 {
  let sz = vec2f(textureDimensions(maskTex));
  let p = clamp(_rzTapUv(uv) * sz, vec2f(0.0), sz - vec2f(1.0));
  return textureLoad(maskTex, vec2<i32>(p), 0).g;
}

/**
 * The scene at uv in LINEAR light, straight rather than premultiplied, with the
 * bloom the composite would add to it. What the view transform takes.
 *
 * Divided back out by coverage, so this is the colour the geometry HAS rather
 * than the colour it contributes. Where nothing was drawn the division is
 * meaningless and the result is black — pair it with rzSceneAlpha, never use it
 * alone.
 */
fn rzScene(uv: vec2f) -> vec3f {
  let p = _rzTapUv(uv);
  let pm = textureSampleLevel(_rzSceneTex, _rzSceneSamp, p, 0.0).rgb;
  let straight = pm / max(rzSceneAlpha(uv), 1e-6);
  let bloom = textureSampleLevel(bloomTex, bloomSamp, p, 0.0).rgb * viewU[1].xyz * viewU[1].w;
  return straight + bloom;
}
/** Alias, matching bgWorldPos and the rest of the bg-prefixed surface. */
fn bgScene(uv: vec2f) -> vec3f { return rzScene(uv); }

/**
 * How far the scene is at ANY pixel — the depth tap, off this fragment's leash.
 *
 * foreground() is handed the depth of its OWN pixel, which answers "is this
 * behind me" and nothing else. Anything that resamples asks a different
 * question: it reads the frame somewhere else and needs to know whether the
 * scene was even drawn THERE. One pixel's depth cannot say, and without it an
 * effect that replaces pixels has to replace all of them — painting the empty
 * sky black because that is what the scene texture holds where nothing drew.
 *
 * NO NEW BINDING. The depth buffer has been in this group since foreground got
 * its argument and linearDepth already inverts it; this only lets an effect name
 * a pixel other than its own. Metres along the view axis, the same units as the
 * depth argument and as rzProject's z, so comparing them is a subtraction.
 *
 * Cleared depth inverts to the far plane, so "nothing here" reads as a very
 * large number rather than as zero.
 */
fn rzSceneDepth(uv: vec2f) -> f32 {
  let sz = vec2f(textureDimensions(depthTex));
  let p = clamp(vec2f(uv.x, 1.0 - uv.y) * sz, vec2f(0.0), sz - vec2f(1.0));
  return linearDepth(vec2<i32>(p));
}

/**
 * The far plane, in the same metres rzSceneDepth answers in.
 *
 * Read straight from the uniform the camera writes it into, NOT re-derived by
 * inverting a cleared depth. That inversion is projB / (1 - projA), whose sign
 * depends on which way round the projection maps z — and when it comes out
 * wrong it does not error, it clamps to the near plane and every rzSceneHit
 * answers false, so the effect silently draws nothing at all.
 */
fn rzSceneFar() -> f32 {
  return max(dofU[2].z, 0.05);
}

/**
 * Was anything actually DRAWN at this uv?
 *
 * The question every effect that resamples has to ask, and the one no author
 * should answer with a constant. The far plane is not a fixed number here — it
 * tracks the camera (roughly its orbit radius times twelve), so a threshold
 * picked by eye is right at one distance and wrong at every other. Guess high
 * and empty sky reads as geometry, and an effect that replaces pixels paints the
 * background black; guess low and the far half of a stage disappears.
 *
 * Compared against the plane itself, with a hair of margin for the precision the
 * inversion loses out there.
 */
fn rzSceneHit(uv: vec2f) -> bool {
  return rzSceneDepth(uv) < rzSceneFar() * 0.995;
}

/**
 * The scene at uv, in the SAME space as the pixel it would replace.
 *
 * rzScene is linear and pre-tone-map. Anything that resamples has to put what it
 * fetched back into display space, and every effect that tried did it by hand —
 * a Reinhard shoulder and a 2.2 gamma, near enough at a glance and wrong
 * everywhere it mattered. The tell is a mosaic or a glitch applied to part of
 * the frame: the treated region reads a shade off from the untreated one beside
 * it, and no amount of tuning closes it, because the engine's curve is Filmic or
 * AgX with the scene's own exposure and grade, not a shoulder.
 *
 * So it is not approximated. This is the composite's own chain — exposure,
 * viewTransform, the grade when the scene has one, then the display gamma —
 * applied to the same sample. Fetch a pixel at its own uv through this and you
 * get the pixel that is already there, so a displacement of zero is invisible
 * and the boundary of any masked effect disappears.
 *
 * The gamma is applied unconditionally rather than behind APPLY_GAMMA: the
 * override is a pipeline constant the field pass need not share, and pow(x, 1)
 * is the identity, so writing it out is both safer and equivalent.
 */
fn rzSceneDisplay(uv: vec2f) -> vec3f {
  var c = max(viewTransform(rzScene(uv) * exp2(viewU[0].x)), vec3f(0.0));
  if (viewU[9].w > 0.5) { c = grade(c); }
  return pow(max(c, vec3f(0.0)), vec3f(viewU[0].y));
}

/**
 * The finished pixel: the scene over its background, exactly as the composite
 * would lay it down. One call, and the answer already carries the ground's
 * fade, its opacity, and the background showing through both.
 *
 * This is what a FILTER wants — anything re-rendering the whole frame on a tube
 * or a low-resolution screen. Reaching for rzSceneDisplay and a depth test
 * instead rebuilds this by hand and gets it wrong in one specific way every
 * time: coverage is not a yes or no, and treating it as one turns every soft
 * edge in the scene into a hard one.
 *
 * NOT INCLUDED: depth of field, background effects, and an equirect background
 * (a 360 dome reads as its average rather than its texture). Foregrounds compose
 * after this in any case.
 */
fn rzSceneFrame(uv: vec2f) -> vec3f {
  let a = clamp(rzSceneAlpha(uv), 0.0, 1.0);
  let bg = viewU[2];
  let bgRgb = bg.rgb * select(0.0, 1.0, bg.w > 0.5);
  return rzSceneDisplay(uv) * a + bgRgb * (1.0 - a);
}

/**
 * The scene's background: rgb is the picked colour in DISPLAY space, and w says
 * what kind it is — 0 transparent (a backdrop or an alpha export is showing
 * through), 1 a flat colour, above 1 an equirect dome.
 *
 * For effects that need to know what sits behind the scene. It is NOT what a
 * background effect drew — those compose in the pass after this one — so an
 * effect that paints over the background with this erases them.
 */
fn rzBackground() -> vec4f { return viewU[2]; }
`
}

/**
 * The same names, for the modules with no finished scene to give.
 *
 * An effect file is spliced into EVERY module it has a mount in, so a water
 * effect with a gridStep compiles its foreground() inside the SIM shader as
 * well — dead code there that nothing calls, but code that still has to
 * resolve. Without this, adding a scene read to any effect that also drives a
 * grid fails to install with "unresolved call target rzScene".
 *
 * Black, not a guess: every one of these modules steps BEFORE the scene pass
 * resolves, so at the moment they run the frame does not exist.
 */
export const SCENE_TAP_STUB = /* wgsl */ `
fn rzScene(uv: vec2f) -> vec3f { return vec3f(0.0); }
fn bgScene(uv: vec2f) -> vec3f { return rzScene(uv); }
/** Far, which in these modules is true: no geometry has been drawn yet. */
fn rzSceneDepth(uv: vec2f) -> f32 { return 1e9; }
fn rzSceneFar() -> f32 { return 1e9; }
fn rzSceneHit(uv: vec2f) -> bool { return false; }
fn rzSceneAlpha(uv: vec2f) -> f32 { return 0.0; }
fn rzSceneDisplay(uv: vec2f) -> vec3f { return vec3f(0.0); }
fn rzSceneFrame(uv: vec2f) -> vec3f { return vec3f(0.0); }
fn rzBackground() -> vec4f { return vec4f(0.0); }
`
