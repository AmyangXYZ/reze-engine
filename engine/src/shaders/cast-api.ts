// The cast, as data — the sibling of the audio and score interfaces, and shaped
// like them: one shared buffer, read through accessors, never touched directly.
//
// WHY THIS FILE EXISTS. There were two of these. The field, grid and lightEmit
// modules read the cast through one implementation; the particle and trail
// modules read the SAME BUFFER through another, written separately, and the
// particle one had no rzAnchor at all — so a particle effect could ask where a
// trail had been but not where a wrist is. Neither was wrong; they had simply
// never been the same code, and the split was invisible until an effect used a
// mount from each family and its own file stopped compiling in one of them.
//
// The two differed only in how the layout reached them: one baked the engine's
// constants, the other took them as a CastLayout. Every caller of that layout
// passed the same five constants, so the parameterisation described a freedom
// that did not exist. Baking them makes this a constant string, which is what
// lets both families share it without either one deciding the shape.
//
// WHAT A HOST MUST SUPPLY. Four names, and deliberately only four:
//
//   _rzCast           the buffer, at whatever binding the module puts it on
//   _rzSlot(i)        the effect's local slot → the scene's, from its alias
//   _rzCastLive()     how many subjects the SCENE holds
//   _rzSubjectMask()  which of them THIS EFFECT applies to, one bit per slot
//
// The third is a host's because the two families genuinely disagree on it: the
// field module reads a count the engine wrote into the view uniform, and the
// particle module scans the buffer, because it has no view uniform to read.
// They agree in value. Unifying them would be a behaviour change to every
// shipped effect for no gain, so the seam stays and is named here instead.
//
// The fourth is a host's because every mount carries it in a uniform of its own
// — the field clock's z, the particle and trail pads, the grid's and the light
// emitter's blocks — and each of those is written once a frame by the engine.

import { EFFECT_ANCHORS, EFFECT_SUBJECTS, EFFECT_TRAIL_BASE, EFFECT_TRAIL_SAMPLES, EFFECT_SUBJECT_VEC4S } from "./cast-layout"

export const CAST_API = /* wgsl */ `
const RZ_SUBJECTS: i32 = ${EFFECT_SUBJECTS};
const RZ_SAMPLES: i32 = ${EFFECT_TRAIL_SAMPLES};
/** The anchor ADDRESS SPACE — how many an effect may declare, not how many it
 *  did. RZ_TRAIL_SLOTS is the per-effect number and is not this one; the two
 *  being one number was the old trail bug. */
const RZ_MAX_ANCHORS: i32 = ${EFFECT_ANCHORS};
const RZ_TRAIL_SAMPLES: i32 = ${EFFECT_TRAIL_SAMPLES};

struct RzSubject {
  /** On the FLOOR, under the body — where a ring or a magic circle belongs. */
  root: vec3f,
  /** At the hips, the middle of the body — where an aura belongs. */
  center: vec3f,
  /** Bounding sphere: xyz centre, w radius. Deliberately generous — cull with it. */
  bounds: vec4f,
  /**
   * How much of this character is still THERE: 1 whole, 0 gone.
   *
   * What setModelDissolve last set on them, which the material pass has already
   * acted on by the time an effect runs — so an effect drawing what LEAVES a
   * dissolving body (sparks, ash, a ghost) reads the same number the body was
   * taken apart with, rather than keeping a clock of its own and hoping the two
   * agree. 1 on a scene that never dissolves anybody.
   */
  dissolve: f32,
  /**
   * Where she LOOKS: a unit direction from between the eyes, in world space.
   * Exact while her eyes are on something (Engine.setEyeTracking); otherwise
   * the motion's own eye turn undone through the range map — an eye bone
   * turns a fraction of the way toward what it looks at, and this gives back
   * the whole of it. An effect that should go where the look goes — a beam,
   * a light — takes this rather than an eye bone's axis, and it holds in
   * either mode. Zero, with looking false, only for a model without eyes.
   */
  gaze: vec3f,
  looking: bool,
  /** False past the end of the cast, and every field is then zero. */
  valid: bool,
}

struct RzAnchor {
  pos: vec3f,
  /** World units per second, from the previous frame. Direction for a trail,
   *  magnitude for anything that should react to how hard someone is moving. */
  vel: vec3f,
  /** The bone's forward axis — which way a foot points, where a head looks. */
  fwd: vec3f,
  /** False when this rig has no such bone. Check it: the alternative is drawing
   *  a hand effect at the world origin on every model that spells it differently. */
  valid: bool,
}

/**
 * How many characters THIS EFFECT is on, up to four.
 *
 * Not how many the scene holds. An effect applies to the whole cast unless the
 * scene says otherwise (Engine.setEffectSubjects), and where it says otherwise
 * this counts only the models named — so an effect aimed at one dancer sees one
 * subject, at index 0, and an author who wrote a loop over the cast gets the
 * targeting for free.
 *
 * Which is the whole design: the FILTER IS IN THE INDEX SPACE, not in the
 * shaders. There is no mask an effect can read, because an effect asking "am I
 * allowed on this one" is an effect that can get the answer wrong, and 18 of the
 * shipped built-ins would have had to be rewritten to ask at all.
 */
fn rzSubjectCount() -> i32 {
  let live = _rzCastLive();
  let mask = _rzSubjectMask();
  var n = 0;
  for (var i = 0; i < live; i++) {
    if ((mask & (1u << u32(i))) != 0u) { n++; }
  }
  return n;
}

/** This effect's i-th subject → the scene's slot, or -1 past its own count.
 *  Every accessor below goes through it, which is what keeps one effect's
 *  subject 1 from being another's. */
fn _rzSubjectSlot(i: i32) -> i32 {
  if (i < 0) { return -1; }
  let live = _rzCastLive();
  let mask = _rzSubjectMask();
  var n = 0;
  for (var s = 0; s < live; s++) {
    if ((mask & (1u << u32(s))) == 0u) { continue; }
    if (n == i) { return s; }
    n++;
  }
  return -1;
}

/** Which model this is, stable across a scene — for per-subject variation. */
fn rzSubjectId(i: i32) -> u32 {
  let g = _rzSubjectSlot(i);
  if (g < 0) { return 0u; }
  return u32(_rzCast[g * ${EFFECT_SUBJECT_VEC4S} + 1].w);
}

fn rzSubject(i: i32) -> RzSubject {
  var s: RzSubject;
  let g = _rzSubjectSlot(i);
  s.valid = g >= 0;
  if (!s.valid) { return s; }
  let b = g * ${EFFECT_SUBJECT_VEC4S};
  s.root = _rzCast[b].xyz;
  s.dissolve = _rzCast[b].w;
  s.center = _rzCast[b + 1].xyz;
  s.bounds = _rzCast[b + 2];
  s.gaze = _rzCast[b + 3].xyz;
  s.looking = _rzCast[b + 3].w > 0.5;
  return s;
}

/**
 * Where a named bone is, this frame.
 *
 * The slot is the author's own: the Nth #anchor in their file, in the order
 * they wrote them. _rzSlot turns that into the scene's address, which is what
 * keeps two effects that both anchor to a wrist from reading each other's.
 */
fn rzAnchor(subject: i32, slot: i32) -> RzAnchor {
  var a: RzAnchor;
  a.valid = false;
  let g = _rzSlot(slot);
  let s = _rzSubjectSlot(subject);
  if (s < 0 || g < 0 || g >= RZ_MAX_ANCHORS) { return a; }
  let b = ${EFFECT_SUBJECTS * EFFECT_SUBJECT_VEC4S} + (g * ${EFFECT_SUBJECTS} + s) * 3;
  a.valid = _rzCast[b].w > 0.5;
  a.pos = _rzCast[b].xyz;
  a.vel = _rzCast[b + 1].xyz;
  a.fwd = _rzCast[b + 2].xyz;
  return a;
}

/**
 * How many samples of a path are recorded — 0 for an anchor that asked for no
 * trail, and for one that has not moved yet.
 *
 * Bounded by the anchor cap, NOT by how many anchors asked for a trail. Those
 * are different index spaces: storage is addressed by anchor slot, so an
 * untrailed #anchor followed by a trailed one put the trail at index 1 with a
 * bound of 1 and rzTrail returned zero — a ribbon that silently did not draw.
 */
fn rzTrailCount(subject: i32, slot: i32) -> i32 {
  let g = _rzSlot(slot);
  let s = _rzSubjectSlot(subject);
  if (s < 0 || g < 0 || g >= RZ_MAX_ANCHORS) { return 0; }
  return i32(_rzCast[${EFFECT_SUBJECTS * EFFECT_SUBJECT_VEC4S} + (g * ${EFFECT_SUBJECTS} + s) * 3 + 2].w);
}

/** Sample i of a path: xyz where it was, w how many seconds ago. i = 0 is now. */
fn rzTrail(subject: i32, slot: i32, i: i32) -> vec4f {
  let n = rzTrailCount(subject, slot);
  if (i < 0 || i >= n) { return vec4f(0.0); }
  let base = ${EFFECT_TRAIL_BASE} + (_rzSlot(slot) * ${EFFECT_SUBJECTS} + _rzSubjectSlot(subject)) * RZ_TRAIL_SAMPLES;
  return _rzCast[base + i];
}
`

/**
 * Which subjects this effect applies to, as WGSL — the second of the two names a
 * mount owes CAST_API (`_rzCastLive` is the first, and a mount with a view
 * uniform already has it from EFFECT_SCENE_API).
 *
 * One bit per cast slot. An EXPRESSION rather than a value: it resolves per
 * draw, out of a uniform the engine writes once a frame, so a module that baked
 * it would stop following a target being changed or a model being hidden.
 */
export const subjectMaskApi = (mask: string): string => /* wgsl */ `
fn _rzSubjectMask() -> u32 { return ${mask}; }
`

/** Every subject the scene has. For a mount that is not an effect — the
 *  composite's own module, where nothing user-written lands and the accessors
 *  exist only so the shared API compiles — and the value the engine writes for
 *  an effect the scene has not aimed at anybody in particular. */
export const CAST_MASK_ALL = "0xfu"
