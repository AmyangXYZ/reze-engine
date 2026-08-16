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
// WHAT A HOST MUST SUPPLY. Three names, and deliberately only three:
//
//   _rzCast          the buffer, at whatever binding the module puts it on
//   _rzSlot(i)       the effect's local slot → the scene's, from its alias
//   rzSubjectCount() how many subjects are live
//
// The last is a host's because the two families genuinely disagree on it: the
// field module reads a count the engine wrote into the view uniform, and the
// particle module scans the buffer, because it has no view uniform to read.
// They agree in value. Unifying them would be a behaviour change to every
// shipped effect for no gain, so the seam stays and is named here instead.

import { EFFECT_ANCHORS, EFFECT_SUBJECTS, EFFECT_TRAIL_BASE, EFFECT_TRAIL_SAMPLES } from "./cast-layout"

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

/** Which model this is, stable across a scene — for per-subject variation. */
fn rzSubjectId(i: i32) -> u32 {
  if (i < 0 || i >= rzSubjectCount()) { return 0u; }
  return u32(_rzCast[i * 3 + 1].w);
}

fn rzSubject(i: i32) -> RzSubject {
  var s: RzSubject;
  s.valid = i >= 0 && i < rzSubjectCount();
  if (!s.valid) { return s; }
  let b = i * 3;
  s.root = _rzCast[b].xyz;
  s.center = _rzCast[b + 1].xyz;
  s.bounds = _rzCast[b + 2];
  return s;
}

/**
 * Where a named bone is, this frame.
 *
 * The slot is the author's own: the Nth @anchor in their file, in the order
 * they wrote them. _rzSlot turns that into the scene's address, which is what
 * keeps two effects that both anchor to a wrist from reading each other's.
 */
fn rzAnchor(subject: i32, slot: i32) -> RzAnchor {
  var a: RzAnchor;
  a.valid = false;
  let g = _rzSlot(slot);
  if (subject < 0 || subject >= rzSubjectCount() || g < 0 || g >= RZ_MAX_ANCHORS) { return a; }
  let b = ${EFFECT_SUBJECTS * 3} + (g * ${EFFECT_SUBJECTS} + subject) * 3;
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
 * untrailed @anchor followed by a trailed one put the trail at index 1 with a
 * bound of 1 and rzTrail returned zero — a ribbon that silently did not draw.
 */
fn rzTrailCount(subject: i32, slot: i32) -> i32 {
  let g = _rzSlot(slot);
  if (subject < 0 || subject >= rzSubjectCount() || g < 0 || g >= RZ_MAX_ANCHORS) { return 0; }
  return i32(_rzCast[${EFFECT_SUBJECTS * 3} + (g * ${EFFECT_SUBJECTS} + subject) * 3 + 2].w);
}

/** Sample i of a path: xyz where it was, w how many seconds ago. i = 0 is now. */
fn rzTrail(subject: i32, slot: i32, i: i32) -> vec4f {
  let n = rzTrailCount(subject, slot);
  if (i < 0 || i >= n) { return vec4f(0.0); }
  let base = ${EFFECT_TRAIL_BASE} + (_rzSlot(slot) * ${EFFECT_SUBJECTS} + subject) * RZ_TRAIL_SAMPLES;
  return _rzCast[base + i];
}
`
