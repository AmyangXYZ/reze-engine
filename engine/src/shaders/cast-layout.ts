/**
 * The caps the cast buffer is built to, shared by every shader that reads it and
 * by the engine that fills it. Interpolated into the WGSL rather than written
 * twice: the layout arithmetic on both sides has to agree exactly, and two
 * literals that must match are two literals that eventually will not.
 *
 * All three are MINIMUMS. Raising one breaks nothing, because effects read
 * through accessors and loop to the count functions; lowering one does.
 *
 * THEIR OWN FILE, and it imports nothing: cast-api.ts needs them to write the
 * accessors, composite.ts needs them for the rest of its module, and cast-api
 * is spliced into composite. Left where they were, that is an import cycle,
 * and the first thing to touch the cycle throws ReferenceError at module load —
 * the whole engine failing to start on an import order nobody chose.
 */
export const EFFECT_SUBJECTS = 4
export const EFFECT_ANCHORS = 8
export const EFFECT_TRAIL_SAMPLES = 128
/** vec4 slot where the trails begin — after the subjects and the anchors. */
export const EFFECT_TRAIL_BASE = EFFECT_SUBJECTS * 3 + EFFECT_ANCHORS * EFFECT_SUBJECTS * 3
