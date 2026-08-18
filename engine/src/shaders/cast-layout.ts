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
/**
 * Bone anchors, for the WHOLE SCENE rather than per effect — this is an address
 * space that every installed effect draws slots from, and an effect asking for
 * one the table has already given away is told so and has it dropped.
 *
 * 8 was reachable, and quietly: two effects each wanting two hands, two feet and
 * a head is ten, so the second one silently lost its ribbons to a diagnostic
 * nobody was reading. 16 is double the headroom for 131KB of storage buffer,
 * where 8 cost 66KB.
 *
 * The cost really is only that. The per-frame upload is bounded by the last
 * TRAILED slot, not by this cap (see the writeBuffer in updateCastBuffer), so a
 * scene using three anchors uploads three anchors' worth whatever this says.
 * The CPU-side path rings are keyed by (model, slot) and allocated on use. And
 * effects never index this directly — they loop to rzTrailCount / rzSubjectCount
 * and read through rzAnchor/rzTrail, both of which bounds-check against it.
 *
 * It stays a compile-time constant because the accessors in cast-api.ts are
 * interpolated into every effect module as WGSL literals; making it dynamic
 * means resizing the buffer and recompiling every installed effect whenever the
 * scene's anchor count grows, which is a different feature from raising a number
 * that was never load-bearing.
 */
export const EFFECT_ANCHORS = 16
/**
 * Path samples kept per trailed anchor — 128 at the 60Hz sampling rate is a
 * ~2.1 second ribbon.
 *
 * Briefly 256, and reverted with the reason, because the reason is the useful
 * part: a ribbon's cost is not its geometry, it is its FRAGMENTS. It is a wide
 * translucent strip blended additively into the HDR target at the pass's sample
 * count, and it overlaps itself — so its cost tracks the screen AREA it covers,
 * and a twice-as-long ribbon covers roughly twice as much.
 *
 * That lands very differently on the two backends. Overdraw and blend bandwidth
 * are what a tile-based GPU pays for most and what an immediate-mode desktop GPU
 * absorbs, which is why ribbons were reported as costing far more on Safari than
 * on Chrome for the same scene. Doubling this doubles the one thing already
 * known to be the bottleneck there.
 *
 * Raising it is still SAFE — effects loop to rzTrailCount and nothing breaks —
 * it is simply not cheap, and the cost shows up on the slower of the two
 * browsers rather than the one it would be measured on.
 */
export const EFFECT_TRAIL_SAMPLES = 128
/** vec4 slot where the trails begin — after the subjects and the anchors. */
export const EFFECT_TRAIL_BASE = EFFECT_SUBJECTS * 3 + EFFECT_ANCHORS * EFFECT_SUBJECTS * 3
