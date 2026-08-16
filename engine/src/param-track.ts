/**
 * A material parameter over time.
 *
 * The other half of step 5: dissolve, teleport, a look that changes on a beat.
 * The per-material channel already exists — setStyleParam writes straight into
 * the style uniform — so what was missing was only a way to drive it from the
 * scene clock rather than from whenever a caller happened to fire.
 *
 * WHY THE SAMPLING LIVES HERE, alone in its own file: everything else in this
 * feature needs a GPU, and this does not. Keeping it pure is what lets the
 * interpolation be tested exhaustively in a headless suite, which matters
 * because it is the part with edge cases — the ends, a single key, keys at the
 * same instant.
 *
 * NOT bezier. Bone animation has curves because a VMD carries them and a body
 * needs ease; a material parameter is a dial, and a dial that needs shaping can
 * be shaped by placing more keys. If eased segments are ever wanted, they
 * belong as a property ON a key, not as a second sampler.
 *
 * DETERMINISTIC BY CONSTRUCTION: the value is a pure function of the scene
 * clock, so an offline export stepping frame by frame produces exactly what
 * playback did. Anything read from wall time would not, which is the same rule
 * the score and audio interfaces already follow.
 */

/** A parameter value: a scalar, or a vector for the vec3 params. */
export type ParamValue = number | [number, number, number]

export type ParamKey = {
  /** Seconds on the SCENE clock — the same clock an export steps. */
  t: number
  v: ParamValue
}

const lerp = (a: number, b: number, k: number): number => a + (b - a) * k

/**
 * The value at `t`.
 *
 * Outside the track it HOLDS rather than extrapolating: a parameter that ran
 * off the end of its keys and kept going would leave the scene somewhere its
 * author never described, and the last key is the last thing they said.
 *
 * Keys must be sorted by `t` — setStyleParamTrack sorts on the way in, so this
 * can binary-search rather than scan, and a track with a thousand keys costs
 * the same per frame as one with four.
 */
export function sampleParamTrack(keys: ParamKey[], t: number): ParamValue | null {
  if (keys.length === 0) return null
  if (keys.length === 1 || t <= keys[0].t) return keys[0].v
  const last = keys[keys.length - 1]
  if (t >= last.t) return last.v

  // The last key at or before t.
  let lo = 0
  let hi = keys.length - 1
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1
    if (keys[mid].t <= t) lo = mid
    else hi = mid - 1
  }
  const a = keys[lo]
  const b = keys[lo + 1]
  const span = b.t - a.t
  // Two keys at the same instant are a STEP, and the later one wins. Dividing
  // by the zero span would be a NaN written into a uniform every frame.
  if (span <= 0) return b.v
  const k = (t - a.t) / span
  if (typeof a.v === "number" && typeof b.v === "number") return lerp(a.v, b.v, k)
  // Mixed scalar and vector keys are an authoring mistake, not something to
  // guess at: hold the segment's start rather than inventing a conversion.
  if (typeof a.v === "number" || typeof b.v === "number") return a.v
  return [lerp(a.v[0], b.v[0], k), lerp(a.v[1], b.v[1], k), lerp(a.v[2], b.v[2], k)]
}

/** Whether a freshly sampled value differs from the last one written. Tracks are
 *  evaluated every frame and most of them are flat most of the time, so this is
 *  what keeps a still scene from writing a uniform per parameter per frame. */
export function paramChanged(a: ParamValue | null, b: ParamValue | null): boolean {
  if (a === null || b === null) return a !== b
  if (typeof a === "number" || typeof b === "number") return a !== b
  return a[0] !== b[0] || a[1] !== b[1] || a[2] !== b[2]
}
