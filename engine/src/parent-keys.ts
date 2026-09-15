/**
 * WHO a model hangs from, over time — MMD's 外部親 keyed on the timeline.
 *
 * A throw is three holds on one object: in her right hand, in the air, in his
 * left hand. Each is a KEY: from its time until the next key's, the model rides
 * that bone with that offset, or stands on its own at that placement. A key
 * switches at its time, or, marked `tween`, is arrived at gradually: across the
 * time since the previous key the model stands free at the blend of the two
 * placements. That is how a flight is a run of free keys and a catch lands in a
 * moving hand.
 *
 * SECONDS on the transport clock, like effect windows, so a 60Hz preview and a
 * 30 or 60fps export switch on the same frame.
 *
 * PURE, in its own file, for the reason effect-schedule.ts is: the boundary is
 * where it goes wrong, and finding the key in force needs no GPU to test.
 */

import type { Quat, Vec3 } from "./math"

/** One hold in a model's parent track. See Engine.setModelParentKeys. */
export type ModelParentKey = {
  /** Transport seconds this hold starts at. The first key also holds before it. */
  time: number
  /** The parent's model key, or null to stand on its own. */
  parent: string | null
  /** Bone on the parent. Omitted rides 全ての親; a name the rig lacks rides its root. */
  bone?: string
  /** With a parent, the offset in the bone's space. Without one, where the model stands. */
  position?: Vec3
  rotation?: Quat
  /** Arrive at this key gradually, from the previous key's placement, rather
   *  than switching at its time. */
  tween?: boolean
}

/** How far past the clock a key may sit and still be reached. The transport
 *  clock is a sum of frame deltas, so a key at 4s reads 3.9999999 at 60fps; a
 *  millisecond is far below any frame and far above that drift. */
export const PARENT_KEY_TOLERANCE = 1e-3

/**
 * The key in force at `time`: the last one whose time has been reached, or the
 * first key before any has. `keys` must be sorted by time. -1 for no keys.
 */
export function parentKeyIndex(keys: readonly { time: number }[], time: number): number {
  if (keys.length === 0) return -1
  let i = 0
  while (i + 1 < keys.length && keys[i + 1].time <= time + PARENT_KEY_TOLERANCE) i++
  return i
}

/**
 * The key in force at `time`, and how far the placement has travelled toward
 * the next key: 0 unless that key tweens, else the fraction of the way from
 * this key's time to its.
 */
export function parentKeySpan(
  keys: readonly { time: number; tween?: boolean }[],
  time: number,
): { index: number; toward: number } {
  const index = parentKeyIndex(keys, time)
  const next = keys[index + 1]
  if (index < 0 || !next?.tween) return { index, toward: 0 }
  const start = keys[index].time
  const span = next.time - start
  if (span <= 0 || time <= start) return { index, toward: 0 }
  return { index, toward: Math.min(1, (time - start) / span) }
}
