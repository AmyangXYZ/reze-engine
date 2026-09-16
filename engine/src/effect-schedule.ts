/**
 * WHEN an effect is on, and how much of it.
 *
 * The engine evaluates this, not the app, and that is not an arbitrary split.
 * Everything time-driven that lives above the engine has to be ticked by every
 * loop that renders — playback, offline export, a warm-up pass — and the export
 * loop already carries a scar about it: a score-driven effect once exported a
 * still keyboard while the rest of the scene animated, because that loop
 * advanced the audio clock and forgot the score's. A schedule evaluated where
 * the scene clock advances cannot be forgotten by anyone.
 *
 * It is also where the three references put it. Unity's Timeline evaluates clip
 * ease in its playable graph, Unreal's Sequencer evaluates section easing, and
 * Blender's NLA evaluates strip influence — in all three the editor only edits
 * the data. This is that.
 *
 * SECONDS, because that is the engine's clock. The document above works in
 * MMD's 30fps frames and converts on the way in; nothing here knows the render
 * rate, which is what lets a 60Hz preview and a 30 or 60fps export agree about
 * which beat an effect lands on.
 *
 * The model is Blender's NLA strip, minus what needs a strip STACK: no
 * extrapolation modes (outside a window is off, always), no auto-blend (it
 * derives from neighbouring strips, and effects have a lane each), no repeat or
 * scale (an author who wants a cycle can wrap their own clock).
 *
 * PURE, in its own file, for the reason param-track.ts is: everything else in
 * this feature needs a GPU and this needs one number, so the edge cases can be
 * tested exhaustively. They are where it goes wrong, and a wrong one is a beat
 * missed in a video someone posts.
 */

/** One strip on an effect's LANE — and a lane holds as many as you place.
 *  Seconds throughout. */
export type EffectWindow = {
  /** When it comes alive, and where its own clock reads zero. */
  start: number
  /** When it stops. Omitted = it runs to the end of the scene. */
  end?: number
  /** Ramp up over this long from `start`. Omitted or 0 is a HARD CUT, which is
   *  the right answer for a flash and the wrong one for a glow. */
  blendIn?: number
  /** Ramp down over this long back from `end`. Needs an `end` to measure from:
   *  an effect with no end has nothing to ramp toward. */
  blendOut?: number
}

/** What an effect is doing at one instant: a weight for its mounts, and where
 *  its own clock has reached. */
export type EffectState = { weight: number; time: number }

/** Off, and stopped at its own beginning. */
const SILENT: EffectState = { weight: 0, time: 0 }

const clamp01 = (n: number) => (n < 0 ? 0 : n > 1 ? 1 : Number.isFinite(n) ? n : 0)

/**
 * One effect's state at one moment of the scene.
 *
 * `influence` is the level it reaches INSIDE the window — Blender's word and
 * Blender's meaning. The blends ramp toward it rather than toward 1, so a
 * permanently half-strength effect and a scheduled one are the same dial
 * instead of two that can disagree.
 *
 * LINEAR ramps, matching the NLA's. A curve is a better default for light and a
 * worse one for everything else, and there is no control to shape it with — a
 * curve nobody asked for is worse than a plain one. If eased blends are ever
 * wanted they belong as a property ON the window, which is the same call
 * param-track.ts made about bezier keys.
 */
export function effectState(
  windows: readonly EffectWindow[] | null,
  influence: number,
  sceneTime: number,
): EffectState {
  const level = clamp01(influence)
  // UNSCHEDULED RUNS WITH THE SCENE, on the scene's own clock — so an effect
  // nobody has scheduled behaves exactly as it did before windows existed.
  if (!windows || windows.length === 0) return { weight: level, time: sceneTime }

  // A LANE HOLDS MANY STRIPS, which is what lets one effect fire more than
  // once. Each entry restarts the effect's own clock at its own start, so a hit
  // placed at bar 8 and again at bar 24 plays its opening both times rather
  // than resuming halfway through itself.
  //
  // The LATEST start that contains the time wins. Strips on one lane are not
  // meant to overlap — that is the rule every NLE enforces within a track, and
  // it keeps "which strip am I in" a question with one answer — but a document
  // can be hand-edited, and "the one most recently entered" is the reading that
  // matches what you would see if they were laid down in order.
  const window = activeWindow(windows, sceneTime)
  if (!window) return SILENT

  const { start, end } = window
  // The picker above already rejected an empty strip (an end at or before the
  // start, reachable by dragging one edge past the other) and anything the
  // time is outside of, so from here the strip is live and has a real length.

  // ITS OWN CLOCK, from its entry. This is the reason the window owns the clock
  // rather than being a visibility flag beside it: an effect entering at bar 33
  // should play its own opening, not join the scene four minutes in.
  const time = sceneTime - start

  let ramp = 1
  const fadeIn = window.blendIn ?? 0
  if (fadeIn > 0) ramp = Math.min(ramp, time / fadeIn)
  const fadeOut = end !== undefined ? (window.blendOut ?? 0) : 0
  if (fadeOut > 0) ramp = Math.min(ramp, (end! - sceneTime) / fadeOut)

  // The MINIMUM of the two ramps, so blends that overlap degrade instead of
  // break: set both longer than the strip and it becomes a triangle — dimmer
  // than asked for, still smooth, never out of range. Multiplying them would
  // dip toward zero in the middle of a short strip, which is not a fade.
  return { weight: level * clamp01(ramp), time }
}

/**
 * The window that holds `sceneTime`: the latest start that contains it, edges
 * inclusive, an end at or before its start holding nothing. Null outside every
 * window and for a missing lane. effectState plays this window.
 */
export function activeWindow(windows: readonly EffectWindow[] | null, sceneTime: number): EffectWindow | null {
  if (!windows) return null
  let window: EffectWindow | null = null
  for (const w of windows) {
    if (sceneTime < w.start) continue
    if (w.end !== undefined && (w.end <= w.start || sceneTime > w.end)) continue
    if (!window || w.start > window.start) window = w
  }
  return window
}

/** The largest single step a scheduled simulation takes, in seconds. A hitch or
 *  a scrub forward advances it this far and no further, and restarts nothing. */
export const SIM_MAX_STEP = 0.1

/** Where a scheduled effect's simulation stood last frame: the start of the
 *  window it was in, and its own time there. Both null outside every window. */
export type SimClock = { start: number | null; time: number | null }

/**
 * One frame of a scheduled effect's simulation: its particle pool and its grid.
 *
 * Those carry state from one frame to the next, so the window's clock alone
 * cannot place them. A burst scheduled at bar 33 has to start there from an
 * empty pool, and play the same way every time the scene reaches bar 33 —
 * in the editor, on a loop, and in an export.
 *
 * - Entering a window RESTARTS it, from an empty pool and an unseeded grid.
 * - Time going back inside a window restarts it too: a scrub back or a loop
 *   wrapping, where what the pool holds is from later.
 * - Forward, it steps by exactly how far the transport moved, at most
 *   SIM_MAX_STEP. A paused transport steps it by zero, and an export steps it
 *   the same way on every render.
 * - Outside every window it does not step.
 */
export function advanceSim(
  prev: SimClock,
  windows: readonly EffectWindow[] | null,
  sceneTime: number,
): { reset: boolean; step: number; clock: SimClock } {
  const window = activeWindow(windows, sceneTime)
  if (!window) return { reset: false, step: 0, clock: { start: null, time: null } }
  const time = sceneTime - window.start
  const clock = { start: window.start, time }
  if (prev.start !== window.start || prev.time === null || time < prev.time) return { reset: true, step: 0, clock }
  return { reset: false, step: Math.min(SIM_MAX_STEP, time - prev.time), clock }
}

/**
 * A repeating dissolve, in seconds within one cycle.
 *
 * Four moments rather than a duration and a delay: every one of them is a thing
 * you can see happen, and an author tuning this is watching for exactly those
 * four frames.
 */
export interface DissolveCycle {
  period: number
  /** She starts to come apart. */
  breakAt: number
  /** Fully gone. */
  hiddenAt: number
  /** She starts to come back. */
  backAt: number
  /** Whole again. */
  doneAt: number
}

/** The four durations an effect declares, in seconds. */
export type DissolveTimings = { apart: number; gone: number; back: number; whole: number }

/** Each duration and the name an effect gives it, as a dial or as a constant. */
export const DISSOLVE_PARAMS = [
  ["apart", "DISSOLVE_APART"],
  ["gone", "DISSOLVE_GONE"],
  ["back", "DISSOLVE_BACK"],
  ["whole", "DISSOLVE_WHOLE"],
] as const satisfies readonly (readonly [keyof DissolveTimings, string])[]

/**
 * The four durations an effect wrote as constants.
 *
 * Where an effect declares them as `#param` dials instead, these are absent and
 * the live dial stands in — which is what lets an author retime a teleport while
 * watching it.
 */
export function dissolveConstants(wgsl: string): DissolveTimings {
  const one = (name: string): number => {
    const m = new RegExp(`^\\s*const\\s+${name}\\s*(?::\\s*f32\\s*)?=\\s*(-?[\\d.]+)`, "m").exec(wgsl)
    const v = m ? Number(m[1]) : 0
    return Number.isFinite(v) && v > 0 ? v : 0
  }
  return { apart: one("DISSOLVE_APART"), gone: one("DISSOLVE_GONE"), back: one("DISSOLVE_BACK"), whole: one("DISSOLVE_WHOLE") }
}

/**
 * Four durations as the four moments the engine performs, or null when they add
 * up to nothing — a period of zero is not a dissolve that never fires, it is a
 * number everything downstream divides by.
 *
 * The cycle STARTS WHOLE: the wait comes first, so it is the gap you see between
 * teleports rather than a tail nobody can find the start of.
 */
export function dissolveCycleOf(t: DissolveTimings): DissolveCycle | null {
  const apart = Math.max(0, t.apart)
  const gone = Math.max(0, t.gone)
  const back = Math.max(0, t.back)
  const whole = Math.max(0, t.whole)
  if (apart + gone + back + whole <= 0.01) return null
  const breakAt = whole
  const hiddenAt = breakAt + apart
  const backAt = hiddenAt + gone
  const doneAt = backAt + back
  return { period: doneAt, breakAt, hiddenAt, backAt, doneAt }
}

/** How much of her is there at one moment of a cycle, 1 whole and 0 gone. */
export function sampleDissolveCycle(c: DissolveCycle, time: number): number {
  const period = Math.max(c.period, 1e-3)
  const t = time - Math.floor(time / period) * period
  if (t >= c.breakAt && t < c.hiddenAt) return 1 - (t - c.breakAt) / Math.max(c.hiddenAt - c.breakAt, 1e-4)
  if (t >= c.hiddenAt && t < c.backAt) return 0
  if (t >= c.backAt && t < c.doneAt) return (t - c.backAt) / Math.max(c.doneAt - c.backAt, 1e-4)
  return 1
}

/**
 * A scheduled effect's dissolve: whole outside every window, and inside one the
 * cycle on that window's own clock.
 *
 * THE WINDOW OPENS ON THE DEPARTURE. The cycle itself starts whole, which is
 * right for an effect that simply repeats, and wrong for a clip someone placed
 * on a beat: they put it where the teleport happens, not three seconds before
 * it. So the clip enters at `breakAt`, and the wait becomes the tail the motes
 * finish falling in.
 */
export function scheduledDissolve(
  windows: readonly EffectWindow[],
  c: DissolveCycle,
  transport: number,
): number {
  const w = activeWindow(windows, transport)
  return w ? sampleDissolveCycle(c, transport - w.start + c.breakAt) : 1
}
