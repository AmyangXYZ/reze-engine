/**
 * A view transform baked by the renderer a stage was made in — Blender's AgX,
 * Filmic, a Look — so the app shows it as that renderer does rather than
 * through the nearest curve it has (Engine.setViewLut).
 *
 * The cube's input is the exposed scene-linear colour, each channel through
 * log2 between these stops — 2^-12.5 (darker than any display shows) to 2^7
 * (past where every film curve has rolled off) — a texel every
 * (hi - lo)/(n - 1) stops. Its texels are display values, encoded. An exporter
 * fills it by asking its renderer's colour management for exactly these
 * inputs, so the two must agree on the range: it is fixed here, not carried.
 */
export const VIEW_LUT_LOG2: readonly [number, number] = [-12.5, 7]
