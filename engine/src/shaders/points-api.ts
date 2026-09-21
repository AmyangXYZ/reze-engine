// Named points on the models, for an effect that puts something at each one.
//
//     #points flame
//
// gives the effect every bone whose name starts with `flame`, on every model in
// the scene — a stage, its parts, a prop, a character — as RzPoint: where the
// bone stands and where its tail points, in world space, this frame. A candle
// stage carries a bone per wick; a flame effect draws one flame on each, sized
// and stood up by the tail.
//
// BONES, because a bone is MMD's own word for a named point in a model. A stage
// that brings its candles as bones needs no side file, a hand-made stage can
// add them in any PMX editor, and a candle a character carries moves with the
// hand that holds it.
//
// Not the cast's anchors: those name up to sixteen bones on four characters,
// one bone per name. A stage is not a character, and fifty wicks share a
// prefix rather than each having a name worth declaring.

/** How many points one effect can read. A candlelit hall is fifty. */
export const MAX_EFFECT_POINTS = 256

/** Floats in a points buffer: a vec4 header (count), then two vec4s a point. */
export const POINTS_FLOATS = 4 + MAX_EFFECT_POINTS * 8

const STRUCT = /* wgsl */ `
/** One named point: where the bone stands, and where its tail points to. */
struct RzPoint {
  pos: vec3f,
  tip: vec3f,
}
`

/**
 * rzPointCount / rzPoint. `bound` is false in every module that cannot read the
 * buffer — the author's whole file is spliced into each module it has a mount
 * in, so the names must resolve there too, and there they read as no points.
 */
export function pointsApi(bound: boolean, group = 0, binding = 0): string {
  if (!bound) {
    return (
      STRUCT +
      /* wgsl */ `
fn rzPointCount() -> u32 { return 0u; }
fn rzPoint(i: u32) -> RzPoint { var p: RzPoint; return p; }
`
    )
  }
  return (
    STRUCT +
    /* wgsl */ `
@group(${group}) @binding(${binding}) var<storage, read> _rzPoints: array<vec4f>;

/** How many named points the scene has for this effect, this frame. */
fn rzPointCount() -> u32 { return min(u32(_rzPoints[0].x), ${MAX_EFFECT_POINTS}u); }

/** Point i, world space. */
fn rzPoint(i: u32) -> RzPoint {
  var p: RzPoint;
  p.pos = _rzPoints[1u + i * 2u].xyz;
  p.tip = _rzPoints[2u + i * 2u].xyz;
  return p;
}
`
  )
}
