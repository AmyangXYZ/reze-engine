// Lyrics, as data — the fourth timing interface beside audio, the score and
// the lights, and shaped like them: one shared buffer, read through accessors,
// never touched directly.
//
// An effect gets the TIMING of the words — which line is live at the scene
// clock, how far through it is — and, in the field module, the words
// themselves: the host rasterises each line once (Canvas2D; CJK rules out
// glyph atlases) into a fixed atlas the effect samples through rzLyricText.
// The look — fill, outline, wipe, motion — is the effect author's, in WGSL.
//
// LAYOUT. A 4-float header, then LYRIC_LINES_MAX records of 8 floats:
//
//   header [0] line count   [1] the widest line's rect ratio   [2..3] padding,
//     which keeps the records vec4-aligned
//   record [0] start s   [1] end s   [2] character count   [3] reserved
//          [4..7] atlas rect: u0, vTop, u1, vBottom
//
// FIXED SIZE, unlike the score: a song carries tens of lines, not thousands
// of notes, so capping at 256 costs 8 KB and buys the property that setLyrics
// is a buffer write — the buffer identity never changes, so nothing ever has
// to re-bind for lyrics arriving late. The atlas holds the same property by
// being allocated once at a fixed size (LYRIC_ATLAS_W × LYRIC_ATLAS_H).

export const LYRIC_LINES_MAX = 256
export const LYRIC_HEADER = 4
export const LYRIC_STRIDE = 8
export const LYRICS_FLOATS = LYRIC_HEADER + LYRIC_LINES_MAX * LYRIC_STRIDE

/** Bounds on the line atlas the host packs rasterised lines into. It is sized
 *  to the track that arrives rather than allocated at the maximum: a scene with
 *  no lyrics carries a 1×1 placeholder, and a song's atlas is as tall as its
 *  own lines need. 8192 is the smallest texture dimension WebGPU guarantees.
 *
 *  WIDTH IS WHAT DECIDES SHARPNESS on a long line. A lyric spans most of the
 *  frame — the shipped effect leaves 6% a side — so at 4K it is drawn across
 *  some 3400 pixels, and 2048 could only ever be stretched to cover them. That
 *  is the stair-stepped text on a 4K render, and no row height fixes it: the
 *  line is already as wide as the sheet. 1080p never showed it, because 1690
 *  pixels of line fit inside 2048. The atlas is r8unorm, one byte a texel, and
 *  only ever as wide as its own longest line — so this ceiling costs nothing
 *  until a song actually needs it. */
export const LYRIC_ATLAS_MAX_W = 4096
export const LYRIC_ATLAS_MAX_H = 8192

export type LyricLine = {
  /** Seconds on the scene clock. */
  start: number
  /** Seconds; a parser that has no better answer uses the next line's start. */
  end: number
  text: string
}

/** Where a rasterised line sits in the atlas: u0, vTop, u1, vBottom, in 0..1. */
export type LyricRect = [number, number, number, number]

/**
 * Parse an .lrc file: `[mm:ss.xx]` tags (several per line share the text),
 * an optional `[offset:±ms]` tag, blank-text tags kept as instrumental gaps'
 * end markers. Lines come out sorted; each line's end is the next LATER
 * stamp's start, and the last line gets a ten-second hold. Several lines on
 * one stamp — an original and its translation, the bilingual .lrc idiom —
 * stay in file order and share the one window, so an effect finds a
 * translation as the consecutive lines whose start equals the live line's.
 * The offset follows the LRC convention: positive shows lines EARLIER — the
 * knob to turn when the words feel late against this particular rip.
 */
export function parseLRC(source: string): LyricLine[] {
  let offset = 0
  const stamped: { start: number; text: string }[] = []
  for (const raw of source.split(/\r?\n/)) {
    const off = /^\s*\[offset:\s*([+-]?\d+)\s*\]/i.exec(raw)
    if (off) {
      offset = parseInt(off[1], 10) / 1000
      continue
    }
    const tags = [...raw.matchAll(/\[(\d+):(\d{1,2})(?:[.:](\d{1,3}))?\]/g)]
    if (tags.length === 0) continue
    const text = raw.slice(tags[tags.length - 1].index! + tags[tags.length - 1][0].length).trim()
    for (const t of tags) {
      const frac = t[3] ? parseInt(t[3], 10) / 10 ** t[3].length : 0
      stamped.push({ start: Math.max(0, parseInt(t[1], 10) * 60 + parseInt(t[2], 10) + frac - offset), text })
    }
  }
  stamped.sort((a, b) => a.start - b.start)
  const lines: LyricLine[] = []
  for (let i = 0; i < stamped.length; i++) {
    // An empty-text stamp is an .lrc idiom for "the previous line ends here";
    // it closes its predecessor and is not a line of its own.
    if (stamped[i].text === "") continue
    // The window closes at the next stamp that is actually LATER. A stamp
    // shared by an original and its translation is one moment; closing the
    // first line at the second's start gave it a zero-length window no clock
    // ever fell inside, and only the translation was ever drawn.
    let j = i + 1
    while (j < stamped.length && stamped[j].start <= stamped[i].start) j++
    const next = stamped[j]
    lines.push({
      start: stamped[i].start,
      end: next ? next.start : stamped[i].start + 10,
      text: stamped[i].text,
    })
  }
  return lines
}

/** Fill the shared buffer's floats from parsed lines, clamped to the cap. */
export function packLyrics(lines: LyricLine[], rects?: LyricRect[]): Float32Array<ArrayBuffer> {
  const out = new Float32Array(new ArrayBuffer(LYRICS_FLOATS * 4))
  const n = Math.min(lines.length, LYRIC_LINES_MAX)
  out[0] = n
  // THE WIDEST LINE IN THE TRACK, which is the one number an effect laying the
  // words out cannot work out for itself: finding it means reading every rect,
  // and a field shader would be doing that at every pixel of a 4K frame to
  // answer something that changes only when the track does. Carried as the
  // rect's own ratio rather than as an aspect, so it is a property of the pack
  // and not of whatever the atlas happened to be sized at; rzLyricWidest turns
  // it into the same number rzLyricAspect answers with.
  let widest = 0
  for (let i = 0; i < n; i++) {
    const b = LYRIC_HEADER + i * LYRIC_STRIDE
    out[b] = lines[i].start
    out[b + 1] = lines[i].end
    out[b + 2] = lines[i].text.length
    const r = rects?.[i]
    if (r) {
      out[b + 4] = r[0]
      out[b + 5] = r[1]
      out[b + 6] = r[2]
      out[b + 7] = r[3]
      const h = r[3] - r[1]
      if (h > 0) widest = Math.max(widest, (r[2] - r[0]) / h)
    }
  }
  out[1] = widest
  return out
}

/** The rzLyric* timing accessors, with the buffer declared at the given binding. */
export function lyricsApi(group: number, binding: number): string {
  return /* wgsl */ `
@group(${group}) @binding(${binding}) var<storage, read> _rzLyrics: array<f32>;

/** Lines in the lyric track; 0 when none is loaded, which every accessor
 *  below tolerates by answering zero rather than reading past the end. */
fn rzLyricCount() -> i32 { return i32(_rzLyrics[0]); }

fn rzLyricStart(i: i32) -> f32 {
  if (i < 0 || i >= rzLyricCount()) { return 0.0; }
  return _rzLyrics[${LYRIC_HEADER} + i * ${LYRIC_STRIDE}];
}

fn rzLyricEnd(i: i32) -> f32 {
  if (i < 0 || i >= rzLyricCount()) { return 0.0; }
  return _rzLyrics[${LYRIC_HEADER} + i * ${LYRIC_STRIDE} + 1];
}

/** Characters in line i — the number a per-character sweep divides by. */
fn rzLyricChars(i: i32) -> f32 {
  if (i < 0 || i >= rzLyricCount()) { return 0.0; }
  return _rzLyrics[${LYRIC_HEADER} + i * ${LYRIC_STRIDE} + 2];
}

/** Where line i sits in the lyric atlas: u0, vTop, u1, vBottom. Zero when the
 *  host never rasterised text — check with rzLyricHasText. */
fn rzLyricRect(i: i32) -> vec4f {
  if (i < 0 || i >= rzLyricCount()) { return vec4f(0.0); }
  let b = ${LYRIC_HEADER} + i * ${LYRIC_STRIDE};
  return vec4f(_rzLyrics[b + 4], _rzLyrics[b + 5], _rzLyrics[b + 6], _rzLyrics[b + 7]);
}

fn rzLyricHasText(i: i32) -> bool {
  let r = rzLyricRect(i);
  return r.z > r.x;
}

/** The line live at time t, or -1 between lines and outside the track. */
fn rzLyricIndex(t: f32) -> i32 {
  let n = rzLyricCount();
  for (var i = 0; i < n; i = i + 1) {
    if (t >= rzLyricStart(i) && t < rzLyricEnd(i)) { return i; }
  }
  return -1;
}

/** How far through line i the clock is, 0..1 — the karaoke sweep. */
fn rzLyricProgress(i: i32, t: f32) -> f32 {
  let s = rzLyricStart(i);
  let e = rzLyricEnd(i);
  if (e <= s) { return 0.0; }
  return clamp((t - s) / (e - s), 0.0, 1.0);
}
`
}

/**
 * The text half — field module only, where the atlas is bound. uv is 0..1
 * across LINE i's own box, y-up like everything else; the return is glyph
 * coverage. textureSampleLevel, so it is legal after any branch.
 */
export function lyricsTextApi(group: number, texBinding: number, samplerName: string): string {
  return /* wgsl */ `
@group(${group}) @binding(${texBinding}) var _rzLyricTex: texture_2d<f32>;

fn rzLyricText(i: i32, uv: vec2f) -> f32 {
  let r = rzLyricRect(i);
  if (r.z <= r.x || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 0.0; }
  let at = vec2f(mix(r.x, r.z, uv.x), mix(r.w, r.y, uv.y));
  return textureSampleLevel(_rzLyricTex, ${samplerName}, at, 0.0).r;
}

/** Line i's width over its height as rasterised — size a box with it so the
 *  glyphs keep their proportions on any canvas. */
fn rzLyricAspect(i: i32) -> f32 {
  let r = rzLyricRect(i);
  let h = r.w - r.y;
  if (h <= 0.0) { return 1.0; }
  let dim = vec2f(textureDimensions(_rzLyricTex));
  return ((r.z - r.x) * dim.x) / (h * dim.y);
}

/**
 * THE WIDEST LINE IN THE TRACK, on the same scale rzLyricAspect answers on.
 *
 * What it is for: setting the type so the longest line in the song fits the
 * frame, once, for every line — which is the only way a column of lyrics can be
 * bounded by the frame without lines changing size down the page. Sizing each
 * line to its own width instead sets a verse smaller than the interjection above
 * it, for a reason the reader cannot see.
 *
 * 0 before the lines are rasterised, and 0 with no track: there is no widest
 * line yet, and a cap derived from it would be a cap on nothing. Guard it.
 */
fn rzLyricWidest() -> f32 {
  let dim = vec2f(textureDimensions(_rzLyricTex));
  return _rzLyrics[1] * dim.x / dim.y;
}

/**
 * Line i's box in ATLAS TEXELS. Divide by the size you draw it at to learn
 * whether you are magnifying or minifying, which is what an edge-sharpening
 * step needs to know — and the honest way to get it, since a derivative
 * builtin is illegal after the branches an effect of this kind opens with.
 */
fn rzLyricPixels(i: i32) -> vec2f {
  let r = rzLyricRect(i);
  return vec2f(r.z - r.x, r.w - r.y) * vec2f(textureDimensions(_rzLyricTex));
}
`
}
