// Lyrics, as data — the fourth timing interface beside audio, the score and
// the lights, and shaped like them: one shared buffer, read through accessors,
// never touched directly.
//
// What an effect gets is the TIMING of the words: which line is live at the
// scene clock, how far through it is, when the next one lands. That is what
// karaoke bars, per-line pulses and line-change bursts are made of, and it is
// export-safe by construction because it reads the same clock everything else
// does. Drawing the TEXT itself is a renderer concern with its own recorded
// design (Canvas2D rasterisation — CJK rules out glyph atlases) and rides on
// top of this, not inside it.
//
// LAYOUT. A 4-float header (count, then padding that keeps the records
// vec4-aligned), then LYRIC_LINES_MAX records of 4 floats:
//
//   [0] start seconds   [1] end seconds   [2] character count   [3] reserved
//
// FIXED SIZE, unlike the score: a song carries tens of lines, not thousands
// of notes, so capping at 256 costs 4 KB and buys the property that setLyrics
// is a buffer write — the buffer identity never changes, so nothing ever has
// to re-bind for lyrics arriving late.

export const LYRIC_LINES_MAX = 256
export const LYRIC_HEADER = 4
export const LYRIC_STRIDE = 4
export const LYRICS_FLOATS = LYRIC_HEADER + LYRIC_LINES_MAX * LYRIC_STRIDE

export type LyricLine = {
  /** Seconds on the scene clock. */
  start: number
  /** Seconds; a parser that has no better answer uses the next line's start. */
  end: number
  text: string
}

/**
 * Parse an .lrc file: `[mm:ss.xx]` tags (several per line share the text),
 * an optional `[offset:±ms]` tag, blank-text tags kept as instrumental gaps'
 * end markers. Lines come out sorted; each line's end is the next line's
 * start, and the last line gets a ten-second hold.
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
      stamped.push({ start: parseInt(t[1], 10) * 60 + parseInt(t[2], 10) + frac + offset, text })
    }
  }
  stamped.sort((a, b) => a.start - b.start)
  const lines: LyricLine[] = []
  for (let i = 0; i < stamped.length; i++) {
    // An empty-text stamp is an .lrc idiom for "the previous line ends here";
    // it closes its predecessor and is not a line of its own.
    if (stamped[i].text === "") continue
    const next = stamped[i + 1]
    lines.push({
      start: stamped[i].start,
      end: next ? next.start : stamped[i].start + 10,
      text: stamped[i].text,
    })
  }
  return lines
}

/** Fill the shared buffer's floats from parsed lines, clamped to the cap. */
export function packLyrics(lines: LyricLine[]): Float32Array<ArrayBuffer> {
  const out = new Float32Array(new ArrayBuffer(LYRICS_FLOATS * 4))
  const n = Math.min(lines.length, LYRIC_LINES_MAX)
  out[0] = n
  for (let i = 0; i < n; i++) {
    const b = LYRIC_HEADER + i * LYRIC_STRIDE
    out[b] = lines[i].start
    out[b + 1] = lines[i].end
    out[b + 2] = lines[i].text.length
  }
  return out
}

/** The rzLyric* accessors, with the buffer declared at the given binding. */
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
