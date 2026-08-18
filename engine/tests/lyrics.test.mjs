import { test } from "node:test"
import assert from "node:assert/strict"
// dist, like every other test here: node cannot load a .ts, so importing the
// source made this the one file in the suite that could never run.
import { parseLRC, packLyrics, LYRIC_HEADER, LYRIC_STRIDE, LYRIC_LINES_MAX, LYRICS_FLOATS } from "../dist/shaders/lyrics-api.js"

test("parses timestamped lines, ends at the next line's start", () => {
  const lines = parseLRC("[00:12.00]first line\n[00:17.20]second line\n")
  assert.equal(lines.length, 2)
  assert.equal(lines[0].start, 12)
  assert.equal(lines[0].end, 17.2)
  assert.equal(lines[0].text, "first line")
  assert.equal(lines[1].start, 17.2)
  assert.equal(lines[1].text, "second line")
})

test("the last line holds for ten seconds", () => {
  const lines = parseLRC("[01:00]only line")
  assert.equal(lines[0].start, 60)
  assert.equal(lines[0].end, 70)
})

test("several tags on one line share its text, sorted into place", () => {
  const lines = parseLRC("[00:30.00][00:10.00]chorus\n[00:20.00]verse")
  assert.deepEqual(
    lines.map((l) => [l.start, l.text]),
    [
      [10, "chorus"],
      [20, "verse"],
      [30, "chorus"],
    ],
  )
})

test("offset follows the LRC convention: positive shows lines earlier", () => {
  assert.equal(parseLRC("[offset:+500]\n[00:10.00]shifted")[0].start, 9.5)
  assert.equal(parseLRC("[offset:-500]\n[00:10.00]shifted")[0].start, 10.5)
  // ...and never drives a line before the song exists.
  assert.equal(parseLRC("[offset:+900]\n[00:00.20]early")[0].start, 0)
})

test("an empty-text stamp closes its predecessor without becoming a line", () => {
  const lines = parseLRC("[00:10.00]sung line\n[00:14.00]\n[00:30.00]after the gap")
  assert.equal(lines.length, 2)
  assert.equal(lines[0].end, 14)
  assert.equal(lines[1].start, 30)
})

test("fractional stamps scale by their digit count", () => {
  assert.equal(parseLRC("[00:01.5]a")[0].start, 1.5)
  assert.equal(parseLRC("[00:01.50]a")[0].start, 1.5)
  assert.equal(parseLRC("[00:01.500]a")[0].start, 1.5)
})

test("metadata tags and plain prose are ignored", () => {
  const lines = parseLRC("[ti:Song]\n[ar:Artist]\nno stamp here\n[00:05.00]real")
  assert.equal(lines.length, 1)
  assert.equal(lines[0].text, "real")
})

test("packing writes count, start, end and character count at the stride", () => {
  const out = packLyrics([{ start: 1.5, end: 4, text: "abcde" }])
  assert.equal(out.length, LYRICS_FLOATS)
  assert.equal(out[0], 1)
  assert.equal(out[LYRIC_HEADER], 1.5)
  assert.equal(out[LYRIC_HEADER + 1], 4)
  assert.equal(out[LYRIC_HEADER + 2], 5)
})

test("packing carries each line's atlas rect when the host rasterised text", () => {
  const out = packLyrics(
    [
      { start: 0, end: 1, text: "a" },
      { start: 1, end: 2, text: "b" },
    ],
    [
      [0.1, 0.2, 0.3, 0.25],
      [0.1, 0.25, 0.5, 0.3],
    ],
  )
  const b = LYRIC_HEADER + LYRIC_STRIDE
  assert.deepEqual([...out.slice(b + 4, b + 8)].map((v) => Math.round(v * 100) / 100), [0.1, 0.25, 0.5, 0.3])
  // Without rects the lane stays zero — rzLyricHasText reads that as "timing only".
  const bare = packLyrics([{ start: 0, end: 1, text: "a" }])
  assert.deepEqual([...bare.slice(LYRIC_HEADER + 4, LYRIC_HEADER + 8)], [0, 0, 0, 0])
})

test("packing clamps to the line cap", () => {
  const many = Array.from({ length: LYRIC_LINES_MAX + 40 }, (_, i) => ({ start: i, end: i + 1, text: "x" }))
  const out = packLyrics(many)
  assert.equal(out[0], LYRIC_LINES_MAX)
  const last = LYRIC_HEADER + (LYRIC_LINES_MAX - 1) * LYRIC_STRIDE
  assert.equal(out[last], LYRIC_LINES_MAX - 1)
})
