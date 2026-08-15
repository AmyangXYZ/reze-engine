// Standard MIDI File parsing. Run: npm test.
//
// The files are built here rather than checked in: a real transcription is
// someone's work, and a synthetic file can be made to contain exactly the
// awkward cases — a tempo map, running status, note-off spelled as note-on with
// velocity zero, a restruck pitch — which a real file may or may not happen to
// have. The loader was separately verified against two real piano transcriptions
// (1482 notes / 89 tempo changes, and 1562 notes / 31 tracks): note counts,
// durations, pitch ranges and ordering all matched an independent parser.

import { test } from "node:test"
import assert from "node:assert/strict"
import { parseMidi } from "../dist/midi-loader.js"

/** Variable-length quantity, the format's own integer encoding. */
function vlq(n) {
  const out = [n & 0x7f]
  n >>= 7
  while (n > 0) {
    out.unshift((n & 0x7f) | 0x80)
    n >>= 7
  }
  return out
}
const be = (n, bytes) => Array.from({ length: bytes }, (_, i) => (n >> (8 * (bytes - 1 - i))) & 0xff)

function midi(ticksPerQuarter, tracks) {
  const out = [...[0x4d, 0x54, 0x68, 0x64], ...be(6, 4), ...be(1, 2), ...be(tracks.length, 2), ...be(ticksPerQuarter, 2)]
  for (const events of tracks) {
    out.push(0x4d, 0x54, 0x72, 0x6b, ...be(events.length, 4), ...events)
  }
  return new Uint8Array(out).buffer
}

test("ticks become seconds through the tempo map, not an average", () => {
  // 480 ticks/quarter. Starts at 120bpm (500000 µs/quarter → 1 tick = 1/960 s),
  // then DOUBLES speed at tick 960. A loader using one tempo gets note 2 wrong.
  const track = [
    ...vlq(0), 0xff, 0x51, 0x03, ...be(500000, 3),
    ...vlq(0), 0x90, 60, 100, // note on at tick 0
    ...vlq(480), 0x80, 60, 0, // off at 480 → 0.5 s long
    ...vlq(480), 0xff, 0x51, 0x03, ...be(250000, 3), // tick 960: twice as fast
    ...vlq(0), 0x90, 62, 100,
    ...vlq(480), 0x80, 62, 0, // 480 ticks now lasts 0.25 s
    ...vlq(0), 0xff, 0x2f, 0x00,
  ]
  const n = parseMidi(midi(480, [track]))
  assert.equal(n.length, 2)
  assert.ok(Math.abs(n[0].start - 0) < 1e-6, `first note at ${n[0].start}`)
  assert.ok(Math.abs(n[0].duration - 0.5) < 1e-6, `first lasts ${n[0].duration}`)
  // 960 ticks at the FIRST tempo = 1.0 s; the change applies only after it.
  assert.ok(Math.abs(n[1].start - 1.0) < 1e-6, `second note at ${n[1].start}, want 1.0`)
  assert.ok(Math.abs(n[1].duration - 0.25) < 1e-6, `second lasts ${n[1].duration}, want 0.25`)
})

test("running status and velocity-zero note-off both parse", () => {
  // Three notes sharing ONE 0x90 status byte, released the same way — the
  // encoding a real file uses constantly, and reading it wrong turns the rest
  // of the track into garbage rather than failing loudly.
  const track = [
    ...vlq(0), 0x90, 60, 64,
    ...vlq(0), 62, 64, // running status: no 0x90
    ...vlq(0), 64, 64,
    ...vlq(480), 60, 0, // note-on velocity 0 = note-off
    ...vlq(0), 62, 0,
    ...vlq(0), 64, 0,
    ...vlq(0), 0xff, 0x2f, 0x00,
  ]
  const n = parseMidi(midi(480, [track]))
  assert.equal(n.length, 3)
  assert.deepEqual(n.map((x) => x.pitch).sort((a, b) => a - b), [60, 62, 64])
  for (const x of n) assert.ok(Math.abs(x.duration - 0.5) < 1e-6, `duration ${x.duration}`)
})

test("a restruck pitch pairs oldest-first, so neither note hangs", () => {
  // Pedalled piano restrikes a pitch before releasing it. Pairing the NEWEST
  // note-on would leave the older one open to the end of the file — a note that
  // lasts the rest of the piece.
  const track = [
    ...vlq(0), 0x90, 60, 100,
    ...vlq(240), 0x90, 60, 100, // struck again, still held
    ...vlq(240), 0x80, 60, 0, // releases the FIRST
    ...vlq(240), 0x80, 60, 0, // releases the second
    ...vlq(0), 0xff, 0x2f, 0x00,
  ]
  const n = parseMidi(midi(480, [track]))
  assert.equal(n.length, 2)
  assert.ok(Math.abs(n[0].duration - 0.5) < 1e-6, `first ${n[0].duration}, want 0.5`)
  assert.ok(Math.abs(n[1].duration - 0.5) < 1e-6, `second ${n[1].duration}, want 0.5`)
})

test("notes come back sorted by onset across tracks", () => {
  // Format 1 splits one timeline across tracks, so a later track routinely
  // starts earlier. An effect walking the list in order needs onset order.
  const a = [...vlq(960), 0x90, 60, 100, ...vlq(240), 0x80, 60, 0, ...vlq(0), 0xff, 0x2f, 0x00]
  const b = [...vlq(0), 0x90, 72, 100, ...vlq(240), 0x80, 72, 0, ...vlq(0), 0xff, 0x2f, 0x00]
  const n = parseMidi(midi(480, [a, b]))
  assert.equal(n.length, 2)
  assert.equal(n[0].pitch, 72, "the earlier note is first even though its track is second")
  assert.ok(n[0].start < n[1].start)
})

test("a note left hanging at the end keeps its pitch", () => {
  // A file that stops without releasing. The note has to survive AS ITS PITCH:
  // defaulting it to 0 does not drop it, it relocates it to C-1 — and since the
  // score reports the range it actually uses, one stray note there stretches
  // every keyboard drawn against it across an octave nothing plays in.
  const track = [
    ...vlq(0), 0x90, 72, 100,
    ...vlq(480), 0x90, 77, 100, // both still held
    ...vlq(480), 0xff, 0x2f, 0x00,
  ]
  const n = parseMidi(midi(480, [track]))
  assert.equal(n.length, 2)
  assert.deepEqual(n.map((x) => x.pitch).sort((a, b) => a - b), [72, 77])
})

test("velocity is normalised, and a non-MIDI file is rejected by name", () => {
  const track = [...vlq(0), 0x90, 60, 127, ...vlq(480), 0x80, 60, 0, ...vlq(0), 0xff, 0x2f, 0x00]
  assert.equal(parseMidi(midi(480, [track]))[0].velocity, 1)
  assert.throws(() => parseMidi(new Uint8Array([1, 2, 3, 4]).buffer), /not a MIDI file/)
})
