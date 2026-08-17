// The score buffer's layout contract. Run: npm test.
//
// The shader reads header fields at hardcoded indices and the engine writes them
// at hardcoded indices, in two different files. Nothing connects the two but
// agreement, and a disagreement is silent: notes would simply appear at the
// wrong time, or a key would never light, with no error anywhere. So the layout
// is pinned here — reorder the header and this fires rather than the effect
// quietly going wrong.

import { test } from "node:test"
import assert from "node:assert/strict"
import { midiApi, MIDI_HEADER, MIDI_KEYS, MIDI_NOTES, MIDI_STRIDE } from "../dist/shaders/midi-api.js"

const wgsl = midiApi(0, 5)

test("the key map sits directly after the header, and notes after it", () => {
  assert.equal(MIDI_NOTES, MIDI_HEADER + MIDI_KEYS)
  assert.equal(MIDI_KEYS, 128, "the MIDI pitch range, not a cap we chose")
  assert.equal(MIDI_STRIDE, 4, "start, duration, pitch, velocity")
})

test("header fields are read where the engine writes them", () => {
  // engine.ts: payload[0]=count, [1]=low, [2]=high, [5]=duration, [6]=release;
  // setMidiTime writes time+playing at float 3.
  for (const [fn, index] of [
    ["rzNoteCount", 0],
    ["rzPitchLow", 1],
    ["rzPitchHigh", 2],
    ["rzMidiTime", 3],
    ["rzMidiPlaying", 4],
    ["rzMidiDuration", 5],
  ]) {
    // Non-greedy: rzMidiPlaying guards on _rzMidi[0] AFTER reading its own
    // slot, and a greedy match would report the guard as the field.
    const body = new RegExp(`fn ${fn}\\(\\)[^}]*?_rzMidi\\[(\\d+)\\]`).exec(wgsl)
    assert.ok(body, `${fn} not found or reads no header slot`)
    assert.equal(Number(body[1]), index, `${fn} reads slot ${body[1]}, engine writes ${index}`)
  }
})

test("notes and keys are indexed off the exported offsets", () => {
  assert.match(
    wgsl,
    new RegExp(`_rzMidi\\[${MIDI_NOTES} \\+ i \\* ${MIDI_STRIDE} \\+ field\\]`),
    "note records must start at MIDI_NOTES with MIDI_STRIDE floats each",
  )
  assert.match(wgsl, new RegExp(`_rzMidi\\[${MIDI_HEADER} \\+ k\\]`), "the key map must start at MIDI_HEADER")
})

test("every accessor tolerates an absent score", () => {
  // rzNoteCount is 0 with no score, so the guards below are what keep an effect
  // written against a score from reading past the fallback buffer.
  assert.match(wgsl, /if \(i < 0 \|\| i >= i32\(_rzMidi\[0\]\)\) \{ return 0\.0; \}/, "note reads must bounds-check")
  assert.match(wgsl, new RegExp(`if \\(k < 0 \\|\\| k >= ${MIDI_KEYS}\\) \\{ return 0\\.0; \\}`), "key reads must bounds-check")
})

test("rzNoteAge is signed — the property the falling-note geometry rests on", () => {
  // time - start, not abs or max: a note not yet played must read NEGATIVE so a
  // bar can sit above the keyboard at a distance proportional to how long is left.
  assert.match(wgsl, /fn rzNoteAge\(i: i32\) -> f32 \{ return _rzMidi\[3\] - rzNoteStart\(i\); \}/)
})
