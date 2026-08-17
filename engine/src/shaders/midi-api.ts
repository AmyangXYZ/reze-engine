// The score data interface, as WGSL — shared verbatim by every effect module.
//
// The sibling of the audio interface, and precomputed for the same reason: an
// export steps the engine frame by frame rather than playing in real time, so
// anything read live from a player would render empty into the exported video.
// A note list is inherently precomputed, so it cannot even be tempted.
//
// WHY THIS EXISTS BESIDE rzAudio* RATHER THAN INSTEAD OF IT. The audio interface
// answers "how loud is the bass right now" — a spectrum, smeared across
// frequency and time. That is the right shape for a reactive backdrop and the
// wrong shape for a note: you cannot recover a discrete onset, a pitch and a
// duration from band energy, and the crispness of a falling note is the whole
// point of the look. The two are complementary — a scene can pulse its fog on
// rzAudioOnset while its notes fall on rzNoteAge.
//
// LAYOUT. An 8-float header, then a 128-float live key map, then the notes at 4
// floats each (start, duration, pitch, velocity). `time` and `playing` are
// rewritten every frame by whoever owns playback, exactly as the audio header
// is, so "now" is always the clock actually running.
//
// THE KEY MAP IS THE POINT OF THE DESIGN. Falling notes index the note list
// directly — one particle per note, which is what makes the geometry trivial.
// But a keyboard glow asks the opposite question, per pixel: is anything
// sounding at THIS pitch? Answering that by scanning the note list would be
// thousands of iterations per fragment. So the engine keeps a 128-entry
// per-pitch energy map, updated once per frame on the CPU, and the glow becomes
// one lookup — the same trade the audio interface makes by precomputing onset
// so "a beat-triggered effect is one comparison instead of a per-pixel history
// scan".

/** Floats before the key map: count, pitch range, clock, duration, release. */
export const MIDI_HEADER = 8
/** One energy slot per MIDI pitch. 128 is the MIDI range, not a cap we chose. */
export const MIDI_KEYS = 128
/** Where the note records begin. */
export const MIDI_NOTES = MIDI_HEADER + MIDI_KEYS
/** Floats per note: start, duration, pitch, velocity. */
export const MIDI_STRIDE = 4

/** The rzNote*/ /* rzKey* accessors, with the buffer declared at the given binding. */
export function midiApi(group: number, binding: number): string {
  return /* wgsl */ `
@group(${group}) @binding(${binding}) var<storage, read> _rzMidi: array<f32>;

/** Notes in the score; 0 when none is loaded, which every accessor below
 *  tolerates by returning zero rather than reading past the end. */
fn rzNoteCount() -> i32 { return i32(_rzMidi[0]); }
/** Where the score is NOW, in seconds — the clock every age below hangs from. */
fn rzMidiTime() -> f32 { return _rzMidi[3]; }
/** 1 while the score is advancing, 0 paused or absent. */
fn rzMidiPlaying() -> f32 { return select(0.0, _rzMidi[4], i32(_rzMidi[0]) > 0); }
/** Last note-off, in seconds — the length of the piece. */
fn rzMidiDuration() -> f32 { return _rzMidi[5]; }
/** Lowest and highest pitch the score actually uses. Lay a keyboard out against
 *  these rather than against 0..127 and a piano piece fills the screen instead
 *  of occupying its middle third. */
fn rzPitchLow() -> f32 { return _rzMidi[1]; }
fn rzPitchHigh() -> f32 { return _rzMidi[2]; }

fn _rzNoteAt(i: i32, field: i32) -> f32 {
  if (i < 0 || i >= i32(_rzMidi[0])) { return 0.0; }
  return _rzMidi[${MIDI_NOTES} + i * ${MIDI_STRIDE} + field];
}

/** When note i begins, in seconds. */
fn rzNoteStart(i: i32) -> f32 { return _rzNoteAt(i, 0); }
/** How long note i sounds, in seconds. */
fn rzNoteLength(i: i32) -> f32 { return _rzNoteAt(i, 1); }
/** MIDI pitch of note i — 60 is middle C. */
fn rzNotePitch(i: i32) -> f32 { return _rzNoteAt(i, 2); }
/** How hard note i was struck, 0..1. */
fn rzNoteVelocity(i: i32) -> f32 { return _rzNoteAt(i, 3); }

/**
 * Seconds since note i began — NEGATIVE before it does.
 *
 * The falling-note function, and the sign is the whole of it: a note's height is
 * -rzNoteAge(i) * speed, so a note not yet played sits above the keyboard at a
 * distance proportional to how long is left, crosses zero exactly as it sounds,
 * and keeps going. One expression, no state, no spawning — which is why one
 * particle per note is the natural mapping and the pool index IS the note index.
 */
fn rzNoteAge(i: i32) -> f32 { return _rzMidi[3] - rzNoteStart(i); }

/** 1 while note i is sounding, 0 either side of it. */
fn rzNoteHeld(i: i32) -> f32 {
  let age = rzNoteAge(i);
  return select(0.0, 1.0, age >= 0.0 && age < rzNoteLength(i));
}

/**
 * Energy at a MIDI pitch right now: 1 the instant it is struck, holding while
 * sounding, then falling away over the release set at install time.
 *
 * Per PITCH rather than per note, because that is the question a keyboard asks
 * and a note list cannot answer cheaply — see the header. Fractional pitches
 * read the nearest key, so a caller can hand this a continuous x across the
 * keyboard without rounding first.
 */
fn rzKeyEnergy(pitch: f32) -> f32 {
  let k = i32(round(pitch));
  if (k < 0 || k >= ${MIDI_KEYS}) { return 0.0; }
  return _rzMidi[${MIDI_HEADER} + k];
}

/**
 * A pitch's place across the score's own range, 0..1 — the x of a falling note
 * and of the key it lands on, which have to agree or the effect is nonsense.
 */
fn rzPitchX(pitch: f32) -> f32 {
  let lo = _rzMidi[1];
  let hi = _rzMidi[2];
  return select(0.5, (pitch - lo) / max(hi - lo, 1.0), hi > lo);
}
`
}
