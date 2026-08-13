// The audio data interface, as WGSL — shared verbatim by every effect module.
//
// The buffer is PRECOMPUTED for the whole track and sampled by time, never fed
// live from an AnalyserNode. That is not an optimisation: an export steps the
// engine frame by frame instead of playing in real time, so a live analyser
// would return silence during a render and every audio-reactive effect would
// quietly vanish from the exported video — the same class of bug the scene
// clock exists to prevent. Layout: an 8-float header [frames, bands,
// secondsPerFrame, audioTime, playing, 3 spare] then per frame
// [level, onset, band0..bandN-1]. audioTime and playing are rewritten every frame by
// whoever owns playback — the editor's audio clock, the viewer's, or the export
// loop — so "now" is always the clock that is actually running.
//
// Every module binds the SAME buffer, so a spawn rule in a particle effect and
// a bar in a background read identical numbers for the same instant.

/** The rzAudio* accessors, with the buffer declared at the given binding. */
export function audioApi(group: number, binding: number): string {
  return /* wgsl */ `
@group(${group}) @binding(${binding}) var<storage, read> _rzAudio: array<f32>;

/** Frames of analysis available; 0 when the scene has no audio. */
fn rzAudioFrames() -> i32 { return i32(_rzAudio[0]); }
/** 1 while the track is actually PLAYING, 0 paused or absent — so an effect can
 *  go calm instead of oscillating over a frozen spectrum. */
fn rzAudioPlaying() -> f32 { return select(0.0, _rzAudio[4], i32(_rzAudio[0]) > 0); }
/** Where the track is NOW, in seconds — the clock rzAudioLevelAt offsets hang
 *  from. An effect that detects an onset k seconds back can anchor a ring to
 *  the ABSOLUTE moment of the hit and age it continuously. */
fn rzAudioTime() -> f32 { return _rzAudio[3]; }
/** Bands per frame — log-spaced, bass first. */
fn rzAudioBandCount() -> i32 { return i32(_rzAudio[1]); }

fn _rzAudioFrameAt(offset: f32) -> i32 {
  let frames = i32(_rzAudio[0]);
  let t = _rzAudio[3] + offset;
  return clamp(i32(t / max(_rzAudio[2], 1e-5)), 0, frames - 1);
}

/**
 * Loudness at now + offset seconds, 0..1.
 *
 * The offset is what makes a WAVEFORM drawable: a column at x samples the
 * envelope at (x - 0.5) * window seconds, and the playhead is the centre of the
 * screen by construction. Offsets past either end clamp to the track's edges.
 */
fn rzAudioLevelAt(offset: f32) -> f32 {
  let frames = i32(_rzAudio[0]);
  if (frames <= 0) { return 0.0; }
  let bands = i32(_rzAudio[1]);
  return _rzAudio[8 + _rzAudioFrameAt(offset) * (bands + 2)];
}

/**
 * The KICK track: how hard the bass is rising at now + offset, 0..1 —
 * precomputed, so a beat-triggered effect is one comparison instead of a
 * per-pixel history scan. Scan a short window of NEGATIVE offsets to find
 * recent hits and age things from them (quantise the hit to the analysis grid
 * so ages stay continuous).
 */
fn rzAudioOnsetAt(offset: f32) -> f32 {
  let frames = i32(_rzAudio[0]);
  if (frames <= 0) { return 0.0; }
  let bands = i32(_rzAudio[1]);
  return _rzAudio[8 + _rzAudioFrameAt(offset) * (bands + 2) + 1];
}
fn rzAudioOnset() -> f32 { return rzAudioOnsetAt(0.0); }
/** Loudness now, 0..1. */
fn rzAudioLevel() -> f32 { return rzAudioLevelAt(0.0); }

/** Band i at now + offset seconds, 0..1. Band 0 is the deepest bass. */
fn rzAudioBandAt(i: i32, offset: f32) -> f32 {
  let frames = i32(_rzAudio[0]);
  if (frames <= 0) { return 0.0; }
  let bands = i32(_rzAudio[1]);
  if (i < 0 || i >= bands) { return 0.0; }
  return _rzAudio[8 + _rzAudioFrameAt(offset) * (bands + 2) + 2 + i];
}
/** Band i now — the visualiser call. */
fn rzAudioBand(i: i32) -> f32 { return rzAudioBandAt(i, 0.0); }
`
}
