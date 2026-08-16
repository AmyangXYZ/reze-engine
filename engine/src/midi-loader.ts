// Standard MIDI File → ScoreNote[], for setScore.
//
// The engine already parses PMX and VMD; this is the third loader and the
// smallest, because a score needs four numbers per note and a .mid carries far
// more than that. Everything not needed to place a note in TIME is skipped:
// instruments, controllers, pitch bend, lyrics, key signatures.
//
// THE TEMPO MAP IS THE WHOLE JOB. A MIDI file measures time in ticks against a
// tempo that changes as often as the music does — the piano transcription this
// was written against has 89 tempo changes across four minutes, because rubato
// is what makes it sound played rather than sequenced. Divide by a single
// average and every note after the first change lands at the wrong moment, and
// the drift compounds. So tempo changes are collected across ALL tracks first
// (format 1 puts them in track 0, but nothing requires that), then ticks are
// integrated through them segment by segment.

import type { ScoreNote } from "./engine"

/** A tempo change: from this tick onward, one quarter note lasts this long. */
type Tempo = { tick: number; usPerQuarter: number }

/** 120 bpm — what the spec says to assume when a file states nothing. */
const DEFAULT_US_PER_QUARTER = 500000

class Reader {
  offset = 0
  constructor(private readonly d: DataView) {}
  get remaining(): number {
    return this.d.byteLength - this.offset
  }
  u8(): number {
    return this.d.getUint8(this.offset++)
  }
  u16(): number {
    const v = this.d.getUint16(this.offset)
    this.offset += 2
    return v
  }
  u32(): number {
    const v = this.d.getUint32(this.offset)
    this.offset += 4
    return v
  }
  bytes(n: number): number {
    // Big-endian integer of n bytes — tempo is three of them.
    let v = 0
    for (let i = 0; i < n; i++) v = (v << 8) | this.u8()
    return v
  }
  /** Variable-length quantity: seven bits per byte, high bit continues. */
  vlq(): number {
    let v = 0
    for (;;) {
      const b = this.u8()
      v = (v << 7) | (b & 0x7f)
      if ((b & 0x80) === 0) return v
    }
  }
  tag(): string {
    return String.fromCharCode(this.u8(), this.u8(), this.u8(), this.u8())
  }
}

/** One note-on waiting for its note-off, keyed by channel and pitch. */
type Pending = { tick: number; velocity: number }

/**
 * Parse a Standard MIDI File into notes on the scene clock, sorted by onset.
 *
 * Throws only on a file that is not a MIDI file at all; a truncated or partly
 * unreadable track yields the notes it managed rather than nothing, because a
 * score that plays most of a piece is worth more than an exception.
 */
export function parseMidi(data: ArrayBuffer): ScoreNote[] {
  const r = new Reader(new DataView(data))
  if (r.remaining < 14 || r.tag() !== "MThd") throw new Error("not a MIDI file (no MThd header)")
  const headerLength = r.u32()
  const headerEnd = r.offset + headerLength
  r.u16() // format: 0, 1 and 2 all parse the same way here — every track is read
  const trackCount = r.u16()
  const division = r.u16()
  r.offset = headerEnd

  if (division & 0x8000) {
    // SMPTE: ticks are absolute frames, so there is no tempo map to apply.
    // Vanishingly rare for music files and it would silently mis-time
    // everything, so say so rather than pretend.
    throw new Error("SMPTE time division is not supported — export the file with a tick-based division")
  }
  const ticksPerQuarter = division || 480

  const tempos: Tempo[] = []
  const raw: { tick: number; on: boolean; pitch: number; velocity: number; channel: number }[] = []

  for (let t = 0; t < trackCount && r.remaining >= 8; t++) {
    if (r.tag() !== "MTrk") break
    const length = r.u32()
    const end = Math.min(r.offset + length, r.offset + r.remaining)
    let tick = 0
    // Running status: an event may omit its status byte and inherit the last
    // one. Dropping this reads the file as garbage from the first omission on.
    let status = 0
    while (r.offset < end) {
      tick += r.vlq()
      let b = r.u8()
      if (b & 0x80) {
        status = b
        b = r.offset < end ? r.u8() : 0
      }
      const kind = status & 0xf0
      if (status === 0xff) {
        const meta = b
        const len = r.vlq()
        if (meta === 0x51 && len === 3) tempos.push({ tick, usPerQuarter: r.bytes(3) })
        else r.offset += len
      } else if (status === 0xf0 || status === 0xf7) {
        // A sysex length was already consumed into `b` above only if the status
        // byte was present; re-read defensively from the current position.
        r.offset -= 1
        r.offset += r.vlq()
      } else if (kind === 0xc0 || kind === 0xd0) {
        // Program change / channel pressure: one data byte, already in `b`.
      } else if (kind === 0x80 || kind === 0x90) {
        const velocity = r.offset < end ? r.u8() : 0
        // Note-on at velocity 0 IS a note-off — the common encoding, because it
        // lets a run of notes share one running status byte.
        const on = kind === 0x90 && velocity > 0
        raw.push({ tick, on, pitch: b, velocity, channel: status & 0x0f })
      } else {
        // Everything else with two data bytes: controller, pitch bend, aftertouch.
        if (r.offset < end) r.u8()
      }
    }
    r.offset = end
  }

  tempos.sort((a, b) => a.tick - b.tick)
  // Ticks → seconds, integrating the tempo map. Walked once in tick order
  // rather than searched per note: the events are already sorted, so this is
  // linear where a per-note scan would be quadratic on a 1500-note file.
  const toSeconds = makeTickClock(tempos, ticksPerQuarter)

  raw.sort((a, b) => a.tick - b.tick || (a.on ? 1 : 0) - (b.on ? 1 : 0))
  const pending = new Map<number, Pending[]>()
  /**
   * Note-offs that found nothing to close, by key, in the order they were read.
   *
   * At the same tick the sort above puts every off AHEAD of every on, which is
   * what makes a restrike pair correctly — the release of the held note is read
   * before the strike that follows it. The cost is a note whose off is at its
   * own onset tick: genuinely zero-length, and its off arrives before the on
   * exists to be closed. Dropping it here left the on unpaired, so the note
   * hung to the end of the file — the one case where the ordering that fixes
   * restrikes creates a note nothing wrote.
   *
   * Kept rather than discarded so the flush below can look again. An off at any
   * OTHER tick really is stray and stays dropped.
   */
  const orphanOffs = new Map<number, number[]>()
  const notes: ScoreNote[] = []
  for (const e of raw) {
    const key = e.channel * 128 + e.pitch
    if (e.on) {
      const list = pending.get(key)
      if (list) list.push({ tick: e.tick, velocity: e.velocity })
      else pending.set(key, [{ tick: e.tick, velocity: e.velocity }])
      continue
    }
    // FIFO against the same key: a pedalled passage can restrike a pitch before
    // releasing it, and pairing the newest would leave the older one hanging
    // forever — the note would last the rest of the piece.
    const list = pending.get(key)
    const start = list?.shift()
    if (!start) {
      const seen = orphanOffs.get(key)
      if (seen) seen.push(e.tick)
      else orphanOffs.set(key, [e.tick])
      continue
    }
    const t0 = toSeconds(start.tick)
    notes.push({
      start: t0,
      duration: Math.max(0, toSeconds(e.tick) - t0),
      pitch: e.pitch,
      velocity: start.velocity / 127,
    })
  }
  // Anything still held at the end of the file gets the file's own length, so a
  // missing note-off shows as a long note rather than as a lost one.
  const lastTick = raw.length ? raw[raw.length - 1].tick : 0
  for (const [key, list] of pending) {
    // The pitch comes back out of the map key. Writing a literal 0 here — as
    // this did — does not lose the note, it MOVES it: every hanging note lands
    // on C-1, which then drags the score's reported pitch range down to it and
    // lays every keyboard out against an octave nothing plays in.
    const pitch = key % 128
    const orphans = orphanOffs.get(key)
    for (const p of list) {
      const t0 = toSeconds(p.tick)
      // The second look: an off read at this note's OWN tick is its off, seen
      // early because same-tick offs sort first. One off closes one note, so it
      // is consumed — two zero-length strikes of a pitch need two offs.
      const i = orphans ? orphans.indexOf(p.tick) : -1
      if (i >= 0) {
        orphans!.splice(i, 1)
        notes.push({ start: t0, duration: 0, pitch, velocity: p.velocity / 127 })
        continue
      }
      notes.push({ start: t0, duration: Math.max(0, toSeconds(lastTick) - t0), pitch, velocity: p.velocity / 127 })
    }
  }

  notes.sort((a, b) => a.start - b.start)
  return notes
}

/** Tick → seconds through the tempo map, as a closure over the segments. */
function makeTickClock(tempos: Tempo[], ticksPerQuarter: number): (tick: number) => number {
  // Precompute the elapsed seconds at each tempo change, so a lookup is a walk
  // over segments rather than a re-integration from zero.
  const marks: { tick: number; seconds: number; usPerQuarter: number }[] = []
  let seconds = 0
  let lastTick = 0
  let us = tempos.length && tempos[0].tick === 0 ? tempos[0].usPerQuarter : DEFAULT_US_PER_QUARTER
  marks.push({ tick: 0, seconds: 0, usPerQuarter: us })
  for (const t of tempos) {
    if (t.tick > lastTick) {
      seconds += ((t.tick - lastTick) * us) / ticksPerQuarter / 1e6
      lastTick = t.tick
    }
    us = t.usPerQuarter
    marks.push({ tick: t.tick, seconds, usPerQuarter: us })
  }
  return (tick: number): number => {
    // Binary search for the last mark at or before `tick`.
    let lo = 0
    let hi = marks.length - 1
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1
      if (marks[mid].tick <= tick) lo = mid
      else hi = mid - 1
    }
    const m = marks[lo]
    return m.seconds + ((tick - m.tick) * m.usPerQuarter) / ticksPerQuarter / 1e6
  }
}
