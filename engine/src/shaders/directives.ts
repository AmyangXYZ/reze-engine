// What an effect DECLARES about itself — parsed once, in one place.
//
// An effect file is WGSL plus a handful of lines that configure how the engine
// mounts it: which resolution it draws at, how it blends, which bones it
// follows, what knobs it exposes. Those lines are not comments. They decide
// whether the file gets what it asks for, and the failure they used to have was
// the worst kind available.
//
// WHY `#` AND NOT `// @`. The old spelling put directives inside comments,
// which meant two things. Authors read them as comments and wrote notes on the
// same line — every parser was anchored to end-of-line, so the note silently
// unmade the directive, and three shipped effects ran at half the resolution
// their own first line said they needed while a fourth quietly stopped being
// additive. Nothing failed and nothing was reported, because nothing had been
// declared to fail. And it could not be fixed by being strict: a comment
// beginning with `@` is a legitimate thing to write, so an unrecognised one can
// only ever be a warning.
//
// WGSL has no `#` syntax of its own. A line starting with `#` is therefore
// unambiguously ours, an unknown one is an ERROR rather than a guess, and the
// sigil is the one every shader author already reads as "directive" from
// `#pragma`. The cost usually quoted for this — that the file stops being valid
// WGSL — is not a cost we pay: an effect calls rzSubject, rzTrail and reads
// `params`, none of which exist until the engine splices them in, so these
// files have never compiled anywhere else.
//
// STRIPPED BY BLANKING, not by deleting: the compiler sees an empty line where
// each directive was, so every diagnostic's line number still points at the
// line the author is looking at.

/** A knob an effect exposes, for a host to build a control from. */
export type EffectParamDecl = {
  name: string
  kind: "float" | "color" | "vec3"
  /** Numbers for float/vec3; `#rrggbb` for a colour, which the host converts. */
  value: number | [number, number, number] | string
  /** float only, and only when the author gave a range. */
  min?: number
  max?: number
}

export type EffectDirectives = {
  /** Bones this effect follows, in declaration order — slot 0 is the first. */
  anchors: { bone: string; trail: boolean }[]
  params: EffectParamDecl[]
  /** Field layer: 0 full, 1 half. Full unless `#halfres` says otherwise. */
  fieldLayer: 0 | 1
  /** The field layer composites additively rather than over. */
  additiveLayer: boolean
  /** Particle blend, which is a different axis from the field layer's. */
  particleBlend: "alpha" | "additive"
  particles: number
  lights: number
  grid: number
  bloom: boolean
  /** This effect takes the cast apart — the host reads the timing. */
  dissolve: boolean
  /**
   * How long ONE firing of this effect lasts, in seconds. 0 = undeclared.
   *
   * An effect is one of two things, and only its author knows which. A HIT has
   * an arc — a circle flares, peaks and is gone — and its length is a fact
   * about it, the way a video clip's length is a fact about the file. An
   * AMBIENT effect (stars, fog, rain) has no length at all; it is a condition
   * the scene is in.
   *
   * Declaring it is what lets a host place the effect instead of making someone
   * construct it: dropping a hit on the timeline gives a strip already the
   * right size, and an effect that declares nothing spans the scene. Every
   * timeline works this way — a clip arrives at its own duration.
   */
  duration: number
}

/** Every directive, with how many words follow it. `rest` means free-form. */
const SPEC = {
  anchor: "rest",
  param: "rest",
  halfres: 0,
  layer: 1,
  blend: 1,
  particles: 1,
  lights: 1,
  grid: 1,
  bloom: 0,
  dissolve: 0,
  duration: 1,
} as const

/** A line that declares something, and what it declares. Exported because an
 *  editor highlights by the same rule the parser reads by — a highlighter with
 *  its own idea of what counts is one that paints a line as configuration that
 *  the engine then ignores. */
export const DIRECTIVE_LINE = /^[ \t]*#([a-zA-Z]+)[ \t]*(.*)$/
const LINE = DIRECTIVE_LINE

/**
 * Split a directive's arguments from a trailing note.
 *
 * `#anchor 左手首 trail — her sword hand` is one line doing two jobs, and
 * refusing it is what made the old spelling dangerous: the note is the natural
 * thing to write, so it has to be the accepted thing to write.
 */
export const DIRECTIVE_NOTE = /(?:^|\s)(?:—|--|\/\/|#)\s/

function argsOf(rest: string): string[] {
  // `^` as well as `\s`: the tag's own trailing space is eaten by LINE, so a
  // note can begin at the very first character — which is what
  // `#halfres — glyph edges` looks like by the time it reaches here.
  const note = rest.search(DIRECTIVE_NOTE)
  return (note >= 0 ? rest.slice(0, note) : rest).trim().split(/\s+/).filter(Boolean)
}

/** A number, or null. Empty is NULL, not zero: `Number("")` is 0, so a missing
 *  default silently became a real one — an author who wrote `#param float D`
 *  and meant to finish the line got a knob quietly pinned at zero. */
const num = (s: string | undefined): number | null => {
  if (!s || !s.trim()) return null
  const v = Number(s)
  return Number.isFinite(v) ? v : null
}

export type DirectiveResult = { directives: EffectDirectives; errors: string[] }

/** Read every declaration in a source. Errors name the line, one-based. */
export function parseDirectives(wgsl: string): DirectiveResult {
  const d: EffectDirectives = {
    anchors: [],
    params: [],
    // FULL unless asked otherwise — the default is what an author gets for
    // saying nothing, so it has to be the answer that cannot silently ruin an
    // effect. `#halfres` is a claim about being cheap, which is a claim only
    // the author can make.
    fieldLayer: 0,
    additiveLayer: false,
    particleBlend: "alpha",
    particles: 0,
    lights: 0,
    grid: 0,
    bloom: false,
    dissolve: false,
    duration: 0,
  }
  const errors: string[] = []
  const lines = wgsl.split("\n")

  lines.forEach((line, i) => {
    const m = LINE.exec(line)
    if (!m) return
    const at = `line ${i + 1}`
    const tag = m[1].toLowerCase()
    if (!(tag in SPEC)) {
      errors.push(`${at}: #${m[1]} is not a directive. Known: ${Object.keys(SPEC).map((k) => `#${k}`).join(", ")}`)
      return
    }
    const args = argsOf(m[2])
    const want = SPEC[tag as keyof typeof SPEC]
    if (want !== "rest" && args.length !== want) {
      errors.push(`${at}: #${tag} takes ${want} argument${want === 1 ? "" : "s"}, got ${args.length}`)
      return
    }

    switch (tag) {
      case "anchor": {
        if (args.length < 1 || args.length > 2 || (args[1] && args[1] !== "trail")) {
          errors.push(`${at}: #anchor takes a bone name and optionally the word "trail"`)
          return
        }
        d.anchors.push({ bone: args[0], trail: args[1] === "trail" })
        return
      }
      case "param": {
        const [kind, name, ...rest] = args
        if (!name || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(name)) {
          errors.push(`${at}: #param needs a WGSL identifier for a name`)
          return
        }
        if (kind === "color") {
          if (!/^#[0-9a-fA-F]{6}$/.test(rest[0] ?? "")) {
            errors.push(`${at}: #param color ${name} needs a default like #3b82f6`)
            return
          }
          d.params.push({ name, kind: "color", value: rest[0] })
          return
        }
        if (kind === "vec3") {
          const v = rest.slice(0, 3).map(num)
          if (v.length !== 3 || v.some((x) => x === null)) {
            errors.push(`${at}: #param vec3 ${name} needs three numbers`)
            return
          }
          d.params.push({ name, kind: "vec3", value: v as [number, number, number] })
          return
        }
        if (kind === "float") {
          const v = num(rest[0])
          if (v === null) {
            errors.push(`${at}: #param float ${name} needs a default`)
            return
          }
          const lo = num(rest[1])
          const hi = num(rest[2])
          // A range is optional and all-or-nothing: half of one is a slider
          // with an end nobody chose.
          if ((rest[1] !== undefined) !== (rest[2] !== undefined) || (rest[1] !== undefined && (lo === null || hi === null))) {
            errors.push(`${at}: #param float ${name} takes both a min and a max, or neither`)
            return
          }
          d.params.push({ name, kind: "float", value: v, ...(lo !== null && hi !== null ? { min: lo, max: hi } : {}) })
          return
        }
        errors.push(`${at}: #param kind must be float, color or vec3`)
        return
      }
      case "halfres":
        d.fieldLayer = 1
        return
      case "layer":
        if (args[0] !== "additive") {
          errors.push(`${at}: #layer takes "additive" — over is the default`)
          return
        }
        d.additiveLayer = true
        return
      case "blend":
        if (args[0] !== "additive") {
          errors.push(`${at}: #blend takes "additive" — alpha is the default`)
          return
        }
        d.particleBlend = "additive"
        return
      case "bloom":
        d.bloom = true
        return
      case "dissolve":
        d.dissolve = true
        return
      case "duration": {
        // SECONDS, like every other time a directive states (see #dissolve).
        // The document above works in frames and converts once; an author
        // writing a shader is thinking about how long a flare takes, not about
        // MMD's frame rate.
        const n = num(args[0])
        if (n === null || n <= 0) {
          errors.push(`${at}: #duration takes a length in seconds`)
          return
        }
        d.duration = n
        return
      }
      default: {
        // The three that take a count.
        const n = num(args[0])
        if (n === null || n < 0) {
          errors.push(`${at}: #${tag} takes a number`)
          return
        }
        if (tag === "particles") d.particles = n
        else if (tag === "lights") d.lights = n
        else if (tag === "grid") d.grid = n
        return
      }
    }
  })

  // Duplicates are an author editing one line and forgetting another; last-wins
  // is a coin toss they never see resolved.
  const names = new Set<string>()
  for (const p of d.params) {
    if (names.has(p.name)) errors.push(`#param ${p.name} is declared twice`)
    names.add(p.name)
  }

  return { directives: d, errors }
}

/**
 * The source as the compiler should see it: every directive line blanked.
 *
 * Blanked rather than removed so line numbers survive — a diagnostic points at
 * the line the author is looking at, which is the whole reason the engine
 * rebases them in the first place.
 */
export function stripDirectives(wgsl: string): string {
  return wgsl
    .split("\n")
    .map((line) => (LINE.test(line) ? "" : line))
    .join("\n")
}
