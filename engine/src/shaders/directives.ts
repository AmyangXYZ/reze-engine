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

/**
 * A bone an effect follows: `#anchor <bone> [trail] [along <d>]`.
 *
 * `along` is for an effect that comes off a point ON the bone rather than at its
 * joint — threads leaving the knuckles, not the wrist; a flame at a sword's tip.
 * The point is `d` model units down the bone's own axis: the direction a PMX
 * editor draws the bone pointing (its tail, as an offset or as the bone it
 * points at), turned by the bone's pose, scaled with the model. A left and a
 * right wrist point opposite ways, so one number mirrors itself. A bone with no
 * tail has no axis, and the point stays at the joint.
 *
 * Everything the anchor feeds reads the moved point: rzAnchor, the trail, the
 * ribbon. Two effects naming the same bone at different `along` are two anchors.
 *
 * `trail step d` records the path the way a game's trail does (Unity's
 * minVertexDistance): the newest point follows the bone every frame, and a
 * point is only LEFT BEHIND once the bone is `d` world units from the last one.
 * Sampled on the clock alone, a slowing hand leaves a knot of tiny segments
 * whose directions jitter, and a wide ribbon fans across them into a starburst;
 * a stepped path has no segment shorter than `d`. Ages stay true — each point
 * keeps the time it was left. Different steps are different anchors.
 */
export type EffectAnchor = { bone: string; trail: boolean; along?: number; step?: number }

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
  /** Bones this effect follows, in declaration order — slot 0 is the first.
   *  `along` (model units, absent = 0) moves the point down the bone's own
   *  axis, its rest tail direction posed with it: see EffectAnchor. */
  anchors: EffectAnchor[]
  params: EffectParamDecl[]
  /** Field layer: 0 full, 1 half. Full unless `#halfres` says otherwise. */
  fieldLayer: 0 | 1
  /** The field layer composites additively rather than over. */
  additiveLayer: boolean
  /** Particle blend, which is a different axis from the field layer's. */
  /** `cutout` is alpha WITH DEPTH: the pipeline writes depth and turns the
   *  fragment's alpha into MSAA sample coverage, so a blade of grass behind
   *  another is rejected by the depth test instead of blended under it. What
   *  every real grass renderer does, and the only thing that bounds a dense
   *  lawn's cost by the pixels it covers rather than by how deep it stacks. */
  particleBlend: "alpha" | "additive" | "cutout"
  /** How RIBBONS land, a separate axis from the particles': additive unless
   *  the file says `#blend over`. Ribbons ignored #blend until that keyword,
   *  and every shipped ribbon was tuned additive — so the default stays, and
   *  `over` is the opt-in. The same line sets particles to plain alpha, which
   *  already IS premultiplied over for them. */
  trailBlend: "additive" | "over"
  /** `#depth always`: the particles are drawn over the scene, not depth
   *  tested — a game glow its material draws with ZTest Always (a moon's halo
   *  in front of the sky dome it sits behind). Depth tested otherwise. */
  depthAlways: boolean
  particles: number
  lights: number
  grid: number
  /** `#points <prefix>`: every bone on every model whose name starts with it,
   *  as rzPoint(i) in the particle stages. Null when the file declares none. */
  points: string | null
  /**
   * `#textures N` (1-4): the particle shading reads N pictures the HOST hands
   * over with the install (setEffects' `textures`), as rzTexture(i, uv) — uv
   * (0,0) at the image's top-left, repeating. A game's own splash or spray is a
   * picture on a card, and drawn procedurally it can only ever be a guess at
   * it; this is how a converted stage's particles carry the real one. The
   * other modules an effect is spliced into get the names as stubs returning
   * zero. 0 when undeclared.
   */
  textures: number
  bloom: boolean
  /** This effect takes the cast apart — the host reads the timing. */
  dissolve: boolean
  /** `#ground #rrggbb [noise]`: how this effect DRESSES THE FLOOR while it is
   *  installed — a colour as linear RGB, and the strength of the ground's own
   *  grain (0 flat, 1 strong), which is what keeps soil from being a swatch.
   *  A lawn wants earth under it, and the earth has to be the engine's own
   *  ground rather than a layer painted over the frame: the ground is geometry
   *  in the scene pass, so a shadow lands on it and it tone-maps with
   *  everything else. The scene's own floor returns when the effect goes.
   *  Null when the file declares none. */
  ground: { color: [number, number, number]; noise: number } | null
  /**
   * This effect IS a mirror: a plane in the scene showing a true reflection of
   * it, folded and re-rendered by the engine.
   *
   * A mount with no shader, and the second of them — `#lights` was the first.
   * The reasoning is identical: a planar reflection re-renders every model from
   * a folded camera, which is a PASS, and no mount is a pass. So the effect
   * declares the mirror and the engine reflects, exactly as a lighting rig
   * declares lamps and the engine shades.
   *
   * The plane comes from the effect's own `#param` dials, by name — see the
   * engine's MIRROR_DIALS. That is what makes it an effect rather than a switch:
   * it publishes, forks, schedules and fades like every other one.
   */
  mirror: boolean
  /**
   * `#stepped`: the cast this effect is aimed at moves ON TWOS — its pose, and
   * its face, held for a few frames and then snapped to where it is now, the way
   * limited animation and stop-motion move. A mount with no shader, on
   * #mirror's footing: it changes WHEN a pose reaches the screen, which no
   * shader can. The rate comes from the effect's own FPS dial by name — see
   * the engine's STEPPED_DIALS.
   */
  stepped: boolean
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
  depth: 1,
  particles: 1,
  lights: 1,
  grid: 1,
  points: 1,
  textures: 1,
  bloom: 0,
  dissolve: 0,
  mirror: 0,
  stepped: 0,
  duration: 1,
  ground: "rest",
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
    trailBlend: "additive",
    depthAlways: false,
    particles: 0,
    lights: 0,
    grid: 0,
    points: null,
    textures: 0,
    bloom: false,
    dissolve: false,
    ground: null,
    mirror: false,
    stepped: false,
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
        // <bone> [trail [step <d>]] [along <d>], in that order.
        const usage = `${at}: #anchor takes a bone name, optionally "trail" (and "step" with a distance), then optionally "along" and a distance`
        if (args.length < 1) {
          errors.push(usage)
          return
        }
        let k = 1
        const trail = args[k] === "trail"
        if (trail) k++
        let step: number | undefined
        if (trail && args[k] === "step") {
          const v = num(args[k + 1])
          if (v === null || v <= 0) {
            errors.push(`${at}: #anchor ... trail step needs a positive distance in world units, like "step 0.24"`)
            return
          }
          step = v
          k += 2
        }
        let along: number | undefined
        if (args[k] === "along") {
          const v = num(args[k + 1])
          if (v === null) {
            errors.push(`${at}: #anchor ... along needs a distance in model units, like "along 0.9"`)
            return
          }
          along = v
          k += 2
        }
        if (k !== args.length) {
          errors.push(usage)
          return
        }
        d.anchors.push({ bone: args[0], trail, ...(step ? { step } : {}), ...(along ? { along } : {}) })
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
        // `over` is for ribbons: laid over the scene by their alpha instead of
        // added. For particles it is the default alpha, which is already over.
        if (args[0] === "over") {
          d.trailBlend = "over"
          d.particleBlend = "alpha"
          return
        }
        if (args[0] !== "additive" && args[0] !== "cutout") {
          errors.push(`${at}: #blend takes "additive", "cutout" or "over" — alpha is the default`)
          return
        }
        d.particleBlend = args[0]
        return
      case "bloom":
        d.bloom = true
        return
      case "depth":
        if (args[0] !== "always") {
          errors.push(`${at}: #depth takes "always" — depth tested is the default`)
          return
        }
        d.depthAlways = true
        return
      case "ground": {
        const m = /^#([0-9a-fA-F]{6})$/.exec(args[0] ?? "")
        const noise = args[1] === undefined ? 0 : num(args[1])
        if (!m || args.length > 2 || noise === null) {
          errors.push(`${at}: #ground takes a colour like #4b4026 and optionally a grain strength 0..1`)
          return
        }
        const v = parseInt(m[1], 16)
        // sRGB in the file, linear in the uniform — the same transfer the
        // ground's own colour goes through on the way in.
        const lin = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4))
        d.ground = {
          color: [lin(((v >> 16) & 255) / 255), lin(((v >> 8) & 255) / 255), lin((v & 255) / 255)],
          noise: Math.min(Math.max(noise, 0), 1),
        }
        return
      }
      case "dissolve":
        d.dissolve = true
        return
      case "points":
        d.points = args[0]
        return
      case "textures": {
        const n = num(args[0])
        if (n === null || !Number.isInteger(n) || n < 1 || n > 4) {
          errors.push(`${at}: #textures takes a count from 1 to 4`)
          return
        }
        d.textures = n
        return
      }
      case "mirror":
        d.mirror = true
        return
      case "stepped":
        d.stepped = true
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
