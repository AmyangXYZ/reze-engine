// Jelly eyes — the MMD 果冻眼 / Live2D highlight wobble, for any PMX.
//
// What every working version of the effect has in common, whether it is keyed
// by hand in MMD, a pendulum in Live2D, or a jiggle bone in Blender: an INPUT
// (a blink, a turn), something that REMEMBERS the last frame (a spring), and a
// cheap OUTPUT. Here the output is the iris's shape, in place.
//
// The blink is the MME 果冻眼's (JellyEye V1.2.2, by 茶叶味香皂): a spring
// driven by how FAST the lids move. The iris widens while they close and rings
// out once they stop, kicked again at each change of pace; the lid presses it
// flatter as it closes; a twist and a drift of a hair ride the spring. On top,
// her head's motion and an idle tap now and then knock it, as before. A
// highlight drawn as its own layer over the iris (a separate material, as MME
// rigs have) follows the iris, then trembles on its own: a new random spot ten
// times a second, no easing — the jiggle the MME is known for. The inputs are
// her blinks and her head, never her eyes' own turn: eye tracking turns the
// eyes after the camera, and the camera must not shake the jelly.
//
// The springs run on the CPU, stepped by the engine's frame delta — the same
// exact delta renderFrame() hands an export, so a blink bounces identically on
// screen and in the file. Their result rides in the model's ObjectLight
// (jelly[0..11], see materials/common.ts): the vertex shader moves the iris —
// the vertices the eye bones carry — about its centre, so every eye graph
// becomes jelly without knowing it. Nothing changes its colour.

import type { Model } from "./model"

export interface JellyEyeOptions {
  /** How much the eyes give: 0 still, 1 the intended bounce, up to 2. Default 1. */
  strength?: number
  /** The idle taps: a ripple every 4–8 s while nothing else moves them.
   *  Default true; false leaves only what her blinks and motion set off. */
  idle?: boolean
}

// The closing morphs, by the names MMD, VRoid and the usual converters give
// them. Group morphs (a smile that closes the eyes) resolve into these through
// the effective weights, so only the leaves need naming.
const BOTH = ["まばたき", "笑い", "はぅ", "blink", "Blink", "eye_close", "Fcl_EYE_Close"]
const LEFT = ["ウィンク", "ウィンク２", "ｳｨﾝｸ", "ｳｨﾝｸ２", "wink", "Wink", "blink_L", "Fcl_EYE_Close_L"]
const RIGHT = ["ウィンク右", "ウィンク２右", "ｳｨﾝｸ右", "ｳｨﾝｸ２右", "wink_R", "Wink_R", "blink_R", "Fcl_EYE_Close_R"]

// The blink, as the MME runs it (its JellyEye_Settings.fxsub defaults). The
// eye's closure (0..1) is differentiated twice. Its speed drives a target the
// spring chases: CLOSE_DRIVE while closing, OPEN_DRIVE while opening, eased by
// how fast. Every change between gaining and losing pace kicks the spring's
// velocity. The spring is an exact under-damped step at SPRING_HZ whose
// envelope halves every HALF_LIFE.
const ACCEL_MIN = 0.5
const ACCEL_FULL = 2
const ACCEL_KICK = 0.32
const RELEASE_KICK = 0.2
const CLOSE_DRIVE = 0.85
const OPEN_DRIVE = 0.2
const DRIVE_SPEED_MIN = 0.04
const DRIVE_SPEED_FULL = 2.5
const DRIVE_CURVE = 0.85
const MORPH_SPEED_LIMIT = 30
const ACCEL_LIMIT = 240
const MAX_STEP = 0.12
const SPRING_HZ = 3
const HALF_LIFE = 0.28
const ELASTIC_S = 2
const STOP_EPS = 0.0025
const MIN_RELEASE = 0.12
const SPRING_LIMIT = 1.25
// What the spring does to the iris, per unit of it: wider far more than
// taller, a little less on the rebound side; the lid's press, flatter by
// SQUEEZE_Y at full closure; a twist (degrees); and a drift (iris radii: the
// MME's model units over a typical iris of 0.1). Twist and drift take both the
// spring's place (_POS) and its velocity (_VEL). The twist is ten and five
// times the MME's defaults (0.2°, 0.8°): at those it stays under a degree and
// the iris never visibly turns as it rings.
const SCALE_GAIN: [number, number] = [0.2, 0.05]
const REBOUND: [number, number] = [0.7, 0.8]
const SQUEEZE_Y = 0.18
const TWIST_POS = 2
const TWIST_VEL = 4
const DRIFT_POS: [number, number] = [0.008, 0.015]
const DRIFT_VEL: [number, number] = [0.012, 0.006]
// The highlight layer's own motion: a held random spot, re-drawn JITTER_HZ
// times a second per axis, up to JITTER of an iris radius; and, on the spring,
// a further sway (iris radii) and turn (degrees) of its own.
const JITTER_HZ: [number, number] = [9, 11]
const JITTER: [number, number] = [0.03, 0.02]
const HL_SWAY: [number, number] = [0.03, 0.045]
const HL_TURN = 10
const HL_SCALE: [number, number] = [0.055, 0.045]
const HL_SCALE_MIN = 0.85
const HL_SCALE_MAX = 1.2
// The roll: while a closing expression holds, the iris turns about its eye
// bone's X axis (左目/右目, the eyeball's centre), rolling down with the lid —
// the angle at weight 1, and simultaneous expressions add. It comes forward
// only with the spring (iris radii): pushed with the expression as well, a
// shut eye's iris came through a lid set close in front of it (白里黑子). The MME rolls with the lid
// exactly, which the lid hides; here the roll follows on a spring twice the
// blink's pace with its decay, so as the eye opens the iris swings up past rest
// and wobbles back, where it can be seen. At the blink's own pace it trailed
// the opening lid and swung a third of an iris past centre: a jump.
const ROLL: Record<string, number> = {
  まばたき: -15, blink: -15, Blink: -15, eye_close: -15, Fcl_EYE_Close: -15, はぅ: -15, 笑い: -5,
  ウィンク: -5, ｳｨﾝｸ: -5, wink: -5, Wink: -5, ウィンク２: -15, ｳｨﾝｸ２: -15, blink_L: -15, Fcl_EYE_Close_L: -15,
  ウィンク右: -5, ｳｨﾝｸ右: -5, wink_R: -5, Wink_R: -5, ウィンク２右: -15, ｳｨﾝｸ２右: -15, blink_R: -15, Fcl_EYE_Close_R: -15,
}
const ROLL_HZ = 6
const ROLL_ZETA = Math.LN2 / HALF_LIFE / (2 * Math.PI * ROLL_HZ)
// How much of the spring's swing about the expression angle shows: a
// twenty-fifth of an iris radius past centre after a blink.
const ROLL_SWING = 0.25
const PUSH_ELASTIC = 0.04
// A highlight sheet by name, for rigs that keep it on its own material.
const HIGHLIGHT = /ハイライト|ﾊｲﾗｲﾄ|高光|highlight|hilight|(^|[^a-z])hl([^a-z]|$)/i
const DEG = Math.PI / 180
// The knock's squash (her motion, the idle taps): nearly round, lightly damped
// so one knock ripples.
const SQUASH_HZ = 1.8
const SQUASH_ZETA = 0.08
// The lag: the picture trails the eye's turn and catches up past it. Radians of
// lag to eye radii of offset; the head's own turn counts for a third, so a
// sudden stop of the head wobbles the eyes without a dance keeping them
// sloshing.
const LAG_HZ = 3.5
const LAG_ZETA = 0.4
// The lag is an angle: how far the iris trails the eye's own turn, as a share
// of the lag in look, capped at a quarter of eye tracking's default 20° reach —
// a wobble on the turn, never a second turn. It becomes iris radii through the
// eye's own lever (iris centre to eye bone), the same displacement a real turn
// of that angle gives, so it scales with whatever the rig's eyes do.
const LAG_GAIN = 0.5
const LAG_MAX = (3 * Math.PI) / 180
// And never past this many iris radii, whatever the lever, at strength 1.
const OFFSET_MAX = 0.2
const HEAD_SHARE = 0.35
// The inertia: the iris has a little mass, hung on a spring from the head, so
// her own motion — a bob, a step, a sway of the dance — leaves it behind and
// it swings back. Model units of lag to iris radii, scaled by INERTIA_GAIN.
const INERTIA_HZ = 3.0
const INERTIA_ZETA = 0.4
const INERTIA_GAIN = 0.04
// The jelly's give: the iris never moves, it swells. The lag and the inertia
// are how hard it is being knocked (in iris radii, as if it could move); that
// knock grows it a little and it ripples back, as the squash does — the bounce on top
// of the blink's squash. Eased, not linear: the swell is PULSE_MAX·tanh(knock /
// PULSE_REF), so the small steady knocks of a dance still show while a jolt
// (play from a pause: the head going from still to moving at once) cannot
// swell it past the cap.
const PULSE_MAX = 0.07
const PULSE_REF = 0.006
// And the same knock squashes it as a blink does — nearly round, on
// the blink's spring — so her motion reads as plainly as a blink. Eased by the
// same curve, up to MOTION_SQUASH (a blink's opening gives about 0.2).
const MOTION_SQUASH = 0.18
const PULSE_HZ = 2.0
const PULSE_ZETA = 0.08
// The idle taps: every 4–8 s, at uneven times, a knock into the
// same springs her motion drives — the iris dips, bounces and settles, then
// rests — the way a jelly on a table is never quite still. A steady sine was
// tried, as size (a zoom) and as wider-then-taller (queasy); a settling tap is
// what reads as jelly. A tap is a full KNOCK, exactly what her motion gives at
// its strongest — a gentler nudge was tried and went unnoticed — held for a
// moment and let go, so it rings through the same springs. The gaps come from
// a hash of the tap's count, so the scene clock alone decides them and an
// export taps where the screen did.
const IDLE_TAP_S = 0.12
// How hard a tap knocks (motion's strongest is 1) and how far apart they fall:
// a full knock every 2–4 s read as the iris zooming constantly; rarer, a touch
// softer, each one reads as a ripple and the eye rests between.
const IDLE_TAP = 0.6
const IDLE_GAP: [number, number] = [4, 8]
const idleGap = (n: number): number => {
  const x = Math.sin(n * 12.9898 + 78.233) * 43758.5453
  return IDLE_GAP[0] + (x - Math.floor(x)) * (IDLE_GAP[1] - IDLE_GAP[0])
}
// Integration step, and the jump past which a delta is a seek, not time.
const SUBSTEP = 1 / 240
const SEEK = 0.25

interface EyeState {
  closure: number
  // The MME blink: last frame's closure and its speed, the last pace trend
  // (+1 gaining, −1 losing, 0 still), the spring's place and velocity, and
  // the seconds since the lids last drove it.
  prev: number
  speed: number
  trend: number
  x: number
  v: number
  age: number
  squash: number
  squashV: number
  // Lagged look (yaw, pitch) and its velocity; the offset is lagged − actual.
  lag: [number, number]
  lagV: [number, number]
  look: [number, number]
  pulse: number
  // The roll (degrees) and its velocity, chasing the expressions' angle.
  roll: number
  rollV: number
  pulseV: number
}

// The head's inertial mass, shared by both eyes: where it hangs (world) and how
// fast it moves, and where the head was last frame.
interface Inertia {
  p: [number, number, number]
  v: [number, number, number]
  head: [number, number, number]
  // Its lag in the head's frame, in iris radii: (right, up).
  offset: [number, number]
}

interface EyeRest {
  // The iris in BIND space — where restPos puts it — and its half extents.
  centre: [number, number, number]
  radius: [number, number]
  // Iris centre to the eye bone's head: what a turn moves the iris by, per
  // radian.
  lever: number
  // The vertices it was measured from (empty for a fallback), and the bone
  // that carries them (−1 when none does).
  verts: number[]
  bone: number
  // The eye bone's head, which the roll turns about (null: no eye bone).
  pivot: [number, number, number] | null
}

// A vertex morph's pull on an iris: the mean offset it gives the iris's
// vertices at weight 1. Morphs add linearly, so the iris's centre under any
// mix of them is the rest centre plus these, weighted.
interface MorphShift {
  morph: number
  d: [number, number, number]
}

const newState = (): EyeState => ({
  closure: 0,
  prev: 0,
  speed: 0,
  trend: 0,
  x: 0,
  v: 0,
  age: ELASTIC_S,
  squash: 0,
  squashV: 0,
  lag: [0, 0],
  lagV: [0, 0],
  look: [0, 0],
  pulse: 0,
  roll: 0,
  rollV: 0,
  pulseV: 0,
})

export class JellyEyes {
  strength: number
  idle: boolean
  // Scene time stepped so far, and the idle taps: how many, and when the next.
  private clock = 0
  private taps = 0
  private nextTap = idleGap(0)
  // The current tap's knock, 1 at the tap and fading over IDLE_TAP_S.
  private tapKnock = 0
  private readonly rest: [EyeRest, EyeRest]
  private readonly state: [EyeState, EyeState] = [newState(), newState()]
  private readonly morphs: [number[], number[], number[]]
  private readonly eyeBones: [number, number]
  private readonly headBone: number
  private fresh = true
  // Each iris's centre THIS frame: the rest centre carried by the morphs that
  // move the iris (a pupil-up, a look-left) — the vertex shader scales the
  // morphed iris, so a centre left at rest scales it about a point it has
  // already left, and it drifts off as it shrinks and grows.
  private readonly shifts: [MorphShift[], MorphShift[]]
  private readonly centre: [[number, number, number], [number, number, number]]
  private readonly inertia: Inertia = { p: [0, 0, 0], v: [0, 0, 0], head: [0, 0, 0], offset: [0, 0] }
  // The highlight layers' draw ids (MaterialUniforms.materialId), up to two.
  readonly highlights: number[]
  // Each eye's closing expressions with their roll, and the angle they set this frame (degrees).
  private readonly rolls: [{ morph: number; deg: number }[], { morph: number; deg: number }[]]
  private readonly roll: [number, number] = [0, 0]

  constructor(model: Model, options: JellyEyeOptions) {
    this.strength = clampStrength(options.strength)
    this.idle = options.idle ?? true
    const bones = model.getSkeleton().bones
    const index = (name: string) => bones.findIndex((b) => b.name === name)
    const both = index("両目")
    const l = index("左目")
    const r = index("右目")
    this.headBone = index("頭")
    const all = model.getMorphing().morphs
    const names = all.map((m) => m.name)
    // The weights read are resolved, so a group whose every target is named
    // (Reze's まばたき is ウィンク２ + ウィンク２右) arrives through its
    // leaves; counted as well, a blink rolled the iris twice as far.
    const known = new Set([...BOTH, ...LEFT, ...RIGHT])
    const resolved = (i: number) => {
      const refs = all[i].groupReferences
      return !!refs?.length && refs.every((r) => known.has(names[r.morphIndex]))
    }
    const morphIndex = (n: string) => {
      const i = names.indexOf(n)
      return i >= 0 && !resolved(i) ? i : -1
    }
    const find = (list: string[]) => list.map(morphIndex).filter((i) => i >= 0)
    this.morphs = [find(BOTH), find(LEFT), find(RIGHT)]
    const rolls = (list: string[]) => list.map((n) => ({ morph: morphIndex(n), deg: ROLL[n] ?? 0 })).filter((r) => r.morph >= 0 && r.deg !== 0)
    this.rolls = [rolls([...BOTH, ...LEFT]), rolls([...BOTH, ...RIGHT])]
    this.rest = findIrises(model, l, r, both)
    // The bone that carries each iris, which may be under the eye bone.
    this.eyeBones = [this.rest[0].bone, this.rest[1].bone]
    this.shifts = [irisShifts(model, this.rest[0].verts), irisShifts(model, this.rest[1].verts)]
    this.centre = [[...this.rest[0].centre], [...this.rest[1].centre]]
    this.highlights = findHighlights(model, this.rest)
  }

  /** Advance both springs by dt seconds of scene time. */
  step(dt: number, model: Model): void {
    const weights = model.getEffectiveMorphWeights()
    for (let e = 0; e < 2; e++) {
      const c = this.centre[e]
      const r = this.rest[e].centre
      c[0] = r[0]
      c[1] = r[1]
      c[2] = r[2]
      for (const { morph, d } of this.shifts[e]) {
        const w = weights[morph] ?? 0
        if (w === 0) continue
        c[0] += d[0] * w
        c[1] += d[1] * w
        c[2] += d[2] * w
      }
    }
    const sum = (list: number[]) => list.reduce((s, i) => s + (weights[i] ?? 0), 0)
    const both = sum(this.morphs[0])
    const closure = [Math.min(1, both + sum(this.morphs[1])), Math.min(1, both + sum(this.morphs[2]))]
    for (let e = 0; e < 2; e++) this.roll[e] = this.rolls[e].reduce((a, r) => a + r.deg * Math.max(0, Math.min(1, weights[r.morph] ?? 0)), 0)
    const head = this.headBone >= 0 ? model.getBoneWorldMatrixAt(this.headBone) : null
    const seek = this.fresh || dt > SEEK || dt < 0
    if (dt > 0) this.clock += dt
    if (dt > 0) this.tapKnock *= Math.exp(-dt / IDLE_TAP_S)
    if (seek) this.tapKnock = 0
    else if (this.idle && this.clock >= this.nextTap) {
      this.tapKnock = IDLE_TAP
      this.taps++
      this.nextTap = this.clock + idleGap(this.taps)
    }
    this.stepInertia(dt, head, seek)
    for (let e = 0; e < 2; e++) {
      const s = this.state[e]
      s.closure = closure[e]
      // Her HEAD's turn, not her eyes': eye tracking turns the eyes after the
      // camera, so the eyes' own turn would make the camera shake the jelly.
      // Only her motion — the head turning and moving (the inertia) — and her
      // blinks do.
      look(head, null, s.look)
      const knock = seek ? 0 : Math.max(this.tapKnock, Math.tanh(Math.hypot(...this.offset(s, e)) / PULSE_REF))
      const squashTarget = MOTION_SQUASH * knock
      if (seek) {
        // Landing on a pose is no blink: the spring rests where it lands.
        s.prev = s.closure
        s.speed = s.trend = s.x = s.v = 0
        s.age = ELASTIC_S
        s.squash = squashTarget
        s.squashV = 0
        s.lag[0] = s.look[0]
        s.lag[1] = s.look[1]
        s.lagV[0] = s.lagV[1] = 0
        s.pulse = s.pulseV = 0
        s.roll = this.roll[e]
        s.rollV = 0
        continue
      }
      blink(s, dt)
      const steps = Math.ceil(dt / SUBSTEP)
      const h = steps > 0 ? dt / steps : 0
      const pulseTarget = PULSE_MAX * knock
      for (let i = 0; i < steps; i++) {
        s.pulseV += h * spring(s.pulse - pulseTarget, s.pulseV, PULSE_HZ, PULSE_ZETA)
        s.pulse += h * s.pulseV
        s.squashV += h * spring(s.squash - squashTarget, s.squashV, SQUASH_HZ, SQUASH_ZETA)
        s.squash += h * s.squashV
        s.rollV += h * spring(s.roll - this.roll[e], s.rollV, ROLL_HZ, ROLL_ZETA)
        s.roll += h * s.rollV
        for (let k = 0; k < 2; k++) {
          s.lagV[k] += h * spring(s.lag[k] - s.look[k], s.lagV[k], LAG_HZ, LAG_ZETA)
          s.lag[k] += h * s.lagV[k]
        }
      }
    }
    this.fresh = false
  }

  /** Write jelly[0..11] into an ObjectLight copy (floats 48..95). */
  write(o: Float32Array): void {
    const [L, R] = this.centre
    const g = this.strength
    o.set([L[0], L[1], L[2], g], 48)
    // The highlight layers' draw ids, two packed in one float (exact to 4095).
    const [a = 0, b = 0] = this.highlights
    o.set([R[0], R[1], R[2], a + b * 4096], 52)
    // One knock swell for both: the mean of the two eyes', so they swell together.
    const pulse = (this.state[0].pulse + this.state[1].pulse) / 2
    const omega = 2 * Math.PI * SPRING_HZ
    // Strength scales every departure from rest, then the bounds hold: the
    // swell under 1.08 tall so it cannot rise past the upper lid.
    const by = (k: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, 1 + (k - 1) * g))
    for (let e = 0; e < 2; e++) {
      const s = this.state[e]
      const rest = this.rest[e]
      const r = (rest.radius[0] + rest.radius[1]) / 2
      // The MME spring's place, and its velocity in the same units.
      const x = s.x
      const vn = s.v / omega
      const gain = (k: 0 | 1) => SCALE_GAIN[k] * (x >= 0 ? 1 : REBOUND[k])
      const kx = (1 + x * gain(0)) * (1 - 0.8 * s.squash + pulse)
      const ky = (1 + x * gain(1)) * (1 - SQUEEZE_Y * s.closure) * (1 - s.squash + pulse)
      o.set([rest.radius[0], rest.radius[1], s.closure, this.eyeBones[e]], 56 + e * 4)
      // [4]/[5]: scale across and down, the iris's twist, the highlight's turn.
      o.set([by(kx, 0.78, 1.25), by(ky, 0.7, 1.08), (x * TWIST_POS + vn * TWIST_VEL) * DEG * g, vn * HL_TURN * DEG * g], 64 + e * 4)
      // [6]/[7]: the iris's drift, then the highlight's on top (bind units).
      // The tremble is held random spots on the scene clock, so an export
      // trembles where the screen did, and a pause holds it.
      const tick = (hz: number, seed: number) => (this.idle ? hashSigned(Math.floor(this.clock * hz) + seed + e * 101) : 0)
      o.set(
        [
          (x * DRIFT_POS[0] + vn * DRIFT_VEL[0]) * r * g,
          (x * DRIFT_POS[1] + vn * DRIFT_VEL[1]) * r * g,
          (vn * HL_SWAY[0] + tick(JITTER_HZ[0], 17) * JITTER[0]) * r * g,
          (vn * HL_SWAY[1] + tick(JITTER_HZ[1], 53) * JITTER[1]) * r * g,
        ],
        72 + e * 4,
      )
      // [8]/[9]: the roll's pivot and angle; [10]/[11]: the highlight's own
      // scale, and how far the iris comes forward (bind units).
      const pivot = rest.pivot
      o.set(pivot ? [pivot[0], pivot[1], pivot[2], (this.roll[e] + (s.roll - this.roll[e]) * ROLL_SWING) * DEG * g] : [0, 0, 0, 0], 80 + e * 4)
      const hs = (k: 0 | 1) => Math.max(HL_SCALE_MIN, Math.min(HL_SCALE_MAX, 1 + x * HL_SCALE[k] * g))
      o.set([hs(0), hs(1), Math.min(Math.abs(x), SPRING_LIMIT) * PUSH_ELASTIC * r * g, 0], 88 + e * 4)
    }
  }

  /** Where the iris sits off rest, in iris radii (right, up): the lag through
   *  the eye's lever, plus the inertia. */
  private offset(s: EyeState, e: number): [number, number] {
    const angle = (x: number) => Math.max(-LAG_MAX, Math.min(LAG_MAX, x * LAG_GAIN))
    const rest = this.rest[e]
    const r = (rest.radius[0] + rest.radius[1]) / 2
    const per = rest.lever / r
    const io = this.inertia.offset
    const x = angle(s.lag[0] - s.look[0]) * per + io[0]
    const y = angle(s.lag[1] - s.look[1]) * per + io[1]
    const n = Math.hypot(x, y)
    const cap = n > OFFSET_MAX ? OFFSET_MAX / n : 1
    return [x * cap, y * cap]
  }

  /** The mass on the head: semi-implicit steps toward where the head is, the
   *  head moving linearly across the frame. */
  private stepInertia(dt: number, head: Float32Array | null, seek: boolean): void {
    const m = this.inertia
    if (!head) {
      m.offset[0] = m.offset[1] = 0
      return
    }
    const now: [number, number, number] = [head[12], head[13], head[14]]
    if (seek) {
      m.p = [...now]
      m.v = [0, 0, 0]
      m.head = [...now]
      m.offset[0] = m.offset[1] = 0
      return
    }
    const hv = [0, 1, 2].map((k) => (dt > 0 ? (now[k] - m.head[k]) / dt : 0))
    const steps = Math.ceil(dt / SUBSTEP)
    const h = steps > 0 ? dt / steps : 0
    for (let i = 1; i <= steps; i++) {
      const t = i / steps
      for (let k = 0; k < 3; k++) {
        const at = m.head[k] + (now[k] - m.head[k]) * t
        m.v[k] += h * spring(m.p[k] - at, m.v[k] - hv[k], INERTIA_HZ, INERTIA_ZETA)
        m.p[k] += h * m.v[k]
      }
    }
    m.head = now
    const u = [m.p[0] - now[0], m.p[1] - now[1], m.p[2] - now[2]]
    const right = [head[0], head[1], head[2]]
    const up = [head[4], head[5], head[6]]
    const size = Math.max(1e-3, (this.rest[0].radius[0] + this.rest[0].radius[1] + this.rest[1].radius[0] + this.rest[1].radius[1]) / 4)
    const cap = (x: number) => Math.max(-OFFSET_MAX, Math.min(OFFSET_MAX, x))
    m.offset[0] = cap((dot(u, right) / Math.hypot(...right) / size) * INERTIA_GAIN)
    m.offset[1] = cap((dot(u, up) / Math.hypot(...up) / size) * INERTIA_GAIN)
  }
}

/** Zero jelly[0..11] — strength 0 is what the shader reads as off. */
export function clearJellyEyes(o: Float32Array): void {
  o.fill(0, 48, 96)
}

/** A held random value in −1..1 for an integer tick, as the MME hashes it. */
function hashSigned(n: number): number {
  const x = Math.sin(n * 12.9898 + 78.233) * 43758.5453
  return (x - Math.floor(x)) * 2 - 1
}

/**
 * One frame of the MME's blink spring for one eye (JellyEye_StateUpdate.fxsub,
 * JE_ComputeNextState), from its closure this frame.
 */
function blink(s: EyeState, dt: number): void {
  if (!(dt > 0)) return
  dt = Math.min(dt, MAX_STEP)
  const omega = 2 * Math.PI * SPRING_HZ
  const clamp = (x: number, m: number) => Math.max(-m, Math.min(m, x))
  const speed = clamp((s.closure - s.prev) / dt, MORPH_SPEED_LIMIT)
  // The change of SPEED, not of signed velocity: gaining pace or losing it.
  const accel = clamp((Math.abs(speed) - Math.abs(s.speed)) / dt, ACCEL_LIMIT)
  const trend = accel > ACCEL_MIN ? 1 : accel < -ACCEL_MIN ? -1 : 0
  // One run of gaining (or losing) pace is one kick.
  const kicked = trend !== 0 && trend !== s.trend
  let latched = trend !== 0 ? trend : s.trend
  if (trend === 0 && Math.abs(speed) < DRIVE_SPEED_MIN && Math.abs(s.speed) < DRIVE_SPEED_MIN) latched = 0
  // While the lids move, a target the spring chases, eased by how fast.
  const amount = Math.max(0, Math.min(1, (Math.abs(speed) - DRIVE_SPEED_MIN) / (DRIVE_SPEED_FULL - DRIVE_SPEED_MIN))) ** DRIVE_CURVE
  const target = amount * (speed > DRIVE_SPEED_MIN ? CLOSE_DRIVE : speed < -DRIVE_SPEED_MIN ? OPEN_DRIVE : 0)
  const driven = Math.abs(target) > 1e-5
  const x0 = s.x
  let v0 = s.v
  if (kicked) v0 += (trend > 0 ? ACCEL_KICK : -RELEASE_KICK) * Math.min(1, Math.abs(accel) / ACCEL_FULL) * omega
  const age = driven || kicked ? 0 : Math.min(s.age + dt, ELASTIC_S)
  let x = 0
  let v = 0
  if (driven || kicked || s.age < ELASTIC_S || Math.abs(x0) > STOP_EPS || Math.abs(v0) / omega > STOP_EPS) {
    // The exact under-damped step about the target.
    const decay = Math.LN2 / HALF_LIFE
    const envelope = Math.exp(-decay * dt)
    const c = Math.cos(omega * dt)
    const sn = Math.sin(omega * dt)
    const rel = x0 - target
    x = target + envelope * (rel * c + ((v0 + decay * rel) * sn) / omega)
    v = envelope * (v0 * c - ((decay * v0 + (decay * decay + omega * omega) * rel) * sn) / omega)
  }
  x = clamp(x, SPRING_LIMIT)
  v = clamp(v, omega * SPRING_LIMIT * 2)
  s.prev = s.closure
  s.speed = speed
  s.trend = latched
  const settled = !driven && age >= MIN_RELEASE && Math.abs(x) <= STOP_EPS && Math.abs(v) / omega <= STOP_EPS
  if (settled || (!driven && age >= ELASTIC_S)) {
    s.x = s.v = 0
    s.age = ELASTIC_S
  } else {
    s.x = x
    s.v = v
    s.age = age
  }
}

/**
 * The highlight layers: a material drawn over an iris — after it, on the same
 * disc, made of vertices the eye bones carry — or one named as a highlight.
 * As draw ids (1-based over the materials that draw, as the engine numbers
 * MaterialUniforms.materialId), at most two. A rig whose highlight is painted
 * into the iris has none, and only the iris moves.
 *
 * A layer drawn on vertices an earlier layer already uses is a copy of that
 * layer (白里黑子's ☆ and ❤ are star and heart irises on the iris's own
 * vertices), so it moves with it. Slid as a highlight, a copy of the dome sits
 * in front of the iris on one half, flipping sides with the tremble.
 */
function findHighlights(model: Model, rest: [EyeRest, EyeRest]): number[] {
  const materials = model.getMaterials?.() ?? []
  const { positions, indices } = model.getGeometry()
  const iris = new Set([...rest[0].verts, ...rest[1].verts])
  if (!materials.length || !indices.length || !iris.size) return []
  const layers: { id: number; name: string; minX: number; maxX: number; minY: number; maxY: number }[] = []
  const used = new Set<number>()
  let start = 0
  let id = 0
  for (const m of materials) {
    const count = m.vertexCount
    if (count === 0) continue
    id++
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    let copy = true
    const mine: number[] = []
    for (let k = start; k < start + count; k++) {
      const v = indices[k]
      if (!iris.has(v)) continue
      if (!used.has(v)) copy = false
      mine.push(v)
      minX = Math.min(minX, positions[v * 3])
      maxX = Math.max(maxX, positions[v * 3])
      minY = Math.min(minY, positions[v * 3 + 1])
      maxY = Math.max(maxY, positions[v * 3 + 1])
    }
    start += count
    for (const v of mine) used.add(v)
    if (minX <= maxX && !copy) layers.push({ id, name: m.name, minX, maxX, minY, maxY })
  }
  const out: number[] = []
  layers.forEach((l, i) => {
    const cx = (l.minX + l.maxX) / 2
    const cy = (l.minY + l.maxY) / 2
    const over = layers.slice(0, i).some((u) => cx >= u.minX && cx <= u.maxX && cy >= u.minY && cy <= u.maxY)
    if ((over || HIGHLIGHT.test(l.name)) && out.length < 2) out.push(l.id)
  })
  return out
}

export function clampStrength(s: number | undefined): number {
  return Math.max(0, Math.min(2, s ?? 1))
}

/** Acceleration of a damped spring at displacement x, velocity v. */
function spring(x: number, v: number, hz: number, zeta: number): number {
  const w = 2 * Math.PI * hz
  return -w * w * x - 2 * zeta * w * v
}

/**
 * Where an eye looks, as (yaw, pitch) radians: the eye in the head's frame,
 * plus a share of the head in the model's. Bone matrices are column-major and
 * MMD rests facing −Z, so forward is the negated third column.
 */
function look(eye: Float32Array | null, head: Float32Array | null, out: [number, number]): void {
  out[0] = out[1] = 0
  if (!eye) return
  const f = [-eye[8], -eye[9], -eye[10]]
  if (head && head !== eye) {
    const right = [head[0], head[1], head[2]]
    const up = [head[4], head[5], head[6]]
    const hf = [-head[8], -head[9], -head[10]]
    const fz = dot(f, hf)
    out[0] = Math.atan2(dot(f, right), fz)
    out[1] = Math.atan2(dot(f, up), fz)
    out[0] += HEAD_SHARE * Math.atan2(hf[0], -hf[2])
    out[1] += HEAD_SHARE * Math.asin(Math.max(-1, Math.min(1, hf[1] / Math.hypot(hf[0], hf[1], hf[2]))))
  } else {
    // Yaw positive toward +X, as in the head-frame branch: the shader warps
    // along bind-space +X, so the two must agree.
    out[0] = Math.atan2(f[0], -f[2])
    out[1] = Math.asin(Math.max(-1, Math.min(1, f[1] / Math.hypot(f[0], f[1], f[2]))))
  }
}

const dot = (a: number[], b: number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

/**
 * The iris out of everything one eye carries: grouped by material, the group
 * whose centre is nearest the eye bone, and any other group centred on it
 * (within its own size) — layers of the same eye. Without materials (or a
 * bone), the set as it came.
 */
function irisOnly(model: Model, set: number[], bone: [number, number, number] | null): number[] {
  const materials = model.getMaterials?.() ?? []
  const { positions, indices } = model.getGeometry()
  if (set.length < 3 || !bone || !materials.length || !indices.length) return set
  const material = new Int32Array(positions.length / 3).fill(-1)
  let start = 0
  materials.forEach((m, i) => {
    for (let k = start; k < start + m.vertexCount; k++) if (material[indices[k]] < 0) material[indices[k]] = i
    start += m.vertexCount
  })
  const groups = new Map<number, number[]>()
  for (const v of set) {
    const g = groups.get(material[v])
    if (g) g.push(v)
    else groups.set(material[v], [v])
  }
  if (groups.size < 2) return set
  const boxes = [...groups.values()].map((vs) => {
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
    for (const v of vs) {
      minX = Math.min(minX, positions[v * 3])
      maxX = Math.max(maxX, positions[v * 3])
      minY = Math.min(minY, positions[v * 3 + 1])
      maxY = Math.max(maxY, positions[v * 3 + 1])
    }
    return { vs, x: (minX + maxX) / 2, y: (minY + maxY) / 2, size: Math.max(maxX - minX, maxY - minY) / 2 }
  })
  const near = boxes.reduce((a, b) => (Math.hypot(b.x - bone[0], b.y - bone[1]) < Math.hypot(a.x - bone[0], a.y - bone[1]) ? b : a))
  return boxes.filter((b) => Math.hypot(b.x - near.x, b.y - near.y) <= near.size).flatMap((b) => b.vs)
}

/** Every vertex morph that moves this iris, with the mean offset it gives it. */
function irisShifts(model: Model, verts: number[]): MorphShift[] {
  if (!verts.length) return []
  const inIris = new Set(verts)
  const out: MorphShift[] = []
  model.getMorphing().morphs.forEach((m, morph) => {
    const d: [number, number, number] = [0, 0, 0]
    for (const o of m.vertexOffsets ?? []) {
      if (!inIris.has(o.vertexIndex)) continue
      d[0] += o.positionOffset[0]
      d[1] += o.positionOffset[1]
      d[2] += o.positionOffset[2]
    }
    d[0] /= verts.length
    d[1] /= verts.length
    d[2] /= verts.length
    if (Math.hypot(...d) > 1e-6) out.push({ morph, d })
  })
  return out
}

/**
 * Each iris in bind space: the vertices the eye bones carry are exactly the
 * part of the eye that turns to look — the iris and pupil — so their centroid
 * is its centre and their spread its size. A rig that carries both eyes on
 * 両目 alone is split by side. With no carried vertices at all, the eye bones'
 * own heads stand in, sized from how far apart they are.
 */
function findIrises(model: Model, l: number, r: number, both: number): [EyeRest, EyeRest] {
  const bones = model.getSkeleton().bones
  const head = (i: number): [number, number, number] | null => {
    if (i < 0) return null
    const p: [number, number, number] = [0, 0, 0]
    for (let b = i; b >= 0; b = bones[b].parentIndex) {
      p[0] += bones[b].bindTranslation[0]
      p[1] += bones[b].bindTranslation[1]
      p[2] += bones[b].bindTranslation[2]
    }
    return p
  }
  const lHead = head(l)
  const rHead = head(r)
  // +X is the model's left; trust the bones when they say otherwise.
  const leftSign = lHead && rHead ? Math.sign(lHead[0] - rHead[0]) || 1 : 1
  const { positions } = model.getGeometry()
  const { joints, weights } = model.getSkinning()
  // Which eye each bone turns with: an eye bone, or any bone under one. Rigs
  // differ in which carries the iris — some hang it on the eye bone itself,
  // some on its tip (左目先/右目先), which turns with it all the same.
  const owner = bones.map((_, i) => {
    for (let b = i; b >= 0; b = bones[b].parentIndex) {
      if (b === l) return 0
      if (b === r) return 1
      if (b === both) return 2
    }
    return -1
  })
  // Per side, the vertices and how much iris weight each bone carries: the
  // heaviest is the one the vertex shader scales by (one bone per eye).
  //
  // The per-eye trees first. 両目 is a fallback for rigs that carry both irises
  // on it alone: where 左目/右目 do carry them, anything hung on 両目 is
  // something else that turns with the eyes — an expression overlay above them
  // (blush lines, sparkles) — and split by side it stretches each "iris" past
  // any iris, so the size check rejects both and the eyes never move.
  const gather = (useBoth: boolean) => {
    const sets: [number[], number[]] = [[], []]
    const n = positions.length / 3
    for (let v = 0; v < n; v++) {
      const w3 = [0, 0, 0]
      for (let k = 0; k < 4; k++) {
        const o = owner[joints[v * 4 + k]]
        if (o >= 0) w3[o] += weights[v * 4 + k]
      }
      let side: 0 | 1
      if (w3[0] >= 128) side = 0
      else if (w3[1] >= 128) side = 1
      else if (useBoth && w3[2] >= 128) side = Math.sign(positions[v * 3]) === leftSign ? 0 : 1
      else continue
      sets[side].push(v)
    }
    return sets
  }
  let gathered = gather(false)
  if (gathered[0].length < 3 && gathered[1].length < 3) gathered = gather(true)
  // Of what turns with an eye, the iris is the mesh nearest its bone, with the
  // layers stacked on it (an eye white, a highlight sheet over the same disc).
  // A mesh elsewhere that the eye bones also carry — an expression overlay
  // above the eyes — is left out, or it stretches the "iris" past any iris.
  const sets: [number[], number[]] = [irisOnly(model, gathered[0], lHead), irisOnly(model, gathered[1], rHead)]
  const carried: [Map<number, number>, Map<number, number>] = [new Map(), new Map()]
  for (const side of [0, 1] as const) {
    for (const v of sets[side]) {
      for (let k = 0; k < 4; k++) {
        const j = joints[v * 4 + k]
        if (owner[j] >= 0) carried[side].set(j, (carried[side].get(j) ?? 0) + weights[v * 4 + k])
      }
    }
  }
  const carrier = (side: 0 | 1): number => {
    let best = -1
    let most = 0
    for (const [j, w] of carried[side]) if (w > most) [best, most] = [j, w]
    return best
  }
  const apart = lHead && rHead ? Math.hypot(lHead[0] - rHead[0], lHead[1] - rHead[1]) : 0
  const fallbackRadius = apart > 0 ? apart * 0.22 : 0.3
  const rest = (set: number[], bone: [number, number, number] | null, side: number, carrierBone: number): EyeRest => {
    if (set.length < 3) {
      const c: [number, number, number] = bone ?? [side * fallbackRadius * 2, 0, 0]
      return { centre: c, radius: [fallbackRadius, fallbackRadius], lever: fallbackRadius * 2, verts: [], bone: -1, pivot: null }
    }
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, z = 0
    for (const v of set) {
      const x = positions[v * 3]
      const y = positions[v * 3 + 1]
      minX = Math.min(minX, x)
      maxX = Math.max(maxX, x)
      minY = Math.min(minY, y)
      maxY = Math.max(maxY, y)
      z += positions[v * 3 + 2]
    }
    // A carried set wider than the gap between the eyes is not an iris (a
    // whole-face rig weighted to the eye bone): fall back rather than wobble
    // half the face.
    const rx = (maxX - minX) / 2
    const ry = (maxY - minY) / 2
    if (apart > 0 && (rx > apart * 0.6 || ry > apart * 0.8)) {
      return { centre: bone ?? [side * fallbackRadius * 2, 0, 0], radius: [fallbackRadius, fallbackRadius], lever: fallbackRadius * 2, verts: [], bone: -1, pivot: null }
    }
    const centre: [number, number, number] = [(minX + maxX) / 2, (minY + maxY) / 2, z / set.length]
    const radius: [number, number] = [Math.max(rx, 1e-3), Math.max(ry, 1e-3)]
    // An eye bone sitting on the iris itself would give no lever; such a rig
    // still turns its eyes visibly, so read it as a typical eyeball's.
    const r = (radius[0] + radius[1]) / 2
    const lever = Math.max(bone ? Math.hypot(centre[0] - bone[0], centre[1] - bone[1], centre[2] - bone[2]) : 0, r * 1.5)
    return { centre, radius, lever, verts: set, bone: carrierBone, pivot: bone }
  }
  return [rest(sets[0], lHead, leftSign, carrier(0)), rest(sets[1], rHead, -leftSign, carrier(1))]
}
