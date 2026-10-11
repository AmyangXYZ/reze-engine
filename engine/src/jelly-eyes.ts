// Jelly eyes — the MMD 果冻眼 / Live2D highlight wobble, for any PMX.
//
// What every working version of the effect has in common, whether it is keyed
// by hand in MMD, a pendulum in Live2D, or a jiggle bone in Blender: an INPUT
// (a blink, a turn), something that REMEMBERS the last frame (a spring), and a
// cheap OUTPUT. Here the output is the iris's SCALE, in place: it flattens and
// bulges as a lid moves (opening too), swells as her head turns or moves, and
// wobbles back. It never slides — tried, it read as the iris coming loose. The
// inputs are her blinks and her head, never her eyes' own turn: eye tracking
// turns the eyes after the camera, and the camera must not shake the jelly.
//
// The springs run on the CPU, stepped by the engine's frame delta — the same
// exact delta renderFrame() hands an export, so a blink bounces identically on
// screen and in the file. Their state rides in the model's ObjectLight
// (jelly[0..5], see materials/common.ts): the vertex shader scales the iris —
// the vertices the eye bones carry — about its centre, and the eye render class
// adds the gem look after its graph (glints, light through the iris, deeper
// colour), so every eye graph becomes jelly without knowing it.

import type { Model } from "./model"

export interface JellyEyeOptions {
  /** How much the eyes give: 0 still, 1 the intended bounce, up to 2. Default 1. */
  strength?: number
}

// The closing morphs, by the names MMD, VRoid and the usual converters give
// them. Group morphs (a smile that closes the eyes) resolve into these through
// the effective weights, so only the leaves need naming.
const BOTH = ["まばたき", "笑い", "はぅ", "blink", "Blink", "eye_close", "Fcl_EYE_Close"]
const LEFT = ["ウィンク", "ウィンク２", "ｳｨﾝｸ", "ｳｨﾝｸ２", "wink", "Wink", "blink_L", "Fcl_EYE_Close_L"]
const RIGHT = ["ウィンク右", "ウィンク２右", "ｳｨﾝｸ右", "ｳｨﾝｸ２右", "wink_R", "Wink_R", "blink_R", "Fcl_EYE_Close_R"]

// The squash: how far a closed eye presses the picture, and the spring that
// follows it — underdamped, so the eye opening overshoots into a stretch and
// settles in two or three wobbles, the "Duang" the effect is named for.
// Only a CHANGE of closure squashes: the eye's closure less a slow follower of
// it, so a blink squeezes then bounces, while a held half-close (a smile)
// leaves the iris at rest instead of squashed for as long as she smiles.
const SQUASH_CLOSED = 0.35
const SQUASH_HZ = 1.4
const SQUASH_ZETA = 0.22
const CLOSURE_FOLLOW_S = 0.35
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
const OFFSET_MAX = 0.06
const HEAD_SHARE = 0.35
// The inertia: the iris has a little mass, hung on a spring from the head, so
// her own motion — a bob, a step, a sway of the dance — leaves it behind and
// it swings back. Model units of lag to iris radii, scaled by INERTIA_GAIN.
const INERTIA_HZ = 3.0
const INERTIA_ZETA = 0.4
const INERTIA_GAIN = 0.04
// The jelly's give: the iris never moves, it swells. The lag and the inertia
// are how hard it is being knocked (in iris radii, as if it could move); that
// knock grows it a little and it settles in a wobble — the scale bounce on top
// of the blink's squash. Iris scale per iris radius of knock.
const PULSE_GAIN = 1.6
const PULSE_HZ = 1.6
const PULSE_ZETA = 0.24
// Integration step, and the jump past which a delta is a seek, not time.
const SUBSTEP = 1 / 240
const SEEK = 0.25

interface EyeState {
  closure: number
  // The slow follower of closure; closure − settled is what squashes.
  settled: number
  squash: number
  squashV: number
  // Lagged look (yaw, pitch) and its velocity; the offset is lagged − actual.
  lag: [number, number]
  lagV: [number, number]
  look: [number, number]
  shine: number
  pulse: number
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
}

const newState = (): EyeState => ({
  closure: 0,
  settled: 0,
  squash: 0,
  squashV: 0,
  lag: [0, 0],
  lagV: [0, 0],
  look: [0, 0],
  shine: 0,
  pulse: 0,
  pulseV: 0,
})

export class JellyEyes {
  strength: number
  private readonly rest: [EyeRest, EyeRest]
  private readonly state: [EyeState, EyeState] = [newState(), newState()]
  private readonly morphs: [number[], number[], number[]]
  private readonly eyeBones: [number, number]
  private readonly headBone: number
  private fresh = true
  private readonly inertia: Inertia = { p: [0, 0, 0], v: [0, 0, 0], head: [0, 0, 0], offset: [0, 0] }

  constructor(model: Model, options: JellyEyeOptions) {
    this.strength = clampStrength(options.strength)
    const bones = model.getSkeleton().bones
    const index = (name: string) => bones.findIndex((b) => b.name === name)
    const both = index("両目")
    const l = index("左目")
    const r = index("右目")
    this.eyeBones = [l >= 0 ? l : both, r >= 0 ? r : both]
    this.headBone = index("頭")
    const names = model.getMorphing().morphs.map((m) => m.name)
    const find = (list: string[]) => list.map((n) => names.indexOf(n)).filter((i) => i >= 0)
    this.morphs = [find(BOTH), find(LEFT), find(RIGHT)]
    this.rest = findIrises(model, l, r, both)
  }

  /** Advance both springs by dt seconds of scene time. */
  step(dt: number, model: Model): void {
    const weights = model.getEffectiveMorphWeights()
    const sum = (list: number[]) => list.reduce((s, i) => s + (weights[i] ?? 0), 0)
    const both = sum(this.morphs[0])
    const closure = [Math.min(1, both + sum(this.morphs[1])), Math.min(1, both + sum(this.morphs[2]))]
    const head = this.headBone >= 0 ? model.getBoneWorldMatrixAt(this.headBone) : null
    const seek = this.fresh || dt > SEEK || dt < 0
    this.stepInertia(dt, head, seek)
    for (let e = 0; e < 2; e++) {
      const s = this.state[e]
      s.closure = closure[e]
      // Her HEAD's turn, not her eyes': eye tracking turns the eyes after the
      // camera, so the eyes' own turn would make the camera shake the jelly.
      // Only her motion — the head turning and moving (the inertia) — and her
      // blinks do.
      look(head, null, s.look)
      if (seek) s.settled = s.closure
      else s.settled += (s.closure - s.settled) * Math.min(1, dt / CLOSURE_FOLLOW_S)
      // Either way a lid moves squashes — opening too, which is the half of a
      // blink you can see (closing, the lid hides the iris anyway).
      const squashTarget = SQUASH_CLOSED * Math.abs(s.closure - s.settled)
      if (seek) {
        s.squash = squashTarget
        s.squashV = 0
        s.lag[0] = s.look[0]
        s.lag[1] = s.look[1]
        s.lagV[0] = s.lagV[1] = 0
        s.shine = 0
        s.pulse = s.pulseV = 0
        continue
      }
      const steps = Math.ceil(dt / SUBSTEP)
      const h = steps > 0 ? dt / steps : 0
      const pulseTarget = PULSE_GAIN * Math.hypot(...this.offset(s, e))
      for (let i = 0; i < steps; i++) {
        s.pulseV += h * spring(s.pulse - pulseTarget, s.pulseV, PULSE_HZ, PULSE_ZETA)
        s.pulse += h * s.pulseV
        s.squashV += h * spring(s.squash - squashTarget, s.squashV, SQUASH_HZ, SQUASH_ZETA)
        s.squash += h * s.squashV
        for (let k = 0; k < 2; k++) {
          s.lagV[k] += h * spring(s.lag[k] - s.look[k], s.lagV[k], LAG_HZ, LAG_ZETA)
          s.lag[k] += h * s.lagV[k]
        }
      }
      // Shine follows the motion: bright while the jelly moves, fading as it
      // settles, eased so a single wobble does not strobe.
      const motion = Math.min(1, 2.5 * Math.abs(s.squash - squashTarget) + 4 * Math.hypot(...this.offset(s, e)))
      s.shine += (motion - s.shine) * Math.min(1, dt * 12)
    }
    this.fresh = false
  }

  /** Write jelly[0..5] into an ObjectLight copy (floats 48..71). */
  write(o: Float32Array): void {
    const [L, R] = this.rest
    o.set([L.centre[0], L.centre[1], L.centre[2], this.strength], 48)
    // One scale bounce for both: the mean of the two eyes', so they swell together.
    const pulse = ((this.state[0].pulse + this.state[1].pulse) / 2) * this.strength
    o.set([R.centre[0], R.centre[1], R.centre[2], pulse], 52)
    for (let e = 0; e < 2; e++) {
      const s = this.state[e]
      const rest = this.rest[e]
      o.set([rest.radius[0], rest.radius[1], s.closure, this.eyeBones[e]], 56 + e * 4)
      // Offset words (65, 66) stay 0: the iris holds its place; only scale.
      o.set([s.squash * this.strength, 0, 0, s.shine], 64 + e * 4)
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

/** Zero jelly[0..5] — strength 0 is what the shader reads as off. */
export function clearJellyEyes(o: Float32Array): void {
  o.fill(0, 48, 72)
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
  const sets: [number[], number[]] = [[], []]
  const n = positions.length / 3
  for (let v = 0; v < n; v++) {
    let wl = 0
    let wr = 0
    let wb = 0
    for (let k = 0; k < 4; k++) {
      const j = joints[v * 4 + k]
      const w = weights[v * 4 + k]
      if (j === l) wl += w
      else if (j === r) wr += w
      else if (j === both) wb += w
    }
    if (wl >= 128) sets[0].push(v)
    else if (wr >= 128) sets[1].push(v)
    else if (wb >= 128) sets[Math.sign(positions[v * 3]) === leftSign ? 0 : 1].push(v)
  }
  const apart = lHead && rHead ? Math.hypot(lHead[0] - rHead[0], lHead[1] - rHead[1]) : 0
  const fallbackRadius = apart > 0 ? apart * 0.22 : 0.3
  const rest = (set: number[], bone: [number, number, number] | null, side: number): EyeRest => {
    if (set.length < 3) {
      const c: [number, number, number] = bone ?? [side * fallbackRadius * 2, 0, 0]
      return { centre: c, radius: [fallbackRadius, fallbackRadius], lever: fallbackRadius * 2 }
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
      return { centre: bone ?? [side * fallbackRadius * 2, 0, 0], radius: [fallbackRadius, fallbackRadius], lever: fallbackRadius * 2 }
    }
    const centre: [number, number, number] = [(minX + maxX) / 2, (minY + maxY) / 2, z / set.length]
    const radius: [number, number] = [Math.max(rx, 1e-3), Math.max(ry, 1e-3)]
    // An eye bone sitting on the iris itself would give no lever; such a rig
    // still turns its eyes visibly, so read it as a typical eyeball's.
    const r = (radius[0] + radius[1]) / 2
    const lever = Math.max(bone ? Math.hypot(centre[0] - bone[0], centre[1] - bone[1], centre[2] - bone[2]) : 0, r * 1.5)
    return { centre, radius, lever }
  }
  return [rest(sets[0], lHead, leftSign), rest(sets[1], rHead, -leftSign)]
}
