// Persistent contact manifolds — the cache that makes warm starting possible.
//
// Bullet 2.75 keeps a btPersistentManifold per body pair holding up to 4 points,
// each carrying the impulse it converged to last step. At setup the solver seeds
// each row with `cp.m_appliedImpulse * m_warmstartingFactor` (0.85) and applies
// it immediately, so a resting stack starts the substep already holding roughly
// the load it needs instead of rediscovering it from zero every time.
//
// That is not a nicety here: penetration recovery rides in the contact velocity
// row as a Baumgarte term, and a row rebuilt from zero each substep overshoots
// it. Warm starting on its own was measured WORSE on this engine — but that was
// against a solver with no bias term and a position-correction pass, which is a
// different system. The two are one design in Bullet and are ported as one.
//
// Points are matched by proximity in each body's own local frame, which is what
// btPersistentManifold does — a world-space match would drift with the body.

const MAX_POINTS = 4
/** Match radius, in model units. Bullet's gContactBreakingThreshold is 0.02;
 *  ours is the contact margin, so a point that merely slid along a face is
 *  still recognised as the same point rather than dropped and rebuilt. */
const MATCH_DIST_SQ = 0.04 * 0.04

interface Point {
  /** Contact point in each body's local frame (lever arms are world-space and
   *  rotate with the body, so they cannot be compared across substeps). */
  lax: number; lay: number; laz: number
  lbx: number; lby: number; lbz: number
  normalImpulse: number
  frictionImpulse1: number
  frictionImpulse2: number
  /** Substeps this point has survived — Bullet disables restitution past
   *  m_restingContactRestitutionThreshold (2) so resting contacts stop bouncing. */
  age: number
  /** Marks points seen this substep; unseen ones are dropped. */
  seen: boolean
}

export class ManifoldCache {
  private pairs = new Map<number, Point[]>()
  private touched = new Set<number>()

  private static key(a: number, b: number): number {
    return a < b ? a * 65536 + b : b * 65536 + a
  }

  /** Look up what this contact point converged to last substep. Returns null
   *  when it is new. */
  find(a: number, b: number, lax: number, lay: number, laz: number): Point | null {
    const list = this.pairs.get(ManifoldCache.key(a, b))
    if (list === undefined) return null
    let best: Point | null = null
    let bestD = MATCH_DIST_SQ
    for (let i = 0; i < list.length; i++) {
      const p = list[i]
      const dx = p.lax - lax, dy = p.lay - lay, dz = p.laz - laz
      const d = dx * dx + dy * dy + dz * dz
      if (d < bestD) { bestD = d; best = p }
    }
    return best
  }

  /** Record what this point converged to, for the next substep to start from. */
  store(
    a: number, b: number,
    lax: number, lay: number, laz: number,
    lbx: number, lby: number, lbz: number,
    normalImpulse: number, frictionImpulse1: number, frictionImpulse2: number,
    age: number,
  ): void {
    const k = ManifoldCache.key(a, b)
    this.touched.add(k)
    let list = this.pairs.get(k)
    if (list === undefined) { list = []; this.pairs.set(k, list) }
    // Replace the nearest existing point, else append; past MAX_POINTS drop the
    // shallowest-held one so the manifold keeps the load-bearing corners.
    let best = -1
    let bestD = MATCH_DIST_SQ
    for (let i = 0; i < list.length; i++) {
      const p = list[i]
      const dx = p.lax - lax, dy = p.lay - lay, dz = p.laz - laz
      const d = dx * dx + dy * dy + dz * dz
      if (d < bestD) { bestD = d; best = i }
    }
    if (best < 0) {
      if (list.length < MAX_POINTS) {
        list.push({ lax, lay, laz, lbx, lby, lbz, normalImpulse, frictionImpulse1, frictionImpulse2, age, seen: true })
        return
      }
      // btPersistentManifold::sortCachedPoints — when a 5th point arrives, drop
      // whichever of the 5 leaves the largest quadrilateral. Area is what keeps
      // a resting box from pivoting; dropping the shallowest instead can leave
      // four nearly-collinear points that pin position but not orientation.
      best = worstAreaIndex(list, lax, lay, laz)
    }
    const p = list[best]
    p.lax = lax; p.lay = lay; p.laz = laz
    p.lbx = lbx; p.lby = lby; p.lbz = lbz
    p.normalImpulse = normalImpulse
    p.frictionImpulse1 = frictionImpulse1
    p.frictionImpulse2 = frictionImpulse2
    p.age = age
    p.seen = true
  }

  /** Drop every point not re-seen this substep, and every pair left empty.
   *  Without this a separated pair keeps handing back a stale impulse. */
  endStep(): void {
    for (const [k, list] of this.pairs) {
      if (!this.touched.has(k)) { this.pairs.delete(k); continue }
      let w = 0
      for (let i = 0; i < list.length; i++) {
        const p = list[i]
        if (!p.seen) continue
        p.seen = false
        list[w++] = p
      }
      list.length = w
      if (w === 0) this.pairs.delete(k)
    }
    this.touched.clear()
  }

  clear(): void {
    this.pairs.clear()
    this.touched.clear()
  }
}


/** Which of the 4 cached points to replace so the surviving quad keeps the most
 *  area once the new point joins it. */
function worstAreaIndex(list: Point[], nx: number, ny: number, nz: number): number {
  let bestIdx = 0
  let bestArea = -1
  for (let drop = 0; drop < list.length; drop++) {
    // The quad is: the new point plus the three survivors.
    const pts: number[][] = [[nx, ny, nz]]
    for (let i = 0; i < list.length; i++) if (i !== drop) pts.push([list[i].lax, list[i].lay, list[i].laz])
    if (pts.length < 4) continue
    // |d0 × d1| over the diagonals — Bullet's area proxy.
    const d0x = pts[0][0] - pts[2][0], d0y = pts[0][1] - pts[2][1], d0z = pts[0][2] - pts[2][2]
    const d1x = pts[1][0] - pts[3][0], d1y = pts[1][1] - pts[3][1], d1z = pts[1][2] - pts[3][2]
    const cx = d0y * d1z - d0z * d1y
    const cy = d0z * d1x - d0x * d1z
    const cz = d0x * d1y - d0y * d1x
    const area = cx * cx + cy * cy + cz * cz
    if (area > bestArea) { bestArea = area; bestIdx = drop }
  }
  return bestIdx
}
