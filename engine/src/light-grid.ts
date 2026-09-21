import { LIGHT_GRID_CELLS, LIGHT_MASK_WORDS } from "./shaders/lights"

// Which document lamps can reach which part of the world, so a fragment walks
// only those.
//
// A uniform grid in WORLD space, built on the CPU whenever the document's lamps
// change. Each cell holds one bit per lamp — 128 lamps is one vec4u — set when
// the lamp's sphere, and its cone for a spot, can touch the cell. A fragment
// finds its cell from its world position and loops the set bits.
//
// WORLD SPACE rather than screen tiles, because the answer then belongs to the
// lamps and not to a camera: the mirror pass, an export at another resolution
// and a preview all read the same grid with no per-view work, and it is rebuilt
// only when a lamp moves, never per frame.
//
// THE BOX is where the lamps cluster — their positions padded by the median
// reach — not the union of every sphere. One stage spot reaching 300 units
// would otherwise stretch the box across the whole set, and a fixed cell budget
// spread over it leaves cells wider than the small lamps they are meant to
// separate. A lamp whose reach leaves the box also sets its bit in the OUTSIDE
// mask, which is what a fragment beyond the box reads, so nothing is lost at
// the edge; it is only coarser there.
//
// CONSERVATIVE, never exact: a bit may be set in a cell the lamp does not
// light, never missing from one it does. A missing bit is a hole in the light;
// a spare one is a few instructions.

/** One document lamp, as the grid sees it. */
export type GridLamp = {
  x: number
  y: number
  z: number
  radius: number
  /** Unit aim, away from the lamp. Zero for a point lamp. */
  ax: number
  ay: number
  az: number
  /** Cosine of the half-angle. -1 for a point lamp, which covers everything. */
  cosOuter: number
}

export type LightGrid = {
  origin: [number, number, number]
  /** Edge length of one cubic cell, world units. */
  cell: number
  dims: [number, number, number]
  /** The lamps that reach past the box, for fragments outside it. */
  outside: Uint32Array
  /** LIGHT_MASK_WORDS words per cell, x fastest, then y, then z. */
  cells: Uint32Array
}

function setBit(words: Uint32Array, base: number, lamp: number): void {
  words[base + (lamp >> 5)] |= 1 << (lamp & 31)
}

export function buildLightGrid(lamps: GridLamp[], maxCells = LIGHT_GRID_CELLS): LightGrid {
  const outside = new Uint32Array(LIGHT_MASK_WORDS)
  const live = lamps
    .map((l, i) => ({ l, i }))
    .filter(({ l }) => l.radius > 0 && [l.x, l.y, l.z, l.radius].every(Number.isFinite))
  if (!live.length) return { origin: [0, 0, 0], cell: 1, dims: [0, 0, 0], outside, cells: new Uint32Array(0) }

  const radii = live.map(({ l }) => l.radius).sort((a, b) => a - b)
  const pad = radii[radii.length >> 1]
  const min = [Infinity, Infinity, Infinity]
  const max = [-Infinity, -Infinity, -Infinity]
  for (const { l } of live) {
    const p = [l.x, l.y, l.z]
    for (let k = 0; k < 3; k++) {
      min[k] = Math.min(min[k], p[k] - pad)
      max[k] = Math.max(max[k], p[k] + pad)
    }
  }
  const extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]]
  let cell = Math.cbrt((extent[0] * extent[1] * extent[2]) / maxCells)
  let dims: [number, number, number]
  // ceil() rounds every axis up, so the first guess can overshoot the budget by
  // a few percent; widen the cell until it fits.
  for (;;) {
    dims = [0, 1, 2].map((k) => Math.max(1, Math.ceil(extent[k] / cell))) as [number, number, number]
    if (dims[0] * dims[1] * dims[2] <= maxCells) break
    cell *= 1.02
  }
  const cells = new Uint32Array(dims[0] * dims[1] * dims[2] * LIGHT_MASK_WORDS)
  // Half the cell's diagonal: the sphere a spot's cone is tested against.
  const halfDiagonal = (cell * Math.sqrt(3)) / 2

  for (const { l, i } of live) {
    // A hair of slack on the reach: the shader finds its cell in f32 from a
    // reciprocal, the builder in f64 from a division, and a fragment on a cell
    // face may land in either neighbour.
    const r = l.radius + cell * 1e-4
    const p = [l.x, l.y, l.z]
    const lo = [0, 0, 0]
    const hi = [0, 0, 0]
    let leaves = false
    for (let k = 0; k < 3; k++) {
      const a = Math.floor((p[k] - r - min[k]) / cell)
      const b = Math.floor((p[k] + r - min[k]) / cell)
      if (a < 0 || b >= dims[k]) leaves = true
      lo[k] = Math.max(a, 0)
      hi[k] = Math.min(b, dims[k] - 1)
    }
    if (leaves) setBit(outside, 0, i)
    if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) continue

    const spot = l.cosOuter > -1
    const outer = Math.acos(Math.min(Math.max(l.cosOuter, -1), 1))
    for (let z = lo[2]; z <= hi[2]; z++) {
      for (let y = lo[1]; y <= hi[1]; y++) {
        for (let x = lo[0]; x <= hi[0]; x++) {
          const b0 = [min[0] + x * cell, min[1] + y * cell, min[2] + z * cell]
          // The sphere against the cell's box: the nearest point of the box.
          let d2 = 0
          for (let k = 0; k < 3; k++) {
            const q = Math.min(Math.max(p[k], b0[k]), b0[k] + cell)
            d2 += (q - p[k]) * (q - p[k])
          }
          if (d2 > r * r) continue
          if (spot) {
            // The cone against the cell's bounding sphere: the cell can be lit
            // when the angle to its centre is within the cone's half-angle plus
            // the angle that sphere subtends.
            const v = [b0[0] + cell / 2 - p[0], b0[1] + cell / 2 - p[1], b0[2] + cell / 2 - p[2]]
            const len = Math.hypot(v[0], v[1], v[2])
            if (len > halfDiagonal) {
              const limit = outer + Math.asin(halfDiagonal / len)
              const cos = (v[0] * l.ax + v[1] * l.ay + v[2] * l.az) / len
              if (limit < Math.PI && Math.acos(Math.min(Math.max(cos, -1), 1)) > limit) continue
            }
          }
          setBit(cells, ((z * dims[1] + y) * dims[0] + x) * LIGHT_MASK_WORDS, i)
        }
      }
    }
  }
  return { origin: [min[0], min[1], min[2]], cell, dims, outside, cells }
}
