// A three-colour sky, fitted to the harmonics an HDRI uses.
//
// The fit is what makes a gradient cost the shader nothing, so the tests are
// about whether it still means the same thing after the round trip: up is the
// sky colour, down is the ground colour, and a flat gradient is the flat world
// it replaces.

import test from "node:test"
import assert from "node:assert/strict"
import { gradientIrradianceSH, evalIrradianceSH } from "../dist/ibl.js"

const rgb = (x, y, z) => ({ x, y, z })
const at = (sh, d) => evalIrradianceSH(sh, { x: d[0], y: d[1], z: d[2] })

test("a uniform gradient is the flat world it replaces", () => {
  // Irradiance from a uniform sky of radiance L is L in every direction — the
  // same identity the flat ambient relies on (E = pi L, Lambert reflects E/pi).
  const sh = gradientIrradianceSH({ sky: rgb(0.4, 0.4, 0.4), equator: rgb(0.4, 0.4, 0.4), ground: rgb(0.4, 0.4, 0.4) })
  for (const d of [[0, 1, 0], [0, -1, 0], [1, 0, 0], [0.577, 0.577, 0.577]]) {
    const c = at(sh, d)
    for (const v of c) assert.ok(Math.abs(v - 0.4) < 0.02, `${v} should be 0.4 along ${d}`)
  }
})

test("up sees the sky and down sees the ground", () => {
  const sh = gradientIrradianceSH({ sky: rgb(0, 0, 1), equator: rgb(0, 0, 0), ground: rgb(1, 0, 0) })
  const up = at(sh, [0, 1, 0])
  const down = at(sh, [0, -1, 0])
  assert.ok(up[2] > up[0], `up is blue: ${up}`)
  assert.ok(down[0] > down[2], `down is red: ${down}`)
  // And it is IRRADIANCE, not the colour itself: a surface facing up is lit by
  // the whole sky above it, so the horizon's black pulls it well under 1.
  assert.ok(up[2] < 1, `up integrates the hemisphere: ${up[2]}`)
})

test("X333's own sky comes out sky-tinted and dim below", () => {
  // The numbers from the scene file: sky, equator, ground.
  const sh = gradientIrradianceSH({
    sky: rgb(0.28622285, 0.78177273, 0.9056604),
    equator: rgb(0, 0.19948709, 0.31132078),
    ground: rgb(0, 0, 0),
  })
  const up = at(sh, [0, 1, 0])
  const down = at(sh, [0, -1, 0])
  assert.ok(up[2] > up[0], "upward fill is blue")
  assert.ok(up[1] > down[1], "a surface facing up is lit more than one facing down")
  assert.ok(down.every((v) => v >= 0), "irradiance is never negative")
})
