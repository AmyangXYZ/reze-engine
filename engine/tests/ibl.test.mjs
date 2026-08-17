// The irradiance projector. Run: npm test.
//
// Checked against analytic skies, because SH projection is exactly the kind of
// arithmetic that looks right while being wrong by a constant factor — and the
// normalisation IS the contract: a uniform sky of radiance 1 must light every
// surface with exactly 1, or an HDRI world is not a drop-in for the flat
// world colour it replaces.

import { test } from "node:test"
import assert from "node:assert/strict"
import { projectIrradianceSH, evalIrradianceSH } from "../dist/ibl.js"

function sky(width, height, fill) {
  const data = new Float32Array(width * height * 4)
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = fill(x / width, y / height)
      const i = (y * width + x) * 4
      data[i] = r
      data[i + 1] = g
      data[i + 2] = b
      data[i + 3] = 1
    }
  }
  return { width, height, data }
}

const NORMALS = [
  { x: 0, y: 1, z: 0 },
  { x: 0, y: -1, z: 0 },
  { x: 1, y: 0, z: 0 },
  { x: 0, y: 0, z: 1 },
  { x: 0.577, y: 0.577, z: 0.577 },
]

test("a uniform sky of radiance 1 lights every normal with exactly 1", () => {
  const sh = projectIrradianceSH(sky(128, 64, () => [1, 0.5, 2]), 1)
  for (const n of NORMALS) {
    const [r, g, b] = evalIrradianceSH(sh, n)
    assert.ok(Math.abs(r - 1) < 0.01, `r at ${JSON.stringify(n)}: ${r}`)
    assert.ok(Math.abs(g - 0.5) < 0.005, `g: ${g}`)
    assert.ok(Math.abs(b - 2) < 0.02, `b: ${b}`)
  }
})

test("a linear-gradient sky matches the analytic irradiance exactly", () => {
  // L(ω) = 1 + ω·up is band-limited at SH1, so the projection is EXACT and the
  // cosine convolution has a closed form: E(n) = 1 + (2/3)·n·up. Up 5/3, side
  // 1, down 1/3 — an answer the code cannot fit to, only be correct about.
  // (A hard-edged cap was tried here first: SH2's known ringing on
  // concentrated sources pushes the down-lobe above the side, and the hand
  // computation agreed with the code to three decimals. Soft skies — every
  // real HDRI — are the operating range.)
  const sh = projectIrradianceSH(sky(256, 128, (u, v) => {
    const ny = Math.cos(Math.PI * (v + 0.5 / 128))
    return [1 + ny, 1 + ny, 1 + ny]
  }), 1)
  const up = evalIrradianceSH(sh, { x: 0, y: 1, z: 0 })[0]
  const side = evalIrradianceSH(sh, { x: 1, y: 0, z: 0 })[0]
  const down = evalIrradianceSH(sh, { x: 0, y: -1, z: 0 })[0]
  assert.ok(Math.abs(up - 5 / 3) < 0.02, `up ${up} vs 5/3`)
  assert.ok(Math.abs(side - 1) < 0.02, `side ${side} vs 1`)
  assert.ok(Math.abs(down - 1 / 3) < 0.02, `down ${down} vs 1/3`)
})

test("a sky lit from +Z lights the +Z-facing surface most — the drawing and the lighting agree", () => {
  // The composite draws u=0.5 at +Z (LH, atan2(x,z) convention). A bright patch
  // at the horizon around u=0.5 must light the normal that FACES it — a
  // convention slip here would light her from behind the visible sun.
  const sh = projectIrradianceSH(
    sky(256, 128, (u, v) => (Math.abs(u - 0.5) < 0.1 && Math.abs(v - 0.5) < 0.2 ? [5, 5, 5] : [0, 0, 0])),
    1,
  )
  const front = evalIrradianceSH(sh, { x: 0, y: 0, z: 1 })[0]
  const back = evalIrradianceSH(sh, { x: 0, y: 0, z: -1 })[0]
  const side = evalIrradianceSH(sh, { x: 1, y: 0, z: 0 })[0]
  assert.ok(front > side && side >= back - 1e-9, `front ${front} / side ${side} / back ${back}`)
})

test("subsampling changes the answer by fractions of a percent, not by its magnitude", () => {
  const gradient = sky(256, 128, (u, v) => [1 + Math.sin(u * 6.28) * 0.5, 1 - v, v * 2])
  const fine = projectIrradianceSH(gradient, 1)
  const coarse = projectIrradianceSH(gradient, 4)
  for (const n of NORMALS) {
    const a = evalIrradianceSH(fine, n)
    const b = evalIrradianceSH(coarse, n)
    for (let c = 0; c < 3; c++) {
      assert.ok(Math.abs(a[c] - b[c]) < 0.03, `stride drift ${a[c]} vs ${b[c]}`)
    }
  }
})
