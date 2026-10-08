// The frame read back as numbers (Engine.readAovs, Engine.probeLight). Run: npm test.
//
// No GPU here, so what is checked is the contract the host decodes against:
// the field list and the stride agree, every binding the engine fills is
// declared exactly once, and the probe's per-lamp block covers every slot.

import { test } from "node:test"
import assert from "node:assert/strict"
import {
  AOV_PIXEL_FIELDS,
  AOV_PIXEL_STRIDE,
  AOV_PIXELS_WGSL,
  AOV_PROBE_HEAD,
  AOV_PROBE_STRIDE,
  AOV_PROBE_WGSL,
} from "../dist/shaders/passes/aov.js"
import { MAX_LIGHTS } from "../dist/shaders/lights.js"

const bindings = (wgsl) => [...wgsl.matchAll(/@group\(0\) @binding\((\d+)\)/g)].map((m) => Number(m[1]))

test("a texel's floats are exactly the named fields", () => {
  assert.equal(AOV_PIXEL_FIELDS.length, AOV_PIXEL_STRIDE)
  assert.equal(new Set(AOV_PIXEL_FIELDS).size, AOV_PIXEL_FIELDS.length)
})

test("the pixel pass declares bindings 0..12 once each", () => {
  const b = bindings(AOV_PIXELS_WGSL).sort((x, y) => x - y)
  assert.deepEqual(b, [...Array(13).keys()])
})

test("the probe pass declares bindings 0..8 once each", () => {
  const b = bindings(AOV_PROBE_WGSL).sort((x, y) => x - y)
  assert.deepEqual(b, [...Array(9).keys()])
})

test("the probe keeps a block for every lamp slot", () => {
  assert.equal(AOV_PROBE_STRIDE, AOV_PROBE_HEAD + MAX_LIGHTS * 4)
  assert.match(AOV_PROBE_WGSL, new RegExp(`l < ${MAX_LIGHTS}u`))
})

test("ids are packed material-low, object-high", () => {
  assert.match(AOV_PIXELS_WGSL, /outId\[i\] = \(id\.y << 16u\) \| \(id\.x & 0xffffu\)/)
})
