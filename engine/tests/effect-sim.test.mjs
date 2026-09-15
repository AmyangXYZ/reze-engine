// A scheduled effect's simulation: when its particles and grid restart, and how
// far they step. Run: npm test.
//
// A window's clock is enough to place a field effect. A particle pool carries
// state from frame to frame, so a burst scheduled at bar 33 has to START there
// from an empty pool and step with the transport, or it arrives mid-flight and
// an export never matches the preview. advanceSim is that decision, pure; the
// source checks at the end pin the engine to it, since there is no device here
// to render a frame with.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import { activeWindow, advanceSim, SIM_MAX_STEP } from "../dist/effect-schedule.js"

const OUT = { start: null, time: null }
const near = (got, want, what) => assert.ok(Math.abs(got - want) < 1e-9, `${what}: ${got} != ${want}`)

test("activeWindow is the latest start that holds the time, edges inclusive", () => {
  const a = { start: 0, end: 10 }
  const b = { start: 10, end: 20 }
  assert.equal(activeWindow([a, b], 5), a)
  assert.equal(activeWindow([a, b], 10), b, "back to back, the later start wins the shared edge")
  assert.equal(activeWindow([a, b], 20), b, "the end is inclusive")
  assert.equal(activeWindow([a, b], 20.01), null)
  const open = { start: 5 }
  assert.equal(activeWindow([open], 1e6), open, "an open end runs on")
  assert.equal(activeWindow([{ start: 5, end: 5 }], 5), null, "an empty window holds nothing")
  assert.equal(activeWindow(null, 3), null)
})

test("before a window nothing steps", () => {
  assert.deepEqual(advanceSim(OUT, [{ start: 10, end: 20 }], 5), { reset: false, step: 0, clock: OUT })
})

test("entering a window restarts the simulation at its own zero", () => {
  assert.deepEqual(advanceSim(OUT, [{ start: 10, end: 20 }], 10), {
    reset: true,
    step: 0,
    clock: { start: 10, time: 0 },
  })
})

test("inside, it steps by exactly how far the transport moved", () => {
  const r = advanceSim({ start: 10, time: 1 }, [{ start: 10, end: 20 }], 11 + 1 / 30)
  assert.equal(r.reset, false)
  near(r.step, 1 / 30, "one export frame")
  near(r.clock.time, 1 + 1 / 30, "the clock follows")
})

test("a paused transport holds it still", () => {
  const r = advanceSim({ start: 10, time: 3 }, [{ start: 10, end: 20 }], 13)
  assert.equal(r.reset, false)
  assert.equal(r.step, 0)
})

test("a hitch or a scrub forward steps at most SIM_MAX_STEP and restarts nothing", () => {
  const r = advanceSim({ start: 10, time: 1 }, [{ start: 10, end: 20 }], 14)
  assert.equal(r.reset, false)
  assert.equal(r.step, SIM_MAX_STEP)
})

test("going back inside a window restarts it", () => {
  assert.equal(advanceSim({ start: 10, time: 5 }, [{ start: 10, end: 20 }], 12).reset, true)
})

test("a loop wrapping back into the same window restarts it", () => {
  const r = advanceSim({ start: 0, time: 9.9 }, [{ start: 0, end: 10 }], 0)
  assert.equal(r.reset, true)
  assert.deepEqual(r.clock, { start: 0, time: 0 })
})

test("the next window, back to back, is a new firing", () => {
  const r = advanceSim({ start: 0, time: 9.98 }, [{ start: 0, end: 10 }, { start: 10, end: 20 }], 10)
  assert.equal(r.reset, true)
  assert.equal(r.clock.start, 10)
})

test("leaving a window and coming back restarts it", () => {
  const lane = [{ start: 10, end: 20 }]
  const left = advanceSim({ start: 10, time: 9 }, lane, 25)
  assert.deepEqual(left, { reset: false, step: 0, clock: OUT })
  assert.equal(advanceSim(left.clock, lane, 15).reset, true)
})

test("a window that moved under the transport restarts; one trimmed around it does not", () => {
  assert.equal(advanceSim({ start: 10, time: 2 }, [{ start: 11, end: 20 }], 12.05).reset, true, "moved")
  const trimmed = advanceSim({ start: 10, time: 2 }, [{ start: 10, end: 30 }], 12 + 1 / 60)
  assert.equal(trimmed.reset, false, "trimmed at the end")
  near(trimmed.step, 1 / 60, "and it keeps stepping")
})

test("the engine restarts and steps scheduled simulations from advanceSim", () => {
  const src = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8")
  const slice = (from, to) => {
    const a = src.indexOf(from)
    assert.ok(a > 0, `${from} exists`)
    return src.slice(a, src.indexOf(to, a))
  }
  const evaluate = slice("private evaluateEffectSchedules", "\n  }")
  assert.match(evaluate, /const sim = advanceSim\(fx\.sim, fx\.window, transport\)/)
  assert.match(evaluate, /fx\.simReset = sim\.reset/)
  assert.match(evaluate, /fx\.simStep = sim\.step/)

  const particles = slice("private stepParticles", "private renderParticles")
  assert.match(particles, /if \(scheduled && !e\.simReset && e\.simStep === 0\) continue/)
  assert.match(particles, /if \(e\.simReset\) encoder\.clearBuffer\(p\.buffer\)/)
  assert.match(particles, /p\.data\[1\] = scheduled \? e\.simStep : Math\.min\(0\.1, Math\.max\(0, deltaTime\)\)/)

  const grid = slice("private stepSim(", "getEffectMounts()")
  assert.match(grid, /if \(scheduled && !e\.simReset && e\.simStep === 0\) continue/)
  assert.match(grid, /loadOp: "clear"/)
  assert.match(grid, /grid\.frame = 0/)
  assert.match(grid, /grid\.data\[1\] = scheduled \? e\.simStep : Math\.min\(0\.1, Math\.max\(0, deltaTime\)\)/)

  // Clearing needs the usages: a pool that cannot be written and a grid that
  // cannot be a render target would reject the whole frame's command buffer.
  assert.match(src, /label: "particle pool",\s*size: count \* PARTICLE_STRIDE,\s*usage: GPUBufferUsage\.STORAGE \| GPUBufferUsage\.COPY_DST/)
  assert.match(src, /usage: GPUTextureUsage\.TEXTURE_BINDING \| GPUTextureUsage\.STORAGE_BINDING \| GPUTextureUsage\.RENDER_ATTACHMENT/)
})
