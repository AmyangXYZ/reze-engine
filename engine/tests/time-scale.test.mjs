// Global playback speed (Engine.setTimeScale). Run: npm test.
//
// One scale on the frame's delta, so every clock advances together: a clock
// left on the wall delta is a part of the scene that keeps full speed while
// the rest goes slow — physics at full pace under a half-speed dance, or a
// sparkle racing past a slow-motion turn. Source patterns, like the other
// engine contracts: the frame needs a device.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"

const engine = readFileSync(new URL("../src/engine.ts", import.meta.url), "utf8").replace(/\r\n/g, "\n")

const frameBody = () => {
  const start = engine.indexOf("  private renderWithDelta(wallDelta: number) {")
  assert.ok(start >= 0, "renderWithDelta takes the wall delta")
  const end = engine.indexOf("\n  private ", start + 10)
  return engine.slice(start, end)
}

test("the frame scales its delta once, at the top", () => {
  const body = frameBody()
  assert.match(body, /const deltaTime = wallDelta \* this\.timeScale/)
})

test("only the frame stats read the wall delta", () => {
  const body = frameBody()
  const uses = body.match(/\bwallDelta\b/g) ?? []
  // The parameter, the scaled delta, and the stats.
  assert.equal(uses.length, 3, `wallDelta read ${uses.length} times`)
  assert.match(body, /this\.updateStats\(wallDelta \* 1000\)/)
})

test("both entry points go through the scaled frame", () => {
  const render = engine.slice(engine.indexOf("  render() {"), engine.indexOf("  render() {") + 400)
  assert.match(render, /this\.renderWithDelta\(deltaTime\)/)
  const frame = engine.slice(engine.indexOf("  renderFrame(deltaSeconds: number) {"), engine.indexOf("  renderFrame(deltaSeconds: number) {") + 300)
  assert.match(frame, /this\.renderWithDelta\(deltaSeconds\)/)
})

test("the scale is clamped and a bad value means 1", () => {
  const body = engine.slice(engine.indexOf("  setTimeScale(scale: number): void {"), engine.indexOf("  setTimeScale(scale: number): void {") + 200)
  assert.match(body, /Number\.isFinite\(scale\) \? Math\.min\(4, Math\.max\(0\.1, scale\)\) : 1/)
})
