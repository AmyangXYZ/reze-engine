#!/usr/bin/env node
// Does a rasterised lyric line still have antialiased edges once it is inside
// the atlas? Draw text on a 2D canvas, copy it into r8unorm both ways the spec
// allows, read the texels back, and count how many are PARTIAL.
//
//   node --import ./tests/register.mjs tools/probe-lyric-atlas.mjs
//
// A coverage texture whose texels are only ever 0 or 255 has lost its
// antialiasing, and no amount of filtering downstream can put it back — the
// glyph edges will stair-step however the effect samples them.

import { writeFileSync, mkdtempSync } from "node:fs"
import { execFileSync } from "node:child_process"
import { tmpdir } from "node:os"
import { join } from "node:path"

const W = 256
const H = 64

const work = mkdtempSync(join(tmpdir(), "lyric-probe-"))
writeFileSync(
  join(work, "probe.html"),
  `<!doctype html><script type="module">
const W = ${W}, H = ${H}
const canvas = document.createElement("canvas")
canvas.width = W; canvas.height = H
const ctx = canvas.getContext("2d")
ctx.clearRect(0, 0, W, H)
ctx.fillStyle = "#ffffff"
ctx.textBaseline = "middle"
ctx.font = "700 40px sans-serif"
ctx.fillText("(Cue!)", 8, H / 2)

const adapter = await navigator.gpu?.requestAdapter()
const device = await adapter.requestDevice()

async function measure(premultiplied, format) {
  device.pushErrorScope("validation")
  const tex = device.createTexture({
    size: [W, H], format,
    // RENDER_ATTACHMENT is REQUIRED for copyExternalImageToTexture — the
    // implementation does the copy as a draw. Without it the call is a
    // validation error and the texture stays zeroed, which reads exactly like
    // a rasteriser that drew nothing.
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
  })
  device.queue.copyExternalImageToTexture({ source: canvas }, { texture: tex, premultipliedAlpha: premultiplied }, [W, H])
  const err = await device.popErrorScope()
  if (err) console.log("LYRIC-PROBE VALIDATION " + format + " pm=" + premultiplied + ": " + err.message.split("\\n")[0])
  const bpp = format === "r8unorm" ? 1 : 4
  const buf = device.createBuffer({ size: W * H * bpp, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder()
  enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: W * bpp }, [W, H])
  device.queue.submit([enc.finish()])
  await buf.mapAsync(GPUMapMode.READ)
  const all = new Uint8Array(buf.getMappedRange())
  let zero = 0, full = 0, partial = 0
  for (let i = 0; i < all.length; i += bpp) {
    const v = all[i]
    if (v === 0) zero++; else if (v === 255) full++; else partial++
  }
  return { zero, full, partial }
}

// What the canvas itself holds, for reference: the ceiling any copy could reach.
const src = ctx.getImageData(0, 0, W, H).data
let sZero = 0, sFull = 0, sPartial = 0
for (let i = 0; i < src.length; i += 4) {
  const v = src[i + 3]
  if (v === 0) sZero++; else if (v === 255) sFull++; else sPartial++
}
console.log("LYRIC-PROBE canvas alpha: partial=" + sPartial + " full=" + sFull + " zero=" + sZero)

for (const format of ["r8unorm", "rgba8unorm"]) {
  for (const pm of [true, false]) {
    const r = await measure(pm, format)
    console.log("LYRIC-PROBE " + format + " premultipliedAlpha=" + pm + ": partial=" + r.partial + " full=" + r.full + " zero=" + r.zero)
  }
}
console.log("LYRIC-PROBE DONE")
window.close()
</script></html>`,
)

const chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
let out = ""
try {
  out = execFileSync(
    "bash",
    [
      "-c",
      `"${chrome}" --headless=new --enable-unsafe-webgpu --enable-logging=stderr --v=0 --no-sandbox "file://${join(work, "probe.html")}" 2>&1`,
    ],
    { encoding: "utf8", timeout: 60_000 },
  )
} catch (e) {
  out = (e.stdout ?? "") + (e.stderr ?? "")
}

const lines = out
  .split("\n")
  .filter((l) => l.includes("LYRIC-PROBE"))
  // Chrome appends ', source: file://…' to every console line.
  .map((l) => l.slice(l.indexOf("LYRIC-PROBE")).replace(/,\s*source:.*$/, ""))
if (lines.length === 0) {
  console.error("no probe output — is Chrome present with WebGPU?")
  process.exit(1)
}
for (const l of lines) console.log(l)
