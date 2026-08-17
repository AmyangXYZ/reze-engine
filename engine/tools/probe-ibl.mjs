#!/usr/bin/env node
// Prove the IBL path on a real device: fill a light-uniform buffer laid out
// EXACTLY as engine.writeWorld does, run the real WORLD_AMBIENT_WGSL over a
// set of normals in a compute shader, and compare against the CPU evaluator.
//
//   node --import ./tests/register.mjs tools/probe-ibl.mjs
//
// This closes the gap the hermetic tests cannot: WGSL struct offsets vs the
// CPU float indices, and the polynomial as the GPU actually executes it.

import { writeFileSync, mkdtempSync } from "node:fs"
import { execFileSync } from "node:child_process"
import { tmpdir } from "node:os"
import { join, dirname } from "node:path"
import { fileURLToPath } from "node:url"

const here = dirname(fileURLToPath(import.meta.url))
const dist = join(here, "..", "dist")
const { WORLD_AMBIENT_WGSL } = await import(`${dist}/shaders/lights.js`)
const { projectIrradianceSH, evalIrradianceSH } = await import(`${dist}/ibl.js`)

// A lopsided analytic sky, so every SH band is exercised.
const w = 64
const h = 32
const data = new Float32Array(w * h * 4)
for (let y = 0; y < h; y++) {
  for (let x = 0; x < w; x++) {
    const ny = Math.cos(Math.PI * ((y + 0.5) / h))
    const phi = ((x + 0.5) / w - 0.5) * 2 * Math.PI
    const i = (y * w + x) * 4
    data[i] = 0.6 + 0.5 * ny
    data[i + 1] = 0.5 + 0.3 * Math.sin(phi) * Math.sqrt(Math.max(0, 1 - ny * ny))
    data[i + 2] = 0.4 + 0.4 * Math.cos(phi) * Math.sqrt(Math.max(0, 1 - ny * ny))
    data[i + 3] = 1
  }
}
const sh = projectIrradianceSH({ width: w, height: h, data }, 1)

// The buffer, filled EXACTLY as writeWorld fills lightData (80 floats).
const strength = 0.7
const light = new Float32Array(80)
light[0] = 9.9 // flat ambient r — a poison value: if the flag path is broken, this leaks through
light[1] = 9.9
light[2] = 9.9
for (let i = 0; i < 9; i++) {
  const b = 36 + i * 4
  light[b] = sh[i * 3] * strength
  light[b + 1] = sh[i * 3 + 1] * strength
  light[b + 2] = sh[i * 3 + 2] * strength
  light[b + 3] = 0
}
light[39] = 1

const NORMALS = [
  [0, 1, 0],
  [0, -1, 0],
  [1, 0, 0],
  [0, 0, 1],
  [0.577, 0.577, 0.577],
  [-0.577, 0.577, -0.577],
]

const shader = `
struct Light { direction: vec4f, color: vec4f, };
struct LightUniforms { ambientColor: vec4f, lights: array<Light, 4>, sh: array<vec4f, 9>, };
@group(0) @binding(0) var<uniform> light: LightUniforms;
@group(0) @binding(1) var<storage, read> normals: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> results: array<vec4f>;
${WORLD_AMBIENT_WGSL}
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  results[gid.x] = vec4f(rzWorldAmbient(normals[gid.x].xyz), 1.0);
}
`

const payload = Buffer.from(
  JSON.stringify({ shader, light: [...light], normals: NORMALS }),
  "utf8",
).toString("base64")

const work = mkdtempSync(join(tmpdir(), "ibl-probe-"))
writeFileSync(
  join(work, "probe.html"),
  `<!doctype html><script type="module">
const p = JSON.parse(new TextDecoder().decode(Uint8Array.from(atob("${payload}"), (c) => c.charCodeAt(0))))
const adapter = await navigator.gpu?.requestAdapter()
const device = await adapter.requestDevice()
const mod = device.createShaderModule({ code: p.shader })
const info = await mod.getCompilationInfo()
for (const m of info.messages) if (m.type === "error") console.log("IBL-PROBE ERROR: " + m.lineNum + " " + m.message)
const lightBuf = device.createBuffer({ size: 320, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
device.queue.writeBuffer(lightBuf, 0, new Float32Array(p.light))
const nBuf = device.createBuffer({ size: p.normals.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
device.queue.writeBuffer(nBuf, 0, new Float32Array(p.normals.flatMap((n) => [...n, 0])))
const rBuf = device.createBuffer({ size: p.normals.length * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC })
const read = device.createBuffer({ size: p.normals.length * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
const pipe = device.createComputePipeline({ layout: "auto", compute: { module: mod, entryPoint: "main" } })
const bind = device.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: [
  { binding: 0, resource: { buffer: lightBuf } },
  { binding: 1, resource: { buffer: nBuf } },
  { binding: 2, resource: { buffer: rBuf } },
]})
const enc = device.createCommandEncoder()
const pass = enc.beginComputePass()
pass.setPipeline(pipe); pass.setBindGroup(0, bind); pass.dispatchWorkgroups(p.normals.length); pass.end()
enc.copyBufferToBuffer(rBuf, 0, read, 0, p.normals.length * 16)
device.queue.submit([enc.finish()])
await read.mapAsync(GPUMapMode.READ)
const out = new Float32Array(read.getMappedRange())
for (let i = 0; i < p.normals.length; i++) {
  console.log("IBL-PROBE RESULT " + i + ": " + out[i * 4] + " " + out[i * 4 + 1] + " " + out[i * 4 + 2])
}
console.log("IBL-PROBE DONE")
window.close()
</script></html>`,
)

const chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
let out = ""
try {
  out = execFileSync(
    "bash",
    ["-c", `"${chrome}" --headless=new --enable-unsafe-webgpu --enable-logging=stderr --v=0 --no-sandbox "file://${join(work, "probe.html")}" 2>&1`],
    { encoding: "utf8", timeout: 60_000 },
  )
} catch (e) {
  out = (e.stdout ?? "") + (e.stderr ?? "")
}

const lines = out.split("\n").filter((l) => l.includes("IBL-PROBE"))
let bad = 0
for (const l of lines.filter((x) => x.includes("ERROR"))) {
  console.error(l.slice(l.indexOf("IBL-PROBE")))
  bad++
}
for (let i = 0; i < NORMALS.length; i++) {
  const m = lines.find((l) => l.includes(`IBL-PROBE RESULT ${i}:`))
  if (!m) {
    console.error(`normal ${i}: no GPU result`)
    bad++
    continue
  }
  // Chrome suffixes console lines with `", source: file://..."` — cut at the
  // closing quote before splitting, or the last number swallows it as NaN.
  const gpu = m
    .slice(m.indexOf(`RESULT ${i}:`) + `RESULT ${i}:`.length)
    .replace(/".*$/, "")
    .trim()
    .split(/\s+/)
    .map(Number)
  const [nx, ny, nz] = NORMALS[i]
  const cpu = evalIrradianceSH(sh, { x: nx, y: ny, z: nz }).map((v) => Math.max(v * strength, 0))
  const ok = cpu.every((v, c) => Math.abs(v - gpu[c]) < 1e-3)
  if (!ok) bad++
  console.log(
    `n=(${NORMALS[i].join(",")})  gpu=[${gpu.map((v) => v.toFixed(4)).join(", ")}]  cpu=[${cpu.map((v) => v.toFixed(4)).join(", ")}]  ${ok ? "MATCH" : "MISMATCH"}`,
  )
  if (gpu[0] > 9) {
    console.error("  ^ the poison flat-ambient leaked through — the flag path is broken")
  }
}
console.log(bad === 0 && lines.some((l) => l.includes("DONE")) ? "IBL layout + polynomial PROVEN on device" : "PROBE FAILED")
process.exit(bad ? 1 : 0)
