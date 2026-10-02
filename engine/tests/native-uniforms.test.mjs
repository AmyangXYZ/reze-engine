// A native look's material values, set at runtime. Run: npm test.
//
// A game animates some of a material's properties over a take (Aether Gazer's
// timeline material curves: a pour stream's colour fading, its dissolve
// rising). The host samples those keys on its own clock and hands the engine
// the values of the frame; the engine owns no track and no clock. What this
// pins: a set value reaches that material's draws over the look's own, only
// that material's, only names its look carries, and a value that did not move
// costs no upload — the host refills a uniform member only when its value
// object changes, and only the bytes of the members that did.

import { test } from "node:test"
import assert from "node:assert/strict"
import { NativeLooks } from "../dist/unity/looks.js"
import { NativeHost, sameValue } from "../dist/unity/host.js"

globalThis.GPUBufferUsage ??= { MAP_READ: 1, MAP_WRITE: 2, COPY_SRC: 4, COPY_DST: 8, INDEX: 16, VERTEX: 32, UNIFORM: 64, STORAGE: 128 }
globalThis.GPUTextureUsage ??= { COPY_SRC: 1, COPY_DST: 2, TEXTURE_BINDING: 4, STORAGE_BINDING: 8, RENDER_ATTACHMENT: 16 }

/** A device that records the buffer uploads and builds nothing real. */
function fakeDevice() {
  const writes = []
  const buffer = (d) => ({ label: d.label, size: d.size, destroy() {}, getMappedRange: () => new ArrayBuffer(d.size), unmap() {} })
  const device = {
    writes,
    createBuffer: buffer,
    createSampler: () => ({}),
    createShaderModule: () => ({}),
    createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
    createTexture: () => ({ createView: () => ({}), destroy() {} }),
    queue: {
      writeBuffer: (buf, offset, data, dataOffset = 0, size) => writes.push({ label: buf.label, offset, size: size ?? data.byteLength - dataOffset }),
      writeTexture() {},
      onSubmittedWorkDone: () => Promise.resolve(),
    },
  }
  return device
}

/** A host that keeps every draw's value sources instead of drawing. */
function fakeHost() {
  const draws = []
  return {
    draws,
    register() {},
    warm: () => Promise.resolve(),
    inputs: () => [],
    release() {},
    draw: (_pass, d) => draws.push(d),
  }
}

function lookup(sources, name) {
  for (const s of sources) if (s[name] !== undefined && s[name] !== null) return s[name]
  return undefined
}

function dressedModel(device) {
  const looks = new NativeLooks(device, fakeHost())
  const vertices = new Float32Array(8 * 3)
  const model = {
    name: "pour",
    vertices,
    indices: new Uint32Array([0, 1, 2, 0, 1, 2]),
    vertexBuffer: {},
    jointsBuffer: {},
    weightsBuffer: {},
    skinMatrixBuffer: {},
    indexBuffer: {},
    draws: [
      { materialName: "water1", firstIndex: 0, count: 3, diffuse: {} },
      { materialName: "water2", firstIndex: 3, count: 3, diffuse: {} },
    ],
    hidden: () => false,
    headSkin: () => null,
    headRest: [0, 0, 0],
    layers: 1,
    additionalUv: () => null,
    morphs: [],
    morphWeights: () => new Float32Array(0),
  }
  const pass = { lightMode: "FORWARDBASE", shader: "fx", state: {} }
  looks.install(model, {
    shaders: [],
    textures: {},
    materials: [
      { materials: ["water1"], queue: 3000, passes: [pass], values: { _DissovleStrength: 0, _Color: [1, 1, 1, 0.045] }, textures: {} },
      { materials: ["water2"], queue: 3000, passes: [pass], values: { _DissovleStrength: 0.27, _Color: [1, 1, 1, 1] }, textures: {} },
    ],
  })
  return looks
}

function drawnValues(looks, name) {
  const host = looks.host
  host.draws.length = 0
  looks.drawTransparent({})
  return Object.fromEntries(host.draws.map((d) => [d.key.split("|")[1], lookup(d.sources, name)]))
}

test("a set value draws over the look's own, on that material alone, until set again", () => {
  const looks = dressedModel(fakeDevice())
  assert.deepEqual(drawnValues(looks, "_DissovleStrength"), { water1: 0, water2: 0.27 }, "before any set: the look's values")
  assert.equal(looks.setUniforms("pour", "water1", { _DissovleStrength: 0.48 }), true)
  assert.deepEqual(drawnValues(looks, "_DissovleStrength"), { water1: 0.48, water2: 0.27 })
  // it stands while nothing sets it, and a vector goes through as given
  looks.setUniforms("pour", "water1", { _Color: [1, 0.86, 0.69, 0.01] })
  assert.deepEqual(drawnValues(looks, "_DissovleStrength"), { water1: 0.48, water2: 0.27 })
  assert.deepEqual(drawnValues(looks, "_Color").water1, [1, 0.86, 0.69, 0.01])
})

test("only a dressed material and the names its look carries are taken", () => {
  const looks = dressedModel(fakeDevice())
  assert.equal(looks.setUniforms("pour", "cup", { _DissovleStrength: 1 }), false, "a material the look does not dress")
  assert.equal(looks.setUniforms("nobody", "water1", { _DissovleStrength: 1 }), false, "a model without a look")
  // a name the material does not carry could be a frame-wide global, whose
  // block is shared by every draw of the shader: it is not this material's to set
  looks.setUniforms("pour", "water1", { _Time: [9, 9, 9, 9] })
  assert.equal(drawnValues(looks, "_Time").water1, undefined)
})

test("a value that did not move keeps the object the host compares by", () => {
  const looks = dressedModel(fakeDevice())
  const first = [1, 0.86, 0.69, 0.04]
  looks.setUniforms("pour", "water1", { _Color: first })
  looks.setUniforms("pour", "water1", { _Color: [1, 0.86, 0.69, 0.04] })
  assert.equal(drawnValues(looks, "_Color").water1, first, "an equal value is the same object: no refill")
  looks.setUniforms("pour", "water1", { _Color: [1, 0.86, 0.69, 0.03] })
  assert.notEqual(drawnValues(looks, "_Color").water1, first, "a moved value is new: refilled")
  assert.ok(sameValue([1, [2, 3]], [1, [2, 3]]) && !sameValue(new Uint32Array([1]), [1]))
})

test("the host uploads only the members whose value object changed", () => {
  const device = fakeDevice()
  const host = new NativeHost(device, { colorFormats: [], depthFormat: "depth32float", sampleCount: 1, reversedZ: false }, {})
  const struct = {
    size: 48,
    members: [
      { name: "_Color", offset: 0, size: 16, type: "vec4<f32>" },
      { name: "_DissovleStrength", offset: 16, size: 4, type: "f32" },
      { name: "_MainTex_ST", offset: 32, size: 16, type: "vec4<f32>" },
    ],
  }
  const spec = { _Color: [1, 1, 1, 1], _DissovleStrength: 0, _MainTex_ST: [1, 1, 0, 0] }
  const set = {}
  const blk = host.newBlock(struct, "block")
  host.fill(struct, [set, spec], blk, true)
  device.writes.length = 0
  host.fill(struct, [set, spec], blk, false)
  assert.equal(device.writes.length, 0, "nothing moved: no upload")
  set._DissovleStrength = 0.5
  host.fill(struct, [set, spec], blk, false)
  assert.deepEqual(device.writes, [{ label: "block", offset: 16, size: 4 }], "one float moved: its four bytes")
  assert.equal(blk.f32[4], 0.5)
  device.writes.length = 0
  host.fill(struct, [set, spec], blk, false)
  assert.equal(device.writes.length, 0, "held: no upload")
})
