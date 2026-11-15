// The camera on a bone: POV through her eyes, selfie held in her hand.
//
// The math only — the engine places camera.mountPose from this every frame.

import { test } from "node:test"
import assert from "node:assert/strict"
import { readFileSync, existsSync, readdirSync, statSync } from "node:fs"
import { fileURLToPath } from "node:url"
import { dirname, join } from "node:path"

const here = dirname(fileURLToPath(import.meta.url))
const { PmxLoader } = await import("../dist/pmx-loader.js")
const { CameraMount, facePoint, headMaterials } = await import("../dist/camera-mount.js")
const { Quat, Vec3 } = await import("../dist/math.js")

const findModels = () => {
  const roots = [join(here, "../../web/public/models"), join(here, "../../../reze-design/public/models")].filter(existsSync)
  const out = []
  const walk = (dir) => {
    for (const entry of readdirSync(dir)) {
      const p = join(dir, entry)
      if (statSync(p).isDirectory()) walk(p)
      else if (p.toLowerCase().endsWith(".pmx")) out.push(p)
    }
  }
  for (const r of roots) walk(r)
  return out
}

const toAB = (b) => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)

const RIG = findModels().find((path) => {
  try {
    const names = PmxLoader.loadFromBuffer(toAB(readFileSync(path))).getSkeleton().bones.map((b) => b.name)
    return ["頭", "左目", "右目", "右手首"].every((n) => names.includes(n))
  } catch {
    return false
  }
})
const SKIP = { skip: RIG ? false : "no rig with a head, eyes and a wrist on this machine" }

const load = () => {
  const model = PmxLoader.loadFromBuffer(toAB(readFileSync(RIG)))
  model.update(0, false)
  return model
}
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z)

test("POV sits just in front of her eyes, looking where her face does, level", SKIP, () => {
  const model = load()
  const pose = new CameraMount(model, { mode: "pov" }).step(1 / 60)
  const face = facePoint(model)
  assert.ok(pose)
  assert.ok(dist(pose.eye, face) < 0.5, `eye ${dist(pose.eye, face)} from her face`)
  // At rest an MMD model faces −Z.
  assert.ok(pose.forward.z < -0.9, `forward ${JSON.stringify(pose.forward)}`)
  assert.deepEqual([pose.up.x, pose.up.y, pose.up.z], [0, 1, 0])
})

test("the mount rides the model's placement", SKIP, () => {
  const model = load()
  const at = new CameraMount(model, { mode: "pov" }).step(1 / 60).eye
  model.position.x += 10
  const moved = new CameraMount(model, { mode: "pov" }).step(1 / 60).eye
  assert.ok(Math.abs(moved.x - at.x - 10) < 1e-3, `${moved.x - at.x}`)
})

test("selfie holds the phone at arm's length toward her hand, aimed at her face", SKIP, () => {
  const model = load()
  const pose = new CameraMount(model, { mode: "selfie", distance: 7 }).step(1 / 60)
  const face = facePoint(model)
  const d = dist(pose.eye, face)
  assert.ok(d >= 7 * 0.55 - 1e-3 && d <= 7 + 1e-3, `held ${d} from her face`)
  // Forward points from the lens back to her face.
  const to = { x: (face.x - pose.eye.x) / d, y: (face.y - pose.eye.y) / d, z: (face.z - pose.eye.z) / d }
  const dot = to.x * pose.forward.x + to.y * pose.forward.y + to.z * pose.forward.z
  assert.ok(dot > 0.999, `aim ${dot}`)
})

test("a turn is eased, not snapped, once seeded", SKIP, () => {
  const model = load()
  const mount = new CameraMount(model, { mode: "pov", smoothing: 0.2 })
  const before = mount.step(1 / 60).forward
  // Turn the whole model half way round: the rig's head faces +Z now.
  model.rotation.setXYZW(0, 1, 0, 0)
  const after = mount.step(1 / 60).forward
  assert.ok(after.z < 0, "one frame later it has not swung all the way")
  assert.ok(after.z > before.z, "but it has started to")
})

test("her head's materials are picked for the POV camera to leave out", SKIP, () => {
  const model = load()
  const head = headMaterials(model)
  const all = model.getMaterials().map((m) => m.name)
  assert.ok(head.length > 0, "some materials are her head")
  assert.ok(head.length < all.length / 2, `${head.length} of ${all.length} is too many`)
})

test("POV steadies a nodding head: little of a beat's shake reaches the view", SKIP, () => {
  const model = load()
  const mount = new CameraMount(model, { mode: "pov" })
  const nod = (deg) => Quat.fromAxisAngle(new Vec3(1, 0, 0), (deg * Math.PI) / 180)
  mount.step(1 / 60)
  // A nod ±10° every other frame: the worst a dance's beat does to a head.
  let view = 0
  let head = 0
  let prev = null
  let prevHead = null
  for (let i = 0; i < 120; i++) {
    model.setBoneRotationOffset("頭", nod(i % 2 ? 10 : -10))
    model.update(1 / 60, false)
    const f = mount.step(1 / 60).forward
    const h = model.getBoneWorldMatrix("頭")
    const hf = { x: -h[8], y: -h[9], z: -h[10] }
    if (prev && i > 60) {
      view += Math.acos(Math.min(1, prev.x * f.x + prev.y * f.y + prev.z * f.z))
      const l = Math.hypot(hf.x, hf.y, hf.z) * Math.hypot(prevHead.x, prevHead.y, prevHead.z)
      head += Math.acos(Math.min(1, (prevHead.x * hf.x + prevHead.y * hf.y + prevHead.z * hf.z) / l))
    }
    prev = f
    prevHead = hf
  }
  assert.ok(head > 0.1, `the head shook (${head})`)
  assert.ok(view < head * 0.05, `the view moved ${view} rad against the head's ${head}`)
})
