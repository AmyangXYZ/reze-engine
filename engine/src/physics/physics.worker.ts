// Physics worker: owns ONE model's RezePhysics world. The host posts the
// frame's bone world matrices as a transferable buffer; the worker steps the
// simulation (identical math to the main-thread path — same class, same code)
// and returns the buffer with the dynamic bones' matrices written back, plus
// its own step cost for the host's stats.
//
// The incoming buffer is COPIED into a persistent local pose (and back out)
// rather than wrapped: transferred ArrayBuffers get a fresh identity on every
// hop, so wrappers could never be cached against them — and two ~20KB copies
// per frame are microseconds.
import { RezePhysics } from "./physics"
import { Mat4 } from "../math"
import type { Rigidbody, Joint } from "./types"
import { RigidbodyType } from "./types"

interface InitMsg {
  cmd: "init"
  rigidbodies: Rigidbody[]
  joints: Joint[]
  inverseBind: Float32Array
}
interface PoseMsg {
  cmd: "step" | "reset"
  dt: number
  bones: ArrayBuffer
}

const ctx = self as unknown as {
  onmessage: ((e: MessageEvent<InitMsg | PoseMsg>) => void) | null
  postMessage(msg: unknown, transfer?: Transferable[]): void
}

let physics: RezePhysics | null = null
let inverseBind: Float32Array<ArrayBufferLike> = new Float32Array(0)
let pose = new Float32Array(0)
let mats: Mat4[] = []

ctx.onmessage = (e) => {
  const msg = e.data
  if (msg.cmd === "init") {
    physics = new RezePhysics(msg.rigidbodies, msg.joints)
    inverseBind = msg.inverseBind
    const boneCount = inverseBind.length / 16
    pose = new Float32Array(boneCount * 16)
    mats = new Array(boneCount)
    for (let i = 0; i < boneCount; i++) mats[i] = new Mat4(pose.subarray(i * 16, i * 16 + 16) as Float32Array)
    // The host copies back only the bones physics can write — the bones of
    // dynamic bodies. Kinematic bones must NOT round-trip: in the pipelined
    // protocol they would drag a stale (frame-old) pose over the live one.
    const dynamicBones: number[] = []
    for (const rb of msg.rigidbodies) {
      if (rb.type === RigidbodyType.Dynamic && rb.mass > 0 && rb.boneIndex >= 0) dynamicBones.push(rb.boneIndex)
    }
    ctx.postMessage({ cmd: "ready", dynamicBones })
    return
  }
  if (!physics) return
  const incoming = new Float32Array(msg.bones)
  pose.set(incoming)
  const t0 = performance.now()
  if (msg.cmd === "step") physics.step(msg.dt, mats, inverseBind)
  else physics.reset(mats)
  const stepMs = performance.now() - t0
  incoming.set(pose)
  ctx.postMessage({ cmd: "stepped", bones: msg.bones, stepMs }, [msg.bones])
}
