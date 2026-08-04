// Main-thread facade over a physics worker, presenting RezePhysics' synchronous
// step()/reset() surface so the engine's frame loop doesn't change shape.
//
// PIPELINED ONE FRAME DEEP: step(N) applies the worker's result for frame N−1
// onto the current pose (dynamic bones only — cloth runs one frame behind its
// anchors, invisible at 60Hz) and posts frame N without waiting. Main-thread
// cost collapses to two ~20KB copies; the simulation itself runs on another
// core, and with one worker per model the wall cost of a multi-model scene is
// the SLOWEST model instead of the sum. If a frame arrives while the worker is
// still busy, its dt accumulates and the next post carries it — the worker's
// own fixed-step accumulator and load-shedding handle catch-up exactly as the
// main-thread path would.
import type { Rigidbody, Joint } from "./types"
import type { Mat4 } from "../math"

interface ReadyMsg {
  cmd: "ready"
  dynamicBones: number[]
}
interface SteppedMsg {
  cmd: "stepped"
  bones: ArrayBuffer
  stepMs: number
}

export class WorkerPhysics {
  private readonly worker: Worker
  private dynamicBones: number[] = []
  private readonly boneCount: number
  /** Transfer buffer when idle; null while a step is in flight. */
  private buf: ArrayBuffer | null
  /** Latest completed pose from the worker (copied out of the transfer buffer). */
  private readonly result: Float32Array
  private hasResult = false
  private pendingDt = 0
  private queuedReset = false
  /** Worker-side cost of the last completed step — for engine stats. */
  stepMs = 0

  private constructor(worker: Worker, boneCount: number) {
    this.worker = worker
    this.boneCount = boneCount
    this.buf = new ArrayBuffer(boneCount * 64)
    this.result = new Float32Array(boneCount * 16)
  }

  static supported(): boolean {
    return typeof Worker !== "undefined"
  }

  static create(rigidbodies: Rigidbody[], joints: Joint[], inverseBind: Float32Array): Promise<WorkerPhysics> {
    return new Promise((resolve, reject) => {
      let worker: Worker
      try {
        worker = new Worker(new URL("./physics.worker.js", import.meta.url), { type: "module" })
      } catch (e) {
        reject(e instanceof Error ? e : new Error(String(e)))
        return
      }
      const wp = new WorkerPhysics(worker, inverseBind.length / 16)
      const fail = (message: string) => {
        worker.terminate()
        reject(new Error(message))
      }
      worker.onerror = (e) => fail(`physics worker failed to boot: ${e.message || "worker error"}`)
      worker.onmessage = (ev: MessageEvent<ReadyMsg>) => {
        if (ev.data?.cmd !== "ready") return
        wp.dynamicBones = ev.data.dynamicBones
        worker.onmessage = (m: MessageEvent<SteppedMsg>) => wp.onStepped(m)
        worker.onerror = null
        resolve(wp)
      }
      // Rigidbody/Joint carry only data (Vec3/Mat4 fields clone as plain
      // objects with the same fields — the physics constructor reads fields,
      // never methods), so structuredClone is a faithful serializer.
      worker.postMessage({ cmd: "init", rigidbodies, joints, inverseBind: inverseBind.slice() })
    })
  }

  /** Same signature as RezePhysics.step — the engine cannot tell them apart.
   *  (inverseBind was shipped to the worker at init; the param is unused.) */
  step(dt: number, boneWorldMatrices: Mat4[], _inverseBind: Float32Array): void {
    this.pendingDt += dt
    // Apply the newest completed simulation onto this frame's pose. Dynamic
    // bones only: kinematic bones must keep the LIVE animation pose.
    if (this.hasResult) {
      const r = this.result
      for (const bi of this.dynamicBones) {
        boneWorldMatrices[bi].values.set(r.subarray(bi * 16, bi * 16 + 16))
      }
    }
    if (this.buf === null) return // worker mid-step: dt accumulated for the next post
    const flat = new Float32Array(this.buf)
    const n = Math.min(this.boneCount, boneWorldMatrices.length)
    for (let i = 0; i < n; i++) flat.set(boneWorldMatrices[i].values, i * 16)
    this.worker.postMessage(
      { cmd: this.queuedReset ? "reset" : "step", dt: this.pendingDt, bones: this.buf },
      [this.buf],
    )
    this.buf = null
    this.pendingDt = 0
    this.queuedReset = false
  }

  /** Reset rides the same pipeline: the next posted frame carries a reset
   *  command instead of a step, and stale results stop applying immediately. */
  reset(boneWorldMatrices: Mat4[]): void {
    this.queuedReset = true
    this.hasResult = false
    // Post right away if idle — reuse step's snapshot/post path with dt 0.
    if (this.buf !== null) this.step(0, boneWorldMatrices, undefined as unknown as Float32Array)
  }

  dispose(): void {
    this.worker.terminate()
  }

  private onStepped(ev: MessageEvent<SteppedMsg>): void {
    const d = ev.data
    if (d?.cmd !== "stepped") return
    this.buf = d.bones
    this.stepMs = d.stepMs
    this.result.set(new Float32Array(this.buf))
    this.hasResult = true
  }
}
