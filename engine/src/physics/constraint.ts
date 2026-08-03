import { Mat4 } from "../math"
import type { Joint, Rigidbody } from "./types"
import { RigidbodyType } from "./types"

// 6DOF spring constraint trimmed to what MMD uses. Connects bodyA and bodyB
// via local-space anchor frames; at simulate time the world frames are
// TA = worldA · frameA, TB = worldB · frameB. The 6 DOFs are the linear
// diff in TA's basis (axes 0..2) and the Euler-XYZ angular diff between
// TA's and TB's basis (axes 3..5).
//
// Springs (when enabled) drive each DOF toward equilibriumPoint[i] with
// stiffness[i]. Per-axis stop ERP is 0.475 — PMX joint limits are tuned
// against this softness.
export interface SixDofSpringConstraint {
  bodyA: number
  bodyB: number
  // Local 4x4 (column-major) anchor frames on each body.
  frameA: Float32Array
  frameB: Float32Array
  // Per-axis limits. For each i: when min[i] > max[i] the axis is free
  // (Bullet's "free" convention); when min[i] === max[i] the axis is locked.
  linearMin: Float32Array  // length 3
  linearMax: Float32Array
  angularMin: Float32Array // length 3, radians
  angularMax: Float32Array
  // Springs.
  springEnabled: Uint8Array     // length 6
  springStiffness: Float32Array // length 6 (k)
  equilibriumPoint: Float32Array// length 6, baked at setup time
  // True for joints that close a cycle in the joint graph (e.g. the
  // horizontal ring welds of cross-linked skirt lattices). Loop edges get
  // reduced limit-correction rates in the solver — a cycle over-determines
  // positions and full-rate corrections chatter (see LOOP_ERP_SCALE).
  isLoop: boolean

  // Per-substep solver cache lives in solver.ts' SolverCache (flat SoA typed
  // arrays indexed by constraint slot) — see F_STRIDE layout there.
  // read by the velocity-only iter loop. None of these depend on lv/av — only
  // on pos/ori/inertia which are constant during solve.
  // Limit rows are unilateral (Bullet-style): the per-substep accumulated
  // impulse is clamped to the corrective sign, so a limit can push a body
  // back into range but never pull it deeper / brake its natural recovery —
  // bilateral limit rows act as motors and pump energy into swinging cloth.
  // Spring rows are velocity-target drives; setup clamps k to the deadbeat
  // stability bound so they cannot pump energy. maxImp bounds the per-substep
  // accumulated impulse by the real spring force (k·|err|·dt) for rows that
  // must stay force-limited (loop-edge welds); Infinity for authored springs.
  // Per-axis angular Jacobians and tensor-multiplied axes: with full inertia
  // tensors the effective mass differs per axis, and impulse application
  // needs I⁻¹·axis per body.
  // Angular spring force clamp (|k·err|·dt) + per-substep accumulator —
  // unclamped spring velocity-drives pump energy slowly (the restoring
  // magnitude on derived euler axes is orientation-dependent by up to ~25%,
  // and a mis-scaled oscillator restoring force injects per cycle).
  // Single geodesic limit row: shortest rotation from the current relative
  // orientation to the euler-clamped target. Per-axis euler limit rows are
  // geometrically inconsistent for large violations (asin singularity) and
  // pump energy instead of converging. Unilateral like the linear limits.
  // Per-axis angular limit rows for the small-violation regime. act:
  // 1 = bilateral (locked axis), 2 = unilateral stop (ranged axis) — a
  // bilateral row on a ranged axis brakes natural recovery and pumps
  // energy into swinging cloth.
}

// Stop-limit ERP. PMX rigs are tuned against MMD's stiff limit response;
// lowering this makes cloth resting against its limits sink visibly deeper
// (equilibrium penetration scales with 1/ERP). Rest chatter at this
// stiffness was fixed at the source (spring double-drive, unilateral
// stops) — don't lower ERP to paper over jitter.
export const STOP_ERP = 0.45

// Build per-joint constraints from PMX data:
//   frameA = (bodyA_worldBind)^-1 · jointWorldBind
//   frameB = (bodyB_worldBind)^-1 · jointWorldBind
// Equilibrium is zero on every axis (both frames coincide at bind pose).
export function buildConstraints(
  rigidbodies: Rigidbody[],
  joints: Joint[],
): SixDofSpringConstraint[] {
  const out: SixDofSpringConstraint[] = []
  const jointWorld = new Float32Array(16)
  const bodyWorld = new Float32Array(16)
  const bodyInv = new Float32Array(16)

  for (let j = 0; j < joints.length; j++) {
    const joint = joints[j]
    const a = joint.rigidbodyIndexA
    const b = joint.rigidbodyIndexB
    if (a < 0 || b < 0 || a >= rigidbodies.length || b >= rigidbodies.length) continue
    if (a === b) continue
    const rbA = rigidbodies[a]
    const rbB = rigidbodies[b]

    // jointWorldBind from PMX (Euler XYZ as written by saba reference).
    const jq = eulerToQuat(joint.rotation.x, joint.rotation.y, joint.rotation.z)
    Mat4.fromPositionRotationInto(
      joint.position.x, joint.position.y, joint.position.z,
      jq.x, jq.y, jq.z, jq.w,
      jointWorld,
    )

    const frameA = new Float32Array(16)
    const frameB = new Float32Array(16)
    if (!buildLocalFrame(rbA, jointWorld, bodyWorld, bodyInv, frameA)) continue
    if (!buildLocalFrame(rbB, jointWorld, bodyWorld, bodyInv, frameB)) continue

    const linearMin = new Float32Array([joint.positionMin.x, joint.positionMin.y, joint.positionMin.z])
    const linearMax = new Float32Array([joint.positionMax.x, joint.positionMax.y, joint.positionMax.z])
    // Some PMX rigs encode "free" angular axes as ±π·N which wraps badly
    // in limit comparisons — normalize to [-π, π] up front.
    const angularMin = new Float32Array([
      normalizeAngle(joint.rotationMin.x),
      normalizeAngle(joint.rotationMin.y),
      normalizeAngle(joint.rotationMin.z),
    ])
    const angularMax = new Float32Array([
      normalizeAngle(joint.rotationMax.x),
      normalizeAngle(joint.rotationMax.y),
      normalizeAngle(joint.rotationMax.z),
    ])

    const springEnabled = new Uint8Array(6)
    const springStiffness = new Float32Array(6)
    springStiffness[0] = joint.springPosition.x
    springStiffness[1] = joint.springPosition.y
    springStiffness[2] = joint.springPosition.z
    springStiffness[3] = joint.springRotation.x
    springStiffness[4] = joint.springRotation.y
    springStiffness[5] = joint.springRotation.z
    for (let i = 0; i < 6; i++) springEnabled[i] = springStiffness[i] !== 0 ? 1 : 0

    out.push({
      bodyA: a,
      bodyB: b,
      frameA,
      frameB,
      linearMin,
      linearMax,
      angularMin,
      angularMax,
      springEnabled,
      springStiffness,
      equilibriumPoint: new Float32Array(6),
      isLoop: false,
    })
  }

  // Mark loop-closing joints via union-find over the joint graph. All
  // bone-follow bodies count as one "world" component (they're each anchored
  // to the skeleton), so a joint bridging two separately anchored subtrees
  // also closes a kinematic cycle. One edge per cycle gets softened, which
  // releases the over-determination regardless of which edge it is.
  const parent = new Int32Array(rigidbodies.length + 1)
  for (let i = 0; i < parent.length; i++) parent[i] = i
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]
      x = parent[x]
    }
    return x
  }
  const world = rigidbodies.length
  for (let i = 0; i < rigidbodies.length; i++) {
    if (rigidbodies[i].type !== RigidbodyType.Dynamic) parent[find(i)] = find(world)
  }
  for (const con of out) {
    const ra = find(con.bodyA)
    const rb = find(con.bodyB)
    if (ra === rb) con.isLoop = true
    else parent[ra] = rb
  }

  return out
}

// frame = bodyWorldBind^-1 · jointWorld. False if bodyWorldBind is singular.
function buildLocalFrame(
  rb: Rigidbody,
  jointWorld: Float32Array,
  bodyWorld: Float32Array,
  bodyInv: Float32Array,
  out: Float32Array,
): boolean {
  const q = eulerToQuat(rb.shapeRotation.x, rb.shapeRotation.y, rb.shapeRotation.z)
  Mat4.fromPositionRotationInto(
    rb.shapePosition.x, rb.shapePosition.y, rb.shapePosition.z,
    q.x, q.y, q.z, q.w,
    bodyWorld,
  )
  if (!Mat4.inverseInto(bodyWorld, bodyInv)) return false
  Mat4.multiplyArrays(bodyInv, 0, jointWorld, 0, out, 0)
  return true
}

function normalizeAngle(a: number): number {
  const twoPi = Math.PI * 2
  a = a % twoPi
  if (a < -Math.PI) a += twoPi
  else if (a > Math.PI) a -= twoPi
  return a
}

// ZXY left-handed Euler → quat (matches Quat.fromEuler), inlined to skip
// the allocation per joint at build time.
function eulerToQuat(rx: number, ry: number, rz: number) {
  const cx = Math.cos(rx * 0.5), sx = Math.sin(rx * 0.5)
  const cy = Math.cos(ry * 0.5), sy = Math.sin(ry * 0.5)
  const cz = Math.cos(rz * 0.5), sz = Math.sin(rz * 0.5)
  const w = cy * cx * cz + sy * sx * sz
  const x = cy * sx * cz + sy * cx * sz
  const y = sy * cx * cz - cy * sx * sz
  const z = cy * cx * sz - sy * sx * cz
  const len = Math.hypot(x, y, z, w) || 1
  const inv = 1 / len
  return { x: x * inv, y: y * inv, z: z * inv, w: w * inv }
}
