// The game pipeline's per-frame globals, computed from the engine's scene.
//
// The game's shaders read what its render pipeline (UnityEngine.Rendering.
// Replica / ReplicaExt, decompiled from GameAssembly.dll) sets before a camera
// renders: the camera, the main light, the Forward+ light table and its bins,
// the cascades, the ambient SH, fog, time. ag-rip's AGSimPipeline.cs and
// AGSimShadows.cs reproduce those calls in Unity, method by method, and the
// compare page proved them against the game's frames; this is that code again,
// fed by the engine instead of by a Unity scene.
//
// SPACE. The game's shaders work in the game's world, and the engine's scene
// is that world as ag-rip exports it to MMD: `scale` PMX units per game unit
// (8: the one-scene-scale rule) and turned half round Y — game (x, y, z) is
// PMX (-x, y, -z), the convention its camera and motion VMDs and its stages
// share. Everything positional is handed over through that map (gameToEngine
// and back) — the camera, the lights and their reach, the cascades — so a
// shader's distances (outline widths, light ranges, fog) and directions mean
// what they meant in the game. A half turn keeps handedness: both are
// left-handed and Y-up.
//
// CLIP SPACE. naga negates y on the way out of every vertex shader (Vulkan to
// WebGPU), so the projection handed to the shaders is the engine's with y
// flipped: the two flips cancel and a native draw lands exactly where the
// engine's own draw of the same triangle would, on the same depth.

import type { NativeValue } from "./host"

export type Mat4Like = ArrayLike<number>

export type UnityLight = {
  /** World position, engine units. */
  position: [number, number, number]
  /** Reach, engine units. */
  radius: number
  /** Linear colour times intensity. */
  color: [number, number, number]
  /** Unit aim, away from the light; zero for a point light. */
  aim: [number, number, number]
  cosOuter: number
  cosInner: number
  /** Rendering-layer bits it reaches. */
  layers: number
  /** What the game's pipeline packs beside a light it lit with, when the light
   *  came from a game stage (stage_native.py): spot direction with w = 1 /
   *  its sphere's radius (the area term reads it), colour alpha, Extra,
   *  WorldToLight and type — all in the game's world, used as they are. */
  unity?: {
    spotDir: number[]
    colorW: number
    extra: number[]
    worldToLight: number[] | null
    lightType: number
    simSpotDir?: number[] | null
    simColorW?: number
  }
}

export type UnityFrameInput = {
  /** Engine units per game unit. */
  scale: number
  view: Mat4Like
  proj: Mat4Like
  eye: [number, number, number]
  near: number
  far: number
  width: number
  height: number
  /** Scene clock and its step, seconds. */
  time: number
  dt: number
  /** The direction the main light travels, unit; null for none. */
  sunDirection: [number, number, number] | null
  sunColor: [number, number, number]
  /** A game stage's own main light as its pipeline packed it (recorded), which
   *  its shaders get in place of sunColor; null for the scene sun's. */
  sunUnity?: { color: number[]; simColor: number[]; simColorNoInt: number[] } | null
  sunShadow: number
  lights: UnityLight[]
  /** Ambient as the engine's folded SH9 (27 floats, ibl.ts), or null for flat. */
  ambientSH: ArrayLike<number> | null
  ambientFlat: [number, number, number]
  /** The cascades as the engine drew them: render view-projections (engine
   *  units, the atlas's own depth convention), their tiles, and the spheres
   *  they were fitted to. */
  shadow: {
    viewProj: Mat4Like[]
    tiles: [number, number][]
    tileSize: number
    atlasSize: number
    spheres: { center: [number, number, number]; radius: number }[]
  } | null
  /** What a loaded game stage sets for its scene (stage.ts): fog, tint, ambient
   *  SH, the environment cube's numbers — the pipeline's per-scene globals, in
   *  place of the defaults below. Never camera, time, lights or shadows. */
  settings?: Record<string, NativeValue> | null
}

const PLUS_MAX = 255

/** Column-major 4x4 multiply. */
export function mul4(a: Mat4Like, b: Mat4Like): Float32Array {
  const o = new Float32Array(16)
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++) {
      let s = 0
      for (let k = 0; k < 4; k++) s += a[k * 4 + r] * b[c * 4 + k]
      o[c * 4 + r] = s
    }
  return o
}

export function invert4(m: Mat4Like): Float32Array {
  const a = m
  const o = new Float32Array(16)
  const a00 = a[0],
    a01 = a[1],
    a02 = a[2],
    a03 = a[3],
    a10 = a[4],
    a11 = a[5],
    a12 = a[6],
    a13 = a[7]
  const a20 = a[8],
    a21 = a[9],
    a22 = a[10],
    a23 = a[11],
    a30 = a[12],
    a31 = a[13],
    a32 = a[14],
    a33 = a[15]
  const b00 = a00 * a11 - a01 * a10,
    b01 = a00 * a12 - a02 * a10,
    b02 = a00 * a13 - a03 * a10
  const b03 = a01 * a12 - a02 * a11,
    b04 = a01 * a13 - a03 * a11,
    b05 = a02 * a13 - a03 * a12
  const b06 = a20 * a31 - a21 * a30,
    b07 = a20 * a32 - a22 * a30,
    b08 = a20 * a33 - a23 * a30
  const b09 = a21 * a32 - a22 * a31,
    b10 = a21 * a33 - a23 * a31,
    b11 = a22 * a33 - a23 * a32
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
  if (!det) return Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
  det = 1 / det
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det
  o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det
  return o
}

/** Game world -> engine world: `s` PMX per unit, half a turn about Y. */
export function gameToEngine(s: number): Float32Array {
  return Float32Array.from([-s, 0, 0, 0, 0, s, 0, 0, 0, 0, -s, 0, 0, 0, 0, 1])
}

/** Engine world -> game world. */
export function engineToGame(s: number): Float32Array {
  return Float32Array.from([-1 / s, 0, 0, 0, 0, 1 / s, 0, 0, 0, 0, -1 / s, 0, 0, 0, 0, 1])
}

/** An engine point in the game's world. */
export function gamePoint(p: ArrayLike<number>, s: number): [number, number, number] {
  return [-p[0] / s, p[1] / s, -p[2] / s]
}

/** An engine direction in the game's world. */
export function gameDir(d: ArrayLike<number>): [number, number, number] {
  return [-d[0], d[1], -d[2]]
}

/** Uniform scale, as a column-major matrix. */
export function scale4(s: number): Float32Array {
  return Float32Array.from([s, 0, 0, 0, 0, s, 0, 0, 0, 0, s, 0, 0, 0, 0, 1])
}

/** The projection with clip y negated — undone by naga's own negation. */
export function flipY(p: Mat4Like): Float32Array {
  const o = Float32Array.from(p)
  for (let c = 0; c < 4; c++) o[c * 4 + 1] = -p[c * 4 + 1]
  return o
}

/**
 * The engine's folded SH9 (ibl.ts: c0 + c1·y + c2·z + c3·x + c4·xy + c5·yz +
 * c6·(3z²−1) + c7·xz + c8·(x²−y²)) as Unity's seven shader constants —
 * SHA = (x, y, z, constant), SHB = (xy, yz, z², zx), SHC = (x²−y²) — the same
 * polynomial, regrouped.
 */
export function unitySH(sh: ArrayLike<number> | null, flat: [number, number, number]): Record<string, number[]> {
  const out: Record<string, number[]> = {}
  const ch = ["r", "g", "b"]
  const c = (i: number, k: number) => (sh ? sh[i * 3 + k] : i === 0 ? flat[k] : 0)
  for (let k = 0; k < 3; k++) {
    out[`_Replica_SHA${ch[k]}`] = [c(3, k), c(1, k), c(2, k), c(0, k) - c(6, k)]
    out[`_Replica_SHB${ch[k]}`] = [c(4, k), c(5, k), 3 * c(6, k), c(7, k)]
  }
  out._Replica_SHC = [c(8, 0), c(8, 1), c(8, 2), 1]
  return out
}

/** Everything the frame sets, by name. */
export function unityFrameGlobals(f: UnityFrameInput): Record<string, NativeValue> {
  const s = f.scale
  const g: Record<string, NativeValue> = {}

  // ---- the camera (UnityPerFrame), in game units
  // Unity's own pair, so what a shader rebuilds from them means what it did
  // in the game: the view in game units, right-handed and looking down -Z;
  // the projection on game-unit depths (the engine's, conjugated by the scale
  // and the z flip, and divided through by s, which leaves the same NDC). Their
  // product puts every vertex exactly where the engine's own draw of it lands.
  const zFlip = Float32Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1])
  const view = mul4(zFlip, mul4(scale4(1 / s), mul4(f.view, gameToEngine(s))))
  const toEngineView = mul4(scale4(s), zFlip)
  const proj = flipY(mul4(f.proj, toEngineView))
  for (let i = 0; i < 16; i++) proj[i] /= s
  const vp = mul4(proj, view)
  const w = Math.max(1, f.width)
  const h = Math.max(1, f.height)
  const near = f.near / s
  const far = f.far / s
  g._WorldSpaceCameraPos = [...gamePoint(f.eye, s), 1]
  g.unity_MatrixV = view
  g.unity_MatrixInvV = invert4(view)
  g.unity_MatrixP = proj
  g.glstate_matrix_projection = proj
  g.unity_MatrixVP = vp
  g._NonJitteredViewProjMatrix = vp
  // x = -1: a flipped projection, as Unity's into a texture.
  g._ProjectionParams = [-1, near, far, 1 / far]
  g._ScreenParams = [w, h, 1 + 1 / w, 1 + 1 / h]
  g._ScaledScreenParams = [w, h, 1 + 1 / w, 1 + 1 / h]
  g._ZBufferParams = [-1 + far / near, 1, (-1 + far / near) / far, 1 / far]
  g.unity_OrthoParams = [0, 0, 0, 0]
  const t = f.time
  g._Time = [t / 20, t, t * 2, t * 3]
  g._SinTime = [Math.sin(t / 8), Math.sin(t / 4), Math.sin(t / 2), Math.sin(t)]
  g._CosTime = [Math.cos(t / 8), Math.cos(t / 4), Math.cos(t / 2), Math.cos(t)]
  const dt = Math.max(f.dt, 1e-4)
  g.unity_DeltaTime = [dt, 1 / dt, dt, 1 / dt]
  // UpdateRenderSettings: sim_Time = (t - floor t, floor t)
  g.sim_Time = [t - Math.floor(t), Math.floor(t), 0, 0]
  g.identity_matrix = scale4(1)

  // ---- LightingFeatures.SetupLightings / PlusLighting SetupMainLightConstants
  if (f.sunDirection) {
    const d = gameDir(f.sunDirection)
    g.SimMainLightDir = [-d[0], -d[1], -d[2], 0]
    const u = f.sunUnity
    g.SimMainLightColor = u ? u.simColor.slice(0, 4) : [...f.sunColor, 1]
    g.SimMainLightColorNoInt = u ? u.simColorNoInt.slice(0, 4) : [...f.sunColor, 1]
    // As the game's pipeline leaves it (recorded): the way the light travels.
    g._MainLightPosition = [d[0], d[1], d[2], 0]
    g.sim_ShadowLightDirection = [-d[0], -d[1], -d[2], 0]
    g._MainLightColor = u ? u.color.slice(0, 4) : [...f.sunColor, 1]
  } else {
    g.SimMainLightDir = [0, 0, 0, 0]
    g.SimMainLightColor = [0, 0, 0, 0]
    g.SimMainLightColorNoInt = [0, 0, 0, 0]
    g._MainLightPosition = [0, 1, 0, 0]
    g._MainLightColor = [0, 0, 0, 0]
  }

  // ---- ForwardFeature.SetupRenderFeature, UpdateRenderSettings: a scene
  // with no stage of the game's has no settings of its own — the defaults the
  // game's SceneSetting starts from.
  g.SimSceneTint = [1, 1, 1, 1]
  g._ReceiveSceneShadow = 1
  g.sim_FogRotation = 0
  g.sim_FogInverseRootRotationMatrix = scale4(1)
  g.sim_FogColor = [0, 0, 0, 0]
  g.sim_FogColor2 = [0, 0, 0, 0]
  g.sim_FogParams = [0, 1, 0, 1]
  g.sim_FogDirectionalColor = [0, 0, 0, 0]
  g.sim_FogDirectionalDir = [0, 0, 1, 0]
  g.sim_FogDirectionalFalloff = 0
  g.sim_DYN_FOG_LINEAR_Variable = 0
  g.sim_DynFogColor = [0, 0, 0, 0]
  g.sim_DynFogParams = [0, 1, 0, 1]
  g._ProbeLightingBase = [1, 1, 1, 1]
  g._ProbeLightingScale = 0
  g._AcceptLightProbe = 0
  g.sim_VertexAmbientScale = 1
  g._EnableShadowScatter = 0
  g._ShadowScatterColor = [0, 0, 0, 0]
  g._ShadowScatterRange = 0
  // realtimeShadowColor's default grey, made linear (ConvertSRGBToActiveColorSpace)
  g._SubtractiveShadowColor = [0.2158605, 0.2158605, 0.2158605, 1]
  g._AmbientOcclusionParam = [0, 0, 0, 0]
  g.sim_EnvCubeScale = 0
  g.sim_EnvCube_BoxMin = [0, 0, 0, 0]
  g.sim_EnvCube_BoxMax = [0, 0, 0, 0]
  g.sim_EnvCube_ProbePosition = [0, 0, 0, 0]

  // ---- SetupEnvironmentLighting: the scene's ambient SH
  // The half turn negates x and z: the terms odd in either (z, x, xy, yz) flip.
  const sh = f.ambientSH ? Array.from(f.ambientSH) : null
  if (sh) for (const i of [2, 3, 4, 5]) for (let k = 0; k < 3; k++) sh[i * 3 + k] = -sh[i * 3 + k]
  Object.assign(g, unitySH(sh, f.ambientFlat))

  // ---- a game stage's own settings over all of the above
  if (f.settings) Object.assign(g, f.settings)

  // ---- PlusLightingFeature: the light table, one tile and one depth bin
  // holding every light (binning only culls; the shader applies each range)
  const n = Math.min(f.lights.length, PLUS_MAX)
  const pos: number[][] = [],
    col: number[][] = [],
    att: number[][] = [],
    dir: number[][] = []
  const extra: number[][] = [],
    w2l: Float32Array[] = []
  const simDir: number[][] = [],
    simColorW: number[] = []
  const types = new Float32Array(PLUS_MAX)
  const masks = new Uint32Array(PLUS_MAX)
  const ident = scale4(1)
  for (let i = 0; i < PLUS_MAX; i++) {
    const l = f.lights[i]
    if (i >= n || !l) {
      pos.push([0, 0, 0, 0])
      col.push([0, 0, 0, 0])
      att.push([0, 1, 0, 1])
      dir.push([0, 0, 0, 0])
      extra.push([0, 0, 0, 0])
      w2l.push(ident)
      continue
    }
    const spot = l.aim[0] !== 0 || l.aim[1] !== 0 || l.aim[2] !== 0
    const r = l.radius / s
    const r2 = r * r
    // LightInfo.GetPunctualLightDistanceAttenuation / GetSpotAngleAttenuation
    let z = 0
    let ww = 1
    if (spot) {
      const inv = 1 / Math.max(l.cosInner - l.cosOuter, 0.001)
      z = inv
      ww = -l.cosOuter * inv
    }
    pos.push([...gamePoint(l.position, s), 1])
    // The engine's lamp brightness is per its own unit; the inverse square in
    // game units is s² larger at the same place, so the colour carries 1/s².
    col.push([l.color[0] / (s * s), l.color[1] / (s * s), l.color[2] / (s * s), 1])
    att.push([1 / Math.max(r2, 1e-4), -r2 / (r2 * 0.64 - r2), z, ww])
    const u = l.unity
    if (u) {
      col[col.length - 1][3] = u.colorW
      dir.push(u.spotDir.slice(0, 4))
      extra.push(u.extra.slice(0, 4))
      w2l.push(u.worldToLight ? Float32Array.from(u.worldToLight) : ident)
      simDir.push((u.simSpotDir ?? u.spotDir).slice(0, 4))
      simColorW.push(u.simColorW ?? 1)
      types[i] = u.lightType
    } else {
      // The pipeline's own defaults for a light it did not author: forward
      // along the aim, w = 1 / the 0.1 sphere every game lamp of these stages has.
      const a = gameDir(l.aim)
      dir.push(spot ? [...a, 10] : [0, 0, 1, 10])
      extra.push([0, 0, 0, 0])
      w2l.push(ident)
      simDir.push(spot ? [...a, 10] : [0, 0, 1, 10])
      simColorW.push(1)
      types[i] = spot ? 0 : 2
    }
    masks[i] = l.layers >>> 0
  }
  g._AdditionalLightsPosition = pos
  g._AdditionalLightsColor = col
  g._AdditionalLightsAttenuation = att
  g._AdditionalLightsSpotDir = dir
  g._AdditionalLightsExtra = extra
  g._AdditionalLightsWorldToLights = w2l
  g._AdditionalLightsLightTypes = types
  g._AdditionalLightsLayerMasks = masks
  g._AdditionalLightsCookieEnableBits = new Float32Array(8)
  const words = Math.max(1, Math.ceil(n / 32))
  const zb = new Uint32Array(4096)
  const tiles = new Uint32Array(16384)
  zb[0] = n > 0 ? ((n - 1) << 16) >>> 0 : 0xffff
  for (let k = 0; k < words; k++) {
    const bits = Math.min(Math.max(n - k * 32, 0), 32)
    const m = bits >= 32 ? 0xffffffff : ((1 << bits) >>> 0) - 1
    zb[1 + k] = m >>> 0
    tiles[k] = m >>> 0
  }
  // LightingFeatures' own copy for the per-vertex lights: the first 16, colour
  // without the intensity in w.
  g.SimAdditionalLightPosition = pos.slice(0, 16)
  g.SimAdditionalLightColor = col
    .slice(0, 16)
    .map((c, i) => (c[0] || c[1] || c[2] ? [c[0], c[1], c[2], simColorW[i] ?? 1] : [0, 0, 0, 0]))
  g.SimAdditionalLightAttenuation = att.slice(0, 16)
  g.SimAdditionalLightSpotDir = simDir.slice(0, 16).concat(dir.slice(simDir.length, 16))
  g._PlusLighting_ZBins = zb
  g._PlusLighting_Tiles = tiles
  g._PlusLightingParams0 = [0, 0, 1, words]
  g._PlusLightingParams1 = [1000, 0, 0, 0]

  // ---- the main light's cascades, sampled from the engine's atlas
  if (f.shadow) {
    const sh = f.shadow
    const unit = sh.tileSize / sh.atlasSize
    const w2s: Float32Array[] = []
    for (let i = 0; i < 4; i++) {
      const vpe = sh.viewProj[i]
      const tile = sh.tiles[i]
      if (!vpe || !tile) {
        w2s.push(scale4(0))
        continue
      }
      // ndc -> the tile's uv (y down, as the atlas is addressed), depth as drawn
      const tx = tile[0] / sh.atlasSize
      const ty = tile[1] / sh.atlasSize
      const toTile = Float32Array.from([0.5 * unit, 0, 0, 0, 0, -0.5 * unit, 0, 0, 0, 0, 1, 0, 0.5 * unit + tx, 0.5 * unit + ty, 0, 1])
      w2s.push(mul4(toTile, mul4(vpe, gameToEngine(s))))
    }
    g._MainLightWorldToShadow = w2s
    for (let i = 0; i < 4; i++) {
      const sp = sh.spheres[i]
      g[`_CascadeShadowSplitSpheres${i}`] = sp ? [...gamePoint(sp.center, s), sp.radius / s] : [0, 0, 0, 0]
    }
    const r = (i: number) => (sh.spheres[i] ? (sh.spheres[i].radius / s) ** 2 : 0)
    g._CascadeShadowSplitSphereRadii = [r(0), r(1), r(2), r(3)]
    const half = 0.5 / sh.atlasSize
    g._MainLightShadowOffset0 = [-half, -half, half, -half]
    g._MainLightShadowOffset1 = [-half, half, half, half]
    g._MainLightShadowmapSize = [1 / sh.atlasSize, 1 / sh.atlasSize, sh.atlasSize, sh.atlasSize]
    g._MainLightShadowParams = [f.sunShadow, 1, 0, 0]
  } else {
    g._MainLightShadowParams = [0, 0, 0, 0]
  }
  return g
}
