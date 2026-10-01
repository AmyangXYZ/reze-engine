// The game's per-character shading inputs, for a model wearing its character
// shaders.
//
// In the game a CharacterEffect component (P08.RenderPipeline, ported in
// ag-rip's charfx/CharacterEffect.cs from the decompiled GameAssembly.dll)
// writes one property block shared by all of a character's renderers: its own
// key light, expressed in a camera-facing frame (_LocalLightDir in
// sim_TowardMatrix, LightingFeatures.SetupTowardMatrix); its face frame for the
// SDF face shadow (_World2Face); its rim lights; its fills; and an ambient of
// its own from three colours (EnvironmentEffect). Nothing here is per scene:
// it is the character's look, carried by the character.
//
// An MMD model has no CharacterEffect. It takes a CharacterRig — the values a
// game character's prefab carries, exported with the look — and the frames
// come from its own bones: the face is the head bone turned to face the way
// the game's faces do (the game's face node rests aligned with the world,
// looking +Z; a PMX model looks −Z, so a half turn about Y).

import type { NativeValue } from "./host"
import { gameToEngine, invert4, mul4 } from "./globals"

/** A game character's CharacterEffect values, as its prefab serialises them. */
export type CharacterRig = {
  localLightInclination: number
  localLightAzimuth: number
  localLightIntensity: number
  localLightColor: [number, number, number, number]
  faceReceiveShadow: boolean
  disableFaceSDFShadow: boolean
  emissionColor: [number, number, number, number]
  fillOuter: [number, number, number, number]
  fillInner: [number, number, number, number]
  fillColor: [number, number, number, number]
  fillRatio: number
  rimlightColor: [number, number, number, number]
  rimlightThreshold: number
  rimlightFade: number
  rimlightRange: number
  rimlightInclination: number
  rimlightAzimuth1: number
  rimlightAzimuth2: number
  characterLayer: number
  /** EnvironmentEffect's trilight, sRGB as serialised. */
  skyColor: [number, number, number]
  equatorColor: [number, number, number]
  groundColor: [number, number, number]
}

const srgbToLinear = (c: number) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4))

/**
 * Unity's ambient probe of a trilight (RenderSettings.ambientProbe in Trilight
 * mode), measured in Unity 6 rather than derived: the colours are taken to
 * linear, then
 *   c0 = (S + 4E + G) / 6        c1 = (√6 / 9)(S − G)
 *   c6 = (5/192)(2E − S − G)     c8 = (5/64)(2E − S − G)
 * and every other coefficient is zero — Unity's raw coefficients, which its
 * shader constants pack as SHA = (c3, c1, c2, c0 − c6), SHB = (c4, c5, 3c6, c7),
 * SHC = c8. Checked against the game's recorded block for (0.8, 0.5, 0.3).
 */
export function trilightProbe(
  sky: [number, number, number],
  equator: [number, number, number],
  ground: [number, number, number],
): number[][] {
  const out: number[][] = []
  for (let k = 0; k < 3; k++) {
    const S = srgbToLinear(sky[k]),
      E = srgbToLinear(equator[k]),
      G = srgbToLinear(ground[k])
    const ring = 2 * E - S - G
    out.push([(S + 4 * E + G) / 6, (Math.sqrt(6) / 9) * (S - G), 0, 0, 0, 0, (5 / 192) * ring, 0, (5 / 64) * ring])
  }
  return out
}

/** Unity's shader constants from raw coefficients (channel, index). */
export function probeConstants(c: number[][]): Record<string, number[]> {
  const ch = ["r", "g", "b"]
  const o: Record<string, number[]> = {}
  for (let k = 0; k < 3; k++) {
    const s = c[k]
    o[`SHA${ch[k]}`] = [s[3], s[1], s[2], s[0] - s[6]]
    o[`SHB${ch[k]}`] = [s[4], s[5], s[6] * 3, s[7]]
  }
  o.SHC = [c[0][8], c[1][8], c[2][8], 1]
  return o
}

/**
 * sim_TowardMatrix (LightingFeatures.SetupTowardMatrix): the frame the
 * character light is expressed in — the camera's backward direction flattened
 * to the ground, as columns (up × d, up, d). Taken immediately, as the DLC
 * poster sets CameraExtension.shadowRotationImmediately.
 */
export function towardMatrix(cameraForward: [number, number, number]): Float32Array {
  const x = -cameraForward[0],
    z = -cameraForward[2]
  const len = Math.hypot(x, z)
  const d = len > 1e-5 ? [x / len, 0, z / len] : [0, 0, 0]
  return Float32Array.from([d[2], 0, -d[0], 0, 0, 1, 0, 0, d[0], d[1], d[2], 0, 0, 0, 0, 1])
}

/** CharacterGlobalParams.Execute: the outline's resolution allowance. */
export function outlineMaxOffsetMultiplier(pixelWidth: number): number {
  return Math.max(pixelWidth / 768 - 1, 0)
}

/**
 * The character's property block (CharacterEffect.LateUpdate with the default
 * look: no dissolve, dither, interference), by name.
 *
 * `head` is the head bone's skinning matrix (its pose relative to the bind
 * pose, engine units) and `headRest` its rest position; `scale` engine units
 * per game unit.
 */
export function characterBlock(
  rig: CharacterRig,
  head: ArrayLike<number> | null,
  headRest: [number, number, number],
  scale: number,
): Record<string, NativeValue> {
  const o: Record<string, NativeValue> = {}
  // _World2Face: the face frame's world-to-local, in game units. The face is
  // the head's pose applied to a frame at the head's rest position looking −Z.
  const faceRest = Float32Array.from([-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, headRest[0], headRest[1], headRest[2], 1])
  const face = head ? mul4(head, faceRest) : faceRest
  o._World2Face = mul4(invert4(face), gameToEngine(scale))
  o._ExtralEmissionColor = rig.emissionColor
  let inner = rig.fillInner,
    outer = rig.fillOuter
  inner = [inner[0], inner[1], inner[2], inner[3] * rig.fillRatio]
  outer = [outer[0], outer[1], outer[2], outer[3] * rig.fillRatio]
  o._FillOuter = outer
  o._FillInner = inner
  o._FillClamp = 0
  o._FillSoft = 0
  o._DissolveFactor = 0
  // inclination about the horizon, azimuth negated — float math as the game's
  const incl = (rig.localLightInclination * Math.PI) / 180
  const az = (-rig.localLightAzimuth * Math.PI) / 180
  const c = Math.cos(incl)
  o._LocalLightDir = [Math.sin(az) * -c, Math.sin(incl), Math.cos(az) * c, 0]
  const k = rig.localLightIntensity
  o._LocalLightColor = [rig.localLightColor[0] * k, rig.localLightColor[1] * k, rig.localLightColor[2] * k, rig.localLightColor[3] * k]
  o._UseFaceReceiveShadow = rig.faceReceiveShadow ? 1 : 0
  o._DisableFaceSDFShadow = rig.disableFaceSDFShadow ? 1 : 0
  o._RimlightColor = rig.rimlightColor
  o._RimlightThreshold = rig.rimlightThreshold
  o._RimlightFade = rig.rimlightFade
  o._RimlightRange = rig.rimlightRange
  const zr = rig.rimlightInclination
  const sr = Math.sqrt(1 - zr * zr)
  const a1 = rig.rimlightAzimuth1 * Math.PI,
    a2 = rig.rimlightAzimuth2 * Math.PI
  o._RimlightDir1 = [Math.cos(a1) * sr, Math.sin(a1) * sr, zr, 0]
  o._RimlightDir2 = [Math.cos(a2) * sr, Math.sin(a2) * sr, zr, 0]
  o._ZOffset_Unit = rig.characterLayer
  o._DitherAlpha = 1
  // EnvironmentEffect: the character's own ambient, in both names it is read by
  const sh = probeConstants(trilightProbe(rig.skyColor, rig.equatorColor, rig.groundColor))
  for (const [n, v] of Object.entries(sh)) {
    o[`unity_${n}`] = v
    o[`_Replica_${n}`] = v
  }
  // CharacterEffect.OnEnable: every character renderer on rendering layer 0x40000001
  o.unity_RenderingLayer = new Uint32Array([0x40000001, 0, 0, 0])
  return o
}
