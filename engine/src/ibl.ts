// Diffuse irradiance from an HDRI, as 9 spherical-harmonic coefficients — the
// Ramamoorthi–Hanrahan formulation every engine uses, because integrating the
// whole sky per fragment is absurd and an SH2 fit of irradiance is within 1%
// for natural skies.
//
// Pure and headlessly testable (the shadow-cascades precedent). The output is
// PRE-FOLDED: all basis and convolution constants are baked into the nine RGB
// vectors here, so the shader is a plain polynomial in the normal — and the
// whole thing is normalised so a uniform sky of radiance 1 lights every
// surface with exactly 1, which is what makes an HDRI world a drop-in for the
// flat world colour it replaces.
//
// Direction convention matches the composite's equirect sampling exactly
// (LH, +Z forward: u = 0.5 + atan2(x, z)/2π, v = 0.5 − asin(y)/π), because a
// sky that lights the character from anywhere but where it is drawn would be
// worse than no IBL at all.

import type { HdrImage } from "./hdr"

/**
 * Project an equirect HDRI to folded irradiance SH.
 *
 * Returns 9 RGB triplets (27 floats): [A, By, Bz, Bx, Cxy, Cyz, Cz2, Cxz, Cx2y2]
 * for  E(n) = A + By·y + Bz·z + Bx·x + Cxy·xy + Cyz·yz + Cz2·(3z²−1) + Cxz·xz + Cx2y2·(x²−y²).
 *
 * `stride` subsamples the image — irradiance is the lowest-frequency thing an
 * HDRI carries, so every 4th texel of a 2K is already overkill.
 */
export function projectIrradianceSH(img: HdrImage, stride = 4): Float32Array {
  const { width: w, height: h, data } = img
  // Accumulated radiance moments L_lm, RGB each.
  const L = new Float64Array(9 * 3)
  const dPhi = (2 * Math.PI) / w
  const dTheta = Math.PI / h
  for (let y = 0; y < h; y += stride) {
    const v = (y + 0.5) / h
    const ny = Math.cos(Math.PI * v)
    const r = Math.sin(Math.PI * v)
    // Solid angle of one texel row-slice, times the stride² it stands in for.
    const dw = r * dPhi * dTheta * stride * stride
    if (dw <= 0) continue
    for (let x = 0; x < w; x += stride) {
      const u = (x + 0.5) / w
      const phi = (u - 0.5) * 2 * Math.PI
      const nx = r * Math.sin(phi)
      const nz = r * Math.cos(phi)
      const i = (y * w + x) * 4
      // Real SH basis, evaluated once and weighted by the solid angle.
      const b0 = 0.282095 * dw
      const b1 = 0.488603 * ny * dw
      const b2 = 0.488603 * nz * dw
      const b3 = 0.488603 * nx * dw
      const b4 = 1.092548 * nx * ny * dw
      const b5 = 1.092548 * ny * nz * dw
      const b6 = 0.315392 * (3 * nz * nz - 1) * dw
      const b7 = 1.092548 * nx * nz * dw
      const b8 = 0.546274 * (nx * nx - ny * ny) * dw
      for (let c = 0; c < 3; c++) {
        const rad = data[i + c]
        L[0 * 3 + c] += rad * b0
        L[1 * 3 + c] += rad * b1
        L[2 * 3 + c] += rad * b2
        L[3 * 3 + c] += rad * b3
        L[4 * 3 + c] += rad * b4
        L[5 * 3 + c] += rad * b5
        L[6 * 3 + c] += rad * b6
        L[7 * 3 + c] += rad * b7
        L[8 * 3 + c] += rad * b8
      }
    }
  }
  // Irradiance convolution (Ramamoorthi's c1..c5), then /π for the Lambert
  // convention — this is the normalisation that makes uniform-1 → exactly 1.
  const c1 = 0.429043
  const c2 = 0.511664
  const c3 = 0.743125
  const c4 = 0.886227
  const c5 = 0.247708
  const out = new Float32Array(27)
  for (let c = 0; c < 3; c++) {
    const l00 = L[0 * 3 + c]
    const l1m1 = L[1 * 3 + c]
    const l10 = L[2 * 3 + c]
    const l11 = L[3 * 3 + c]
    const l2m2 = L[4 * 3 + c]
    const l2m1 = L[5 * 3 + c]
    const l20 = L[6 * 3 + c]
    const l21 = L[7 * 3 + c]
    const l22 = L[8 * 3 + c]
    // c3·L20·z² is refolded onto the (3z²−1) basis: the z²-free remainder
    // joins the constant term, so the shader's polynomial needs no z² row.
    out[0 * 3 + c] = (c4 * l00 - c5 * l20 + (c3 / 3) * l20) / Math.PI // A
    out[1 * 3 + c] = (2 * c2 * l1m1) / Math.PI // By
    out[2 * 3 + c] = (2 * c2 * l10) / Math.PI // Bz
    out[3 * 3 + c] = (2 * c2 * l11) / Math.PI // Bx
    out[4 * 3 + c] = (2 * c1 * l2m2) / Math.PI // Cxy
    out[5 * 3 + c] = (2 * c1 * l2m1) / Math.PI // Cyz
    out[6 * 3 + c] = ((c3 / 3) * l20) / Math.PI // Cz2 · (3z²−1)
    out[7 * 3 + c] = (2 * c1 * l21) / Math.PI // Cxz
    out[8 * 3 + c] = (2 * c1 * l22) / Math.PI // Cx2y2
  }
  return out
}

/** Evaluate the folded coefficients at a normal — the shader's polynomial, in
 *  JS, so tests and any future CPU consumer share one definition. */
export function evalIrradianceSH(sh: Float32Array, n: { x: number; y: number; z: number }): [number, number, number] {
  const { x, y, z } = n
  const basis = [1, y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y]
  const out: [number, number, number] = [0, 0, 0]
  for (let b = 0; b < 9; b++) {
    for (let c = 0; c < 3; c++) out[c] += sh[b * 3 + c] * basis[b]
  }
  return out
}
