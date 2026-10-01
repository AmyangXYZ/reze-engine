// A Unity pass's fixed-function state (Blend, ZTest, Cull, Stencil, ColorMask,
// Offset), as its ShaderLab names or UnityEngine.Rendering enum numbers, into
// WebGPU's. Ported from the compare page's unity.js, where these were checked
// draw for draw against the game's frames.

/** A pass's state as the exporter writes it: each key's ShaderLab arguments. */
export type PassState = Record<string, (string | number)[]>

const BLEND: GPUBlendFactor[] = [
  "zero",
  "one",
  "dst",
  "src",
  "one-minus-dst",
  "src-alpha",
  "one-minus-src",
  "dst-alpha",
  "one-minus-dst-alpha",
  "src-alpha-saturated",
  "one-minus-src-alpha",
]
const BLEND_NAMES: Record<string, GPUBlendFactor> = {
  zero: "zero",
  one: "one",
  dstcolor: "dst",
  srccolor: "src",
  oneminusdstcolor: "one-minus-dst",
  srcalpha: "src-alpha",
  oneminussrccolor: "one-minus-src",
  dstalpha: "dst-alpha",
  oneminusdstalpha: "one-minus-dst-alpha",
  srcalphasaturate: "src-alpha-saturated",
  oneminussrcalpha: "one-minus-src-alpha",
}
const isNum = (v: unknown) => typeof v === "number" || /^-?\d+(\.\d+)?$/.test(String(v))

export function blendFactor(v: string | number | undefined): GPUBlendFactor {
  if (v == null) return "one"
  if (isNum(v)) return BLEND[Math.round(+v)] ?? "one"
  return BLEND_NAMES[String(v).toLowerCase()] ?? "one"
}

const BLEND_OP: GPUBlendOperation[] = ["add", "subtract", "reverse-subtract", "min", "max"]
export function blendOp(v: string | number | undefined): GPUBlendOperation {
  if (v == null) return "add"
  if (isNum(v)) return BLEND_OP[+v] ?? "add"
  const s = String(v).toLowerCase()
  return s === "sub" ? "subtract" : s === "revsub" ? "reverse-subtract" : s === "min" ? "min" : s === "max" ? "max" : "add"
}

// CompareFunction: 0 Disabled 1 Never 2 Less 3 Equal 4 LessEqual 5 Greater 6 NotEqual 7 GreaterEqual 8 Always
const CMP: (GPUCompareFunction | null)[] = [null, "never", "less", "equal", "less-equal", "greater", "not-equal", "greater-equal", "always"]
const CMP_NAMES: Record<string, number> = {
  never: 1,
  less: 2,
  equal: 3,
  lequal: 4,
  lessequal: 4,
  greater: 5,
  notequal: 6,
  gequal: 7,
  greaterequal: 7,
  always: 8,
  disabled: 8,
}
export function compare(v: string | number | undefined, fallback: GPUCompareFunction = "always"): GPUCompareFunction {
  const n = v != null && isNum(v) ? +v : CMP_NAMES[String(v ?? "").toLowerCase()]
  if (n == null) return fallback
  return CMP[n] ?? "always"
}

const REVERSE: Partial<Record<GPUCompareFunction, GPUCompareFunction>> = {
  less: "greater",
  "less-equal": "greater-equal",
  greater: "less",
  "greater-equal": "less-equal",
}
/** A ZTest, flipped when the depth buffer is reversed — as Unity flips it on D3D. */
export function depthCompare(v: string | number | undefined, reversed: boolean): GPUCompareFunction {
  const c = compare(v ?? "LEqual", "less-equal")
  return reversed ? (REVERSE[c] ?? c) : c
}

// StencilOp: 0 Keep 1 Zero 2 Replace 3 IncrSat 4 DecrSat 5 Invert 6 IncrWrap 7 DecrWrap
const SOP: GPUStencilOperation[] = [
  "keep",
  "zero",
  "replace",
  "increment-clamp",
  "decrement-clamp",
  "invert",
  "increment-wrap",
  "decrement-wrap",
]
const SOP_NAMES: Record<string, number> = {
  keep: 0,
  zero: 1,
  replace: 2,
  incrsat: 3,
  decrsat: 4,
  invert: 5,
  incrwrap: 6,
  decrwrap: 7,
}
export function stencilOp(v: string | number | undefined): GPUStencilOperation {
  if (v == null) return "keep"
  const n = isNum(v) ? +v : SOP_NAMES[String(v).toLowerCase()]
  return SOP[n ?? 0] ?? "keep"
}

// CullMode: 0 Off 1 Front 2 Back
export function cull(v: string | number | undefined): GPUCullMode {
  if (v == null) return "back"
  const s = String(v).toLowerCase()
  if (s === "0" || s === "off") return "none"
  if (s === "1" || s === "front") return "front"
  return "back"
}

export function zwrite(v: string | number | undefined): boolean {
  if (v == null) return true
  const s = String(v).toLowerCase()
  return !(s === "0" || s === "off" || s === "false")
}

/** Unity ColorWriteMask (A 1, B 2, G 4, R 8, or letters) to WebGPU's (R 1, G 2, B 4, A 8). */
export function colorMask(v: string | number | undefined): number {
  if (v == null) return 0xf
  const s = String(v).toUpperCase()
  if (/^\d+$/.test(s)) {
    const n = +s
    return (n & 8 ? 1 : 0) | (n & 4 ? 2 : 0) | (n & 2 ? 4 : 0) | (n & 1 ? 8 : 0)
  }
  return (s.includes("R") ? 1 : 0) | (s.includes("G") ? 2 : 0) | (s.includes("B") ? 4 : 0) | (s.includes("A") ? 8 : 0)
}

/** The colour blend a pass declares, or undefined for opaque (One Zero). */
export function blendState(st: PassState): GPUBlendState | undefined {
  const b = st.Blend?.map(String)
  if (!b?.length || b[0].toLowerCase() === "off") return undefined
  const color: GPUBlendComponent = {
    srcFactor: blendFactor(b[0]),
    dstFactor: blendFactor(b[1] ?? "zero"),
    operation: blendOp(st.BlendOp?.[0]),
  }
  const alpha: GPUBlendComponent =
    b.length >= 4
      ? {
          srcFactor: blendFactor(b[2]),
          dstFactor: blendFactor(b[3]),
          operation: blendOp(st.BlendOp?.[1] ?? st.BlendOp?.[0]),
        }
      : color
  const opaque = (c: GPUBlendComponent) => c.srcFactor === "one" && c.dstFactor === "zero"
  return opaque(color) && opaque(alpha) ? undefined : { color, alpha }
}

/** The stencil a pass declares, or undefined when it touches none. */
export function stencilState(st: PassState): { face: GPUStencilFaceState; read: number; write: number; ref: number } | undefined {
  if (!st.StencilRef && !st.StencilComp) return undefined
  return {
    face: {
      compare: compare(st.StencilComp?.[0] ?? "Always"),
      passOp: stencilOp(st.StencilPass?.[0]),
      failOp: stencilOp(st.StencilFail?.[0]),
      depthFailOp: stencilOp(st.StencilZFail?.[0]),
    },
    read: +(st.StencilReadMask?.[0] ?? 255),
    write: +(st.StencilWriteMask?.[0] ?? 255),
    ref: st.StencilRef ? +st.StencilRef[0] : 0,
  }
}
