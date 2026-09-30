// Pictures an effect is handed by its host — `#textures N`.
//
//     #textures 2
//
// gives the PARTICLE SHADING rzTexture(i, uv) and rzTextureLod(i, uv, lod) over
// N images the host passes with the install (setEffects' `textures`, slot i =
// picture i). uv (0,0) is the image's top-left row, the style graph's
// tex_image convention, and it repeats. A slot the host left empty reads white.
//
// WHY: a game's splash is a picture on a card, and an effect that may only draw
// procedurally can only ever guess at it — shape, coverage, veins, each a guess
// to be corrected. A converted stage carries the game's own textures, and this
// is how its particles draw them.
//
// Fragment-only, because textureSample is: the compute stage and every other
// module the author's file is spliced into get STUBS returning zero, so one
// effect file compiles in every mount it has.

/** How many pictures one effect can read. */
export const MAX_EFFECT_TEXTURES = 4

/**
 * rzTexture / rzTextureLod. `count` is the effect's `#textures` (0 = stubs);
 * the textures take bindings first..first+count-1 in `group` and the one
 * sampler `sampler`.
 */
export function textureApi(count: number, group = 0, first = 0, sampler = 0): string {
  if (count <= 0) {
    return /* wgsl */ `
fn rzTexture(i: u32, uv: vec2f) -> vec4f { return vec4f(0.0); }
fn rzTextureLod(i: u32, uv: vec2f, lod: f32) -> vec4f { return vec4f(0.0); }
`
  }
  const n = Math.min(count, MAX_EFFECT_TEXTURES)
  let decl = `@group(${group}) @binding(${sampler}) var _rzTexSampler: sampler;\n`
  for (let k = 0; k < n; k++) decl += `@group(${group}) @binding(${first + k}) var _rzTex${k}: texture_2d<f32>;\n`
  // Every slot is sampled unconditionally and the one asked for is selected:
  // textureSample must be reached in uniform control flow, and a switch on a
  // per-fragment index is not.
  const pick = (call: (k: number) => string) => {
    let body = "  var c = vec4f(0.0);\n"
    for (let k = 0; k < n; k++) body += `  let s${k} = ${call(k)};\n  if (i == ${k}u) { c = s${k}; }\n`
    return body + "  return c;\n"
  }
  return (
    decl +
    /* wgsl */ `
/** Picture i of the effect's #textures, at uv ((0,0) top-left, repeating). */
fn rzTexture(i: u32, uv: vec2f) -> vec4f {
${pick((k) => `textureSample(_rzTex${k}, _rzTexSampler, uv)`)}}
/** The same at an explicit mip level — usable where derivatives are not. */
fn rzTextureLod(i: u32, uv: vec2f, lod: f32) -> vec4f {
${pick((k) => `textureSampleLevel(_rzTex${k}, _rzTexSampler, uv, lod)`)}}
`
  )
}
