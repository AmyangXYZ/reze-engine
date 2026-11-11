// Words an effect is handed — `#param text NAME <default words>`.
//
// The engine sets every string an effect's text dials can show (each dial's
// value and every key on its track) into ONE picture per effect, a row a
// string, and a dial's slot in the params block holds the rect of the string it
// shows now. Changing the words, by hand or by a key, is a rect write; the
// picture is redrawn only when a string it has never held arrives.
//
// Canvas2D, as the lyric atlas: it sets every script the platform does, CJK
// included, which a glyph atlas cannot promise. The look — fill, edge, glow —
// is the author's, in WGSL, from the coverage rzText_NAME answers.
//
// FIELD MODULE ONLY, where the picture is bound. Every other module the file is
// spliced into gets stubs answering zero, so one effect file compiles in every
// mount it has.

import type { LyricRect } from "./lyrics-api"

/** The font size the rows are set at, in texels. A boss's name drawn 60 pixels
 *  tall on a 4K frame is still minified from this. */
export const EFFECT_TEXT_PX = 96
const PAD = 8
const MAX_W = 4096
const MAX_H = 8192

export type EffectTextFont = { css: string; tracking: number } | null

/** The CSS font for a size: `#font`'s words with the size put where CSS wants
 *  it — after style, variant and weight, before the family. */
export function effectTextFont(font: EffectTextFont, px: number): string {
  const css = font?.css ?? "serif"
  const words = css.split(/\s+/)
  let k = 0
  while (k < words.length - 1 && /^(normal|italic|oblique|small-caps|bold|bolder|lighter|\d{3})$/i.test(words[k])) k++
  return [...words.slice(0, k), `${px}px`, words.slice(k).join(" ")].join(" ").trim()
}

type Ctx2D = CanvasRenderingContext2D | OffscreenCanvasRenderingContext2D

function canvas2d(w: number, h: number): { canvas: HTMLCanvasElement | OffscreenCanvas; ctx: Ctx2D } | null {
  if (typeof OffscreenCanvas !== "undefined") {
    const canvas = new OffscreenCanvas(w, h)
    const ctx = canvas.getContext("2d")
    if (ctx) return { canvas, ctx }
  }
  if (typeof document === "undefined") return null
  const canvas = document.createElement("canvas")
  canvas.width = w
  canvas.height = h
  const ctx = canvas.getContext("2d")
  return ctx ? { canvas, ctx } : null
}

export type EffectTextSheet = {
  source: HTMLCanvasElement | OffscreenCanvas
  width: number
  height: number
  /** Where each string sits: u0, vTop, u1, vBottom, in 0..1. */
  rects: Map<string, LyricRect>
}

/**
 * Set `strings` into one sheet, a row each. Every row is the font's full box
 * (its ascent and descent, not the string's own ink), so two strings in one
 * face have one height and their aspects compare. A string too wide for the
 * sheet is set smaller in its own row rather than cut. Null without a canvas.
 */
export function rasterEffectText(strings: readonly string[], font: EffectTextFont): EffectTextSheet | null {
  const list = [...new Set(strings.filter((s) => s.length > 0))]
  const probe = canvas2d(1, 1)
  if (!probe) return null
  const css = effectTextFont(font, EFFECT_TEXT_PX)
  const tracking = (font?.tracking ?? 0) * EFFECT_TEXT_PX
  const setUp = (ctx: Ctx2D, px: number) => {
    ctx.font = px === EFFECT_TEXT_PX ? css : effectTextFont(font, px)
    if ("letterSpacing" in ctx) (ctx as { letterSpacing: string }).letterSpacing = `${(tracking * px) / EFFECT_TEXT_PX}px`
    ctx.textBaseline = "alphabetic"
    ctx.textAlign = "left"
  }
  setUp(probe.ctx, EFFECT_TEXT_PX)
  const box = probe.ctx.measureText("Hg国")
  const ascent = Math.ceil(box.fontBoundingBoxAscent || EFFECT_TEXT_PX * 0.9)
  const descent = Math.ceil(box.fontBoundingBoxDescent || EFFECT_TEXT_PX * 0.25)
  const rowH = ascent + descent + PAD * 2
  const rows = list.map((s) => {
    // Tracking adds after every glyph, the last one too; trim it off the box.
    const w = probe.ctx.measureText(s).width - tracking
    const px = w + PAD * 2 > MAX_W ? Math.floor((EFFECT_TEXT_PX * (MAX_W - PAD * 2)) / w) : EFFECT_TEXT_PX
    return { s, px, w: Math.min(MAX_W, Math.ceil((w * px) / EFFECT_TEXT_PX) + PAD * 2) }
  })
  const width = Math.max(1, ...rows.map((r) => r.w))
  const height = Math.max(1, Math.min(MAX_H, rows.length * rowH))
  const sheet = canvas2d(width, height)
  if (!sheet) return null
  const { ctx } = sheet
  ctx.clearRect(0, 0, width, height)
  ctx.fillStyle = "#ffffff"
  const rects = new Map<string, LyricRect>()
  rows.forEach((r, i) => {
    const top = i * rowH
    if (top + rowH > height) return
    setUp(ctx, r.px)
    // A smaller row stays centred on the same baseline band.
    const base = top + rowH / 2 + ((ascent - descent) * r.px) / (2 * EFFECT_TEXT_PX)
    ctx.fillText(r.s, PAD, base)
    rects.set(r.s, [0, top / height, r.w / width, (top + rowH) / height])
  })
  return { source: sheet.canvas, width, height, rects }
}

/**
 * rzText_NAME(uv) and rzTextAspect_NAME() for each text dial. uv runs 0..1
 * across the string's own box, y up, like rzLyricText; the answer is glyph
 * coverage. textureSampleLevel, so it is legal after any branch.
 */
export function effectTextWgsl(names: readonly string[], real: boolean, binding: number, sampler: string): string {
  if (names.length === 0) return ""
  if (!real) {
    return names
      .map((n) => `fn rzText_${n}(uv: vec2f) -> f32 { return 0.0; }\nfn rzTextAspect_${n}() -> f32 { return 1.0; }\n`)
      .join("")
  }
  return (
    /* wgsl */ `
@group(0) @binding(${binding}) var _rzEffectText: texture_2d<f32>;

fn _rzTextAt(r: vec4f, uv: vec2f) -> f32 {
  if (r.z <= r.x || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) { return 0.0; }
  let at = vec2f(mix(r.x, r.z, uv.x), mix(r.w, r.y, uv.y));
  return textureSampleLevel(_rzEffectText, ${sampler}, at, 0.0).r;
}

fn _rzTextAspect(r: vec4f) -> f32 {
  let h = r.w - r.y;
  if (h <= 0.0) { return 1.0; }
  let dim = vec2f(textureDimensions(_rzEffectText));
  return ((r.z - r.x) * dim.x) / (h * dim.y);
}
` +
    names
      .map((n) => `fn rzText_${n}(uv: vec2f) -> f32 { return _rzTextAt(params.${n}, uv); }\nfn rzTextAspect_${n}() -> f32 { return _rzTextAspect(params.${n}); }\n`)
      .join("")
  )
}
