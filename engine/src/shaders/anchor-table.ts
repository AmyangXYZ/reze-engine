// The scene's anchor table: one deduplicated set of bones for every effect in
// the scene, plus the per-effect alias that maps an author's slot onto it.
//
// THE PROBLEM IT SOLVES. A slot number is currently two things at once: the
// author's name for a bone, and its storage address in the cast buffer. Those
// coincide only while one effect exists. Install two and slot 0 means a
// different bone to each of them, so whichever wrote last wins and the other
// silently reads someone else's hand.
//
// The fix is to stop conflating them. Bones are allocated ONCE for the scene,
// deduplicated, and each effect is spliced a private `_rzSlot(local) -> global`
// that the accessors route through. Author source is untouched — a published
// effect keeps compiling and keeps its line numbers, which matters because line
// numbers are what an author debugs against.
//
// Three things fall out of it beyond fixing the clash:
//   · two effects wanting the same trail SHARE one ring, and trails are the
//     expensive resource — 128 samples × 4 subjects each;
//   · the cap becomes 8 distinct bones per SCENE rather than per file;
//   · it is the same indirection the skeleton data interface needs later, so
//     this is a bridge rather than a detour.

interface AnchorRequest {
  bone: string
  trail: boolean
}

/** The empty table — a scene with no effect installed asks for no bones. */
export const EMPTY_ANCHOR_TABLE: AnchorTable = { entries: [], alias: [], dropped: [] }

export interface AnchorTable {
  /** The scene's bones, deduplicated, in allocation order. Storage addresses. */
  entries: AnchorRequest[]
  /** Per effect, local slot → global slot; -1 for a request the cap refused. */
  alias: number[][]
  /** What the cap refused, so an install can say so instead of going quiet. */
  dropped: { effect: number; bone: string }[]
}

/**
 * Allocate the scene's anchors from what each effect asked for.
 *
 * Deduplicated by BONE, not by (bone, trail): a request for a trail and a
 * request for the bare position are the same bone, and one entry with the trail
 * turned on satisfies both. Keying on the pair would spend two of eight slots
 * describing one wrist.
 *
 * Order is first-come, so a single effect gets the identity alias and the whole
 * mechanism is a no-op until a second effect exists — which is what makes this
 * safe to land before setEffects does.
 */
export function buildAnchorTable(requests: AnchorRequest[][], max: number): AnchorTable {
  const entries: AnchorRequest[] = []
  const index = new Map<string, number>()
  const alias: number[][] = []
  const dropped: { effect: number; bone: string }[] = []

  for (let e = 0; e < requests.length; e++) {
    const local: number[] = []
    for (const req of requests[e]) {
      let g = index.get(req.bone)
      if (g === undefined) {
        if (entries.length >= max) {
          // Refused, and the effect still installs: an effect that loses one of
          // its anchors draws that one wrong, where refusing the install would
          // lose the whole scene's visuals over a bone.
          dropped.push({ effect: e, bone: req.bone })
          local.push(-1)
          continue
        }
        g = entries.length
        index.set(req.bone, g)
        entries.push({ bone: req.bone, trail: req.trail })
      } else if (req.trail) {
        // A later request for the same bone can only ever ADD the trail — the
        // ring is shared, and turning it on for one reader turns it on for all.
        entries[g].trail = true
      }
      local.push(g)
    }
    alias.push(local)
  }

  return { entries, alias, dropped }
}

/**
 * Ribbon index → LOCAL anchor slot, for the trail draw.
 *
 * A THIRD index space, and the one that bit. The trail pass draws one ribbon per
 * TRAILED anchor, so its instance index counts 0,1,2… over trailed anchors only
 * — while the cast buffer is addressed by DECLARATION slot. Those coincide
 * exactly when every anchor is trailed, which is true of all 14 library effects
 * and is why this stayed latent: declare `@anchor 頭` then `@anchor 左手首 trail`
 * and ribbon 0 asked for the trail of 頭, which has none, so the ribbon silently
 * did not draw.
 *
 * Feeding this into _rzSlot afterwards is what makes the chain complete:
 * ribbon → local slot → scene slot.
 */
export function ribbonSlotWgsl(localSlots: number[]): string {
  const identity = localSlots.every((s, i) => s === i)
  if (identity) {
    return `
/** Ribbon index → local anchor slot. Identity: every anchor is trailed. */
fn _rzRibbonSlot(ribbon: i32) -> i32 { return ribbon; }
`
  }
  const cases = localSlots.map((s, i) => `    case ${i}: { return ${s}; }`).join("\n")
  return `
/** Ribbon index → local anchor slot, skipping the anchors with no trail. */
fn _rzRibbonSlot(ribbon: i32) -> i32 {
  switch ribbon {
${cases}
    default: { return -1; }
  }
}
`
}

/**
 * The alias as WGSL, spliced into each effect's module.
 *
 * A switch rather than an array because a const array indexed by a runtime value
 * lowers badly on the Metal backend — the same reason the filmic curve became a
 * texture. With one effect this compiles to `return local`, which every backend
 * folds away.
 */
export function anchorAliasWgsl(alias: number[]): string {
  // NO @anchor AT ALL. Every slot is unmapped, and rzAnchor() must report
  // .valid false for all of them.
  //
  // This used to fall through to the identity branch below — `[].every()` is
  // true — so an effect that declared no bone and called rzAnchor(c, 0) got
  // slot 0 of the SCENE's table: whichever bone the first effect that did
  // declare one happened to name. Alone in a scene it read a zeroed buffer and
  // looked correct; add a second effect and it silently anchored to that
  // effect's wrist. Cross-effect bleed, and invisible from the file you were
  // reading.
  if (alias.length === 0) {
    return `
/** No @anchor in this effect: every slot is unmapped, so rzAnchor() is invalid
 *  rather than reading whichever bone another effect declared first. */
fn _rzSlot(local: i32) -> i32 { return -1; }
`
  }
  const identity = alias.every((g, i) => g === i)
  if (identity) {
    return `
/** Local slot → scene slot. Identity here: this effect owns the table. */
fn _rzSlot(local: i32) -> i32 { return local; }
`
  }
  const cases = alias.map((g, i) => `    case ${i}: { return ${g}; }`).join("\n")
  return `
/** Local slot → scene slot, from the deduplicated table this effect shares. */
fn _rzSlot(local: i32) -> i32 {
  switch local {
${cases}
    default: { return -1; }
  }
}
`
}
