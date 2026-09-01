import { castDistanceStub } from "./cast-distance"
import { audioApi } from "../audio-api"
import { lyricsApi } from "../lyrics-api"
import { anchorAliasWgsl } from "../anchor-table"
import { midiApi } from "../midi-api"
import { EFFECT_SCENE_API } from "./composite"
import { type CastLayout } from "./particles"
import { clockApi, trailSlotsApi, viewportApi } from "./hosted-api"
import { idApi } from "../id-api"

// A persistent grid an effect can step and read: the one thing an effect could
// not have before, which is MEMORY.
//
// Every other mount is a pure function of position and time. A field shader
// recomputes its noise from scratch every frame; a trail reads history the
// engine happens to keep. Neither can build on what it produced last frame, and
// a whole class of effect is nothing but that:
//
//   Fog that a dancer walks through does not "look displaced near her" — it IS
//   displaced, and stays displaced, and creeps back over seconds. Water does not
//   look rippled where a foot landed; a ripple leaves and propagates and
//   reflects off the far edge long after the foot is gone. Both are the same
//   shape of thing — a grid, stepped by its own previous value.
//
// It is also the only way to get the LOOK of a fluid, not merely its motion.
// Advection compounds: each frame's field is the last one pushed sideways, and a
// few hundred frames of that stretches and folds a smooth field into filaments
// and vortex sheets finer than the grid storing them. No noise function
// produces that, at any octave count, because the structure is not a function of
// position at all — it is a function of history.
//
// The engine owns the ping-pong, the dispatch and the sampling. The author owns
// the KERNEL, which is the part that differs: advection for smoke, the wave
// equation for water, a decaying stamp for footprints. That split is the same
// one the particle mounts make, and for the same reason — the loop is identical
// in every such effect and the physics never is.

// A megatexel of rgba16float is 8MB, and two of those is the whole cost. What
// actually bounds this is the STEP: one invocation per texel per frame, each
// taking a dozen samples, so doubling the side quadruples the work.
export const GRID_MAX = 1024
/** rgba16float, ping-ponged. Two of these is the whole memory cost. */
export const SIM_FORMAT: GPUTextureFormat = "rgba16float"

/** Whether this effect drives a grid at all. */
export function gridEntryPoint(wgsl: string): boolean {
  return /\bfn\s+gridStep\s*\(/.test(wgsl)
}

/**
 * `rzGrid(uv)`, for a shader that READS the grid rather than steps it.
 *
 * Always compiled in, even for an effect with no grid — it then samples a 1×1
 * of zeroes. An accessor that exists only sometimes is one an author has to
 * guard, and a missing function is a compile error rather than a blank result.
 */
export function gridReadApi(group: number, tex: number, samp: number, size: number): string {
  return /* wgsl */ `
const RZ_GRID_SIZE: f32 = ${size > 0 ? size : 1}.0;
@group(${group}) @binding(${tex}) var _rzGridTex: texture_2d<f32>;
@group(${group}) @binding(${samp}) var _rzGridSamp: sampler;

/**
 * The grid, bilinearly sampled. uv is 0..1 across it, and what that MEANS in
 * the world is the effect's own business — the engine deliberately does not
 * impose a mapping, because a fog laid over a stage and a ripple field around a
 * character want different ones and both are two lines of arithmetic.
 */
fn rzGrid(uv: vec2f) -> vec4f {
  return textureSampleLevel(_rzGridTex, _rzGridSamp, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0);
}
fn rzGridSize() -> f32 { return RZ_GRID_SIZE; }
fn rzGridTexel() -> f32 { return 1.0 / RZ_GRID_SIZE; }

// ── The step-only half, defined here too and never called here ──
//
// One effect file is spliced into EVERY module it has a mount in, so a file
// with a grid compiles its gridStep inside the field and particle shaders as
// well — where it is dead code that nothing calls, but still code that has to
// resolve. Leaving these out is a compile error on a function the author was
// right to write. rzGridPrev reads the current grid rather than a previous one,
// which is the honest answer outside the step: there is no previous frame here.
fn rzGridPrev(uv: vec2f) -> vec4f { return rzGrid(uv); }
fn rzGridFrame() -> i32 { return 1; }
`
}

const SIM_UNIFORMS = /* wgsl */ `
struct SimU {
  time: f32,
  dt: f32,
  size: f32,
  frame: f32,
}
`

/**
 * The step shader: one invocation per texel, once per frame.
 *
 * The previous grid is bound as a SAMPLED texture and the next as a storage
 * texture, which is what makes the two different resources and the whole thing
 * legal — a shader cannot read and write one texture coherently, and the
 * ping-pong is not an optimisation but the only correct way to do this.
 *
 * Sampled rather than only fetched, because advection needs to read BETWEEN
 * texels: a semi-Lagrangian step asks "what was at the place this parcel came
 * from", and that place is almost never a texel centre. Point-sampling it is
 * what turns smoke into a staircase.
 */
export function buildSimShader(wgsl: string, size: number, cast: CastLayout): string {
  return (
    SIM_UNIFORMS +
    /* wgsl */ `
@group(0) @binding(0) var<uniform> su: SimU;
@group(0) @binding(1) var _rzGridPrevTex: texture_2d<f32>;
@group(0) @binding(2) var _rzGridSamp: sampler;
@group(0) @binding(3) var _rzGridOut: texture_storage_2d<${SIM_FORMAT}, write>;
@group(0) @binding(4) var<storage, read> _rzCast: array<vec4f>;
// The same view uniform the composite reads, so the scene API below is the real
// thing here rather than a stub — a kernel can ask where a bone is, and a
// foreground spliced into this module resolves its camera calls.
@group(0) @binding(6) var<uniform> viewU: array<vec4<f32>, 15>;
${clockApi("su.time", "su.dt")}${viewportApi("viewU[6].w")}${trailSlotsApi(cast.trailCount)}
/** Texels per side. */
fn rzGridSize() -> f32 { return su.size; }
/** One texel, in uv — the step a neighbour lookup takes. */
fn rzGridTexel() -> f32 { return 1.0 / su.size; }
/**
 * Steps since this effect was installed. Zero on the very first one, which is
 * the only chance to seed a grid: everything after it builds on what is there.
 */
fn rzGridFrame() -> i32 { return i32(su.frame); }

/** The grid as it was LAST frame, bilinear. The only input a kernel really has. */
fn rzGridPrev(uv: vec2f) -> vec4f {
  return textureSampleLevel(_rzGridPrevTex, _rzGridSamp, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0);
}
` +
    EFFECT_SCENE_API +
    // The alias _rzSlot, which the scene API's accessors route through. Every
    // module embedding EFFECT_SCENE_API must splice this; the sim module was
    // the one that did not, and every gridStep effect failed to install with
    // "unresolved call target" from the day the identity fallback was removed.
    // Found by tools/validate-wgsl.mjs on its first run.
    anchorAliasWgsl(cast.alias) +
    audioApi(0, 5) +
    midiApi(0, 7) +
    lyricsApi(0, 8) +
    // rzGrid itself, so a kernel may read the grid it is writing — through the
    // PREVIOUS frame's texture, which is the only version of it that exists
    // while the current one is still being written.
    /* wgsl */ `
fn rzGrid(uv: vec2f) -> vec4f { return rzGridPrev(uv); }
` +
    // Stubbed: this module cannot read an attachment the scene pass writes — see
    // id-api.ts. The author's whole file compiles here, so the names must exist.
    idApi(false, 0, 0) + castDistanceStub() +
    "\n// ── user effect (setEffect) ──\n" +
    wgsl +
    /* wgsl */ `

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let n = u32(${size});
  if (gid.x >= n || gid.y >= n) { return; }
  let xy = vec2i(i32(gid.x), i32(gid.y));
  // The texel CENTRE, so a kernel that reads its own uv back through rzGridPrev
  // lands on itself rather than a quarter-texel off.
  let uv = (vec2f(f32(gid.x), f32(gid.y)) + vec2f(0.5)) / su.size;
  textureStore(_rzGridOut, xy, gridStep(uv, textureLoad(_rzGridPrevTex, xy, 0), su.dt));
}
`
  )
}
