import { sceneFsOutWgsl } from "./scene-contract"

/**
 * The field layer, drawn INTO the scene.
 *
 * Field effects render into their own targets at their own resolution — that
 * part was always right. What was wrong was where the result landed: the
 * composite sampled it AFTER tone mapping, which is why a field effect could
 * never bloom, could never light anything, and tone mapped differently from the
 * particle and ribbon mounts drawing the same kind of glow.
 *
 * This blit puts it back in the scene. Two draws inside the scene pass — the
 * background before any geometry, the foreground after all of it — so the
 * bloom pyramid sees field effects and one tone map covers the whole frame.
 *
 * NO DEPTH, neither written nor tested. A foreground is handed the scene's
 * depth as a parameter and does its own occlusion; testing it here would
 * occlude it a second time, and a background has nothing to test against.
 */
export function fieldBlitShaderWgsl(): string {
  return /* wgsl */ `
@group(0) @binding(0) var _rzLayer: texture_2d<f32>;
@group(0) @binding(1) var _rzLayerSamp: sampler;
// (bloom, _, fullW, fullH). The size is the SCENE target's, not the layer's:
// the layer is often half resolution, and normalised uv is what makes that
// invisible.
@group(0) @binding(2) var<uniform> _rzBlitU: vec4f;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32((vi & 1u) << 2u) - 1.0;
  let y = f32((vi & 2u) << 1u) - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

${sceneFsOutWgsl()}
@fragment fn fs(@builtin(position) fragCoord: vec4f) -> FSOut {
  var out: FSOut;
  // Bilinear, because the layer is usually half resolution and the whole point
  // of that choice is that low-frequency effects upsample invisibly.
  let uv = fragCoord.xy / max(_rzBlitU.zw, vec2f(1.0));
  let c = textureSampleLevel(_rzLayer, _rzLayerSamp, uv, 0.0);
  // PREMULTIPLIED, and blitted with a premultiplied-over blend — the field
  // targets have been premultiplied since N effects had to compose into them
  // associatively. Handing this to the material blend would scale rgb by alpha
  // a second time and every layer would come out dark.
  out.color = c;
  // The aux target: bloom mask, then coverage. THIS is what finally makes
  // // @bloom mean something on a field effect — the flag has been parsed and
  // ignored for every mount but particles and ribbons, which is the kind of
  // author-surface lie the install-time guards exist to kill.
  //
  // The mask is written UNWEIGHTED and the blend weights it by src.a — the
  // convention the materials already follow (they write vec4f(1, 1, 0, alpha)).
  // Pre-weighting here would apply alpha twice.
  out.mask = vec4f(_rzBlitU.x, 1.0, 0.0, c.a);
  return out;
}
`
}
