// Default material — Blender 3.6 Principled BSDF defaults, no NPR stack.
// Metallic=0, Specular=0.5 (F0=0.04), Roughness=0.5. Serves as the EEVEE reference
// path that every NPR material mixes against in its final stage.

import { NODES_WGSL } from "./nodes"
import { COMMON_MATERIAL_PRELUDE_WGSL } from "./common"

export const DEFAULT_SHADER_WGSL = /* wgsl */ `

${NODES_WGSL}
${COMMON_MATERIAL_PRELUDE_WGSL}

const DEFAULT_SPECULAR: f32 = 0.5;
const DEFAULT_ROUGHNESS: f32 = 0.5;

@fragment fn fs(input: VertexOutput) -> FSOut {
  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  // MMD alpha semantics: material alpha × texture alpha. Hair/lace/accessory
  // textures cut their shapes in the alpha channel — ignoring it renders each
  // card's full quad with the texture's padding color (white shimmer on dark
  // hair for models whose textures pad with white).
  let alpha = material.alpha * tex_s.a;
  if (alpha < 0.001) { discard; }

  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);

  let albedo = tex_s.rgb;

  let color = eval_principled(
    PrincipledIn(albedo, 0.0, DEFAULT_SPECULAR, DEFAULT_ROUGHNESS, 10.0, 0.0, 0.0),
    n, l, v, sun, amb, shadow
  );

  var out: FSOut;
  out.color = vec4f(color, alpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}

`
