// M_MMDClassic — authentic MMD/MikuMikuDance fixed-pipeline material. Uses the
// PMX material data every model author actually tuned: ambient + diffuse×light
// base, toon-ramp shading (ramp carries the terminator, NOT an N·L multiply),
// sphere map (sph multiply / spa add), and Blinn-Phong specular with the PMX
// shininess. This is the universal fallback for materials no preset map or
// name heuristic covers — it renders what the author saw in MMD/PMXEditor.

import { COMMON_MATERIAL_PRELUDE_WGSL } from "./common"

export const MMD_CLASSIC_SHADER_WGSL = /* wgsl */ `

${COMMON_MATERIAL_PRELUDE_WGSL}

@fragment fn fs(input: VertexOutput) -> FSOut {
  let tex_s = textureSample(diffuseTexture, diffuseSampler, input.uv);
  // MMD alpha semantics: material alpha × texture alpha.
  let alpha = material.alpha * tex_s.a;
  if (alpha < 0.001) { discard; }

  let n = safe_normal(input.normal);
  let v = normalize(camera.viewPos - input.worldPos);
  let l = -light.lights[0].direction.xyz;
  let sun = light.lights[0].color.xyz * light.lights[0].color.w;
  let amb = light.ambientColor.xyz;
  let shadow = sampleShadow(input.worldPos, n);

  // Sphere-map UV from the view-space normal — sampled up front to keep the
  // texture reads in uniform control flow.
  let vn = normalize((camera.view * vec4f(n, 0.0)).xyz);
  let sph_uv = vec2f(vn.x * 0.5 + 0.5, 0.5 - vn.y * 0.5);
  let sphere_rgb = textureSample(sphereTexture, diffuseSampler, sph_uv).rgb;

  // Toon ramp: v runs light (0) → shadow (1); the ramp texture carries the
  // terminator shape and shadow tint. Self-shadow pushes toward the shadow
  // end by attenuating N·L. Sampled up front for the same uniformity reason.
  let ndl = dot(n, l) * shadow;
  let toon_v = clamp(0.5 - 0.5 * ndl, 0.0, 1.0);
  let toon_rgb = textureSample(toonTexture, diffuseSampler, vec2f(0.5, toon_v)).rgb;

  // MMD base: saturate(ambient + diffuse × light color), modulated by texture.
  // The engine's ambient light scales the material-ambient floor so scene
  // lighting still has authority over overall exposure.
  let base = clamp(material.ambient * max(amb, vec3f(0.35)) + material.diffuseColor * sun, vec3f(0.0), vec3f(1.0));
  var color = base * tex_s.rgb * toon_rgb;

  // Sphere map: 1 = sph (multiply), 2 = spa (add). Mode 3 (sub-texture) and
  // 0 fall through unchanged.
  if (material.sphereMode == 1.0) {
    color *= sphere_rgb;
  } else if (material.sphereMode == 2.0) {
    color += sphere_rgb;
  }

  // Blinn-Phong specular with PMX shininess. shininess 0 disables (many
  // rigs author 0 to mean "no highlight").
  if (material.shininess > 0.0) {
    let h = normalize(l + v);
    let spec = pow(max(dot(n, h), 0.0), max(material.shininess, 1.0));
    color += material.specular * sun * (spec * shadow);
  }

  var out: FSOut;
  out.color = vec4f(color, alpha);
  out.mask = vec4f(1.0, 1.0, 0.0, out.color.a);
  return out;
}

`
