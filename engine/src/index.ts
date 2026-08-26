export {
  Engine,
  DEFAULT_BLOOM_OPTIONS,
  DEFAULT_DEPTH_OF_FIELD_OPTIONS,
  DEFAULT_VIEW_TRANSFORM,
  DEFAULT_COLOR_GRADING,
  type ColorGradingOptions,
  type EngineStats,
  type EngineOptions,
  type BloomOptions,
  type DepthOfFieldOptions,
  type ViewTransformOptions,
  type LoadModelFromFilesOptions,
  type MaterialPreset,
  type MaterialPresetMap,
  type ModelTransform,
  type GizmoDragEvent,
  type GizmoDragCallback,
  type GizmoDragKind,
  type DissolveCycle,
  type EffectParamValue,
  type EffectResult,
  type CullDiagnostics,
  type MidiNote,
} from "./engine"
export { parsePmxFolderInput, pmxFileAtRelativePath, type PmxFolderInputResult } from "./folder-upload"
export { parseMidi } from "./midi-loader"
// Radiance .hdr, for HDRI worlds — the host fetches the file and hands the
// parsed image to setBackdropEquirect.
export { parseHDR, type HdrImage } from "./hdr"
// Lyrics timing (.lrc) — the host parses the file and hands the lines to setLyrics.
export { LYRIC_ATLAS_MAX_H, LYRIC_ATLAS_MAX_W, parseLRC, type LyricLine, type LyricRect } from "./shaders/lyrics-api"
// A material parameter over time. The sampler is exported too: a host that
// wants to draw a track, or scrub one, should read the same curve the engine
// plays rather than reimplementing it a second time.
export { sampleParamTrack, type ParamKey, type ParamValue } from "./param-track"
// The strip an effect is scheduled by, and the pure evaluator behind it —
// exported so a caller can draw a lane against the same numbers the engine
// renders from, rather than a second copy of the ramp maths.
export { effectState, type EffectWindow, type EffectState } from "./effect-schedule"
export {
  compileGraph,
  validateGraph,
  assignStyleSlots,
  type CompileOptions,
  type CompileResult,
  type StyleSlot,
} from "./graph/compile"
export type {
  ShaderGraph,
  GraphNode,
  GraphLink,
  ExposedParam,
  SocketValue,
  Diagnostic,
} from "./graph/schema"
export { NODE_REGISTRY, type NodeSpec, type SockT } from "./graph/registry"
export { RENDER_CLASSES, type RenderClass, type AlphaMode, type RenderClassInfo } from "./graph/render-class"
export type {
  StyleGroup,
  GroupImage,
  GroupImageSource,
  GroupDiagnostic,
  ApplyStyleGroupsResult,
  ApplyStyleGroupResult,
} from "./graph/style-group"
export { HAIR_GRAPH } from "./graph/presets/hair"
export { DEFAULT_GRAPH } from "./graph/presets/default"
export { CLOTH_SMOOTH_GRAPH } from "./graph/presets/cloth_smooth"
export { CLOTH_ROUGH_GRAPH } from "./graph/presets/cloth_rough"
export { METAL_GRAPH } from "./graph/presets/metal"
export { BODY_GRAPH } from "./graph/presets/body"
export { STOCKINGS_GRAPH } from "./graph/presets/stockings"
export { EYE_GRAPH } from "./graph/presets/eye"
export { FACE_GRAPH } from "./graph/presets/face"
export { UNLIT_GRAPH } from "./graph/presets/unlit"
// What an effect declares, and the source with those lines blanked. Exported
// because a host builds parameter controls from the declarations and needs the
// same answer the engine got — two parsers is how they disagree.
export {
  parseDirectives,
  stripDirectives,
  DIRECTIVE_LINE,
  DIRECTIVE_NOTE,
  type EffectDirectives,
  type EffectParamDecl,
} from "./shaders/directives"
export {
  Model,
  MATERIAL_MORPH_MULTIPLY,
  MATERIAL_MORPH_ADD,
  type ClipEventInfo,
  type RootMotionProfile,
  type Morph,
  type Morphing,
  type BoneMorphOffset,
  type MaterialMorphOffset,
  type UvMorphOffset,
} from "./model"
export { Vec3, Quat, Mat4, easeInOut, type EulerOrder } from "./math"
export type {
  AnimationClip,
  AnimationPlayOptions,
  AnimationProgress,
  BlendEntry,
  BoneKeyframe,
  IkKeyframe,
  MorphKeyframe,
  BoneInterpolation,
  ControlPoint,
} from "./animation"
export {
  LocomotionController,
  type LocomotionClips,
  type LocomotionOptions,
  type LocomotionPose,
  type StrafeClipEntry,
  type TurnClipEntry,
  type RunTurnClipEntry,
  type StopClipEntry,
} from "./locomotion"
export { AnimationStateMachine, type AnimStateDef, type AnimTransitionDef, type StateMachineOptions } from "./state-machine"
export {
  FPS,
  bezierInterpolate,
  interpolateControlPoints,
  rawInterpolationToBoneInterpolation,
} from "./animation"
export { VMDLoader, DEFAULT_CAMERA_INTERPOLATION, type CameraKeyframe, type IkFrame } from "./vmd-loader"
export { VMDWriter, type VmdTrackSelection } from "./vmd-writer"
export { PmxLoader } from "./pmx-loader"
export { CameraAnimation, type CameraPose } from "./camera-animation"
export { RezePhysics } from "./physics"
export type { WindOptions } from "./physics/world"
