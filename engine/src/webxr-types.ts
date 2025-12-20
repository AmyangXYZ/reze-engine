export {}

declare global {
  // Base WebXR types (minimal). Some TS lib.dom builds omit these.
  interface XRViewport {
    readonly x: number
    readonly y: number
    readonly width: number
    readonly height: number
  }

  type XRReferenceSpaceType = string

  interface XRReferenceSpace {}

  interface XRViewerPose {
    readonly views: XRView[]
  }

  interface XRRigidTransform {
    readonly inverse: XRRigidTransform
    readonly matrix: Float32Array
    readonly position: DOMPointReadOnly
  }

  interface XRView {
    readonly projectionMatrix: Float32Array
    readonly transform: XRRigidTransform
  }

  type XRFrameRequestCallback = (time: number, frame: XRFrame) => void

  interface XRFrame {
    getViewerPose(referenceSpace: XRReferenceSpace): XRViewerPose | null
  }

  interface XRSession extends EventTarget {
    requestReferenceSpace(type: XRReferenceSpaceType): Promise<XRReferenceSpace>
    requestAnimationFrame(callback: XRFrameRequestCallback): number
    cancelAnimationFrame(handle: number): void
    end(): Promise<void>
  }

  interface XRSystem {
    isSessionSupported(mode: string): Promise<boolean>
    requestSession(mode: string, options?: unknown): Promise<XRSession>
  }

  interface Navigator {
    readonly xr?: XRSystem
  }

  // WebXR WebGPU draft types (minimal; runtime feature-detected).
  interface XRProjectionLayer {
    readonly textureWidth: number
    readonly textureHeight: number
  }

  interface XRWebGPUSubImage {
    readonly viewport: XRViewport
    readonly colorTexture: GPUTexture
    readonly depthStencilTexture?: GPUTexture
    readonly imageIndex?: number
  }

  interface XRGPUBinding {
    createProjectionLayer(options?: unknown): XRProjectionLayer
    getViewSubImage(layer: XRProjectionLayer, view: XRView): XRWebGPUSubImage
  }

  // eslint-disable-next-line no-var
  var XRGPUBinding:
    | {
        prototype: XRGPUBinding
        new (session: XRSession, device: GPUDevice): XRGPUBinding
      }
    | undefined
}
