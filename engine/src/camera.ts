import { Mat4, Quat, Vec3 } from "./math"
import type { CameraPose } from "./camera-animation"

/** Far cap / zoom limit; large enough for wide shots without clipping distant ground */
const FAR_CAP = 8000
const FAR_MIN = 200
/**
 * Near-plane floor and cap, in two sets, because what a near plane costs depends
 * entirely on the depth buffer underneath it.
 *
 * The single set that used to live here was written against a projection that
 * silently doubled it (see Mat4.perspectiveInto), so the real clip was at 1.0 and
 * 10 rather than the 0.5 and 5 stated. UNORM keeps exactly that behaviour, now
 * said out loud: identical precision, identical z-fighting margin on coplanar
 * cloth, no change to any scene that renders correctly today.
 *
 * REVERSED gets the floor the content actually wants. Float depth read backwards
 * holds roughly constant RELATIVE precision across the range, so pulling the near
 * plane in by twenty times costs almost nothing — and a 4 mm near plane is closer
 * than any camera VMD will ever push.
 */
const NEAR_MIN_UNORM = 1.0
const NEAR_MAX_UNORM = 10
const NEAR_MIN_REVERSED = 0.05
const NEAR_MAX_REVERSED = 5

export class Camera {
  alpha: number
  beta: number
  radius: number
  target: Vec3
  fov: number
  aspect: number = 1
  /** Depth precision scales with the near plane: at 0.05 the 24-bit buffer wasted
   *  nearly all of it within a hand's reach and coplanar cloth/body layers z-fought
   *  (white crack flashes, worse the further the camera). 0.5 MMD units ≈ 4 cm —
   *  closer than any real framing — and buys 10× precision at character range. */
  near: number = NEAR_MIN_UNORM
  far: number = FAR_CAP
  /** Set once by the engine, from whether it got a float depth format. Decides
   *  both the near-plane floor and which way round the projection maps z. */
  reversedZ = false

  // Input state
  private canvas: HTMLCanvasElement | null = null
  private inputLocked: boolean = false
  private isDragging: boolean = false
  private mouseButton: number | null = null // Track which mouse button is pressed (0 = left, 2 = right)
  private lastMousePos = { x: 0, y: 0 }
  private lastTouchPos = { x: 0, y: 0 }
  private touchIdentifier: number | null = null
  private isPinching: boolean = false
  private lastPinchDistance: number = 0
  private lastPinchMidpoint = { x: 0, y: 0 } // Midpoint of two fingers for panning
  private initialPinchDistance: number = 0 // Initial distance when pinch started

  // Camera settings
  angularSensitivity: number = 0.005
  panSensitivity: number = 0.0002 // Sensitivity for right-click panning
  wheelPrecision: number = 0.01
  pinchPrecision: number = 0.05
  minZ: number = 0.05
  maxZ: number = FAR_CAP
  lowerBetaLimit: number = 0.001
  upperBetaLimit: number = Math.PI - 0.001

  // Reused each frame so getViewMatrix/getProjectionMatrix don't allocate a Mat4 per call.
  private _viewMat = new Mat4(new Float32Array(16))
  private _projMat = new Mat4(new Float32Array(16))

  // ── VMD camera drive ──
  // When vmdDriven, getViewMatrix builds the shot from a sampled MMD camera pose (target /
  // rotation-euler / distance / fov) instead of the orbit params. Toggled by the engine.
  vmdDriven: boolean = false
  private _vmdTarget = new Vec3(0, 0, 0)
  private _vmdRotation = new Vec3(0, 0, 0) // euler radians
  private _vmdDistance = -45
  private _savedFov = Math.PI / 4
  private _quatScratch = new Float32Array(16)

  constructor(alpha: number, beta: number, radius: number, target: Vec3, fov: number = Math.PI / 4) {
    this.alpha = alpha
    this.beta = beta
    this.radius = radius
    this.target = target
    this.fov = fov
    this.updateFarFromRadius()

    // Bind event handlers
    this.onMouseDown = this.onMouseDown.bind(this)
    this.onMouseMove = this.onMouseMove.bind(this)
    this.onMouseUp = this.onMouseUp.bind(this)
    this.onWheel = this.onWheel.bind(this)
    this.onContextMenu = this.onContextMenu.bind(this)
    this.onTouchStart = this.onTouchStart.bind(this)
    this.onTouchMove = this.onTouchMove.bind(this)
    this.onTouchEnd = this.onTouchEnd.bind(this)
  }

  getPosition(): Vec3 {
    // Convert spherical coordinates to Cartesian position
    const x = this.target.x + this.radius * Math.sin(this.beta) * Math.sin(this.alpha)
    const y = this.target.y + this.radius * Math.cos(this.beta)
    const z = this.target.z + this.radius * Math.sin(this.beta) * Math.cos(this.alpha)
    return new Vec3(x, y, z)
  }

  /** Enter/leave VMD-camera drive. Backs up the orbit fov on enter, restores it on leave
   *  (the VMD camera animates fov, so orbit's value would otherwise be clobbered). */
  setVmdDriven(enabled: boolean): void {
    if (enabled === this.vmdDriven) return
    if (enabled) this._savedFov = this.fov
    else this.fov = this._savedFov
    this.vmdDriven = enabled
  }

  /** Feed the next sampled MMD camera pose (engine calls this each frame while driving). */
  /**
   * The shot as MMD states one: a point, an orientation about it, how far back,
   * and how wide. The five channels an MMD camera keyframe carries.
   *
   * ONE SHAPE FOR BOTH MODES, which is the point of it existing. A VMD-driven
   * camera stores exactly this; an orbiting one holds alpha/beta/radius/target
   * instead, which is the same statement with no roll. A caller that wants to
   * know where the shot is should not have to ask which of the two is driving,
   * and should certainly not have to take a view matrix apart to find out —
   * both modes have the answer already, unfactored.
   */
  getPose(): CameraPose {
    if (this.vmdDriven) {
      return {
        target: new Vec3(this._vmdTarget.x, this._vmdTarget.y, this._vmdTarget.z),
        rotation: new Vec3(this._vmdRotation.x, this._vmdRotation.y, this._vmdRotation.z),
        distance: this._vmdDistance,
        fov: this.fov,
      }
    }
    // Orbit. alpha and beta are the same yaw and pitch the VMD euler carries,
    // in the same sense the shot is built from — see vmdEye. Distance is
    // NEGATIVE to match: in a VMD the camera sits behind its target.
    return {
      target: new Vec3(this.target.x, this.target.y, this.target.z),
      rotation: new Vec3(this.beta - Math.PI / 2, -this.alpha, 0),
      distance: -this.radius,
      fov: this.fov,
    }
  }

  setVmdPose(pose: CameraPose): void {
    this._vmdTarget.set(pose.target)
    this._vmdRotation.set(pose.rotation)
    this._vmdDistance = pose.distance
    this.fov = pose.fov // drives the projection
  }

  // MMD camera: look at `target` from `distance` back, oriented by the euler rotation.
  // forward = q·(0,0,1) (LH), up = q·(0,1,0) — read as columns 2 and 1 of the rot matrix.
  // eye = target + forward·distance (distance is negative in VMD, so eye sits behind).
  // Euler is NEGATED on all three axes: babylon-mmd builds the shot with
  // RotationYawPitchRoll(-ry, -rx, -rz), and our fromEuler(rx,ry,rz) equals
  // RotationYawPitchRoll(ry,rx,rz) — so the MMD-authentic rotation is fromEuler(-r).
  //
  // Leaves the rotation matrix in `_quatScratch`; getViewMatrix reads its up vector.
  private vmdEye(): Vec3 {
    const r = this._vmdRotation
    const q = Quat.fromEuler(-r.x, -r.y, -r.z)
    Mat4.fromQuatInto(q.x, q.y, q.z, q.w, this._quatScratch, 0)
    const s = this._quatScratch
    const t = this._vmdTarget
    const d = this._vmdDistance
    return new Vec3(t.x + s[8] * d, t.y + s[9] * d, t.z + s[10] * d)
  }

  /**
   * Where the shot is actually taken from: the VMD eye while a camera VMD drives, the
   * orbit position otherwise.
   *
   * `getPosition()` deliberately stays orbit-only — pan, zoom and framing reason in the
   * spherical coordinates a camera VMD never touches. Anything asking "where is the
   * viewer" wants THIS. Feeding the orbit position to the shader's `camera.viewPos`
   * while rendering from the VMD camera left every view-dependent term — specular, rim,
   * fresnel, sphere maps, the eye visibility gate — evaluated from a camera that was not
   * the one taking the picture, with the error depending on wherever the orbit happened
   * to be left.
   */
  getEyePosition(): Vec3 {
    return this.vmdDriven ? this.vmdEye() : this.getPosition()
  }

  getViewMatrix(): Mat4 {
    if (this.vmdDriven) {
      const eye = this.vmdEye()
      const s = this._quatScratch
      // View = Rᵀ · T(−eye), built straight from the quaternion basis instead of
      // lookAt(eye, target): same matrix for a normal d<0 shot, but position-baked
      // tracks store distance = 0 on every frame (eye == target), which degenerates
      // lookAt's normalize to an all-zero basis. The orientation never depended on
      // the eye→target line anyway — it IS the euler rotation.
      const o = this._viewMat.values
      o[0] = s[0]; o[1] = s[4]; o[2] = s[8]; o[3] = 0
      o[4] = s[1]; o[5] = s[5]; o[6] = s[9]; o[7] = 0
      o[8] = s[2]; o[9] = s[6]; o[10] = s[10]; o[11] = 0
      o[12] = -(s[0] * eye.x + s[1] * eye.y + s[2] * eye.z)
      o[13] = -(s[4] * eye.x + s[5] * eye.y + s[6] * eye.z)
      o[14] = -(s[8] * eye.x + s[9] * eye.y + s[10] * eye.z)
      o[15] = 1
      return this._viewMat
    }
    const eye = this.getPosition()
    const t = this.target
    Mat4.lookAtInto(this._viewMat.values, eye.x, eye.y, eye.z, t.x, t.y, t.z, 0, 1, 0)
    return this._viewMat
  }

  // Get camera's right and up vectors for panning
  // Uses a more robust calculation similar to BabylonJS
  private getCameraVectors(): { right: Vec3; up: Vec3 } {
    const eye = this.getPosition()
    const forward = this.target.subtract(eye)
    const forwardLen = forward.length()

    // Handle edge case where camera is at target
    if (forwardLen < 0.0001) {
      return { right: new Vec3(1, 0, 0), up: new Vec3(0, 1, 0) }
    }

    const forwardNorm = forward.scale(1 / forwardLen)
    const worldUp = new Vec3(0, 1, 0)

    // Calculate right vector: right = worldUp × forward
    // Use a more stable calculation that handles parallel vectors
    let right = worldUp.cross(forwardNorm)
    const rightLen = right.length()

    // If forward is parallel to worldUp, use a fallback
    if (rightLen < 0.0001) {
      // Camera is looking straight up or down, use X-axis as right
      right = new Vec3(1, 0, 0)
    } else {
      right = right.scale(1 / rightLen)
    }

    // Calculate camera up vector: up = forward × right (ensures orthogonality)
    let up = forwardNorm.cross(right)
    const upLen = up.length()

    if (upLen < 0.0001) {
      // Fallback to world up
      up = new Vec3(0, 1, 0)
    } else {
      up = up.scale(1 / upLen)
    }

    return { right, up }
  }

  // Pan the camera target based on mouse movement
  // Uses screen-space to world-space translation similar to BabylonJS
  private panCamera(deltaX: number, deltaY: number) {
    const { right, up } = this.getCameraVectors()

    // Calculate pan distance based on camera distance
    // The pan amount is proportional to the camera distance (radius) for consistent feel
    // This makes panning feel natural at all zoom levels
    const panDistance = this.radius * this.panSensitivity

    // Horizontal movement: drag right pans left (opposite direction)
    // Vertical movement: drag up pans up (positive up vector)
    const panRight = right.scale(-deltaX * panDistance)
    const panUp = up.scale(deltaY * panDistance)

    // Update target position smoothly
    this.target = this.target.add(panRight).add(panUp)
  }

  /** Far plane grows with zoom-out so big floors / distant geometry stay visible */
  private updateFarFromRadius(): void {
    const margin = 600
    this.far = Math.min(FAR_CAP, Math.max(FAR_MIN, this.radius * 12 + margin))
  }

  /**
   * Near plane scales with how far out you are framing.
   *
   * A 24-bit non-reversed depth buffer spends most of its precision just past
   * the near plane, so what survives at distance is governed by the far/near
   * ratio. 0.5 was chosen for character framing and fixed coplanar cloth there
   * (see `near` above) — but a stage puts the floor 5–20× further out, where the
   * same ratio leaves neighbouring surfaces sharing a depth value. They then win
   * and lose per pixel as the camera turns, which reads as flickering bands and
   * cracks across the floor.
   *
   * Tying near to the framing keeps character shots byte-identical (radius ~26
   * still lands on the 0.5 floor) and buys back an order of magnitude once you
   * pull out to see a whole stage. Capped so it can never clip something the
   * user is deliberately close to.
   */
  private updateNearFromRadius(): void {
    const lo = this.reversedZ ? NEAR_MIN_REVERSED : NEAR_MIN_UNORM
    const hi = this.reversedZ ? NEAR_MAX_REVERSED : NEAR_MAX_UNORM
    this.near = Math.min(hi, Math.max(lo, this.radius / 50))
  }

  getProjectionMatrix(): Mat4 {
    this.updateFarFromRadius()
    this.updateNearFromRadius()
    // Reversed-Z is the same matrix with near and far handed over the other way
    // round — near → 1, far → 0. One projection function, and the reversal is a
    // property of the depth buffer rather than a second matrix to keep in step.
    if (this.reversedZ) Mat4.perspectiveInto(this._projMat.values, this.fov, this.aspect, this.far, this.near)
    else Mat4.perspectiveInto(this._projMat.values, this.fov, this.aspect, this.near, this.far)
    return this._projMat
  }

  attachControl(canvas: HTMLCanvasElement) {
    this.canvas = canvas

    // Attach mouse event listeners
    // mousedown on canvas, but move/up on window so dragging works everywhere
    this.canvas.addEventListener("mousedown", this.onMouseDown)
    window.addEventListener("mousemove", this.onMouseMove)
    window.addEventListener("mouseup", this.onMouseUp)
    this.canvas.addEventListener("wheel", this.onWheel, { passive: false })
    this.canvas.addEventListener("contextmenu", this.onContextMenu)

    // Attach touch event listeners for mobile
    this.canvas.addEventListener("touchstart", this.onTouchStart, { passive: false })
    window.addEventListener("touchmove", this.onTouchMove, { passive: false })
    window.addEventListener("touchend", this.onTouchEnd)
  }

  detachControl() {
    if (!this.canvas) return

    // Remove mouse event listeners
    this.canvas.removeEventListener("mousedown", this.onMouseDown)
    window.removeEventListener("mousemove", this.onMouseMove)
    window.removeEventListener("mouseup", this.onMouseUp)
    this.canvas.removeEventListener("wheel", this.onWheel)
    this.canvas.removeEventListener("contextmenu", this.onContextMenu)

    // Remove touch event listeners
    this.canvas.removeEventListener("touchstart", this.onTouchStart)
    window.removeEventListener("touchmove", this.onTouchMove)
    window.removeEventListener("touchend", this.onTouchEnd)

    this.canvas = null
  }

  setInputLocked(locked: boolean) {
    this.inputLocked = locked
    if (locked) {
      this.isDragging = false
      this.isPinching = false
      this.touchIdentifier = null
    }
  }

  private onMouseDown(e: MouseEvent) {
    if (this.inputLocked) return
    this.isDragging = true
    this.mouseButton = e.button
    this.lastMousePos = { x: e.clientX, y: e.clientY }
  }

  private onMouseMove(e: MouseEvent) {
    if (this.inputLocked || this.vmdDriven) return // VMD owns the camera; orbit/pan is inert
    if (!this.isDragging) return

    const deltaX = e.clientX - this.lastMousePos.x
    const deltaY = e.clientY - this.lastMousePos.y

    if (this.mouseButton === 2) {
      // Right-click: pan the camera target
      this.panCamera(deltaX, deltaY)
    } else {
      // Left-click (or default): rotate the camera
      this.alpha += deltaX * this.angularSensitivity
      this.beta -= deltaY * this.angularSensitivity

      // Clamp beta to prevent flipping
      this.beta = Math.max(this.lowerBetaLimit, Math.min(this.upperBetaLimit, this.beta))
    }

    this.lastMousePos = { x: e.clientX, y: e.clientY }
  }

  private onMouseUp() {
    this.isDragging = false
    this.mouseButton = null
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault()
    if (this.vmdDriven) return // VMD owns the camera; zoom is inert

    // Update camera radius (zoom)
    this.radius += e.deltaY * this.wheelPrecision

    // Clamp radius to reasonable bounds
    this.radius = Math.max(this.minZ, Math.min(this.maxZ, this.radius))
    this.updateFarFromRadius()
  }

  private onContextMenu(e: Event) {
    e.preventDefault()
  }

  private onTouchStart(e: TouchEvent) {
    if (this.inputLocked) return
    e.preventDefault()

    if (e.touches.length === 1) {
      // Single touch - rotation
      const touch = e.touches[0]
      this.isDragging = true
      this.isPinching = false
      this.touchIdentifier = touch.identifier
      this.lastTouchPos = { x: touch.clientX, y: touch.clientY }
    } else if (e.touches.length === 2) {
      // Two touches - can be pinch zoom or pan
      this.isDragging = false
      this.isPinching = true
      const touch1 = e.touches[0]
      const touch2 = e.touches[1]
      const dx = touch2.clientX - touch1.clientX
      const dy = touch2.clientY - touch1.clientY
      this.lastPinchDistance = Math.sqrt(dx * dx + dy * dy)
      this.initialPinchDistance = this.lastPinchDistance

      // Calculate initial midpoint for panning
      this.lastPinchMidpoint = {
        x: (touch1.clientX + touch2.clientX) / 2,
        y: (touch1.clientY + touch2.clientY) / 2,
      }
    }
  }

  private onTouchMove(e: TouchEvent) {
    if (this.inputLocked) return
    // This listener lives on window (so drags keep tracking off-canvas), but the
    // camera only OWNS a gesture that started on the canvas (touchstart sets
    // isDragging/isPinching there). preventDefault-ing unconditionally cancelled
    // native touch scrolling in every UI panel layered above the canvas.
    if (!this.isDragging && !this.isPinching) return
    e.preventDefault()
    if (this.vmdDriven) return // VMD owns the camera; pinch/pan/rotate is inert

    if (this.isPinching && e.touches.length === 2) {
      // Two-finger gesture: can be pinch zoom or pan
      const touch1 = e.touches[0]
      const touch2 = e.touches[1]
      const dx = touch2.clientX - touch1.clientX
      const dy = touch2.clientY - touch1.clientY
      const distance = Math.sqrt(dx * dx + dy * dy)

      // Calculate current midpoint
      const currentMidpoint = {
        x: (touch1.clientX + touch2.clientX) / 2,
        y: (touch1.clientY + touch2.clientY) / 2,
      }

      // Calculate distance change and midpoint movement
      const distanceDelta = Math.abs(distance - this.lastPinchDistance)
      const midpointDeltaX = currentMidpoint.x - this.lastPinchMidpoint.x
      const midpointDeltaY = currentMidpoint.y - this.lastPinchMidpoint.y
      const midpointDelta = Math.sqrt(midpointDeltaX * midpointDeltaX + midpointDeltaY * midpointDeltaY)

      // Determine gesture type based on relative changes
      // Calculate relative change in distance (as percentage of initial distance)
      const distanceChangeRatio = distanceDelta / Math.max(this.initialPinchDistance, 10.0)

      // Threshold: if distance changes more than 3% of initial, it's primarily a zoom gesture
      // Otherwise, if midpoint moves significantly, it's a pan gesture
      const ZOOM_THRESHOLD = 0.03
      const PAN_THRESHOLD = 2.0 // Minimum pixels of midpoint movement for pan

      const isZoomGesture = distanceChangeRatio > ZOOM_THRESHOLD
      const isPanGesture = midpointDelta > PAN_THRESHOLD && distanceChangeRatio < ZOOM_THRESHOLD * 2

      if (isZoomGesture) {
        // Primary gesture is zoom (pinch)
        const delta = this.lastPinchDistance - distance
        this.radius += delta * this.pinchPrecision

        // Clamp radius to reasonable bounds
        this.radius = Math.max(this.minZ, Math.min(this.maxZ, this.radius))
        this.updateFarFromRadius()
      }

      if (isPanGesture) {
        // Primary gesture is pan (two-finger drag)
        // Use panning similar to right-click pan
        this.panCamera(midpointDeltaX, midpointDeltaY)
      }

      // Update tracking values
      this.lastPinchDistance = distance
      this.lastPinchMidpoint = currentMidpoint
    } else if (this.isDragging && this.touchIdentifier !== null) {
      // Single-finger rotation
      // Find the touch we're tracking
      let touch: Touch | null = null
      for (let i = 0; i < e.touches.length; i++) {
        if (e.touches[i].identifier === this.touchIdentifier) {
          touch = e.touches[i]
          break
        }
      }

      if (!touch) return

      const deltaX = touch.clientX - this.lastTouchPos.x
      const deltaY = touch.clientY - this.lastTouchPos.y

      this.alpha += deltaX * this.angularSensitivity
      this.beta -= deltaY * this.angularSensitivity

      // Clamp beta to prevent flipping
      this.beta = Math.max(this.lowerBetaLimit, Math.min(this.upperBetaLimit, this.beta))

      this.lastTouchPos = { x: touch.clientX, y: touch.clientY }
    }
  }

  private onTouchEnd(e: TouchEvent) {
    if (e.touches.length === 0) {
      // All touches ended
      this.isDragging = false
      this.isPinching = false
      this.touchIdentifier = null
      this.initialPinchDistance = 0
    } else if (e.touches.length === 1 && this.isPinching) {
      // Went from 2 fingers to 1 - switch to rotation
      const touch = e.touches[0]
      this.isPinching = false
      this.isDragging = true
      this.touchIdentifier = touch.identifier
      this.lastTouchPos = { x: touch.clientX, y: touch.clientY }
      this.initialPinchDistance = 0
    } else if (this.touchIdentifier !== null) {
      // Check if our tracked touch ended
      let touchStillActive = false
      for (let i = 0; i < e.touches.length; i++) {
        if (e.touches[i].identifier === this.touchIdentifier) {
          touchStillActive = true
          break
        }
      }

      if (!touchStillActive) {
        this.isDragging = false
        this.touchIdentifier = null
      }
    }
  }
}
