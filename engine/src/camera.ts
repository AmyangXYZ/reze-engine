import { Mat4, Vec3 } from "./math"

const FAR = 1000

export class Camera {
  alpha: number
  beta: number
  radius: number
  target: Vec3
  fov: number
  aspect: number = 1
  near: number = 0.05
  far: number = FAR

  // Input state
  private canvas: HTMLCanvasElement | null = null
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
  minZ: number = 0.1
  maxZ: number = FAR
  lowerBetaLimit: number = 0.001
  upperBetaLimit: number = Math.PI - 0.001

  constructor(alpha: number, beta: number, radius: number, target: Vec3, fov: number = Math.PI / 4) {
    this.alpha = alpha
    this.beta = beta
    this.radius = radius
    this.target = target
    this.fov = fov

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

  getViewMatrix(): Mat4 {
    const eye = this.getPosition()
    const up = new Vec3(0, 1, 0)
    return Mat4.lookAt(eye, this.target, up)
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

  getProjectionMatrix(): Mat4 {
    return Mat4.perspective(this.fov, this.aspect, this.near, this.far)
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

  private onMouseDown(e: MouseEvent) {
    this.isDragging = true
    this.mouseButton = e.button
    this.lastMousePos = { x: e.clientX, y: e.clientY }
  }

  private onMouseMove(e: MouseEvent) {
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

    // Update camera radius (zoom)
    this.radius += e.deltaY * this.wheelPrecision

    // Clamp radius to reasonable bounds
    this.radius = Math.max(this.minZ, Math.min(this.maxZ, this.radius))
    // Expand far plane to keep scene visible when zooming out
    this.far = Math.max(FAR, this.radius * 4)
  }

  private onContextMenu(e: Event) {
    e.preventDefault()
  }

  private onTouchStart(e: TouchEvent) {
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
    e.preventDefault()

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
        // Expand far plane for pinch zoom as well
        this.far = Math.max(FAR, this.radius * 4)
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
