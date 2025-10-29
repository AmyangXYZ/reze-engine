import { Mat4, Vec3 } from "./math"

export class Camera {
  alpha: number
  beta: number
  radius: number
  target: Vec3
  fov: number
  aspect: number = 1
  near: number = 0.1
  far: number = 100

  // Input state
  private canvas: HTMLCanvasElement | null = null
  private isDragging: boolean = false
  private lastMousePos = { x: 0, y: 0 }
  private lastTouchPos = { x: 0, y: 0 }
  private touchIdentifier: number | null = null

  // Camera settings
  angularSensitivity: number = 0.005
  wheelPrecision: number = 0.001
  minZ: number = 0.5
  maxZ: number = 10
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
    this.onTouchStart = this.onTouchStart.bind(this)
    this.onTouchMove = this.onTouchMove.bind(this)
    this.onTouchEnd = this.onTouchEnd.bind(this)
  }

  getViewMatrix(): Mat4 {
    // Convert spherical coordinates to Cartesian position
    const x = this.target.x + this.radius * Math.sin(this.beta) * Math.sin(this.alpha)
    const y = this.target.y + this.radius * Math.cos(this.beta)
    const z = this.target.z + this.radius * Math.sin(this.beta) * Math.cos(this.alpha)

    const eye = new Vec3(x, y, z)
    const up = new Vec3(0, 1, 0)

    return Mat4.lookAt(eye, this.target, up)
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
    this.canvas.addEventListener("wheel", this.onWheel)

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

    // Remove touch event listeners
    this.canvas.removeEventListener("touchstart", this.onTouchStart)
    window.removeEventListener("touchmove", this.onTouchMove)
    window.removeEventListener("touchend", this.onTouchEnd)

    this.canvas = null
  }

  private onMouseDown(e: MouseEvent) {
    this.isDragging = true
    this.lastMousePos = { x: e.clientX, y: e.clientY }
  }

  private onMouseMove(e: MouseEvent) {
    if (!this.isDragging) return

    const deltaX = e.clientX - this.lastMousePos.x
    const deltaY = e.clientY - this.lastMousePos.y

    this.alpha -= deltaX * this.angularSensitivity
    this.beta -= deltaY * this.angularSensitivity

    // Clamp beta to prevent flipping
    this.beta = Math.max(this.lowerBetaLimit, Math.min(this.upperBetaLimit, this.beta))

    this.lastMousePos = { x: e.clientX, y: e.clientY }
  }

  private onMouseUp() {
    this.isDragging = false
  }

  private onWheel(e: WheelEvent) {
    e.preventDefault()

    // Update camera radius (zoom)
    this.radius += e.deltaY * this.wheelPrecision

    // Clamp radius to reasonable bounds
    this.radius = Math.max(this.minZ, Math.min(this.maxZ, this.radius))
  }

  private onTouchStart(e: TouchEvent) {
    e.preventDefault()

    if (e.touches.length === 1) {
      // Single touch - rotation
      const touch = e.touches[0]
      this.isDragging = true
      this.touchIdentifier = touch.identifier
      this.lastTouchPos = { x: touch.clientX, y: touch.clientY }
    }
  }

  private onTouchMove(e: TouchEvent) {
    e.preventDefault()

    if (!this.isDragging || this.touchIdentifier === null) return

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

    this.alpha -= deltaX * this.angularSensitivity
    this.beta -= deltaY * this.angularSensitivity

    // Clamp beta to prevent flipping
    this.beta = Math.max(this.lowerBetaLimit, Math.min(this.upperBetaLimit, this.beta))

    this.lastTouchPos = { x: touch.clientX, y: touch.clientY }
  }

  private onTouchEnd(e: TouchEvent) {
    // Check if our tracked touch ended
    if (this.touchIdentifier !== null) {
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
