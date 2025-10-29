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

    // Attach event listeners
    // mousedown on canvas, but move/up on window so dragging works everywhere
    this.canvas.addEventListener("mousedown", this.onMouseDown)
    window.addEventListener("mousemove", this.onMouseMove)
    window.addEventListener("mouseup", this.onMouseUp)
    this.canvas.addEventListener("wheel", this.onWheel)
  }

  detachControl() {
    if (!this.canvas) return

    // Remove event listeners
    this.canvas.removeEventListener("mousedown", this.onMouseDown)
    window.removeEventListener("mousemove", this.onMouseMove)
    window.removeEventListener("mouseup", this.onMouseUp)
    this.canvas.removeEventListener("wheel", this.onWheel)

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
}
