import { Vec3, Quat, Mat4 } from "../lib/math"
import { Camera } from "../lib/camera"
import modelData from "../model.json"

interface Model {
  vertices: Float32Array
  indices: Uint32Array
  textures: Texture[]
  materials: Material[]
  bones: Bone[]
  skinning: Skinning
}

interface Texture {
  path: string
  name: string
}

interface Material {
  name: string
  diffuseTextureIndex: number
  vertexCount: number
}

interface Bone {
  name: string
  parentIndex: number
  bindTranslation: Vec3
  appendParentIndex?: number // index of the bone to inherit from
  appendRatio?: number // 0..1
  appendRotate?: boolean
  appendMove?: boolean
}

interface Skinning {
  joints: Uint16Array
  weights: Uint8Array
}

interface BoneState {
  localRotation: Quat
  worldMatrix: Mat4
  inverseBindMatrix: Mat4 // Transform from model space to bone's local space
}

// Basic engine with arc rotate camera
export class EngineV4 {
  private canvas: HTMLCanvasElement
  private device!: GPUDevice
  private context!: GPUCanvasContext
  private presentationFormat!: GPUTextureFormat
  private pipeline!: GPURenderPipeline
  private vertexBuffer!: GPUBuffer
  private indexBuffer!: GPUBuffer
  private renderPassDescriptor!: GPURenderPassDescriptor
  private shaderModule!: GPUShaderModule

  private model!: Model

  private camera!: Camera
  private cameraUniformBuffer!: GPUBuffer
  private cameraMatrixData = new Float32Array(36)
  private bindGroup!: GPUBindGroup
  private materialBindGroups: GPUBindGroup[] = []
  private textures: GPUTexture[] = []
  private sampler!: GPUSampler
  private depthTexture!: GPUTexture
  private animationFrameId: number | null = null

  private boneStates: BoneState[] = [] // Runtime bone states (CPU)
  private jointsBuffer!: GPUBuffer // Bone indices per vertex
  private weightsBuffer!: GPUBuffer // Bone weights per vertex
  private worldMatrixBuffer!: GPUBuffer // Current bone transforms
  private inverseBindMatrixBuffer!: GPUBuffer // Bind pose transforms
  private skinMatrixBuffer!: GPUBuffer // Final skinning matrices (GPU output)
  private skinMatrixComputePipeline!: GPUComputePipeline
  private skinMatrixComputeBindGroup!: GPUBindGroup
  private boneCountBuffer!: GPUBuffer

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  public async init() {
    this.loadModel()
    await this.initDevice()
    this.initContext()
    await this.initTexture()
    this.initShader()
    this.initVertexBuffers()
    this.initSkinning()
    this.initBoneBuffers()
    this.initPipeline()
    this.setupCamera()
    this.createBindGroups()
  }

  private loadModel() {
    const rawModel = modelData as unknown as Model

    this.model = {
      vertices: new Float32Array(rawModel.vertices),
      indices: new Uint32Array(rawModel.indices),
      textures: rawModel.textures,
      materials: rawModel.materials,
      bones: rawModel.bones.map((bone) => {
        // Parse bindTranslation from JSON (can be array or object)
        let translation: Vec3
        if (Array.isArray(bone.bindTranslation)) {
          translation = new Vec3(bone.bindTranslation[0], bone.bindTranslation[1], bone.bindTranslation[2])
        } else {
          translation = new Vec3(bone.bindTranslation.x, bone.bindTranslation.y, bone.bindTranslation.z)
        }

        return {
          ...bone,
          bindTranslation: translation,
        }
      }),
      skinning: {
        joints: new Uint16Array(rawModel.skinning.joints),
        weights: new Uint8Array(rawModel.skinning.weights),
      },
    }

    this.boneStates = this.model.bones.map(() => ({
      localRotation: new Quat(0, 0, 0, 1),
      worldMatrix: Mat4.identity(),
      inverseBindMatrix: Mat4.identity(),
    }))

    this.calculateInverseBindMatrices()
  }

  private calculateInverseBindMatrices() {
    // Calculate bind pose world matrix for each bone (identity rotation + bindTranslation)
    for (let i = 0; i < this.model.bones.length; i++) {
      const bone = this.model.bones[i]
      const state = this.boneStates[i]

      const localMatrix = Mat4.fromPositionRotation(bone.bindTranslation, new Quat(0, 0, 0, 1))

      if (bone.parentIndex === -1) {
        state.worldMatrix = localMatrix
      } else {
        const parentWorld = this.boneStates[bone.parentIndex].worldMatrix
        state.worldMatrix = parentWorld.multiply(localMatrix)
      }
    }

    // Invert each world matrix to get inverse bind matrix
    for (let i = 0; i < this.boneStates.length; i++) {
      this.boneStates[i].inverseBindMatrix = this.boneStates[i].worldMatrix.inverse()
    }

    // Reset world matrices (they'll be recalculated by evaluatePose)
    for (let i = 0; i < this.boneStates.length; i++) {
      this.boneStates[i].worldMatrix = Mat4.identity()
    }
  }

  private async initDevice() {
    const adapter = await navigator.gpu?.requestAdapter()
    const device = await adapter?.requestDevice()
    if (!device) {
      throw new Error("WebGPU is not supported in this browser.")
    }
    this.device = device
  }

  private initContext() {
    const context = this.canvas.getContext("webgpu")
    if (!context) {
      throw new Error("Failed to get WebGPU context.")
    }
    this.context = context

    this.presentationFormat = navigator.gpu.getPreferredCanvasFormat()
    this.context.configure({
      device: this.device,
      format: this.presentationFormat,
    })

    // Set canvas size with device pixel ratio (like engine.ts)
    const displayWidth = this.canvas.clientWidth
    const displayHeight = this.canvas.clientHeight
    const dpr = window.devicePixelRatio || 1
    const width = Math.floor(displayWidth * dpr)
    const height = Math.floor(displayHeight * dpr)
    this.canvas.width = width
    this.canvas.height = height

    // Create depth texture
    this.depthTexture = this.device.createTexture({
      size: [width, height],
      format: "depth24plus",
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    })

    this.renderPassDescriptor = {
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
      depthStencilAttachment: {
        view: this.depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: "clear",
        depthStoreOp: "store",
      },
    }
  }

  private initShader() {
    this.shaderModule = this.device.createShaderModule({
      code: /* wgsl */ `
        struct CameraUniforms {
          view: mat4x4f,
          projection: mat4x4f,
          viewPos: vec3f,
          _padding: f32,
        };

        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) uv: vec2<f32>,
        };

        @group(0) @binding(0) var<uniform> camera: CameraUniforms;
        @group(0) @binding(1) var<storage, read> skinMats: array<mat4x4f>;
        @group(1) @binding(0) var texture: texture_2d<f32>;
        @group(1) @binding(1) var textureSampler: sampler;

        @vertex
        fn vs(
          @location(0) position: vec3<f32>,
          @location(1) normal: vec3<f32>,
          @location(2) uv: vec2<f32>,
          @location(3) joints0: vec4<u32>,
          @location(4) weights0: vec4<f32>
        ) -> VertexOutput {
          var output: VertexOutput;
          let pos4 = vec4f(position, 1.0);
          
          // Normalize weights to ensure they sum to 1.0
          let weightSum = weights0.x + weights0.y + weights0.z + weights0.w;
          var normalizedWeights: vec4f;
          if (weightSum > 0.0001) {
            normalizedWeights = weights0 / weightSum;
          } else {
            normalizedWeights = vec4f(1.0, 0.0, 0.0, 0.0);
          }
          
          // Apply skinning: blend position by bone influences
          var skinnedPos = vec4f(0.0, 0.0, 0.0, 0.0);
          for (var i = 0u; i < 4u; i++) {
            let j = joints0[i];
            let w = normalizedWeights[i];
            let m = skinMats[j];
            skinnedPos += (m * pos4) * w;
          }
          
          let worldPos = skinnedPos.xyz;
          output.position = camera.projection * camera.view * vec4f(worldPos, 1.0);
          output.uv = uv;
          return output;
        }

        @fragment
        fn fs(input: VertexOutput) -> @location(0) vec4<f32> {
          return vec4<f32>(textureSample(texture, textureSampler, input.uv).rgb, 1.0);
        }
      `,
    })
  }

  private initVertexBuffers() {
    const vertices = Float32Array.from(this.model.vertices)
    this.vertexBuffer = this.device.createBuffer({
      label: "model vertex buffer",
      size: vertices.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.vertexBuffer, 0, vertices.buffer)

    // Create index buffer
    const indices = Uint32Array.from(this.model.indices)
    this.indexBuffer = this.device.createBuffer({
      label: "model index buffer",
      size: indices.byteLength,
      usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(this.indexBuffer, 0, indices.buffer)
  }

  private initSkinning() {
    // Create joints buffer (bone indices per vertex)
    this.jointsBuffer = this.device.createBuffer({
      label: "joints buffer",
      size: this.model.skinning.joints.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(
      this.jointsBuffer,
      0,
      this.model.skinning.joints.buffer,
      this.model.skinning.joints.byteOffset,
      this.model.skinning.joints.byteLength
    )

    // Create weights buffer (bone weights per vertex)
    this.weightsBuffer = this.device.createBuffer({
      label: "weights buffer",
      size: this.model.skinning.weights.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    })
    this.device.queue.writeBuffer(
      this.weightsBuffer,
      0,
      this.model.skinning.weights.buffer,
      this.model.skinning.weights.byteOffset,
      this.model.skinning.weights.byteLength
    )
  }

  private initPipeline() {
    this.pipeline = this.device.createRenderPipeline({
      label: "v4 pipeline",
      layout: "auto",
      vertex: {
        module: this.shaderModule,
        buffers: [
          {
            arrayStride: 8 * 4, // 8 floats * 4 bytes each = 32 bytes per vertex (position + normal + UV)
            attributes: [
              {
                shaderLocation: 0, // position
                offset: 0,
                format: "float32x3",
              },
              {
                shaderLocation: 1, // normal
                offset: 3 * 4,
                format: "float32x3",
              },
              {
                shaderLocation: 2, // UV
                offset: 6 * 4,
                format: "float32x2",
              },
            ],
          },
          {
            arrayStride: 4 * 2, // 4 uint16 values (joints)
            attributes: [
              {
                shaderLocation: 3, // joints
                offset: 0,
                format: "uint16x4",
              },
            ],
          },
          {
            arrayStride: 4, // 4 uint8 values (weights)
            attributes: [
              {
                shaderLocation: 4, // weights
                offset: 0,
                format: "unorm8x4",
              },
            ],
          },
        ],
      },
      fragment: {
        module: this.shaderModule,
        targets: [{ format: this.presentationFormat }],
      },
      primitive: { cullMode: "none" },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: "less",
        format: "depth24plus",
      },
    })
  }

  private setupCamera() {
    this.cameraUniformBuffer = this.device.createBuffer({
      label: "camera uniforms",
      size: 40 * 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })

    this.camera = new Camera(Math.PI, Math.PI / 2.5, 27, new Vec3(0, 12, 0))

    this.camera.aspect = this.canvas.width / this.canvas.height
    this.camera.attachControl(this.canvas)
  }

  private async initTexture() {
    // Collect all unique texture indices used by materials
    const textureIndices = new Set<number>()
    for (const material of this.model.materials) {
      if (material.diffuseTextureIndex >= 0 && material.diffuseTextureIndex < this.model.textures.length) {
        textureIndices.add(material.diffuseTextureIndex)
      }
    }
    const textureDir = "/models/塞尔凯特"
    // Load all textures referenced by materials
    const textureLoadPromises = Array.from(textureIndices).map(async (textureIndex) => {
      const texturePath = `${textureDir}/${this.model.textures[textureIndex].path}`
      const response = await fetch(texturePath)
      if (!response.ok) {
        throw new Error(`Failed to load texture: ${texturePath}`)
      }

      const imageBitmap = await createImageBitmap(await response.blob(), {
        premultiplyAlpha: "none",
        colorSpaceConversion: "none",
      })

      // Create texture
      const texture = this.device.createTexture({
        label: `texture: ${this.model.textures[textureIndex].name}`,
        size: [imageBitmap.width, imageBitmap.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
      })

      // Upload image data to texture
      this.device.queue.copyExternalImageToTexture({ source: imageBitmap }, { texture }, [
        imageBitmap.width,
        imageBitmap.height,
      ])

      return { textureIndex, texture }
    })

    const loadedTextures = await Promise.all(textureLoadPromises)

    // Store textures in array indexed by texture index
    this.textures = new Array(this.model.textures.length)
    for (const { textureIndex, texture } of loadedTextures) {
      this.textures[textureIndex] = texture
    }

    // Create sampler
    this.sampler = this.device.createSampler({
      label: "texture sampler",
      magFilter: "linear",
      minFilter: "linear",
      addressModeU: "repeat",
      addressModeV: "repeat",
    })
  }

  private createBindGroups() {
    this.bindGroup = this.device.createBindGroup({
      label: "camera bind group",
      layout: this.pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.cameraUniformBuffer } },
        { binding: 1, resource: { buffer: this.skinMatrixBuffer } },
      ],
    })

    // Create bind group for each material with its own texture
    for (const material of this.model.materials) {
      const textureIndex = material.diffuseTextureIndex
      const materialBindGroup = this.device.createBindGroup({
        label: `texture bind group: ${material.name}`,
        layout: this.pipeline.getBindGroupLayout(1),
        entries: [
          { binding: 0, resource: this.textures[textureIndex].createView() },
          { binding: 1, resource: this.sampler },
        ],
      })

      this.materialBindGroups.push(materialBindGroup)
    }
  }

  private updateCameraUniforms() {
    const viewMatrix = this.camera.getViewMatrix()
    const projectionMatrix = this.camera.getProjectionMatrix()
    const cameraPos = this.camera.getPosition()
    this.cameraMatrixData.set(viewMatrix.values, 0)
    this.cameraMatrixData.set(projectionMatrix.values, 16)
    this.cameraMatrixData[32] = cameraPos.x
    this.cameraMatrixData[33] = cameraPos.y
    this.cameraMatrixData[34] = cameraPos.z
    this.device.queue.writeBuffer(this.cameraUniformBuffer, 0, this.cameraMatrixData)
  }

  private evaluatePose() {
    // Calculate world matrices with proper MMD append logic
    for (let i = 0; i < this.model.bones.length; i++) {
      const bone = this.model.bones[i]
      const state = this.boneStates[i]

      // Start with bone's local rotation
      let finalRotation = state.localRotation

      // Apply append rotation (if exists) - uses append parent's LOCAL rotation
      if (bone.appendRotate && bone.appendParentIndex !== undefined && bone.appendParentIndex >= 0) {
        const appendParent = this.boneStates[bone.appendParentIndex]
        const ratio = bone.appendRatio ?? 1.0

        // Blend identity with append parent's LOCAL rotation
        const identityQuat = new Quat(0, 0, 0, 1)
        const blendedAppendRot = Quat.slerp(identityQuat, appendParent.localRotation, ratio)

        // Apply append rotation BEFORE local rotation: appendRot * localRot
        finalRotation = blendedAppendRot.multiply(state.localRotation)
      }

      // Build local matrix: bindTranslation + rotation
      const localMatrix = Mat4.fromPositionRotation(bone.bindTranslation, finalRotation)

      // Multiply with parent world matrix to get final world matrix
      if (bone.parentIndex === -1) {
        state.worldMatrix = localMatrix
      } else {
        const parentWorld = this.boneStates[bone.parentIndex].worldMatrix
        state.worldMatrix = parentWorld.multiply(localMatrix)
      }
    }

    // Upload to GPU and compute skin matrices
    if (this.worldMatrixBuffer) {
      this.uploadBoneMatricesToGPU()
      this.computeSkinMatrices()
    }
  }

  private initBoneBuffers() {
    // Create storage buffers for bone matrices
    const boneCount = this.model.bones.length
    const matrixSize = boneCount * 16 * 4 // 16 floats per matrix, 4 bytes per float

    this.worldMatrixBuffer = this.device.createBuffer({
      label: "world matrices",
      size: Math.max(256, matrixSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    this.inverseBindMatrixBuffer = this.device.createBuffer({
      label: "inverse bind matrices",
      size: Math.max(256, matrixSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    })

    this.skinMatrixBuffer = this.device.createBuffer({
      label: "skin matrices",
      size: Math.max(256, matrixSize),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX,
    })

    // Upload inverse bind matrices (static, only once)
    const inverseBindMatrices = new Float32Array(boneCount * 16)
    for (let i = 0; i < boneCount; i++) {
      inverseBindMatrices.set(this.boneStates[i].inverseBindMatrix.values, i * 16)
    }
    this.device.queue.writeBuffer(this.inverseBindMatrixBuffer, 0, inverseBindMatrices)

    // Create bone count uniform buffer
    this.boneCountBuffer = this.device.createBuffer({
      label: "bone count uniform",
      size: 32, // Minimum uniform buffer size
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    })
    const boneCountData = new Uint32Array(8) // 32 bytes total
    boneCountData[0] = boneCount
    this.device.queue.writeBuffer(this.boneCountBuffer, 0, boneCountData)

    // Create compute shader for skinning
    this.createSkinMatrixComputePipeline()

    // Calculate initial bind pose and upload to GPU
    this.evaluatePose()
  }

  private createSkinMatrixComputePipeline() {
    const computeShader = this.device.createShaderModule({
      label: "skin matrix compute",
      code: /* wgsl */ `
        struct BoneCountUniform {
          count: u32,
          _padding1: u32,
          _padding2: u32,
          _padding3: u32,
          _padding4: vec4<u32>,
        };
        
        @group(0) @binding(0) var<uniform> boneCount: BoneCountUniform;
        @group(0) @binding(1) var<storage, read> worldMatrices: array<mat4x4f>;
        @group(0) @binding(2) var<storage, read> inverseBindMatrices: array<mat4x4f>;
        @group(0) @binding(3) var<storage, read_write> skinMatrices: array<mat4x4f>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
          let boneIndex = globalId.x;
          if (boneIndex >= boneCount.count) {
            return;
          }
          let worldMat = worldMatrices[boneIndex];
          let invBindMat = inverseBindMatrices[boneIndex];
          skinMatrices[boneIndex] = worldMat * invBindMat;
        }
      `,
    })

    this.skinMatrixComputePipeline = this.device.createComputePipeline({
      label: "skin matrix compute pipeline",
      layout: "auto",
      compute: {
        module: computeShader,
      },
    })

    // Create compute bind group (reused every frame)
    this.skinMatrixComputeBindGroup = this.device.createBindGroup({
      layout: this.skinMatrixComputePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.boneCountBuffer } },
        { binding: 1, resource: { buffer: this.worldMatrixBuffer } },
        { binding: 2, resource: { buffer: this.inverseBindMatrixBuffer } },
        { binding: 3, resource: { buffer: this.skinMatrixBuffer } },
      ],
    })
  }

  private uploadBoneMatricesToGPU() {
    // Convert boneStates world matrices to flat Float32Array
    const worldMatrices = new Float32Array(this.model.bones.length * 16)
    for (let i = 0; i < this.boneStates.length; i++) {
      worldMatrices.set(this.boneStates[i].worldMatrix.values, i * 16)
    }
    this.device.queue.writeBuffer(this.worldMatrixBuffer, 0, worldMatrices)
  }

  private computeSkinMatrices() {
    const boneCount = this.model.bones.length
    const workgroupSize = 64
    const workgroupCount = Math.ceil(boneCount / workgroupSize)

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginComputePass()
    pass.setPipeline(this.skinMatrixComputePipeline)
    pass.setBindGroup(0, this.skinMatrixComputeBindGroup)
    pass.dispatchWorkgroups(workgroupCount)
    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  public rotateBone(boneName: string, rotation: Quat) {
    const index = this.model.bones.findIndex((b) => b.name === boneName)
    if (index < 0) return

    // Normalize the rotation quaternion and update pose
    this.boneStates[index].localRotation = rotation.normalize()
    this.evaluatePose() // This now handles GPU upload and compute
  }

  public render() {
    // Update render target views
    ;(this.renderPassDescriptor.colorAttachments as GPURenderPassColorAttachment[])[0].view = this.context
      .getCurrentTexture()
      .createView()

    // Update camera uniforms
    this.updateCameraUniforms()

    const encoder = this.device.createCommandEncoder()
    const pass = encoder.beginRenderPass(this.renderPassDescriptor)
    pass.setPipeline(this.pipeline)
    pass.setBindGroup(0, this.bindGroup)
    pass.setVertexBuffer(0, this.vertexBuffer)
    pass.setVertexBuffer(1, this.jointsBuffer)
    pass.setVertexBuffer(2, this.weightsBuffer)
    pass.setIndexBuffer(this.indexBuffer, "uint32")

    // Render each material separately with its own texture
    let firstIndex = 0
    for (let i = 0; i < this.model.materials.length; i++) {
      const material = this.model.materials[i]
      if (material.vertexCount === 0) continue

      const bindGroup = this.materialBindGroups[i]
      pass.setBindGroup(1, bindGroup)
      pass.drawIndexed(material.vertexCount, 1, firstIndex)
      firstIndex += material.vertexCount
    }

    pass.end()
    this.device.queue.submit([encoder.finish()])
  }

  public runRenderLoop() {
    const loop = () => {
      this.render()
      this.animationFrameId = requestAnimationFrame(loop)
    }

    this.animationFrameId = requestAnimationFrame(loop)
  }

  public dispose() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId)
      this.animationFrameId = null
    }
    if (this.camera) {
      this.camera.detachControl()
    }
    this.context.unconfigure()
  }
}
