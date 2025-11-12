import { Model, Texture, Material, Bone, Skeleton, Skinning } from "./model"
import { Mat4, Vec3 } from "./math"
import { Rigidbody, Joint, RigidbodyShape, RigidbodyType } from "./physics"

export class PmxLoader {
  private view: DataView
  private offset = 0
  private decoder!: TextDecoder
  private encoding = 0
  private additionalVec4Count = 0
  private vertexIndexSize = 0
  private textureIndexSize = 0
  private materialIndexSize = 0
  private boneIndexSize = 0
  private morphIndexSize = 0
  private rigidBodyIndexSize = 0
  private textures: Texture[] = []
  private materials: Material[] = []
  private bones: Bone[] = []
  private inverseBindMatrices: Float32Array | null = null
  private joints0: Uint16Array | null = null
  private weights0: Uint8Array | null = null
  private rigidbodies: Rigidbody[] = []
  private joints: Joint[] = []

  private constructor(buffer: ArrayBuffer) {
    this.view = new DataView(buffer)
  }

  static async load(url: string): Promise<Model> {
    const loader = new PmxLoader(await fetch(url).then((r) => r.arrayBuffer()))
    return loader.parse()
  }

  private parse(): Model {
    this.parseHeader()
    const { positions, normals, uvs } = this.parseVertices()
    const indices = this.parseIndices()
    this.parseTextures()
    this.parseMaterials()
    this.parseBones()
    // Skip morphs and display frames before parsing rigidbodies
    this.skipMorphs()
    this.skipDisplayFrames()
    this.parseRigidbodies()
    this.parseJoints()
    this.computeInverseBind()
    return this.toModel(positions, normals, uvs, indices)
  }

  private parseHeader() {
    if (this.getString(3) !== "PMX") throw new Error("Not a PMX file")

    // PMX: 1-byte alignment after signature
    this.getUint8()

    // PMX: version (float32)
    const version = this.getFloat32()
    if (version < 2.0 || version > 2.2) {
      // Continue, but warn for unexpected version
      console.warn(`PMX version ${version} may not be fully supported`)
    }

    // PMX: globals count (uint8) followed by that many bytes describing encoding and index sizes
    const globalsCount = this.getUint8()

    // Validate globalsCount (must be at least 8 for PMX 2.x)
    if (globalsCount < 8) {
      throw new Error(`Invalid globalsCount: ${globalsCount}, expected at least 8`)
    }

    // Read globals (first 8 bytes are standard for PMX 2.x)
    this.encoding = this.getUint8() // 0:utf16le, 1:utf8
    this.additionalVec4Count = this.getUint8()
    this.vertexIndexSize = this.getUint8()
    this.textureIndexSize = this.getUint8()
    this.materialIndexSize = this.getUint8()
    this.boneIndexSize = this.getUint8()
    this.morphIndexSize = this.getUint8()
    this.rigidBodyIndexSize = this.getUint8()

    // Skip any extra globals beyond the first 8 (for future format extensions)
    if (globalsCount > 8) {
      for (let i = 8; i < globalsCount; i++) {
        this.getUint8()
      }
    }

    this.decoder = new TextDecoder(this.encoding === 0 ? "utf-16le" : "utf-8")

    // Skip model info (4 text fields)
    this.getText()
    this.getText()
    this.getText()
    this.getText()
  }

  private parseVertices() {
    const count = this.getInt32()
    const positions: number[] = []
    const normals: number[] = []
    const uvs: number[] = []
    // Prepare skinning arrays (4 influences per vertex)
    const joints = new Uint16Array(count * 4)
    const weights = new Uint8Array(count * 4) // UNORM8, will be normalized to 255

    for (let i = 0; i < count; i++) {
      const px = this.getFloat32()
      const py = this.getFloat32()
      const pz = this.getFloat32()
      positions.push(px, py, pz)

      const nx = this.getFloat32()
      const ny = this.getFloat32()
      const nz = this.getFloat32()
      normals.push(nx, ny, nz)

      const u = this.getFloat32()
      const v = this.getFloat32()
      // PMX UVs are in the same orientation as WebGPU sampling; no flip
      uvs.push(u, v)

      this.offset += this.additionalVec4Count * 16
      const type = this.getUint8()
      const base = i * 4
      // Initialize defaults
      joints[base] = 0
      joints[base + 1] = 0
      joints[base + 2] = 0
      joints[base + 3] = 0
      weights[base] = 255
      weights[base + 1] = 0
      weights[base + 2] = 0
      weights[base + 3] = 0

      if (type === 0) {
        // BDEF1
        const j0 = this.getNonVertexIndex(this.boneIndexSize)
        joints[base] = j0 >= 0 ? j0 : 0
        // weight stays [255,0,0,0]
      } else if (type === 1 || type === 3) {
        // BDEF2 or SDEF (treated as BDEF2)
        const j0 = this.getNonVertexIndex(this.boneIndexSize)
        const j1 = this.getNonVertexIndex(this.boneIndexSize)
        const w0f = this.getFloat32()
        const w0 = Math.max(0, Math.min(255, Math.round(w0f * 255)))
        const w1 = Math.max(0, Math.min(255, 255 - w0))
        joints[base] = j0 >= 0 ? j0 : 0
        joints[base + 1] = j1 >= 0 ? j1 : 0
        weights[base] = w0
        weights[base + 1] = w1
        // SDEF has extra 3 vec3 (C, R0, R1)
        if (type === 3) {
          this.offset += 36 // 9 floats * 4 bytes
        }
      } else if (type === 2 || type === 4) {
        // BDEF4 or QDEF (treat as LBS4)
        let sum = 0
        for (let k = 0; k < 4; k++) {
          const j = this.getNonVertexIndex(this.boneIndexSize)
          joints[base + k] = j >= 0 ? j : 0
        }
        const wf = [this.getFloat32(), this.getFloat32(), this.getFloat32(), this.getFloat32()]
        const ws = wf.map((x) => Math.max(0, Math.min(1, x)))
        const w8 = ws.map((x) => Math.round(x * 255))
        sum = w8[0] + w8[1] + w8[2] + w8[3]
        if (sum === 0) {
          weights[base] = 255
        } else {
          // Normalize to 255
          const scale = 255 / sum
          let accum = 0
          for (let k = 0; k < 3; k++) {
            const v = Math.max(0, Math.min(255, Math.round(w8[k] * scale)))
            weights[base + k] = v
            accum += v
          }
          weights[base + 3] = Math.max(0, Math.min(255, 255 - accum))
        }
      } else {
        throw new Error(`Invalid bone weight type: ${type}`)
      }
      this.offset += 4 // edge scale
    }

    this.joints0 = joints
    this.weights0 = weights
    return { positions, normals, uvs }
  }

  // (removed) skipBoneWeight – replaced by inline parsing

  private parseIndices() {
    const count = this.getInt32()
    const indices: number[] = []

    for (let i = 0; i < count; i++) {
      indices.push(this.getIndex(this.vertexIndexSize))
    }

    return indices
  }

  private parseTextures() {
    try {
      const count = this.getInt32()
      this.textures = []

      for (let i = 0; i < count; i++) {
        const textureName = this.getText()
        this.textures.push({
          path: textureName,
          name: textureName.split("/").pop() || textureName, // Extract filename
        })
      }
    } catch (error) {
      console.error("Error parsing textures:", error)
      this.textures = []
    }
  }

  private parseMaterials() {
    try {
      const count = this.getInt32()
      this.materials = []

      for (let i = 0; i < count; i++) {
        const name = this.getText()
        this.getText() // englishName (skip)

        const diffuse = [this.getFloat32(), this.getFloat32(), this.getFloat32(), this.getFloat32()] as [
          number,
          number,
          number,
          number
        ]
        const specular = [this.getFloat32(), this.getFloat32(), this.getFloat32()] as [number, number, number]
        const shininess = this.getFloat32()
        const ambient = [this.getFloat32(), this.getFloat32(), this.getFloat32()] as [number, number, number]

        const flag = this.getUint8()
        // edgeColor vec4
        const edgeColor: [number, number, number, number] = [
          this.getFloat32(),
          this.getFloat32(),
          this.getFloat32(),
          this.getFloat32(),
        ]
        // edgeSize float
        const edgeSize = this.getFloat32()

        const textureIndex = this.getNonVertexIndex(this.textureIndexSize)
        const sphereTextureIndex = this.getNonVertexIndex(this.textureIndexSize)
        const sphereTextureMode = this.getUint8()

        const isSharedToonTexture = this.getUint8() === 1
        const toonTextureIndex = isSharedToonTexture ? this.getUint8() : this.getNonVertexIndex(this.textureIndexSize)

        this.getText() // comment (skip)
        const vertexCount = this.getInt32()

        // PMX material flag bits:
        // Bit 0 (0x01): Double-sided rendering
        // Bit 4 (0x10): Edge drawing (outline)
        const mat: Material = {
          name,
          diffuse,
          specular,
          ambient,
          shininess,
          diffuseTextureIndex: textureIndex,
          normalTextureIndex: -1, // Not used in basic PMX
          sphereTextureIndex,
          sphereMode: sphereTextureMode,
          toonTextureIndex,
          edgeFlag: flag,
          edgeColor,
          edgeSize,
          vertexCount,
        }

        // Classify materials based on name
        const materialName = name.toLowerCase()

        // Classify eye materials
        mat.isEye =
          materialName.includes("目") || // Japanese "eye"
          materialName.includes("瞳") || // Japanese "pupil"
          materialName.includes("eye") ||
          materialName.includes("pupil") ||
          materialName.includes("iris") ||
          materialName.includes("目白") ||
          materialName.includes("眼") ||
          materialName.includes("睛") ||
          materialName.includes("眉")

        // Classify face materials
        mat.isFace = materialName.includes("face") || materialName.includes("脸")

        // Classify hair materials
        mat.isHair = materialName.includes("hair_f")

        this.materials.push(mat)
      }
    } catch (error) {
      console.error("Error parsing materials:", error)
      this.materials = []
    }
  }

  private parseBones() {
    try {
      const count = this.getInt32()
      const bones: Bone[] = []
      // Collect absolute positions, then convert to parent-relative offsets
      type AbsBone = {
        name: string
        parent: number
        x: number
        y: number
        z: number
        appendParent?: number
        appendRatio?: number
        appendRotate?: boolean
        appendMove?: boolean
      }
      const abs: AbsBone[] = new Array(count)
      // PMX 2.x bone flags (best-effort common masks)
      const FLAG_TAIL_IS_BONE = 0x0001
      const FLAG_IK = 0x0020
      const FLAG_APPEND_ROTATE = 0x0100
      const FLAG_APPEND_MOVE = 0x0200
      const FLAG_AXIS_LIMIT = 0x0400
      const FLAG_LOCAL_AXIS = 0x0800
      const FLAG_EXTERNAL_PARENT = 0x2000
      for (let i = 0; i < count; i++) {
        const name = this.getText()
        this.getText() // englishName (skip)
        // PMX is left-handed, engine is now left-handed - no conversion needed
        const x = this.getFloat32()
        const y = this.getFloat32()
        const z = this.getFloat32()
        const parentIndex = this.getNonVertexIndex(this.boneIndexSize)
        this.getInt32() // transform order (skip)
        const flags = this.getUint16()

        // Tail: bone index or offset vector3
        if ((flags & FLAG_TAIL_IS_BONE) !== 0) {
          this.getNonVertexIndex(this.boneIndexSize)
        } else {
          // tail offset vec3
          this.getFloat32()
          this.getFloat32()
          this.getFloat32()
        }

        // Append transform (inherit/ratio)
        let appendParent: number | undefined = undefined
        let appendRatio: number | undefined = undefined
        let appendRotate = false
        let appendMove = false
        if ((flags & (FLAG_APPEND_ROTATE | FLAG_APPEND_MOVE)) !== 0) {
          appendParent = this.getNonVertexIndex(this.boneIndexSize) // append parent
          appendRatio = this.getFloat32() // ratio
          appendRotate = (flags & FLAG_APPEND_ROTATE) !== 0
          appendMove = (flags & FLAG_APPEND_MOVE) !== 0
        }

        // Axis limit
        if ((flags & FLAG_AXIS_LIMIT) !== 0) {
          this.getFloat32()
          this.getFloat32()
          this.getFloat32()
        }

        // Local axis (two vectors x and z)
        if ((flags & FLAG_LOCAL_AXIS) !== 0) {
          // local axis X
          this.getFloat32()
          this.getFloat32()
          this.getFloat32()
          // local axis Z
          this.getFloat32()
          this.getFloat32()
          this.getFloat32()
        }

        // External parent transform
        if ((flags & FLAG_EXTERNAL_PARENT) !== 0) {
          this.getInt32()
        }

        // IK block
        if ((flags & FLAG_IK) !== 0) {
          this.getNonVertexIndex(this.boneIndexSize) // target
          this.getInt32() // iteration
          this.getFloat32() // rotationConstraint
          const linksCount = this.getInt32()
          for (let li = 0; li < linksCount; li++) {
            this.getNonVertexIndex(this.boneIndexSize) // link target
            const hasLimit = this.getUint8() === 1
            if (hasLimit) {
              // min and max angles (vec3 each)
              this.getFloat32()
              this.getFloat32()
              this.getFloat32()
              this.getFloat32()
              this.getFloat32()
              this.getFloat32()
            }
          }
        }
        // Stash minimal bone info; append data will be merged later
        abs[i] = { name, parent: parentIndex, x, y, z, appendParent, appendRatio, appendRotate, appendMove }
      }
      for (let i = 0; i < count; i++) {
        const a = abs[i]
        if (a.parent >= 0 && a.parent < count) {
          const p = abs[a.parent]
          bones.push({
            name: a.name,
            parentIndex: a.parent,
            bindTranslation: [a.x - p.x, a.y - p.y, a.z - p.z],
            children: [], // Will be populated later when building skeleton
            appendParentIndex: a.appendParent,
            appendRatio: a.appendRatio,
            appendRotate: a.appendRotate,
            appendMove: a.appendMove,
          })
        } else {
          bones.push({
            name: a.name,
            parentIndex: a.parent,
            bindTranslation: [a.x, a.y, a.z],
            children: [], // Will be populated later when building skeleton
            appendParentIndex: a.appendParent,
            appendRatio: a.appendRatio,
            appendRotate: a.appendRotate,
            appendMove: a.appendMove,
          })
        }
      }
      this.bones = bones
    } catch (e) {
      console.warn("Error parsing bones:", e)
      this.bones = []
    }
  }

  private skipMorphs(): boolean {
    try {
      // Check if we have enough bytes to read the count
      if (this.offset + 4 > this.view.buffer.byteLength) {
        return false
      }
      const count = this.getInt32()
      if (count < 0 || count > 100000) {
        // Suspicious count, likely corrupted - restore offset
        this.offset -= 4
        return false
      }
      for (let i = 0; i < count; i++) {
        // Check bounds before reading each morph
        if (this.offset >= this.view.buffer.byteLength) {
          return false
        }
        try {
          this.getText() // name
          this.getText() // englishName
          this.getUint8() // panelType
          const morphType = this.getUint8()
          const offsetCount = this.getInt32()

          // Skip offsets based on morph type
          for (let j = 0; j < offsetCount; j++) {
            if (this.offset >= this.view.buffer.byteLength) {
              return false
            }
            if (morphType === 0) {
              // Group morph
              this.getNonVertexIndex(this.morphIndexSize) // morphIndex
              this.getFloat32() // ratio
            } else if (morphType === 1) {
              // Vertex morph
              this.getIndex(this.vertexIndexSize) // vertexIndex
              this.getFloat32() // x
              this.getFloat32() // y
              this.getFloat32() // z
            } else if (morphType === 2) {
              // Bone morph
              this.getNonVertexIndex(this.boneIndexSize) // boneIndex
              this.getFloat32() // x
              this.getFloat32() // y
              this.getFloat32() // z
              this.getFloat32() // rx
              this.getFloat32() // ry
              this.getFloat32() // rz
            } else if (morphType === 3) {
              // UV morph
              this.getIndex(this.vertexIndexSize) // vertexIndex
              this.getFloat32() // u
              this.getFloat32() // v
            } else if (morphType === 4 || morphType === 5 || morphType === 6 || morphType === 7) {
              // UV morph types 4-7 (additional UV channels)
              this.getIndex(this.vertexIndexSize) // vertexIndex
              this.getFloat32() // u
              this.getFloat32() // v
            } else if (morphType === 8) {
              // Material morph
              this.getNonVertexIndex(this.materialIndexSize) // materialIndex
              this.getUint8() // offsetType
              this.getFloat32() // diffuse r
              this.getFloat32() // diffuse g
              this.getFloat32() // diffuse b
              this.getFloat32() // diffuse a
              this.getFloat32() // specular r
              this.getFloat32() // specular g
              this.getFloat32() // specular b
              this.getFloat32() // specular power
              this.getFloat32() // ambient r
              this.getFloat32() // ambient g
              this.getFloat32() // ambient b
              this.getFloat32() // edgeColor r
              this.getFloat32() // edgeColor g
              this.getFloat32() // edgeColor b
              this.getFloat32() // edgeColor a
              this.getFloat32() // edgeSize
              this.getFloat32() // textureCoeff r
              this.getFloat32() // textureCoeff g
              this.getFloat32() // textureCoeff b
              this.getFloat32() // textureCoeff a
              this.getFloat32() // sphereCoeff r
              this.getFloat32() // sphereCoeff g
              this.getFloat32() // sphereCoeff b
              this.getFloat32() // sphereCoeff a
              this.getFloat32() // toonCoeff r
              this.getFloat32() // toonCoeff g
              this.getFloat32() // toonCoeff b
              this.getFloat32() // toonCoeff a
            }
          }
        } catch (e) {
          // If we fail to read a morph, stop skipping
          console.warn(`Error reading morph ${i}:`, e)
          return false
        }
      }
      return true
    } catch (e) {
      console.warn("Error skipping morphs:", e)
      return false
    }
  }

  private skipDisplayFrames(): boolean {
    try {
      // Check if we have enough bytes to read the count
      if (this.offset + 4 > this.view.buffer.byteLength) {
        return false
      }
      const count = this.getInt32()
      if (count < 0 || count > 100000) {
        // Suspicious count, likely corrupted - restore offset
        this.offset -= 4
        return false
      }
      for (let i = 0; i < count; i++) {
        // Check bounds before reading each frame
        if (this.offset >= this.view.buffer.byteLength) {
          return false
        }
        try {
          this.getText() // name
          this.getText() // englishName
          this.getUint8() // flag
          const elementCount = this.getInt32()
          for (let j = 0; j < elementCount; j++) {
            if (this.offset >= this.view.buffer.byteLength) {
              return false
            }
            const elementType = this.getUint8()
            if (elementType === 0) {
              // Bone frame
              this.getNonVertexIndex(this.boneIndexSize)
            } else if (elementType === 1) {
              // Morph frame
              this.getNonVertexIndex(this.morphIndexSize)
            }
          }
        } catch (e) {
          // If we fail to read a frame, stop skipping
          console.warn(`Error reading display frame ${i}:`, e)
          return false
        }
      }
      return true
    } catch (e) {
      console.warn("Error skipping display frames:", e)
      return false
    }
  }

  private parseRigidbodies(): void {
    try {
      // Note: morphs and display frames are already skipped in parse() before calling this
      // Check bounds before reading rigidbody count
      if (this.offset + 4 > this.view.buffer.byteLength) {
        console.warn("Not enough bytes for rigidbody count")
        this.rigidbodies = []
        return
      }

      const count = this.getInt32()
      if (count < 0 || count > 10000) {
        // Suspicious count
        console.warn(`Suspicious rigidbody count: ${count}`)
        this.rigidbodies = []
        return
      }

      this.rigidbodies = []

      for (let i = 0; i < count; i++) {
        try {
          // Check bounds before reading each rigidbody
          if (this.offset >= this.view.buffer.byteLength) {
            console.warn(`Reached end of buffer while reading rigidbody ${i} of ${count}`)
            break
          }

          const name = this.getText()
          const englishName = this.getText()
          const boneIndex = this.getNonVertexIndex(this.boneIndexSize)
          const group = this.getUint8()
          const collisionMask = this.getUint16()

          const shape = this.getUint8() // 0=sphere, 1=box, 2=capsule

          // Size parameters (depends on shape)
          const sizeX = this.getFloat32()
          const sizeY = this.getFloat32()
          const sizeZ = this.getFloat32()

          const posX = this.getFloat32()
          const posY = this.getFloat32()
          const posZ = this.getFloat32()

          // Rotation (Euler angles in RADIANS in PMX file - PMX Editor displays them as degrees for convenience)
          // ZXY order (left-handed system)
          const rotX = this.getFloat32()
          const rotY = this.getFloat32()
          const rotZ = this.getFloat32()

          // Physical properties
          const mass = this.getFloat32()
          const linearDamping = this.getFloat32()
          const angularDamping = this.getFloat32()
          const restitution = this.getFloat32()
          const friction = this.getFloat32()
          const type = this.getUint8() // 0=static, 1=dynamic, 2=kinematic

          this.rigidbodies.push({
            name,
            englishName,
            boneIndex,
            group,
            collisionMask,
            shape: shape as RigidbodyShape,
            size: new Vec3(sizeX, sizeY, sizeZ),
            shapePosition: new Vec3(posX, posY, posZ),
            shapeRotation: new Vec3(rotX, rotY, rotZ),
            mass,
            linearDamping,
            angularDamping,
            restitution,
            friction,
            type: type as RigidbodyType,
            bodyOffsetMatrixInverse: Mat4.identity(),
          })
        } catch (e) {
          console.warn(`Error reading rigidbody ${i} of ${count}:`, e)
          // Stop parsing if we encounter an error
          break
        }
      }
    } catch (e) {
      console.warn("Error parsing rigidbodies:", e)
      this.rigidbodies = []
    }
  }

  private parseJoints(): void {
    try {
      // Check bounds before reading joint count
      if (this.offset + 4 > this.view.buffer.byteLength) {
        console.warn("Not enough bytes for joint count")
        this.joints = []
        return
      }

      const count = this.getInt32()
      if (count < 0 || count > 10000) {
        console.warn(`Suspicious joint count: ${count}`)
        this.joints = []
        return
      }

      this.joints = []

      for (let i = 0; i < count; i++) {
        try {
          // Check bounds before reading each joint
          if (this.offset >= this.view.buffer.byteLength) {
            console.warn(`Reached end of buffer while reading joint ${i} of ${count}`)
            break
          }

          const name = this.getText()
          const englishName = this.getText()
          const type = this.getUint8()

          // Rigidbody indices use rigidBodyIndexSize (not boneIndexSize)
          const rigidbodyIndexA = this.getNonVertexIndex(this.rigidBodyIndexSize)
          const rigidbodyIndexB = this.getNonVertexIndex(this.rigidBodyIndexSize)

          const posX = this.getFloat32()
          const posY = this.getFloat32()
          const posZ = this.getFloat32()

          // Rotation (Euler angles in RADIANS in PMX file - PMX Editor displays them as degrees for convenience)
          // ZXY order (left-handed system)
          const rotX = this.getFloat32()
          const rotY = this.getFloat32()
          const rotZ = this.getFloat32()

          const posMinX = this.getFloat32()
          const posMinY = this.getFloat32()
          const posMinZ = this.getFloat32()

          const posMaxX = this.getFloat32()
          const posMaxY = this.getFloat32()
          const posMaxZ = this.getFloat32()

          // Rotation constraints (in RADIANS in PMX file)
          const rotMinX = this.getFloat32()
          const rotMinY = this.getFloat32()
          const rotMinZ = this.getFloat32()

          const rotMaxX = this.getFloat32()
          const rotMaxY = this.getFloat32()
          const rotMaxZ = this.getFloat32()

          // Spring parameters
          const springPosX = this.getFloat32()
          const springPosY = this.getFloat32()
          const springPosZ = this.getFloat32()

          // Spring rotation (stiffness values in PMX file)
          const springRotX = this.getFloat32()
          const springRotY = this.getFloat32()
          const springRotZ = this.getFloat32()

          // Store rotations and springs as Vec3 (Euler angles), not Quat
          this.joints.push({
            name,
            englishName,
            type,
            rigidbodyIndexA,
            rigidbodyIndexB,
            position: new Vec3(posX, posY, posZ),
            rotation: new Vec3(rotX, rotY, rotZ), // Store as Vec3 (Euler angles)
            positionMin: new Vec3(posMinX, posMinY, posMinZ),
            positionMax: new Vec3(posMaxX, posMaxY, posMaxZ),
            rotationMin: new Vec3(rotMinX, rotMinY, rotMinZ), // Store as Vec3 (Euler angles)
            rotationMax: new Vec3(rotMaxX, rotMaxY, rotMaxZ), // Store as Vec3 (Euler angles)
            springPosition: new Vec3(springPosX, springPosY, springPosZ),
            springRotation: new Vec3(springRotX, springRotY, springRotZ), // Store as Vec3 (stiffness values)
          })
        } catch (e) {
          console.warn(`Error reading joint ${i} of ${count}:`, e)
          // Stop parsing if we encounter an error
          break
        }
      }
    } catch (e) {
      console.warn("Error parsing joints:", e)
      this.joints = []
    }
  }

  private computeInverseBind() {
    if (!this.bones || this.bones.length === 0) {
      this.inverseBindMatrices = new Float32Array(0)
      return
    }
    const n = this.bones.length
    const world = new Array<Mat4 | null>(n).fill(null)
    const inv = new Float32Array(n * 16)

    const computeWorld = (i: number): Mat4 => {
      if (world[i]) return world[i] as Mat4
      const bone = this.bones[i]
      const local = Mat4.identity().translateInPlace(
        bone.bindTranslation[0],
        bone.bindTranslation[1],
        bone.bindTranslation[2]
      )
      let w: Mat4
      if (bone.parentIndex >= 0 && bone.parentIndex < n) {
        w = computeWorld(bone.parentIndex).multiply(local)
      } else {
        w = local
      }
      world[i] = w
      return w
    }

    for (let i = 0; i < n; i++) {
      const w = computeWorld(i)
      const invm = Mat4.identity().translateInPlace(-w.values[12], -w.values[13], -w.values[14])
      inv.set(invm.values, i * 16)
    }
    this.inverseBindMatrices = inv
  }

  private toModel(positions: number[], normals: number[], uvs: number[], indices: number[]): Model {
    // Create indexed vertex buffer
    const vertexCount = positions.length / 3
    const vertexData = new Float32Array(vertexCount * 8)

    for (let i = 0; i < vertexCount; i++) {
      const pi = i * 3
      const ui = i * 2
      const vi = i * 8

      vertexData[vi] = positions[pi]
      vertexData[vi + 1] = positions[pi + 1]
      vertexData[vi + 2] = positions[pi + 2]
      vertexData[vi + 3] = normals[pi]
      vertexData[vi + 4] = normals[pi + 1]
      vertexData[vi + 5] = normals[pi + 2]
      vertexData[vi + 6] = uvs[ui]
      vertexData[vi + 7] = uvs[ui + 1]
    }

    // Create index buffer
    const indexData = new Uint32Array(indices)

    // Create skeleton (required)
    const skeleton: Skeleton = {
      bones: this.bones.length > 0 ? this.bones : [],
      inverseBindMatrices: this.inverseBindMatrices || new Float32Array(0),
    }

    let skinning: Skinning
    if (this.joints0 && this.weights0) {
      // Clamp joints to valid range now that we know bone count, and renormalize weights
      const boneCount = this.bones.length
      const joints = this.joints0
      const weights = this.weights0
      for (let i = 0; i < joints.length; i += 4) {
        // First pass: identify and zero out invalid joints, collect valid weight sum
        let validWeightSum = 0
        let validCount = 0
        for (let k = 0; k < 4; k++) {
          const j = joints[i + k]
          if (j < 0 || j >= boneCount) {
            // Invalid joint: zero the weight but keep joint index for debugging
            weights[i + k] = 0
            // Optionally clamp to valid range
            if (j < 0) {
              joints[i + k] = 0
            } else {
              joints[i + k] = boneCount > 0 ? boneCount - 1 : 0
            }
          } else {
            validWeightSum += weights[i + k]
            validCount++
          }
        }

        // If no valid weights, assign all weight to first bone (bone 0)
        if (validWeightSum === 0 || validCount === 0) {
          weights[i] = 255
          weights[i + 1] = 0
          weights[i + 2] = 0
          weights[i + 3] = 0
          joints[i] = boneCount > 0 ? 0 : 0
          joints[i + 1] = 0
          joints[i + 2] = 0
          joints[i + 3] = 0
        } else if (validWeightSum !== 255) {
          // Normalize valid weights to sum to exactly 255
          const scale = 255 / validWeightSum
          let accum = 0
          for (let k = 0; k < 3; k++) {
            if (joints[i + k] >= 0 && joints[i + k] < boneCount) {
              const v = Math.max(0, Math.min(255, Math.round(weights[i + k] * scale)))
              weights[i + k] = v
              accum += v
            } else {
              weights[i + k] = 0
            }
          }
          // Handle the 4th weight
          if (joints[i + 3] >= 0 && joints[i + 3] < boneCount) {
            weights[i + 3] = Math.max(0, Math.min(255, 255 - accum))
          } else {
            weights[i + 3] = 0
            // Redistribute the remainder to the last valid weight
            if (accum < 255) {
              for (let k = 2; k >= 0; k--) {
                if (joints[i + k] >= 0 && joints[i + k] < boneCount && weights[i + k] > 0) {
                  weights[i + k] = Math.min(255, weights[i + k] + (255 - accum))
                  break
                }
              }
            }
          }

          // Final safety check: ensure sum is exactly 255
          const finalSum = weights[i] + weights[i + 1] + weights[i + 2] + weights[i + 3]
          if (finalSum !== 255) {
            const diff = 255 - finalSum
            // Add/subtract difference to/from the largest valid weight
            let maxIdx = 0
            let maxWeight = weights[i]
            for (let k = 1; k < 4; k++) {
              if (weights[i + k] > maxWeight && joints[i + k] >= 0 && joints[i + k] < boneCount) {
                maxWeight = weights[i + k]
                maxIdx = k
              }
            }
            if (joints[i + maxIdx] >= 0 && joints[i + maxIdx] < boneCount) {
              weights[i + maxIdx] = Math.max(0, Math.min(255, weights[i + maxIdx] + diff))
            }
          }
        }
      }
      skinning = { joints, weights }
    } else {
      // Create default skinning (single bone per vertex)
      const vertexCount = positions.length / 3
      const joints = new Uint16Array(vertexCount * 4)
      const weights = new Uint8Array(vertexCount * 4)
      for (let i = 0; i < vertexCount; i++) {
        joints[i * 4] = 0
        weights[i * 4] = 255
      }
      skinning = { joints, weights }
    }

    return new Model(
      vertexData,
      indexData,
      this.textures,
      this.materials,
      skeleton,
      skinning,
      this.rigidbodies,
      this.joints
    )
  }

  private getUint8() {
    if (this.offset >= this.view.buffer.byteLength) {
      throw new RangeError(`Offset ${this.offset} exceeds buffer bounds ${this.view.buffer.byteLength}`)
    }
    return this.view.getUint8(this.offset++)
  }

  private getUint16() {
    if (this.offset + 2 > this.view.buffer.byteLength) {
      throw new RangeError(`Offset ${this.offset} + 2 exceeds buffer bounds ${this.view.buffer.byteLength}`)
    }
    const v = this.view.getUint16(this.offset, true)
    this.offset += 2
    return v
  }

  // Vertex index: 1->uint8, 2->uint16, 4->int32
  private getVertexIndex(size: number) {
    if (size === 1) return this.getUint8()
    if (size === 2) {
      const v = this.view.getUint16(this.offset, true)
      this.offset += 2
      return v
    }
    return this.getInt32()
  }

  // Non-vertex indices (texture/material/bone/morph/rigid): 1->int8, 2->int16, 4->int32
  private getNonVertexIndex(size: number) {
    if (size === 1) {
      const v = this.view.getInt8(this.offset)
      this.offset += 1
      return v
    }
    if (size === 2) {
      const v = this.view.getInt16(this.offset, true)
      this.offset += 2
      return v
    }
    return this.getInt32()
  }

  private getInt32() {
    if (this.offset + 4 > this.view.buffer.byteLength) {
      throw new RangeError(`Offset ${this.offset} + 4 exceeds buffer bounds ${this.view.buffer.byteLength}`)
    }
    const v = this.view.getInt32(this.offset, true)
    this.offset += 4
    return v
  }

  private getFloat32() {
    if (this.offset + 4 > this.view.buffer.byteLength) {
      throw new RangeError(`Offset ${this.offset} + 4 exceeds buffer bounds ${this.view.buffer.byteLength}`)
    }
    const v = this.view.getFloat32(this.offset, true)
    this.offset += 4
    return v
  }

  private getString(len: number) {
    const bytes = new Uint8Array(this.view.buffer, this.offset, len)
    this.offset += len
    return String.fromCharCode(...bytes)
  }

  private getText() {
    const len = this.getInt32()
    if (len <= 0) return ""

    // Debug: log problematic string lengths
    if (len > 1000 || len < -1000) {
      throw new RangeError(`Suspicious string length: ${len} at offset ${this.offset - 4}`)
    }

    // Ensure we don't read beyond buffer bounds
    if (this.offset + len > this.view.buffer.byteLength) {
      throw new RangeError(`String length ${len} exceeds buffer bounds at offset ${this.offset - 4}`)
    }

    const bytes = new Uint8Array(this.view.buffer, this.offset, len)
    this.offset += len
    return this.decoder.decode(bytes)
  }

  private getIndex(size: number) {
    // Backward-compat helper: defaults to vertex index behavior
    return this.getVertexIndex(size)
  }
}
