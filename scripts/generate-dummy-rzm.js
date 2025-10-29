// Generate dummy.rzm model file from humanoid geometry
// Run with: node scripts/generate-dummy-rzm.js

const fs = require('fs');
const path = require('path');

const RZM_MAGIC = 0x455A4552; // "REZE" in little-endian
const RZM_VERSION = 1;
const VERTEX_STRIDE = 8; // floats per vertex: position(3) + normal(3) + uv(2)

// Dummy humanoid positions (simple stick figure made of boxes)
const positions = new Float32Array([
  // Head (cube at top)
  -0.15, 0.9, 0.15, 0.15, 0.9, 0.15, 0.15, 1.2, 0.15, -0.15, 0.9, 0.15, 0.15, 1.2, 0.15, -0.15, 1.2, 0.15,
  -0.15, 0.9, -0.15, 0.15, 1.2, -0.15, 0.15, 0.9, -0.15, -0.15, 0.9, -0.15, -0.15, 1.2, -0.15, 0.15, 1.2, -0.15,
  -0.15, 1.2, 0.15, 0.15, 1.2, 0.15, 0.15, 1.2, -0.15, -0.15, 1.2, 0.15, 0.15, 1.2, -0.15, -0.15, 1.2, -0.15,
  -0.15, 0.9, 0.15, 0.15, 0.9, -0.15, 0.15, 0.9, 0.15, -0.15, 0.9, 0.15, -0.15, 0.9, -0.15, 0.15, 0.9, -0.15,
  0.15, 0.9, 0.15, 0.15, 0.9, -0.15, 0.15, 1.2, -0.15, 0.15, 0.9, 0.15, 0.15, 1.2, -0.15, 0.15, 1.2, 0.15,
  -0.15, 0.9, 0.15, -0.15, 1.2, -0.15, -0.15, 0.9, -0.15, -0.15, 0.9, 0.15, -0.15, 1.2, 0.15, -0.15, 1.2, -0.15,

  // Torso
  -0.25, 0.2, 0.1, 0.25, 0.2, 0.1, 0.25, 0.9, 0.1, -0.25, 0.2, 0.1, 0.25, 0.9, 0.1, -0.25, 0.9, 0.1,
  -0.25, 0.2, -0.1, 0.25, 0.9, -0.1, 0.25, 0.2, -0.1, -0.25, 0.2, -0.1, -0.25, 0.9, -0.1, 0.25, 0.9, -0.1,
  -0.25, 0.9, 0.1, 0.25, 0.9, 0.1, 0.25, 0.9, -0.1, -0.25, 0.9, 0.1, 0.25, 0.9, -0.1, -0.25, 0.9, -0.1,
  -0.25, 0.2, 0.1, 0.25, 0.2, -0.1, 0.25, 0.2, 0.1, -0.25, 0.2, 0.1, -0.25, 0.2, -0.1, 0.25, 0.2, -0.1,
  0.25, 0.2, 0.1, 0.25, 0.2, -0.1, 0.25, 0.9, -0.1, 0.25, 0.2, 0.1, 0.25, 0.9, -0.1, 0.25, 0.9, 0.1,
  -0.25, 0.2, 0.1, -0.25, 0.9, -0.1, -0.25, 0.2, -0.1, -0.25, 0.2, 0.1, -0.25, 0.9, 0.1, -0.25, 0.9, -0.1,

  // Left Leg
  -0.25, -0.8, 0.08, -0.08, -0.8, 0.08, -0.08, 0.2, 0.08, -0.25, -0.8, 0.08, -0.08, 0.2, 0.08, -0.25, 0.2, 0.08,
  -0.25, -0.8, -0.08, -0.08, 0.2, -0.08, -0.08, -0.8, -0.08, -0.25, -0.8, -0.08, -0.25, 0.2, -0.08, -0.08, 0.2, -0.08,
  -0.25, -0.8, 0.08, -0.25, 0.2, 0.08, -0.25, 0.2, -0.08, -0.25, -0.8, 0.08, -0.25, 0.2, -0.08, -0.25, -0.8, -0.08,
  -0.08, -0.8, 0.08, -0.08, 0.2, -0.08, -0.08, 0.2, 0.08, -0.08, -0.8, 0.08, -0.08, -0.8, -0.08, -0.08, 0.2, -0.08,

  // Right Leg
  0.08, -0.8, 0.08, 0.25, -0.8, 0.08, 0.25, 0.2, 0.08, 0.08, -0.8, 0.08, 0.25, 0.2, 0.08, 0.08, 0.2, 0.08,
  0.08, -0.8, -0.08, 0.25, 0.2, -0.08, 0.25, -0.8, -0.08, 0.08, -0.8, -0.08, 0.08, 0.2, -0.08, 0.25, 0.2, -0.08,
  0.08, -0.8, 0.08, 0.08, 0.2, 0.08, 0.08, 0.2, -0.08, 0.08, -0.8, 0.08, 0.08, 0.2, -0.08, 0.08, -0.8, -0.08,
  0.25, -0.8, 0.08, 0.25, 0.2, -0.08, 0.25, 0.2, 0.08, 0.25, -0.8, 0.08, 0.25, -0.8, -0.08, 0.25, 0.2, -0.08,

  // Left Arm
  -0.25, 0.5, 0.05, -0.15, 0.5, 0.05, -0.15, 0.85, 0.05, -0.25, 0.5, 0.05, -0.15, 0.85, 0.05, -0.25, 0.85, 0.05,
  -0.25, 0.5, -0.05, -0.15, 0.85, -0.05, -0.15, 0.5, -0.05, -0.25, 0.5, -0.05, -0.25, 0.85, -0.05, -0.15, 0.85, -0.05,
  -0.25, 0.5, 0.05, -0.25, 0.85, 0.05, -0.25, 0.85, -0.05, -0.25, 0.5, 0.05, -0.25, 0.85, -0.05, -0.25, 0.5, -0.05,
  -0.15, 0.5, 0.05, -0.15, 0.85, -0.05, -0.15, 0.85, 0.05, -0.15, 0.5, 0.05, -0.15, 0.5, -0.05, -0.15, 0.85, -0.05,

  // Right Arm
  0.15, 0.5, 0.05, 0.25, 0.5, 0.05, 0.25, 0.85, 0.05, 0.15, 0.5, 0.05, 0.25, 0.85, 0.05, 0.15, 0.85, 0.05,
  0.15, 0.5, -0.05, 0.25, 0.85, -0.05, 0.25, 0.5, -0.05, 0.15, 0.5, -0.05, 0.15, 0.85, -0.05, 0.25, 0.85, -0.05,
  0.15, 0.5, 0.05, 0.15, 0.85, 0.05, 0.15, 0.85, -0.05, 0.15, 0.5, 0.05, 0.15, 0.85, -0.05, 0.15, 0.5, -0.05,
  0.25, 0.5, 0.05, 0.25, 0.85, -0.05, 0.25, 0.85, 0.05, 0.25, 0.5, 0.05, 0.25, 0.5, -0.05, 0.25, 0.85, -0.05,
]);

// Convert positions to interleaved vertex data
const vertexCount = positions.length / 3;
const vertexData = new Float32Array(vertexCount * VERTEX_STRIDE);

for (let i = 0; i < vertexCount; i++) {
  const posIdx = i * 3;
  const vertIdx = i * VERTEX_STRIDE;

  // Position
  vertexData[vertIdx + 0] = positions[posIdx + 0];
  vertexData[vertIdx + 1] = positions[posIdx + 1];
  vertexData[vertIdx + 2] = positions[posIdx + 2];

  // Normal (dummy, pointing up)
  vertexData[vertIdx + 3] = 0;
  vertexData[vertIdx + 4] = 1;
  vertexData[vertIdx + 5] = 0;

  // UV (dummy)
  vertexData[vertIdx + 6] = 0;
  vertexData[vertIdx + 7] = 0;
}

// Create RZM file buffer
const headerSize = 64;
const vertexDataSize = vertexData.byteLength;
const totalSize = headerSize + vertexDataSize;

const buffer = Buffer.alloc(totalSize);

// Write header
buffer.writeUInt32LE(RZM_MAGIC, 0);       // magic
buffer.writeUInt32LE(RZM_VERSION, 4);     // version
buffer.writeUInt32LE(0, 8);               // flags
buffer.writeUInt32LE(vertexCount, 12);    // vertexCount
buffer.writeUInt32LE(0, 16);              // indexCount
buffer.writeUInt32LE(0, 20);              // materialCount
buffer.writeUInt32LE(0, 24);              // boneCount

// Write vertex data
Buffer.from(vertexData.buffer).copy(buffer, headerSize);

// Ensure output directory exists
const outputDir = path.join(__dirname, '..', 'public', 'models');
if (!fs.existsSync(outputDir)) {
  fs.mkdirSync(outputDir, { recursive: true });
}

// Write file
const outputPath = path.join(outputDir, 'dummy.rzm');
fs.writeFileSync(outputPath, buffer);

console.log(`Generated dummy.rzm: ${vertexCount} vertices, ${totalSize} bytes`);
console.log(`Output: ${outputPath}`);

