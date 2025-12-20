import type { NextConfig } from "next"
import path from "path"
// import { join } from "path"

const nextConfig: NextConfig = {
  experimental: {
    externalDir: true,
  },
  // Next.js 16 uses Turbopack by default; avoid webpack customization.
  turbopack: {
    resolveAlias: {
      // When importing engine sources from ../engine, resolve runtime deps from this app's node_modules.
      "@fred3d/ammo": path.resolve(__dirname, "node_modules/@fred3d/ammo"),
    },
  },
  // outputFileTracingRoot: join(__dirname, ".."),
}

export default nextConfig
