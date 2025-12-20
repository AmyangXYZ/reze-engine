import type { NextConfig } from "next"
// import { join } from "path"

const nextConfig: NextConfig = {
  experimental: {
    externalDir: true,
  },
  // Rely on tsconfig `paths` for resolving engine runtime deps when importing from ../engine.
  turbopack: {},
  // outputFileTracingRoot: join(__dirname, ".."),
}

export default nextConfig
