import type { NextConfig } from "next"
// import { join } from "path"

const nextConfig: NextConfig = {
  experimental: {
    externalDir: true,
  },
  // outputFileTracingRoot: join(__dirname, ".."),
}

export default nextConfig
