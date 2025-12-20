import type { NextConfig } from "next"
import path from "path"
// import { join } from "path"

const nextConfig: NextConfig = {
  experimental: {
    externalDir: true,
  },
  webpack: (config) => {
    config.resolve = config.resolve ?? {}
    config.resolve.modules = [
      ...(config.resolve.modules ?? []),
      // Ensure external monorepo sources (e.g. ../engine/src) can resolve deps from this app's node_modules.
      path.resolve(__dirname, "node_modules"),
    ]
    return config
  },
  // outputFileTracingRoot: join(__dirname, ".."),
}

export default nextConfig
