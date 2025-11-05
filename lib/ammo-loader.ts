"use client"
import type { AmmoInstance } from "@fred3d/ammo"

let ammoInstance: AmmoInstance | null = null
let ammoPromise: Promise<AmmoInstance> | null = null

export async function loadAmmo(): Promise<AmmoInstance> {
  // Return cached instance if available
  if (ammoInstance) {
    return ammoInstance
  }

  // Return existing promise if already loading
  if (ammoPromise) {
    return ammoPromise
  }

  // Start loading Ammo
  ammoPromise = (async () => {
    try {
      const { Ammo } = await import("@fred3d/ammo")
      ammoInstance = await Ammo()
      return ammoInstance
    } catch (error) {
      console.error("[Ammo] Failed to load:", error)
      ammoPromise = null // Reset promise so it can be retried
      throw error
    }
  })()

  return ammoPromise
}

export function getAmmoInstance(): AmmoInstance | null {
  return ammoInstance
}
