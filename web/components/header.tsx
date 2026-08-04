"use client"

import { useState } from "react"
import Image from "next/image"
import Link from "next/link"
import { Button } from "./ui/button"
import { EngineStats } from "reze-engine"
import { BookOpenText, Rocket, WandSparkles } from "lucide-react"

/** Stats worth a row in the dropdown, in reading order. Everything else in
 *  EngineStats is either internal or derivable from these. */
const STAT_ROWS: { key: keyof EngineStats; label: string; unit?: string }[] = [
  { key: "fps", label: "FPS" },
  { key: "fps1PercentLow", label: "1% low" },
  { key: "frameTime", label: "frame avg", unit: "ms" },
  { key: "frameTimeMax", label: "frame max", unit: "ms" },
  { key: "jitter", label: "jitter", unit: "ms" },
  { key: "cpuAnimMs", label: "anim", unit: "ms" },
  { key: "cpuPhysicsMs", label: "physics (main)", unit: "ms" },
  { key: "cpuPhysicsWorkerMs", label: "physics (worker)", unit: "ms" },
  { key: "cpuRenderMs", label: "render prep", unit: "ms" },
]

export default function Header({ stats }: { stats: EngineStats | null }) {
  const [statsOpen, setStatsOpen] = useState(false)
  return (
    <header className="absolute top-0 left-0 right-0 px-4 md:px-6 py-2 flex items-center gap-2 z-50 w-full select-none flex flex-row justify-between">
      <div className="flex items-center gap-2">
        <Link href="/">
          <h1
            className="text-2xl font-light tracking-[0.2em] md:tracking-[0.3em] ext-white uppercase letter-spacing-wider"
            style={{
              textShadow: "0 0 20px rgba(255, 255, 255, 0.3), 0 2px 10px rgba(0, 0, 0, 0.5)",
              fontFamily: "var(--font-geist-sans)",
              fontWeight: 400,
            }}
          >
            Reze Engine
          </h1>
        </Link>
      </div>

      {stats && (
        <div className="relative ml-auto hidden md:block">
          {/* Click for the full engine.getStats() readout — replaces the old
              window.engine console handle. */}
          <button
            onClick={() => setStatsOpen((v) => !v)}
            className="flex h-7 cursor-pointer items-center gap-3 rounded-full bg-black/30 px-3 font-mono text-xs font-medium text-white/90 backdrop-blur-sm select-none hover:bg-black/50 md:px-4"
          >
            <span className="tabular-nums">FPS: {stats.fps}</span>
          </button>
          {statsOpen && (
            <div className="absolute right-0 mt-2 w-56 rounded-xl bg-black/70 p-3 font-mono text-xs text-white/90 backdrop-blur-sm">
              {STAT_ROWS.map(({ key, label, unit }) => {
                const v = stats[key]
                if (typeof v !== "number") return null
                return (
                  <div key={key} className="flex items-center justify-between py-0.5">
                    <span className="text-white/60">{label}</span>
                    <span className="tabular-nums">
                      {v}
                      {unit ? ` ${unit}` : ""}
                    </span>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      )}

      <div className="pointer-events-auto flex flex-row items-center gap-0 px-1 bg-black/30 backdrop-blur-sm rounded-full h-7 ">
        <Button variant="ghost" size="icon" asChild className="hover:bg-black hover:text-white rounded-full">
          <Link href="https://reze.design" target="_blank">
            <WandSparkles />
          </Link>
        </Button>
        <Button variant="ghost" size="icon" asChild className="hover:bg-black hover:text-white rounded-full">
          <Link href="https://reze.studio" target="_blank">
            <Rocket />
          </Link>
        </Button>
        <Button variant="ghost" size="icon" asChild className="hover:bg-black hover:text-white rounded-full">
          <Link href="/tutorial">
            <BookOpenText />
          </Link>
        </Button>
        <Button variant="ghost" size="icon" asChild className="hover:bg-black hover:text-white rounded-full">
          <Link href="https://github.com/AmyangXYZ/reze-engine" target="_blank">
            <Image src="/github-mark-white.svg" alt="GitHub" width={17} height={17} />
          </Link>
        </Button>
      </div>
    </header>
  )
}
