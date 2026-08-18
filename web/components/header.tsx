"use client"

import { useEffect, useState } from "react"
import type { RefObject } from "react"
import Image from "next/image"
import Link from "next/link"
import { Button } from "./ui/button"
import { Engine, EngineStats } from "reze-engine"
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
  { key: "cpuPhysicsMs", label: "physics", unit: "ms" },
  { key: "cpuRenderMs", label: "render prep", unit: "ms" },
]

/**
 * The GPU passes, in frame order, named as the engine's architecture figure
 * names them. Keys are the engine's TIMED_PASSES verbatim; a pass that did not
 * run reports 0 and is drawn at zero rather than hidden — "the mirror cost
 * nothing" and "there is no mirror row" are different facts.
 */
const PASS_ROWS: { key: string; label: string }[] = [
  { key: "cull", label: "frustum cull" },
  { key: "morph", label: "vertex morphs" },
  { key: "shadow", label: "shadow maps" },
  { key: "mirror", label: "planar reflection" },
  { key: "scene", label: "scene pass" },
  { key: "field", label: "field effects" },
  { key: "bloom", label: "bloom pyramid" },
  { key: "composite", label: "composite" },
]

type GpuTimings = Record<string, number> | null

/**
 * getGpuTimings shipped after the version this demo currently pins, so it is
 * read through a narrow structural shape: compiles against the old package,
 * lights up the moment the dependency is bumped. Reading it is also what
 * ENROLS the engine in the per-frame timestamp readback — the engine does no
 * timing work until something asks, which is why this is only called while the
 * dropdown is open.
 */
function gpuTimingsOf(engine: Engine | null): GpuTimings {
  if (!engine || !("getGpuTimings" in engine)) return null
  return (engine as unknown as { getGpuTimings(): GpuTimings }).getGpuTimings()
}

export default function Header({ stats, engineRef }: { stats: EngineStats | null; engineRef?: RefObject<Engine | null> }) {
  const [statsOpen, setStatsOpen] = useState(false)
  const [passes, setPasses] = useState<GpuTimings>(null)
  useEffect(() => {
    if (!statsOpen || !engineRef) return
    const sample = () => setPasses(gpuTimingsOf(engineRef.current))
    sample()
    const id = window.setInterval(sample, 250)
    return () => window.clearInterval(id)
  }, [statsOpen, engineRef])
  const passMax = passes ? Math.max(...PASS_ROWS.map((r) => passes[r.key] ?? 0), 0.01) : 0.01
  // Some devices (WebKit today) report nearly the same span — about the whole
  // frame — for every pass: they are not resolving timings per pass, and bars
  // drawn from that look like a finding. Detected, not hard-coded per browser.
  const busy = passes ? PASS_ROWS.map((r) => passes[r.key] ?? 0).filter((v) => v > 0.01) : []
  const degenerate =
    busy.length >= 3 &&
    (stats?.frameTime ?? 0) > 0 &&
    passMax > (stats?.frameTime ?? 0) * 0.6 &&
    Math.min(...busy) > passMax * 0.6
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
            <div className="absolute right-0 mt-2 max-h-[70vh] w-72 overflow-y-auto rounded-xl bg-black/70 p-3 font-mono text-xs text-white/90 backdrop-blur-sm">
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
              {/* GPU passes — the architecture figure's boxes, in frame order.
                  Bars are RELATIVE TO THE LARGEST PASS, not the frame: the
                  timestamps are coarsely quantised and passes overlap, so
                  proportions across a change are the comparison the instrument
                  supports, and a budget that sums to the frame is not. */}
              <div className="mt-2 border-t border-white/15 pt-2">
                <div className="mb-1 text-white/60">GPU passes</div>
                {passes === null ? (
                  <div className="text-white/40">
                    {engineRef ? "measuring… (needs timestamp queries)" : "unavailable"}
                  </div>
                ) : degenerate ? (
                  <div className="text-amber-300">
                    This device reports ~the whole frame for every pass — it is not resolving timings per pass.
                    Compare the frame rows above instead.
                  </div>
                ) : (
                  PASS_ROWS.map(({ key, label }) => {
                    const ms = passes[key] ?? 0
                    return (
                      <div key={key} className="py-0.5">
                        <div className="flex items-center justify-between">
                          <span className={ms > 0 ? "text-white/90" : "text-white/40"}>{label}</span>
                          <span className="tabular-nums">{ms.toFixed(2)} ms</span>
                        </div>
                        <div className="mt-0.5 h-1 w-full overflow-hidden rounded bg-white/10">
                          <div
                            className="h-full rounded bg-sky-400"
                            style={{ width: `${Math.min(100, (ms / passMax) * 100)}%` }}
                          />
                        </div>
                      </div>
                    )
                  })
                )}
              </div>
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
