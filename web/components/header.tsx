import Image from "next/image"
import Link from "next/link"
import { Button } from "./ui/button"
import type { EngineStats } from "@/lib/engine"

export default function Header({ stats }: { stats: EngineStats }) {
  return (
    <header className="absolute top-0 left-0 right-0 px-4 md:px-6 py-3 md:py-4 pointer-events-none z-10 flex items-center gap-2">
      <h1 className="text-xl font-extrabold tracking-tight">Reze Engine</h1>

      <div className="ml-auto flex items-center gap-3 text-xs text-white/90 pointer-events-none bg-black py-2 px-4 rounded-full font-mono font-medium">
        {/* Mobile: show only FPS */}
        <div className="md:hidden tabular-nums">
          FPS: <span>{stats.fps}</span>
        </div>

        {/* Desktop: show essential stats */}
        <div className="hidden md:flex items-center gap-4 tabular-nums">
          <div>
            FPS: <span>{stats.fps}</span>
          </div>
          <div>
            Frame: <span>{stats.frameTime.toFixed(2)} ms</span>
          </div>
          <div>
            Calls: <span>{stats.drawCalls}</span>
          </div>
        </div>
      </div>

      {/* GitHub button (clickable) */}
      <div className="pointer-events-auto md:ml-2">
        <Button size="icon" asChild className="bg-black text-white hover:bg-black hover:text-white rounded-full">
          <Link href="https://github.com/AmyangXYZ/reze-engine" target="_blank">
            <Image src="/github-mark-white.svg" alt="GitHub" width={20} height={20} />
          </Link>
        </Button>
      </div>
    </header>
  )
}
