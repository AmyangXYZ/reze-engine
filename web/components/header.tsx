import Image from "next/image"
import Link from "next/link"
import { Button } from "./ui/button"
import { EngineStats } from "reze-engine"

export default function Header({ stats }: { stats: EngineStats | null }) {
  return (
    <header className="absolute top-0 left-0 right-0 px-4 md:px-6 py-3 md:py-4 flex items-center gap-2 z-50 w-full select-none">
      <Link href="/">
        <h1 className="text-xl font-extrabold tracking-tight mr-8">Reze Engine</h1>
      </Link>

      <Link href="/tutorial" className="hidden md:block">
        <Button className="cursor-pointer text-white " variant="link" size="sm">
          Tutorial
        </Button>
      </Link>

      <Link href="https://github.com/AmyangXYZ/reze-engine" target="_blank" className="hidden md:block z-50">
        <Button className="cursor-pointer text-white" variant="link" size="sm">
          GitHub
        </Button>
      </Link>

      {stats && (
        <div className="ml-auto flex items-center gap-3 text-xs text-white/90 pointer-events-none bg-black py-2 rounded-full font-mono font-medium">
          <div className="flex items-center justify-end gap-4 tabular-nums flex-wrap md:hidden">
            <div>
              FPS: <span>{stats.fps}</span>
            </div>
            <div className="pointer-events-auto md:hidden">
              <Button size="icon" asChild className="bg-black text-white hover:bg-black hover:text-white rounded-full">
                <Link href="https://github.com/AmyangXYZ/reze-engine" target="_blank">
                  <Image src="/github-mark-white.svg" alt="GitHub" width={20} height={20} />
                </Link>
              </Button>
            </div>
          </div>

          <div className="flex items-center gap-4 tabular-nums hidden md:flex">
            <div>
              FPS: <span>{stats.fps}</span>
            </div>
            <div>
              Frame: <span>{stats.frameTime.toFixed(2)} ms</span>
            </div>
            <div>
              GPU: <span>{stats.gpuMemory.toFixed(1)} MB</span>
            </div>
          </div>
        </div>
      )}
    </header >
  )
}
