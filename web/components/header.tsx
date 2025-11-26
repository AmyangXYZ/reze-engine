import Image from "next/image"
import Link from "next/link"
import { Button } from "./ui/button"
import { EngineStats } from "reze-engine"
import { BookOpenText } from "lucide-react"

export default function Header({ stats }: { stats: EngineStats | null }) {
  return (
    <header className="absolute top-0 left-0 right-0 px-4 md:px-6 py-3 md:py-4 flex items-center gap-2 z-50 w-full select-none flex flex-row justify-between">
      <div className="flex items-center gap-2">
        <Link href="/">
          <h1 className="text-2xl md:text-4xl font-light tracking-[0.2em] md:tracking-[0.28em] ext-white uppercase letter-spacing-wider" style={{
            textShadow: '0 0 20px rgba(255, 255, 255, 0.3), 0 2px 10px rgba(0, 0, 0, 0.5)',
            fontFamily: 'var(--font-geist-sans)',
            fontWeight: 400
          }}>Reze Engine</h1>
        </Link>

        {/* <Link href="/tutorial" className="hidden md:block">
          <Button className="cursor-pointer text-white " variant="link" size="sm">
            Tutorial
          </Button>
        </Link>

        <Link href="https://github.com/AmyangXYZ/reze-engine" target="_blank" className="hidden md:block z-50">
          <Button className="cursor-pointer text-white" variant="link" size="sm">
            GitHub
          </Button>
        </Link> */}
      </div>

      {stats && (
        <div className="ml-auto flex items-center gap-3 text-xs text-white/90 pointer-events-none bg-black h-7 px-3 md:py-2 md:px-4 rounded-full font-mono font-medium  hidden md:flex">
          {/* <div className="flex items-center justify-end gap-4 tabular-nums flex-wrap md:hidden">
            <div>
              FPS: <span>{stats.fps}</span>
            </div>


          </div> */}

          <div className="flex items-center gap-4 tabular-nums">
            <div>
              FPS: <span>{stats.fps}</span>
            </div>
          </div>
        </div>
      )}

      <div className="pointer-events-auto flex items-center gap-1 bg-black rounded-full h-7">
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
    </header >
  )
}
