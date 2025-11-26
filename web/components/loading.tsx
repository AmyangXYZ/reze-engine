"use client"

import { Progress } from "./ui/progress"
import { useEffect, useState } from "react"

export default function Loading({ loading }: { loading: boolean }) {
  const [progress, setProgress] = useState(0)

  useEffect(() => {
    if (loading) {
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            return 0
          }
          return prev + 1
        })
      }, 50)

      return () => clearInterval(interval)
    }
  }, [loading])

  return (
    <div className="absolute inset-0 w-screen h-screen flex items-center justify-center text-white p-6 z-50 bg-black">
      <Progress value={progress} className="rounded-none w-full z-50 w-sm" />
    </div>
  )
}
