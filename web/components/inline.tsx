import React from "react"

export default function Inline({ children }: { children: React.ReactNode }) {
  return (
    <code className="bg-zinc-800 px-1.5 py-0.5 rounded text-sm font-mono">
      {children}
    </code>
  )
}