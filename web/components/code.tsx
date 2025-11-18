"use client"

import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { oneDark } from "react-syntax-highlighter/dist/esm/styles/prism"

interface CodeBlockProps {
  children: string
  language: string
  className?: string
}

export default function Code({ children, language, className = "" }: CodeBlockProps) {
  return (
    <div className={`rounded-md border border-zinc-700 overflow-hidden w-full ${className}`}>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        customStyle={{
          margin: 0,
          padding: "0.5rem 1rem",
          background: "#0d1117",
        }}
        codeTagProps={{
          style: {
            fontWeight: "600",
            fontSize: "0.9rem",
          }
        }}
        wrapLongLines
      >
        {children}
      </SyntaxHighlighter>
    </div>
  )
}