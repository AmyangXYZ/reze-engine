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
    <div className={`w-full text-sm ${className}`}>
      <SyntaxHighlighter
        language={language}
        style={oneDark}
        customStyle={{
          margin: 0,
          padding: "0.5rem 1rem",
          background: "#0d1117",
        }}
        codeTagProps={{
          style: {},
        }}
        wrapLongLines
      >
        {children}
      </SyntaxHighlighter>
    </div>
  )
}
