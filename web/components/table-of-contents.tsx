"use client"

import { useEffect, useState, useRef } from "react"
import Link from "next/link"

interface Heading {
  id: string
  text: string
  level: number
}

function generateId(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "") // Remove special characters
    .replace(/\s+/g, "-") // Replace spaces with hyphens
    .replace(/-+/g, "-") // Replace multiple hyphens with single hyphen
    .trim()
}

function extractHeadings(): Heading[] {
  const headings: Heading[] = []
  const headingElements = document.querySelectorAll("h1, h2, h3, h4, h5, h6")

  headingElements.forEach((element) => {
    const tagName = element.tagName.toLowerCase()
    const level = parseInt(tagName.charAt(1))
    const text = element.textContent || ""

    // Get or generate ID
    let id = element.id
    if (!id) {
      id = generateId(text)
      element.id = id
    }

    headings.push({ id, text, level })
  })

  return headings.filter((heading) => heading.text != "Reze Engine")
}

export default function TableOfContents() {
  const [headings, setHeadings] = useState<Heading[]>([])
  const [activeId, setActiveId] = useState<string>("")
  const isManualScrollRef = useRef(false)

  useEffect(() => {
    // Extract headings from the DOM (defer state update to avoid synchronous setState)
    const extractedHeadings = extractHeadings()

    // Defer state update to next tick
    const timeoutId = setTimeout(() => {
      setHeadings(extractedHeadings)
    }, 0)

    // Function to find the active heading based on scroll position
    const findActiveHeading = () => {
      // Skip if we're in a manual scroll
      if (isManualScrollRef.current) return

      const scrollOffset = 100 // Offset from top of viewport
      let activeHeading: Heading | null = null
      let minDistance = Infinity

      extractedHeadings.forEach((heading) => {
        const element = document.getElementById(heading.id)
        if (!element) return

        const rect = element.getBoundingClientRect()
        const distanceFromTop = Math.abs(rect.top - scrollOffset)

        // If heading is above the threshold or very close to it, consider it active
        if (rect.top <= scrollOffset + 50) {
          if (distanceFromTop < minDistance) {
            minDistance = distanceFromTop
            activeHeading = heading
          }
        }
      })

      // If no heading is above threshold, use the first one that's visible
      if (!activeHeading) {
        for (const heading of extractedHeadings) {
          const element = document.getElementById(heading.id)
          if (element) {
            const rect = element.getBoundingClientRect()
            if (rect.top < window.innerHeight && rect.bottom > 0) {
              activeHeading = heading
              break
            }
          }
        }
      }

      if (activeHeading) {
        setActiveId(activeHeading.id)
      }
    }

    // Initial check
    findActiveHeading()

    // Set up scroll listener with throttling
    let ticking = false
    const handleScroll = () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          findActiveHeading()
          ticking = false
        })
        ticking = true
      }
    }

    window.addEventListener("scroll", handleScroll, { passive: true })

    return () => {
      clearTimeout(timeoutId)
      window.removeEventListener("scroll", handleScroll)
    }
  }, [])

  if (headings.length === 0) {
    return null
  }

  return (
    <nav className="hidden lg:block w-full max-w-64">
      <div className="border-l-2 border-zinc-800 pl-4">
        <h3 className="font-semibold text-zinc-400 mb-3 uppercase tracking-wider">Contents</h3>
        <ul className="space-y-2">
          {headings.map((heading) => (
            <li key={heading.id} className={heading.level > 2 ? "ml-4" : ""}>
              <Link
                href={`#${heading.id}`}
                className={`block text-sm transition-colors hover:${
                  activeId === heading.id ? "text-blue-400" : "text-blue-400/70"
                } ${activeId === heading.id ? "text-blue-400 font-medium" : "text-zinc-500"}`}
                onClick={(e) => {
                  e.preventDefault()
                  // Immediately set active to clicked heading
                  setActiveId(heading.id)

                  // Mark as manual scroll to prevent intermediate updates
                  isManualScrollRef.current = true

                  const element = document.getElementById(heading.id)
                  if (element) {
                    const offset = 80
                    const elementPosition = element.getBoundingClientRect().top
                    const offsetPosition = elementPosition + window.pageYOffset - offset

                    window.scrollTo({
                      top: offsetPosition,
                      behavior: "smooth",
                    })

                    // Detect when scroll completes by checking scroll position
                    const checkScrollComplete = () => {
                      const currentPosition = window.pageYOffset
                      const targetPosition = offsetPosition
                      const distance = Math.abs(currentPosition - targetPosition)

                      if (distance < 5) {
                        // Scroll is complete, re-enable tracking
                        isManualScrollRef.current = false
                      } else {
                        // Check again after a short delay
                        setTimeout(checkScrollComplete, 50)
                      }
                    }

                    // Start checking after a brief delay
                    setTimeout(checkScrollComplete, 100)
                  }
                }}
              >
                {heading.text}
              </Link>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  )
}
