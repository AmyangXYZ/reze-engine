import type { Metadata } from "next"
import { Geist, Geist_Mono } from "next/font/google"
import "./globals.css"
import { Analytics } from "@vercel/analytics/next"

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
})

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
})

export const metadata: Metadata = {
  title: "Reze Engine",
  description: "WebGPU 3D Engine for real-time 3D anime character MMD model rendering",
  keywords: ["WebGPU", "3D", "Engine", "MMD", "Model", "Rendering", "Animation", "Reze Engine"],
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="dark fixed inset-0 w-full h-full overflow-hidden m-0 p-0">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased fixed inset-0 w-full h-full overflow-hidden m-0 p-0`}
      >
        {children}
        <Analytics />
      </body>
    </html>
  )
}
