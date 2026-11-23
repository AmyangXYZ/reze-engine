/** Unified binary I/O for PMX/VMD/textures: HTTP(s) or user folder (File map). */

export type AssetReader = {
  readBinary(logicalPath: string): Promise<ArrayBuffer>
}

/** Normalize PMX-style paths: backslashes, trim, NFC, and `.` / `..` segments
 *  resolved. NFC because macOS hands over file names decomposed (NFD) while a
 *  PMX spells them composed: ド in a folder name would otherwise never match. */
export function normalizeAssetPath(p: string): string {
  const s = p.replace(/\\/g, "/").trim().normalize("NFC")
  const out: string[] = []
  const parts = s.split("/")
  parts.forEach((part) => {
    if (part === ".") return
    if (part === ".." && out.length && out[out.length - 1] !== ".." && out[out.length - 1] !== "") out.pop()
    else out.push(part)
  })
  return out.join("/")
}

/** Join PMX directory prefix and texture-relative path (both may be ""). */
export function joinAssetPath(baseDir: string, relative: string): string {
  const rel = normalizeAssetPath(relative)
  if (!rel) return normalizeAssetPath(baseDir)
  const base = baseDir.endsWith("/") ? baseDir.slice(0, -1) : baseDir
  if (!base) return rel
  return normalizeAssetPath(`${base}/${rel}`)
}

/** Same rules as the original engine string split: supports absolute site paths like `/models/a/b.pmx`. */
export function deriveBasePathFromPmxPath(pmxPath: string): string {
  const pathParts = pmxPath.replace(/\\/g, "/").split("/")
  pathParts.pop()
  return pathParts.join("/") + (pathParts.length > 0 ? "/" : "")
}

const fetchReader: AssetReader = {
  async readBinary(logicalPath: string) {
    const r = await fetch(logicalPath)
    if (!r.ok) throw new Error(`Failed to fetch ${logicalPath}: ${r.status} ${r.statusText}`)
    return r.arrayBuffer()
  },
}

/** The one reader for URLs: a URL names the same file whoever asks. */
export function createFetchAssetReader(): AssetReader {
  return fetchReader
}

/** Keys must be normalized paths relative to the selected folder root (see fileListToMap). */
export function createFileMapAssetReader(files: Map<string, File>): AssetReader {
  return {
    async readBinary(logicalPath: string) {
      const key = normalizeAssetPath(logicalPath)
      // The path as written, then in any case (PMX paths are Windows paths),
      // then the bare name: mobile pickers cannot select folders, so a flat
      // selection's `body.png` has to answer for `tex/body.png`. The two
      // fallbacks answer only when ONE file matches. A model can hold hair.png,
      // spa/hair.png and toon/hair.png, and picking among them by upload order
      // is how a sphere map got drawn as hair.
      const lower = key.toLowerCase()
      const base = lower.split("/").pop()!
      const only = (match: (k: string) => boolean) => {
        const hits = [...files.keys()].filter(match)
        if (hits.length > 1) console.warn(`[reze] ${key} is ambiguous in this folder: ${hits.join(", ")}`)
        return hits.length === 1 ? files.get(hits[0]) : undefined
      }
      const file =
        files.get(key) ?? only((k) => k.toLowerCase() === lower) ?? only((k) => k.toLowerCase().split("/").pop() === base)
      if (!file) throw new Error(`Missing file in folder: ${key}`)
      return file.arrayBuffer()
    },
  }
}

export function fileListToMap(files: FileList | File[]): Map<string, File> {
  const m = new Map<string, File>()
  for (const f of Array.from(files)) {
    // `||`, NOT `??`: files from a non-directory input (flat multi-select, drag
    // & drop) have webkitRelativePath === "" — `"" ?? name` keeps the empty
    // string, which keyed EVERY file at "" (each overwriting the last) and broke
    // all flat uploads ("no pmx" / textureless white models).
    const rel = (f as File & { webkitRelativePath?: string }).webkitRelativePath || f.name
    m.set(normalizeAssetPath(rel), f)
  }
  return m
}

export function findFirstPmxFileInList(files: FileList | File[]): File | null {
  const list = Array.from(files).filter((f) => f.name.toLowerCase().endsWith(".pmx"))
  if (list.length === 0) return null
  list.sort((a, b) => {
    const pa = (a as File & { webkitRelativePath?: string }).webkitRelativePath || a.name
    const pb = (b as File & { webkitRelativePath?: string }).webkitRelativePath || b.name
    return pa < pb ? -1 : pa > pb ? 1 : 0
  })
  return list[0] ?? null
}
