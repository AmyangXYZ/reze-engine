/** Unified binary I/O for PMX/VMD/textures: HTTP(s) or user folder (File map). */

export type AssetReader = {
  readBinary(logicalPath: string): Promise<ArrayBuffer>
}

/** Normalize PMX-style paths: backslashes, trim, strip leading ./ */
export function normalizeAssetPath(p: string): string {
  let s = p.replace(/\\/g, "/").trim()
  if (s.startsWith("./")) s = s.slice(2)
  return s
}

/** Join PMX directory prefix and texture-relative path (both may be ""). */
export function joinAssetPath(baseDir: string, relative: string): string {
  const rel = normalizeAssetPath(relative)
  if (!rel) return normalizeAssetPath(baseDir)
  const base = baseDir.endsWith("/") ? baseDir.slice(0, -1) : baseDir
  if (!base) return rel
  return `${base}/${rel}`
}

/** Same rules as the original engine string split: supports absolute site paths like `/models/a/b.pmx`. */
export function deriveBasePathFromPmxPath(pmxPath: string): string {
  const pathParts = pmxPath.replace(/\\/g, "/").split("/")
  pathParts.pop()
  return pathParts.join("/") + (pathParts.length > 0 ? "/" : "")
}

export function createFetchAssetReader(): AssetReader {
  return {
    async readBinary(logicalPath: string) {
      const r = await fetch(logicalPath)
      if (!r.ok) throw new Error(`Failed to fetch ${logicalPath}: ${r.status} ${r.statusText}`)
      return r.arrayBuffer()
    },
  }
}

/** Keys must be normalized paths relative to the selected folder root (see fileListToMap). */
export function createFileMapAssetReader(files: Map<string, File>): AssetReader {
  return {
    async readBinary(logicalPath: string) {
      const key = normalizeAssetPath(logicalPath)
      let file = files.get(key)
      if (!file) {
        const lower = key.toLowerCase()
        for (const [k, f] of files) {
          if (k.toLowerCase() === lower) {
            file = f
            break
          }
        }
      }
      if (!file) {
        // Last resort: basename match. Mobile pickers can't select folders — a flat
        // multi-file selection loses directory structure, so `tex/body.png` must
        // find a bare `body.png`. Also rescues wrongly-cased directory names.
        const base = key.split("/").pop()!.toLowerCase()
        for (const [k, f] of files) {
          if (k.split("/").pop()!.toLowerCase() === base) {
            file = f
            break
          }
        }
      }
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
    return pa.localeCompare(pb)
  })
  return list[0] ?? null
}
