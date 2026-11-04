import { useEffect, useState } from 'react'

/** One pre-processed result bundled under public/showcase/. Paths are relative to /showcase/. */
export interface ShowcaseItem {
  title: string
  glb: string
  thumbnail?: string
  stats?: { triangles?: number; vertices?: number; texture_size?: number; seconds?: number }
}

export const showcaseUrl = (path: string) => `${import.meta.env.BASE_URL}showcase/${path.replace(/^\/+/, '')}`

/** Accepts either `[...]` or `{ "items": [...] }`. Returns [] for a missing or malformed manifest. */
export async function loadManifest(): Promise<ShowcaseItem[]> {
  try {
    const res = await fetch(showcaseUrl('manifest.json'), { cache: 'no-cache' })
    if (!res.ok) return []
    const body = await res.json()
    const list: unknown = Array.isArray(body) ? body : body?.items
    return Array.isArray(list)
      ? list.filter((i): i is ShowcaseItem => typeof i?.title === 'string' && typeof i?.glb === 'string')
      : []
  } catch {
    return []
  }
}

/** `null` while loading. */
export function useShowcase(): ShowcaseItem[] | null {
  const [items, setItems] = useState<ShowcaseItem[] | null>(null)
  useEffect(() => {
    let cancelled = false
    void loadManifest().then((i) => !cancelled && setItems(i))
    return () => {
      cancelled = true
    }
  }, [])
  return items
}
