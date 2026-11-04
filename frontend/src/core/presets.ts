import type { Preset, ReconstructSettings } from './types.ts'

export interface PresetInfo {
  id: Preset
  label: string
  /** Frames the pipeline samples from the capture (docs/PLAN.md, ingest). */
  frames: number
  hint: string
}

export const PRESETS: readonly PresetInfo[] = [
  { id: 'draft', label: 'Draft', frames: 24, hint: 'Fastest. Check framing and masks.' },
  { id: 'standard', label: 'Standard', frames: 40, hint: 'Balanced quality and time.' },
  { id: 'high', label: 'High', frames: 64, hint: 'Most frames, best detail. Slowest.' },
]

export const TEXTURE_SIZES = [2048, 4096] as const

export const DEFAULT_SETTINGS: ReconstructSettings = {
  preset: 'standard',
  triangle_budget: 'auto',
  texture_size: 2048,
}

export const MIN_TRIANGLES = 500
export const MAX_TRIANGLES = 2_000_000

/** Clamp a user-typed triangle budget; returns 'auto' when the value is not a usable number. */
export function parseTriangleBudget(raw: string): ReconstructSettings['triangle_budget'] {
  const n = Math.round(Number(raw))
  if (!Number.isFinite(n) || n <= 0) return 'auto'
  return Math.min(MAX_TRIANGLES, Math.max(MIN_TRIANGLES, n))
}
