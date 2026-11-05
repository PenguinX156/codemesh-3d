import type { Job, MetricValue } from './types.ts'

export interface ResultSummary {
  iou: number | null
  psnr: number | null
  ssim: number | null
  triangles: number | null
  textureSize: number | null
  engine: string | null
  /** True when the user gave a real-world dimension, so sizes are measured rather than estimated. */
  measuredScale: boolean
}

const num = (v: MetricValue | undefined): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : null)

/** Pull the headline numbers out of `job.metrics` and every stage's metrics (stage metrics win on clashes). */
export function summarize(job: Job): ResultSummary {
  const m: Record<string, MetricValue> = { ...job.metrics }
  for (const s of job.stages) Object.assign(m, s.metrics)
  const geometry = job.stages.find((s) => s.name === 'geometry')?.metrics
  const engine = geometry?.engine ?? m.engine
  const known = (job.settings?.reconstruct as { known_dimension?: unknown } | undefined)?.known_dimension
  return {
    iou: num(m.iou_holdout_mean) ?? num(m.iou_mean),
    psnr: num(m.psnr_mean),
    ssim: num(m.ssim_mean),
    triangles: num(m.triangles),
    textureSize: num(m.texture_size),
    engine: typeof engine === 'string' ? engine : null,
    measuredScale: !!known,
  }
}

/** Artifact entries whose key starts with `compare`, in key order (evaluate's render-vs-photo images). */
export function compareArtifacts(job: Job): [string, string][] {
  return Object.entries(job.artifacts)
    .filter((e): e is [string, string] => e[0].startsWith('compare') && typeof e[1] === 'string')
    .sort(([a], [b]) => a.localeCompare(b, undefined, { numeric: true }))
}
