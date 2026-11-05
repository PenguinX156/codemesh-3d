import type { Frame } from './types.ts'

/**
 * Position in `frames` to show first: the backend's recommendation (`bestIndex`, else the frame flagged `best`) when present,
 * else the sharpest frame in the middle 60% of the list (ends of a capture are usually transitions).
 */
export function pickDefaultFrame(frames: Frame[], bestIndex?: number | null): number {
  const given = frames.findIndex((f) => (bestIndex == null ? f.best : f.index === bestIndex))
  if (given >= 0) return given
  if (!frames.length) return 0
  const lo = Math.floor(frames.length * 0.2)
  const hi = Math.max(lo + 1, Math.ceil(frames.length * 0.8))
  let best = lo
  for (let i = lo; i < hi; i++) if (frames[i].sharpness > frames[best].sharpness) best = i
  return best
}
