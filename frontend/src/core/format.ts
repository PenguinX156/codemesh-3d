/** Small pure formatters shared by all UIs. */

export function formatCount(n: number): string {
  return Math.round(n).toLocaleString('en-US')
}

export function formatSeconds(s: number | null | undefined): string {
  if (s == null) return ''
  if (s < 60) return `${s.toFixed(1)} s`
  return `${Math.floor(s / 60)} m ${Math.round(s % 60)} s`
}

export function formatPercent(p: number): string {
  return `${Math.round(Math.min(1, Math.max(0, p)) * 100)}%`
}

export function titleCase(s: string): string {
  return s.replace(/_/g, ' ').replace(/^\w/, (c) => c.toUpperCase())
}
