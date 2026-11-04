import type { Api } from './api.ts'
import { TERMINAL_STATUSES, type Job } from './types.ts'

export interface PollOptions {
  /** Delay between successful polls. */
  intervalMs?: number
  /** Upper bound of the exponential backoff after errors. */
  maxBackoffMs?: number
  onUpdate: (job: Job) => void
  onError?: (error: unknown) => void
}

/**
 * Poll GET /jobs/{id} until the job is done or failed. Errors back off (x1.6, capped) and
 * never stop the poller: a restarting backend should not strand the UI.
 * Returns `stop()`. Uses only setTimeout, so it works in React Native too.
 */
export function pollJob(api: Api, jobId: string, opts: PollOptions): () => void {
  const interval = opts.intervalMs ?? 1000
  const maxBackoff = opts.maxBackoffMs ?? 8000
  let stopped = false
  let timer: ReturnType<typeof setTimeout> | undefined
  let delay = interval

  const tick = async () => {
    try {
      const job = await api.getJob(jobId)
      if (stopped) return
      delay = interval
      opts.onUpdate(job)
      if (TERMINAL_STATUSES.includes(job.status)) return
    } catch (e) {
      if (stopped) return
      delay = Math.min(maxBackoff, Math.round(delay * 1.6))
      opts.onError?.(e)
    }
    timer = setTimeout(tick, delay)
  }

  void tick()
  return () => {
    stopped = true
    if (timer) clearTimeout(timer)
  }
}
