import { useCallback, useEffect, useState } from 'react'
import { pollJob } from '../core/jobPoller'
import type { Capabilities, Job } from '../core/types'
import { api } from './client'

export type BackendState =
  | { status: 'checking' }
  | { status: 'offline' }
  | { status: 'online'; version: string; capabilities: Capabilities | null }

/**
 * Runs `start` while the tab is visible and calls the stop() it returns when the tab is hidden or the
 * component unmounts. Background tabs make no requests, so the hosted server can scale to zero.
 */
function whileVisible(start: () => () => void): () => void {
  let stop: (() => void) | undefined
  const sync = () => {
    if (document.hidden) {
      stop?.()
      stop = undefined
    } else if (!stop) {
      stop = start()
    }
  }
  sync()
  document.addEventListener('visibilitychange', sync)
  return () => {
    document.removeEventListener('visibilitychange', sync)
    stop?.()
  }
}

/**
 * Checks /health (and loads /capabilities) when the tab is shown. Once the backend answers it stops;
 * only an offline backend is re-checked every 10 s. An idle open tab must not keep the server awake.
 */
export function useBackend(): BackendState {
  const [state, setState] = useState<BackendState>({ status: 'checking' })
  useEffect(
    () =>
      whileVisible(() => {
        let cancelled = false
        let timer: ReturnType<typeof setTimeout> | undefined
        const check = async () => {
          try {
            const health = await api.health()
            const capabilities = await api.capabilities().catch(() => null)
            if (!cancelled) setState({ status: 'online', version: health.version, capabilities })
            return
          } catch {
            if (!cancelled) setState({ status: 'offline' })
          }
          if (!cancelled) timer = setTimeout(check, 10_000)
        }
        void check()
        return () => {
          cancelled = true
          clearTimeout(timer)
        }
      }),
    [],
  )
  return state
}

/** Live job state. `reload()` restarts polling after a user action (select, reconstruct, refine). */
export function useJob(id: string): { job: Job | null; error: string | null; reload: () => void } {
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  useEffect(
    () =>
      whileVisible(() =>
        pollJob(api, id, {
          onUpdate: (j) => {
            setError(null)
            setJob(j)
          },
          onError: (e) => setError(e instanceof Error ? e.message : String(e)),
        }),
      ),
    [id, tick],
  )

  const reload = useCallback(() => setTick((t) => t + 1), [])
  // Ignore a stale job left over from the previous route id.
  return { job: job?.id === id ? job : null, error, reload }
}
