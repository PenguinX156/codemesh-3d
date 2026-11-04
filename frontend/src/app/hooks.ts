import { useCallback, useEffect, useState } from 'react'
import { pollJob } from '../core/jobPoller'
import type { Capabilities, Job } from '../core/types'
import { api } from './client'

export type BackendState =
  | { status: 'checking' }
  | { status: 'offline' }
  | { status: 'online'; version: string; capabilities: Capabilities | null }

/** Checks /health now and every 10 s; loads /capabilities once when the backend is reachable. */
export function useBackend(): BackendState {
  const [state, setState] = useState<BackendState>({ status: 'checking' })
  useEffect(() => {
    let cancelled = false
    const timer: { id?: ReturnType<typeof setTimeout> } = {}
    const check = async () => {
      try {
        const health = await api.health()
        const capabilities = await api.capabilities().catch(() => null)
        if (!cancelled) setState({ status: 'online', version: health.version, capabilities })
      } catch {
        if (!cancelled) setState({ status: 'offline' })
      }
      if (!cancelled) timer.id = setTimeout(check, 10_000)
    }
    void check()
    return () => {
      cancelled = true
      clearTimeout(timer.id)
    }
  }, [])
  return state
}

/** Live job state. `reload()` restarts polling after a user action (select, reconstruct, refine). */
export function useJob(id: string): { job: Job | null; error: string | null; reload: () => void } {
  const [job, setJob] = useState<Job | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [tick, setTick] = useState(0)

  useEffect(() => {
    return pollJob(api, id, {
      onUpdate: (j) => {
        setError(null)
        setJob(j)
      },
      onError: (e) => setError(e instanceof Error ? e.message : String(e)),
    })
  }, [id, tick])

  const reload = useCallback(() => setTick((t) => t + 1), [])
  // Ignore a stale job left over from the previous route id.
  return { job: job?.id === id ? job : null, error, reload }
}
