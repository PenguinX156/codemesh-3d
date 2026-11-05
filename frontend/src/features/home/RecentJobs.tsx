import { useCallback, useEffect, useState } from 'react'
import { api } from '../../app/client'
import { hrefs } from '../../app/router'
import { titleCase } from '../../core/format'
import type { Job } from '../../core/types'

export function RecentJobs() {
  const [jobs, setJobs] = useState<Job[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  const refresh = useCallback(() => {
    api
      .listJobs()
      .then(setJobs)
      .catch((e) => setError(e instanceof Error ? e.message : 'Could not load jobs'))
  }, [])
  useEffect(refresh, [refresh])

  const remove = async (id: string) => {
    if (!window.confirm('Delete this job and all its files?')) return
    try {
      await api.deleteJob(id)
      refresh()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not delete job')
    }
  }

  return (
    <section aria-labelledby="recent-title">
      <h2 id="recent-title">Recent jobs</h2>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {jobs?.length === 0 && <p className="muted">No jobs yet.</p>}
      {jobs && jobs.length > 0 && (
        <ul className="list">
          {jobs.map((j) => (
            <li key={j.id}>
              <a href={hrefs.job(j.id)}>
                <span className="num">{j.id.slice(0, 8)}</span>
                <span className="muted small">
                  {titleCase(j.mode)} &middot; <time dateTime={new Date(j.created_at * 1000).toISOString()}>{new Date(j.created_at * 1000).toLocaleString()}</time>
                </span>
              </a>
              <span className={`pill ${j.status}`}>{titleCase(j.status)}</span>
              <button type="button" className="btn link" onClick={() => remove(j.id)} aria-label={`Delete job ${j.id.slice(0, 8)}`}>
                Delete
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
