import { lazy, Suspense } from 'react'
import { api } from '../../app/client'
import { useJob } from '../../app/hooks'
import { hrefs } from '../../app/router'
import { formatCount, formatSeconds } from '../../core/format'
import { DownloadButton } from '../export/DownloadButton'
import { ProgressView } from '../progress/ProgressView'
import { SelectView } from '../select/SelectView'
import { SettingsPanel } from '../settings/SettingsPanel'

// three.js is large; only load it when a model is actually shown.
const ModelViewer = lazy(() => import('../viewer/ModelViewer'))

/** One job; the view follows `job.status` (created -> select -> segment -> reconstruct -> done). */
export function JobPage({ id }: { id: string }) {
  const { job, error, reload } = useJob(id)

  if (!job)
    return (
      <div className="narrow">
        {error ? (
          <p className="error" role="alert">
            Could not load this job: {error}. <a href={hrefs.home}>Back to home</a>
          </p>
        ) : (
          <p className="muted">Loading job…</p>
        )}
      </div>
    )

  const offlineBanner = error && (
    <p className="warn" role="status">
      Connection problem, retrying: {error}
    </p>
  )

  switch (job.status) {
    case 'ready_to_select':
      return (
        <>
          {offlineBanner}
          <SelectView job={job} onSegmenting={reload} />
        </>
      )
    case 'ready_to_reconstruct':
      return (
        <>
          {offlineBanner}
          <SelectView job={job} onSegmenting={reload}>
            <SettingsPanel jobId={job.id} onStarted={reload} />
          </SelectView>
        </>
      )
    case 'done': {
      const glb = job.artifacts.model_glb ? api.artifactUrl(job.id, job.artifacts.model_glb) : null
      const total = job.stages.reduce((s, st) => s + (st.seconds ?? 0), 0)
      return (
        <>
          {offlineBanner}
          <div className="workspace">
            <div className="stage">
              {glb ? (
                <Suspense fallback={<p className="muted pad">Loading viewer…</p>}>
                  <ModelViewer url={glb} />
                </Suspense>
              ) : (
                <p className="error pad" role="alert">
                  The job finished but reported no model_glb artifact.
                </p>
              )}
            </div>
            <div className="panel">
              <section className="card" aria-labelledby="result-title">
                <h2 id="result-title">Result</h2>
                <dl className="kv num">
                  <dt>Total time</dt>
                  <dd>{formatSeconds(total)}</dd>
                  {Object.entries(job.metrics).map(([k, v]) => (
                    <MetricRow key={k} name={k} value={v} />
                  ))}
                </dl>
                {glb && <DownloadButton url={glb} filename="model.glb" />}
              </section>
              <SettingsPanel jobId={job.id} onStarted={reload} submitLabel="Reconstruct again" />
            </div>
          </div>
        </>
      )
    }
    default:
      return (
        <div className="narrow">
          {offlineBanner}
          <ProgressView job={job} />
          {job.status === 'failed' && (
            <p>
              <a className="btn" href={hrefs.home}>
                Back to home
              </a>
            </p>
          )}
        </div>
      )
  }
}

function MetricRow({ name, value }: { name: string; value: unknown }) {
  if (value == null || typeof value === 'object') return null
  const text = typeof value === 'number' ? (Number.isInteger(value) ? formatCount(value) : value.toFixed(3)) : String(value)
  return (
    <>
      <dt>{name.replace(/_/g, ' ')}</dt>
      <dd>{text}</dd>
    </>
  )
}
