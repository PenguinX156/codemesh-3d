import { lazy, Suspense } from 'react'
import { api } from '../../app/client'
import { useJob } from '../../app/hooks'
import { hrefs } from '../../app/router'
import { ResultPanel } from '../result/ResultPanel'
import { ProgressView } from '../progress/ProgressView'
import { MaskReview } from '../select/MaskReview'
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
          <MaskReview job={job} reload={reload} />
        </>
      )
    case 'done': {
      const glb = job.artifacts.model_glb ? api.artifactUrl(job.id, job.artifacts.model_glb) : null
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
              <ResultPanel job={job} glb={glb} />
              <SettingsPanel jobId={job.id} onStarted={reload} submitLabel="Reconstruct again" />
            </div>
          </div>
        </>
      )
    }
    default:
      // A failed run keeps the ingested frames, so let the user fix the selection or settings and retry.
      if (job.status === 'failed' && job.artifacts.frames_json)
        return (
          <>
            {offlineBanner}
            <MaskReview job={job} reload={reload} notice={<ProgressView job={job} />} />
          </>
        )
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
