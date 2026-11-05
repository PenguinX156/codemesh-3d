import { useState, type ReactNode } from 'react'
import { api } from '../../app/client'
import type { Job } from '../../core/types'
import { SettingsPanel } from '../settings/SettingsPanel'
import { SelectView } from './SelectView'

interface Props {
  job: Job
  /** Resume polling after a request (new selection or reconstruct). */
  reload: () => void
  notice?: ReactNode
}

/** Shows the mask preview; "Looks right" starts reconstruction, "Try again" goes back to the select screen. */
export function MaskReview({ job, reload, notice }: Props) {
  const [retry, setRetry] = useState(false)
  if (retry || !job.artifacts.masks_preview) return <SelectView job={job} onSegmenting={reload} notice={notice} />
  return (
    <div className="narrow select">
      {notice}
      <h1 className="select-title">Is this your object?</h1>
      <div className="frame-wrap">
        <img
          className="mask-preview"
          src={api.artifactUrl(job.id, job.artifacts.masks_preview)}
          alt="Preview of the detected object in sampled frames"
        />
      </div>
      <p className="muted small">The highlighted area should cover the whole object in every frame shown.</p>
      <SettingsPanel jobId={job.id} onStarted={reload} advanced submitLabel="Looks right → Build 3D model" />
      <button type="button" className="btn" onClick={() => setRetry(true)}>
        Try again
      </button>
    </div>
  )
}
