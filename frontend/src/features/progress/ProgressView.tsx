import { formatPercent, formatSeconds, titleCase } from '../../core/format'
import type { Job, StageStatus } from '../../core/types'

const STATUS_LABEL: Record<StageStatus, string> = {
  pending: 'Waiting',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
  skipped: 'Skipped',
}

/** Overall bar, current message, and the per-stage list with timings and the error, if any. */
export function ProgressView({ job }: { job: Job }) {
  const failed = job.status === 'failed'
  return (
    <section className="card progress" aria-labelledby="progress-title">
      <h2 id="progress-title">{failed ? 'Processing failed' : titleCase(job.stage ?? job.status)}</h2>

      <div
        className="bar"
        role="progressbar"
        aria-label="Overall progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(job.progress * 100)}
      >
        <div className={`bar-fill${failed ? ' failed' : ''}`} style={{ width: formatPercent(job.progress) }} />
      </div>
      <p className="num small">
        {formatPercent(job.progress)} <span className="muted" aria-live="polite">&middot; {job.message}</span>
      </p>

      {job.error && (
        <p className="error" role="alert">
          {job.error}
        </p>
      )}

      <ol className="stages">
        {job.stages.map((s) => (
          <li key={s.name} className={`stage-row ${s.status}`}>
            <span className="dot" aria-hidden="true" />
            <span className="stage-name">{titleCase(s.name)}</span>
            <span className="muted small">{STATUS_LABEL[s.status]}</span>
            <span className="num small stage-secs">{formatSeconds(s.seconds)}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
