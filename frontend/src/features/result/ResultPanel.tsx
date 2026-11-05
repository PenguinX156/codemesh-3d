import { useEffect, useState } from 'react'
import { api } from '../../app/client'
import { formatCount, formatSeconds, titleCase } from '../../core/format'
import { compareArtifacts, summarize } from '../../core/summary'
import type { Job } from '../../core/types'
import { DownloadButton } from '../export/DownloadButton'

const fixed = (v: number | null, digits: number, unit = '') => (v == null ? 'n/a' : `${v.toFixed(digits)}${unit}`)

/** Finished-job side panel: headline metrics, per-stage timings, scale note, download and held-out comparisons. */
export function ResultPanel({ job, glb }: { job: Job; glb: string | null }) {
  const s = summarize(job)
  const total = job.stages.reduce((sum, st) => sum + (st.seconds ?? 0), 0)
  const rows: [string, string][] = [
    ['Silhouette IoU', fixed(s.iou, 3)],
    ['Masked PSNR', fixed(s.psnr, 1, ' dB')],
    ['Masked SSIM', fixed(s.ssim, 3)],
    ['Triangles', s.triangles == null ? 'n/a' : formatCount(s.triangles)],
    ['Texture', s.textureSize == null ? 'n/a' : `${s.textureSize} × ${s.textureSize}`],
    ['Geometry engine', s.engine ?? 'n/a'],
  ]
  const compares = compareArtifacts(job)

  return (
    <>
      <section className="card" aria-labelledby="result-title">
        <h2 id="result-title">Result</h2>
        <dl className="kv num">
          {rows.map(([k, v]) => (
            <Row key={k} name={k} value={v} />
          ))}
        </dl>
        <p className="muted small" data-testid="scale-note">
          {s.measuredScale
            ? 'Measured: the size is scaled to the dimension you entered.'
            : 'Estimated: with no known dimension the size is the geometry model’s estimate, not a measurement. Enter one real dimension and reconstruct again to get a measured scale.'}{' '}
          Quality scores are measured on held-out photos against the SAM masks and the photos themselves.
        </p>
        {glb && <DownloadButton url={glb} filename="model.glb" />}
      </section>

      <section className="card" aria-labelledby="timings-title">
        <h2 id="timings-title">Stage timings</h2>
        <dl className="kv num">
          {job.stages.map((st) => (
            <Row key={st.name} name={titleCase(st.name)} value={st.status === 'done' ? formatSeconds(st.seconds) : st.status} />
          ))}
          <Row name="Total" value={formatSeconds(total)} />
        </dl>
      </section>

      {compares.length > 0 && (
        <section className="card" aria-labelledby="compare-title">
          <h2 id="compare-title">Render vs photo</h2>
          <p className="muted small">Left to right: photo, texture render, silhouette check (green = match, red = render only, blue = mask only).</p>
          <div className="compare-list">
            {compares.map(([key, path]) => (
              <a key={key} href={api.artifactUrl(job.id, path)} target="_blank" rel="noreferrer">
                <img src={api.artifactUrl(job.id, path)} alt={`Comparison ${titleCase(key)}`} loading="lazy" />
              </a>
            ))}
          </div>
        </section>
      )}

      <AgentCard job={job} />
    </>
  )
}

type AgentReport = {
  analysis?: { category?: string; plan?: string } | null
  tool_calls?: { round: number; name: string; arguments: Record<string, unknown>; result: string }[]
}

/** Muse Spark agent decisions (agent/report.json): category, plan and every tool call it made. */
function AgentCard({ job }: { job: Job }) {
  const path = job.artifacts.agent_report
  const [report, setReport] = useState<AgentReport | null>(null)
  useEffect(() => {
    if (!path) return
    fetch(api.artifactUrl(job.id, path))
      .then((r) => (r.ok ? r.json() : null))
      .then(setReport)
      .catch(() => setReport(null))
  }, [job.id, path, job.status])
  if (!report) return null
  return (
    <section className="card" aria-labelledby="agent-title">
      <h2 id="agent-title">Muse Spark agent</h2>
      <dl className="kv">
        <Row name="Category" value={report.analysis?.category ?? 'n/a'} />
      </dl>
      {report.analysis?.plan && <p className="small">{report.analysis.plan}</p>}
      <ol className="small">
        {(report.tool_calls ?? []).map((c, i) => {
          const { rationale, ...args } = c.arguments
          return (
            <li key={i}>
              <code>{`${c.name}(${Object.entries(args).map(([k, v]) => `${k}=${String(v)}`).join(', ')})`}</code>{' '}
              <span className="muted">round {c.round}: {String(rationale ?? c.result)}</span>
            </li>
          )
        })}
      </ol>
    </section>
  )
}

function Row({ name, value }: { name: string; value: string }) {
  return (
    <>
      <dt>{name}</dt>
      <dd>{value}</dd>
    </>
  )
}
