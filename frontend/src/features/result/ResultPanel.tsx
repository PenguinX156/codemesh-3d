import { useEffect, useState } from 'react'
import { api } from '../../app/client'
import { formatCount, formatSeconds, titleCase } from '../../core/format'
import { compareArtifacts, summarize } from '../../core/summary'
import type { Job } from '../../core/types'
import { DownloadButton } from '../export/DownloadButton'

const fixed = (v: number | null, digits: number, unit = '') => (v == null ? 'n/a' : `${v.toFixed(digits)}${unit}`)

type Quality = { grade?: string; passed?: boolean; failed_checks?: string[]; hints?: string[] }

const gradeTone = (g?: string) => (g === 'A' || g === 'B' ? 'ok' : g === 'C' ? 'warn' : 'bad')

/** Finished-job side panel: download first, then headline metrics, quality grade, agent decisions, timings. */
export function ResultPanel({ job, glb }: { job: Job; glb: string | null }) {
  const s = summarize(job)
  const total = job.stages.reduce((sum, st) => sum + (st.seconds ?? 0), 0)
  const quality = job.stages.find((st) => st.name === 'quality')?.metrics as Quality | undefined
  const metrics: { label: string; value: string; hint: string }[] = [
    { label: 'Shape score', value: s.iou == null ? 'n/a' : `${Math.round(s.iou * 100)}%`, hint: `Silhouette IoU ${fixed(s.iou, 3)}` },
    { label: 'Triangles', value: s.triangles == null ? 'n/a' : formatCount(s.triangles), hint: 'Game-ready mesh' },
    { label: 'Texture', value: s.textureSize == null ? 'n/a' : `${s.textureSize}²`, hint: 'Baked albedo' },
    { label: 'Photo match', value: fixed(s.psnr, 1, ' dB'), hint: `Masked SSIM ${fixed(s.ssim, 3)}` },
  ]
  const compares = compareArtifacts(job)

  return (
    <>
      <section className="card result-head" aria-labelledby="result-title">
        <div className="result-title-row">
          <h2 id="result-title">Your 3D asset</h2>
          {quality?.grade && (
            <span className={`grade ${gradeTone(quality.grade)}`} title={quality.passed ? 'All quality checks passed' : 'Some quality checks failed'}>
              Grade {quality.grade}
            </span>
          )}
        </div>
        {glb && <DownloadButton url={glb} filename="model.glb" />}
        <p className="muted small" data-testid="scale-note">
          {s.measuredScale
            ? 'Measured: the size is scaled to the dimension you entered.'
            : 'Size is estimated by the geometry model. Enter one real dimension and reconstruct again for a measured scale.'}
        </p>
      </section>

      <div className="metric-grid" role="list">
        {metrics.map((m) => (
          <div key={m.label} className="metric" role="listitem">
            <span className="metric-label">{m.label}</span>
            <span className="metric-value num">{m.value}</span>
            <span className="metric-hint">{m.hint}</span>
          </div>
        ))}
      </div>

      {quality && ((quality.failed_checks?.length ?? 0) > 0 || (quality.hints?.length ?? 0) > 0) && (
        <section className="card" aria-labelledby="quality-title">
          <h2 id="quality-title">Quality check</h2>
          {(quality.failed_checks?.length ?? 0) > 0 && (
            <p className="small">
              Needs work:{' '}
              {quality.failed_checks!.map((c) => (
                <code key={c}>{c}</code>
              ))}
            </p>
          )}
          {quality.hints?.map((h) => (
            <p key={h} className="small muted hint">
              {h}
            </p>
          ))}
        </section>
      )}

      <AgentCard job={job} />

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
      {report.analysis?.category && (
        <p className="small">
          Detected: <strong>{report.analysis.category}</strong>
        </p>
      )}
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
