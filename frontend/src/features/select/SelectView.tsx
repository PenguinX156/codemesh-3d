import { useEffect, useRef, useState, type KeyboardEvent, type MouseEvent, type ReactNode } from 'react'
import { api } from '../../app/client'
import type { Frame, Job, PromptPoint } from '../../core/types'

const KEY_STEP = 0.02
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

interface Props {
  job: Job
  /** Called after the selection was sent so the page can resume polling. */
  onSegmenting: () => void
  /** Extra panel content, e.g. reconstruct settings once masks exist. */
  children?: ReactNode
  /** Shown above the selection card, e.g. why the previous run failed. */
  notice?: ReactNode
}

/**
 * Pick the object: click = include, Alt/right-click or "Exclude" mode = exclude.
 * Keyboard: focus the frame, arrows move the cursor, Enter includes, Shift+Enter excludes.
 */
export function SelectView({ job, onSegmenting, children, notice }: Props) {
  const [frames, setFrames] = useState<Frame[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [current, setCurrent] = useState(0)
  // Points belong to a single frame: adding on another frame starts a new prompt there.
  const [prompt, setPrompt] = useState<{ frame: number; points: PromptPoint[] }>({ frame: -1, points: [] })
  const [text, setText] = useState('')
  const [excludeMode, setExcludeMode] = useState(false)
  const [showMask, setShowMask] = useState(true)
  const [cursor, setCursor] = useState({ x: 0.5, y: 0.5 })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const stage = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let cancelled = false
    api
      .getFrames(job.id)
      .then((f) => !cancelled && setFrames(f))
      .catch((e) => !cancelled && setLoadError(e instanceof Error ? e.message : 'Could not load frames'))
    return () => {
      cancelled = true
    }
  }, [job.id])

  const frame = frames[current]
  const maskUrl = job.artifacts.masks_preview ? api.artifactUrl(job.id, job.artifacts.masks_preview) : null
  const framePoints = prompt.frame === frame?.index ? prompt.points : []

  const addPoint = (x: number, y: number, label: 1 | 0) => {
    if (!frame) return
    setPrompt({ frame: frame.index, points: [...framePoints, { x: clamp01(x), y: clamp01(y), label }] })
  }

  const onClick = (e: MouseEvent, forceExclude = false) => {
    const rect = stage.current?.getBoundingClientRect()
    if (!rect) return
    const exclude = forceExclude || e.altKey || excludeMode
    addPoint((e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height, exclude ? 0 : 1)
  }

  const onKeyDown = (e: KeyboardEvent) => {
    const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key]
    if (d) {
      e.preventDefault()
      setCursor((c) => ({ x: clamp01(c.x + d[0] * KEY_STEP), y: clamp01(c.y + d[1] * KEY_STEP) }))
    } else if (e.key === 'Enter') {
      e.preventDefault()
      addPoint(cursor.x, cursor.y, e.shiftKey || excludeMode ? 0 : 1)
    }
  }

  const canSegment = !busy && !!frame && (framePoints.some((p) => p.label === 1) || text.trim().length > 0)

  const segment = async () => {
    if (!frame) return
    setBusy(true)
    setError(null)
    try {
      await api.select(job.id, { frame_index: frame.index, points: framePoints, text: text.trim() || undefined })
      onSegmenting()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Segmentation request failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="workspace">
      <div className="stage">
        {loadError && (
          <p className="error" role="alert">
            {loadError}
          </p>
        )}
        {!frame && !loadError && <p className="muted pad">Loading frames…</p>}
        {frame && (
          <div className="frame-wrap">
            <div
              ref={stage}
              className="frame-stage"
              role="application"
              aria-label="Frame view. Arrow keys move the cursor, Enter includes a point, Shift+Enter excludes."
              tabIndex={0}
              onClick={(e) => onClick(e)}
              onContextMenu={(e) => {
                e.preventDefault()
                onClick(e, true)
              }}
              onKeyDown={onKeyDown}
            >
              <img src={api.resolveUrl(frame.url)} alt={`Frame ${frame.index}`} draggable={false} />
              {showMask && maskUrl && <img className="mask-overlay" src={maskUrl} alt="" draggable={false} />}
              {framePoints.map((p, i) => (
                <span
                  key={i}
                  className={`point ${p.label ? 'include' : 'exclude'}`}
                  style={{ left: `${p.x * 100}%`, top: `${p.y * 100}%` }}
                />
              ))}
              <span className="kbd-cursor" style={{ left: `${cursor.x * 100}%`, top: `${cursor.y * 100}%` }} />
            </div>
          </div>
        )}
        <ul className="strip" aria-label="Frames">
          {frames.map((f, i) => (
            <li key={f.index}>
              <button
                type="button"
                aria-pressed={i === current}
                aria-label={`Frame ${f.index}`}
                title={`Frame ${f.index} · sharpness ${f.sharpness.toFixed(0)}`}
                onClick={() => setCurrent(i)}
              >
                <img src={api.resolveUrl(f.thumb_url)} alt="" loading="lazy" draggable={false} />
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="panel">
        {notice}
        <section className="card" aria-labelledby="select-title">
          <h2 id="select-title">Select the object</h2>
          <p className="muted small">
            Click the object to include it. Alt-click or right-click marks background to exclude. On touch, switch to
            Exclude.
          </p>
          <fieldset className="segmented">
            <legend>Point type</legend>
            <label>
              <input type="radio" name="pt" checked={!excludeMode} onChange={() => setExcludeMode(false)} />
              <span>Include</span>
            </label>
            <label>
              <input type="radio" name="pt" checked={excludeMode} onChange={() => setExcludeMode(true)} />
              <span>Exclude</span>
            </label>
          </fieldset>
          <label className="field">
            <span>Description (optional)</span>
            <input
              type="text"
              value={text}
              maxLength={120}
              placeholder="e.g. ceramic mug"
              onChange={(e) => setText(e.target.value)}
            />
          </label>
          <p className="num small">
            {framePoints.filter((p) => p.label === 1).length} include &middot;{' '}
            {framePoints.filter((p) => p.label === 0).length} exclude
            <button type="button" className="btn link" onClick={() => setPrompt({ frame: -1, points: [] })} disabled={!framePoints.length}>
              Clear
            </button>
          </p>
          {maskUrl && (
            <label className="check">
              <input type="checkbox" checked={showMask} onChange={(e) => setShowMask(e.target.checked)} />
              <span>Show mask preview</span>
            </label>
          )}
          {error && (
            <p className="error" role="alert">
              {error}
            </p>
          )}
          <button type="button" className="btn primary" disabled={!canSegment} onClick={segment}>
            {busy ? 'Starting…' : maskUrl ? 'Re-run segmentation' : 'Segment object'}
          </button>
        </section>
        {children}
      </div>
    </div>
  )
}
