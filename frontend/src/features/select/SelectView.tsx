import { useEffect, useRef, useState, type KeyboardEvent, type MouseEvent, type ReactNode } from 'react'
import { api } from '../../app/client'
import { pickDefaultFrame } from '../../core/frames'
import type { Frame, Job, PromptPoint } from '../../core/types'

const KEY_STEP = 0.02
const clamp01 = (v: number) => Math.min(1, Math.max(0, v))

interface Props {
  job: Job
  /** Called after the selection was sent so the page can resume polling. */
  onSegmenting: () => void
  /** Shown above the selection card, e.g. why the previous run failed. */
  notice?: ReactNode
}

/**
 * Pick the object: click = include, Alt/right-click or "Exclude" mode = exclude.
 * Keyboard: focus the frame, arrows move the cursor, Enter includes, Shift+Enter excludes.
 */
export function SelectView({ job, onSegmenting, notice }: Props) {
  const [frames, setFrames] = useState<Frame[]>([])
  const [loadError, setLoadError] = useState<string | null>(null)
  const [picked, setPicked] = useState<number | null>(null)
  const [showStrip, setShowStrip] = useState(false)
  // Points belong to a single frame: adding on another frame starts a new prompt there.
  const [prompt, setPrompt] = useState<{ frame: number; points: PromptPoint[] }>({ frame: -1, points: [] })
  const [text, setText] = useState(() => (job.settings?.selection as { text?: string } | undefined)?.text ?? '')
  const [excludeMode, setExcludeMode] = useState(false)
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

  const current = picked ?? pickDefaultFrame(frames, job.best_frame)
  const frame = frames[current]
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
    <div className="narrow select">
      {notice}
      <h1 className="select-title">
        <label htmlFor="object-text">What is the object?</label>
      </h1>
      <input
        id="object-text"
        className="select-text"
        type="text"
        value={text}
        maxLength={120}
        placeholder="e.g. black coffee mug"
        autoFocus
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => e.key === 'Enter' && canSegment && segment()}
      />
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
            aria-label="Reference frame. Optional: click the object to add a point. Arrow keys move the cursor, Enter includes a point, Shift+Enter excludes."
            tabIndex={0}
            onClick={(e) => onClick(e)}
            onContextMenu={(e) => {
              e.preventDefault()
              onClick(e, true)
            }}
            onKeyDown={onKeyDown}
          >
            <img src={api.resolveUrl(frame.url)} alt={`Reference frame ${frame.index}`} draggable={false} />
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
      <p className="muted small">
        Optional: click the object to pin it down. Right-click or Alt-click marks background.
        {framePoints.length > 0 && (
          <span className="num">
            {' '}
            {framePoints.filter((p) => p.label === 1).length} include &middot;{' '}
            {framePoints.filter((p) => p.label === 0).length} exclude.{' '}
            <button type="button" className="btn link" onClick={() => setPrompt({ frame: -1, points: [] })}>
              Clear
            </button>
          </span>
        )}
      </p>
      <label className="check">
        <input type="checkbox" checked={excludeMode} onChange={(e) => setExcludeMode(e.target.checked)} />
        <span>Taps mark background (for touch screens)</span>
      </label>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <button type="button" className="btn primary big" disabled={!canSegment} onClick={segment}>
        {busy ? 'Starting…' : 'Find object in all frames'}
      </button>
      {!canSegment && !busy && frame && <p className="muted small">Describe the object or click it to continue.</p>}
      <button
        type="button"
        className="btn link"
        aria-expanded={showStrip}
        aria-controls="frame-strip"
        onClick={() => setShowStrip((v) => !v)}
      >
        {showStrip ? 'Hide other frames' : 'Use a different frame'}
      </button>
      {showStrip && (
        <ul className="strip" id="frame-strip" aria-label="Frames">
          {frames.map((f, i) => (
            <li key={f.index}>
              <button
                type="button"
                aria-pressed={i === current}
                aria-label={`Frame ${f.index}`}
                title={`Frame ${f.index} · sharpness ${f.sharpness.toFixed(0)}`}
                onClick={() => setPicked(i)}
              >
                <img src={api.resolveUrl(f.thumb_url)} alt="" loading="lazy" draggable={false} />
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
