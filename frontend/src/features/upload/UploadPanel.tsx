import { useRef, useState, type DragEvent } from 'react'
import { api } from '../../app/client'
import { hrefs, navigate } from '../../app/router'
import type { JobMode } from '../../core/types'
import { Disclosure } from './Disclosure'
import { validateFiles } from './validateFiles'

const ACK_KEY = 'codemesh.disclosure-ack'

const readAck = () => {
  try {
    return localStorage.getItem(ACK_KEY) === '1'
  } catch {
    return false
  }
}

export function UploadPanel({ disabled }: { disabled: boolean }) {
  const [files, setFiles] = useState<File[]>([])
  const [mode, setMode] = useState<JobMode>('walkaround')
  const [ack, setAck] = useState(readAck)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const input = useRef<HTMLInputElement>(null)

  const problem = files.length ? validateFiles(files) : null
  const canSubmit = !disabled && !busy && ack && files.length > 0 && !problem

  const pick = (list: FileList | null) => {
    setError(null)
    setFiles(list ? Array.from(list) : [])
  }

  const onDrop = (e: DragEvent) => {
    e.preventDefault()
    setDragging(false)
    pick(e.dataTransfer.files)
  }

  const toggleAck = (v: boolean) => {
    setAck(v)
    try {
      localStorage.setItem(ACK_KEY, v ? '1' : '0')
    } catch {
      /* storage unavailable: the checkbox still works for this session */
    }
  }

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      const job = await api.createJob(files, mode)
      api.warmup().catch(() => {}) // warm the GPU while ingest/segmentation prepare
      navigate(hrefs.job(job.id))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed')
      setBusy(false)
    }
  }

  return (
    <section className="card" aria-labelledby="upload-title">
      <h2 id="upload-title">New capture</h2>

      <button
        type="button"
        className={`dropzone${dragging ? ' is-dragging' : ''}`}
        onClick={() => input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        disabled={disabled}
      >
        {files.length === 0 ? (
          <>
            <span className="dropzone-title">Drop a video or photos here</span>
            <span className="muted">or press to browse &middot; one video, or many photos</span>
          </>
        ) : (
          <>
            <span className="dropzone-title">
              {files.length === 1 ? files[0].name : `${files.length} files selected`}
            </span>
            <span className="muted num">
              {(files.reduce((s, f) => s + f.size, 0) / 1e6).toFixed(1)} MB &middot; press to change
            </span>
          </>
        )}
      </button>
      <input
        ref={input}
        type="file"
        accept="video/*,image/*"
        multiple
        hidden
        onChange={(e) => pick(e.target.files)}
      />

      <fieldset className="segmented">
        <legend>Capture style</legend>
        <label>
          <input type="radio" name="mode" checked={mode === 'walkaround'} onChange={() => setMode('walkaround')} />
          <span>Walk-around</span>
        </label>
        <label>
          <input type="radio" name="mode" checked={mode === 'turntable'} onChange={() => setMode('turntable')} />
          <span>Turntable</span>
        </label>
      </fieldset>
      <p className="muted small">
        {mode === 'walkaround'
          ? 'You moved the camera around a stationary object.'
          : 'The object rotated in front of a fixed camera.'}
      </p>

      <Disclosure />
      <label className="check">
        <input type="checkbox" checked={ack} onChange={(e) => toggleAck(e.target.checked)} />
        <span>I understand and my upload contains no personal or sensitive content.</span>
      </label>

      {(problem || error) && (
        <p className="error" role="alert">
          {problem ?? error}
        </p>
      )}
      <button type="button" className="btn primary" disabled={!canSubmit} onClick={submit}>
        {busy ? 'Uploading…' : 'Upload and start'}
      </button>
      {disabled && <p className="muted small">The backend is offline, so uploading is unavailable.</p>}
    </section>
  )
}
