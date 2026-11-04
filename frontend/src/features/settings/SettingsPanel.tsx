import { useState } from 'react'
import { api } from '../../app/client'
import { DEFAULT_SETTINGS, PRESETS, TEXTURE_SIZES, parseTriangleBudget } from '../../core/presets'
import type { Axis, ReconstructSettings } from '../../core/types'

interface Props {
  jobId: string
  /** Called after the request was accepted so the page can resume polling. */
  onStarted: () => void
  submitLabel?: string
}

/** Reconstruction settings draft + submit. Cached stages mean re-running with new texture settings is cheap. */
export function SettingsPanel({ jobId, onStarted, submitLabel = 'Reconstruct' }: Props) {
  const [settings, setSettings] = useState<ReconstructSettings>(DEFAULT_SETTINGS)
  const [fixedBudget, setFixedBudget] = useState(false)
  const [budgetText, setBudgetText] = useState('20000')
  const [useDim, setUseDim] = useState(false)
  const [axis, setAxis] = useState<Axis>('y')
  const [meters, setMeters] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const meterValue = Number(meters)
  const dimInvalid = useDim && !(meterValue > 0)

  const submit = async () => {
    setBusy(true)
    setError(null)
    try {
      await api.reconstruct(jobId, {
        ...settings,
        triangle_budget: fixedBudget ? parseTriangleBudget(budgetText) : 'auto',
        known_dimension: useDim ? { axis, meters: meterValue } : undefined,
      })
      onStarted()
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not start reconstruction')
    } finally {
      setBusy(false)
    }
  }

  return (
    <section className="card" aria-labelledby="settings-title">
      <h2 id="settings-title">Settings</h2>

      <fieldset className="segmented">
        <legend>Quality preset</legend>
        {PRESETS.map((p) => (
          <label key={p.id}>
            <input
              type="radio"
              name="preset"
              checked={settings.preset === p.id}
              onChange={() => setSettings({ ...settings, preset: p.id })}
            />
            <span>{p.label}</span>
          </label>
        ))}
      </fieldset>
      <p className="muted small">
        {PRESETS.find((p) => p.id === settings.preset)?.hint}{' '}
        <span className="num">{PRESETS.find((p) => p.id === settings.preset)?.frames} frames.</span>
      </p>

      <fieldset className="segmented">
        <legend>Triangle budget</legend>
        <label>
          <input
            type="radio"
            name="budget"
            checked={!fixedBudget}
            onChange={() => setFixedBudget(false)}
          />
          <span>Auto</span>
        </label>
        <label>
          <input
            type="radio"
            name="budget"
            checked={fixedBudget}
            onChange={() => setFixedBudget(true)}
          />
          <span>Fixed</span>
        </label>
      </fieldset>
      {fixedBudget && (
        <label className="field">
          <span>Triangles</span>
          <input
            type="number"
            inputMode="numeric"
            className="num"
            min={500}
            step={1000}
            value={budgetText}
            onChange={(e) => setBudgetText(e.target.value)}
          />
        </label>
      )}
      {!fixedBudget && <p className="muted small">Chosen from the object&apos;s complexity.</p>}

      <fieldset className="segmented">
        <legend>Texture size</legend>
        {TEXTURE_SIZES.map((t) => (
          <label key={t}>
            <input
              type="radio"
              name="tex"
              checked={settings.texture_size === t}
              onChange={() => setSettings({ ...settings, texture_size: t })}
            />
            <span className="num">{t} px</span>
          </label>
        ))}
      </fieldset>

      <label className="check">
        <input type="checkbox" checked={useDim} onChange={(e) => setUseDim(e.target.checked)} />
        <span>I know one real-world dimension</span>
      </label>
      {useDim && (
        <div className="row">
          <label className="field">
            <span>Axis</span>
            <select value={axis} onChange={(e) => setAxis(e.target.value as Axis)}>
              <option value="x">X (width)</option>
              <option value="y">Y (height)</option>
              <option value="z">Z (depth)</option>
            </select>
          </label>
          <label className="field">
            <span>Meters</span>
            <input
              type="number"
              inputMode="decimal"
              className="num"
              min={0}
              step="any"
              value={meters}
              aria-invalid={dimInvalid}
              onChange={(e) => setMeters(e.target.value)}
            />
          </label>
        </div>
      )}

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <button type="button" className="btn primary" disabled={busy || dimInvalid} onClick={submit}>
        {busy ? 'Starting…' : submitLabel}
      </button>
    </section>
  )
}
