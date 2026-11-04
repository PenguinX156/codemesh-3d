import { useState } from 'react'

/** Downloads via fetch + blob so it also works when the backend is on another origin (download= is ignored there). */
export function DownloadButton({ url, filename, label = 'Download model.glb' }: { url: string; filename: string; label?: string }) {
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const download = async () => {
    setBusy(true)
    setError(null)
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
      const href = URL.createObjectURL(await res.blob())
      const a = document.createElement('a')
      a.href = href
      a.download = filename
      a.click()
      URL.revokeObjectURL(href)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Download failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <button type="button" className="btn primary" onClick={download} disabled={busy}>
        {busy ? 'Preparing…' : label}
      </button>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </>
  )
}
