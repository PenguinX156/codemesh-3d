import { lazy, Suspense } from 'react'
import { hrefs } from '../../app/router'
import { formatCount } from '../../core/format'
import { DownloadButton } from '../export/DownloadButton'
import { showcaseUrl, useShowcase } from './manifest'

const ModelViewer = lazy(() => import('../viewer/ModelViewer'))

export function ShowcasePage({ index }: { index: number }) {
  const items = useShowcase()
  const item = items?.[index]

  if (items === null) return <p className="muted pad">Loading…</p>
  if (!item)
    return (
      <p className="pad">
        That showcase item does not exist. <a href={hrefs.home}>Back to home</a>
      </p>
    )

  const url = showcaseUrl(item.glb)
  return (
    <div className="workspace">
      <div className="stage">
        <Suspense fallback={<p className="muted pad">Loading viewer…</p>}>
          <ModelViewer url={url} />
        </Suspense>
      </div>
      <div className="panel">
        <section className="card">
          <h2>{item.title}</h2>
          {item.stats && (
            <dl className="kv num">
              {item.stats.triangles != null && (
                <>
                  <dt>Triangles</dt>
                  <dd>{formatCount(item.stats.triangles)}</dd>
                </>
              )}
              {item.stats.texture_size != null && (
                <>
                  <dt>Texture</dt>
                  <dd>{item.stats.texture_size} px</dd>
                </>
              )}
            </dl>
          )}
          <DownloadButton url={url} filename={item.glb.split('/').pop() ?? 'model.glb'} label="Download GLB" />
          <a className="btn" href={hrefs.home}>
            Back
          </a>
        </section>
      </div>
    </div>
  )
}
