import { hrefs } from '../../app/router'
import { formatCount } from '../../core/format'
import { showcaseUrl, useShowcase } from './manifest'

/** Grid of bundled example results; usable without a backend (e.g. the Vercel build). */
export function Showcase() {
  const items = useShowcase()
  return (
    <section aria-labelledby="showcase-title">
      <h2 id="showcase-title">Showcase</h2>
      {items === null && <p className="muted">Loading…</p>}
      {items?.length === 0 && (
        <p className="muted">
          No showcase models bundled yet. Add GLB files and list them in <code>public/showcase/manifest.json</code>.
        </p>
      )}
      {items && items.length > 0 && (
        <ul className="grid">
          {items.map((it, i) => (
            <li key={it.glb}>
              <a className="tile" href={hrefs.showcase(i)}>
                {it.thumbnail ? (
                  <img src={showcaseUrl(it.thumbnail)} alt="" loading="lazy" />
                ) : (
                  <span className="tile-empty" aria-hidden="true" />
                )}
                <span className="tile-title">{it.title}</span>
                {it.stats?.triangles != null && (
                  <span className="muted small num">{formatCount(it.stats.triangles)} triangles</span>
                )}
              </a>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
