import { useBackend, type BackendState } from './app/hooks'
import { hrefs, useRoute } from './app/router'
import { HomePage } from './features/home/HomePage'
import { JobPage } from './features/job/JobPage'
import { ShowcasePage } from './features/showcase/ShowcasePage'

function BackendBadge({ backend }: { backend: BackendState }) {
  const cap = backend.status === 'online' ? backend.capabilities : null
  const label =
    backend.status === 'checking' ? 'Checking backend' : backend.status === 'offline' ? 'Backend offline' : 'Backend online'
  return (
    <span className={`badge ${backend.status}`} role="status">
      <span className="dot" aria-hidden="true" />
      {label}
      {cap && (
        <span className="muted small">
          {' '}
          &middot; {cap.geometry_engine} &middot; {cap.device}
        </span>
      )}
    </span>
  )
}

export default function App() {
  const route = useRoute()
  const backend = useBackend()

  return (
    <>
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <a className="brand" href={hrefs.home}>
          CodeMesh <span>3D</span>
        </a>
        <BackendBadge backend={backend} />
      </header>
      <main id="main" tabIndex={-1}>
        {route.name === 'home' && <HomePage backend={backend} />}
        {route.name === 'job' && <JobPage key={route.id} id={route.id} />}
        {route.name === 'showcase' && <ShowcasePage index={route.index} />}
      </main>
    </>
  )
}
