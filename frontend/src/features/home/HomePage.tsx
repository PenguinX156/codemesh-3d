import type { BackendState } from '../../app/hooks'
import { Showcase } from '../showcase/Showcase'
import { UploadPanel } from '../upload/UploadPanel'
import { RecentJobs } from './RecentJobs'

const STEPS = [
  { title: 'Capture', body: 'Walk around the object with your phone.', credit: 'MapAnything geometry' },
  { title: 'Describe', body: 'Click the object once; the mask follows it through the video.', credit: 'SAM 3.1 masks' },
  { title: 'Get the asset', body: 'An agent cleans, retopologises and textures it.', credit: 'Muse Spark agent' },
]

export function HomePage({ backend }: { backend: BackendState }) {
  const online = backend.status === 'online'
  return (
    <div className="home-wrap">
      <header className="hero">
        <h1>Phone video in. Game-ready 3D out.</h1>
        <p>Film an object, tell us which one, and get a textured, optimised GLB in minutes.</p>
      </header>
      <ol className="steps" aria-label="How it works">
        {STEPS.map((s, i) => (
          <li key={s.title}>
            <span className="step-n num">{i + 1}</span>
            <div>
              <strong>{s.title}</strong>
              <span className="muted small">{s.body}</span>
              <span className="credit">{s.credit}</span>
            </div>
          </li>
        ))}
      </ol>
    <div className="home">
      <div className="home-main">
        <UploadPanel disabled={!online} />
        {online && <RecentJobs />}
      </div>
      <div className="home-side">
        {backend.status === 'offline' && (
          <p className="warn" role="status">
            Backend offline. Start the API on port 8000 (see the README), or browse the showcase below.
          </p>
        )}
        <Showcase />
      </div>
    </div>
    </div>
  )
}
