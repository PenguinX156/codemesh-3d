import type { BackendState } from '../../app/hooks'
import { Showcase } from '../showcase/Showcase'
import { UploadPanel } from '../upload/UploadPanel'
import { RecentJobs } from './RecentJobs'

export function HomePage({ backend }: { backend: BackendState }) {
  const online = backend.status === 'online'
  return (
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
  )
}
