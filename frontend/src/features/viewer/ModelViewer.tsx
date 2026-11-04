import { Bounds, Environment, Lightformer, OrbitControls } from '@react-three/drei'
import { Canvas } from '@react-three/fiber'
import { Component, Suspense, useCallback, useState, type ReactNode } from 'react'
import { formatCount } from '../../core/format'
import { Model, type DisplayOptions } from './Model'
import type { ModelStats } from './measure'

class ViewerErrorBoundary extends Component<{ children: ReactNode }, { error: string | null }> {
  state = { error: null as string | null }
  static getDerivedStateFromError(e: unknown) {
    return { error: e instanceof Error ? e.message : String(e) }
  }
  render() {
    if (this.state.error)
      return (
        <p className="error pad" role="alert">
          Could not load the model: {this.state.error}
        </p>
      )
    return this.props.children
  }
}

/** Procedural studio lighting: no HDR download, so it works offline and on a static host. */
function StudioLight() {
  return (
    <Environment resolution={256}>
      <Lightformer form="rect" intensity={2.2} position={[0, 5, 2]} scale={[8, 4, 1]} />
      <Lightformer form="rect" intensity={1.1} position={[-5, 1, 3]} scale={[3, 5, 1]} />
      <Lightformer form="rect" intensity={1.1} position={[5, 1, -2]} scale={[3, 5, 1]} />
      <Lightformer form="rect" intensity={0.5} position={[0, -4, 0]} rotation-x={Math.PI / 2} scale={[8, 8, 1]} />
    </Environment>
  )
}

const TOGGLES: { key: keyof DisplayOptions; label: string }[] = [
  { key: 'texture', label: 'Texture' },
  { key: 'wireframe', label: 'Wireframe' },
  { key: 'normals', label: 'Normals' },
]

/** The hero 3D viewport: orbit, auto-framed model, display toggles and a geometry readout. */
export default function ModelViewer({ url }: { url: string }) {
  const [display, setDisplay] = useState<DisplayOptions>({ texture: true, wireframe: false, normals: false })
  const [stats, setStats] = useState<ModelStats | null>(null)
  const onStats = useCallback((s: ModelStats) => setStats(s), [])

  return (
    <div className="viewer">
      <ViewerErrorBoundary key={url}>
        <Canvas frameloop="demand" dpr={[1, 2]} camera={{ fov: 40, position: [2, 1.5, 3] }} aria-label="3D model viewer">
          <color attach="background" args={['#1b1e22']} />
          <StudioLight />
          <Suspense fallback={null}>
            <Bounds fit clip observe margin={1.25}>
              <Model url={url} display={display} onStats={onStats} />
            </Bounds>
          </Suspense>
          <OrbitControls makeDefault enableDamping={false} />
        </Canvas>
      </ViewerErrorBoundary>

      <div className="viewer-toolbar" role="group" aria-label="Display options">
        {TOGGLES.map((t) => (
          <button
            key={t.key}
            type="button"
            className="chip"
            aria-pressed={display[t.key]}
            onClick={() => setDisplay((d) => ({ ...d, [t.key]: !d[t.key] }))}
          >
            {t.label}
          </button>
        ))}
      </div>

      {stats && (
        <dl className="viewer-stats num" aria-label="Model statistics">
          <div>
            <dt>Triangles</dt>
            <dd>{formatCount(stats.triangles)}</dd>
          </div>
          <div>
            <dt>Vertices</dt>
            <dd>{formatCount(stats.vertices)}</dd>
          </div>
          <div>
            <dt>Texture</dt>
            <dd>{stats.textureSize ? `${stats.textureSize} px` : 'none'}</dd>
          </div>
        </dl>
      )}
    </div>
  )
}
