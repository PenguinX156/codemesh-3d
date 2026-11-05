import { useGLTF } from '@react-three/drei'
import { useThree } from '@react-three/fiber'
import { useEffect, useMemo } from 'react'
import { meshesOf, measure, type ModelStats } from './measure'
import { Box3, MeshBasicMaterial, MeshNormalMaterial, MeshStandardMaterial, type Material, type Mesh } from 'three'

export interface DisplayOptions {
  texture: boolean
  wireframe: boolean
  normals: boolean
}

// three.js materials are mutable scene state, not React state.
const setMaterial = (mesh: Mesh, material: Mesh['material']) => {
  mesh.material = material
}

/** A glTF scene whose materials follow the display toggles. Original materials are restored on toggle-off. */
export function Model({ url, display, onStats, onFloor }: { url: string; display: DisplayOptions; onStats: (s: ModelStats) => void; onFloor?: (y: number) => void }) {
  const { scene } = useGLTF(url)
  const invalidate = useThree((s) => s.invalidate)
  const root = useMemo(() => scene.clone(true), [scene])
  const originals = useMemo(() => new Map(meshesOf(root).map((m) => [m, m.material] as const)), [root])

  useEffect(() => onStats(measure(root)), [root, onStats])
  useEffect(() => onFloor?.(new Box3().setFromObject(root).min.y), [root, onFloor])

  useEffect(() => {
    const made: Material[] = []
    const track = <T extends Material>(m: T): T => (made.push(m), m)
    const plain = track(new MeshStandardMaterial({ color: '#b9bec4', roughness: 0.7, metalness: 0 }))
    const normal = track(new MeshNormalMaterial())

    for (const [mesh, original] of originals) {
      if (display.normals) setMaterial(mesh, normal)
      else if (!display.texture) setMaterial(mesh, plain)
      else setMaterial(mesh, original)
    }
    if (display.wireframe) {
      // Wireframe replaces the surface so the topology is readable at any density.
      const wire = track(new MeshBasicMaterial({ color: '#3dd6c3', wireframe: true }))
      for (const mesh of originals.keys()) setMaterial(mesh, wire)
    }
    invalidate() // the canvas renders on demand; material swaps are not React state
    return () => made.forEach((m) => m.dispose())
  }, [originals, display, invalidate])

  return <primitive object={root} />
}
