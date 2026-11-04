import type { BufferGeometry, Material, Mesh, MeshStandardMaterial, Object3D, Texture } from 'three'

export interface ModelStats {
  triangles: number
  vertices: number
  /** Largest base-color texture edge in pixels, or null when untextured. */
  textureSize: number | null
}

export const meshesOf = (root: Object3D): Mesh[] => {
  const out: Mesh[] = []
  root.traverse((o) => (o as Mesh).isMesh && out.push(o as Mesh))
  return out
}

const triangleCount = (g: BufferGeometry) => (g.index ? g.index.count : g.getAttribute('position').count) / 3

export function measure(root: Object3D): ModelStats {
  let triangles = 0
  let vertices = 0
  let textureSize: number | null = null
  for (const m of meshesOf(root)) {
    triangles += triangleCount(m.geometry)
    vertices += m.geometry.getAttribute('position').count
    for (const mat of [m.material].flat() as Material[]) {
      const map = (mat as MeshStandardMaterial).map as Texture | null | undefined
      const img = map?.image as { width?: number; height?: number } | undefined
      if (img?.width) textureSize = Math.max(textureSize ?? 0, img.width, img.height ?? 0)
    }
  }
  return { triangles, vertices, textureSize }
}

