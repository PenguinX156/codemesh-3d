/** Mirrors the v1 API contract in docs/PLAN.md. Platform-agnostic: no DOM types. */

export type JobMode = 'walkaround' | 'turntable'

export type JobStatus =
  | 'created'
  | 'ready_to_select'
  | 'segmenting'
  | 'ready_to_reconstruct'
  | 'reconstructing'
  | 'refining'
  | 'done'
  | 'failed'

export type StageStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped'

export type MetricValue = number | string | boolean | null

export interface Stage {
  name: string
  status: StageStatus
  seconds: number | null
  metrics: Record<string, MetricValue>
}

/** Values are artifact paths (relative to /jobs/{id}/artifacts/) or full URLs. */
export interface JobArtifacts {
  model_glb?: string
  masks_preview?: string
  [key: string]: string | undefined
}

export interface Job {
  id: string
  /** Unix epoch seconds (backend `time.time()`). */
  created_at: number
  mode: JobMode
  status: JobStatus
  stage: string | null
  progress: number
  message: string
  stages: Stage[]
  artifacts: JobArtifacts
  metrics: Record<string, MetricValue>
  error?: string | null
  /** User choices as stored by the backend (selection, reconstruct, ...). */
  settings?: Record<string, unknown>
  /** Frame index the backend recommends for selection (optional). */
  best_frame?: number | null
}

export interface Frame {
  index: number
  url: string
  thumb_url: string
  sharpness: number
  width: number
  height: number
  /** True on the one frame the backend recommends for selection. */
  best?: boolean
}

export interface PromptPoint {
  /** Normalized 0..1 from the top-left of the frame. */
  x: number
  y: number
  /** 1 = belongs to the object, 0 = background. */
  label: 1 | 0
}

export interface SelectRequest {
  frame_index: number
  points: PromptPoint[]
  text?: string
}

export type Preset = 'draft' | 'standard' | 'high'
export type Axis = 'x' | 'y' | 'z'

export interface ReconstructSettings {
  preset: Preset
  triangle_budget: 'auto' | number
  texture_size: 2048 | 4096
  known_dimension?: { axis: Axis; meters: number }
}

export interface Health {
  ok: boolean
  version: string
}

export interface Capabilities {
  geometry_engine: string
  device: string
  precision: string
  blender: boolean
  meta_api: boolean
  engines: Record<string, boolean>
  heic: boolean
  max_upload_bytes: number
}

/** Statuses after which the job will not change without a user action. */
export const TERMINAL_STATUSES: readonly JobStatus[] = ['done', 'failed']

/** Statuses where the backend is working and the UI should show progress. */
export const BUSY_STATUSES: readonly JobStatus[] = ['created', 'segmenting', 'reconstructing', 'refining']
