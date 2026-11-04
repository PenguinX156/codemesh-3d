import type {
  Capabilities,
  Frame,
  Health,
  Job,
  JobMode,
  ReconstructSettings,
  SelectRequest,
} from './types.ts'

/** A picked file. A browser `File` fits; React Native can pass its `{uri,name,type}` blob-like object. */
export type UploadFile = Blob & { name?: string }

export class ApiError extends Error {
  readonly status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** FastAPI errors look like `{detail: string | {msg}[]}`; fall back to the HTTP status text. */
async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json()
    const d = body?.detail
    if (typeof d === 'string') return d
    if (Array.isArray(d)) return d.map((e: { msg?: string }) => e.msg ?? 'invalid request').join('; ')
  } catch {
    /* body was not JSON */
  }
  return `${res.status} ${res.statusText}`.trim()
}

/**
 * Typed client for the CodeMesh API. `baseUrl` is injected (e.g. "/api" in the browser,
 * "http://192.168.1.5:8000/api" from a phone) so the same code runs on web and Expo.
 */
export function createApi(baseUrl: string, fetchImpl: typeof fetch = (...a) => fetch(...a)) {
  const base = baseUrl.replace(/\/+$/, '')

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    let res: Response
    try {
      res = await fetchImpl(`${base}${path}`, init)
    } catch (e) {
      throw new ApiError(e instanceof Error ? e.message : 'Network error', 0)
    }
    if (!res.ok) throw new ApiError(await errorMessage(res), res.status)
    return (res.status === 204 ? undefined : await res.json()) as T
  }

  const json = (method: string, body: unknown): RequestInit => ({
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

  const isAbsolute = (u: string) => /^(https?:|blob:|data:)/i.test(u)

  /** Turn a URL or path returned by the API into something fetchable/embeddable. */
  function resolveUrl(url: string): string {
    if (isAbsolute(url)) return url
    if (url.startsWith('/api/')) return `${base}${url.slice(4)}`
    return `${base}${url.startsWith('/') ? '' : '/'}${url}`
  }

  return {
    baseUrl: base,
    resolveUrl,
    /** Artifact values may be relative paths ("model.glb") or full URLs. */
    artifactUrl(jobId: string, artifact: string): string {
      if (isAbsolute(artifact) || artifact.startsWith('/')) return resolveUrl(artifact)
      return `${base}/jobs/${jobId}/artifacts/${artifact}`
    },
    health: () => request<Health>('/health'),
    capabilities: () => request<Capabilities>('/capabilities'),
    listJobs: () => request<Job[]>('/jobs'),
    getJob: (id: string) => request<Job>(`/jobs/${id}`),
    getFrames: (id: string) => request<Frame[]>(`/jobs/${id}/frames`),
    createJob(files: UploadFile[], mode: JobMode) {
      const form = new FormData()
      form.append('mode', mode)
      for (const f of files) form.append('files', f, f.name)
      return request<Job>('/jobs', { method: 'POST', body: form })
    },
    select: (id: string, body: SelectRequest) => request<unknown>(`/jobs/${id}/select`, json('POST', body)),
    reconstruct: (id: string, body: ReconstructSettings) =>
      request<unknown>(`/jobs/${id}/reconstruct`, json('POST', body)),
    refine: (id: string, instruction?: string) => request<unknown>(`/jobs/${id}/refine`, json('POST', { instruction })),
    deleteJob: (id: string) => request<void>(`/jobs/${id}`, { method: 'DELETE' }),
  }
}

export type Api = ReturnType<typeof createApi>
