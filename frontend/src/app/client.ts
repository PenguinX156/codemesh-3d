import { createApi } from '../core/api'

/** `VITE_API_BASE_URL` points a hosted (Vercel) build at a remote backend, e.g. "https://box.example.com/api". */
const configured = import.meta.env.VITE_API_BASE_URL as string | undefined

export const api = createApi(configured || '/api')

/** True when this build talks to a backend other than the local dev proxy. */
export const isRemoteBackend = Boolean(configured)
