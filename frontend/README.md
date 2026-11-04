# CodeMesh 3D frontend

React + TypeScript + Vite + React Three Fiber. Plain CSS with tokens in `src/styles.css`.

```
npm install
npm run dev      # http://localhost:5173, /api proxied to http://localhost:8000
npm run build    # tsc + vite build -> dist/
npm run lint     # oxlint
npm test         # node:test on src/core (no bundler, no DOM)
```

## Layout

- `src/core/` platform-agnostic TS (types mirroring the API contract in `docs/PLAN.md`, `createApi(baseUrl)`, `pollJob`, presets, formatters). No DOM/window/document, so an Expo app can import it unchanged. Relative imports use `.ts` so Node can run the tests directly.
- `src/app/` browser wiring: API instance (`client.ts`), hash router, `useBackend` (/health), `useJob` (polling).
- `src/features/<name>/` UI per feature: `upload`, `select`, `progress`, `viewer`, `settings`, `export`, `showcase`, `home`, `job`. `job/JobPage.tsx` picks the view from `job.status`.

## Configuration

- `VITE_API_BASE_URL` (see `.env.example`): backend base including `/api`. Unset locally (dev proxy). On Vercel set it to a reachable backend that allows the site origin via CORS; without it the hosted app shows "Backend offline" and only the showcase works.
- Showcase: put GLB (+ optional thumbnail) files in `public/showcase/` and list them in `public/showcase/manifest.json` as `[{ "title", "glb", "thumbnail", "stats": { "triangles", "vertices", "texture_size" } }]` (paths relative to `/showcase/`).

## Assumptions about the backend

- `job.artifacts` values are paths relative to `/jobs/{id}/artifacts/` or full URLs; `masks_preview` is an image drawn over the frame (use a transparent overlay).
- Frame `url` / `thumb_url` may start with `/api/` or be absolute; both are resolved against the injected base.
- `POST /select`, `/reconstruct` response bodies are ignored; the UI re-polls `GET /jobs/{id}`.
