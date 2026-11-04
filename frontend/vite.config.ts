import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev: /api is proxied to the local FastAPI backend. Hosted builds set VITE_API_BASE_URL instead.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  },
  build: { chunkSizeWarningLimit: 1400 }, // three.js is lazy-loaded, but still large
})
