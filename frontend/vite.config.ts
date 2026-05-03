import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  envDir: "..",   // read .env from repo root (shared with backend)
  server: {
    port: 7550,
    proxy: {
      "/chat": "http://localhost:8000",
      "/users": "http://localhost:8000",
      "/session": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/evals": "http://localhost:8000",
      "/feedback": "http://localhost:8000",
      "/debug": "http://localhost:8000",
      "/ink": "http://localhost:8000",
    },
  },
})
