import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 7550,
    proxy: {
      "/chat": "http://localhost:8000",
      "/users": "http://localhost:8000",
      "/session": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
})
