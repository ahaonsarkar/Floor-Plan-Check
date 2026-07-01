import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/upload-floor-plan': 'https://floor-plan-check-smoky.vercel.app',
      '/health': 'https://floor-plan-check-smoky.vercel.app',
    },
  }
})