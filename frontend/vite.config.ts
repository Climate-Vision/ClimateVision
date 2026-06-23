import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: mode === 'development'
      ? {
          '/api': {
            target: 'http://127.0.0.1:8000',
            changeOrigin: true,
          },
        }
      : undefined,
  },
}))
