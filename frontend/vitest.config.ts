import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'

// Test configuration is kept separate from vite.config.ts so the dev/build
// pipeline stays untouched. Vitest picks up this file automatically.
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    css: false,
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
})
