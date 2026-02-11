import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Default port for Vite
    port: 5173,
    // Optional: Proxy API requests if you wanted to remove CORS from backend
    // proxy: {
    //   '/api': 'http://localhost:8000'
    // }
  }
})