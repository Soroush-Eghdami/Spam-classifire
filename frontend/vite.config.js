import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During local development the frontend calls /api and Vite proxies it to
// Django.  In Docker, nginx does the same proxying.  Either way the browser
// sees a single origin, so there is no CORS setup to remember.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 3000,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  preview: {
    port: 4173,
  },
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
