import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Send /api calls to Django during local development.
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
