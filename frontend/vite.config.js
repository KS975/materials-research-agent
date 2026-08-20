import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  // Emit ./assets/... so the built dist can be hosted under /Q&A/ or any subpath.
  base: "./",
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Keep source code identical between local development and the unit server.
      // Local Vite strips /agent-api before forwarding to FastAPI on :8000.
      "/agent-api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/agent-api/, ""),
      },
    },
  },
});
