import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

// The build lands inside the Python package so `flyweight serve` ships it.
// Hashed filenames under assets/ let the server cache them immutably while
// index.html stays no-cache.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../src/flyweight/ui",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
    chunkSizeWarningLimit: 1500,
    // Keep every font a file: the server CSP allows font-src self only.
    assetsInlineLimit: 0,
  },
  server: {
    port: 5173,
    proxy: {
      "/v1": "http://127.0.0.1:8000",
      "/health": "http://127.0.0.1:8000",
      "/props": "http://127.0.0.1:8000",
      "/slots": "http://127.0.0.1:8000",
      "/tokenize": "http://127.0.0.1:8000",
      "/detokenize": "http://127.0.0.1:8000",
      "/agent": "http://127.0.0.1:8000",
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
  },
});
