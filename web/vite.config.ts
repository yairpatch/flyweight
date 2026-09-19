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
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts", "src/**/*.test.tsx"],
    // pdf.js ships two builds and asks Node to use the legacy one; the
    // warning it prints under vitest is not advisory. Its default build is
    // compiled for current browsers and calls Promise.try (V8 13.1, node 24)
    // and Uint8Array.prototype.toHex (node 26) -- on CI's node 22 the first
    // of those rejects inside the worker's message handler, which nothing
    // awaits, so pdfText simply never settles and the two cases time out at
    // five seconds. A developer on node 26 sees them pass.
    //
    // Only the test build is aliased. The browser keeps the default build,
    // which is the point of shipping it.
    // Anchored, because a bare string alias matches as a prefix and would
    // rewrite the `pdfjs-dist/build/pdf.worker.mjs?url` import too.
    alias: [
      { find: /^pdfjs-dist$/, replacement: "pdfjs-dist/legacy/build/pdf.mjs" },
    ],
  },
});
