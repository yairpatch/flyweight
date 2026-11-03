import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Flyweight Desktop: the app the desktop window shows. Built into the Python
// package beside the chat UI and served by `flyweight app`, which also
// proxies nothing: the chat talks to the engine's own port directly.
export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "../src/flyweight/desktop_ui",
    emptyOutDir: true,
    sourcemap: false,
    target: "es2022",
    chunkSizeWarningLimit: 2500,
    assetsInlineLimit: 0,
    rollupOptions: { input: "desktop.html" },
  },
  server: { port: 5174 },
});
