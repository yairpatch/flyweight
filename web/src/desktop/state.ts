// Desktop app state: the backend snapshot (polled), the engine option
// schemas (fetched once), the page, and a toast. The chat keeps using the
// chat UI's own store (../store), pointed at the running engine.
import { create } from "zustand";
import { desk, type LogLine, type Schemas, type Snapshot } from "./api";

export type Page = "chat" | "models" | "server" | "monitor" | "tools" | "activity" | "system";

export interface DeskToast {
  id: number;
  text: string;
  kind: "info" | "error" | "ok";
}

interface DeskState {
  snapshot: Snapshot | null;
  logs: LogLine[];
  schemas: Schemas | null;
  page: Page;
  connected: boolean;
  toasts: DeskToast[];
  setPage: (page: Page) => void;
  poll: () => Promise<void>;
  loadSchemas: () => Promise<void>;
  call: <T = Record<string, unknown>>(path: string, body?: unknown) => Promise<T | null>;
  toast: (text: string, kind?: DeskToast["kind"]) => void;
  clearLogs: () => void;
}

let lastSeq = 0;
let toastId = 0;

export const useDesk = create<DeskState>()((set, get) => ({
  snapshot: null,
  logs: [],
  schemas: null,
  page: "chat",
  connected: true,
  toasts: [],

  setPage: (page) => set({ page }),

  poll: async () => {
    try {
      const fresh = await desk<Snapshot>(`/api/state?since=${lastSeq}`);
      lastSeq = fresh.lastSeq;
      const logs = fresh.logs.length ? [...get().logs, ...fresh.logs].slice(-4000) : get().logs;
      if (!get().connected) get().toast("Reconnected", "ok");
      set({ snapshot: fresh, logs, connected: true });
      if (fresh.notice) get().toast(fresh.notice, "ok");
    } catch {
      if (get().connected) get().toast("Lost contact with Flyweight. Is it still running?", "error");
      set({ connected: false });
    }
  },

  loadSchemas: async () => {
    try {
      set({ schemas: await desk<Schemas>("/api/schema") });
    } catch {
      /* retried on the next visit to a page that needs it */
    }
  },

  call: async <T,>(path: string, body: unknown = {}) => {
    try {
      const result = await desk<T & { ok?: boolean; error?: string }>(path, body);
      if (result && result.ok === false && result.error) get().toast(result.error, "error");
      void get().poll();
      return result as T;
    } catch (error) {
      get().toast(error instanceof Error ? error.message : String(error), "error");
      return null;
    }
  },

  toast: (text, kind = "info") => {
    const id = ++toastId;
    set({ toasts: [...get().toasts, { id, text, kind }].slice(-4) });
    window.setTimeout(() => set({ toasts: get().toasts.filter((t) => t.id !== id) }), kind === "error" ? 7000 : 3200);
  },

  clearLogs: () => set({ logs: [] }),
}));

// ---- Formatting -----------------------------------------------------------

export function gb(bytes: number | null | undefined, digits = 1): string {
  if (bytes == null || !Number.isFinite(bytes)) return "—";
  if (bytes >= 2 ** 30) return `${(bytes / 2 ** 30).toFixed(digits)} GB`;
  if (bytes >= 2 ** 20) return `${Math.round(bytes / 2 ** 20)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

export function clock(seconds: number): string {
  const s = Math.max(0, Math.round(seconds));
  return s >= 3600
    ? `${Math.floor(s / 3600)}:${String(Math.floor((s % 3600) / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`
    : `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** "--cache-type-k" -> "Cache type k"; the dest is the fallback. */
export function labelOf(flag: string, dest: string): string {
  const base = (flag || dest).replace(/^-+/, "").replace(/[-_]/g, " ");
  const words = base.split(" ").map((w) =>
    ({ kv: "KV", gpu: "GPU", cpu: "CPU", mtp: "MTP", api: "API", cors: "CORS", mib: "MiB", moe: "MoE",
       mmproj: "MMProj", swa: "SWA", sse: "SSE", ngl: "NGL", ttft: "TTFT", imatrix: "Imatrix" } as Record<string, string>)[w] ?? w);
  const text = words.join(" ");
  return text.charAt(0).toUpperCase() + text.slice(1);
}
