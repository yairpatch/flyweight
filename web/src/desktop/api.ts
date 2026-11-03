// The desktop app's backend (flyweight/desktop.py), reached on the page's own
// origin with the per-run token. The engine itself is reached through the
// chat UI's lib/api.ts, pointed at the engine's port.

const TOKEN = (() => {
  const match = location.hash.match(/t=([\w-]+)/);
  if (match) {
    try {
      sessionStorage.setItem("fw-desktop-token", match[1]);
    } catch {
      /* private mode */
    }
    history.replaceState(null, "", location.pathname);
    return match[1];
  }
  try {
    return sessionStorage.getItem("fw-desktop-token") ?? "";
  } catch {
    return "";
  }
})();

export async function desk<T = Record<string, unknown>>(path: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "X-Flyweight-Token": TOKEN, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return (await response.json()) as T;
}

// ---- Shapes the backend returns ------------------------------------------

export type Status = "idle" | "loading" | "running" | "stopping" | "failed";
export type Scalar = string | number | boolean | null;
export type Values = Record<string, Scalar>;

export interface ModelCheck {
  ok: boolean;
  kind: string;
  path: string;
  message: string;
  mmproj: string;
  name: string;
  folder: string;
  mmprojName: string;
}

export interface LogLine {
  seq: number;
  text: string;
  kind: string;
}

export interface DoctorLine {
  status: "ok" | "warn" | "fail";
  name: string;
  detail: string;
  fix: string;
}

export interface Fit {
  level: "great" | "good" | "slow" | "no";
  label: string;
}

export interface HubFile {
  path: string;
  size: number;
  sha256: string;
  quant?: string;
}

export interface CatalogEntry {
  id: string;
  name: string;
  repo: string;
  description?: string;
  tags?: string[];
  rank?: number;
  moe?: boolean;
  available?: boolean;
  error?: string;
  quant?: string;
  size?: number;
  files?: HubFile[];
  fit?: Fit;
}

export interface DownloadJob {
  id: string;
  repo: string;
  name: string;
  files: HubFile[];
  total: number;
  done: number;
  state: "queued" | "downloading" | "verifying" | "paused" | "done" | "failed";
  error: string;
  speed: number;
  result: string;
}

export interface LibraryModel {
  path: string;
  name: string;
  file: string;
  repo: string;
  size: number;
  folder: string;
  modified: number;
}

export interface Machine {
  ram: number;
  vram: number;
  gpu: string;
  disk_free: number;
}

export interface GpuStatus {
  state: "ready" | "needs-setup" | "needs-driver" | "old-driver" | "none";
  title: string;
  detail: string;
  driver?: string;
  major?: number;
  download_mb?: number;
}

export interface ModelInfo {
  name: string;
  architecture: string;
  format: string;
  file_size: number;
  tensor_count: number;
  parameters: number;
  quant_mix: Array<{ type: string; bytes: number; share: number }>;
  config: Record<string, number | string>;
}

export interface JobSummary {
  id: string;
  command: string;
  title: string;
  state: "running" | "done" | "failed" | "cancelled";
}

export interface JobDetail extends JobSummary {
  argv: string[];
  returncode: number | null;
  stdout: string;
  log: string[];
  result: unknown;
  error: string;
  started: number;
  ended: number;
}

export interface Snapshot {
  status: Status;
  hint: string;
  port: number | null;
  chatUrl: string | null;
  lanUrl: string | null;
  elapsed: number;
  settings: { model: string; open_chat: boolean; recent_models: string[]; models_dir: string; profiles: Record<string, Values> };
  model: ModelCheck;
  recent: Array<{ path: string; name: string; folder: string; exists: boolean }>;
  logs: LogLine[];
  lastSeq: number;
  lastLine: string;
  doctor: { running: boolean; ran: boolean; lines: DoctorLine[] };
  version: string;
  platform: string;
  profile: Values;
  ownProfile: boolean;
  engineKey: string;
  jobs: JobSummary[];
  modelInfo: ModelInfo | null;
  gpu: GpuStatus | null;
  gpuSetup: { running: boolean; message: string; fraction: number; error: string };
  update: { version: string; url: string } | null;
  driverUrl: string;
  notice: string;
  models: {
    machine: Machine | null;
    summary: string;
    catalog: CatalogEntry[];
    loading: boolean;
    downloads: DownloadJob[];
    library: LibraryModel[];
    dir: string;
  };
}

export interface OptionSpec {
  dest: string;
  flags: string[];
  flag: string;
  positional: boolean;
  required: boolean;
  kind: "bool" | "choice" | "int" | "float" | "path" | "dir" | "text";
  choices: Scalar[] | null;
  default: Scalar;
  store: "true" | "false";
  metavar: string;
  help: string;
}

export interface CommandSchema {
  command: string;
  description: string;
  groups: Array<{ title: string; description: string; options: OptionSpec[] }>;
}

export type Schemas = Record<string, CommandSchema>;

export interface RequestRow {
  time: string;
  endpoint: string;
  prompt: number | null;
  cached: number | null;
  ttft: number | null;
  out: number | null;
  rate: number | null;
  finish: string;
  total: number | null;
  error: string;
  at: number;
}

export interface MonitorData {
  system: {
    memory: { total: number | null; available: number | null };
    cpu: number | null;
    gpus: Array<{ name: string; used: number | null; total: number | null; util: number | null; temp: number | null; power: number | null }>;
    engine_rss: number | null;
    at: number;
  };
  health: Record<string, unknown> | null;
  props: Record<string, unknown> | null;
  slots: { slots?: Array<Record<string, unknown>> } | null;
  requests: RequestRow[];
  status: Status;
  startedAt: number;
  readyAt: number;
}
