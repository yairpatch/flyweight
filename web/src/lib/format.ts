// Small formatting helpers shared by the transcript and the dashboard.

export function formatBytes(bytes: number | undefined | null): string {
  if (bytes === undefined || bytes === null || !Number.isFinite(bytes)) return "–";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value < 10 ? value.toFixed(2) : value < 100 ? value.toFixed(1) : Math.round(value)} ${units[unit]}`;
}

export function formatCompact(value: number | undefined | null, digits = 1): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return "–";
  const abs = Math.abs(value);
  if (abs < 1000) return Number.isInteger(value) ? String(value) : value.toFixed(digits);
  if (abs < 1_000_000) return `${(value / 1000).toFixed(digits)}K`;
  if (abs < 1_000_000_000) return `${(value / 1_000_000).toFixed(digits)}M`;
  return `${(value / 1_000_000_000).toFixed(digits)}B`;
}

export function formatInteger(value: number | undefined | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return "–";
  return new Intl.NumberFormat().format(Math.round(value));
}

export function formatSeconds(seconds: number | undefined | null): string {
  if (seconds === undefined || seconds === null || !Number.isFinite(seconds)) return "–";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
}

export function formatNanoseconds(ns: number | undefined | null): string {
  if (ns === undefined || ns === null || !Number.isFinite(ns)) return "–";
  return formatSeconds(ns / 1e9);
}

export function formatPercent(ratio: number | undefined | null, digits = 1): string {
  if (ratio === undefined || ratio === null || !Number.isFinite(ratio)) return "–";
  return `${(ratio * 100).toFixed(digits)}%`;
}

export function formatRate(value: number | undefined | null): string {
  if (value === undefined || value === null || !Number.isFinite(value)) return "–";
  return `${value < 10 ? value.toFixed(2) : value.toFixed(1)} tok/s`;
}

export function formatTime(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(timestamp);
}

export function formatDate(timestamp: number): string {
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(timestamp);
}

export function relativeDay(timestamp: number, now = Date.now()): string {
  const day = 86_400_000;
  const start = new Date(now);
  start.setHours(0, 0, 0, 0);
  const diff = start.getTime() - timestamp;
  if (diff < 0) return "Today";
  if (diff < day) return "Yesterday";
  if (diff < 7 * day) return "Previous 7 days";
  if (diff < 30 * day) return "Previous 30 days";
  return "Older";
}

export function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9֐-׿؀-ۿ]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60) || "conversation";
}

export function identifier(prefix = "id"): string {
  const random =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${random}`;
}

export function titleFromPrompt(text: string, limit = 48): string {
  const line = text.replace(/\s+/g, " ").trim();
  if (!line) return "New conversation";
  if (line.length <= limit) return line;
  const cut = line.slice(0, limit);
  const space = cut.lastIndexOf(" ");
  return `${space > limit / 2 ? cut.slice(0, space) : cut}…`;
}

export function prettyJson(text: string): string {
  try {
    return JSON.stringify(JSON.parse(text), null, 2);
  } catch {
    return text;
  }
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
