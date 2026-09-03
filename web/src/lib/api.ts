// Thin fetch layer: auth header, JSON helpers, and typed endpoint wrappers.
import type {
  HealthPayload,
  ModelInfo,
  PropsPayload,
  SlotInfo,
} from "../types";

const KEY_STORAGE = "flyweight.api-key";

export function getApiKey(): string {
  try {
    return sessionStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function setApiKey(key: string): void {
  try {
    if (key) sessionStorage.setItem(KEY_STORAGE, key);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    /* hardened profiles may refuse storage */
  }
}

export class ApiError extends Error {
  status: number;
  kind: string;
  constructor(status: number, message: string, kind = "api_error") {
    super(message);
    this.status = status;
    this.kind = kind;
  }
}

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const key = getApiKey();
  if (key) headers["Authorization"] = `Bearer ${key}`;
  return headers;
}

async function errorFromResponse(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`;
  let kind = "api_error";
  try {
    const payload = await response.json();
    const error = payload?.error ?? payload;
    if (error?.message) message = String(error.message);
    if (error?.type) kind = String(error.type);
  } catch {
    /* body was not JSON */
  }
  return new ApiError(response.status, message, kind);
}

export async function fetchJson<T>(
  url: string,
  init: RequestInit & { timeoutMs?: number } = {},
): Promise<T> {
  const { timeoutMs, ...rest } = init;
  const signal = rest.signal ?? (timeoutMs ? AbortSignal.timeout(timeoutMs) : undefined);
  const response = await fetch(url, {
    ...rest,
    signal,
    headers: authHeaders({
      ...(rest.body ? { "Content-Type": "application/json" } : {}),
      ...((rest.headers as Record<string, string>) ?? {}),
    }),
  });
  if (!response.ok) throw await errorFromResponse(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function postJson<T>(url: string, body: unknown, init: RequestInit & { timeoutMs?: number } = {}): Promise<T> {
  return fetchJson<T>(url, { ...init, method: "POST", body: JSON.stringify(body) });
}

// ---- Endpoints -----------------------------------------------------------

export const api = {
  health: () => fetchJson<HealthPayload>("/health", { timeoutMs: 15000 }),
  props: () => fetchJson<PropsPayload>("/props", { timeoutMs: 15000 }),
  models: async () => (await fetchJson<{ data: ModelInfo[] }>("/v1/models", { timeoutMs: 15000 })).data,
  slots: async () => (await fetchJson<{ slots: SlotInfo[] }>("/slots", { timeoutMs: 15000 })).slots,
  tokenize: (content: string) => postJson<{ tokens: number[]; count: number }>("/tokenize", { content }),
  detokenize: (tokens: number[]) => postJson<{ content: string }>("/detokenize", { tokens }),
  countAnthropic: (body: unknown) => postJson<{ input_tokens: number }>("/v1/messages/count_tokens", body),
  countResponses: (body: unknown) => postJson<{ input_tokens: number }>("/v1/responses/input_tokens", body),
  getResponse: (id: string) => fetchJson<Record<string, unknown>>(`/v1/responses/${encodeURIComponent(id)}`),
  deleteResponse: (id: string) => fetchJson<Record<string, unknown>>(`/v1/responses/${encodeURIComponent(id)}`, { method: "DELETE" }),
};

/**
 * Open a streaming POST and hand back the response; throws ApiError on a
 * non-2xx status so callers see the server's message instead of a broken
 * SSE stream.
 */
export async function openStream(url: string, body: unknown, signal: AbortSignal): Promise<Response> {
  const response = await fetch(url, {
    method: "POST",
    headers: authHeaders({ "Content-Type": "application/json", Accept: "text/event-stream" }),
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await errorFromResponse(response);
  if (!response.body) throw new ApiError(0, "The server returned no body");
  return response;
}
