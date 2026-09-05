// Runs a tool's user-authored JavaScript handler for the agentic loop. The
// handler executes inside the same sandboxed preview shell as the code
// preview: preview.html is served with a sandboxing CSP that gives it an
// opaque origin with no access to this app's storage or API key, and the
// document travels in the URL fragment so it never reaches the server. The
// handler reports its result back over postMessage.
import { identifier } from "./format";

export interface ToolExecution {
  ok: boolean;
  /** The handler's result on success; a human-readable reason on failure. */
  result: string;
}

export const EXECUTOR_TIMEOUT_MS = 30_000;

/** JSON that stays inert inside a <script> block. */
function inlineJson(value: string): string {
  return JSON.stringify(value).replace(/</g, "\\u003c");
}

/**
 * The document the sandbox runs: parse the call's arguments, run the handler
 * body as an async function of `args`, and post the result to the parent. A
 * string return travels as-is; anything else is JSON-stringified.
 */
export function executorDocument(source: string, argsText: string, token: string): string {
  return `<!DOCTYPE html><html><head><meta charset="utf-8"></head><body><script>
const post = (ok, result) => parent.postMessage({ flyweightTool: ${inlineJson(token)}, ok, result: String(result) }, "*");
addEventListener("unhandledrejection", (event) => post(false, String((event.reason && event.reason.message) || event.reason || "unhandled rejection")));
(async () => {
  let args = {};
  try { args = JSON.parse(${inlineJson(argsText)}); } catch { /* malformed arguments stay {} */ }
  try {
    const AsyncFunction = Object.getPrototypeOf(async () => {}).constructor;
    const value = await new AsyncFunction("args", ${inlineJson(source)})(args);
    post(true, value === undefined || value === null ? "" : typeof value === "string" ? value : (JSON.stringify(value, null, 2) ?? String(value)));
  } catch (error) {
    post(false, String((error && error.message) || error));
  }
})();
</${"script"}></body></html>`;
}

export interface ExecutorOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

/** Execute one handler in a hidden sandboxed iframe; never rejects. */
export function runToolExecutor(source: string, argsText: string, options: ExecutorOptions = {}): Promise<ToolExecution> {
  const timeoutMs = options.timeoutMs ?? EXECUTOR_TIMEOUT_MS;
  return new Promise((resolve) => {
    const token = identifier("exec");
    const iframe = document.createElement("iframe");
    iframe.setAttribute("sandbox", "allow-scripts");
    iframe.style.display = "none";
    iframe.title = "Tool execution";
    let settled = false;
    const finish = (ok: boolean, result: string) => {
      if (settled) return;
      settled = true;
      window.clearTimeout(timer);
      window.removeEventListener("message", onMessage);
      options.signal?.removeEventListener("abort", onAbort);
      iframe.remove();
      resolve({ ok, result });
    };
    const onMessage = (event: MessageEvent) => {
      if (event.source !== iframe.contentWindow) return;
      const data = event.data as { flyweightTool?: string; ok?: boolean; result?: string } | null;
      if (data && data.flyweightTool === token) finish(Boolean(data.ok), String(data.result ?? ""));
    };
    const onAbort = () => finish(false, "stopped");
    const timer = window.setTimeout(() => finish(false, `handler timed out after ${Math.round(timeoutMs / 1000)}s`), timeoutMs);
    window.addEventListener("message", onMessage);
    options.signal?.addEventListener("abort", onAbort);
    if (options.signal?.aborted) {
      onAbort();
      return;
    }
    iframe.src = `/preview.html#${encodeURIComponent(executorDocument(source, argsText, token))}`;
    document.body.appendChild(iframe);
  });
}
