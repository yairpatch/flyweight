// Built-in tools for agent runs, backed by the server's /agent/* endpoints.
// They exist only when `serve --agent-workspace DIR` is running: /props then
// advertises the workspace, and these definitions are sent alongside any
// user-defined tools. Everything is confined to that directory server-side.
import { postJson } from "./api";
import type { PropsPayload, ToolDefinition } from "../types";

/** Shell commands are the one built-in that asks before it runs. */
export const APPROVAL_TOOL = "run_command";

export function workspaceRoot(props: PropsPayload | null): string | null {
  return props?.agent_workspace ?? null;
}

export function hasWorkspace(props: PropsPayload | null): boolean {
  return Boolean(workspaceRoot(props));
}

interface BuiltinTool {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  /** POST target under /agent. */
  endpoint: string;
}

const BUILTINS: BuiltinTool[] = [
  {
    name: "list_dir",
    description:
      "List files and directories inside the agent workspace. Use it to explore before reading or writing. Paths are relative to the workspace root.",
    endpoint: "/agent/fs/list",
    parameters: {
      type: "object",
      properties: { path: { type: "string", description: "Directory relative to the workspace root; defaults to the root." } },
    },
  },
  {
    name: "read_file",
    description: "Read a UTF-8 text file from the agent workspace. Long files come back truncated with a flag saying so.",
    endpoint: "/agent/fs/read",
    parameters: {
      type: "object",
      properties: { path: { type: "string", description: "File relative to the workspace root." } },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description: "Create or overwrite a text file in the agent workspace, making parent directories as needed. Write the file's complete new contents.",
    endpoint: "/agent/fs/write",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File relative to the workspace root." },
        content: { type: "string", description: "The complete file contents." },
      },
      required: ["path", "content"],
    },
  },
  {
    name: APPROVAL_TOOL,
    description:
      "Run a shell command in the agent workspace and return its exit code, stdout, and stderr. The user approves each command before it runs. Use it to build, test, search, or inspect.",
    endpoint: "/agent/exec",
    parameters: {
      type: "object",
      properties: {
        command: { type: "string", description: "The shell command line to run." },
        timeout_seconds: { type: "number", description: "Seconds before the command is killed (default 30, max 300)." },
      },
      required: ["command"],
    },
  },
  {
    name: "fetch_url",
    description: "Fetch an http(s) URL from the server and return its status and body text. Not subject to browser CORS.",
    endpoint: "/agent/fetch",
    parameters: {
      type: "object",
      properties: { url: { type: "string", description: "The absolute http or https URL." } },
      required: ["url"],
    },
  },
];

const BY_NAME = new Map(BUILTINS.map((tool) => [tool.name, tool]));

export function isBuiltinTool(name: string): boolean {
  return BY_NAME.has(name);
}

/** The built-ins as tool definitions for the request builder. */
export function builtinToolDefinitions(): ToolDefinition[] {
  return BUILTINS.map((tool) => ({
    id: `builtin-${tool.name}`,
    name: tool.name,
    description: tool.description,
    parameters: JSON.stringify(tool.parameters, null, 2),
    enabled: true,
  }));
}

export interface BuiltinResult {
  ok: boolean;
  /** Text handed back to the model as the tool result. */
  result: string;
}

/**
 * Run one built-in call. `argsText` is the raw JSON the model streamed; a
 * malformed body is reported to the model rather than thrown, so the run can
 * continue and the model can correct itself.
 */
export async function runBuiltinTool(name: string, argsText: string, signal?: AbortSignal): Promise<BuiltinResult> {
  const tool = BY_NAME.get(name);
  if (!tool) return { ok: false, result: `No built-in tool named ${name}` };
  let args: unknown;
  try {
    args = argsText.trim() ? JSON.parse(argsText) : {};
  } catch {
    return { ok: false, result: "Arguments were not valid JSON; call the tool again with a well-formed arguments object." };
  }
  try {
    const payload = await postJson<Record<string, unknown>>(tool.endpoint, args, { signal });
    return { ok: true, result: formatResult(name, payload) };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, result: message };
  }
}

/** Shape each result for the model: compact, and honest about truncation. */
function formatResult(name: string, payload: Record<string, unknown>): string {
  if (name === "read_file") {
    const suffix = payload.truncated ? `\n[truncated: showing the first part of ${payload.size} bytes]` : "";
    return `${payload.content ?? ""}${suffix}`;
  }
  if (name === "write_file") return `Wrote ${payload.bytes} bytes to ${payload.path}`;
  if (name === "list_dir") {
    const entries = (payload.entries as Array<{ name: string; kind: string; size?: number }>) ?? [];
    if (!entries.length) return `${payload.path} is empty`;
    const lines = entries.map((entry) => (entry.kind === "dir" ? `${entry.name}/` : `${entry.name}${entry.size === undefined ? "" : ` (${entry.size} B)`}`));
    if (payload.truncated) lines.push("[truncated: more entries exist]");
    return lines.join("\n");
  }
  if (name === APPROVAL_TOOL) {
    const parts: string[] = [payload.timed_out ? "timed out" : `exit code ${payload.exit_code}`];
    const stdout = String(payload.stdout ?? "");
    const stderr = String(payload.stderr ?? "");
    if (stdout) parts.push(`stdout:\n${stdout}${payload.stdout_truncated ? "\n[truncated]" : ""}`);
    if (stderr) parts.push(`stderr:\n${stderr}${payload.stderr_truncated ? "\n[truncated]" : ""}`);
    if (!stdout && !stderr) parts.push("(no output)");
    return parts.join("\n");
  }
  if (name === "fetch_url") {
    return `HTTP ${payload.status} ${payload.content_type ?? ""}\n\n${payload.body ?? ""}${payload.truncated ? "\n[truncated]" : ""}`;
  }
  return JSON.stringify(payload, null, 2);
}

/** The system prompt an agent run prepends when the workspace tools exist. */
export function agentSystemPrompt(root: string): string {
  return [
    `You are an agent working in the directory ${root} on the user's machine.`,
    "Use the provided tools to inspect and change files, run commands, and fetch URLs; every path is relative to that directory.",
    "Work in small steps: look before you edit, and check your work by running something afterwards.",
    "When a command needs approval the user sees it first, so state what you are about to run and why.",
    "Stop and answer the user when the task is done or when you need a decision only they can make.",
  ].join(" ");
}
