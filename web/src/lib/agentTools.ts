// Built-in tools for agent runs, backed by the server's /agent/* endpoints.
// They exist only when `serve --agent-workspace DIR` is running: /props then
// advertises the workspace, and these definitions are sent alongside any
// user-defined tools. Everything is confined to that directory server-side.
import { postJson } from "./api";
import type { PropsPayload, ToolDefinition } from "../types";

/** Shell commands are the one built-in that asks before it runs. */
export const APPROVAL_TOOL = "run_command";

/** Whether a called name is the shell tool, however the model spelled it. */
export function needsApproval(name: string): boolean {
  return canonicalName(name) === APPROVAL_TOOL;
}

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

/**
 * A called name as the built-ins know it. Names arrive as the model wrote
 * them: some templates namespace a call ("functions.read_file") and streamed
 * names can carry stray whitespace. Matching loosely here is the difference
 * between the loop running the call and pausing for a manual result.
 */
function canonicalName(name: string): string {
  const trimmed = name.trim();
  const dot = trimmed.lastIndexOf(".");
  return dot === -1 ? trimmed : trimmed.slice(dot + 1);
}

export function isBuiltinTool(name: string): boolean {
  return BY_NAME.has(canonicalName(name));
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
  const tool = BY_NAME.get(canonicalName(name));
  if (!tool) return { ok: false, result: `No built-in tool named ${name}` };
  let args: unknown;
  try {
    args = argsText.trim() ? JSON.parse(argsText) : {};
  } catch {
    return { ok: false, result: "Arguments were not valid JSON; call the tool again with a well-formed arguments object." };
  }
  try {
    const payload = await postJson<Record<string, unknown>>(tool.endpoint, args, { signal });
    return { ok: true, result: formatResult(tool.name, payload) };
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

/**
 * Why the loop stopped short of running the calls it was handed, in the words
 * the user needs to fix it. A built-in name with no workspace is the common
 * case: the model asked for a file or a shell because agent runs advertise
 * those tools, but the server was started without a directory to confine them
 * to, so nothing can run them.
 */
export function missingHandlerReason(names: string[], workspaceLive: boolean): string {
  const list = names.join(", ");
  const plural = names.length > 1;
  const builtins = names.filter(isBuiltinTool);
  if (!workspaceLive && builtins.length) {
    return `${builtins.join(", ")} ${plural ? "are workspace tools" : "is a workspace tool"}, and this server has no agent workspace. Restart it with --agent-workspace DIR to let the agent list, read, and write files, run commands, and fetch URLs inside DIR — or answer the call by hand below.`;
  }
  return `Nothing here can run ${list}: ${plural ? "these tools have" : "this tool has"} no JavaScript handler. Add one in the Tools panel, or answer the call by hand below.`;
}

/** The turn cap stopped the loop; say what the cap is and where to raise it. */
export function turnCapReason(turns: number, cap: number): string {
  return `The agent used its ${cap}-turn budget (${turns} model turns). Raise the turn cap in the Tools panel, or answer the call by hand below to keep going.`;
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
