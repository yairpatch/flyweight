// Built-in tools for agent runs, backed by the server's /agent/* endpoints.
// They exist only when `serve --agent-workspace DIR` is running: /props then
// advertises the workspace, and these definitions are sent alongside any
// user-defined tools. Everything is confined to that directory server-side.
import { postJson } from "./api";
import type { AgentPlatform, PropsPayload, ToolDefinition } from "../types";

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

export function workspacePlatform(props: PropsPayload | null): AgentPlatform | null {
  return props?.agent_platform ?? null;
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
    description:
      "Read a text file from the agent workspace. Long files come back truncated with a flag saying so. Read a file before editing it: edit_file needs the exact text that is in it.",
    endpoint: "/agent/fs/read",
    parameters: {
      type: "object",
      properties: { path: { type: "string", description: "File relative to the workspace root." } },
      required: ["path"],
    },
  },
  {
    name: "edit_file",
    description:
      "Replace an exact snippet of an existing file with new text. Prefer this over write_file for any change to a file that already exists: it leaves the rest of the file untouched. old_string must match the file exactly, including indentation, and must be unique unless replace_all is true — include surrounding lines to make it unique. Write both strings with plain \\n newlines; the file's own line endings are preserved.",
    endpoint: "/agent/fs/edit",
    parameters: {
      type: "object",
      properties: {
        path: { type: "string", description: "File relative to the workspace root." },
        old_string: { type: "string", description: "The exact text to replace, copied from the file." },
        new_string: { type: "string", description: "The text to put in its place; empty to delete the snippet." },
        replace_all: { type: "boolean", description: "Replace every occurrence instead of requiring a unique one." },
      },
      required: ["path", "old_string", "new_string"],
    },
  },
  {
    name: "write_file",
    description:
      "Create a file, or replace one whole file's contents, making parent directories as needed. Use edit_file to change part of an existing file. Write plain \\n newlines; an existing file keeps its own line endings and encoding.",
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
      "Run a shell command in the agent workspace and return its exit code, stdout, and stderr. The user approves each command before it runs. Use it to build, test, search, or inspect. The shell is the host's own — the system prompt says which one, and the command must be written for it.",
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
    description:
      "Fetch an http(s) URL from the server and return its readable text with the markup, navigation and scripts stripped. Not subject to browser CORS. Always pass query: a page is far larger than the answer you want from it, and query makes the server send back the passages that match instead of the top of the page. Raise max_chars only when you truly need more, or page through a long document with offset.",
    endpoint: "/agent/fetch",
    parameters: {
      type: "object",
      properties: {
        url: { type: "string", description: "The absolute http or https URL." },
        query: { type: "string", description: "What you are looking for on the page, in words that would appear in it." },
        max_chars: { type: "number", description: "Characters of text to return (default 6000, max 40000)." },
        offset: { type: "number", description: "Skip this many characters first; use it to read on where the last call stopped." },
      },
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
  if (name === "write_file") {
    return `${payload.created ? "Created" : "Wrote"} ${payload.path} (${payload.bytes} bytes, ${payload.line_ending ?? "lf"} line endings)`;
  }
  if (name === "edit_file") {
    const count = Number(payload.replacements ?? 0);
    return `Replaced ${count} occurrence${count === 1 ? "" : "s"} in ${payload.path} (${payload.bytes} bytes)`;
  }
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
    const head = [`HTTP ${payload.status} ${payload.content_type ?? ""}`.trim(), payload.title ? String(payload.title) : ""].filter(Boolean).join(" — ");
    // Saying what was left behind, and how to reach it, is what keeps the
    // model from re-fetching the same page to look for the rest.
    const total = Number(payload.total_chars ?? 0);
    const shown = Number(payload.chars ?? 0);
    const note = payload.truncated
      ? payload.selection === "query"
        ? `\n\n[showing the passages matching your query: ${shown} of ${total} characters. Raise max_chars or drop the query for more.]`
        : `\n\n[showing ${shown} of ${total} characters. Call again with offset ${payload.next_offset ?? shown} for the next part, or pass a query to jump to what you need.]`
      : "";
    return `${head}\n\n${payload.body ?? ""}${note}`;
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

/**
 * What the model needs to know about the machine it is working on. Without it
 * a model defaults to Unix habits — `ls -la`, `grep -r`, `rm -rf`, forward
 * slashes — and on a Windows host half of those fail or, worse, half-succeed.
 * The server reports the shell it will actually spawn, so the prompt names it.
 */
function platformLines(platform: AgentPlatform): string[] {
  const windows = platform.os === "windows";
  const lines = [`The machine runs ${platform.os} and commands go to ${platform.shell}, so write every command in that shell's syntax.`];
  if (windows && (platform.shell === "powershell" || platform.shell === "pwsh")) {
    // The aliases are real, so say so: a model told only "this is Windows"
    // reaches for cmd builtins like `dir /b` and `type`, which PowerShell
    // parses differently.
    lines.push(
      "PowerShell aliases ls, cat, cp, mv, rm and pwd to its own cmdlets, so those work; grep, sed, awk and && do not. Use Select-String instead of grep and separate statements with ; instead of &&.",
      // The single most common way an agent's first Windows command fails.
      'Run a program whose path you had to quote with the call operator: & "C:\\path with spaces\\tool.exe" --flag. Without the &, PowerShell prints the path instead of running it.',
    );
  } else if (windows) {
    lines.push("Use cmd.exe syntax: dir, type, copy, del, and %VAR% for variables.");
  }
  lines.push(
    `Paths on this host use ${platform.path_separator === "\\" ? "backslashes" : "forward slashes"}, and the tools accept either; keep tool paths relative to the workspace root.`,
  );
  if (platform.line_ending === "crlf") {
    lines.push("Files here tend to use CRLF line endings; write \\n in tool arguments and the server keeps each file's existing endings.");
  }
  return lines;
}

/** The system prompt an agent run prepends when the workspace tools exist. */
export function agentSystemPrompt(root: string, platform?: AgentPlatform | null): string {
  return [
    `You are an agent working in the directory ${root} on the user's machine.`,
    "Use the provided tools to inspect and change files, run commands, and fetch URLs; every path is relative to that directory.",
    ...(platform ? platformLines(platform) : []),
    "Work in small steps: look before you edit, and check your work by running something afterwards.",
    "To change an existing file, read it and then call edit_file with an exact snippet; write_file replaces a whole file and is for new ones.",
    "When a command needs approval the user sees it first, so state what you are about to run and why.",
    "Stop and answer the user when the task is done or when you need a decision only they can make.",
  ].join(" ");
}
