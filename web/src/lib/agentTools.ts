// Built-in tools for agent runs, backed by the server's /agent/* endpoints.
// They exist only when `serve --agent-workspace DIR` is running: /props then
// advertises the workspace, and these definitions are sent alongside any
// user-defined tools. Everything is confined to that directory server-side.
import { postJson } from "./api";
import type { AgentPermissions, AgentPlatform, AgentSearchInfo, AgentWorkspaceInfo, PropsPayload, ToolDefinition } from "../types";

/** Shell commands are the one built-in that asks before it runs. */
export const APPROVAL_TOOL = "run_command";

/**
 * The temperature ceiling for a run with workspace tools, whatever the chat
 * slider says. A run of eight spaces and a run of nine are adjacent tokens,
 * so chat temperature flips exactly the near-ties that put a wrong indent
 * into new_string; code wants the argmax, not variety.
 */
export const AGENT_TEMPERATURE_CAP = 0.2;

/** Whether a called name is the shell tool, however the model spelled it. */
export function needsApproval(name: string): boolean {
  return canonicalName(name) === APPROVAL_TOOL;
}

/** The server's default workspace directory, if it has one ready. */
export function workspaceRoot(props: PropsPayload | null): string | null {
  return props?.agent_workspace ?? null;
}

export function hasWorkspace(props: PropsPayload | null): boolean {
  return Boolean(workspaceRoot(props));
}

/** Whether the server has agent tools at all (it may still have no directory). */
export function agentToolsAvailable(props: PropsPayload | null): boolean {
  return Array.isArray(props?.agent_workspaces);
}

/** Every directory the server offers, in its order: flag roots first. */
export function workspaceList(props: PropsPayload | null): AgentWorkspaceInfo[] {
  return props?.agent_workspaces ?? [];
}

/**
 * The directory a run works in. A run names one by id; a run saved before
 * the registry existed has none and gets the server's default, which is what
 * it had at the time.
 */
export function workspaceFor(props: PropsPayload | null, workspaceId: string | undefined): AgentWorkspaceInfo | null {
  const list = workspaceList(props);
  if (workspaceId) return list.find((workspace) => workspace.id === workspaceId) ?? null;
  const root = workspaceRoot(props);
  return list.find((workspace) => workspace.path === root) ?? null;
}

// ---- Permissions -------------------------------------------------------------

export const DEFAULT_PERMISSIONS: AgentPermissions = "workspace-write";

export interface PermissionPreset {
  value: AgentPermissions;
  label: string;
  /** One sentence on what the preset lets the run do. */
  description: string;
}

/**
 * The presets a run chooses between. Their order is the order the UI shows
 * them, safest first. Every preset confines the file tools to the workspace;
 * none confines a shell command, so the description says what the approval
 * prompt is doing rather than promising a sandbox.
 */
export const PERMISSION_PRESETS: PermissionPreset[] = [
  {
    value: "read-only",
    label: "Read only",
    description: "The model can list and read files; the server refuses every write and edit. Commands still run only with your approval, and nothing stops an approved command from changing files.",
  },
  {
    value: "workspace-write",
    label: "Workspace write, ask",
    description: "The model can read, write, and edit files inside the workspace. Every shell command waits for your approval.",
  },
  {
    value: "auto-approve",
    label: "Auto-approve commands",
    description: "As above, but shell commands run without asking. Commands are not sandboxed: use this only in a directory you can afford to lose.",
  },
];

export function permissionPreset(value: AgentPermissions | undefined): PermissionPreset {
  return PERMISSION_PRESETS.find((preset) => preset.value === (value ?? DEFAULT_PERMISSIONS)) ?? PERMISSION_PRESETS[1];
}

/** The file-effect mode the server enforces for a preset. */
export function modeOf(permissions: AgentPermissions | undefined): "read-only" | "workspace-write" {
  return permissions === "read-only" ? "read-only" : "workspace-write";
}

/** What a tool call carries so the server runs it in the right place, the right way. */
export interface RunContext {
  /** The workspace id; omitted, the server uses its default. */
  workspace?: string;
  mode?: "read-only" | "workspace-write";
}

export function workspacePlatform(props: PropsPayload | null): AgentPlatform | null {
  return props?.agent_platform ?? null;
}

/** What the server says about web search; null on a server too old to say. */
export function searchInfo(props: PropsPayload | null): AgentSearchInfo | null {
  return props?.agent_search ?? null;
}

/**
 * Whether to offer web_search this run. A server that cannot search — no key
 * for the provider it was pointed at — is one where every search is a turn
 * spent on a 400, so the tool is withheld instead. A server too old to report
 * anything has no /agent/search either.
 */
export function searchAvailable(props: PropsPayload | null): boolean {
  return searchInfo(props)?.ready === true;
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
    name: "web_search",
    description:
      "Search the web and get back a handful of results: title, url, and a sentence each. Use it when you do not already know the page that answers the question, then read the most promising result with fetch_url — the snippets are there to choose between, not to answer from. One search with the words that would appear on the page beats three vague ones.",
    endpoint: "/agent/search",
    parameters: {
      type: "object",
      properties: {
        query: { type: "string", description: "What to search for, as you would type it into a search engine." },
        count: { type: "number", description: "How many results to return (default 5, max 20)." },
      },
      required: ["query"],
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

/**
 * The built-ins as tool definitions for the request builder. A read-only run
 * is not offered the writers at all: a tool the model cannot use is a call it
 * will make anyway and a turn spent on the refusal.
 */
export function builtinToolDefinitions(permissions: AgentPermissions = DEFAULT_PERMISSIONS, search = true): ToolDefinition[] {
  const withheld = new Set<string>();
  if (permissions === "read-only") {
    withheld.add("write_file").add("edit_file");
  }
  if (!search) withheld.add("web_search");
  const offered = BUILTINS.filter((tool) => !withheld.has(tool.name));
  return offered.map((tool) => ({
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
export async function runBuiltinTool(name: string, argsText: string, signal?: AbortSignal, run: RunContext = {}): Promise<BuiltinResult> {
  const tool = BY_NAME.get(canonicalName(name));
  if (!tool) return { ok: false, result: `No built-in tool named ${name}` };
  let args: unknown;
  try {
    args = argsText.trim() ? JSON.parse(argsText) : {};
  } catch {
    return { ok: false, result: "Arguments were not valid JSON; call the tool again with a well-formed arguments object." };
  }
  if (!args || typeof args !== "object" || Array.isArray(args)) {
    return { ok: false, result: "Arguments must be a JSON object matching the tool's schema." };
  }
  // The run's workspace and mode ride on every call, after the model's
  // arguments so a model cannot pick a different directory by naming one.
  const body = { ...(args as Record<string, unknown>), ...(run.workspace ? { workspace: run.workspace } : {}), ...(run.mode ? { mode: run.mode } : {}) };
  try {
    const payload = await postJson<Record<string, unknown>>(tool.endpoint, body, { signal });
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
    const content = String(payload.content ?? "");
    // An empty file is a real answer, but a result with no characters in it
    // reads to the model as a turn that said nothing at all.
    if (!content) return `${payload.path ?? "the file"} is empty (0 bytes)`;
    return `${content}${suffix}`;
  }
  if (name === "write_file") {
    return `${payload.created ? "Created" : "Wrote"} ${payload.path} (${payload.bytes} bytes, ${payload.line_ending ?? "lf"} line endings)`;
  }
  if (name === "edit_file") {
    const count = Number(payload.replacements ?? 0);
    const head = `Replaced ${count} occurrence${count === 1 ? "" : "s"} in ${payload.path} (${payload.bytes} bytes)`;
    // new_string is the one string no exact match guards: a miscounted indent
    // lands silently and surfaces turns later as an interpreter error. The
    // echoed region lets the model see the lines it wrote while the mistake
    // is still one edit_file away from fixed. Older servers send no snippet.
    if (!payload.snippet) return head;
    return `${head}. The edited region now reads, from line ${payload.snippet_line}:\n${payload.snippet}`;
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
  if (name === "web_search") {
    const results = (payload.results as Array<{ title?: string; url?: string; snippet?: string }>) ?? [];
    if (!results.length) return `No results for "${payload.query}" (${payload.provider}). Try different words, or fetch_url a page you already know.`;
    // Numbered, url on its own line: the next call is a fetch of one of
    // these, and the model has to copy the url exactly to make it.
    return results
      .map((result, index) => `${index + 1}. ${result.title || "(untitled)"}\n   ${result.url}\n   ${result.snippet || "(no summary)"}`)
      .join("\n");
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
    return `${builtins.join(", ")} ${plural ? "are workspace tools" : "is a workspace tool"}, and this run has no workspace. Add a directory in the Agent tab (from a browser on the server's machine) or restart the server with --agent-workspace DIR — or answer the call by hand below.`;
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
/** The host under the name it is known by; the server reports sys.platform. */
function osLabel(os: string): string {
  return { windows: "Windows", darwin: "macOS", linux: "Linux" }[os] ?? os;
}

function platformLines(platform: AgentPlatform): string[] {
  const windows = platform.os === "windows";
  const lines = [`The machine runs ${osLabel(platform.os)} and commands go to ${platform.shell}, so write every command in that shell's syntax.`];
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
  } else if (platform.shell === "sh") {
    // The POSIX counterpart of the PowerShell problem: /bin/sh is dash on
    // most Linux distributions, so the bash a model writes by reflex --
    // [[ ]], arrays, source, pipefail -- fails on syntax, not on the task.
    lines.push(
      "That shell is /bin/sh, which is not bash on most Linux systems: [[ ]], arrays, source and pipefail are not available. Write portable sh, or run bash -c '...' explicitly when you need it.",
    );
  }
  lines.push(
    `Paths on this host use ${platform.path_separator === "\\" ? "backslashes" : "forward slashes"}, and the tools accept either; keep tool paths relative to the workspace root.`,
  );
  if (platform.line_ending === "crlf") {
    lines.push("Files here tend to use CRLF line endings; write \\n in tool arguments and the server keeps each file's existing endings.");
  }
  return lines;
}

/**
 * When each tool is the right one, in one line each. The schemas the model
 * receives say what a tool takes; this says what it is *for*, which is the
 * choice a run actually gets wrong -- rewriting a file it meant to edit,
 * reading a directory to find a string, fetching a page whole.
 */
const TOOL_GUIDANCE: Record<string, string> = {
  list_dir: "see what is in a directory before guessing at a name",
  read_file: "read a file; do this before editing one, because edit_file needs text you have actually seen",
  edit_file: "change part of a file that exists — the normal way to edit",
  write_file: "create a file, or replace one whole; not for a small change, and never for a formatting fix",
  [APPROVAL_TOOL]: "build, test, search, inspect. The user approves each command before it runs",
  web_search: "find pages when you do not already have a url; read the one you pick with fetch_url",
  fetch_url: "read a web page as text; always pass query so you get the part you need",
};

function toolLines(names: string[]): string[] {
  const known = names.filter((name) => TOOL_GUIDANCE[name]);
  const other = names.filter((name) => !TOOL_GUIDANCE[name]);
  const lines = known.map((name) => `- ${name}: ${TOOL_GUIDANCE[name]}.`);
  if (other.length) lines.push(`- Also available: ${other.join(", ")}. Their descriptions say what they do.`);
  return lines;
}

/** What the run knows about itself when it writes its prompt. */
export interface AgentPromptContext {
  /** The workspace directory every path is relative to. */
  root: string;
  /** The host, from /props; omitted when the server is too old to report it. */
  platform?: AgentPlatform | null;
  /** Tool names being sent with this request, built-in and user-defined. */
  tools?: string[];
  /** The turn cap that will stop the run. */
  turnCap?: number;
  /** What the run may do; shapes the WORKING rules. */
  permissions?: AgentPermissions;
}

/**
 * The system prompt an agent run prepends when the workspace tools exist.
 *
 * Written for the models this server runs, which are small: they follow short
 * imperative rules under headings far better than a paragraph of prose, and
 * they fail in specific, predictable ways -- inventing a tool result rather
 * than waiting for one, retrying a failed call unchanged, reading a whole
 * directory to find a string, announcing an edit they never made, looping
 * long after the answer was ready. Each section below is aimed at one of
 * those, so the prompt is a list of the ways a run goes wrong rather than a
 * description of the job.
 */
export function agentSystemPrompt(context: AgentPromptContext): string {
  const { root, platform, tools = [], turnCap, permissions = DEFAULT_PERMISSIONS } = context;
  const sections: string[] = [
    [
      `You are a coding agent working in ${root} on the user's machine.`,
      "Carry the task out yourself with the tools you have; do not hand the user instructions for work you could do.",
      "Every path you pass to a tool is relative to that directory.",
    ].join(" "),
  ];

  if (platform) sections.push(["HOST", ...platformLines(platform).map((line) => `- ${line}`)].join("\n"));
  if (tools.length) sections.push(["TOOLS", ...toolLines(tools)].join("\n"));

  sections.push(
    [
      "CALLING A TOOL",
      "- One call at a time. Emit it, wait for the result, then decide the next step from what came back.",
      "- Never write a tool result yourself, and never say a file was changed, a command ran, or a test passed unless a result said so.",
      "- Arguments are a JSON object matching the tool's schema: the exact tool name, every required field, no commentary around it.",
      "- A failed call fails the same way when repeated. Read the error, change the arguments or the approach, then try again.",
      "- If two attempts at one approach fail, take a different one or ask the user rather than a third.",
    ].join("\n"),
    [
      "WORKING",
      ...(permissions === "read-only"
        ? [
            "- This run is read-only: you have no tools that change files, and you must not run a command that creates, modifies, or deletes anything. Read, search, and report; if the task needs a change, describe it for the user instead.",
          ]
        : [
            "- Look before you write: list or read first, edit second, then run something that proves it worked.",
            "- Change files with edit_file. Reach for write_file only for a new file — never to fix indentation or formatting, which a rewrite makes worse; repair the broken lines with edit_file, or run a formatter.",
            "- An edit's result shows the region it changed. Check its indentation, and fix it now if it is wrong.",
          ]),
      "- Every result stays in your context for the rest of the run, so ask narrowly: search with a command instead of reading files one by one, read the file you need rather than everything near it, and give fetch_url a query instead of pulling a whole page.",
      ...(permissions === "auto-approve"
        ? ["- Commands run without the user's approval in this run, so be conservative: no command that deletes, overwrites, or reaches outside the workspace unless the task plainly requires it."]
        : [
            "- Before a command needs approval, say in one line what it will do and why, so the user can decide without reading the flags.",
            "- If the user denies a command, do not send it again. Find another way or ask what they would prefer.",
          ]),
    ].join("\n"),
    [
      "FINISHING",
      "- Stop as soon as the task is done and answer in prose: what you changed, what you ran, and what it said.",
      "- Stop and ask when you are blocked or when the next step is a decision only the user can make.",
      "- Report what actually happened, failures included. A wrong answer costs the user more than an unfinished one.",
    ].join("\n"),
  );

  // A model that knows its budget spends it on the task; one that does not
  // explores until the cap stops it mid-step. Only the cap goes here: this
  // prompt is the prefix of every request in the run, and the server's prefix
  // cache dies at the first changed token.
  if (turnCap) {
    sections.push(
      `This run has a budget of ${turnCap} tool-calling turns; when few are left a note under the newest message says so. Finish what matters most by then and report where you got to.`,
    );
  }
  return sections.join("\n\n");
}

/**
 * How many turns from the cap the countdown starts. Before that there is no
 * note at all: see turnBudgetNote for why silence is worth more than a number.
 */
export const TURN_WARNING_AT = 3;

/**
 * The warning appended after the newest message when the budget runs short.
 *
 * It used to count down on every turn, which cost more than it was worth. A
 * note appended to the newest message is not part of the run's history, so
 * next turn that message goes out without it — and a request that rewrites
 * even the last few characters of the previous one is no longer an extension
 * of it. On a runtime whose KV cannot rewind (any recurrent layer, or
 * speculative drafts) reuse then falls back to a sparse checkpoint, which
 * measured 40% of a 10k-token prompt re-evaluated every turn. Every turn that
 * sends no note is a strict extension of the last, which is the case the
 * cache handles perfectly, so the note is saved for the turns where knowing
 * actually changes what the model should do.
 */
export function turnBudgetNote(turn: number, turnCap: number): string {
  const left = turnCap - turn;
  if (left <= 0 || left > TURN_WARNING_AT) return "";
  return `[${left} tool-calling turn${left === 1 ? "" : "s"} left in this run.]`;
}
