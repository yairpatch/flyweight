import { afterEach, describe, expect, it, vi } from "vitest";
import {
  PERMISSION_PRESETS,
  agentSystemPrompt,
  builtinToolDefinitions,
  modeOf,
  workspaceFor,
  isBuiltinTool,
  missingHandlerReason,
  needsApproval,
  runBuiltinTool,
  searchAvailable,
  TURN_WARNING_AT,
  turnBudgetNote,
  turnCapReason,
} from "./agentTools";
import type { PropsPayload } from "../types";

function respondWith(payload: unknown, ok = true) {
  const fetchMock = vi.fn(async (_url: string, _init?: RequestInit) =>
    new Response(JSON.stringify(payload), { status: ok ? 200 : 403, headers: { "Content-Type": "application/json" } }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("builtinToolDefinitions", () => {
  it("declares each workspace tool with a parseable schema", () => {
    const definitions = builtinToolDefinitions();
    expect(definitions.map((tool) => tool.name).sort()).toEqual([
      "edit_file",
      "fetch_url",
      "list_dir",
      "read_file",
      "run_command",
      "web_search",
      "write_file",
    ]);
    for (const tool of definitions) {
      expect(tool.enabled).toBe(true);
      expect(tool.description).not.toBe("");
      expect(JSON.parse(tool.parameters).type).toBe("object");
      expect(isBuiltinTool(tool.name)).toBe(true);
    }
    expect(isBuiltinTool("get_weather")).toBe(false);
  });
});

describe("permissions", () => {
  it("withholds the writing tools from a read-only run", () => {
    const names = builtinToolDefinitions("read-only").map((tool) => tool.name);
    expect(names).not.toContain("write_file");
    expect(names).not.toContain("edit_file");
    expect(names).toContain("read_file");
    expect(builtinToolDefinitions("auto-approve").map((tool) => tool.name)).toContain("write_file");
  });

  it("withholds web_search from a server that cannot search", () => {
    // A tool the server would refuse is a turn spent on the refusal, so a
    // server with no working backend is not offered the tool at all.
    expect(builtinToolDefinitions("workspace-write", false).map((tool) => tool.name)).not.toContain("web_search");
    expect(searchAvailable({ agent_search: { provider: "brave", ready: false, detail: "needs a key" } } as PropsPayload)).toBe(false);
    expect(searchAvailable({ agent_search: { provider: "duckduckgo", ready: true } } as PropsPayload)).toBe(true);
    // A server too old to report search has no /agent/search either.
    expect(searchAvailable({} as PropsPayload)).toBe(false);
    expect(searchAvailable(null)).toBe(false);
  });

  it("maps each preset to the mode the server enforces", () => {
    expect(modeOf("read-only")).toBe("read-only");
    expect(modeOf("workspace-write")).toBe("workspace-write");
    expect(modeOf("auto-approve")).toBe("workspace-write");
    expect(modeOf(undefined)).toBe("workspace-write");
  });

  it("lists the presets safest first and never promises a command sandbox", () => {
    expect(PERMISSION_PRESETS.map((preset) => preset.value)).toEqual(["read-only", "workspace-write", "auto-approve"]);
    for (const preset of PERMISSION_PRESETS) expect(preset.description).not.toMatch(/sandboxed\./);
    expect(PERMISSION_PRESETS[2].description).toContain("not sandboxed");
  });

  it("tells a read-only run not to change anything, and an auto-approve run to be careful", () => {
    const readOnly = agentSystemPrompt({ root: "/w", permissions: "read-only" });
    expect(readOnly).toContain("read-only");
    expect(readOnly).not.toContain("Change files with edit_file");
    const auto = agentSystemPrompt({ root: "/w", permissions: "auto-approve" });
    expect(auto).toContain("without the user's approval");
    expect(auto).not.toContain("Before a command needs approval");
    const asks = agentSystemPrompt({ root: "/w" });
    expect(asks).toContain("Before a command needs approval");
  });
});

describe("workspaceFor", () => {
  const props = {
    agent_workspace: "/srv/one",
    agent_workspaces: [
      { id: "a", path: "/srv/one", title: "one", source: "flag" as const, exists: true },
      { id: "b", path: "/srv/two", title: "two", source: "registered" as const, exists: false },
    ],
  };

  it("finds a run's workspace by id, and an old run's by the server default", () => {
    expect(workspaceFor(props, "b")?.path).toBe("/srv/two");
    expect(workspaceFor(props, undefined)?.id).toBe("a");
    expect(workspaceFor(props, "gone")).toBeNull();
    expect(workspaceFor(null, "a")).toBeNull();
  });
});

describe("name matching", () => {
  it("recognizes a namespaced or padded call as the same built-in", async () => {
    expect(isBuiltinTool("functions.list_dir")).toBe(true);
    expect(isBuiltinTool(" read_file\n")).toBe(true);
    expect(isBuiltinTool("functions.get_weather")).toBe(false);
    const fetchMock = respondWith({ path: "a.txt", content: "hi", size: 2, truncated: false });
    const result = await runBuiltinTool("functions.read_file", '{"path":"a.txt"}');
    expect(result).toEqual({ ok: true, result: "hi" });
    expect(fetchMock.mock.calls[0][0]).toBe("/agent/fs/read");
  });

  it("still asks for approval when the shell tool is namespaced", () => {
    expect(needsApproval("functions.run_command")).toBe(true);
    expect(needsApproval("run_command")).toBe(true);
    expect(needsApproval("read_file")).toBe(false);
  });
});

describe("agentSystemPrompt", () => {
  const windows = { os: "windows", shell: "powershell", path_separator: "\\", line_ending: "crlf" } as const;
  const linux = { os: "linux", shell: "sh", path_separator: "/", line_ending: "lf" } as const;

  it("tells the model which shell it is writing for on Windows", () => {
    const prompt = agentSystemPrompt({ root: "C:\\work", platform: windows });
    expect(prompt).toContain("C:\\work");
    expect(prompt).toContain("powershell");
    expect(prompt).toContain("Select-String");
    expect(prompt).toContain("call operator");
    expect(prompt).toContain("backslashes");
    expect(prompt).toContain("CRLF");
  });

  it("warns a POSIX host about its own shell instead", () => {
    const prompt = agentSystemPrompt({ root: "/srv/work", platform: linux });
    expect(prompt).toContain("Linux");
    expect(prompt).toContain("commands go to sh");
    // /bin/sh is dash on most distributions, which is the POSIX equivalent of
    // the PowerShell trap: bash syntax fails before the task is attempted.
    expect(prompt).toContain("not bash");
    expect(prompt).toContain("forward slashes");
    expect(prompt).not.toContain("Select-String");
    expect(prompt).not.toContain("CRLF");
  });

  it("names macOS rather than the platform string the server reports", () => {
    const prompt = agentSystemPrompt({ root: "/Users/me/work", platform: { ...linux, os: "darwin" } });
    expect(prompt).toContain("The machine runs macOS");
    expect(prompt).not.toContain("darwin");
  });

  it("leaves a shell it was told nothing about alone", () => {
    // FLYWEIGHT_AGENT_SHELL=fish, say: naming the shell is still right, but
    // none of the sh advice applies to it.
    const prompt = agentSystemPrompt({ root: "/srv/work", platform: { ...linux, shell: "fish" } });
    expect(prompt).toContain("commands go to fish");
    expect(prompt).not.toContain("not bash");
  });

  it("names the loop's own failure modes, not just the job", () => {
    const prompt = agentSystemPrompt({ root: "/srv/work", platform: linux, tools: builtinToolDefinitions().map((tool) => tool.name) });
    for (const heading of ["HOST", "TOOLS", "CALLING A TOOL", "WORKING", "FINISHING"]) {
      expect(prompt).toContain(heading);
    }
    // The specific ways a small model breaks a run.
    expect(prompt).toContain("Never write a tool result yourself");
    expect(prompt).toContain("fails the same way when repeated");
    expect(prompt).toContain("One call at a time");
    expect(prompt).toContain("stays in your context");
    expect(prompt).toContain("Stop as soon as the task is done");
    // The indentation spiral: a slipped indent, then a whole-file rewrite
    // that slips more. The prompt blocks the rewrite and points at the echo.
    expect(prompt).toContain("never to fix indentation");
    expect(prompt).toContain("Check its indentation");
  });

  it("says what each tool it was given is for, and nothing about the rest", () => {
    const prompt = agentSystemPrompt({ root: "/srv/work", tools: ["read_file", "edit_file", "get_weather"] });
    expect(prompt).toContain("- edit_file: change part of a file that exists");
    expect(prompt).toContain("Also available: get_weather");
    // A tool this run cannot call has no business being described to it.
    expect(prompt).not.toContain("- run_command:");
    expect(prompt).not.toContain("- fetch_url:");
  });

  it("states the budget as a fixed fact, so the prompt is byte-identical every turn", () => {
    // The countdown lives in turnBudgetNote at the request's tail; a number
    // that changed here would invalidate the server's cached prefix per turn.
    const prompt = agentSystemPrompt({ root: "/w", turnCap: 8 });
    expect(prompt).toContain("budget of 8 tool-calling turns");
    expect(prompt).toContain("note under the newest message");
    expect(agentSystemPrompt({ root: "/w" })).not.toContain("budget");
  });

  it("still works when the server reports no platform", () => {
    const prompt = agentSystemPrompt({ root: "/srv/work" });
    expect(prompt).toContain("/srv/work");
    expect(prompt).toContain("edit_file");
    expect(prompt).not.toContain("HOST");
  });
});

describe("turnBudgetNote", () => {
  it("warns near the cap and says nothing before it", () => {
    expect(turnBudgetNote(6, 8)).toBe("[2 tool-calling turns left in this run.]");
    expect(turnBudgetNote(7, 8)).toBe("[1 tool-calling turn left in this run.]");
    // Spent: nothing to promise.
    expect(turnBudgetNote(8, 8)).toBe("");
  });

  it("stays silent while there is budget, so the request keeps growing by appends only", () => {
    // A note is not part of the run's history, so the message it hangs on
    // goes out without it next turn -- and a request that rewrites the tail
    // of the last one is no longer an extension of it. On a runtime whose KV
    // cannot rewind that costs the whole cached prefix back to a checkpoint,
    // every turn, which is far more than a countdown is worth.
    for (let turn = 1; turn < 8 - TURN_WARNING_AT; turn += 1) expect(turnBudgetNote(turn, 8)).toBe("");
    expect(turnBudgetNote(8 - TURN_WARNING_AT, 8)).not.toBe("");
  });
});

describe("pause reasons", () => {
  it("says where a workspace comes from when the model asked for a workspace tool", () => {
    const reason = missingHandlerReason(["list_dir"], false);
    expect(reason).toContain("list_dir is a workspace tool");
    expect(reason).toContain("Agent tab");
    expect(reason).toContain("--agent-workspace DIR");
  });

  it("points at the Tools panel when a user tool has no handler", () => {
    const reason = missingHandlerReason(["get_weather"], true);
    expect(reason).toContain("get_weather");
    expect(reason).toContain("no JavaScript handler");
    expect(reason).not.toContain("--agent-workspace");
  });

  it("says what the turn cap was", () => {
    expect(turnCapReason(8, 8)).toContain("8-turn budget");
  });
});

describe("runBuiltinTool", () => {
  it("posts the model's arguments to the matching endpoint", async () => {
    const fetchMock = respondWith({ path: "notes.md", content: "hello", size: 5, truncated: false });
    const result = await runBuiltinTool("read_file", '{"path":"notes.md"}');
    expect(result).toEqual({ ok: true, result: "hello" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/agent/fs/read");
    expect(JSON.parse(String(init?.body))).toEqual({ path: "notes.md" });
  });

  it("names the run's workspace and mode after the model's arguments, so the model cannot pick another", async () => {
    const fetchMock = respondWith({ path: "notes.md", content: "hello", size: 5, truncated: false });
    await runBuiltinTool("read_file", '{"path":"notes.md","workspace":"elsewhere","mode":"workspace-write"}', undefined, {
      workspace: "ws-1",
      mode: "read-only",
    });
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(String(init?.body))).toEqual({ path: "notes.md", workspace: "ws-1", mode: "read-only" });
  });

  it("refuses arguments that are not an object before they reach the server", async () => {
    const fetchMock = respondWith({});
    const result = await runBuiltinTool("list_dir", "[1, 2]");
    expect(result.ok).toBe(false);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("tells the model when a result was clipped", async () => {
    respondWith({ path: "big.txt", content: "xxx", size: 999999, truncated: true });
    const result = await runBuiltinTool("read_file", '{"path":"big.txt"}');
    expect(result.result).toContain("truncated");
    expect(result.result).toContain("999999");
  });

  it("summarizes a command by exit code and streams", async () => {
    respondWith({ command: "pytest", exit_code: 1, timed_out: false, stdout: "1 failed", stderr: "", stdout_truncated: false, stderr_truncated: false });
    const result = await runBuiltinTool("run_command", '{"command":"pytest"}');
    expect(result.ok).toBe(true);
    expect(result.result).toContain("exit code 1");
    expect(result.result).toContain("1 failed");
  });

  it("reports a refused path as a failed result rather than throwing", async () => {
    respondWith({ error: { message: "path is outside the agent workspace /tmp/ws" } }, false);
    const result = await runBuiltinTool("read_file", '{"path":"../../etc/passwd"}');
    expect(result.ok).toBe(false);
    expect(result.result).toContain("outside the agent workspace");
  });

  it("reports an edit by what it changed, counts alone on a server with no snippet", async () => {
    const fetchMock = respondWith({ path: "app.py", replacements: 2, bytes: 512, line_ending: "crlf" });
    const result = await runBuiltinTool("edit_file", '{"path":"app.py","old_string":"a","new_string":"b","replace_all":true}');
    expect(result.ok).toBe(true);
    expect(result.result).toBe("Replaced 2 occurrences in app.py (512 bytes)");
    expect(fetchMock.mock.calls[0][0]).toBe("/agent/fs/edit");
  });

  it("echoes the edited region so the model sees the lines it wrote", async () => {
    // new_string is the one string no exact match guards; without the echo a
    // slipped indent surfaces turns later, as an interpreter error.
    respondWith({ path: "app.py", replacements: 1, bytes: 512, line_ending: "lf", snippet: "def main():\n     x = 2\n    return x", snippet_line: 4 });
    const result = await runBuiltinTool("edit_file", '{"path":"app.py","old_string":"    x = 1","new_string":"     x = 2"}');
    expect(result.result).toContain("Replaced 1 occurrence in app.py");
    expect(result.result).toContain("from line 4:\ndef main():\n     x = 2");
  });

  it("passes a failed edit's advice through to the model", async () => {
    respondWith({ error: { message: "old_string appears 3 times in app.py; include the surrounding lines to make it unique, or pass replace_all" } }, false);
    const result = await runBuiltinTool("edit_file", '{"path":"app.py","old_string":"x","new_string":"y"}');
    expect(result.ok).toBe(false);
    expect(result.result).toContain("replace_all");
  });

  it("says whether a write created the file and what endings it kept", async () => {
    respondWith({ path: "new.txt", bytes: 12, created: true, line_ending: "lf" });
    const result = await runBuiltinTool("write_file", '{"path":"new.txt","content":"hello"}');
    expect(result.result).toBe("Created new.txt (12 bytes, lf line endings)");
  });

  it("says what a page's extract left out and how to reach the rest", async () => {
    respondWith({
      url: "https://example.com/docs",
      status: 200,
      content_type: "text/html",
      title: "Widget docs",
      body: "The timeout is eight seconds.",
      chars: 29,
      total_chars: 40000,
      next_offset: 29,
      selection: "query",
      truncated: true,
    });
    const result = await runBuiltinTool("fetch_url", '{"url":"https://example.com/docs","query":"timeout"}');
    expect(result.result).toContain("Widget docs");
    expect(result.result).toContain("The timeout is eight seconds.");
    expect(result.result).toContain("passages matching your query: 29 of 40000");
  });

  it("offers the next offset when it sent the head of a long page", async () => {
    respondWith({ status: 200, content_type: "text/plain", body: "x", chars: 6000, total_chars: 20000, next_offset: 6000, selection: "head", truncated: true });
    const result = await runBuiltinTool("fetch_url", '{"url":"https://example.com/log"}');
    expect(result.result).toContain("offset 6000");
    expect(result.result).toContain("pass a query");
  });

  it("lists search results as numbered links to fetch next", async () => {
    respondWith({
      query: "gguf runtime",
      provider: "duckduckgo",
      count: 2,
      results: [
        { title: "Flyweight", url: "https://example.com/a", snippet: "A GGUF runtime." },
        { title: "", url: "https://example.org/b", snippet: "" },
      ],
    });
    const result = await runBuiltinTool("web_search", '{"query":"gguf runtime"}');
    expect(result.ok).toBe(true);
    expect(result.result).toContain("1. Flyweight");
    expect(result.result).toContain("https://example.com/a");
    expect(result.result).toContain("A GGUF runtime.");
    // A result with nothing in it still has to be pickable by its number.
    expect(result.result).toContain("2. (untitled)");
    expect(result.result).toContain("(no summary)");
  });

  it("tells the model what to do when a search finds nothing", async () => {
    respondWith({ query: "asdfqwer", provider: "duckduckgo", count: 0, results: [] });
    const result = await runBuiltinTool("web_search", '{"query":"asdfqwer"}');
    expect(result.result).toContain("No results");
    expect(result.result).toContain("fetch_url");
  });

  it("hands malformed arguments back to the model instead of calling the server", async () => {
    const fetchMock = respondWith({});
    const result = await runBuiltinTool("write_file", "{not json");
    expect(result.ok).toBe(false);
    expect(result.result).toContain("valid JSON");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
