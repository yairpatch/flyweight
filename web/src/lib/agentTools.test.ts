import { afterEach, describe, expect, it, vi } from "vitest";
import {
  agentSystemPrompt,
  builtinToolDefinitions,
  isBuiltinTool,
  missingHandlerReason,
  needsApproval,
  runBuiltinTool,
  turnBudgetNote,
  turnCapReason,
} from "./agentTools";

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
  it("counts down what the system prompt only promises", () => {
    expect(turnBudgetNote(1, 8)).toBe("[7 tool-calling turns left in this run.]");
    expect(turnBudgetNote(7, 8)).toBe("[1 tool-calling turn left in this run.]");
    // Spent: nothing to promise.
    expect(turnBudgetNote(8, 8)).toBe("");
  });
});

describe("pause reasons", () => {
  it("names the missing server flag when the model asked for a workspace tool", () => {
    const reason = missingHandlerReason(["list_dir"], false);
    expect(reason).toContain("list_dir is a workspace tool");
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

  it("hands malformed arguments back to the model instead of calling the server", async () => {
    const fetchMock = respondWith({});
    const result = await runBuiltinTool("write_file", "{not json");
    expect(result.ok).toBe(false);
    expect(result.result).toContain("valid JSON");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
