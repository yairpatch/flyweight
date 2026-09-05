import { afterEach, describe, expect, it, vi } from "vitest";
import { builtinToolDefinitions, isBuiltinTool, missingHandlerReason, needsApproval, runBuiltinTool, turnCapReason } from "./agentTools";

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

  it("hands malformed arguments back to the model instead of calling the server", async () => {
    const fetchMock = respondWith({});
    const result = await runBuiltinTool("write_file", "{not json");
    expect(result.ok).toBe(false);
    expect(result.result).toContain("valid JSON");
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
