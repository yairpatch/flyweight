import { describe, expect, it } from "vitest";
import { budgetChars, compactMessages, compactionNote, contextOverflow, conversationChars, observedCharsPerToken, retryBudget } from "./compaction";
import type { Message, RequestRecord } from "../types";

let counter = 0;
function message(role: Message["role"], content: string, extra: Partial<Message> = {}): Message {
  counter += 1;
  return { id: `m${counter}`, role, content, createdAt: counter, ...extra };
}

/** A run: the task, then N steps of "call a tool, get a big result". */
function run(steps: number, resultChars = 2000): Message[] {
  const messages: Message[] = [message("user", "List the workspace and summarize the project.")];
  for (let index = 0; index < steps; index += 1) {
    const callId = `call-${index}`;
    messages.push(message("assistant", "", { toolCalls: [{ id: callId, name: "read_file", arguments: `{"path":"f${index}.py"}` }] }));
    messages.push(message("tool", "x".repeat(resultChars), { toolCallId: callId, toolName: "read_file", auto: true }));
  }
  return messages;
}

describe("compactMessages", () => {
  it("leaves a run that fits completely alone", () => {
    const messages = run(3);
    const outcome = compactMessages(messages, 1_000_000);
    expect(outcome.messages).toBe(messages);
    expect(outcome.removedChars).toBe(0);
  });

  it("stubs the oldest tool results and keeps the newest intact", () => {
    const messages = run(8);
    const outcome = compactMessages(messages, 8000);
    expect(outcome.stubbed).toBeGreaterThan(0);
    expect(conversationChars(outcome.messages)).toBeLessThanOrEqual(8000);
    const stub = outcome.messages.find((item) => item.content.startsWith("[read_file result removed"));
    expect(stub?.content).toContain("2000 characters");
    // The last exchange is what the model is working from; it survives whole.
    expect(outcome.messages[outcome.messages.length - 1].content).toBe("x".repeat(2000));
  });

  it("keeps the task that opened the run", () => {
    const messages = run(12);
    const outcome = compactMessages(messages, 4000);
    expect(outcome.messages[0].content).toContain("List the workspace");
  });

  it("never leaves a tool result without the call it answers", () => {
    const messages = run(12);
    const outcome = compactMessages(messages, 3000);
    expect(outcome.dropped).toBeGreaterThan(0);
    const calls = new Set(outcome.messages.flatMap((item) => (item.toolCalls ?? []).map((call) => call.id)));
    for (const item of outcome.messages) {
      if (item.role === "tool" && item.toolCallId) expect(calls.has(item.toolCallId)).toBe(true);
    }
  });

  it("clips a result that is larger than the window on its own", () => {
    const messages = [
      message("user", "Read the log and tell me what failed."),
      message("assistant", "", { toolCalls: [{ id: "c1", name: "read_file", arguments: '{"path":"build.log"}' }] }),
      message("tool", `START${"y".repeat(80000)}`, { toolCallId: "c1", toolName: "read_file", auto: true }),
    ];
    const outcome = compactMessages(messages, 6000);
    const result = outcome.messages[2];
    expect(result.content.startsWith("START")).toBe(true);
    expect(result.content).toContain("more characters removed");
    expect(conversationChars(outcome.messages)).toBeLessThanOrEqual(6000);
    // Nothing was dropped: the step is the only one there is.
    expect(outcome.dropped).toBe(0);
  });

  it("tells the model what it lost", () => {
    const outcome = compactMessages(run(10), 5000);
    const note = compactionNote(outcome);
    expect(note).toContain("removed to fit the context window");
    expect(compactionNote({ messages: [], removedChars: 0, stubbed: 0, dropped: 0 })).toBe("");
  });
});

describe("budgetChars", () => {
  it("reserves room for the answer and the scaffolding", () => {
    expect(budgetChars(undefined, 4096)).toBe(Number.POSITIVE_INFINITY);
    expect(budgetChars(32768, 4096)).toBeLessThan(32768 * 3.5);
    expect(budgetChars(32768, 4096)).toBeGreaterThan(0);
    // A window swallowed entirely by the output cap still leaves a floor.
    expect(budgetChars(4096, 4096)).toBe(2000);
  });

  it("uses the model's measured ratio when one is known", () => {
    expect(budgetChars(32768, 4096, 5)).toBeGreaterThan(budgetChars(32768, 4096));
    // Usage from a real request calibrates it; nonsense ratios are ignored.
    expect(observedCharsPerToken(40000, 10000)).toBe(4);
    expect(observedCharsPerToken(40000, 100)).toBeNull();
    expect(observedCharsPerToken(40000, 200)).toBeNull();
    expect(observedCharsPerToken(0, 10000)).toBeNull();
    expect(observedCharsPerToken(40000, undefined)).toBeNull();
  });
});

describe("contextOverflow", () => {
  const record = (patch: Partial<RequestRecord>): RequestRecord => ({
    id: "r1",
    at: 0,
    protocol: "chat",
    url: "/v1/chat/completions",
    body: {},
    rawEvents: [],
    ...patch,
  });

  it("reads the server's numbers out of the message", () => {
    const overflow = contextOverflow(record({ error: "prompt is too long: 41000 tokens > 32768 maximum", errorCode: "context_length_exceeded" }));
    expect(overflow).toEqual({ promptTokens: 41000, contextWindow: 32768 });
  });

  it("matches on the code alone when the prose differs", () => {
    expect(contextOverflow(record({ error: "no room left", errorCode: "context_length_exceeded" }))).toEqual({});
    expect(contextOverflow(record({ error: "model is loading" }))).toBeNull();
    expect(contextOverflow(record({}))).toBeNull();
  });

  it("calibrates the retry budget from the tokens the server counted", () => {
    const overflow = { promptTokens: 40000, contextWindow: 32000 };
    // 140k characters made 40k tokens, so 3.5 chars a token; three quarters of
    // the window is 24k tokens, i.e. 84k characters.
    expect(retryBudget(overflow, 140000, Number.POSITIVE_INFINITY)).toBeCloseTo(84000, 0);
    // Without numbers, halve what the last attempt allowed.
    expect(retryBudget({}, 140000, 100000)).toBe(50000);
  });
});
