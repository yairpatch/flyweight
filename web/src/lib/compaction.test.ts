import { describe, expect, it } from "vitest";
import { budgetChars, compactMessages, compactionNote, contextOverflow, conversationChars, isSilentTurn, observedCharsPerToken, overheadTokens, retryBudget } from "./compaction";
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

  it("counts the prompt and schemas it was actually given", () => {
    // A 4k window with a 1k answer: the default allowance leaves ~1500 tokens
    // of messages, but an agent prompt and six schemas eat most of that, and
    // pretending otherwise is how a request overflows the window.
    const measured = budgetChars(4096, 1024, 3.5, overheadTokens(3300 + 4000));
    expect(measured).toBeLessThan(budgetChars(4096, 1024));
    expect(overheadTokens(3500, 3.5)).toBe(1200);
    // A big window barely notices, which is the point of measuring.
    expect(budgetChars(131072, 4096, 3.5, overheadTokens(7300))).toBeGreaterThan(budgetChars(131072, 4096) * 0.9);
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

describe("compaction and the prefix cache", () => {
  /** What the request looks like as text, which is what the cache matches on. */
  const render = (messages: Message[]) => messages.map((m) => `${m.role}:${m.content}`).join("\n");
  const sharedPrefix = (a: string, b: string) => {
    let index = 0;
    while (index < a.length && index < b.length && a[index] === b[index]) index += 1;
    return index;
  };

  it("keeps a compacted run byte-identical from turn to turn", () => {
    // The failure this prevents, measured on a real 71k-token run: once
    // compaction engaged, every single turn came back 0% cached and took
    // 280 s to first token instead of 80 s. Compaction removes the OLDEST
    // material, which is the FRONT of the prompt, so a boundary that moves
    // even one message per turn invalidates the entire cached prefix every
    // turn. The boundary must therefore be a checkpoint, not a per-turn
    // recomputation -- and the transcript keeps growing, so "trim until it
    // fits" moves it every turn by construction.
    const budget = 60_000;
    let messages = run(12);
    let through: string | undefined;
    let previous = "";
    const shares: number[] = [];
    let checkpoints = 0;
    for (let turn = 0; turn < 10; turn += 1) {
      const callId = `late-${turn}`;
      messages = [
        ...messages,
        message("assistant", "", { toolCalls: [{ id: callId, name: "read_file", arguments: "{}" }] }),
        message("tool", "y".repeat(2000), { toolCallId: callId, toolName: "read_file" }),
      ];
      const outcome = compactMessages(messages, budget, through);
      if (outcome.through !== through) checkpoints += 1;
      through = outcome.through;
      const prompt = render(outcome.messages);
      if (previous) shares.push(sharedPrefix(previous, prompt) / previous.length);
      previous = prompt;
    }
    // One checkpoint, not ten: the boundary moves when the run crosses the
    // ceiling and then stays put.
    expect(checkpoints).toBeLessThanOrEqual(2);
    // Every turn that is not a checkpoint sends the previous prompt entire,
    // plus its new messages. That is what the server can reuse.
    const appended = shares.filter((share) => share === 1).length;
    expect(appended).toBeGreaterThanOrEqual(shares.length - 2);
  });

  it("trims well below the ceiling so the next turns need no trimming", () => {
    const budget = 60_000;
    const messages = run(20);
    const outcome = compactMessages(messages, budget);
    const size = conversationChars(outcome.messages);
    expect(size).toBeLessThanOrEqual(budget * 0.7);
    // Room left over is the point: it is how many turns pass before the next
    // cold prefill.
    expect(budget - size).toBeGreaterThan(10_000);
  });

  it("re-applies a boundary even when the run would now fit without it", () => {
    // Once a result has been stubbed it stays stubbed. Restoring it because
    // there is room again would rewrite the front of the prompt for a saving
    // the model never asked for.
    const messages = run(12);
    const compacted = compactMessages(messages, 20_000);
    expect(compacted.stubbed).toBeGreaterThan(0);
    const roomy = compactMessages(messages, 1_000_000, compacted.through);
    expect(roomy.stubbed).toBe(compacted.stubbed);
    expect(render(roomy.messages)).toBe(render(compacted.messages));
  });

  it("still fits a run whose newest result is bigger than the whole budget", () => {
    // The last resort has to keep working: a boundary cannot help when the
    // message over the line is one the boundary may not touch.
    const messages = [...run(2), message("tool", "z".repeat(80_000), { toolCallId: "big", toolName: "read_file" })];
    const outcome = compactMessages(messages, 20_000);
    expect(conversationChars(outcome.messages)).toBeLessThanOrEqual(20_000);
  });
});

describe("the checkpoint survives the last resort", () => {
  it("keeps the boundary when a recent result had to be clipped", () => {
    // clipRecent rebuilt the outcome and dropped `through` on the way out, so
    // any turn that clipped forgot the checkpoint and the next one re-derived
    // the whole compaction — the churn the boundary exists to prevent,
    // reintroduced by the path that runs when a single result is enormous.
    const messages = [...run(8), message("tool", "z".repeat(60_000), { toolCallId: "big", toolName: "read_file" })];
    const outcome = compactMessages(messages, 20_000);
    expect(conversationChars(outcome.messages)).toBeLessThanOrEqual(20_000);
    expect(outcome.through).toBeTruthy();
  });

  it("does not churn while the budget wobbles turn to turn", () => {
    // The store re-measures characters-per-token after every request, so the
    // budget is never twice the same number. The low-water mark has to absorb
    // that: a boundary that moved on a 3% wobble would cost the cache as
    // surely as one that moved on every turn.
    let messages = run(10);
    let through: string | undefined;
    let checkpoints = 0;
    for (let turn = 0; turn < 12; turn += 1) {
      const callId = `wobble-${turn}`;
      messages = [
        ...messages,
        message("assistant", "", { toolCalls: [{ id: callId, name: "read_file", arguments: "{}" }] }),
        message("tool", `result ${turn} `.repeat(120), { toolCallId: callId, toolName: "read_file" }),
      ];
      const outcome = compactMessages(messages, 60_000 * (1 + 0.03 * Math.sin(turn)), through);
      if (outcome.through !== through) checkpoints += 1;
      through = outcome.through;
    }
    expect(checkpoints).toBeLessThanOrEqual(1);
  });
});

describe("turns that produced nothing", () => {
  it("keeps a failed or stopped turn out of the request", () => {
    // The failure this prevents: a generation that errors or is stopped
    // before its first token leaves an assistant message with the error and
    // no content. Sent back on the next request it is an empty assistant
    // turn, and the model answers the only way it can — "the user's message
    // is empty". They accumulate, so a run that has hit a few failures says
    // it more and more often.
    const failed = message("assistant", "", { error: "vision workspace allocation failed (233 MiB)" });
    const stopped = message("assistant", "", { finishReason: "stopped" });
    expect(isSilentTurn(failed)).toBe(true);
    expect(isSilentTurn(stopped)).toBe(true);
  });

  it("does not mistake a tool-calling turn for an empty one", () => {
    // No text is exactly what a turn that only calls a tool looks like.
    const calling = message("assistant", "", { toolCalls: [{ id: "c", name: "read_file", arguments: "{}" }] });
    expect(isSilentTurn(calling)).toBe(false);
    // Nor a turn that only thought.
    expect(isSilentTurn(message("assistant", "", { reasoning: "considering the options" }))).toBe(false);
    // And never anything the user or a tool said, however short.
    expect(isSilentTurn(message("user", ""))).toBe(false);
    expect(isSilentTurn(message("tool", "", { toolCallId: "c" }))).toBe(false);
  });
});
