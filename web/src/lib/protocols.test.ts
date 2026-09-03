import { describe, expect, it } from "vitest";
import {
  buildAnthropicRequest,
  buildChatRequest,
  buildResponsesRequest,
  parseAnthropicFrame,
  parseChatFrame,
  parseResponsesFrame,
} from "./protocols";
import { DEFAULT_SETTINGS } from "./settings";
import type { Message, ToolDefinition } from "../types";

const user = (content: string, images?: string[]): Message => ({ id: "u", role: "user", content, images, createdAt: 0 });
const tools: ToolDefinition[] = [
  { id: "t", name: "get_weather", description: "Weather", parameters: '{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}', enabled: true },
  { id: "off", name: "disabled_tool", description: "", parameters: "{}", enabled: false },
];

describe("chat completions request", () => {
  it("omits reasoning_effort unless thinking is on and a level is chosen", () => {
    const auto = buildChatRequest({ model: "m", messages: [user("hi")], settings: { ...DEFAULT_SETTINGS, thinking: true, reasoningEffort: "auto" }, tools: [] });
    expect(auto).not.toHaveProperty("reasoning_effort");
    const off = buildChatRequest({ model: "m", messages: [user("hi")], settings: { ...DEFAULT_SETTINGS, thinking: false, reasoningEffort: "high" }, tools: [] });
    expect(off).not.toHaveProperty("reasoning_effort");
    expect(off.enable_thinking).toBe(false);
    const high = buildChatRequest({ model: "m", messages: [user("hi")], settings: { ...DEFAULT_SETTINGS, thinking: true, reasoningEffort: "xhigh" }, tools: [] });
    expect(high.reasoning_effort).toBe("xhigh");
  });

  it("sends images as content parts ahead of the text", () => {
    const body = buildChatRequest({ model: "m", messages: [user("look", ["data:image/png;base64,AAAA"])], settings: DEFAULT_SETTINGS, tools: [] });
    const messages = body.messages as Array<{ role: string; content: unknown }>;
    expect(messages[0].content).toEqual([
      { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
      { type: "text", text: "look" },
    ]);
  });

  it("only sends enabled tools and a well-formed tool_choice", () => {
    const body = buildChatRequest({ model: "m", messages: [user("hi")], settings: { ...DEFAULT_SETTINGS, toolChoice: "get_weather", parallelToolCalls: false }, tools });
    expect(body.tools).toHaveLength(1);
    expect(body.tool_choice).toEqual({ type: "function", function: { name: "get_weather" } });
    expect(body.parallel_tool_calls).toBe(false);
  });

  it("replays tool calls and tool results", () => {
    const messages: Message[] = [
      user("weather?"),
      { id: "a", role: "assistant", content: "", createdAt: 0, toolCalls: [{ id: "call_1", name: "get_weather", arguments: '{"city":"Haifa"}' }] },
      { id: "t", role: "tool", content: "22C", toolCallId: "call_1", createdAt: 0 },
    ];
    const body = buildChatRequest({ model: "m", messages, settings: DEFAULT_SETTINGS, tools });
    const out = body.messages as Array<Record<string, unknown>>;
    expect(out[1].tool_calls).toEqual([{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Haifa"}' } }]);
    expect(out[2]).toEqual({ role: "tool", tool_call_id: "call_1", content: "22C" });
  });
});

describe("chat completions stream", () => {
  it("splits content, reasoning, tool calls, metrics and usage", () => {
    const events = parseChatFrame({
      data: JSON.stringify({
        choices: [{ delta: { content: "Hi", reasoning_content: "hmm", tool_calls: [{ index: 0, id: "c1", function: { name: "f", arguments: "{" } }] }, finish_reason: null }],
        flyweight: { generated_tokens: 3, decode_elapsed_seconds: 0.5 },
      }),
      raw: "",
    });
    expect(events.map((event) => event.type)).toEqual(["metrics", "text", "reasoning", "tool_call_start", "tool_call_delta"]);
    const usage = parseChatFrame({ data: JSON.stringify({ choices: [], usage: { prompt_tokens: 10, completion_tokens: 4, prompt_tokens_details: { cached_tokens: 8 }, completion_tokens_details: { reasoning_tokens: 2 } } }), raw: "" });
    expect(usage[0]).toEqual({ type: "usage", usage: { prompt_tokens: 10, completion_tokens: 4, total_tokens: 14, cached_tokens: 8, reasoning_tokens: 2 } });
    expect(parseChatFrame({ data: "[DONE]", raw: "" })).toEqual([]);
    const withId = parseChatFrame({ data: JSON.stringify({ id: "chatcmpl-1", choices: [{ delta: { content: "x" } }] }), raw: "" });
    expect(withId[0]).toEqual({ type: "id", id: "chatcmpl-1" });
  });
});

describe("anthropic messages", () => {
  it("builds blocks with base64 images, thinking, tool_use and merged tool_result", () => {
    const messages: Message[] = [
      user("see", ["data:image/jpeg;base64,QUJD"]),
      { id: "a", role: "assistant", content: "calling", reasoning: "think", createdAt: 0, toolCalls: [{ id: "tu_1", name: "get_weather", arguments: '{"city":"X"}' }, { id: "tu_2", name: "get_weather", arguments: "{}" }] },
      { id: "r1", role: "tool", content: "a", toolCallId: "tu_1", createdAt: 0 },
      { id: "r2", role: "tool", content: "b", toolCallId: "tu_2", createdAt: 0 },
    ];
    const body = buildAnthropicRequest({ model: "m", messages, settings: { ...DEFAULT_SETTINGS, systemPrompt: "sys", stop: ["END"], thinking: true, reasoningBudget: 512 }, tools });
    expect(body.system).toBe("sys");
    expect(body.stop_sequences).toEqual(["END"]);
    expect(body.thinking).toEqual({ type: "enabled", budget_tokens: 512 });
    expect(body.max_tokens).toBe(DEFAULT_SETTINGS.maxTokens);
    const out = body.messages as Array<{ role: string; content: Array<Record<string, unknown>> }>;
    expect(out[0].content[0]).toEqual({ type: "image", source: { type: "base64", media_type: "image/jpeg", data: "QUJD" } });
    expect(out[1].content.map((block) => block.type)).toEqual(["thinking", "text", "tool_use", "tool_use"]);
    expect(out[1].content[2].input).toEqual({ city: "X" });
    expect(out).toHaveLength(3);
    expect(out[2].content).toHaveLength(2);
    expect((body.tools as unknown[])).toHaveLength(1);
    expect(body.tool_choice).toEqual({ type: "auto" });
  });

  it("parses named events into unified events", () => {
    const frames = [
      { event: "message_start", data: JSON.stringify({ type: "message_start", message: { id: "msg_1" } }) },
      { event: "content_block_start", data: JSON.stringify({ type: "content_block_start", index: 1, content_block: { type: "tool_use", id: "tu", name: "f" } }) },
      { event: "content_block_delta", data: JSON.stringify({ type: "content_block_delta", index: 1, delta: { type: "input_json_delta", partial_json: '{"a"' } }) },
      { event: "content_block_delta", data: JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "thinking_delta", thinking: "t" } }) },
      { event: "content_block_delta", data: JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "x" }, flyweight: { generated_tokens: 1, decode_elapsed_seconds: 0 } }) },
      { event: "message_delta", data: JSON.stringify({ type: "message_delta", delta: { stop_reason: "tool_use" }, usage: { input_tokens: 5, output_tokens: 2, cache_read_input_tokens: 3 } }) },
    ].map((frame) => ({ ...frame, raw: "" }));
    const events = frames.flatMap(parseAnthropicFrame);
    expect(events.map((event) => event.type)).toEqual(["id", "tool_call_start", "tool_call_delta", "reasoning", "metrics", "text", "usage", "finish"]);
    expect(events[events.length - 1]).toEqual({ type: "finish", reason: "tool_calls" });
    expect(events[6]).toEqual({ type: "usage", usage: { prompt_tokens: 8, completion_tokens: 2, total_tokens: 10, cached_tokens: 3 } });
  });
});

describe("responses api", () => {
  it("builds input items and text.format", () => {
    const messages: Message[] = [
      user("q"),
      { id: "a", role: "assistant", content: "", createdAt: 0, toolCalls: [{ id: "call_9", name: "f", arguments: "{}" }] },
      { id: "t", role: "tool", content: "out", toolCallId: "call_9", createdAt: 0 },
    ];
    const body = buildResponsesRequest({ model: "m", messages, settings: { ...DEFAULT_SETTINGS, systemPrompt: "i", responseFormat: "json_object", thinking: true, reasoningEffort: "low" }, tools });
    expect(body.instructions).toBe("i");
    expect(body.reasoning).toEqual({ effort: "low" });
    expect(body.text).toEqual({ format: { type: "json_object" } });
    const input = body.input as Array<Record<string, unknown>>;
    expect(input.map((item) => item.type)).toEqual(["message", "function_call", "function_call_output"]);
    expect(input[1].call_id).toBe("call_9");
  });

  it("parses deltas and completion", () => {
    const events = [
      { event: "response.output_item.added", data: JSON.stringify({ type: "response.output_item.added", output_index: 0, item: { type: "function_call", call_id: "c", name: "f" } }) },
      { event: "response.function_call_arguments.delta", data: JSON.stringify({ type: "response.function_call_arguments.delta", output_index: 0, delta: "{}" }) },
      { event: "response.output_text.delta", data: JSON.stringify({ type: "response.output_text.delta", delta: "hey" }) },
      { event: "response.completed", data: JSON.stringify({ type: "response.completed", response: { output: [{ type: "function_call" }], usage: { input_tokens: 1, output_tokens: 2, input_tokens_details: { cached_tokens: 0 } } } }) },
    ].map((frame) => ({ ...frame, raw: "" })).flatMap(parseResponsesFrame);
    expect(events.map((event) => event.type)).toEqual(["tool_call_start", "tool_call_delta", "text", "usage", "finish"]);
    expect(events[4]).toEqual({ type: "finish", reason: "tool_calls" });
  });
});
