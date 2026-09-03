// Request builders and stream parsers for the three chat protocols the
// runtime serves: OpenAI chat completions, Anthropic messages, and the
// OpenAI Responses API. Each yields the same unified StreamEvent stream so
// the transcript code never cares which wire format produced a token.
import type {
  GenerationSettings,
  Message,
  Protocol,
  StreamEvent,
  ToolDefinition,
  Usage,
} from "../types";
import type { SseFrame } from "./sse";

export const PROTOCOL_URLS: Record<Protocol, string> = {
  chat: "/v1/chat/completions",
  anthropic: "/v1/messages",
  responses: "/v1/responses",
};

export const PROTOCOL_LABELS: Record<Protocol, string> = {
  chat: "OpenAI chat",
  anthropic: "Anthropic messages",
  responses: "OpenAI responses",
};

export interface RequestContext {
  model: string;
  messages: Message[];
  settings: GenerationSettings;
  tools: ToolDefinition[];
}

// ---- Helpers ---------------------------------------------------------------

export function parseJsonObject(text: string, fallback: Record<string, unknown> = {}): Record<string, unknown> {
  const trimmed = text.trim();
  if (!trimmed) return fallback;
  try {
    const value = JSON.parse(trimmed);
    return value && typeof value === "object" && !Array.isArray(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

function enabledTools(tools: ToolDefinition[]): ToolDefinition[] {
  return tools.filter((tool) => tool.enabled && tool.name.trim());
}

function toolParameters(tool: ToolDefinition): Record<string, unknown> {
  return parseJsonObject(tool.parameters, { type: "object", properties: {} });
}

function dataUrlParts(url: string): { mediaType: string; data: string } | null {
  const match = /^data:([^;,]+)?(;base64)?,(.*)$/s.exec(url);
  if (!match || !match[2]) return null;
  return { mediaType: match[1] || "image/png", data: match[3] };
}

function samplingFields(settings: GenerationSettings): Record<string, unknown> {
  const fields: Record<string, unknown> = {
    temperature: settings.temperature,
    top_k: settings.topK,
    top_p: settings.topP,
    min_p: settings.minP,
    repetition_penalty: settings.repetitionPenalty,
    presence_penalty: settings.presencePenalty,
    frequency_penalty: settings.frequencyPenalty,
    penalty_window: settings.penaltyWindow,
  };
  if (settings.seed !== null && Number.isFinite(settings.seed)) fields.seed = settings.seed;
  return fields;
}

function reasoningEffort(settings: GenerationSettings): string | undefined {
  // "auto" sends nothing so the checkpoint's own level stays in force, and
  // a level only means something when thinking is on.
  if (!settings.thinking || settings.reasoningEffort === "auto") return undefined;
  return settings.reasoningEffort;
}

// ---- OpenAI chat completions -------------------------------------------

function chatMessages(messages: Message[]): unknown[] {
  return messages.map((message) => {
    if (message.role === "tool") {
      return { role: "tool", tool_call_id: message.toolCallId, content: message.content };
    }
    if (message.role === "assistant") {
      const out: Record<string, unknown> = { role: "assistant", content: message.content };
      if (message.toolCalls?.length) {
        out.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: "function",
          function: { name: call.name, arguments: call.arguments },
        }));
      }
      return out;
    }
    if (message.role === "user" && message.images?.length) {
      return {
        role: "user",
        content: [
          ...message.images.map((url) => ({ type: "image_url", image_url: { url } })),
          { type: "text", text: message.content },
        ],
      };
    }
    return { role: message.role, content: message.content };
  });
}

export function buildChatRequest(ctx: RequestContext): Record<string, unknown> {
  const { settings } = ctx;
  const body: Record<string, unknown> = {
    model: ctx.model,
    messages: [
      ...(settings.systemPrompt.trim() ? [{ role: "system", content: settings.systemPrompt }] : []),
      ...chatMessages(ctx.messages),
    ],
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: settings.maxTokens,
    ...samplingFields(settings),
    enable_thinking: settings.thinking,
  };
  if (settings.stop.length) body.stop = settings.stop;
  const effort = reasoningEffort(settings);
  if (effort) body.reasoning_effort = effort;
  if (settings.thinking && settings.reasoningBudget) body.reasoning_budget_tokens = settings.reasoningBudget;
  if (settings.preserveThinking) body.preserve_thinking = true;
  const tools = enabledTools(ctx.tools);
  if (tools.length) {
    body.tools = tools.map((tool) => ({
      type: "function",
      function: { name: tool.name, description: tool.description, parameters: toolParameters(tool) },
    }));
    const choice = settings.toolChoice;
    body.tool_choice =
      choice === "auto" || choice === "none" || choice === "required"
        ? choice
        : { type: "function", function: { name: choice } };
    body.parallel_tool_calls = settings.parallelToolCalls;
  }
  if (settings.responseFormat === "json_object") body.response_format = { type: "json_object" };
  else if (settings.responseFormat === "json_schema") {
    body.response_format = {
      type: "json_schema",
      json_schema: { name: "response", schema: parseJsonObject(settings.jsonSchema, { type: "object" }) },
    };
  }
  const kwargs = parseJsonObject(settings.chatTemplateKwargs);
  if (Object.keys(kwargs).length) body.chat_template_kwargs = kwargs;
  return body;
}

export function parseChatFrame(frame: SseFrame): StreamEvent[] {
  if (frame.data.trim() === "[DONE]") return [];
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(frame.data);
  } catch {
    return [];
  }
  const events: StreamEvent[] = [];
  if (payload.error) {
    const error = payload.error as { message?: string };
    events.push({ type: "error", message: error.message ?? "stream error" });
    return events;
  }
  const metrics = payload.flyweight as { generated_tokens?: number; decode_elapsed_seconds?: number; phase?: string } | undefined;
  if (metrics && typeof metrics.generated_tokens === "number") {
    events.push({
      type: "metrics",
      tokens: metrics.generated_tokens,
      decodeSeconds: metrics.decode_elapsed_seconds ?? 0,
      phase: metrics.phase,
    });
  }
  if (typeof payload.id === "string" && payload.id) events.push({ type: "id", id: payload.id });
  const choice = (payload.choices as Array<Record<string, unknown>> | undefined)?.[0];
  if (choice) {
    const delta = (choice.delta ?? {}) as Record<string, unknown>;
    if (typeof delta.content === "string" && delta.content) events.push({ type: "text", text: delta.content });
    const reasoning = (delta.reasoning_content ?? delta.reasoning) as string | undefined;
    if (typeof reasoning === "string" && reasoning) events.push({ type: "reasoning", text: reasoning });
    const calls = delta.tool_calls as Array<Record<string, unknown>> | undefined;
    if (Array.isArray(calls)) {
      for (const call of calls) {
        const index = (call.index as number) ?? 0;
        const fn = (call.function ?? {}) as { name?: string; arguments?: string };
        if (call.id || fn.name) {
          events.push({ type: "tool_call_start", index, id: String(call.id ?? ""), name: fn.name ?? "" });
        }
        if (fn.arguments) events.push({ type: "tool_call_delta", index, arguments: fn.arguments });
      }
    }
    if (typeof choice.finish_reason === "string") events.push({ type: "finish", reason: choice.finish_reason });
  }
  if (payload.usage && typeof payload.usage === "object") {
    events.push({ type: "usage", usage: normalizeOpenAiUsage(payload.usage as Record<string, unknown>) });
  }
  return events;
}

export function normalizeOpenAiUsage(usage: Record<string, unknown>): Usage {
  const promptDetails = (usage.prompt_tokens_details ?? usage.input_tokens_details ?? {}) as Record<string, number>;
  const completionDetails = (usage.completion_tokens_details ?? usage.output_tokens_details ?? {}) as Record<string, number>;
  const prompt = (usage.prompt_tokens ?? usage.input_tokens) as number | undefined;
  const completion = (usage.completion_tokens ?? usage.output_tokens) as number | undefined;
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: (usage.total_tokens as number | undefined) ?? ((prompt ?? 0) + (completion ?? 0)),
    cached_tokens: promptDetails.cached_tokens,
    reasoning_tokens: completionDetails.reasoning_tokens,
  };
}

// ---- Anthropic messages ------------------------------------------------

function anthropicMessages(messages: Message[]): unknown[] {
  const out: Array<{ role: string; content: unknown[] }> = [];
  for (const message of messages) {
    if (message.role === "system") continue;
    if (message.role === "tool") {
      const block = { type: "tool_result", tool_use_id: message.toolCallId, content: message.content };
      const last = out[out.length - 1];
      if (last && last.role === "user" && (last.content[0] as { type?: string })?.type === "tool_result") {
        last.content.push(block);
      } else {
        out.push({ role: "user", content: [block] });
      }
      continue;
    }
    if (message.role === "assistant") {
      const content: unknown[] = [];
      if (message.reasoning) content.push({ type: "thinking", thinking: message.reasoning, signature: "" });
      if (message.content) content.push({ type: "text", text: message.content });
      for (const call of message.toolCalls ?? []) {
        content.push({ type: "tool_use", id: call.id, name: call.name, input: parseJsonObject(call.arguments) });
      }
      if (content.length) out.push({ role: "assistant", content });
      continue;
    }
    const content: unknown[] = [];
    for (const url of message.images ?? []) {
      const parts = dataUrlParts(url);
      content.push(
        parts
          ? { type: "image", source: { type: "base64", media_type: parts.mediaType, data: parts.data } }
          : { type: "image", source: { type: "url", url } },
      );
    }
    if (message.content || !content.length) content.push({ type: "text", text: message.content });
    out.push({ role: "user", content });
  }
  return out;
}

export function buildAnthropicRequest(ctx: RequestContext): Record<string, unknown> {
  const { settings } = ctx;
  const body: Record<string, unknown> = {
    model: ctx.model,
    max_tokens: settings.maxTokens,
    messages: anthropicMessages(ctx.messages),
    stream: true,
    ...samplingFields(settings),
  };
  if (settings.systemPrompt.trim()) body.system = settings.systemPrompt;
  if (settings.stop.length) body.stop_sequences = settings.stop;
  body.thinking = settings.thinking
    ? { type: "enabled", budget_tokens: settings.reasoningBudget ?? Math.max(1024, Math.floor(settings.maxTokens / 2)) }
    : { type: "disabled" };
  if (settings.preserveThinking) body.preserve_thinking = true;
  const tools = enabledTools(ctx.tools);
  if (tools.length) {
    body.tools = tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      input_schema: toolParameters(tool),
    }));
    const choice = settings.toolChoice;
    const toolChoice: Record<string, unknown> =
      choice === "auto" ? { type: "auto" }
      : choice === "none" ? { type: "none" }
      : choice === "required" ? { type: "any" }
      : { type: "tool", name: choice };
    if (!settings.parallelToolCalls) toolChoice.disable_parallel_tool_use = true;
    body.tool_choice = toolChoice;
  }
  return body;
}

const ANTHROPIC_STOP: Record<string, string> = {
  end_turn: "stop",
  max_tokens: "length",
  tool_use: "tool_calls",
  stop_sequence: "stop",
};

export function parseAnthropicFrame(frame: SseFrame): StreamEvent[] {
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(frame.data);
  } catch {
    return [];
  }
  const events: StreamEvent[] = [];
  const type = (payload.type as string) ?? frame.event ?? "";
  const metrics = payload.flyweight as { generated_tokens?: number; decode_elapsed_seconds?: number; phase?: string } | undefined;
  if (metrics && typeof metrics.generated_tokens === "number") {
    events.push({ type: "metrics", tokens: metrics.generated_tokens, decodeSeconds: metrics.decode_elapsed_seconds ?? 0, phase: metrics.phase });
  }
  switch (type) {
    case "error": {
      const error = payload.error as { message?: string } | undefined;
      events.push({ type: "error", message: error?.message ?? "stream error" });
      break;
    }
    case "message_start": {
      const id = (payload.message as { id?: string } | undefined)?.id;
      if (id) events.push({ type: "id", id });
      break;
    }
    case "content_block_start": {
      const block = payload.content_block as { type?: string; id?: string; name?: string } | undefined;
      if (block?.type === "tool_use") {
        events.push({ type: "tool_call_start", index: (payload.index as number) ?? 0, id: block.id ?? "", name: block.name ?? "" });
      }
      break;
    }
    case "content_block_delta": {
      const delta = payload.delta as { type?: string; text?: string; thinking?: string; partial_json?: string } | undefined;
      if (delta?.type === "text_delta" && delta.text) events.push({ type: "text", text: delta.text });
      else if (delta?.type === "thinking_delta" && delta.thinking) events.push({ type: "reasoning", text: delta.thinking });
      else if (delta?.type === "input_json_delta" && delta.partial_json) {
        events.push({ type: "tool_call_delta", index: (payload.index as number) ?? 0, arguments: delta.partial_json });
      }
      break;
    }
    case "message_delta": {
      const delta = payload.delta as { stop_reason?: string } | undefined;
      const usage = payload.usage as Record<string, number> | undefined;
      if (usage) {
        const cached = usage.cache_read_input_tokens ?? 0;
        events.push({
          type: "usage",
          usage: {
            prompt_tokens: (usage.input_tokens ?? 0) + cached,
            completion_tokens: usage.output_tokens,
            total_tokens: (usage.input_tokens ?? 0) + cached + (usage.output_tokens ?? 0),
            cached_tokens: cached,
          },
        });
      }
      if (delta?.stop_reason) events.push({ type: "finish", reason: ANTHROPIC_STOP[delta.stop_reason] ?? delta.stop_reason });
      break;
    }
    default:
      break;
  }
  return events;
}

// ---- OpenAI Responses --------------------------------------------------

function responseItems(messages: Message[]): unknown[] {
  const items: unknown[] = [];
  for (const message of messages) {
    if (message.role === "system") continue;
    if (message.role === "tool") {
      items.push({ type: "function_call_output", call_id: message.toolCallId, output: message.content });
      continue;
    }
    if (message.role === "assistant") {
      if (message.content) {
        items.push({ type: "message", role: "assistant", content: [{ type: "output_text", text: message.content }] });
      }
      for (const call of message.toolCalls ?? []) {
        items.push({ type: "function_call", call_id: call.id, name: call.name, arguments: call.arguments });
      }
      continue;
    }
    const content: unknown[] = [];
    for (const url of message.images ?? []) content.push({ type: "input_image", image_url: url });
    content.push({ type: "input_text", text: message.content });
    items.push({ type: "message", role: "user", content });
  }
  return items;
}

export function buildResponsesRequest(ctx: RequestContext): Record<string, unknown> {
  const { settings } = ctx;
  const body: Record<string, unknown> = {
    model: ctx.model,
    input: responseItems(ctx.messages),
    stream: true,
    max_output_tokens: settings.maxTokens,
    ...samplingFields(settings),
  };
  if (settings.systemPrompt.trim()) body.instructions = settings.systemPrompt;
  const effort = reasoningEffort(settings);
  if (effort) body.reasoning = { effort };
  const tools = enabledTools(ctx.tools);
  if (tools.length) {
    body.tools = tools.map((tool) => ({
      type: "function",
      name: tool.name,
      description: tool.description,
      parameters: toolParameters(tool),
    }));
    const choice = settings.toolChoice;
    body.tool_choice =
      choice === "auto" || choice === "none" || choice === "required" ? choice : { type: "function", name: choice };
    body.parallel_tool_calls = settings.parallelToolCalls;
  }
  if (settings.responseFormat === "json_object") body.text = { format: { type: "json_object" } };
  else if (settings.responseFormat === "json_schema") {
    body.text = { format: { type: "json_schema", name: "response", schema: parseJsonObject(settings.jsonSchema, { type: "object" }) } };
  }
  return body;
}

export function parseResponsesFrame(frame: SseFrame): StreamEvent[] {
  let payload: Record<string, unknown>;
  try {
    payload = JSON.parse(frame.data);
  } catch {
    return [];
  }
  const events: StreamEvent[] = [];
  const type = (payload.type as string) ?? frame.event ?? "";
  const metrics = payload.flyweight as { generated_tokens?: number; decode_elapsed_seconds?: number; phase?: string } | undefined;
  if (metrics && typeof metrics.generated_tokens === "number") {
    events.push({ type: "metrics", tokens: metrics.generated_tokens, decodeSeconds: metrics.decode_elapsed_seconds ?? 0, phase: metrics.phase });
  }
  switch (type) {
    case "error":
      events.push({ type: "error", message: String((payload.error as { message?: string })?.message ?? payload.message ?? "stream error") });
      break;
    case "response.output_text.delta":
      if (typeof payload.delta === "string" && payload.delta) events.push({ type: "text", text: payload.delta });
      break;
    case "response.output_item.added": {
      const item = payload.item as { type?: string; call_id?: string; id?: string; name?: string } | undefined;
      if (item?.type === "function_call") {
        events.push({
          type: "tool_call_start",
          index: (payload.output_index as number) ?? 0,
          id: item.call_id ?? item.id ?? "",
          name: item.name ?? "",
        });
      }
      break;
    }
    case "response.function_call_arguments.delta":
      if (typeof payload.delta === "string") {
        events.push({ type: "tool_call_delta", index: (payload.output_index as number) ?? 0, arguments: payload.delta });
      }
      break;
    case "response.completed":
    case "response.incomplete": {
      const response = (payload.response ?? {}) as Record<string, unknown>;
      if (response.usage) events.push({ type: "usage", usage: normalizeOpenAiUsage(response.usage as Record<string, unknown>) });
      const incomplete = response.incomplete_details as { reason?: string } | undefined;
      const output = (response.output ?? []) as Array<{ type?: string }>;
      const reason =
        type === "response.incomplete"
          ? incomplete?.reason === "max_output_tokens" ? "length" : incomplete?.reason ?? "incomplete"
          : output.some((item) => item.type === "function_call") ? "tool_calls" : "stop";
      events.push({ type: "finish", reason });
      break;
    }
    default:
      break;
  }
  return events;
}

// ---- Dispatch ----------------------------------------------------------------

export function buildRequest(protocol: Protocol, ctx: RequestContext): Record<string, unknown> {
  switch (protocol) {
    case "anthropic":
      return buildAnthropicRequest(ctx);
    case "responses":
      return buildResponsesRequest(ctx);
    default:
      return buildChatRequest(ctx);
  }
}

export function parseFrame(protocol: Protocol, frame: SseFrame): StreamEvent[] {
  switch (protocol) {
    case "anthropic":
      return parseAnthropicFrame(frame);
    case "responses":
      return parseResponsesFrame(frame);
    default:
      return parseChatFrame(frame);
  }
}

/** Response id from a Responses stream, for the inspector's retrieve/delete. */
export function responseIdFromFrame(frame: SseFrame): string | undefined {
  if (frame.event !== "response.created") return undefined;
  try {
    const payload = JSON.parse(frame.data);
    return payload?.response?.id as string | undefined;
  } catch {
    return undefined;
  }
}
