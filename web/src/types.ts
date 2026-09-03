// Shared domain types for the Flyweight web UI.

export type Role = "system" | "user" | "assistant" | "tool";

export interface ToolCall {
  id: string;
  name: string;
  /** Raw JSON text of the arguments as streamed; may be partial while live. */
  arguments: string;
}

export interface Usage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  cached_tokens?: number;
  reasoning_tokens?: number;
}

export interface MessageMetrics {
  /** Tokens generated, as reported by the server's live metrics. */
  tokens: number;
  /** Seconds spent decoding (excludes prefill), from the server when present. */
  decodeSeconds: number;
  /** Wall time from request start to first visible token. */
  ttftSeconds?: number;
  /** Wall time from request start to end of stream. */
  totalSeconds?: number;
  /** Sample points (elapsedMs, tokens) for the throughput sparkline. */
  samples?: Array<[number, number]>;
}

export interface Message {
  id: string;
  role: Role;
  content: string;
  /** Separate reasoning channel (thinking). */
  reasoning?: string;
  reasoningSeconds?: number;
  images?: string[];
  toolCalls?: ToolCall[];
  /** For role=tool: which call this answers. */
  toolCallId?: string;
  toolName?: string;
  createdAt: number;
  finishReason?: string;
  usage?: Usage;
  metrics?: MessageMetrics;
  error?: string;
  /** Which API produced the turn; informational. */
  protocol?: Protocol;
  generating?: boolean;
}

export type Protocol = "chat" | "anthropic" | "responses";

export type ReasoningEffort = "auto" | "low" | "medium" | "high" | "xhigh";

export type ResponseFormatKind = "text" | "json_object" | "json_schema";

export interface ToolDefinition {
  id: string;
  name: string;
  description: string;
  /** JSON Schema text for the parameters object. */
  parameters: string;
  enabled: boolean;
}

export interface GenerationSettings {
  systemPrompt: string;
  maxTokens: number;
  temperature: number;
  topK: number;
  topP: number;
  minP: number;
  seed: number | null;
  repetitionPenalty: number;
  presencePenalty: number;
  frequencyPenalty: number;
  penaltyWindow: number;
  stop: string[];
  thinking: boolean;
  reasoningEffort: ReasoningEffort;
  reasoningBudget: number | null;
  preserveThinking: boolean;
  responseFormat: ResponseFormatKind;
  jsonSchema: string;
  toolChoice: "auto" | "none" | "required" | string;
  parallelToolCalls: boolean;
  chatTemplateKwargs: string;
  protocol: Protocol;
  /** True after the user edits anything; until then defaults track /props. */
  customized: boolean;
}

export interface Preset {
  id: string;
  name: string;
  settings: GenerationSettings;
  tools: ToolDefinition[];
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  pinned?: boolean;
  messages: Message[];
  /** Per-conversation settings override; undefined means use global. */
  model?: string;
}

// ---- Server payloads ----------------------------------------------------

export interface HealthPayload {
  status: string;
  model: string;
  loaded_at?: number;
  busy?: boolean;
  active_requests?: number;
  request_capacity?: number;
  context_window?: number;
  prefix_cache?: Record<string, number>;
  execution?: Record<string, unknown> & {
    backend?: string;
    vision?: Record<string, unknown> | null;
  };
}

export interface PropsPayload {
  model_path?: string;
  model_alias?: string;
  total_slots?: number;
  max_output_tokens?: number;
  context_window?: number;
  chat_template?: string;
  chat_template_source?: string;
  generation_defaults?: Record<string, number>;
  generation_defaults_source?: string;
  capabilities?: string[];
}

export interface ModelInfo {
  id: string;
  created?: number;
  owned_by?: string;
}

export interface SlotInfo {
  id: number;
  is_processing: boolean;
  model: string;
}

// ---- Unified stream events ---------------------------------------------

export type StreamEvent =
  | { type: "id"; id: string }
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "tool_call_start"; index: number; id: string; name: string }
  | { type: "tool_call_delta"; index: number; arguments: string }
  | { type: "metrics"; tokens: number; decodeSeconds: number; phase?: string }
  | { type: "usage"; usage: Usage }
  | { type: "finish"; reason: string }
  | { type: "error"; message: string };

export interface RawLogEntry {
  at: number;
  line: string;
}

export interface RequestRecord {
  id: string;
  at: number;
  protocol: Protocol;
  url: string;
  body: unknown;
  status?: number;
  durationMs?: number;
  rawEvents: RawLogEntry[];
  error?: string;
}
