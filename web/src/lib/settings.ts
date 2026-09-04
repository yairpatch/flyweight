// Generation settings: defaults, normalization, and the mapping from the
// server's /props generation_defaults.
import type { GenerationSettings, PropsPayload, ReasoningEffort } from "../types";
import { clamp } from "./format";

export const SETTINGS_KEY = "flyweight.settings.v2";
export const REASONING_EFFORTS: ReasoningEffort[] = ["auto", "low", "medium", "high", "xhigh"];

export const DEFAULT_SETTINGS: GenerationSettings = {
  systemPrompt: "",
  maxTokens: 4096,
  temperature: 0.8,
  topK: 40,
  topP: 0.95,
  minP: 0.05,
  seed: null,
  repetitionPenalty: 1.0,
  presencePenalty: 0,
  frequencyPenalty: 0,
  penaltyWindow: 64,
  stop: [],
  thinking: true,
  reasoningEffort: "auto",
  reasoningBudget: null,
  preserveThinking: false,
  responseFormat: "text",
  jsonSchema: '{\n  "type": "object",\n  "properties": {\n    "answer": { "type": "string" }\n  },\n  "required": ["answer"]\n}',
  toolChoice: "auto",
  parallelToolCalls: true,
  chatTemplateKwargs: "",
  protocol: "chat",
  customized: false,
};

function num(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function normalizeSettings(input: Partial<GenerationSettings> | null | undefined): GenerationSettings {
  const raw = input ?? {};
  const base = DEFAULT_SETTINGS;
  const effort = REASONING_EFFORTS.includes(raw.reasoningEffort as ReasoningEffort)
    ? (raw.reasoningEffort as ReasoningEffort)
    : base.reasoningEffort;
  const seed = raw.seed === null || raw.seed === undefined || raw.seed === ("" as unknown) ? null : Math.trunc(num(raw.seed, 0));
  return {
    systemPrompt: typeof raw.systemPrompt === "string" ? raw.systemPrompt : base.systemPrompt,
    maxTokens: clamp(Math.trunc(num(raw.maxTokens, base.maxTokens)), 1, 1_000_000),
    temperature: clamp(num(raw.temperature, base.temperature), 0, 5),
    topK: clamp(Math.trunc(num(raw.topK, base.topK)), 0, 10_000),
    topP: clamp(num(raw.topP, base.topP), 0.01, 1),
    minP: clamp(num(raw.minP, base.minP), 0, 1),
    seed,
    repetitionPenalty: clamp(num(raw.repetitionPenalty, base.repetitionPenalty), 1, 2),
    presencePenalty: clamp(num(raw.presencePenalty, base.presencePenalty), -2, 2),
    frequencyPenalty: clamp(num(raw.frequencyPenalty, base.frequencyPenalty), -2, 2),
    penaltyWindow: clamp(Math.trunc(num(raw.penaltyWindow, base.penaltyWindow)), 0, 100_000),
    stop: Array.isArray(raw.stop) ? raw.stop.filter((s): s is string => typeof s === "string" && s.length > 0).slice(0, 16) : base.stop,
    thinking: typeof raw.thinking === "boolean" ? raw.thinking : base.thinking,
    reasoningEffort: effort,
    reasoningBudget:
      raw.reasoningBudget === null || raw.reasoningBudget === undefined ? null : Math.max(0, Math.trunc(num(raw.reasoningBudget, 0))) || null,
    preserveThinking: Boolean(raw.preserveThinking),
    responseFormat: raw.responseFormat === "json_object" || raw.responseFormat === "json_schema" ? raw.responseFormat : "text",
    jsonSchema: typeof raw.jsonSchema === "string" ? raw.jsonSchema : base.jsonSchema,
    toolChoice: typeof raw.toolChoice === "string" && raw.toolChoice ? raw.toolChoice : "auto",
    parallelToolCalls: raw.parallelToolCalls !== false,
    chatTemplateKwargs: typeof raw.chatTemplateKwargs === "string" ? raw.chatTemplateKwargs : "",
    protocol: raw.protocol === "anthropic" || raw.protocol === "responses" ? raw.protocol : "chat",
    customized: Boolean(raw.customized),
  };
}

/** Settings seeded from the server's own defaults for the loaded model. */
export function settingsFromProps(props: PropsPayload | null, current: GenerationSettings): GenerationSettings {
  const defaults = props?.generation_defaults ?? {};
  const contextWindow = props?.context_window ?? Infinity;
  const maxOutput = props?.max_output_tokens ?? Infinity;
  const cap = Math.min(contextWindow, maxOutput);
  const maxTokens = Math.min(Number.isFinite(cap) ? cap : DEFAULT_SETTINGS.maxTokens, num(defaults.max_new_tokens, DEFAULT_SETTINGS.maxTokens));
  return normalizeSettings({
    ...current,
    maxTokens,
    temperature: num(defaults.temperature, DEFAULT_SETTINGS.temperature),
    topK: num(defaults.top_k, DEFAULT_SETTINGS.topK),
    topP: num(defaults.top_p, DEFAULT_SETTINGS.topP),
    minP: num(defaults.min_p, DEFAULT_SETTINGS.minP),
    repetitionPenalty: num(defaults.repetition_penalty, DEFAULT_SETTINGS.repetitionPenalty),
    presencePenalty: num(defaults.presence_penalty, DEFAULT_SETTINGS.presencePenalty),
    frequencyPenalty: num(defaults.frequency_penalty, DEFAULT_SETTINGS.frequencyPenalty),
    penaltyWindow: num(defaults.penalty_window, DEFAULT_SETTINGS.penaltyWindow),
    customized: false,
  });
}

export function loadSettings(): GenerationSettings {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    return normalizeSettings(raw ? JSON.parse(raw) : null);
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export function saveSettings(settings: GenerationSettings): void {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    /* ignore quota errors */
  }
}
