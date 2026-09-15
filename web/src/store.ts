// Application state. One zustand store holds conversations, settings, tool
// definitions, runtime telemetry, and the live generation; persistence goes
// to IndexedDB (conversations, presets) and localStorage (settings, tools,
// UI preferences).
import { create } from "zustand";
import { api, ApiError, openStream } from "./lib/api";
import { db, migrateLegacyHistory } from "./lib/db";
import { readSse } from "./lib/sse";
import { generate } from "./lib/generate";
import { identifier, titleFromPrompt } from "./lib/format";
import { buildRequest } from "./lib/protocols";
import { holdPartialTag, splitThinking } from "./lib/thinking";
import { attachmentImages, forgetSources } from "./lib/attachments";
import { availableEfforts, loadSettings, saveSettings, settingsFromProps, DEFAULT_SETTINGS } from "./lib/settings";
import type {
  Attachment,
  Conversation,
  GenerationSettings,
  HealthPayload,
  Message,
  MessageMetrics,
  ModelInfo,
  Preset,
  PrefillProgress,
  PropsPayload,
  RequestRecord,
  SlotInfo,
  StreamEvent,
  ToolCall,
  ToolDefinition,
  ImageRecord,
} from "./types";

export type Panel = "settings" | "tools" | "runtime" | "tokenizer" | "playground" | "inspector" | null;
/** Which workspace the main area shows. */
export type Mode = "chat" | "images";

export interface ImageSettings {
  aspect: "1:1" | "3:2" | "2:3" | "16:9" | "9:16";
  /** The longer side in pixels; the other follows the aspect, both rounded to 16. */
  size: number;
  steps: number;
  /** A fixed seed, or null for a fresh one per render. */
  seed: number | null;
}

export interface ImageProgress {
  step: number;
  steps: number;
  startedAt: number;
}
export type ThemePreference = "system" | "light" | "dark";
export type RuntimeStatus = "connecting" | "online" | "busy" | "offline" | "locked";

export interface Toast {
  id: string;
  kind: "info" | "error" | "success";
  text: string;
}

export interface HealthSample {
  at: number;
  health: HealthPayload;
}

const TOOLS_KEY = "flyweight.tools.v1";
const THEME_KEY = "flyweight.theme";
const SIDEBAR_KEY = "flyweight.sidebar";
const MODEL_KEY = "flyweight.model";
const MODE_KEY = "flyweight.mode";
const IMAGE_SETTINGS_KEY = "flyweight.images.settings.v1";

const DEFAULT_IMAGE_SETTINGS: ImageSettings = { aspect: "1:1", size: 1024, steps: 8, seed: null };

/** Pixel dimensions for the studio's aspect and size, multiples of 16 within `limit`. */
export function imageDimensions(settings: ImageSettings, limit: number): { width: number; height: number } {
  const ratios: Record<ImageSettings["aspect"], [number, number]> = { "1:1": [1, 1], "3:2": [3, 2], "2:3": [2, 3], "16:9": [16, 9], "9:16": [9, 16] };
  const [rw, rh] = ratios[settings.aspect];
  const long = Math.min(settings.size, limit);
  const round16 = (value: number) => Math.max(256, Math.round(value / 16) * 16);
  if (rw >= rh) return { width: round16(long), height: round16((long * rh) / rw) };
  return { width: round16((long * rw) / rh), height: round16(long) };
}
const HEALTH_HISTORY = 180;
const REQUEST_HISTORY = 25;

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}

function writeJson(key: string, value: unknown): void {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* ignore */
  }
}

function readString(key: string, fallback: string): string {
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

const DEFAULT_TOOLS: ToolDefinition[] = [
  {
    id: "tool-weather",
    name: "get_weather",
    description: "Get the current weather for a city.",
    parameters: JSON.stringify(
      {
        type: "object",
        properties: {
          city: { type: "string", description: "City name" },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] },
        },
        required: ["city"],
      },
      null,
      2,
    ),
    enabled: false,
  },
];

interface StoreState {
  ready: boolean;
  mode: Mode;
  images: ImageRecord[];
  imageSettings: ImageSettings;
  currentImageId: string | null;
  imageProgress: ImageProgress | null;
  imageError: string | null;
  imagePrompt: string;
  conversations: Conversation[];
  activeId: string | null;
  settings: GenerationSettings;
  tools: ToolDefinition[];
  presets: Preset[];
  model: string;
  models: ModelInfo[];
  health: HealthPayload | null;
  healthHistory: HealthSample[];
  props: PropsPayload | null;
  slots: SlotInfo[];
  status: RuntimeStatus;
  statusDetail: string;
  generating: {
    conversationId: string;
    messageId: string;
    controller: AbortController;
    /** The stream id the server reported, for stop_thinking. */
    requestId?: string;
    /** "thinking" while the model is inside its reasoning block. */
    phase?: string;
    /**
     * How far the server has got evaluating the prompt. Set while prefill is
     * running and cleared by the first output, which is what tells the UI to
     * stop showing a bar and start showing an answer.
     */
    prefill?: PrefillProgress;
  } | null;
  requests: RequestRecord[];
  panel: Panel;
  theme: ThemePreference;
  sidebarOpen: boolean;
  paletteOpen: boolean;
  toasts: Toast[];
  pendingAttachments: Attachment[];
  draft: string;
  previewSource: { language: string; code: string } | null;

  // lifecycle
  init: () => Promise<void>;
  pollRuntime: () => Promise<void>;

  // conversations
  newConversation: () => string;
  selectConversation: (id: string | null) => void;
  deleteConversation: (id: string) => Promise<void>;
  renameConversation: (id: string, title: string) => void;
  togglePin: (id: string) => void;
  clearMessages: (id: string) => void;
  importConversation: (data: unknown) => Promise<string | null>;
  active: () => Conversation | null;

  // messages
  sendMessage: (text: string, attachments?: Attachment[]) => Promise<void>;
  stopGeneration: () => void;
  stopThinking: () => Promise<void>;
  regenerate: (messageId: string) => Promise<void>;
  editMessage: (messageId: string, text: string, resend: boolean) => Promise<void>;
  deleteMessage: (messageId: string) => void;
  submitToolResult: (assistantMessageId: string, callId: string, content: string, continueGeneration: boolean) => Promise<void>;
  continueGeneration: () => Promise<void>;

  // settings and tools
  updateSettings: (patch: Partial<GenerationSettings>) => void;
  resetSettings: () => void;
  setModel: (model: string) => void;
  setTools: (tools: ToolDefinition[]) => void;
  savePreset: (name: string) => Promise<void>;
  applyPreset: (id: string) => void;
  deletePreset: (id: string) => Promise<void>;

  // ui
  setPanel: (panel: Panel) => void;
  setMode: (mode: Mode) => void;
  setImagePrompt: (prompt: string) => void;
  updateImageSettings: (patch: Partial<ImageSettings>) => void;
  generateImage: (options?: { seed?: number | null }) => Promise<void>;
  cancelImage: () => void;
  selectImage: (id: string | null) => void;
  deleteImage: (id: string) => Promise<void>;
  setTheme: (theme: ThemePreference) => void;
  toggleSidebar: (open?: boolean) => void;
  setPaletteOpen: (open: boolean) => void;
  toast: (text: string, kind?: Toast["kind"]) => void;
  dismissToast: (id: string) => void;
  setPendingAttachments: (attachments: Attachment[]) => void;
  setDraft: (draft: string) => void;
  setPreviewSource: (source: { language: string; code: string } | null) => void;
  clearRequests: () => void;
}

function applyTheme(theme: ThemePreference): void {
  const root = document.documentElement;
  if (theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
  const dark = theme === "dark" || (theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", dark ? "#0f1115" : "#ffffff");
}

function isNarrow(): boolean {
  return typeof window !== "undefined" && window.innerWidth < 820;
}

function touch(conversation: Conversation): Conversation {
  return { ...conversation, updatedAt: Date.now() };
}

let imageAbort: AbortController | null = null;

export const useStore = create<StoreState>()((set, get) => {
  const persistTimers = new Map<string, number>();

  const persist = (conversation: Conversation, immediate = false) => {
    const write = () => {
      persistTimers.delete(conversation.id);
      const latest = get().conversations.find((item) => item.id === conversation.id);
      if (!latest) return;
      const stored: Conversation = {
        ...latest,
        messages: latest.messages.map((message) => ({ ...message, generating: undefined })),
      };
      void db.conversations.put(stored).catch(() => get().toast("Could not save the conversation", "error"));
    };
    const pending = persistTimers.get(conversation.id);
    if (pending) clearTimeout(pending);
    if (immediate) write();
    else persistTimers.set(conversation.id, window.setTimeout(write, 1500));
  };

  const updateConversation = (id: string, update: (conversation: Conversation) => Conversation, immediate = true) => {
    let next: Conversation | undefined;
    set((state) => ({
      conversations: state.conversations.map((conversation) => {
        if (conversation.id !== id) return conversation;
        next = update(conversation);
        return next;
      }),
    }));
    if (next) persist(next, immediate);
  };

  const updateMessage = (conversationId: string, messageId: string, update: (message: Message) => Message, immediate = false) => {
    updateConversation(
      conversationId,
      (conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) => (message.id === messageId ? update(message) : message)),
      }),
      immediate,
    );
  };

  const ensureConversation = (): Conversation => {
    const current = get().active();
    if (current) return current;
    const id = get().newConversation();
    return get().conversations.find((conversation) => conversation.id === id)!;
  };

  /** Run a generation for the conversation as it stands, appending an assistant turn. */
  const runGeneration = async (conversationId: string) => {
    const state = get();
    if (state.generating) {
      state.toast("A generation is already running", "info");
      return;
    }
    const conversation = state.conversations.find((item) => item.id === conversationId);
    if (!conversation) return;
    const protocol = state.settings.protocol;
    const assistant: Message = {
      id: identifier("msg"),
      role: "assistant",
      content: "",
      createdAt: Date.now(),
      generating: true,
      protocol,
    };
    updateConversation(conversationId, (item) => touch({ ...item, messages: [...item.messages, assistant] }));
    const controller = new AbortController();
    set({ generating: { conversationId, messageId: assistant.id, controller } });

    const body = buildRequest(protocol, {
      model: state.model || state.health?.model || "local",
      messages: conversation.messages,
      settings: state.settings,
      tools: state.tools,
      // Only when the server says it understands the field: an older
      // flyweight, or another OpenAI-compatible server, may reject it.
      prefillProgress: (state.props?.capabilities ?? []).includes("prefill_progress"),
    });

    // Accumulate in a local draft and flush on animation frames so a fast
    // decode does not re-render the transcript per token.
    let draft: Message = { ...assistant };
    let toolCalls: ToolCall[] = [];
    let metrics: MessageMetrics = { tokens: 0, decodeSeconds: 0, samples: [] };
    const startedAt = performance.now();
    let firstTokenAt: number | null = null;
    let reasoningStartedAt: number | null = null;
    let reasoningEndedAt: number | null = null;
    let frame: number | null = null;
    let dirty = false;
    let responseId: string | undefined;

    const flush = () => {
      frame = null;
      if (!dirty) return;
      dirty = false;
      const snapshot: Message = { ...draft, toolCalls: toolCalls.length ? toolCalls.map((call) => ({ ...call })) : undefined, metrics: { ...metrics, samples: metrics.samples?.slice() } };
      updateMessage(conversationId, assistant.id, () => snapshot);
    };
    const scheduleFlush = () => {
      dirty = true;
      if (frame === null) frame = requestAnimationFrame(flush);
    };

    // The server reports the phase in its live metrics; when it does not
    // (a runtime that streams inline <think> text), infer it from the text.
    let serverPhase = false;
    const setPhase = (phase: string) => {
      const live = get().generating;
      if (live && live.messageId === assistant.id && (live.phase ?? "") !== phase) set({ generating: { ...live, phase } });
    };
    const inferPhase = () => {
      if (serverPhase) return;
      if (draft.reasoning !== undefined) setPhase(draft.content ? "" : "thinking");
      else setPhase(splitThinking(holdPartialTag(draft.content)).open ? "thinking" : "");
    };
    /** The prompt is done being read; whatever comes next is the answer. */
    const clearPrefill = () => {
      const live = get().generating;
      if (live?.prefill && live.messageId === assistant.id) set({ generating: { ...live, prefill: undefined } });
    };
    const onEvent = (event: StreamEvent) => {
      switch (event.type) {
        case "id":
          if (get().generating?.messageId === assistant.id && get().generating?.requestId !== event.id) {
            set((current) => (current.generating ? { generating: { ...current.generating, requestId: event.id } } : {}));
          }
          return;
        case "prefill": {
          // Only while the answer is still empty: the server sends a final
          // 100% frame, and a bar that outlived the first token would sit
          // over the answer it was waiting for.
          const { type: _prefill, ...progress } = event;
          const live = get().generating;
          if (live && live.messageId === assistant.id && firstTokenAt === null) {
            set({ generating: { ...live, prefill: progress } });
          }
          return;
        }
        case "text":
          if (firstTokenAt === null) firstTokenAt = performance.now();
          if (reasoningStartedAt !== null && reasoningEndedAt === null) reasoningEndedAt = performance.now();
          clearPrefill();
          draft = { ...draft, content: draft.content + event.text };
          inferPhase();
          break;
        case "reasoning":
          if (firstTokenAt === null) firstTokenAt = performance.now();
          if (reasoningStartedAt === null) reasoningStartedAt = performance.now();
          clearPrefill();
          draft = { ...draft, reasoning: (draft.reasoning ?? "") + event.text };
          inferPhase();
          break;
        case "tool_call_start": {
          if (firstTokenAt === null) firstTokenAt = performance.now();
          clearPrefill();
          const existing = toolCalls[event.index];
          toolCalls[event.index] = existing
            ? { ...existing, id: existing.id || event.id, name: existing.name || event.name }
            : { id: event.id || identifier("call"), name: event.name, arguments: "" };
          break;
        }
        case "tool_call_delta": {
          const existing = toolCalls[event.index] ?? { id: identifier("call"), name: "", arguments: "" };
          toolCalls[event.index] = { ...existing, arguments: existing.arguments + event.arguments };
          break;
        }
        case "metrics": {
          metrics = { ...metrics, tokens: event.tokens, decodeSeconds: event.decodeSeconds };
          if (event.phase !== undefined) {
            serverPhase = true;
            setPhase(event.phase);
          }
          const samples = metrics.samples ?? [];
          samples.push([performance.now() - startedAt, event.tokens]);
          if (samples.length > 600) samples.splice(0, samples.length - 600);
          metrics.samples = samples;
          break;
        }
        case "usage":
          draft = { ...draft, usage: event.usage };
          break;
        case "finish":
          draft = { ...draft, finishReason: event.reason };
          break;
        case "error":
          draft = { ...draft, error: event.message };
          if (event.message.toLowerCase().includes("api key") || /401/.test(event.message)) {
            set({ panel: "settings", status: "locked" });
          }
          break;
      }
      scheduleFlush();
    };

    const record = await generate({
      protocol,
      body,
      signal: controller.signal,
      onEvent,
      onResponseId: (id) => {
        responseId = id;
      },
      onRecord: (rec) => {
        set((current) => {
          const others = current.requests.filter((item) => item.id !== rec.id);
          return { requests: [{ ...rec }, ...others].slice(0, REQUEST_HISTORY) };
        });
      },
    });

    if (frame !== null) cancelAnimationFrame(frame);
    const now = performance.now();
    metrics = {
      ...metrics,
      ttftSeconds: firstTokenAt === null ? undefined : (firstTokenAt - startedAt) / 1000,
      totalSeconds: (now - startedAt) / 1000,
    };
    if (reasoningStartedAt !== null) {
      draft.reasoningSeconds = ((reasoningEndedAt ?? now) - reasoningStartedAt) / 1000;
    }
    const finished: Message = {
      ...draft,
      generating: undefined,
      toolCalls: toolCalls.length ? toolCalls : undefined,
      metrics,
      finishReason: draft.finishReason ?? (controller.signal.aborted ? "stopped" : toolCalls.length ? "tool_calls" : draft.finishReason),
    };
    if (responseId) (finished as Message & { responseId?: string }).responseId = responseId;
    updateMessage(conversationId, assistant.id, () => finished, true);
    set({ generating: null });
    if (record.error && record.status === 401) set({ status: "locked", panel: "settings" });
    else void get().pollRuntime();
  };

  return {
    ready: false,
    mode: (readString(MODE_KEY, "chat") === "images" ? "images" : "chat") as Mode,
    images: [],
    imageSettings: { ...DEFAULT_IMAGE_SETTINGS, ...readJson<Partial<ImageSettings>>(IMAGE_SETTINGS_KEY, {}) },
    currentImageId: null,
    imageProgress: null,
    imageError: null,
    imagePrompt: "",
    conversations: [],
    activeId: null,
    settings: loadSettings(),
    tools: readJson<ToolDefinition[]>(TOOLS_KEY, DEFAULT_TOOLS),
    presets: [],
    model: readString(MODEL_KEY, ""),
    models: [],
    health: null,
    healthHistory: [],
    props: null,
    slots: [],
    status: "connecting",
    statusDetail: "Connecting…",
    generating: null,
    requests: [],
    panel: null,
    theme: (readString(THEME_KEY, "system") as ThemePreference) || "system",
    sidebarOpen: readString(SIDEBAR_KEY, "visible") !== "hidden" && !isNarrow(),
    paletteOpen: false,
    toasts: [],
    pendingAttachments: [],
    draft: "",
    previewSource: null,

    init: async () => {
      applyTheme(get().theme);
      matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(get().theme));
      try {
        const migrated = await migrateLegacyHistory();
        if (migrated) get().toast(`Imported ${migrated} conversation${migrated === 1 ? "" : "s"} from the previous UI`, "success");
      } catch {
        /* ignore */
      }
      const [conversations, presets, images] = await Promise.all([
        db.conversations.orderBy("updatedAt").reverse().toArray(),
        db.presets.toArray(),
        db.images.orderBy("createdAt").reverse().toArray().catch(() => [] as ImageRecord[]),
      ]);
      set({ conversations, presets, images, currentImageId: images[0]?.id ?? null, ready: true, activeId: conversations[0]?.id ?? null });
      await get().pollRuntime();
    },

    pollRuntime: async () => {
      const state = get();
      try {
        const health = await api.health();
        const busy = Boolean(health.busy) || Boolean(state.generating);
        set((current) => ({
          health,
          healthHistory: [...current.healthHistory, { at: Date.now(), health }].slice(-HEALTH_HISTORY),
          status: busy ? "busy" : "online",
          statusDetail: busy ? "Generating" : "Ready",
        }));
        if (!state.props || !state.models.length) {
          const [props, models, slots] = await Promise.all([
            api.props().catch(() => null),
            api.models().catch(() => [] as ModelInfo[]),
            api.slots().catch(() => [] as SlotInfo[]),
          ]);
          const patch: Partial<StoreState> = { props, models, slots };
          const current = get();
          if (!current.model || !models.some((model) => model.id === current.model)) {
            patch.model = models[0]?.id ?? health.model ?? "";
          }
          if (props && !current.settings.customized) {
            patch.settings = settingsFromProps(props, current.settings);
          }
          // A saved effort the loaded checkpoint does not name -- "high",
          // stored against one that reads xhigh -- would leave the picker
          // showing a value it no longer offers. The server clamps such a
          // request rather than failing it, so this is cosmetic, but the honest
          // label for "the checkpoint decides" is the one the UI already has.
          const settings = patch.settings ?? current.settings;
          if (props && !availableEfforts(props).includes(settings.reasoningEffort)) {
            patch.settings = { ...settings, reasoningEffort: "auto" };
          }
          set(patch);
        } else {
          api.slots().then((slots) => set({ slots })).catch(() => undefined);
        }
      } catch (error) {
        if (error instanceof ApiError && error.status === 401) {
          set({ status: "locked", statusDetail: "API key required" });
        } else if (get().generating) {
          // A poll that times out while a stream is running means the
          // runtime is busy, not gone; the stream itself reports failures.
          set({ status: "busy", statusDetail: "Generating" });
        } else if (error instanceof ApiError) {
          set({ status: "offline", statusDetail: error.message });
        } else {
          set({ status: "offline", statusDetail: "Server unreachable" });
        }
      }
    },

    active: () => {
      const { conversations, activeId } = get();
      return conversations.find((conversation) => conversation.id === activeId) ?? null;
    },

    newConversation: () => {
      const conversation: Conversation = {
        id: identifier("conv"),
        title: "New conversation",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        messages: [],
      };
      forgetSources();
      set((state) => ({ conversations: [conversation, ...state.conversations], activeId: conversation.id, pendingAttachments: [] }));
      persist(conversation, true);
      return conversation.id;
    },

    selectConversation: (id) => {
      forgetSources();
      set({ activeId: id, pendingAttachments: [], ...(isNarrow() ? { sidebarOpen: false } : {}) });
    },

    deleteConversation: async (id) => {
      const state = get();
      if (state.generating?.conversationId === id) state.stopGeneration();
      const remaining = state.conversations.filter((conversation) => conversation.id !== id);
      set({ conversations: remaining, activeId: state.activeId === id ? remaining[0]?.id ?? null : state.activeId });
      await db.conversations.delete(id);
    },

    renameConversation: (id, title) => {
      const trimmed = title.trim();
      if (!trimmed) return;
      updateConversation(id, (conversation) => ({ ...conversation, title: trimmed }));
    },

    togglePin: (id) => updateConversation(id, (conversation) => ({ ...conversation, pinned: !conversation.pinned })),

    clearMessages: (id) => {
      const state = get();
      if (state.generating?.conversationId === id) state.stopGeneration();
      updateConversation(id, (conversation) => touch({ ...conversation, messages: [] }));
    },

    importConversation: async (data) => {
      const raw = data as Partial<Conversation> | { messages?: unknown };
      if (!raw || typeof raw !== "object" || !Array.isArray((raw as Conversation).messages)) return null;
      const source = raw as Partial<Conversation>;
      const now = Date.now();
      const conversation: Conversation = {
        id: identifier("conv"),
        title: typeof source.title === "string" && source.title ? source.title : "Imported conversation",
        createdAt: now,
        updatedAt: now,
        messages: (source.messages as Message[])
          .filter((message) => message && typeof message.content === "string" && typeof message.role === "string")
          .map((message, index) => ({
            ...message,
            id: message.id ?? `${now}-${index}`,
            createdAt: message.createdAt ?? now,
            generating: undefined,
          })),
      };
      set((state) => ({ conversations: [conversation, ...state.conversations], activeId: conversation.id }));
      await db.conversations.put(conversation);
      return conversation.id;
    },

    sendMessage: async (text, attachments = []) => {
      const trimmed = text.trim();
      const usable = attachments.filter((attachment) => !attachment.error);
      if (!trimmed && !usable.length) return;
      const conversation = ensureConversation();
      const images = attachmentImages(usable);
      const message: Message = {
        id: identifier("msg"),
        role: "user",
        content: trimmed,
        images: images.length ? images : undefined,
        // The pictures live in `images`; keeping a second copy on every
        // attachment would double what IndexedDB stores per turn.
        attachments: usable.length ? usable.map(({ url: _url, pages: _pages, ...rest }) => rest) : undefined,
        createdAt: Date.now(),
      };
      updateConversation(conversation.id, (item) =>
        touch({
          ...item,
          title: item.messages.length === 0 && item.title === "New conversation" ? titleFromPrompt(trimmed || usable[0]?.name || "Attachment") : item.title,
          messages: [...item.messages, message],
        }),
      );
      forgetSources();
      set({ pendingAttachments: [], draft: "" });
      await runGeneration(conversation.id);
    },

    stopGeneration: () => {
      const { generating } = get();
      if (!generating) return;
      generating.controller.abort();
    },

    stopThinking: async () => {
      const { generating, settings } = get();
      if (!generating?.requestId || settings.protocol === "responses") return;
      try {
        await api.stopThinking(settings.protocol, generating.requestId);
        get().toast("Asked the model to answer now", "info");
      } catch (error) {
        get().toast(error instanceof Error ? error.message : "Could not interrupt thinking", "error");
      }
    },

    regenerate: async (messageId) => {
      const conversation = get().active();
      if (!conversation || get().generating) return;
      const index = conversation.messages.findIndex((message) => message.id === messageId);
      if (index === -1) return;
      // Drop this assistant turn and everything after it, then re-run.
      updateConversation(conversation.id, (item) => touch({ ...item, messages: item.messages.slice(0, index) }));
      await runGeneration(conversation.id);
    },

    editMessage: async (messageId, text, resend) => {
      const conversation = get().active();
      if (!conversation) return;
      const index = conversation.messages.findIndex((message) => message.id === messageId);
      if (index === -1) return;
      if (resend) {
        if (get().generating) return;
        updateConversation(conversation.id, (item) =>
          touch({
            ...item,
            messages: [...item.messages.slice(0, index), { ...item.messages[index], content: text }],
          }),
        );
        await runGeneration(conversation.id);
      } else {
        updateMessage(conversation.id, messageId, (message) => ({ ...message, content: text }), true);
      }
    },

    deleteMessage: (messageId) => {
      const conversation = get().active();
      if (!conversation) return;
      updateConversation(conversation.id, (item) => touch({ ...item, messages: item.messages.filter((message) => message.id !== messageId) }));
    },

    submitToolResult: async (assistantMessageId, callId, content, continueGeneration) => {
      const conversation = get().active();
      if (!conversation) return;
      const assistant = conversation.messages.find((message) => message.id === assistantMessageId);
      const call = assistant?.toolCalls?.find((item) => item.id === callId);
      const result: Message = {
        id: identifier("msg"),
        role: "tool",
        content,
        toolCallId: callId,
        toolName: call?.name,
        createdAt: Date.now(),
      };
      updateConversation(conversation.id, (item) => {
        const index = item.messages.findIndex((message) => message.id === assistantMessageId);
        // Insert after the assistant turn and any existing tool results for it.
        let insertAt = index + 1;
        while (insertAt < item.messages.length && item.messages[insertAt].role === "tool") insertAt += 1;
        const messages = [...item.messages];
        messages.splice(insertAt, 0, result);
        return touch({ ...item, messages });
      });
      if (continueGeneration) await runGeneration(conversation.id);
    },

    continueGeneration: async () => {
      const conversation = get().active();
      if (!conversation || !conversation.messages.length) return;
      await runGeneration(conversation.id);
    },

    updateSettings: (patch) => {
      const next = { ...get().settings, ...patch, customized: true };
      saveSettings(next);
      set({ settings: next });
    },

    resetSettings: () => {
      const { props, settings } = get();
      const next = settingsFromProps(props, { ...DEFAULT_SETTINGS, protocol: settings.protocol });
      saveSettings(next);
      set({ settings: next });
    },

    setModel: (model) => {
      writeJson(MODEL_KEY, model);
      try {
        localStorage.setItem(MODEL_KEY, model);
      } catch {
        /* ignore */
      }
      set({ model });
    },

    setTools: (tools) => {
      writeJson(TOOLS_KEY, tools);
      set({ tools });
    },

    savePreset: async (name) => {
      const { settings, tools } = get();
      const preset: Preset = { id: identifier("preset"), name: name.trim() || "Preset", settings: { ...settings }, tools: tools.map((tool) => ({ ...tool })) };
      await db.presets.put(preset);
      set((state) => ({ presets: [...state.presets, preset] }));
    },

    applyPreset: (id) => {
      const preset = get().presets.find((item) => item.id === id);
      if (!preset) return;
      const settings = { ...preset.settings, customized: true };
      saveSettings(settings);
      writeJson(TOOLS_KEY, preset.tools);
      set({ settings, tools: preset.tools.map((tool) => ({ ...tool })) });
      get().toast(`Applied preset “${preset.name}”`, "success");
    },

    deletePreset: async (id) => {
      await db.presets.delete(id);
      set((state) => ({ presets: state.presets.filter((preset) => preset.id !== id) }));
    },

    setPanel: (panel) => set((state) => ({ panel: state.panel === panel ? null : panel })),

    setMode: (mode) => {
      try {
        localStorage.setItem(MODE_KEY, mode);
      } catch {
        /* ignore */
      }
      set({ mode });
    },
    setImagePrompt: (imagePrompt) => set({ imagePrompt }),
    updateImageSettings: (patch) => {
      const imageSettings = { ...get().imageSettings, ...patch };
      try {
        localStorage.setItem(IMAGE_SETTINGS_KEY, JSON.stringify(imageSettings));
      } catch {
        /* ignore */
      }
      set({ imageSettings });
    },
    selectImage: (currentImageId) => set({ currentImageId }),
    deleteImage: async (id) => {
      await db.images.delete(id);
      set((state) => {
        const images = state.images.filter((image) => image.id !== id);
        const currentImageId = state.currentImageId === id ? (images[0]?.id ?? null) : state.currentImageId;
        return { images, currentImageId };
      });
    },
    cancelImage: () => {
      imageAbort?.abort();
    },
    generateImage: async (options = {}) => {
      const state = get();
      const prompt = state.imagePrompt.trim();
      if (!prompt || state.imageProgress) return;
      const info = state.health?.execution?.images as { max_size?: string } | null | undefined;
      const limit = parseInt(String(info?.max_size ?? "1024x1024").split("x")[0] ?? "1024", 10) || 1024;
      const { width, height } = imageDimensions(state.imageSettings, limit);
      const seed = options.seed === undefined ? state.imageSettings.seed : options.seed;
      const body: Record<string, unknown> = { prompt, size: `${width}x${height}`, steps: state.imageSettings.steps, stream: true };
      if (seed !== null) body.seed = seed;
      const abort = new AbortController();
      imageAbort = abort;
      set({ imageProgress: { step: 0, steps: state.imageSettings.steps, startedAt: Date.now() }, imageError: null });
      try {
        const response = await openStream("/v1/images/generations", body, abort.signal);
        if (!response.body) throw new Error("empty response");
        for await (const frame of readSse(response.body)) {
          if (frame.data === "[DONE]") break;
          let event: Record<string, unknown>;
          try {
            event = JSON.parse(frame.data);
          } catch {
            continue;
          }
          if (event.error && typeof event.error === "object") {
            throw new Error(String((event.error as { message?: string }).message ?? "image generation failed"));
          }
          if (event.type === "progress") {
            set((current) => (current.imageProgress ? { imageProgress: { ...current.imageProgress, step: Number(event.step), steps: Number(event.steps) } } : {}));
          } else if (event.type === "image") {
            const bytes = Uint8Array.from(atob(String(event.b64_json)), (character) => character.charCodeAt(0));
            const record: ImageRecord = {
              id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
              createdAt: Date.now(),
              prompt,
              seed: Number(event.seed),
              width,
              height,
              steps: state.imageSettings.steps,
              seconds: Number(event.seconds),
              blob: new Blob([bytes], { type: "image/png" }),
            };
            await db.images.put(record).catch(() => get().toast("Could not save the picture", "error"));
            set((current) => ({ images: [record, ...current.images], currentImageId: record.id }));
          }
        }
      } catch (failure) {
        if (!abort.signal.aborted) {
          const message = failure instanceof ApiError ? failure.message : String(failure);
          set({ imageError: message });
          get().toast(message, "error");
        }
      } finally {
        if (imageAbort === abort) imageAbort = null;
        set({ imageProgress: null });
      }
    },

    setTheme: (theme) => {
      try {
        localStorage.setItem(THEME_KEY, theme);
      } catch {
        /* ignore */
      }
      applyTheme(theme);
      set({ theme });
    },

    toggleSidebar: (open) => {
      const next = open ?? !get().sidebarOpen;
      try {
        localStorage.setItem(SIDEBAR_KEY, next ? "visible" : "hidden");
      } catch {
        /* ignore */
      }
      set({ sidebarOpen: next });
    },

    setPaletteOpen: (open) => set({ paletteOpen: open }),

    toast: (text, kind = "info") => {
      const toast: Toast = { id: identifier("toast"), kind, text };
      set((state) => ({ toasts: [...state.toasts, toast].slice(-4) }));
      window.setTimeout(() => get().dismissToast(toast.id), kind === "error" ? 6000 : 3200);
    },

    dismissToast: (id) => set((state) => ({ toasts: state.toasts.filter((toast) => toast.id !== id) })),
    setPendingAttachments: (attachments) => {
      forgetSources(attachments);
      set({ pendingAttachments: attachments });
    },
    setDraft: (draft) => set({ draft }),
    setPreviewSource: (source) => set({ previewSource: source }),
    clearRequests: () => set({ requests: [] }),
  };
});

export function useActiveConversation(): Conversation | null {
  return useStore((state) => state.conversations.find((conversation) => conversation.id === state.activeId) ?? null);
}
