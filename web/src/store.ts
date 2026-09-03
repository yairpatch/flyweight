// Application state. One zustand store holds conversations, settings, tool
// definitions, runtime telemetry, and the live generation; persistence goes
// to IndexedDB (conversations, presets) and localStorage (settings, tools,
// UI preferences).
import { create } from "zustand";
import { api, ApiError } from "./lib/api";
import { db, migrateLegacyHistory } from "./lib/db";
import { generate } from "./lib/generate";
import { identifier, titleFromPrompt } from "./lib/format";
import { buildRequest } from "./lib/protocols";
import { loadSettings, saveSettings, settingsFromProps, DEFAULT_SETTINGS } from "./lib/settings";
import type {
  Conversation,
  GenerationSettings,
  HealthPayload,
  Message,
  MessageMetrics,
  ModelInfo,
  Preset,
  PropsPayload,
  RequestRecord,
  SlotInfo,
  StreamEvent,
  ToolCall,
  ToolDefinition,
} from "./types";

export type Panel = "settings" | "tools" | "runtime" | "tokenizer" | "playground" | "inspector" | null;
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
  generating: { conversationId: string; messageId: string; controller: AbortController } | null;
  requests: RequestRecord[];
  panel: Panel;
  theme: ThemePreference;
  sidebarOpen: boolean;
  paletteOpen: boolean;
  toasts: Toast[];
  pendingImages: string[];
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
  sendMessage: (text: string, images?: string[]) => Promise<void>;
  stopGeneration: () => void;
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
  setTheme: (theme: ThemePreference) => void;
  toggleSidebar: (open?: boolean) => void;
  setPaletteOpen: (open: boolean) => void;
  toast: (text: string, kind?: Toast["kind"]) => void;
  dismissToast: (id: string) => void;
  setPendingImages: (images: string[]) => void;
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

    const onEvent = (event: StreamEvent) => {
      switch (event.type) {
        case "text":
          if (firstTokenAt === null) firstTokenAt = performance.now();
          if (reasoningStartedAt !== null && reasoningEndedAt === null) reasoningEndedAt = performance.now();
          draft = { ...draft, content: draft.content + event.text };
          break;
        case "reasoning":
          if (firstTokenAt === null) firstTokenAt = performance.now();
          if (reasoningStartedAt === null) reasoningStartedAt = performance.now();
          draft = { ...draft, reasoning: (draft.reasoning ?? "") + event.text };
          break;
        case "tool_call_start": {
          if (firstTokenAt === null) firstTokenAt = performance.now();
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
    pendingImages: [],
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
      const [conversations, presets] = await Promise.all([
        db.conversations.orderBy("updatedAt").reverse().toArray(),
        db.presets.toArray(),
      ]);
      set({ conversations, presets, ready: true, activeId: conversations[0]?.id ?? null });
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
      set((state) => ({ conversations: [conversation, ...state.conversations], activeId: conversation.id, pendingImages: [] }));
      persist(conversation, true);
      return conversation.id;
    },

    selectConversation: (id) => set({ activeId: id, pendingImages: [], ...(isNarrow() ? { sidebarOpen: false } : {}) }),

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

    sendMessage: async (text, images = []) => {
      const trimmed = text.trim();
      if (!trimmed && !images.length) return;
      const conversation = ensureConversation();
      const message: Message = {
        id: identifier("msg"),
        role: "user",
        content: trimmed,
        images: images.length ? images : undefined,
        createdAt: Date.now(),
      };
      updateConversation(conversation.id, (item) =>
        touch({
          ...item,
          title: item.messages.length === 0 && item.title === "New conversation" ? titleFromPrompt(trimmed || "Image") : item.title,
          messages: [...item.messages, message],
        }),
      );
      set({ pendingImages: [], draft: "" });
      await runGeneration(conversation.id);
    },

    stopGeneration: () => {
      const { generating } = get();
      if (!generating) return;
      generating.controller.abort();
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
    setPendingImages: (images) => set({ pendingImages: images }),
    setDraft: (draft) => set({ draft }),
    setPreviewSource: (source) => set({ previewSource: source }),
    clearRequests: () => set({ requests: [] }),
  };
});

export function useActiveConversation(): Conversation | null {
  return useStore((state) => state.conversations.find((conversation) => conversation.id === state.activeId) ?? null);
}
