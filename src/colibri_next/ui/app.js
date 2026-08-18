const STORAGE_KEY = "colibri.chat.v1";
const SETTINGS_KEY = "colibri.settings.v1";
const THEME_KEY = "colibri.theme";
const SIDEBAR_KEY = "colibri.sidebar";
const API_KEY = "colibri.api-key";

const DEFAULT_SETTINGS = Object.freeze({
  systemPrompt: "",
  maxTokens: 64,
  temperature: 0.7,
  topP: 0.95,
  topK: 20,
  thinking: false,
  // "auto" sends nothing and leaves the checkpoint's own level, which is the
  // only honest default: a template that does not grade its reasoning ignores
  // the field, and one that does has a level it was tuned for.
  reasoningEffort: "auto",
});

const REASONING_EFFORTS = ["auto", "low", "medium", "high"];
// The composer chip cycles the whole reasoning state, because thinking and its
// level are one decision: "off" is off, and any level implies thinking. Keeping
// them as two controls in a dialog meant the one people actually want to change
// per prompt was three clicks and a modal away, and the level could be set to
// something the toggle then made inert.
const REASONING_STATES = ["off", "auto", "low", "medium", "high"];
const REASONING_LABELS = Object.freeze({
  off: "Thinking off",
  auto: "Thinking on",
  low: "Thinking · low",
  medium: "Thinking · medium",
  high: "Thinking · high",
});

const elements = {
  body: document.body,
  sidebar: document.querySelector("#sidebar"),
  sidebarOpen: document.querySelector("#sidebar-open"),
  sidebarClose: document.querySelector("#sidebar-close"),
  sidebarScrim: document.querySelector("#sidebar-scrim"),
  brand: document.querySelector(".brand-row .brand"),
  searchBox: document.querySelector("#search-box"),
  newChat: document.querySelector("#new-chat"),
  conversationSearch: document.querySelector("#conversation-search"),
  conversationList: document.querySelector("#conversation-list"),
  statusDot: document.querySelector("#status-dot"),
  modelStatus: document.querySelector("#model-status"),
  modelSelectorWrap: document.querySelector("#model-selector-wrap"),
  modelSelector: document.querySelector("#model-selector"),
  runtimePill: document.querySelector("#runtime-pill"),
  deviceLabel: document.querySelector("#device-label"),
  chatScroll: document.querySelector("#chat-scroll"),
  welcome: document.querySelector("#welcome"),
  messages: document.querySelector("#messages"),
  scrollBottom: document.querySelector("#scroll-bottom"),
  promptInput: document.querySelector("#prompt-input"),
  composerShell: document.querySelector("#composer-shell"),
  sendButton: document.querySelector("#send-button"),
  thinkingChip: document.querySelector("#thinking-chip"),
  tokenChip: document.querySelector("#token-chip"),
  quickSettings: document.querySelector("#quick-settings"),
  clearChat: document.querySelector("#clear-chat"),
  exportChat: document.querySelector("#export-chat"),
  themeToggle: document.querySelector("#theme-toggle"),
  openSettings: document.querySelector("#open-settings"),
  importChat: document.querySelector("#import-chat"),
  settingsDialog: document.querySelector("#settings-dialog"),
  settingsForm: document.querySelector("#settings-form"),
  apiKey: document.querySelector("#api-key"),
  systemPrompt: document.querySelector("#system-prompt"),
  maxTokens: document.querySelector("#max-tokens"),
  temperature: document.querySelector("#temperature"),
  topP: document.querySelector("#top-p"),
  topK: document.querySelector("#top-k"),
  thinkingSetting: document.querySelector("#thinking-setting"),
  reasoningEffort: document.querySelector("#reasoning-effort"),
  resetSettings: document.querySelector("#reset-settings"),
  saveSettings: document.querySelector("#save-settings"),
  toastRegion: document.querySelector("#toast-region"),
};

const state = {
  conversations: loadConversations(),
  activeId: null,
  settings: loadSettings(),
  settingsCustomized: hasPersistedSettings(),
  apiKey: readSession(API_KEY),
  theme: "system",
  sidebarHidden: false,
  model: null,
  models: [],
  health: null,
  properties: null,
  maxOutputTokens: 64,
  controller: null,
  generating: false,
  stickToBottom: true,
};

function identifier() {
  if (globalThis.crypto?.randomUUID) {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function loadConversations() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((conversation) => (
      conversation
      && typeof conversation.id === "string"
      && Array.isArray(conversation.messages)
    ));
  } catch {
    return [];
  }
}

function loadSettings() {
  try {
    const parsed = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    return normalizeSettings({ ...DEFAULT_SETTINGS, ...parsed });
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

function hasPersistedSettings() {
  try {
    return localStorage.getItem(SETTINGS_KEY) !== null;
  } catch {
    return false;
  }
}

function normalizeSettings(value) {
  return {
    systemPrompt: typeof value.systemPrompt === "string" ? value.systemPrompt : "",
    maxTokens: clampInteger(
      value.maxTokens,
      1,
      Number.MAX_SAFE_INTEGER,
      DEFAULT_SETTINGS.maxTokens,
    ),
    temperature: clampNumber(value.temperature, 0, 2, DEFAULT_SETTINGS.temperature),
    topP: clampNumber(value.topP, 0.01, 1, DEFAULT_SETTINGS.topP),
    topK: clampInteger(value.topK, 0, 200, DEFAULT_SETTINGS.topK),
    thinking: Boolean(value.thinking),
    reasoningEffort: REASONING_EFFORTS.includes(value.reasoningEffort)
      ? value.reasoningEffort
      : DEFAULT_SETTINGS.reasoningEffort,
  };
}

function clampNumber(value, minimum, maximum, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.min(maximum, Math.max(minimum, number)) : fallback;
}

function clampInteger(value, minimum, maximum, fallback) {
  return Math.round(clampNumber(value, minimum, maximum, fallback));
}

function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

function readSession(key) {
  try {
    return sessionStorage.getItem(key) || "";
  } catch {
    return "";
  }
}

function saveSession(key, value) {
  try {
    if (value) {
      sessionStorage.setItem(key, value);
    } else {
      sessionStorage.removeItem(key);
    }
  } catch {
    // Session storage can be unavailable in hardened browser profiles.
  }
}

// localStorage holds a few megabytes at most, and both the write here and the
// parse at startup are synchronous on the main thread. History is capped by
// size as well as by count so a few long chats cannot stall the next launch.
const HISTORY_LIMIT = 100;
const HISTORY_BYTE_BUDGET = 2_000_000;

function historyPayload(limit) {
  const ordered = [...state.conversations]
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .slice(0, limit);
  const retained = [];
  let bytes = 2;
  for (const conversation of ordered) {
    // A persisted message is never mid-generation. The flag drives the caret
    // and suppresses the message actions, so a snapshot taken while an answer
    // was still drawing would reload as an answer that never finishes.
    const serialized = JSON.stringify(
      conversation,
      (key, value) => (key === "generating" ? undefined : value),
    );
    if (retained.length && bytes + serialized.length + 1 > HISTORY_BYTE_BUDGET) {
      break;
    }
    retained.push(serialized);
    bytes += serialized.length + 1;
  }
  return `[${retained.join(",")}]`;
}

function persistConversations() {
  try {
    localStorage.setItem(STORAGE_KEY, historyPayload(HISTORY_LIMIT));
  } catch {
    // The quota is shared with everything else this origin stored, so retry
    // with a much shorter history before giving up on the write entirely.
    try {
      localStorage.setItem(STORAGE_KEY, historyPayload(5));
    } catch {
      toast("Conversation history could not be saved.", "error");
    }
  }
}

function persistSettings() {
  state.settingsCustomized = true;
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
}

function settingsFromModelDefaults() {
  const defaults = state.properties?.generation_defaults || {};
  return normalizeSettings({
    ...DEFAULT_SETTINGS,
    maxTokens: defaults.max_new_tokens ?? DEFAULT_SETTINGS.maxTokens,
    temperature: defaults.temperature ?? DEFAULT_SETTINGS.temperature,
    topP: defaults.top_p ?? DEFAULT_SETTINGS.topP,
    topK: defaults.top_k ?? DEFAULT_SETTINGS.topK,
  });
}

// The entry point for every "new chat" affordance: the logo, the sidebar
// button and Ctrl+K. An unused conversation is not worth a second one, so
// asking for a blank chat while already sitting in one does nothing but return
// the cursor to the composer. Without this, every click of the logo left
// another identical "New conversation" in the history.
function startNewConversation() {
  if (state.generating) {
    stopGeneration();
  }
  const active = activeConversation();
  if (active && !active.messages.length) {
    closeSidebar();
    elements.promptInput.focus();
    return active;
  }
  const blank = state.conversations.find((conversation) => !conversation.messages.length);
  if (blank) {
    selectConversation(blank.id);
    elements.promptInput.focus();
    return blank;
  }
  return createConversation();
}

function createConversation({ activate = true } = {}) {
  const now = Date.now();
  const conversation = {
    id: identifier(),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    messages: [],
  };
  // Any blank chat left behind elsewhere in the history is dead weight, and
  // keeping them is how the list fills up with placeholders.
  state.conversations = state.conversations.filter(
    (existing) => existing.messages.length || existing.id === state.activeId,
  );
  state.conversations.unshift(conversation);
  if (activate) {
    state.activeId = conversation.id;
  }
  persistConversations();
  render();
  closeSidebar();
  elements.promptInput.focus();
  return conversation;
}

function activeConversation() {
  return state.conversations.find((conversation) => conversation.id === state.activeId) || null;
}

function ensureActiveConversation() {
  let conversation = activeConversation();
  if (!conversation) {
    conversation = createConversation();
  }
  return conversation;
}

function selectConversation(id) {
  if (state.generating && id !== state.activeId) {
    stopGeneration();
  }
  if (!state.conversations.some((conversation) => conversation.id === id)) {
    return;
  }
  state.activeId = id;
  render();
  closeSidebar();
  requestAnimationFrame(() => scrollToBottom());
}

function deleteConversation(id) {
  if (id === state.activeId && state.generating) {
    stopGeneration();
  }
  state.conversations = state.conversations.filter((conversation) => conversation.id !== id);
  if (state.activeId === id) {
    state.activeId = state.conversations[0]?.id || null;
  }
  if (!state.activeId) {
    createConversation();
    return;
  }
  persistConversations();
  render();
}

function startRenameConversation(conversation, nameElement) {
  const input = document.createElement("input");
  input.type = "text";
  input.className = "rename-input";
  input.dir = "auto";
  input.value = conversation.title;
  nameElement.replaceWith(input);
  input.focus();
  input.select();
  const commit = () => {
    const val = input.value.trim();
    if (val && val !== conversation.title) {
      conversation.title = val;
      persistConversations();
    }
    renderConversationList();
  };
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); input.blur(); }
    if (e.key === "Escape") { input.value = conversation.title; input.blur(); }
  });
}

function importConversation() {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = ".json";
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const data = JSON.parse(reader.result);
        if (!data || !Array.isArray(data.messages)) {
          toast("Invalid conversation file.", "error");
          return;
        }
        const conversation = {
          id: identifier(),
          title: data.title || "Imported conversation",
          createdAt: data.createdAt || Date.now(),
          updatedAt: data.updatedAt || Date.now(),
          messages: data.messages.filter((m) => m && typeof m.role === "string"),
        };
        state.conversations.unshift(conversation);
        state.activeId = conversation.id;
        persistConversations();
        render();
        closeSidebar();
        toast("Conversation imported.");
      } catch {
        toast("Failed to parse conversation file.", "error");
      }
    };
    reader.readAsText(file);
  });
  input.click();
}

function clearConversation() {
  const conversation = activeConversation();
  if (!conversation || !conversation.messages.length) {
    return;
  }
  if (!confirm("Clear all messages in this conversation?")) {
    return;
  }
  stopGeneration();
  conversation.messages = [];
  conversation.title = "New conversation";
  conversation.updatedAt = Date.now();
  persistConversations();
  render();
}

function render() {
  renderConversationList();
  renderConversation();
  renderSettingsSummary();
  updateSendButton();
}

function renderConversationList() {
  const query = elements.conversationSearch.value.trim().toLowerCase();
  const conversations = [...state.conversations]
    .sort((left, right) => right.updatedAt - left.updatedAt)
    .filter((conversation) => conversation.title.toLowerCase().includes(query));
  elements.conversationList.replaceChildren();

  if (!conversations.length) {
    const empty = document.createElement("div");
    empty.className = "empty-history";
    empty.textContent = query ? "No conversations match your search." : "Your conversations will appear here.";
    elements.conversationList.append(empty);
    return;
  }

  for (const conversation of conversations) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `conversation-item${conversation.id === state.activeId ? " active" : ""}`;
    button.setAttribute("aria-current", conversation.id === state.activeId ? "page" : "false");

    let clickTimer = null;
    button.addEventListener("click", () => {
      if (clickTimer) return;
      clickTimer = setTimeout(() => {
        clickTimer = null;
        selectConversation(conversation.id);
      }, 200);
    });

    const icon = svgIcon("message");
    const name = document.createElement("span");
    name.className = "conversation-name";
    name.dir = "auto";
    name.textContent = conversation.title;
    button.addEventListener("dblclick", (event) => {
      event.preventDefault();
      if (clickTimer) {
        clearTimeout(clickTimer);
        clickTimer = null;
      }
      startRenameConversation(conversation, name);
    });
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "conversation-delete";
    remove.setAttribute("aria-label", `Delete ${conversation.title}`);
    remove.append(svgIcon("trash"));
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      deleteConversation(conversation.id);
    });
    button.append(icon, name, remove);
    elements.conversationList.append(button);
  }
}

function renderConversation() {
  const conversation = activeConversation();
  const messages = conversation?.messages || [];
  elements.welcome.hidden = messages.length > 0;
  elements.messages.replaceChildren();
  for (const message of messages) {
    elements.messages.append(renderMessage(message));
  }
}

// The transcript reads as a document rather than a table: the answer is plain
// prose at full width and the prompt is a single bubble above it. Authorship is
// carried by shape and placement, so no avatar or byline is needed on any turn.
function renderMessage(message) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;
  article.dataset.messageId = message.id;

  const body = document.createElement("div");
  body.className = "message-body";

  const content = document.createElement("div");
  content.className = "message-content";
  renderMessageContent(content, message);

  body.append(content, renderMessageToolbar(message));
  article.append(body);
  return article;
}

function messageAction(icon, label, onClick) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "msg-action";
  button.setAttribute("aria-label", label);
  button.dataset.tooltip = label;
  button.append(svgIcon(icon));
  button.addEventListener("click", onClick);
  return button;
}

function renderMessageToolbar(message) {
  const toolbar = document.createElement("div");
  toolbar.className = "message-toolbar";

  const actions = document.createElement("div");
  actions.className = "message-actions";
  if (message.content && !message.generating) {
    actions.append(messageAction("copy", "Copy message", () => copyText(answerText(message))));
  }
  if (message.role === "user" && !message.generating) {
    actions.append(messageAction("edit", "Edit message", () => startEditMessage(message)));
  }
  if (message.role === "assistant" && !message.generating && !message.error) {
    actions.append(messageAction("refresh", "Regenerate", () => regenerateFrom(message)));
  }

  const metrics = document.createElement("span");
  metrics.className = "message-metrics";
  metrics.textContent = message.metrics || "";

  const time = document.createElement("time");
  time.className = "message-time";
  time.dateTime = new Date(message.createdAt).toISOString();
  time.textContent = formatTime(message.createdAt);

  toolbar.append(actions, metrics, time);
  return toolbar;
}

// What the reader actually sees, which is what "copy" should hand back — the
// reasoning trace is collapsed in the transcript and does not belong on the
// clipboard.
function answerText(message) {
  return splitThinking(message, message.content || "")
    .answer
    .replace(/<think>[\s\S]*?<\/think>/g, "")
    .trim();
}

function renderMessageContent(content, message) {
  content.replaceChildren();
  const parts = splitThinking(message, message.content || "");
  applyMessageDirection(content, message, parts.answer);
  if (parts.reasoning !== null && parts.reasoning.trim()) {
    const panel = renderThinkingPanel();
    updateThinkingPanel(panel, {
      reasoning: parts.reasoning,
      live: Boolean(message.generating) && !parts.settled,
      seconds: message.thinkingSeconds,
    });
    content.append(panel);
  }
  renderRichText(content, parts.answer);
  if (message.generating) {
    const cursor = document.createElement("span");
    cursor.className = "stream-cursor";
    cursor.setAttribute("aria-label", "Generating");
    cursor.hidden = parts.reasoning !== null && !parts.settled;
    content.append(cursor);
  }
  if (message.error) {
    const error = document.createElement("div");
    error.className = "error-card";
    const errText = document.createElement("span");
    errText.textContent = message.error;
    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "retry-button";
    retryBtn.textContent = "Retry";
    retryBtn.addEventListener("click", () => retryFromMessage(message));
    error.append(errText, retryBtn);
    content.append(error);
  }
  for (const toolCall of message.toolCalls || []) {
    content.append(renderToolCall(toolCall));
  }
}

function startEditMessage(message) {
  const article = elements.messages.querySelector(`[data-message-id="${message.id}"]`);
  const contentEl = article?.querySelector(".message-content");
  if (!contentEl) return;
  contentEl.replaceChildren();
  const textarea = document.createElement("textarea");
  textarea.className = "edit-textarea";
  textarea.dir = "auto";
  textarea.value = message.content;
  textarea.rows = Math.min(Math.max(message.content.split("\n").length + 1, 2), 12);
  const btnRow = document.createElement("div");
  btnRow.className = "edit-actions";
  const save = document.createElement("button");
  save.type = "button";
  save.className = "primary-button";
  save.textContent = "Save";
  const cancel = document.createElement("button");
  cancel.type = "button";
  cancel.className = "text-button";
  cancel.textContent = "Cancel";
  save.addEventListener("click", () => {
    const newContent = textarea.value.trim();
    if (newContent && newContent !== message.content) {
      message.content = newContent;
      // The latched direction described the previous wording.
      message.dir = null;
      persistConversations();
    }
    renderConversation();
  });
  cancel.addEventListener("click", () => renderConversation());
  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      save.click();
    }
    if (e.key === "Escape") cancel.click();
  });
  btnRow.append(cancel, save);
  contentEl.append(textarea, btnRow);
  textarea.focus();
  textarea.setSelectionRange(textarea.value.length, textarea.value.length);
}

function regenerateFrom(assistantMessage) {
  const conversation = activeConversation();
  if (!conversation || state.generating) return;
  const msgIndex = conversation.messages.findIndex((m) => m.id === assistantMessage.id);
  if (msgIndex < 0) return;
  conversation.messages = conversation.messages.slice(0, msgIndex);
  conversation.updatedAt = Date.now();
  persistConversations();
  const lastUser = [...conversation.messages].reverse().find((m) => m.role === "user");
  if (lastUser) {
    sendMessage(lastUser.content);
  }
}

function retryFromMessage(errorMessage) {
  const conversation = activeConversation();
  if (!conversation || state.generating) return;
  const errIndex = conversation.messages.findIndex((m) => m.id === errorMessage.id);
  if (errIndex < 0) return;
  conversation.messages = conversation.messages.slice(0, errIndex);
  conversation.updatedAt = Date.now();
  persistConversations();
  const lastUser = [...conversation.messages].reverse().find((m) => m.role === "user");
  if (lastUser) {
    sendMessage(lastUser.content);
  }
}

// Streaming repaints are coalesced into a single animation frame, and only the
// text after the last completed block is rebuilt. Re-rendering the whole
// message on every token is quadratic and freezes the tab on long answers.
const stream = {
  messageId: null,
  content: null,
  committedEnd: 0,
  thinkingPanel: null,
  tailNodes: [],
  toolHost: null,
};

// Tokens land in bursts: a prefill pause, then a clump, then another pause.
// Painting each burst the instant it arrives is what makes the transcript hard
// to read. The reveal head advances on its own clock instead, draining whatever
// has been received over a short window, so text appears at a steady pace no
// matter how uneven decode is. The buffer is bounded by DRAIN_SECONDS, so the
// reveal never falls meaningfully behind the model.
const REVEAL_DRAIN_SECONDS = 0.22;
const REVEAL_MIN_CPS = 26;
const REVEAL_MAX_CPS = 1200;

const smooth = {
  message: null,
  revealed: 0,
  painted: -1,
  // Reasoning is not revealed on the same clock as the answer -- it arrives on
  // its own field and is shown as it lands -- so its own painted length is
  // what says whether there is anything new to draw.
  paintedReasoning: -1,
  finishing: false,
  instant: false,
  onDone: null,
  last: 0,
  speed: REVEAL_MIN_CPS,
  frame: null,
};

function beginSmoothStream(message) {
  flushSmoothStream();
  smooth.message = message;
  smooth.revealed = 0;
  smooth.painted = -1;
  smooth.paintedReasoning = -1;
  smooth.finishing = false;
  smooth.instant = false;
  smooth.onDone = null;
  smooth.last = performance.now();
  smooth.speed = REVEAL_MIN_CPS;
  smooth.frame = requestAnimationFrame(smoothTick);
}

function smoothTick(now) {
  smooth.frame = null;
  const message = smooth.message;
  if (!message) {
    return;
  }
  const target = (message.content || "").length;
  // Real time not spent on a frame is still time the reveal owes the reader, so
  // the whole delta is carried. Clamping it (this once capped at 0.25 s) threw
  // the excess away, which left a throttled tab losing ground on every long
  // frame with no way to catch up. `revealed` is bounded by `target` below, so
  // a large delta can only ever close the gap, never overrun it.
  const elapsed = Math.max(0, (now - smooth.last) / 1000);
  smooth.last = now;
  if (smooth.instant) {
    smooth.revealed = target;
  } else if (smooth.revealed < target) {
    // Chasing the backlog is right while tokens are still arriving. Once the
    // request is over the backlog only shrinks, and a speed proportional to it
    // decays exponentially through the last characters -- a slow-motion tail
    // exactly where the reader is looking. Finishing therefore holds the pace
    // the stream had reached and runs out at it, only ever revising upwards so
    // a large unrevealed remainder still drains promptly.
    const chase = clampNumber(
      (target - smooth.revealed) / REVEAL_DRAIN_SECONDS,
      REVEAL_MIN_CPS,
      REVEAL_MAX_CPS,
      REVEAL_MIN_CPS,
    );
    smooth.speed = smooth.finishing ? Math.max(smooth.speed, chase) : chase;
    smooth.revealed = Math.min(target, smooth.revealed + smooth.speed * elapsed);
  }

  const head = Math.floor(smooth.revealed);
  // The answer's reveal is not the only thing that moves. A reasoning model
  // spends its first seconds emitting nothing but `reasoning_content`, so
  // `content` -- and with it `head` -- stays at zero, and a repaint gated on
  // `head` alone left the thinking panel frozen at whatever had arrived by the
  // first frame. It filled in only once the answer began, which is precisely
  // when it stops being interesting.
  const reasoningLength = (message.reasoning || "").length;
  if (head !== smooth.painted || reasoningLength !== smooth.paintedReasoning) {
    smooth.painted = head;
    smooth.paintedReasoning = reasoningLength;
    // Asked before the text lands, never after: this frame's own lines would
    // otherwise register as the reader having scrolled away from the bottom.
    const following = isFollowingStream();
    if (!updateStreamingMessage(message, revealedText(message, head))) {
      // The message left the DOM (conversation switched, transcript rebuilt).
      // Nothing left to animate against, so settle the request immediately.
      finishSmoothStream();
      return;
    }
    if (following) {
      scrollToBottom();
    }
  }

  if (smooth.finishing && smooth.revealed >= target) {
    finishSmoothStream();
    return;
  }
  smooth.frame = requestAnimationFrame(smoothTick);
}

// A slice can land in the middle of a `<think>` tag, and half a tag rendered as
// prose flickers into view for a frame. Only a trailing run that could still
// become one of those tags is held back — a lone `<` in prose ("a < b") is not.
const THINKING_TAGS = ["<think>", "</think>"];

function revealedText(message, head) {
  const text = (message.content || "").slice(0, head);
  const open = text.lastIndexOf("<");
  if (open === -1) {
    return text;
  }
  const tail = text.slice(open);
  return THINKING_TAGS.some((tag) => tag.startsWith(tail)) ? text.slice(0, open) : text;
}

function endSmoothStream(done, { instant = false } = {}) {
  if (!smooth.message) {
    done?.();
    return;
  }
  smooth.finishing = true;
  smooth.instant = smooth.instant || instant;
  smooth.onDone = done;
}

// requestAnimationFrame is suspended while the page is hidden, but the SSE
// reader is not, so the reveal head can be an entire answer behind by the time
// the reader comes back. Replaying that as an animation shows text arriving
// long after the runtime finished. Returning to the page therefore snaps the
// head to everything received and settles a request that ended while away --
// the frame this cancels would otherwise be the one that ran `onDone`.
function catchUpSmoothStream() {
  if (!smooth.message) {
    return;
  }
  smooth.revealed = (smooth.message.content || "").length;
  smooth.last = performance.now();
  if (smooth.finishing) {
    // `onDone` re-renders the transcript from the message itself, so the full
    // text lands without painting it here first.
    finishSmoothStream();
  }
}

// Reveals whatever is left at once and settles the request. Used when the
// transcript is about to be rebuilt under a stream that is still draining.
function flushSmoothStream() {
  const done = smooth.onDone;
  const pending = Boolean(smooth.message);
  cancelSmoothStream();
  if (pending) {
    done?.();
  }
}

function finishSmoothStream() {
  const done = smooth.onDone;
  cancelSmoothStream();
  done?.();
}

function cancelSmoothStream() {
  if (smooth.frame !== null) {
    cancelAnimationFrame(smooth.frame);
  }
  smooth.frame = null;
  smooth.message = null;
  smooth.onDone = null;
  smooth.revealed = 0;
  smooth.painted = -1;
  smooth.finishing = false;
  smooth.instant = false;
  smooth.speed = REVEAL_MIN_CPS;
}

function resetStreamState(messageId = null, content = null) {
  stream.messageId = messageId;
  stream.content = content;
  stream.committedEnd = 0;
  stream.thinkingPanel = null;
  stream.tailNodes = [];
  stream.toolHost = null;
}

// The furthest point in the answer that can no longer change: the last blank
// line that sits outside a code fence. Everything before it stays in the DOM
// untouched for the rest of the generation. Only the text after the previous
// boundary is scanned, which keeps a repaint proportional to the tail rather
// than to the whole answer. No fence is ever open at `from`, because a commit
// only ever happens outside one.
function stableBoundary(body, from) {
  let open = false;
  let boundary = from;
  let at = from;
  while (at < body.length) {
    const fence = body.indexOf("```", at);
    const blank = body.indexOf("\n\n", at);
    if (blank !== -1 && (fence === -1 || blank < fence)) {
      let end = blank;
      while (end < body.length && body[end] === "\n") {
        end += 1;
      }
      if (!open) {
        boundary = end;
      }
      at = end;
    } else if (fence !== -1) {
      open = !open;
      at = fence + 3;
    } else {
      break;
    }
  }
  return boundary;
}

function updateStreamingMessage(message, visible) {
  const article = elements.messages.querySelector(`[data-message-id="${message.id}"]`);
  const content = article?.querySelector(".message-content");
  if (!content) {
    return false;
  }
  updateMessageMetrics(article, message);
  const cursor = content.querySelector(".stream-cursor");
  if (!cursor || !message.generating) {
    resetStreamState();
    renderMessageContent(content, message);
    return true;
  }
  // Anchored to the element as well as the id: a full re-render mid-stream
  // replaces the node, and the committed offsets would no longer describe it.
  if (stream.messageId !== message.id || stream.content !== content) {
    resetStreamState(message.id, content);
    content.replaceChildren(cursor);
  }

  const text = visible ?? message.content ?? "";
  const parts = splitThinking(message, text);
  applyMessageDirection(content, message, parts.answer);

  if (parts.reasoning !== null) {
    if (!stream.thinkingPanel) {
      stream.thinkingPanel = renderThinkingPanel();
      content.insertBefore(stream.thinkingPanel, content.firstChild);
    }
    updateThinkingPanel(stream.thinkingPanel, {
      reasoning: parts.reasoning,
      live: !parts.settled,
      seconds: message.thinkingSeconds,
    });
  }
  // While the model is still reasoning there is no answer for the cursor to
  // trail, and a dot parked under the "Thinking" line just reads as debris.
  // The shimmer on that line is the activity indicator until prose starts.
  cursor.hidden = parts.reasoning !== null && !parts.settled;
  const body = parts.answer;

  for (const node of stream.tailNodes) {
    node.remove();
  }
  stream.tailNodes = [];

  // Frozen text is rendered once with the full markdown renderer, so finished
  // code blocks, lists and headings look the same while streaming as they do
  // once the answer lands. Only the unfinished tail uses the cheap renderer.
  const boundary = stableBoundary(body, stream.committedEnd);
  if (boundary > stream.committedEnd) {
    const settled = document.createDocumentFragment();
    renderRichTextSegment(settled, body.slice(stream.committedEnd, boundary));
    content.insertBefore(settled, cursor);
    stream.committedEnd = boundary;
  }

  const tail = body.slice(stream.committedEnd);
  if (tail) {
    const pending = document.createDocumentFragment();
    appendStreamMarkdown(pending, tail);
    stream.tailNodes = [...pending.childNodes];
    content.insertBefore(pending, cursor);
  }

  if (message.toolCalls?.length) {
    if (!stream.toolHost) {
      stream.toolHost = document.createElement("div");
      stream.toolHost.className = "stream-tools";
      content.append(stream.toolHost);
    }
    stream.toolHost.replaceChildren(...message.toolCalls.map(renderToolCall));
  }
  return true;
}

/* =============================================================================
   Reasoning

   The runtime puts the opening `<think>` in the *prompt* (see
   `format_messages` in v2_server.py), so a thinking response arrives already
   inside the block: raw reasoning first, then `</think>`, then the answer. A
   reader looking for the reply should never have to wade through that, so it
   lives behind one collapsed line \u2014 a live "Thinking" while it runs, and how
   long it took once it is done.
   ========================================================================== */

function splitThinking(message, text) {
  // Streamed separately by the server; stored conversations may still carry
  // the tags inline, which the rest of this function handles.
  if (message && typeof message.reasoning === "string" && message.reasoning) {
    return { reasoning: message.reasoning, answer: text, settled: Boolean(text) };
  }
  const open = text.indexOf("<think>");
  const close = text.indexOf("</think>");
  if (open !== -1) {
    // An explicit opening tag, as older stored conversations have.
    if (close > open) {
      return {
        reasoning: text.slice(open + 7, close),
        answer: text.slice(0, open) + text.slice(close + 8),
        settled: true,
      };
    }
    return { reasoning: text.slice(open + 7), answer: text.slice(0, open), settled: false };
  }
  if (close !== -1) {
    return { reasoning: text.slice(0, close), answer: text.slice(close + 8), settled: true };
  }
  // No closing tag yet. Whether what has arrived so far is reasoning or the
  // answer is decided by the flag the request was made with, not by guesswork.
  if (message.thinking) {
    return { reasoning: text, answer: "", settled: false };
  }
  return { reasoning: null, answer: text, settled: true };
}

function renderThinkingPanel() {
  const panel = document.createElement("details");
  panel.className = "thinking";

  const summary = document.createElement("summary");
  summary.className = "thinking-summary";
  const label = document.createElement("span");
  label.className = "thinking-label";
  const chevron = svgIcon("chevron");
  chevron.classList.add("thinking-chevron");
  summary.append(label, chevron);

  const body = document.createElement("div");
  body.className = "thinking-body";

  panel.append(summary, body);
  return panel;
}

function updateThinkingPanel(panel, { reasoning, live, seconds }) {
  panel.classList.toggle("live", Boolean(live));
  const label = panel.querySelector(".thinking-label");
  const next = thinkingLabel(live, seconds);
  if (label.textContent !== next) {
    label.textContent = next;
  }
  const body = panel.querySelector(".thinking-body");
  const trimmed = reasoning.trim();
  if (body.textContent !== trimmed) {
    body.textContent = trimmed;
    applyDirection(body, trimmed);
    // Expanded mid-thought, the newest line is the one worth seeing.
    if (live && panel.open) {
      body.scrollTop = body.scrollHeight;
    }
  }
}

function thinkingLabel(live, seconds) {
  if (live) {
    return "Thinking";
  }
  if (!Number.isFinite(seconds) || seconds < 1) {
    return "Thought for a moment";
  }
  const whole = Math.round(seconds);
  return `Thought for ${whole} second${whole === 1 ? "" : "s"}`;
}

function appendStreamMarkdown(container, text) {
  if (!text) return;
  // Only an odd number of fences leaves a code block open. Treating the last
  // fence as an opener regardless put every paragraph that followed a finished
  // code block inside a "streaming" code box until the answer completed.
  const fences = [...text.matchAll(/```([^\n`]*)/g)];
  let incompleteCode = null;
  if (fences.length % 2 === 1) {
    const lastOpen = fences[fences.length - 1];
    incompleteCode = {
      lang: lastOpen[1].trim(),
      code: text.slice(lastOpen.index + lastOpen[0].length),
      start: lastOpen.index,
    };
  }
  const textPart = incompleteCode ? text.slice(0, incompleteCode.start) : text;
  if (textPart) {
    const paragraphs = textPart.split(/\n{2,}/);
    for (const paragraphText of paragraphs) {
      if (!paragraphText) continue;
      const p = document.createElement("p");
      applyDirection(p, paragraphText);
      const lines = paragraphText.split("\n");
      lines.forEach((line, idx) => {
        appendStreamInline(p, line);
        if (idx + 1 < lines.length) p.append(document.createElement("br"));
      });
      container.append(p);
    }
  }
  if (incompleteCode) {
    const pre = document.createElement("pre");
    // Source is left-to-right whatever language surrounds it. Without this an
    // RTL answer mirrors its own code samples.
    pre.dir = "ltr";
    const header = document.createElement("div");
    header.className = "code-header";
    const label = document.createElement("span");
    label.textContent = incompleteCode.lang || "code";
    const badge = document.createElement("span");
    badge.className = "streaming-badge";
    badge.textContent = "streaming";
    header.append(label, badge);
    const code = document.createElement("code");
    if (incompleteCode.lang) {
      code.className = `language-${incompleteCode.lang.toLowerCase()}`;
    }
    code.textContent = incompleteCode.code;
    pre.append(header, code);
    container.append(pre);
  }
}

function appendStreamInline(parent, text) {
  if (!text) return;
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*|\[[^\]]+\]\([^)]+\))/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      parent.append(document.createTextNode(text.slice(cursor, match.index)));
    }
    const fragment = match[0];
    if (fragment.startsWith("`")) {
      const code = document.createElement("code");
      code.textContent = fragment.slice(1, -1);
      parent.append(code);
    } else if (fragment.startsWith("**")) {
      const strong = document.createElement("strong");
      strong.textContent = fragment.slice(2, -2);
      parent.append(strong);
    } else if (fragment.startsWith("*")) {
      const em = document.createElement("em");
      em.textContent = fragment.slice(1, -1);
      parent.append(em);
    } else if (fragment.startsWith("[")) {
      const linkMatch = fragment.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        const a = document.createElement("a");
        a.textContent = linkMatch[1];
        a.href = linkMatch[2];
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        parent.append(a);
      } else {
        parent.append(document.createTextNode(fragment));
      }
    }
    cursor = pattern.lastIndex;
  }
  if (cursor < text.length) {
    parent.append(document.createTextNode(text.slice(cursor)));
  }
}

function updateMessageMetrics(article, message) {
  const metrics = article.querySelector(".message-toolbar > .message-metrics");
  if (metrics) {
    metrics.textContent = message.metrics || "";
  }
}

/* =============================================================================
   Bidirectional text

   `dir="auto"` alone is not enough for chat: it looks only at the first strong
   character, so one English product name at the head of a Hebrew or Arabic
   answer lays the whole paragraph out left-to-right, with the punctuation
   stranded on the wrong side. Direction is decided by weight instead — a
   paragraph is RTL as soon as a meaningful share of its letters are, which is
   how mixed technical prose actually reads.
   ========================================================================== */

// Hebrew, Arabic, Syriac, Thaana, N'Ko, Samaritan and the Arabic presentation
// forms. Vowel points and diacritics count too: they only ever attach to RTL
// letters, so they reinforce the reading rather than skew it.
const RTL_LETTERS = /[\u0591-\u07FF\u0860-\u08FF\uFB1D-\uFDFD\uFE70-\uFEFC]/g;
// Latin, Greek, Cyrillic, Armenian and the scripts that sit above Hebrew.
const LTR_LETTERS = /[A-Za-z\u00C0-\u02B8\u0370-\u058F\u0900-\u1FFF\u2C00-\uD7FF]/g;

// The share of strong letters that must be RTL for a block to flip. Low enough
// that a Hebrew sentence quoting an English identifier still reads right-to-
// left, high enough that an English sentence quoting one Hebrew word does not.
const RTL_THRESHOLD = 0.34;

// Enough letters to trust the ratio. Below this the reading is provisional and
// is not cached, so a paragraph that opens with a stray Latin character is not
// locked into the wrong direction for the rest of the answer.
const DIRECTION_CONFIDENCE = 8;

function countMatches(text, pattern) {
  pattern.lastIndex = 0;
  return text.match(pattern)?.length || 0;
}

// Long answers are sampled rather than scanned end to end: the opening of a
// block settles its direction, and this runs on every streamed frame.
const DIRECTION_SAMPLE = 400;

function detectDirection(text) {
  if (!text) {
    return null;
  }
  const sample = text.length > DIRECTION_SAMPLE ? text.slice(0, DIRECTION_SAMPLE) : text;
  const rtl = countMatches(sample, RTL_LETTERS);
  const ltr = countMatches(sample, LTR_LETTERS);
  const total = rtl + ltr;
  if (!total) {
    return null;
  }
  return { dir: rtl / total >= RTL_THRESHOLD ? "rtl" : "ltr", strength: total };
}

// Only ever sets `dir` when the text actually says something about direction.
// Blocks that are all digits or punctuation stay unset and inherit the
// message's direction, which keeps a numbered list from breaking apart.
function applyDirection(element, text) {
  const reading = detectDirection(text);
  if (reading) {
    element.dir = reading.dir;
  }
  return reading?.dir || null;
}

// The message-level direction is latched once it is confident. Re-deciding it
// on every streamed frame would swing the whole block from side to side as the
// first sentence arrives.
function applyMessageDirection(content, message, text) {
  if (message.dir) {
    content.dir = message.dir;
    return message.dir;
  }
  const reading = detectDirection(text);
  if (!reading) {
    return null;
  }
  content.dir = reading.dir;
  if (reading.strength >= DIRECTION_CONFIDENCE) {
    message.dir = reading.dir;
  }
  return reading.dir;
}

// The leading reasoning block is peeled off by splitThinking before this runs.
// Any further pairs are a model repeating itself mid-answer; they collapse the
// same way rather than spilling raw reasoning into the prose.
function renderRichText(container, text) {
  if (!text) {
    return;
  }
  const thinkingPattern = /<think>([\s\S]*?)<\/think>/g;
  let tCursor = 0;
  let tMatch;
  while ((tMatch = thinkingPattern.exec(text)) !== null) {
    renderRichTextSegment(container, text.slice(tCursor, tMatch.index));
    const panel = renderThinkingPanel();
    updateThinkingPanel(panel, { reasoning: tMatch[1], live: false, seconds: null });
    container.append(panel);
    tCursor = thinkingPattern.lastIndex;
  }
  renderRichTextSegment(container, text.slice(tCursor));
}

function renderRichTextSegment(container, text) {
  if (!text) return;
  const fencePattern = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let cursor = 0;
  let match;
  while ((match = fencePattern.exec(text)) !== null) {
    appendMarkdownBlock(container, text.slice(cursor, match.index));
    appendCodeBlock(container, match[1].trim(), match[2].replace(/\n$/, ""));
    cursor = fencePattern.lastIndex;
  }
  const remainder = text.slice(cursor);
  const open = remainder.match(/```([^\n`]*)\n([\s\S]*)$/);
  if (open) {
    appendMarkdownBlock(container, remainder.slice(0, open.index));
    appendCodeBlock(container, open[1].trim(), open[2].replace(/\n$/, ""));
    return;
  }
  appendMarkdownBlock(container, remainder);
}

function appendMarkdownBlock(container, text) {
  if (!text) return;
  const lines = text.split("\n");
  const startsHandledBlock = (line) => (
    /^#{1,6}\s/.test(line)
    || /^>\s/.test(line)
    || /^[-*]\s/.test(line)
    || /^\d+\.\s/.test(line)
    || /^[-*_]{3,}\s*$/.test(line)
  );
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^#{1,6}\s/.test(line)) {
      const level = line.match(/^(#{1,6})\s/)[1].length;
      const el = document.createElement(`h${level}`);
      const heading = line.replace(/^#{1,6}\s+/, "");
      applyDirection(el, heading);
      appendInline(el, heading);
      container.append(el);
      i++;
    } else if (/^>\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^>\s/.test(lines[i])) {
        items.push(lines[i].replace(/^>\s+/, ""));
        i++;
      }
      const bq = document.createElement("blockquote");
      applyDirection(bq, items.join("\n"));
      appendMarkdownBlock(bq, items.join("\n"));
      container.append(bq);
    } else if (/^[-*]\s/.test(line)) {
      const ul = document.createElement("ul");
      const items = [];
      while (i < lines.length && /^[-*]\s/.test(lines[i])) {
        const item = lines[i].replace(/^[-*]\s+/, "");
        const li = document.createElement("li");
        appendDirectedInline(li, item);
        ul.append(li);
        items.push(item);
        i++;
      }
      // The marker sits on whichever side the list itself reads from, so the
      // direction is taken from the list as a whole rather than per item.
      applyDirection(ul, items.join("\n"));
      container.append(ul);
    } else if (/^\d+\.\s/.test(line)) {
      const ol = document.createElement("ol");
      const items = [];
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        const item = lines[i].replace(/^\d+\.\s+/, "");
        const li = document.createElement("li");
        appendDirectedInline(li, item);
        ol.append(li);
        items.push(item);
        i++;
      }
      applyDirection(ol, items.join("\n"));
      container.append(ol);
    } else if (/^[-*_]{3,}\s*$/.test(line)) {
      container.append(document.createElement("hr"));
      i++;
    } else if (line.trim() === "") {
      i++;
    } else {
      const paraLines = [];
      while (
        i < lines.length
        && lines[i].trim() !== ""
        && !startsHandledBlock(lines[i])
      ) {
        paraLines.push(lines[i]);
        i++;
      }
      // Defensive forward progress: a newly added block detector must never
      // strand the parser on a line without a matching handler. That turns one
      // generated Markdown token into an infinite animation-frame loop and
      // freezes the entire tab.
      if (!paraLines.length && i < lines.length) {
        paraLines.push(lines[i]);
        i++;
      }
      if (paraLines.length) {
        const p = document.createElement("p");
        applyDirection(p, paraLines.join("\n"));
        paraLines.forEach((pl, idx) => {
          appendInline(p, pl);
          if (idx + 1 < paraLines.length) p.append(document.createElement("br"));
        });
        container.append(p);
      }
    }
  }
}

function appendDirectedInline(element, text) {
  applyDirection(element, text);
  appendInline(element, text);
}

function appendInline(parent, text) {
  let remaining = text;
  const pattern = /(\*\*(.+?)\*\*|\*(.+?)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(remaining)) !== null) {
    if (match.index > cursor) {
      parent.append(document.createTextNode(remaining.slice(cursor, match.index)));
    }
    if (match[2]) {
      const strong = document.createElement("strong");
      strong.textContent = match[2];
      parent.append(strong);
    } else if (match[3]) {
      const em = document.createElement("em");
      em.textContent = match[3];
      parent.append(em);
    } else if (match[4]) {
      const code = document.createElement("code");
      code.textContent = match[4];
      parent.append(code);
    } else if (match[5] && match[6]) {
      const a = document.createElement("a");
      a.textContent = match[5];
      a.href = match[6];
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      parent.append(a);
    }
    cursor = pattern.lastIndex;
  }
  parent.append(document.createTextNode(remaining.slice(cursor)));
}

const RUNNABLE_LANGUAGES = new Set(["html", "htm", "svg", "js", "javascript"]);

function appendCodeBlock(container, language, code) {
  const pre = document.createElement("pre");
  pre.dir = "ltr";
  const header = document.createElement("div");
  header.className = "code-header";
  const label = document.createElement("span");
  label.textContent = language || "code";
  const actions = document.createElement("span");
  actions.className = "code-actions";
  if (RUNNABLE_LANGUAGES.has(language.toLowerCase())) {
    const run = document.createElement("button");
    run.type = "button";
    run.className = "run-code";
    run.textContent = "Run";
    run.addEventListener("click", () => togglePreview(pre, run, language, code));
    actions.append(run);
  }
  const download = document.createElement("button");
  download.type = "button";
  download.className = "download-code";
  download.textContent = "Download";
  download.addEventListener("click", () => downloadCode(code, language));
  actions.append(download);
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy-code";
  copy.textContent = "Copy";
  copy.addEventListener("click", () => copyText(code));
  actions.append(copy);
  header.append(label, actions);
  const codeElement = document.createElement("code");
  if (language) {
    codeElement.className = `language-${language.toLowerCase()}`;
  }
  highlightCode(codeElement, code, language);
  pre.append(header, codeElement);
  container.append(pre);
}

const CODE_EXTENSIONS = {
  javascript: "js", js: "js", typescript: "ts", ts: "ts",
  python: "py", py: "py", html: "html", htm: "html",
  css: "css", rust: "rs", go: "go", c: "c", cpp: "cpp",
  java: "java", ruby: "rb", bash: "sh", shell: "sh",
  json: "json", xml: "xml", svg: "svg", sql: "sql",
  markdown: "md", md: "md", yaml: "yaml", yml: "yaml",
};

function downloadCode(code, language) {
  const ext = CODE_EXTENSIONS[(language || "").toLowerCase()] || "txt";
  const blob = new Blob([code], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `code.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

const HL_RULES = {
  js: [
    { pattern: /(\/\/.*$)/gm, cls: "hl-comment" },
    { pattern: /(\/\*[\s\S]*?\*\/)/g, cls: "hl-comment" },
    { pattern: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)/g, cls: "hl-string" },
    { pattern: /\b(const|let|var|function|return|if|else|for|while|do|switch|case|break|continue|new|this|class|extends|import|from|export|default|async|await|try|catch|throw|finally|typeof|instanceof|in|of|null|undefined|true|false)\b/g, cls: "hl-keyword" },
    { pattern: /\b(\d+\.?\d*)\b/g, cls: "hl-number" },
    { pattern: /\b([A-Z][a-zA-Z0-9]*)\b/g, cls: "hl-type" },
  ],
  python: [
    { pattern: /(#.*$)/gm, cls: "hl-comment" },
    { pattern: /("""[\s\S]*?"""|'''[\s\S]*?''')/g, cls: "hl-string" },
    { pattern: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, cls: "hl-string" },
    { pattern: /\b(def|class|return|if|elif|else|for|while|break|continue|import|from|as|with|try|except|finally|raise|lambda|yield|global|nonlocal|pass|del|in|not|and|or|is|None|True|False|self|print)\b/g, cls: "hl-keyword" },
    { pattern: /\b(\d+\.?\d*)\b/g, cls: "hl-number" },
    { pattern: /\b([A-Z][a-zA-Z0-9]*)\b/g, cls: "hl-type" },
  ],
  html: [
    { pattern: /(<!--[\s\S]*?-->)/g, cls: "hl-comment" },
    { pattern: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, cls: "hl-string" },
    { pattern: /(&lt;\/?[a-zA-Z][a-zA-Z0-9]*)/g, cls: "hl-keyword" },
    { pattern: /\b([a-zA-Z-]+)=/g, cls: "hl-attr" },
  ],
  css: [
    { pattern: /(\/\*[\s\S]*?\*\/)/g, cls: "hl-comment" },
    { pattern: /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g, cls: "hl-string" },
    { pattern: /(#[0-9a-fA-F]{3,8})\b/g, cls: "hl-number" },
    { pattern: /\b([a-zA-Z-]+)\s*(?=:)/g, cls: "hl-keyword" },
    { pattern: /\b(\d+\.?\d*(px|em|rem|%|vh|vw|s|ms)?)\b/g, cls: "hl-number" },
  ],
  rust: [
    { pattern: /(\/\/.*$)/gm, cls: "hl-comment" },
    { pattern: /(\/\*[\s\S]*?\*\/)/g, cls: "hl-comment" },
    { pattern: /("(?:[^"\\]|\\.)*")/g, cls: "hl-string" },
    { pattern: /\b(fn|let|mut|const|struct|enum|impl|trait|pub|use|mod|crate|self|super|return|if|else|for|while|loop|break|continue|match|as|in|ref|move|async|await|where|type|dyn|unsafe|extern|true|false)\b/g, cls: "hl-keyword" },
    { pattern: /\b(\d+\.?\d*)\b/g, cls: "hl-number" },
    { pattern: /\b([A-Z][a-zA-Z0-9]*)\b/g, cls: "hl-type" },
  ],
  go: [
    { pattern: /(\/\/.*$)/gm, cls: "hl-comment" },
    { pattern: /(\/\*[\s\S]*?\*\/)/g, cls: "hl-comment" },
    { pattern: /("(?:[^"\\]|\\.)*"|`[^`]*`)/g, cls: "hl-string" },
    { pattern: /\b(func|return|if|else|for|range|switch|case|default|break|continue|go|defer|select|chan|map|struct|interface|package|import|var|const|type|struct|true|false|null)\b/g, cls: "hl-keyword" },
    { pattern: /\b(\d+\.?\d*)\b/g, cls: "hl-number" },
    { pattern: /\b([A-Z][a-zA-Z0-9]*)\b/g, cls: "hl-type" },
  ],
};

function highlightCode(el, code, lang) {
  const key = (lang || "").toLowerCase();
  const rules = HL_RULES[key] || HL_RULES.js;
  const escaped = code.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  let result = escaped;
  const replacements = [];
  for (const rule of rules) {
    const re = new RegExp(rule.pattern.source, rule.pattern.flags);
    let m;
    while ((m = re.exec(result)) !== null) {
      replacements.push({ start: m.index, end: m.index + m[0].length, text: m[0], cls: rule.cls });
    }
  }
  replacements.sort((a, b) => a.start - b.start);
  const merged = [];
  let lastEnd = 0;
  for (const r of replacements) {
    if (r.start >= lastEnd) {
      merged.push(r);
      lastEnd = r.end;
    }
  }
  let out = "";
  let pos = 0;
  for (const r of merged) {
    if (r.start > pos) out += escaped.slice(pos, r.start);
    out += `<span class="${r.cls}">${r.text}</span>`;
    pos = r.end;
  }
  if (pos < escaped.length) out += escaped.slice(pos);
  el.innerHTML = out;
}

function togglePreview(pre, button, language, code) {
  const existing = document.querySelector(".code-preview-overlay");
  if (existing) {
    existing.remove();
    button.textContent = "Run";
    return;
  }
  const overlay = document.createElement("div");
  overlay.className = "code-preview-overlay";
  const toolbar = document.createElement("div");
  toolbar.className = "preview-toolbar";
  const title = document.createElement("span");
  title.className = "preview-title";
  title.textContent = language || "Preview";
  const dismiss = () => {
    overlay.remove();
    button.textContent = "Run";
    document.removeEventListener("keydown", escapeHandler);
  };
  const escapeHandler = (event) => {
    if (event.key === "Escape") {
      dismiss();
    }
  };
  const close = document.createElement("button");
  close.type = "button";
  close.className = "preview-close";
  close.setAttribute("aria-label", "Close preview");
  close.append(svgIcon("close"));
  close.addEventListener("click", dismiss);
  toolbar.append(title, close);
  const frame = document.createElement("iframe");
  frame.className = "code-preview-frame";
  frame.setAttribute("sandbox", "allow-scripts allow-modals");
  frame.setAttribute("title", "Code preview");
  frame.src = `preview.html#${encodeURIComponent(previewDocument(language, code))}`;
  overlay.append(toolbar, frame);
  document.body.append(overlay);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) {
      dismiss();
    }
  });
  document.addEventListener("keydown", escapeHandler);
  button.textContent = "Hide";
}

function previewDocument(language, code) {
  const kind = language.toLowerCase();
  if (kind === "js" || kind === "javascript") {
    const safe = code.replace(/<\/script/gi, "<\\/script");
    return `<!DOCTYPE html><html><body><script>${safe}<\/script></body></html>`;
  }
  if (kind === "svg") {
    return `<!DOCTYPE html><html><body>${code}</body></html>`;
  }
  return code;
}



function renderToolCall(toolCall) {
  const card = document.createElement("section");
  card.className = "tool-card";
  const header = document.createElement("div");
  header.className = "tool-card-header";
  header.append(svgIcon("tool"));
  const name = document.createElement("span");
  name.textContent = toolCall.name || "Function call";
  header.append(name);
  const argumentsBlock = document.createElement("pre");
  argumentsBlock.dir = "ltr";
  argumentsBlock.textContent = prettyJson(toolCall.arguments || "{}");
  card.append(header, argumentsBlock);
  return card;
}

function prettyJson(value) {
  try {
    return JSON.stringify(JSON.parse(value), null, 2);
  } catch {
    return value;
  }
}

function svgIcon(name) {
  const paths = {
    message: '<path d="M5 5h14v11H9l-4 3V5Z"/>',
    trash: '<path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13"/>',
    copy: '<rect x="8" y="8" width="10" height="10" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/>',
    tool: '<path d="m14.7 6.3 3-3a5 5 0 0 1-6.4 6.4L5.5 15.5a2.1 2.1 0 0 0 3 3l5.8-5.8a5 5 0 0 1 6.4-6.4l-3 3"/>',
    edit: '<path d="M17 3a2.8 2.8 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>',
    refresh: '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    close: '<path d="m6 6 12 12M18 6 6 18"/>',
    down: '<path d="M12 5v13M6 13l6 6 6-6"/>',
    brain: '<path d="M9 18h6M10 22h4M8.2 14.8A7 7 0 1 1 15.8 14.8c-.7.5-.8 1.2-.8 2.2H9c0-1-.1-1.7-.8-2.2Z"/>',
    chevron: '<path d="m9 6 6 6-6 6"/>',
  };
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.innerHTML = paths[name] || paths.message;
  return svg;
}

async function sendMessage(promptOverride = null) {
  if (state.generating) {
    stopGeneration();
    return;
  }
  const prompt = (promptOverride ?? elements.promptInput.value).trim();
  if (!prompt) {
    return;
  }
  // A previous answer can still be draining its buffer. Settle it before the
  // transcript is rebuilt, or its finishing pass would never run.
  flushSmoothStream();

  const conversation = ensureActiveConversation();
  const now = Date.now();
  conversation.messages.push({
    id: identifier(),
    role: "user",
    content: prompt,
    createdAt: now,
  });
  if (conversation.messages.filter((message) => message.role === "user").length === 1) {
    conversation.title = titleFromPrompt(prompt);
  }
  const assistant = {
    id: identifier(),
    role: "assistant",
    content: "",
    toolCalls: [],
    createdAt: Date.now(),
    generating: true,
    // Recorded per message rather than read from settings at render time: the
    // toggle can be flipped while an answer is still on screen, and that must
    // not change how an answer already produced is read.
    thinking: state.settings.thinking,
  };
  conversation.messages.push(assistant);
  conversation.updatedAt = Date.now();
  elements.promptInput.value = "";
  resizePrompt();
  state.generating = true;
  state.controller = new AbortController();
  persistConversations();
  render();
  scrollToBottom();
  beginSmoothStream(assistant);

  let settled = false;
  const requestStarted = performance.now();
  let firstTokenAt = null;
  let lastTokenAt = null;
  let decodeSeconds = 0;
  try {
    const payloadMessages = conversation.messages
      .slice(0, -1)
      .filter((message) => !message.error)
      .map(messageForAPI);
    if (state.settings.systemPrompt.trim()) {
      payloadMessages.unshift({ role: "system", content: state.settings.systemPrompt.trim() });
    }
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: requestHeaders(),
      body: JSON.stringify({
        model: state.model || undefined,
        messages: payloadMessages,
        stream: true,
        stream_options: { include_usage: true },
        max_tokens: state.settings.maxTokens,
        temperature: state.settings.temperature,
        top_p: state.settings.topP,
        top_k: state.settings.topK,
        enable_thinking: state.settings.thinking,
        // Omitted at "auto": the server treats an absent field as "the
        // request did not say" and leaves the checkpoint's own level, where
        // sending a value would override it.
        ...(state.settings.thinking && state.settings.reasoningEffort !== "auto"
          ? { reasoning_effort: state.settings.reasoningEffort }
          : {}),
      }),
      signal: state.controller.signal,
    });

    if (!response.ok) {
      throw new Error(await responseError(response));
    }
    if (!response.body) {
      throw new Error("The server returned an empty stream.");
    }

    let usage = null;
    await readSSE(response.body, (data) => {
      if (data === "[DONE]") {
        return;
      }
      const chunk = JSON.parse(data);
      if (chunk.usage) {
        usage = chunk.usage;
      }
      const generatedTokens = chunk.colibri?.generated_tokens;
      if (Number.isFinite(generatedTokens)) {
        const now = performance.now();
        firstTokenAt ??= now;
        lastTokenAt = now;
        const serverDecodeSeconds = chunk.colibri?.decode_elapsed_seconds;
        decodeSeconds = Number.isFinite(serverDecodeSeconds)
          ? serverDecodeSeconds
          : (lastTokenAt - firstTokenAt) / 1000;
        assistant.metrics = formatGenerationMetrics(
          generatedTokens,
          decodeSeconds,
          true,
          (now - requestStarted) / 1000,
        );
      }
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) {
        return;
      }
      // Reasoning arrives on its own field: it is the model's notes, and the
      // server no longer folds it into the answer.
      if (typeof delta.reasoning_content === "string" && delta.reasoning_content) {
        assistant.thinkingStartedAt ??= Date.now();
        assistant.reasoning = (assistant.reasoning || "") + delta.reasoning_content;
      }
      if (typeof delta.content === "string") {
        assistant.thinkingStartedAt ??= Date.now();
        // Timed off the raw stream, not the reveal, so the reported duration is
        // the model's and not the animation's. The answer beginning is what
        // ends the thinking, whether it was marked by a tag in the content or
        // delivered on the reasoning field.
        if (assistant.thinkingSeconds === undefined
          && delta.content
          && (assistant.reasoning
            || (assistant.thinking && (assistant.content + delta.content).includes("</think>")))) {
          assistant.thinkingSeconds = (Date.now() - assistant.thinkingStartedAt) / 1000;
        }
        assistant.content += delta.content;
      }
      mergeToolCalls(assistant, delta.tool_calls);
    });
    const totalSeconds = (performance.now() - requestStarted) / 1000;
    const outputTokens = usage?.completion_tokens;
    assistant.metrics = outputTokens
      ? formatGenerationMetrics(
          outputTokens, decodeSeconds, false, totalSeconds,
        )
      : `${totalSeconds.toFixed(1)}s`;
  } catch (error) {
    if (error.name === "AbortError") {
      settled = true;
      if (!assistant.content && !assistant.toolCalls.length) {
        assistant.error = "Generation stopped.";
      } else {
        assistant.metrics = "Stopped";
      }
    } else {
      settled = true;
      assistant.error = error.message || "Generation failed.";
      toast(assistant.error, "error");
      if (/API key|401|Unauthorized/i.test(assistant.error)) {
        openSettings();
      }
    }
  } finally {
    // The request is over, but the reveal may still be a few hundred
    // milliseconds behind. The composer is released immediately while the tail
    // of the answer finishes drawing; stopping or failing skips the wait.
    state.generating = false;
    state.controller = null;
    updateSendButton();
    // The answer is complete the moment the request ends; the reveal may still
    // be drawing it, and while the page is hidden it is not drawing at all.
    // Saving here rather than only in the settle callback below means a tab
    // closed mid-reveal cannot lose a response the server already delivered.
    conversation.updatedAt = Date.now();
    persistConversations();
    endSmoothStream(() => {
      const following = isFollowingStream();
      resetStreamState();
      assistant.generating = false;
      conversation.updatedAt = Date.now();
      persistConversations();
      render();
      if (following) {
        scrollToBottom();
      }
      pollRuntime();
    }, { instant: settled });
  }
}

function formatGenerationMetrics(tokens, decodeSeconds, live, totalSeconds) {
  // The first token ends prompt evaluation/TTFT. Decode throughput therefore
  // measures the remaining token intervals, matching native benchmark-v2.
  const decodeIntervals = Math.max(0, tokens - 1);
  const rate = decodeIntervals > 0 && decodeSeconds > 0
    ? `${(decodeIntervals / decodeSeconds).toFixed(1)} tok/s`
    : "measuring decode…";
  const total = Number.isFinite(totalSeconds)
    ? ` · ${totalSeconds.toFixed(1)}s total`
    : "";
  return `${tokens} tokens · ${rate}${live ? " · live" : ""}${total}`;
}

function messageForAPI(message) {
  if (message.toolCalls?.length) {
    return {
      role: "assistant",
      content: message.content || null,
      tool_calls: message.toolCalls.map((toolCall) => ({
        id: toolCall.id || `call_${identifier().replaceAll("-", "")}`,
        type: "function",
        function: {
          name: toolCall.name,
          arguments: toolCall.arguments || "{}",
        },
      })),
    };
  }
  return { role: message.role, content: message.content };
}

function mergeToolCalls(message, deltas) {
  if (!Array.isArray(deltas)) {
    return;
  }
  for (const delta of deltas) {
    const index = Number.isInteger(delta.index) ? delta.index : message.toolCalls.length;
    if (!message.toolCalls[index]) {
      message.toolCalls[index] = { id: "", name: "", arguments: "" };
    }
    const call = message.toolCalls[index];
    if (delta.id) {
      call.id = delta.id;
    }
    if (delta.function?.name) {
      call.name += delta.function.name;
    }
    if (delta.function?.arguments) {
      call.arguments += delta.function.arguments;
    }
  }
}

async function readSSE(responseBody, onData) {
  const reader = responseBody.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      const data = block
        .split(/\r?\n/)
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) {
        onData(data);
      }
    }
    if (done) {
      if (buffer.trim()) {
        const data = buffer
          .split(/\r?\n/)
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (data) {
          onData(data);
        }
      }
      break;
    }
  }
}

function stopGeneration() {
  state.controller?.abort();
}

function requestHeaders() {
  const headers = { "Content-Type": "application/json" };
  if (state.apiKey) {
    headers.Authorization = `Bearer ${state.apiKey}`;
  }
  return headers;
}

async function responseError(response) {
  try {
    const payload = await response.json();
    return payload.error?.message || `Request failed with status ${response.status}.`;
  } catch {
    return `Request failed with status ${response.status}.`;
  }
}

// A poll that outlives its interval must not start another one. The browser
// only keeps six connections per origin, so polls queued behind a loading or
// busy server would otherwise stall the page's own requests.
let pollInFlight = false;

function pollRequest(path) {
  return fetch(path, {
    headers: authHeaders(),
    signal: AbortSignal.timeout?.(15000),
  });
}

async function pollRuntime() {
  if (pollInFlight) {
    return;
  }
  pollInFlight = true;
  try {
    const response = await pollRequest("/health");
    if (!response.ok) {
      if (response.status === 401) {
        setRuntimeStatus("locked");
        return;
      }
      throw new Error(String(response.status));
    }
    state.health = await response.json();
    const [modelResponse, propertiesResponse] = await Promise.all([
      state.model ? null : pollRequest("/v1/models"),
      state.properties ? null : pollRequest("/props"),
    ]);
    if (modelResponse?.ok) {
      const models = await modelResponse.json();
      const modelList = models.data || [];
      state.models = modelList;
      if (modelList.length > 1) {
        state.model = state.model || modelList[0]?.id || null;
        renderModelSelector();
      } else {
        state.model = modelList[0]?.id || null;
        elements.modelSelectorWrap.hidden = true;
      }
    }
    if (propertiesResponse?.ok) {
      const properties = await propertiesResponse.json();
      // Context window and output ceiling are fixed for the loaded model, so
      // this endpoint is read once rather than on every poll.
      state.properties = properties;
      if (!state.settingsCustomized) {
        state.settings = settingsFromModelDefaults();
        renderSettingsSummary();
      }
      state.maxOutputTokens = clampInteger(
        Math.min(
          properties.context_window || Number.MAX_SAFE_INTEGER,
          properties.max_output_tokens || Number.MAX_SAFE_INTEGER,
        ),
        1,
        Number.MAX_SAFE_INTEGER,
        64,
      );
      if (state.settings.maxTokens > state.maxOutputTokens) {
        state.settings.maxTokens = state.maxOutputTokens;
        persistSettings();
        renderSettingsSummary();
      }
    }
    setRuntimeStatus(state.health.busy ? "busy" : "online");
  } catch {
    setRuntimeStatus("offline");
  } finally {
    pollInFlight = false;
  }
}

function authHeaders() {
  return state.apiKey ? { Authorization: `Bearer ${state.apiKey}` } : {};
}

function setRuntimeStatus(status) {
  elements.statusDot.className = `status-dot ${status}`;
  if (status === "locked") {
    elements.modelStatus.textContent = "API key required · open Settings";
    elements.runtimePill.hidden = true;
    return;
  }
  if (status === "offline") {
    elements.modelStatus.textContent = "Server unavailable";
    elements.runtimePill.hidden = true;
    return;
  }
  const model = state.model || state.health?.model || "Local model";
  elements.modelStatus.textContent = status === "busy" ? `${model} · generating` : `${model} · ready`;
  const execution = state.health?.execution;
  if (!execution) {
    elements.runtimePill.hidden = true;
    return;
  }
  elements.runtimePill.hidden = false;
  const nativeCuda = execution.backend === "native-v2-cpp-cuda";
  const usesCuda = nativeCuda || execution.device === "cuda";
  elements.deviceLabel.textContent = usesCuda ? "CUDA" : "CPU";
  // Expert mode and cache figures matter when they are being investigated, not
  // on every glance at the transcript. The bar keeps the one word that changes
  // how you read a slow response; the rest is a hover away.
  elements.runtimePill.dataset.tooltip = runtimeDetail(execution);
}

function runtimeDetail(execution) {
  const lines = [];
  if (execution.expert_mode) {
    const cacheMiB = Math.round((execution.expert_cache_bytes || 0) / (1024 * 1024));
    const alias = execution.requested_expert_mode
      && execution.requested_expert_mode !== execution.expert_mode
      ? ` (requested ${execution.requested_expert_mode})`
      : "";
    lines.push(`${execution.expert_mode}${alias} experts · ${cacheMiB} MB GPU`);
    if (execution.expert_fallback_reason) {
      lines.push(execution.expert_fallback_reason);
    }
  } else if (execution.device === "cuda") {
    lines.push(`KV cache ${execution.cache_used_mib || 0} / ${execution.cache_limit_mib || 0} MB`);
  } else {
    lines.push("Portable CPU execution");
  }
  return lines.join("\n");
}

function renderModelSelector() {
  if (state.models.length <= 1) {
    elements.modelSelectorWrap.hidden = true;
    return;
  }
  elements.modelSelectorWrap.hidden = false;
  elements.modelSelector.innerHTML = "";
  for (const m of state.models) {
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = m.id;
    if (m.id === state.model) opt.selected = true;
    elements.modelSelector.append(opt);
  }
}

function openSettings() {
  elements.apiKey.value = state.apiKey;
  elements.systemPrompt.value = state.settings.systemPrompt;
  elements.maxTokens.max = state.maxOutputTokens;
  elements.maxTokens.value = Math.min(
    state.settings.maxTokens,
    state.maxOutputTokens,
  );
  elements.temperature.value = state.settings.temperature;
  elements.topP.value = state.settings.topP;
  elements.topK.value = state.settings.topK;
  elements.thinkingSetting.checked = state.settings.thinking;
  elements.reasoningEffort.value = state.settings.reasoningEffort;
  syncReasoningEffortAvailability();
  if (!elements.settingsDialog.open) {
    elements.settingsDialog.showModal();
  }
}

function saveSettings() {
  state.settings = normalizeSettings({
    systemPrompt: elements.systemPrompt.value.trim(),
    maxTokens: elements.maxTokens.value,
    temperature: elements.temperature.value,
    topP: elements.topP.value,
    topK: elements.topK.value,
    thinking: elements.thinkingSetting.checked,
    reasoningEffort: elements.reasoningEffort.value,
  });
  state.settings.maxTokens = Math.min(
    state.settings.maxTokens,
    state.maxOutputTokens,
  );
  state.apiKey = elements.apiKey.value.trim();
  saveSession(API_KEY, state.apiKey);
  persistSettings();
  elements.settingsDialog.close();
  renderSettingsSummary();
  pollRuntime();
  toast("Settings saved.");
}

function resetSettings() {
  state.settings = settingsFromModelDefaults();
  elements.systemPrompt.value = state.settings.systemPrompt;
  elements.maxTokens.value = state.settings.maxTokens;
  elements.temperature.value = state.settings.temperature;
  elements.topP.value = state.settings.topP;
  elements.topK.value = state.settings.topK;
  elements.thinkingSetting.checked = state.settings.thinking;
  elements.reasoningEffort.value = state.settings.reasoningEffort;
  syncReasoningEffortAvailability();
}

// A graded effort only means something while the model is thinking: the
// template that reads it skips the instruction entirely when thinking is off,
// so an enabled control there would silently do nothing.
function syncReasoningEffortAvailability() {
  elements.reasoningEffort.disabled = !elements.thinkingSetting.checked;
}

function renderSettingsSummary() {
  const reasoning = reasoningState();
  elements.thinkingChip.setAttribute("aria-pressed", String(reasoning !== "off"));
  elements.thinkingChip.lastChild.textContent = ` ${REASONING_LABELS[reasoning]}`;
  elements.thinkingChip.title =
    reasoning === "off"
      ? "Reasoning off. Click to let the model use its own level."
      : reasoning === "auto"
        ? "Reasoning at the model's own level. Click to set one."
        : `Reasoning effort ${reasoning}. Click to change.`;
  elements.tokenChip.textContent = `${state.settings.maxTokens} tokens`;
}

function reasoningState() {
  if (!state.settings.thinking) return "off";
  const effort = state.settings.reasoningEffort;
  return REASONING_EFFORTS.includes(effort) ? effort : "auto";
}

function applyReasoningState(next) {
  state.settings.thinking = next !== "off";
  state.settings.reasoningEffort = next === "off" ? "auto" : next;
  if (elements.settingsDialog.open) {
    elements.thinkingSetting.checked = state.settings.thinking;
    elements.reasoningEffort.value = state.settings.reasoningEffort;
    syncReasoningEffortAvailability();
  }
  persistSettings();
  renderSettingsSummary();
}

function cycleReasoning() {
  const current = REASONING_STATES.indexOf(reasoningState());
  applyReasoningState(
    REASONING_STATES[(current + 1) % REASONING_STATES.length],
  );
}

/* =============================================================================
   Theme

   Three states, not two. "System" is the default and a first-class choice that
   is stored like any other: it keeps tracking the OS for the life of the tab,
   including a change made while the app is open. The old toggle read the OS
   once at startup and then latched to whatever it found, so a machine that
   switched to dark in the evening left this window bright until someone
   noticed the button.

   index.html resolves and applies the same preference inline before first
   paint. This module is the authority afterwards; the two agree by reading the
   same key and following the same rule.
   ========================================================================== */

const THEME_PREFERENCES = ["system", "light", "dark"];
const THEME_LABELS = { system: "System", light: "Light", dark: "Dark" };
const THEME_COLORS = { dark: "#212121", light: "#ffffff" };

const prefersDark = matchMedia("(prefers-color-scheme: dark)");

function loadThemePreference() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    return THEME_PREFERENCES.includes(saved) ? saved : "system";
  } catch {
    return "system";
  }
}

function resolveTheme(preference) {
  if (preference === "system") {
    return prefersDark.matches ? "dark" : "light";
  }
  return preference;
}

function applyTheme(preference) {
  const theme = resolveTheme(preference);
  document.documentElement.dataset.theme = theme;
  document.documentElement.dataset.themePreference = preference;
  // Keeps the browser's own chrome — mobile address bar, window borders on
  // some platforms — in step with the page instead of framing it in the
  // opposite theme.
  document.querySelector('meta[name="theme-color"]')?.setAttribute("content", THEME_COLORS[theme]);
  const next = THEME_PREFERENCES[(THEME_PREFERENCES.indexOf(preference) + 1) % THEME_PREFERENCES.length];
  const label = preference === "system"
    ? `System (${THEME_LABELS[theme].toLowerCase()})`
    : THEME_LABELS[preference];
  elements.themeToggle.setAttribute("aria-label", `Theme: ${label}. Switch to ${THEME_LABELS[next]}.`);
  elements.themeToggle.dataset.tooltip = `Theme: ${label}`;
}

function cycleTheme() {
  const index = THEME_PREFERENCES.indexOf(state.theme);
  state.theme = THEME_PREFERENCES[(index + 1) % THEME_PREFERENCES.length];
  try {
    localStorage.setItem(THEME_KEY, state.theme);
  } catch {
    // A preference that cannot be stored still applies for this session.
  }
  applyTheme(state.theme);
  toast(`Theme: ${state.theme === "system"
    ? `System (${THEME_LABELS[resolveTheme("system")].toLowerCase()})`
    : THEME_LABELS[state.theme]}`);
}

function initializeTheme() {
  state.theme = loadThemePreference();
  applyTheme(state.theme);
  prefersDark.addEventListener("change", () => {
    if (state.theme === "system") {
      applyTheme("system");
    }
  });
}

function resizePrompt() {
  elements.promptInput.style.height = "auto";
  elements.promptInput.style.height = `${Math.min(elements.promptInput.scrollHeight, 180)}px`;
  updateSendButton();
}

function updateSendButton() {
  elements.sendButton.classList.toggle("generating", state.generating);
  elements.sendButton.disabled = !state.generating && !elements.promptInput.value.trim();
  elements.sendButton.setAttribute("aria-label", state.generating ? "Stop generation" : "Send message");
}

/* =============================================================================
   Following the stream

   Reading back through an answer while it is still being written has to win
   over keeping the newest line in view. Two things make that work: a scroll
   the app performed is never mistaken for one the reader performed, and an
   upward gesture detaches immediately rather than waiting to see how far it
   travelled. Re-attaching is deliberate — the reader returns to the bottom, or
   presses the button.
   ========================================================================== */

// Being pinned means sitting on the last line, not merely near it. Anything
// looser is a window the reader can land inside after a deliberate scroll, and
// then the next painted frame drags them back out of it. A few pixels of
// tolerance only absorbs sub-pixel rounding at fractional zoom levels.
const STICK_THRESHOLD = 4;

// The jump button appears well before that, so a reader who has drifted a
// little still has an obvious way back.
const AFFORDANCE_THRESHOLD = 72;

let touchAnchor = null;

function scrollDistanceFromBottom() {
  return elements.chatScroll.scrollHeight
    - elements.chatScroll.scrollTop
    - elements.chatScroll.clientHeight;
}

// Answers the question the streaming loop needs answered, and it has to be
// asked *before* the frame's text is inserted: once new lines are in the DOM
// the reader is no longer at the bottom by definition, and every frame would
// look like they had scrolled away.
function isFollowingStream() {
  return state.stickToBottom && scrollDistanceFromBottom() <= STICK_THRESHOLD;
}

function scrollToBottom({ smoothly = false } = {}) {
  // A streaming repaint has to land in the same frame as the text it follows,
  // so the default is an immediate jump. Smooth scrolling is only for the
  // deliberate "jump to latest" gesture.
  if (smoothly) {
    elements.chatScroll.scrollTo({ top: elements.chatScroll.scrollHeight, behavior: "smooth" });
  } else {
    elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
  }
  state.stickToBottom = true;
  updateScrollAffordance();
}

function handleScroll() {
  // Following resumes only from the very bottom. Ending up close to it — after
  // scrolling back through an answer, say — is not a request to be pinned.
  state.stickToBottom = scrollDistanceFromBottom() <= STICK_THRESHOLD;
  updateScrollAffordance();
}

// Wheel and touch are read directly as well, so the intent registers even when
// the compositor has not produced a scroll event yet.
function detachFromBottom() {
  if (state.stickToBottom) {
    state.stickToBottom = false;
    updateScrollAffordance();
  }
}

function handleWheel(event) {
  if (event.deltaY < 0) {
    detachFromBottom();
  }
}

function handleTouchStart(event) {
  touchAnchor = event.touches[0]?.clientY ?? null;
}

function handleTouchMove(event) {
  const y = event.touches[0]?.clientY;
  if (touchAnchor === null || y === undefined) {
    return;
  }
  // Finger travelling down drags the content down, which is a scroll up.
  if (y - touchAnchor > 4) {
    detachFromBottom();
  }
  touchAnchor = y;
}

function updateScrollAffordance() {
  elements.scrollBottom.hidden = scrollDistanceFromBottom() <= AFFORDANCE_THRESHOLD;
}

/* =============================================================================
   Tooltips

   One floating label, moved to whichever control is hovered or focused, rather
   than a pseudo-element per button. Two reasons it is not CSS: the sidebar and
   the main panel both clip their overflow for the collapse animation, so a
   label attached to a button near their edges would be cut off; and a shared
   element positioned with `fixed` can be nudged back inside the viewport and
   flipped above the control when there is no room below it.

   This replaces `title`, which browsers draw at the pointer after their own
   delay in the OS's styling — none of which is ours to place.
   ========================================================================== */

const TOOLTIP_DELAY = 140;
const TOOLTIP_GAP = 8;
const TOOLTIP_MARGIN = 8;

const tooltip = document.createElement("div");
tooltip.className = "tooltip";
tooltip.setAttribute("role", "tooltip");
tooltip.hidden = true;
document.body.append(tooltip);

let tooltipTimer = null;
let tooltipTarget = null;

function positionTooltip(target) {
  const anchor = target.getBoundingClientRect();
  const label = tooltip.getBoundingClientRect();
  let top = anchor.bottom + TOOLTIP_GAP;
  if (top + label.height > window.innerHeight - TOOLTIP_MARGIN) {
    top = anchor.top - label.height - TOOLTIP_GAP;
  }
  const centred = anchor.left + (anchor.width - label.width) / 2;
  const left = Math.min(
    Math.max(TOOLTIP_MARGIN, centred),
    window.innerWidth - label.width - TOOLTIP_MARGIN,
  );
  tooltip.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
}

function showTooltip(target) {
  const label = target.dataset.tooltip;
  if (!label) {
    return;
  }
  tooltipTarget = target;
  tooltip.textContent = label;
  tooltip.hidden = false;
  // Measured only once it is laid out, which is why it is unhidden first.
  positionTooltip(target);
}

function hideTooltip() {
  clearTimeout(tooltipTimer);
  tooltipTimer = null;
  tooltipTarget = null;
  tooltip.hidden = true;
}

function queueTooltip(target) {
  if (tooltipTarget === target) {
    return;
  }
  clearTimeout(tooltipTimer);
  // Shown at once when moving between controls, so sweeping along a row of
  // buttons does not restart the delay on each one.
  const delay = tooltipTarget ? 0 : TOOLTIP_DELAY;
  tooltipTarget = null;
  tooltip.hidden = true;
  tooltipTimer = setTimeout(() => showTooltip(target), delay);
}

function bindTooltips() {
  document.addEventListener("pointerover", (event) => {
    const target = event.target.closest?.("[data-tooltip]");
    if (target) {
      queueTooltip(target);
    } else if (tooltipTarget || tooltipTimer) {
      hideTooltip();
    }
  });
  // Keyboard users get the same label, without needing a pointer.
  document.addEventListener("focusin", (event) => {
    const target = event.target.closest?.("[data-tooltip]");
    if (target) {
      queueTooltip(target);
    }
  });
  document.addEventListener("focusout", hideTooltip);
  document.addEventListener("pointerdown", hideTooltip);
  window.addEventListener("blur", hideTooltip);
  // A label anchored to a control that has since moved is worse than none.
  window.addEventListener("scroll", hideTooltip, { capture: true, passive: true });
  window.addEventListener("resize", hideTooltip, { passive: true });
}

/* =============================================================================
   Sidebar

   Two different behaviours behind one pair of buttons. Below the layout
   breakpoint the sidebar is a drawer that slides over the page and is dismissed
   by picking something, tapping the scrim or pressing Escape — transient, so
   its state is never stored. On a wide screen it is a column of the layout, and
   hiding it is a lasting preference about how much room the transcript gets.
   ========================================================================== */

const mobileLayout = matchMedia("(max-width: 900px)");

function toggleSidebar() {
  if (mobileLayout.matches) {
    elements.body.classList.toggle("sidebar-visible");
    return;
  }
  setSidebarHidden(!state.sidebarHidden);
}

function setSidebarHidden(hidden) {
  state.sidebarHidden = hidden;
  elements.body.classList.toggle("sidebar-hidden", hidden);
  try {
    localStorage.setItem(SIDEBAR_KEY, hidden ? "hidden" : "visible");
  } catch {
    // A layout preference that cannot be stored still applies for this session.
  }
  const label = hidden ? "Show sidebar" : "Hide sidebar";
  for (const control of [elements.sidebarOpen, elements.sidebarClose]) {
    control.setAttribute("aria-label", label);
    control.dataset.tooltip = label;
  }
  updateRailLabels();
}

// In the rail the controls are icons with nothing written beside them, so each
// one carries its name as a tooltip. Expanded, that name is already on screen
// and a tooltip repeating it is just noise — so the attribute comes and goes
// with the layout, including when a window resize crosses the breakpoint.
function updateRailLabels() {
  const rail = state.sidebarHidden && !mobileLayout.matches;
  const labelled = [
    [elements.brand, "Colibri"],
    [elements.newChat, "New conversation"],
    [elements.searchBox, "Search chats"],
    [elements.openSettings, "Settings"],
    [elements.importChat, "Import"],
  ];
  for (const [control, label] of labelled) {
    if (rail) {
      control.dataset.tooltip = label;
    } else {
      delete control.dataset.tooltip;
    }
  }
}

function initializeSidebar() {
  let hidden = false;
  try {
    hidden = localStorage.getItem(SIDEBAR_KEY) === "hidden";
  } catch {
    hidden = false;
  }
  setSidebarHidden(hidden);
  // Crossing the breakpoint swaps the rail for the drawer, and the labels that
  // suit one are wrong for the other.
  mobileLayout.addEventListener("change", updateRailLabels);
}

// Only ever dismisses the mobile drawer. Selecting a conversation calls this,
// and on a wide screen that must not collapse the column out from under the
// click that just used it.
function closeSidebar() {
  elements.body.classList.remove("sidebar-visible");
}

function titleFromPrompt(prompt) {
  const firstLine = prompt.replace(/\s+/g, " ").trim();
  return firstLine.length > 42 ? `${firstLine.slice(0, 41)}…` : firstLine;
}

function formatTime(timestamp) {
  return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(timestamp);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied to clipboard.");
  } catch {
    toast("Clipboard access is unavailable.", "error");
  }
}

function exportConversation() {
  const conversation = activeConversation();
  if (!conversation?.messages.length) {
    toast("There is nothing to export.");
    return;
  }
  const blob = new Blob([JSON.stringify(conversation, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${conversation.title.replace(/[^a-z0-9]+/gi, "-").toLowerCase() || "conversation"}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function toast(message, kind = "info") {
  const item = document.createElement("div");
  item.className = `toast ${kind}`;
  item.textContent = message;
  elements.toastRegion.append(item);
  setTimeout(() => item.remove(), 3200);
}

function bindEvents() {
  elements.brand.addEventListener("click", (event) => {
    event.preventDefault();
    startNewConversation();
  });
  // In rail mode the field itself is hidden, so the icon stands in for it.
  elements.searchBox.addEventListener("click", (event) => {
    if (!state.sidebarHidden || mobileLayout.matches) {
      return;
    }
    event.preventDefault();
    setSidebarHidden(false);
    elements.conversationSearch.focus();
  });
  elements.newChat.addEventListener("click", startNewConversation);
  elements.conversationSearch.addEventListener("input", debounce(renderConversationList, 150));
  elements.sidebarOpen.addEventListener("click", toggleSidebar);
  elements.sidebarClose.addEventListener("click", toggleSidebar);
  elements.sidebarScrim.addEventListener("click", closeSidebar);
  elements.sendButton.addEventListener("click", () => sendMessage());
  elements.promptInput.addEventListener("input", resizePrompt);
  elements.promptInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      sendMessage();
    }
  });
  elements.chatScroll.addEventListener("scroll", handleScroll, { passive: true });
  elements.chatScroll.addEventListener("wheel", handleWheel, { passive: true });
  elements.chatScroll.addEventListener("touchstart", handleTouchStart, { passive: true });
  elements.chatScroll.addEventListener("touchmove", handleTouchMove, { passive: true });
  elements.scrollBottom.addEventListener("click", () => {
    scrollToBottom({ smoothly: true });
    elements.promptInput.focus();
  });
  elements.openSettings.addEventListener("click", openSettings);
  elements.quickSettings.addEventListener("click", openSettings);
  elements.importChat.addEventListener("click", importConversation);
  elements.thinkingChip.addEventListener("click", cycleReasoning);
  elements.saveSettings.addEventListener("click", (event) => {
    event.preventDefault();
    saveSettings();
  });
  elements.resetSettings.addEventListener("click", resetSettings);
  // Toggling thinking inside the dialog has to enable or grey the effort
  // control with it, or the two read as contradicting each other.
  elements.thinkingSetting.addEventListener(
    "change", syncReasoningEffortAvailability);
  elements.clearChat.addEventListener("click", clearConversation);
  elements.exportChat.addEventListener("click", exportConversation);
  elements.themeToggle.addEventListener("click", cycleTheme);
  elements.modelSelector.addEventListener("change", () => {
    state.model = elements.modelSelector.value;
    setRuntimeStatus(state.health?.busy ? "busy" : "online");
    toast(`Switched to ${state.model}`);
  });
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      startNewConversation();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "b") {
      event.preventDefault();
      toggleSidebar();
    }
    if (event.key === "Escape") {
      closeSidebar();
    }
  });
}

initializeTheme();
initializeSidebar();
bindEvents();
bindTooltips();
if (!state.conversations.length) {
  createConversation();
} else {
  state.activeId = [...state.conversations].sort((left, right) => right.updatedAt - left.updatedAt)[0].id;
  render();
}
resizePrompt();
pollRuntime();
setInterval(() => {
  if (!document.hidden) {
    pollRuntime();
  }
}, 5000);
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    catchUpSmoothStream();
    pollRuntime();
  }
});
