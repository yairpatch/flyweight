const STORAGE_KEY = "colibri.chat.v1";
const SETTINGS_KEY = "colibri.settings.v1";
const THEME_KEY = "colibri.theme";
const API_KEY = "colibri.api-key";

const DEFAULT_SETTINGS = Object.freeze({
  systemPrompt: "",
  maxTokens: 64,
  temperature: 0.7,
  topP: 0.95,
  topK: 20,
  thinking: false,
});

const elements = {
  body: document.body,
  sidebar: document.querySelector("#sidebar"),
  sidebarOpen: document.querySelector("#sidebar-open"),
  sidebarClose: document.querySelector("#sidebar-close"),
  sidebarScrim: document.querySelector("#sidebar-scrim"),
  brand: document.querySelector(".brand"),
  newChat: document.querySelector("#new-chat"),
  conversationSearch: document.querySelector("#conversation-search"),
  conversationList: document.querySelector("#conversation-list"),
  conversationTitle: document.querySelector("#conversation-title"),
  statusDot: document.querySelector("#status-dot"),
  modelStatus: document.querySelector("#model-status"),
  modelSelectorWrap: document.querySelector("#model-selector-wrap"),
  modelSelector: document.querySelector("#model-selector"),
  runtimePill: document.querySelector("#runtime-pill"),
  deviceLabel: document.querySelector("#device-label"),
  cacheLabel: document.querySelector("#cache-label"),
  chatScroll: document.querySelector("#chat-scroll"),
  welcome: document.querySelector("#welcome"),
  messages: document.querySelector("#messages"),
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
  resetSettings: document.querySelector("#reset-settings"),
  saveSettings: document.querySelector("#save-settings"),
  toastRegion: document.querySelector("#toast-region"),
};

const state = {
  conversations: loadConversations(),
  activeId: null,
  settings: loadSettings(),
  apiKey: readSession(API_KEY),
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
    topK: clampInteger(value.topK, 1, 200, DEFAULT_SETTINGS.topK),
    thinking: Boolean(value.thinking),
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
    const serialized = JSON.stringify(conversation);
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
  localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
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
  requestAnimationFrame(scrollToBottom);
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
  input.value = conversation.title;
  nameElement.replaceWith(input);
  input.focus();
  input.select();
  const commit = () => {
    const val = input.value.trim();
    if (val && val !== conversation.title) {
      conversation.title = val;
      if (conversation.id === state.activeId) {
        elements.conversationTitle.textContent = val;
      }
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
  elements.conversationTitle.textContent = conversation?.title || "New conversation";
  elements.welcome.hidden = messages.length > 0;
  elements.messages.replaceChildren();
  for (const message of messages) {
    elements.messages.append(renderMessage(message));
  }
}

function renderMessage(message) {
  const article = document.createElement("article");
  article.className = `message ${message.role}`;
  article.dataset.messageId = message.id;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = message.role === "assistant" ? "C" : "YOU";

  const body = document.createElement("div");
  body.className = "message-body";
  const meta = document.createElement("div");
  meta.className = "message-meta";
  const author = document.createElement("span");
  author.className = "message-author";
  author.textContent = message.role === "assistant" ? "Colibri" : "You";
  const time = document.createElement("time");
  time.className = "message-time";
  time.dateTime = new Date(message.createdAt).toISOString();
  time.textContent = formatTime(message.createdAt);
  meta.append(author, time);

  if (message.content) {
    const copy = document.createElement("button");
    copy.type = "button";
    copy.className = "message-copy";
    copy.setAttribute("aria-label", "Copy message");
    copy.append(svgIcon("copy"));
    copy.addEventListener("click", () => copyText(message.content));
    meta.append(copy);
  }

  const actions = document.createElement("div");
  actions.className = "message-actions";
  if (message.role === "user" && !message.generating) {
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "msg-action";
    edit.setAttribute("aria-label", "Edit message");
    edit.append(svgIcon("edit"));
    edit.addEventListener("click", () => startEditMessage(message));
    actions.append(edit);
  }
  if (message.role === "assistant" && !message.generating && !message.error) {
    const regen = document.createElement("button");
    regen.type = "button";
    regen.className = "msg-action";
    regen.setAttribute("aria-label", "Regenerate");
    regen.append(svgIcon("refresh"));
    regen.addEventListener("click", () => regenerateFrom(message));
    actions.append(regen);
  }
  if (actions.childNodes.length) meta.append(actions);

  const content = document.createElement("div");
  content.className = "message-content";
  renderMessageContent(content, message);
  body.append(meta, content);
  if (message.metrics) {
    const metrics = document.createElement("span");
    metrics.className = "message-metrics";
    metrics.textContent = message.metrics;
    body.append(metrics);
  }
  article.append(avatar, body);
  return article;
}

function renderMessageContent(content, message) {
  content.replaceChildren();
  renderRichText(content, message.content || "");
  if (message.generating) {
    const cursor = document.createElement("span");
    cursor.className = "stream-cursor";
    cursor.setAttribute("aria-label", "Generating");
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
  bodyStart: 0,
  committedEnd: 0,
  thinkingSettled: false,
  tailNodes: [],
  toolHost: null,
};

let streamFrame = null;
let streamMessage = null;

function scheduleStreamRender(message) {
  streamMessage = message;
  if (streamFrame !== null) {
    return;
  }
  streamFrame = requestAnimationFrame(() => {
    streamFrame = null;
    const pending = streamMessage;
    streamMessage = null;
    if (!pending) {
      return;
    }
    updateStreamingMessage(pending);
    if (state.stickToBottom) {
      scrollToBottom();
    }
  });
}

function cancelStreamRender() {
  if (streamFrame !== null) {
    cancelAnimationFrame(streamFrame);
    streamFrame = null;
  }
  streamMessage = null;
}

function resetStreamState(messageId = null, content = null) {
  stream.messageId = messageId;
  stream.content = content;
  stream.bodyStart = 0;
  stream.committedEnd = 0;
  stream.thinkingSettled = false;
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

function updateStreamingMessage(message) {
  const article = elements.messages.querySelector(`[data-message-id="${message.id}"]`);
  const content = article?.querySelector(".message-content");
  if (!content) {
    renderConversation();
    return;
  }
  updateMessageMetrics(article, message);
  const cursor = content.querySelector(".stream-cursor");
  if (!cursor || !message.generating) {
    resetStreamState();
    renderMessageContent(content, message);
    return;
  }
  // Anchored to the element as well as the id: a full re-render mid-stream
  // replaces the node, and the committed offsets would no longer describe it.
  if (stream.messageId !== message.id || stream.content !== content) {
    resetStreamState(message.id, content);
    content.replaceChildren(cursor);
  }

  const text = message.content || "";
  const thinkingClose = text.indexOf("</think>");
  const thinkingOpen = text.indexOf("<think>");
  let body = "";

  if (thinkingClose !== -1) {
    if (!stream.thinkingSettled) {
      content.querySelector(".thinking-streaming")?.remove();
      if (thinkingOpen !== -1 && thinkingOpen < thinkingClose) {
        content.insertBefore(
          renderThinkingBlock(text.slice(thinkingOpen + 8, thinkingClose)),
          cursor,
        );
      }
      stream.thinkingSettled = true;
      stream.bodyStart = thinkingClose + 9;
    }
    body = text.slice(stream.bodyStart);
  } else if (thinkingOpen !== -1) {
    let indicator = content.querySelector(".thinking-streaming");
    if (!indicator) {
      indicator = renderThinkingIndicator();
      content.insertBefore(indicator, cursor);
    }
    const thinkingBody = indicator.querySelector(".thinking-stream-body");
    if (thinkingBody) {
      thinkingBody.textContent = text.slice(thinkingOpen + 8).trim()
        || "Waiting for thoughts...";
    }
  } else {
    body = text;
  }

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
}

function renderThinkingBlock(content) {
  const details = document.createElement("details");
  details.className = "thinking-block";
  const summary = document.createElement("summary");
  summary.className = "thinking-summary";
  summary.textContent = "Thought process";
  const body = document.createElement("div");
  body.className = "thinking-body";
  const p = document.createElement("p");
  p.textContent = content.trim();
  body.append(p);
  details.append(summary, body);
  return details;
}

function renderThinkingIndicator() {
  const wrapper = document.createElement("div");
  wrapper.className = "thinking-streaming";
  
  const header = document.createElement("div");
  header.className = "thinking-stream-header";
  header.append(svgIcon("brain"));
  const label = document.createElement("span");
  label.textContent = "Thinking";
  const dots = document.createElement("span");
  dots.className = "thinking-ellipsis";
  dots.textContent = "...";
  header.append(label, dots);
  
  const chevron = document.createElement("span");
  chevron.className = "thinking-chevron";
  chevron.textContent = "\u25B6";
  header.append(chevron);
  
  const body = document.createElement("div");
  body.className = "thinking-stream-body";
  body.style.display = "none";
  
  header.addEventListener("click", () => {
    const open = body.style.display !== "none";
    body.style.display = open ? "none" : "block";
    chevron.classList.toggle("open", !open);
  });
  
  wrapper.append(header, body);
  return wrapper;
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
  let metrics = article.querySelector(".message-body > .message-metrics");
  if (!message.metrics) {
    metrics?.remove();
    return;
  }
  if (!metrics) {
    metrics = document.createElement("span");
    metrics.className = "message-metrics";
    article.querySelector(".message-body")?.append(metrics);
  }
  metrics.textContent = message.metrics;
}

function renderRichText(container, text) {
  if (!text) {
    return;
  }
  const thinkingPattern = /<think>([\s\S]*?)<\/think>/g;
  let tCursor = 0;
  let tMatch;
  while ((tMatch = thinkingPattern.exec(text)) !== null) {
    renderRichTextSegment(container, text.slice(tCursor, tMatch.index));
    container.append(renderThinkingBlock(tMatch[1]));
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
      appendInline(el, line.replace(/^#{1,6}\s+/, ""));
      container.append(el);
      i++;
    } else if (/^>\s/.test(line)) {
      const items = [];
      while (i < lines.length && /^>\s/.test(lines[i])) {
        items.push(lines[i].replace(/^>\s+/, ""));
        i++;
      }
      const bq = document.createElement("blockquote");
      appendMarkdownBlock(bq, items.join("\n"));
      container.append(bq);
    } else if (/^[-*]\s/.test(line)) {
      const ul = document.createElement("ul");
      while (i < lines.length && /^[-*]\s/.test(lines[i])) {
        const li = document.createElement("li");
        appendInline(li, lines[i].replace(/^[-*]\s+/, ""));
        ul.append(li);
        i++;
      }
      container.append(ul);
    } else if (/^\d+\.\s/.test(line)) {
      const ol = document.createElement("ol");
      while (i < lines.length && /^\d+\.\s/.test(lines[i])) {
        const li = document.createElement("li");
        appendInline(li, lines[i].replace(/^\d+\.\s+/, ""));
        ol.append(li);
        i++;
      }
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
        paraLines.forEach((pl, idx) => {
          appendInline(p, pl);
          if (idx + 1 < paraLines.length) p.append(document.createElement("br"));
        });
        container.append(p);
      }
    }
  }
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
    brain: '<path d="M9 18h6M10 22h4M8.2 14.8A7 7 0 1 1 15.8 14.8c-.7.5-.8 1.2-.8 2.2H9c0-1-.1-1.7-.8-2.2Z"/>',
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
      if (typeof delta.content === "string") {
        assistant.content += delta.content;
      }
      mergeToolCalls(assistant, delta.tool_calls);
      scheduleStreamRender(assistant);
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
      if (!assistant.content && !assistant.toolCalls.length) {
        assistant.error = "Generation stopped.";
      } else {
        assistant.metrics = "Stopped";
      }
    } else {
      assistant.error = error.message || "Generation failed.";
      toast(assistant.error, "error");
      if (/API key|401|Unauthorized/i.test(assistant.error)) {
        openSettings();
      }
    }
  } finally {
    cancelStreamRender();
    resetStreamState();
    assistant.generating = false;
    state.generating = false;
    state.controller = null;
    conversation.updatedAt = Date.now();
    persistConversations();
    render();
    scrollToBottom();
    pollRuntime();
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
  if (execution.expert_mode) {
    const cacheMiB = Math.round((execution.expert_cache_bytes || 0) / (1024 * 1024));
    const alias = execution.requested_expert_mode
      && execution.requested_expert_mode !== execution.expert_mode
      ? ` (${execution.requested_expert_mode})`
      : "";
    elements.cacheLabel.textContent = `${execution.expert_mode}${alias} experts · ${cacheMiB} MB GPU`;
    elements.cacheLabel.title = execution.expert_fallback_reason || "";
  } else {
    elements.cacheLabel.textContent = execution.device === "cuda"
      ? `${execution.cache_used_mib || 0} / ${execution.cache_limit_mib || 0} MB`
      : "Portable";
    elements.cacheLabel.title = "";
  }
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
  state.settings = { ...DEFAULT_SETTINGS };
  elements.systemPrompt.value = state.settings.systemPrompt;
  elements.maxTokens.value = state.settings.maxTokens;
  elements.temperature.value = state.settings.temperature;
  elements.topP.value = state.settings.topP;
  elements.topK.value = state.settings.topK;
  elements.thinkingSetting.checked = state.settings.thinking;
}

function renderSettingsSummary() {
  elements.thinkingChip.setAttribute("aria-pressed", String(state.settings.thinking));
  elements.thinkingChip.lastChild.textContent = state.settings.thinking ? " Thinking on" : " Thinking off";
  elements.tokenChip.textContent = `${state.settings.maxTokens} tokens`;
}

function toggleThinking() {
  state.settings.thinking = !state.settings.thinking;
  persistSettings();
  renderSettingsSummary();
}

function toggleTheme() {
  const current = document.documentElement.dataset.theme || "dark";
  const theme = current === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = theme;
  localStorage.setItem(THEME_KEY, theme);
}

function initializeTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  const theme = saved || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
  document.documentElement.dataset.theme = theme;
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

function scrollToBottom() {
  elements.chatScroll.scrollTop = elements.chatScroll.scrollHeight;
  state.stickToBottom = true;
}

function handleScroll() {
  const distance = elements.chatScroll.scrollHeight
    - elements.chatScroll.scrollTop
    - elements.chatScroll.clientHeight;
  state.stickToBottom = distance < 100;
}

function openSidebar() {
  elements.body.classList.add("sidebar-visible");
}

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
    if (state.generating) {
      stopGeneration();
    }
    createConversation();
  });
  elements.newChat.addEventListener("click", () => {
    if (state.generating) {
      stopGeneration();
    }
    createConversation();
  });
  elements.conversationSearch.addEventListener("input", debounce(renderConversationList, 150));
  elements.sidebarOpen.addEventListener("click", openSidebar);
  elements.sidebarClose.addEventListener("click", closeSidebar);
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
  elements.openSettings.addEventListener("click", openSettings);
  elements.quickSettings.addEventListener("click", openSettings);
  elements.importChat.addEventListener("click", importConversation);
  elements.thinkingChip.addEventListener("click", toggleThinking);
  elements.saveSettings.addEventListener("click", (event) => {
    event.preventDefault();
    saveSettings();
  });
  elements.resetSettings.addEventListener("click", resetSettings);
  elements.clearChat.addEventListener("click", clearConversation);
  elements.exportChat.addEventListener("click", exportConversation);
  elements.themeToggle.addEventListener("click", toggleTheme);
  elements.modelSelector.addEventListener("change", () => {
    state.model = elements.modelSelector.value;
    setRuntimeStatus(state.health?.busy ? "busy" : "online");
    toast(`Switched to ${state.model}`);
  });
  document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => sendMessage(button.dataset.prompt));
  });
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      createConversation();
    }
    if (event.key === "Escape") {
      closeSidebar();
    }
  });
}

initializeTheme();
bindEvents();
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
    pollRuntime();
  }
});
