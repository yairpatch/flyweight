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
  health: null,
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

function persistConversations() {
  const compact = state.conversations.slice(0, 100);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
  } catch {
    toast("Conversation history could not be saved.", "error");
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
    button.addEventListener("click", () => selectConversation(conversation.id));

    const icon = svgIcon("message");
    const name = document.createElement("span");
    name.className = "conversation-name";
    name.textContent = conversation.title;
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
    error.textContent = message.error;
    content.append(error);
  }
  for (const toolCall of message.toolCalls || []) {
    content.append(renderToolCall(toolCall));
  }
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
    renderMessageContent(content, message);
    return;
  }
  for (const child of [...content.childNodes]) {
    if (child !== cursor) {
      child.remove();
    }
  }
  const text = document.createDocumentFragment();
  renderRichText(text, message.content || "");
  content.insertBefore(text, cursor);
  for (const toolCall of message.toolCalls || []) {
    content.append(renderToolCall(toolCall));
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
  const fencePattern = /```([^\n`]*)\n?([\s\S]*?)```/g;
  let cursor = 0;
  let match;
  while ((match = fencePattern.exec(text)) !== null) {
    appendTextBlock(container, text.slice(cursor, match.index));
    appendCodeBlock(container, match[1].trim(), match[2].replace(/\n$/, ""));
    cursor = fencePattern.lastIndex;
  }
  const remainder = text.slice(cursor);
  const open = remainder.match(/```([^\n`]*)\n([\s\S]*)$/);
  if (open) {
    appendTextBlock(container, remainder.slice(0, open.index));
    appendCodeBlock(container, open[1].trim(), open[2].replace(/\n$/, ""));
    return;
  }
  appendTextBlock(container, remainder);
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
  const copy = document.createElement("button");
  copy.type = "button";
  copy.className = "copy-code";
  copy.textContent = "Copy";
  copy.addEventListener("click", () => copyText(code));
  actions.append(copy);
  header.append(label, actions);
  const codeElement = document.createElement("code");
  codeElement.textContent = code;
  pre.append(header, codeElement);
  container.append(pre);
}

function togglePreview(pre, button, language, code) {
  const existing = pre.nextElementSibling;
  if (existing?.classList.contains("code-preview")) {
    existing.remove();
    button.textContent = "Run";
    return;
  }
  const preview = document.createElement("div");
  preview.className = "code-preview";
  const frame = document.createElement("iframe");
  frame.className = "code-preview-frame";
  frame.setAttribute("sandbox", "allow-scripts allow-modals");
  frame.setAttribute("title", "Code preview");
  frame.src = `preview.html#${encodeURIComponent(previewDocument(language, code))}`;
  preview.append(frame);
  pre.after(preview);
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

function appendTextBlock(container, text) {
  if (!text) {
    return;
  }
  const paragraphs = text.split(/\n{2,}/);
  for (const paragraphText of paragraphs) {
    if (!paragraphText) {
      continue;
    }
    const paragraph = document.createElement("p");
    const lines = paragraphText.split("\n");
    lines.forEach((line, index) => {
      appendInlineCode(paragraph, line);
      if (index + 1 < lines.length) {
        paragraph.append(document.createElement("br"));
      }
    });
    container.append(paragraph);
  }
}

function appendInlineCode(parent, text) {
  const pattern = /`([^`]+)`/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    parent.append(document.createTextNode(text.slice(cursor, match.index)));
    const code = document.createElement("code");
    code.textContent = match[1];
    parent.append(code);
    cursor = pattern.lastIndex;
  }
  parent.append(document.createTextNode(text.slice(cursor)));
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

  const started = performance.now();
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
        const seconds = (performance.now() - started) / 1000;
        assistant.metrics = formatGenerationMetrics(generatedTokens, seconds, true);
      }
      const delta = chunk.choices?.[0]?.delta;
      if (!delta) {
        return;
      }
      if (typeof delta.content === "string") {
        assistant.content += delta.content;
      }
      mergeToolCalls(assistant, delta.tool_calls);
      updateStreamingMessage(assistant);
      if (state.stickToBottom) {
        scrollToBottom();
      }
    });
    const seconds = (performance.now() - started) / 1000;
    const outputTokens = usage?.completion_tokens;
    assistant.metrics = outputTokens
      ? formatGenerationMetrics(outputTokens, seconds, false)
      : `${seconds.toFixed(1)}s`;
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

function formatGenerationMetrics(tokens, seconds, live) {
  const rate = seconds > 0 ? (tokens / seconds).toFixed(1) : "0.0";
  return `${tokens} tokens · ${rate} tok/s${live ? " · live" : ""}`;
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

async function readSSE(stream, onData) {
  const reader = stream.getReader();
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

async function pollRuntime() {
  try {
    const response = await fetch("/health", { headers: authHeaders() });
    if (!response.ok) {
      if (response.status === 401) {
        setRuntimeStatus("locked");
        return;
      }
      throw new Error(String(response.status));
    }
    state.health = await response.json();
    const [modelResponse, propertiesResponse] = await Promise.all([
      state.model ? null : fetch("/v1/models", { headers: authHeaders() }),
      fetch("/props", { headers: authHeaders() }),
    ]);
    if (modelResponse?.ok) {
      const models = await modelResponse.json();
      state.model = models.data?.[0]?.id || null;
    }
    if (propertiesResponse.ok) {
      const properties = await propertiesResponse.json();
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
  elements.deviceLabel.textContent = execution.device === "cuda" ? "CUDA" : "CPU";
  elements.cacheLabel.textContent = execution.device === "cuda"
    ? `${execution.cache_used_mib || 0} / ${execution.cache_limit_mib || 0} MB`
    : "Portable";
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
  elements.conversationSearch.addEventListener("input", renderConversationList);
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
  elements.thinkingChip.addEventListener("click", toggleThinking);
  elements.saveSettings.addEventListener("click", (event) => {
    event.preventDefault();
    saveSettings();
  });
  elements.resetSettings.addEventListener("click", resetSettings);
  elements.clearChat.addEventListener("click", clearConversation);
  elements.exportChat.addEventListener("click", exportConversation);
  elements.themeToggle.addEventListener("click", toggleTheme);
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
setInterval(pollRuntime, 5000);
