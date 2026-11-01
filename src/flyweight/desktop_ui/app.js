// Flyweight Desktop page. Polls /api/state, renders it, and posts the
// user's choices back. Plain DOM, no build step: this ships in the wheel.
"use strict";

const ICONS = {
  home: '<path d="M3 10.5 12 3l9 7.5"/><path d="M5 9.5V20a1 1 0 0 0 1 1h4v-6h4v6h4a1 1 0 0 0 1-1V9.5"/>',
  chat: '<path d="M21 12a8 8 0 0 1-11.6 7.1L4 20.5l1.4-4.8A8 8 0 1 1 21 12Z"/>',
  activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  shield: '<path d="M12 3 4.5 6v5.5c0 4.6 3.1 8.3 7.5 9.5 4.4-1.2 7.5-4.9 7.5-9.5V6L12 3Z"/><path d="m9 12 2 2 4-4"/>',
  sidebar: '<rect x="3" y="4" width="18" height="16" rx="3"/><path d="M9 4v16"/>',
  cube: '<path d="M12 2.8 20 7.2v9.6l-8 4.4-8-4.4V7.2l8-4.4Z"/><path d="m4 7.2 8 4.4 8-4.4M12 11.6v9.6"/>',
  file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5"/>',
  folder: '<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7Z"/>',
  play: '<path d="M7 4.8v14.4a.8.8 0 0 0 1.2.7l11.3-7.2a.8.8 0 0 0 0-1.4L8.2 4.1a.8.8 0 0 0-1.2.7Z" fill="currentColor"/>',
  stop: '<rect x="6" y="6" width="12" height="12" rx="2.5" fill="currentColor"/>',
  external: '<path d="M14 4h6v6M20 4l-9 9"/><path d="M19 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2h4"/>',
  copy: '<rect x="9" y="9" width="11" height="11" rx="2.5"/><path d="M15 9V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7a2 2 0 0 0 2 2h3"/>',
  trash: '<path d="M4 7h16M10 11v6M14 11v6M6 7l1 12a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-12M9 7V4h6v3"/>',
  refresh: '<path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  chevron: '<path d="m6 9 6 6 6-6"/>',
  check: '<path d="m5 12.5 4.5 4.5L19 7.5"/>',
  alert: '<path d="M12 8v5M12 16.5v.01"/><circle cx="12" cy="12" r="9"/>',
  bang: '<path d="M12 6v8M12 18v.01"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  globe: '<circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18"/>',
};

const icon = (name) =>
  `<svg data-icon="${name}" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name] || ""}</svg>`;

function paintIcons(root = document) {
  root.querySelectorAll("svg[data-icon]:not([data-painted])").forEach((svg) => {
    svg.setAttribute("viewBox", "0 0 24 24");
    svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = ICONS[svg.dataset.icon] || "";
    svg.dataset.painted = "1";
  });
}

const $ = (id) => document.getElementById(id);
const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// The per-run token arrives in the fragment (never sent to a server) and is
// kept for reloads in this window only.
const TOKEN = (() => {
  const match = location.hash.match(/t=([\w-]+)/);
  if (match) {
    try { sessionStorage.setItem("fw-token", match[1]); } catch { /* private mode */ }
    history.replaceState(null, "", location.pathname);
    return match[1];
  }
  try { return sessionStorage.getItem("fw-token") || ""; } catch { return ""; }
})();

async function api(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "X-Flyweight-Token": TOKEN, "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json();
}

// ---------------------------------------------------------------------------
// State

let state = null;
let lastSeq = 0;
let view = "home";
let formLoaded = false;
let lastStatus = null;
let logLines = [];
let disconnected = false;

function toast(message, kind = "") {
  const node = $("toast");
  node.textContent = message;
  node.className = `toast ${kind}`;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, kind === "error" ? 6000 : 2600);
}

function greeting() {
  const hour = new Date().getHours();
  if (hour < 5) return "Working late";
  if (hour < 12) return "Good morning";
  if (hour < 18) return "Good afternoon";
  return "Good evening";
}

const minutes = (seconds) => {
  const s = Math.max(0, Math.round(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
};

// ---------------------------------------------------------------------------
// Navigation

function show(name) {
  if (name === "chat" && !(state && state.chatUrl)) {
    // still allowed: the empty state explains what to do
  }
  view = name;
  for (const section of document.querySelectorAll(".view")) {
    section.hidden = section.id !== `view-${name}`;
  }
  for (const item of document.querySelectorAll(".nav-item")) {
    item.classList.toggle("active", item.dataset.view === name);
  }
  document.getElementById("app").classList.toggle("collapsed",
    name === "chat" ? !userExpanded : userCollapsed);
  if (name === "activity") scrollLog(true);
}

let userCollapsed = false;
let userExpanded = false;

// ---------------------------------------------------------------------------
// Rendering

function render() {
  if (!state) return;
  const s = state;
  const settings = s.settings;
  const running = s.status === "running";
  const busy = s.status === "loading" || s.status === "running" || s.status === "stopping";

  // sidebar
  const labels = { idle: "Stopped", loading: "Loading…", running: "Running",
    stopping: "Stopping…", failed: "Couldn't start" };
  $("engine-status").textContent = labels[s.status] || s.status;
  $("engine-dot").className = `status-dot ${s.status}`;
  $("engine-model").textContent = busy || s.status === "failed"
    ? (s.model.name || "—") : (s.model.ok ? s.model.name : "No model chosen");
  $("nav-chat-dot").classList.toggle("on", running);
  $("version").textContent = `Flyweight ${s.version}`;
  const failures = s.doctor.lines.filter((l) => l.status === "fail").length;
  $("nav-system-badge").hidden = failures === 0;
  $("nav-system-badge").textContent = failures;

  // home: greeting
  $("greeting").textContent = running ? "Your model is running"
    : s.status === "loading" ? "Getting your model ready" : greeting();
  $("lede").textContent = running
    ? "Chat with it here, or point any OpenAI- or Anthropic-compatible app at the address below."
    : s.status === "loading" ? "Big models take a few minutes to load. You can watch progress in Activity."
    : s.status === "failed" ? "Something stopped the model from loading. The bar below says what to do."
    : "Choose a model and press Start. Everything runs on this computer.";

  renderModel(s);
  renderRecent(s);
  if (!formLoaded) { loadForm(settings); formLoaded = true; }
  $("row-mmproj").hidden = !(s.model.ok && s.model.mmproj);
  if (s.model.mmprojName) $("mmproj-sub").textContent = `Use ${s.model.mmprojName}, found beside the model.`;
  $("settings").disabled = busy;
  $("locked-note").hidden = !busy;
  for (const node of document.querySelectorAll("[data-browse], #path-input")) node.disabled = busy;
  for (const node of document.querySelectorAll("[data-linux]")) node.hidden = s.platform === "win32";

  renderActionBar(s);
  renderChat(s);
  renderDoctor(s);

  // transitions
  if (lastStatus && lastStatus !== s.status) {
    if (s.status === "running" && settings.open_chat) show("chat");
    if (s.status === "failed" && view === "chat") show("home");
    if (s.status === "idle" && view === "chat") show("home");
  }
  lastStatus = s.status;
}

function renderModel(s) {
  const m = s.model;
  const has = Boolean(s.settings.model);
  $("model-empty").hidden = has && m.ok;
  $("model-chosen").hidden = !(has && m.ok);
  if (m.ok) {
    $("model-name").textContent = m.name;
    const kind = m.kind === "gguf" ? "GGUF" : "Safetensors";
    const size = m.kind === "gguf" ? (m.message.match(/[\d.]+ [GM]iB/) || [""])[0] : "";
    $("model-meta").innerHTML =
      `<span class="chip accent">${kind}</span>` +
      (size ? `<span class="chip">${esc(size)}</span>` : "") +
      (m.mmproj ? `<span class="chip">Vision</span>` : "") +
      `<span class="path" title="${esc(m.folder)}">${esc(m.folder)}</span>`;
  }
  const input = $("path-input");
  if (document.activeElement !== input) input.value = s.settings.model;
  const message = $("model-message");
  const note = m.ok ? (m.kind === "safetensors" || /Split/.test(m.message) ? m.message : "") : m.message;
  message.hidden = !(has && note);
  message.className = `model-message ${m.ok ? "info" : "error"}`;
  message.innerHTML = `${icon(m.ok ? "alert" : "alert")}<span>${esc(note)}</span>`;
}

function renderRecent(s) {
  const items = s.recent.filter((r) => r.path !== s.model.path).slice(0, 5);
  $("recent").hidden = items.length === 0;
  const key = JSON.stringify(items) + (state.status === "idle" || state.status === "failed");
  if (renderRecent.key === key) return;
  renderRecent.key = key;
  $("recent-list").innerHTML = items.map((r) => `
    <div class="recent-item ${r.exists ? "" : "missing"}" role="button" tabindex="0" data-path="${esc(r.path)}">
      <span class="ri-icon">${icon("cube")}</span>
      <span class="ri-text"><div class="ri-name">${esc(r.name)}</div><div class="ri-folder">${esc(r.folder)}</div></span>
      <button class="icon-button ri-remove" data-forget="${esc(r.path)}" title="Remove from recent" aria-label="Remove from recent">${icon("x")}</button>
    </div>`).join("");
}

function renderActionBar(s) {
  const title = $("action-title");
  const sub = $("action-sub");
  const buttons = $("action-buttons");
  $("progress").classList.toggle("on", s.status === "loading" || s.status === "stopping");
  sub.className = "action-sub";
  let html = "";
  let key = s.status;
  if (s.status === "loading") {
    title.textContent = `Loading ${s.model.name || "model"} · ${minutes(s.elapsed)}`;
    sub.classList.add("mono");
    sub.textContent = s.lastLine || "Starting the engine…";
    html = `<button class="button" data-act="stop">Cancel</button>`;
  } else if (s.status === "stopping") {
    title.textContent = "Stopping…";
    sub.textContent = "Freeing memory.";
  } else if (s.status === "running") {
    title.innerHTML = `<i class="status-dot running"></i>Running · loaded in ${Math.round(s.elapsed)} s`;
    sub.innerHTML = `<a data-copy="${esc(s.chatUrl)}" title="Copy">${esc(s.chatUrl)}</a>` +
      (s.lanUrl ? ` · on your network <a data-copy="${esc(s.lanUrl)}" title="Copy">${esc(s.lanUrl)}</a>` : "") +
      ` · API at <a data-copy="${esc(s.chatUrl)}v1" title="Copy">/v1</a>`;
    html = `<button class="button" data-act="stop">${icon("stop")}Stop</button>
            <button class="button primary" data-act="chat">${icon("chat")}Open chat</button>`;
    key += s.chatUrl + s.lanUrl;
  } else if (s.status === "failed") {
    title.innerHTML = `${icon("alert")}Couldn't start the model`;
    sub.classList.add("wrap");
    sub.textContent = s.hint;
    html = `<button class="button ghost" data-act="activity">View activity</button>
            <button class="button primary" data-act="start">${icon("refresh")}Try again</button>`;
    key += s.hint;
  } else {
    const ready = s.model.ok;
    const device = { auto: "automatic device", cuda: "NVIDIA GPU", rocm: "AMD GPU", cpu: "CPU only" }[s.settings.backend];
    title.textContent = ready ? `Ready to start ${s.model.name}` : "Choose a model to begin";
    sub.textContent = ready
      ? `${device} · ${s.settings.context ? `${s.settings.context / 1024}K context` : "automatic context"}${s.settings.share_on_network ? " · shared on your network" : ""}`
      : "Pick a .gguf file or a safetensors folder above.";
    html = `<button class="button primary large" data-act="start" ${ready ? "" : "disabled"}>${icon("play")}Start</button>`;
    key += ready + sub.textContent;
  }
  if (renderActionBar.key !== key) {
    renderActionBar.key = key;
    buttons.innerHTML = html;
  }
}

function renderChat(s) {
  const frame = $("chat-frame");
  const running = s.status === "running" && s.chatUrl;
  $("chat-empty").hidden = Boolean(running);
  $("chat-bar").hidden = !running;
  frame.hidden = !running || !s.framed;
  if (running) {
    $("chat-model").textContent = s.model.name;
    $("chat-url").textContent = s.chatUrl;
    if (s.framed && frame.dataset.src !== s.chatUrl) {
      frame.dataset.src = s.chatUrl;
      frame.src = s.chatUrl;
    }
  } else if (frame.dataset.src) {
    frame.dataset.src = "";
    frame.removeAttribute("src");
  }
}

const CHECK_NAMES = {
  python: "Python", jinja2: "Jinja2", package: "Package", command: "Command",
  "native runtime": "Native runtime", sources: "Sources", cmake: "CMake",
  "c++ compiler": "C++ compiler", "build tool": "Build tool", "nvidia gpu": "NVIDIA GPU",
  "amd gpu": "AMD GPU", "cuda headers": "CUDA headers", check: "System check",
};
const checkName = (name) => CHECK_NAMES[name.toLowerCase()] || name.charAt(0).toUpperCase() + name.slice(1);

function renderDoctor(s) {
  const d = s.doctor;
  const summary = $("doctor-summary");
  const failures = d.lines.filter((l) => l.status === "fail").length;
  $("doctor-run").disabled = d.running;
  $("doctor-run").innerHTML = `${icon("refresh")}${d.running ? "Checking…" : d.ran ? "Run again" : "Run check"}`;
  if (d.running) {
    summary.className = "summary pending";
    summary.innerHTML = `${icon("refresh")}Checking this computer…`;
  } else if (!d.ran) {
    summary.className = "summary pending";
    summary.innerHTML = `${icon("shield")}Run the check to see whether this computer is ready.`;
  } else if (failures) {
    summary.className = "summary fail";
    summary.innerHTML = `${icon("alert")}${failures} problem${failures > 1 ? "s" : ""} to fix before models can run.`;
  } else {
    summary.className = "summary ok";
    summary.innerHTML = `${icon("check")}This computer can run Flyweight.`;
  }
  const key = JSON.stringify(d.lines);
  if (renderDoctor.key === key) return;
  renderDoctor.key = key;
  const order = { fail: 0, warn: 1, ok: 2 };
  const lines = [...d.lines].sort((a, b) => order[a.status] - order[b.status]);
  $("doctor-list").innerHTML = lines.map((l) => `
    <div class="check">
      <div class="check-icon ${l.status}">${icon(l.status === "ok" ? "check" : l.status === "warn" ? "bang" : "x")}</div>
      <div class="check-body">
        <div class="check-name">${esc(checkName(l.name))}</div>
        <div class="check-detail">${esc(l.detail)}</div>
        ${l.fix ? `<div class="check-fix"><span>${esc(l.fix)}</span><button class="icon-button" data-copy="${esc(l.fix)}" title="Copy" aria-label="Copy">${icon("copy")}</button></div>` : ""}
      </div>
    </div>`).join("");
}

function appendLogs(entries) {
  if (!entries.length) return;
  const log = $("log");
  const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 40;
  const fragment = document.createDocumentFragment();
  for (const entry of entries) {
    logLines.push(entry.text);
    const line = document.createElement("div");
    if (entry.kind) line.className = `l-${entry.kind}`;
    line.textContent = entry.text || " ";
    fragment.appendChild(line);
  }
  log.appendChild(fragment);
  while (log.childElementCount > 4000) log.firstElementChild.remove();
  logLines = logLines.slice(-4000);
  if (atBottom) scrollLog(true);
}

function scrollLog(force) {
  const log = $("log");
  if (force) log.scrollTop = log.scrollHeight;
}

// ---------------------------------------------------------------------------
// Settings form

function loadForm(settings) {
  for (const group of document.querySelectorAll(".segmented[data-setting]")) {
    const value = String(settings[group.dataset.setting]);
    for (const button of group.querySelectorAll("button")) {
      button.classList.toggle("on", button.dataset.value === value);
    }
  }
  for (const input of document.querySelectorAll("input[data-setting], select[data-setting]")) {
    const value = settings[input.dataset.setting];
    if (input.type === "checkbox") input.checked = Boolean(value);
    else input.value = value ?? "";
  }
}

let pending = {};
let pendingTimer = null;
function saveSetting(name, value, delay = 0) {
  pending[name] = value;
  clearTimeout(pendingTimer);
  pendingTimer = setTimeout(async () => {
    const changes = pending;
    pending = {};
    try {
      const fresh = await api("/api/settings", changes);
      state = { ...state, settings: fresh.settings, model: fresh.model, recent: fresh.recent };
      render();
    } catch (error) { toast(`Could not save: ${error.message}`, "error"); }
  }, delay);
}

// ---------------------------------------------------------------------------
// Actions

async function act(name) {
  try {
    if (name === "start") {
      const result = await api("/api/start", {});
      if (!result.ok) toast(result.error, "error");
      await poll();
    } else if (name === "stop") {
      await api("/api/stop", {});
      await poll();
    } else if (name === "chat") {
      show("chat");
      if (state && !state.framed) await api("/api/open", { url: state.chatUrl });
    } else if (name === "activity") {
      show("activity");
    }
  } catch (error) { toast(error.message, "error"); }
}

async function browse(kind) {
  try {
    const result = await api("/api/browse", { kind });
    if (!result.ok) toast(result.error, "error");
    await poll();
  } catch (error) { toast(error.message, "error"); }
}

async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied"); }
  catch {
    const area = document.createElement("textarea");
    area.value = text; document.body.appendChild(area); area.select();
    document.execCommand("copy"); area.remove(); toast("Copied");
  }
}

document.addEventListener("click", (event) => {
  const target = event.target.closest("button, [data-copy], .recent-item, [data-goto]");
  if (!target) return;
  if (target.dataset.view) return show(target.dataset.view);
  if (target.dataset.goto) return show(target.dataset.goto);
  if (target.dataset.act) return act(target.dataset.act);
  if (target.dataset.browse) return browse(target.dataset.browse);
  if (target.dataset.copy) return copy(target.dataset.copy);
  if (target.dataset.forget) {
    event.stopPropagation();
    return api("/api/forget", { path: target.dataset.forget }).then(poll);
  }
  if (target.classList.contains("recent-item")) {
    if (state && (state.status === "idle" || state.status === "failed")) saveSetting("model", target.dataset.path);
    return;
  }
  if (target.parentElement && target.parentElement.classList.contains("segmented")) {
    const group = target.parentElement;
    for (const b of group.querySelectorAll("button")) b.classList.toggle("on", b === target);
    const raw = target.dataset.value;
    return saveSetting(group.dataset.setting, "number" in group.dataset ? Number(raw) : raw);
  }
});

document.addEventListener("keydown", (event) => {
  if (event.target.classList && event.target.classList.contains("recent-item") && event.key === "Enter") {
    event.target.click();
  }
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && state) {
    if (state.status === "idle" || state.status === "failed") act("start");
  }
});

document.addEventListener("change", (event) => {
  const input = event.target;
  if (!input.dataset || !input.dataset.setting) return;
  if (input.type === "checkbox") return saveSetting(input.dataset.setting, input.checked);
  const value = "number" in input.dataset ? Number(input.value || 0) : input.value;
  saveSetting(input.dataset.setting, value);
});
document.addEventListener("input", (event) => {
  const input = event.target;
  if (input.id === "path-input") return saveSetting("model", input.value.trim().replace(/^["']|["']$/g, ""), 350);
  if (input.dataset && input.dataset.setting && input.type !== "checkbox" && input.tagName === "INPUT") {
    const value = "number" in input.dataset ? Number(input.value || 0) : input.value;
    saveSetting(input.dataset.setting, value, 500);
  }
});

$("collapse").addEventListener("click", () => {
  const app = $("app");
  const collapsed = !app.classList.contains("collapsed");
  app.classList.toggle("collapsed", collapsed);
  if (view === "chat") userExpanded = !collapsed; else userCollapsed = collapsed;
});
$("chat-external").addEventListener("click", () => state && api("/api/open", { url: state.chatUrl }));
$("log-copy").addEventListener("click", () => copy(logLines.join("\n")));
$("log-clear").addEventListener("click", () => { $("log").innerHTML = ""; logLines = []; });
$("doctor-run").addEventListener("click", () => api("/api/doctor", {}).then(poll));
$("shortcut").addEventListener("click", async () => {
  const result = await api("/api/shortcut", {});
  if (result.ok) {
    toast("Added to the app menu");
    $("shortcut-sub").textContent = `Added: ${result.paths.join(", ")}`;
  } else toast(result.error, "error");
});

// ---------------------------------------------------------------------------
// Polling

async function poll() {
  try {
    const fresh = await api(`/api/state?since=${lastSeq}`);
    if (disconnected) { disconnected = false; toast("Reconnected"); }
    appendLogs(fresh.logs);
    lastSeq = fresh.lastSeq;
    state = fresh;
    render();
  } catch {
    if (!disconnected) { disconnected = true; toast("Lost contact with Flyweight. Is it still running?", "error"); }
  }
}

async function loop() {
  await poll();
  const busy = state && (state.status === "loading" || state.status === "stopping" || state.doctor.running);
  setTimeout(loop, busy ? 500 : 1000);
}

paintIcons();
show("home");
loop().then(() => {
  if (state && state.doctor.running) show("system");
});
