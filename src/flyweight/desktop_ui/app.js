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
  gpu: '<rect x="3" y="6" width="18" height="12" rx="2"/><circle cx="15" cy="12" r="2.5"/><path d="M7 10h3M7 14h3M6 18v2M10 18v2"/>',
  download: '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/>',
  sparkle: '<path d="M12 3l1.8 4.6L18.5 9.5l-4.7 1.9L12 16l-1.8-4.6L5.5 9.5l4.7-1.9L12 3Z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15Z"/>',
  edit: '<path d="M4 20h4L19 9l-4-4L4 16v4Z"/><path d="m13.5 6.5 4 4"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  pause: '<rect x="6.5" y="5" width="3.5" height="14" rx="1" fill="currentColor"/><rect x="14" y="5" width="3.5" height="14" rx="1" fill="currentColor"/>',
  memory: '<rect x="3" y="7" width="18" height="10" rx="2"/><path d="M7 7v10M11 7v10M15 7v10M3 12h18"/>',
  disk: '<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"/><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"/>',
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
  renderGpu(s);
  renderModels(s);
  if (s.notice) toast(s.notice);
  $("update").hidden = !s.update;
  if (s.update) $("update-text").textContent = `Update to ${s.update.version}`;

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
  $("chat-unframed").hidden = !running || s.framed;
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

// What the GPU needs, and the one button that provides it.
function gpuView(s) {
  const g = s.gpu;
  const job = s.gpuSetup;
  if (!g) return null;
  if (job.running) {
    const percent = Math.round(job.fraction * 100);
    return { kind: "", title: "Setting up GPU support…", detail: "Downloading NVIDIA's compiler and headers. You can keep using the app.",
      extra: `<div class="meter"><i style="width:${percent}%"></i></div><div class="meter-label">${esc(job.message)}</div>`, buttons: "" };
  }
  if (job.error) {
    return { kind: "error", title: "GPU setup did not finish", detail: job.error,
      buttons: `<button class="button primary" data-gpu="setup">${icon("refresh")}Try again</button>` };
  }
  if (g.state === "needs-setup") {
    return { kind: "", title: g.title, detail: `${g.detail} About ${g.download_mb} MB, once.`,
      buttons: `<button class="button primary" data-gpu="setup">${icon("download")}Set up GPU</button>` };
  }
  if (g.state === "needs-driver" || g.state === "old-driver") {
    return { kind: "warn", title: g.title, detail: `${g.detail} Install it from NVIDIA, restart if asked, then press "Check again".`,
      buttons: `<button class="button" data-gpu="refresh">${icon("refresh")}Check again</button>
                <button class="button primary" data-gpu="driver">${icon("external")}Get the driver</button>` };
  }
  return null;
}

function renderGpu(s) {
  const view = gpuView(s);
  const notice = $("gpu-notice");
  const key = JSON.stringify([view, s.status]);
  if (renderGpu.key === key) return;
  renderGpu.key = key;
  const busy = s.status !== "idle" && s.status !== "failed";
  // Home shows it only while something needs doing.
  notice.hidden = !view;
  if (view) {
    notice.className = `notice ${view.kind}`;
    notice.innerHTML = `<div class="notice-icon">${icon("gpu")}</div>
      <div class="notice-text"><div class="notice-title">${esc(view.title)}</div>
      <div class="notice-detail">${esc(view.detail)}</div>${view.extra || ""}</div>
      <div class="button-row">${view.buttons}</div>`;
  }
  const g = s.gpu || { title: "Checking graphics…", detail: "", state: "none" };
  const card = view || { kind: g.state === "ready" ? "" : "", title: g.title, detail: g.detail, buttons:
    `<button class="button ghost small" data-gpu="refresh">${icon("refresh")}Check again</button>` };
  $("gpu-card").className = `card gpu-card ${view ? view.kind : g.state === "ready" ? "ok" : ""}`;
  $("gpu-card").innerHTML = `<div class="notice-icon">${icon(g.state === "ready" ? "check" : "gpu")}</div>
      <div class="notice-text"><div class="notice-title">${esc(card.title)}</div>
      <div class="notice-detail">${esc(card.detail)}</div>${card.extra || ""}</div>
      <div class="button-row">${card.buttons}</div>`;
}

// ---------------------------------------------------------------------------
// Get a model

const gb = (bytes) => bytes >= 2 ** 30 ? `${(bytes / 2 ** 30).toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 2 ** 20))} MB`;
const LEVEL_ORDER = { great: 0, good: 1, slow: 2, no: 3 };
let searchResults = null;     // null: no search yet
let searchBusy = false;
let searchError = "";
const repoDetails = {};       // repo -> details | {loading} | {error}
const openRepos = new Set();

function setHtml(id, html) {
  const node = $(id);
  if (node.dataset.html === html) return;
  node.dataset.html = html;
  node.innerHTML = html;
}

function eta(job) {
  if (!job.speed) return "";
  const seconds = (job.total - job.done) / job.speed;
  if (seconds < 60) return "less than a minute left";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min left`;
  return `${(seconds / 3600).toFixed(1)} h left`;
}

function jobFor(s, repo) {
  return s.models.downloads.find((j) => j.repo === repo && j.state !== "done");
}

function downloadedPath(s, entry) {
  if (!entry.files) return null;
  const file = entry.files[0].path.split("/").pop();
  const hit = s.models.library.find((m) => m.repo === entry.repo && m.file === file);
  return hit ? hit.path : null;
}

function renderModels(s) {
  const m = s.models;
  const active = m.downloads.filter((j) => ["queued", "downloading", "verifying"].includes(j.state));
  $("nav-models-dot").classList.toggle("on", active.length > 0);
  const locked = s.status !== "idle" && s.status !== "failed";

  // this computer
  const mc = m.machine;
  setHtml("machine", mc ? [
    `<span class="chip">${icon("memory")}${gb(mc.ram)} memory</span>`,
    `<span class="chip">${icon("gpu")}${mc.gpu ? `${esc(mc.gpu)} · ${gb(mc.vram)}` : "No GPU — models run on the CPU"}</span>`,
    `<span class="chip">${icon("disk")}${gb(mc.disk_free)} free</span>`,
  ].join("") : `<span class="chip">Measuring this computer…</span>`);

  // downloads in progress (and just finished)
  setHtml("downloads", m.downloads.map((j) => {
    const pct = j.total ? Math.min(100, Math.round((100 * j.done) / j.total)) : 0;
    let sub = "", buttons = "", kind = "";
    if (j.state === "done") {
      kind = "done";
      const inUse = s.settings.model === j.result;
      sub = inUse ? "Downloaded · selected on Home" : "Downloaded";
      buttons = inUse ? `<button class="button primary small" data-goto="home">${icon("play")}Go to Start</button>`
        : `<button class="button small" data-use="${esc(j.result)}" ${locked ? "disabled" : ""}>Use</button>`;
    } else if (j.state === "failed") {
      kind = "failed";
      sub = j.error;
      buttons = `<button class="button ghost small" data-job="cancel" data-id="${j.id}">Remove</button>
                 <button class="button primary small" data-job="resume" data-id="${j.id}">${icon("refresh")}Resume</button>`;
    } else if (j.state === "paused") {
      sub = `Paused · ${gb(j.done)} of ${gb(j.total)}`;
      buttons = `<button class="button ghost small" data-job="cancel" data-id="${j.id}">Cancel</button>
                 <button class="button primary small" data-job="resume" data-id="${j.id}">${icon("play")}Resume</button>`;
    } else {
      sub = j.state === "queued" ? "Waiting…" : j.state === "verifying" ? "Checking the file…"
        : `${gb(j.done)} of ${gb(j.total)}${j.speed ? ` · ${gb(j.speed)}/s · ${eta(j)}` : ""}`;
      buttons = `<button class="button ghost small" data-job="cancel" data-id="${j.id}">Cancel</button>
                 <button class="button small" data-job="pause" data-id="${j.id}">${icon("pause")}Pause</button>`;
    }
    const meter = j.state === "done" || j.state === "failed" ? ""
      : `<div class="meter"><i style="width:${pct}%"></i></div>`;
    return `<div class="dl ${kind}">
      <div class="notice-icon">${icon(kind === "done" ? "check" : kind === "failed" ? "alert" : "download")}</div>
      <div class="dl-body"><div class="dl-title">${esc(j.name)}</div>${meter}<div class="dl-sub">${esc(sub)}</div></div>
      <div class="button-row">${buttons}</div></div>`;
  }).join(""));

  // the catalog
  if (m.loading && !m.catalog.length) {
    setHtml("catalog", `<div class="skeleton"></div><div class="skeleton"></div><div class="skeleton"></div>`);
  } else if (!m.catalog.length) {
    setHtml("catalog", `<div class="card empty-row">Could not load the list of models. Check the internet connection, then
      <button class="button ghost small" data-models="refresh">${icon("refresh")}Try again</button></div>`);
  } else {
    const entries = [...m.catalog].sort((a, b) =>
      (a.available === b.available ? 0 : a.available ? -1 : 1)
      || LEVEL_ORDER[(a.fit || {}).level || "no"] - LEVEL_ORDER[(b.fit || {}).level || "no"]
      || (b.rank || 0) - (a.rank || 0));
    const fitting = entries.filter((e) => e.available && e.fit && (e.fit.level === "great" || e.fit.level === "good"));
    const best = fitting.length ? fitting.reduce((x, y) => ((y.rank || 0) > (x.rank || 0) ? y : x)) : null;
    if (best) entries.splice(entries.indexOf(best), 1), entries.unshift(best);
    setHtml("catalog", entries.map((e) => {
      const job = e.available && jobFor(s, e.repo);
      const have = e.available && downloadedPath(s, e);
      const tags = (e.tags || []).map((t) => `<span class="chip">${esc(t)}</span>`).join("");
      let foot;
      if (!e.available) {
        foot = `<span class="size">Unavailable: ${esc(e.error || "")}</span>`;
      } else if (job) {
        const pct = job.total ? Math.round((100 * job.done) / job.total) : 0;
        foot = `<div style="flex:1"><div class="meter"><i style="width:${pct}%"></i></div>
                <div class="meter-label">${job.state === "paused" ? "Paused" : `${pct}%`}</div></div>`;
      } else if (have) {
        const inUse = s.settings.model === have;
        foot = `<span class="size">${gb(e.size)} · downloaded</span>` + (inUse
          ? `<span class="chip accent">${icon("check")}Selected</span>`
          : `<button class="button small" data-use="${esc(have)}" ${locked ? "disabled" : ""}>Use</button>`);
      } else {
        foot = `<span class="size">${gb(e.size)} · ${esc(e.quant || "")}</span>
                <button class="button ${e === best ? "primary" : ""} small" data-download="${esc(e.id)}" ${e.fit.level === "no" ? "disabled" : ""}>${icon("download")}Download</button>`;
      }
      const fitBadge = e.available && e.fit ? `<span class="fit ${e.fit.level}">${esc(e.fit.label)}</span>` : "";
      return `<div class="mcard ${e === best ? "best" : ""} ${!e.available || (e.fit && e.fit.level === "no") ? "dim" : ""}">
        ${e === best ? `<span class="ribbon">${icon("sparkle")}Best match</span>` : ""}
        <div class="mcard-head"><div class="mcard-name">${esc(e.name)}</div>${fitBadge}</div>
        <div class="mcard-desc">${esc(e.description || "")}</div>
        <div class="chips">${tags}</div>
        <div class="mcard-foot">${foot}</div></div>`;
    }).join(""));
  }

  // the library
  setHtml("library", m.library.length ? m.library.map((x) => {
    const inUse = s.settings.model === x.path;
    return `<div class="lib">
      <div class="model-tile" style="width:38px;height:38px;border-radius:10px">${icon("cube")}</div>
      <div class="lib-text"><div class="lib-name">${esc(x.name)}</div>
        <div class="lib-sub">${esc(x.file)} · ${gb(x.size)}</div></div>
      ${inUse ? `<span class="chip accent">Selected</span>` : `<button class="button small" data-use="${esc(x.path)}" ${locked ? "disabled" : ""}>Use</button>`}
      <button class="icon-button" data-reveal="${esc(x.path)}" title="Show in folder" aria-label="Show in folder">${icon("folder")}</button>
      <button class="icon-button" data-delete="${esc(x.folder)}" data-name="${esc(x.name)}" title="Delete" aria-label="Delete">${icon("trash")}</button>
    </div>`;
  }).join("") : `<div class="empty-row">Models you download appear here.</div>`);
  setHtml("models-dir", `Saved in ${esc(m.dir)}`);

  renderSearch(s);
}

function renderSearch(s) {
  if (searchBusy) return setHtml("search-results", `<div class="card empty-row">Searching…</div>`);
  if (searchError) return setHtml("search-results", `<div class="card empty-row">${esc(searchError)}</div>`);
  if (searchResults === null) return setHtml("search-results", "");
  if (!searchResults.length) return setHtml("search-results", `<div class="card empty-row">No GGUF models matched.</div>`);
  const support = { yes: "Supported", no: "Not supported", unknown: "Unknown" };
  setHtml("search-results", `<div class="card results">${searchResults.map((r) => {
    const open = openRepos.has(r.repo);
    const d = repoDetails[r.repo];
    let body = "";
    if (open) {
      if (!d || d.loading) body = `<div class="result-note">Loading files…</div>`;
      else if (d.error) body = `<div class="result-note">${esc(d.error)}</div>`;
      else {
        const sup = d.supported;
        const note = d.gated ? "This model needs a Hugging Face login, which Flyweight does not do yet."
          : sup === "no" ? `Flyweight cannot run the ${esc(d.architecture)} architecture.`
          : sup === "unknown" ? "Hugging Face does not say which architecture this is; it may not load." : "";
        body = (note ? `<div class="result-note">${note}</div>` : "") + `<div class="variants">${d.variants.map((v, i) => {
          const files = v.files.concat(d.projector ? [d.projector] : []);
          const job = s.models.downloads.find((j) => j.repo === d.repo && j.files[0].path === files[0].path && j.state !== "done");
          const blocked = d.gated || sup === "no" || v.fit.level === "no";
          return `<div class="variant"><span class="q">${esc(v.quant)}</span><span class="s">${gb(v.size)}</span>
            <span class="fit ${v.fit.level}">${esc(v.fit.label)}</span>
            ${job ? `<span class="s">${Math.round((100 * job.done) / (job.total || 1))}%</span>`
              : `<button class="button small" data-hub="${esc(d.repo)}" data-variant="${i}" ${blocked ? "disabled" : ""}>${icon("download")}Download</button>`}</div>`;
        }).join("") || `<div class="result-note">No complete GGUF files in this repository.</div>`}</div>`;
      }
    }
    return `<div class="result"><div class="result-head" data-repo="${esc(r.repo)}">
      <span class="result-name">${esc(r.repo)}</span>
      <span class="result-meta">${r.downloads.toLocaleString()} downloads</span>
      <span class="support ${r.supported}">${support[r.supported] || r.supported}</span>
      ${icon("chevron")}</div>${body}</div>`;
  }).join("")}</div>`);
}

async function searchHub(query) {
  searchBusy = true; searchError = ""; render();
  try {
    const result = await api("/api/models/search", { query });
    if (result.ok) searchResults = result.results; else searchError = result.error;
  } catch (error) { searchError = error.message; }
  searchBusy = false; render();
}

async function toggleRepo(repo) {
  if (openRepos.has(repo)) { openRepos.delete(repo); return render(); }
  openRepos.add(repo);
  if (!repoDetails[repo] || repoDetails[repo].error) {
    repoDetails[repo] = { loading: true }; render();
    try {
      const result = await api("/api/models/repo", { repo });
      repoDetails[repo] = result.ok ? result.repo : { error: result.error };
    } catch (error) { repoDetails[repo] = { error: error.message }; }
  }
  render();
}

async function modelsCall(verb, body = {}) {
  try {
    const result = await api(`/api/models/${verb}`, body);
    if (result && result.ok === false) toast(result.error, "error");
    await poll();
    return result;
  } catch (error) { toast(error.message, "error"); return null; }
}

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
  const target = event.target.closest("button, [data-copy], .recent-item, [data-goto], [data-repo]");
  if (!target) return;
  if (target.dataset.view) return show(target.dataset.view);
  if (target.dataset.goto) return show(target.dataset.goto);
  if (target.dataset.act) return act(target.dataset.act);
  if (target.dataset.browse) return browse(target.dataset.browse);
  if (target.dataset.copy) return copy(target.dataset.copy);
  if (target.dataset.gpu) {
    const what = target.dataset.gpu;
    if (what === "driver") return api("/api/open", { url: state.driverUrl });
    target.disabled = true;
    return api(`/api/gpu/${what}`, {}).then((r) => { if (r && r.ok === false) toast(r.error, "error"); return poll(); });
  }
  if (target.id === "update") {
    return state && state.update && api("/api/open", { url: state.update.url });
  }
  if (target.dataset.models) return modelsCall(target.dataset.models);
  if (target.dataset.download) return modelsCall("download", { id: target.dataset.download });
  if (target.dataset.use) return modelsCall("use", { path: target.dataset.use }).then(() => toast("Selected. Press Start on Home."));
  if (target.dataset.reveal) return modelsCall("reveal", { path: target.dataset.reveal });
  if (target.dataset.job) return modelsCall(target.dataset.job, { job: target.dataset.id });
  if (target.dataset.delete) {
    if (!confirm(`Delete ${target.dataset.name} from this computer?`)) return;
    return modelsCall("delete", { folder: target.dataset.delete });
  }
  if (target.dataset.hub) {
    const d = repoDetails[target.dataset.hub];
    const v = d.variants[Number(target.dataset.variant)];
    const files = v.files.concat(d.projector ? [d.projector] : []);
    return modelsCall("download", { repo: d.repo, name: `${d.repo.split("/").pop()} ${v.quant}`, files });
  }
  if (target.dataset.repo) return toggleRepo(target.dataset.repo);
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

$("search-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const query = $("search-input").value.trim();
  if (query) searchHub(query);
});

$("collapse").addEventListener("click", () => {
  const app = $("app");
  const collapsed = !app.classList.contains("collapsed");
  app.classList.toggle("collapsed", collapsed);
  if (view === "chat") userExpanded = !collapsed; else userCollapsed = collapsed;
});
$("chat-external").addEventListener("click", () => state && api("/api/open", { url: state.chatUrl }));
$("chat-open-browser").addEventListener("click", () => state && api("/api/open", { url: state.chatUrl }));
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
  const busy = state && (state.status === "loading" || state.status === "stopping"
    || state.doctor.running || state.gpuSetup.running || !state.gpu || state.models.loading
    || state.models.downloads.some((j) => j.state === "downloading" || j.state === "verifying"));
  setTimeout(loop, busy ? 500 : 1000);
}

paintIcons();
show("home");
loop().then(() => {
  if (state && state.doctor.running) show("system");
});
