import { useEffect, useRef, useState } from "react";
import {
  Activity, Box, ChevronDown, Cpu, FolderOpen, Gauge, MessageSquare, Play, Server, ShieldCheck,
  Sparkles, Square, Wrench, Download, FileBox, Check,
} from "lucide-react";
import { useDesk, gb, clock, type Page } from "./state";
import { useStore } from "../store";
import { setApiBase, setApiKey } from "../lib/api";
import { ChatPage } from "./pages/ChatPage";
import { ServerPage } from "./pages/ServerPage";
import { MonitorPage } from "./pages/MonitorPage";
import { ToolsPage } from "./pages/ToolsPage";
import { ModelsPage } from "./pages/ModelsPage";
import { ActivityPage } from "./pages/ActivityPage";
import { SystemPage } from "./pages/SystemPage";

const NAV: Array<{ page: Page; label: string; icon: typeof MessageSquare }> = [
  { page: "chat", label: "Chat", icon: MessageSquare },
  { page: "models", label: "Models", icon: Sparkles },
  { page: "server", label: "Server", icon: Server },
  { page: "monitor", label: "Monitor", icon: Gauge },
  { page: "tools", label: "Tools", icon: Wrench },
  { page: "activity", label: "Activity", icon: Activity },
  { page: "system", label: "System", icon: ShieldCheck },
];

/** Point the chat store at the engine while it runs; keep its view fresh. */
function useEngineBridge() {
  const status = useDesk((s) => s.snapshot?.status);
  const port = useDesk((s) => s.snapshot?.port);
  const key = useDesk((s) => s.snapshot?.engineKey ?? "");
  const inited = useRef(false);

  useEffect(() => {
    if (!inited.current) {
      inited.current = true;
      setApiBase(""); // this app's own origin answers 404 until an engine runs
      void useStore.getState().init();
    }
  }, []);

  useEffect(() => {
    if (status === "running" && port) {
      setApiBase(`http://127.0.0.1:${port}`);
      setApiKey(key);
      void useStore.getState().pollRuntime();
      const timer = window.setInterval(() => void useStore.getState().pollRuntime(), 4000);
      return () => window.clearInterval(timer);
    }
    setApiBase("");
    return undefined;
  }, [status, port, key]);
}

function ModelMenu() {
  const snapshot = useDesk((s) => s.snapshot);
  const call = useDesk((s) => s.call);
  const setPage = useDesk((s) => s.setPage);
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (event: MouseEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  if (!snapshot) return null;
  const busy = snapshot.status !== "idle" && snapshot.status !== "failed";
  const model = snapshot.model;
  const library = snapshot.models.library;
  const recent = snapshot.recent.filter((r) => r.exists && !library.some((m) => m.path === r.path));
  const pick = (path: string) => {
    setOpen(false);
    void call("/api/settings", { model: path });
  };
  return (
    <div className="d-modelmenu" ref={ref}>
      <button className="d-modelmenu__button" disabled={busy} onClick={() => setOpen(!open)}
              title={busy ? "Stop the model to switch" : "Choose the model to run"}>
        <FileBox size={16} />
        <span className="d-modelmenu__name">{model.ok ? model.name : snapshot.settings.model ? "Model not found" : "Choose a model"}</span>
        {model.ok && <span className="d-pill">{model.kind === "gguf" ? "GGUF" : "Safetensors"}</span>}
        <ChevronDown size={14} />
      </button>
      {open && (
        <div className="d-menu">
          {library.length > 0 && <div className="d-menu__label">Downloaded</div>}
          {library.map((m) => (
            <button key={m.path} className="d-menu__item" onClick={() => pick(m.path)}>
              {snapshot.settings.model === m.path ? <Check size={14} /> : <Box size={14} />}
              <span className="d-menu__text">{m.name}<small>{m.file} · {gb(m.size)}</small></span>
            </button>
          ))}
          {recent.length > 0 && <div className="d-menu__label">Recent</div>}
          {recent.map((r) => (
            <button key={r.path} className="d-menu__item" onClick={() => pick(r.path)}>
              {snapshot.settings.model === r.path ? <Check size={14} /> : <Box size={14} />}
              <span className="d-menu__text">{r.name}<small>{r.folder}</small></span>
            </button>
          ))}
          <div className="d-menu__sep" />
          <button className="d-menu__item" onClick={() => { setOpen(false); void call("/api/browse", { kind: "file" }); }}>
            <FolderOpen size={14} /><span className="d-menu__text">Open a .gguf file…</span>
          </button>
          <button className="d-menu__item" onClick={() => { setOpen(false); void call("/api/browse", { kind: "folder" }); }}>
            <FolderOpen size={14} /><span className="d-menu__text">Open a safetensors folder…</span>
          </button>
          <button className="d-menu__item" onClick={() => { setOpen(false); setPage("models"); }}>
            <Download size={14} /><span className="d-menu__text">Get a model…</span>
          </button>
        </div>
      )}
    </div>
  );
}

function RunBar() {
  const snapshot = useDesk((s) => s.snapshot);
  const call = useDesk((s) => s.call);
  const setPage = useDesk((s) => s.setPage);
  if (!snapshot) return <header className="d-runbar" />;
  const { status } = snapshot;
  const canStart = snapshot.model.ok && (status === "idle" || status === "failed");
  const label = { idle: "Stopped", loading: "Loading", running: "Running", stopping: "Stopping", failed: "Failed" }[status];
  let detail = "";
  if (status === "loading") detail = `${clock(snapshot.elapsed)} · ${snapshot.lastLine}`;
  else if (status === "running") detail = `${snapshot.chatUrl ?? ""}${snapshot.lanUrl ? ` · ${snapshot.lanUrl}` : ""}`;
  else if (status === "failed") detail = snapshot.hint;
  return (
    <header className="d-runbar">
      <ModelMenu />
      <button className={`d-pill d-pill--button${snapshot.ownProfile ? " d-pill--accent" : ""}`} onClick={() => setPage("server")}
              title="The engine options this model starts with">
        <Cpu size={13} />{snapshot.ownProfile ? "Custom settings" : "Default settings"}
      </button>
      <div className={`d-runstate d-runstate--${status}`}>
        <i className="d-dot" />
        <span className="d-runstate__label">{label}</span>
        <span className="d-runstate__detail" title={detail}>{detail}</span>
      </div>
      {status === "loading" || status === "running" ? (
        <button className="button d-runbutton" onClick={() => void call("/api/stop")}><Square size={14} />Stop</button>
      ) : (
        <button className="button button--primary d-runbutton" disabled={!canStart} onClick={() => void call("/api/start")}>
          <Play size={14} />Start
        </button>
      )}
      {(status === "loading" || status === "stopping") && <div className="d-runbar__progress" />}
    </header>
  );
}

function Toasts() {
  const toasts = useDesk((s) => s.toasts);
  return (
    <div className="d-toasts" role="status">
      {toasts.map((t) => <div key={t.id} className={`d-toast d-toast--${t.kind}`}>{t.text}</div>)}
    </div>
  );
}

export function App() {
  const snapshot = useDesk((s) => s.snapshot);
  const page = useDesk((s) => s.page);
  const setPage = useDesk((s) => s.setPage);
  const poll = useDesk((s) => s.poll);
  const loadSchemas = useDesk((s) => s.loadSchemas);
  useEngineBridge();

  useEffect(() => {
    void loadSchemas();
    let stop = false;
    const loop = async () => {
      if (stop) return;
      await poll();
      const status = useDesk.getState().snapshot?.status;
      window.setTimeout(loop, status === "loading" || status === "stopping" ? 600 : 1200);
    };
    void loop();
    return () => { stop = true; };
  }, [poll, loadSchemas]);

  // Open on the chat once the model is ready, as the toggle on Server asks.
  const lastStatus = useRef<string | undefined>(undefined);
  useEffect(() => {
    const status = snapshot?.status;
    if (lastStatus.current && lastStatus.current !== status && status === "running" && snapshot?.settings.open_chat) setPage("chat");
    lastStatus.current = status;
  }, [snapshot?.status, snapshot?.settings.open_chat, setPage]);

  const failures = snapshot?.doctor.lines.filter((l) => l.status === "fail").length ?? 0;
  const downloading = snapshot?.models.downloads.some((d) => d.state === "downloading" || d.state === "queued") ?? false;
  const jobRunning = snapshot?.jobs.some((j) => j.state === "running") ?? false;

  return (
    <div className="d-app">
      <nav className="d-nav" aria-label="Sections">
        <div className="d-brand"><img src="/favicon.svg" alt="" width={26} height={26} /><span>Flyweight</span></div>
        {NAV.map(({ page: p, label, icon: Icon }) => (
          <button key={p} className={`d-nav__item${page === p ? " d-nav__item--active" : ""}`} onClick={() => setPage(p)}>
            <Icon size={17} /><span>{label}</span>
            {p === "chat" && snapshot?.status === "running" && <i className="d-nav__dot d-nav__dot--ok" />}
            {p === "models" && downloading && <i className="d-nav__dot d-nav__dot--busy" />}
            {p === "tools" && jobRunning && <i className="d-nav__dot d-nav__dot--busy" />}
            {p === "system" && failures > 0 && <b className="d-nav__badge">{failures}</b>}
          </button>
        ))}
        <div className="d-nav__foot">
          {snapshot?.update && (
            <button className="d-update" onClick={() => void useDesk.getState().call("/api/open", { url: snapshot.update!.url })}>
              <Download size={14} />Update to {snapshot.update.version}
            </button>
          )}
          <div className="d-version">Flyweight {snapshot?.version ?? ""}</div>
        </div>
      </nav>
      <div className="d-main">
        <RunBar />
        <div className="d-page">
          {page === "chat" && <ChatPage />}
          {page === "models" && <ModelsPage />}
          {page === "server" && <ServerPage />}
          {page === "monitor" && <MonitorPage />}
          {page === "tools" && <ToolsPage />}
          {page === "activity" && <ActivityPage />}
          {page === "system" && <SystemPage />}
        </div>
      </div>
      <Toasts />
    </div>
  );
}
