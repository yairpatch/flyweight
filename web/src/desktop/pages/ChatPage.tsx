import { useMemo, useState } from "react";
import {
  Image as ImageIcon, MessageSquare, Pin, Plus, Search, SlidersHorizontal, Trash2, Wrench, Braces, Hash, Play, X,
} from "lucide-react";
import { useStore, useActiveConversation } from "../../store";
import { Transcript } from "../../components/Transcript";
import { Composer } from "../../components/Composer";
import { Studio } from "../../components/Studio";
import { PreviewOverlay } from "../../components/PreviewOverlay";
import { Toasts as ChatToasts } from "../../components/Toasts";
import { SettingsPanel } from "../../components/panels/SettingsPanel";
import { ToolsPanel } from "../../components/panels/ToolsPanel";
import { InspectorPanel } from "../../components/panels/InspectorPanel";
import { TokenizerPanel } from "../../components/panels/TokenizerPanel";
import { useDesk, clock } from "../state";
import { relativeDay } from "../../lib/format";

type Side = "settings" | "tools" | "inspector" | "tokenizer" | null;

const SIDES: Array<{ id: Exclude<Side, null>; label: string; icon: typeof Wrench }> = [
  { id: "settings", label: "Generation", icon: SlidersHorizontal },
  { id: "tools", label: "Tools", icon: Wrench },
  { id: "inspector", label: "Requests", icon: Braces },
  { id: "tokenizer", label: "Tokenizer", icon: Hash },
];

function Conversations() {
  const conversations = useStore((s) => s.conversations);
  const activeId = useStore((s) => s.activeId);
  const select = useStore((s) => s.selectConversation);
  const create = useStore((s) => s.newConversation);
  const remove = useStore((s) => s.deleteConversation);
  const togglePin = useStore((s) => s.togglePin);
  const mode = useStore((s) => s.mode);
  const setMode = useStore((s) => s.setMode);
  const imagesEnabled = useStore((s) => Boolean(s.health?.execution?.images));
  const [query, setQuery] = useState("");

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const list = conversations
      .filter((c) => !needle || c.title.toLowerCase().includes(needle)
        || c.messages.some((m) => m.content.toLowerCase().includes(needle)))
      .sort((a, b) => Number(Boolean(b.pinned)) - Number(Boolean(a.pinned)) || b.updatedAt - a.updatedAt);
    const out: Array<[string, typeof list]> = [];
    for (const c of list) {
      const group = c.pinned ? "Pinned" : relativeDay(c.updatedAt);
      const last = out[out.length - 1];
      if (last && last[0] === group) last[1].push(c);
      else out.push([group, [c]]);
    }
    return out;
  }, [conversations, query]);

  return (
    <aside className="d-convs">
      <div className="d-convs__top">
        <button className="button button--primary d-convs__new" onClick={() => { setMode("chat"); create(); }}>
          <Plus size={15} />New chat
        </button>
        {(imagesEnabled || mode === "images") && (
          <button className={`d-iconbtn${mode === "images" ? " d-iconbtn--on" : ""}`} title="Image studio"
                  onClick={() => setMode(mode === "images" ? "chat" : "images")}>
            <ImageIcon size={16} />
          </button>
        )}
      </div>
      <label className="d-search">
        <Search size={14} />
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search chats" />
      </label>
      <div className="d-convs__list">
        {groups.length === 0 && <div className="d-convs__empty">Your conversations appear here. They stay on this computer.</div>}
        {groups.map(([title, items]) => (
          <div key={title}>
            <div className="d-convs__group">{title}</div>
            {items.map((c) => (
              <div key={c.id} className={`d-conv${c.id === activeId && mode === "chat" ? " d-conv--active" : ""}`}
                   onClick={() => { setMode("chat"); select(c.id); }} role="button" tabIndex={0}
                   onKeyDown={(e) => { if (e.key === "Enter") { setMode("chat"); select(c.id); } }}>
                <MessageSquare size={14} />
                <span className="d-conv__title">{c.title || "New chat"}</span>
                <span className="d-conv__actions">
                  <button title={c.pinned ? "Unpin" : "Pin"} onClick={(e) => { e.stopPropagation(); togglePin(c.id); }}>
                    <Pin size={13} />
                  </button>
                  <button title="Delete" onClick={(e) => { e.stopPropagation(); if (confirm(`Delete "${c.title || "this chat"}"?`)) void remove(c.id); }}>
                    <Trash2 size={13} />
                  </button>
                </span>
              </div>
            ))}
          </div>
        ))}
      </div>
    </aside>
  );
}

function NotRunning() {
  const snapshot = useDesk((s) => s.snapshot);
  const call = useDesk((s) => s.call);
  const setPage = useDesk((s) => s.setPage);
  if (!snapshot) return null;
  const { status, model } = snapshot;
  return (
    <div className="d-empty">
      <div className="d-empty__icon"><MessageSquare size={24} /></div>
      {status === "loading" ? (
        <>
          <h2>Loading {model.name}…</h2>
          <p className="d-muted">{clock(snapshot.elapsed)} · {snapshot.lastLine}</p>
          <div className="d-bar"><i /></div>
        </>
      ) : status === "failed" ? (
        <>
          <h2>The model did not start</h2>
          <p className="d-muted">{snapshot.hint}</p>
          <div className="d-row">
            <button className="button" onClick={() => setPage("activity")}>View activity</button>
            <button className="button" onClick={() => setPage("server")}>Change settings</button>
            <button className="button button--primary" onClick={() => void call("/api/start")}><Play size={14} />Try again</button>
          </div>
        </>
      ) : model.ok ? (
        <>
          <h2>Start {model.name} to chat</h2>
          <p className="d-muted">Everything runs on this computer. Conversations are saved here too.</p>
          <div className="d-row">
            <button className="button" onClick={() => setPage("server")}>Settings</button>
            <button className="button button--primary" onClick={() => void call("/api/start")}><Play size={14} />Start</button>
          </div>
        </>
      ) : (
        <>
          <h2>Choose a model to chat with</h2>
          <p className="d-muted">Download one picked for this computer, or open a .gguf you already have.</p>
          <div className="d-row">
            <button className="button" onClick={() => void call("/api/browse", { kind: "file" })}>Open a file…</button>
            <button className="button button--primary" onClick={() => setPage("models")}>Get a model</button>
          </div>
        </>
      )}
    </div>
  );
}

export function ChatPage() {
  const running = useDesk((s) => s.snapshot?.status === "running");
  const mode = useStore((s) => s.mode);
  const conversation = useActiveConversation();
  const [side, setSide] = useState<Side>(null);

  return (
    <div className={`d-chat${side ? " d-chat--side" : ""}`}>
      <Conversations />
      <section className="d-chat__main">
        <div className="d-chat__head">
          <h1 className="d-chat__title">{mode === "images" ? "Image studio" : conversation?.title || "New chat"}</h1>
          <div className="d-chat__tabs">
            {SIDES.map(({ id, label, icon: Icon }) => (
              <button key={id} className={`d-tab${side === id ? " d-tab--on" : ""}`} onClick={() => setSide(side === id ? null : id)}>
                <Icon size={14} /><span>{label}</span>
              </button>
            ))}
          </div>
        </div>
        {/* The chat UI's own components, with this app's frame around them. */}
        <main className="main d-chat__body">
          {!running ? <NotRunning /> : mode === "images" ? <Studio /> : (<><Transcript /><Composer /></>)}
        </main>
      </section>
      {side && (
        <aside className="d-chat__side">
          <div className="d-chat__sidehead">
            <span>{SIDES.find((s) => s.id === side)?.label}</span>
            <button className="d-iconbtn" onClick={() => setSide(null)} aria-label="Close"><X size={15} /></button>
          </div>
          <div className="d-chat__sidebody panel__body">
            {side === "settings" && <SettingsPanel managedKey />}
            {side === "tools" && <ToolsPanel />}
            {side === "inspector" && <InspectorPanel />}
            {side === "tokenizer" && <TokenizerPanel />}
          </div>
        </aside>
      )}
      <PreviewOverlay />
      <ChatToasts />
    </div>
  );
}
