import { useMemo, useRef, useState } from "react";
import {
  Download,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Pin,
  PinOff,
  Search,
  Trash2,
  Upload,
} from "lucide-react";
import { useStore } from "../store";
import { relativeDay, slugify } from "../lib/format";
import type { Conversation } from "../types";

export function Sidebar() {
  const conversations = useStore((state) => state.conversations);
  const activeId = useStore((state) => state.activeId);
  const open = useStore((state) => state.sidebarOpen);
  const toggleSidebar = useStore((state) => state.toggleSidebar);
  const newConversation = useStore((state) => state.newConversation);
  const selectConversation = useStore((state) => state.selectConversation);
  const importConversation = useStore((state) => state.importConversation);
  const toast = useStore((state) => state.toast);
  const [query, setQuery] = useState("");
  const fileInput = useRef<HTMLInputElement>(null);

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = conversations.filter((conversation) => {
      if (!needle) return true;
      if (conversation.title.toLowerCase().includes(needle)) return true;
      return conversation.messages.some((message) => message.content.toLowerCase().includes(needle));
    });
    const pinned = filtered.filter((conversation) => conversation.pinned);
    const rest = filtered.filter((conversation) => !conversation.pinned);
    const byDay = new Map<string, Conversation[]>();
    for (const conversation of rest) {
      const label = relativeDay(conversation.updatedAt);
      byDay.set(label, [...(byDay.get(label) ?? []), conversation]);
    }
    const out: Array<[string, Conversation[]]> = [];
    if (pinned.length) out.push(["Pinned", pinned]);
    for (const label of ["Today", "Yesterday", "Previous 7 days", "Previous 30 days", "Older"]) {
      const items = byDay.get(label);
      if (items?.length) out.push([label, items]);
    }
    return out;
  }, [conversations, query]);

  const onImport = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      const imported = Array.isArray(data) ? data : [data];
      let count = 0;
      for (const item of imported) if (await importConversation(item)) count += 1;
      toast(count ? `Imported ${count} conversation${count === 1 ? "" : "s"}` : "Nothing importable in that file", count ? "success" : "error");
    } catch {
      toast("That file is not valid JSON", "error");
    }
    if (fileInput.current) fileInput.current.value = "";
  };

  return (
    <aside className="sidebar" aria-label="Conversations">
      <div className="sidebar__top">
        <button className="icon-button" onClick={() => toggleSidebar()} title={open ? "Collapse sidebar (Ctrl+B)" : "Expand sidebar (Ctrl+B)"} aria-label="Toggle sidebar">
          {open ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
        </button>
        <button className="button button--primary sidebar__new" onClick={() => newConversation()} title="New conversation (Ctrl+Shift+O)">
          <MessageSquarePlus size={16} />
          <span className="sidebar__label">New chat</span>
        </button>
      </div>
      <label className="sidebar__search">
        <Search size={14} />
        <input
          type="search"
          placeholder="Search conversations"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          aria-label="Search conversations"
        />
      </label>
      <nav className="sidebar__list">
        {groups.length === 0 && <p className="sidebar__empty">{query ? "No matches." : "No conversations yet."}</p>}
        {groups.map(([label, items]) => (
          <section key={label} className="sidebar__group">
            <h2 className="sidebar__group-title">{label}</h2>
            {items.map((conversation) => (
              <ConversationRow key={conversation.id} conversation={conversation} active={conversation.id === activeId} onSelect={() => selectConversation(conversation.id)} />
            ))}
          </section>
        ))}
      </nav>
      <div className="sidebar__bottom">
        <button className="button button--ghost" onClick={() => fileInput.current?.click()} title="Import conversations from JSON">
          <Upload size={15} />
          <span className="sidebar__label">Import</span>
        </button>
        <button
          className="button button--ghost"
          onClick={() => {
            const active = useStore.getState().active();
            if (!active) return toast("Nothing to export");
            downloadJson(active, `${slugify(active.title)}.json`);
          }}
          title="Export the active conversation as JSON"
        >
          <Download size={15} />
          <span className="sidebar__label">Export</span>
        </button>
        <input ref={fileInput} type="file" accept="application/json,.json" hidden onChange={(event) => void onImport(event.target.files)} />
      </div>
    </aside>
  );
}

function ConversationRow({ conversation, active, onSelect }: { conversation: Conversation; active: boolean; onSelect: () => void }) {
  const renameConversation = useStore((state) => state.renameConversation);
  const deleteConversation = useStore((state) => state.deleteConversation);
  const togglePin = useStore((state) => state.togglePin);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(conversation.title);
  const generating = useStore((state) => state.generating?.conversationId === conversation.id);

  const commit = () => {
    setEditing(false);
    renameConversation(conversation.id, title);
  };

  return (
    <div className={`conv${active ? " conv--active" : ""}`} aria-current={active ? "page" : undefined}>
      {editing ? (
        <input
          className="conv__rename"
          autoFocus
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          onBlur={commit}
          onKeyDown={(event) => {
            if (event.key === "Enter") commit();
            if (event.key === "Escape") {
              setTitle(conversation.title);
              setEditing(false);
            }
          }}
          aria-label="Conversation title"
        />
      ) : (
        <button
          className="conv__title"
          onClick={onSelect}
          onDoubleClick={() => {
            setTitle(conversation.title);
            setEditing(true);
          }}
          title={conversation.title}
        >
          {generating && <span className="conv__pulse" aria-label="Generating" />}
          <span className="sidebar__label">{conversation.title}</span>
        </button>
      )}
      <div className="conv__actions">
        <button className="icon-button icon-button--small" onClick={() => togglePin(conversation.id)} title={conversation.pinned ? "Unpin" : "Pin"} aria-label={conversation.pinned ? "Unpin" : "Pin"}>
          {conversation.pinned ? <PinOff size={13} /> : <Pin size={13} />}
        </button>
        <button
          className="icon-button icon-button--small icon-button--danger"
          onClick={() => {
            if (confirm(`Delete “${conversation.title}”?`)) void deleteConversation(conversation.id);
          }}
          title="Delete"
          aria-label="Delete conversation"
        >
          <Trash2 size={13} />
        </button>
      </div>
    </div>
  );
}

export function downloadJson(data: unknown, filename: string): void {
  downloadBlob(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }), filename);
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}
