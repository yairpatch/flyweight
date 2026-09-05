import { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import { downloadJson } from "./Sidebar";
import { slugify } from "../lib/format";
import { conversationToMarkdown } from "../lib/export";
import { downloadBlob } from "./Sidebar";

interface Command {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
}

export function CommandPalette() {
  const open = useStore((state) => state.paletteOpen);
  const setOpen = useStore((state) => state.setPaletteOpen);
  const conversations = useStore((state) => state.conversations);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const input = useRef<HTMLInputElement>(null);

  const commands = useMemo<Command[]>(() => {
    const s = useStore.getState();
    const base: Command[] = [
      { id: "new", label: "New conversation", hint: "Ctrl+Shift+O", run: () => s.newConversation("chat") },
      { id: "new-agent", label: "New agent run", run: () => s.newConversation("agent") },
      { id: "settings", label: "Open generation settings", hint: "Ctrl+,", run: () => s.setPanel("settings") },
      { id: "tools", label: "Open tools", run: () => s.setPanel("tools") },
      { id: "runtime", label: "Open runtime dashboard", run: () => s.setPanel("runtime") },
      { id: "tokenizer", label: "Open tokenizer", run: () => s.setPanel("tokenizer") },
      { id: "playground", label: "Open completions playground", run: () => s.setPanel("playground") },
      { id: "inspector", label: "Open request inspector", run: () => s.setPanel("inspector") },
      { id: "theme-dark", label: "Theme: dark", run: () => s.setTheme("dark") },
      { id: "theme-light", label: "Theme: light", run: () => s.setTheme("light") },
      { id: "theme-system", label: "Theme: system", run: () => s.setTheme("system") },
      { id: "sidebar", label: "Toggle sidebar", hint: "Ctrl+B", run: () => s.toggleSidebar() },
      { id: "thinking", label: s.settings.thinking ? "Turn thinking off" : "Turn thinking on", run: () => s.updateSettings({ thinking: !s.settings.thinking }) },
      { id: "proto-chat", label: "Protocol: OpenAI chat completions", run: () => s.updateSettings({ protocol: "chat" }) },
      { id: "proto-anthropic", label: "Protocol: Anthropic messages", run: () => s.updateSettings({ protocol: "anthropic" }) },
      { id: "proto-responses", label: "Protocol: OpenAI responses", run: () => s.updateSettings({ protocol: "responses" }) },
      {
        id: "export-json",
        label: "Export conversation as JSON",
        run: () => {
          const active = s.active();
          if (active) downloadJson(active, `${slugify(active.title)}.json`);
        },
      },
      {
        id: "export-md",
        label: "Export conversation as Markdown",
        run: () => {
          const active = s.active();
          if (active) downloadBlob(new Blob([conversationToMarkdown(active)], { type: "text/markdown" }), `${slugify(active.title)}.md`);
        },
      },
      {
        id: "clear",
        label: "Clear messages in this conversation",
        run: () => {
          const active = s.active();
          if (active && confirm("Clear all messages in this conversation?")) s.clearMessages(active.id);
        },
      },
      { id: "reset", label: "Reset settings to model defaults", run: () => s.resetSettings() },
    ];
    const convs: Command[] = conversations.slice(0, 50).map((conversation) => ({
      id: `conv-${conversation.id}`,
      label: conversation.title,
      hint: "conversation",
      run: () => s.selectConversation(conversation.id),
    }));
    return [...base, ...convs];
  }, [conversations, open]);

  const results = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return commands.slice(0, 14);
    return commands.filter((command) => command.label.toLowerCase().includes(needle)).slice(0, 14);
  }, [commands, query]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setCursor(0);
      window.setTimeout(() => input.current?.focus(), 0);
    }
  }, [open]);

  useEffect(() => setCursor(0), [query]);

  if (!open) return null;

  const run = (command: Command | undefined) => {
    if (!command) return;
    setOpen(false);
    command.run();
  };

  return (
    <div className="palette-backdrop" onClick={() => setOpen(false)}>
      <div className="palette" role="dialog" aria-label="Command palette" onClick={(event) => event.stopPropagation()}>
        <input
          ref={input}
          className="palette__input"
          placeholder="Type a command or conversation…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "ArrowDown") {
              event.preventDefault();
              setCursor((c) => Math.min(results.length - 1, c + 1));
            } else if (event.key === "ArrowUp") {
              event.preventDefault();
              setCursor((c) => Math.max(0, c - 1));
            } else if (event.key === "Enter") {
              event.preventDefault();
              run(results[cursor]);
            }
          }}
        />
        <ul className="palette__list" role="listbox">
          {results.map((command, index) => (
            <li
              key={command.id}
              role="option"
              aria-selected={index === cursor}
              className={`palette__item${index === cursor ? " palette__item--active" : ""}`}
              onMouseEnter={() => setCursor(index)}
              onClick={() => run(command)}
            >
              <span>{command.label}</span>
              {command.hint && <kbd>{command.hint}</kbd>}
            </li>
          ))}
          {results.length === 0 && <li className="palette__empty">No matching commands</li>}
        </ul>
      </div>
    </div>
  );
}
