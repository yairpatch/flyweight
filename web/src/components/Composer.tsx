import { useEffect, useRef, useState, type ClipboardEvent, type DragEvent } from "react";
import { ArrowUp, Bot, Braces, Brain, Paperclip, Square, Wrench, X, Zap } from "lucide-react";
import { useStore } from "../store";
import { ACCEPT_ATTRIBUTE, MAX_ATTACHMENTS, filesFrom, readAttachment, setPdfMode } from "../lib/attachments";
import { AttachmentChip } from "./AttachmentChip";
import { api } from "../lib/api";
import { PROTOCOL_LABELS } from "../lib/protocols";
import type { Attachment, ReasoningEffort } from "../types";

const REASONING_CYCLE: Array<{ thinking: boolean; effort: ReasoningEffort; label: string }> = [
  { thinking: false, effort: "auto", label: "Thinking off" },
  { thinking: true, effort: "auto", label: "Thinking on" },
  { thinking: true, effort: "low", label: "Thinking · low" },
  { thinking: true, effort: "medium", label: "Thinking · medium" },
  { thinking: true, effort: "high", label: "Thinking · high" },
  { thinking: true, effort: "xhigh", label: "Thinking · xhigh" },
];

export function Composer() {
  const draft = useStore((state) => state.draft);
  const setDraft = useStore((state) => state.setDraft);
  const pendingAttachments = useStore((state) => state.pendingAttachments);
  const setPendingAttachments = useStore((state) => state.setPendingAttachments);
  const sendMessage = useStore((state) => state.sendMessage);
  const stopGeneration = useStore((state) => state.stopGeneration);
  const generating = useStore((state) => Boolean(state.generating));
  const canStopThinking = useStore(
    (state) =>
      Boolean(state.generating?.requestId) &&
      state.generating?.phase === "thinking" &&
      state.settings.protocol !== "responses" &&
      (state.props?.capabilities ?? []).includes("stop_thinking"),
  );
  const stopThinking = useStore((state) => state.stopThinking);
  const settings = useStore((state) => state.settings);
  const updateSettings = useStore((state) => state.updateSettings);
  const mode = useStore((state) => state.mode);
  const tools = useStore((state) => state.tools);
  const setPanel = useStore((state) => state.setPanel);
  const health = useStore((state) => state.health);
  const status = useStore((state) => state.status);
  const toast = useStore((state) => state.toast);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [tokenCount, setTokenCount] = useState<number | null>(null);
  const [reading, setReading] = useState(0);

  const props = useStore((state) => state.props);
  const vision = Boolean(health?.execution?.vision);
  const readContext = { vision, contextWindow: props?.context_window ?? health?.context_window };
  const enabledTools = tools.filter((tool) => tool.enabled).length;
  const reasoningIndex = Math.max(
    0,
    REASONING_CYCLE.findIndex((entry) => entry.thinking === settings.thinking && (entry.thinking ? entry.effort === settings.reasoningEffort : true)),
  );

  useEffect(() => {
    const element = textarea.current;
    if (!element) return;
    element.style.height = "auto";
    element.style.height = `${Math.min(element.scrollHeight, 320)}px`;
  }, [draft]);

  // Live token count for the draft, debounced; skipped when offline.
  useEffect(() => {
    if (!draft.trim() || status === "offline" || status === "locked") {
      setTokenCount(null);
      return;
    }
    const timer = window.setTimeout(() => {
      api.tokenize(draft).then((result) => setTokenCount(result.count)).catch(() => setTokenCount(null));
    }, 400);
    return () => window.clearTimeout(timer);
  }, [draft, status]);

  const attach = async (files: File[]) => {
    if (!files.length) return;
    const room = MAX_ATTACHMENTS - pendingAttachments.length;
    if (room <= 0) {
      toast(`A turn takes at most ${MAX_ATTACHMENTS} attachments`, "error");
      return;
    }
    if (files.length > room) toast(`Only the first ${room} of ${files.length} files were attached`, "info");
    setReading((count) => count + 1);
    try {
      const read = await Promise.all(files.slice(0, room).map((file) => readAttachment(file, readContext)));
      for (const attachment of read) {
        if (attachment.error) toast(`${attachment.name}: ${attachment.error}`, "error");
      }
      setPendingAttachments([...useStore.getState().pendingAttachments, ...read].slice(0, MAX_ATTACHMENTS));
    } finally {
      setReading((count) => count - 1);
    }
  };

  const togglePages = async (attachment: Attachment) => {
    setReading((count) => count + 1);
    try {
      const next = await setPdfMode(attachment, attachment.mode === "pages" ? "text" : "pages", readContext);
      if (next.error) toast(`${next.name}: ${next.error}`, "error");
      setPendingAttachments(useStore.getState().pendingAttachments.map((item) => (item.id === attachment.id ? next : item)));
    } finally {
      setReading((count) => count - 1);
    }
  };

  const usable = pendingAttachments.filter((attachment) => !attachment.error);

  const submit = () => {
    if (generating || reading) return;
    if (!draft.trim() && !usable.length) return;
    void sendMessage(draft, pendingAttachments);
  };

  const onPaste = (event: ClipboardEvent<HTMLTextAreaElement>) => {
    const files = filesFrom(event.clipboardData?.items);
    if (files.length) {
      event.preventDefault();
      void attach(files);
    }
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    void attach(filesFrom(event.dataTransfer?.files));
  };

  const cycleReasoning = () => {
    const next = REASONING_CYCLE[(reasoningIndex + 1) % REASONING_CYCLE.length];
    updateSettings({ thinking: next.thinking, reasoningEffort: next.effort });
  };

  return (
    <div className={`composer${dragging ? " composer--drag" : ""}`} onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
      <div className="composer__box">
        {pendingAttachments.length > 0 && (
          <div className="composer__attachments">
            {pendingAttachments.map((attachment) => {
              const remove = () => setPendingAttachments(pendingAttachments.filter((item) => item.id !== attachment.id));
              if (attachment.kind === "image" && attachment.url) {
                return (
                  <div key={attachment.id} className="composer__thumb">
                    <img src={attachment.url} alt={attachment.name} />
                    <button className="composer__remove" onClick={remove} aria-label={`Remove ${attachment.name}`}>
                      <X size={12} />
                    </button>
                  </div>
                );
              }
              return (
                <AttachmentChip
                  key={attachment.id}
                  attachment={attachment}
                  onRemove={remove}
                  onTogglePages={attachment.kind === "pdf" && vision ? () => void togglePages(attachment) : undefined}
                  busy={reading > 0}
                />
              );
            })}
          </div>
        )}
        <textarea
          ref={textarea}
          className="composer__input"
          placeholder={status === "locked" ? "Enter the API key in settings to start" : mode === "agent" ? "Give the agent a task…" : "Message the model…"}
          value={draft}
          rows={1}
          onChange={(event) => setDraft(event.target.value)}
          onPaste={onPaste}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
              event.preventDefault();
              submit();
            }
          }}
          aria-label="Message"
        />
        <div className="composer__bar">
          <div className="composer__chips">
            <button className={`chip chip--button${settings.thinking ? " chip--on" : ""}`} onClick={cycleReasoning} aria-pressed={settings.thinking} title="Cycle thinking level">
              <Brain size={13} /> {REASONING_CYCLE[reasoningIndex].label}
            </button>
            <button className={`chip chip--button${enabledTools ? " chip--on" : ""}`} onClick={() => setPanel("tools")} title="Tools">
              <Wrench size={13} /> {enabledTools ? `${enabledTools} tool${enabledTools === 1 ? "" : "s"}` : "Tools"}
            </button>
            {mode === "agent" && (
              <button className="chip chip--button chip--on" onClick={() => setPanel("tools")} title={`Agent run: tool calls run through their handlers automatically, up to ${settings.agentMaxTurns} turns`}>
                <Bot size={13} /> Agent · {settings.agentMaxTurns} turns
              </button>
            )}
            {settings.responseFormat !== "text" && (
              <button className="chip chip--button chip--on" onClick={() => setPanel("settings")} title="Response format">
                <Braces size={13} /> {settings.responseFormat === "json_object" ? "JSON" : "JSON schema"}
              </button>
            )}
            <select className="chip chip--select" value={settings.protocol} onChange={(event) => updateSettings({ protocol: event.target.value as typeof settings.protocol })} aria-label="API protocol" title="API protocol used for this conversation">
              {(Object.keys(PROTOCOL_LABELS) as Array<keyof typeof PROTOCOL_LABELS>).map((key) => (
                <option key={key} value={key}>
                  {PROTOCOL_LABELS[key]}
                </option>
              ))}
            </select>
            <button
              className="chip chip--button"
              onClick={() => fileInput.current?.click()}
              disabled={reading > 0}
              title={vision ? "Attach files: images, PDF, Word, Excel, text and code" : "Attach files: PDF, Word, Excel, text and code (no vision tower for images)"}
            >
              <Paperclip size={13} /> {reading > 0 ? "Reading…" : "Attach"}
            </button>
            <input ref={fileInput} type="file" accept={ACCEPT_ATTRIBUTE} multiple hidden onChange={(event) => { void attach(filesFrom(event.target.files)); event.target.value = ""; }} />
          </div>
          <div className="composer__right">
            {tokenCount !== null && <span className="composer__count" title="Tokens in the draft">{tokenCount} tok</span>}
            {canStopThinking && (
              <button className="chip chip--button chip--on" onClick={() => void stopThinking()} title="Close the thinking block and answer now">
                <Zap size={13} /> Answer now
              </button>
            )}
            {generating ? (
              <button className="send send--stop" onClick={stopGeneration} aria-label="Stop generating" title="Stop (Esc)">
                <Square size={14} />
              </button>
            ) : (
              <button className="send" onClick={submit} disabled={reading > 0 || (!draft.trim() && !usable.length)} aria-label="Send" title="Send (Enter)">
                <ArrowUp size={16} />
              </button>
            )}
          </div>
        </div>
      </div>
      <p className="composer__hint">Enter to send · Shift+Enter for a new line · Ctrl+K for commands</p>
    </div>
  );
}
