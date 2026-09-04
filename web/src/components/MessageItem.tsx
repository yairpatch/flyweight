import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Bot, Brain, Check, Copy, Pencil, RefreshCw, Trash2, User, Wrench, AlertTriangle, ChevronRight, Play } from "lucide-react";
import type { Message, ToolCall } from "../types";
import { useStore } from "../store";
import { StreamingMarkdown } from "./Markdown";
import { Sparkline } from "./charts";
import { AttachmentChip } from "./AttachmentChip";
import { detectDirection } from "../lib/direction";
import { formatRate, formatSeconds, formatTime, prettyJson } from "../lib/format";
import { holdPartialTag, splitThinking } from "../lib/thinking";

interface Props {
  message: Message;
  previous?: Message;
  isLast: boolean;
}

export const MessageItem = memo(function MessageItem({ message, previous, isLast }: Props) {
  const [editing, setEditing] = useState(false);
  const generating = Boolean(message.generating);
  // Reasoning arrives on its own channel normally; fall back to inline
  // <think> blocks for stored turns and servers that do not split it.
  const { reasoning, answer, reasoningLive } = useMemo(() => {
    const raw = generating ? holdPartialTag(message.content) : message.content;
    if (message.reasoning !== undefined || message.role !== "assistant") {
      return { reasoning: message.reasoning, answer: raw, reasoningLive: false };
    }
    const split = splitThinking(raw);
    return { reasoning: split.reasoning, answer: split.answer, reasoningLive: split.open };
  }, [message.content, message.reasoning, message.role, generating]);
  const direction = useMemo(() => detectDirection(answer || reasoning || ""), [answer, reasoning]);

  if (message.role === "tool") return <ToolResult message={message} />;

  return (
    <article className={`msg msg--${message.role}${generating ? " msg--live" : ""}`} dir={direction}>
      <div className="msg__avatar" aria-hidden="true">
        {message.role === "user" ? <User size={15} /> : <Bot size={15} />}
      </div>
      <div className="msg__body">
        <header className="msg__meta">
          <span className="msg__role">{message.role === "user" ? "You" : "Assistant"}</span>
          <time className="msg__time" dateTime={new Date(message.createdAt).toISOString()}>
            {formatTime(message.createdAt)}
          </time>
          {message.protocol && message.role === "assistant" && <span className="msg__proto">{message.protocol}</span>}
        </header>
        {(message.attachments ?? []).some((attachment) => attachment.kind !== "image") && (
          <div className="msg__attachments">
            {(message.attachments ?? [])
              .filter((attachment) => attachment.kind !== "image")
              .map((attachment) => (
                <AttachmentChip key={attachment.id} attachment={attachment} />
              ))}
          </div>
        )}
        {message.images && message.images.length > 0 && (
          <div className="msg__images">
            {message.images.map((url, index) => (
              <a key={index} href={url} target="_blank" rel="noopener noreferrer">
                <img src={url} alt={`Attachment ${index + 1}`} loading="lazy" />
              </a>
            ))}
          </div>
        )}
        {reasoning !== undefined && reasoning !== "" && (
          <ReasoningPanel text={reasoning} seconds={message.reasoningSeconds} live={generating && (reasoningLive || !answer)} />
        )}
        {editing ? (
          <EditBox message={message} onDone={() => setEditing(false)} />
        ) : message.role === "user" ? (
          <div className="msg__content msg__content--plain">{message.content}</div>
        ) : (
          <div className="msg__content markdown">
            {answer ? <StreamingMarkdown text={answer} live={generating} /> : null}
            {generating && !reasoning && !answer && !message.toolCalls?.length && <span className="caret" aria-label="Generating" />}
            {generating && answer && <span className="caret caret--inline" />}
          </div>
        )}
        {message.toolCalls?.map((call) => (
          <ToolCallCard key={call.id} call={call} messageId={message.id} live={generating} isLast={isLast} />
        ))}
        {message.error && (
          <div className="msg__error" role="alert">
            <AlertTriangle size={15} />
            <span>{message.error}</span>
            <button className="button button--small" onClick={() => void useStore.getState().regenerate(message.id)}>
              Retry
            </button>
          </div>
        )}
        {!editing && <Toolbar message={message} previous={previous} onEdit={() => setEditing(true)} />}
      </div>
    </article>
  );
});

function ReasoningPanel({ text, seconds, live }: { text: string; seconds?: number; live: boolean }) {
  // Collapsed by default, live or not: the header's shimmer says thinking is
  // in progress, and the transcript stays short. Opening it while live
  // follows the reasoning as it streams.
  const [open, setOpen] = useState(false);
  const body = useRef<HTMLDivElement>(null);
  // Follows the reasoning while open and live, the same way the transcript
  // follows the answer: a wheel-up inside the panel detaches, reaching the
  // bottom again re-attaches.
  const follow = useRef(true);
  const label = live ? "Thinking…" : seconds ? `Thought for ${formatSeconds(seconds)}` : "Thinking";
  useEffect(() => {
    const element = body.current;
    if (open && live && follow.current && element) element.scrollTop = element.scrollHeight;
  }, [text, open, live]);
  useEffect(() => {
    if (open) follow.current = true;
  }, [open]);
  const onBodyScroll = () => {
    const element = body.current;
    if (!element) return;
    follow.current = element.scrollHeight - element.scrollTop - element.clientHeight <= 8;
  };
  return (
    <details className={`thinking${live ? " thinking--live" : ""}`} open={open} onToggle={(event) => setOpen((event.target as HTMLDetailsElement).open)}>
      <summary>
        <Brain size={14} />
        <span>{label}</span>
        <ChevronRight size={14} className="thinking__chevron" />
      </summary>
      <div className="thinking__body markdown" dir={detectDirection(text)} ref={body} onScroll={onBodyScroll} onWheel={(event) => { if (event.deltaY < 0) follow.current = false; }}>
        <StreamingMarkdown text={text} live={live} />
      </div>
    </details>
  );
}

function ToolCallCard({ call, messageId, live, isLast }: { call: ToolCall; messageId: string; live: boolean; isLast: boolean }) {
  const conversation = useStore((state) => state.conversations.find((item) => item.id === state.activeId));
  const submitToolResult = useStore((state) => state.submitToolResult);
  const busy = useStore((state) => Boolean(state.generating));
  const [result, setResult] = useState("");
  const answered = conversation?.messages.some((message) => message.role === "tool" && message.toolCallId === call.id);
  const args = useMemo(() => prettyJson(call.arguments), [call.arguments]);

  return (
    <div className={`tool${live ? " tool--live" : ""}`} dir="ltr">
      <div className="tool__head">
        <Wrench size={14} />
        <span className="tool__name">{call.name || "tool"}</span>
        <span className="tool__id">{call.id}</span>
      </div>
      <pre className="tool__args">
        <code>{args}</code>
      </pre>
      {!live && !answered && isLast && (
        <div className="tool__reply">
          <textarea
            className="tool__input"
            placeholder="Paste the tool's result here (text or JSON), then continue"
            value={result}
            onChange={(event) => setResult(event.target.value)}
            rows={3}
          />
          <div className="tool__actions">
            <button className="button button--small" disabled={busy} onClick={() => void submitToolResult(messageId, call.id, result, false)}>
              Add result
            </button>
            <button className="button button--small button--primary" disabled={busy} onClick={() => void submitToolResult(messageId, call.id, result, true)}>
              <Play size={13} /> Add and continue
            </button>
          </div>
        </div>
      )}
      {answered && <div className="tool__answered">Result provided</div>}
    </div>
  );
}

function ToolResult({ message }: { message: Message }) {
  const deleteMessage = useStore((state) => state.deleteMessage);
  return (
    <article className="msg msg--tool" dir="ltr">
      <div className="msg__avatar" aria-hidden="true">
        <Wrench size={15} />
      </div>
      <div className="msg__body">
        <header className="msg__meta">
          <span className="msg__role">Tool result · {message.toolName ?? message.toolCallId}</span>
          <time className="msg__time">{formatTime(message.createdAt)}</time>
        </header>
        <pre className="tool__args">
          <code>{prettyJson(message.content) || "(empty)"}</code>
        </pre>
        <div className="msg__toolbar">
          <button className="icon-button icon-button--small" onClick={() => deleteMessage(message.id)} title="Delete" aria-label="Delete tool result">
            <Trash2 size={13} />
          </button>
        </div>
      </div>
    </article>
  );
}

function EditBox({ message, onDone }: { message: Message; onDone: () => void }) {
  const editMessage = useStore((state) => state.editMessage);
  const [text, setText] = useState(message.content);
  const save = (resend: boolean) => {
    void editMessage(message.id, text, resend);
    onDone();
  };
  return (
    <div className="edit">
      <textarea
        className="edit__input"
        value={text}
        autoFocus
        rows={Math.min(14, Math.max(3, text.split("\n").length + 1))}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if ((event.ctrlKey || event.metaKey) && event.key === "Enter") save(message.role === "user");
          if (event.key === "Escape") onDone();
        }}
      />
      <div className="edit__actions">
        <button className="button button--small" onClick={onDone}>
          Cancel
        </button>
        <button className="button button--small" onClick={() => save(false)}>
          Save
        </button>
        {message.role === "user" && (
          <button className="button button--small button--primary" onClick={() => save(true)}>
            Save and resend
          </button>
        )}
      </div>
    </div>
  );
}

function Toolbar({ message, previous, onEdit }: { message: Message; previous?: Message; onEdit: () => void }) {
  const regenerate = useStore((state) => state.regenerate);
  const deleteMessage = useStore((state) => state.deleteMessage);
  const busy = useStore((state) => Boolean(state.generating));
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.reasoning !== undefined ? message.content : splitThinking(message.content).answer);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      useStore.getState().toast("Clipboard unavailable", "error");
    }
  };
  const metrics = message.metrics;
  const trend = useMemo(() => throughputTrend(metrics?.samples), [metrics?.samples]);
  if (message.generating) return null;
  const rate = metrics && metrics.tokens > 1 && metrics.decodeSeconds > 0 ? (metrics.tokens - 1) / metrics.decodeSeconds : null;

  return (
    <div className="msg__toolbar">
      <button className="icon-button icon-button--small" onClick={() => void copy()} title="Copy" aria-label="Copy message">
        {copied ? <Check size={13} /> : <Copy size={13} />}
      </button>
      <button className="icon-button icon-button--small" onClick={onEdit} title="Edit" aria-label="Edit message">
        <Pencil size={13} />
      </button>
      {message.role === "assistant" && previous && (
        <button className="icon-button icon-button--small" disabled={busy} onClick={() => void regenerate(message.id)} title="Regenerate" aria-label="Regenerate">
          <RefreshCw size={13} />
        </button>
      )}
      <button className="icon-button icon-button--small" onClick={() => deleteMessage(message.id)} title="Delete" aria-label="Delete message">
        <Trash2 size={13} />
      </button>
      {message.role === "assistant" && metrics && (
        <span className="msg__metrics" title={metricsTooltip(message)}>
          {metrics.tokens} tok
          {rate !== null && <> · {formatRate(rate)}</>}
          {metrics.ttftSeconds !== undefined && <> · TTFT {formatSeconds(metrics.ttftSeconds)}</>}
          {message.usage?.cached_tokens ? <> · {message.usage.cached_tokens} cached</> : null}
          {message.finishReason && message.finishReason !== "stop" && <> · {message.finishReason}</>}
          {trend.length > 4 && (
            <span className="msg__spark">
              <Sparkline values={trend} height={18} compact />
            </span>
          )}
        </span>
      )}
    </div>
  );
}

function metricsTooltip(message: Message): string {
  const parts: string[] = [];
  const metrics = message.metrics;
  if (metrics) {
    parts.push(`${metrics.tokens} generated tokens`);
    parts.push(`decode ${formatSeconds(metrics.decodeSeconds)}`);
    if (metrics.totalSeconds !== undefined) parts.push(`total ${formatSeconds(metrics.totalSeconds)}`);
  }
  const usage = message.usage;
  if (usage) {
    if (usage.prompt_tokens !== undefined) parts.push(`prompt ${usage.prompt_tokens} tokens`);
    if (usage.cached_tokens) parts.push(`${usage.cached_tokens} from prefix cache`);
    if (usage.reasoning_tokens) parts.push(`${usage.reasoning_tokens} reasoning tokens`);
  }
  if (message.finishReason) parts.push(`finish: ${message.finishReason}`);
  return parts.join("\n");
}

/** Instantaneous tokens per second over ~250 ms windows from the live samples. */
export function throughputTrend(samples?: Array<[number, number]>): number[] {
  if (!samples || samples.length < 3) return [];
  const out: number[] = [];
  let windowStart = 0;
  for (let index = 1; index < samples.length; index += 1) {
    const [time, tokens] = samples[index];
    const [startTime, startTokens] = samples[windowStart];
    if (time - startTime >= 250) {
      out.push(((tokens - startTokens) * 1000) / (time - startTime));
      windowStart = index;
    }
  }
  return out;
}
