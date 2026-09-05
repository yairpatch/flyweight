import { memo, useEffect, useMemo, useRef, useState } from "react";
import { Ban, Bot, Brain, Check, Copy, PauseCircle, Pencil, RefreshCw, ShieldAlert, Trash2, User, Wrench, AlertTriangle, ChevronRight, Play } from "lucide-react";
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
          {message.compaction && (
            <span
              className="msg__proto msg__proto--warn"
              title={`The run no longer fit the context window, so ${describeCompaction(message.compaction)} were left out of this request. The transcript still has them.`}
            >
              compacted
            </span>
          )}
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
        {message.toolCalls?.length ? <AgentPauseNote messageId={message.id} /> : null}
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

/** What the prompt lost to the context window, for the "compacted" tooltip. */
function describeCompaction({ stubbed, dropped }: NonNullable<Message["compaction"]>): string {
  const parts: string[] = [];
  if (stubbed) parts.push(`${stubbed} older tool result${stubbed === 1 ? "" : "s"}`);
  if (dropped) parts.push(`${dropped} older message${dropped === 1 ? "" : "s"}`);
  return parts.join(" and ") || "older steps";
}

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
  // The agent loop owns this turn's calls while it runs them, whether it is
  // executing one or waiting on an approval; the manual reply box stays away
  // until it is done.
  const loopActive = useStore((state) => state.generating?.messageId === messageId && state.generating.phase !== undefined);
  const executing = useStore((state) => state.generating?.phase === "tools" && state.generating.messageId === messageId);
  const approval = useStore((state) => (state.approval?.callId === call.id ? state.approval : null));
  const [result, setResult] = useState("");
  const answered = conversation?.messages.some((message) => message.role === "tool" && message.toolCallId === call.id);
  const parsed = useMemo(() => parseArguments(call.arguments), [call.arguments]);
  const subject = parsed ? callSubject(parsed) : "";

  return (
    <div className={`tool${live ? " tool--live" : ""}${approval ? " tool--approval" : ""}`} dir="ltr">
      <div className="tool__head">
        <Wrench size={14} />
        <span className="tool__name">{call.name || "tool"}</span>
        {subject && <span className="tool__subject">{subject}</span>}
        <span className="tool__id">{call.id}</span>
      </div>
      <ToolArguments name={call.name} text={call.arguments} parsed={parsed} />
      {approval && <ApprovalPrompt command={approval.command} />}
      {executing && !answered && !approval && <div className="tool__answered">Running…</div>}
      {!live && !answered && !loopActive && isLast && (
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

/** The call's arguments, when the model has finished writing them. */
function parseArguments(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (!trimmed) return null;
  try {
    const value: unknown = JSON.parse(trimmed);
    return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

/** The one argument that says what a call is about, for the card's header. */
function callSubject(args: Record<string, unknown>): string {
  for (const key of ["path", "command", "url", "query"]) {
    const value = args[key];
    if (typeof value === "string" && value.trim()) return value.length > 90 ? `${value.slice(0, 90)}…` : value;
  }
  return "";
}

/**
 * A call's arguments, shaped by what they are.
 *
 * Arguments stream in a character at a time, and a half-written JSON object
 * has no formatting to give it: printing the partial text as it arrives is
 * the wall of dense text a long file write looks like. So while it streams
 * the card says how much has arrived and shows the tail; once the object
 * parses it becomes fields, with the strings big enough to matter -- a file's
 * new contents, an edit's two sides -- in their own blocks.
 */
function ToolArguments({ name, text, parsed }: { name: string; text: string; parsed: Record<string, unknown> | null }) {
  const [open, setOpen] = useState(false);
  if (!parsed) {
    return (
      <div className="tool__pending">
        <span className="tool__pending-label">Writing arguments… {text.length} chars</span>
        <code className="tool__pending-tail">{text.slice(-80)}</code>
      </div>
    );
  }
  const short = Object.entries(parsed).filter(([, value]) => typeof value !== "string" || value.length <= 120);
  const long = Object.entries(parsed).filter(([, value]) => typeof value === "string" && value.length > 120) as [string, string][];
  const edit = name.endsWith("edit_file") && typeof parsed.old_string === "string" && typeof parsed.new_string === "string";
  return (
    <div className="tool__args">
      {short.length > 0 && (
        <dl className="tool__fields">
          {short.map(([key, value]) => (
            <div className="tool__field" key={key}>
              <dt>{key}</dt>
              <dd>{typeof value === "string" ? value : JSON.stringify(value)}</dd>
            </div>
          ))}
        </dl>
      )}
      {edit ? (
        <div className="tool__diff">
          <CodeBlock label="− old_string" text={String(parsed.old_string)} tone="del" />
          <CodeBlock label="+ new_string" text={String(parsed.new_string)} tone="ins" />
        </div>
      ) : (
        long.map(([key, value]) => <CodeBlock key={key} label={key} text={value} />)
      )}
      {(long.length > 0 || edit) && (
        <button className="tool__raw-toggle" onClick={() => setOpen(!open)}>
          {open ? "Hide raw JSON" : "Show raw JSON"}
        </button>
      )}
      {open && (
        <pre className="tool__raw">
          <code>{prettyJson(text)}</code>
        </pre>
      )}
    </div>
  );
}

/** A labelled block of text, folded down to its head when it is long. */
function CodeBlock({ label, text, tone }: { label: string; text: string; tone?: "del" | "ins" }) {
  const [open, setOpen] = useState(false);
  const lines = text.split("\n");
  const folded = !open && lines.length > FOLD_LINES;
  const shown = folded ? lines.slice(0, FOLD_LINES).join("\n") : text;
  return (
    <figure className={`code-block${tone ? ` code-block--${tone}` : ""}`}>
      <figcaption>
        <span>{label}</span>
        <span className="code-block__size">
          {lines.length} line{lines.length === 1 ? "" : "s"} · {text.length} chars
        </span>
      </figcaption>
      <pre>
        <code>{shown}</code>
      </pre>
      {lines.length > FOLD_LINES && (
        <button className="code-block__toggle" onClick={() => setOpen(!open)}>
          {folded ? `Show all ${lines.length} lines` : "Collapse"}
        </button>
      )}
    </figure>
  );
}

/** Lines of a long block shown before it folds. */
const FOLD_LINES = 14;

/**
 * An agent run that stopped on these calls says why here. Without it the only
 * sign is the manual reply box, which looks like the loop simply ignored the
 * call rather than having nowhere to run it.
 */
function AgentPauseNote({ messageId }: { messageId: string }) {
  const pause = useStore((state) => (state.agentPause?.messageId === messageId ? state.agentPause : null));
  const setPanel = useStore((state) => state.setPanel);
  if (!pause) return null;
  return (
    <div className="agent-pause" role="status">
      <PauseCircle size={14} />
      <span>{pause.reason}</span>
      <button className="button button--small" onClick={() => setPanel("tools")}>
        Tools panel
      </button>
    </div>
  );
}

/**
 * The agent wants to run a shell command in the workspace. Nothing runs until
 * one of these buttons is pressed; stopping the run counts as a denial.
 */
function ApprovalPrompt({ command }: { command: string }) {
  const resolveApproval = useStore((state) => state.resolveApproval);
  const approve = useRef<HTMLButtonElement>(null);
  useEffect(() => approve.current?.focus(), []);
  return (
    <div className="approval" role="group" aria-label="Approve shell command">
      <div className="approval__head">
        <ShieldAlert size={14} />
        <span>Run this command in the agent workspace?</span>
      </div>
      <pre className="approval__command">
        <code>{command}</code>
      </pre>
      <div className="approval__actions">
        <button className="button button--small button--primary" ref={approve} onClick={() => resolveApproval("approve")}>
          <Play size={13} /> Run
        </button>
        <button className="button button--small" onClick={() => resolveApproval("approve-all")}>
          Always allow in this run
        </button>
        <button className="button button--small button--danger" onClick={() => resolveApproval("deny")}>
          <Ban size={13} /> Deny
        </button>
      </div>
    </div>
  );
}

function ToolResult({ message }: { message: Message }) {
  const deleteMessage = useStore((state) => state.deleteMessage);
  // A result is a file, a page, or a build log: it is long by nature, and
  // printing it whole buries the conversation it belongs to. The head is
  // almost always the part being read -- a listing's first entries, a
  // command's first error -- so that is what stays open.
  const text = prettyJson(message.content) || "(empty)";
  return (
    <article className="msg msg--tool" dir="ltr">
      <div className="msg__avatar" aria-hidden="true">
        <Wrench size={15} />
      </div>
      <div className="msg__body">
        <header className="msg__meta">
          <span className="msg__role">Tool result · {message.toolName ?? message.toolCallId}</span>
          <time className="msg__time">{formatTime(message.createdAt)}</time>
          {message.auto && <span className="msg__proto">auto</span>}
        </header>
        <CodeBlock label={message.toolName ?? "result"} text={text} />
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
