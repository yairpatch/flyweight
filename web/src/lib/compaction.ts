// Keeping an agent run inside the context window.
//
// A loop grows its own prompt: every command's output and every file it reads
// comes back as a tool result and stays there for the rest of the run. Left
// alone the run dies on "prompt is too long" a few turns in, with the failure
// landing on whichever step happened to cross the line.
//
// Compaction here is mechanical and costs no model call: old tool results
// shrink to a line saying what was dropped, and whole old steps go if that is
// not enough. It applies to the request only — the transcript keeps every
// message, so the user can still read what the agent actually did.
import type { Message, RequestRecord } from "../types";

/**
 * Characters per token, deliberately low. Guessing small compacts a little
 * early rather than a little late, and the server's own error is the backstop
 * when the guess is wrong anyway.
 */
const CHARS_PER_TOKEN = 3.5;
/** Recent messages are never touched: the model needs its last steps intact. */
const RECENT_KEEP = 6;
/** Tool results shorter than this cost less than the stub explaining them. */
const STUB_MIN = 300;
/** How much of a result survives when even the recent ones must be clipped. */
const CLIP_CHARS = 2000;
/**
 * Room left for everything in the request that is not the messages, when the
 * caller cannot measure it: the system prompt, the tool schemas, and the chat
 * template's own text. A caller that knows what it is about to send should
 * pass the real figure -- an agent run's prompt and six tool schemas are well
 * over this, and a budget that ignores them overflows the window it was
 * supposed to fit.
 */
const OVERHEAD_TOKENS = 1500;

export interface CompactionOutcome {
  /** The messages to send; the same array when nothing had to go. */
  messages: Message[];
  /** Prompt characters removed. */
  removedChars: number;
  /** Tool results replaced by a stub. */
  stubbed: number;
  /** Whole messages dropped. */
  dropped: number;
}

export function messageChars(message: Message): number {
  const attachments = (message.attachments ?? []).reduce((total, item) => total + (item.text?.length ?? 0), 0);
  const calls = (message.toolCalls ?? []).reduce((total, call) => total + call.name.length + call.arguments.length, 0);
  return message.content.length + attachments + calls;
}

export function conversationChars(messages: Message[]): number {
  return messages.reduce((total, message) => total + messageChars(message), 0);
}

/**
 * The character budget the messages of a run may occupy, from the model's
 * context window minus what the answer and the scaffolding need. Infinite when
 * the window is unknown, which leaves the server's error as the only trigger.
 */
export function budgetChars(
  contextWindow: number | undefined,
  maxOutputTokens: number,
  charsPerToken = CHARS_PER_TOKEN,
  overheadTokens = OVERHEAD_TOKENS,
): number {
  if (!contextWindow || contextWindow <= 0) return Number.POSITIVE_INFINITY;
  const room = contextWindow - Math.max(0, maxOutputTokens) - Math.max(0, overheadTokens);
  return Math.max(2000, room * Math.max(1, charsPerToken));
}

/**
 * What the non-message part of a request will cost, from the text of it. The
 * template wraps each tool and each turn in markup of its own, which no
 * client can see, so the count is rounded up rather than taken at face value.
 */
export function overheadTokens(chars: number, charsPerToken = CHARS_PER_TOKEN): number {
  return Math.ceil(chars / Math.max(1, charsPerToken)) + 200;
}

/**
 * What the last request's prompt actually measured, from the characters sent
 * and the tokens the server counted. It runs a little low — the prompt also
 * carries the system text and tool schemas, which are not in `sentChars` — and
 * low is the safe direction for a budget.
 */
export function observedCharsPerToken(sentChars: number, promptTokens: number | undefined): number | null {
  if (!promptTokens || promptTokens < 200 || sentChars <= 0) return null;
  const ratio = sentChars / promptTokens;
  return ratio >= 1 && ratio <= 12 ? ratio : null;
}

/**
 * Fit `messages` into `budget` characters. The task that opened the run and
 * the last few messages are never touched; between them, big tool results are
 * stubbed first because they are the bulk of a run, and only then are whole
 * steps dropped. An assistant turn leaves together with the tool results that
 * answer it: a result whose call is gone is a malformed request.
 */
export function compactMessages(messages: Message[], budget: number): CompactionOutcome {
  const total = conversationChars(messages);
  if (!Number.isFinite(budget) || total <= budget) return { messages, removedChars: 0, stubbed: 0, dropped: 0 };

  const firstUser = messages.findIndex((message) => message.role === "user");
  const keepFrom = Math.max(0, messages.length - RECENT_KEEP);
  const touchable = (index: number) => index < keepFrom && index !== firstUser && messages[index].role !== "system";
  const working = messages.slice();
  let removed = 0;
  let stubbed = 0;
  let dropped = 0;

  for (let index = 0; index < working.length && total - removed > budget; index += 1) {
    const message = working[index];
    if (!touchable(index) || message.role !== "tool" || message.content.length < STUB_MIN) continue;
    const stub = `[${message.toolName ?? "tool"} result removed to fit the context window: ${message.content.length} characters]`;
    removed += message.content.length - stub.length;
    working[index] = { ...message, content: stub };
    stubbed += 1;
  }

  if (total - removed > budget) {
    const doomed = new Set<string>();
    for (let index = 0; index < working.length && total - removed > budget; index += 1) {
      if (!touchable(index) || working[index].role === "tool") continue;
      const group = [index];
      const calls = new Set((working[index].toolCalls ?? []).map((call) => call.id));
      for (let next = index + 1; next < working.length; next += 1) {
        const candidate = working[next];
        if (candidate.role !== "tool" || !candidate.toolCallId || !calls.has(candidate.toolCallId)) break;
        group.push(next);
      }
      // A step is dropped whole or not at all, so a kept result never loses
      // the call it answers.
      if (group.some((position) => !touchable(position))) continue;
      for (const position of group) {
        doomed.add(working[position].id);
        removed += messageChars(working[position]);
        dropped += 1;
      }
      index = group[group.length - 1];
    }
    if (doomed.size) {
      const kept = working.filter((message) => !doomed.has(message.id));
      return clipRecent({ messages: kept, removedChars: removed, stubbed, dropped }, conversationChars(kept), budget);
    }
  }

  return clipRecent({ messages: working, removedChars: removed, stubbed, dropped }, total - removed, budget);
}

/**
 * Last resort: one result — a large file, a verbose build — is bigger than the
 * window on its own, so even the recent messages have to give. Clipping keeps
 * the head of each, where a listing's first entries and a command's error live,
 * instead of letting the run die on a prompt that cannot be sent.
 */
function clipRecent(outcome: CompactionOutcome, current: number, budget: number): CompactionOutcome {
  if (current <= budget) return outcome;
  const messages = outcome.messages.slice();
  let { removedChars, stubbed, dropped } = outcome;
  let size = current;
  for (let index = 0; index < messages.length && size > budget; index += 1) {
    const message = messages[index];
    if (message.role !== "tool" || message.content.length <= CLIP_CHARS) continue;
    const gone = message.content.length - CLIP_CHARS;
    const clipped = `${message.content.slice(0, CLIP_CHARS)}\n[${gone} more characters removed to fit the context window]`;
    size -= message.content.length - clipped.length;
    removedChars += message.content.length - clipped.length;
    messages[index] = { ...message, content: clipped };
    stubbed += 1;
  }
  return { messages, removedChars, stubbed, dropped };
}

/** What the model is told about the gap, appended to the run's system prompt. */
export function compactionNote(outcome: CompactionOutcome): string {
  if (!outcome.removedChars) return "";
  const parts: string[] = [];
  if (outcome.stubbed) parts.push(`${outcome.stubbed} earlier tool result${outcome.stubbed === 1 ? "" : "s"}`);
  if (outcome.dropped) parts.push(`${outcome.dropped} earlier message${outcome.dropped === 1 ? "" : "s"}`);
  return `Some of this run has been removed to fit the context window (${parts.join(" and ")}). Read a file or re-run a command if you need something that is no longer shown.`;
}

/** The server's numbers when a request died on a full context window. */
export interface ContextOverflow {
  /** Tokens the prompt needed, when the message says. */
  promptTokens?: number;
  /** The window it exceeded, when the message says. */
  contextWindow?: number;
}

/**
 * Whether this request failed because the prompt did not fit. The server sends
 * OpenAI's `context_length_exceeded` code and Anthropic's "prompt is too long"
 * prose; other runtimes send one or the other, so both are matched.
 */
export function contextOverflow(record: RequestRecord): ContextOverflow | null {
  const message = record.error ?? "";
  const matched = record.errorCode === "context_length_exceeded" || /prompt is too long|context length|too many tokens/i.test(message);
  if (!matched) return null;
  const numbers = /prompt is too long: (\d+) tokens > (\d+)/.exec(message);
  if (!numbers) return {};
  return { promptTokens: Number(numbers[1]), contextWindow: Number(numbers[2]) };
}

/**
 * The budget to retry an overflowed request on. The server said how many
 * tokens the prompt was, and we know how many characters we sent, so the real
 * ratio for this model replaces the guess; three quarters of the window leaves
 * room for the answer.
 */
export function retryBudget(overflow: ContextOverflow, sentChars: number, previous: number): number {
  if (overflow.promptTokens && overflow.contextWindow && sentChars > 0) {
    const charsPerToken = sentChars / overflow.promptTokens;
    return Math.max(2000, overflow.contextWindow * 0.75 * charsPerToken);
  }
  return Math.max(2000, (Number.isFinite(previous) ? previous : sentChars) / 2);
}
