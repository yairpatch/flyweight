// Legacy inline thinking: some stored turns (and servers that do not split
// reasoning) carry `<think>…</think>` inside the content. Split it out so the
// answer never shows the tags, and hold back a partial tag while streaming
// so half of one never flashes.

export interface ThinkingSplit {
  reasoning: string | undefined;
  answer: string;
  /** True while inside an unterminated think block. */
  open: boolean;
}

const OPEN = "<think>";
const CLOSE = "</think>";

export function splitThinking(content: string): ThinkingSplit {
  const start = content.indexOf(OPEN);
  if (start === -1 || content.slice(0, start).trim() !== "") {
    return { reasoning: undefined, answer: content, open: false };
  }
  const bodyStart = start + OPEN.length;
  const end = content.indexOf(CLOSE, bodyStart);
  if (end === -1) {
    return { reasoning: content.slice(bodyStart), answer: "", open: true };
  }
  const answer = content.slice(end + CLOSE.length).replace(/^\s+/, "");
  return { reasoning: content.slice(bodyStart, end).trim(), answer, open: false };
}

/** Trim a trailing fragment of a think tag so it is not rendered mid-stream. */
export function holdPartialTag(text: string): string {
  for (const tag of [CLOSE, OPEN]) {
    for (let length = tag.length - 1; length > 0; length -= 1) {
      if (text.endsWith(tag.slice(0, length))) return text.slice(0, -length);
    }
  }
  return text;
}
