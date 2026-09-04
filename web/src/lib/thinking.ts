// Legacy inline thinking: some stored turns (and servers that do not split
// reasoning) carry `<think>…</think>` or `<ifm|think>…</ifm|think>` inside the content.
// Split it out so the answer never shows the tags, and hold back a partial tag while
// streaming so half of one never flashes.

export interface ThinkingSplit {
  reasoning: string | undefined;
  answer: string;
  /** True while inside an unterminated think block. */
  open: boolean;
}

const OPEN_TAGS = ["<think>", "<ifm|think>", "<ifm|think_fast>", "<ifm|think_faster>"];
const CLOSE_TAGS = ["</think>", "</ifm|think>", "</ifm|think_fast>", "</ifm|think_faster>"];

const ALL_TAGS = [...CLOSE_TAGS, ...OPEN_TAGS].sort((a, b) => b.length - a.length);

export function splitThinking(content: string): ThinkingSplit {
  // Find earliest open tag
  let earliestOpen = -1;
  let matchingOpenTag = "";
  for (const tag of OPEN_TAGS) {
    const idx = content.indexOf(tag);
    if (idx !== -1 && (earliestOpen === -1 || idx < earliestOpen)) {
      earliestOpen = idx;
      matchingOpenTag = tag;
    }
  }

  if (earliestOpen !== -1 && content.slice(0, earliestOpen).trim() === "") {
    const bodyStart = earliestOpen + matchingOpenTag.length;
    let earliestClose = -1;
    let matchingCloseTag = "";
    for (const tag of CLOSE_TAGS) {
      const idx = content.indexOf(tag, bodyStart);
      if (idx !== -1 && (earliestClose === -1 || idx < earliestClose)) {
        earliestClose = idx;
        matchingCloseTag = tag;
      }
    }

    if (earliestClose === -1) {
      return { reasoning: content.slice(bodyStart), answer: "", open: true };
    }
    const answer = content.slice(earliestClose + matchingCloseTag.length).replace(/^\s+/, "");
    return { reasoning: content.slice(bodyStart, earliestClose).trim(), answer, open: false };
  }

  // Bare closing tag (when the prompt opened thinking and only the closing tag appears in output)
  let earliestClose = -1;
  let matchingCloseTag = "";
  for (const tag of CLOSE_TAGS) {
    const idx = content.indexOf(tag);
    if (idx !== -1 && (earliestClose === -1 || idx < earliestClose)) {
      earliestClose = idx;
      matchingCloseTag = tag;
    }
  }

  if (earliestClose !== -1) {
    const prefix = content.slice(0, earliestClose);
    if (!OPEN_TAGS.some((tag) => prefix.includes(tag))) {
      const answer = content.slice(earliestClose + matchingCloseTag.length).replace(/^\s+/, "");
      return { reasoning: prefix.trim() || undefined, answer, open: false };
    }
  }

  return { reasoning: undefined, answer: content, open: false };
}

/** Trim a trailing fragment of a think tag so it is not rendered mid-stream. */
export function holdPartialTag(text: string): string {
  for (const tag of ALL_TAGS) {
    for (let length = tag.length - 1; length > 0; length -= 1) {
      if (text.endsWith(tag.slice(0, length))) return text.slice(0, -length);
    }
  }
  return text;
}
