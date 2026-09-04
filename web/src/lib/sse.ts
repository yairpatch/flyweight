// Server-sent events reader over a fetch body. Tolerates CRLF, comment
// lines (keepalives), multi-line data, and flushes a trailing frame.

export interface SseFrame {
  event?: string;
  data: string;
  /** The raw text of the frame, for the request inspector. */
  raw: string;
}

export async function* readSse(
  body: ReadableStream<Uint8Array>,
  onRaw?: (line: string) => void,
): AsyncGenerator<SseFrame> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const parseFrame = (block: string): SseFrame | null => {
    const lines = block.split("\n");
    let event: string | undefined;
    const data: string[] = [];
    for (const line of lines) {
      if (!line || line.startsWith(":")) continue;
      const colon = line.indexOf(":");
      const field = colon === -1 ? line : line.slice(0, colon);
      let value = colon === -1 ? "" : line.slice(colon + 1);
      if (value.startsWith(" ")) value = value.slice(1);
      if (field === "event") event = value;
      else if (field === "data") data.push(value);
    }
    if (data.length === 0) return null;
    return { event, data: data.join("\n"), raw: block };
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      buffer = buffer.replace(/\r\n/g, "\n");
      let boundary: number;
      while ((boundary = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        onRaw?.(block);
        const frame = parseFrame(block);
        if (frame) yield frame;
      }
    }
    buffer += decoder.decode();
    const tail = buffer.trim();
    if (tail) {
      onRaw?.(tail);
      const frame = parseFrame(tail);
      if (frame) yield frame;
    }
  } finally {
    reader.releaseLock();
  }
}
