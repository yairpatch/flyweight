import { describe, expect, it } from "vitest";
import { readSse } from "./sse";
import { detectDirection } from "./direction";
import { splitStable } from "../components/Markdown";
import { throughputTrend } from "../components/MessageItem";

function stream(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

async function collect(chunks: string[]) {
  const frames = [];
  for await (const frame of readSse(stream(chunks))) frames.push(frame);
  return frames;
}

describe("readSse", () => {
  it("splits frames across chunk boundaries, tolerates CRLF and keepalive comments", async () => {
    const frames = await collect(["event: a\r\ndata: {\"x\":1}\r\n\r\n: keepalive\n\ndata: pa", "rt\ndata: two\n\n", "data: tail"]);
    expect(frames).toHaveLength(3);
    expect(frames[0]).toMatchObject({ event: "a", data: '{"x":1}' });
    expect(frames[1]).toMatchObject({ data: "part\ntwo" });
    expect(frames[2]).toMatchObject({ data: "tail" });
  });

  it("does not split a multibyte character across chunks", async () => {
    const bytes = new TextEncoder().encode("data: שלום\n\n");
    const decoder = new TextDecoder();
    const first = decoder.decode(bytes.slice(0, 8));
    // Rebuild from raw bytes to exercise the streaming decoder.
    const chunks: Uint8Array[] = [bytes.slice(0, 8), bytes.slice(8)];
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    });
    const frames = [];
    for await (const frame of readSse(body)) frames.push(frame);
    expect(first).not.toBe("data: שלום");
    expect(frames[0].data).toBe("שלום");
  });
});

describe("detectDirection", () => {
  it("weighs the whole sample instead of the first strong character", () => {
    expect(detectDirection("HTTP שלום עולם מה שלומך היום חברים")).toBe("rtl");
    expect(detectDirection("שלום everyone, this is a mostly english sentence")).toBe("ltr");
    expect(detectDirection("ok")).toBe("ltr");
    expect(detectDirection("```\nשלום שלום שלום שלום\n```")).toBe("ltr");
  });
});

describe("splitStable", () => {
  it("keeps an open fence in the live tail", () => {
    const [stable, tail] = splitStable("para one\n\n```js\nlet a\n\nlet b");
    expect(stable).toBe("para one\n\n");
    expect(tail).toBe("```js\nlet a\n\nlet b");
  });
  it("returns everything as tail without a blank line", () => {
    expect(splitStable("just text")).toEqual(["", "just text"]);
  });
});

describe("throughputTrend", () => {
  it("buckets samples into ~250ms windows", () => {
    const samples: Array<[number, number]> = [[0, 0], [100, 2], [260, 5], [520, 10], [800, 12]];
    const trend = throughputTrend(samples);
    expect(trend).toHaveLength(3);
    expect(trend[0]).toBeCloseTo(5000 / 260, 3);
  });
});
