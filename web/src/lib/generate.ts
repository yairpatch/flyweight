// Drives one streaming generation: opens the request, parses frames through
// the protocol adapter, and reports unified events plus the raw wire log.
import type { Protocol, RequestRecord, StreamEvent } from "../types";
import { ApiError, openStream } from "./api";
import { PROTOCOL_URLS, parseFrame, responseIdFromFrame } from "./protocols";
import { readSse } from "./sse";
import { identifier } from "./format";

export interface GenerateOptions {
  protocol: Protocol;
  body: Record<string, unknown>;
  signal: AbortSignal;
  onEvent: (event: StreamEvent) => void;
  onRecord?: (record: RequestRecord) => void;
  onResponseId?: (id: string) => void;
}

const RAW_LOG_LIMIT = 4000;

export async function generate(options: GenerateOptions): Promise<RequestRecord> {
  const url = PROTOCOL_URLS[options.protocol];
  const record: RequestRecord = {
    id: identifier("req"),
    at: Date.now(),
    protocol: options.protocol,
    url,
    body: options.body,
    rawEvents: [],
  };
  const started = performance.now();
  const pushRaw = (line: string) => {
    if (record.rawEvents.length < RAW_LOG_LIMIT) record.rawEvents.push({ at: performance.now() - started, line });
  };
  options.onRecord?.(record);
  try {
    const response = await openStream(url, options.body, options.signal);
    record.status = response.status;
    for await (const frame of readSse(response.body!, pushRaw)) {
      if (options.signal.aborted) break;
      const responseId = responseIdFromFrame(frame);
      if (responseId) options.onResponseId?.(responseId);
      for (const event of parseFrame(options.protocol, frame)) options.onEvent(event);
    }
  } catch (error) {
    if (options.signal.aborted) {
      // A user stop is not an error.
    } else if (error instanceof ApiError) {
      record.status = error.status;
      record.error = error.message;
      options.onEvent({ type: "error", message: error.message });
    } else {
      const message = error instanceof Error ? error.message : String(error);
      record.error = message;
      options.onEvent({ type: "error", message: /fetch/i.test(message) ? "Could not reach the server" : message });
    }
  } finally {
    record.durationMs = performance.now() - started;
    options.onRecord?.(record);
  }
  return record;
}
