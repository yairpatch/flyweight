import { describe, expect, it } from "vitest";
import { DEFAULT_SETTINGS, maxTokensCeiling, normalizeSettings, settingsFromProps } from "./settings";
import type { PropsPayload } from "../types";

/** A server started with the shipped defaults: --context 32768, --max-tokens 4096. */
const props: PropsPayload = {
  context_window: 32768,
  max_output_tokens: 4096,
  generation_defaults: { max_new_tokens: 4096, temperature: 0.7, top_k: 20, top_p: 0.8, min_p: 0 },
};

describe("maxTokensCeiling", () => {
  it("is the context window, not the server's default output length", () => {
    // --max-tokens is what a request gets when it names no limit; the server
    // honors a larger one, so offering only 4096 was the UI's own ceiling.
    expect(maxTokensCeiling(props)).toBe(32768);
    expect(maxTokensCeiling({ ...props, context_window: 131072 })).toBe(131072);
  });

  it("falls back when the server says nothing useful", () => {
    expect(maxTokensCeiling(null)).toBe(131072);
    expect(maxTokensCeiling({ max_output_tokens: 4096 })).toBe(131072);
    expect(maxTokensCeiling({ context_window: 0 })).toBe(131072);
  });
});

describe("settingsFromProps", () => {
  it("seeds max tokens from the server's default", () => {
    expect(settingsFromProps(props, DEFAULT_SETTINGS).maxTokens).toBe(4096);
    expect(settingsFromProps({ ...props, generation_defaults: { max_new_tokens: 16384 } }, DEFAULT_SETTINGS).maxTokens).toBe(16384);
  });

  it("never seeds a value the window cannot hold", () => {
    const tiny = { context_window: 2048, max_output_tokens: 4096, generation_defaults: { max_new_tokens: 4096 } };
    expect(settingsFromProps(tiny, DEFAULT_SETTINGS).maxTokens).toBe(2048);
  });

  it("uses max_output_tokens only when there are no generation defaults", () => {
    expect(settingsFromProps({ context_window: 32768, max_output_tokens: 8192 }, DEFAULT_SETTINGS).maxTokens).toBe(8192);
    expect(settingsFromProps(null, DEFAULT_SETTINGS).maxTokens).toBe(DEFAULT_SETTINGS.maxTokens);
  });

  it("takes the sampling defaults the server reports", () => {
    const seeded = settingsFromProps(props, DEFAULT_SETTINGS);
    expect(seeded.temperature).toBe(0.7);
    expect(seeded.topK).toBe(20);
    expect(seeded.customized).toBe(false);
  });
});

describe("normalizeSettings", () => {
  it("keeps a max tokens the user raised past the server's default", () => {
    // The setting the user changed is the request's own max_tokens, and the
    // server honors it; clamping here would undo the fix above.
    expect(normalizeSettings({ maxTokens: 32768 }).maxTokens).toBe(32768);
    expect(normalizeSettings({ maxTokens: 0 }).maxTokens).toBe(1);
  });
});
