import { describe, expect, it } from "vitest";
import { formatEta, prefillDetail } from "./PrefillStatus";

describe("prefillDetail", () => {
  it("says how far the prompt has got, and what the cache spared", () => {
    const line = prefillDetail({ processed: 4300, total: 10600, cached: 4300, tokensPerSecond: 940, etaSeconds: 6.7 });
    expect(line).toContain("4.3K of 10.6K tokens");
    // The number that separates "this prompt is long" from "this prompt is new".
    expect(line).toContain("4.3K reused from cache");
    expect(line).toContain("940 tok/s");
    expect(line).toContain("about 7s left");
  });

  it("promises nothing it does not know", () => {
    // The first frame arrives before any of the prompt has been evaluated:
    // a total, and no rate to estimate from.
    const first = prefillDetail({ processed: 0, total: 8192, cached: 0, tokensPerSecond: 0 });
    expect(first).toBe("0 of 8.2K tokens");
    expect(first).not.toContain("left");
    expect(first).not.toContain("cache");
  });

  it("drops the rate and the estimate once the prompt is read", () => {
    const done = prefillDetail({ processed: 8192, total: 8192, cached: 1024, tokensPerSecond: 900, etaSeconds: 0 });
    expect(done).toBe("8.2K of 8.2K tokens · 1.0K reused from cache");
  });
});

describe("formatEta", () => {
  it("reads as a person would say it", () => {
    expect(formatEta(0.4)).toBe("a second");
    expect(formatEta(7.2)).toBe("7s");
    expect(formatEta(65)).toBe("1m 5s");
    expect(formatEta(120)).toBe("2m");
  });
});
