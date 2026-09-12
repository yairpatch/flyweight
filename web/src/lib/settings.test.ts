import { describe, expect, it } from "vitest";
import { availableEfforts, REASONING_EFFORTS } from "./settings";

describe("availableEfforts", () => {
  it("offers only the levels the checkpoint names", () => {
    // Qwen3.8-Flash-Next: its template reads xhigh / medium / low and raises on
    // "high", which the server clamps onto xhigh. Offering both would present
    // one behaviour as two choices.
    expect(availableEfforts({ reasoning_efforts: ["low", "medium", "xhigh"] })).toEqual([
      "auto",
      "low",
      "medium",
      "xhigh",
    ]);
  });

  it("keeps auto, which is the UI's own entry rather than a level", () => {
    expect(availableEfforts({ reasoning_efforts: ["low"] })).toEqual(["auto", "low"]);
  });

  it("falls back to the full ladder when the server reports nothing", () => {
    // An older server, or one whose checkpoint has no opinion.
    expect(availableEfforts(null)).toEqual(REASONING_EFFORTS);
    expect(availableEfforts({})).toEqual(REASONING_EFFORTS);
    expect(availableEfforts({ reasoning_efforts: [] })).toEqual(REASONING_EFFORTS);
  });

  it("falls back when it recognizes none of the reported levels", () => {
    // A vocabulary this build has never heard of would otherwise collapse the
    // picker to "auto" alone, hiding controls that do work.
    expect(availableEfforts({ reasoning_efforts: ["turbo", "glacial"] })).toEqual(REASONING_EFFORTS);
  });
});
