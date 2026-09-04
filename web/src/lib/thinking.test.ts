import { describe, expect, it } from "vitest";
import { holdPartialTag, splitThinking } from "./thinking";

describe("splitThinking", () => {
  it("splits standard <think> blocks", () => {
    const res = splitThinking("<think>mulling things over</think>Hello world");
    expect(res).toEqual({
      reasoning: "mulling things over",
      answer: "Hello world",
      open: false,
    });
  });

  it("handles streaming unclosed <think>", () => {
    const res = splitThinking("<think>mulling things over");
    expect(res).toEqual({
      reasoning: "mulling things over",
      answer: "",
      open: true,
    });
  });

  it("splits K2-Horizon <ifm|think> blocks", () => {
    const res = splitThinking("<ifm|think>calculating 2+2</ifm|think>The answer is 4.");
    expect(res).toEqual({
      reasoning: "calculating 2+2",
      answer: "The answer is 4.",
      open: false,
    });
  });

  it("splits K2-Horizon <ifm|think_fast> and <ifm|think_faster> blocks", () => {
    expect(splitThinking("<ifm|think_fast>quick plan</ifm|think_fast>Done")).toEqual({
      reasoning: "quick plan",
      answer: "Done",
      open: false,
    });
    expect(splitThinking("<ifm|think_faster>fast plan</ifm|think_faster>Done")).toEqual({
      reasoning: "fast plan",
      answer: "Done",
      open: false,
    });
  });

  it("handles streaming unclosed <ifm|think>", () => {
    expect(splitThinking("<ifm|think>thinking in progress")).toEqual({
      reasoning: "thinking in progress",
      answer: "",
      open: true,
    });
  });

  it("handles bare closing tags when prompt opened thinking", () => {
    expect(splitThinking("deliberating options\n</think>\nFinal answer")).toEqual({
      reasoning: "deliberating options",
      answer: "Final answer",
      open: false,
    });
    expect(splitThinking("analyzing math\n</ifm|think>\n42")).toEqual({
      reasoning: "analyzing math",
      answer: "42",
      open: false,
    });
    expect(splitThinking("</ifm|think>Direct answer")).toEqual({
      reasoning: undefined,
      answer: "Direct answer",
      open: false,
    });
  });

  it("leaves text without thinking tags untouched", () => {
    expect(splitThinking("Just a normal response")).toEqual({
      reasoning: undefined,
      answer: "Just a normal response",
      open: false,
    });
  });
});

describe("holdPartialTag", () => {
  it("holds back partial closing tags", () => {
    expect(holdPartialTag("Hello </th")).toBe("Hello ");
    expect(holdPartialTag("Hello </ifm|")).toBe("Hello ");
    expect(holdPartialTag("Hello </ifm|think_fa")).toBe("Hello ");
  });

  it("holds back partial opening tags", () => {
    expect(holdPartialTag("<th")).toBe("");
    expect(holdPartialTag("<ifm|th")).toBe("");
  });

  it("does not hold back complete tags or normal text", () => {
    expect(holdPartialTag("Hello world")).toBe("Hello world");
    expect(holdPartialTag("Look at 5 < 10")).toBe("Look at 5 < 10");
  });
});
