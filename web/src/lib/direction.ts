// Weighted text-direction detection. `dir="auto"` keys off the first strong
// character, which flips a Hebrew answer that opens with a Latin identifier.
// Instead we count strong letters over a sample and flip only when the RTL
// share is decisive, with a small confidence floor so a stray word does not
// swing an empty message.

const RTL_LETTERS = /[֐-׿؀-ۿݐ-ݿࢠ-ࣿיִ-﷿ﹰ-﻿]/g;
const LTR_LETTERS = /[A-Za-zÀ-ɏͰ-ϿЀ-ӿ԰-֏぀-ヿ一-鿿가-힯]/g;

const SAMPLE_CHARS = 400;
const RTL_RATIO = 0.34;
const MIN_LETTERS = 8;

export type Direction = "ltr" | "rtl";

function stripCode(text: string): string {
  return text.replace(/```[\s\S]*?```/g, " ").replace(/`[^`\n]*`/g, " ");
}

export function detectDirection(text: string, fallback: Direction = "ltr"): Direction {
  const sample = stripCode(text).slice(0, SAMPLE_CHARS);
  const rtl = (sample.match(RTL_LETTERS) ?? []).length;
  const ltr = (sample.match(LTR_LETTERS) ?? []).length;
  const total = rtl + ltr;
  if (total < MIN_LETTERS) return fallback;
  return rtl / total >= RTL_RATIO ? "rtl" : "ltr";
}
