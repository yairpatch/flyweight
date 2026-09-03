import { describe as group, expect, it } from "vitest";
import { attachmentImages, attachmentPrelude, describe, formatBytes, readAttachment, setPdfMode } from "./attachments";
import type { Attachment, Message } from "../types";

function attachment(patch: Partial<Attachment>): Attachment {
  return { id: "a1", name: "notes.txt", kind: "text", mediaType: "text/plain", size: 12, ...patch };
}

function turn(attachments: Attachment[], content = "what does this say?"): Message {
  return { id: "m1", role: "user", content, createdAt: 0, attachments };
}

group("attachmentPrelude", () => {
  it("fences extracted text ahead of the typed message", () => {
    const prelude = attachmentPrelude(turn([attachment({ name: "main.py", language: "python", text: "print(1)" })]));
    expect(prelude).toContain("Attached file: main.py");
    expect(prelude).toContain("```python\nprint(1)\n```");
    expect(prelude.endsWith("\n\n")).toBe(true);
  });

  it("uses a longer fence than any run of backticks inside the file", () => {
    const prelude = attachmentPrelude(turn([attachment({ name: "readme.md", text: "```js\nx\n```" })]));
    expect(prelude).toContain("````\n```js\nx\n```\n````");
  });

  it("marks truncated files so the model knows text is missing", () => {
    const prelude = attachmentPrelude(turn([attachment({ text: "a\nb", truncated: { shown: 2, total: 9, unit: "lines" } })]));
    expect(prelude).toContain("[truncated: 7 more lines omitted]");
  });

  it("inlines nothing for pictures or for PDFs sent as pages", () => {
    const images = turn([attachment({ kind: "image", url: "data:image/png;base64,AA" })]);
    expect(attachmentPrelude(images)).toBe("");
    const pages = turn([attachment({ kind: "pdf", mode: "pages", text: "leftover text", pages: ["data:image/jpeg;base64,BB"] })]);
    expect(attachmentPrelude(pages)).toBe("");
  });

  it("skips attachments that failed to extract", () => {
    expect(attachmentPrelude(turn([attachment({ error: "unsupported file type", text: "x" })]))).toBe("");
  });

  it("is empty for a turn with no attachments", () => {
    expect(attachmentPrelude({ id: "m", role: "user", content: "hi", createdAt: 0 })).toBe("");
  });
});

group("attachmentImages", () => {
  it("collects pictures and rendered pages, dropping failures", () => {
    expect(
      attachmentImages([
        attachment({ id: "a", kind: "image", url: "one" }),
        attachment({ id: "b", kind: "pdf", mode: "pages", pages: ["two", "three"] }),
        attachment({ id: "c", kind: "image", url: "four", error: "no vision tower" }),
      ]),
    ).toEqual(["one", "two", "three"]);
  });
});

group("readAttachment", () => {
  it("reads a text file and reports its shape", async () => {
    const file = new File(["alpha\nbeta\n"], "notes.txt", { type: "text/plain" });
    const result = await readAttachment(file, { vision: false, contextWindow: 8192 });
    expect(result.kind).toBe("text");
    expect(result.text).toBe("alpha\nbeta");
    expect(result.truncated).toBeUndefined();
  });

  it("infers a language from the extension", async () => {
    const result = await readAttachment(new File(["x = 1"], "s.py", { type: "" }), { vision: false });
    expect(result.language).toBe("python");
  });

  it("truncates on line boundaries when the file overruns the context share", async () => {
    const body = Array.from({ length: 5000 }, (_, index) => `line ${index}`).join("\n");
    const result = await readAttachment(new File([body], "big.log", { type: "text/plain" }), { vision: false, contextWindow: 4096 });
    expect(result.truncated).toBeDefined();
    expect(result.truncated!.total).toBe(5000);
    expect(result.truncated!.shown).toBeLessThan(5000);
    expect(result.text!.split("\n").length).toBe(result.truncated!.shown);
    expect(result.text!.startsWith("line 0\n")).toBe(true);
    expect(describe(result)).toContain(`of 5000 lines`);
  });

  it("falls back to a character cut when the file is one huge line", async () => {
    const body = `{"rows":[${Array.from({ length: 20000 }, (_, index) => `"row ${index}"`).join(",")}]}`;
    const result = await readAttachment(new File([body], "bundle.json", { type: "application/json" }), { vision: false, contextWindow: 4096 });
    expect(result.truncated).toEqual({ shown: expect.any(Number), total: body.length, unit: "characters" });
    expect(result.text!.length).toBeLessThan(body.length);
    expect(body.startsWith(result.text!)).toBe(true);
  });

  it("takes an extensionless text file but refuses binary", async () => {
    const config = await readAttachment(new File(["key = value"], "gitconfig", { type: "" }), { vision: false });
    expect(config.error).toBeUndefined();
    expect(config.kind).toBe("text");

    const binary = new File([new Uint8Array([0, 1, 2, 3, 0, 255])], "blob.bin", { type: "" });
    expect((await readAttachment(binary, { vision: false })).error).toBe("unsupported file type");
  });

  it("refuses an image when no vision tower is loaded", async () => {
    const file = new File([new Uint8Array([1, 2])], "shot.png", { type: "image/png" });
    expect((await readAttachment(file, { vision: false })).error).toMatch(/vision tower/);
  });
});

group("setPdfMode", () => {
  it("drops a page-count marker when a PDF whose source is gone goes back to text", async () => {
    const pages = attachment({ kind: "pdf", mode: "pages", pageCount: 20, text: "leftover", truncated: { shown: 8, total: 20, unit: "pages" } });
    const back = await setPdfMode(pages, "text", { vision: true });
    expect(back.mode).toBe("text");
    expect(back.pages).toBeUndefined();
    expect(back.truncated).toBeUndefined();
  });

  it("refuses page rendering without a vision tower", async () => {
    const result = await setPdfMode(attachment({ kind: "pdf", mode: "text" }), "pages", { vision: false });
    expect(result.error).toMatch(/vision tower|no longer available/);
  });
});

group("formatBytes", () => {
  it("scales the unit", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(5 * 1024 * 1024)).toBe("5.0 MB");
  });
});
