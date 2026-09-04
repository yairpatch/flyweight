// Composer attachments. Images ride the vision path unchanged; every other
// file is extracted to text in the browser and inlined into the user turn,
// because the server only accepts text and image content parts.
import { api } from "./api";
import { readImageFile } from "./images";
import type { Attachment, AttachmentKind, Message } from "../types";

export const MAX_ATTACHMENTS = 8;
/** Bigger than this is refused outright: reading it would hang the tab. */
export const MAX_FILE_BYTES = 32 * 1024 * 1024;
/** Share of the context window all attached text may occupy, per file. */
const CONTEXT_SHARE = 0.25;
const FALLBACK_BUDGET_TOKENS = 8192;
/** Rough bytes per token, used when the server cannot be reached. */
const CHARS_PER_TOKEN = 3.5;

const TEXT_EXTENSIONS: Record<string, string> = {
  bash: "bash", c: "c", cc: "cpp", cfg: "ini", cjs: "javascript", conf: "ini", cpp: "cpp",
  cs: "csharp", css: "css", csv: "csv", cu: "cpp", cuh: "cpp", diff: "diff", env: "ini",
  go: "go", gradle: "groovy", h: "c", hpp: "cpp", htm: "html", html: "html", ini: "ini",
  java: "java", js: "javascript", json: "json", jsonl: "json", jsx: "jsx", kt: "kotlin",
  less: "less", log: "", lua: "lua", m: "objectivec", md: "markdown", mdx: "markdown",
  mjs: "javascript", mm: "objectivec", patch: "diff", php: "php", pl: "perl", proto: "protobuf",
  ps1: "powershell", py: "python", pyi: "python", r: "r", rb: "ruby", rs: "rust", rst: "rst",
  scala: "scala", scss: "scss", sh: "bash", sql: "sql", srt: "", svg: "xml", swift: "swift",
  tex: "latex", toml: "toml", ts: "typescript", tsv: "tsv", tsx: "tsx", txt: "", vtt: "",
  vue: "vue", xml: "xml", yaml: "yaml", yml: "yaml", zsh: "bash",
};

const PDF_TYPES = ["application/pdf"];
const DOCX_TYPES = [
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
];
const SHEET_TYPES = [
  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  "application/vnd.ms-excel.sheet.macroEnabled.12",
];

/** What the file picker advertises; drag-and-drop still takes anything. */
export const ACCEPT_ATTRIBUTE = [
  "image/*",
  ".pdf",
  ".docx",
  ".xlsx",
  ".xlsm",
  "text/*",
  ...Object.keys(TEXT_EXTENSIONS).map((extension) => `.${extension}`),
].join(",");

// The source File stays reachable while the attachment is pending, so a PDF
// can be re-read when the user flips it between text and page images. Sent
// attachments drop out of the map; nothing re-reads them after the fact.
const sources = new Map<string, File>();

export function forgetSources(keep: Attachment[] = []): void {
  const live = new Set(keep.map((item) => item.id));
  for (const id of Array.from(sources.keys())) if (!live.has(id)) sources.delete(id);
}

function extensionOf(name: string): string {
  const match = /\.([A-Za-z0-9]+)$/.exec(name);
  return match ? match[1].toLowerCase() : "";
}

export function classify(file: File): AttachmentKind | "unknown" {
  const extension = extensionOf(file.name);
  if (file.type.startsWith("image/")) return "image";
  if (PDF_TYPES.includes(file.type) || extension === "pdf") return "pdf";
  if (DOCX_TYPES.includes(file.type) || extension === "docx") return "document";
  if (SHEET_TYPES.includes(file.type) || extension === "xlsx" || extension === "xlsm") return "sheet";
  if (extension in TEXT_EXTENSIONS) return "text";
  if (file.type.startsWith("text/")) return "text";
  if (/^application\/(json|xml|x-yaml|yaml|javascript|x-sh|toml|sql)/.test(file.type)) return "text";
  return "unknown";
}

/**
 * Decide whether an unrecognised file is really text. Anything that decodes as
 * UTF-8 without NULs and is mostly printable gets treated as a text file, so
 * extensionless configs and dotfiles still attach.
 */
async function looksTextual(file: File): Promise<boolean> {
  const head = new Uint8Array(await file.slice(0, 64 * 1024).arrayBuffer());
  if (!head.length) return true;
  try {
    const text = new TextDecoder("utf-8", { fatal: true }).decode(head);
    let control = 0;
    for (const character of text) {
      const code = character.charCodeAt(0);
      if (code === 0) return false;
      if (code < 9 || (code > 13 && code < 32)) control += 1;
    }
    return control / text.length < 0.01;
  } catch {
    return false;
  }
}

export interface ReadContext {
  /** True when a vision tower is attached; gates images and PDF page mode. */
  vision: boolean;
  contextWindow?: number;
}

function budgetFor(context: ReadContext): number {
  const window = context.contextWindow;
  if (!window || !Number.isFinite(window)) return FALLBACK_BUDGET_TOKENS;
  return Math.max(1024, Math.floor(window * CONTEXT_SHARE));
}

let counter = 0;
function identifier(): string {
  counter += 1;
  return `att-${Date.now().toString(36)}-${counter}`;
}

export async function readAttachment(file: File, context: ReadContext): Promise<Attachment> {
  const base: Attachment = {
    id: identifier(),
    name: file.name || "attachment",
    kind: "text",
    mediaType: file.type || "application/octet-stream",
    size: file.size,
  };
  sources.set(base.id, file);

  if (file.size > MAX_FILE_BYTES) {
    return { ...base, error: `too large (${formatBytes(file.size)}); the limit is ${formatBytes(MAX_FILE_BYTES)}` };
  }

  let kind = classify(file);
  if (kind === "unknown") kind = (await looksTextual(file)) ? "text" : "unknown";
  if (kind === "unknown") {
    return { ...base, error: "unsupported file type" };
  }

  try {
    if (kind === "image") {
      if (!context.vision) return { ...base, kind, error: "no vision tower is loaded (--mmproj)" };
      return { ...base, kind, url: await readImageFile(file) };
    }
    if (kind === "pdf") return await readPdf(base, file, context);
    if (kind === "document") {
      const markdown = await (await import("./extract/docx")).docxToMarkdown(file);
      return withText({ ...base, kind, language: "markdown" }, markdown, context);
    }
    if (kind === "sheet") {
      const { workbookToTables, tableToCsv } = await import("./extract/sheet");
      const tables = await workbookToTables(file);
      const body = tables
        .map((table) => {
          const omitted = table.omittedRows ? `\n... ${table.omittedRows} more rows` : "";
          return `# ${table.name} (${table.rows.length} rows)\n${tableToCsv(table.rows)}${omitted}`;
        })
        .join("\n\n");
      return withText({ ...base, kind, language: "csv" }, body, context);
    }
    return withText({ ...base, kind: "text", language: TEXT_EXTENSIONS[extensionOf(file.name)] ?? "" }, await file.text(), context);
  } catch (error) {
    return { ...base, kind, error: error instanceof Error ? error.message : "could not read this file" };
  }
}

/** One page-marked block per page, plus how much text there was to find. */
async function pdfBody(file: File): Promise<{ body: string; characters: number; pageCount: number }> {
  const { pdfText } = await import("./extract/pdf");
  const { pages, pageCount } = await pdfText(file);
  return {
    body: pages.map((text, index) => `--- page ${index + 1} ---\n${text}`).join("\n\n"),
    characters: pages.join("").trim().length,
    pageCount,
  };
}

async function readPdf(base: Attachment, file: File, context: ReadContext): Promise<Attachment> {
  const { body, characters, pageCount } = await pdfBody(file);
  // A scan has a near-empty text layer; the rendered pages are the only way
  // to read it, and that needs a vision tower.
  const scanned = characters / Math.max(1, pageCount) < 100;
  const attachment: Attachment = { ...base, kind: "pdf", pageCount, mode: "text", language: "" };
  const extracted = characters ? await withText(attachment, body, context) : attachment;
  if (scanned) {
    if (context.vision) return setPdfMode(extracted, "pages", context);
    if (!characters) {
      return { ...attachment, error: "no text layer; start the server with --mmproj to read the pages" };
    }
  }
  return extracted;
}

/** Flip a PDF between its text layer and rendered page images. */
export async function setPdfMode(attachment: Attachment, mode: "text" | "pages", context: ReadContext): Promise<Attachment> {
  if (attachment.kind !== "pdf") return attachment;
  const source = sources.get(attachment.id);
  if (mode === "text") {
    // The page count that was truncated describes the images, not the text, so
    // the text layer is re-read rather than carrying a marker that now lies.
    if (!source) {
      const stale = attachment.truncated?.unit === "pages";
      return { ...attachment, mode, pages: undefined, truncated: stale ? undefined : attachment.truncated };
    }
    const { body, characters } = await pdfBody(source);
    const base = { ...attachment, mode, pages: undefined, truncated: undefined, error: undefined };
    return characters ? withText(base, body, context) : { ...base, error: "no text layer in this PDF" };
  }
  const file = source;
  if (!file) return { ...attachment, error: "the original file is no longer available" };
  if (!context.vision) return { ...attachment, error: "no vision tower is loaded (--mmproj)" };
  try {
    const { pdfPageImages } = await import("./extract/pdf");
    const images = await pdfPageImages(file);
    const total = attachment.pageCount ?? images.length;
    return {
      ...attachment,
      mode,
      pages: images,
      error: undefined,
      truncated: total > images.length ? { shown: images.length, total, unit: "pages" } : undefined,
    };
  } catch (error) {
    return { ...attachment, error: error instanceof Error ? error.message : "could not render this PDF" };
  }
}

async function withText(attachment: Attachment, text: string, context: ReadContext): Promise<Attachment> {
  const trimmed = text.trim();
  if (!trimmed) return { ...attachment, error: "no readable text in this file" };
  const fitted = await fitToBudget(trimmed, budgetFor(context));
  return { ...attachment, text: fitted.text, truncated: fitted.truncated };
}

interface Fitted {
  text: string;
  truncated?: { shown: number; total: number; unit: string };
}

/**
 * Cut `text` to a token budget, on a line boundary whenever there is one. The
 * count comes from the server's tokenizer when it is reachable and from a
 * bytes-per-token estimate when it is not; either way the caller gets a visible
 * truncation marker.
 */
async function fitToBudget(text: string, budget: number): Promise<Fitted> {
  // Never POST a whole book to /tokenize just to learn it is too long.
  let cut = cutTo(text, Math.ceil(budget * CHARS_PER_TOKEN * 4));
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const tokens = await countTokens(cut.text);
    if (tokens <= budget) break;
    const next = cutTo(cut.text, Math.max(1, Math.floor(cut.text.length * (budget / tokens) * 0.92)));
    if (next.text === cut.text) break;
    cut = next;
  }
  if (cut.text.length >= text.length) return { text };
  return {
    text: cut.text,
    truncated: cut.byCharacter
      ? { shown: cut.text.length, total: text.length, unit: "characters" }
      : { shown: countLines(cut.text), total: countLines(text), unit: "lines" },
  };
}

/**
 * The longest prefix of `text` within `characters`, ending on a line boundary.
 * A minified file is one enormous line with no boundary to cut on, so it falls
 * back to a character cut and says so.
 */
function cutTo(text: string, characters: number): { text: string; byCharacter: boolean } {
  if (text.length <= characters) return { text, byCharacter: false };
  const slice = text.slice(0, characters);
  const boundary = slice.lastIndexOf("\n");
  if (boundary > 0) return { text: slice.slice(0, boundary), byCharacter: false };
  return { text: slice, byCharacter: true };
}

async function countTokens(text: string): Promise<number> {
  try {
    return (await api.tokenize(text)).count;
  } catch {
    return Math.ceil(text.length / CHARS_PER_TOKEN);
  }
}

function countLines(text: string): number {
  let lines = 1;
  for (let index = 0; index < text.length; index += 1) if (text.charCodeAt(index) === 10) lines += 1;
  return lines;
}

// ---- Prompt assembly ----------------------------------------------------

/** Extracted text for every attachment on a turn, ahead of the user's words. */
export function attachmentPrelude(message: Message): string {
  const blocks = (message.attachments ?? [])
    .map(attachmentBlock)
    .filter((block) => block !== "");
  return blocks.length ? `${blocks.join("\n\n")}\n\n` : "";
}

function attachmentBlock(attachment: Attachment): string {
  if (attachment.error || !attachment.text) return "";
  if (attachment.kind === "image") return "";
  if (attachment.kind === "pdf" && attachment.mode === "pages") return "";
  const fence = "`".repeat(Math.max(3, longestFence(attachment.text) + 1));
  const header = `Attached file: ${attachment.name} (${describe(attachment)})`;
  const footer = attachment.truncated
    ? `\n[truncated: ${attachment.truncated.total - attachment.truncated.shown} more ${attachment.truncated.unit} omitted]`
    : "";
  return `${header}\n\n${fence}${attachment.language ?? ""}\n${attachment.text}\n${fence}${footer}`;
}

function longestFence(text: string): number {
  let longest = 0;
  for (const run of text.match(/`+/g) ?? []) longest = Math.max(longest, run.length);
  return longest;
}

/** Every picture a turn sends: image attachments plus rendered PDF pages. */
export function attachmentImages(attachments: Attachment[]): string[] {
  const images: string[] = [];
  for (const attachment of attachments) {
    if (attachment.error) continue;
    if (attachment.url) images.push(attachment.url);
    if (attachment.pages?.length) images.push(...attachment.pages);
  }
  return images;
}

// ---- Presentation -------------------------------------------------------

export function describe(attachment: Attachment): string {
  const parts: string[] = [];
  if (attachment.kind === "pdf" && attachment.pageCount) {
    parts.push(`${attachment.pageCount} page${attachment.pageCount === 1 ? "" : "s"}`);
  }
  if (attachment.truncated) {
    const { shown, total, unit } = attachment.truncated;
    parts.push(`${shown} of ${total} ${unit}`);
  } else if (attachment.text) {
    const lines = countLines(attachment.text);
    parts.push(`${lines} line${lines === 1 ? "" : "s"}`);
  }
  parts.push(formatBytes(attachment.size));
  return parts.join(", ");
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/** Files out of a paste or a drop, whatever their type. */
export function filesFrom(items: DataTransferItemList | FileList | null | undefined): File[] {
  if (!items) return [];
  const files: File[] = [];
  for (const item of Array.from(items as ArrayLike<DataTransferItem | File>)) {
    if (item instanceof File) files.push(item);
    else if (item.kind === "file") {
      const file = item.getAsFile();
      if (file) files.push(file);
    }
  }
  return files;
}
