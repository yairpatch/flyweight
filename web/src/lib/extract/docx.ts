// Word documents through mammoth's prebuilt browser bundle, loaded on demand.
// Markdown keeps headings, lists and tables legible to the model, which plain
// text extraction would flatten away.

interface MammothResult {
  value: string;
  messages: Array<{ type: string; message: string }>;
}
interface Mammoth {
  convertToMarkdown(input: { arrayBuffer: ArrayBuffer }): Promise<MammothResult>;
}

export async function docxToMarkdown(file: File): Promise<string> {
  const module = (await import("mammoth/mammoth.browser.js")) as unknown as { default?: Mammoth } & Mammoth;
  const mammoth = module.default ?? module;
  const result = await mammoth.convertToMarkdown({ arrayBuffer: await file.arrayBuffer() });
  return result.value.trim();
}
