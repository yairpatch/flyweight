// PDF extraction, loaded on demand so pdf.js only ships to a session that
// actually attaches a PDF. Two modes: the embedded text layer (cheap, exact)
// and rendered page images for the vision tower (works on scans).
import * as pdfjs from "pdfjs-dist";
import workerUrl from "pdfjs-dist/build/pdf.worker.mjs?url";
import type { PDFDocumentLoadingTask } from "pdfjs-dist";

pdfjs.GlobalWorkerOptions.workerSrc = workerUrl;

/** Rendering every page of a long PDF would swamp the context; cap it. */
export const MAX_PAGE_IMAGES = 8;
const PAGE_MAX_SIDE = 1400;
const PAGE_QUALITY = 0.85;

/** The loading task owns the worker, so the caller destroys that, not the doc. */
async function open(file: File): Promise<PDFDocumentLoadingTask> {
  const data = new Uint8Array(await file.arrayBuffer());
  return pdfjs.getDocument({ data });
}

export interface PdfText {
  /** One entry per page, in order. */
  pages: string[];
  pageCount: number;
}

export async function pdfText(file: File): Promise<PdfText> {
  const task = await open(file);
  try {
    const doc = await task.promise;
    const pages: string[] = [];
    for (let index = 1; index <= doc.numPages; index += 1) {
      const page = await doc.getPage(index);
      const content = await page.getTextContent();
      let text = "";
      for (const item of content.items) {
        if (!("str" in item)) continue;
        text += item.str;
        if (item.hasEOL) text += "\n";
        else if (item.str && !item.str.endsWith(" ")) text += " ";
      }
      pages.push(text.replace(/[ \t]+\n/g, "\n").trim());
      page.cleanup();
    }
    return { pages, pageCount: doc.numPages };
  } finally {
    await task.destroy();
  }
}

/** Render the first pages to JPEG data URLs for a vision tower. */
export async function pdfPageImages(file: File, limit = MAX_PAGE_IMAGES): Promise<string[]> {
  const task = await open(file);
  try {
    const doc = await task.promise;
    const images: string[] = [];
    const count = Math.min(doc.numPages, limit);
    for (let index = 1; index <= count; index += 1) {
      const page = await doc.getPage(index);
      const base = page.getViewport({ scale: 1 });
      const scale = Math.min(2, PAGE_MAX_SIDE / Math.max(base.width, base.height));
      const viewport = page.getViewport({ scale });
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(1, Math.round(viewport.width));
      canvas.height = Math.max(1, Math.round(viewport.height));
      const context = canvas.getContext("2d");
      if (!context) break;
      context.fillStyle = "#ffffff";
      context.fillRect(0, 0, canvas.width, canvas.height);
      // pdf.js wants the canvas itself in newer builds and the 2d context in
      // older ones; handing it both keeps either happy.
      await page.render({ canvas, canvasContext: context, viewport } as Parameters<typeof page.render>[0]).promise;
      images.push(canvas.toDataURL("image/jpeg", PAGE_QUALITY));
      page.cleanup();
    }
    return images;
  } finally {
    await task.destroy();
  }
}
