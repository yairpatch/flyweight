import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { beforeAll, describe, expect, it } from "vitest";
import * as pdfjs from "pdfjs-dist";
import { tableToCsv, workbookToTables } from "./sheet";
import { docxToMarkdown } from "./docx";
import { pdfText } from "./pdf";
import { readAttachment } from "../attachments";

// vitest runs from web/, and jsdom rewrites import.meta.url to a page URL.
function fixture(name: string): File {
  return new File([readFileSync(resolve("src/lib/extract/__fixtures__", name))], name);
}

describe("workbookToTables", () => {
  it("reads every sheet, keeping blank cells in their column", async () => {
    const tables = await workbookToTables(fixture("sample.xlsx"));
    expect(tables.map((table) => table.name)).toEqual(["Sales", "Notes"]);

    const sales = tables[0];
    expect(sales.rows[0]).toEqual(["region", "units", "ok"]);
    expect(sales.rows[1]).toEqual(["north", "12", "TRUE"]);
    // B3 is empty in the file: "south" must not slide into the units column.
    expect(sales.rows[2]).toEqual(["south", "", "FALSE"]);
    expect(sales.rows[3]).toEqual(["", "", "", "trailing, comma"]);
    expect(sales.omittedRows).toBe(0);

    expect(tables[1].rows[0]).toEqual(['he said "hi"', "3.5"]);
  });

  it("quotes CSV cells that carry commas or quotes", async () => {
    const tables = await workbookToTables(fixture("sample.xlsx"));
    expect(tableToCsv(tables[0].rows)).toContain(',,,"trailing, comma"');
    expect(tableToCsv(tables[1].rows)).toBe('"he said ""hi""",3.5');
  });

  it("rejects a file that is not a workbook", async () => {
    await expect(workbookToTables(fixture("sample.docx"))).rejects.toThrow(/workbook/);
  });
});

describe("docxToMarkdown", () => {
  it("keeps headings, lists and table cells", async () => {
    const markdown = await docxToMarkdown(fixture("sample.docx"));
    expect(markdown).toContain("# Quarterly report");
    // mammoth escapes a period after a digit so it cannot read as a list.
    expect(markdown).toContain("Revenue grew by 12%");
    expect(markdown).toContain("- first bullet");
    expect(markdown).toContain("north");
  });
});

describe("pdfText", () => {
  beforeAll(() => {
    // In the browser the worker is a Vite-emitted asset URL. Node has no
    // origin to fetch that from, so point pdf.js at the file on disk.
    pdfjs.GlobalWorkerOptions.workerSrc = resolve("node_modules/pdfjs-dist/build/pdf.worker.mjs");
  });

  it("returns one entry per page from the text layer", async () => {
    const result = await pdfText(fixture("sample.pdf"));
    expect(result.pageCount).toBe(2);
    expect(result.pages[0]).toBe("Quarterly report\nRevenue grew by 12 percent in the northern region.");
    expect(result.pages[1]).toBe("Second page: outlook is stable.");
  });

  it("reaches the prompt as page-marked text when a text layer exists", async () => {
    const attachment = await readAttachment(fixture("sample.pdf"), { vision: false, contextWindow: 8192 });
    expect(attachment.kind).toBe("pdf");
    expect(attachment.mode).toBe("text");
    expect(attachment.pageCount).toBe(2);
    expect(attachment.error).toBeUndefined();
    expect(attachment.text).toContain("--- page 1 ---\nQuarterly report");
    expect(attachment.text).toContain("--- page 2 ---\nSecond page");
  });
});
