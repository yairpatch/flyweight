// Spreadsheets to CSV. The npm `xlsx` package is abandoned at 0.18.5 with an
// open prototype-pollution advisory, so this unzips the workbook with fflate
// and reads the sheet XML directly -- enough for cell values, which is all a
// model needs.
import { unzipSync, strFromU8 } from "fflate";

/** A very wide or very long sheet would eat the whole context on its own. */
export const MAX_ROWS_PER_SHEET = 2000;

export interface SheetTable {
  name: string;
  rows: string[][];
  /** Rows present in the file beyond the ones returned. */
  omittedRows: number;
}

export async function workbookToTables(file: File): Promise<SheetTable[]> {
  const zip = unzipSync(new Uint8Array(await file.arrayBuffer()));
  const read = (path: string): string | null => (zip[path] ? strFromU8(zip[path]) : null);
  const parse = (xml: string): Document => new DOMParser().parseFromString(xml, "application/xml");

  const workbookXml = read("xl/workbook.xml");
  if (!workbookXml) throw new Error("not an Office Open XML workbook");
  const shared = sharedStrings(read("xl/sharedStrings.xml"), parse);
  const targets = relationshipTargets(read("xl/_rels/workbook.xml.rels"), parse);

  const tables: SheetTable[] = [];
  const sheets = Array.from(parse(workbookXml).getElementsByTagName("sheet"));
  for (const [index, sheet] of sheets.entries()) {
    const name = sheet.getAttribute("name") ?? `Sheet${index + 1}`;
    const relation = sheet.getAttribute("r:id") ?? sheet.getAttributeNS(REL_NS, "id");
    const target = (relation && targets.get(relation)) || `worksheets/sheet${index + 1}.xml`;
    const xml = read(`xl/${target.replace(/^\/?xl\//, "")}`);
    if (!xml) continue;
    tables.push({ name, ...sheetRows(parse(xml), shared) });
  }
  return tables;
}

const REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";

function sharedStrings(xml: string | null, parse: (xml: string) => Document): string[] {
  if (!xml) return [];
  return Array.from(parse(xml).getElementsByTagName("si")).map((item) =>
    Array.from(item.getElementsByTagName("t"))
      .map((node) => node.textContent ?? "")
      .join(""),
  );
}

function relationshipTargets(xml: string | null, parse: (xml: string) => Document): Map<string, string> {
  const map = new Map<string, string>();
  if (!xml) return map;
  for (const node of Array.from(parse(xml).getElementsByTagName("Relationship"))) {
    const id = node.getAttribute("Id");
    const target = node.getAttribute("Target");
    if (id && target) map.set(id, target);
  }
  return map;
}

function sheetRows(document: Document, shared: string[]): { rows: string[][]; omittedRows: number } {
  const rows: string[][] = [];
  const all = Array.from(document.getElementsByTagName("row"));
  for (const row of all.slice(0, MAX_ROWS_PER_SHEET)) {
    const cells: string[] = [];
    for (const cell of Array.from(row.getElementsByTagName("c"))) {
      const column = columnIndex(cell.getAttribute("r"));
      const at = column >= 0 ? column : cells.length;
      while (cells.length < at) cells.push("");
      cells[at] = cellValue(cell, shared);
    }
    rows.push(cells);
  }
  // Drop trailing rows and columns that are entirely blank.
  while (rows.length && rows[rows.length - 1].every((value) => value === "")) rows.pop();
  return { rows, omittedRows: Math.max(0, all.length - MAX_ROWS_PER_SHEET) };
}

/** "BC12" -> 54. Returns -1 when the reference is missing or malformed. */
function columnIndex(reference: string | null): number {
  if (!reference) return -1;
  let index = 0;
  for (const character of reference) {
    const position = character.charCodeAt(0) - 64;
    if (position < 1 || position > 26) break;
    index = index * 26 + position;
  }
  return index - 1;
}

function cellValue(cell: Element, shared: string[]): string {
  const type = cell.getAttribute("t");
  if (type === "inlineStr") {
    return Array.from(cell.getElementsByTagName("t"))
      .map((node) => node.textContent ?? "")
      .join("");
  }
  const raw = cell.getElementsByTagName("v")[0]?.textContent ?? "";
  if (type === "s") return shared[Number(raw)] ?? "";
  if (type === "b") return raw === "1" ? "TRUE" : "FALSE";
  if (type === "e") return raw;
  return raw;
}

export function tableToCsv(rows: string[][]): string {
  return rows.map((row) => row.map(csvCell).join(",")).join("\n");
}

function csvCell(value: string): string {
  return /[",\n\r]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}
