import { memo, useMemo, useState, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { Check, Copy, Download, Play } from "lucide-react";
import { useStore } from "../store";
import { downloadBlob } from "./Sidebar";

const RUNNABLE = new Set(["html", "htm", "svg", "js", "javascript"]);

const EXTENSIONS: Record<string, string> = {
  javascript: "js", js: "js", typescript: "ts", ts: "ts", tsx: "tsx", jsx: "jsx", python: "py", py: "py",
  html: "html", htm: "html", css: "css", json: "json", yaml: "yml", yml: "yml", markdown: "md", md: "md",
  bash: "sh", sh: "sh", shell: "sh", zsh: "sh", fish: "fish", rust: "rs", rs: "rs", go: "go", java: "java",
  c: "c", cpp: "cpp", "c++": "cpp", h: "h", hpp: "hpp", cs: "cs", csharp: "cs", sql: "sql", svg: "svg",
  xml: "xml", toml: "toml", ini: "ini", kotlin: "kt", swift: "swift", ruby: "rb", php: "php", r: "r",
  cuda: "cu", cmake: "txt", diff: "diff", text: "txt", txt: "txt",
};

interface HastNode {
  type: string;
  value?: string;
  tagName?: string;
  properties?: Record<string, unknown>;
  children?: HastNode[];
}

function hastText(node: HastNode | undefined): string {
  if (!node) return "";
  if (node.type === "text") return node.value ?? "";
  return (node.children ?? []).map(hastText).join("");
}

function languageOf(node: HastNode | undefined): string {
  const classes = node?.properties?.className;
  const list = Array.isArray(classes) ? classes.map(String) : typeof classes === "string" ? classes.split(/\s+/) : [];
  const match = list.find((name) => name.startsWith("language-"));
  return match ? match.slice("language-".length).toLowerCase() : "";
}

function CodeBlock({ language, code, children }: { language: string; code: string; children: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const setPreviewSource = useStore((state) => state.setPreviewSource);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      useStore.getState().toast("Clipboard unavailable", "error");
    }
  };
  const download = () => {
    const extension = EXTENSIONS[language] ?? (language || "txt");
    downloadBlob(new Blob([code], { type: "text/plain" }), `snippet.${extension}`);
  };
  return (
    <div className="code" dir="ltr">
      <div className="code__bar">
        <span className="code__lang">{language || "text"}</span>
        <div className="code__actions">
          {RUNNABLE.has(language) && (
            <button className="code__action" onClick={() => setPreviewSource({ language, code })} title="Run in a sandbox">
              <Play size={13} /> Run
            </button>
          )}
          <button className="code__action" onClick={download} title="Download">
            <Download size={13} />
          </button>
          <button className="code__action" onClick={() => void copy()} title="Copy">
            {copied ? <Check size={13} /> : <Copy size={13} />}
          </button>
        </div>
      </div>
      <pre>{children}</pre>
    </div>
  );
}

const components: Components = {
  pre({ node, children }) {
    const codeNode = (node as HastNode | undefined)?.children?.find((child) => child.tagName === "code");
    const language = languageOf(codeNode);
    const code = hastText(codeNode).replace(/\n$/, "");
    return (
      <CodeBlock language={language} code={code}>
        {children}
      </CodeBlock>
    );
  },
  a({ href, children }) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  },
  table({ children }) {
    return (
      <div className="table-wrap">
        <table>{children}</table>
      </div>
    );
  },
};

const remarkPlugins = [remarkGfm, remarkMath];
const rehypePlugins = [[rehypeKatex, { output: "html", throwOnError: false }], [rehypeHighlight, { detect: false, ignoreMissing: true }]] as never;

export const Markdown = memo(function Markdown({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={remarkPlugins} rehypePlugins={rehypePlugins} components={components}>
      {text}
    </ReactMarkdown>
  );
});

/**
 * While streaming, the text before the last blank line outside a code fence
 * is stable and re-renders from the memo; only the tail re-parses per frame.
 */
export function splitStable(text: string): [string, string] {
  let inFence = false;
  let boundary = -1;
  const lines = text.split("\n");
  let offset = 0;
  for (const line of lines) {
    if (/^\s*(```|~~~)/.test(line)) inFence = !inFence;
    if (!inFence && line.trim() === "" && offset > 0) boundary = offset + line.length + 1;
    offset += line.length + 1;
  }
  if (boundary <= 0) return ["", text];
  return [text.slice(0, boundary), text.slice(boundary)];
}

export function StreamingMarkdown({ text, live }: { text: string; live: boolean }) {
  const [stable, tail] = useMemo(() => (live ? splitStable(text) : ["", text]), [text, live]);
  if (!live) return <Markdown text={text} />;
  return (
    <>
      {stable && <Markdown text={stable} />}
      {tail && <Markdown text={tail} />}
    </>
  );
}
