import { useEffect, useMemo } from "react";
import { X } from "lucide-react";
import { useStore } from "../store";

/**
 * Runs model-generated HTML/SVG/JS inside a sandboxed iframe. The document
 * travels in the URL fragment, so it never reaches the server, and
 * preview.html is served with a sandboxing CSP that gives it an opaque
 * origin with no access to this app's storage or API.
 */
export function PreviewOverlay() {
  const source = useStore((state) => state.previewSource);
  const close = () => useStore.getState().setPreviewSource(null);

  const src = useMemo(() => {
    if (!source) return "";
    return `/preview.html#${encodeURIComponent(previewDocument(source.language, source.code))}`;
  }, [source]);

  useEffect(() => {
    if (!source) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [source]);

  if (!source) return null;
  return (
    <div className="overlay" onClick={close} role="dialog" aria-label="Code preview">
      <div className="overlay__frame" onClick={(event) => event.stopPropagation()}>
        <div className="overlay__bar">
          <span className="overlay__title">Preview · {source.language}</span>
          <button className="icon-button" onClick={close} aria-label="Close preview">
            <X size={16} />
          </button>
        </div>
        <iframe className="overlay__iframe" title="Preview" sandbox="allow-scripts allow-modals" src={src} />
      </div>
    </div>
  );
}

export function previewDocument(language: string, code: string): string {
  const lang = language.toLowerCase();
  if (lang === "svg") return `<!DOCTYPE html><html><body style="margin:0;display:grid;place-items:center;min-height:100vh;background:#fff">${code}</body></html>`;
  if (lang === "js" || lang === "javascript") {
    const escaped = code.replace(/<\/script/gi, "<\\/script");
    return `<!DOCTYPE html><html><head><meta charset="utf-8"><style>body{font:14px/1.5 ui-monospace,monospace;padding:16px;white-space:pre-wrap}</style></head><body><pre id="out"></pre><script>
const out=document.getElementById('out');
const log=(...a)=>{out.textContent+=a.map(v=>typeof v==='string'?v:JSON.stringify(v,null,2)).join(' ')+'\\n';};
console.log=log;console.error=log;console.warn=log;console.info=log;
window.onerror=(m)=>log('Error: '+m);
try{${escaped}}catch(e){log('Error: '+(e&&e.message||e));}
</script></body></html>`;
  }
  return code;
}
