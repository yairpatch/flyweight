import { useEffect, useRef, useState } from "react";
import { Activity, Copy, Search, Trash2 } from "lucide-react";
import { useDesk } from "../state";

export function ActivityPage() {
  const logs = useDesk((s) => s.logs);
  const clear = useDesk((s) => s.clearLogs);
  const toast = useDesk((s) => s.toast);
  const [filter, setFilter] = useState("");
  const [errorsOnly, setErrorsOnly] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  const needle = filter.trim().toLowerCase();
  const shown = logs.filter((l) => (!needle || l.text.toLowerCase().includes(needle)) && (!errorsOnly || l.kind === "error"));

  useEffect(() => {
    if (stick.current && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [shown.length]);

  return (
    <div className="d-activity">
      <div className="d-pagehead">
        <div><h2><Activity size={20} /> Activity</h2><p className="d-muted">The engine's output, the app's notes, and every command it ran.</p></div>
        <div className="d-row">
          <label className="d-search d-search--inline"><Search size={14} />
            <input value={filter} onChange={(e) => setFilter(e.target.value)} placeholder="Filter" /></label>
          <label className="d-check"><input type="checkbox" checked={errorsOnly} onChange={(e) => setErrorsOnly(e.target.checked)} />Errors only</label>
          <button className="button button--small" onClick={() => { void navigator.clipboard.writeText(shown.map((l) => l.text).join("\n")); toast("Copied", "ok"); }}><Copy size={13} />Copy</button>
          <button className="button button--small" onClick={clear}><Trash2 size={13} />Clear</button>
        </div>
      </div>
      <div className="d-log" ref={box}
           onScroll={(e) => { const el = e.currentTarget; stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40; }}>
        {shown.length === 0 && <div className="d-muted">Nothing yet. Start a model and its output appears here.</div>}
        {shown.map((l) => <div key={l.seq} className={l.kind ? `d-log--${l.kind}` : undefined}>{l.text || " "}</div>)}
      </div>
    </div>
  );
}
