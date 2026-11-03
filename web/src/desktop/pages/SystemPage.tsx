import { AlertTriangle, Check, Copy, Cpu, Download, ExternalLink, Plus, RefreshCw, ShieldCheck, X } from "lucide-react";
import { useDesk } from "../state";

const NAMES: Record<string, string> = {
  python: "Python", jinja2: "Jinja2", package: "Package", command: "Command", "native runtime": "Native runtime",
  sources: "Sources", cmake: "CMake", "c++ compiler": "C++ compiler", "build tool": "Build tool",
  "nvidia gpu": "NVIDIA GPU", "amd gpu": "AMD GPU", "cuda headers": "CUDA headers", check: "System check",
};

function GpuCard() {
  const snapshot = useDesk((s) => s.snapshot)!;
  const call = useDesk((s) => s.call);
  const g = snapshot.gpu;
  const job = snapshot.gpuSetup;
  if (!g) return <div className="d-card d-gpu"><p className="d-muted">Checking graphics…</p></div>;
  const tone = job.error ? "error" : g.state === "ready" ? "ok" : g.state === "none" ? "" : "warn";
  return (
    <div className={`d-card d-gpu d-gpu--${tone}`}>
      <div className="d-gpu__icon">{g.state === "ready" ? <Check size={18} /> : <Cpu size={18} />}</div>
      <div className="d-gpu__text">
        <b>{job.running ? "Setting up GPU support…" : job.error ? "GPU setup did not finish" : g.title}</b>
        <p className="d-muted d-small">{job.running ? job.message : job.error || g.detail}{g.state === "needs-setup" && !job.running ? ` About ${g.download_mb} MB, once.` : ""}</p>
        {job.running && <div className="d-meter"><i style={{ width: `${Math.round(job.fraction * 100)}%` }} /></div>}
      </div>
      <div className="d-row">
        {(g.state === "needs-setup" || job.error) && !job.running && (
          <button className="button button--primary button--small" onClick={() => void call("/api/gpu/setup")}><Download size={13} />Set up GPU</button>
        )}
        {(g.state === "needs-driver" || g.state === "old-driver") && (
          <button className="button button--primary button--small" onClick={() => void call("/api/open", { url: snapshot.driverUrl })}><ExternalLink size={13} />Get the driver</button>
        )}
        <button className="button button--ghost button--small" onClick={() => void call("/api/gpu/refresh")}><RefreshCw size={13} />Check again</button>
      </div>
    </div>
  );
}

export function SystemPage() {
  const snapshot = useDesk((s) => s.snapshot);
  const call = useDesk((s) => s.call);
  const toast = useDesk((s) => s.toast);
  if (!snapshot) return null;
  const d = snapshot.doctor;
  const order = { fail: 0, warn: 1, ok: 2 };
  const lines = [...d.lines].sort((a, b) => order[a.status] - order[b.status]);
  const failures = lines.filter((l) => l.status === "fail").length;
  return (
    <div className="d-scroll">
      <div className="d-pagewrap d-pagewrap--narrow">
        <div className="d-pagehead">
          <div><h2><ShieldCheck size={20} /> System</h2><p className="d-muted">Whether this computer can run models, and how to fix what is missing.</p></div>
          <button className="button" disabled={d.running} onClick={() => void call("/api/doctor")}><RefreshCw size={14} />{d.running ? "Checking…" : d.ran ? "Run again" : "Run check"}</button>
        </div>
        <h3 className="d-section">Graphics</h3>
        <GpuCard />
        <h3 className="d-section">Check</h3>
        <div className={`d-summary d-summary--${d.running || !d.ran ? "pending" : failures ? "fail" : "ok"}`}>
          {d.running ? <><RefreshCw size={16} />Checking this computer…</>
            : !d.ran ? <><ShieldCheck size={16} />Run the check to see whether this computer is ready.</>
            : failures ? <><AlertTriangle size={16} />{failures} problem{failures > 1 ? "s" : ""} to fix before models can run.</>
            : <><Check size={16} />This computer can run Flyweight.</>}
        </div>
        {lines.length > 0 && (
          <div className="d-card d-checks">
            {lines.map((l, i) => (
              <div key={i} className="d-check-row">
                <span className={`d-check-icon d-check-icon--${l.status}`}>{l.status === "ok" ? <Check size={12} /> : l.status === "warn" ? "!" : <X size={12} />}</span>
                <div className="d-check-body">
                  <b>{NAMES[l.name.toLowerCase()] ?? l.name}</b>
                  <div className="d-muted d-small">{l.detail}</div>
                  {l.fix && (
                    <div className="d-fix"><span>{l.fix}</span>
                      <button className="d-iconbtn" title="Copy" onClick={() => { void navigator.clipboard.writeText(l.fix); toast("Copied", "ok"); }}><Copy size={13} /></button>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
        <h3 className="d-section">Desktop</h3>
        <div className="d-card d-rowcard">
          <div><b>Add to the app menu</b><p className="d-muted d-small">Open Flyweight from your applications menu without a terminal.</p></div>
          <button className="button button--small" onClick={async () => {
            const r = await call<{ ok: boolean; paths?: string[] }>("/api/shortcut");
            if (r?.ok) toast("Added to the app menu", "ok");
          }}><Plus size={13} />Add shortcut</button>
        </div>
        <p className="d-muted d-small">Flyweight {snapshot.version} · {snapshot.platform}</p>
      </div>
    </div>
  );
}
