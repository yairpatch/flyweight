import { useState } from "react";
import {
  Box, Check, ChevronDown, Cpu, Download, FolderOpen, HardDrive, MemoryStick, Pause, Pencil, Play, RefreshCw, Search, Sparkles, Trash2,
} from "lucide-react";
import { desk, type CatalogEntry, type DownloadJob, type Fit, type HubFile } from "../api";
import { useDesk, gb } from "../state";

const LEVEL: Record<Fit["level"], number> = { great: 0, good: 1, slow: 2, no: 3 };

interface SearchResult { repo: string; downloads: number; supported: "yes" | "no" | "unknown"; gated: boolean; architecture: string }
interface Variant { quant: string; size: number; files: HubFile[]; fit: Fit }
interface RepoDetails { repo: string; architecture: string; supported: string; gated: boolean; variants: Variant[]; projector: HubFile | null }

function eta(job: DownloadJob): string {
  if (!job.speed) return "";
  const s = (job.total - job.done) / job.speed;
  return s < 60 ? "under a minute left" : s < 3600 ? `${Math.round(s / 60)} min left` : `${(s / 3600).toFixed(1)} h left`;
}

function Downloads({ jobs, selected, locked }: { jobs: DownloadJob[]; selected: string; locked: boolean }) {
  const call = useDesk((s) => s.call);
  const setPage = useDesk((s) => s.setPage);
  if (!jobs.length) return null;
  return (
    <div className="d-downloads">
      {jobs.map((j) => {
        const pct = j.total ? Math.min(100, Math.round((100 * j.done) / j.total)) : 0;
        const active = ["queued", "downloading", "verifying"].includes(j.state);
        return (
          <div key={j.id} className={`d-dl d-dl--${j.state}`}>
            <div className="d-dl__icon">{j.state === "done" ? <Check size={18} /> : <Download size={18} />}</div>
            <div className="d-dl__body">
              <div className="d-dl__title">{j.name}</div>
              {j.state !== "done" && j.state !== "failed" && <div className="d-meter"><i style={{ width: `${pct}%` }} /></div>}
              <div className="d-dl__sub">
                {j.state === "done" ? (selected === j.result ? "Downloaded · selected" : "Downloaded")
                  : j.state === "failed" ? j.error
                  : j.state === "paused" ? `Paused · ${gb(j.done)} of ${gb(j.total)}`
                  : j.state === "queued" ? "Waiting…" : j.state === "verifying" ? "Checking the file…"
                  : `${gb(j.done)} of ${gb(j.total)}${j.speed ? ` · ${gb(j.speed)}/s · ${eta(j)}` : ""}`}
              </div>
            </div>
            <div className="d-row">
              {j.state === "done" && selected !== j.result && <button className="button button--small" disabled={locked} onClick={() => void call("/api/models/use", { path: j.result })}>Use</button>}
              {j.state === "done" && selected === j.result && <button className="button button--primary button--small" onClick={() => setPage("chat")}>Go to chat</button>}
              {active && <button className="button button--small" onClick={() => void call("/api/models/pause", { job: j.id })}><Pause size={12} />Pause</button>}
              {(j.state === "paused" || j.state === "failed") && <button className="button button--primary button--small" onClick={() => void call("/api/models/resume", { job: j.id })}><Play size={12} />Resume</button>}
              {j.state !== "done" && <button className="button button--ghost button--small" onClick={() => void call("/api/models/cancel", { job: j.id })}>Cancel</button>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Catalog() {
  const snapshot = useDesk((s) => s.snapshot)!;
  const call = useDesk((s) => s.call);
  const m = snapshot.models;
  const locked = snapshot.status !== "idle" && snapshot.status !== "failed";
  if (m.loading && !m.catalog.length) return <div className="d-catalog">{[0, 1, 2].map((i) => <div key={i} className="d-skeleton" />)}</div>;
  if (!m.catalog.length) {
    return (
      <div className="d-card d-empty-row">Could not load the list of models.{" "}
        <button className="button button--small" onClick={() => void call("/api/models/refresh")}><RefreshCw size={12} />Try again</button>
      </div>
    );
  }
  const entries = [...m.catalog].sort((a, b) =>
    Number(Boolean(b.available)) - Number(Boolean(a.available))
    || LEVEL[a.fit?.level ?? "no"] - LEVEL[b.fit?.level ?? "no"] || (b.rank ?? 0) - (a.rank ?? 0));
  const fitting = entries.filter((e) => e.available && (e.fit?.level === "great" || e.fit?.level === "good"));
  const best = fitting.reduce<CatalogEntry | null>((x, y) => (!x || (y.rank ?? 0) > (x.rank ?? 0) ? y : x), null);
  if (best) entries.splice(entries.indexOf(best), 1), entries.unshift(best);
  const have = (e: CatalogEntry) => {
    const file = e.files?.[0]?.path.split("/").pop();
    return m.library.find((x) => x.repo === e.repo && x.file === file)?.path ?? null;
  };
  return (
    <div className="d-catalog">
      {entries.map((e) => {
        const job = m.downloads.find((j) => j.repo === e.repo && j.state !== "done");
        const path = e.available ? have(e) : null;
        const pct = job?.total ? Math.round((100 * job.done) / job.total) : 0;
        return (
          <div key={e.id} className={`d-mcard${e === best ? " d-mcard--best" : ""}${!e.available || e.fit?.level === "no" ? " d-mcard--dim" : ""}`}>
            {e === best && <span className="d-ribbon"><Sparkles size={11} />Best match</span>}
            <div className="d-mcard__name">{e.name}</div>
            {e.available && e.fit && <span className={`d-fit d-fit--${e.fit.level}`}>{e.fit.label}</span>}
            <p className="d-mcard__desc">{e.description}</p>
            <div className="d-chips">{(e.tags ?? []).map((t) => <span key={t} className="d-chip">{t}</span>)}</div>
            <div className="d-mcard__foot">
              {!e.available ? <span className="d-muted d-small">Unavailable: {e.error}</span>
                : job ? <div style={{ flex: 1 }}><div className="d-meter"><i style={{ width: `${pct}%` }} /></div><small className="d-muted">{job.state === "paused" ? "Paused" : `${pct}%`}</small></div>
                : path ? (<>
                    <span className="d-muted d-small">{gb(e.size)} · downloaded</span>
                    {snapshot.settings.model === path ? <span className="d-pill d-pill--accent"><Check size={12} />Selected</span>
                      : <button className="button button--small" disabled={locked} onClick={() => void call("/api/models/use", { path })}>Use</button>}
                  </>)
                : (<>
                    <span className="d-muted d-small">{gb(e.size)} · {e.quant}</span>
                    <button className={`button button--small${e === best ? " button--primary" : ""}`} disabled={e.fit?.level === "no"}
                            onClick={() => void call("/api/models/download", { id: e.id })}><Download size={13} />Download</button>
                  </>)}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function HubSearch() {
  const snapshot = useDesk((s) => s.snapshot)!;
  const call = useDesk((s) => s.call);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState<Record<string, RepoDetails | "loading" | { error: string }>>({});

  const search = async () => {
    setBusy(true);
    setError("");
    try {
      const r = await desk<{ ok: boolean; results?: SearchResult[]; error?: string }>("/api/models/search", { query });
      if (r.ok) setResults(r.results ?? []);
      else setError(r.error ?? "Search failed");
    } catch (e) {
      setError(String(e));
    }
    setBusy(false);
  };
  const toggle = async (repo: string) => {
    if (open[repo]) {
      const next = { ...open };
      delete next[repo];
      setOpen(next);
      return;
    }
    setOpen({ ...open, [repo]: "loading" });
    const r = await desk<{ ok: boolean; repo?: RepoDetails; error?: string }>("/api/models/repo", { repo });
    setOpen((o) => ({ ...o, [repo]: r.ok && r.repo ? r.repo : { error: r.error ?? "failed" } }));
  };
  const label = { yes: "Supported", no: "Not supported", unknown: "Unknown" } as const;
  return (
    <>
      <form className="d-searchbar" onSubmit={(e) => { e.preventDefault(); if (query.trim()) void search(); }}>
        <Search size={15} />
        <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search GGUF models on Hugging Face, e.g. qwen3 coder" />
        <button className="button" type="submit" disabled={busy}>{busy ? "Searching…" : "Search"}</button>
      </form>
      {error && <div className="d-alert d-alert--error">{error}</div>}
      {results && (
        <div className="d-card d-results">
          {!results.length && <div className="d-empty-row">No GGUF models matched.</div>}
          {results.map((r) => {
            const d = open[r.repo];
            return (
              <div key={r.repo} className="d-result">
                <button className="d-result__head" onClick={() => void toggle(r.repo)}>
                  <span className="d-result__name">{r.repo}</span>
                  <span className="d-muted d-small">{r.downloads.toLocaleString()} downloads</span>
                  <span className={`d-support d-support--${r.supported}`}>{label[r.supported]}</span>
                  <ChevronDown size={14} />
                </button>
                {d === "loading" && <div className="d-result__note">Loading files…</div>}
                {d && d !== "loading" && "error" in d && <div className="d-result__note">{d.error}</div>}
                {d && d !== "loading" && "variants" in d && (
                  <div className="d-variants">
                    {(d.gated || d.supported !== "yes") && (
                      <div className="d-result__note">
                        {d.gated ? "This model needs a Hugging Face login, which Flyweight does not do yet."
                          : d.supported === "no" ? `Flyweight cannot run the ${d.architecture} architecture.`
                          : "Hugging Face does not say which architecture this is; it may not load."}
                      </div>
                    )}
                    {d.variants.map((v) => {
                      const files = v.files.concat(d.projector ? [d.projector] : []);
                      const job = snapshot.models.downloads.find((j) => j.repo === d.repo && j.files[0]?.path === files[0].path && j.state !== "done");
                      return (
                        <div key={v.quant + v.size} className="d-variant">
                          <code>{v.quant}</code><span>{gb(v.size)}</span><span className={`d-fit d-fit--${v.fit.level}`}>{v.fit.label}</span>
                          {job ? <span className="d-muted">{Math.round((100 * job.done) / (job.total || 1))}%</span> : (
                            <button className="button button--small" disabled={d.gated || d.supported === "no" || v.fit.level === "no"}
                                    onClick={() => void call("/api/models/download", { repo: d.repo, name: `${d.repo.split("/").pop()} ${v.quant}`, files })}>
                              <Download size={12} />Download
                            </button>
                          )}
                        </div>
                      );
                    })}
                    {!d.variants.length && <div className="d-result__note">No complete GGUF files in this repository.</div>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </>
  );
}

export function ModelsPage() {
  const snapshot = useDesk((s) => s.snapshot);
  const call = useDesk((s) => s.call);
  if (!snapshot) return null;
  const m = snapshot.models;
  const mc = m.machine;
  const locked = snapshot.status !== "idle" && snapshot.status !== "failed";
  return (
    <div className="d-scroll">
      <div className="d-pagewrap">
        <div className="d-pagehead">
          <div><h2><Sparkles size={20} /> Models</h2>
            <p className="d-muted">Recommended for this computer, downloaded from Hugging Face, kept on this computer.</p></div>
        </div>
        <div className="d-chips d-machine">
          {mc ? (<>
            <span className="d-chip"><MemoryStick size={13} />{gb(mc.ram, 0)} memory</span>
            <span className="d-chip"><Cpu size={13} />{mc.gpu ? `${mc.gpu} · ${gb(mc.vram, 0)}` : "No GPU: models run on the CPU"}</span>
            <span className="d-chip"><HardDrive size={13} />{gb(mc.disk_free, 0)} free</span>
          </>) : <span className="d-chip">Measuring this computer…</span>}
        </div>
        <Downloads jobs={m.downloads} selected={snapshot.settings.model} locked={locked} />
        <h3 className="d-section">Recommended</h3>
        <Catalog />
        <div className="d-section d-section--row">
          <h3>Your models</h3>
          <div className="d-row">
            <button className="button button--ghost button--small" onClick={() => void call("/api/models/reveal", {})}><FolderOpen size={13} />Open folder</button>
            <button className="button button--ghost button--small" onClick={() => void call("/api/models/folder")}><Pencil size={13} />Change folder</button>
          </div>
        </div>
        <div className="d-card d-library">
          {!m.library.length && <div className="d-empty-row">Models you download appear here.</div>}
          {m.library.map((x) => (
            <div key={x.path} className="d-lib">
              <Box size={18} />
              <div className="d-lib__text"><b>{x.name}</b><small className="d-muted">{x.file} · {gb(x.size)}</small></div>
              {snapshot.settings.model === x.path ? <span className="d-pill d-pill--accent">Selected</span>
                : <button className="button button--small" disabled={locked} onClick={() => void call("/api/models/use", { path: x.path })}>Use</button>}
              <button className="d-iconbtn" title="Show in folder" onClick={() => void call("/api/models/reveal", { path: x.path })}><FolderOpen size={15} /></button>
              <button className="d-iconbtn" title="Delete" onClick={() => { if (confirm(`Delete ${x.name} from this computer?`)) void call("/api/models/delete", { folder: x.folder }); }}><Trash2 size={15} /></button>
            </div>
          ))}
        </div>
        <p className="d-muted d-small">Saved in {m.dir}</p>
        <h3 className="d-section">Search Hugging Face</h3>
        <HubSearch />
      </div>
    </div>
  );
}
