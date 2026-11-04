import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { ClipboardPaste, Copy, Cpu, Layers, Play, RotateCcw, Search, Terminal, X } from "lucide-react";
import { useDesk, gb } from "../state";
import type { OptionSpec, Scalar, Values } from "../api";
import { OptionField, invalid, isDefault } from "../components/OptionField";
import { useJob, latestJob } from "../components/useJob";
import { TuneCard } from "../components/TuneCard";

// The CLI groups flags by what they touch; "hardware and placement" is long
// enough to deserve sub-sections. Anything not named here keeps its group.
const SUBGROUPS: Record<string, string> = {
  expert_mode: "GPU and CPU placement", gpu_cache_mib: "GPU and CPU placement", gpu_layers: "GPU and CPU placement",
  cpu_threads: "GPU and CPU placement", parallel_sequences: "GPU and CPU placement", scratch_context: "GPU and CPU placement",
  dense_requant: "GPU and CPU placement",
  prompt_cache_mib: "Memory and caches", prompt_cache_dir: "Memory and caches", kv_dtype: "Memory and caches",
  cache_type_k: "Memory and caches", cache_type_v: "Memory and caches",
  mtp_drafts: "Speculative decoding", mtp_model: "Speculative decoding",
  mmproj: "Vision", image_max_tokens: "Vision", image_urls: "Vision",
  image_model: "Image generation", image_transformer: "Image generation", image_max_size: "Image generation",
  image_reserve: "Image generation", image_weights: "Image generation", image_precision: "Image generation",
};

const ESSENTIALS = [
  "backend", "context_window", "max_new_tokens", "gpu_layers", "expert_mode", "kv_dtype",
  "parallel_sequences", "reasoning_effort", "temperature", "host", "port", "api_key",
];

const TITLES: Record<string, string> = {
  model: "Model", server: "Server", "generation defaults": "Generation defaults", backend: "Backend",
  "advanced tuning": "Advanced tuning",
};

interface Section {
  id: string;
  title: string;
  description: string;
  options: OptionSpec[];
}

function argvPreview(options: OptionSpec[], values: Values, model: string, extra: string): string[] {
  const parts = ["flyweight", "serve", model || "MODEL"];
  for (const spec of options) {
    if (spec.positional) continue;
    const value = values[spec.dest];
    if (isDefault(spec, value ?? undefined)) continue;
    if (spec.kind === "bool") {
      if (Boolean(value) === (spec.store === "true")) parts.push(spec.flag);
    } else parts.push(spec.flag, String(value));
  }
  if (extra.trim()) parts.push(extra.trim());
  return parts;
}

function quote(word: string): string {
  return /^[\w@%+=:,./-]+$/.test(word) ? word : `'${word.replace(/'/g, "'\\''")}'`;
}

function PlanCard({ values }: { values: Values }) {
  const call = useDesk((s) => s.call);
  const model = useDesk((s) => s.snapshot?.model);
  const running = useDesk((s) => s.snapshot?.status === "running");
  const [jobId, setJobId] = useState<string | null>(null);
  const job = useJob(jobId);
  useEffect(() => {
    void latestJob("plan").then((j) => j && setJobId(j.id));
  }, []);
  const run = async () => {
    const r = await call<{ ok: boolean; job?: string }>("/api/jobs/run", { command: "plan", values });
    if (r?.ok && r.job) setJobId(r.job);
  };
  const notes = job?.log.filter((l) => l.startsWith("[flyweight]")).map((l) => l.replace("[flyweight] ", "")) ?? [];
  return (
    <section className="d-card">
      <div className="d-card__head">
        <Layers size={16} /><h3>Memory plan</h3>
        <span className="d-spacer" />
        {job?.state === "running" ? (
          <button className="button button--small" onClick={() => void call("/api/jobs/cancel", { id: job.id })}>Cancel</button>
        ) : (
          <button className="button button--small" disabled={!model?.ok} onClick={() => void run()}>
            <Play size={13} />{job ? "Run again" : "Preview"}
          </button>
        )}
      </div>
      <p className="d-muted d-small">
        Loads the model with these settings and shows what lands on the GPU and in RAM, without serving.
        {running && " The running model holds memory, so the plan sees less free."}
      </p>
      {job?.state === "running" && <div className="d-bar"><i /></div>}
      {job?.state === "done" && <pre className="d-pre">{job.stdout.trim()}</pre>}
      {job?.state === "failed" && <div className="d-alert d-alert--error">{job.error}</div>}
      {notes.length > 0 && job?.state !== "running" && (
        <details className="d-details"><summary>Placement notes ({notes.length})</summary>
          <ul className="d-notes">{notes.map((n, i) => <li key={i}>{n}</li>)}</ul>
        </details>
      )}
    </section>
  );
}

function ModelCard() {
  const snapshot = useDesk((s) => s.snapshot);
  const call = useDesk((s) => s.call);
  const path = snapshot?.model.ok ? snapshot.model.path : "";
  const info = snapshot?.modelInfo;
  const asked = useRef("");
  useEffect(() => {
    if (path && !info && asked.current !== path) {
      asked.current = path;
      void call("/api/inspect", { path });
    }
  }, [path, info, call]);
  if (!snapshot?.model.ok) return null;
  const params = info?.parameters ?? 0;
  return (
    <section className="d-card">
      <div className="d-card__head"><Cpu size={16} /><h3>Model</h3></div>
      {!info ? (
        <p className="d-muted d-small">Reading the model's metadata…</p>
      ) : (
        <>
          <div className="d-kv">
            <span>Architecture</span><b>{info.architecture || "—"}</b>
            <span>Parameters</span><b>{params >= 1e9 ? `${(params / 1e9).toFixed(1)} B` : `${(params / 1e6).toFixed(0)} M`}</b>
            <span>File</span><b>{gb(info.file_size)}</b>
            {Object.entries(info.config).filter(([k]) => k !== "architecture").map(([k, v]) => (
              <Fragment key={k}><span>{k.replace(/_/g, " ")}</span><b>{String(v)}</b></Fragment>
            ))}
          </div>
          {info.quant_mix.length > 0 && (
            <div className="d-mix" title="Share of the file by tensor type">
              {info.quant_mix.map((q) => (
                <span key={q.type} style={{ flexGrow: Math.max(q.share, 0.02) }} className="d-mix__part">
                  {q.share > 0.08 ? `${q.type} ${Math.round(q.share * 100)}%` : ""}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </section>
  );
}

export function ServerPage() {
  const snapshot = useDesk((s) => s.snapshot);
  const schemas = useDesk((s) => s.schemas);
  const call = useDesk((s) => s.call);
  const toast = useDesk((s) => s.toast);
  const [values, setValues] = useState<Values>({});
  const [loadedFor, setLoadedFor] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [onlyChanged, setOnlyChanged] = useState(false);
  const [section, setSection] = useState("essentials");
  const [importing, setImporting] = useState(false);
  const [importText, setImportText] = useState("");
  const saveTimer = useRef(0);
  const dirtyFor = useRef<string | null>(null);

  const model = snapshot?.settings.model ?? "";
  const serve = schemas?.serve;
  const options = useMemo(() => serve?.groups.flatMap((g) => g.options) ?? [], [serve]);
  const byDest = useMemo(() => Object.fromEntries(options.map((o) => [o.dest, o])), [options]);

  // Load the profile when the model changes (not on every poll: that would
  // undo typing).
  useEffect(() => {
    if (snapshot && loadedFor !== model && dirtyFor.current === null) {
      setValues({ ...snapshot.profile });
      setLoadedFor(model);
    }
  }, [snapshot, model, loadedFor]);

  const save = (next: Values) => {
    setValues(next);
    dirtyFor.current = model;
    window.clearTimeout(saveTimer.current);
    saveTimer.current = window.setTimeout(async () => {
      const bad = options.find((o) => invalid(o, next[o.dest] ?? undefined));
      if (bad) return; // the field shows why; nothing half-valid is saved
      await call("/api/profile", { model, values: next });
      dirtyFor.current = null;
    }, 450);
  };
  const set = (dest: string, value: Scalar | undefined) => {
    const next = { ...values };
    if (value === undefined) delete next[dest];
    else next[dest] = value;
    save(next);
  };

  const sections: Section[] = useMemo(() => {
    if (!serve) return [];
    const out: Section[] = [{
      id: "essentials", title: "Essentials", description: "The settings most people change. Everything else is in the sections below.",
      options: ESSENTIALS.map((d) => byDest[d]).filter(Boolean),
    }];
    for (const group of serve.groups) {
      const visible = group.options.filter((o) => !o.positional);
      if (!visible.length) continue;
      const subs = new Map<string, OptionSpec[]>();
      for (const o of visible) {
        const key = SUBGROUPS[o.dest] ?? (TITLES[group.title] ?? group.title.charAt(0).toUpperCase() + group.title.slice(1));
        if (!subs.has(key)) subs.set(key, []);
        subs.get(key)!.push(o);
      }
      for (const [title, opts] of subs) {
        out.push({ id: title, title, description: SUBGROUPS[opts[0].dest] ? "" : group.description, options: opts });
      }
    }
    return out;
  }, [serve, byDest]);

  if (!snapshot || !serve) return <div className="d-loading">Loading engine options…</div>;

  const needle = query.trim().toLowerCase();
  const matches = (o: OptionSpec) =>
    (!needle || o.flags.some((f) => f.toLowerCase().includes(needle)) || o.help.toLowerCase().includes(needle) || o.dest.includes(needle))
    && (!onlyChanged || !isDefault(o, values[o.dest] ?? undefined));
  const filtering = Boolean(needle) || onlyChanged;
  const changedCount = options.filter((o) => !o.positional && !isDefault(o, values[o.dest] ?? undefined)).length;
  const running = snapshot.status === "running" || snapshot.status === "loading";
  const extra = String(values._extra ?? "");
  const command = argvPreview(options, values, snapshot.model.ok ? snapshot.model.path : model, extra).map(quote).join(" ");
  const shown = filtering ? sections.slice(1) : sections.filter((s) => s.id === section);

  const doImport = async () => {
    const r = await call<{ ok: boolean; values?: Values; model?: string | null }>("/api/import", { text: importText });
    if (!r?.ok || !r.values) return;
    const next: Values = { ...r.values };
    if (values._extra) next._extra = values._extra;
    save(next);
    setImporting(false);
    setImportText("");
    toast(r.model && r.model !== model ? `Imported. The command named ${r.model}; this profile stays on the selected model.` : "Imported", "ok");
  };

  return (
    <div className="d-server">
      <div className="d-server__nav">
        <label className="d-search">
          <Search size={14} />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={`Search ${options.length} options`} />
          {query && <button onClick={() => setQuery("")} aria-label="Clear"><X size={13} /></button>}
        </label>
        <label className="d-check"><input type="checkbox" checked={onlyChanged} onChange={(e) => setOnlyChanged(e.target.checked)} />Only changed ({changedCount})</label>
        {sections.map((s) => {
          const n = s.options.filter((o) => !isDefault(o, values[o.dest] ?? undefined)).length;
          return (
            <button key={s.id} className={`d-subnav${!filtering && section === s.id ? " d-subnav--on" : ""}`}
                    onClick={() => { setQuery(""); setOnlyChanged(false); setSection(s.id); }}>
              {s.title}{n > 0 && s.id !== "essentials" && <b>{n}</b>}
            </button>
          );
        })}
        <button className={`d-subnav${!filtering && section === "extra" ? " d-subnav--on" : ""}`} onClick={() => setSection("extra")}>
          Extra flags{extra.trim() && <b>•</b>}
        </button>
      </div>

      <div className="d-server__body">
        <div className="d-server__title">
          <div>
            <h2>Server settings</h2>
            <p className="d-muted">
              {snapshot.model.ok ? <>For <b>{snapshot.model.name}</b> · </> : null}
              {snapshot.ownProfile ? "saved for this model" : "using the defaults for new models"}
              {running && " · changes apply on the next start"}
            </p>
          </div>
          <div className="d-row">
            <button className="button button--small" onClick={() => setImporting(true)}><ClipboardPaste size={13} />Paste a command</button>
            {snapshot.model.ok && (
              <button className="button button--small" title="Use these settings for every model without its own"
                      onClick={async () => { await call("/api/profile", { model: "", values }); toast("Saved as the default for new models", "ok"); }}>
                Make default
              </button>
            )}
            <button className="button button--small" disabled={!changedCount && !extra}
                    onClick={async () => {
                      if (!confirm("Reset every option for this model to the defaults?")) return;
                      await call("/api/profile", { model, reset: true });
                      setLoadedFor(null);
                    }}>
              <RotateCcw size={13} />Reset
            </button>
            {running && (
              <button className="button button--primary button--small" onClick={async () => {
                await call("/api/stop");
                const wait = async () => {
                  for (let i = 0; i < 100; i++) {
                    await new Promise((r) => setTimeout(r, 300));
                    const st = useDesk.getState().snapshot?.status;
                    if (st === "idle" || st === "failed") break;
                  }
                  await call("/api/start");
                };
                void wait();
              }}>Restart to apply</button>
            )}
          </div>
        </div>

        {importing && (
          <div className="d-card d-import">
            <p className="d-small">Paste a <code>flyweight serve …</code> line. Its options replace this profile's.</p>
            <textarea className="mono" rows={3} value={importText} onChange={(e) => setImportText(e.target.value)}
                      placeholder="flyweight serve model.gguf --context 65536 --expert-mode cpu --cache-type-k q8_0" autoFocus />
            <div className="d-row">
              <button className="button button--small" onClick={() => setImporting(false)}>Cancel</button>
              <button className="button button--primary button--small" disabled={!importText.trim()} onClick={() => void doImport()}>Import</button>
            </div>
          </div>
        )}

        {section === "extra" && !filtering ? (
          <section className="d-group">
            <h3>Extra flags</h3>
            <p className="d-muted d-small">Passed to <code>flyweight serve</code> as typed, after everything above. For flags a newer engine knows and this app does not show yet.</p>
            <textarea className="mono" rows={3} value={extra} spellCheck={false}
                      onChange={(e) => set("_extra", e.target.value || undefined)} placeholder="--some-new-flag 1" />
            <label className="d-check">
              <input type="checkbox" checked={values._auto_mmproj !== false}
                     onChange={(e) => set("_auto_mmproj", e.target.checked ? undefined : false)} />
              Use a vision projector (mmproj) found beside the model
            </label>
            <label className="d-check">
              <input type="checkbox" checked={snapshot.settings.open_chat}
                     onChange={(e) => void call("/api/settings", { open_chat: e.target.checked })} />
              Open the chat when the model is ready
            </label>
          </section>
        ) : (
          shown.map((s) => {
            const opts = s.options.filter(matches);
            if (!opts.length) return null;
            return (
              <section key={s.id} className="d-group">
                <h3>{s.title}</h3>
                {s.description && !filtering && <p className="d-muted d-small">{s.description}</p>}
                {opts.map((o) => (
                  <OptionField key={`${s.id}-${o.dest}`} spec={o} value={values[o.dest] ?? undefined} onChange={(v) => set(o.dest, v)} />
                ))}
              </section>
            );
          })
        )}
        {filtering && !sections.slice(1).some((s) => s.options.some(matches)) && (
          <div className="d-muted d-pad">No option matches.</div>
        )}
      </div>

      <aside className="d-server__side">
        <section className="d-card">
          <div className="d-card__head">
            <Terminal size={16} /><h3>Command</h3><span className="d-spacer" />
            <button className="d-iconbtn" title="Copy" onClick={() => { void navigator.clipboard.writeText(command); toast("Copied", "ok"); }}>
              <Copy size={14} />
            </button>
          </div>
          <pre className="d-pre d-pre--wrap">{command}</pre>
          <p className="d-muted d-small">{changedCount} of {options.filter((o) => !o.positional).length} options changed. The app picks a free port if this one is busy.</p>
        </section>
        <TuneCard values={values} onApply={(changes, context) => {
          const next: Values = { ...values, context_window: context };
          for (const [dest, value] of Object.entries(changes)) {
            if (value === null) delete next[dest];
            else next[dest] = value;
          }
          save(next);
          toast("Applied. Start the model to use them.", "ok");
        }} />
        <PlanCard values={values} />
        <ModelCard />
      </aside>
    </div>
  );
}
