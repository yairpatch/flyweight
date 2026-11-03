import { useEffect, useMemo, useState } from "react";
import { Copy, FlaskConical, Play, Square, Wrench } from "lucide-react";
import { useDesk, labelOf } from "../state";
import type { JobDetail, Scalar, Values } from "../api";
import { OptionField, isDefault, invalid } from "../components/OptionField";
import { useJob, latestJob } from "../components/useJob";
import { Markdown } from "../../components/Markdown";
import { StatTile, KeyValueTable } from "../../components/charts";

type Tool = "benchmark" | "generate" | "imatrix" | "transcript-audit";

const TOOLS: Array<{ id: Tool; title: string; blurb: string; usesModel: boolean }> = [
  { id: "benchmark", title: "Benchmark", blurb: "Measure prompt processing and decode speed with the model's settings, without serving.", usesModel: true },
  { id: "generate", title: "Generate", blurb: "One prompt, one answer, then the model unloads. Handy for checking a setting quickly.", usesModel: true },
  { id: "imatrix", title: "Importance matrix", blurb: "Gather an importance matrix over calibration text, for packing safetensors to IQ formats.", usesModel: true },
  { id: "transcript-audit", title: "Transcript audit", blurb: "Explain a coding harness's bad edits from a request dump folder.", usesModel: false },
];

function seconds(value: unknown): string {
  return typeof value === "number" ? `${value.toFixed(2)} s` : "—";
}

function BenchmarkResult({ result }: { result: Record<string, unknown> }) {
  const rate = result.decode_tokens_per_second as number | undefined;
  const prompt = result.prompt_tokens as number | undefined;
  const first = result.prompt_and_first_token_seconds as number | undefined;
  return (
    <>
      <div className="stat-grid d-stats">
        <StatTile label="Decode" value={rate ? `${rate.toFixed(1)} tok/s` : "—"} />
        <StatTile label="Prompt + first token" value={seconds(first)} hint={`${prompt ?? "?"} prompt tokens`} />
        <StatTile label="Prompt speed" value={first && prompt ? `${Math.round(prompt / first)} tok/s` : "—"} hint="Approximate: includes the first decode step" />
        <StatTile label="Load" value={seconds(result.prepare_seconds)} />
      </div>
      <KeyValueTable rows={[
        ["Execution", String(result.execution ?? "—")],
        ["Generated", `${(result.generated_tokens as unknown[] | undefined)?.length ?? 0} tokens`],
        ["First window", typeof result.decode_first_window_tokens_per_second === "number" ? `${(result.decode_first_window_tokens_per_second as number).toFixed(1)} tok/s` : "—"],
        ["Last window", typeof result.decode_last_window_tokens_per_second === "number" ? `${(result.decode_last_window_tokens_per_second as number).toFixed(1)} tok/s` : "—"],
        ["Median step", typeof result.decode_median_seconds === "number" ? `${((result.decode_median_seconds as number) * 1000).toFixed(1)} ms` : "—"],
        ["Request", seconds(result.request_seconds)],
      ]} />
      {typeof result.generated_text === "string" && result.generated_text && (
        <details className="d-details"><summary>Generated text</summary><pre className="d-pre d-pre--wrap">{result.generated_text}</pre></details>
      )}
    </>
  );
}

function JobView({ job }: { job: JobDetail }) {
  const call = useDesk((s) => s.call);
  const toast = useDesk((s) => s.toast);
  const command = `flyweight ${job.argv.join(" ")}`;
  const elapsed = (job.ended || Date.now() / 1000) - job.started;
  return (
    <section className="d-card">
      <div className="d-card__head">
        <FlaskConical size={16} /><h3>{job.title}</h3>
        <span className={`d-state d-state--${job.state}`}>{job.state} · {elapsed.toFixed(0)} s</span>
        <span className="d-spacer" />
        <button className="d-iconbtn" title="Copy the command" onClick={() => { void navigator.clipboard.writeText(command); toast("Copied", "ok"); }}><Copy size={14} /></button>
        {job.state === "running" && <button className="button button--small" onClick={() => void call("/api/jobs/cancel", { id: job.id })}><Square size={12} />Stop</button>}
      </div>
      {job.state === "running" && <div className="d-bar"><i /></div>}
      {job.state === "failed" && <div className="d-alert d-alert--error">{job.error}</div>}
      {job.state === "done" && job.command === "benchmark" && typeof job.result === "object" && job.result && (
        <BenchmarkResult result={job.result as Record<string, unknown>} />
      )}
      {job.state === "done" && job.command === "generate" && (
        <div className="d-answer"><Markdown text={job.stdout} /></div>
      )}
      {job.state === "done" && (job.command === "transcript-audit" || job.command === "imatrix") && job.stdout.trim() && (
        <pre className="d-pre d-pre--wrap">{job.stdout}</pre>
      )}
      <details className="d-details" open={job.state === "running"}>
        <summary>Engine output ({job.log.length} lines)</summary>
        <pre className="d-pre d-pre--log">{job.log.slice(-200).join("\n")}</pre>
      </details>
    </section>
  );
}

export function ToolsPage() {
  const snapshot = useDesk((s) => s.snapshot);
  const schemas = useDesk((s) => s.schemas);
  const call = useDesk((s) => s.call);
  const [tool, setTool] = useState<Tool>("benchmark");
  const [useServer, setUseServer] = useState(true);
  const [values, setValues] = useState<Record<Tool, Values>>({ benchmark: {}, generate: {}, imatrix: {}, "transcript-audit": {} });
  const [jobIds, setJobIds] = useState<Partial<Record<Tool, string>>>({});
  const job = useJob(jobIds[tool] ?? null);

  useEffect(() => {
    for (const t of TOOLS) {
      void latestJob(t.id).then((j) => j && setJobIds((ids) => ({ ...ids, [t.id]: ids[t.id] ?? j.id })));
    }
  }, []);

  const spec = schemas?.[tool];
  const serveDests = useMemo(() => new Set(schemas?.serve.groups.flatMap((g) => g.options.map((o) => o.dest)) ?? []), [schemas]);
  if (!snapshot || !spec) return <div className="d-loading">Loading tools…</div>;
  const meta = TOOLS.find((t) => t.id === tool)!;
  const profile = snapshot.profile;
  const mine = values[tool];
  const effective: Values = { ...(useServer && meta.usesModel ? profile : {}), ...mine };
  const set = (dest: string, value: Scalar | undefined) => {
    const next = { ...mine };
    if (value === undefined) delete next[dest];
    else next[dest] = value;
    setValues({ ...values, [tool]: next });
  };
  const options = spec.groups.flatMap((g) => g.options);
  const missing = options.filter((o) => o.positional || (o.required && !o.flag)).filter((o) => o.dest !== "model" || !meta.usesModel)
    .filter((o) => isDefault(o, effective[o.dest] ?? undefined));
  const requiredFlags = options.filter((o) => o.required && o.flag && isDefault(o, effective[o.dest] ?? undefined));
  const bad = options.some((o) => invalid(o, effective[o.dest] ?? undefined));
  const engineRunning = snapshot.status === "running" || snapshot.status === "loading";
  const canRun = (!meta.usesModel || snapshot.model.ok) && !missing.length && !requiredFlags.length && !bad && job?.state !== "running";

  const run = async () => {
    const positional: Record<string, string> = {};
    for (const o of options.filter((x) => x.positional)) {
      const v = effective[o.dest];
      if (v != null && v !== "") positional[o.dest] = String(v);
    }
    const r = await call<{ ok: boolean; job?: string }>("/api/jobs/run", { command: tool, values: effective, positional });
    if (r?.ok && r.job) setJobIds({ ...jobIds, [tool]: r.job });
  };

  return (
    <div className="d-scroll">
      <div className="d-pagewrap">
        <div className="d-pagehead">
          <div><h2><Wrench size={20} /> Tools</h2><p className="d-muted">The engine's other commands, without a terminal.</p></div>
        </div>
        <div className="d-tabs">
          {TOOLS.map((t) => (
            <button key={t.id} className={`d-tab${tool === t.id ? " d-tab--on" : ""}`} onClick={() => setTool(t.id)}>{t.title}</button>
          ))}
        </div>
        <p className="d-muted">{meta.blurb}</p>
        {meta.usesModel && (
          <div className="d-card d-toolmodel">
            <span>Model: <b>{snapshot.model.ok ? snapshot.model.name : "none chosen"}</b></span>
            <label className="d-check"><input type="checkbox" checked={useServer} onChange={(e) => setUseServer(e.target.checked)} />
              Start from this model's server settings</label>
            {engineRunning && <span className="d-warn">The server is running and holds memory; stop it for clean numbers.</span>}
          </div>
        )}
        <div className="d-grid2 d-grid2--tool">
          <div>
            {spec.groups.map((g) => {
              const opts = g.options.filter((o) => !(o.dest === "model" && meta.usesModel));
              // Options shared with serve are already in the server settings;
              // show the command's own ones first and fold the shared ones.
              const own = opts.filter((o) => !serveDests.has(o.dest) || !meta.usesModel || o.positional);
              const shared = opts.filter((o) => !own.includes(o));
              if (!own.length && !shared.length) return null;
              return (
                <section key={g.title} className="d-group">
                  <h3>{labelOf(g.title, g.title)}</h3>
                  {own.map((o) => <OptionField key={o.dest} spec={o} value={effective[o.dest] ?? undefined} onChange={(v) => set(o.dest, v)} />)}
                  {shared.length > 0 && (
                    <details className="d-details">
                      <summary>{shared.length} option{shared.length > 1 ? "s" : ""} shared with the server settings</summary>
                      {shared.map((o) => <OptionField key={o.dest} spec={o} value={effective[o.dest] ?? undefined} onChange={(v) => set(o.dest, v)} />)}
                    </details>
                  )}
                </section>
              );
            })}
          </div>
          <div className="d-toolside">
            <button className="button button--primary d-runwide" disabled={!canRun} onClick={() => void run()}>
              <Play size={15} />Run {meta.title.toLowerCase()}
            </button>
            {(missing.length > 0 || requiredFlags.length > 0) && (
              <p className="d-small d-warn">Needs {[...missing, ...requiredFlags].map((o) => labelOf(o.flag, o.dest)).join(", ")}.</p>
            )}
            {job ? <JobView job={job} /> : <p className="d-muted d-small">Results appear here.</p>}
          </div>
        </div>
      </div>
    </div>
  );
}
