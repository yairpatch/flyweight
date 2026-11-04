import { useEffect, useState } from "react";
import { Check, Gauge, Play } from "lucide-react";
import { useDesk } from "../state";
import type { Scalar, Values } from "../api";
import { useJob, latestJob } from "./useJob";

type Goal = "balanced" | "prefill" | "decode";

interface TuneRun {
  label: string;
  ok: boolean;
  prefill_tps: number;
  decode_tps: number;
  gain_percent: number | null;
  error: string;
}

interface TuneReport {
  context: number;
  goal: Goal;
  runs: TuneRun[];
  best: {
    ok: boolean;
    changes: Record<string, Scalar>;
    argv: string[];
    prefill_tps: number;
    decode_tps: number;
    gain_percent: number | null;
  };
  verified: boolean | null;
  verify_error: string;
  stopped: string;
}

const GOALS: { id: Goal; label: string }[] = [
  { id: "balanced", label: "Balanced" },
  { id: "prefill", label: "Prompt speed" },
  { id: "decode", label: "Generation speed" },
];

/** Search for the fastest settings with real servers, then apply the winner. */
export function TuneCard({ values, onApply }: {
  values: Values;
  onApply: (changes: Record<string, Scalar>, context: number) => void;
}) {
  const call = useDesk((s) => s.call);
  const model = useDesk((s) => s.snapshot?.model);
  const engineUp = useDesk((s) => s.snapshot?.status === "running" || s.snapshot?.status === "loading");
  const [context, setContext] = useState(String(values.context_window ?? 32768));
  const [goal, setGoal] = useState<Goal>("balanced");
  const [minutes, setMinutes] = useState("20");
  const [jobId, setJobId] = useState<string | null>(null);
  const [applied, setApplied] = useState(false);
  const job = useJob(jobId);

  useEffect(() => {
    void latestJob("tune").then((j) => j && setJobId(j.id));
  }, []);
  useEffect(() => setContext(String(values.context_window ?? 32768)), [values.context_window]);

  const contextN = Number(context);
  const minutesN = Number(minutes);
  const valid = Number.isInteger(contextN) && contextN >= 1024 && minutesN > 0;
  const run = async () => {
    if (engineUp && !confirm("The settings search needs the GPU to itself. Stop the running model?")) return;
    setApplied(false);
    const r = await call<{ ok: boolean; job?: string }>("/api/jobs/run", {
      command: "tune",
      values: { ...values, context_window: contextN, goal, time_limit: minutesN },
    });
    if (r?.ok && r.job) setJobId(r.job);
  };

  const progress = job?.log.filter((l) => l.startsWith("[tune] ")).map((l) => l.slice(7)) ?? [];
  const report = job?.state === "done" && job.result && typeof job.result === "object" ? job.result as TuneReport : null;
  const best = report?.best;
  const changed = best ? Object.keys(best.changes).length > 0 : false;

  return (
    <section className="d-card">
      <div className="d-card__head">
        <Gauge size={16} /><h3>Find best settings</h3>
        <span className="d-spacer" />
        {job?.state === "running" ? (
          <button className="button button--small" onClick={() => void call("/api/jobs/cancel", { id: job.id })}>Cancel</button>
        ) : (
          <button className="button button--small" disabled={!model?.ok || !valid} onClick={() => void run()}>
            <Play size={13} />{job ? "Search again" : "Search"}
          </button>
        )}
      </div>
      <p className="d-muted d-small">
        Starts the model with different settings, times real requests against each, and keeps what is
        fastest on this computer at the context you need.
      </p>
      {job?.state !== "running" && (
        <div className="d-kv d-tune__form">
          <span>Context</span>
          <input className="d-input d-input--num mono" inputMode="numeric" value={context} onChange={(e) => setContext(e.target.value)} />
          <span>Optimize for</span>
          <select className="d-select" value={goal} onChange={(e) => setGoal(e.target.value as Goal)}>
            {GOALS.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
          </select>
          <span>Time limit (min)</span>
          <input className="d-input d-input--num mono" inputMode="numeric" value={minutes} onChange={(e) => setMinutes(e.target.value)} />
        </div>
      )}
      {job?.state === "running" && (
        <>
          <div className="d-bar"><i /></div>
          <ul className="d-notes">{progress.slice(-8).map((line, i) => <li key={i}>{line}</li>)}</ul>
        </>
      )}
      {job?.state === "failed" && <div className="d-alert d-alert--error">{job.error}</div>}
      {report && best && (
        <>
          {best.ok ? (
            <div className="d-tune__best">
              <b>{changed ? "Faster settings found"
                : report.runs.length <= 1 ? "Only the current settings were measured"
                : "The current settings are already the fastest found"}</b>
              <span className="d-muted d-small">
                Prompt {Math.round(best.prefill_tps).toLocaleString()} tok/s · generation {best.decode_tps.toFixed(1)} tok/s
                {best.gain_percent ? ` · ${best.gain_percent > 0 ? "+" : ""}${best.gain_percent}% for “${GOALS.find((g) => g.id === report.goal)?.label}”` : ""}
              </span>
              {report.verified === false && (
                <div className="d-alert d-alert--error">It did not hold a full {report.context.toLocaleString()}-token prompt: {report.verify_error}</div>
              )}
              {report.stopped === "time limit" && (
                <span className="d-muted d-small">
                  {report.runs.length <= 1 ? "The time limit ran out before any other setup could be tried. Give it longer."
                    : "Stopped at the time limit; a longer search may find more."}
                </span>
              )}
              {(changed || Number(values.context_window ?? 32768) !== report.context) && (
                <button className="button button--primary button--small" disabled={applied}
                        onClick={() => { onApply(best.changes, report.context); setApplied(true); }}>
                  <Check size={13} />{applied ? "Applied" : "Apply to this model's settings"}
                </button>
              )}
            </div>
          ) : (
            <div className="d-alert d-alert--error">No setup ran at a {report.context.toLocaleString()}-token context. Try a smaller one.</div>
          )}
          <details className="d-details">
            <summary>Every setup tried ({report.runs.length})</summary>
            <div className="d-tablewrap">
              <table className="d-table">
                <thead><tr><th>Setup</th><th>Prompt</th><th>Generation</th><th>vs current</th></tr></thead>
                <tbody>
                  {report.runs.map((r, i) => r.ok ? (
                    <tr key={i}>
                      <td className="mono">{r.label}</td>
                      <td>{Math.round(r.prefill_tps).toLocaleString()}</td>
                      <td>{r.decode_tps.toFixed(1)}</td>
                      <td>{r.gain_percent == null ? "" : `${r.gain_percent > 0 ? "+" : ""}${r.gain_percent}%`}</td>
                    </tr>
                  ) : (
                    <tr key={i} className="d-table__bad"><td className="mono">{r.label}</td><td colSpan={3} title={r.error}>did not run</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </section>
  );
}
