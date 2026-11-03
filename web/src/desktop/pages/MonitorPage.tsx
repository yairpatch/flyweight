import { useEffect, useRef, useState } from "react";
import { Gauge, Trash2 } from "lucide-react";
import { desk, type MonitorData, type RequestRow } from "../api";
import { useDesk, gb, clock } from "../state";
import { Meter, Sparkline, StatTile, KeyValueTable } from "../../components/charts";

interface Sample {
  at: number;
  cpu: number | null;
  ram: number | null;
  vram: number | null;
  util: number | null;
}

const num = (value: unknown): number => (typeof value === "number" && Number.isFinite(value) ? value : 0);
const pct = (a: number, b: number) => (b > 0 ? `${Math.round((100 * a) / b)}%` : "—");

/** Keep polling while the page is open; history survives page switches. */
let history: Sample[] = [];
let requests: RequestRow[] = [];
let lastAt = 0;

export function MonitorPage() {
  const status = useDesk((s) => s.snapshot?.status);
  const [data, setData] = useState<MonitorData | null>(null);
  const [, force] = useState(0);
  const timer = useRef(0);

  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const fresh = await desk<MonitorData>(`/api/monitor?since=${lastAt}`);
        if (stop) return;
        const sys = fresh.system;
        const gpu = sys.gpus[0];
        history = [...history, {
          at: sys.at, cpu: sys.cpu,
          ram: sys.memory.total && sys.memory.available != null ? sys.memory.total - sys.memory.available : null,
          vram: gpu?.used ?? null, util: gpu?.util ?? null,
        }].slice(-150);
        if (fresh.requests.length) {
          requests = [...requests, ...fresh.requests].slice(-300);
          lastAt = fresh.requests[fresh.requests.length - 1].at;
        }
        setData(fresh);
        force((n) => n + 1);
      } catch {
        /* next tick */
      }
      if (!stop) timer.current = window.setTimeout(tick, 2000);
    };
    void tick();
    return () => {
      stop = true;
      window.clearTimeout(timer.current);
    };
  }, []);

  if (!data) return <div className="d-loading">Reading this computer…</div>;
  const sys = data.system;
  const health = data.health as Record<string, unknown> | null;
  const exec = (health?.execution ?? {}) as Record<string, unknown>;
  const prefix = (health?.prefix_cache ?? {}) as Record<string, unknown>;
  const props = data.props as Record<string, unknown> | null;
  const done = requests.filter((r) => r.rate);
  const rates = done.map((r) => r.rate ?? 0);
  const ttfts = requests.filter((r) => r.ttft != null).map((r) => r.ttft ?? 0);
  const last = done[done.length - 1];
  const avg = rates.length ? rates.slice(-10).reduce((a, b) => a + b, 0) / Math.min(10, rates.length) : 0;
  const memTotal = sys.memory.total ?? 0;
  const memUsed = memTotal - (sys.memory.available ?? memTotal);
  const uptime = data.readyAt ? Date.now() / 1000 - data.readyAt : 0;
  const expertHits = num(exec.expert_cache_hits);
  const expertMiss = num(exec.expert_cache_misses);
  const mtpDraft = num(exec.mtp_draft_tokens);
  const mtpAccepted = num(exec.mtp_accepted_tokens);
  const slots = data.slots?.slots ?? [];

  return (
    <div className="d-scroll">
      <div className="d-pagewrap">
        <div className="d-pagehead">
          <div>
            <h2><Gauge size={20} /> Monitor</h2>
            <p className="d-muted">
              {status === "running" ? `Serving ${String(health?.model ?? "")} for ${clock(uptime)}` : "No model is running; showing this computer."}
            </p>
          </div>
          <button className="button button--small" onClick={() => { requests = []; lastAt = Date.now() / 1000; force((n) => n + 1); }}>
            <Trash2 size={13} />Clear requests
          </button>
        </div>

        <div className="stat-grid d-stats">
          <StatTile label="Decode speed" value={last ? `${last.rate?.toFixed(1)} tok/s` : "—"} hint="Last request" trend={rates.slice(-40)} />
          <StatTile label="Average (last 10)" value={avg ? `${avg.toFixed(1)} tok/s` : "—"} />
          <StatTile label="Time to first token" value={ttfts.length ? `${ttfts[ttfts.length - 1].toFixed(2)} s` : "—"} trend={ttfts.slice(-40)} />
          <StatTile label="Requests" value={String(requests.length)}
                    hint="Since the monitor opened" >
            <small className="d-muted">{num(health?.active_requests)} active of {num(health?.request_capacity) || "—"}</small>
          </StatTile>
        </div>

        <div className="d-grid2">
          <section className="d-card">
            <div className="d-card__head"><h3>Memory</h3></div>
            {sys.gpus.map((g, i) => (
              <Meter key={i} label={`${g.name}${sys.gpus.length > 1 ? ` #${i}` : ""}`}
                     ratio={g.total ? (g.used ?? 0) / g.total : null}
                     detail={g.total ? `${gb(g.used)} of ${gb(g.total)}` : "—"} />
            ))}
            <Meter label="System memory" ratio={memTotal ? memUsed / memTotal : null} detail={memTotal ? `${gb(memUsed)} of ${gb(memTotal)}` : "—"} />
            <Meter label="Engine process" ratio={memTotal && sys.engine_rss ? sys.engine_rss / memTotal : null}
                   detail={sys.engine_rss ? gb(sys.engine_rss) : "not running"} severity="accent" />
            <div className="d-sparkrow">
              <div><small className="d-muted">CPU {sys.cpu != null ? `${Math.round(sys.cpu)}%` : "—"}</small>
                <Sparkline values={history.map((h) => h.cpu ?? 0)} height={44} compact /></div>
              {sys.gpus[0] && (
                <div><small className="d-muted">GPU {sys.gpus[0].util != null ? `${Math.round(sys.gpus[0].util)}%` : "—"}
                  {sys.gpus[0].temp != null ? ` · ${sys.gpus[0].temp}°C` : ""}{sys.gpus[0].power != null ? ` · ${Math.round(sys.gpus[0].power)} W` : ""}</small>
                  <Sparkline values={history.map((h) => h.util ?? 0)} height={44} compact /></div>
              )}
            </div>
          </section>

          <section className="d-card">
            <div className="d-card__head"><h3>Decode speed per request</h3></div>
            <Sparkline values={rates.slice(-60)} labels={done.slice(-60).map((r) => `${r.time} · ${r.endpoint} · ${r.out ?? 0} tokens`)}
                       height={150} format={(v) => `${v.toFixed(1)} tok/s`} ariaLabel="Decode speed per request" />
            {rates.length < 2 && <p className="d-muted d-small">Each finished request adds a point; the line appears after two.</p>}
          </section>
        </div>

        {status === "running" && health && (
          <div className="d-grid2">
            <section className="d-card">
              <div className="d-card__head"><h3>Engine</h3></div>
              <KeyValueTable rows={[
                ["Backend", `${String(exec.backend ?? "—")} · experts ${String(exec.expert_mode ?? exec.moe_device ?? "—")}`],
                ["Context window", `${num(health.context_window).toLocaleString()} tokens`],
                ["Slots", `${slots.length} (${slots.filter((s) => s.is_processing).length} busy)`],
                ["KV cache", `${typeof exec.cache_type_k === "string" ? `${exec.cache_type_k}/${String(exec.cache_type_v)} · ` : ""}${gb(num(exec.kv_reserved_bytes) || num(prefix.kv_reserved_bytes))} reserved`],
                ["GPU allocated", gb(num(exec.gpu_allocated_bytes))],
                ["Weights on GPU", gb(num(exec.static_tensor_bytes))],
                ["Layers", `${num(exec.layers)}${num(exec.expert_count) ? ` · ${num(exec.expert_count)} experts, ${num(exec.expert_used_count)} per token` : ""}`],
                ["Chat template", String(props?.chat_template ?? "—")],
              ]} />
            </section>
            <section className="d-card">
              <div className="d-card__head"><h3>Caches</h3></div>
              <KeyValueTable rows={[
                ["Prompt reuse", `${num(prefix.hits)} hits · ${num(prefix.misses)} misses · ${num(prefix.reused_tokens).toLocaleString()} tokens reused`],
                ["Prompt cache (RAM)", `${num(prefix.ram_entries)} entries · ${gb(num(prefix.ram_bytes))}`],
                ...(expertHits + expertMiss > 0
                  ? [["Expert cache", `${pct(expertHits, expertHits + expertMiss)} hit rate · ${gb(num(exec.expert_cache_bytes))} · ${num(exec.expert_cache_slots)} slots`] as [string, string]]
                  : []),
                ...(num(exec.mtp_enabled)
                  ? [["Speculative drafts", `${pct(mtpAccepted, mtpDraft)} accepted (${mtpAccepted.toLocaleString()} of ${mtpDraft.toLocaleString()})`] as [string, string]]
                  : []),
                ["Prefill", `${num(exec.prefill_tokens).toLocaleString()} tokens in ${(num(exec.prefill_nanoseconds) / 1e9).toFixed(1)} s`
                  + (num(exec.prefill_nanoseconds) ? ` · ${Math.round(num(exec.prefill_tokens) / (num(exec.prefill_nanoseconds) / 1e9))} tok/s` : "")],
              ]} />
            </section>
          </div>
        )}

        <section className="d-card">
          <div className="d-card__head"><h3>Requests</h3><span className="d-spacer" /><small className="d-muted">{requests.length} recorded</small></div>
          {requests.length === 0 ? <p className="d-muted d-small">Chats and API calls appear here as they finish.</p> : (
            <div className="d-tablewrap">
              <table className="d-table">
                <thead><tr><th>Time</th><th>Endpoint</th><th>Prompt</th><th>Cached</th><th>TTFT</th><th>Output</th><th>tok/s</th><th>Finish</th><th>Total</th></tr></thead>
                <tbody>
                  {requests.slice(-60).reverse().map((r, i) => (
                    <tr key={`${r.at}-${i}`} className={r.error || /^\d{3}$/.test(r.finish) ? "d-table__bad" : ""} title={r.error}>
                      <td>{r.time}</td><td>{r.endpoint}</td><td>{r.prompt?.toLocaleString() ?? "—"}</td>
                      <td>{r.cached != null ? `${r.cached}%` : "—"}</td><td>{r.ttft != null ? `${r.ttft}s` : "—"}</td>
                      <td>{r.out ?? "—"}</td><td>{r.rate ?? "—"}</td><td>{r.finish}</td><td>{r.total != null ? `${r.total}s` : "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
