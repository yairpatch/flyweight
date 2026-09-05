import { useMemo, useState } from "react";
import { useStore } from "../../store";
import { KeyValueTable, Meter, Sparkline, StatTile } from "../charts";
import { formatBytes, formatCompact, formatInteger, formatNanoseconds, formatPercent, formatRate, formatSeconds, formatTime } from "../../lib/format";
import { runtimeDeviceLabel } from "../TopBar";

type Exec = Record<string, unknown>;

function n(exec: Exec, key: string): number | undefined {
  const value = exec[key];
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function ratio(hits?: number, misses?: number): number | null {
  if (hits === undefined || misses === undefined) return null;
  const total = hits + misses;
  return total > 0 ? hits / total : null;
}

function formatValue(key: string, value: unknown): string {
  if (value === null || value === undefined) return "–";
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "number") {
    if (/bytes$/.test(key)) return formatBytes(value);
    if (/nanoseconds$/.test(key)) return formatNanoseconds(value);
    if (/seconds$/.test(key)) return formatSeconds(value);
    return formatInteger(value);
  }
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

export function RuntimePanel() {
  const health = useStore((state) => state.health);
  const history = useStore((state) => state.healthHistory);
  const props = useStore((state) => state.props);
  const slots = useStore((state) => state.slots);
  const models = useStore((state) => state.models);
  const status = useStore((state) => state.status);
  const [showAll, setShowAll] = useState(false);
  const [showRaw, setShowRaw] = useState(false);

  const exec: Exec = health?.execution ?? {};
  const prefix = health?.prefix_cache ?? {};

  // Throughput over time from the decode counters between health polls.
  const throughput = useMemo(() => {
    const values: number[] = [];
    const labels: string[] = [];
    for (let index = 1; index < history.length; index += 1) {
      const a = history[index - 1];
      const b = history[index];
      const ea = (a.health.execution ?? {}) as Exec;
      const eb = (b.health.execution ?? {}) as Exec;
      const tokens = (n(eb, "decode_calls") ?? 0) + (n(eb, "mtp_accepted_tokens") ?? 0) - (n(ea, "decode_calls") ?? 0) - (n(ea, "mtp_accepted_tokens") ?? 0);
      const ns = (n(eb, "decode_nanoseconds") ?? 0) - (n(ea, "decode_nanoseconds") ?? 0);
      values.push(ns > 0 && tokens > 0 ? tokens / (ns / 1e9) : 0);
      labels.push(formatTime(b.at));
    }
    return { values, labels };
  }, [history]);

  if (!health) {
    return <p className="muted">{status === "offline" ? "The server is unreachable." : status === "locked" ? "Enter the API key in settings to read runtime telemetry." : "Waiting for the first health sample…"}</p>;
  }

  const decodeCalls = n(exec, "decode_calls");
  const decodeNs = n(exec, "decode_nanoseconds");
  const mtpAccepted = n(exec, "mtp_accepted_tokens");
  const lifetimeRate = decodeCalls && decodeNs ? ((decodeCalls + (mtpAccepted ?? 0)) / (decodeNs / 1e9)) : undefined;
  const prefillTokens = n(exec, "prefill_tokens");
  const prefillNs = n(exec, "prefill_nanoseconds");
  const prefillRate = prefillTokens && prefillNs ? prefillTokens / (prefillNs / 1e9) : undefined;

  const gpuAllocated = n(exec, "gpu_allocated_bytes");
  const memoryParts: Array<[string, number | undefined]> = [
    ["Static tensors", n(exec, "static_tensor_bytes")],
    ["Expert tensors", n(exec, "expert_tensor_bytes")],
    ["Expert cache", n(exec, "expert_cache_bytes")],
    ["KV reserved", n(exec, "kv_reserved_bytes") ?? (typeof prefix.kv_reserved_bytes === "number" ? prefix.kv_reserved_bytes : undefined)],
    ["Workspace", n(exec, "workspace_bytes")],
    ["State", n(exec, "state_bytes")],
    ["Expert staging", n(exec, "expert_staging_bytes")],
    ["MTP tensors", n(exec, "mtp_tensor_bytes")],
  ];

  const expertHits = n(exec, "expert_cache_hits");
  const expertMisses = n(exec, "expert_cache_misses");
  const prefixHits = n(exec, "prefix_cache_hits") ?? (typeof prefix.hits === "number" ? prefix.hits : undefined);
  const prefixMisses = n(exec, "prefix_cache_misses") ?? (typeof prefix.misses === "number" ? prefix.misses : undefined);
  const kvPeak = n(exec, "kv_peak_tokens") ?? (typeof prefix.kv_peak_tokens === "number" ? prefix.kv_peak_tokens : undefined);
  const kvMax = n(exec, "kv_peak_tokens_max") ?? (typeof prefix.kv_peak_tokens_max === "number" ? prefix.kv_peak_tokens_max : undefined) ?? health.context_window;
  const mtpDraft = n(exec, "mtp_draft_tokens");
  const mtpRejected = n(exec, "mtp_rejected_tokens");
  const grammarSteps = n(exec, "grammar_constrained_steps");
  const vision = exec.vision as Record<string, unknown> | null | undefined;
  const uptime = health.loaded_at ? Date.now() / 1000 - health.loaded_at : undefined;

  const allRows = Object.keys(exec)
    .filter((key) => key !== "vision")
    .sort()
    .map((key) => [key, formatValue(key, exec[key])] as [string, string]);

  return (
    <div className="runtime">
      <section className="runtime__section">
        <h3>Model</h3>
        <KeyValueTable
          rows={[
            ["Model", health.model],
            ["Backend", String(exec.backend ?? "–")],
            ["Device", runtimeDeviceLabel(exec)],
            ...(exec.architecture ? [["Architecture", String(exec.architecture)] as [string, string]] : []),
            ["Context window", formatInteger(health.context_window ?? props?.context_window)],
            // --max-tokens: what a request gets when it asks for nothing, not
            // a limit on what it may ask for.
            ["Default max tokens", formatInteger(props?.max_output_tokens)],
            ...(exec.cache_type_k ? [["KV cache types", `${exec.cache_type_k} / ${exec.cache_type_v}`] as [string, string]] : []),
            ...(n(exec, "layers") !== undefined ? [["Layers", `${formatInteger(n(exec, "layers"))}${n(exec, "attention_layers") !== undefined ? ` (${n(exec, "attention_layers")} attn, ${n(exec, "deltanet_layers") ?? 0} deltanet, ${n(exec, "swa_layers") ?? 0} swa)` : ""}`] as [string, string]] : []),
            ...(n(exec, "expert_count") !== undefined ? [["Experts", `${formatInteger(n(exec, "expert_count"))} total · ${formatInteger(n(exec, "expert_used_count"))} active`] as [string, string]] : []),
            ["Chat template", `${props?.chat_template ?? "–"}${props?.chat_template_source ? ` (${props.chat_template_source})` : ""}`],
            ["Loaded", uptime !== undefined ? `${formatSeconds(uptime)} ago` : "–"],
            ["Requests", `${health.active_requests ?? 0} active / ${health.request_capacity ?? "–"} capacity`],
            ["Slots", slots.length ? slots.map((slot) => `#${slot.id} ${slot.is_processing ? "busy" : "idle"}`).join(", ") : "–"],
            ["Models served", models.map((model) => model.id).join(", ") || health.model],
            ["Capabilities", (props?.capabilities ?? []).join(", ") || "–"],
          ]}
        />
      </section>

      <section className="runtime__section">
        <h3>Throughput</h3>
        <div className="stat-grid">
          <StatTile label="Lifetime decode" value={formatRate(lifetimeRate)} hint="(decode steps + MTP accepted) / decode time" />
          <StatTile label="Prefill" value={formatRate(prefillRate)} hint="prefill tokens / prefill time" />
          <StatTile label="Decode steps" value={formatCompact(decodeCalls)} />
          <StatTile label="Prefill tokens" value={formatCompact(prefillTokens)} />
        </div>
        <div className="chart-card">
          <div className="chart-card__title">Decode tokens per second, by health poll</div>
          {throughput.values.length > 1 ? <Sparkline values={throughput.values} labels={throughput.labels} format={(v) => formatRate(v)} ariaLabel="Decode throughput over time" /> : <p className="muted">Collecting samples…</p>}
        </div>
      </section>

      {gpuAllocated !== undefined && (
        <section className="runtime__section">
          <h3>Memory</h3>
          <div className="stat-grid">
            <StatTile label="GPU allocated" value={formatBytes(gpuAllocated)} />
            <StatTile label="Host available" value={formatBytes(n(exec, "host_available_bytes"))} />
            <StatTile label="Host FFN" value={formatBytes(n(exec, "host_ffn_bytes"))} />
            <StatTile label="Prompt cache" value={formatBytes(n(exec, "prompt_cache_used_bytes"))} hint={`${formatInteger(n(exec, "prompt_cache_entries"))} entries`} />
          </div>
          {memoryParts
            .filter(([, value]) => value !== undefined && value > 0)
            .map(([label, value]) => (
              <Meter key={label} label={label} ratio={gpuAllocated ? value! / gpuAllocated : null} detail={formatBytes(value)} severity="accent" />
            ))}
        </section>
      )}

      <section className="runtime__section">
        <h3>KV and prefix cache</h3>
        <Meter label="KV peak occupancy" ratio={kvPeak !== undefined && kvMax ? kvPeak / kvMax : null} detail={`${formatInteger(kvPeak)} / ${formatInteger(kvMax)} tokens`} />
        <Meter label="Prefix cache hit rate" ratio={ratio(prefixHits, prefixMisses)} detail={`${formatInteger(prefixHits)} hits · ${formatInteger(prefixMisses)} misses`} severity="accent" />
        <KeyValueTable
          rows={[
            ["Reused tokens", formatInteger(n(exec, "prefix_cache_reused_tokens") ?? (prefix.reused_tokens as number))],
            ["Re-prefilled tokens", formatInteger(n(exec, "prefix_cache_reprefilled_tokens"))],
            ["Last prompt / reused", `${formatInteger(prefix.last_prompt_tokens as number)} / ${formatInteger(prefix.last_reused_tokens as number)}`],
            ["Entries", `${formatInteger(prefix.entries as number)} of ${formatInteger(prefix.capacity as number)}${typeof prefix.ram_entries === "number" ? ` · ${prefix.ram_entries} in RAM (${formatBytes(prefix.ram_bytes as number)})` : ""}`],
            ["Evictions", formatInteger(prefix.evictions as number)],
            ["Donations", `${formatInteger(n(exec, "prefix_donations") ?? (prefix.donations as number))} · ${formatInteger(n(exec, "prefix_donated_tokens") ?? (prefix.donated_tokens as number))} tokens`],
            ["KV slots", formatInteger(n(exec, "kv_slots"))],
            ["KV peak live", formatBytes(n(exec, "kv_peak_live_bytes") ?? (prefix.kv_peak_live_bytes as number))],
          ]}
        />
      </section>

      {expertHits !== undefined && (
        <section className="runtime__section">
          <h3>Expert cache</h3>
          <Meter label="Hit rate" ratio={ratio(expertHits, expertMisses)} detail={`${formatCompact(expertHits)} hits · ${formatCompact(expertMisses)} misses`} severity="accent" />
          <KeyValueTable
            rows={[
              ["Mode", `${exec.expert_mode ?? "–"}${exec.requested_expert_mode && exec.requested_expert_mode !== exec.expert_mode ? ` (requested ${exec.requested_expert_mode})` : ""}`],
              ...(exec.expert_fallback_reason ? [["Fallback reason", String(exec.expert_fallback_reason)] as [string, string]] : []),
              ["Cache", `${formatBytes(n(exec, "expert_cache_bytes"))} · ${formatInteger(n(exec, "expert_cache_slots"))} slots`],
              ["Evictions", formatInteger(n(exec, "expert_cache_evictions"))],
              ["Admissions / rejections", `${formatInteger(n(exec, "expert_cache_admissions"))} / ${formatInteger(n(exec, "expert_cache_rejections"))}`],
              ["Deferred / unused admissions", `${formatInteger(n(exec, "expert_cache_deferred_admissions"))} / ${formatInteger(n(exec, "expert_cache_unused_admissions"))}`],
              ["Prompt bypasses", formatInteger(n(exec, "expert_cache_prompt_bypasses"))],
              ["Residency epochs", `${formatInteger(n(exec, "expert_residency_epochs"))}${exec.expert_residency_frozen ? " (frozen)" : ""}`],
            ]}
          />
        </section>
      )}

      {(exec.mtp_available !== undefined || mtpDraft !== undefined) && (
        <section className="runtime__section">
          <h3>Multi-token prediction</h3>
          <Meter label="Draft acceptance" ratio={mtpDraft ? (mtpAccepted ?? 0) / mtpDraft : null} detail={`${formatCompact(mtpAccepted)} accepted · ${formatCompact(mtpRejected)} rejected`} severity="accent" />
          <KeyValueTable
            rows={[
              ["Available / enabled", `${formatValue("", exec.mtp_available)} / ${formatValue("", exec.mtp_enabled)}`],
              ["Drafts per step", formatInteger(n(exec, "mtp_drafts"))],
              ["Draft time", formatNanoseconds(n(exec, "mtp_draft_nanoseconds"))],
              ["Verify time", formatNanoseconds(n(exec, "mtp_verify_nanoseconds"))],
              ["Rollback time", formatNanoseconds(n(exec, "mtp_rollback_nanoseconds"))],
            ]}
          />
        </section>
      )}

      {decodeNs !== undefined && (
        <section className="runtime__section">
          <h3>Time breakdown</h3>
          {(
            [
              ["Decode", decodeNs],
              ["Prefill", prefillNs],
              ["Route wait", n(exec, "route_wait_nanoseconds")],
              ["Expert paging", n(exec, "expert_page_nanoseconds")],
              ["Expert compute", n(exec, "expert_compute_nanoseconds")],
              ["Tail wait", n(exec, "tail_wait_nanoseconds")],
              ["Sampling", n(exec, "sampling_nanoseconds")],
              ["Dense on host", n(exec, "dense_host_nanoseconds")],
            ] as Array<[string, number | undefined]>
          )
            .filter(([, value]) => value !== undefined && value > 0)
            .map(([label, value]) => (
              <Meter key={label} label={label} ratio={(decodeNs + (prefillNs ?? 0)) > 0 ? value! / (decodeNs + (prefillNs ?? 0)) : null} detail={formatNanoseconds(value)} severity="accent" />
            ))}
        </section>
      )}

      {grammarSteps !== undefined && (
        <section className="runtime__section">
          <h3>Grammar enforcement</h3>
          <KeyValueTable
            rows={[
              ["Constrained steps", formatInteger(grammarSteps)],
              ["Rejected candidates", formatInteger(n(exec, "grammar_rejected_candidates"))],
              ["Empty candidate sets", formatInteger(n(exec, "grammar_empty_candidate_sets"))],
            ]}
          />
        </section>
      )}

      {vision && (
        <section className="runtime__section">
          <h3>Vision</h3>
          <KeyValueTable rows={Object.entries(vision).map(([key, value]) => [key, formatValue(key, value)] as [string, string])} />
        </section>
      )}

      <section className="runtime__section">
        <button className="button button--small" onClick={() => setShowAll((value) => !value)}>
          {showAll ? "Hide" : "Show"} all {allRows.length} execution fields
        </button>
        {showAll && <KeyValueTable rows={allRows} />}
        <button className="button button--small" onClick={() => setShowRaw((value) => !value)}>
          {showRaw ? "Hide" : "Show"} raw /health and /props
        </button>
        {showRaw && (
          <pre className="raw">
            <code>{JSON.stringify({ health, props }, null, 2)}</code>
          </pre>
        )}
        <p className="muted">
          Hit rate = {formatPercent(ratio(prefixHits, prefixMisses))} prefix · {formatPercent(ratio(expertHits, expertMisses))} experts. Polled every 5 s while the tab is visible.
        </p>
      </section>
    </div>
  );
}
