import { useStore } from "../store";
import { formatCompact } from "../lib/format";
import type { PrefillProgress } from "../types";

/**
 * What the server is doing before it can answer.
 *
 * Prefill is the one phase that produces nothing to show: on a long prompt
 * the model can spend twenty seconds reading it, and a blinking cursor says
 * only "something, or nothing, is happening". This says which, how far along
 * it is, and when it will be done.
 */
export function PrefillStatus({ messageId }: { messageId: string }) {
  const prefill = useStore((state) => (state.generating?.messageId === messageId ? state.generating.prefill : undefined));
  if (!prefill) return null;
  const { processed, total } = prefill;
  const percent = total > 0 ? Math.min(100, Math.round((100 * processed) / total)) : 0;
  return (
    <div className="prefill" role="status" aria-live="polite">
      <div className="prefill__head">
        <span className="prefill__label">Reading the prompt</span>
        <span className="prefill__percent">{percent}%</span>
      </div>
      <div
        className="prefill__track"
        role="progressbar"
        aria-valuemin={0}
        aria-valuemax={total}
        aria-valuenow={processed}
        aria-valuetext={`${percent}% of ${total} tokens`}
      >
        <div className="prefill__fill" style={{ width: `${percent}%` }} />
      </div>
      <div className="prefill__detail">{prefillDetail(prefill)}</div>
    </div>
  );
}

/**
 * The line under the bar. The cached share is the interesting number when a
 * conversation is being continued -- it is the difference between a prompt
 * that is long and a prompt that is new -- and the estimate is what tells a
 * user whether to wait.
 */
export function prefillDetail(prefill: PrefillProgress): string {
  const { processed, total, cached, tokensPerSecond, etaSeconds } = prefill;
  const parts = [`${formatCompact(processed)} of ${formatCompact(total)} tokens`];
  if (cached > 0) parts.push(`${formatCompact(cached)} reused from cache`);
  if (tokensPerSecond > 0 && processed < total) parts.push(`${Math.round(tokensPerSecond)} tok/s`);
  if (etaSeconds !== undefined && processed < total) parts.push(`about ${formatEta(etaSeconds)} left`);
  return parts.join(" · ");
}

/** Seconds a person would say out loud, not a decimal. */
export function formatEta(seconds: number): string {
  if (seconds < 1) return "a second";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds - minutes * 60);
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}
