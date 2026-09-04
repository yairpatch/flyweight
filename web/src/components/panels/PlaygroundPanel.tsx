import { useRef, useState } from "react";
import { Play, Square } from "lucide-react";
import { useStore } from "../../store";
import { ApiError, openStream } from "../../lib/api";
import { readSse } from "../../lib/sse";
import { formatRate, formatSeconds } from "../../lib/format";
import { normalizeOpenAiUsage } from "../../lib/protocols";
import type { Usage } from "../../types";

/** Raw text completion through /v1/completions: no chat template, no tools. */
export function PlaygroundPanel() {
  const settings = useStore((state) => state.settings);
  const model = useStore((state) => state.model);
  const [prompt, setPrompt] = useState("The three laws of thermodynamics, stated plainly:\n\n1.");
  const [output, setOutput] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [finish, setFinish] = useState<string | null>(null);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [elapsed, setElapsed] = useState<number | null>(null);
  const [maxTokens, setMaxTokens] = useState(256);
  const [echo, setEcho] = useState(true);
  const controller = useRef<AbortController | null>(null);

  const run = async () => {
    if (running) return;
    setRunning(true);
    setOutput("");
    setError(null);
    setFinish(null);
    setUsage(null);
    const abort = new AbortController();
    controller.current = abort;
    const started = performance.now();
    const body = {
      model,
      prompt,
      stream: true,
      max_tokens: maxTokens,
      temperature: settings.temperature,
      top_p: settings.topP,
      min_p: settings.minP,
      top_k: settings.topK,
      repetition_penalty: settings.repetitionPenalty,
      presence_penalty: settings.presencePenalty,
      frequency_penalty: settings.frequencyPenalty,
      penalty_window: settings.penaltyWindow,
      ...(settings.seed !== null ? { seed: settings.seed } : {}),
      ...(settings.stop.length ? { stop: settings.stop } : {}),
    };
    try {
      const response = await openStream("/v1/completions", body, abort.signal);
      for await (const frame of readSse(response.body!)) {
        if (frame.data.trim() === "[DONE]") break;
        let payload: Record<string, unknown>;
        try {
          payload = JSON.parse(frame.data);
        } catch {
          continue;
        }
        const choice = (payload.choices as Array<Record<string, unknown>> | undefined)?.[0];
        if (choice?.text) setOutput((current) => current + String(choice.text));
        if (typeof choice?.finish_reason === "string") setFinish(choice.finish_reason);
        if (payload.usage) setUsage(normalizeOpenAiUsage(payload.usage as Record<string, unknown>));
      }
    } catch (err) {
      if (!abort.signal.aborted) setError(err instanceof ApiError ? err.message : err instanceof Error ? err.message : String(err));
    } finally {
      setElapsed((performance.now() - started) / 1000);
      setRunning(false);
      controller.current = null;
    }
  };

  const completionTokens = usage?.completion_tokens;
  const rate = completionTokens && elapsed ? completionTokens / elapsed : null;

  return (
    <div className="playground">
      <p className="muted">Sends the prompt verbatim to /v1/completions with the sampling settings from the settings panel. No chat template is applied, so this is the place to probe the base model or a custom template.</p>
      <label className="field">
        <span className="field__label">Prompt</span>
        <textarea rows={8} className="mono" value={prompt} onChange={(event) => setPrompt(event.target.value)} spellCheck={false} />
      </label>
      <div className="inline inline--space">
        <label className="field field--inline">
          <span className="field__label">Max tokens</span>
          <input type="number" min={1} value={maxTokens} onChange={(event) => setMaxTokens(Math.max(1, Number(event.target.value) || 1))} />
        </label>
        <label className="toggle">
          <input type="checkbox" checked={echo} onChange={(event) => setEcho(event.target.checked)} />
          <span>Show prompt before output</span>
        </label>
        {running ? (
          <button className="button button--small" onClick={() => controller.current?.abort()}>
            <Square size={13} /> Stop
          </button>
        ) : (
          <button className="button button--small button--primary" onClick={() => void run()} disabled={!prompt}>
            <Play size={13} /> Run
          </button>
        )}
      </div>
      <pre className="raw playground__output" dir="auto">
        {echo && <span className="playground__prompt">{prompt}</span>}
        <span>{output}</span>
        {running && <span className="caret caret--inline" />}
      </pre>
      {error && <p className="error-text">{error}</p>}
      {(finish || usage) && (
        <p className="muted">
          {finish && <>finish: {finish}</>}
          {usage?.prompt_tokens !== undefined && <> · prompt {usage.prompt_tokens} tok</>}
          {completionTokens !== undefined && <> · output {completionTokens} tok</>}
          {elapsed !== null && <> · {formatSeconds(elapsed)}</>}
          {rate !== null && <> · {formatRate(rate)} incl. prefill</>}
        </p>
      )}
    </div>
  );
}
