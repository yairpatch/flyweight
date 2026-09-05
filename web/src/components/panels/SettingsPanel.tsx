import { useState } from "react";
import { Save, Trash2 } from "lucide-react";
import { useStore } from "../../store";
import { getApiKey, setApiKey } from "../../lib/api";
import { PROTOCOL_LABELS } from "../../lib/protocols";
import { maxTokensCeiling, REASONING_EFFORTS } from "../../lib/settings";
import type { GenerationSettings, Protocol } from "../../types";

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span className="field__label">
        {label}
        {hint && <small>{hint}</small>}
      </span>
      {children}
    </label>
  );
}

function Range({ label, value, min, max, step, onChange, hint }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void; hint?: string }) {
  return (
    <div className="field field--range">
      <span className="field__label">
        {label}
        {hint && <small>{hint}</small>}
      </span>
      <div className="range">
        <input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} aria-label={label} />
        <input type="number" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} aria-label={`${label} value`} />
      </div>
    </div>
  );
}

function validJson(text: string): boolean {
  if (!text.trim()) return true;
  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
}

export function SettingsPanel() {
  const settings = useStore((state) => state.settings);
  const update = useStore((state) => state.updateSettings);
  const reset = useStore((state) => state.resetSettings);
  const props = useStore((state) => state.props);
  const presets = useStore((state) => state.presets);
  const savePreset = useStore((state) => state.savePreset);
  const applyPreset = useStore((state) => state.applyPreset);
  const deletePreset = useStore((state) => state.deletePreset);
  const pollRuntime = useStore((state) => state.pollRuntime);
  const toast = useStore((state) => state.toast);
  const [apiKey, setKey] = useState(getApiKey());
  const [presetName, setPresetName] = useState("");
  const [stopDraft, setStopDraft] = useState("");

  const set = <K extends keyof GenerationSettings>(key: K) => (value: GenerationSettings[K]) => update({ [key]: value } as Partial<GenerationSettings>);
  const protocol = settings.protocol;
  // The window is the only ceiling; max_output_tokens is the server's default
  // for a request that names none, and it honors a larger one.
  const maxCap = maxTokensCeiling(props);
  const notOn = (list: Protocol[]) => (list.includes(protocol) ? `not sent on ${PROTOCOL_LABELS[protocol]}` : undefined);

  return (
    <div className="settings">
      <section className="settings__section">
        <h3>Connection</h3>
        <Field label="API key" hint="kept for this tab only">
          <div className="inline">
            <input type="password" value={apiKey} onChange={(event) => setKey(event.target.value)} placeholder="Only if the server was started with --api-key" autoComplete="off" />
            <button
              className="button button--small"
              onClick={() => {
                setApiKey(apiKey);
                toast("API key saved for this session", "success");
                void pollRuntime();
              }}
            >
              Save
            </button>
          </div>
        </Field>
        <Field label="Protocol" hint="which server API the chat goes through">
          <select value={protocol} onChange={(event) => set("protocol")(event.target.value as Protocol)}>
            {(Object.keys(PROTOCOL_LABELS) as Protocol[]).map((key) => (
              <option key={key} value={key}>
                {PROTOCOL_LABELS[key]} · {key === "chat" ? "/v1/chat/completions" : key === "anthropic" ? "/v1/messages" : "/v1/responses"}
              </option>
            ))}
          </select>
        </Field>
      </section>

      <section className="settings__section">
        <h3>Prompt</h3>
        <Field label="System prompt">
          <textarea rows={4} value={settings.systemPrompt} onChange={(event) => set("systemPrompt")(event.target.value)} placeholder="Optional instructions sent ahead of the conversation" />
        </Field>
        <Field label="Chat template kwargs" hint={notOn(["anthropic", "responses"]) ?? "JSON passed to the chat template"}>
          <textarea rows={2} value={settings.chatTemplateKwargs} onChange={(event) => set("chatTemplateKwargs")(event.target.value)} placeholder='{"enable_thinking": true}' className={validJson(settings.chatTemplateKwargs) ? "" : "invalid"} spellCheck={false} />
        </Field>
      </section>

      <section className="settings__section">
        <h3>Sampling</h3>
        <Range label="Max tokens" value={settings.maxTokens} min={1} max={maxCap} step={1} onChange={set("maxTokens")} hint={props ? `up to the ${maxCap} context window` : undefined} />
        <Range label="Temperature" value={settings.temperature} min={0} max={2} step={0.01} onChange={set("temperature")} />
        <Range label="Top-p" value={settings.topP} min={0.01} max={1} step={0.01} onChange={set("topP")} />
        <Range label="Min-p" value={settings.minP} min={0} max={1} step={0.01} onChange={set("minP")} hint="relative to the best token; 0 disables" />
        <Range label="Top-k" value={settings.topK} min={0} max={200} step={1} onChange={set("topK")} hint="0 disables" />
        <Range label="Repetition penalty" value={settings.repetitionPenalty} min={1} max={2} step={0.01} onChange={set("repetitionPenalty")} />
        <Range label="Presence penalty" value={settings.presencePenalty} min={-2} max={2} step={0.01} onChange={set("presencePenalty")} />
        <Range label="Frequency penalty" value={settings.frequencyPenalty} min={-2} max={2} step={0.01} onChange={set("frequencyPenalty")} />
        <Range label="Penalty window" value={settings.penaltyWindow} min={0} max={4096} step={1} onChange={set("penaltyWindow")} hint="tokens; 0 disables penalties" />
        <Field label="Seed" hint="blank for random">
          <input type="number" value={settings.seed ?? ""} onChange={(event) => set("seed")(event.target.value === "" ? null : Number(event.target.value))} placeholder="random" />
        </Field>
        <Field label="Stop sequences" hint="Enter to add">
          <div className="tags">
            {settings.stop.map((stop) => (
              <span key={stop} className="tag">
                <code>{JSON.stringify(stop)}</code>
                <button onClick={() => set("stop")(settings.stop.filter((item) => item !== stop))} aria-label={`Remove ${stop}`}>
                  ×
                </button>
              </span>
            ))}
            <input
              value={stopDraft}
              onChange={(event) => setStopDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && stopDraft) {
                  event.preventDefault();
                  if (!settings.stop.includes(stopDraft)) set("stop")([...settings.stop, stopDraft]);
                  setStopDraft("");
                }
              }}
              placeholder="add…"
            />
          </div>
        </Field>
      </section>

      <section className="settings__section">
        <h3>Reasoning</h3>
        <label className="toggle">
          <input type="checkbox" checked={settings.thinking} onChange={(event) => set("thinking")(event.target.checked)} />
          <span>Enable thinking</span>
        </label>
        <Field label="Reasoning effort" hint={notOn(["anthropic"]) ?? "auto keeps the checkpoint default"}>
          <select value={settings.reasoningEffort} disabled={!settings.thinking} onChange={(event) => set("reasoningEffort")(event.target.value as GenerationSettings["reasoningEffort"])}>
            {REASONING_EFFORTS.map((effort) => (
              <option key={effort} value={effort}>
                {effort}
              </option>
            ))}
          </select>
        </Field>
        <Field label="Reasoning budget" hint={notOn(["responses"]) ?? "tokens; blank = half of max tokens"}>
          <input type="number" min={0} disabled={!settings.thinking} value={settings.reasoningBudget ?? ""} onChange={(event) => set("reasoningBudget")(event.target.value === "" ? null : Number(event.target.value))} placeholder="auto" />
        </Field>
        <label className="toggle">
          <input type="checkbox" checked={settings.preserveThinking} onChange={(event) => set("preserveThinking")(event.target.checked)} />
          <span>Preserve thinking across turns</span>
          <small>{notOn(["responses"]) ?? "keeps earlier reasoning in the prompt"}</small>
        </label>
      </section>

      <section className="settings__section">
        <h3>Output format</h3>
        <Field label="Response format" hint={notOn(["anthropic"]) ?? "enforced by the sampler grammar"}>
          <select value={settings.responseFormat} onChange={(event) => set("responseFormat")(event.target.value as GenerationSettings["responseFormat"])}>
            <option value="text">Text</option>
            <option value="json_object">JSON object</option>
            <option value="json_schema">JSON schema</option>
          </select>
        </Field>
        {settings.responseFormat === "json_schema" && (
          <Field label="JSON schema">
            <textarea rows={8} spellCheck={false} className={validJson(settings.jsonSchema) ? "mono" : "mono invalid"} value={settings.jsonSchema} onChange={(event) => set("jsonSchema")(event.target.value)} />
          </Field>
        )}
      </section>

      <section className="settings__section">
        <h3>Presets</h3>
        <div className="inline">
          <input value={presetName} onChange={(event) => setPresetName(event.target.value)} placeholder="Preset name" />
          <button
            className="button button--small"
            disabled={!presetName.trim()}
            onClick={() => {
              void savePreset(presetName);
              setPresetName("");
            }}
          >
            <Save size={13} /> Save
          </button>
        </div>
        <ul className="presets">
          {presets.map((preset) => (
            <li key={preset.id}>
              <button className="button button--ghost" onClick={() => applyPreset(preset.id)}>
                {preset.name}
              </button>
              <button className="icon-button icon-button--small icon-button--danger" onClick={() => void deletePreset(preset.id)} aria-label={`Delete preset ${preset.name}`}>
                <Trash2 size={13} />
              </button>
            </li>
          ))}
          {presets.length === 0 && <li className="muted">No presets saved yet. A preset stores settings and tools together.</li>}
        </ul>
      </section>

      <section className="settings__section settings__section--footer">
        <button className="button" onClick={reset}>
          Reset to model defaults
        </button>
        {props?.generation_defaults_source && <small className="muted">defaults from {props.generation_defaults_source}</small>}
      </section>
    </div>
  );
}
