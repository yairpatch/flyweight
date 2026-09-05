import { useRef } from "react";
import { Plus, Trash2, Upload } from "lucide-react";
import { useStore } from "../../store";
import { clamp, identifier } from "../../lib/format";
import type { ToolDefinition } from "../../types";

function validJson(text: string): boolean {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object";
  } catch {
    return false;
  }
}

export function ToolsPanel() {
  const tools = useStore((state) => state.tools);
  const setTools = useStore((state) => state.setTools);
  const settings = useStore((state) => state.settings);
  const updateSettings = useStore((state) => state.updateSettings);
  const toast = useStore((state) => state.toast);
  const fileInput = useRef<HTMLInputElement>(null);

  const patch = (id: string, changes: Partial<ToolDefinition>) => setTools(tools.map((tool) => (tool.id === id ? { ...tool, ...changes } : tool)));
  const add = () =>
    setTools([
      ...tools,
      { id: identifier("tool"), name: `tool_${tools.length + 1}`, description: "", parameters: '{\n  "type": "object",\n  "properties": {}\n}', enabled: true },
    ]);

  const importTools = async (files: FileList | null) => {
    const file = files?.[0];
    if (!file) return;
    try {
      const data = JSON.parse(await file.text());
      const list: unknown[] = Array.isArray(data) ? data : Array.isArray(data?.tools) ? data.tools : [data];
      const imported: ToolDefinition[] = [];
      for (const item of list as Array<Record<string, unknown>>) {
        const fn = (item.function ?? item) as Record<string, unknown>;
        const name = String(fn.name ?? "").trim();
        if (!name) continue;
        const parameters = fn.parameters ?? fn.input_schema ?? { type: "object", properties: {} };
        imported.push({ id: identifier("tool"), name, description: String(fn.description ?? ""), parameters: JSON.stringify(parameters, null, 2), enabled: true });
      }
      if (!imported.length) throw new Error("no tools");
      setTools([...tools, ...imported]);
      toast(`Imported ${imported.length} tool${imported.length === 1 ? "" : "s"}`, "success");
    } catch {
      toast("Expected OpenAI or Anthropic tool definitions", "error");
    }
    if (fileInput.current) fileInput.current.value = "";
  };

  const enabled = tools.filter((tool) => tool.enabled);

  return (
    <div className="tools-panel">
      <p className="muted">
        Tools are sent with every request while enabled. The runtime's sampler enforces the call grammar: a declared name, required parameters present,
        and arguments that parse as JSON. In a chat, paste the result into the card and continue; in an agent run, a tool's handler answers the call
        automatically.
      </p>
      <div className="settings__section">
        <h3>Agent runs</h3>
        <p className="muted">
          In the sidebar's <strong>Agent</strong> tab, tool calls run through their handlers automatically and the results go back to the model until
          it answers. Handlers run in the same sandboxed frame as the code preview: an opaque origin with no access to this app's storage or API key,
          and network requests subject to CORS. A call whose tool has no handler pauses the run for a manual result; Stop (Esc) interrupts it.
        </p>
        <label className="field">
          <span className="field__label">
            Turn cap <small>model turns per run</small>
          </span>
          <input
            type="number"
            min={1}
            max={100}
            value={settings.agentMaxTurns}
            onChange={(event) => updateSettings({ agentMaxTurns: clamp(Math.trunc(Number(event.target.value) || 1), 1, 100) })}
          />
        </label>
      </div>
      <div className="settings__section">
        <h3>Calling policy</h3>
        <label className="field">
          <span className="field__label">Tool choice</span>
          <select value={settings.toolChoice} onChange={(event) => updateSettings({ toolChoice: event.target.value })} disabled={!enabled.length}>
            <option value="auto">auto · model decides</option>
            <option value="required">required · must call a tool</option>
            <option value="none">none · never call</option>
            {enabled.map((tool) => (
              <option key={tool.id} value={tool.name}>
                force · {tool.name}
              </option>
            ))}
          </select>
        </label>
        <label className="toggle">
          <input type="checkbox" checked={settings.parallelToolCalls} onChange={(event) => updateSettings({ parallelToolCalls: event.target.checked })} />
          <span>Allow parallel tool calls</span>
        </label>
      </div>
      <div className="settings__section">
        <div className="inline inline--space">
          <h3>Definitions</h3>
          <div className="inline">
            <button className="button button--small" onClick={() => fileInput.current?.click()}>
              <Upload size={13} /> Import
            </button>
            <button className="button button--small button--primary" onClick={add}>
              <Plus size={13} /> Add
            </button>
          </div>
          <input ref={fileInput} type="file" accept="application/json,.json" hidden onChange={(event) => void importTools(event.target.files)} />
        </div>
        {tools.length === 0 && <p className="muted">No tools defined.</p>}
        {tools.map((tool) => (
          <details key={tool.id} className="tool-def" open={tool.enabled}>
            <summary>
              <input type="checkbox" checked={tool.enabled} onChange={(event) => patch(tool.id, { enabled: event.target.checked })} onClick={(event) => event.stopPropagation()} aria-label={`Enable ${tool.name}`} />
              <code>{tool.name || "unnamed"}</code>
              <span className="tool-def__desc">{tool.description}</span>
              <button className="icon-button icon-button--small icon-button--danger" onClick={(event) => { event.preventDefault(); setTools(tools.filter((item) => item.id !== tool.id)); }} aria-label="Delete tool">
                <Trash2 size={13} />
              </button>
            </summary>
            <label className="field">
              <span className="field__label">Name</span>
              <input value={tool.name} onChange={(event) => patch(tool.id, { name: event.target.value.replace(/[^A-Za-z0-9_.-]/g, "_") })} spellCheck={false} />
            </label>
            <label className="field">
              <span className="field__label">Description</span>
              <textarea rows={2} value={tool.description} onChange={(event) => patch(tool.id, { description: event.target.value })} />
            </label>
            <label className="field">
              <span className="field__label">
                Parameters <small>JSON schema</small>
              </span>
              <textarea rows={8} className={validJson(tool.parameters) ? "mono" : "mono invalid"} spellCheck={false} value={tool.parameters} onChange={(event) => patch(tool.id, { parameters: event.target.value })} />
            </label>
            <label className="field">
              <span className="field__label">
                Handler <small>async JavaScript · used by agent runs</small>
              </span>
              <textarea
                rows={6}
                className="mono"
                spellCheck={false}
                placeholder={"// Runs sandboxed with the call's parsed arguments as `args`.\n// Return a string or any JSON-able value; `await fetch(...)` works.\nreturn `sunny in ${args.city}`;"}
                value={tool.executor ?? ""}
                onChange={(event) => patch(tool.id, { executor: event.target.value })}
              />
            </label>
          </details>
        ))}
      </div>
    </div>
  );
}
