import { useState } from "react";
import { Copy, Trash2 } from "lucide-react";
import { useStore } from "../../store";
import { api, getApiKey } from "../../lib/api";
import { formatSeconds } from "../../lib/format";
import type { RequestRecord } from "../../types";

function toCurl(record: RequestRecord): string {
  const key = getApiKey();
  const origin = window.location.origin;
  const body = JSON.stringify(record.body).replace(/'/g, "'\\''");
  return [
    `curl -N ${origin}${record.url}`,
    `  -H 'Content-Type: application/json'`,
    ...(key ? [`  -H 'Authorization: Bearer ${"*".repeat(8)}'`] : []),
    `  -d '${body}'`,
  ].join(" \\\n");
}

/** Every request the chat sent, with the exact body and the raw SSE frames. */
export function InspectorPanel() {
  const requests = useStore((state) => state.requests);
  const clear = useStore((state) => state.clearRequests);
  const toast = useStore((state) => state.toast);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tab, setTab] = useState<"request" | "stream" | "curl">("request");
  const [responseLookup, setResponseLookup] = useState("");
  const [lookupResult, setLookupResult] = useState<string | null>(null);

  const selected = requests.find((record) => record.id === selectedId) ?? requests[0];

  const copy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      toast("Copied", "success");
    } catch {
      toast("Clipboard unavailable", "error");
    }
  };

  const lookup = async (remove: boolean) => {
    const id = responseLookup.trim();
    if (!id) return;
    try {
      const result = remove ? await api.deleteResponse(id) : await api.getResponse(id);
      setLookupResult(JSON.stringify(result, null, 2));
    } catch (err) {
      setLookupResult(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  return (
    <div className="inspector">
      <div className="inline inline--space">
        <p className="muted">Last {requests.length} request{requests.length === 1 ? "" : "s"} from this tab.</p>
        <button className="button button--small" onClick={clear} disabled={!requests.length}>
          <Trash2 size={13} /> Clear
        </button>
      </div>
      <ul className="inspector__list">
        {requests.map((record) => (
          <li key={record.id}>
            <button className={`inspector__row${selected?.id === record.id ? " inspector__row--active" : ""}`} onClick={() => setSelectedId(record.id)}>
              <span className={`badge badge--${record.error ? "error" : record.status ? "ok" : "pending"}`}>{record.status ?? "…"}</span>
              <code>{record.url}</code>
              <span className="muted">{record.durationMs !== undefined ? formatSeconds(record.durationMs / 1000) : "live"}</span>
            </button>
          </li>
        ))}
      </ul>
      {selected && (
        <div className="inspector__detail">
          <div className="tabs" role="tablist">
            {(["request", "stream", "curl"] as const).map((name) => (
              <button key={name} role="tab" aria-selected={tab === name} className={`tab${tab === name ? " tab--active" : ""}`} onClick={() => setTab(name)}>
                {name === "request" ? "Request body" : name === "stream" ? `Stream (${selected.rawEvents.length})` : "curl"}
              </button>
            ))}
            <button className="icon-button icon-button--small" onClick={() => void copy(tab === "request" ? JSON.stringify(selected.body, null, 2) : tab === "stream" ? selected.rawEvents.map((entry) => entry.line).join("\n\n") : toCurl(selected))} aria-label="Copy">
              <Copy size={13} />
            </button>
          </div>
          {selected.error && <p className="error-text">{selected.error}</p>}
          {tab === "request" && (
            <pre className="raw">
              <code>{JSON.stringify(selected.body, null, 2)}</code>
            </pre>
          )}
          {tab === "stream" && (
            <div className="frames">
              {selected.rawEvents.map((entry, index) => (
                <div key={index} className="frame">
                  <span className="frame__at">{(entry.at / 1000).toFixed(3)}s</span>
                  <pre>
                    <code>{entry.line}</code>
                  </pre>
                </div>
              ))}
              {selected.rawEvents.length === 0 && <p className="muted">No frames received.</p>}
            </div>
          )}
          {tab === "curl" && (
            <pre className="raw">
              <code>{toCurl(selected)}</code>
            </pre>
          )}
        </div>
      )}
      <section className="settings__section">
        <h3>Stored responses</h3>
        <p className="muted">The Responses API keeps completed responses server-side (128 most recent). Retrieve or delete one by id.</p>
        <div className="inline">
          <input value={responseLookup} onChange={(event) => setResponseLookup(event.target.value)} placeholder="resp_…" spellCheck={false} />
          <button className="button button--small" onClick={() => void lookup(false)}>
            Get
          </button>
          <button className="button button--small" onClick={() => void lookup(true)}>
            Delete
          </button>
        </div>
        {lookupResult && (
          <pre className="raw">
            <code>{lookupResult}</code>
          </pre>
        )}
      </section>
    </div>
  );
}
