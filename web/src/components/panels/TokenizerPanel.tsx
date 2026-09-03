import { useEffect, useMemo, useState } from "react";
import { api } from "../../lib/api";
import { useActiveConversation, useStore } from "../../store";
import { buildAnthropicRequest, buildResponsesRequest } from "../../lib/protocols";
import { formatInteger } from "../../lib/format";

const TOKEN_HUES = 6;

export function TokenizerPanel() {
  const [text, setText] = useState("Hello, world! שלום עולם 🚀");
  const [tokens, setTokens] = useState<number[]>([]);
  const [pieces, setPieces] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [idsInput, setIdsInput] = useState("");
  const [decoded, setDecoded] = useState<string | null>(null);
  const [counts, setCounts] = useState<{ anthropic?: number; responses?: number; error?: string } | null>(null);
  const conversation = useActiveConversation();
  const settings = useStore((state) => state.settings);
  const tools = useStore((state) => state.tools);
  const model = useStore((state) => state.model);

  useEffect(() => {
    const timer = window.setTimeout(async () => {
      if (!text) {
        setTokens([]);
        setPieces([]);
        return;
      }
      try {
        const result = await api.tokenize(text);
        setTokens(result.tokens);
        setError(null);
        // Per-token text: detokenize each id individually, in one batch.
        const parts = await Promise.all(result.tokens.slice(0, 400).map((id) => api.detokenize([id]).then((r) => r.content).catch(() => "")));
        setPieces(parts);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    }, 350);
    return () => window.clearTimeout(timer);
  }, [text]);

  const chars = text.length;
  const ratio = useMemo(() => (tokens.length ? (chars / tokens.length).toFixed(2) : "–"), [chars, tokens.length]);

  const decode = async () => {
    const ids = idsInput
      .split(/[\s,]+/)
      .map((piece) => Number(piece))
      .filter((id) => Number.isInteger(id) && id >= 0);
    if (!ids.length) return setDecoded("");
    try {
      setDecoded((await api.detokenize(ids)).content);
    } catch (err) {
      setDecoded(`Error: ${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const countConversation = async () => {
    if (!conversation) return;
    const ctx = { model, messages: conversation.messages, settings, tools };
    const next: { anthropic?: number; responses?: number; error?: string } = {};
    try {
      const anthropicBody = buildAnthropicRequest(ctx);
      delete anthropicBody.stream;
      next.anthropic = (await api.countAnthropic(anthropicBody)).input_tokens;
    } catch (err) {
      next.error = err instanceof Error ? err.message : String(err);
    }
    try {
      const responsesBody = buildResponsesRequest(ctx);
      delete responsesBody.stream;
      next.responses = (await api.countResponses(responsesBody)).input_tokens;
    } catch (err) {
      next.error = err instanceof Error ? err.message : String(err);
    }
    setCounts(next);
  };

  return (
    <div className="tokenizer">
      <section className="settings__section">
        <h3>Tokenize</h3>
        <textarea rows={4} value={text} onChange={(event) => setText(event.target.value)} aria-label="Text to tokenize" />
        <div className="stat-grid stat-grid--3">
          <div className="stat">
            <div className="stat__label">Tokens</div>
            <div className="stat__value">{formatInteger(tokens.length)}</div>
          </div>
          <div className="stat">
            <div className="stat__label">Characters</div>
            <div className="stat__value">{formatInteger(chars)}</div>
          </div>
          <div className="stat">
            <div className="stat__label">Chars / token</div>
            <div className="stat__value">{ratio}</div>
          </div>
        </div>
        {error && <p className="error-text">{error}</p>}
        <div className="token-strip" dir="auto">
          {tokens.slice(0, 400).map((id, index) => (
            <span key={index} className={`token token--${index % TOKEN_HUES}`} title={`id ${id}`}>
              {pieces[index] === undefined ? "…" : pieces[index] === "" ? "·" : pieces[index].replace(/\n/g, "⏎").replace(/ /g, "␣")}
            </span>
          ))}
          {tokens.length > 400 && <span className="muted">… {tokens.length - 400} more</span>}
        </div>
        <details>
          <summary>Token ids</summary>
          <pre className="raw">
            <code>{tokens.join(", ")}</code>
          </pre>
        </details>
      </section>

      <section className="settings__section">
        <h3>Detokenize</h3>
        <textarea rows={2} value={idsInput} onChange={(event) => setIdsInput(event.target.value)} placeholder="9707, 11, 1879" aria-label="Token ids" />
        <button className="button button--small" onClick={() => void decode()}>
          Decode
        </button>
        {decoded !== null && (
          <pre className="raw">
            <code>{decoded || "(empty)"}</code>
          </pre>
        )}
      </section>

      <section className="settings__section">
        <h3>Count the current conversation</h3>
        <p className="muted">Runs the server's own prompt assembly (system prompt, tools, template) and reports the input tokens for the Anthropic and Responses translators.</p>
        <button className="button button--small" onClick={() => void countConversation()} disabled={!conversation?.messages.length}>
          Count input tokens
        </button>
        {counts && (
          <div className="stat-grid stat-grid--2">
            <div className="stat">
              <div className="stat__label">/v1/messages/count_tokens</div>
              <div className="stat__value">{formatInteger(counts.anthropic)}</div>
            </div>
            <div className="stat">
              <div className="stat__label">/v1/responses/input_tokens</div>
              <div className="stat__value">{formatInteger(counts.responses)}</div>
            </div>
            {counts.error && <p className="error-text">{counts.error}</p>}
          </div>
        )}
      </section>
    </div>
  );
}
