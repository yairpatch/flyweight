import { useEffect, useRef, useState } from "react";
import { Download, Image as ImageIcon, Wand2 } from "lucide-react";
import { useStore } from "../../store";
import { ApiError, postJson } from "../../lib/api";
import { formatSeconds } from "../../lib/format";

interface ImagesInfo {
  model?: string;
  max_size?: string;
  default_steps?: number;
  busy?: boolean;
  generated?: number;
}

interface Rendered {
  id: string;
  url: string;
  prompt: string;
  seed: number;
  size: string;
  steps: number;
  seconds: number;
}

interface GenerationResponse {
  size: string;
  steps: number;
  data: Array<{ b64_json: string; seed: number; seconds: number }>;
}

/** Side lengths offered up to the server's limit: every 64 px from 256, the limit itself included. */
function sidesUpTo(limit: number): number[] {
  const sides: number[] = [];
  for (let side = 256; side <= limit; side += 64) sides.push(side);
  if (sides[sides.length - 1] !== limit && limit % 16 === 0) sides.push(limit);
  return sides;
}

function maxSide(info: ImagesInfo | null): number {
  const text = info?.max_size ?? "512x512";
  const side = parseInt(text.split("x")[0] ?? "512", 10);
  return Number.isFinite(side) && side > 0 ? side : 512;
}

/** Text-to-image through /v1/images/generations, when the server carries an image model. */
export function ImagesPanel() {
  const health = useStore((state) => state.health);
  const toast = useStore((state) => state.toast);
  const info = (health?.execution?.images as ImagesInfo | null | undefined) ?? null;
  const limit = maxSide(info);
  const [prompt, setPrompt] = useState("");
  const [width, setWidth] = useState(Math.min(1024, limit));
  const [height, setHeight] = useState(Math.min(1024, limit));
  const [steps, setSteps] = useState(info?.default_steps ?? 8);
  const [seed, setSeed] = useState("");
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [gallery, setGallery] = useState<Rendered[]>([]);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!running) {
      if (timer.current) window.clearInterval(timer.current);
      timer.current = null;
      return;
    }
    const started = performance.now();
    timer.current = window.setInterval(() => setElapsed((performance.now() - started) / 1000), 200);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [running]);

  if (!info) {
    return (
      <div className="images">
        <p className="muted">
          This server has no image model. Start it with <code>--image-model DIR</code> pointing at a Z-Image-Turbo
          snapshot (text_encoder/, transformer/, vae/) to render pictures here.
        </p>
      </div>
    );
  }

  const run = async () => {
    if (running || !prompt.trim()) return;
    setRunning(true);
    setError(null);
    setElapsed(0);
    const body: Record<string, unknown> = { prompt, size: `${width}x${height}`, steps };
    const seedValue = seed.trim() === "" ? null : Number(seed);
    if (seedValue !== null && Number.isInteger(seedValue) && seedValue >= 0) body.seed = seedValue;
    try {
      const response = await postJson<GenerationResponse>("/v1/images/generations", body, { timeoutMs: 600000 });
      const rendered = response.data.map((item, index) => ({
        id: `${Date.now()}-${index}`,
        url: `data:image/png;base64,${item.b64_json}`,
        prompt,
        seed: item.seed,
        size: response.size,
        steps: response.steps,
        seconds: item.seconds,
      }));
      setGallery((current) => [...rendered, ...current].slice(0, 24));
    } catch (failure) {
      const message = failure instanceof ApiError ? failure.message : String(failure);
      setError(message);
      toast(message, "error");
    } finally {
      setRunning(false);
    }
  };

  const sides = sidesUpTo(limit);

  return (
    <div className="images">
      <p className="muted">
        {info.model ? <>Rendering with <strong>{info.model}</strong>. </> : null}
        Up to {info.max_size}; the size is fixed when the server starts (<code>--image-max-size</code>).
      </p>
      <label className="field">
        <span className="field__label">Prompt</span>
        <textarea
          rows={4}
          value={prompt}
          onChange={(event) => setPrompt(event.target.value)}
          placeholder="a red bicycle leaning on a brick wall, afternoon light"
          onKeyDown={(event) => {
            if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
              event.preventDefault();
              void run();
            }
          }}
        />
      </label>
      <div className="inline inline--space images__controls">
        <label className="field field--inline">
          <span className="field__label">Width</span>
          <select className="select select--compact" value={width} onChange={(event) => setWidth(Number(event.target.value))}>
            {sides.map((side) => (
              <option key={side} value={side}>{side}</option>
            ))}
          </select>
        </label>
        <label className="field field--inline">
          <span className="field__label">Height</span>
          <select className="select select--compact" value={height} onChange={(event) => setHeight(Number(event.target.value))}>
            {sides.map((side) => (
              <option key={side} value={side}>{side}</option>
            ))}
          </select>
        </label>
        <label className="field field--inline">
          <span className="field__label">Steps</span>
          <input type="number" min={1} max={50} value={steps} onChange={(event) => setSteps(Math.max(1, Math.min(50, Number(event.target.value) || 1)))} />
        </label>
        <label className="field field--inline">
          <span className="field__label">Seed</span>
          <input type="text" inputMode="numeric" placeholder="random" value={seed} onChange={(event) => setSeed(event.target.value)} />
        </label>
        <button className="button button--small button--primary" onClick={() => void run()} disabled={running || !prompt.trim()}>
          <Wand2 size={14} /> {running ? `Rendering ${formatSeconds(elapsed)}` : "Generate"}
        </button>
      </div>
      {error && <p className="error-text">{error}</p>}
      {gallery.length === 0 ? (
        <p className="muted images__empty">
          <ImageIcon size={16} /> Pictures appear here. Ctrl+Enter renders.
        </p>
      ) : (
        <div className="images__gallery">
          {gallery.map((item) => (
            <figure key={item.id} className="images__item">
              <img src={item.url} alt={item.prompt} width={width} height={height} />
              <figcaption>
                <span title={item.prompt}>{item.prompt}</span>
                <span className="muted">
                  {item.size} · {item.steps} steps · seed {item.seed} · {formatSeconds(item.seconds)}
                </span>
                <a className="button button--small button--ghost" href={item.url} download={`flyweight-${item.seed}.png`}>
                  <Download size={13} /> PNG
                </a>
              </figcaption>
            </figure>
          ))}
        </div>
      )}
    </div>
  );
}
