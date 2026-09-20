import { useEffect, useMemo, useRef, useState } from "react";
import { Copy, Dices, Download, Image as ImageIcon, ImagePlus, Lock, RefreshCw, Square, Trash2, Wand2, X } from "lucide-react";
import { imageDimensions, useStore, type ImageSettings } from "../store";
import { formatSeconds } from "../lib/format";

interface ImagesInfo {
  model?: string;
  max_size?: string;
  weights?: string;
  precision?: string;
  /** What a side must divide by, and the step count the model was tuned for. */
  size_multiple?: number;
  default_steps?: number;
  alpha?: boolean;
  /** Whether the model takes reference images (Qwen-Image-2.1). */
  edit?: boolean;
}

const ASPECTS: Array<{ id: ImageSettings["aspect"]; label: string }> = [
  { id: "1:1", label: "1:1" },
  { id: "3:2", label: "3:2" },
  { id: "2:3", label: "2:3" },
  { id: "16:9", label: "16:9" },
  { id: "9:16", label: "9:16" },
];

function sizesUpTo(limit: number): number[] {
  const sides: number[] = [];
  for (let side = 512; side <= limit; side += 128) sides.push(side);
  if (sides[sides.length - 1] !== limit) sides.push(limit);
  return sides;
}

/** Text-to-image workspace: the canvas, the render progress, the prompt bar, and the settings column. */
export function Studio() {
  const health = useStore((state) => state.health);
  const info = (health?.execution?.images as ImagesInfo | null | undefined) ?? null;
  const limit = parseInt(String(info?.max_size ?? "1024x1024").split("x")[0] ?? "1024", 10) || 1024;
  // Both follow from which image model the server loaded, so they come off the
  // health payload rather than being pinned to one model's.
  const multiple = Number(info?.size_multiple) || 16;
  const defaultSteps = Number(info?.default_steps) || 8;
  const images = useStore((state) => state.images);
  const currentImageId = useStore((state) => state.currentImageId);
  const progress = useStore((state) => state.imageProgress);
  const error = useStore((state) => state.imageError);
  const prompt = useStore((state) => state.imagePrompt);
  const setPrompt = useStore((state) => state.setImagePrompt);
  const settings = useStore((state) => state.imageSettings);
  const update = useStore((state) => state.updateImageSettings);
  const generateImage = useStore((state) => state.generateImage);
  const cancelImage = useStore((state) => state.cancelImage);
  const deleteImage = useStore((state) => state.deleteImage);
  const toast = useStore((state) => state.toast);
  const refs = useStore((state) => state.imageRefs);
  const setRefs = useStore((state) => state.setImageRefs);
  const canEdit = Boolean(info?.edit);
  const fileInput = useRef<HTMLInputElement>(null);
  const attach = (files: FileList | null) => {
    if (!files) return;
    const readers = Array.from(files).slice(0, 10 - refs.length).map(
      (file) => new Promise<string>((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result));
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      }),
    );
    void Promise.all(readers).then((urls) => setRefs([...refs, ...urls])).catch(() => toast("Could not read that image", "error"));
  };
  const current = images.find((image) => image.id === currentImageId) ?? null;
  const url = useMemo(() => (current ? URL.createObjectURL(current.blob) : null), [current]);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  const [elapsed, setElapsed] = useState(0);
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!progress) return;
    const timer = window.setInterval(() => setElapsed((Date.now() - progress.startedAt) / 1000), 250);
    return () => window.clearInterval(timer);
  }, [progress]);

  const { width, height } = imageDimensions(settings, limit, multiple);
  const steps = settings.steps || defaultSteps;
  const running = progress !== null;
  const canRender = !running && prompt.trim().length > 0 && info !== null;

  const copy = async () => {
    if (!current) return;
    try {
      await navigator.clipboard.write([new ClipboardItem({ "image/png": current.blob })]);
      toast("Copied to the clipboard", "success");
    } catch {
      toast("The browser refused the clipboard", "error");
    }
  };

  const reuse = () => {
    if (!current) return;
    setPrompt(current.prompt);
    textarea.current?.focus();
  };

  if (!info) {
    return (
      <section className="studio">
        <div className="studio__stage">
          <div className="empty">
            <div className="empty__badge"><ImageIcon size={22} /></div>
            <h2>No image model loaded</h2>
            <p>Start the server with <code>--image-model DIR</code> pointing at a Z-Image-Turbo or Qwen-Image-2.1 snapshot to render pictures here.</p>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="studio">
      <div className="studio__stage">
        <div className="studio__canvas">
          {current && url ? (
            <img src={url} alt={current.prompt} className={running ? "studio__image studio__image--dim" : "studio__image"} />
          ) : (
            !running && (
              <div className="empty">
                <div className="empty__badge"><ImageIcon size={22} /></div>
                <h2>What should we draw?</h2>
                <p>Describe the picture below. Ctrl+Enter renders; the history on the left keeps everything.</p>
              </div>
            )
          )}
          {running && (
            <div className="studio__progress" role="status">
              <div className="studio__bar">
                <div className="studio__bar-fill" style={{ width: `${Math.max(4, (100 * progress.step) / Math.max(1, progress.steps))}%` }} />
              </div>
              <span>
                {progress.step === 0 ? "Encoding the prompt" : `Step ${progress.step} of ${progress.steps}`} · {formatSeconds(elapsed)}
              </span>
            </div>
          )}
        </div>
        {current && !running && (
          <div className="studio__caption">
            <span className="studio__caption-text" title={current.prompt}>{current.prompt}</span>
            <span className="muted">{current.width}x{current.height} · {current.steps} steps · seed {current.seed} · {formatSeconds(current.seconds)}</span>
          </div>
        )}
        {error && <p className="error-text studio__error">{error}</p>}
        {refs.length > 0 && (
          <div className="studio__refs" aria-label="Reference images">
            {refs.map((ref, index) => (
              <div key={index} className="studio__ref">
                <img src={ref} alt={`Reference ${index + 1}`} />
                <button className="icon-button icon-button--small studio__ref-remove" onClick={() => setRefs(refs.filter((_, i) => i !== index))} aria-label="Remove reference image" disabled={running}>
                  <X size={12} />
                </button>
              </div>
            ))}
            <span className="muted">Editing: the picture takes the last reference's shape unless a size is set.</span>
          </div>
        )}
        <div className="studio__prompt">
          {canEdit && (
            <>
              <input ref={fileInput} type="file" accept="image/*" multiple hidden onChange={(event) => { attach(event.target.files); event.target.value = ""; }} />
              <button className="icon-button" onClick={() => fileInput.current?.click()} title="Add reference images to edit" aria-label="Add reference images" disabled={running || refs.length >= 10}>
                <ImagePlus size={16} />
              </button>
            </>
          )}
          <textarea
            ref={textarea}
            rows={2}
            value={prompt}
            placeholder={refs.length ? "make the bicycle blue" : "a red bicycle leaning on a brick wall, afternoon light"}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                if (canRender) void generateImage();
              }
            }}
            aria-label="Prompt"
          />
          {running ? (
            <button className="button button--small" onClick={cancelImage} title="Stop rendering">
              <Square size={14} /> Stop
            </button>
          ) : (
            <button className="button button--primary" onClick={() => void generateImage()} disabled={!canRender} title="Render (Ctrl+Enter)">
              <Wand2 size={15} /> Generate
            </button>
          )}
        </div>
      </div>

      <aside className="studio__settings" aria-label="Image settings">
        <section className="settings__section">
          <h3>Size</h3>
          <div className="studio__aspects" role="radiogroup" aria-label="Aspect ratio">
            {ASPECTS.map((aspect) => (
              <button
                key={aspect.id}
                role="radio"
                aria-checked={settings.aspect === aspect.id}
                className={`studio__aspect${settings.aspect === aspect.id ? " studio__aspect--active" : ""}`}
                onClick={() => update({ aspect: aspect.id })}
              >
                <span className={`studio__aspect-box studio__aspect-box--${aspect.id.replace(":", "-")}`} aria-hidden="true" />
                {aspect.label}
              </button>
            ))}
          </div>
          <label className="field field--inline">
            <span className="field__label">Long side</span>
            <select className="select select--compact" value={Math.min(settings.size, limit)} onChange={(event) => update({ size: Number(event.target.value) })}>
              {sizesUpTo(limit).map((side) => (
                <option key={side} value={side}>{side}</option>
              ))}
            </select>
          </label>
          <p className="muted">{width} x {height}, up to {info.max_size} on this server.</p>
        </section>

        <section className="settings__section">
          <h3>Sampling</h3>
          <label className="field field--inline">
            <span className="field__label">Steps</span>
            <input type="number" min={1} max={50} value={steps} onChange={(event) => update({ steps: Math.max(1, Math.min(50, Number(event.target.value) || 1)) })} />
          </label>
          <label className="field field--inline">
            <span className="field__label">Seed</span>
            <input
              type="text"
              inputMode="numeric"
              placeholder="random"
              value={settings.seed === null ? "" : String(settings.seed)}
              onChange={(event) => {
                const text = event.target.value.trim();
                update({ seed: text === "" ? null : Math.max(0, Number(text) || 0) });
              }}
            />
            <button
              className="icon-button icon-button--small"
              title={settings.seed === null ? "Lock the seed" : "Random seed each time"}
              aria-label={settings.seed === null ? "Lock the seed" : "Random seed each time"}
              onClick={() => update({ seed: settings.seed === null ? (current?.seed ?? Math.floor(Math.random() * 1e9)) : null })}
            >
              {settings.seed === null ? <Dices size={13} /> : <Lock size={13} />}
            </button>
          </label>
          <p className="muted">
            {info.precision === "exact" ? "Exact precision" : "Fast precision"} · weights on {info.weights === "host" ? "host" : "GPU"} · {info.model}
          </p>
        </section>

        <section className="settings__section">
          <h3>This picture</h3>
          <div className="studio__actions">
            <button className="button button--small" disabled={!current || running} onClick={() => current && void generateImage({ seed: current.seed })} title="Render the same prompt and seed again">
              <RefreshCw size={13} /> Again
            </button>
            <button className="button button--small" disabled={!current || running} onClick={() => { reuse(); void generateImage({ seed: null }); }} title="Same prompt, fresh seed">
              <Dices size={13} /> Variation
            </button>
            <a className={`button button--small${current ? "" : " button--disabled"}`} href={url ?? undefined} download={current ? `flyweight-${current.seed}.png` : undefined} aria-disabled={!current}>
              <Download size={13} /> PNG
            </a>
            <button className="button button--small" disabled={!current} onClick={() => void copy()}>
              <Copy size={13} /> Copy
            </button>
            <button className="button button--small button--ghost" disabled={!current} onClick={reuse} title="Put this picture's prompt back in the box">
              Reuse prompt
            </button>
            <button className="button button--small button--ghost" disabled={!current || running} onClick={() => current && void deleteImage(current.id)}>
              <Trash2 size={13} /> Delete
            </button>
          </div>
        </section>
      </aside>
    </section>
  );
}
