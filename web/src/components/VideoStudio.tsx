import { useEffect, useMemo, useRef, useState } from "react";
import { Clapperboard, Dices, Download, Lock, RefreshCw, Square, Trash2, Wand2 } from "lucide-react";
import { useStore, videoDimensions, videoFrameChoices, type VideoSettings } from "../store";
import { formatSeconds } from "../lib/format";

interface VideosInfo {
  model?: string;
  max_size?: string;
  max_frames?: number;
  fps?: number;
  weights?: string;
  precision?: string;
}

const ASPECTS: Array<{ id: VideoSettings["aspect"]; label: string }> = [
  { id: "16:9", label: "16:9" },
  { id: "9:16", label: "9:16" },
  { id: "1:1", label: "1:1" },
  { id: "4:3", label: "4:3" },
  { id: "3:4", label: "3:4" },
];

function sizesUpTo(limit: number): number[] {
  const sides: number[] = [];
  for (let side = 256; side <= limit; side += 64) sides.push(side);
  if (sides[sides.length - 1] !== limit) sides.push(limit);
  return sides;
}

/** Text-to-video workspace: the stage, the render progress, the prompt bar, and the settings column. */
export function VideoStudio() {
  const health = useStore((state) => state.health);
  const info = (health?.execution?.videos as VideosInfo | null | undefined) ?? null;
  const [maxWidth, maxHeight] = String(info?.max_size ?? "640x384").split("x").map((side) => parseInt(side, 10) || 32);
  const maxFrames = Number(info?.max_frames ?? 124);
  const fps = Number(info?.fps ?? 24);
  const videos = useStore((state) => state.videos);
  const currentVideoId = useStore((state) => state.currentVideoId);
  const progress = useStore((state) => state.videoProgress);
  const error = useStore((state) => state.videoError);
  const prompt = useStore((state) => state.videoPrompt);
  const setPrompt = useStore((state) => state.setVideoPrompt);
  const settings = useStore((state) => state.videoSettings);
  const update = useStore((state) => state.updateVideoSettings);
  const generateVideo = useStore((state) => state.generateVideo);
  const cancelVideo = useStore((state) => state.cancelVideo);
  const deleteVideo = useStore((state) => state.deleteVideo);
  const current = videos.find((video) => video.id === currentVideoId) ?? null;
  const url = useMemo(() => (current ? URL.createObjectURL(current.blob) : null), [current]);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  const [elapsed, setElapsed] = useState(0);
  const textarea = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!progress) return;
    const timer = window.setInterval(() => setElapsed((Date.now() - progress.startedAt) / 1000), 250);
    return () => window.clearInterval(timer);
  }, [progress]);

  const { width, height } = videoDimensions(settings, maxWidth, maxHeight);
  const frameChoices = videoFrameChoices(maxFrames);
  const frames = frameChoices.includes(settings.frames) ? settings.frames : frameChoices[frameChoices.length - 1];
  const running = progress !== null;
  const canRender = !running && prompt.trim().length > 0 && info !== null;

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
            <div className="empty__badge"><Clapperboard size={22} /></div>
            <h2>No video model loaded</h2>
            <p>Start the server with <code>--video-model DIR</code> pointing at a MiniMax-H3 model directory to render clips here.</p>
          </div>
        </div>
      </section>
    );
  }

  const stageLabel = progress
    ? progress.stage === "encode"
      ? "Encoding the prompt"
      : progress.step >= progress.steps - 1
        ? "Decoding the frames"
        : `Step ${progress.step} of ${progress.steps - 1}`
    : "";

  return (
    <section className="studio">
      <div className="studio__stage">
        <div className="studio__canvas">
          {current && url ? (
            <video key={current.id} src={url} className={running ? "studio__video studio__video--dim" : "studio__video"} controls autoPlay loop muted playsInline aria-label={current.prompt} />
          ) : (
            !running && (
              <div className="empty">
                <div className="empty__badge"><Clapperboard size={22} /></div>
                <h2>What should we film?</h2>
                <p>Describe the shot below. Ctrl+Enter renders; a clip takes a few minutes, and the history on the left keeps everything.</p>
              </div>
            )
          )}
          {running && (
            <div className="studio__progress" role="status">
              <div className="studio__bar">
                <div className="studio__bar-fill" style={{ width: `${Math.max(4, (100 * progress.step) / Math.max(1, progress.steps))}%` }} />
              </div>
              <span>{stageLabel} · {formatSeconds(elapsed)}</span>
            </div>
          )}
        </div>
        {current && !running && (
          <div className="studio__caption">
            <span className="studio__caption-text" title={current.prompt}>{current.prompt}</span>
            <span className="muted">
              {current.width}x{current.height} · {current.frames} frames ({(current.frames / current.fps).toFixed(1)} s) · {current.steps} steps · seed {current.seed} · {formatSeconds(current.seconds)}
            </span>
          </div>
        )}
        {error && <p className="error-text studio__error">{error}</p>}
        <div className="studio__prompt">
          <textarea
            ref={textarea}
            rows={2}
            value={prompt}
            placeholder="a paper boat drifting down a rain-filled gutter, close up, soft afternoon light"
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
                event.preventDefault();
                if (canRender) void generateVideo();
              }
            }}
            aria-label="Prompt"
          />
          {running ? (
            <button className="button button--small" onClick={cancelVideo} title="Stop rendering">
              <Square size={14} /> Stop
            </button>
          ) : (
            <button className="button button--primary" onClick={() => void generateVideo()} disabled={!canRender} title="Render (Ctrl+Enter)">
              <Wand2 size={15} /> Generate
            </button>
          )}
        </div>
      </div>

      <aside className="studio__settings" aria-label="Video settings">
        <section className="settings__section">
          <h3>Canvas</h3>
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
            <select className="select select--compact" value={Math.min(settings.size, Math.max(maxWidth, maxHeight))} onChange={(event) => update({ size: Number(event.target.value) })}>
              {sizesUpTo(Math.max(maxWidth, maxHeight)).map((side) => (
                <option key={side} value={side}>{side}</option>
              ))}
            </select>
          </label>
          <label className="field field--inline">
            <span className="field__label">Length</span>
            <select className="select select--compact" value={frames} onChange={(event) => update({ frames: Number(event.target.value) })}>
              {frameChoices.map((count) => (
                <option key={count} value={count}>{(count / fps).toFixed(1)} s · {count} frames</option>
              ))}
            </select>
          </label>
          <p className="muted">{width} x {height} at {fps} fps, up to {info.max_size} and {maxFrames} frames on this server.</p>
        </section>

        <section className="settings__section">
          <h3>Sampling</h3>
          <label className="field field--inline">
            <span className="field__label">Steps</span>
            <input type="number" min={2} max={50} value={settings.steps} onChange={(event) => update({ steps: Math.max(2, Math.min(50, Number(event.target.value) || 2)) })} />
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
            {info.precision === "exact" ? "Exact" : info.precision === "balanced" ? "Balanced" : "Fast"} precision · weights on {info.weights === "host" ? "host" : "GPU"} · {info.model}
          </p>
        </section>

        <section className="settings__section">
          <h3>This clip</h3>
          <div className="studio__actions">
            <button className="button button--small" disabled={!current || running} onClick={() => current && void generateVideo({ seed: current.seed })} title="Render the same prompt and seed again">
              <RefreshCw size={13} /> Again
            </button>
            <button className="button button--small" disabled={!current || running} onClick={() => { reuse(); void generateVideo({ seed: null }); }} title="Same prompt, fresh seed">
              <Dices size={13} /> Variation
            </button>
            <a className={`button button--small${current ? "" : " button--disabled"}`} href={url ?? undefined} download={current ? `flyweight-${current.seed}.mp4` : undefined} aria-disabled={!current}>
              <Download size={13} /> MP4
            </a>
            <button className="button button--small button--ghost" disabled={!current} onClick={reuse} title="Put this clip's prompt back in the box">
              Reuse prompt
            </button>
            <button className="button button--small button--ghost" disabled={!current || running} onClick={() => current && void deleteVideo(current.id)}>
              <Trash2 size={13} /> Delete
            </button>
          </div>
        </section>
      </aside>
    </section>
  );
}
