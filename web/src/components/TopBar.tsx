import type React from "react";
import { Activity, Braces, FlaskConical, Hash, Menu, Monitor, Moon, Settings2, Sun, Wrench, Command } from "lucide-react";
import { useStore, type Panel, type ThemePreference } from "../store";
import { useActiveConversation } from "../store";
import { formatBytes } from "../lib/format";

const THEME_ORDER: ThemePreference[] = ["system", "light", "dark"];

export function TopBar() {
  const status = useStore((state) => state.status);
  const statusDetail = useStore((state) => state.statusDetail);
  const health = useStore((state) => state.health);
  const models = useStore((state) => state.models);
  const model = useStore((state) => state.model);
  const setModel = useStore((state) => state.setModel);
  const panel = useStore((state) => state.panel);
  const setPanel = useStore((state) => state.setPanel);
  const theme = useStore((state) => state.theme);
  const setTheme = useStore((state) => state.setTheme);
  const setPaletteOpen = useStore((state) => state.setPaletteOpen);
  const conversation = useActiveConversation();

  const execution = health?.execution ?? {};
  const backend = String(execution.backend ?? "");
  const device = runtimeDeviceLabel(execution);
  const detail = runtimeDetail(execution);

  const panelButton = (id: Panel, icon: React.ReactNode, label: string, shortcut?: string, secondary = false) => (
    <button
      className={`icon-button${panel === id ? " icon-button--active" : ""}${secondary ? " topbar__secondary" : ""}`}
      onClick={() => setPanel(id)}
      title={shortcut ? `${label} (${shortcut})` : label}
      aria-label={label}
      aria-pressed={panel === id}
    >
      {icon}
    </button>
  );

  return (
    <header className="topbar">
      <div className="topbar__left">
        <button className="icon-button topbar__menu" onClick={() => useStore.getState().toggleSidebar()} aria-label="Toggle conversations">
          <Menu size={18} />
        </button>
        <h1 className="topbar__title" title={conversation?.title}>
          {conversation?.title ?? "Flyweight Chat"}
        </h1>
      </div>
      <div className="topbar__center">
        <div className={`status status--${status}`} title={statusDetail}>
          <span className="status__dot" aria-hidden="true" />
          <span className="status__text">{statusDetail}</span>
        </div>
        {backend && (
          <span className="chip chip--runtime" title={detail}>
            {device}
          </span>
        )}
        {models.length > 1 ? (
          <select className="select select--compact" value={model} onChange={(event) => setModel(event.target.value)} aria-label="Model">
            {models.map((item) => (
              <option key={item.id} value={item.id}>
                {item.id}
              </option>
            ))}
          </select>
        ) : (
          model && (
            <span className="chip chip--model" title={health?.model}>
              {model}
            </span>
          )
        )}
      </div>
      <div className="topbar__right">
        <button className="icon-button" onClick={() => setPaletteOpen(true)} title="Command palette (Ctrl+K)" aria-label="Command palette">
          <Command size={17} />
        </button>
        {panelButton("settings", <Settings2 size={17} />, "Generation settings", "Ctrl+,")}
        {panelButton("tools", <Wrench size={17} />, "Tools")}
        {panelButton("runtime", <Activity size={17} />, "Runtime dashboard")}
        {panelButton("tokenizer", <Hash size={17} />, "Tokenizer", undefined, true)}
        {panelButton("playground", <FlaskConical size={17} />, "Completions playground", undefined, true)}
        {panelButton("inspector", <Braces size={17} />, "Request inspector", undefined, true)}
        <button
          className="icon-button"
          onClick={() => setTheme(THEME_ORDER[(THEME_ORDER.indexOf(theme) + 1) % THEME_ORDER.length])}
          title={`Theme: ${theme}`}
          aria-label="Cycle theme"
        >
          {theme === "system" ? <Monitor size={17} /> : theme === "light" ? <Sun size={17} /> : <Moon size={17} />}
        </button>
      </div>
    </header>
  );
}

export function runtimeDeviceLabel(execution: Record<string, unknown>): string {
  const backend = String(execution.backend ?? "");
  if (backend === "native-v2-cpp-cuda") {
    const mode = String(execution.expert_mode ?? "");
    return mode ? `CUDA · experts ${mode}` : "CUDA";
  }
  if (backend === "native-v2-bailingmoe3") return execution.device === "gpu" ? "GPU" : "CPU";
  if (backend.startsWith("native-v2-deepseek4")) return backend.endsWith("hybrid") ? "Hybrid" : "CPU";
  if (typeof execution.cuda_ready === "boolean") return execution.cuda_ready ? "CUDA" : "CPU";
  return backend.replace(/^native-v2-?/, "") || "runtime";
}

export function runtimeDetail(execution: Record<string, unknown>): string {
  const lines: string[] = [];
  if (execution.backend) lines.push(`backend ${execution.backend}`);
  if (execution.architecture) lines.push(`arch ${execution.architecture}`);
  if (execution.expert_mode) {
    const requested = execution.requested_expert_mode;
    lines.push(`experts ${execution.expert_mode}${requested && requested !== execution.expert_mode ? ` (requested ${requested})` : ""}`);
  }
  if (execution.expert_fallback_reason) lines.push(`fallback: ${execution.expert_fallback_reason}`);
  if (typeof execution.gpu_allocated_bytes === "number") lines.push(`GPU allocated ${formatBytes(execution.gpu_allocated_bytes)}`);
  if (typeof execution.kv_reserved_bytes === "number") lines.push(`KV reserved ${formatBytes(execution.kv_reserved_bytes)}`);
  if (typeof execution.mtp_drafts === "number" && (execution.mtp_drafts as number) > 0) lines.push(`MTP drafts ${execution.mtp_drafts}`);
  if (execution.vision) lines.push("vision tower loaded");
  return lines.join("\n");
}
