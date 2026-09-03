import { X } from "lucide-react";
import { useStore } from "../store";
import { SettingsPanel } from "./panels/SettingsPanel";
import { ToolsPanel } from "./panels/ToolsPanel";
import { RuntimePanel } from "./panels/RuntimePanel";
import { TokenizerPanel } from "./panels/TokenizerPanel";
import { PlaygroundPanel } from "./panels/PlaygroundPanel";
import { InspectorPanel } from "./panels/InspectorPanel";

const TITLES = {
  settings: "Generation settings",
  tools: "Tools",
  runtime: "Runtime",
  tokenizer: "Tokenizer",
  playground: "Completions playground",
  inspector: "Request inspector",
} as const;

export function SidePanel() {
  const panel = useStore((state) => state.panel);
  const setPanel = useStore((state) => state.setPanel);
  if (!panel) return null;
  return (
    <aside className={`panel panel--${panel}`} aria-label={TITLES[panel]}>
      <header className="panel__header">
        <h2>{TITLES[panel]}</h2>
        <button className="icon-button" onClick={() => setPanel(null)} aria-label="Close panel">
          <X size={16} />
        </button>
      </header>
      <div className="panel__body">
        {panel === "settings" && <SettingsPanel />}
        {panel === "tools" && <ToolsPanel />}
        {panel === "runtime" && <RuntimePanel />}
        {panel === "tokenizer" && <TokenizerPanel />}
        {panel === "playground" && <PlaygroundPanel />}
        {panel === "inspector" && <InspectorPanel />}
      </div>
    </aside>
  );
}
