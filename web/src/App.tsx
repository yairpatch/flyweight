import { useEffect } from "react";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { Transcript } from "./components/Transcript";
import { Composer } from "./components/Composer";
import { Toasts } from "./components/Toasts";
import { PreviewOverlay } from "./components/PreviewOverlay";
import { CommandPalette } from "./components/CommandPalette";
import { SidePanel } from "./components/SidePanel";
import { useStore } from "./store";

const POLL_MS = 5000;

export function App() {
  const init = useStore((state) => state.init);
  const pollRuntime = useStore((state) => state.pollRuntime);
  const ready = useStore((state) => state.ready);
  const sidebarOpen = useStore((state) => state.sidebarOpen);
  const panel = useStore((state) => state.panel);

  useEffect(() => {
    void init();
  }, [init]);

  // Poll the runtime while the tab is visible; catch up on return.
  useEffect(() => {
    let inFlight = false;
    const tick = async () => {
      if (document.hidden || inFlight) return;
      inFlight = true;
      try {
        await pollRuntime();
      } finally {
        inFlight = false;
      }
    };
    const timer = window.setInterval(tick, POLL_MS);
    const onVisible = () => {
      if (!document.hidden) void tick();
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [pollRuntime]);

  // Global keyboard shortcuts.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const meta = event.ctrlKey || event.metaKey;
      const state = useStore.getState();
      if (meta && event.key.toLowerCase() === "k") {
        event.preventDefault();
        state.setPaletteOpen(!state.paletteOpen);
        return;
      }
      if (meta && event.shiftKey && event.key.toLowerCase() === "o") {
        event.preventDefault();
        state.newConversation();
        return;
      }
      if (meta && event.key.toLowerCase() === "b") {
        event.preventDefault();
        state.toggleSidebar();
        return;
      }
      if (meta && event.key === ",") {
        event.preventDefault();
        state.setPanel("settings");
        return;
      }
      if (event.key === "Escape") {
        if (state.previewSource) state.setPreviewSource(null);
        else if (state.paletteOpen) state.setPaletteOpen(false);
        else if (state.generating) state.stopGeneration();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className={`shell${sidebarOpen ? "" : " shell--rail"}${panel ? " shell--panel" : ""}`}>
      <Sidebar />
      <div className="scrim" onClick={() => useStore.getState().toggleSidebar(false)} aria-hidden="true" />
      <main className="main" aria-busy={!ready}>
        <TopBar />
        <Transcript />
        <Composer />
      </main>
      <SidePanel />
      <Toasts />
      <PreviewOverlay />
      <CommandPalette />
    </div>
  );
}
