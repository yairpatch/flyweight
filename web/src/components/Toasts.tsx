import { X } from "lucide-react";
import { useStore } from "../store";

export function Toasts() {
  const toasts = useStore((state) => state.toasts);
  const dismiss = useStore((state) => state.dismissToast);
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast toast--${toast.kind}`}>
          <span>{toast.text}</span>
          <button className="icon-button icon-button--small" onClick={() => dismiss(toast.id)} aria-label="Dismiss">
            <X size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}
