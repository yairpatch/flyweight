import { FileCode, FileSpreadsheet, FileText, Images, ScanText, TriangleAlert, X } from "lucide-react";
import { describe } from "../lib/attachments";
import type { Attachment } from "../types";

function Icon({ attachment }: { attachment: Attachment }) {
  const size = 14;
  if (attachment.kind === "sheet") return <FileSpreadsheet size={size} />;
  if (attachment.kind === "pdf") return attachment.mode === "pages" ? <Images size={size} /> : <FileText size={size} />;
  if (attachment.kind === "document") return <FileText size={size} />;
  return attachment.language ? <FileCode size={size} /> : <FileText size={size} />;
}

interface Props {
  attachment: Attachment;
  /** Absent in the transcript, where an attachment can no longer be changed. */
  onRemove?: () => void;
  onTogglePages?: () => void;
  busy?: boolean;
}

export function AttachmentChip({ attachment, onRemove, onTogglePages, busy }: Props) {
  const failed = Boolean(attachment.error);
  return (
    <div className={`attachment${failed ? " attachment--error" : ""}${busy ? " attachment--busy" : ""}`} title={attachment.error ?? attachment.name}>
      <span className="attachment__icon">{failed ? <TriangleAlert size={14} /> : <Icon attachment={attachment} />}</span>
      <span className="attachment__text">
        <span className="attachment__name">{attachment.name}</span>
        <span className="attachment__meta">{attachment.error ?? describe(attachment)}</span>
      </span>
      {onTogglePages && attachment.kind === "pdf" && !failed && (
        <button
          className="attachment__action"
          onClick={onTogglePages}
          disabled={busy}
          title={attachment.mode === "pages" ? "Send the text layer instead" : "Send the pages to the vision tower instead"}
        >
          {attachment.mode === "pages" ? <ScanText size={12} /> : <Images size={12} />}
        </button>
      )}
      {onRemove && (
        <button className="attachment__action" onClick={onRemove} aria-label={`Remove ${attachment.name}`}>
          <X size={12} />
        </button>
      )}
    </div>
  );
}
