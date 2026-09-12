import { useState } from "react";
import { FolderPlus, Trash2 } from "lucide-react";
import { useStore } from "../store";
import { PERMISSION_PRESETS, DEFAULT_PERMISSIONS, permissionPreset, searchInfo, workspaceFor, workspaceList } from "../lib/agentTools";
import type { AgentPermissions, Conversation } from "../types";

/**
 * Where a run works and what it may do. Shown in the agent tab's empty state,
 * before the first message, which is the only time the directory can change:
 * a transcript full of paths relative to one root cannot move to another.
 * The permissions preset can change at any time; it applies from the next
 * tool call.
 */
export function AgentSetup({ conversation }: { conversation: Conversation | null }) {
  const props = useStore((state) => state.props);
  const canRegister = useStore((state) => state.canRegisterWorkspaces);
  const platform = props?.agent_platform ?? null;
  const newConversation = useStore((state) => state.newConversation);
  const setRunWorkspace = useStore((state) => state.setRunWorkspace);
  const setRunPermissions = useStore((state) => state.setRunPermissions);
  const addWorkspace = useStore((state) => state.addWorkspace);
  const removeWorkspace = useStore((state) => state.removeWorkspace);
  const [path, setPath] = useState("");
  const [adding, setAdding] = useState(false);

  const workspaces = workspaceList(props);
  const current = workspaceFor(props, conversation?.workspaceId);
  const permissions = conversation?.permissions ?? DEFAULT_PERMISSIONS;
  const locked = Boolean(conversation?.messages.length);
  const toolsOff = !Array.isArray(props?.agent_workspaces);
  const search = searchInfo(props);

  /** The run these choices belong to; made on demand so a choice is never lost. */
  const runId = () => conversation?.id ?? newConversation("agent");

  const choose = (workspaceId: string) => {
    if (!workspaceId) return;
    setRunWorkspace(runId(), workspaceId);
  };
  const submit = async () => {
    const trimmed = path.trim();
    if (!trimmed || adding) return;
    setAdding(true);
    const id = await addWorkspace(trimmed);
    setAdding(false);
    if (id) {
      setPath("");
      if (!locked) setRunWorkspace(runId(), id);
    }
  };

  if (toolsOff) {
    return (
      <p className="muted">
        This server's agent tools are off (<code>--no-agent-tools</code>), so only tools with a JavaScript handler can run here.
      </p>
    );
  }

  const separator = platform?.path_separator === "\\" ? "C:\\projects\\app" : "/home/you/projects/app";

  return (
    <div className="agent-setup">
      <label className="field">
        <span className="field__label">
          Workspace <small>{locked ? "fixed for this run" : "the directory the run works in"}</small>
        </span>
        {workspaces.length ? (
          <div className="inline">
            <select className="select" value={current?.id ?? ""} onChange={(event) => choose(event.target.value)} disabled={locked} aria-label="Workspace">
              {!current && <option value="">Choose a directory…</option>}
              {workspaces.map((workspace) => (
                <option key={workspace.id} value={workspace.id} disabled={!workspace.exists}>
                  {workspace.title} — {workspace.path}
                  {workspace.exists ? "" : " (missing)"}
                </option>
              ))}
            </select>
            {canRegister && current?.source === "registered" && !locked && (
              <button className="button button--ghost button--small" onClick={() => void removeWorkspace(current.id)} title="Forget this directory (files are untouched)">
                <Trash2 size={14} />
              </button>
            )}
          </div>
        ) : (
          <p className="muted">
            {canRegister
              ? "No directory yet. Add one below; the server remembers it."
              : "No directory yet. Start the server with --agent-workspace DIR, or open this page in a browser on the server's own machine to add one here."}
          </p>
        )}
      </label>
      {canRegister && (
        <div className="field">
          <span className="field__label">
            Add a directory <small>an absolute path on the server's machine</small>
          </span>
          <div className="inline">
            <input
              className="input"
              value={path}
              placeholder={separator}
              onChange={(event) => setPath(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") void submit();
              }}
              aria-label="Directory to add"
            />
            <button className="button button--small" onClick={() => void submit()} disabled={!path.trim() || adding}>
              <FolderPlus size={14} /> Add
            </button>
          </div>
        </div>
      )}
      <label className="field">
        <span className="field__label">
          Permissions <small>what this run may do</small>
        </span>
        <select className="select" value={permissions} onChange={(event) => setRunPermissions(runId(), event.target.value as AgentPermissions)} aria-label="Permissions">
          {PERMISSION_PRESETS.map((preset) => (
            <option key={preset.value} value={preset.value}>
              {preset.label}
            </option>
          ))}
        </select>
        <small className="muted">{permissionPreset(permissions).description}</small>
      </label>
      <p className="muted">
        {/* Every preset reaches the internet: none of them fences the network,
            and a user choosing "read only" for the disk should not have to
            discover that from a transcript. */}
        Web: this run can read public pages with <code>fetch_url</code>
        {search?.ready ? (
          <>
            {" "}
            and search with <code>web_search</code> ({search.provider}).
          </>
        ) : (
          <>
            . Search is off{search?.detail ? ` — ${search.detail}` : ""}.
          </>
        )}
      </p>
    </div>
  );
}
