import type { ReactElement } from "react";
import { FolderOpen, RotateCcw } from "lucide-react";
import type { OptionSpec, Scalar } from "../api";
import { labelOf, useDesk } from "../state";

export function isDefault(spec: OptionSpec, value: Scalar | undefined): boolean {
  if (value === undefined || value === null || value === "") return true;
  if (spec.kind === "bool") return Boolean(value) === Boolean(spec.default ?? false);
  return String(value) === String(spec.default ?? "");
}

export function invalid(spec: OptionSpec, value: Scalar | undefined): string {
  if (isDefault(spec, value)) return "";
  if (spec.kind === "int" && !/^-?\d+$/.test(String(value).trim())) return "a whole number";
  if (spec.kind === "float" && !Number.isFinite(Number(value))) return "a number";
  return "";
}

/** The default as a person reads it: the CLI's sentinels named. */
function shownDefault(spec: OptionSpec): string {
  const d = spec.default;
  if (spec.required) return "required";
  if (d === null || d === undefined || d === "") return spec.kind === "path" || spec.kind === "dir" ? "none" : "automatic";
  if (spec.kind === "bool") return d ? "on" : "off";
  if (d === 4294967295) return "automatic";
  return String(d);
}

interface Props {
  spec: OptionSpec;
  value: Scalar | undefined;
  onChange: (value: Scalar | undefined) => void;
  disabled?: boolean;
  compact?: boolean;
}

export function OptionField({ spec, value, onChange, disabled, compact }: Props) {
  const call = useDesk((s) => s.call);
  const changed = !isDefault(spec, value);
  const problem = invalid(spec, value);
  const label = labelOf(spec.flag, spec.dest);
  const id = `opt-${spec.dest}`;

  let control: ReactElement;
  if (spec.kind === "bool") {
    const on = value === undefined || value === null ? Boolean(spec.default) : Boolean(value);
    control = (
      <label className="d-switch">
        <input id={id} type="checkbox" checked={on} disabled={disabled}
               onChange={(e) => onChange(e.target.checked === Boolean(spec.default ?? false) ? undefined : e.target.checked)} />
        <span />
      </label>
    );
  } else if (spec.kind === "choice" && spec.choices) {
    const current = value ?? spec.default ?? "";
    const short = spec.choices.length <= 5 && spec.choices.every((c) => String(c).length <= 10);
    control = short ? (
      <div className="d-seg" role="radiogroup" aria-labelledby={`${id}-label`}>
        {spec.default === null && (
          <button className={current === "" ? "on" : ""} disabled={disabled} onClick={() => onChange(undefined)}>auto</button>
        )}
        {spec.choices.map((c) => (
          <button key={String(c)} className={String(current) === String(c) ? "on" : ""} disabled={disabled}
                  onClick={() => onChange(String(c) === String(spec.default) ? undefined : c)}>{String(c)}</button>
        ))}
      </div>
    ) : (
      <select id={id} className="d-select" value={String(current)} disabled={disabled}
              onChange={(e) => onChange(e.target.value === String(spec.default ?? "") ? undefined : e.target.value)}>
        {spec.default === null && <option value="">automatic</option>}
        {spec.choices.map((c) => <option key={String(c)} value={String(c)}>{String(c)}</option>)}
      </select>
    );
  } else {
    const numeric = spec.kind === "int" || spec.kind === "float";
    const pickable = spec.kind === "path" || spec.kind === "dir";
    control = (
      <div className="d-inputrow">
        <input id={id} className={`d-input${numeric ? " d-input--num" : ""}${pickable ? " mono" : ""}${problem ? " invalid" : ""}`}
               value={value === undefined || value === null ? "" : String(value)} disabled={disabled}
               inputMode={numeric ? "decimal" : undefined} spellCheck={false}
               type={spec.dest === "api_key" ? "password" : "text"}
               placeholder={shownDefault(spec)}
               onChange={(e) => onChange(e.target.value === "" ? undefined : e.target.value)} />
        {pickable && (
          <button className="d-iconbtn" disabled={disabled} title="Browse"
                  onClick={async () => {
                    const r = await call<{ ok: boolean; path: string | null }>("/api/pick", { kind: spec.kind === "dir" ? "folder" : "file" });
                    if (r?.ok && r.path) onChange(r.path);
                  }}>
            <FolderOpen size={15} />
          </button>
        )}
      </div>
    );
  }

  return (
    <div className={`d-opt${changed ? " d-opt--changed" : ""}${compact ? " d-opt--compact" : ""}`}>
      <div className="d-opt__text">
        <div className="d-opt__label" id={`${id}-label`}>
          <label htmlFor={id}>{label}</label>
          <code>{spec.flag}</code>
          {changed && (
            <button className="d-opt__reset" title={`Back to the default (${shownDefault(spec)})`} onClick={() => onChange(undefined)} disabled={disabled}>
              <RotateCcw size={12} />
            </button>
          )}
        </div>
        {!compact && <div className="d-opt__help">{spec.help}</div>}
        {problem && <div className="d-opt__error">Needs {problem}.</div>}
      </div>
      <div className="d-opt__control">{control}</div>
    </div>
  );
}
