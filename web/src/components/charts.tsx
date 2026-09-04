// Chart primitives for the runtime dashboard: stat tile, meter, sparkline.
// Colors come from CSS custom properties (see styles/app.css --viz-*), so
// light and dark each get their own validated step.
import { useId, useState, type ReactNode } from "react";

export function StatTile({ label, value, hint, trend, children }: { label: string; value: ReactNode; hint?: string; trend?: number[]; children?: ReactNode }) {
  return (
    <div className="stat" title={hint}>
      <div className="stat__label">{label}</div>
      <div className="stat__value">{value}</div>
      {trend && trend.length > 1 && <Sparkline values={trend} height={28} compact />}
      {children}
    </div>
  );
}

export function Meter({ label, ratio, detail, severity }: { label: string; ratio: number | null; detail?: string; severity?: "accent" | "warning" | "danger" }) {
  const clamped = ratio === null || !Number.isFinite(ratio) ? 0 : Math.max(0, Math.min(1, ratio));
  const tone = severity ?? (clamped > 0.92 ? "danger" : clamped > 0.75 ? "warning" : "accent");
  return (
    <div className="meter" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(clamped * 100)} aria-label={label}>
      <div className="meter__row">
        <span className="meter__label">{label}</span>
        <span className="meter__detail">{detail ?? (ratio === null ? "–" : `${Math.round(clamped * 100)}%`)}</span>
      </div>
      <div className={`meter__track meter__track--${tone}`}>
        <div className="meter__fill" style={{ width: `${clamped * 100}%` }} />
      </div>
    </div>
  );
}

export interface SparklineProps {
  values: number[];
  labels?: string[];
  height?: number;
  compact?: boolean;
  format?: (value: number) => string;
  ariaLabel?: string;
}

/** Single-series line with a wash, an end marker, and a crosshair tooltip. */
export function Sparkline({ values, labels, height = 96, compact = false, format = (v) => String(Math.round(v * 10) / 10), ariaLabel }: SparklineProps) {
  const [hover, setHover] = useState<number | null>(null);
  const id = useId();
  const width = 320;
  const padX = compact ? 4 : 6;
  const padY = compact ? 4 : 10;
  const count = values.length;
  if (count < 2) return <div className="spark spark--empty" style={{ height }} />;
  const max = Math.max(...values, 0);
  const min = Math.min(...values, 0);
  const span = max - min || 1;
  const x = (index: number) => padX + (index / (count - 1)) * (width - padX * 2);
  const y = (value: number) => height - padY - ((value - min) / span) * (height - padY * 2);
  const path = values.map((value, index) => `${index === 0 ? "M" : "L"}${x(index).toFixed(1)},${y(value).toFixed(1)}`).join(" ");
  const area = `${path} L${x(count - 1).toFixed(1)},${(height - padY).toFixed(1)} L${x(0).toFixed(1)},${(height - padY).toFixed(1)} Z`;
  const last = count - 1;
  const active = hover ?? last;

  return (
    <div className={`spark${compact ? " spark--compact" : ""}`} style={{ height }}>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={ariaLabel ?? `Trend, latest ${format(values[last])}`}
        onMouseMove={(event) => {
          const rect = event.currentTarget.getBoundingClientRect();
          const ratio = (event.clientX - rect.left) / rect.width;
          setHover(Math.max(0, Math.min(last, Math.round(ratio * last))));
        }}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id={`${id}-wash`} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="var(--viz-1)" stopOpacity="0.16" />
            <stop offset="1" stopColor="var(--viz-1)" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {!compact && <line x1={padX} x2={width - padX} y1={height - padY} y2={height - padY} className="spark__baseline" />}
        <path d={area} fill={`url(#${id}-wash)`} />
        <path d={path} className="spark__line" vectorEffect="non-scaling-stroke" />
        {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={padY} y2={height - padY} className="spark__crosshair" vectorEffect="non-scaling-stroke" />}
        <circle cx={x(active)} cy={y(values[active])} r={compact ? 3 : 4} className="spark__dot" vectorEffect="non-scaling-stroke" />
      </svg>
      {!compact && (
        <div className="spark__tip" style={{ left: `${(x(active) / width) * 100}%` }}>
          <strong>{format(values[active])}</strong>
          {labels?.[active] && <span>{labels[active]}</span>}
        </div>
      )}
    </div>
  );
}

export function KeyValueTable({ rows }: { rows: Array<[string, ReactNode]> }) {
  return (
    <table className="kv">
      <tbody>
        {rows.map(([key, value]) => (
          <tr key={key}>
            <th scope="row">{key}</th>
            <td>{value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
