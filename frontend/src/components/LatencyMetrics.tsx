import type { LatencyLog } from "../types";

interface Props {
  latency: LatencyLog | null;
}

const FIELDS: { key: keyof LatencyLog; label: string }[] = [
  { key: "t_sensing", label: "Sensing" },
  { key: "t_intent", label: "Intent" },
  { key: "t_retrieval", label: "Retrieval" },
  { key: "t_generation", label: "Generation" },
  { key: "t_total", label: "Total" },
];

export function LatencyMetrics({ latency }: Props) {
  if (!latency) return <p className="no-metrics">No turn yet</p>;

  return (
    <div className="latency-metrics">
      <h3>Latency</h3>
      {FIELDS.map(({ key, label }) => (
        <div key={key} className="metric-row">
          <span className="metric-label">{label}</span>
          <span className="metric-value">{(latency[key] ?? 0).toFixed(3)}s</span>
        </div>
      ))}
    </div>
  );
}
