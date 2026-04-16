import { useState } from "react";
import type { EvalScores } from "../types";

interface Props {
  evalScores: EvalScores;
}

function ScoreBar({ value }: { value: number }) {
  const pct = Math.min(value * 100, 100);
  const color = pct > 70 ? "#4caf50" : pct > 40 ? "#ff9800" : "#f44336";
  return (
    <div className="score-bar">
      <div className="score-bar-fill" style={{ width: `${pct}%`, background: color }} />
    </div>
  );
}

function StarRating({
  value,
  onChange,
}: {
  value: number | null;
  onChange: (v: number) => void;
}) {
  const [hover, setHover] = useState(0);
  return (
    <div className="star-rating">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          className={`star ${star <= (hover || (value ?? 0)) ? "active" : ""}`}
          onMouseEnter={() => setHover(star)}
          onMouseLeave={() => setHover(0)}
          onClick={() => onChange(star)}
        >
          ★
        </button>
      ))}
      {value !== null && <span className="star-label">{value}/5</span>}
    </div>
  );
}

export function EvalPanel({ evalScores }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [likert, setLikert] = useState<number | null>(null);

  return (
    <div className="eval-panel">
      <button
        className="eval-toggle"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? "▾" : "▸"} Eval Metrics
        {evalScores.slo_passed ? (
          <span className="slo-badge pass">SLO ✓</span>
        ) : (
          <span className="slo-badge fail">SLO ✗</span>
        )}
        {likert !== null && (
          <span className="slo-badge">{likert}/5 ★</span>
        )}
      </button>

      {expanded && (
        <div className="eval-details">
          <div className="eval-section">
            <div className="section-title">Factual Faithfulness</div>
            {evalScores.no_evidence ? (
              <div className="eval-na">N/A — no evidence retrieved</div>
            ) : (
              <>
                <div className="metric-row">
                  <span>Groundedness</span>
                  <span className="metric-value">{(evalScores.groundedness * 100).toFixed(0)}%</span>
                </div>
                <ScoreBar value={evalScores.groundedness} />
                <div className="metric-row">
                  <span>Hallucination Rate</span>
                  <span className={`metric-value ${evalScores.hallucination_rate > 0.2 ? "fail" : "pass"}`}>
                    {(evalScores.hallucination_rate * 100).toFixed(0)}%
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="eval-section">
            <div className="section-title">Communication Efficiency</div>
            <div className="metric-row">
              <span>Response Time</span>
              <span className={`metric-value ${evalScores.slo_passed ? "pass" : "fail"}`}>
                {evalScores.t_total_s.toFixed(2)}s
                {evalScores.slo_passed ? " ✓" : " ✗"}
              </span>
            </div>
            <div className="metric-row sub">
              <span>SLO Target</span>
              <span className="metric-value">
                &lt; {evalScores.slo_target_s.toFixed(1)}s (margin: {evalScores.slo_margin_s.toFixed(2)}s)
              </span>
            </div>
          </div>

          <div className="eval-section">
            <div className="section-title">Multimodal Alignment</div>
            <div className="metric-row">
              <span>Overall</span>
              <span className="metric-value">{(evalScores.multimodal_alignment * 100).toFixed(0)}%</span>
            </div>
            <ScoreBar value={evalScores.multimodal_alignment} />
            <div className="metric-row sub">
              <span>Affect</span>
              <span className="metric-value">{(evalScores.affect_alignment * 100).toFixed(0)}%</span>
            </div>
            <div className="metric-row sub">
              <span>Gesture</span>
              <span className="metric-value">{(evalScores.gesture_alignment * 100).toFixed(0)}%</span>
            </div>
            <div className="metric-row sub">
              <span>Gaze</span>
              <span className="metric-value">{(evalScores.gaze_alignment * 100).toFixed(0)}%</span>
            </div>
          </div>

          <div className="eval-section">
            <div className="section-title">Perceived Authenticity</div>
            <div className="metric-row">
              <span>Rate this response</span>
            </div>
            <StarRating value={likert} onChange={setLikert} />
          </div>
        </div>
      )}
    </div>
  );
}
