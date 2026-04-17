import { useState } from "react";
import { submitRating } from "../lib/api";
import type { EvalScores } from "../types";

interface Props {
  runId: string;
  userId: string;
  latencyTotal: number;
  sloTarget?: number;
  evalScores?: EvalScores | null;
}

function gradeClass(score: number): string {
  if (score >= 0.75) return "good";
  if (score >= 0.4) return "mid";
  return "bad";
}

function fmt(score: number): string {
  return (score * 100).toFixed(0) + "%";
}

export function EvalPanel({
  runId,
  userId,
  latencyTotal,
  sloTarget = 6.0,
  evalScores,
}: Props) {
  const [value, setValue] = useState<number | null>(null);
  const [hover, setHover] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const sloPassed = evalScores
    ? evalScores.slo_passed
    : latencyTotal > 0 && latencyTotal < sloTarget;
  const effectiveLatency = evalScores?.t_total_s ?? latencyTotal;

  async function rate(stars: number) {
    if (submitting || value !== null) return;
    setSubmitting(true);
    try {
      await submitRating({
        run_id: runId,
        user_id: userId,
        authenticity: stars,
      });
      setValue(stars);
    } catch (e) {
      console.error("rating submit failed", e);
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="eval-panel">
      <div className="eval-row">
        {effectiveLatency > 0 && (
          <span
            className={`slo-badge ${sloPassed ? "pass" : "fail"}`}
            title={`SLO target ${sloTarget.toFixed(1)}s`}
          >
            {effectiveLatency.toFixed(2)}s {sloPassed ? "✓" : "✗"}
          </span>
        )}
        {evalScores && (
          <>
            <span
              className={`eval-pill ${
                evalScores.no_evidence ? "muted" : gradeClass(evalScores.groundedness)
              }`}
              title={
                evalScores.no_evidence
                  ? "No retrieved evidence — groundedness not scored"
                  : `Groundedness: fraction of response sentences supported by retrieved memories (hallucination ${fmt(
                      evalScores.hallucination_rate
                    )})`
              }
            >
              grounded {evalScores.no_evidence ? "—" : fmt(evalScores.groundedness)}
            </span>
            <span
              className={`eval-pill ${gradeClass(evalScores.affect_alignment)}`}
              title="Affect alignment: does the response tone match the detected facial affect?"
            >
              affect {fmt(evalScores.affect_alignment)}
            </span>
            <span
              className={`eval-pill ${gradeClass(evalScores.gesture_alignment)}`}
              title="Gesture alignment: does the response acknowledge the detected hand gesture?"
            >
              gesture {fmt(evalScores.gesture_alignment)}
            </span>
            <span
              className={`eval-pill ${gradeClass(evalScores.gaze_alignment)}`}
              title="Gaze alignment: did retrieved chunks come from the bucket the user was looking at?"
            >
              gaze {fmt(evalScores.gaze_alignment)}
            </span>
          </>
        )}
        <div className="star-rating" title="Rate authenticity (1-5)">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              className={`star ${star <= (hover || (value ?? 0)) ? "active" : ""}`}
              onMouseEnter={() => setHover(star)}
              onMouseLeave={() => setHover(0)}
              onClick={() => rate(star)}
              disabled={value !== null || submitting}
            >
              ★
            </button>
          ))}
          {value !== null && <span className="star-label">{value}/5</span>}
        </div>
      </div>
    </div>
  );
}
