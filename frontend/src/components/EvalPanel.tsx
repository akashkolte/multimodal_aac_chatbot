import { useState } from "react";
import { submitRating } from "../lib/api";

interface Props {
  runId: string;
  userId: string;
  latencyTotal: number;
  sloTarget?: number;
}

export function EvalPanel({ runId, userId, latencyTotal, sloTarget = 6.0 }: Props) {
  const [value, setValue] = useState<number | null>(null);
  const [hover, setHover] = useState(0);
  const [submitting, setSubmitting] = useState(false);
  const sloPassed = latencyTotal > 0 && latencyTotal < sloTarget;

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
        {latencyTotal > 0 && (
          <span className={`slo-badge ${sloPassed ? "pass" : "fail"}`}>
            {latencyTotal.toFixed(2)}s {sloPassed ? "✓" : "✗"}
          </span>
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
