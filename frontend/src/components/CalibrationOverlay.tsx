interface Props {
  active: boolean;
  progress: number;  // 0 → 1
  onCancel?: () => void;
}

export function CalibrationOverlay({ active, progress, onCancel }: Props) {
  if (!active) return null;

  const pct = Math.round(progress * 100);
  const secondsLeft = Math.max(0, Math.ceil(5 - progress * 5));

  return (
    <div className="calibration-overlay" role="dialog" aria-live="polite">
      <div className="calibration-card">
        <h2>Calibrating sensing</h2>
        <p className="calibration-instructions">
          Look at the camera with a relaxed, neutral expression.
          <br />
          We're learning your baseline so detection works on your face.
        </p>
        <div className="calibration-bar">
          <div
            className="calibration-bar-fill"
            style={{ width: `${pct}%` }}
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            role="progressbar"
          />
        </div>
        <p className="calibration-countdown">
          {secondsLeft > 0 ? `${secondsLeft}s remaining` : "Finishing…"}
        </p>
        {onCancel && (
          <button
            type="button"
            className="calibration-cancel"
            onClick={onCancel}
          >
            Skip
          </button>
        )}
      </div>
    </div>
  );
}
