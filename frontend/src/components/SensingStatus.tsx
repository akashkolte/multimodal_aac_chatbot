import type { SensingState } from "../types";

const AFFECT_EMOJI: Record<string, string> = {
  HAPPY: "\ud83d\ude0a",
  FRUSTRATED: "\ud83d\ude24",
  NEUTRAL: "\ud83d\ude10",
  SURPRISED: "\ud83d\ude32",
};

interface Props {
  sensing: SensingState;
  webcamActive: boolean;
}

export function SensingStatus({ sensing, webcamActive }: Props) {
  if (!webcamActive) {
    return <p className="sensing-off">Webcam off</p>;
  }

  return (
    <div className="sensing-status">
      <div className="sensing-row">
        <span className="sensing-label">Affect</span>
        <span className="sensing-value">
          {AFFECT_EMOJI[sensing.affect ?? "NEUTRAL"]}{" "}
          {sensing.affect ?? "NEUTRAL"}
        </span>
      </div>
      <div className="sensing-row">
        <span className="sensing-label">Gesture</span>
        <span className="sensing-value">
          {sensing.gestureTag ?? "none"}
        </span>
      </div>
      <div className="sensing-row">
        <span className="sensing-label">Gaze</span>
        <span className="sensing-value">
          {sensing.gazeBucket ?? "none"}
        </span>
      </div>
      {sensing.airWrittenText && (
        <div className="sensing-row">
          <span className="sensing-label">Air-written</span>
          <span className="sensing-value">{sensing.airWrittenText}</span>
        </div>
      )}
    </div>
  );
}
