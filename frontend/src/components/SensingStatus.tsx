import { useState } from "react";
import type { MemoryBucket, SensingState } from "../types";

const AFFECT_EMOJI: Record<string, string> = {
  HAPPY: "😊",
  FRUSTRATED: "😤",
  NEUTRAL: "😐",
  SURPRISED: "😲",
};

const ZONE_LABEL: Record<MemoryBucket, string> = {
  family:        "👨‍👩‍👧 Family",
  medical:       "🏥 Medical",
  social:        "👥 Social",
  hobbies:       "🎨 Hobbies",
  daily_routine: "🌅 Routine",
};

function GazeZoneMap({ active }: { active: MemoryBucket | null }) {
  const zone = (bucket: MemoryBucket) => (
    <div
      key={bucket}
      className={`gaze-zone${active === bucket ? " gaze-zone--active" : ""}`}
    >
      {ZONE_LABEL[bucket]}
    </div>
  );

  return (
    <div className="gaze-zone-map">
      <div className="gaze-zone-row">
        {zone("family")}
        {zone("medical")}
      </div>
      <div className="gaze-zone-row">
        {zone("social")}
      </div>
      <div className="gaze-zone-row">
        {zone("hobbies")}
        {zone("daily_routine")}
      </div>
    </div>
  );
}

interface Props {
  sensing: SensingState;
  webcamActive: boolean;
  calibrated?: boolean;
  onRecalibrate?: () => void;
}

export function SensingStatus({ sensing, webcamActive, calibrated, onRecalibrate }: Props) {
  const [expanded, setExpanded] = useState(false);

  if (!webcamActive) {
    return <p className="sensing-off">Webcam off</p>;
  }

  return (
    <div className="sensing-status">
      {calibrated !== undefined && (
        <div className="sensing-row">
          <span className="sensing-label">Calibration</span>
          <span className="sensing-value">
            {calibrated ? "✓ ready" : "—"}
            {onRecalibrate && (
              <button
                type="button"
                className="recalibrate-btn"
                onClick={onRecalibrate}
              >
                Recalibrate
              </button>
            )}
          </span>
        </div>
      )}

      <div className="sensing-row" style={{ marginTop: "12px", marginBottom: "8px" }}>
        <button 
          type="button" 
          onClick={() => setExpanded(!expanded)}
          className="recalibrate-btn"
          style={{ width: "100%", padding: "6px", fontSize: "0.85rem", opacity: 0.9 }}
        >
          {expanded ? "▲ Hide Raw Metrics" : "▼ Show Raw Metrics"}
        </button>
      </div>

      {expanded && (
        <>
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
          <GazeZoneMap active={sensing.gazeZone} />
          <div className="sensing-row">
            <span className="sensing-label">Head</span>
            <span className="sensing-value">{sensing.headSignal ?? "steady"}</span>
          </div>
          <div className="sensing-row sensing-debug">
            <span className="sensing-label">  ↳ p/y/r</span>
            <span className="sensing-value">
              {sensing.headDebug.pitch}° / {sensing.headDebug.yaw}° / {sensing.headDebug.roll}°
              {"  "}(x{sensing.headDebug.crossings})
            </span>
          </div>
          <div className="sensing-row">
            <span className="sensing-label">Air-writing</span>
            <span className="sensing-value">
              {sensing.airWritingActive
                ? "✏️ drawing…"
                : sensing.airWrittenText
                ? sensing.airWrittenText
                : "none"}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
