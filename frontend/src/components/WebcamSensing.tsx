import type { RefObject } from "react";

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  active: boolean;
  error: string | null;
}

export function WebcamSensing({ videoRef, active, error }: Props) {
  return (
    <div className="webcam-container">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        style={{
          width: "100%",
          borderRadius: 8,
          display: active ? "block" : "none",
          transform: "scaleX(-1)",
        }}
      />
      {!active && !error && (
        <div className="webcam-placeholder">Camera off</div>
      )}
      {error && <div className="webcam-error">{error}</div>}
    </div>
  );
}
