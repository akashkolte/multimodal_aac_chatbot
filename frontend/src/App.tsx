import { useState, useCallback, useEffect, useRef } from "react";
import type { Persona, Affect, ChatMessage, LatencyLog } from "./types";
import { resetSession, checkHealth } from "./lib/api";
import { useWebcam } from "./hooks/useWebcam";
import { useSensing } from "./hooks/useSensing";
import { PersonaSelector } from "./components/PersonaSelector";
import { ChatPanel } from "./components/ChatPanel";
import { WebcamSensing } from "./components/WebcamSensing";
import { SensingStatus } from "./components/SensingStatus";
import { LatencyMetrics } from "./components/LatencyMetrics";
import "./App.css";

function App() {
  const [persona, setPersona] = useState<Persona | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [latency, setLatency] = useState<LatencyLog | null>(null);
  const [webcamEnabled, setWebcamEnabled] = useState(false);
  const [affectOverride, setAffectOverride] = useState<Affect | null>(null);
  const [backendReady, setBackendReady] = useState(false);
  const healthPoll = useRef<ReturnType<typeof setInterval>>(undefined);

  useEffect(() => {
    async function poll() {
      const ready = await checkHealth();
      if (ready) {
        setBackendReady(true);
        clearInterval(healthPoll.current);
      }
    }
    poll();
    healthPoll.current = setInterval(poll, 2000);
    return () => clearInterval(healthPoll.current);
  }, []);

  const {
    sensing,
    ready,
    initError,
    init,
    processFrame,
    clearAirWrittenText,
    clearHeadSignal,
    calibrateHeadPose,
    resetCalibration,
  } = useSensing();

  const onFrame = useCallback(
    (video: HTMLVideoElement, timestamp: number) => {
      processFrame(video, timestamp);
    },
    [processFrame]
  );

  const { videoRef, active, error } = useWebcam({
    enabled: webcamEnabled && ready,
    onFrame,
  });

  async function handleWebcamToggle() {
    if (!webcamEnabled) {
      const ok = await init();
      if (ok) setWebcamEnabled(true);
    } else {
      setWebcamEnabled(false);
      resetCalibration();
    }
  }

  async function handlePersonaSelect(p: Persona) {
    setPersona(p);
    setMessages([]);
    setLatency(null);
    try {
      await resetSession(p.id);
    } catch {
      // Session reset failed — non-critical, continue with fresh UI state
    }
  }

  return (
    <div className="app-layout">
      <aside className="sidebar">
        <h1 className="app-title">AAC Chatbot</h1>

        <PersonaSelector
          selected={persona?.id ?? null}
          onSelect={handlePersonaSelect}
        />

        <div className="sidebar-section">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={webcamEnabled}
              onChange={handleWebcamToggle}
            />
            Enable webcam
          </label>
          <WebcamSensing videoRef={videoRef} active={active} error={error || initError} />
          <SensingStatus sensing={sensing} webcamActive={active} />
          <button
            type="button"
            className="calibrate-btn"
            disabled={!active}
            onClick={() => calibrateHeadPose()}
          >
            {sensing.headCalibrated
              ? "Re-calibrate head pose"
              : "Calibrate head pose"}
          </button>
        </div>

        <div className="sidebar-section">
          <label htmlFor="affect-override">Affect override</label>
          <select
            id="affect-override"
            value={affectOverride ?? "auto"}
            onChange={(e) =>
              setAffectOverride(
                e.target.value === "auto" ? null : (e.target.value as Affect)
              )
            }
          >
            <option value="auto">Auto (webcam)</option>
            <option value="HAPPY">HAPPY</option>
            <option value="FRUSTRATED">FRUSTRATED</option>
            <option value="NEUTRAL">NEUTRAL</option>
            <option value="SURPRISED">SURPRISED</option>
          </select>
        </div>

        <LatencyMetrics latency={latency} />
      </aside>

      <main className="main-content">
        <ChatPanel
          userId={persona?.id ?? null}
          personaName={persona?.name ?? ""}
          sensing={sensing}
          affectOverride={affectOverride}
          onAirTextConsumed={clearAirWrittenText}
          onHeadSignalConsumed={clearHeadSignal}
          messages={messages}
          setMessages={setMessages}
          onLatency={setLatency}
          backendReady={backendReady}
        />
      </main>
    </div>
  );
}

export default App;
