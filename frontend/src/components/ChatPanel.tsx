import { useState, useRef, useEffect, useCallback } from "react";
import type { ChatMessage, SensingState, Affect, LatencyLog } from "../types";
import { sendChat, sendTurnaround } from "../lib/api";
import { EvalPanel } from "./EvalPanel";

interface Props {
  userId: string | null;
  personaName: string;
  sensing: SensingState;
  affectOverride: Affect | null;
  onAirTextConsumed: () => void;
  onHeadSignalConsumed: () => void;
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  onLatency: (latency: LatencyLog) => void;
  backendReady: boolean;
}

const TURNAROUND_WINDOW_MS = 5000;

export function ChatPanel({
  userId,
  personaName,
  sensing,
  affectOverride,
  onAirTextConsumed,
  onHeadSignalConsumed,
  messages,
  setMessages,
  onLatency,
  backendReady,
}: Props) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [turnaroundLoading, setTurnaroundLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastResponseTsRef = useRef<number>(0);
  const lastTurnIdRef = useRef<number | null>(null);
  // turn_id of the most recent turn that was already turned around — guards
  // against the new turnaround bubble's own head-signal re-firing turnaround
  // on itself.
  const turnaroundConsumedTurnRef = useRef<number | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Reset per-turn state when the persona changes (parent clears `messages`
  // and resets the backend session — the frontend turn counter must follow).
  useEffect(() => {
    lastTurnIdRef.current = null;
    turnaroundConsumedTurnRef.current = null;
    lastResponseTsRef.current = 0;
  }, [userId]);

  const handleTurnaround = useCallback(
    async (reason: "head" | "manual") => {
      if (!userId || !backendReady || turnaroundLoading || loading) return;
      const targetTurnId = lastTurnIdRef.current;
      if (targetTurnId === null) return;
      if (turnaroundConsumedTurnRef.current === targetTurnId) return;

      turnaroundConsumedTurnRef.current = targetTurnId;
      setTurnaroundLoading(true);
      try {
        const res = await sendTurnaround({
          user_id: userId,
          turn_id: targetTurnId,
          head_signal: reason === "head" ? sensing.headSignal : null,
        });

        lastTurnIdRef.current = res.turn_id;
        turnaroundConsumedTurnRef.current = res.turn_id;

        setMessages((prev) => {
          const next = [...prev];
          for (let i = next.length - 1; i >= 0; i--) {
            if (next[i].role === "aac_user" && !next[i].isTurnaround) {
              next[i] = { ...next[i], rephrased: true };
              break;
            }
          }
          next.push({
            role: "aac_user",
            content: res.response,
            latency: res.latency,
            affect: res.affect,
            runId: res.run_id,
            turnId: res.turn_id,
            isTurnaround: true,
          });
          return next;
        });
        onLatency(res.latency);
        // Do NOT advance lastResponseTsRef — keep the original turn's window so
        // the user can't head-shake the turnaround itself into another loop.
      } catch (e) {
        setMessages((prev) => [
          ...prev,
          {
            role: "aac_user",
            content: `Error rephrasing: ${
              e instanceof Error ? e.message : "request failed"
            }`,
            isTurnaround: true,
          },
        ]);
      } finally {
        if (reason === "head") onHeadSignalConsumed();
        setTurnaroundLoading(false);
      }
    },
    [
      userId,
      backendReady,
      turnaroundLoading,
      loading,
      sensing.headSignal,
      setMessages,
      onLatency,
      onHeadSignalConsumed,
    ]
  );

  useEffect(() => {
    if (
      sensing.headSignal !== "HEAD_NOD_DISSATISFIED" &&
      sensing.headSignal !== "HEAD_SHAKE"
    ) {
      return;
    }
    const targetTurnId = lastTurnIdRef.current;
    const eligible =
      targetTurnId !== null &&
      turnaroundConsumedTurnRef.current !== targetTurnId &&
      lastResponseTsRef.current > 0 &&
      performance.now() - lastResponseTsRef.current <= TURNAROUND_WINDOW_MS;

    if (eligible) {
      handleTurnaround("head");
      return;
    }
    // Not eligible — keep the chip visible briefly so the user can see that
    // detection fired, then clear it. (Instant clear made detection invisible.)
    const id = window.setTimeout(() => onHeadSignalConsumed(), 1500);
    return () => window.clearTimeout(id);
  }, [sensing.headSignal, handleTurnaround, onHeadSignalConsumed]);

  async function handleSend() {
    if (!input.trim() || !userId || !backendReady || loading) return;

    const query = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "partner", content: query }]);
    setLoading(true);

    const airText = sensing.airWrittenText || null;
    try {
      const res = await sendChat({
        user_id: userId,
        query,
        affect_override: affectOverride ?? sensing.affect,
        gesture_tag: sensing.gestureTag,
        gaze_bucket: sensing.gazeBucket,
        air_written_text: airText,
        head_signal: sensing.headSignal,
      });

      lastTurnIdRef.current = res.turn_id;
      setMessages((prev) => [
        ...prev,
        {
          role: "aac_user",
          content: res.response,
          latency: res.latency,
          affect: res.affect,
          runId: res.run_id,
          turnId: res.turn_id,
        },
      ]);
      onLatency(res.latency);
      lastResponseTsRef.current = performance.now();
    } catch (e) {
      setMessages((prev) => [
        ...prev,
        {
          role: "aac_user",
          content: `Error: ${e instanceof Error ? e.message : "request failed"}`,
        },
      ]);
    } finally {
      if (airText) onAirTextConsumed();
      setLoading(false);
    }
  }

  const canTurnaround =
    !!userId &&
    backendReady &&
    !loading &&
    !turnaroundLoading &&
    lastTurnIdRef.current !== null;

  return (
    <div className="chat-panel">
      <div className="chat-header">
        Talking as: {personaName || "select a persona"}
      </div>
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`chat-bubble ${msg.role}${
              msg.rephrased ? " rephrased" : ""
            }${msg.isTurnaround ? " turnaround" : ""}`}
          >
            <span className="chat-role">
              {msg.role === "partner" ? "Partner" : "AAC User"}
              {msg.rephrased && (
                <span className="badge badge-rephrased"> rephrased</span>
              )}
              {msg.isTurnaround && (
                <span className="badge badge-turnaround"> ↻ turnaround</span>
              )}
            </span>
            <p>{msg.content}</p>
            {msg.role === "aac_user" && msg.runId && userId && (
              <EvalPanel
                runId={msg.runId}
                userId={userId}
                latencyTotal={msg.latency?.t_total ?? 0}
              />
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-bubble aac_user loading">
            <span className="chat-role">AAC User</span>
            <p>Generating...</p>
          </div>
        )}
        {turnaroundLoading && (
          <div className="chat-bubble aac_user loading">
            <span className="chat-role">AAC User</span>
            <p>↻ Rephrasing...</p>
          </div>
        )}
        <div ref={bottomRef} />
      </div>
      <div className="chat-input-row">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && handleSend()}
          placeholder={backendReady ? "Type as the communication partner..." : "Waiting for backend to load models..."}
          disabled={!userId || loading || !backendReady}
        />
        <button onClick={handleSend} disabled={!userId || loading || !backendReady || !input.trim()}>
          Send
        </button>
        <button
          type="button"
          className="turnaround-btn"
          onClick={() => handleTurnaround("manual")}
          disabled={!canTurnaround}
          title="Re-plan the last response (also triggered by a head shake / sharp nod)"
        >
          ↻ Not quite right
        </button>
      </div>
    </div>
  );
}
