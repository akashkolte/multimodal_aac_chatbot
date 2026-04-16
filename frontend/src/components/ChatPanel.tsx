import { useState, useRef, useEffect } from "react";
import type { ChatMessage, SensingState, Affect, LatencyLog } from "../types";
import { sendChat } from "../lib/api";

interface Props {
  userId: string | null;
  personaName: string;
  sensing: SensingState;
  affectOverride: Affect | null;
  onAirTextConsumed: () => void;
  messages: ChatMessage[];
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>;
  onLatency: (latency: LatencyLog) => void;
  backendReady: boolean;
}

export function ChatPanel({
  userId,
  personaName,
  sensing,
  affectOverride,
  onAirTextConsumed,
  messages,
  setMessages,
  onLatency,
  backendReady,
}: Props) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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
      });

      setMessages((prev) => [
        ...prev,
        {
          role: "aac_user",
          content: res.response,
          latency: res.latency,
          affect: res.affect,
        },
      ]);
      onLatency(res.latency);
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

  return (
    <div className="chat-panel">
      <div className="chat-header">
        Talking as: {personaName || "select a persona"}
      </div>
      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble ${msg.role}`}>
            <span className="chat-role">
              {msg.role === "partner" ? "Partner" : "AAC User"}
            </span>
            <p>{msg.content}</p>
          </div>
        ))}
        {loading && (
          <div className="chat-bubble aac_user loading">
            <span className="chat-role">AAC User</span>
            <p>Generating...</p>
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
      </div>
    </div>
  );
}
