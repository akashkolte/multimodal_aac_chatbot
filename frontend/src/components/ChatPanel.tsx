import { useState, useRef, useEffect, useCallback } from "react";
import type {
  Affect,
  Candidate,
  ChatMessage,
  LatencyLog,
  SensingState,
} from "../types";
import {
  pollEvals,
  sendPick,
  sendTurnaround,
  streamChat,
  streamRegenerate,
} from "../lib/api";
import { EvalPanel } from "./EvalPanel";

const STRATEGY_LABELS: Record<string, string> = {
  broad: "broad — all memories",
  focused: "focused — top memory",
  serendipitous: "serendipitous — other memory",
  side_index: "like last time",
  present_good: "feeling good",
  present_fine: "doing okay",
  present_rough: "not great",
  pending: "",
};

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

// Batches token deltas per (msgIdx, candIdx) and flushes them in a single
// setState call per animation frame. Streaming tokens at 30-60/s × 3 candidates
// otherwise causes a rerender per token. Non-token events (start/done/complete)
// flush the pending deltas first to preserve ordering.
//
// INVARIANT: keys are message indices into the messages[] array. Callers must
// ensure no message is inserted *before* a streaming message for the duration
// of its stream — appending to the end is fine, mid-list insert is not. Today
// every path appends to the end; if that changes, switch to a stable message
// id (e.g. the placeholder's runId or a freshly-minted uuid).
function useTokenBatcher(
  setMessages: React.Dispatch<React.SetStateAction<ChatMessage[]>>,
) {
  // Lazy-init refs to avoid allocating a fresh Map on every render.
  const pending = useRef<Map<number, Map<number, string>> | null>(null);
  if (pending.current === null) pending.current = new Map();
  const rafId = useRef<number | null>(null);

  const flush = useCallback(() => {
    rafId.current = null;
    const batch = pending.current;
    if (!batch || batch.size === 0) return;
    pending.current = new Map();
    setMessages((prev) =>
      prev.map((m, i) => {
        const perCand = batch.get(i);
        if (!perCand) return m;
        const cands = [...(m.candidates ?? [])];
        for (const [ci, delta] of perCand) {
          if (cands[ci]) {
            cands[ci] = { ...cands[ci], text: cands[ci].text + delta };
          }
        }
        return { ...m, candidates: cands };
      }),
    );
  }, [setMessages]);

  const queueToken = useCallback(
    (msgIdx: number, candIdx: number, delta: string) => {
      const batch = pending.current!;
      let perMsg = batch.get(msgIdx);
      if (!perMsg) {
        perMsg = new Map();
        batch.set(msgIdx, perMsg);
      }
      perMsg.set(candIdx, (perMsg.get(candIdx) ?? "") + delta);
      if (rafId.current === null) {
        rafId.current = window.requestAnimationFrame(flush);
      }
    },
    [flush],
  );

  const flushNow = useCallback(() => {
    if (rafId.current !== null) {
      window.cancelAnimationFrame(rafId.current);
      rafId.current = null;
    }
    flush();
  }, [flush]);

  // Cancel any pending rAF on unmount — otherwise a persona switch mid-stream
  // leaves a scheduled flush that calls setMessages against the new state.
  useEffect(() => {
    return () => {
      if (rafId.current !== null) {
        window.cancelAnimationFrame(rafId.current);
        rafId.current = null;
      }
      pending.current = null;
    };
  }, []);

  return { queueToken, flushNow };
}

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
  const [regenerateLoading, setRegenerateLoading] = useState(false);
  const { queueToken, flushNow } = useTokenBatcher(setMessages);
  const bottomRef = useRef<HTMLDivElement>(null);
  const lastResponseTsRef = useRef<number>(0);
  const lastTurnIdRef = useRef<number | null>(null);
  // turn_id of the most recent turn that was already turned around — guards
  // against the new turnaround bubble's own head-signal re-firing turnaround
  // on itself.
  const turnaroundConsumedTurnRef = useRef<number | null>(null);
  const evalPollAbortsRef = useRef<Set<AbortController>>(new Set());

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Reset per-turn state when the persona changes (parent clears `messages`
  // and resets the backend session — the frontend turn counter must follow).
  useEffect(() => {
    lastTurnIdRef.current = null;
    turnaroundConsumedTurnRef.current = null;
    lastResponseTsRef.current = 0;
    evalPollAbortsRef.current.forEach((ac) => ac.abort());
    evalPollAbortsRef.current.clear();
  }, [userId]);

  useEffect(() => {
    const active = evalPollAbortsRef.current;
    return () => {
      active.forEach((ac) => ac.abort());
      active.clear();
    };
  }, []);

  const startEvalPolling = useCallback(
    (runId: string | null | undefined) => {
      if (!runId) return;
      const ac = new AbortController();
      evalPollAbortsRef.current.add(ac);
      void pollEvals(runId, { signal: ac.signal })
        .then((scores) => {
          if (ac.signal.aborted || !scores) return;
          setMessages((prev) =>
            prev.map((m) =>
              m.runId === runId ? { ...m, evalScores: scores } : m
            )
          );
        })
        .finally(() => {
          evalPollAbortsRef.current.delete(ac);
        });
    },
    [setMessages]
  );

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
              next[i] = { ...next[i], rephrased: true, picked: true };
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
            evalScores: null,
            isTurnaround: true,
            candidates: res.candidates ?? [],
            picked: true,
          });
          return next;
        });
        onLatency(res.latency);
        startEvalPolling(res.run_id);
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
      startEvalPolling,
    ]
  );

  const handleRegenerate = useCallback(
    async (msgIdx: number) => {
      if (!userId || !backendReady || regenerateLoading || loading) return;
      const msg = messages[msgIdx];
      if (!msg || !msg.candidates || msg.picked || msg.turnId === undefined) return;

      const currentRound = msg.candidates;
      const priorRounds = msg.rejectedRounds ?? [];
      const rejected_texts = [
        ...priorRounds.flat().map((c) => c.text),
        ...currentRound.map((c) => c.text),
      ];

      setRegenerateLoading(true);

      // Move the current round into rejectedRounds + clear candidates so the
      // UI shows empty-card placeholders while streams fill in.
      setMessages((prev) =>
        prev.map((m, i) =>
          i === msgIdx
            ? {
                ...m,
                candidates: [],
                rejectedRounds: [...priorRounds, currentRound],
                picked: false,
              }
            : m,
        ),
      );

      const updateMsg = (
        updater: (m: ChatMessage) => ChatMessage,
      ) => {
        setMessages((prev) =>
          prev.map((m, i) => (i === msgIdx ? updater(m) : m)),
        );
      };

      try {
        await streamRegenerate(
          {
            user_id: userId,
            turn_id: msg.turnId,
            rejected_texts,
          },
          (evt) => {
            if (evt.type === "token") {
              queueToken(msgIdx, evt.idx, evt.delta);
              return;
            }
            flushNow();
            if (evt.type === "candidate_start") {
              updateMsg((m) => {
                const cands = [...(m.candidates ?? [])];
                while (cands.length <= evt.idx) {
                  cands.push({
                    text: "",
                    strategy: "pending",
                    grounded_buckets: [],
                  });
                }
                cands[evt.idx] = {
                  text: "",
                  strategy: evt.strategy,
                  grounded_buckets: evt.grounded_buckets,
                };
                return { ...m, candidates: cands };
              });
            } else if (evt.type === "candidate_done") {
              updateMsg((m) => {
                const cands = [...(m.candidates ?? [])];
                if (cands[evt.idx]) {
                  cands[evt.idx] = { ...cands[evt.idx], text: evt.text };
                }
                return { ...m, candidates: cands };
              });
            } else if (evt.type === "complete") {
              const res = evt.response;
              lastTurnIdRef.current = res.turn_id;
              updateMsg((m) => ({
                ...m,
                content: res.response,
                latency: res.latency,
                affect: res.affect,
                runId: res.run_id,
                turnId: res.turn_id,
                evalScores: null,
                candidates: res.candidates ?? m.candidates ?? [],
                picked: false,
              }));
              onLatency(res.latency);
              startEvalPolling(res.run_id);
            }
          },
        );
      } catch (e) {
        flushNow();
        console.warn("streamRegenerate failed", e);
      } finally {
        setRegenerateLoading(false);
      }
    },
    [
      userId,
      backendReady,
      regenerateLoading,
      loading,
      messages,
      setMessages,
      queueToken,
      flushNow,
      onLatency,
      startEvalPolling,
    ]
  );

  useEffect(() => {
    if (
      sensing.headSignal !== "HEAD_NOD_DISSATISFIED" &&
      sensing.headSignal !== "HEAD_SHAKE"
    ) {
      return;
    }

    // If the most recent AAC message has an open picker, head-signal means
    // "regenerate" — the user hasn't committed, so there's nothing to
    // "rephrase" yet.
    let openPickerIdx = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      const m = messages[i];
      if (m.role !== "aac_user") continue;
      if (!m.picked && (m.candidates?.length ?? 0) > 1) openPickerIdx = i;
      break;
    }
    if (openPickerIdx !== -1) {
      handleRegenerate(openPickerIdx);
      onHeadSignalConsumed();
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
  }, [
    sensing.headSignal,
    handleTurnaround,
    handleRegenerate,
    onHeadSignalConsumed,
    messages,
  ]);

  async function handleSend() {
    if (!input.trim() || !userId || !backendReady || loading) return;

    const query = input.trim();
    setInput("");
    setLoading(true);

    const airText = sensing.airWrittenText || null;

    // Push the partner bubble, and a placeholder AAC message we'll fill in
    // progressively. We need the placeholder's index to target updates — use
    // a ref captured from the setter so we don't rely on stale state.
    let placeholderIdx = -1;
    setMessages((prev) => {
      const next = [
        ...prev,
        { role: "partner" as const, content: query },
        {
          role: "aac_user" as const,
          content: "",
          candidates: [] as Candidate[],
          picked: false,
        },
      ];
      placeholderIdx = next.length - 1;
      return next;
    });

    const updatePlaceholder = (
      updater: (m: ChatMessage) => ChatMessage,
    ) => {
      setMessages((prev) =>
        prev.map((m, i) => (i === placeholderIdx ? updater(m) : m)),
      );
    };

    try {
      await streamChat(
        {
          user_id: userId,
          query,
          affect_override: affectOverride ?? sensing.affect,
          gesture_tag: sensing.gestureTag,
          gaze_bucket: sensing.gazeBucket,
          air_written_text: airText,
          head_signal: sensing.headSignal,
        },
        (evt) => {
          if (evt.type === "token") {
            queueToken(placeholderIdx, evt.idx, evt.delta);
            return;
          }
          // Any non-token event must see the latest text — flush the queue first.
          flushNow();
          if (evt.type === "candidate_start") {
            updatePlaceholder((m) => {
              const cands = [...(m.candidates ?? [])];
              while (cands.length <= evt.idx) {
                cands.push({
                  text: "",
                  strategy: "pending",
                  grounded_buckets: [],
                });
              }
              cands[evt.idx] = {
                text: "",
                strategy: evt.strategy,
                grounded_buckets: evt.grounded_buckets,
              };
              return { ...m, candidates: cands };
            });
          } else if (evt.type === "candidate_done") {
            updatePlaceholder((m) => {
              const cands = [...(m.candidates ?? [])];
              if (cands[evt.idx]) {
                cands[evt.idx] = { ...cands[evt.idx], text: evt.text };
              }
              return { ...m, candidates: cands };
            });
          } else if (evt.type === "complete") {
            const res = evt.response;
            lastTurnIdRef.current = res.turn_id;
            updatePlaceholder((m) => ({
              ...m,
              content: res.response,
              latency: res.latency,
              affect: res.affect,
              runId: res.run_id,
              turnId: res.turn_id,
              evalScores: null,
              candidates: res.candidates ?? m.candidates ?? [],
              picked: (res.candidates ?? []).length <= 1,
            }));
            onLatency(res.latency);
            lastResponseTsRef.current = performance.now();
            startEvalPolling(res.run_id);
          }
        },
      );
    } catch (e) {
      flushNow();
      updatePlaceholder((m) => ({
        ...m,
        content: `Error: ${e instanceof Error ? e.message : "request failed"}`,
      }));
    } finally {
      if (airText) onAirTextConsumed();
      setLoading(false);
    }
  }

  const handlePick = useCallback(
    async (msgIdx: number, candIdx: number) => {
      const msg = messages[msgIdx];
      if (!msg || !msg.candidates || !msg.runId || !userId) return;
      if (msg.picked) return;
      const picked = msg.candidates[candIdx];
      if (!picked) return;

      setMessages((prev) =>
        prev.map((m, i) =>
          i === msgIdx
            ? {
                ...m,
                content: picked.text,
                picked: true,
                pickedIdx: candIdx,
              }
            : m
        )
      );
      try {
        await sendPick({
          run_id: msg.runId,
          user_id: userId,
          picked_idx: candIdx,
        });
      } catch (e) {
        console.warn("sendPick failed", e);
      }
    },
    [messages, setMessages, userId]
  );

  return (
    <div className="chat-panel">
      <div className="chat-header">
        Talking as: {personaName || "select a persona"}
      </div>
      <div className="chat-messages">
        {messages.map((msg, i) => {
          const hasRegenerated = (msg.rejectedRounds?.length ?? 0) > 0;
          const showPicker =
            msg.role === "aac_user" &&
            !msg.picked &&
            !!msg.candidates &&
            (msg.candidates.length > 1 || hasRegenerated);

          if (showPicker) {
            const priorRounds = msg.rejectedRounds ?? [];
            return (
              <div key={i} className="chat-bubble aac_user picker">
                <span className="chat-role">
                  AAC User
                  <span className="badge badge-picker">
                    pick one ({msg.candidates!.length} options)
                  </span>
                </span>
                {priorRounds.map((round, ri) => (
                  <div key={`r${ri}`} className="candidate-list rejected-round">
                    <div className="rejected-round-label">
                      rejected round {ri + 1}
                    </div>
                    {round.map((cand, ci) => (
                      <div key={ci} className="candidate-card rejected">
                        <div className="candidate-strategy">
                          {STRATEGY_LABELS[cand.strategy] ?? cand.strategy}
                        </div>
                        <div className="candidate-text">{cand.text}</div>
                      </div>
                    ))}
                  </div>
                ))}
                <div className="candidate-list">
                  {msg.candidates!.map((cand, ci) => (
                    <button
                      key={ci}
                      type="button"
                      className="candidate-card"
                      onClick={() => handlePick(i, ci)}
                      disabled={regenerateLoading}
                      title="Click to send this one"
                    >
                      <div className="candidate-strategy">
                        {STRATEGY_LABELS[cand.strategy] ?? cand.strategy}
                      </div>
                      <div className="candidate-text">{cand.text}</div>
                    </button>
                  ))}
                  <button
                    type="button"
                    className="candidate-card try-again"
                    onClick={() => handleRegenerate(i)}
                    disabled={regenerateLoading}
                    title="None of these fit — generate fresh options"
                  >
                    <div className="candidate-strategy">try again</div>
                    <div className="candidate-text">
                      {regenerateLoading
                        ? "Regenerating…"
                        : "↻ None of these fit — try different angles"}
                    </div>
                  </button>
                </div>
              </div>
            );
          }

          return (
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
                {msg.picked && msg.pickedIdx !== undefined && msg.candidates && msg.candidates[msg.pickedIdx] && (
                  <span className="badge badge-picked">
                    ✓ {STRATEGY_LABELS[msg.candidates[msg.pickedIdx].strategy] ?? msg.candidates[msg.pickedIdx].strategy}
                  </span>
                )}
              </span>
              <p>{msg.content}</p>
              {msg.role === "aac_user" && msg.runId && userId && (
                <EvalPanel
                  runId={msg.runId}
                  userId={userId}
                  latencyTotal={msg.latency?.t_total ?? 0}
                  evalScores={msg.evalScores ?? null}
                />
              )}
            </div>
          );
        })}
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
      </div>
    </div>
  );
}
