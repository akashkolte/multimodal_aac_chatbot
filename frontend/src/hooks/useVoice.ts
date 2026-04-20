import { useCallback, useEffect, useRef, useState } from "react";

// Thin wrapper around the Web Speech API. Chrome/Edge expose
// `webkitSpeechRecognition`; Safari/Firefox don't — we no-op gracefully.

type SRCtor = new () => SpeechRecognitionLike;

interface SpeechRecognitionLike {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((e: SpeechRecognitionEventLike) => void) | null;
  onerror: ((e: { error: string }) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

interface SpeechRecognitionEventLike {
  results: {
    length: number;
    [index: number]: {
      isFinal: boolean;
      length: number;
      [index: number]: { transcript: string; confidence: number };
    };
  };
}

function getRecognitionCtor(): SRCtor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as {
    SpeechRecognition?: SRCtor;
    webkitSpeechRecognition?: SRCtor;
  };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export interface VoiceCapture {
  transcript: string;
  confidence: number;
}

const WINDOW_MS = 3500;

export function useVoice() {
  const [supported] = useState(() => getRecognitionCtor() !== null);
  const [listening, setListening] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recRef = useRef<SpeechRecognitionLike | null>(null);
  const resolveRef = useRef<((v: VoiceCapture) => void) | null>(null);
  const rejectRef = useRef<((err: Error) => void) | null>(null);
  const bestRef = useRef<VoiceCapture>({ transcript: "", confidence: 0 });
  const timerRef = useRef<number | null>(null);

  const cleanup = useCallback(() => {
    if (timerRef.current !== null) {
      window.clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    const rec = recRef.current;
    if (rec) {
      try {
        rec.abort();
      } catch {
        // ignore — some browsers throw if already stopped
      }
      rec.onresult = null;
      rec.onerror = null;
      rec.onend = null;
    }
    recRef.current = null;
    resolveRef.current = null;
    rejectRef.current = null;
    setListening(false);
  }, []);

  // Unmount teardown: reject any in-flight promise so `await voice.capture()`
  // in a parent component doesn't hang forever when the tree unmounts mid-listen.
  useEffect(
    () => () => {
      const rj = rejectRef.current;
      cleanup();
      if (rj) rj(new Error("unmounted"));
    },
    [cleanup]
  );

  const capture = useCallback((): Promise<VoiceCapture> => {
    const Ctor = getRecognitionCtor();
    if (!Ctor) {
      return Promise.reject(new Error("Speech recognition not supported"));
    }
    if (recRef.current) {
      return Promise.reject(new Error("Already listening"));
    }

    return new Promise<VoiceCapture>((resolve, reject) => {
      const rec = new Ctor();
      rec.lang = navigator.language || "en-US";
      rec.continuous = false;
      rec.interimResults = false;
      rec.maxAlternatives = 1;

      bestRef.current = { transcript: "", confidence: 0 };
      resolveRef.current = resolve;
      rejectRef.current = reject;
      recRef.current = rec;
      setError(null);

      rec.onresult = (e) => {
        for (let i = 0; i < e.results.length; i++) {
          const res = e.results[i];
          if (!res.isFinal) continue;
          const alt = res[0];
          if (alt && alt.transcript.trim().length > 0) {
            if (alt.confidence > bestRef.current.confidence) {
              bestRef.current = {
                transcript: alt.transcript.trim(),
                confidence: alt.confidence,
              };
            }
          }
        }
      };

      rec.onerror = (e) => {
        const msg = e.error || "recognition error";
        setError(msg);
        const rj = rejectRef.current;
        cleanup();
        if (rj) rj(new Error(msg));
      };

      rec.onend = () => {
        const rs = resolveRef.current;
        const best = bestRef.current;
        cleanup();
        if (rs) rs(best);
      };

      try {
        rec.start();
        setListening(true);
        timerRef.current = window.setTimeout(() => {
          try {
            rec.stop();
          } catch {
            // onend will still fire
          }
        }, WINDOW_MS);
      } catch (err) {
        const msg = err instanceof Error ? err.message : "failed to start";
        setError(msg);
        cleanup();
        reject(new Error(msg));
      }
    });
  }, [cleanup]);

  const cancel = useCallback(() => {
    const rj = rejectRef.current;
    cleanup();
    if (rj) rj(new Error("cancelled"));
  }, [cleanup]);

  return { supported, listening, error, capture, cancel };
}
