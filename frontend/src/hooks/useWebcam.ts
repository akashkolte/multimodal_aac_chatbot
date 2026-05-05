import { useRef, useEffect, useState } from "react";

interface UseWebcamOptions {
  enabled: boolean;
  onFrame?: (video: HTMLVideoElement, timestamp: number) => void;
  processEveryN?: number;
}

export function useWebcam({
  enabled,
  onFrame,
  processEveryN = 3,
}: UseWebcamOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameCount = useRef(0);
  const rafId = useRef(0);
  const onFrameRef = useRef(onFrame);
  const [error, setError] = useState<string | null>(null);
  const [active, setActive] = useState(false);

  useEffect(() => {
    onFrameRef.current = onFrame;
  }, [onFrame]);

  function teardown() {
    if (rafId.current) cancelAnimationFrame(rafId.current);
    rafId.current = 0;
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  }

  useEffect(() => {
    if (!enabled) {
      teardown();
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setActive(false);
      return;
    }

    let cancelled = false;

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 640, height: 480 },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        if (cancelled) {
          teardown();
          return;
        }
        setActive(true);
        setError(null);

        function loop(timestamp: number) {
          if (cancelled) return;
          frameCount.current++;
          if (
            frameCount.current % processEveryN === 0 &&
            videoRef.current &&
            onFrameRef.current
          ) {
            onFrameRef.current(videoRef.current, timestamp);
          }
          rafId.current = requestAnimationFrame(loop);
        }
        rafId.current = requestAnimationFrame(loop);
      } catch (e) {
        if (!cancelled) {
          setError(
            e instanceof Error ? e.message : "Webcam access denied"
          );
          setActive(false);
        }
      }
    }

    start();

    return () => {
      cancelled = true;
      teardown();
      setActive(false);
    };
  }, [enabled, processEveryN]);

  return { videoRef, active, error };
}
