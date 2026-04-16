import { useRef, useCallback, useState, useEffect } from "react";
import {
  FaceLandmarker,
  HandLandmarker,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import type { SensingState } from "../types";
import {
  computeAffectVector,
  classifyAffect,
  classifyGesture,
  GazeTracker,
  AirWriter,
} from "../lib/sensing";

const EMA_ALPHA = 0.3;

export function useSensing() {
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const gazeTrackerRef = useRef(new GazeTracker());
  const airWriterRef = useRef(new AirWriter());
  const neutralLCPRef = useRef<number | null>(null);
  const smoothedRef = useRef({ MAR: 0, EAR: 0.3, BRI: -0.3, LCP: 0 });
  const initingRef = useRef(false);
  const [ready, setReady] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [sensing, setSensing] = useState<SensingState>({
    affect: null,
    gestureTag: null,
    gazeBucket: null,
    airWrittenText: "",
  });

  // Cleanup MediaPipe resources on unmount
  useEffect(() => {
    return () => {
      faceLandmarkerRef.current?.close();
      handLandmarkerRef.current?.close();
      faceLandmarkerRef.current = null;
      handLandmarkerRef.current = null;
    };
  }, []);

  const init = useCallback(async (): Promise<boolean> => {
    if (faceLandmarkerRef.current || initingRef.current) return true;
    initingRef.current = true;
    try {
      const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
      );
      faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: false,
          outputFacialTransformationMatrixes: false,
        }
      );
      handLandmarkerRef.current = await HandLandmarker.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
          numHands: 1,
        }
      );
      setReady(true);
      return true;
    } catch (e) {
      setInitError(
        e instanceof Error ? e.message : "Failed to load MediaPipe models"
      );
      return false;
    } finally {
      initingRef.current = false;
    }
  }, []);

  const processFrame = useCallback(
    (video: HTMLVideoElement, timestamp: number) => {
      const faceLandmarker = faceLandmarkerRef.current;
      const handLandmarker = handLandmarkerRef.current;
      if (!faceLandmarker || !handLandmarker) return;

      let affect: SensingState["affect"] = null;
      let gazeBucket: SensingState["gazeBucket"] = null;

      const faceResult = faceLandmarker.detectForVideo(video, timestamp);
      if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
        const landmarks = faceResult.faceLandmarks[0];

        if (neutralLCPRef.current === null) {
          neutralLCPRef.current =
            (landmarks[61].x + landmarks[291].x) / 2;
        }

        const raw = computeAffectVector(landmarks, neutralLCPRef.current);

        const prev = smoothedRef.current;
        const smoothed = {
          MAR: EMA_ALPHA * raw.MAR + (1 - EMA_ALPHA) * prev.MAR,
          EAR: EMA_ALPHA * raw.EAR + (1 - EMA_ALPHA) * prev.EAR,
          BRI: EMA_ALPHA * raw.BRI + (1 - EMA_ALPHA) * prev.BRI,
          LCP: EMA_ALPHA * raw.LCP + (1 - EMA_ALPHA) * prev.LCP,
        };
        smoothedRef.current = smoothed;

        affect = classifyAffect(smoothed);
        gazeBucket = gazeTrackerRef.current.process(landmarks);
      }

      let gestureTag: SensingState["gestureTag"] = null;

      const handResult = handLandmarker.detectForVideo(video, timestamp);
      if (handResult.landmarks && handResult.landmarks.length > 0) {
        const handLandmarks = handResult.landmarks[0];
        gestureTag = classifyGesture(handLandmarks);
        airWriterRef.current.processHandLandmarks(
          handLandmarks,
          video.videoWidth,
          video.videoHeight
        );
      } else {
        airWriterRef.current.noHand();
      }

      const newAirText = airWriterRef.current.getText();

      setSensing((prev) => ({
        affect: affect ?? prev.affect,
        gestureTag: gestureTag ?? prev.gestureTag,
        gazeBucket: gazeBucket ?? prev.gazeBucket,
        airWrittenText: newAirText
          ? prev.airWrittenText + newAirText
          : prev.airWrittenText,
      }));
    },
    []
  );

  const clearAirWrittenText = useCallback(() => {
    setSensing((prev) => ({ ...prev, airWrittenText: "" }));
  }, []);

  const resetCalibration = useCallback(() => {
    neutralLCPRef.current = null;
    smoothedRef.current = { MAR: 0, EAR: 0.3, BRI: -0.3, LCP: 0 };
    gazeTrackerRef.current.reset();
    setSensing({ affect: null, gestureTag: null, gazeBucket: null, airWrittenText: "" });
  }, []);

  return { sensing, ready, initError, init, processFrame, clearAirWrittenText, resetCalibration };
}
