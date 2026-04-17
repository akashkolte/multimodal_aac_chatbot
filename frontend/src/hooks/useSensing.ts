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
  HeadPoseTracker,
} from "../lib/sensing";
import { DEFAULT_AIR_TEMPLATES } from "../lib/airTemplates";

const EMA_ALPHA = 0.3;

export function useSensing() {
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const handLandmarkerRef = useRef<HandLandmarker | null>(null);
  const gazeTrackerRef = useRef(new GazeTracker());
  const airWriterRef = useRef(new AirWriter(DEFAULT_AIR_TEMPLATES));
  const headTrackerRef = useRef(new HeadPoseTracker());
  const calibratePendingRef = useRef(false);
  const headDebugRef = useRef({ dx: 0, dy: 0, maxAbsDx: 0, maxAbsDy: 0, crossings: 0 });
  const neutralLCPRef = useRef<number | null>(null);
  const calibBufferRef = useRef<number[]>([]);
  const smoothedRef = useRef({ MAR: 0, EAR: 0.3, BRI: -0.3, LCP: 0 });
  const initingRef = useRef(false);
  const [ready, setReady] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [sensing, setSensing] = useState<SensingState>({
    affect: null,
    gestureTag: null,
    gazeBucket: null,
    airWrittenText: "",
    headSignal: null,
    headCalibrated: false,
    headDebug: { dx: 0, dy: 0, maxAbsDx: 0, maxAbsDy: 0, crossings: 0 },
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
      let headSignal: SensingState["headSignal"] = null;

      const faceResult = faceLandmarker.detectForVideo(video, timestamp);
      if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
        const landmarks = faceResult.faceLandmarks[0];

        // Average the raw LCP (vertical corner pull, pre-offset) over ~30 frames
        // of the user's face before locking neutral. Single-frame calibration is
        // too noisy and tended to bake in a momentary smile as "neutral".
        // During calibration, affect stays null but gaze/head/gesture still flow.
        if (neutralLCPRef.current === null) {
          const raw0 = computeAffectVector(landmarks, 0);
          calibBufferRef.current.push(raw0.LCP);
          if (calibBufferRef.current.length >= 30) {
            const sum = calibBufferRef.current.reduce((a, b) => a + b, 0);
            neutralLCPRef.current = sum / calibBufferRef.current.length;
            calibBufferRef.current = [];
          }
        }

        if (calibratePendingRef.current) {
          headTrackerRef.current.calibrate(landmarks);
          calibratePendingRef.current = false;
        }

        if (neutralLCPRef.current !== null) {
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
        }

        gazeBucket = gazeTrackerRef.current.process(landmarks);
        headSignal = headTrackerRef.current.process(landmarks);
        headDebugRef.current = headTrackerRef.current.debug;
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
        headSignal: headSignal ?? prev.headSignal,
        headCalibrated: headTrackerRef.current.calibrated,
        headDebug: headDebugRef.current,
      }));
    },
    []
  );

  const clearAirWrittenText = useCallback(() => {
    setSensing((prev) => ({ ...prev, airWrittenText: "" }));
  }, []);

  const clearHeadSignal = useCallback(() => {
    setSensing((prev) => ({ ...prev, headSignal: null }));
  }, []);

  const calibrateHeadPose = useCallback(() => {
    calibratePendingRef.current = true;
    setSensing((prev) => ({ ...prev, headSignal: null }));
  }, []);

  const resetCalibration = useCallback(() => {
    neutralLCPRef.current = null;
    calibBufferRef.current = [];
    smoothedRef.current = { MAR: 0, EAR: 0.3, BRI: -0.3, LCP: 0 };
    gazeTrackerRef.current.reset();
    headTrackerRef.current.reset();
    setSensing({
      affect: null,
      gestureTag: null,
      gazeBucket: null,
      airWrittenText: "",
      headSignal: null,
      headCalibrated: false,
      headDebug: { dx: 0, dy: 0, maxAbsDx: 0, maxAbsDy: 0, crossings: 0 },
    });
  }, []);

  return {
    sensing,
    ready,
    initError,
    init,
    processFrame,
    clearAirWrittenText,
    clearHeadSignal,
    calibrateHeadPose,
    resetCalibration,
  };
}
