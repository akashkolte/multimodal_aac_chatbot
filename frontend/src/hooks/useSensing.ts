import { useRef, useCallback, useState, useEffect } from "react";
import {
  FaceLandmarker,
  GestureRecognizer,
  FilesetResolver,
} from "@mediapipe/tasks-vision";
import type { SensingState } from "../types";
import {
  classifyAffect,
  mapGestureLabel,
  GazeTracker,
  AirWriter,
  HeadPoseTracker,
  Calibrator,
  worldGazeXY,
  extractAngles,
  faceBboxSize,
} from "../lib/sensing";
import { recognizeInkStroke } from "../lib/inkRecognizer";

const GESTURE_DEBOUNCE_MS = 100;
const AFFECT_DEBOUNCE_MS  = 270;

const AIRWRITING_ENABLED  = import.meta.env.VITE_AIRWRITING_ENABLED !== "false";
const GAZE_ENABLED        = import.meta.env.VITE_GAZE_ENABLED !== "false";
const CALIBRATION_ENABLED = import.meta.env.VITE_CALIBRATION_ENABLED !== "false";

export function useSensing() {
  const faceLandmarkerRef = useRef<FaceLandmarker | null>(null);
  const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);
  const calibratorRef = useRef(new Calibrator());
  const gazeTrackerRef = useRef(new GazeTracker());
  const airWriterRef = useRef(new AirWriter());
  const inkBusyRef = useRef(false);
  const headTrackerRef = useRef(new HeadPoseTracker());
  const headDebugRef = useRef({ pitch: 0, yaw: 0, roll: 0, crossings: 0 });
  const gestureCountRef = useRef<{ tag: SensingState["gestureTag"]; since: number }>({ tag: null, since: 0 });
  const affectCountRef = useRef<{ affect: SensingState["affect"]; since: number }>({ affect: null, since: 0 });
  const initingRef = useRef(false);

  const [ready, setReady] = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [sensing, setSensing] = useState<SensingState>({
    affect: null,
    gestureTag: null,
    gazeZone: null,
    gazeBucket: null,
    airWrittenText: "",
    airWritingActive: false,
    headSignal: null,
    headDebug: { pitch: 0, yaw: 0, roll: 0, crossings: 0 },
  });

  useEffect(() => {
    return () => {
      faceLandmarkerRef.current?.close();
      gestureRecognizerRef.current?.close();
      faceLandmarkerRef.current = null;
      gestureRecognizerRef.current = null;
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
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        }
      );
      gestureRecognizerRef.current = await GestureRecognizer.createFromOptions(
        vision,
        {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
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

  const startCalibration = useCallback(() => {
    if (!CALIBRATION_ENABLED) {
      setIsCalibrated(true);
      return;
    }
    calibratorRef.current.start();
    setIsCalibrating(true);
    setIsCalibrated(false);
    setCalibrationProgress(0);
    // Reset the per-detector state so post-calibration baselines aren't
    // mixed with stale pre-calibration history.
    gazeTrackerRef.current.reset();
    headTrackerRef.current.reset();
    gestureCountRef.current = { tag: null, since: 0 };
    affectCountRef.current = { affect: null, since: 0 };
  }, []);

  const cancelCalibration = useCallback(() => {
    calibratorRef.current.cancel();
    setIsCalibrating(false);
    setIsCalibrated(false);
    setCalibrationProgress(0);
  }, []);

  const processFrame = useCallback(
    (video: HTMLVideoElement, timestamp: number) => {
      const faceLandmarker = faceLandmarkerRef.current;
      const gestureRecognizer = gestureRecognizerRef.current;
      if (!faceLandmarker || !gestureRecognizer) return;

      const calibrator = calibratorRef.current;
      const calibrating = calibrator.isActive;
      const baseline = calibrator.getBaseline();

      let affect: SensingState["affect"] = null;
      let gazeBucket: SensingState["gazeBucket"] = null;
      let headSignal: SensingState["headSignal"] = null;

      const faceResult = faceLandmarker.detectForVideo(video, timestamp);
      if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
        const matrix = faceResult.facialTransformationMatrixes?.[0] ?? null;
        const landmarks = faceResult.faceLandmarks[0];

        const bs: Record<string, number> = {};
        if (faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
          for (const cat of faceResult.faceBlendshapes[0].categories) {
            bs[cat.categoryName] = cat.score;
          }
        }

        if (calibrating) {
          calibrator.addSample({
            blendshapes: bs,
            gaze: matrix ? worldGazeXY(matrix, bs) : null,
            head: matrix ? extractAngles(matrix.data) : null,
            faceBboxSize: faceBboxSize(landmarks),
          });
          setCalibrationProgress(Math.round(calibrator.progress * 100) / 100);
          if (calibrator.isReady) {
            setIsCalibrating(false);
            setIsCalibrated(true);
            setCalibrationProgress(1);
          }
          return;
        }

        affect = classifyAffect(bs, baseline);

        if (GAZE_ENABLED) {
          gazeBucket = gazeTrackerRef.current.process(matrix, bs, baseline);
        }

        if (matrix) {
          headSignal = headTrackerRef.current.process(matrix, baseline);
          headDebugRef.current = headTrackerRef.current.debug;
        }
      } else if (calibrating) {
        setCalibrationProgress(Math.round(calibrator.progress * 100) / 100);
        return;
      }

      let gestureTag: SensingState["gestureTag"] = null;

      const gestureResult = gestureRecognizer.recognizeForVideo(video, timestamp);
      if (gestureResult.gestures && gestureResult.gestures.length > 0) {
        const topGesture = gestureResult.gestures[0][0];
        gestureTag = mapGestureLabel(topGesture.categoryName);
        if (AIRWRITING_ENABLED) {
          const handLandmarks = gestureResult.landmarks[0];
          airWriterRef.current.processHandLandmarks(
            handLandmarks,
            video.videoWidth,
            video.videoHeight
          );
        }
      } else if (AIRWRITING_ENABLED) {
        airWriterRef.current.noHand();
      }

      if (AIRWRITING_ENABLED) {
        const completedStroke = airWriterRef.current.getCompletedStroke();
        if (completedStroke && !inkBusyRef.current) {
          inkBusyRef.current = true;
          recognizeInkStroke(completedStroke).then((text) => {
            inkBusyRef.current = false;
            if (text) {
              setSensing((prev) => ({
                ...prev,
                airWrittenText: prev.airWrittenText + text,
              }));
            }
          });
        }
      }

      const now = performance.now();
      if (gestureTag !== gestureCountRef.current.tag) {
        gestureCountRef.current = { tag: gestureTag, since: now };
      }
      const stableGesture =
        now - gestureCountRef.current.since >= GESTURE_DEBOUNCE_MS
          ? gestureTag
          : null;

      if (affect !== affectCountRef.current.affect) {
        affectCountRef.current = { affect, since: now };
      }
      const stableAffect =
        now - affectCountRef.current.since >= AFFECT_DEBOUNCE_MS
          ? affect
          : null;

      const activeZone = GAZE_ENABLED ? gazeTrackerRef.current.activeZone : null;
      const airWritingActive = airWriterRef.current.strokeActive;
      const headDebug = headDebugRef.current;

      setSensing((prev) => {
        const nextAffect = stableAffect ?? prev.affect;
        const nextGazeBucket = gazeBucket ?? prev.gazeBucket;
        const nextHeadSignal = headSignal ?? prev.headSignal;
        const debugChanged =
          headDebug.pitch !== prev.headDebug.pitch ||
          headDebug.yaw !== prev.headDebug.yaw ||
          headDebug.roll !== prev.headDebug.roll ||
          headDebug.crossings !== prev.headDebug.crossings;
        if (
          nextAffect === prev.affect &&
          stableGesture === prev.gestureTag &&
          activeZone === prev.gazeZone &&
          nextGazeBucket === prev.gazeBucket &&
          airWritingActive === prev.airWritingActive &&
          nextHeadSignal === prev.headSignal &&
          !debugChanged
        ) {
          return prev;
        }
        return {
          ...prev,
          affect: nextAffect,
          gestureTag: stableGesture,
          gazeZone: activeZone,
          gazeBucket: nextGazeBucket,
          airWritingActive,
          headSignal: nextHeadSignal,
          headDebug: debugChanged ? headDebug : prev.headDebug,
        };
      });
    },
    []
  );

  const clearAirWrittenText = useCallback(() => {
    setSensing((prev) => ({ ...prev, airWrittenText: "" }));
  }, []);

  const clearHeadSignal = useCallback(() => {
    setSensing((prev) => ({ ...prev, headSignal: null }));
  }, []);

  const resetCalibration = useCallback(() => {
    gestureCountRef.current = { tag: null, since: 0 };
    affectCountRef.current = { affect: null, since: 0 };
    gazeTrackerRef.current.reset();
    headTrackerRef.current.reset();
    calibratorRef.current.cancel();
    setIsCalibrating(false);
    setIsCalibrated(false);
    setCalibrationProgress(0);
    setSensing({
      affect: null,
      gestureTag: null,
      gazeZone: null,
      gazeBucket: null,
      airWrittenText: "",
      airWritingActive: false,
      headSignal: null,
      headDebug: { pitch: 0, yaw: 0, roll: 0, crossings: 0 },
    });
  }, []);

  return {
    sensing,
    ready,
    initError,
    isCalibrating,
    isCalibrated,
    calibrationProgress,
    init,
    processFrame,
    startCalibration,
    cancelCalibration,
    clearAirWrittenText,
    clearHeadSignal,
    resetCalibration,
  };
}
