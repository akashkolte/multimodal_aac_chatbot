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
} from "../lib/sensing";
import { recognizeInkStroke } from "../lib/inkRecognizer";

const GESTURE_DEBOUNCE_FRAMES = 3;
const AFFECT_DEBOUNCE_FRAMES  = 8;

const AIRWRITING_ENABLED = import.meta.env.VITE_AIRWRITING_ENABLED !== "false";
const GAZE_ENABLED       = import.meta.env.VITE_GAZE_ENABLED !== "false";

export function useSensing() {
  const faceLandmarkerRef    = useRef<FaceLandmarker | null>(null);
  const gestureRecognizerRef = useRef<GestureRecognizer | null>(null);
  const gazeTrackerRef  = useRef(new GazeTracker());
  const airWriterRef    = useRef(new AirWriter());
  const inkBusyRef      = useRef(false);
  const headTrackerRef  = useRef(new HeadPoseTracker());
  const headDebugRef    = useRef({ pitch: 0, yaw: 0, roll: 0, crossings: 0 });
  const gestureCountRef = useRef<{ tag: SensingState["gestureTag"]; count: number }>({ tag: null, count: 0 });
  const affectCountRef  = useRef<{ affect: SensingState["affect"]; count: number }>({ affect: null, count: 0 });
  const initingRef      = useRef(false);

  const [ready, setReady]       = useState(false);
  const [initError, setInitError] = useState<string | null>(null);
  const [sensing, setSensing]   = useState<SensingState>({
    affect: null,
    gestureTag: null,
    gazeZone: null,
    gazeBucket: null,
    airWrittenText: "",
    airWritingActive: false,
    headSignal: null,
    headCalibrated: false,
    headDebug: { pitch: 0, yaw: 0, roll: 0, crossings: 0 },
  });

  useEffect(() => {
    return () => {
      faceLandmarkerRef.current?.close();
      gestureRecognizerRef.current?.close();
      faceLandmarkerRef.current    = null;
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
      faceLandmarkerRef.current = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
      gestureRecognizerRef.current = await GestureRecognizer.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numHands: 1,
      });
      setReady(true);
      return true;
    } catch (e) {
      setInitError(e instanceof Error ? e.message : "Failed to load MediaPipe models");
      return false;
    } finally {
      initingRef.current = false;
    }
  }, []);

  const processFrame = useCallback(
    (video: HTMLVideoElement, timestamp: number) => {
      const faceLandmarker    = faceLandmarkerRef.current;
      const gestureRecognizer = gestureRecognizerRef.current;
      if (!faceLandmarker || !gestureRecognizer) return;

      let affect: SensingState["affect"]       = null;
      let gazeBucket: SensingState["gazeBucket"] = null;
      let headSignal: SensingState["headSignal"] = null;

      const faceResult = faceLandmarker.detectForVideo(video, timestamp);
      if (faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0) {
        const matrix = faceResult.facialTransformationMatrixes?.[0] ?? null;

        const bs: Record<string, number> = {};
        if (faceResult.faceBlendshapes && faceResult.faceBlendshapes.length > 0) {
          for (const cat of faceResult.faceBlendshapes[0].categories) {
            bs[cat.categoryName] = cat.score;
          }
          affect = classifyAffect(bs);
        }

        if (GAZE_ENABLED) {
          gazeBucket = gazeTrackerRef.current.process(matrix, bs);
        }

        if (matrix) {
          headSignal = headTrackerRef.current.process(matrix);
          headDebugRef.current = headTrackerRef.current.debug;
        }
      }

      // Always call recognizeForVideo every frame — VIDEO-mode models maintain
      // internal temporal state and produce stale results if frames are skipped.
      const gestureResult = gestureRecognizer.recognizeForVideo(video, timestamp);

      let gestureTag: SensingState["gestureTag"] = null;

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
              setSensing((prev) => ({ ...prev, airWrittenText: prev.airWrittenText + text }));
            }
          });
        }
      }

      if (gestureTag === gestureCountRef.current.tag) {
        gestureCountRef.current.count++;
      } else {
        gestureCountRef.current = { tag: gestureTag, count: 1 };
      }
      const stableGesture = gestureCountRef.current.count >= GESTURE_DEBOUNCE_FRAMES
        ? gestureTag
        : null;

      if (affect === affectCountRef.current.affect) {
        affectCountRef.current.count++;
      } else {
        affectCountRef.current = { affect, count: 1 };
      }
      const stableAffect = affectCountRef.current.count >= AFFECT_DEBOUNCE_FRAMES
        ? affect
        : null;

      setSensing((prev) => ({
        affect: stableAffect ?? prev.affect,
        gestureTag: stableGesture,
        gazeZone: GAZE_ENABLED ? gazeTrackerRef.current.activeZone : null,
        gazeBucket: gazeBucket ?? prev.gazeBucket,
        airWrittenText: prev.airWrittenText,
        airWritingActive: airWriterRef.current.strokeActive,
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

  const resetCalibration = useCallback(() => {
    gestureCountRef.current = { tag: null, count: 0 };
    affectCountRef.current  = { affect: null, count: 0 };
    gazeTrackerRef.current.reset();
    headTrackerRef.current.reset();
    setSensing({
      affect: null,
      gestureTag: null,
      gazeZone: null,
      gazeBucket: null,
      airWrittenText: "",
      airWritingActive: false,
      headSignal: null,
      headCalibrated: false,
      headDebug: { pitch: 0, yaw: 0, roll: 0, crossings: 0 },
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
    resetCalibration,
  };
}
