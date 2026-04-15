"""
L1 — Facial affect detection via MediaPipe 2D Face Mesh.

Extracts 4 geometric features from 478 landmarks at ~10 fps:
  MAR — Mouth Aspect Ratio     (surprise / speech attempt)
  EAR — Eye Aspect Ratio       (frustration / blink)
  BRI — Brow Raise Index       (surprise / questioning)
  LCP — Lip Corner Pull        (smile vs frown)

These form the affect vector fed into MobileNetV3-Small affect classifier,
which maps to one of 4 actionable states: HAPPY | FRUSTRATED | NEUTRAL | SURPRISED.

EMA smoothing (α=0.3) prevents transient expressions (sneezes, blinks)
from destabilising the detected state across turns.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from config.settings import settings
from pipeline.state import AffectState, AffectVector

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ── MediaPipe landmark indices (from proposal §5.2) ───────────────────────────

# MAR — mouth vertical / horizontal ratio
_MOUTH_TOP    = 13
_MOUTH_BOTTOM = 14
_MOUTH_LEFT   = 61
_MOUTH_RIGHT  = 291

# EAR — eye vertical / horizontal ratio (right eye)
_EYE_TOP    = 159
_EYE_BOTTOM = 145
_EYE_LEFT   = 33
_EYE_RIGHT  = 133

# BRI — brow vertical displacement relative to eye centre
_BROW_LEFT  = 70
_BROW_RIGHT = 300

# LCP — mouth corner horizontal displacement from neutral baseline
_CORNER_LEFT  = 61
_CORNER_RIGHT = 291


# ── Affect classes ────────────────────────────────────────────────────────────

AFFECT_CLASSES = ["HAPPY", "FRUSTRATED", "NEUTRAL", "SURPRISED"]


@dataclass
class AffectDetector:
    """
    Stateful detector that maintains EMA-smoothed affect across frames.
    Create one instance per session and call `process_frame` each frame.
    """
    _smoothed: AffectVector = field(default_factory=lambda: AffectVector(MAR=0.0, EAR=0.3, BRI=0.0, LCP=0.0))
    _neutral_lcp: float = 0.0          # calibrated at session start
    _calibrated: bool = False

    def __post_init__(self):
        if not _MP_AVAILABLE:
            raise ImportError("mediapipe is required: pip install mediapipe")
        if not _CV2_AVAILABLE:
            raise ImportError("opencv-python is required: pip install opencv-python")

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,       # enables iris landmarks (468-477)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, bgr_frame: np.ndarray) -> AffectState | None:
        """
        Process one BGR frame from OpenCV and return the current AffectState,
        or None if no face is detected.
        """
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        lm = result.multi_face_landmarks[0].landmark
        h, w = bgr_frame.shape[:2]

        def pt(idx):
            l = lm[idx]
            return np.array([l.x * w, l.y * h])

        raw = self._compute_features(pt)

        if not self._calibrated:
            self._neutral_lcp = raw["LCP"]
            self._calibrated = True

        raw["LCP"] = raw["LCP"] - self._neutral_lcp  # relative to neutral baseline

        alpha = settings.affect_ema_alpha
        smoothed = AffectVector(
            MAR=alpha * raw["MAR"] + (1 - alpha) * self._smoothed["MAR"],
            EAR=alpha * raw["EAR"] + (1 - alpha) * self._smoothed["EAR"],
            BRI=alpha * raw["BRI"] + (1 - alpha) * self._smoothed["BRI"],
            LCP=alpha * raw["LCP"] + (1 - alpha) * self._smoothed["LCP"],
        )
        self._smoothed = smoothed

        emotion = self._classify(smoothed)
        return AffectState(emotion=emotion, vector=raw, smoothed=smoothed)

    def _compute_features(self, pt) -> dict:
        # MAR
        mouth_v = np.linalg.norm(pt(_MOUTH_TOP) - pt(_MOUTH_BOTTOM))
        mouth_h = np.linalg.norm(pt(_MOUTH_LEFT) - pt(_MOUTH_RIGHT))
        MAR = mouth_v / (mouth_h + 1e-6)

        # EAR
        eye_v = np.linalg.norm(pt(_EYE_TOP) - pt(_EYE_BOTTOM))
        eye_h = np.linalg.norm(pt(_EYE_LEFT) - pt(_EYE_RIGHT))
        EAR = eye_v / (eye_h + 1e-6)

        # BRI — average brow displacement relative to eye centre
        eye_center = (pt(_EYE_LEFT) + pt(_EYE_RIGHT)) / 2
        inter_ocular = eye_h
        brow_mid = (pt(_BROW_LEFT) + pt(_BROW_RIGHT)) / 2
        BRI = (eye_center[1] - brow_mid[1]) / (inter_ocular + 1e-6)

        # LCP — average horizontal mouth corner displacement
        LCP = float((pt(_CORNER_LEFT)[0] + pt(_CORNER_RIGHT)[0]) / 2)

        return {"MAR": float(MAR), "EAR": float(EAR), "BRI": float(BRI), "LCP": float(LCP)}

    @staticmethod
    def _classify(v: AffectVector) -> str:
        """
        Rule-based classifier over the 4 geometric features.
        Replace with MobileNetV3-Small for final evaluation.
        """
        if v["BRI"] > 0.25 and v["MAR"] > 0.3:
            return "SURPRISED"
        if v["EAR"] < 0.15 and v["LCP"] < -5:
            return "FRUSTRATED"
        if v["LCP"] > 5:
            return "HAPPY"
        return "NEUTRAL"

    def release(self):
        self._face_mesh.close()
