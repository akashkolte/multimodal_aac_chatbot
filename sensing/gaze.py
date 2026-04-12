"""
L1 — Gaze-based retrieval activation (Bonus feature, proposal §5.2).

Uses MediaPipe iris landmarks (468-472) to estimate gaze direction as
a 2D screen-coordinate vector. Sustained fixation (> 1.5 s dwell time)
on a defined UI region pre-biases the retrieval layer toward the
corresponding memory bucket.

UI region → bucket mapping:
  top-left quadrant     → family
  top-right quadrant    → medical
  bottom-left quadrant  → hobbies
  bottom-right quadrant → daily_routine
  centre strip          → social
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from config.settings import settings

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# ── Iris landmark indices ──────────────────────────────────────────────────────
# MediaPipe refine_landmarks=True adds iris landmarks 468-477
_LEFT_IRIS_CENTER  = 468
_RIGHT_IRIS_CENTER = 473

# ── Screen region → bucket map ─────────────────────────────────────────────────
# Defined as (x_min, y_min, x_max, y_max) in normalised [0,1] coords
_REGION_BUCKET: list[tuple[tuple[float, float, float, float], str]] = [
    ((0.0, 0.0, 0.5, 0.5), "family"),
    ((0.5, 0.0, 1.0, 0.5), "medical"),
    ((0.0, 0.5, 0.5, 1.0), "hobbies"),
    ((0.5, 0.5, 1.0, 1.0), "daily_routine"),
    ((0.3, 0.3, 0.7, 0.7), "social"),   # centre strip (checked last → lowest priority)
]


@dataclass
class GazeTracker:
    """
    Stateful gaze tracker. Call `process_frame` each frame.
    Returns the bucket name when dwell threshold is exceeded, else None.
    """
    _dwell_start: float = field(default=0.0)
    _current_region: str | None = field(default=None)

    def __post_init__(self):
        if not _MP_AVAILABLE:
            raise ImportError("mediapipe is required: pip install mediapipe")
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, bgr_frame) -> str | None:
        """
        Returns the hinted bucket name once dwell threshold is exceeded,
        then resets the dwell timer. Returns None otherwise.
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            self._reset()
            return None

        lm = result.multi_face_landmarks[0].landmark

        # Average left + right iris centres for gaze estimate
        gaze_x = (lm[_LEFT_IRIS_CENTER].x + lm[_RIGHT_IRIS_CENTER].x) / 2
        gaze_y = (lm[_LEFT_IRIS_CENTER].y + lm[_RIGHT_IRIS_CENTER].y) / 2

        bucket = self._region_for(gaze_x, gaze_y)

        if bucket != self._current_region:
            self._current_region = bucket
            self._dwell_start = time.time()
            return None

        dwell = time.time() - self._dwell_start
        if dwell >= settings.gaze_dwell_threshold_s and bucket is not None:
            self._reset()
            return bucket

        return None

    @staticmethod
    def _region_for(x: float, y: float) -> str | None:
        for (x0, y0, x1, y1), bucket in _REGION_BUCKET:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return bucket
        return None

    def _reset(self):
        self._dwell_start = 0.0
        self._current_region = None

    def release(self):
        self._face_mesh.close()
