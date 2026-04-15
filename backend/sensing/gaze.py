# Gaze-based retrieval bucket hinting via MediaPipe iris landmarks.
from __future__ import annotations

import time
from dataclasses import dataclass, field

from backend.config.settings import settings

mp = None


# ── Iris landmark indices ──────────────────────────────────────────────────────
# MediaPipe refine_landmarks=True adds iris landmarks 468-477
_LEFT_IRIS_CENTER = 468
_RIGHT_IRIS_CENTER = 473

# ── Screen region → bucket map ─────────────────────────────────────────────────
# Defined as (x_min, y_min, x_max, y_max) in normalised [0,1] coords
_REGION_BUCKET: list[tuple[tuple[float, float, float, float], str]] = [
    ((0.3, 0.3, 0.7, 0.7), "social"),  # centre checked first (most specific)
    ((0.0, 0.0, 0.5, 0.5), "family"),
    ((0.5, 0.0, 1.0, 0.5), "medical"),
    ((0.0, 0.5, 0.5, 1.0), "hobbies"),
    ((0.5, 0.5, 1.0, 1.0), "daily_routine"),
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
        global mp
        import mediapipe as mp

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, bgr_frame) -> str | None:
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
