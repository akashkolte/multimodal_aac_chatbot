"""
L1 — Air writing recognition via index-finger tip trajectory (proposal §5.2).

Tracks MediaPipe Hands landmark 8 (index fingertip) across frames.
Stroke segmentation uses velocity thresholding:
  - stroke starts when velocity > START_VEL px/frame
  - stroke ends when velocity < END_VEL px/frame for > GAP_MS ms

Segmented strokes are classified against a template library using
Dynamic Time Warping (DTW). Supports:
  - 26 uppercase English letters (A-Z)
  - 10 digits (0-9)
  - 10 most frequent Devanagari characters (for Arjun's Hindi inputs)

Recognised characters are concatenated and returned as a text string
to the intent decomposition layer.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config.settings import settings

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False

# ── Landmark index ─────────────────────────────────────────────────────────────
_INDEX_TIP = 8


@dataclass
class AirWriter:
    """
    Stateful air-writing recogniser. Feed frames from a webcam loop.
    Call `get_text()` to retrieve and clear the current buffer.
    """
    _trajectory: list[tuple[float, float]] = field(default_factory=list)
    _in_stroke: bool = False
    _stroke_end_time: float = field(default=0.0)
    _text_buffer: list[str] = field(default_factory=list)
    _templates: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        if not _MP_AVAILABLE:
            raise ImportError("mediapipe is required: pip install mediapipe")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._prev_pt: tuple[float, float] | None = None
        self._templates = _load_templates()

    def process_frame(self, bgr_frame) -> str | None:
        """
        Process one frame. Returns a recognised character when a stroke
        completes, or None otherwise.
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            self._prev_pt = None
            return self._check_stroke_end()

        h, w = bgr_frame.shape[:2]
        lm = result.multi_hand_landmarks[0].landmark
        tip = (lm[_INDEX_TIP].x * w, lm[_INDEX_TIP].y * h)

        velocity = 0.0
        if self._prev_pt is not None:
            velocity = np.linalg.norm(np.array(tip) - np.array(self._prev_pt))
        self._prev_pt = tip

        start_v = settings.air_write_velocity_start
        end_v   = settings.air_write_velocity_end

        if velocity > start_v:
            self._in_stroke = True
            self._trajectory.append(tip)
            self._stroke_end_time = 0.0
        elif self._in_stroke and velocity < end_v:
            if self._stroke_end_time == 0.0:
                self._stroke_end_time = time.time()
            return self._check_stroke_end()

        return None

    def _check_stroke_end(self) -> str | None:
        if not self._in_stroke or self._stroke_end_time == 0.0:
            return None
        gap_s = settings.air_write_end_gap_ms / 1000.0
        if time.time() - self._stroke_end_time >= gap_s:
            char = self._recognise(self._trajectory)
            self._trajectory = []
            self._in_stroke = False
            self._stroke_end_time = 0.0
            if char:
                self._text_buffer.append(char)
            return char
        return None

    def _recognise(self, trajectory: list[tuple[float, float]]) -> str | None:
        if len(trajectory) < 5 or not self._templates:
            return None
        query = _normalise_trajectory(np.array(trajectory))
        best_char, best_dist = None, float("inf")
        for char, template in self._templates.items():
            dist = _dtw_distance(query, template)
            if dist < best_dist:
                best_dist = dist
                best_char = char
        return best_char

    def get_text(self) -> str:
        """Return and clear the accumulated air-written text."""
        text = "".join(self._text_buffer)
        self._text_buffer.clear()
        return text

    def release(self):
        self._hands.close()


# ── DTW helpers ───────────────────────────────────────────────────────────────

def _normalise_trajectory(pts: np.ndarray) -> np.ndarray:
    """Scale trajectory to unit bounding box, resample to 32 points."""
    pts = pts - pts.min(axis=0)
    scale = pts.max(axis=0) + 1e-6
    pts = pts / scale
    # Resample to fixed length via linear interpolation
    t_old = np.linspace(0, 1, len(pts))
    t_new = np.linspace(0, 1, 32)
    return np.column_stack([
        np.interp(t_new, t_old, pts[:, 0]),
        np.interp(t_new, t_old, pts[:, 1]),
    ])


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Simple O(n²) DTW — trajectories are short (32 pts), so this is fine."""
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = np.linalg.norm(a[i - 1] - b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _load_templates() -> dict[str, np.ndarray]:
    """
    Load pre-recorded stroke templates from disk.
    Template files should be numpy arrays of shape (32, 2) stored as .npy.
    Returns an empty dict if no template directory exists yet.
    """
    from pathlib import Path
    template_dir = Path("data/air_write_templates")
    if not template_dir.exists():
        return {}
    templates = {}
    for f in template_dir.glob("*.npy"):
        char = f.stem    # filename = character label
        templates[char] = np.load(f)
    return templates
