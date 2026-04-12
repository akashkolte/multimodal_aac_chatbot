"""
L1 — Hand gesture recognition via MediaPipe Hands.

Recognises 4 gestures from 21 3D hand landmarks at ~15 fps using
normalised joint-angle rules (no ML model needed at this stage):

  THUMBS_UP    → [TONE:AFFIRMATIVE]
  THUMBS_DOWN  → [TONE:NEGATIVE]
  POINTING     → [INTENT:REFERENTIAL]
  WAVING       → [INTENT:GREETING]

Each detected gesture is mapped to a stylistic constraint tag that is
injected into the generation prompt by the planner node.
"""
from __future__ import annotations

import numpy as np

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# Gesture → prompt constraint tag mapping
GESTURE_TO_TAG: dict[str, str] = {
    "THUMBS_UP":   "[GESTURE:THUMBS_UP][TONE:AFFIRMATIVE]",
    "THUMBS_DOWN": "[GESTURE:THUMBS_DOWN][TONE:NEGATIVE]",
    "POINTING":    "[GESTURE:POINTING][INTENT:REFERENTIAL]",
    "WAVING":      "[GESTURE:WAVING][INTENT:GREETING]",
}


class GestureClassifier:
    """
    Stateful classifier — create one instance per session.
    Feed MediaPipe hand landmark results each frame.
    """

    def __init__(self):
        if not _MP_AVAILABLE:
            raise ImportError("mediapipe is required: pip install mediapipe")
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

    def process_frame(self, bgr_frame) -> str | None:
        """
        Returns a gesture label string or None if no clear gesture is detected.
        """
        import cv2
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            return None

        lm = result.multi_hand_landmarks[0].landmark
        pts = np.array([[l.x, l.y, l.z] for l in lm])

        return self._classify(pts)

    def gesture_tag(self, bgr_frame) -> str | None:
        """Convenience: returns the prompt tag directly, or None."""
        gesture = self.process_frame(bgr_frame)
        return GESTURE_TO_TAG.get(gesture) if gesture else None

    @staticmethod
    def _classify(pts: np.ndarray) -> str | None:
        """
        Rule-based gesture classification over normalised joint positions.

        MediaPipe hand landmark indices:
          0=WRIST, 1-4=THUMB, 5-8=INDEX, 9-12=MIDDLE, 13-16=RING, 17-20=PINKY
        """
        # Normalise: wrist at origin, scale by palm width
        wrist = pts[0]
        palm_width = np.linalg.norm(pts[5] - pts[17]) + 1e-6
        p = (pts - wrist) / palm_width

        thumb_tip   = p[4]
        index_tip   = p[8]
        middle_tip  = p[12]
        ring_tip    = p[16]
        pinky_tip   = p[20]
        index_mcp   = p[5]   # knuckle

        # THUMBS_UP: thumb tip above wrist, other fingers curled
        fingers_curled = all(
            np.linalg.norm(tip) < np.linalg.norm(p[mcp])
            for tip, mcp in [(index_tip, p[5]), (middle_tip, p[9]), (ring_tip, p[13])]
        )
        if thumb_tip[1] < -0.3 and fingers_curled:
            return "THUMBS_UP"

        # THUMBS_DOWN: thumb tip below wrist, other fingers curled
        if thumb_tip[1] > 0.3 and fingers_curled:
            return "THUMBS_DOWN"

        # POINTING: index extended, others curled
        index_extended = np.linalg.norm(index_tip) > np.linalg.norm(index_mcp) * 1.3
        others_curled  = all(
            np.linalg.norm(tip) < 0.5
            for tip in [middle_tip, ring_tip, pinky_tip]
        )
        if index_extended and others_curled:
            return "POINTING"

        # WAVING: all fingers extended, hand roughly vertical
        all_extended = all(
            np.linalg.norm(tip) > 0.5
            for tip in [index_tip, middle_tip, ring_tip, pinky_tip]
        )
        if all_extended:
            return "WAVING"

        return None

    def release(self):
        self._hands.close()
