import re

_POSITIVE = {
    "glad",
    "love",
    "lucky",
    "happy",
    "great",
    "grateful",
    "fun",
    "wonderful",
    "nice",
    "amazing",
    "delighted",
    "pleased",
    "yes",
    "solid",
}
_NEGATIVE = {
    "tired",
    "hard",
    "sorry",
    "unfortunately",
    "bad",
    "awful",
    "regrettably",
    "difficult",
    "frustrating",
    "no",
    "stop",
}

_AFFECT_TARGET = {
    "HAPPY": 1.0,
    "FRUSTRATED": -0.5,
    "NEUTRAL": 0.0,
    "SURPRISED": 0.0,
}

_GESTURE_OPENER_PATTERNS = {
    "THUMBS_UP": re.compile(r"^\s*(yes|yeah|totally|for sure|absolutely|sure)\b", re.I),
    "THUMBS_DOWN": re.compile(r"^\s*(no|nah|not really|i'd rather not)\b", re.I),
    "WAVING": re.compile(r"^\s*(hi|hey|hello)\b", re.I),
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def _sentiment_score(text: str) -> float:
    toks = _tokens(text)
    pos = len(toks & _POSITIVE)
    neg = len(toks & _NEGATIVE)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _affect_alignment(response: str, affect: str | None) -> float:
    if not affect:
        return 0.0
    target = _AFFECT_TARGET.get(affect, 0.0)
    score = _sentiment_score(response)
    # distance in [0, 2] → similarity in [0, 1]
    return max(0.0, 1.0 - abs(score - target) / 2.0)


def _gesture_alignment(response: str, gesture_tag: str | None) -> float:
    if not gesture_tag:
        return 0.0
    pattern = _GESTURE_OPENER_PATTERNS.get(gesture_tag)
    if pattern is None:
        return 0.5  # gesture has no testable opener; give partial credit
    return 1.0 if pattern.search(response) else 0.0


def _gaze_alignment(chunks: list[dict], gaze_bucket: str | None) -> float:
    if not gaze_bucket or not chunks:
        return 0.0
    matches = sum(1 for c in chunks if c.get("bucket") == gaze_bucket)
    return matches / len(chunks)


def compute_multimodal_alignment(
    response: str,
    affect: str | None,
    gesture_tag: str | None,
    gaze_bucket: str | None,
    chunks: list[dict],
) -> dict:
    scores: dict[str, float] = {}
    if affect:
        scores["affect_alignment"] = _affect_alignment(response, affect)
    if gesture_tag:
        scores["gesture_alignment"] = _gesture_alignment(response, gesture_tag)
    if gaze_bucket:
        scores["gaze_alignment"] = _gaze_alignment(chunks, gaze_bucket)
    overall = sum(scores.values()) / len(scores) if scores else 0.0
    return {
        "overall_score": round(overall, 4),
        "affect_alignment": round(scores.get("affect_alignment", 0.0), 4),
        "gesture_alignment": round(scores.get("gesture_alignment", 0.0), 4),
        "gaze_alignment": round(scores.get("gaze_alignment", 0.0), 4),
    }
