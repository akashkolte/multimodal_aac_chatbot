_BUCKET_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    # AAC air-writing templates (help/water/stop/done/more) are mapped here too —
    # when a partner/user signals one of these, retrieval pulls from the matching bucket.
    (
        "medical",
        (
            "medication",
            "medicine",
            "doctor",
            "health",
            "allergic",
            "therapy",
            "help",
            "stop",
        ),
    ),
    ("family", ("family", "mom", "dad", "brother", "sister", "parents")),
    ("hobbies", ("hobby", "like to do", "enjoy", "weekend", "fun")),
    (
        "daily_routine",
        ("routine", "morning", "wake", "sleep", "daily", "water", "done", "more"),
    ),
    ("social", ("friend", "social", "people", "party", "community", "hi")),
]


def infer_bucket(query: str) -> str | None:
    q = query.lower()
    for bucket, words in _BUCKET_KEYWORDS:
        if any(w in q for w in words):
            return bucket
    return None
