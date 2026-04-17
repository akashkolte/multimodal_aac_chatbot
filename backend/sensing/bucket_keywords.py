_BUCKET_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("medical", ("medication", "medicine", "doctor", "health", "allergic", "therapy")),
    ("family", ("family", "mom", "dad", "brother", "sister", "parents")),
    ("hobbies", ("hobby", "like to do", "enjoy", "weekend", "fun")),
    ("daily_routine", ("routine", "morning", "wake", "sleep", "daily")),
    ("social", ("friend", "social", "people", "party", "community")),
]


def infer_bucket(query: str) -> str | None:
    q = query.lower()
    for bucket, words in _BUCKET_KEYWORDS:
        if any(w in q for w in words):
            return bucket
    return None
