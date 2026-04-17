GESTURE_DIRECTIVES: dict[str, dict[str, str]] = {
    "THUMBS_UP": {
        "tone": "[GESTURE:THUMBS_UP][TONE:AFFIRMATIVE]",
        "opener_hint": "Open with an affirmation (Yes / Totally / For sure).",
    },
    "THUMBS_DOWN": {
        "tone": "[GESTURE:THUMBS_DOWN][TONE:NEGATIVE]",
        "opener_hint": "Open by declining or disagreeing briefly.",
    },
    "POINTING": {
        "tone": "[GESTURE:POINTING][INTENT:REFERENTIAL]",
        "opener_hint": "Treat the query as referring to a specific named thing.",
    },
    "WAVING": {
        "tone": "[GESTURE:WAVING][INTENT:GREETING]",
        "opener_hint": "Open with a greeting.",
    },
}
