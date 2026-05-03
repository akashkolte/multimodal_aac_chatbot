GESTURE_DIRECTIVES: dict[str, dict[str, str]] = {
    "THUMBS_UP": {
        "tone": "[GESTURE:THUMBS_UP][TONE:AFFIRMATIVE]",
        "opener_hint": "Open with an affirmation (Yes / Totally / For sure).",
    },
    "THUMBS_DOWN": {
        "tone": "[GESTURE:THUMBS_DOWN][TONE:NEGATIVE]",
        "opener_hint": "Open by declining or disagreeing briefly.",
    },
    "POINTING_UP": {
        "tone": "[GESTURE:POINTING_UP][INTENT:REFERENTIAL]",
        "opener_hint": "Treat the query as referring to a specific named thing.",
    },
    "CLOSED_FIST": {
        "tone": "[GESTURE:CLOSED_FIST][TONE:EMPHATIC]",
        "opener_hint": "Respond with emphasis or urgency — something important needs saying.",
    },
    "OPEN_PALM": {
        "tone": "[GESTURE:OPEN_PALM][INTENT:GREETING]",
        "opener_hint": "Open with a warm greeting.",
    },
    "VICTORY": {
        "tone": "[GESTURE:VICTORY][TONE:CELEBRATORY]",
        "opener_hint": "Open with celebration or excitement.",
    },
    "I_LOVE_YOU": {
        "tone": "[GESTURE:I_LOVE_YOU][TONE:AFFECTIONATE]",
        "opener_hint": "Open with warmth and affection.",
    },
}
