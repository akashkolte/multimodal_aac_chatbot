# Intent decomposition node — regex-split fragments + BGE zero-shot classifier.
from __future__ import annotations

import copy
import re
import time
from functools import lru_cache

import torch

from backend.config.settings import settings
from backend.pipeline.state import (
    GenerationConfig,
    IntentRoute,
    PipelineState,
    SubIntent,
)
from backend.retrieval.vector_store import get_device, get_embedder
from backend.sensing.bucket_keywords import infer_bucket

_CLASS_EXEMPLARS: dict[str, list[str]] = {
    "PERSONAL": [
        "what is your favourite food",
        "tell me about your family",
        "what do you do for work",
        "did you grow up around here",
        "what was your childhood like",
    ],
    "PRESENT_STATE": [
        "how are you feeling today",
        "are you tired right now",
        "what are you doing at the moment",
        "did you sleep well last night",
        "are you in pain today",
        "how is your day going",
        "are you having a good day",
    ],
    "CONTEXTUAL": [
        "what did you just say",
        "what did I ask earlier",
        "you mentioned something before",
        "can you repeat that",
        "what were we talking about",
    ],
    "OPEN_DOMAIN": [
        "what is the capital of france",
        "how many planets are there",
        "who wrote hamlet",
        "when was world war two",
        "what does photosynthesis mean",
    ],
}

_CLASSIFIER_THRESHOLD = (
    0.45  # below this → PERSONAL fallback (safe default for OOV / typos / short input)
)
_CONTEXTUAL_MARGIN_MIN = (
    0.08  # CONTEXTUAL must beat runner-up by at least this — it over-matches without it
)
_PRESENT_STATE_MARGIN_MIN = (
    0.05  # PRESENT_STATE skips retrieval, so a narrow win against PERSONAL would silently
    # drop persona memories. Require a clear margin before going down that path.
)
_MIN_FRAGMENT_WORDS = 3
_MAX_FRAGMENTS = 4

_CONTEXTUAL_MARKERS = (
    "earlier",
    "before",
    "mentioned",
    "said",
    "asked",
    "just",
    "repeat",
)
_CONTEXTUAL_MARKER_PATTERN = re.compile(
    r"\b(" + "|".join(_CONTEXTUAL_MARKERS) + r")\b",
    flags=re.IGNORECASE,
)

_SPLIT_PATTERN = re.compile(
    r"\s+(?:and|but|also|plus)\s+|[;.?!]+\s+|,\s+(?=\w)",
    flags=re.IGNORECASE,
)

_AFFECT_CONFIG: dict[str, GenerationConfig] = {
    "HAPPY": {
        "max_tokens": settings.max_tokens_happy,
        "tone_tag": "[TONE:WARM]",
        "retrieval_mode": "full",
        "persona_mod": "amplify_quirks",
        "style": {
            "tone_tag": "[TONE:WARM]",
            "register": "warm, upbeat, affectionate",
            "prefer_words": [
                "glad",
                "love",
                "lucky",
                "happy",
                "great",
                "grateful",
                "fun",
            ],
            "avoid_words": ["unfortunately", "frankly", "tired", "hard", "sorry"],
            "opener_hint": None,
            "exemplar": "Yeah — honestly, that made my week.",
        },
    },
    "FRUSTRATED": {
        "max_tokens": settings.max_tokens_frustrated,
        "tone_tag": "[TONE:DIRECT_EMPATHETIC]",
        "retrieval_mode": "fast",
        "persona_mod": "suppress_humor",
        "style": {
            "tone_tag": "[TONE:DIRECT_EMPATHETIC]",
            "register": "direct, short, validating — no jokes",
            "prefer_words": ["okay", "yes", "right", "i hear you", "fair"],
            "avoid_words": ["hilarious", "ha", "lol", "cheerful", "delightful"],
            "opener_hint": "Acknowledge the feeling in 3-5 words before the answer.",
            "exemplar": "Yeah. That's a lot. Short answer: yes.",
        },
    },
    "NEUTRAL": {
        "max_tokens": settings.max_tokens_neutral,
        "tone_tag": "[TONE:DEFAULT]",
        "retrieval_mode": "full",
        "persona_mod": "baseline",
        "style": {
            "tone_tag": "[TONE:DEFAULT]",
            "register": "natural, conversational",
            "prefer_words": [],
            "avoid_words": [],
            "opener_hint": None,
            # Empty on purpose — let the persona's own example_phrases carry the register.
            "exemplar": "",
        },
    },
    "SURPRISED": {
        "max_tokens": settings.max_tokens_surprised,
        "tone_tag": "[TONE:CLARIFYING]",
        "retrieval_mode": "full",
        "persona_mod": "add_confirmation",
        "style": {
            "tone_tag": "[TONE:CLARIFYING]",
            "register": "curious, clarifying",
            "prefer_words": ["really", "wait", "huh", "oh"],
            "avoid_words": [],
            "opener_hint": "Mirror surprise briefly, then ask a clarifying question.",
            "exemplar": "Oh — wait, really? Did you mean the Friday one?",
        },
    },
}


@lru_cache(maxsize=1)
def _exemplar_matrices() -> dict[str, torch.Tensor]:
    embedder = get_embedder()
    device = get_device()
    return {
        cls: embedder.encode(
            exemplars,
            convert_to_tensor=True,
            normalize_embeddings=True,
            device=device,
        )
        for cls, exemplars in _CLASS_EXEMPLARS.items()
    }


def _split_query(query: str) -> list[str]:
    raw = [p.strip() for p in _SPLIT_PATTERN.split(query) if p and p.strip()]
    keep = [p for p in raw if len(p.split()) >= _MIN_FRAGMENT_WORDS]
    if not keep:
        keep = [query.strip()] if query.strip() else []
    return keep[:_MAX_FRAGMENTS]


def _classify(fragment: str) -> str:
    embedder = get_embedder()
    device = get_device()
    vec = embedder.encode(
        [fragment],
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=device,
    )[0]

    scores: dict[str, float] = {}
    for cls, mat in _exemplar_matrices().items():
        scores[cls] = float((mat @ vec).max())

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_cls, best_score = ranked[0]
    runner_up_score = ranked[1][1]

    if best_score < _CLASSIFIER_THRESHOLD:
        return "PERSONAL"  # conservative default: treat as a question about the persona

    # CONTEXTUAL is the riskiest class — if wrong, we lose all persona grounding.
    # Require it to clearly beat the runner-up and for the fragment to mention
    # prior discourse (matched at word boundaries, so "just" doesn't match "unjust").
    if best_cls == "CONTEXTUAL":
        margin = best_score - runner_up_score
        has_discourse_marker = bool(_CONTEXTUAL_MARKER_PATTERN.search(fragment))
        if margin < _CONTEXTUAL_MARGIN_MIN or not has_discourse_marker:
            return "PERSONAL"

    # PRESENT_STATE skips retrieval entirely, so a narrow win over PERSONAL
    # would silently drop persona memories with no recovery path. Demote to
    # PERSONAL if the win isn't decisive — better to over-retrieve than to
    # answer a personal question with no chunks.
    if best_cls == "PRESENT_STATE":
        margin = best_score - runner_up_score
        if margin < _PRESENT_STATE_MARGIN_MIN:
            return "PERSONAL"

    return best_cls


def run(state: PipelineState) -> dict:
    t0 = time.perf_counter()

    # --fast mode: intent_route already resolved by keyword routing in main.py
    if state.get("intent_route") and state.get("generation_config"):
        return {}

    affect_state = state.get("affect") or {}
    emotion: str = affect_state.get("emotion", "NEUTRAL")
    query: str = state["raw_query"]
    # Deep-copy: callers may mutate gen_config downstream; never hand them the shared constant.
    gen_config = copy.deepcopy(_AFFECT_CONFIG.get(emotion, _AFFECT_CONFIG["NEUTRAL"]))

    fragments = _split_query(query)
    priority = "fast" if emotion == "FRUSTRATED" else "normal"

    sub_intents: list[SubIntent] = []
    for frag in fragments:
        cls = _classify(frag)
        bucket_hint = infer_bucket(frag) if cls == "PERSONAL" else None
        sub_intents.append(
            {
                "type": cls,
                "query": frag,
                "bucket_hint": bucket_hint,
                "priority": priority,
            }
        )

    if not sub_intents:
        sub_intents = [
            {
                "type": "PERSONAL",
                "query": query,
                "bucket_hint": None,
                "priority": priority,
            }
        ]

    # Prefer resolved_intent.text when the frontend did voice⇄air reconciliation;
    # fall back to raw air_written_text when no voice was captured.
    resolved = state.get("resolved_intent") or {}
    supplement = (resolved.get("text") or "").strip() or state.get("air_written_text")
    if supplement:
        # Classify the supplement the same way as a normal fragment so a
        # present-tense supplement ("tired") on a present-state question
        # doesn't silently flip the route to PERSONAL and re-enable retrieval.
        sup_cls = _classify(supplement)
        sub_intents.append(
            {
                "type": sup_cls,
                "query": supplement,
                "bucket_hint": infer_bucket(supplement)
                if sup_cls == "PERSONAL"
                else None,
                "priority": priority,
            }
        )

    route: IntentRoute = {
        "sub_intents": sub_intents,
        "style_constraints": dict(gen_config),
        "affect": emotion,
    }

    latency_log = dict(state.get("latency_log") or {})
    latency_log["t_intent"] = round(time.perf_counter() - t0, 4)

    return {
        "intent_route": route,
        "generation_config": gen_config,
        "latency_log": latency_log,
    }
