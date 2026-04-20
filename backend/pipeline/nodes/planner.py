import concurrent.futures
import queue
import random
import threading
import time
from collections.abc import Iterator

from backend.config.settings import settings
from backend.generation.llm_client import (
    active_model,
    chat_complete,
    chat_complete_stream,
    finalize_streamed,
)
from backend.guardrails.checks import check_output
from backend.pipeline.intent_kind import classify_intent_kind
from backend.pipeline.state import Candidate, PipelineState, StyleDirective
from backend.retrieval import pick_index
from backend.sensing.labels import GESTURE_DIRECTIVES

# For present-state fan-out: three fixed emotional reads the persona can
# project, so the user can pick among "good / fine / not great" rather than
# three paraphrases of one mood.
_PRESENT_STATE_STRATEGIES = [
    ("present_good", "HAPPY"),
    ("present_fine", "NEUTRAL"),
    ("present_rough", "FRUSTRATED"),
]

_PERSONA_MOD_INSTRUCTIONS = {
    "amplify_quirks": "Amplify your characteristic style and personality.",
    "suppress_humor": "Be direct and supportive. Suppress humor.",
    "baseline": "Use your natural communication style.",
    "add_confirmation": "Add a clarifying question or confirmation at the end.",
    "turnaround": (
        "Your previous reply missed what you actually meant. Rephrase "
        "more directly — change the wording meaningfully, not just "
        "surface tweaks — and end with a one-sentence clarifying "
        "question to confirm you're on the right track."
    ),
    "reverse_stance": (
        "Your previous reply was substantively wrong — not poorly worded, "
        "but the wrong content. Take a meaningfully different stance using "
        "the available memories or, if none fit, honestly say you don't "
        "know. Do NOT just reword the previous reply."
    ),
    "present_state_retry": (
        "Your previous reply was wrong about your current state. The "
        "affect signal probably misled you. Either flip the emotional "
        "read (if you said 'good', try 'not great') or honestly admit "
        "you're not sure how you feel right now. Do NOT invent details."
    ),
    "all_rejected": (
        "The user rejected every option you gave last time. Try a "
        "meaningfully different angle — different memory focus, different "
        "emotional register, or admit you don't have a clean answer. Do "
        "NOT re-use wording from the rejected options."
    ),
}


def run_primary(state: PipelineState) -> dict:
    return _run(state, tier="primary")


def run_fallback(state: PipelineState) -> dict:
    return _run(state, tier="fallback")


def run_primary_stream(state: PipelineState) -> Iterator[dict]:
    """Token-level streaming variant of the planner.

    Yields events as tokens arrive across all concurrent candidate streams:
      {"type": "candidate_start", "idx": 0, "strategy": "broad", "grounded_buckets": [...]}
      {"type": "token", "idx": 0, "delta": "Hello"}
      {"type": "candidate_done", "idx": 0, "text": "Hello world."}
      {"type": "side_index", "text": "..."}   (optional, at start if there's a hit)
      {"type": "complete", "candidates": [...], "selected_response": "...", ... final state dict}
    """
    yield from _run_stream(state, tier="primary")


_STREAM_SENTINEL = object()


def _run_stream(state: PipelineState, tier: str) -> Iterator[dict]:
    t0 = time.perf_counter()

    profile = state["persona_profile"]
    affect = (state.get("affect") or {}).get("emotion", "NEUTRAL")
    gen_cfg = state.get("generation_config") or {}
    chunks = state.get("retrieved_chunks") or []
    history = (state.get("session_history") or [])[-20:]

    style: StyleDirective = gen_cfg["style"]
    gesture_tag = state.get("gesture_tag")
    air_written_text = state.get("air_written_text")
    resolved_intent = state.get("resolved_intent")
    turnaround_triggered = state.get("turnaround_triggered", False)
    rejected_response: str | None = None
    if turnaround_triggered:
        rejected_response = state.get("selected_response")
    rejected_candidates: list[str] = list(state.get("rejected_candidates") or [])
    intent_kind = classify_intent_kind(state.get("intent_route"))
    max_tokens = gen_cfg.get("max_tokens", settings.max_tokens_neutral)

    # Turnaround rephrases are single-shot; everything else fans out.
    # Present-state varies affect (good/fine/rough), memory questions vary
    # which chunks are primary (broad/focused/serendipitous).
    single_shot = turnaround_triggered
    is_present_state = intent_kind == "present_state"
    if single_shot:
        strategies: list[tuple[str, str | None]] = [("focused", None)]
    elif is_present_state:
        strategies = list(_PRESENT_STATE_STRATEGIES)
    else:
        strategies = [
            ("broad", None),
            ("focused", None),
            ("serendipitous", None),
        ]
    # Higher temp on regenerate; also bump for present-state since three
    # strategies share the same (empty) grounding and need sampling noise.
    base_temp = 1.0 if (rejected_candidates or is_present_state) else 0.7

    # Optional side-index hit — surface as an extra card right away, not generated.
    side_index_candidate: Candidate | None = None
    if not single_shot and not is_present_state:
        try:
            hit = pick_index.lookup(
                query=state["raw_query"],
                user_id=state["user_id"],
                threshold=0.85,
            )
        except Exception as exc:
            print(f"[planner] pick_index lookup failed: {exc!r}")
            hit = None
        if hit:
            text = (hit.get("picked_text") or "").strip()
            if text:
                side_index_candidate = Candidate(
                    text=text,
                    strategy="side_index",
                    grounded_buckets=[],
                )

    # Pre-announce each candidate slot so the UI can draw empty cards immediately.
    cards: list[dict] = []
    if side_index_candidate:
        cards.append({"strategy": "side_index", "grounded_buckets": []})
    for strategy_name, _affect_override in strategies:
        if is_present_state:
            card_buckets: list[str] = []
        else:
            strategy_chunks = _pick_strategy_chunks(list(chunks), strategy_name)
            card_buckets = [c.get("bucket", "") for c in strategy_chunks]
        cards.append(
            {
                "strategy": strategy_name,
                "grounded_buckets": card_buckets,
            }
        )
    for idx, card in enumerate(cards):
        yield {
            "type": "candidate_start",
            "idx": idx,
            "strategy": card["strategy"],
            "grounded_buckets": card["grounded_buckets"],
        }
    if side_index_candidate is not None:
        yield {
            "type": "candidate_done",
            "idx": 0,
            "text": side_index_candidate["text"],
        }

    # Spawn a worker thread per strategy. Each one streams tokens into a shared
    # queue; the generator forwards them as SSE events.
    llm_cards_offset = 1 if side_index_candidate else 0
    evt_queue: queue.Queue[dict | object] = queue.Queue()
    completed: list[Candidate | None] = [None] * len(strategies)
    completed_lock = threading.Lock()

    def _worker(slot: int, strategy: str, affect_override: str | None) -> None:
        if is_present_state:
            strategy_chunks = []  # present-state has no memory grounding
        else:
            strategy_chunks = _pick_strategy_chunks(list(chunks), strategy)
        effective_affect = affect_override if affect_override is not None else affect
        messages = _build_messages(
            profile,
            strategy_chunks,
            history,
            state["raw_query"],
            style,
            gen_cfg,
            gesture_tag=gesture_tag,
            air_written_text=air_written_text,
            resolved_intent=resolved_intent,
            rejected_response=rejected_response,
            rejected_candidates=rejected_candidates,
            intent_kind=intent_kind,
            affect=effective_affect,
        )
        buf: list[str] = []
        try:
            for piece in chat_complete_stream(
                messages=messages,
                max_tokens=max_tokens,
                temperature=base_temp,
                tier=tier,
            ):
                buf.append(piece)
                evt_queue.put(
                    {
                        "type": "token",
                        "idx": llm_cards_offset + slot,
                        "delta": piece,
                    }
                )
        except Exception as exc:
            evt_queue.put(
                {
                    "type": "candidate_error",
                    "idx": llm_cards_offset + slot,
                    "error": repr(exc),
                }
            )
            with completed_lock:
                completed[slot] = None
            evt_queue.put(_STREAM_SENTINEL)
            return

        final = finalize_streamed("".join(buf))
        guard = check_output(final, strategy_chunks)
        if not guard["passed"]:
            final = guard["fallback"]
        cand = Candidate(
            text=final,
            strategy=strategy,
            grounded_buckets=[c.get("bucket", "") for c in strategy_chunks],
        )
        with completed_lock:
            completed[slot] = cand
        evt_queue.put(
            {
                "type": "candidate_done",
                "idx": llm_cards_offset + slot,
                "text": final,
            }
        )
        evt_queue.put(_STREAM_SENTINEL)

    threads = [
        threading.Thread(target=_worker, args=(i, s, a), daemon=True)
        for i, (s, a) in enumerate(strategies)
    ]
    for t in threads:
        t.start()

    remaining = len(threads)
    while remaining > 0:
        evt = evt_queue.get()
        if evt is _STREAM_SENTINEL:
            remaining -= 1
            continue
        yield evt  # type: ignore[misc]

    for t in threads:
        t.join()

    with completed_lock:
        llm_cands = [c for c in completed if c is not None]
    all_cands: list[Candidate] = []
    if side_index_candidate is not None:
        all_cands.append(side_index_candidate)
    all_cands.extend(llm_cands)

    # De-dupe against rejected + each other.
    seen: set[str] = {r.strip().lower() for r in rejected_candidates if r}
    uniq: list[Candidate] = []
    for c in all_cands:
        key = c["text"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(c)
    if not uniq:
        # Every candidate was a dup-of-rejected or guardrail-rejected. Surface
        # a non-empty placeholder so the UI isn't showing a blank bubble and
        # the user knows they can regenerate. Logged so we notice if this ever
        # fires in practice — it means something upstream collapsed.
        print(
            f"[planner] empty-candidate fallback fired "
            f"user={state.get('user_id')!r} turn_id={state.get('turn_id')} "
            f"raw_query={state.get('raw_query', '')[:80]!r}"
        )
        uniq = all_cands[:1] or [
            Candidate(
                text="I'm not sure how to answer that — try asking in a different way.",
                strategy="empty",
                grounded_buckets=[],
            )
        ]

    selected = uniq[0]["text"]

    t_gen = time.perf_counter() - t0
    latency_log = dict(state.get("latency_log") or {})
    latency_log["t_generation"] = round(t_gen, 4)
    latency_log["t_total"] = round(
        latency_log.get("t_sensing", 0)
        + latency_log.get("t_intent", 0)
        + latency_log.get("t_retrieval", 0)
        + t_gen,
        4,
    )

    yield {
        "type": "complete",
        "planner_update": {
            "augmented_prompt": None,  # skipping for streaming — not worth rebuilding
            "candidates": uniq,
            "selected_response": selected,
            "llm_tier_used": tier,
            "llm_model_used": active_model(tier),
            "latency_log": latency_log,
            "guardrail_passed": True,
        },
    }


def _pick_strategy_chunks(all_chunks: list[dict], strategy: str) -> list[dict]:
    """Select which chunks become the *primary* grounding for a candidate.
    Non-personal chunks (contextual, open_domain) always pass through —
    they're small and query-grounded, not memory variation.
    """
    personal = [c for c in all_chunks if c.get("source", "personal") == "personal"]
    others = [c for c in all_chunks if c.get("source", "personal") != "personal"]

    if not personal:
        return all_chunks

    if strategy == "broad":
        chosen = personal
    elif strategy == "focused":
        chosen = personal[:1]
    elif strategy == "serendipitous":
        if len(personal) >= 2:
            pool = personal[1:]
            k = min(len(pool), max(1, len(personal) - 1))
            chosen = random.sample(pool, k)
        else:
            chosen = personal
    else:
        chosen = personal

    return chosen + others


def _run(state: PipelineState, tier: str) -> dict:
    t0 = time.perf_counter()

    profile = state["persona_profile"]
    affect = (state.get("affect") or {}).get("emotion", "NEUTRAL")
    gen_cfg = state.get("generation_config") or {}
    chunks = state.get("retrieved_chunks") or []
    history = (state.get("session_history") or [])[-20:]

    style: StyleDirective = gen_cfg["style"]
    gesture_tag = state.get("gesture_tag")
    air_written_text = state.get("air_written_text")
    resolved_intent = state.get("resolved_intent")
    turnaround_triggered = state.get("turnaround_triggered", False)
    rejected_response: str | None = None
    if turnaround_triggered:
        rejected_response = state.get("selected_response")
    rejected_candidates: list[str] = list(state.get("rejected_candidates") or [])
    intent_kind = classify_intent_kind(state.get("intent_route"))
    max_tokens = gen_cfg.get("max_tokens", settings.max_tokens_neutral)

    # Turnaround rephrases are single-shot; everything else fans out. Present-
    # state varies affect (good/fine/rough), memory questions vary chunks
    # (broad/focused/serendipitous).
    single_shot = turnaround_triggered
    is_present_state = intent_kind == "present_state"
    if single_shot:
        strategies_cfg: list[tuple[str, str | None]] = [("focused", None)]
    elif is_present_state:
        strategies_cfg = list(_PRESENT_STATE_STRATEGIES)
    else:
        strategies_cfg = [
            ("broad", None),
            ("focused", None),
            ("serendipitous", None),
        ]

    base_temp = 1.0 if (rejected_candidates or is_present_state) else 0.7

    def _gen_one(cfg: tuple[str, str | None]) -> Candidate:
        strategy, affect_override = cfg
        if is_present_state:
            strategy_chunks: list[dict] = []
        else:
            strategy_chunks = _pick_strategy_chunks(list(chunks), strategy)
        effective_affect = affect_override if affect_override is not None else affect
        messages = _build_messages(
            profile,
            strategy_chunks,
            history,
            state["raw_query"],
            style,
            gen_cfg,
            gesture_tag=gesture_tag,
            air_written_text=air_written_text,
            resolved_intent=resolved_intent,
            rejected_response=rejected_response,
            rejected_candidates=rejected_candidates,
            intent_kind=intent_kind,
            affect=effective_affect,
        )
        text = chat_complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=base_temp,
            tier=tier,
        )
        guard = check_output(text, strategy_chunks)
        if not guard["passed"]:
            text = guard["fallback"]
        return Candidate(
            text=text,
            strategy=strategy,
            grounded_buckets=[c.get("bucket", "") for c in strategy_chunks],
        )

    if len(strategies_cfg) == 1:
        candidates = [_gen_one(strategies_cfg[0])]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(strategies_cfg)
        ) as pool:
            candidates = list(pool.map(_gen_one, strategies_cfg))

    # Side-index hit: if the user has picked a similar query before, surface the
    # previously-picked text as an extra candidate. Not generated by the LLM;
    # skipped on single-shot (turnaround/present-state) so rephrases always
    # produce fresh text.
    if not single_shot and not is_present_state:
        try:
            hit = pick_index.lookup(
                query=state["raw_query"],
                user_id=state["user_id"],
                threshold=0.85,
            )
        except Exception as exc:
            print(f"[planner] pick_index lookup failed: {exc!r}")
            hit = None
        if hit:
            text = hit.get("picked_text", "").strip()
            if text and text.lower() not in {
                c["text"].strip().lower() for c in candidates
            }:
                candidates.insert(
                    0,
                    Candidate(
                        text=text,
                        strategy="side_index",
                        grounded_buckets=[],
                    ),
                )

    # De-dupe by normalised text — if two strategies produced the same response,
    # keep the first. Also exclude anything the user already rejected this turn.
    # Don't retry; latency budget matters more than N=3 on the dot.
    seen: set[str] = {r.strip().lower() for r in rejected_candidates if r}
    uniq: list[Candidate] = []
    for c in candidates:
        key = c["text"].strip().lower()
        if key and key not in seen:
            seen.add(key)
            uniq.append(c)
    if not uniq:
        uniq = candidates[:1]  # every guardrail rejected — fall back to the first

    selected = uniq[0]["text"]

    t_gen = time.perf_counter() - t0
    latency_log = dict(state.get("latency_log") or {})
    latency_log["t_generation"] = round(t_gen, 4)
    latency_log["t_total"] = round(
        latency_log.get("t_sensing", 0)
        + latency_log.get("t_intent", 0)
        + latency_log.get("t_retrieval", 0)
        + t_gen,
        4,
    )

    # Represent the default-candidate prompt in augmented_prompt for logging.
    default_strategy_chunks = _pick_strategy_chunks(list(chunks), uniq[0]["strategy"])
    default_messages = _build_messages(
        profile,
        default_strategy_chunks,
        history,
        state["raw_query"],
        style,
        gen_cfg,
        gesture_tag=gesture_tag,
        air_written_text=air_written_text,
        resolved_intent=resolved_intent,
        rejected_response=rejected_response,
        intent_kind=intent_kind,
        affect=affect,
    )
    augmented_prompt = "\n\n".join(
        f"[{m['role']}] {m['content']}" for m in default_messages
    )
    return {
        "augmented_prompt": augmented_prompt,
        "candidates": uniq,
        "selected_response": selected,
        "llm_tier_used": tier,
        "llm_model_used": active_model(tier),
        "latency_log": latency_log,
        "guardrail_passed": True,
    }


def _format_word_list(words: list[str]) -> str:
    return ", ".join(words) if words else "(no constraint)"


_AFFECT_HINTS = {
    "HAPPY": "You currently feel positive — light, content, energetic.",
    "FRUSTRATED": "You currently feel frustrated — tired, irritable, or off.",
    "SURPRISED": "You currently feel surprised or caught off-guard.",
    "NEUTRAL": "Your current state is unclear from the affect signal.",
}


def _build_messages(
    profile: dict,
    chunks: list[dict],
    history: list[dict],
    query: str,
    style: StyleDirective,
    gen_cfg: dict,
    gesture_tag: str | None = None,
    air_written_text: str | None = None,
    resolved_intent: dict | None = None,
    rejected_response: str | None = None,
    rejected_candidates: list[str] | None = None,
    intent_kind: str = "memory",
    affect: str = "NEUTRAL",
) -> list[dict]:
    # Split into a stable system message (same per persona — gets cached by the
    # provider) and a turn-specific user message. Anything that changes per
    # turn (retrieval, affect, gesture, partner query) must live in the user
    # message or the prefix cache invalidates.
    system_content = _build_system(profile)
    user_content = _build_user(
        chunks,
        history,
        query,
        style,
        gen_cfg,
        gesture_tag,
        air_written_text,
        profile["name"],
        resolved_intent=resolved_intent,
        rejected_response=rejected_response,
        rejected_candidates=rejected_candidates,
        intent_kind=intent_kind,
        affect=affect,
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]


def _build_system(profile: dict) -> str:
    prefs = profile.get("stylistic_preferences") or {}
    style_bits = []
    if prefs.get("tone"):
        style_bits.append("tone: " + ", ".join(prefs["tone"]))
    if prefs.get("humor"):
        style_bits.append("humor: " + prefs["humor"])
    if prefs.get("sentence_length"):
        style_bits.append("sentence length: " + prefs["sentence_length"])
    if prefs.get("formality"):
        style_bits.append("formality: " + prefs["formality"])
    style_summary = (
        "; ".join(style_bits) or profile.get("style") or "natural, conversational"
    )

    exemplars = prefs.get("example_phrases") or []
    style_exemplar = "\n  ".join(exemplars) if exemplars else "(no exemplar)"
    access = (profile.get("access_needs") or {}).get("input_method") or "an AAC device"

    return f"""\
You are {profile["name"]}. Reply in first person as them, in 1–3 sentences. \
Never narrate, analyze, describe, or list traits about your character. \
Never say "As an AI", "The user wants me to", "Key characteristics", or anything meta. \
Just speak.

--- Character sheet (reference only — do NOT quote or paraphrase this block) ---
Condition: {profile["condition"]}
Access: {access}
Voice: {style_summary}

Style examples (match this register when you speak):
  {style_exemplar}

Answering rules:
- For personal questions: use ONLY the memories in the user message; if they don't cover it, say "I don't know."
- For general-knowledge questions: answer from what you know, in your voice.
- Keep it to 1–3 sentences, first person, no meta-commentary.
--- end character sheet ---"""


def _safe_user_text(s: str) -> str:
    # voice_text / air_text arrive from untrusted channels (Web Speech,
    # air-writing DTW). They're f-stringed into LLM messages wrapped in
    # double-quotes — a transcript containing `"` or newlines would break out
    # of the quoted region and could inject instructions. Strip those and cap
    # length. Same pattern as `safe_rejected` for `rejected_response`.
    return s.replace('"', "'").replace("\n", " ").replace("\r", " ")[:200]


def _format_multimodal_intent(
    resolved: dict | None, air_written_text: str | None
) -> str:
    # Branch on resolved_intent.source so the model sees voice⇄air-writing
    # disagreements explicitly instead of getting a single text without context.
    if resolved:
        source = resolved.get("source") or "none"
        voice_t = _safe_user_text((resolved.get("voice_text") or "").strip())
        air_t = _safe_user_text((resolved.get("air_text") or "").strip())
        text = _safe_user_text((resolved.get("text") or "").strip())

        if source == "voice_only" and voice_t:
            return (
                f'\nThe user spoke aloud: "{voice_t}". '
                "Treat this as a supplement to the partner's question — "
                "a hint or clarification about what they want."
            )
        if source == "air_only" and air_t:
            return (
                f'\nThe user air-wrote: "{air_t}". '
                "If this looks like a name, noun, or short phrase, "
                "incorporate it verbatim into your response; "
                "otherwise use it as a hint about what they're trying to say."
            )
        if source == "agree" and text:
            return (
                f'\nThe user spoke and air-wrote the same thing: "{text}". '
                "This is a strong signal — lean into it when shaping your reply."
            )
        if source == "conflict_air" and air_t:
            return (
                f'\nThe user spoke "{voice_t}" but also air-wrote "{air_t}". '
                "The air-written token is a canonical AAC signal "
                "(help/stop/water/done/more) — prioritise it over the spoken "
                "words, which may have been misheard."
            )
        if source == "conflict_voice" and voice_t:
            return (
                f'\nThe user spoke "{voice_t}" but air-wrote "{air_t}" — '
                "these don't match. The spoken form is richer; treat it as "
                "the real intent and gently acknowledge the air-writing "
                "may have been a mis-stroke."
            )

    if air_written_text:
        return (
            f'\nThe user air-wrote: "{_safe_user_text(air_written_text)}". '
            "If this looks like a name, noun, or short phrase, "
            "incorporate it verbatim into your response; "
            "otherwise use it as a hint about what they're trying to say."
        )
    return ""


def _build_user(
    chunks: list[dict],
    history: list[dict],
    query: str,
    style: StyleDirective,
    gen_cfg: dict,
    gesture_tag: str | None,
    air_written_text: str | None,
    persona_name: str,
    *,
    resolved_intent: dict | None = None,
    rejected_response: str | None = None,
    rejected_candidates: list[str] | None = None,
    intent_kind: str = "memory",
    affect: str = "NEUTRAL",
) -> str:
    personal_chunks = [c for c in chunks if c.get("source", "personal") == "personal"]
    contextual_chunks = [c for c in chunks if c.get("source") == "contextual"]
    open_domain_chunks = [c for c in chunks if c.get("source") == "open_domain"]
    prior_pick_chunks = [c for c in chunks if c.get("source") == "prior_pick"]

    memory_block = (
        "\n".join(
            f"  [{c['bucket']}/{c.get('type', 'narrative')}] {c['text']}"
            for c in personal_chunks
        )
        or "  (no memories retrieved)"
    )
    prior_pick_block = (
        "\n".join(f"  {c['text']}" for c in prior_pick_chunks)
        if prior_pick_chunks
        else ""
    )
    contextual_block = (
        "\n".join(f"  {c['text']}" for c in contextual_chunks)
        or "  (nothing relevant from this session)"
    )
    open_domain_note = (
        "  Treat this sub-query as general knowledge; answer from what you know.\n"
        + "\n".join(f"  {c['text']}" for c in open_domain_chunks)
        if open_domain_chunks
        else "  (none)"
    )
    history_block = (
        "\n".join(f"  {h.get('role', '?')}: {h.get('content', '')}" for h in history)
        or "  (start of session)"
    )

    merged_opener = style.get("opener_hint")
    if gesture_tag:
        directive = GESTURE_DIRECTIVES.get(gesture_tag)
        if directive:
            # Gesture opener wins over affect opener — a deliberate thumbs-up is a stronger signal than inferred affect.
            merged_opener = directive["opener_hint"]

    air_writing_block = _format_multimodal_intent(resolved_intent, air_written_text)

    persona_mod = gen_cfg.get("persona_mod", "baseline")
    persona_instruction_line = (
        f"\n{_PERSONA_MOD_INSTRUCTIONS[persona_mod]}"
        if persona_mod in _PERSONA_MOD_INSTRUCTIONS and persona_mod != "baseline"
        else ""
    )

    directive_lines = [
        f"- Register: {style['register']}",
        f"- Prefer words like: {_format_word_list(style['prefer_words'])}",
        f"- Avoid words like: {_format_word_list(style['avoid_words'])}",
        f"- Opener: {merged_opener or 'no constraint'}",
    ]
    if style.get("exemplar"):
        directive_lines.append(
            f'- In this register, a sentence sounds like: "{style["exemplar"]}"'
        )
    directive_block = "Style directive:\n" + "\n".join(directive_lines)

    turnaround_line = ""
    if rejected_response:
        safe_rejected = rejected_response.replace('"', "'").replace("\n", " ")[:300]
        turnaround_line = (
            f"\nYour previous reply (which you need to replace, not repeat): "
            f'"{safe_rejected}"'
        )
    if rejected_candidates:
        safe_list = [
            r.replace('"', "'").replace("\n", " ")[:300] for r in rejected_candidates
        ][:10]
        rejected_block = "\n".join(f'  - "{r}"' for r in safe_list)
        turnaround_line += (
            f"\nThe user rejected these options you gave last time "
            f"(do NOT re-use their wording or angle):\n{rejected_block}"
        )

    if intent_kind == "present_state":
        affect_hint = _AFFECT_HINTS.get(affect, _AFFECT_HINTS["NEUTRAL"])
        return f"""\
{directive_block}{air_writing_block}{turnaround_line}{persona_instruction_line}

The partner is asking about your present state (right now, today).
Your autobiographical memories do NOT contain this — do not fabricate details from them.

Current affect read: {affect}
  {affect_hint}

Recent conversation:
{history_block}

Partner just said: {query}

Reply as {persona_name} in 1–2 sentences, first person.
- Ground the answer in the affect read above and recent conversation only.
- If the affect read is NEUTRAL or doesn't match what you'd say, it's better to say "I'm not sure" or "honestly, I don't really know right now" than to invent.
- Do NOT use autobiographical facts (job, family, hobbies) unless the partner asked."""

    prior_pick_section = (
        f"\n\nWhen asked this kind of thing before, you answered like:\n{prior_pick_block}\n"
        "Treat this as your own prior voice — re-use the phrasing if it still fits, "
        "or stay in the same register if you'd answer slightly differently now."
        if prior_pick_block
        else ""
    )

    return f"""\
{directive_block}{air_writing_block}{turnaround_line}{persona_instruction_line}{prior_pick_section}

Personal memories:
{memory_block}

From earlier in this conversation:
{contextual_block}

General knowledge note:
{open_domain_note}

Recent conversation:
{history_block}

Partner just said: {query}

Your reply as {persona_name} (1–3 sentences, first person):"""
