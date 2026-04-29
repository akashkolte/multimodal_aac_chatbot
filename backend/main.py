# CLI entry point for the AAC chatbot pipeline.
import argparse
import copy
import json
import os
import sys
import time

from backend.config.settings import settings
from backend.guardrails.checks import check_input
from backend.pipeline.graph import run_pipeline
from backend.pipeline.nodes.intent import _AFFECT_CONFIG
from backend.pipeline.state import GenerationConfig, PipelineState
from backend.retrieval.priors import BUCKETS, CHUNK_TYPES, uniform
from backend.retrieval.vector_store import _get_embedder
from backend.sensing.bucket_keywords import infer_bucket


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AAC Chatbot CLI")
    p.add_argument("--user", type=str, default=None, help="Persona user_id")
    p.add_argument("--debug", action="store_true", help="Print latency table each turn")
    p.add_argument(
        "--fast",
        action="store_true",
        help="Skip LLM intent call — use keyword routing instead (faster local dev)",
    )
    p.add_argument(
        "--tier",
        type=str,
        default=None,
        choices=["primary", "fallback"],
        help="Override LLM tier (default: settings.active_llm_tier)",
    )
    return p.parse_args()


# ── Fast keyword-based intent routing (bypasses the slow LLM intent call) ──────


def _keyword_intent(query: str) -> tuple[dict, GenerationConfig]:
    """Replicate milestone-1 keyword routing as a fast local-dev shortcut."""
    q = query.lower()
    bucket = infer_bucket(query)

    intent_type = (
        "CONTEXTUAL"
        if any(w in q for w in ["you just said", "earlier", "you mentioned"])
        else "PERSONAL"
    )

    # `style_constraints` is vestigial — planner reads `generation_config` (below) as the source of truth.
    route = {
        "sub_intents": [
            {
                "type": intent_type,
                "query": query,
                "bucket_hint": bucket,
                "priority": "normal",
            }
        ],
        "style_constraints": {
            "tone_tag": "[TONE:DEFAULT]",
            "max_tokens": 100,
            "retrieval_mode": "full",
            "persona_mod": "baseline",
        },
        "affect": "NEUTRAL",
    }
    # Deep-copy: callers may mutate gen_config downstream; never hand them the shared constant.
    gen_config: GenerationConfig = copy.deepcopy(_AFFECT_CONFIG["NEUTRAL"])
    return route, gen_config


def load_users() -> dict[str, dict]:
    with open(settings.users_json) as f:
        return {u["id"]: u for u in json.load(f)["users"]}


def load_persona_profile(user_id: str) -> dict:
    with open(settings.memories_dir / f"{user_id}.json") as f:
        return json.load(f)["profile"]


def select_user(users: dict[str, dict], user_arg: str | None) -> str:
    if user_arg:
        if user_arg not in users:
            print(f"Unknown user '{user_arg}'. Available: {list(users)}")
            sys.exit(1)
        return user_arg
    print("\nAvailable personas:")
    for uid, u in users.items():
        print(f"  {uid:20s} — {u['name']} ({u['condition']})")
    uid = input("\nSelect user id: ").strip()
    if uid not in users:
        print("Invalid id.")
        sys.exit(1)
    return uid


def print_latency(log: dict, turn: int) -> None:
    fields = ["t_sensing", "t_intent", "t_retrieval", "t_generation", "t_total"]
    labels = ["sensing", "intent", "retrieval", "generation", "TOTAL"]
    vals = [f"{log.get(f, 0):.3f}s" for f in fields]
    widths = [max(len(l), len(v)) for l, v in zip(labels, vals)]
    sep = " | "
    print(f"\n[turn {turn} latency]")
    print(sep.join(l.ljust(w) for l, w in zip(labels, widths)))
    print(sep.join(v.ljust(w) for v, w in zip(vals, widths)))


def main() -> None:
    args = parse_args()

    # Optionally override the LLM tier at runtime
    if args.tier:
        os.environ["ACTIVE_LLM_TIER"] = args.tier

    users = load_users()
    user_id = select_user(users, args.user)
    profile = load_persona_profile(user_id)

    # Warm up models
    print(f"\nLoading models for {profile['name']} …", end=" ", flush=True)
    _get_embedder()
    print("ready.\n")

    session_history: list[dict] = []
    bucket_priors = uniform(BUCKETS)
    type_priors = uniform(CHUNK_TYPES)
    turn_id = 0

    print(f"Chatting as {profile['name']}. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Partner: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if query.lower() in {"quit", "exit", "q"}:
            break
        if not query:
            continue

        guard = check_input(query)
        if not guard["allowed"]:
            print(f"AAC Bot: {guard['fallback']}\n")
            continue

        turn_id += 1

        # --fast: resolve intent via keywords, skip the slow LLM intent node
        t_intent_fast = 0.0
        if args.fast:
            t0 = time.perf_counter()
            pre_route, pre_gen_config = _keyword_intent(query)
            t_intent_fast = time.perf_counter() - t0
        else:
            pre_route, pre_gen_config = None, None

        state = PipelineState(
            user_id=user_id,
            persona_profile=profile,
            session_history=session_history,
            turn_id=turn_id,
            affect=None,
            gesture_tag=None,
            gaze_bucket=None,
            air_written_text=None,
            voice_text=None,
            resolved_intent=None,
            raw_query=query,
            intent_route=pre_route,  # pre-filled → intent node sees it and skips LLM call
            generation_config=pre_gen_config,
            retrieved_chunks=[],
            bucket_priors=bucket_priors,
            type_priors=type_priors,
            retrieval_mode_used="",
            augmented_prompt=None,
            candidates=[],
            rejected_candidates=[],
            selected_response=None,
            llm_tier_used="",
            latency_log={
                "t_sensing": 0.0,
                "t_intent": round(t_intent_fast, 4),
                "t_retrieval": 0.0,
                "t_generation": 0.0,
                "t_total": 0.0,
            },
            run_id=None,
            guardrail_passed=True,
        )

        result: PipelineState = run_pipeline(state)

        print(f"AAC Bot: {result['selected_response']}\n")

        session_history = result["session_history"]
        bucket_priors = result["bucket_priors"]
        type_priors = result["type_priors"]

        if args.debug:
            print_latency(result.get("latency_log") or {}, turn_id)
            print(
                f"  tier={result.get('llm_tier_used')} | "
                f"retrieval={result.get('retrieval_mode_used')} | "
                f"affect={(result.get('affect') or {}).get('emotion', '?')}\n"
            )


if __name__ == "__main__":
    main()
