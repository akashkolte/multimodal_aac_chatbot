# Pipeline orchestrator: intent → retrieval → generation → feedback.
from backend.config.settings import settings
from backend.pipeline.nodes import feedback, intent, planner, retrieval
from backend.pipeline.state import PipelineState


def _route_by_affect(state: PipelineState) -> str:
    emotion = (state.get("affect") or {}).get("emotion", "NEUTRAL")
    return "fast" if emotion == "FRUSTRATED" else "full"


def _route_by_latency(state: PipelineState) -> str:
    log = state.get("latency_log") or {}
    elapsed = log.get("t_intent", 0.0) + log.get("t_retrieval", 0.0)
    return "fallback" if elapsed > settings.fallback_latency_threshold else "primary"


def _merge(state: PipelineState, update: dict) -> None:
    state.update(update)  # type: ignore[typeddict-item]


def run_pipeline(state: PipelineState) -> PipelineState:
    _merge(state, intent.run(state))

    if _route_by_affect(state) == "fast":
        _merge(state, retrieval.run_fast(state))
    else:
        _merge(state, retrieval.run_full(state))

    if _route_by_latency(state) == "fallback":
        _merge(state, planner.run_fallback(state))
    else:
        _merge(state, planner.run_primary(state))

    _merge(state, feedback.run(state))
    return state


def run_until_planner(state: PipelineState) -> PipelineState:
    """Run intent + retrieval only. Used by the streaming endpoint so it can
    then drive the planner's token stream itself and call feedback at the end.
    """
    _merge(state, intent.run(state))
    if _route_by_affect(state) == "fast":
        _merge(state, retrieval.run_fast(state))
    else:
        _merge(state, retrieval.run_full(state))
    return state


def choose_planner_tier(state: PipelineState) -> str:
    return _route_by_latency(state)
