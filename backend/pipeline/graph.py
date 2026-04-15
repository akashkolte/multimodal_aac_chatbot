# LangGraph pipeline graph — intent → retrieval → generation → feedback.
from langgraph.graph import END, StateGraph

from backend.pipeline.nodes import feedback, intent, planner, retrieval
from backend.pipeline.state import PipelineState


def _route_by_affect(state: PipelineState) -> str:
    """Conditional edge: FRUSTRATED → fast path, otherwise full retrieval."""
    emotion = (state.get("affect") or {}).get("emotion", "NEUTRAL")
    return "fast" if emotion == "FRUSTRATED" else "full"


def _route_by_latency(state: PipelineState) -> str:
    """Conditional edge: if cumulative latency > threshold, use fallback LLM."""
    from backend.config.settings import settings

    log = state.get("latency_log") or {}
    elapsed = log.get("t_intent", 0.0) + log.get("t_retrieval", 0.0)
    return "fallback" if elapsed > settings.fallback_latency_threshold else "primary"


def build_graph() -> StateGraph:
    graph = StateGraph(PipelineState)

    # ── Nodes ──────────────────────────────────────────────────────────────────
    graph.add_node("intent", intent.run)
    graph.add_node("fast_retrieval", retrieval.run_fast)
    graph.add_node("full_retrieval", retrieval.run_full)
    graph.add_node("primary_gen", planner.run_primary)
    graph.add_node("fallback_gen", planner.run_fallback)
    graph.add_node("feedback", feedback.run)

    # ── Entry ──────────────────────────────────────────────────────────────────
    graph.set_entry_point("intent")

    # ── Affect-aware routing after intent ─────────────────────────────────────
    graph.add_conditional_edges(
        "intent",
        _route_by_affect,
        {"fast": "fast_retrieval", "full": "full_retrieval"},
    )

    # ── Latency-aware routing after retrieval ─────────────────────────────────
    graph.add_conditional_edges(
        "fast_retrieval",
        _route_by_latency,
        {"primary": "primary_gen", "fallback": "fallback_gen"},
    )
    graph.add_conditional_edges(
        "full_retrieval",
        _route_by_latency,
        {"primary": "primary_gen", "fallback": "fallback_gen"},
    )

    # ── Feedback loop ─────────────────────────────────────────────────────────
    graph.add_edge("primary_gen", "feedback")
    graph.add_edge("fallback_gen", "feedback")
    graph.add_edge("feedback", END)

    return graph.compile()


# Module-level compiled graph — import this everywhere
aac_graph = build_graph()
