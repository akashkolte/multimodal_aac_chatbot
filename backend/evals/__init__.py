# Evaluation metrics — compute after pipeline returns, before API response.
from backend.evals.efficiency import compute_efficiency
from backend.evals.faithfulness import compute_faithfulness
from backend.evals.multimodal_alignment import compute_multimodal_alignment


def compute_evals(
    response: str,
    chunks: list[dict],
    latency_log: dict,
    affect: str | None,
    gesture_tag: str | None,
    gaze_bucket: str | None,
    slo_target: float = 6.0,
) -> dict:
    """Run all eval scorers and return a unified EvalScores dict."""
    faith = compute_faithfulness(response, chunks)
    eff = compute_efficiency(latency_log, slo_target)
    align = compute_multimodal_alignment(
        response, affect, gesture_tag, gaze_bucket, chunks
    )

    return {
        "groundedness": faith["groundedness"],
        "hallucination_rate": faith["hallucination_rate"],
        "no_evidence": faith["no_evidence"],
        "t_total_s": eff["t_total"],
        "slo_target_s": eff["slo_target"],
        "slo_passed": eff["slo_passed"],
        "slo_margin_s": eff["margin_s"],
        "multimodal_alignment": align["overall_score"],
        "affect_alignment": align["affect_alignment"],
        "gesture_alignment": align["gesture_alignment"],
        "gaze_alignment": align["gaze_alignment"],
    }
