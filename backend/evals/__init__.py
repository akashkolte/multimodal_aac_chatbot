# Evaluation metrics — compute after pipeline returns, before API response.
import torch

from backend.evals.diversity import compute_candidate_diversity
from backend.evals.efficiency import compute_efficiency
from backend.evals.faithfulness import compute_faithfulness, compute_faithfulness_batch
from backend.evals.multimodal_alignment import compute_multimodal_alignment
from backend.evals.relevance import compute_relevance


def _score_candidates_batched(
    candidates: list[dict],
    chunks: list[dict],
    query: str,
) -> tuple[list[dict], "torch.Tensor | None"]:
    """One BGE pass + one NLI pass across all candidates. Returns per-candidate
    score dicts and the candidate vector matrix (for diversity reuse), or None
    when no embedding pass was needed."""
    texts = [c.get("text", "") for c in candidates]
    faiths = compute_faithfulness_batch(texts, chunks)

    cand_vecs: torch.Tensor | None = None
    if query.strip() and any(t.strip() for t in texts):
        from backend.retrieval.vector_store import embed_texts

        vecs = embed_texts([query] + texts)
        q_vec = vecs[0]
        cand_vecs = vecs[1:]
        relevances = [
            round(max(0.0, float(q_vec @ cand_vecs[i])), 4) for i in range(len(texts))
        ]
    else:
        relevances = [0.0] * len(texts)

    scores = [{**f, "relevance": r} for f, r in zip(faiths, relevances, strict=True)]
    return scores, cand_vecs


def _diversity_from_vecs(cand_vecs: "torch.Tensor") -> dict:
    n = cand_vecs.shape[0]
    sims = cand_vecs @ cand_vecs.T
    iu = torch.triu_indices(n, n, offset=1)
    return {
        "candidate_diversity": round(float(1.0 - sims[iu[0], iu[1]].mean().item()), 4),
        "n_candidates": n,
    }


def compute_evals(
    response: str,
    chunks: list[dict],
    latency_log: dict,
    affect: str | None,
    gesture_tag: str | None,
    gaze_bucket: str | None,
    slo_target: float = 6.0,
    query: str = "",
    candidates: list[dict] | None = None,
    selected_idx: int | None = None,
) -> dict:
    """Run all eval scorers and return a unified EvalScores dict.

    When candidates are provided, scoring is batched: one BGE encode for
    query + all candidates, one NLI predict across all (sentence, chunk)
    pairs, then sliced per candidate. The selected candidate's scores are
    reused as the top-level fields so the existing UI pills keep working.
    """
    eff = compute_efficiency(latency_log, slo_target)
    align = compute_multimodal_alignment(
        response, affect, gesture_tag, gaze_bucket, chunks
    )

    per_cand: list[dict] = []
    cand_vecs = None
    if candidates:
        # The planner serves uniq[0] as `selected_response`, so when caller
        # didn't pass selected_idx explicitly, default to 0 rather than
        # text-matching (which can collide on duplicate candidate texts).
        if selected_idx is None:
            selected_idx = 0
        scored, cand_vecs = _score_candidates_batched(candidates, chunks, query)
        per_cand = [
            {
                "idx": i,
                "strategy": c.get("strategy", "unknown"),
                "selected": (selected_idx is not None and i == selected_idx),
                **scored[i],
            }
            for i, c in enumerate(candidates)
        ]

    if per_cand and selected_idx is not None and 0 <= selected_idx < len(per_cand):
        # Strip per-candidate-only keys before reusing as top-level scores.
        top = {
            k: v
            for k, v in per_cand[selected_idx].items()
            if k not in ("idx", "strategy", "selected")
        }
    else:
        faith = compute_faithfulness(response, chunks)
        top = {**faith, "relevance": compute_relevance(response, query)["relevance"]}

    out = {
        **top,
        "t_total_s": eff["t_total"],
        "slo_target_s": eff["slo_target"],
        "slo_passed": eff["slo_passed"],
        "slo_margin_s": eff["margin_s"],
        "multimodal_alignment": align["overall_score"],
        "affect_alignment": align["affect_alignment"],
        "gesture_alignment": align["gesture_alignment"],
        "gaze_alignment": align["gaze_alignment"],
        "explain": align.get("explain", {}),
    }

    if per_cand:
        out["candidates_eval"] = per_cand
        n = len(candidates)
        if n < 2:
            out["candidate_diversity"] = 0.0
            out["n_candidates"] = n
        elif cand_vecs is not None:
            # Reuse vectors from the relevance pass.
            out.update(_diversity_from_vecs(cand_vecs))
        else:
            # Standalone BGE encode (e.g. when query was empty so the relevance
            # pass was skipped).
            out.update(compute_candidate_diversity(candidates))
    else:
        out["candidate_diversity"] = 0.0
        out["n_candidates"] = 1 if response else 0

    return out
