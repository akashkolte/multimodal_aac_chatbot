"""Offline aggregator: reads turns.jsonl, evals.jsonl, ratings.jsonl and prints
per-persona metrics. Run:  python -m backend.evals.aggregate
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from backend.config.settings import settings

# Mean pairwise cosine distance below this means the picker showed near-paraphrases.
_DIVERSITY_FLOOR = 0.10


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(
            f"[aggregate] skipped {skipped} malformed lines in {path}",
            file=sys.stderr,
        )
    return out


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = max(0, min(len(values) - 1, int(round(q * (len(values) - 1)))))
    return sorted(values)[idx]


def _fmt_ms(s: float) -> str:
    return f"{s * 1000:.0f}ms"


def report_latency(turns: list[dict]) -> None:
    print("\n=== Communication Efficiency (latency) ===")
    by_group: dict[tuple[str, str], list[float]] = defaultdict(list)
    for t in turns:
        key = (t.get("user_id", "?"), t.get("llm_tier", "?"))
        by_group[key].append(t.get("latency", {}).get("t_total", 0.0))

    slo = settings.slo_target_s
    print(f"SLO target: < {slo}s")
    print(
        f"{'user_id':<18} {'tier':<10} {'n':>5} {'p50':>8} {'p95':>8} {'p99':>8} {'pass%':>7}"
    )
    for (uid, tier), lats in sorted(by_group.items()):
        if not lats:
            continue
        p50 = _quantile(lats, 0.5)
        p95 = _quantile(lats, 0.95)
        p99 = _quantile(lats, 0.99)
        passed = sum(1 for x in lats if x < slo) / len(lats) * 100
        print(
            f"{uid:<18} {tier:<10} {len(lats):>5} "
            f"{_fmt_ms(p50):>8} {_fmt_ms(p95):>8} {_fmt_ms(p99):>8} {passed:>6.1f}%"
        )


def report_faithfulness(evals: list[dict]) -> None:
    print("\n=== Factual Faithfulness ===")
    scored = [e for e in evals if not e.get("no_evidence")]
    if not scored:
        print("(no turns with retrieved evidence)")
        return
    by_user: dict[str, list[dict]] = defaultdict(list)
    for e in scored:
        by_user[e.get("user_id", "?")].append(e)

    print(f"{'user_id':<18} {'n':>5} {'groundedness':>14} {'hallucination':>14}")
    for uid, rows in sorted(by_user.items()):
        g = statistics.mean(r["groundedness"] for r in rows)
        h = statistics.mean(r["hallucination_rate"] for r in rows)
        print(f"{uid:<18} {len(rows):>5} {g:>13.2%} {h:>13.2%}")


def _mean_nonzero(rows: list[dict], key: str) -> tuple[float, float]:
    # Coverage % undercounts real zeros (a genuinely 0.0-aligned response looks
    # identical to one where the signal was absent). Fixable by serializing
    # null for absent signals in compute_multimodal_alignment.
    vals = [float(r.get(key, 0.0)) for r in rows]
    nonzero = [v for v in vals if v > 0]
    if not nonzero:
        return 0.0, 0.0
    return statistics.mean(nonzero), len(nonzero) / len(vals)


def _fmt_mean_cov(rows: list[dict], key: str) -> str:
    mean, cov = _mean_nonzero(rows, key)
    return f"{mean:>5.0%}|{cov:>5.0%}"


def report_multimodal(evals: list[dict]) -> None:
    print("\n=== Multimodal Alignment (mean among non-zero | coverage) ===")
    if not evals:
        print("(no evals logged)")
        return
    by_user: dict[str, list[dict]] = defaultdict(list)
    for e in evals:
        by_user[e.get("user_id", "?")].append(e)

    print(f"{'user_id':<18} {'n':>5} {'affect':>16} {'gesture':>16} {'gaze':>16}")
    for uid, rows in sorted(by_user.items()):
        print(
            f"{uid:<18} {len(rows):>5} "
            f"{_fmt_mean_cov(rows, 'affect_alignment'):>16} "
            f"{_fmt_mean_cov(rows, 'gesture_alignment'):>16} "
            f"{_fmt_mean_cov(rows, 'gaze_alignment'):>16}"
        )


def report_authenticity(ratings: list[dict]) -> None:
    print("\n=== Perceived Authenticity (Likert 1-5) ===")
    by_user: dict[str, list[int]] = defaultdict(list)
    for r in ratings:
        raw = r.get("authenticity")
        try:
            score = int(raw)
        except (TypeError, ValueError):
            continue
        if not 1 <= score <= 5:
            continue
        by_user[r.get("user_id", "?")].append(score)

    if not by_user:
        print("(no valid ratings logged yet)")
        return

    print(f"{'user_id':<18} {'n':>5} {'mean':>6} {'dist (1..5)':>22}")
    for uid, scores in sorted(by_user.items()):
        mean = statistics.mean(scores)
        dist = [scores.count(i) for i in range(1, 6)]
        dist_str = "/".join(str(x) for x in dist)
        print(f"{uid:<18} {len(scores):>5} {mean:>6.2f} {dist_str:>22}")


def report_picker(turns: list[dict], picks: list[dict], evals: list[dict]) -> None:
    """Picker behaviour: pick rate, regenerate rate, strategy win rate, and
    whether the user's pick beat candidate 0 on grounded/relevance.

    Sources:
      - turns.jsonl    one row per turn, includes `candidates` and `n_candidates`
      - picks.jsonl    one row per /chat/pick — strategy, picked_idx, run_id
      - evals.jsonl    candidates_eval[] with per-candidate grounded + relevance
    """
    print("\n=== Picker Behaviour ===")
    multi = [t for t in turns if (t.get("n_candidates") or 0) >= 2]
    if not multi:
        print(
            "(no multi-candidate turns logged — older format or single-candidate runs)"
        )
        return

    picks_by_run = {p["run_id"]: p for p in picks if p.get("run_id")}
    evals_by_run = {e["run_id"]: e for e in evals if e.get("run_id")}

    n_multi = len(multi)
    n_picked = sum(1 for t in multi if t["run_id"] in picks_by_run)
    # A (user_id, turn_id) seen more than once means the planner re-ran for
    # the same partner query — that's a regenerate. The denominator is the
    # number of distinct (user, turn) conversations that had at least one
    # multi-candidate run, not the raw row count.
    seen: dict[tuple[str, int], int] = defaultdict(int)
    for t in multi:
        seen[(t.get("user_id", "?"), t.get("turn_id", -1))] += 1
    n_regenerated_turns = sum(1 for c in seen.values() if c > 1)
    n_distinct_turns = max(1, len(seen))
    print(
        f"multi-candidate turns: {n_multi} ({n_distinct_turns} distinct)   "
        f"pick rate: {n_picked / n_multi:.0%}   "
        f"regenerate rate: {n_regenerated_turns / n_distinct_turns:.0%} "
        f"(% of distinct turns that re-ran)"
    )

    # Strategy win rate — among multi-candidate picks only, how often does
    # each strategy win. Picks on single-candidate turns aren't a real "win"
    # (no alternative to lose to) so we filter them out.
    multi_run_ids = {t["run_id"] for t in multi}
    strategy_count: dict[str, int] = defaultdict(int)
    for run_id, p in picks_by_run.items():
        if run_id in multi_run_ids:
            strategy_count[p.get("strategy", "unknown")] += 1
    if strategy_count:
        total = sum(strategy_count.values())
        print(f"\nStrategy win rate (n={total} picks):")
        print(f"  {'strategy':<16} {'picks':>6} {'pct':>6}")
        for s, n in sorted(strategy_count.items(), key=lambda x: -x[1]):
            print(f"  {s:<16} {n:>6} {n / total:>5.0%}")

    # Did the picker beat candidate 0? Only meaningful when we have per-candidate
    # eval scores AND the user picked a non-zero index. A "win" = picked
    # candidate scored strictly higher on the metric than candidate 0.
    head_to_head = []
    for run_id, pick in picks_by_run.items():
        ev = evals_by_run.get(run_id)
        if not ev or not ev.get("candidates_eval"):
            continue
        cands = ev["candidates_eval"]
        if len(cands) < 2:
            continue
        picked_idx = pick.get("picked_idx", 0)
        if picked_idx == 0 or picked_idx >= len(cands):
            continue
        head_to_head.append(
            {
                "picked_grounded": cands[picked_idx]["groundedness"],
                "cand0_grounded": cands[0]["groundedness"],
                "picked_relevance": cands[picked_idx].get("relevance", 0.0),
                "cand0_relevance": cands[0].get("relevance", 0.0),
            }
        )

    if head_to_head:
        n = len(head_to_head)
        beat_grounded = sum(
            1 for h in head_to_head if h["picked_grounded"] > h["cand0_grounded"]
        )
        tied_grounded = sum(
            1 for h in head_to_head if h["picked_grounded"] == h["cand0_grounded"]
        )
        beat_rel = sum(
            1 for h in head_to_head if h["picked_relevance"] > h["cand0_relevance"]
        )
        print(f"\nDid picker beat candidate 0? (n={n} picks where picked_idx > 0)")
        print(
            f"  groundedness:  picker > cand0 = {beat_grounded}/{n} ({beat_grounded / n:.0%}), "
            f"tied = {tied_grounded}/{n}"
        )
        print(f"  relevance:     picker > cand0 = {beat_rel}/{n} ({beat_rel / n:.0%})")
    else:
        print(
            "\n(no picks of candidate 1+ with per-candidate eval data — can't measure picker quality yet)"
        )

    # Diversity: among multi-candidate turns with eval data, how often is the
    # picker showing near-paraphrases (the "aloha" problem)?
    div_scored = [
        ev
        for ev in evals_by_run.values()
        if ev.get("n_candidates", 0) >= 2 and "candidate_diversity" in ev
    ]
    if div_scored:
        diversities = [float(e["candidate_diversity"]) for e in div_scored]
        low = sum(1 for d in diversities if d < _DIVERSITY_FLOOR)
        print(
            f"\nCandidate diversity (n={len(div_scored)} turns): "
            f"mean={statistics.mean(diversities):.2f}  "
            f"low (<{_DIVERSITY_FLOOR:.2f}): {low}/{len(div_scored)} ({low / len(div_scored):.0%})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate AAC eval metrics")
    parser.add_argument("--logs", type=Path, default=settings.logs_dir)
    args = parser.parse_args()

    turns = _load(args.logs / "turns.jsonl")
    evals = _load(args.logs / "evals.jsonl")
    ratings = _load(args.logs / "ratings.jsonl")
    picks = _load(args.logs / "picks.jsonl")

    print(
        f"Loaded: {len(turns)} turns, {len(evals)} evals, "
        f"{len(picks)} picks, {len(ratings)} ratings"
    )
    report_latency(turns)
    report_faithfulness(evals)
    report_multimodal(evals)
    report_picker(turns, picks, evals)
    report_authenticity(ratings)


if __name__ == "__main__":
    main()
