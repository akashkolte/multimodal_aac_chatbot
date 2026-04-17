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


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate AAC eval metrics")
    parser.add_argument("--logs", type=Path, default=settings.logs_dir)
    args = parser.parse_args()

    turns = _load(args.logs / "turns.jsonl")
    evals = _load(args.logs / "evals.jsonl")
    ratings = _load(args.logs / "ratings.jsonl")

    print(f"Loaded: {len(turns)} turns, {len(evals)} evals, {len(ratings)} ratings")
    report_latency(turns)
    report_faithfulness(evals)
    report_multimodal(evals)
    report_authenticity(ratings)


if __name__ == "__main__":
    main()
